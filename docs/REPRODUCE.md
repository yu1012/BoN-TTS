# Reproduction guide

End-to-end walkthrough with runtimes, disk requirements and expected outputs. Runtimes are for a single RTX 4090; the pipeline shards cleanly across GPUs.

## Design: one pool, many selections

```
prepare_pc_data  ->  synthesize_pool  ->  score_pool   ->  select_candidates  ->  evaluate  ->  report
   (CPU, mins)        (GPU, ~6h)          (GPU, per       (CPU, seconds)          (GPU, per    (CPU)
                                           verifier)                              evaluator)
```

Synthesis happens **once**. Every configuration — each verifier, each rank ensemble, each N, and the oracle — then selects from that shared candidate pool. Two reasons this matters beyond cost:

1. **Comparability.** Configurations differ only in which candidate they picked, never in the underlying audio. A WER difference cannot be a resampling artifact.
2. **Nesting.** N=3 uses the first 3 candidates of the N=5 pool, which uses the first 5 of the N=10 pool. A change across N reflects the extra candidates, not a different random draw.

Candidate 0 always uses the base seed (42), so the single-shot baseline is literally inside every pool, and all selection strategies break ties toward the lower index — a tie never moves you off the baseline sample.

## 0. Prerequisites

- Docker with the NVIDIA container toolkit, one GPU with ≥16 GB (24 GB comfortable).
- **Disk.** The N=10 pool is 11,270 clips written as 24 kHz 16-bit WAV, i.e. ~48 kB per second of audio. LibriSpeech-PC test-clean cross-sentence targets run 4–10 s, so the pool lands around **3–5 GB**. Budget **~15 GB** in total: selection runs *copy* the chosen audio by default, and 12 configurations × 1127 clips roughly doubles the pool's footprint (use `--link` to avoid this). Model checkpoints need a further ~10 GB, under `CACHE_DIR`.
- `librispeech_pc_test_clean_cross_sentence.lst` from the [F5-TTS repository](https://github.com/SWivid/F5-TTS) `data/` directory.
- LibriSpeech `test-clean` audio from [OpenSLR](https://www.openslr.org/12) (LibriSpeech-PC republishes only transcripts).

```bash
make build
make test     # 60 unit tests, CPU only, ~35 s
```

Point the mounts at real storage — the defaults write into the repo:

```bash
export OUTPUT_DIR=/data/bon-tts/outputs
export CACHE_DIR=/data/bon-tts/cache
export DATA_DIR=/data/corpora
```

## 1. Prepare the evaluation set

```bash
make shell
python scripts/prepare_pc_data.py \
    --lst-file /data/librispeech_pc_test_clean_cross_sentence.lst \
    --librispeech-root /data/LibriSpeech/test-clean
```

Decodes the 1483 unique FLAC files the protocol references to WAV and writes `metadata.json`. FLAC decoding inside the synthesis loop would otherwise dominate its runtime.

Expect `wrote 1127 entries`. A different count means the wrong `.lst` — WER will not be comparable with published numbers, and the script warns.

## 2. Synthesize the candidate pool

```bash
make pool                 # ~6 h for 1127 x 10 candidates
```

Shard across GPUs instead:

```bash
for gpu in 0 1 2 3; do
    python scripts/synthesize_pool.py --gpu $gpu \
        --offset $((gpu * 282)) --limit 282 &
done; wait
```

Each shard writes its own `pool_<offset>_<end>.json`; downstream scripts merge them. Re-running is safe — existing candidates are skipped unless `--overwrite`.

## 3. Score the pool with each verifier

```bash
make score                # w2v2-base, distil-sm, distil-v3
```

`w2v2-base` takes ~15 min; `distil-v3` (756M, autoregressive) is the slow one at ~2 h. Results cache to `<pool>/scores/<verifier>__<metric>__<normalization>.json`, so this is paid once per verifier regardless of how many strategies or N values you later evaluate.

> **Normalization.** The default is `simple` (lowercase + strip), which is what produced the published numbers. `f5_official` normalizes both sides identically and is cleaner for new work, but shifts the results — keep the default when reproducing. Score files are keyed by normalization, so both can coexist.

## 4. Select

```bash
make select               # every strategy x N in {3,5,10}; seconds, no GPU
```

Or individually:

```bash
python scripts/select_candidates.py --strategy single   --verifiers distil-v3            --n 10
python scripts/select_candidates.py --strategy rank_avg --verifiers w2v2-base distil-v3  --n 5
python scripts/select_candidates.py --strategy max_rank --verifiers w2v2-base distil-v3  --n 5
```

Each run writes `<output_dir>/select_<strategy>_<verifiers>_n<N>/` with the selected audio, per-utterance scores, the switch rate (how often it moved off candidate 0) and an RTF estimate.

Pass `--link` to symlink the selected audio instead of copying it — 12 configurations × 1127 clips adds up. Copies are the default because symlinks break if the pool moves.

## 5. Evaluate

```bash
make evaluate             # baseline + every run x 3 evaluators
```

`fwhisper-lgv3` runs ~20 min per configuration; the CTC evaluators are faster. Add `--sim --utmos` for quality metrics (another ~15 min per run).

Run **every** configuration under **every** evaluator. A configuration measured under one evaluator only cannot be interpreted — that is the paper's point, and it applies to your own runs too.

## 6. Report

```bash
make report
```

Prints the configuration × evaluator table with paired permutation p-values, Holm–Bonferroni within each evaluator column, and a `sig` column counting how many evaluators each row reaches p<0.05 under. `--bootstrap` adds 95% CIs.

## 7. Oracle headroom

```bash
python scripts/evaluate_pool.py  --evaluator fwhisper-lgv3   # ~3 h: all 10 candidates
python scripts/analyze_oracle.py --runs select_single_distil-v3_n10 select_max_rank_w2v2-base+distil-v3_n5
```

`evaluate_pool.py` is the expensive step (N× a normal evaluation). `analyze_oracle.py` then reads the cache and needs no GPU. Reported per evaluator: single-shot WER, oracle WER, headroom, and each run's recovery rate.

## Validation against the original runs

The scripts that produced the published tables were lost. This repo is a reimplementation, validated against the surviving output artifacts of those runs:

```bash
python scripts/validate_reproduction.py \
    --reference-dir /path/to/pc_oracle_candidates \
    --metadata /path/to/data/librispeech_pc_test_clean/metadata.json
```

Five checks:

| Check | What it pins down | Result |
|---|---|---|
| A | Text normalization vs. the reference normalized references | 1127/1127 exact, all 3 evaluators |
| B | Corpus WER aggregation (total edits / total words) | exact to <1e-9, all 3 |
| C | Oracle selection reproduces the reference oracle WER | exact to <1e-9, all 3 |
| D | Selection strategies reproduce the recorded winners | needs freshly scored verifiers |
| E | Bounds the one known normalizer divergence | 1/1127 hypotheses, ΔWER 0.005 pp |

Check B is deliberately scored on the artifact strings **as stored** (already normalized), so it tests aggregation in isolation rather than folding normalization differences into it.

**The one known divergence (check E).** Our normalizer takes its CJK punctuation set from `zhon.hanzi.punctuation`, exactly as upstream F5-TTS does. The original runs used a hand-written subset that omitted a few Unicode characters — notably U+2014 em dash. On the reference data this affects **one** Whisper hypothesis out of 1127 and moves corpus WER by **0.005 points**, from 2.0431% to 2.0382%.

We kept the stricter, upstream-matching normalizer rather than degrading it to force an exact match, since the paper claims to run the official F5-TTS pipeline. Check E asserts against a tolerance of `DEFAULT_MAX_WER_DELTA = 0.0005` (0.05 points — an order of magnitude of slack over the observed 0.005) rather than ignoring the difference, so any future change that shifts WER by even a tenth of a point still fails the check.

To run check D, score the pool first, then:

```bash
python scripts/validate_reproduction.py \
    --reference-dir /path/to/pc_oracle_candidates \
    --config configs/librispeech_pc.yaml \
    --compare-selection /path/to/pc_bon_n5_max_rank/synthesis_results_0000_1127.json \
    --strategy max_rank --verifiers w2v2-base distil-v3 --n 5
```

It reports the fraction of per-utterance winners that match and prints the score matrices for the first few disagreements.

## Troubleshooting

**`Could not load library libcudnn_ops_infer.so`** — CTranslate2 needs cuDNN 9 by soname. The Docker image provides it system-wide; on bare metal add pip's cuDNN to the library path:
```bash
export LD_LIBRARY_PATH=$(python -c 'import nvidia.cudnn.lib, os; print(os.path.dirname(nvidia.cudnn.lib.__file__))'):$LD_LIBRARY_PATH
```

**Opaque worker crashes during synthesis** — shared memory. The container sets `shm_size: 8gb`; Docker's 64 MB default is not enough.

**`no pool*.json in ...`** — `synthesize_pool.py` has not run for that pool, or `output_dir` differs between steps. `BON_TTS_OUTPUT_DIR` overrides the config; an explicit `--set output_dir=...` overrides both.

**Verifier score file not found** — score files are keyed by `verifier__metric__normalization`. Selecting with a different `--metric` or `--normalization` than you scored with looks like a missing file; the error message prints the exact `score_pool.py` command to run.

**A rank ensemble behaves like a single verifier** — check the warning from `select_candidates.py`. Two checkpoints from the same family (e.g. `distil-sm` + `distil-v3`, both Whisper) reproduce that family's bias instead of cancelling it.
