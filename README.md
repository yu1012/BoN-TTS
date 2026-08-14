# BoN-TTS

Best-of-N inference for zero-shot TTS, and the evaluation problem hiding inside it.

Reference implementation for **"Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignment"** (ICML 2026 Audio Workshop).

## What this is about

Best-of-N (BoN) is the standard inference-time fix for content errors in flow-matching TTS: synthesize N candidates, transcribe each with an ASR *verifier*, keep the one that best matches the target text. It reliably buys 10–30% relative WER.

The catch is how that gain is measured. Verifier quality is almost always reported under a single fixed *evaluator* ASR — and a verifier looks strong when the evaluator shares its ASR lineage. On LibriSpeech-PC test-clean:

- Verifier rankings **reverse** across Whisper, wav2vec 2.0 and HuBERT evaluators.
- Each evaluator's best verifier is the one from its **own ASR family**, *on identical audio*: under the wav2vec 2.0 evaluator the wav2vec 2.0 verifier recovers 26.1% of the oracle headroom against the Whisper verifier's 18.2%; under the Whisper evaluator the ordering flips, 26.0% against 7.9%.
- Representation similarity does not explain it: the most disagreeing evaluator pair shares **0.978 linear CKA**. The pattern instead looks like identity/lineage coupling — the speech analog of LLM-as-a-judge self-bias.

The effect has limits worth stating up front: it is carried by verifiers strong enough to rank candidates at all (a 166M verifier shows none of it, even under its own family), and the CKA evidence rests on 10 evaluator pairs. [`docs/RESULTS.md`](docs/RESULTS.md) is explicit about both.

So this repo ships two things: the BoN machinery, and the guardrails that keep a BoN result honest — **cross-family rank ensembles** for selection, and **cross-evaluator triangulation** for reporting.

## Headline results

LibriSpeech-PC test-clean cross-sentence, 1127 utterances, F5-TTS v1 base, 32 ODE steps. WER in %, under the official F5-TTS evaluator (`faster-whisper-large-v3`).

| Method (N=3) | WER | CER | RTF | rel. | p |
|---|---|---|---|---|---|
| F5-TTS (baseline) | 2.06 | 0.71 | 0.190 | — | — |
| BoN w2v2-base | 1.99 | 0.62 | 0.383 | −3.5% | 0.31 |
| BoN distil-sm | 2.04 | 0.59 | 0.388 | −1.0% | 0.93 |
| **BoN distil-v3** | **1.88** | **0.53** | 0.379 | **−8.7%** | **0.030** |
| BoN ens3 | 1.91 | 0.54 | 0.429 | −7.2% | 0.057 |

Scaling N, mean WER across all three evaluators — and the reason to prefer an ensemble:

| Method | N=3 | N=5 | N=10 |
|---|---|---|---|
| w2v2-base | 1.71 | 1.79 ↑ | 1.70 |
| distil-v3 | 1.69 | 1.68 | **1.61** |
| rank-avg | 1.73 | 1.65 | **1.61** |
| max-rank | 1.71 | 1.63 | **1.61** |

The single `w2v2-base` verifier *regresses* under the official evaluator at N=5 (2.20 vs. 2.06 baseline) while improving under its own family — exactly the failure the ensembles are for. **`rank-avg` at N=5 is the only configuration that reduces WER under all three evaluators simultaneously at p<0.05.** That simultaneity, not a single p-value, is the bar we recommend — and it is a strict one: `max-rank` at N=5 wins on both the official evaluator and the mean, yet misses it (p=.10 under `w2v2-lv60`).

Oracle headroom bounds what better verifiers could still win: at N=3 the per-utterance oracle reaches 1.42% WER against a 2.04% single-shot, and the best N=3 verifier recovers only 26% of that gap.

Full numbers, including per-evaluator breakdowns and recovery rates: [`docs/RESULTS.md`](docs/RESULTS.md).

## Install

Docker is the supported path — it pins CUDA, cuDNN and the ASR stack together.

```bash
git clone <repo-url> && cd bon-tts
make build
make test          # unit tests, CPU only
```

Mount points (all outside the image; the pool plus the per-configuration copies run to ~10 GB):

```bash
make shell OUTPUT_DIR=/data/bon-tts/outputs CACHE_DIR=/data/bon-tts/cache DATA_DIR=/data/corpora
```

<details>
<summary>Bare-metal install</summary>

```bash
# 1. torch/torchaudio for your CUDA version — see https://pytorch.org
pip install torch torchaudio
# 2. everything else
pip install -e .
```

`faster-whisper` (CTranslate2) needs cuDNN 9 on the library path. The Docker image gets it from the base image; on bare metal you may need `export LD_LIBRARY_PATH=$(python -c 'import nvidia.cudnn.lib,os;print(os.path.dirname(nvidia.cudnn.lib.__file__))'):$LD_LIBRARY_PATH`.

`requirements.lock.txt` holds the exact resolved versions from the container used for the runs.
</details>

## Quickstart

Synthesize one utterance with the recommended cross-family ensemble:

```bash
python scripts/infer.py \
    --ref-audio prompt.wav \
    --ref-text "Transcript of the prompt audio." \
    --gen-text "Text you want spoken." \
    --n 5 --strategy rank_avg --verifiers w2v2-base distil-v3 \
    --out out.wav
```

Or in Python:

```python
from bon_tts.f5_backend import F5CandidateGenerator, load_f5tts
from bon_tts.pipeline import bon_infer
from bon_tts.verifiers import load_verifiers

generator = F5CandidateGenerator(load_f5tts(device="cuda"))
verifiers = load_verifiers(["w2v2-base", "distil-v3"], device="cuda")

result = bon_infer(
    generator, verifiers,
    ref_file="prompt.wav", ref_text="Transcript of the prompt.",
    gen_text="Text you want spoken.",
    n_candidates=5, strategy="rank_avg",
)
print(result.best_idx, result.rtf)
```

Picking a configuration:

| Goal | Configuration |
|---|---|
| Robust across evaluators (recommended) | `--n 5 --strategy rank_avg --verifiers w2v2-base distil-v3` |
| Best WER under a Whisper-family evaluator | `--strategy single --verifiers distil-v3` |
| Best mean WER across evaluators | `--n 5 --strategy max_rank --verifiers w2v2-base distil-v3` |
| Lowest latency | `--n 3`; RTF is roughly linear in N |

Whatever you pick, **do not report a WER from one evaluator alone** — that is the whole point of the paper.

## Reproducing the paper

The pipeline separates expensive synthesis from cheap selection, so every configuration is compared on one shared candidate pool. Differences between configurations then cannot come from different audio, only from different choices.

```
prepare_pc_data  ->  synthesize_pool  ->  score_pool   ->  select   ->  evaluate  ->  report
   (once)             (GPU, hours)        (GPU, per     (CPU, fast)   (GPU, per    (CPU)
                                           verifier)                   evaluator)
```

```bash
# 0. Official protocol data (needs the F5-TTS .lst + LibriSpeech test-clean)
python scripts/prepare_pc_data.py --lst-file .../librispeech_pc_test_clean_cross_sentence.lst \
                                  --librispeech-root .../LibriSpeech/test-clean

make pool       # 1. N=10 candidate pool, 1127 x 10 clips
make score      # 2. score the pool with each verifier (cached)
make select     # 3. all strategies x N in {3,5,10} — seconds, no GPU
make evaluate   # 4. baseline + every run, under all three evaluators
make report     # 5. cross-evaluator table with paired permutation tests
```

Oracle headroom and recovery rates:

```bash
python scripts/evaluate_pool.py --evaluator fwhisper-lgv3     # transcribe all candidates
python scripts/analyze_oracle.py --runs select_single_distil-v3_n10 ...
```

Step-by-step walkthrough with runtimes and expected outputs: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Repository layout

```
bon_tts/
  selection.py       # single / rank_avg / max_rank — the paper's core method
  verifiers.py       # ASR verifiers (CTC + Whisper families)
  evaluators.py      # independent evaluators incl. the official F5-TTS one
  normalization.py   # F5-TTS official text normalization
  f5_backend.py      # F5-TTS candidate generation (ODE solver loop)
  pipeline.py        # online BoN inference
  quality.py         # SIM-o, UTMOS
  stats.py           # paired permutation, bootstrap CI, Holm-Bonferroni
  pool.py            # candidate-pool manifests and score caching
  audio.py cli.py config.py
scripts/             # the reproduction pipeline, in order
tests/               # unit tests for selection, normalization, statistics
docker/              # Dockerfile + compose
docs/                # RESULTS.md, REPRODUCE.md
```

## Provenance

This is a **reimplementation**. The original experiment scripts were lost; what survived is their output — per-candidate evaluator transcripts, per-candidate WERs, and the winner each configuration picked per utterance. The measurement pipeline here was rebuilt against those artifacts and validated by `scripts/validate_reproduction.py`:

- text normalization reproduces the stored normalized references **character-for-character**, 1127/1127 on all three evaluators;
- corpus WER recomputed from the stored transcripts matches the stored value **to <1e-9** — confirming corpus-level aggregation (total edits / total reference words), **not** the mean of per-utterance WERs;
- oracle selection reproduces the stored oracle WER to the same precision.

One known, intentional divergence: our normalizer takes its CJK punctuation set from `zhon.hanzi.punctuation` exactly as upstream F5-TTS does, while the original runs used a hand-written subset missing a few Unicode characters (notably U+2014 em dash). This affects 1 hypothesis in 1127 and moves corpus WER by 0.005 points (2.0431% → 2.0382%). We kept the upstream-matching normalizer rather than degrading it to force an exact match; the validation bounds the impact instead of ignoring it. Details in [`docs/REPRODUCE.md`](docs/REPRODUCE.md#validation-against-the-original-runs).

Two more things worth knowing:

- Verifier text normalization defaults to `simple` (lowercase + strip), which is what produced the published numbers. `f5_official` is cleaner for new work but shifts the results — see `bon_tts/verifiers.py`.
- Exploratory machinery that did not survive into the paper (conservative selection gaps, adaptive early exit, CTC-confidence and Whisper-attention verifier metrics, and the coarse-to-fine SAVR variant that failed its RTF budget) is deliberately **not** ported here.

## Citation

```bibtex
@inproceedings{bontts2026,
  title     = {Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignment},
  booktitle = {ICML 2026 Workshop on Audio},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The F5-TTS checkpoint and the ASR models carry their own licenses.
