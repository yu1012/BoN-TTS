#!/usr/bin/env python3
"""Synthesize the shared candidate pool: N candidates per utterance.

Every BoN configuration in the paper — each verifier, each rank ensemble, each
N, and the oracle — selects from *this same pool*. Synthesizing once and
selecting many times is not just cheaper: it means differences between
configurations cannot come from differences in the audio, only from which
candidate each strategy picked. Candidate 0 (seed 42) is the single-shot
baseline, so the baseline lives inside the pool too.

Writes ``{idx:04d}_c{n}.wav`` plus ``pool.json`` recording seeds and per-
candidate generation times (used later to reconstruct RTF).

Usage:
    python scripts/synthesize_pool.py --config configs/librispeech_pc.yaml --gpu 0
    python scripts/synthesize_pool.py --config ... --offset 0 --limit 200  # shard
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.audio import TTS_SAMPLE_RATE, save_audio  # noqa: E402
from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config, resolve_output_dir  # noqa: E402
from bon_tts.f5_backend import F5CandidateGenerator, candidate_seeds, load_f5tts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--pool-name", type=str, default=None, help="Override pool directory name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="First metadata index to process")
    parser.add_argument("--limit", type=int, default=None, help="Number of utterances to process")
    parser.add_argument("--overwrite", action="store_true", help="Re-synthesize existing candidates")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    tts_cfg = config["tts"]
    pool_cfg = config["pool"]
    n_candidates = pool_cfg["n_candidates"]
    pool_name = args.pool_name or pool_cfg["name"]

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    data_dir = Path(config["output_dir"]) / "data" / config["data"]["name"]
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run scripts/prepare_pc_data.py first")
    with open(meta_path) as handle:
        metadata = json.load(handle)

    end = None if args.limit is None else args.offset + args.limit
    shard = metadata[args.offset : end]
    logger.info(
        "synthesizing %d candidates for %d utterances (offset=%d) on %s",
        n_candidates, len(shard), args.offset, device,
    )

    pool_dir = resolve_output_dir(config, pool_name)
    audio_dir = pool_dir / "candidates"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading F5-TTS (%s)", tts_cfg["model_type"])
    tts = load_f5tts(
        model_type=tts_cfg["model_type"],
        ckpt_file=tts_cfg.get("ckpt_file", ""),
        vocab_file=tts_cfg.get("vocab_file", ""),
        ode_method=tts_cfg.get("ode_method", "euler"),
        device=device,
    )
    generator = F5CandidateGenerator(
        tts_api=tts,
        total_steps=tts_cfg["steps"],
        cfg_strength=tts_cfg["cfg_strength"],
        sway_sampling_coef=tts_cfg["sway_sampling_coef"],
        target_rms=tts_cfg["target_rms"],
        speed=tts_cfg.get("speed", 1.0),
    )
    seeds = candidate_seeds(
        n_candidates,
        base_seed=tts_cfg["seed"],
        seed_offset=pool_cfg["seed_offset"],
    )
    logger.info("candidate seeds: %s", seeds)

    records = []
    for item in tqdm(shard, desc="Synthesizing pool"):
        idx = item["idx"]
        paths = [audio_dir / f"{idx:04d}_c{n}.wav" for n in range(n_candidates)]

        if not args.overwrite and all(p.exists() for p in paths):
            records.append({"idx": idx, "seeds": seeds, "skipped": True})
            continue

        t0 = time.time()
        candidates = generator.generate(
            ref_file=item["ref_audio"],
            ref_text=item["ref_text"],
            gen_text=item["gen_text"],
            seeds=seeds,
        )
        wall_time = time.time() - t0

        durations = []
        for candidate, path in zip(candidates, paths, strict=True):
            save_audio(candidate.wav, path, sr=TTS_SAMPLE_RATE)
            durations.append(len(candidate.wav) / TTS_SAMPLE_RATE)

        records.append(
            {
                "idx": idx,
                "seeds": seeds,
                "gen_text": item["gen_text"],
                "ref_audio": item["ref_audio"],
                "candidate_audio": [str(p) for p in paths],
                "candidate_gen_times_s": [c.gen_time_s for c in candidates],
                "candidate_durations_s": durations,
                "pool_wall_time_s": wall_time,
                "skipped": False,
            }
        )

    suffix = f"_{args.offset:05d}_{args.offset + len(shard):05d}" if args.limit is not None else ""
    pool_manifest = pool_dir / f"pool{suffix}.json"
    with open(pool_manifest, "w") as handle:
        json.dump(
            {
                "pool_name": pool_name,
                "n_candidates": n_candidates,
                "seeds": seeds,
                "tts": tts_cfg,
                "data": config["data"]["name"],
                "records": records,
            },
            handle,
            indent=2,
        )

    synthesized = sum(1 for r in records if not r["skipped"])
    logger.info("done: %d synthesized, %d skipped", synthesized, len(records) - synthesized)
    logger.info("manifest: %s", pool_manifest)


if __name__ == "__main__":
    main()
