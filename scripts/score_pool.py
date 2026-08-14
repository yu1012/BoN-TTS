#!/usr/bin/env python3
"""Score every pooled candidate with one verifier and cache the result.

Verifier scores are the expensive part of Best-of-N and they do not depend on
the selection strategy, so they are computed once per (pool, verifier, metric)
and cached. ``scripts/select_candidates.py`` then combines cached score files — which is
what makes a rank ensemble free to evaluate once its members have been scored.

Recorded scoring time is used later to attribute the verifier's share of RTF.

Usage:
    python scripts/score_pool.py --config configs/librispeech_pc.yaml --verifier w2v2-base
    python scripts/score_pool.py --config ... --verifier distil-v3 --metric wer_cer
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

from bon_tts.audio import load_audio  # noqa: E402
from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.pool import load_pool_records  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir
from bon_tts.verifiers import Verifier, resolve_verifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--pool-name", type=str, default=None)
    parser.add_argument("--verifier", type=str, required=True, help="Alias or HF model id")
    parser.add_argument("--metric", type=str, default=None, choices=["wer", "cer", "wer_cer"])
    parser.add_argument("--composite-alpha", type=float, default=None)
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["simple", "f5_official"],
        help="'simple' reproduces the published numbers; see bon_tts.verifiers",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    verifier_cfg = config.get("verifier", {})
    metric = args.metric or verifier_cfg.get("metric", "wer_cer")
    composite_alpha = (
        args.composite_alpha
        if args.composite_alpha is not None
        else verifier_cfg.get("composite_alpha", 0.5)
    )
    normalization = args.normalization or verifier_cfg.get("normalization", "simple")

    pool_name = args.pool_name or config["pool"]["name"]
    pool_path = get_pool_dir(config, pool_name)
    records = load_pool_records(pool_path)
    logger.info("pool %s: %d utterances", pool_name, len(records))

    scores_dir = pool_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = scores_dir / f"{args.verifier}__{metric}__{normalization}.json"
    if out_path.exists() and not args.overwrite:
        logger.info("%s exists — pass --overwrite to recompute", out_path)
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    verifier = Verifier(
        model_name=args.verifier,
        device=device,
        metric=metric,
        composite_alpha=composite_alpha,
        normalization=normalization,
    )

    entries = []
    total_score_time = 0.0
    for record in tqdm(records, desc=f"Scoring [{args.verifier}]"):
        wavs = [load_audio(p, target_sr=16_000)[0].squeeze() for p in record["candidate_audio"]]
        t0 = time.time()
        scores = verifier.score(wavs, record["gen_text"])
        elapsed = time.time() - t0
        total_score_time += elapsed
        entries.append({"idx": record["idx"], "scores": scores, "score_time_s": elapsed})

    payload = {
        "verifier": args.verifier,
        "checkpoint": resolve_verifier(args.verifier),
        "family": verifier.family,
        "metric": metric,
        "composite_alpha": composite_alpha,
        "normalization": normalization,
        "pool_name": pool_name,
        "n_utterances": len(entries),
        "total_score_time_s": total_score_time,
        "per_utterance": entries,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("scored %d utterances in %.1f s", len(entries), total_score_time)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
