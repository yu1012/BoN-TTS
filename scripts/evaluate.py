#!/usr/bin/env python3
"""Evaluate a selection run under one independent ASR evaluator.

Run this once per evaluator and compare the results: a WER improvement that
appears under only one ASR family is the confound this repo is about, not a
finding. ``--evaluator fwhisper-lgv3`` is the official F5-TTS evaluator and the
one to quote; the others exist to check the conclusion survives.

Writes ``eval_{evaluator}.json`` next to the run's ``selection.json``, holding
corpus WER/CER, per-utterance WER (for ``scripts/analyze_significance.py``) and
optional SIM-o / UTMOS.

Usage:
    # baseline: candidate 0 of the pool, i.e. plain single-shot F5-TTS
    python scripts/evaluate.py --config ... --baseline --evaluator fwhisper-lgv3

    # a selection run, under all three evaluators
    RUN=select_max_rank_w2v2-base+distil-v3_n5
    for e in fwhisper-lgv3 w2v2-lv60 hubert-lg; do
        python scripts/evaluate.py --config ... --run-name $RUN --evaluator $e
    done
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.evaluators import (  # noqa: E402
    EVALUATOR_SPECS,
    Evaluator,
    corpus_cer,
    corpus_wer,
    per_sample_wer,
)
from bon_tts.pool import BASELINE_RUN_NAME, load_pool_records  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_run(config: dict, run_name: str) -> tuple[Path, list[dict]]:
    """Load a selection run's chosen audio."""
    run_dir = Path(config["output_dir"]) / run_name
    selection_path = run_dir / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} not found — run scripts/select_candidates.py first")
    with open(selection_path) as handle:
        payload = json.load(handle)
    return run_dir, payload["selections"]


def load_baseline(config: dict, pool_name: str) -> tuple[Path, list[dict]]:
    """Treat candidate 0 of the pool as the single-shot baseline system."""
    records = load_pool_records(get_pool_dir(config, pool_name))
    run_dir = Path(config["output_dir"]) / BASELINE_RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "idx": record["idx"],
            "gen_audio": record["candidate_audio"][0],
            "gen_text": record["gen_text"],
            "ref_audio": record["ref_audio"],
        }
        for record in records
    ]
    return run_dir, items


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--run-name", type=str, default=None, help="Selection run to evaluate")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate pool candidate 0 (single-shot F5-TTS) instead of a selection run",
    )
    parser.add_argument("--pool-name", type=str, default=None)
    parser.add_argument(
        "--evaluator", type=str, default="fwhisper-lgv3",
        help=f"One of {sorted(EVALUATOR_SPECS)} or 'backend:checkpoint'",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="HF-backend batch size")
    parser.add_argument("--sim", action="store_true", help="Also compute SIM-o")
    parser.add_argument("--utmos", action="store_true", help="Also compute UTMOS")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    if bool(args.run_name) == bool(args.baseline):
        parser.error("pass exactly one of --run-name or --baseline")

    config = apply_overrides(load_config(args.config), args.overrides)
    pool_name = args.pool_name or config["pool"]["name"]

    if args.baseline:
        run_dir, items = load_baseline(config, pool_name)
        run_name = BASELINE_RUN_NAME
    else:
        run_dir, items = load_run(config, args.run_name)
        run_name = args.run_name

    out_path = run_dir / f"eval_{args.evaluator}.json"
    if out_path.exists() and not args.overwrite:
        logger.info("%s exists — pass --overwrite to recompute", out_path)
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    gen_paths = [item["gen_audio"] for item in items]
    references = [item["gen_text"] for item in items]

    logger.info("evaluating %s (%d utterances) with %s", run_name, len(items), args.evaluator)
    evaluator = Evaluator(args.evaluator, device=device)
    hypotheses = evaluator.transcribe(gen_paths, batch_size=args.batch_size)

    results = {
        "run_name": run_name,
        "evaluator": args.evaluator,
        "evaluator_family": evaluator.family,
        "checkpoint": evaluator.checkpoint,
        "normalization": "f5_official",
        "n_utterances": len(items),
        "corpus_wer": corpus_wer(references, hypotheses),
        "corpus_cer": corpus_cer(references, hypotheses),
    }
    sample_wers = per_sample_wer(references, hypotheses)
    results["mean_per_sample_wer"] = float(sum(sample_wers) / len(sample_wers))
    results["per_sample"] = [
        {"idx": item["idx"], "reference": ref, "hypothesis": hyp, "wer": w}
        for item, ref, hyp, w in zip(items, references, hypotheses, sample_wers, strict=True)
    ]

    if args.sim:
        from bon_tts.quality import SpeakerSimilarity

        sim = SpeakerSimilarity(device=device).score(
            gen_paths, [item["ref_audio"] for item in items]
        )
        results["sim_mean"] = sim["sim_mean"]
        results["sim_details"] = sim

    if args.utmos:
        from bon_tts.quality import Utmos

        utmos = Utmos(device=device).score(gen_paths)
        results["utmos_mean"] = utmos["utmos_mean"]
        results["utmos_details"] = utmos

    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)

    logger.info("%-28s corpus WER %.4f  CER %.4f", run_name, results["corpus_wer"], results["corpus_cer"])
    if args.sim:
        logger.info("%-28s SIM-o      %.4f", run_name, results["sim_mean"])
    if args.utmos:
        logger.info("%-28s UTMOS      %.4f", run_name, results["utmos_mean"])
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
