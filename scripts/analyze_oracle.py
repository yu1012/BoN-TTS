#!/usr/bin/env python3
"""Oracle headroom and per-evaluator recovery rates.

Reads the cached per-candidate evaluator WERs from ``scripts/evaluate_pool.py``
and, for each selection run, reports what fraction of the available oracle
headroom that run actually captured:

    recovery = (single_shot - selected) / (single_shot - oracle)

Recovery is the quantity that exposes the family-alignment confound. Raw WER
conflates "good verifier" with "verifier judged by its own family"; recovery
normalizes by the headroom each evaluator actually offers, which makes the
same-family versus cross-family gap directly comparable across evaluators.

Because recovery is computed from *evaluator* transcripts of the very candidates
each run selected, no additional GPU work is needed here.

Usage:
    python scripts/analyze_oracle.py --config ... --runs select_single_distil-v3_n10 ...
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.evaluators import corpus_wer  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir  # noqa: E402
from bon_tts.selection import recovery_rate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_candidate_evals(pool_dir: Path, evaluator: str) -> dict:
    """Load the newest per-candidate eval cache for ``evaluator``."""
    matches = sorted((pool_dir / "candidate_evals").glob(f"eval_{evaluator}_n*.json"))
    if not matches:
        raise FileNotFoundError(
            f"no candidate evals for {evaluator} in {pool_dir / 'candidate_evals'} — run:\n"
            f"  python scripts/evaluate_pool.py --evaluator {evaluator}"
        )
    # Highest N carries the most information; filenames end in _n<N>.json.
    best = max(matches, key=lambda p: int(p.stem.rsplit("_n", 1)[1]))
    with open(best) as handle:
        return json.load(handle)


def load_selection(config: dict, run_name: str) -> dict:
    path = Path(config["output_dir"]) / run_name / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/select_candidates.py first")
    with open(path) as handle:
        return json.load(handle)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--pool-name", type=str, default=None)
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Selection run names")
    parser.add_argument("--evaluators", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Write the report as JSON")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    pool_name = args.pool_name or config["pool"]["name"]
    pool_path = get_pool_dir(config, pool_name)
    evaluators = args.evaluators or list(config["evaluation"]["evaluators"])

    report: dict = {"pool_name": pool_name, "evaluators": {}}

    for evaluator in evaluators:
        pool_eval = load_candidate_evals(pool_path, evaluator)
        by_idx = {e["idx"]: e for e in pool_eval["per_utterance"]}

        single_shot = pool_eval["single_shot_corpus_wer"]
        oracle = pool_eval["oracle_corpus_wer"]
        entry = {
            "single_shot_wer": single_shot,
            "oracle_wer": oracle,
            "headroom": single_shot - oracle,
            "n_candidates": pool_eval["n_candidates"],
            "runs": {},
        }

        for run_name in args.runs:
            selection = load_selection(config, run_name)
            indices, hyps, refs = [], [], []
            for item in selection["selections"]:
                idx, chosen = item["idx"], item["best_idx"]
                if idx not in by_idx:
                    continue
                cached = by_idx[idx]
                if chosen >= len(cached["candidate_hyps"]):
                    # The run selected a candidate this evaluator never scored.
                    continue
                indices.append(idx)
                hyps.append(cached["candidate_hyps"][chosen])
                refs.append(cached["reference"])

            if len(indices) < len(selection["selections"]):
                logger.warning(
                    "%s under %s: %d/%d utterances usable (missing candidate evals)",
                    run_name, evaluator, len(indices), len(selection["selections"]),
                )
            if not indices:
                continue

            selected_wer = corpus_wer(refs, hyps)
            entry["runs"][run_name] = {
                "wer": selected_wer,
                "relative_change_pct": 100 * (selected_wer - single_shot) / single_shot,
                "recovery_pct": recovery_rate(single_shot, selected_wer, oracle),
                "strategy": selection["strategy"],
                "verifiers": selection["verifiers"],
                "n": selection["n"],
                "n_utterances": len(indices),
            }

        report["evaluators"][evaluator] = entry

    # Text report
    for evaluator, entry in report["evaluators"].items():
        print(f"\n=== {evaluator} (N={entry['n_candidates']}) ===")
        print(f"  single-shot WER {entry['single_shot_wer']*100:6.2f}%")
        print(f"  oracle WER      {entry['oracle_wer']*100:6.2f}%   headroom {entry['headroom']*100:5.2f}pp")
        print(f"  {'run':<44} {'WER':>7} {'rel':>8} {'recovery':>9}")
        for run_name, run in sorted(entry["runs"].items(), key=lambda kv: kv[1]["wer"]):
            print(
                f"  {run_name:<44} {run['wer']*100:6.2f}% "
                f"{run['relative_change_pct']:+7.1f}% {run['recovery_pct']:8.1f}%"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
