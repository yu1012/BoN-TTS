#!/usr/bin/env python3
"""Build the cross-evaluator results table with paired significance tests.

Produces the paper's main table shape: one row per configuration, one WER column
per evaluator, a mean across evaluators, and a paired permutation p-value
against the single-shot baseline.

Reading the table:

* Compare each row *across* columns. A configuration that wins under one
  evaluator and loses under another has not been shown to help — that spread is
  the family-alignment confound, and it is why a single-evaluator number is not
  sufficient evidence.
* ``sig`` counts how many evaluators the row reaches p<0.05 under. Reaching it
  under all of them simultaneously is the bar this repo recommends, and it is a
  much harder bar than any single p-value.
* Holm–Bonferroni is applied *within* each evaluator column, over the
  configurations compared there. It does not correct across columns; the
  simultaneity criterion is what handles that.

CPU-only — it reads the JSON written by ``scripts/evaluate.py``.

Usage:
    python scripts/report.py --config ... \\
        --runs select_single_distil-v3_n10 select_max_rank_w2v2-base+distil-v3_n5
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.pool import BASELINE_RUN_NAME  # noqa: E402
from bon_tts.stats import bootstrap_mean_ci, holm_bonferroni, paired_permutation_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIGNIFICANCE_ALPHA = 0.05


def load_eval(config: dict, run_name: str, evaluator: str) -> dict | None:
    path = Path(config["output_dir"]) / run_name / f"eval_{evaluator}.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def format_p(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    parser.add_argument("--baseline-run", type=str, default=BASELINE_RUN_NAME)
    parser.add_argument("--evaluators", type=str, nargs="+", default=None)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", action="store_true", help="Also report bootstrap 95%% CIs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    evaluators = args.evaluators or list(config["evaluation"]["evaluators"])
    all_runs = [args.baseline_run] + [r for r in args.runs if r != args.baseline_run]

    # Collect corpus WER and per-utterance WER per (run, evaluator).
    corpus: dict[tuple[str, str], float] = {}
    per_sample: dict[tuple[str, str], list[float]] = {}
    extras: dict[str, dict] = {}
    missing: list[tuple[str, str]] = []

    for run_name in all_runs:
        for evaluator in evaluators:
            payload = load_eval(config, run_name, evaluator)
            if payload is None:
                missing.append((run_name, evaluator))
                continue
            corpus[(run_name, evaluator)] = payload["corpus_wer"]
            by_idx = {s["idx"]: s["wer"] for s in payload["per_sample"]}
            per_sample[(run_name, evaluator)] = [by_idx[i] for i in sorted(by_idx)]
            if "sim_mean" in payload or "utmos_mean" in payload:
                extras.setdefault(run_name, {}).update(
                    {k: payload[k] for k in ("sim_mean", "utmos_mean") if k in payload}
                )

    for run_name, evaluator in missing:
        logger.warning(
            "no eval_%s.json for run %s — that cell is reported as '--'", evaluator, run_name
        )

    # Paired tests against the baseline, then Holm within each evaluator column.
    p_values: dict[tuple[str, str], float] = {}
    for evaluator in evaluators:
        base = per_sample.get((args.baseline_run, evaluator))
        if base is None:
            continue
        column_runs, column_ps = [], []
        for run_name in all_runs[1:]:
            treatment = per_sample.get((run_name, evaluator))
            if treatment is None or len(treatment) != len(base):
                continue
            result = paired_permutation_test(
                base, treatment, n_permutations=args.permutations, seed=args.seed
            )
            p_values[(run_name, evaluator)] = result.p_value
            column_runs.append(run_name)
            column_ps.append(result.p_value)
        if column_ps:
            rejections = holm_bonferroni(column_ps, SIGNIFICANCE_ALPHA)
            for run_name, rejected in zip(column_runs, rejections, strict=True):
                p_values[(run_name, evaluator, "holm")] = rejected

    # Render
    name_width = max(len(r) for r in all_runs) + 2
    header = f"{'configuration':<{name_width}}" + "".join(f"{e:>16}" for e in evaluators)
    header += f"{'mean':>8}{'sig':>6}"
    print("\nCorpus WER (%) by evaluator; p vs baseline, paired permutation "
          f"({args.permutations} perms). * = survives Holm within column.")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for run_name in all_runs:
        cells = []
        values = []
        n_significant = 0
        for evaluator in evaluators:
            key = (run_name, evaluator)
            if key not in corpus:
                cells.append(f"{'--':>16}")
                continue
            value = corpus[key] * 100
            values.append(value)
            if run_name == args.baseline_run:
                cells.append(f"{value:>10.2f}      ")
            else:
                p = p_values.get(key)
                holm = p_values.get((run_name, evaluator, "holm"), False)
                mark = "*" if holm else (" " if p is None or p >= SIGNIFICANCE_ALPHA else "+")
                if p is not None and p < SIGNIFICANCE_ALPHA:
                    n_significant += 1
                cells.append(f"{value:>10.2f}{mark}{format_p(p) if p else '':>5}")
        mean = f"{sum(values)/len(values):>8.2f}" if values else f"{'--':>8}"
        sig = "" if run_name == args.baseline_run else f"{n_significant}/{len(evaluators)}"
        print(f"{run_name:<{name_width}}" + "".join(cells) + mean + f"{sig:>6}")

    print("=" * len(header))

    if extras:
        print("\nQuality metrics (where computed):")
        for run_name, values in extras.items():
            parts = [f"{k.replace('_mean','')} {v:.4f}" for k, v in values.items()]
            print(f"  {run_name:<{name_width}} " + "  ".join(parts))

    if args.bootstrap:
        print("\nBootstrap 95% CI of mean per-utterance WER (%):")
        for run_name in all_runs:
            for evaluator in evaluators:
                values = per_sample.get((run_name, evaluator))
                if not values:
                    continue
                ci = bootstrap_mean_ci(values, confidence=0.95, seed=args.seed)
                print(
                    f"  {run_name:<{name_width}} {evaluator:<16} "
                    f"{ci.mean*100:5.2f}  [{ci.ci_low*100:5.2f}, {ci.ci_high*100:5.2f}]"
                )

    if args.output:
        payload = {
            "evaluators": evaluators,
            "baseline_run": args.baseline_run,
            "permutations": args.permutations,
            "rows": [
                {
                    "run_name": run_name,
                    "corpus_wer": {e: corpus.get((run_name, e)) for e in evaluators},
                    "p_value": {e: p_values.get((run_name, e)) for e in evaluators},
                    "holm_significant": {
                        e: p_values.get((run_name, e, "holm")) for e in evaluators
                    },
                    **extras.get(run_name, {}),
                }
                for run_name in all_runs
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
