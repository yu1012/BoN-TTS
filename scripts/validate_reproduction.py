#!/usr/bin/env python3
"""Check this implementation against reference artifacts from the original runs.

The scripts that produced the published tables were lost; this repo is a
reimplementation. What survived is the output side of those runs — per-candidate
evaluator transcripts, per-candidate WERs, and the per-utterance winner each
configuration picked. That is enough to pin down the parts of the pipeline where
a silent difference would change every number downstream:

A. **Text normalization** — our :func:`bon_tts.normalization.normalize_en` must
   reproduce the reference normalized references character-for-character.
B. **WER aggregation** — corpus WER recomputed from the reference transcripts
   must match the reference corpus WER to floating-point precision, confirming
   corpus-level (total edits / total words) rather than mean-of-per-utterance.
C. **Oracle selection** — :func:`bon_tts.selection.select_oracle` over the
   reference per-candidate WERs must reproduce the reference oracle WER.
D. **Selection strategies** *(optional)* — with freshly computed verifier scores,
   our selections must match the winners the original runs recorded.
E. **Normalizer parity** — bounds the one known, intentional divergence: our
   normalizer takes its CJK punctuation set from ``zhon`` (as upstream F5-TTS
   does) and so strips a few Unicode characters the original runs left in.

A-C and E need no GPU and validate the measurement pipeline. D needs verifier
scores from ``scripts/score_pool.py`` and validates the selection logic end to end.

Usage:
    # A-C, E
    python scripts/validate_reproduction.py --reference-dir /path/to/pc_oracle_candidates

    # add D
    python scripts/validate_reproduction.py --reference-dir ... \\
        --config configs/librispeech_pc.yaml \\
        --compare-selection /path/to/pc_bon_n5_max_rank/synthesis_results_0000_1127.json \\
        --strategy max_rank --verifiers w2v2-base distil-v3 --n 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jiwer import wer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.evaluators import corpus_wer  # noqa: E402
from bon_tts.normalization import normalize_en  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir  # noqa: E402
from bon_tts.pool import verifier_score_path  # noqa: E402
from bon_tts.selection import select, select_oracle  # noqa: E402

# The reference artifacts store WERs as float64; agreement should be exact to
# within accumulated summation error, not merely close.
FLOAT_TOLERANCE = 1e-9

# Ceiling on the corpus-WER shift our stricter normalizer may cause (check E).
# WER is a fraction here, so 0.0005 == 0.05 percentage points. Observed on the
# reference artifacts: 0.00005 (= 0.005 pp; one em dash in 1127 Whisper
# hypotheses, 2.0431% -> 2.0382%), so this leaves an order of magnitude of slack
# while still failing on any change that moves WER by a tenth of a point.
DEFAULT_MAX_WER_DELTA = 0.0005


class Report:
    """Accumulates pass/fail checks and exits non-zero if any failed."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def skip(self, name: str, reason: str) -> None:
        print(f"  [SKIP] {name} — {reason}")

    def summary(self) -> int:
        failed = [name for name, passed, _ in self.results if not passed]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        if failed:
            print("failed: " + ", ".join(failed))
            return 1
        return 0


def find_reference_evals(reference_dir: Path) -> list[Path]:
    """Reference per-candidate eval files (the ``_f5exact`` normalization)."""
    return sorted(reference_dir.glob("eval_*_f5exact.json"))


def validate_normalization(
    report: Report,
    payloads: list[tuple[Path, dict]],
    metadata: list[dict] | None,
) -> None:
    """Check A: our normalizer reproduces the reference normalized references."""
    if metadata is None:
        report.skip("A. normalization", "--metadata not provided")
        return

    raw_by_idx = {item["idx"]: item["gen_text"] for item in metadata}
    for path, payload in payloads:
        mismatches = []
        compared = 0
        for entry in payload["per_sample"]:
            raw = raw_by_idx.get(entry["idx"])
            if raw is None:
                continue
            compared += 1
            if normalize_en(raw) != entry["ref"]:
                mismatches.append((entry["idx"], normalize_en(raw), entry["ref"]))

        detail = f"{compared - len(mismatches)}/{compared} references match exactly"
        report.check(f"A. normalization [{path.name}]", not mismatches and compared > 0, detail)
        for idx, ours, theirs in mismatches[:3]:
            print(f"         idx {idx}\n           ours:  {ours!r}\n           ref:   {theirs!r}")


def _hyps_key(entry: dict) -> str:
    """Artifact field name for candidate transcripts (it varies by vintage)."""
    return "candidate_hyps" if "candidate_hyps" in entry else "cand_hyps"


def _wers_key(entry: dict) -> str:
    return "candidate_wers" if "candidate_wers" in entry else "cand_wers"


def validate_aggregation(report: Report, payloads: list[tuple[Path, dict]]) -> None:
    """Check B: the WER *aggregation* reproduces the reference value.

    Deliberately scored on the artifact strings as stored — they are already
    normalized, and re-normalizing here would fold normalization differences
    into a check about aggregation. Check E covers the normalizer separately.
    """
    for path, payload in payloads:
        entries = payload["per_sample"]
        refs = [e["ref"] for e in entries]
        hyps = [e[_hyps_key(e)][0] for e in entries]
        ours = float(wer(refs, hyps))
        theirs = payload["random_wer"]
        report.check(
            f"B. corpus WER aggregation [{path.name}]",
            abs(ours - theirs) < FLOAT_TOLERANCE,
            f"ours {ours:.12f} vs reference {theirs:.12f}",
        )


def validate_oracle(report: Report, payloads: list[tuple[Path, dict]]) -> None:
    """Check C: oracle selection reproduces the reference oracle WER."""
    for path, payload in payloads:
        entries = payload["per_sample"]
        refs, hyps = [], []
        for entry in entries:
            chosen = select_oracle(entry[_wers_key(entry)])
            refs.append(entry["ref"])
            hyps.append(entry[_hyps_key(entry)][chosen])
        ours = float(wer(refs, hyps))
        theirs = payload["oracle_wer"]
        report.check(
            f"C. oracle selection [{path.name}]",
            abs(ours - theirs) < FLOAT_TOLERANCE,
            f"ours {ours:.12f} vs reference {theirs:.12f}",
        )


def validate_normalizer_parity(
    report: Report, payloads: list[tuple[Path, dict]], max_wer_delta: float
) -> None:
    """Check E: quantify where our normalizer differs from the reference one.

    The artifact strings were already normalized by the original pipeline, so
    ``normalize_en`` should be a no-op over them. Where it is not, our
    normalizer is strictly more aggressive — the original one missed a handful
    of Unicode punctuation characters that ``zhon.hanzi.punctuation`` covers
    (notably U+2014 em dash in Whisper output).

    This is a real, intentional divergence: our normalizer matches upstream
    F5-TTS, the original runs' did not quite. The check bounds its impact rather
    than hiding it, so a future change that shifts WER materially still fails.
    """
    for path, payload in payloads:
        entries = payload["per_sample"]
        refs = [e["ref"] for e in entries]
        hyps = [e[_hyps_key(e)][0] for e in entries]

        altered = [
            (e["idx"], h) for e, h in zip(entries, hyps, strict=True) if normalize_en(h) != h
        ]
        as_stored = float(wer(refs, hyps))
        renormalized = corpus_wer(refs, hyps)
        delta = abs(renormalized - as_stored)

        report.check(
            f"E. normalizer parity [{path.name}]",
            delta <= max_wer_delta,
            f"{len(altered)}/{len(entries)} hypotheses altered, "
            f"corpus WER {as_stored:.6f} -> {renormalized:.6f} (delta {delta:.6f})",
        )
        for idx, hyp in altered[:3]:
            differing = sorted({c for c in hyp if c not in normalize_en(hyp)})
            print(f"         idx {idx}: extra punctuation {differing}")


def validate_selection(
    report: Report,
    config: dict,
    reference_selection: Path,
    strategy: str,
    verifiers: list[str],
    n: int,
    metric: str,
    normalization: str,
) -> None:
    """Check D: our selection matches the winners the original run recorded."""
    pool_path = get_pool_dir(config)
    score_paths = [
        verifier_score_path(pool_path, v, metric, normalization) for v in verifiers
    ]
    missing = [p for p in score_paths if not p.exists()]
    if missing:
        report.skip(
            f"D. selection [{strategy}/{'+'.join(verifiers)}/n={n}]",
            f"missing verifier scores: {', '.join(p.name for p in missing)}",
        )
        return

    score_maps = []
    for path in score_paths:
        with open(path) as handle:
            payload = json.load(handle)
        score_maps.append({e["idx"]: e["scores"] for e in payload["per_utterance"]})

    with open(reference_selection) as handle:
        reference = json.load(handle)
    reference_by_idx = {item["idx"]: item["best_idx"] for item in reference}

    agree = compared = 0
    disagreements = []
    for idx, expected in sorted(reference_by_idx.items()):
        if any(idx not in m for m in score_maps):
            continue
        matrix = [m[idx][:n] for m in score_maps]
        ours = select(matrix, strategy=strategy)
        compared += 1
        if ours == expected:
            agree += 1
        elif len(disagreements) < 5:
            disagreements.append((idx, ours, expected, matrix))

    if compared == 0:
        report.skip(
            f"D. selection [{strategy}/{'+'.join(verifiers)}/n={n}]",
            "no overlapping utterances between scores and reference",
        )
        return

    rate = agree / compared
    report.check(
        f"D. selection [{strategy}/{'+'.join(verifiers)}/n={n}]",
        rate == 1.0,
        f"{agree}/{compared} winners match ({100*rate:.2f}%)",
    )
    for idx, ours, expected, matrix in disagreements:
        print(f"         idx {idx}: ours {ours}, reference {expected}")
        for name, row in zip(verifiers, matrix, strict=True):
            print(f"           {name:<12} " + " ".join(f"{s:.4f}" for s in row))


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Directory holding eval_*_f5exact.json reference artifacts")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="metadata.json with raw gen_text, enabling check A")
    parser.add_argument("--config", type=Path, default=None, help="Required for check D")
    parser.add_argument("--compare-selection", type=Path, default=None,
                        help="Reference selection JSON (list of {idx, best_idx}) for check D")
    parser.add_argument("--strategy", type=str, default="single")
    parser.add_argument("--verifiers", type=str, nargs="+", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--metric", type=str, default="wer_cer")
    parser.add_argument("--normalization", type=str, default="simple")
    parser.add_argument(
        "--max-wer-delta",
        type=float,
        default=DEFAULT_MAX_WER_DELTA,
        help="Check E ceiling on the corpus-WER shift from our stricter normalizer",
    )
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    eval_paths = find_reference_evals(args.reference_dir)
    if not eval_paths:
        raise FileNotFoundError(f"no eval_*_f5exact.json in {args.reference_dir}")

    payloads = []
    for path in eval_paths:
        with open(path) as handle:
            payloads.append((path, json.load(handle)))

    metadata = None
    if args.metadata:
        with open(args.metadata) as handle:
            metadata = json.load(handle)

    print(f"validating against {len(payloads)} reference artifact(s) in {args.reference_dir}\n")
    report = Report()

    print("Check A — text normalization")
    validate_normalization(report, payloads, metadata)
    print("\nCheck B — WER aggregation")
    validate_aggregation(report, payloads)
    print("\nCheck C — oracle selection")
    validate_oracle(report, payloads)

    print("\nCheck D — selection strategies")
    if args.compare_selection:
        if not (args.config and args.verifiers and args.n):
            parser.error("--compare-selection requires --config, --verifiers and --n")
        config = apply_overrides(load_config(args.config), args.overrides)
        validate_selection(
            report, config, args.compare_selection, args.strategy,
            args.verifiers, args.n, args.metric, args.normalization,
        )
    else:
        report.skip("D. selection", "--compare-selection not provided")

    print("\nCheck E — normalizer parity")
    validate_normalizer_parity(report, payloads, args.max_wer_delta)

    sys.exit(report.summary())


if __name__ == "__main__":
    main()
