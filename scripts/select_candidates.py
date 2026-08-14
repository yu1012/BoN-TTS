#!/usr/bin/env python3
"""Apply a Best-of-N selection strategy to cached verifier scores.

Reads one cached score file per verifier, truncates to the first N candidates,
and picks a winner per utterance. Because scoring already happened, this is
CPU-only and takes seconds — so a rank ensemble over already-scored verifiers
costs nothing extra, and sweeping N is free.

Truncating to the *first* N candidates keeps the nesting property: the N=3 pool
is a subset of the N=5 pool, so a change between them reflects the extra
candidates and not a different random draw.

Usage:
    # single-verifier BoN
    python scripts/select_candidates.py --config ... --strategy single --verifiers distil-v3 --n 10

    # cross-family rank ensemble (the paper's proposed mitigation)
    python scripts/select_candidates.py --config ... --strategy rank_avg --verifiers w2v2-base distil-v3 --n 5
    python scripts/select_candidates.py --config ... --strategy max_rank --verifiers w2v2-base distil-v3 --n 5
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config  # noqa: E402
from bon_tts.pool import load_pool_records, load_verifier_scores  # noqa: E402
from bon_tts.pool import pool_dir as get_pool_dir  # noqa: E402
from bon_tts.selection import STRATEGIES, select  # noqa: E402
from bon_tts.verifiers import VERIFIER_ALIASES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Family per verifier checkpoint, used to warn when an "ensemble" is single-family.
VERIFIER_FAMILY = {
    "w2v2-base": "wav2vec2",
    "distil-sm": "whisper",
    "distil-v3": "whisper",
}


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--pool-name", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="single", choices=list(STRATEGIES))
    parser.add_argument(
        "--verifiers",
        type=str,
        nargs="+",
        required=True,
        help=f"Verifier aliases (known: {', '.join(sorted(VERIFIER_ALIASES))})",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of candidates to select from")
    parser.add_argument("--metric", type=str, default=None, choices=["wer", "cer", "wer_cer"])
    parser.add_argument("--normalization", type=str, default=None, choices=["simple", "f5_official"])
    parser.add_argument("--run-name", type=str, default=None, help="Output directory name")
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink selected audio instead of copying (saves disk, breaks if the pool moves)",
    )
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    verifier_cfg = config.get("verifier", {})
    metric = args.metric or verifier_cfg.get("metric", "wer_cer")
    normalization = args.normalization or verifier_cfg.get("normalization", "simple")

    if args.strategy == "single" and len(args.verifiers) != 1:
        parser.error("--strategy single takes exactly one --verifiers entry")
    if args.strategy != "single" and len(args.verifiers) < 2:
        parser.error(f"--strategy {args.strategy} needs at least two --verifiers entries")

    families = {VERIFIER_FAMILY.get(v, v) for v in args.verifiers}
    if args.strategy != "single" and len(families) == 1:
        logger.warning(
            "all verifiers belong to the %s family — a same-family ensemble reproduces "
            "that family's bias rather than cancelling it; the paper's ensembles are cross-family",
            next(iter(families)),
        )

    pool_name = args.pool_name or config["pool"]["name"]
    pool_path = get_pool_dir(config, pool_name)
    records = {r["idx"]: r for r in load_pool_records(pool_path)}

    score_files = [
        load_verifier_scores(pool_path, v, metric, normalization) for v in args.verifiers
    ]
    n_pool = config["pool"]["n_candidates"]
    if args.n > n_pool:
        parser.error(f"--n {args.n} exceeds the pool size ({n_pool} candidates)")

    scores_by_verifier = [
        {entry["idx"]: entry for entry in payload["per_utterance"]} for payload in score_files
    ]
    common_indices = sorted(set.intersection(*(set(s) for s in scores_by_verifier)) & set(records))
    logger.info(
        "selecting with strategy=%s verifiers=%s N=%d over %d utterances",
        args.strategy, args.verifiers, args.n, len(common_indices),
    )

    run_name = args.run_name or (
        f"select_{args.strategy}_{'+'.join(args.verifiers)}_n{args.n}"
    )
    run_dir = Path(config["output_dir"]) / run_name
    audio_dir = run_dir / "generated"
    audio_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    for idx in common_indices:
        record = records[idx]
        matrix = [entry_map[idx]["scores"][: args.n] for entry_map in scores_by_verifier]
        chosen = select(matrix, strategy=args.strategy)

        source = Path(record["candidate_audio"][chosen])
        destination = audio_dir / f"{idx:04d}.wav"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if args.link:
            destination.symlink_to(source.resolve())
        else:
            shutil.copyfile(source, destination)

        # RTF attribution: synthesizing the N candidates plus this verifier's
        # share of the cached scoring time, over the selected clip's duration.
        # The verifier share is prorated from the pool's N, which assumes
        # scoring cost is linear in candidate count.
        synth_time = sum(record["candidate_gen_times_s"][: args.n])
        verify_time = sum(
            entry_map[idx]["score_time_s"] * args.n / n_pool for entry_map in scores_by_verifier
        )
        duration = record["candidate_durations_s"][chosen]

        selections.append(
            {
                "idx": idx,
                "best_idx": chosen,
                "best_seed": record["seeds"][chosen],
                "gen_audio": str(destination),
                "gen_text": record["gen_text"],
                "ref_audio": record["ref_audio"],
                "scores": matrix,
                "synth_time_s": synth_time,
                "verify_time_s": verify_time,
                "duration_s": duration,
                "rtf": (synth_time + verify_time) / duration if duration else None,
            }
        )

    switch_rate = sum(1 for s in selections if s["best_idx"] != 0) / max(len(selections), 1)
    mean_rtf = sum(s["rtf"] for s in selections if s["rtf"]) / max(len(selections), 1)

    payload = {
        "run_name": run_name,
        "strategy": args.strategy,
        "verifiers": args.verifiers,
        "verifier_families": sorted(families),
        "n": args.n,
        "metric": metric,
        "normalization": normalization,
        "pool_name": pool_name,
        "n_utterances": len(selections),
        "switch_rate": switch_rate,
        "mean_rtf": mean_rtf,
        "selections": selections,
    }
    out_path = run_dir / "selection.json"
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(
        "selected %d utterances; switched away from candidate 0 on %.1f%%; mean RTF %.3f",
        len(selections), 100 * switch_rate, mean_rtf,
    )
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
