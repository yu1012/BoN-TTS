"""Access to a synthesized candidate pool.

Pool manifests are written per shard by ``scripts/synthesize_pool.py`` so that
synthesis can be split across GPUs, then merged here on read.
"""

from __future__ import annotations

import json
from pathlib import Path

# Run name under which pool candidate 0 — the single-shot baseline — is evaluated.
BASELINE_RUN_NAME = "baseline_single_shot"


def pool_dir(config: dict, pool_name: str | None = None) -> Path:
    """Directory holding a pool's candidates, scores and evaluations."""
    name = pool_name or config["pool"]["name"]
    return Path(config["output_dir"]) / name


def load_pool_records(pool_dir: Path) -> list[dict]:
    """Merge every ``pool*.json`` shard manifest, ordered by utterance index.

    Shards may overlap when a range is re-run: records without
    ``candidate_audio`` are skip markers from an interrupted run and are dropped
    in favour of a real record for the same index.
    """
    manifests = sorted(pool_dir.glob("pool*.json"))
    if not manifests:
        raise FileNotFoundError(
            f"no pool*.json in {pool_dir} — run scripts/synthesize_pool.py first"
        )

    by_idx: dict[int, dict] = {}
    for manifest in manifests:
        with open(manifest) as handle:
            payload = json.load(handle)
        for record in payload.get("records", []):
            if "candidate_audio" in record:
                by_idx[record["idx"]] = record

    if not by_idx:
        raise ValueError(
            f"{pool_dir}: manifests contain no synthesized candidates "
            "(every record was a skip marker)"
        )
    return [by_idx[idx] for idx in sorted(by_idx)]


def load_verifier_scores(
    pool_dir: Path, verifier: str, metric: str, normalization: str
) -> dict:
    """Load a cached verifier score file, or explain how to produce it."""
    path = pool_dir / "scores" / f"{verifier}__{metric}__{normalization}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run:\n"
            f"  python scripts/score_pool.py --verifier {verifier} "
            f"--metric {metric} --normalization {normalization}"
        )
    with open(path) as handle:
        return json.load(handle)


def verifier_score_path(
    pool_dir: Path, verifier: str, metric: str, normalization: str
) -> Path:
    """Where :func:`load_verifier_scores` expects a score file to live."""
    return pool_dir / "scores" / f"{verifier}__{metric}__{normalization}.json"
