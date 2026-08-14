"""End-to-end test of the pool -> select pipeline on a synthetic pool.

Exercises the plumbing the unit tests do not: shard-manifest merging, cached
score loading, candidate truncation to N, audio copying, and the shape of
``selection.json``. Runs the script as a subprocess so the CLI wiring is covered
too. No GPU and no models — the "audio" is silence.

The fixture is built so the three strategies provably disagree. Two verifiers
rank the N=5 candidates in exactly opposite orders:

    v0 (w2v2-base) scores [1.00, 0.95, 0.90, 0.85, 0.80] -> ranks [4, 3, 2, 1, 0]
    v1 (distil-v3) scores [0.00, 0.05, 0.10, 0.15, 0.20] -> ranks [0, 1, 2, 3, 4]

    single  (v1 only) -> argmin of v1          -> 0
    rank_avg          -> mean ranks all 2.0    -> tie -> 0 (lowest index)
    max_rank          -> worst ranks [4,3,2,3,4] -> 2

That ``max_rank`` lands on the compromise candidate while ``rank_avg`` ties is
the whole point of having both: they are not interchangeable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECT_SCRIPT = REPO_ROOT / "scripts" / "select_candidates.py"

N_CANDIDATES = 10
N_UTTERANCES = 6
SELECT_N = 5
BASE_SEED = 42
SEED_OFFSET = 1000


@pytest.fixture
def synthetic_pool(tmp_path: Path) -> Path:
    """Build a pool with two shard manifests and two disagreeing verifiers."""
    output_dir = tmp_path / "outputs"
    pool = output_dir / "pool_n10"
    (pool / "candidates").mkdir(parents=True)
    (pool / "scores").mkdir(parents=True)

    seeds = [BASE_SEED + i * SEED_OFFSET for i in range(N_CANDIDATES)]
    ref_audio = tmp_path / "ref.wav"
    sf.write(str(ref_audio), np.zeros(24_000, dtype="float32"), 24_000)

    records = []
    for idx in range(N_UTTERANCES):
        paths = []
        for c in range(N_CANDIDATES):
            path = pool / "candidates" / f"{idx:04d}_c{c}.wav"
            # Distinct lengths so a wrong candidate index is detectable.
            sf.write(str(path), np.zeros(int(24_000 * (1.0 + 0.1 * c)), dtype="float32"), 24_000)
            paths.append(str(path))
        records.append(
            {
                "idx": idx,
                "seeds": seeds,
                "gen_text": f"utterance {idx}",
                "ref_audio": str(ref_audio),
                "candidate_audio": paths,
                "candidate_gen_times_s": [0.5] * N_CANDIDATES,
                "candidate_durations_s": [1.0 + 0.1 * c for c in range(N_CANDIDATES)],
                "pool_wall_time_s": 5.0,
                "skipped": False,
            }
        )

    # Two shards, so the merge path in bon_tts.pool is exercised.
    for low, high in ((0, 3), (3, N_UTTERANCES)):
        manifest = pool / f"pool_{low:05d}_{high:05d}.json"
        manifest.write_text(
            json.dumps(
                {
                    "pool_name": "pool_n10",
                    "n_candidates": N_CANDIDATES,
                    "seeds": seeds,
                    "tts": {},
                    "data": "synthetic",
                    "records": records[low:high],
                }
            )
        )

    verifier_scores = {
        "w2v2-base": [1.0 - 0.05 * c for c in range(N_CANDIDATES)],
        "distil-v3": [0.05 * c for c in range(N_CANDIDATES)],
    }
    for name, scores in verifier_scores.items():
        path = pool / "scores" / f"{name}__wer_cer__simple.json"
        path.write_text(
            json.dumps(
                {
                    "verifier": name,
                    "checkpoint": name,
                    "family": "synthetic",
                    "metric": "wer_cer",
                    "composite_alpha": 0.5,
                    "normalization": "simple",
                    "pool_name": "pool_n10",
                    "n_utterances": N_UTTERANCES,
                    "total_score_time_s": 1.0,
                    "per_utterance": [
                        {"idx": i, "scores": list(scores), "score_time_s": 0.1}
                        for i in range(N_UTTERANCES)
                    ],
                }
            )
        )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(output_dir),
                "data": {"name": "synthetic"},
                "pool": {
                    "name": "pool_n10",
                    "n_candidates": N_CANDIDATES,
                    "seed_offset": SEED_OFFSET,
                },
                "verifier": {
                    "metric": "wer_cer",
                    "composite_alpha": 0.5,
                    "normalization": "simple",
                },
                "evaluation": {"evaluators": ["fwhisper-lgv3"]},
            }
        )
    )
    return config_path


def pool_output_dir(config_path: Path) -> str:
    """The fixture's output_dir, to be passed back as ``--set output_dir=...``.

    Every subprocess invocation needs it: the container sets BON_TTS_OUTPUT_DIR,
    which outranks the config file and would redirect the run out of tmp_path.
    """
    return yaml.safe_load(config_path.read_text())["output_dir"]


def run_select(config_path: Path, strategy: str, verifiers: list[str], n: int = SELECT_N):
    """Invoke select_candidates.py and return its parsed selection.json."""
    output_dir = pool_output_dir(config_path)
    result = subprocess.run(
        [
            sys.executable, str(SELECT_SCRIPT),
            "--config", str(config_path),
            "--set", f"output_dir={output_dir}",
            "--strategy", strategy,
            "--verifiers", *verifiers,
            "--n", str(n),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"select failed:\n{result.stdout}\n{result.stderr}"

    run_name = f"select_{strategy}_{'+'.join(verifiers)}_n{n}"
    selection_path = Path(output_dir) / run_name / "selection.json"
    assert selection_path.exists(), f"{selection_path} not written"
    return json.loads(selection_path.read_text())


class TestSelectionPipeline:
    def test_single_verifier_picks_its_argmin(self, synthetic_pool):
        payload = run_select(synthetic_pool, "single", ["distil-v3"])
        assert [s["best_idx"] for s in payload["selections"]] == [0] * N_UTTERANCES
        assert payload["switch_rate"] == 0.0

    def test_rank_avg_ties_to_baseline_candidate(self, synthetic_pool):
        # Opposite rankings average to a flat 2.0 across all five candidates;
        # the tie must resolve to candidate 0, never drifting off the baseline.
        payload = run_select(synthetic_pool, "rank_avg", ["w2v2-base", "distil-v3"])
        assert [s["best_idx"] for s in payload["selections"]] == [0] * N_UTTERANCES

    def test_max_rank_picks_the_compromise_candidate(self, synthetic_pool):
        payload = run_select(synthetic_pool, "max_rank", ["w2v2-base", "distil-v3"])
        assert [s["best_idx"] for s in payload["selections"]] == [2] * N_UTTERANCES
        assert payload["switch_rate"] == 1.0

    def test_strategies_are_not_interchangeable(self, synthetic_pool):
        rank_avg = run_select(synthetic_pool, "rank_avg", ["w2v2-base", "distil-v3"])
        max_rank = run_select(synthetic_pool, "max_rank", ["w2v2-base", "distil-v3"])
        assert [s["best_idx"] for s in rank_avg["selections"]] != [
            s["best_idx"] for s in max_rank["selections"]
        ]


class TestSelectionOutput:
    def test_records_seed_matching_chosen_candidate(self, synthetic_pool):
        payload = run_select(synthetic_pool, "max_rank", ["w2v2-base", "distil-v3"])
        for item in payload["selections"]:
            assert item["best_seed"] == BASE_SEED + item["best_idx"] * SEED_OFFSET

    def test_copies_selected_audio(self, synthetic_pool):
        payload = run_select(synthetic_pool, "single", ["distil-v3"])
        for item in payload["selections"]:
            audio = Path(item["gen_audio"])
            assert audio.exists()
            # Candidate lengths differ by index, so duration confirms identity.
            info = sf.info(str(audio))
            assert info.duration == pytest.approx(1.0 + 0.1 * item["best_idx"], abs=1e-3)

    def test_score_matrix_shape(self, synthetic_pool):
        payload = run_select(synthetic_pool, "rank_avg", ["w2v2-base", "distil-v3"])
        for item in payload["selections"]:
            assert len(item["scores"]) == 2
            assert all(len(row) == SELECT_N for row in item["scores"])

    def test_merges_both_shard_manifests(self, synthetic_pool):
        payload = run_select(synthetic_pool, "single", ["distil-v3"])
        assert payload["n_utterances"] == N_UTTERANCES
        assert [s["idx"] for s in payload["selections"]] == list(range(N_UTTERANCES))

    def test_rtf_accounts_for_synthesis_and_verification(self, synthetic_pool):
        payload = run_select(synthetic_pool, "single", ["distil-v3"])
        item = payload["selections"][0]
        # 5 candidates x 0.5 s synthesis, plus the prorated verifier share.
        assert item["synth_time_s"] == pytest.approx(2.5)
        assert item["verify_time_s"] == pytest.approx(0.1 * SELECT_N / N_CANDIDATES)
        expected = (item["synth_time_s"] + item["verify_time_s"]) / item["duration_s"]
        assert item["rtf"] == pytest.approx(expected)


class TestSelectionValidation:
    def test_rejects_n_larger_than_pool(self, synthetic_pool):
        result = subprocess.run(
            [
                sys.executable, str(SELECT_SCRIPT),
                "--config", str(synthetic_pool),
                "--set", f"output_dir={pool_output_dir(synthetic_pool)}",
                "--strategy", "single", "--verifiers", "distil-v3",
                "--n", str(N_CANDIDATES + 1),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "exceeds the pool size" in result.stderr

    def test_rejects_ensemble_with_one_verifier(self, synthetic_pool):
        result = subprocess.run(
            [
                sys.executable, str(SELECT_SCRIPT),
                "--config", str(synthetic_pool),
                "--set", f"output_dir={pool_output_dir(synthetic_pool)}",
                "--strategy", "rank_avg", "--verifiers", "distil-v3", "--n", "3",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "at least two" in result.stderr

    def test_reports_missing_score_file_with_remedy(self, synthetic_pool):
        result = subprocess.run(
            [
                sys.executable, str(SELECT_SCRIPT),
                "--config", str(synthetic_pool),
                "--set", f"output_dir={pool_output_dir(synthetic_pool)}",
                "--strategy", "single", "--verifiers", "never-scored", "--n", "3",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "score_pool.py" in result.stderr
