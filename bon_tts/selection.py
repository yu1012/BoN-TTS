"""Candidate selection strategies for Best-of-N TTS.

Every strategy maps a score matrix to the index of one chosen candidate. The
matrix is ``[n_verifiers, n_candidates]`` and scores are *lower-is-better*
(WER, CER, or the WER+CER composite of :mod:`bon_tts.verifiers`).

Three strategies:

``single``
    Plain Best-of-N: the argmin of the one verifier's scores. This is what the
    TTS literature means by "BoN with verifier X".

``rank_avg``
    Each verifier ranks the candidates independently; pick the candidate with
    the lowest mean rank. Aggregating *ranks* rather than raw scores keeps a
    verifier that reports systematically larger WERs from dominating the vote.

``max_rank``
    Pick the candidate with the lowest worst-case rank across verifiers, so the
    winner must rank well under *every* family. This conjunctive form explicitly
    penalizes single-family inflation.

``rank_avg`` and ``max_rank`` are the cross-family rank ensembles the paper
proposes as a mitigation for the verifier–evaluator family alignment confound.
Their point is that the verifier list spans *disjoint ASR lineages* (e.g.
wav2vec 2.0 + Distil-Whisper); running them over two checkpoints of one family
reproduces that family's bias rather than cancelling it.

All strategies break ties toward the lower candidate index. Candidate order is
the seed order fixed by :func:`bon_tts.f5_backend.candidate_seeds`, so candidate
0 is always the single-shot baseline sample and a tie never counts as an
improvement over it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

STRATEGIES = ("single", "rank_avg", "max_rank")


def competition_ranks(scores: Sequence[float]) -> np.ndarray:
    """Rank ``scores`` ascending, 0-based, ties sharing the lowest rank.

    ``[0.1, 0.3, 0.1, 0.5] -> [0, 2, 0, 3]``. Standard competition ranking:
    tied candidates are equally good, and the next distinct score is pushed
    down by the size of the tie so a rank is always "how many candidates are
    strictly better".
    """
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=int)
    rank_of_group = 0
    for position, idx in enumerate(order):
        if position > 0 and arr[idx] > arr[order[position - 1]]:
            rank_of_group = position
        ranks[idx] = rank_of_group
    return ranks


def _argmin_lowest_index(values: Sequence[float]) -> int:
    """Argmin of ``values``, breaking ties toward the lower index."""
    arr = np.asarray(values, dtype=float)
    return int(np.flatnonzero(arr == arr.min())[0])


def select(score_matrix: Sequence[Sequence[float]], strategy: str = "single") -> int:
    """Return the index of the selected candidate.

    Args:
        score_matrix: ``[n_verifiers, n_candidates]`` lower-is-better scores.
            ``single`` requires exactly one verifier row.
        strategy: one of :data:`STRATEGIES`.
    """
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2-D [n_verifiers, n_candidates], got {matrix.shape}")
    if matrix.size == 0:
        raise ValueError("score_matrix is empty")

    if strategy == "single":
        if matrix.shape[0] != 1:
            raise ValueError(
                f"strategy 'single' needs exactly 1 verifier, got {matrix.shape[0]}. "
                "Use 'rank_avg' or 'max_rank' to combine verifiers."
            )
        return _argmin_lowest_index(matrix[0])

    if strategy not in ("rank_avg", "max_rank"):
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    ranks = np.stack([competition_ranks(row) for row in matrix])  # [V, N]
    aggregate = ranks.mean(axis=0) if strategy == "rank_avg" else ranks.max(axis=0)
    return _argmin_lowest_index(aggregate)


def select_oracle(evaluator_scores: Sequence[float]) -> int:
    """Index of the candidate an omniscient verifier would pick.

    Takes *evaluator* scores, not verifier scores, so the result is an upper
    bound on what any verifier could achieve on this candidate pool rather than
    a deployable strategy.
    """
    return _argmin_lowest_index(evaluator_scores)


def recovery_rate(single_shot: float, selected: float, oracle: float) -> float:
    """Fraction of the oracle headroom a selection actually recovered.

    ``(single_shot - selected) / (single_shot - oracle)``, as a percentage.
    Returns NaN when there is no headroom to recover (oracle ties single-shot),
    which keeps a degenerate subset from inflating a mean recovery.
    """
    headroom = single_shot - oracle
    if headroom <= 0:
        return float("nan")
    return 100.0 * (single_shot - selected) / headroom
