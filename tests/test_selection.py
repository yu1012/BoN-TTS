"""Tests for candidate selection strategies."""

import numpy as np
import pytest

from bon_tts.selection import (
    competition_ranks,
    recovery_rate,
    select,
    select_oracle,
)


class TestCompetitionRanks:
    def test_orders_ascending(self):
        assert competition_ranks([0.3, 0.1, 0.2]).tolist() == [2, 0, 1]

    def test_ties_share_lowest_rank(self):
        # Two candidates tied for best both get rank 0; the next distinct score
        # is pushed to rank 2, so a rank always counts strictly-better candidates.
        assert competition_ranks([0.1, 0.3, 0.1, 0.5]).tolist() == [0, 2, 0, 3]

    def test_all_tied(self):
        assert competition_ranks([0.2, 0.2, 0.2]).tolist() == [0, 0, 0]

    def test_single_candidate(self):
        assert competition_ranks([0.7]).tolist() == [0]


class TestSingleStrategy:
    def test_picks_argmin(self):
        assert select([[0.5, 0.1, 0.3]], strategy="single") == 1

    def test_tie_breaks_to_lower_index(self):
        # Candidate 0 is the single-shot baseline; a tie must not move off it.
        assert select([[0.2, 0.2, 0.4]], strategy="single") == 0

    def test_rejects_multiple_verifiers(self):
        with pytest.raises(ValueError, match="exactly 1 verifier"):
            select([[0.1, 0.2], [0.2, 0.1]], strategy="single")


class TestRankAvg:
    def test_averages_ranks_across_verifiers(self):
        # v0 ranks: c1 < c0 < c2  -> [1, 0, 2]
        # v1 ranks: c1 < c2 < c0  -> [2, 0, 1]
        # mean:                      [1.5, 0, 1.5] -> candidate 1
        scores = [[0.2, 0.1, 0.3], [0.3, 0.1, 0.2]]
        assert select(scores, strategy="rank_avg") == 1

    def test_rank_based_not_score_based(self):
        # v1 reports much larger magnitudes but the same ordering. Averaging raw
        # scores would let v1 dominate; averaging ranks gives both equal weight,
        # so v0's preference for candidate 0 decides the tie by lower index.
        scores = [[0.10, 0.20], [9.0, 1.0]]
        ranks_choice = select(scores, strategy="rank_avg")
        mean_score_choice = int(np.argmin(np.mean(scores, axis=0)))
        assert ranks_choice == 0
        assert mean_score_choice == 1

    def test_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            select([0.1, 0.2], strategy="rank_avg")


class TestMaxRank:
    def test_minimises_worst_case_rank(self):
        # c0: ranks (0, 2) -> worst 2
        # c1: ranks (1, 1) -> worst 1  <- wins, ranks acceptably under both
        # c2: ranks (2, 0) -> worst 2
        scores = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
        assert select(scores, strategy="max_rank") == 1

    def test_differs_from_rank_avg(self):
        # c0 mean rank 1.0 beats c1's 1.5, but c0's worst rank (2) is beaten by
        # c1's (2)... constructed so the two strategies disagree:
        # v0 ranks [0, 1, 2]; v1 ranks [2, 1, 0]
        # rank_avg -> [1.0, 1.0, 1.0] -> tie -> candidate 0
        # max_rank -> [2, 1, 2]       -> candidate 1
        scores = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
        assert select(scores, strategy="rank_avg") == 0
        assert select(scores, strategy="max_rank") == 1

    def test_unanimous_winner(self):
        scores = [[0.5, 0.1], [0.6, 0.2]]
        assert select(scores, strategy="max_rank") == 1


class TestValidation:
    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            select([[0.1, 0.2]], strategy="borda")

    def test_empty_matrix(self):
        with pytest.raises(ValueError, match="empty"):
            select(np.empty((0, 0)), strategy="rank_avg")


class TestOracle:
    def test_picks_best_evaluator_score(self):
        assert select_oracle([0.4, 0.0, 0.2]) == 1

    def test_tie_breaks_to_lower_index(self):
        assert select_oracle([0.0, 0.0]) == 0


class TestRecoveryRate:
    def test_full_recovery(self):
        assert recovery_rate(single_shot=0.02, selected=0.01, oracle=0.01) == 100.0

    def test_no_recovery(self):
        assert recovery_rate(single_shot=0.02, selected=0.02, oracle=0.01) == 0.0

    def test_partial_recovery(self):
        assert recovery_rate(single_shot=0.02, selected=0.015, oracle=0.01) == pytest.approx(50.0)

    def test_regression_is_negative(self):
        assert recovery_rate(single_shot=0.02, selected=0.025, oracle=0.01) == pytest.approx(-50.0)

    def test_no_headroom_is_nan(self):
        # Without headroom the ratio is undefined; NaN keeps it out of means
        # instead of silently reporting 0% or dividing by zero.
        assert np.isnan(recovery_rate(single_shot=0.01, selected=0.01, oracle=0.01))
