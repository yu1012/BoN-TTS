"""Tests for the resampling significance tests."""

import numpy as np
import pytest

from bon_tts.stats import bootstrap_mean_ci, holm_bonferroni, paired_permutation_test


class TestPairedPermutation:
    def test_identical_systems_are_not_significant(self):
        values = [0.0, 0.1, 0.2, 0.0, 0.05] * 20
        result = paired_permutation_test(values, values, n_permutations=1000, seed=0)
        assert result.difference == 0.0
        assert result.p_value == pytest.approx(1.0)

    def test_consistent_improvement_is_significant(self):
        rng = np.random.default_rng(0)
        baseline = rng.uniform(0.0, 0.2, size=200)
        treatment = baseline - 0.05  # uniformly better on every utterance
        result = paired_permutation_test(baseline, treatment, n_permutations=2000, seed=0)
        assert result.difference < 0
        assert result.p_value < 0.01

    def test_p_value_is_never_zero(self):
        # The (hits + 1) / (n + 1) correction keeps p reportable as "< 1/(n+1)"
        # rather than as a false exact zero.
        baseline = np.ones(100)
        treatment = np.zeros(100)
        result = paired_permutation_test(baseline, treatment, n_permutations=100, seed=0)
        assert result.p_value > 0
        assert result.p_value == pytest.approx(1 / 101)

    def test_is_two_sided(self):
        rng = np.random.default_rng(1)
        baseline = rng.uniform(0.0, 0.2, size=200)
        better = baseline - 0.05
        worse = baseline + 0.05
        p_better = paired_permutation_test(baseline, better, n_permutations=2000, seed=0).p_value
        p_worse = paired_permutation_test(baseline, worse, n_permutations=2000, seed=0).p_value
        assert p_better == pytest.approx(p_worse, abs=0.02)

    def test_deterministic_under_seed(self):
        rng = np.random.default_rng(2)
        a, b = rng.uniform(size=50), rng.uniform(size=50)
        first = paired_permutation_test(a, b, n_permutations=500, seed=7).p_value
        second = paired_permutation_test(a, b, n_permutations=500, seed=7).p_value
        assert first == second

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_permutation_test([0.1, 0.2], [0.1], n_permutations=10)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            paired_permutation_test([], [], n_permutations=10)


class TestBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.02, 0.01, size=500)
        result = bootstrap_mean_ci(values, n_resamples=2000, seed=0)
        assert result.ci_low < result.mean < result.ci_high

    def test_wider_interval_at_higher_confidence(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.02, 0.01, size=200)
        narrow = bootstrap_mean_ci(values, n_resamples=2000, confidence=0.80, seed=0)
        wide = bootstrap_mean_ci(values, n_resamples=2000, confidence=0.99, seed=0)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_constant_input_has_zero_width(self):
        result = bootstrap_mean_ci([0.5] * 50, n_resamples=200, seed=0)
        assert result.ci_low == result.ci_high == 0.5

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_mean_ci([], n_resamples=10)


class TestHolmBonferroni:
    def test_single_hypothesis_matches_alpha(self):
        assert holm_bonferroni([0.04], alpha=0.05) == [True]
        assert holm_bonferroni([0.06], alpha=0.05) == [False]

    def test_step_down_stops_at_first_failure(self):
        # Sorted: 0.001 (<=0.05/3), 0.03 (>0.05/2 -> fails), so 0.04 also fails
        # even though 0.04 <= 0.05/1.
        assert holm_bonferroni([0.001, 0.03, 0.04], alpha=0.05) == [True, False, False]

    def test_preserves_input_order(self):
        assert holm_bonferroni([0.9, 0.001], alpha=0.05) == [False, True]

    def test_all_rejected_when_all_tiny(self):
        assert holm_bonferroni([1e-6, 1e-6, 1e-6], alpha=0.05) == [True, True, True]
