"""Significance testing for paired WER comparisons.

BoN and baseline are evaluated on the *same* utterances, so the comparison is
paired and an unpaired test would throw away most of the power. Both routines
here are resampling-based: WER differences per utterance are heavily zero-
inflated and long-tailed, which makes the normal-approximation assumptions
behind a t-test unreliable at these effect sizes (tenths of a percentage point).

The mean of per-utterance WERs used here is not the corpus WER of
:func:`bon_tts.evaluators.corpus_wer` — corpus WER weights by utterance length,
so the two differ slightly. Rankings are preserved; report which one you used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PermutationResult:
    """Outcome of a two-sided paired permutation test."""

    mean_a: float
    mean_b: float
    difference: float  # mean_b - mean_a, so negative means b improved on a
    p_value: float
    n_permutations: int
    n_samples: int


@dataclass
class BootstrapResult:
    """Bootstrap point estimate and percentile interval."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int


def paired_permutation_test(
    baseline: Sequence[float],
    treatment: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 0,
) -> PermutationResult:
    """Two-sided paired permutation (sign-flip) test on per-utterance scores.

    Under the null hypothesis the two systems are interchangeable on each
    utterance, so flipping the sign of a paired difference leaves the
    distribution unchanged. We compare the observed mean difference against the
    distribution of mean differences under random sign flips.

    The p-value uses the ``(hits + 1) / (n + 1)`` correction, which keeps it
    strictly positive — with 10,000 permutations the smallest reportable value
    is ~1e-4, so report such cases as ``p < 0.001`` rather than as exact zeros.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must have equal length, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("paired inputs are empty")

    diff = b - a
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, diff.size))
    null_means = (signs * diff).mean(axis=1)
    hits = int(np.sum(np.abs(null_means) >= abs(observed) - 1e-15))

    return PermutationResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        difference=observed,
        p_value=(hits + 1) / (n_permutations + 1),
        n_permutations=n_permutations,
        n_samples=int(diff.size),
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for the mean of per-utterance scores."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values is empty")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])

    return BootstrapResult(
        mean=float(arr.mean()),
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down correction; returns per-hypothesis rejections.

    Scanning many (config x evaluator) cells for ``p < 0.05`` inflates the
    family-wise error rate, so report corrected decisions alongside raw
    p-values and state which family the correction covers.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            # Step-down: once a hypothesis fails, all larger p-values fail too.
            break
    return reject.tolist()
