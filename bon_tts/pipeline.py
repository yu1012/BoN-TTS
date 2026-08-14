"""Online Best-of-N inference: synthesize, verify and select in one call.

This is the deployable form of the method. The reproduction scripts instead
split synthesis and selection into cached stages so that many configurations can
be compared on one candidate pool; use this when you just want better audio out
of a single utterance.

Cost is ``N`` full syntheses plus one verifier pass over ``N`` clips, so RTF
grows roughly linearly in ``N``. The candidate that shares the base seed is
always index 0, so ties never move you off the plain single-shot sample.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from bon_tts.audio import ASR_SAMPLE_RATE, TTS_SAMPLE_RATE, resample
from bon_tts.f5_backend import F5CandidateGenerator, candidate_seeds
from bon_tts.selection import select
from bon_tts.verifiers import Verifier

logger = logging.getLogger(__name__)


@dataclass
class BoNResult:
    """Selected audio plus the decision trail behind it."""

    wav: torch.Tensor
    sample_rate: int
    best_idx: int
    best_seed: int
    scores: list[list[float]]  # [n_verifiers, n_candidates]
    strategy: str
    n_candidates: int
    synth_time_s: float
    verify_time_s: float
    rtf: float


def bon_infer(
    generator: F5CandidateGenerator,
    verifiers: Sequence[Verifier],
    ref_file: str,
    ref_text: str,
    gen_text: str,
    n_candidates: int = 5,
    strategy: str = "single",
    base_seed: int = 42,
    seed_offset: int = 1000,
) -> BoNResult:
    """Synthesize ``n_candidates``, score them, and return the selected audio.

    Args:
        generator: configured :class:`~bon_tts.f5_backend.F5CandidateGenerator`.
        verifiers: one verifier for ``single``; two or more (ideally from
            different ASR families) for ``rank_avg`` / ``max_rank``.
        strategy: see :mod:`bon_tts.selection`.
    """
    if not verifiers:
        raise ValueError("at least one verifier is required")
    if strategy == "single" and len(verifiers) != 1:
        raise ValueError(f"strategy 'single' takes exactly one verifier, got {len(verifiers)}")

    seeds = candidate_seeds(n_candidates, base_seed=base_seed, seed_offset=seed_offset)

    t0 = time.time()
    candidates = generator.generate(ref_file, ref_text, gen_text, seeds)
    synth_time = time.time() - t0

    wavs_16k = [resample(c.wav, from_sr=TTS_SAMPLE_RATE, to_sr=ASR_SAMPLE_RATE) for c in candidates]

    t1 = time.time()
    scores = [verifier.score(wavs_16k, gen_text) for verifier in verifiers]
    verify_time = time.time() - t1

    best_idx = select(scores, strategy=strategy)
    best = candidates[best_idx]
    duration = len(best.wav) / TTS_SAMPLE_RATE

    logger.info(
        "BoN(%s, N=%d): picked candidate %d (seed %d) in %.1fs synth + %.1fs verify",
        strategy, n_candidates, best_idx, best.seed, synth_time, verify_time,
    )

    return BoNResult(
        wav=best.wav,
        sample_rate=TTS_SAMPLE_RATE,
        best_idx=best_idx,
        best_seed=best.seed,
        scores=scores,
        strategy=strategy,
        n_candidates=n_candidates,
        synth_time_s=synth_time,
        verify_time_s=verify_time,
        rtf=(synth_time + verify_time) / duration if duration else float("inf"),
    )
