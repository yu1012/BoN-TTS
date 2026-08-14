"""Speaker similarity (SIM-o) and naturalness (UTMOS) metrics.

BoN optimizes a content metric, so it can in principle buy WER at the cost of
timbre or naturalness. These two metrics are the guard against that: a WER
improvement only counts if SIM-o and UTMOS are preserved.

Both are automatic proxies, not listening tests. Treat differences at the third
decimal place as noise rather than as evidence of quality change.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from bon_tts.audio import load_audio

logger = logging.getLogger(__name__)

SIM_MODEL = "microsoft/wavlm-base-plus-sv"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_MODEL = "utmos22_strong"

# WavLM x-vector inference is quadratic in memory; cap very long inputs.
_MAX_SIM_SECONDS = 30


class SpeakerSimilarity:
    """Cosine similarity between WavLM x-vectors of generated and prompt audio.

    Both sides are resampled to 16 kHz through the same path, so neither gets a
    resampling advantage — SIM-o is sensitive enough to asymmetric resampling to
    shift in the third decimal.
    """

    def __init__(self, device: str = "cuda", model_name: str = SIM_MODEL):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
        logger.info("loaded speaker similarity model %s", model_name)

    @torch.no_grad()
    def _embed(self, path: str) -> torch.Tensor:
        wav, _ = load_audio(path, target_sr=16_000)
        wav = wav.squeeze()
        wav = wav[: 16_000 * _MAX_SIM_SECONDS]
        inputs = self.feature_extractor(
            wav.numpy(), sampling_rate=16_000, return_tensors="pt"
        ).to(self.device)
        return self.model(**inputs).embeddings.squeeze()

    def score(self, gen_paths: Sequence[str], ref_paths: Sequence[str]) -> dict:
        """Per-pair cosine similarity plus summary statistics."""
        similarities = []
        for gen_path, ref_path in tqdm(
            list(zip(gen_paths, ref_paths, strict=True)), desc="SIM-o", total=len(gen_paths)
        ):
            gen_emb = self._embed(gen_path)
            ref_emb = self._embed(ref_path)
            similarities.append(
                float(torch.nn.functional.cosine_similarity(gen_emb, ref_emb, dim=0).item())
            )
        return {
            "sim_mean": float(np.mean(similarities)),
            "sim_std": float(np.std(similarities)),
            "per_sample": similarities,
        }


class Utmos:
    """UTMOS22-strong naturalness predictor (loaded from torch.hub)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load(UTMOS_REPO, UTMOS_MODEL, trust_repo=True)
        self.model = self.model.to(device).eval()
        logger.info("loaded UTMOS predictor %s", UTMOS_MODEL)

    @torch.no_grad()
    def score(self, audio_paths: Sequence[str]) -> dict:
        """Per-file UTMOS plus summary statistics."""
        scores = []
        for path in tqdm(audio_paths, desc="UTMOS"):
            wav, sr = load_audio(path, target_sr=16_000)
            scores.append(float(self.model(wav.to(self.device), sr).item()))
        return {
            "utmos_mean": float(np.mean(scores)),
            "utmos_std": float(np.std(scores)),
            "per_sample": scores,
        }
