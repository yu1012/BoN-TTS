"""ASR verifiers that score BoN candidates against the target text.

A verifier transcribes each candidate waveform and scores it by edit distance
to the target transcript. Scores are lower-is-better and only their *ordering*
within one utterance matters — :mod:`bon_tts.selection` consumes ranks.

Two model interfaces are supported, covering the three ASR families in the
paper: CTC (wav2vec 2.0, HuBERT) and encoder–decoder (Whisper, Distil-Whisper).

Text normalization
------------------
``normalization="simple"`` (default) lowercases and strips, which is what
produced every number reported in the paper. Note the asymmetry it leaves: CTC
models emit no punctuation while the LibriSpeech-PC targets carry it, so a
wav2vec 2.0 verifier sees a constant WER offset. The offset is identical across
the candidates of an utterance, so candidate *ranking* — all a verifier is used
for — is essentially unaffected.

``normalization="f5_official"`` instead applies :func:`bon_tts.normalization.normalize_en`
to both sides, removing that asymmetry. It is the cleaner choice for new work,
but it does not reproduce the published tables; keep the default when
reproducing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from jiwer import cer, wer

from bon_tts.normalization import normalize_en

logger = logging.getLogger(__name__)

METRICS = ("wer", "cer", "wer_cer")
NORMALIZATIONS = ("simple", "f5_official")

# Short aliases for the checkpoints used in the paper.
VERIFIER_ALIASES = {
    "w2v2-base": "facebook/wav2vec2-base-960h",
    "distil-sm": "distil-whisper/distil-small.en",
    "distil-v3": "distil-whisper/distil-large-v3",
}


def resolve_verifier(name: str) -> str:
    """Expand a verifier alias to a HuggingFace model id (pass-through if unknown)."""
    return VERIFIER_ALIASES.get(name, name)


class Verifier:
    """Frozen ASR model that scores candidate waveforms against a target text."""

    def __init__(
        self,
        model_name: str = "w2v2-base",
        device: str = "cuda",
        metric: str = "wer_cer",
        composite_alpha: float = 0.5,
        normalization: str = "simple",
    ):
        """
        Args:
            model_name: HuggingFace model id or an alias from :data:`VERIFIER_ALIASES`.
            device: torch device string.
            metric: ``wer`` | ``cer`` | ``wer_cer`` (composite, see ``composite_alpha``).
            composite_alpha: WER weight in ``wer_cer``; score is
                ``alpha * WER + (1 - alpha) * CER``.
            normalization: ``simple`` | ``f5_official`` — see the module docstring.
        """
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
        if normalization not in NORMALIZATIONS:
            raise ValueError(
                f"unknown normalization {normalization!r}; expected one of {NORMALIZATIONS}"
            )
        if not 0.0 <= composite_alpha <= 1.0:
            raise ValueError(f"composite_alpha must be in [0, 1], got {composite_alpha}")

        self.name = model_name
        self.model_name = resolve_verifier(model_name)
        self.device = device
        self.metric = metric
        self.composite_alpha = composite_alpha
        self.normalization = normalization
        self._load()

    def _load(self) -> None:
        name = self.model_name.lower()
        if "hubert" in name:
            from transformers import HubertForCTC, Wav2Vec2Processor

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = HubertForCTC.from_pretrained(self.model_name)
            self.family = "ctc"
        elif "wav2vec2" in name:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.family = "ctc"
        elif "whisper" in name:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.family = "whisper"
        else:
            raise ValueError(
                f"unsupported verifier {self.model_name!r}: expected a wav2vec2, HuBERT, "
                "Whisper or Distil-Whisper checkpoint"
            )

        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        logger.info("loaded verifier %s (%s family)", self.model_name, self.family)

    def _normalize(self, text: str) -> str:
        if self.normalization == "f5_official":
            return normalize_en(text)
        return text.strip().lower()

    def _whisper_generate_kwargs(self) -> dict:
        """``generate`` kwargs, omitting language/task for English-only checkpoints."""
        is_multilingual = getattr(
            self.model.generation_config,
            "is_multilingual",
            getattr(self.model.config, "is_multilingual", True),
        )
        kwargs: dict = {"max_new_tokens": 200}
        if is_multilingual:
            kwargs["language"] = "en"
            kwargs["task"] = "transcribe"
        return kwargs

    @torch.no_grad()
    def transcribe(self, wavs_16k: Sequence[torch.Tensor]) -> list[str]:
        """Transcribe a batch of 16 kHz mono waveforms in one forward pass."""
        arrays = [w.detach().cpu().numpy() for w in wavs_16k]
        inputs = self.processor(
            arrays, sampling_rate=16_000, return_tensors="pt", padding=True
        ).to(self.device)

        if self.family == "whisper":
            features = inputs.input_features.to(dtype=self.model.dtype)
            predicted = self.model.generate(features, **self._whisper_generate_kwargs())
            hyps = self.processor.batch_decode(predicted, skip_special_tokens=True)
        else:
            logits = self.model(**inputs).logits  # [B, T, V]
            predicted = logits.argmax(dim=-1)
            hyps = self.processor.batch_decode(predicted)

        return [h.strip().lower() for h in hyps]

    def score(self, wavs_16k: Sequence[torch.Tensor], target_text: str) -> list[float]:
        """Score each candidate against ``target_text`` (lower is better)."""
        if not wavs_16k:
            return []

        target = self._normalize(target_text)
        scores = []
        for hyp in self.transcribe(wavs_16k):
            hyp = self._normalize(hyp)
            if not hyp or not target:
                # An empty transcript carries no ranking information; score it
                # worst rather than letting jiwer divide by zero.
                scores.append(1.0)
                continue
            if self.metric == "wer":
                scores.append(wer(target, hyp))
            elif self.metric == "cer":
                scores.append(cer(target, hyp))
            else:
                alpha = self.composite_alpha
                scores.append(alpha * wer(target, hyp) + (1 - alpha) * cer(target, hyp))
        return scores


def load_verifiers(
    names: Sequence[str],
    device: str = "cuda",
    metric: str = "wer_cer",
    composite_alpha: float = 0.5,
    normalization: str = "simple",
) -> list[Verifier]:
    """Load several verifiers with shared settings, in the given order."""
    return [
        Verifier(
            model_name=name,
            device=device,
            metric=metric,
            composite_alpha=composite_alpha,
            normalization=normalization,
        )
        for name in names
    ]
