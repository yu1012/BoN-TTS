"""Independent ASR evaluators for cross-evaluator triangulation.

The paper's central claim is that a BoN verifier looks strong under an evaluator
from its own ASR family and weak under a different one, so *which* evaluator is
reported changes the conclusion. Guarding against that means scoring every
configuration under several evaluators with disjoint training lineages — this
module supplies them behind one interface.

``fwhisper-lgv3`` is the official F5-TTS evaluator: Whisper-large-v3 served via
the faster-whisper (CTranslate2) runtime with ``beam_size=5``, matching
``f5_tts/eval/eval_librispeech_test_clean.py``. Reported numbers should use it
so they stay comparable with the published F5-TTS / CosyVoice 2 / MaskGCT
results; the other evaluators exist to check that a conclusion is not an
artifact of that one choice.

Both reference and hypothesis go through the F5-TTS official normalization
(:func:`bon_tts.normalization.normalize_en`) before scoring. Whisper's own
English text normalizer is deliberately *not* applied on top: it would only
touch the Whisper-family evaluators and reintroduce a family-specific advantage
into the very comparison this module exists to make fair.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch
from jiwer import cer, wer

from bon_tts.audio import load_audio
from bon_tts.normalization import normalize_en

logger = logging.getLogger(__name__)

# alias -> (backend, checkpoint)
EVALUATOR_SPECS: dict[str, tuple[str, str]] = {
    "fwhisper-lgv3": ("faster_whisper", "large-v3"),
    "w2v2-lv60": ("ctc", "facebook/wav2vec2-large-960h-lv60-self"),
    "hubert-lg": ("ctc", "facebook/hubert-large-ls960-ft"),
    "distil-sm": ("whisper", "distil-whisper/distil-small.en"),
    "whisper-med": ("whisper", "openai/whisper-medium.en"),
}

# ASR lineage per evaluator. Two evaluators sharing a family are *not*
# independent for the purposes of triangulation.
EVALUATOR_FAMILY: dict[str, str] = {
    "fwhisper-lgv3": "whisper",
    "distil-sm": "whisper",
    "whisper-med": "whisper",
    "w2v2-lv60": "wav2vec2",
    "hubert-lg": "hubert",
}

DEFAULT_EVALUATORS = ("fwhisper-lgv3", "w2v2-lv60", "hubert-lg")


class Evaluator:
    """Transcribes generated audio files for WER/CER scoring."""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
    ):
        """
        Args:
            name: an alias from :data:`EVALUATOR_SPECS`, or ``"backend:checkpoint"``.
            compute_type: faster-whisper precision (ignored by other backends).
            beam_size: faster-whisper beam size; 5 matches the F5-TTS official script.
        """
        self.name = name
        self.device = device
        self.beam_size = beam_size

        if name in EVALUATOR_SPECS:
            self.backend, self.checkpoint = EVALUATOR_SPECS[name]
        elif ":" in name:
            self.backend, self.checkpoint = name.split(":", 1)
        else:
            raise ValueError(
                f"unknown evaluator {name!r}; use one of {sorted(EVALUATOR_SPECS)} "
                "or 'backend:checkpoint'"
            )

        self.family = EVALUATOR_FAMILY.get(name, self.backend)
        self._load(compute_type)

    def _load(self, compute_type: str) -> None:
        if self.backend == "faster_whisper":
            from faster_whisper import WhisperModel

            device = "cuda" if str(self.device).startswith("cuda") else "cpu"
            device_index = 0
            if ":" in str(self.device):
                device_index = int(str(self.device).split(":")[1])
            if device == "cpu":
                compute_type = "int8"
            self.model = WhisperModel(
                self.checkpoint,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
            )
        elif self.backend == "whisper":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
            self.model = (
                WhisperForConditionalGeneration.from_pretrained(self.checkpoint)
                .to(self.device)
                .eval()
            )
        elif self.backend == "ctc":
            from transformers import AutoModelForCTC, Wav2Vec2Processor

            self.processor = Wav2Vec2Processor.from_pretrained(self.checkpoint)
            self.model = AutoModelForCTC.from_pretrained(self.checkpoint).to(self.device).eval()
        else:
            raise ValueError(f"unknown evaluator backend {self.backend!r}")

        logger.info("loaded evaluator %s (%s, %s)", self.name, self.backend, self.checkpoint)

    def _whisper_generate_kwargs(self) -> dict:
        is_multilingual = getattr(
            self.model.generation_config,
            "is_multilingual",
            getattr(self.model.config, "is_multilingual", True),
        )
        kwargs: dict = {"max_new_tokens": 440}
        if is_multilingual:
            kwargs["language"] = "en"
            kwargs["task"] = "transcribe"
        return kwargs

    @torch.no_grad()
    def transcribe(self, audio_paths: Sequence[str], batch_size: int = 8) -> list[str]:
        """Transcribe audio files, preserving input order."""
        if self.backend == "faster_whisper":
            hyps = []
            for path in audio_paths:
                segments, _ = self.model.transcribe(
                    str(path), beam_size=self.beam_size, language="en"
                )
                hyps.append(" ".join(segment.text for segment in segments))
            return hyps

        hyps = []
        for start in range(0, len(audio_paths), batch_size):
            batch = audio_paths[start : start + batch_size]
            arrays = [load_audio(p, target_sr=16_000)[0].squeeze().numpy() for p in batch]
            inputs = self.processor(
                arrays, sampling_rate=16_000, return_tensors="pt", padding=True
            ).to(self.device)

            if self.backend == "whisper":
                features = inputs.input_features.to(dtype=self.model.dtype)
                predicted = self.model.generate(features, **self._whisper_generate_kwargs())
                hyps.extend(self.processor.batch_decode(predicted, skip_special_tokens=True))
            else:
                logits = self.model(**inputs).logits
                hyps.extend(self.processor.batch_decode(logits.argmax(dim=-1)))

        return hyps


def corpus_wer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Corpus WER: total word edits over total reference words.

    This is the aggregation the F5-TTS evaluation reports, and it is *not* the
    mean of per-utterance WERs — long utterances carry proportionally more
    weight. Both are reported in this repo; keep them labelled.
    """
    refs = [normalize_en(r) for r in references]
    hyps = [normalize_en(h) for h in hypotheses]
    return float(wer(refs, hyps))


def corpus_cer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Corpus CER, normalized the same way as :func:`corpus_wer`."""
    refs = [normalize_en(r) for r in references]
    hyps = [normalize_en(h) for h in hypotheses]
    return float(cer(refs, hyps))


def per_sample_wer(references: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
    """Per-utterance WER, for paired significance tests and bootstrap CIs."""
    out = []
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_n, hyp_n = normalize_en(ref), normalize_en(hyp)
        if not ref_n:
            out.append(0.0)
        elif not hyp_n:
            out.append(1.0)
        else:
            out.append(float(wer(ref_n, hyp_n)))
    return out
