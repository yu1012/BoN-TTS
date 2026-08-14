"""Audio I/O helpers.

``load_audio`` is torchaudio-first with a soundfile fallback: recent torchaudio
releases route file decoding through torchcodec, which needs system FFmpeg
libraries that lean containers do not always ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ASR_SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000


def load_audio(
    path: str | Path,
    target_sr: int | None = TTS_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load ``path`` as mono ``[1, T]`` float32, optionally resampled."""
    path = str(path)
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_audio(wav: torch.Tensor | np.ndarray, path: str | Path, sr: int = TTS_SAMPLE_RATE) -> None:
    """Write a waveform to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    sf.write(str(path), np.asarray(wav).squeeze(), sr)


def resample(wav: torch.Tensor, from_sr: int, to_sr: int = ASR_SAMPLE_RATE) -> torch.Tensor:
    """Resample a 1-D waveform."""
    if from_sr == to_sr:
        return wav
    return torchaudio.functional.resample(wav.unsqueeze(0), from_sr, to_sr).squeeze(0)


def mel_to_waveform(vocoder, mel: torch.Tensor) -> torch.Tensor:
    """Decode a ``[1, T, C]`` mel spectrogram to a 1-D CPU waveform."""
    with torch.no_grad():
        wav = vocoder.decode(mel.permute(0, 2, 1).float())
    return wav.squeeze().cpu().float()


def compute_rtf(gen_time_s: float, audio_duration_s: float) -> float:
    """Real-time factor: generation wall-clock over synthesized duration."""
    if audio_duration_s == 0:
        return float("inf")
    return gen_time_s / audio_duration_s
