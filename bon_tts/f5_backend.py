"""F5-TTS candidate generation.

We drive the flow-matching ODE ourselves instead of calling ``F5TTS.infer`` once
per candidate. The conditioning tensors (reference mel, tokenized text, duration)
depend only on the utterance, so preparing them once and reusing them across the
N candidates removes the redundant preprocessing that N separate ``infer`` calls
would repeat. The ODE loop itself mirrors ``f5_tts.model.cfm``: EPSS timesteps,
optional sway sampling, and classifier-free guidance via the transformer's fused
``cfg_infer`` path.

Candidates differ only in the seed of the initial noise, which is the diversity
source Best-of-N selects over.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torchaudio

from bon_tts.audio import TTS_SAMPLE_RATE, load_audio, mel_to_waveform

logger = logging.getLogger(__name__)

HOP_LENGTH = 256


def candidate_seeds(n: int, base_seed: int = 42, seed_offset: int = 1000) -> list[int]:
    """Seeds for ``n`` candidates: ``42, 1042, 2042, ...`` with the defaults.

    Candidate 0 uses ``base_seed``, so it is bit-identical to the single-shot
    baseline sample. Every BoN result is therefore measured against a baseline
    that is literally inside its own candidate pool.
    """
    return [base_seed + i * seed_offset for i in range(n)]


def load_f5tts(
    model_type: str = "F5TTS_v1_Base",
    ckpt_file: str = "",
    vocab_file: str = "",
    ode_method: str = "euler",
    device: str = "cuda",
):
    """Load the F5-TTS API object (downloads the public checkpoint if unset)."""
    from f5_tts.api import F5TTS

    return F5TTS(
        model=model_type,
        ckpt_file=ckpt_file or "",
        vocab_file=vocab_file or "",
        ode_method=ode_method,
        device=device,
    )


@dataclass
class Candidate:
    """One synthesized candidate."""

    seed: int
    wav: torch.Tensor  # 1-D waveform at TTS_SAMPLE_RATE
    gen_time_s: float


@dataclass
class UtteranceContext:
    """Conditioning tensors shared by every candidate of one utterance."""

    cond_padded: torch.Tensor
    cond_mask_padded: torch.Tensor
    step_cond: torch.Tensor
    text: torch.Tensor
    ref_audio_len: int
    n_channels: int
    max_duration: int
    rms: float
    prep_time_s: float = field(default=0.0)


class F5CandidateGenerator:
    """Generates seed-diverse F5-TTS candidates for one utterance at a time."""

    def __init__(
        self,
        tts_api,
        total_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float | None = -1.0,
        target_rms: float = 0.1,
        speed: float = 1.0,
    ):
        self.tts = tts_api
        self.device = tts_api.device
        self.total_steps = total_steps
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.target_rms = target_rms
        self.speed = speed

    def _timesteps(self, dtype: torch.dtype) -> torch.Tensor:
        """EPSS timestep schedule with optional sway sampling."""
        from f5_tts.model.cfm import get_epss_timesteps

        t = get_epss_timesteps(self.total_steps, device=self.device, dtype=dtype)
        if self.sway_sampling_coef is not None:
            t = t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        return t

    def prepare(self, ref_file: str, ref_text: str, gen_text: str) -> UtteranceContext:
        """Build the conditioning tensors shared across candidates."""
        from f5_tts.infer.utils_infer import convert_char_to_pinyin, preprocess_ref_audio_text
        from f5_tts.model.utils import exists, lens_to_mask, list_str_to_idx, list_str_to_tensor

        t0 = time.time()
        cfm = self.tts.ema_model
        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        audio, sr = load_audio(ref_file, target_sr=None)
        rms_val = float(torch.sqrt(torch.mean(torch.square(audio))).item())
        if rms_val < self.target_rms:
            audio = audio * self.target_rms / rms_val
        if sr != TTS_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, TTS_SAMPLE_RATE)
        audio = audio.to(self.device)

        # F5-TTS concatenates reference and target text; a 1-byte final character
        # means the reference has no trailing separator yet.
        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        final_text_list = convert_char_to_pinyin([ref_text + gen_text])

        ref_audio_len = audio.shape[-1] // HOP_LENGTH
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(
            ref_audio_len / max(ref_text_len, 1) * gen_text_len / self.speed
        )

        cfm.eval()
        cond = cfm.mel_spec(audio).permute(0, 2, 1)
        cond = cond.to(next(cfm.parameters()).dtype)
        batch, cond_seq_len = cond.shape[:2]
        n_channels = cond.shape[-1]

        lens = torch.full((batch,), cond_seq_len, device=self.device, dtype=torch.long)
        cond_mask = lens_to_mask(lens)

        if exists(cfm.vocab_char_map):
            text = list_str_to_idx(final_text_list, cfm.vocab_char_map).to(self.device)
        else:
            text = list_str_to_tensor(final_text_list).to(self.device)

        duration_t = torch.full((batch,), duration, device=self.device, dtype=torch.long)
        duration_t = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration_t
        )
        max_duration = int(duration_t.amax().item())

        cond_padded = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask_padded = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask_padded = cond_mask_padded.unsqueeze(-1)
        step_cond = torch.where(cond_mask_padded, cond_padded, torch.zeros_like(cond_padded))

        return UtteranceContext(
            cond_padded=cond_padded,
            cond_mask_padded=cond_mask_padded,
            step_cond=step_cond,
            text=text,
            ref_audio_len=ref_audio_len,
            n_channels=n_channels,
            max_duration=max_duration,
            rms=rms_val,
            prep_time_s=time.time() - t0,
        )

    @torch.no_grad()
    def _solve_ode(self, ctx: UtteranceContext, seed: int, t_all: torch.Tensor) -> torch.Tensor:
        """Run the full ODE from seeded noise and return the generated mel."""
        cfm = self.tts.ema_model

        torch.manual_seed(seed)
        y_t = torch.randn(
            ctx.max_duration, ctx.n_channels, device=self.device, dtype=ctx.step_cond.dtype
        ).unsqueeze(0)

        for i in range(self.total_steps):
            t_batch = t_all[i].unsqueeze(0)
            dt = (t_all[i + 1] - t_all[i]).item()

            if self.cfg_strength >= 1e-5:
                # Fused conditional/unconditional pass: one transformer call
                # returns both branches stacked on the batch dimension.
                pred_cfg = cfm.transformer(
                    x=y_t, cond=ctx.step_cond, text=ctx.text, time=t_batch,
                    mask=None, cfg_infer=True, cache=True,
                )
                pred_cond, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                velocity = pred_cond + (pred_cond - null_pred) * self.cfg_strength
            else:
                velocity = cfm.transformer(
                    x=y_t, cond=ctx.step_cond, text=ctx.text, time=t_batch,
                    mask=None, drop_audio_cond=False, drop_text=False, cache=True,
                )

            y_t = y_t + velocity * dt

        cfm.transformer.clear_cache()

        out = torch.where(ctx.cond_mask_padded, ctx.cond_padded, y_t)
        return out[:, ctx.ref_audio_len:, :]

    def generate(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        seeds: Sequence[int],
        ctx: UtteranceContext | None = None,
    ) -> list[Candidate]:
        """Synthesize one candidate per seed.

        Args:
            ctx: reuse a context from :meth:`prepare`; built here when omitted.
        """
        if ctx is None:
            ctx = self.prepare(ref_file, ref_text, gen_text)

        t_all = self._timesteps(ctx.step_cond.dtype)
        candidates = []

        for seed in seeds:
            t0 = time.time()
            gen_mel = self._solve_ode(ctx, seed, t_all)
            wav = mel_to_waveform(self.tts.vocoder, gen_mel)
            # Undo the input gain applied to a quiet reference so candidate
            # loudness matches the original recording.
            if ctx.rms < self.target_rms:
                wav = wav * ctx.rms / self.target_rms
            candidates.append(Candidate(seed=seed, wav=wav, gen_time_s=time.time() - t0))

        return candidates

    @property
    def nfe_per_candidate(self) -> int:
        """Function evaluations per candidate (CFG needs no extra call here)."""
        return self.total_steps
