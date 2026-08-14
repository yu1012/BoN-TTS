#!/usr/bin/env python3
"""Synthesize one utterance with Best-of-N selection.

The practical entry point: give it a reference clip, that clip's transcript, and
the text to speak. Use ``--strategy max_rank`` with two verifiers from different
ASR families for the recommended configuration.

Usage:
    # single-verifier BoN
    python scripts/infer.py --ref-audio prompt.wav --ref-text "..." \\
        --gen-text "Text to synthesize." --n 5 --out out.wav

    # cross-family rank ensemble (recommended)
    python scripts/infer.py --ref-audio prompt.wav --ref-text "..." \\
        --gen-text "Text to synthesize." --n 5 --out out.wav \\
        --strategy max_rank --verifiers w2v2-base distil-v3
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.audio import save_audio  # noqa: E402
from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.f5_backend import F5CandidateGenerator, load_f5tts  # noqa: E402
from bon_tts.pipeline import bon_infer  # noqa: E402
from bon_tts.selection import STRATEGIES  # noqa: E402
from bon_tts.verifiers import VERIFIER_ALIASES, load_verifiers  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference (prompt) audio file")
    parser.add_argument("--ref-text", type=str, required=True, help="Transcript of the reference audio")
    parser.add_argument("--gen-text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--out", type=Path, required=True, help="Output WAV path")
    parser.add_argument("--n", type=int, default=5, help="Number of candidates")
    parser.add_argument("--strategy", type=str, default="single", choices=list(STRATEGIES))
    parser.add_argument(
        "--verifiers", type=str, nargs="+", default=["distil-v3"],
        help=f"Verifier aliases (known: {', '.join(sorted(VERIFIER_ALIASES))})",
    )
    parser.add_argument("--metric", type=str, default="wer_cer", choices=["wer", "cer", "wer_cer"])
    parser.add_argument("--steps", type=int, default=32, help="ODE solver steps")
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42, help="Seed of candidate 0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-candidates", type=Path, default=None,
                        help="Directory to also write every candidate for inspection")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    tts = load_f5tts(device=device)
    generator = F5CandidateGenerator(
        tts_api=tts,
        total_steps=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
    )
    verifiers = load_verifiers(args.verifiers, device=device, metric=args.metric)

    result = bon_infer(
        generator=generator,
        verifiers=verifiers,
        ref_file=args.ref_audio,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        n_candidates=args.n,
        strategy=args.strategy,
        base_seed=args.seed,
    )

    save_audio(result.wav, args.out, sr=result.sample_rate)

    if args.save_candidates:
        # Re-running with the same seeds reproduces the candidates bit-exactly.
        from bon_tts.f5_backend import candidate_seeds

        seeds = candidate_seeds(args.n, base_seed=args.seed)
        candidates = generator.generate(args.ref_audio, args.ref_text, args.gen_text, seeds)
        for i, candidate in enumerate(candidates):
            save_audio(candidate.wav, args.save_candidates / f"candidate_{i}_seed{candidate.seed}.wav")
        logger.info("wrote %d candidates to %s", len(candidates), args.save_candidates)

    print(f"\nselected candidate {result.best_idx} (seed {result.best_seed}) of {result.n_candidates}")
    for verifier, row in zip(args.verifiers, result.scores, strict=True):
        print(f"  {verifier:<12} " + " ".join(f"{s:.4f}" for s in row))
    print(f"RTF {result.rtf:.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
