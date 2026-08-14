#!/usr/bin/env python3
"""Build the LibriSpeech-PC test-clean cross-sentence evaluation set.

This is the F5-TTS official protocol: 1127 utterances, 4-10 s reference prompts,
each prompt drawn from a *different* utterance by the same speaker
(cross-sentence). Because F5-TTS, CosyVoice 2, MaskGCT, E2 TTS and VALL-E 2 all
report on this subset, using the published list verbatim is what makes the WER
numbers externally comparable.

Two inputs are required:

1. ``librispeech_pc_test_clean_cross_sentence.lst`` from the F5-TTS repository
   (``data/`` directory of https://github.com/SWivid/F5-TTS). Tab-separated:
   ``ref_id, ref_dur, ref_text, gen_id, gen_dur, gen_text``. The transcripts
   carry true case and punctuation, unlike stock LibriSpeech.
2. LibriSpeech ``test-clean`` audio (the standard OpenSLR release; PC only
   republishes transcripts). FLAC is decoded to WAV once here, since decoding
   FLAC inside the synthesis loop dominates its runtime.

Usage:
    python scripts/prepare_pc_data.py \\
        --lst-file /path/to/librispeech_pc_test_clean_cross_sentence.lst \\
        --librispeech-root /path/to/LibriSpeech/test-clean \\
        --config configs/librispeech_pc.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bon_tts.cli import build_parser  # noqa: E402
from bon_tts.config import apply_overrides, load_config, resolve_output_dir  # noqa: E402

EXPECTED_SAMPLES = 1127


def parse_lst(lst_path: Path) -> list[dict]:
    """Parse the official cross-sentence list."""
    entries = []
    with open(lst_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 6:
                raise ValueError(
                    f"{lst_path}:{line_no}: expected 6 tab-separated fields, got {len(fields)}"
                )
            ref_id, ref_dur, ref_text, gen_id, gen_dur, gen_text = fields
            entries.append(
                {
                    "ref_id": ref_id,
                    "ref_duration": float(ref_dur),
                    "ref_text": ref_text,
                    "gen_id": gen_id,
                    "gen_duration": float(gen_dur),
                    "gen_text": gen_text,
                }
            )
    return entries


def locate_flac(librispeech_root: Path, utterance_id: str) -> Path:
    """Map ``4992-41806-0009`` to ``4992/41806/4992-41806-0009.flac``."""
    speaker, chapter, _ = utterance_id.split("-")
    path = librispeech_root / speaker / chapter / f"{utterance_id}.flac"
    if not path.exists():
        raise FileNotFoundError(
            f"missing audio for {utterance_id}: {path}\n"
            "Check --librispeech-root points at the test-clean directory that "
            "contains per-speaker subdirectories."
        )
    return path


def convert_to_wav(flac_path: Path, wav_dir: Path) -> Path:
    """Decode one FLAC to WAV, skipping files already converted."""
    wav_path = wav_dir / f"{flac_path.stem}.wav"
    if not wav_path.exists():
        audio, sr = sf.read(str(flac_path), always_2d=False)
        sf.write(str(wav_path), audio, sr)
    return wav_path


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--lst-file", type=Path, required=True, help="Official cross-sentence .lst")
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        required=True,
        help="LibriSpeech test-clean directory (contains speaker subdirectories)",
    )
    parser.add_argument("--config", type=Path, default="configs/librispeech_pc.yaml")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    data_dir = resolve_output_dir(config, "data", config["data"]["name"])
    wav_dir = data_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_lst(args.lst_file)
    print(f"parsed {len(entries)} entries from {args.lst_file}")
    if len(entries) != EXPECTED_SAMPLES:
        print(
            f"warning: expected {EXPECTED_SAMPLES} entries for the official subset, "
            f"got {len(entries)} — WER will not be comparable with published numbers",
            file=sys.stderr,
        )

    metadata = []
    for idx, entry in enumerate(tqdm(entries, desc="Converting audio")):
        ref_wav = convert_to_wav(locate_flac(args.librispeech_root, entry["ref_id"]), wav_dir)
        gt_wav = convert_to_wav(locate_flac(args.librispeech_root, entry["gen_id"]), wav_dir)
        metadata.append(
            {
                "idx": idx,
                "speaker_id": entry["ref_id"].split("-")[0],
                "ref_audio": str(ref_wav),
                # Kept exactly as published: true case with punctuation. F5-TTS
                # expects this; lowercasing PC text measurably changes prosody.
                "ref_text": entry["ref_text"],
                "gen_text": entry["gen_text"],
                "gt_audio": str(gt_wav),
                "ref_duration": entry["ref_duration"],
                "gen_duration": entry["gen_duration"],
            }
        )

    meta_path = data_dir / "metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    unique_files = len(list(wav_dir.glob("*.wav")))
    print(f"wrote {len(metadata)} entries to {meta_path}")
    print(f"{unique_files} unique WAV files in {wav_dir}")


if __name__ == "__main__":
    main()
