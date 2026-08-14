"""Text normalization used by the F5-TTS official evaluation pipeline.

The published F5-TTS evaluation code (``src/f5_tts/eval/utils_eval.py::run_asr_wer``)
normalizes both the reference and the hypothesis by *deleting* every punctuation
character, collapsing the double spaces that deletion leaves behind, and
lowercasing. Deletion — rather than replacement with a space — is why
``"proceeded-"`` becomes ``proceeded`` and ``'me"!'`` becomes ``me``.

We reproduce that behaviour verbatim so our WER numbers are directly comparable
with the F5-TTS, CosyVoice 2, MaskGCT and E2 TTS papers, all of which report on
this pipeline. See ``docs/REPRODUCE.md`` for the bit-exact check against the
reference artifacts.
"""

from __future__ import annotations

import string

from zhon.hanzi import punctuation as _CJK_PUNCTUATION

# F5-TTS concatenates ASCII punctuation with ``zhon.hanzi.punctuation``. We take
# the set from zhon itself rather than transcribing it, so this is identical to
# upstream by construction. The CJK half rarely fires on English text, but it
# does cover the Unicode dashes and curly quotes Whisper emits (U+2013, U+2014,
# U+2018/9, U+201C/D) — which is exactly where a hand-copied set goes wrong.
PUNCTUATION = string.punctuation + _CJK_PUNCTUATION


def normalize_en(text: str) -> str:
    """Apply the F5-TTS official English normalization to ``text``.

    Deletes punctuation, collapses the resulting double spaces, lowercases and
    strips. Returns a single-space-delimited string ready for WER scoring.
    """
    for char in PUNCTUATION:
        text = text.replace(char, "")
    text = text.replace("  ", " ")
    return " ".join(text.lower().split())
