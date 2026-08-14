"""Best-of-N zero-shot TTS: verifiers, cross-family rank ensembles, evaluation.

Reference implementation for "Best-of-N TTS Evaluation is Confounded by ASR
Family Alignment". See the README for the reproduction pipeline.
"""

from bon_tts.selection import STRATEGIES, recovery_rate, select, select_oracle

__version__ = "0.1.0"

__all__ = ["STRATEGIES", "select", "select_oracle", "recovery_rate", "__version__"]
