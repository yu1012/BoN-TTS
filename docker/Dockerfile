# Reproduction environment for BoN-TTS.
#
# Base image supplies torch + torchaudio built against CUDA 12.4 and, critically,
# a system-wide cuDNN 9 — CTranslate2 (which backs the official faster-whisper
# evaluator) loads cuDNN by soname at runtime and fails if it is only present
# inside pip's nvidia-* wheels.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep model downloads on a mounted volume so rebuilds don't refetch ~10 GB
    # of checkpoints and so the container never writes them into the image.
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    BON_TTS_OUTPUT_DIR=/outputs

# ffmpeg: audio decoding for torchaudio/librosa fallbacks.
# libsndfile1: soundfile backend.
# git: f5-tts pulls a git dependency chain during install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependency layer first: source edits then rebuild in seconds, not minutes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # Test and lint tooling, so `make test` / `make lint` work in the image the
    # reproduction actually runs in.
    && pip install --no-cache-dir "pytest>=8.0" "ruff>=0.6"

COPY pyproject.toml README.md ./
COPY bon_tts ./bon_tts
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

# Fail fast on a broken image rather than at the end of a long run: import the
# package, the F5-TTS symbols f5_backend needs, and the CTranslate2 evaluator.
RUN python -c "\
import bon_tts, torch, transformers, faster_whisper;\
from f5_tts.model.cfm import get_epss_timesteps;\
from f5_tts.api import F5TTS;\
print('bon-tts', bon_tts.__version__);\
print('torch', torch.__version__, 'transformers', transformers.__version__)"

CMD ["/bin/bash"]
