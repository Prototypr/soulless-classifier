# Audio AI-detector sidecar for Coolify. Build context = ./classifier
# Ships the full ensemble: SONICS (HF, downloaded at first boot) + lofcz
# vocoder-fakeprint (models/ai_music_detector.onnx) + the open-set Suno version
# head (models/version_head.*). Falls back to the baseline only if a dep/model
# is missing.
FROM python:3.12-slim

# ffmpeg + libsndfile are needed by librosa/audioread to decode MP3s; git to
# pip-install the sonics package from GitHub.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-sonics.txt ./
# The build host's IPv6 path to PyPI drops TLS mid-handshake (SSL EOF), which
# stalls pip. Prefer IPv4 in glibc's resolver so pip reaches PyPI reliably.
RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf
# Base stack, then the ensemble stack (torch/torchaudio/onnxruntime/sonics).
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt \
  && pip install --no-cache-dir --retries 5 --timeout 300 -r requirements-sonics.txt \
  && pip install --no-cache-dir huggingface_hub

# Copies app.py, version_features.py, and models/ (lofcz onnx + version head).
COPY . .

ENV PORT=8000 \
    USE_LOFCZ=1 \
    LOFCZ_MODEL=/app/models/ai_music_detector.onnx \
    SONICS_MODELS=awsaf49/sonics-spectttra-gamma-5s,awsaf49/sonics-spectttra-beta-5s \
    # SONICS (~2.9 GB) downloads here on first boot — mount a persistent volume
    # at /app/.hf-cache in Coolify so it survives restarts.
    HF_HOME=/app/.hf-cache
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
