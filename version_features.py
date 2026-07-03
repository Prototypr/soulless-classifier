"""Shared audio feature extractor for Suno-version attribution.

Imported by BOTH the training pipeline (build_version_features.py) and the
inference path (app.py), so training and serving compute identical features —
the train/serve-skew lesson from the SONICS normalisation bug.

The vector is a spectral "fingerprint" (log-mel + MFCC statistics + spectral
shape descriptors) computed on a fixed 30s crop at 16 kHz, chosen to match the
production input distribution (30s Deezer previews). It is deliberately
librosa-only (no torch/onnx) so it stays cheap and portable.
"""
from __future__ import annotations

import numpy as np

SR = 16000
CROP_SEC = 30
N_MELS = 64
N_MFCC = 20

# Fixed feature order; FEATURE_DIM is asserted so a mismatch fails loudly rather
# than silently mislabelling.
FEATURE_DIM = 2 * N_MELS + 2 * N_MFCC + 12 + 5  # mel(mean+std)+mfcc(mean+std)+chroma+shape


def _crop(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to SR and take a centred CROP_SEC window (pad if shorter)."""
    import librosa

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    n = CROP_SEC * SR
    if len(y) >= n:
        start = (len(y) - n) // 2
        y = y[start : start + n]
    else:
        y = np.pad(y, (0, n - len(y)))
    # Unit-peak normalise so loudness doesn't leak into the fingerprint.
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak).astype(np.float32)


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return a fixed-length FEATURE_DIM float32 fingerprint for one clip."""
    import librosa

    y = _crop(np.asarray(y, dtype=np.float32), sr)

    melspec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=2048, hop_length=512)
    logmel = librosa.power_to_db(melspec + 1e-10)
    mel_mean = logmel.mean(axis=1)
    mel_std = logmel.std(axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=SR).mean(axis=1)

    shape = np.array([
        float(librosa.feature.spectral_centroid(y=y, sr=SR).mean()),
        float(librosa.feature.spectral_bandwidth(y=y, sr=SR).mean()),
        float(librosa.feature.spectral_rolloff(y=y, sr=SR).mean()),
        float(librosa.feature.spectral_flatness(y=y).mean()),
        float(librosa.feature.zero_crossing_rate(y).mean()),
    ], dtype=np.float32)

    feat = np.concatenate([mel_mean, mel_std, mfcc_mean, mfcc_std, chroma, shape]).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    assert feat.shape[0] == FEATURE_DIM, f"{feat.shape[0]} != {FEATURE_DIM}"
    return feat
