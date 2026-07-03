"""
Soulless audio AI-detector sidecar.

A small FastAPI service that takes a 30s preview MP3 (by URL), extracts
spectral + temporal features with librosa, and returns a Human / Processed-AI /
Pure-AI verdict. The Next app's `classify:audio` cron calls POST /classify.

Two modes:
  • If a trained model file exists at MODEL_PATH (a joblib-saved scikit-learn
    classifier with predict_proba), it is used.
  • Otherwise a transparent BASELINE heuristic runs so the service works
    end-to-end out of the box. The baseline is a starting point, NOT
    authoritative — train a real model with train.py and mount it to upgrade.

Env:
  MODEL_PATH   path to a joblib model (default: ./model.joblib if present)
  PORT         default 8000
"""
import os

# Pin native thread pools to 1 BEFORE importing numpy/numba/librosa. Their
# multi-threaded backends (OpenBLAS/OMP/numba) can segfault when called from a
# request worker thread; single-threaded is plenty for 30s clips and stable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import tempfile

import librosa
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
# Ensemble of SONICS (ICLR 2025) SpecTTTra variants — MIT-licensed pretrained
# transformers. 5s variants match our 30s previews (we feed a 5s chunk). Each is
# a separate "detector" in the ensemble; comma-separate to add/remove variants.
# Set USE_SONICS=0 to disable. Different sizes (alpha/beta/gamma) decorrelate the
# errors a little; the ensemble's real diversity comes from the metadata signal
# (below) and the non-audio signals in the Next app's combined score.
SONICS_MODELS = [
    s.strip()
    for s in os.environ.get(
        "SONICS_MODELS",
        "awsaf49/sonics-spectttra-gamma-5s,awsaf49/sonics-spectttra-beta-5s",
    ).split(",")
    if s.strip()
]

app = FastAPI(title="Soulless audio classifier")

# Detector registry. Audio learned models (SONICS) + a metadata tag scan are
# combined per request; a local joblib model and the transparent baseline are
# fallbacks when SONICS isn't installed.
_sonics: list[tuple[str, object]] = []  # [(name, model)]
_torch = None
if os.environ.get("USE_SONICS", "1") != "0":
    try:
        import torch as _torch
        from sonics import HFAudioClassifier

        for name in SONICS_MODELS:
            try:
                m = HFAudioClassifier.from_pretrained(name)
                m.eval()
                _sonics.append((name.split("/")[-1], m))
                print(f"Loaded SONICS model: {name}")
            except Exception as e:
                print(f"  skip {name}: {e}")
    except Exception as e:  # torch/sonics missing entirely
        print(f"SONICS unavailable ({e}); falling back.")

# lofcz vocoder-fakeprint detector (MIT, ~15KB ONNX). Excellent at NEWER
# Suno/Udio: it keys on neural-vocoder upsampling artefacts rather than learned
# style, so it generalises where SONICS (trained on older data) under-fires.
_lofcz = None
_lofcz_in = None
_lofcz_feat = None
LOFCZ_MODEL = os.environ.get("LOFCZ_MODEL", "models/ai_music_detector.onnx")
if os.environ.get("USE_LOFCZ", "1") != "0" and os.path.exists(LOFCZ_MODEL):
    try:
        if _torch is None:
            import torch as _torch
        import onnxruntime as _ort
        import torchaudio as _ta
        from scipy.ndimage import minimum_filter1d as _minfilt

        _lofcz = _ort.InferenceSession(LOFCZ_MODEL)
        _lofcz_in = _lofcz.get_inputs()[0].name
        _lofcz_feat = _lofcz.get_inputs()[0].shape[1]
        # Fakeprint params (config.yaml of lofcz/ai-music-detector).
        _LOFCZ_SR, _LOFCZ_NFFT = 16000, 8192
        _LOFCZ_FMIN, _LOFCZ_FMAX = 1000, 8000
        _LOFCZ_HULL, _LOFCZ_MAXDB, _LOFCZ_MINDB = 10, 5.0, -45.0
        _lofcz_stft = _ta.transforms.Spectrogram(
            n_fft=_LOFCZ_NFFT, power=2, normalized=False
        )
        _fb = np.linspace(0, _LOFCZ_SR / 2, (_LOFCZ_NFFT // 2) + 1)
        _lofcz_fmask = (_fb >= _LOFCZ_FMIN) & (_fb <= _LOFCZ_FMAX)
        print(f"Loaded lofcz vocoder-fakeprint model: {LOFCZ_MODEL}")
    except Exception as e:
        _lofcz = None
        print(f"lofcz unavailable ({e}); skipping that detector.")

_model = None
if not _sonics and not _lofcz and os.path.exists(MODEL_PATH):
    import joblib

    _model = joblib.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")
elif not _sonics and not _lofcz:
    print("No model file found — using the transparent baseline heuristic.")


# --- Suno version-attribution head (open-set) -------------------------------
# Portable numpy forward pass exported by train_version.py: standardise → PCA →
# logistic regression → softmax. Class list + threshold come from the model file
# (never hard-coded), so adding a version later needs no change here. Runs only
# on tracks the detector already thinks are AI; low confidence → "uncertain".
_vhead = None
try:
    import json as _json
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _vnpz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "version_head.npz")
    _vjson = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "version_head.json")
    if os.path.exists(_vnpz) and os.path.exists(_vjson):
        _vp = np.load(_vnpz)
        with open(_vjson) as _f:
            _vmeta = _json.load(_f)
        _vhead = {
            "scaler_mean": _vp["scaler_mean"],
            "scaler_scale": _vp["scaler_scale"],
            "pca_mean": _vp["pca_mean"],
            "pca_components": _vp["pca_components"],
            "lr_coef": _vp["lr_coef"],
            "lr_intercept": _vp["lr_intercept"],
            "classes": list(_vmeta["classes"]),
            "threshold": float(_vmeta["threshold"]),
        }
        print(f"Loaded version head: {_vhead['classes']} (thr={_vhead['threshold']:.2f})")
except Exception as _e:
    print(f"version head not loaded: {_e}")
    _vhead = None


def _pretty_version(label: str) -> str:
    # "suno_v3_5" -> "Suno v3.5"; generator-level classes get plain names.
    if label.startswith("suno_v"):
        return "Suno v" + label[len("suno_v"):].replace("_", ".")
    if label == "udio":
        return "Udio"
    return label


def predict_version(y: np.ndarray, sr: int):
    """Open-set version guess: {label, confidence, uncertain, distribution}."""
    if _vhead is None:
        return None
    from version_features import extract_features as _vfeat

    x = _vfeat(y, sr).astype(np.float32)
    xs = (x - _vhead["scaler_mean"]) / _vhead["scaler_scale"]
    xp = (xs - _vhead["pca_mean"]) @ _vhead["pca_components"].T
    logits = xp @ _vhead["lr_coef"].T + _vhead["lr_intercept"]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    classes = _vhead["classes"]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    uncertain = conf < _vhead["threshold"]
    top = classes[idx]
    # Family fallback: no single version is confident, but nearly all the mass
    # sits inside one generator's classes → name the generator, not a shrug.
    label = "uncertain"
    if uncertain:
        suno_mass = sum(float(probs[i]) for i, c in enumerate(classes) if c.startswith("suno_"))
        if suno_mass >= 0.85:
            label = "Suno (version uncertain)"
    return {
        "label": label if uncertain else _pretty_version(top),
        "topVersion": _pretty_version(top),
        "confidence": round(conf, 4),
        "uncertain": uncertain,
        "distribution": {
            _pretty_version(classes[i]): round(float(probs[i]), 4)
            for i in range(len(classes))
        },
    }


def _active_model() -> str:
    parts = [n for n, _ in _sonics]
    if _lofcz:
        parts.append("lofcz")
    if parts:
        return "ensemble(" + "+".join(parts) + "+metadata)"
    return "trained" if _model else "baseline"


class ClassifyRequest(BaseModel):
    url: str


class ClassifyResponse(BaseModel):
    verdict: str  # 'human' | 'processed' | 'pure'
    aiProbability: float  # 0..1 (ensemble result)
    signals: dict = {}  # per-detector probabilities + metadata hits
    features: dict
    model: str  # e.g. 'ensemble(gamma-5s+beta-5s+metadata)' | 'trained' | 'baseline'
    likelySource: dict | None = None  # open-set version guess (AI tracks only)


def extract_features(y: np.ndarray, sr: int) -> dict:
    """Spectral + temporal descriptors used by both the model and baseline."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    flat = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # tempo moved around across librosa versions; try the known locations.
    tempo = 0.0
    for fn in (
        getattr(getattr(librosa.feature, "rhythm", None), "tempo", None),
        getattr(librosa.feature, "tempo", None),
        getattr(librosa.beat, "tempo", None),
    ):
        if fn is None:
            continue
        try:
            tempo = float(fn(onset_envelope=onset_env, sr=sr)[0])
            break
        except Exception:
            continue
    # Micro-timing / dynamics variability: humans wobble, generators are steadier.
    rms = librosa.feature.rms(y=y)
    return {
        "mfcc_mean": [float(x) for x in mfcc.mean(axis=1)],
        "mfcc_var": [float(x) for x in mfcc.var(axis=1)],
        "spectral_centroid_mean": float(cent.mean()),
        "spectral_flatness_mean": float(flat.mean()),
        "spectral_rolloff_mean": float(rolloff.mean()),
        "zcr_mean": float(zcr.mean()),
        "onset_var": float(onset_env.var()),
        "rms_var": float(rms.var()),
        "tempo": tempo,
    }


def baseline_probability(f: dict) -> float:
    """A transparent, weak heuristic so the service runs without a trained model.
    AI/generated audio tends to be over-smooth: low onset/dynamics variance and
    high spectral flatness. This is NOT accurate — replace with a trained model.
    """
    # Normalise a few signals into 0..1 "AI-ness" contributions.
    flat = min(1.0, f["spectral_flatness_mean"] / 0.10)  # high flatness → AI
    steadiness = 1.0 - min(1.0, f["onset_var"] / 5.0)  # low variance → AI
    dyn = 1.0 - min(1.0, f["rms_var"] / 0.01)  # low dynamics → AI
    prob = 0.45 * flat + 0.35 * steadiness + 0.20 * dyn
    return float(max(0.02, min(0.98, prob)))


def to_verdict(prob: float) -> str:
    if prob >= 0.80:
        return "pure"
    if prob >= 0.50:
        return "processed"
    return "human"


# SONICS window aggregation. A single fixed 5s chunk is a lottery: per-window
# P(AI) swings 0.05→0.73 within one 30s preview, and the old fixed 3/4-point
# chunk landed on the quietest window of the benchmark track (verdict "human"
# at 4% for a track every other window flagged). Sweep every 5s window instead
# and average the top-K — one strong window can't be hidden by a quiet outro,
# and one freak window can't flag a human track on its own.
SONICS_TOP_K = int(os.environ.get("SONICS_TOP_K", "3"))
_SONICS_WIN = 16000 * 5
_SONICS_STRIDE = _SONICS_WIN // 2  # 2.5s hop → ~12 windows per 30s preview


def _sonics_windows(y: np.ndarray, sr: int):
    """All 5s windows @16kHz, each unit-variance normalised (essential — without
    the std-normalisation the model reads everything as silence/human).
    Returns a [N, 80000] float tensor."""
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    if len(y16) < _SONICS_WIN:
        y16 = np.pad(y16, (0, _SONICS_WIN - len(y16)))
    starts = list(range(0, max(len(y16) - _SONICS_WIN, 0) + 1, _SONICS_STRIDE))
    if starts[-1] != len(y16) - _SONICS_WIN:  # cover the tail
        starts.append(len(y16) - _SONICS_WIN)
    wins = []
    for s in starts:
        seg = y16[s : s + _SONICS_WIN]
        wins.append(seg / max(float(np.std(seg)), 1e-6))
    return _torch.from_numpy(np.ascontiguousarray(np.stack(wins))).float()


def sonics_probs(y: np.ndarray, sr: int) -> dict:
    """Sweep every loaded SONICS variant over all 5s windows. Per model, the
    score is the mean of its top-K windows; window stats are reported too."""
    x = _sonics_windows(y, sr)
    out = {}
    fracs = []
    with _torch.no_grad():
        for name, model in _sonics:
            probs = _torch.sigmoid(model(x)).flatten()
            k = min(SONICS_TOP_K, probs.numel())
            out[name] = float(probs.topk(k).values.mean().item())
            out[f"{name}_win_max"] = float(probs.max().item())
            out[f"{name}_win_mean"] = float(probs.mean().item())
            fracs.append(float((probs >= 0.5).float().mean().item()))
    # Share of the clip's windows that read as AI, averaged across models.
    # A partial/hybrid edit (AI hook in a human track, or vice versa) shows up
    # here as a mid fraction — some windows hot, the rest cold — where a fully
    # generated song trends toward 1.0 and a human one toward 0.0.
    out["sonics_win_frac"] = sum(fracs) / len(fracs) if fracs else 0.0
    out["sonics_windows"] = float(x.shape[0])
    return out


def lofcz_probability(y: np.ndarray, sr: int) -> float:
    """lofcz fakeprint → tiny ONNX classifier. STFT → mean dB spectrum → subtract
    its lower hull (minimum filter) over the 1–8kHz band → normalise → model.
    Mirrors vendor/lofcz/src/python/inference.py exactly. Returns P(AI)."""
    y16 = (
        librosa.resample(y, orig_sr=sr, target_sr=_LOFCZ_SR)
        if sr != _LOFCZ_SR
        else y
    )
    audio = _torch.from_numpy(np.ascontiguousarray(y16)).unsqueeze(0)
    spec = _lofcz_stft(audio)
    spec_db = 10 * _torch.log10(_torch.clamp(spec, min=1e-10, max=1e6))
    mean_spectrum = spec_db.mean(dim=(0, 2)).numpy()
    fs = mean_spectrum[_lofcz_fmask]
    hull = np.clip(_minfilt(fs, size=_LOFCZ_HULL, mode="nearest"), _LOFCZ_MINDB, None)
    residue = np.clip(np.clip(fs - hull, 0, None), 0, _LOFCZ_MAXDB)
    fp = (residue / (float(residue.max()) + 1e-6)).astype(np.float32)
    if len(fp) != _lofcz_feat:
        fp = np.interp(
            np.linspace(0, 1, _lofcz_feat), np.linspace(0, 1, len(fp)), fp
        ).astype(np.float32)
    return float(_lofcz.run(None, {_lofcz_in: fp.reshape(1, -1)})[0][0, 0])


# Generator fingerprints that AI tools often leave in file tags/metadata. On
# Deezer previews these are usually stripped, but it's a free, near‑zero‑FP
# positive when present (e.g. classifying an original upload).
_GEN_TAGS = ("suno", "udio", "musicgen", "audioldm", "stable audio", "riffusion", "mubert")


def metadata_probability(raw: bytes) -> tuple:
    """Scan the raw file bytes for AI‑generator strings. Returns (prob, hits)."""
    head = raw[:4096].lower() + raw[-2048:].lower()
    hits = [t for t in _GEN_TAGS if t.encode() in head]
    return (0.97 if hits else 0.0, hits)


def noisy_or(probs) -> float:
    """Probabilistic OR: any confident positive pushes the result up, and a low
    score never drags a high one down (the asymmetry we want — SONICS missing a
    newer‑Suno track shouldn't veto the metadata/other signals)."""
    keep = 1.0
    for p in probs:
        keep *= 1.0 - max(0.0, min(1.0, p))
    return 1.0 - keep


# Operating-point calibration. Benchmarked on real Suno v5.5 (Kukedlc dataset)
# vs human artists: a raw cutoff of 0.65 gave 100% recall + 0% false positives
# (a few heavily-electronic human tracks sat at 0.5–0.64). This logistic recentre
# maps raw 0.65 → 0.5 so the standard verdict thresholds (0.5/0.8) become that
# optimal operating point. Tune via CALIB_PIVOT / CALIB_SLOPE.
import math

CALIB_PIVOT = float(os.environ.get("CALIB_PIVOT", "0.65"))
CALIB_SLOPE = float(os.environ.get("CALIB_SLOPE", "9.0"))

# Corroboration knobs (see the ensemble comment in classify()). lofcz at or
# above LOFCZ_CORROB (or a metadata hit) unlocks SONICS at full strength;
# otherwise SONICS is remapped from [SOLO_MIN, SOLO_MAX] onto [0, 0.95].
LOFCZ_CORROB = float(os.environ.get("LOFCZ_CORROB", "0.5"))
LOFCZ_FLOOR = float(os.environ.get("LOFCZ_FLOOR", "0.75"))
SONICS_SOLO_MIN = float(os.environ.get("SONICS_SOLO_MIN", "0.55"))
SONICS_SOLO_MAX = float(os.environ.get("SONICS_SOLO_MAX", "0.95"))


def calibrate(p: float) -> float:
    return 1.0 / (1.0 + math.exp(-CALIB_SLOPE * (p - CALIB_PIVOT)))


@app.get("/health")
def health():
    return {"ok": True, "model": _active_model()}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    try:
        resp = requests.get(req.url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"download failed: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        try:
            y, sr = librosa.load(tmp.name, sr=22050, mono=True, duration=30)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"decode failed: {e}")

    if y.size == 0:
        raise HTTPException(status_code=422, detail="empty audio")

    feats = extract_features(y, sr)
    signals: dict = {}

    if _sonics or _lofcz:
        # Ensemble of audio detectors + metadata, combined with noisy-OR: any
        # confident positive pushes the result up and a low score never vetoes a
        # high one (so SONICS missing newer Suno can't suppress lofcz/metadata).
        # The transparent heuristic is reported but kept OUT of the OR (too noisy
        # → false positives).
        #
        # Corroboration rule (benchmarked 2026-07-02 on 15 human + 8 confirmed-AI
        # Deezer previews): SONICS is the false-positive-prone detector — heavily
        # produced human tracks (reggaeton/EDM) reach 0.5-0.7 while lofcz reads
        # 0.00 on EVERY human track and ~1.0 on confirmed AI. So a moderate
        # SONICS score only enters the OR at full strength when lofcz or
        # metadata corroborates it; uncorroborated SONICS is remapped so it must
        # be extremely confident (~0.82+) to cross into an AI verdict on its own.
        ors = []
        lp = None
        meta_prob, meta_hits = metadata_probability(resp.content)
        if _lofcz:
            lp = lofcz_probability(y, sr)
            signals["lofcz"] = round(lp, 4)
            # lofcz's own decision boundary is 0.5 = AI, and no human track in
            # the benchmark has ever read above 0.00 — so once it crosses its
            # boundary, floor the contribution at LOFCZ_FLOOR. Otherwise a mid
            # reading (e.g. 0.59, SONICS blind) dies against the 0.65 pivot.
            ors.append(max(lp, LOFCZ_FLOOR) if lp >= LOFCZ_CORROB else lp)
        if _sonics:
            sp = sonics_probs(y, sr)
            signals.update({k: round(v, 4) for k, v in sp.items()})
            # Mean across model variants of their top-K-window scores (the
            # extra *_win_* / sonics_windows keys are stats, not detectors).
            model_scores = [sp[name] for name, _ in _sonics]
            sonics_mean = sum(model_scores) / len(model_scores)
            signals["sonics_mean"] = round(sonics_mean, 4)
            corroborated = (lp is not None and lp >= LOFCZ_CORROB) or meta_prob > 0
            if corroborated:
                sonics_eff = sonics_mean
            else:
                # Uncorroborated: judge by the STRONGEST variant, not the mean.
                # Post-processed AI can silence lofcz entirely while one SONICS
                # variant stays very confident (Nightwhisper: beta 0.85, gamma
                # 0.47 → mean 0.66 was squashed to "human"); a human track has
                # never pushed a single variant past ~0.74 in the benchmark
                # (Beéle 0.74 → still well under the verdict line here).
                sonics_solo = max(model_scores)
                sonics_eff = (
                    max(0.0, min(1.0, (sonics_solo - SONICS_SOLO_MIN) / (SONICS_SOLO_MAX - SONICS_SOLO_MIN)))
                    * 0.95
                )
            signals["sonics_corroborated"] = 1.0 if corroborated else 0.0
            signals["sonics_effective"] = round(sonics_eff, 4)
            ors.append(sonics_eff)

        signals["metadata"] = meta_prob
        if meta_hits:
            signals["metadata_hits"] = meta_hits
        ors.append(meta_prob)
        signals["heuristic"] = round(baseline_probability(feats), 4)

        raw = noisy_or(ors)
        signals["ensemble_raw"] = round(raw, 4)
        prob = calibrate(raw)  # recentre to the benchmarked 0.65 operating point
        model_name = _active_model()
    elif _model is not None:
        # Trained model: flatten features in the same order train.py uses.
        vec = (
            feats["mfcc_mean"]
            + feats["mfcc_var"]
            + [
                feats["spectral_centroid_mean"],
                feats["spectral_flatness_mean"],
                feats["spectral_rolloff_mean"],
                feats["zcr_mean"],
                feats["onset_var"],
                feats["rms_var"],
                feats["tempo"],
            ]
        )
        prob = float(_model.predict_proba([vec])[0][1])
        signals["trained"] = round(prob, 4)
        model_name = "trained"
    else:
        prob = baseline_probability(feats)
        signals["baseline"] = round(prob, 4)
        model_name = "baseline"

    # Version attribution only makes sense once we think it's AI. Low-confidence
    # guesses come back as "uncertain" (e.g. a v4 track we have no data for).
    verdict = to_verdict(prob)
    likely_source = None
    if verdict != "human":
        try:
            likely_source = predict_version(y, sr)
        except Exception:
            likely_source = None

    return ClassifyResponse(
        verdict=verdict,
        aiProbability=prob,
        signals=signals,
        features=feats,
        model=model_name,
        likelySource=likely_source,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
