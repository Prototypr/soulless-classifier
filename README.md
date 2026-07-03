# Soulless audio AI-detector

The open-source classifier behind [soullessmusic.com](https://soullessmusic.com)'s
"Is this song AI?" detector. A small FastAPI service: give it an audio URL
(a 30-second store preview is enough), it returns a **Human / Processed-AI /
Pure-AI** verdict, per-detector signals, and an open-set guess at which
generator made it (Suno v2–v5.5, Udio).

We build this in the open — every miss, fix and benchmark is written up in the
[build log](https://soullessmusic.com/articles).

## How it works

Three detectors, combined with a corroboration-aware noisy-OR:

- **SONICS** ([ICLR 2025](https://github.com/awsaf49/sonics), MIT) — two
  SpecTTTra spectrogram-transformer variants, swept over every 5s window of
  the clip (2.5s hop); each model scored as the mean of its top-3 windows.
  Style-based, so it can be tempted by heavily-produced human music —
  which is why it never convicts alone.
- **Vocoder fakeprint** ([lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector),
  MIT, ~15 KB ONNX) — hears the periodic 1–8 kHz reconstruction artefacts
  neural vocoders leave behind. Physics, not style: it reads 0.00 on every
  human track in our benchmark, and ~1.0 on normally-encoded AI.
- **Metadata scan** — generator strings (suno, udio, …) left in file tags.

**Corroboration rules** (each one exists because of a documented real-world
miss or false positive):
- SONICS enters the OR at full strength only when the fakeprint (≥0.5) or a
  metadata hit corroborates it; uncorroborated, its **strongest variant**
  must be extremely confident (~0.82+) to matter.
- A fakeprint reading past its own 0.5 decision boundary is floored at 0.75 —
  humans read 0.00, so a mid reading is already damning.
- The result is recalibrated so the standard 0.5/0.8 verdict thresholds sit
  at the benchmarked operating point.

A **version head** (standardise → PCA → logistic regression, exported as a
plain numpy forward pass) attributes AI verdicts to a generator/version:
Suno v2, v3.x, v4, v4.5, v5.5, or Udio — trained exclusively on AI-generated
audio from the generators' own public CDNs and published research datasets.
Low confidence returns "uncertain" (or "Suno (version uncertain)" when the
family is clear but the version isn't) rather than forcing a wrong label.

## API

```
POST /classify   { "url": "<audio url>" }
→ { "verdict": "pure", "aiProbability": 0.91,
    "signals": { "sonics_mean": 0.65, "lofcz": 0.73, "sonics_win_frac": 0.27, ... },
    "likelySource": { "label": "Suno v4.5", "confidence": 0.86, ... },
    "model": "ensemble(...)" }

GET  /health     → { "ok": true, "model": "ensemble(...)" }
```

`sonics_win_frac` is the share of 5s windows that individually read as AI —
a partial/hybrid-content indicator (an AI hook inside a human track lands in
the middle).

## Run it

```bash
docker compose up -d --build     # http://localhost:8000
# or:
pip install -r requirements.txt -r requirements-sonics.txt
uvicorn app:app --port 8000
```

First boot downloads the SONICS weights (~2.9 GB) from Hugging Face into
`HF_HOME`; everything after that runs fully offline, CPU-only. We run it on a
Mac mini.

## Retraining the version head

```bash
PER_CLASS=300 python build_version_features.py   # harvest + featurise
python add_udio_features.py                      # optional: Udio class from SONICS zips
python train_version.py                          # → models/version_head.{npz,json}
```

Classes are derived from labels — add a new source in
`build_version_features.py` and the head grows a class on the next train.

## Tuning knobs (env)

| Var | Default | |
|---|---|---|
| `SONICS_MODELS` | gamma-5s,beta-5s | comma-separated HF model ids |
| `SONICS_TOP_K` | 3 | windows averaged per model |
| `LOFCZ_CORROB` | 0.5 | fakeprint level that unlocks SONICS |
| `LOFCZ_FLOOR` | 0.75 | floor once fakeprint crosses its boundary |
| `SONICS_SOLO_MIN/MAX` | 0.55 / 0.95 | uncorroborated SONICS remap range |
| `CALIB_PIVOT/SLOPE` | 0.65 / 9.0 | operating-point calibration |

## Ethics

- Human music is never stored and never trained on: previews are streamed,
  scored, deleted. Training data is AI-generated audio only.
- Everything runs on your own hardware — no audio is sent to any third party.
- A verdict is an indicator, not an accusation. Keep humans in the loop.

## Credits

[SONICS](https://github.com/awsaf49/sonics) (MIT) ·
[lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector) (MIT).
Thank you for publishing your work openly — this project exists because you did.
