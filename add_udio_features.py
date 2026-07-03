"""Append a `udio` class to data/version_features.npz from the SONICS dataset.

SONICS (ICLR 2025, awsaf49/sonics) ships 19,516 labelled Udio tracks inside
fake_songs/part_*.zip; filenames carry the generator ("fake_53278_udio_0"), so
one ~3.8GB part yields far more than a class worth. Files are extracted one at
a time from the zip (never unpacked wholesale), featurised with the same
version_features fingerprint, and appended.

  SONICS_ZIP=/path/part_01.zip PER_CLASS=300 ./.venv/bin/python add_udio_features.py
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import zipfile

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_features import FEATURE_DIM, extract_features  # noqa: E402

PER_CLASS = int(os.environ.get("PER_CLASS", "300"))
ZIP = os.environ.get("SONICS_ZIP", "sonics_part01.zip")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "version_features.npz")
random.seed(42)


def main():
    zf = zipfile.ZipFile(ZIP)
    udio = [n for n in zf.namelist() if "_udio_" in n and n.endswith(".mp3")]
    print(f"{ZIP}: {len(udio)} udio files available")
    random.shuffle(udio)

    X, ok = [], 0
    for name in udio:
        if ok >= PER_CLASS:
            break
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(zf.read(name))
                path = f.name
            try:
                y, sr = librosa.load(path, sr=16000, mono=True, duration=35)
            finally:
                os.remove(path)
            if len(y) < 16000:
                continue
            X.append(extract_features(y, sr))
            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{PER_CLASS}", flush=True)
        except Exception as e:
            print(f"  skip {name}: {e}")

    Xn = np.array(X, dtype=np.float32)
    yn = np.array(["udio"] * len(Xn))
    prev = np.load(OUT, allow_pickle=True)
    merged_X = np.concatenate([prev["X"], Xn])
    merged_y = np.concatenate([prev["y"], yn])
    np.savez(OUT, X=merged_X, y=merged_y, Xv4=prev.get("Xv4", np.zeros((0, FEATURE_DIM), np.float32)))
    from collections import Counter
    print("merged class counts:", dict(Counter(merged_y.tolist())))
    print(f"Saved -> {OUT}  (X={merged_X.shape})")


if __name__ == "__main__":
    main()
