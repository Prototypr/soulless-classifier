"""
Train a real AI-vs-human audio classifier to replace the baseline heuristic.

This is a starting point. You provide a labelled dataset; it extracts the same
features app.py uses and fits a classifier, saving model.joblib. Mount that file
into the container (or bake it in) and the service switches from baseline →
trained automatically.

Dataset layout:
    data/human/*.mp3     # real human recordings
    data/ai/*.mp3        # known AI output (Suno / Udio / etc.)

Run:
    pip install -r requirements.txt
    python train.py
"""
import glob
import os

import joblib
import librosa
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app import extract_features  # reuse the exact same feature extraction


def feature_vector(path: str) -> list[float]:
    y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
    f = extract_features(y, sr)
    return (
        f["mfcc_mean"]
        + f["mfcc_var"]
        + [
            f["spectral_centroid_mean"],
            f["spectral_flatness_mean"],
            f["spectral_rolloff_mean"],
            f["zcr_mean"],
            f["onset_var"],
            f["rms_var"],
            f["tempo"],
        ]
    )


def main():
    X, y = [], []
    for label, folder in [(0, "data/human"), (1, "data/ai")]:
        files = glob.glob(os.path.join(folder, "*.mp3"))
        print(f"{folder}: {len(files)} files")
        for p in files:
            try:
                X.append(feature_vector(p))
                y.append(label)
            except Exception as e:
                print(f"  skip {p}: {e}")

    if len(set(y)) < 2:
        raise SystemExit("Need both human and ai examples. Add files to data/.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier()
    clf.fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte)))

    joblib.dump(clf, "model.joblib")
    print("Saved model.joblib — restart the service to use it.")


if __name__ == "__main__":
    main()
