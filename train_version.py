"""Train the open-set Suno-version attribution head.

Reads the feature matrix from build_version_features.py, trains a version
classifier, calibrates an "uncertain" threshold using the held-out v4 tracks
(a version with no training data), and exports a PORTABLE model — plain numpy
arrays + a small JSON — so the inference path (app.py) needs no sklearn and
isn't coupled to a pickled sklearn version.

Class list is derived entirely from the data (`y`), so adding a new version
later = add its vectors + retrain; nothing downstream is hard-coded.

  ./.venv/bin/python train_version.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "version_features.npz")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)

# Merge near-identical generations into coarser, more honest buckets. v3 and v3.5
# are the same era and blur together in feature space — "v3.x" is both more
# accurate and more honest than forcing a v3-vs-v3.5 call. Set MERGE={} to keep
# every version separate.
MERGE = {"suno_v3": "suno_v3_x", "suno_v3_5": "suno_v3_x"}


def main():
    d = np.load(DATA, allow_pickle=True)
    X, y, Xv4 = d["X"], d["y"], d["Xv4"]
    y = np.array([MERGE.get(v, v) for v in y.tolist()])
    classes = sorted(set(y.tolist()))
    print(f"X={X.shape}  classes={classes}  v4_holdout={len(Xv4)}")

    n_comp = int(min(40, X.shape[1], X.shape[0] - 1))
    scaler = StandardScaler()
    pca = PCA(n_components=n_comp, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    Xs = scaler.fit_transform(X)
    Xp = pca.fit_transform(Xs)

    # Cross-validated accuracy + confusion (honest, out-of-fold).
    proba_oof = cross_val_predict(clf, Xp, y, cv=5, method="predict_proba")
    pred_oof = np.array(classes)[proba_oof.argmax(1)]
    acc = float((pred_oof == y).mean())
    print(f"\n5-fold accuracy: {acc*100:.1f}%")
    cm = confusion_matrix(y, pred_oof, labels=classes)
    print("confusion (rows=true, cols=pred):")
    print("      " + "  ".join(f"{c[-4:]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[-6:]:>6} " + "  ".join(f"{cm[i, j]:>6}" for j in range(len(classes))))

    # Fit final model on everything.
    clf.fit(Xp, y)

    # --- Open-set threshold from confidence distributions ---
    known_conf = proba_oof.max(1)  # out-of-fold max-proba on KNOWN versions
    v4_in_set = "suno_v4" in classes
    if len(Xv4):
        v4p = clf.predict_proba(pca.transform(scaler.transform(Xv4)))
        v4_conf = v4p.max(1)
    else:
        v4_conf = np.array([])

    if len(v4_conf) and not v4_in_set:
        # v4 is unseen → use it as the open-set holdout: pick the threshold
        # maximising Youden's J between "known" (ABOVE) and "v4" (BELOW).
        grid = np.linspace(0.3, 0.95, 66)
        best_t, best_j = 0.6, -1
        for t in grid:
            tpr = (known_conf >= t).mean()      # known kept confident
            fpr = (v4_conf >= t).mean()          # v4 wrongly confident
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, float(t)
        threshold = best_t
        v4_uncertain = float((v4_conf < threshold).mean())
        known_kept = float((known_conf >= threshold).mean())
        print(f"\nthreshold={threshold:.3f}  →  v4→uncertain {v4_uncertain*100:.0f}%  |  known-confident kept {known_kept*100:.0f}%")
        print(f"v4 confidence: mean={v4_conf.mean():.3f} median={np.median(v4_conf):.3f}")
    else:
        # No unseen-version holdout (v4 is now trained; v5 has no public data
        # yet) → threshold from the known-confidence distribution alone.
        threshold = float(np.quantile(known_conf, 0.15))
        print(f"\nthreshold={threshold:.3f} (15th pct of known confidences)")
        if len(v4_conf) and v4_in_set:
            # Sanity check: the 48 nyuuzyou v4 tracks are in-distribution now —
            # they should mostly classify AS suno_v4, confidently.
            v4_pred = clf.classes_[v4p.argmax(1)]
            as_v4 = float((v4_pred == "suno_v4").mean())
            kept = float((v4_conf >= threshold).mean())
            print(f"v4 sanity: {as_v4*100:.0f}% predicted suno_v4, {kept*100:.0f}% above threshold")

    # --- Export portable params (numpy fwd pass, no sklearn at inference) ---
    np.savez(
        os.path.join(MODELS, "version_head.npz"),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        lr_coef=clf.coef_.astype(np.float32),
        lr_intercept=clf.intercept_.astype(np.float32),
    )
    with open(os.path.join(MODELS, "version_head.json"), "w") as f:
        json.dump(
            {
                "classes": list(clf.classes_),
                "threshold": threshold,
                "feature_dim": int(X.shape[1]),
                "cv_accuracy": acc,
                "n_train": int(X.shape[0]),
            },
            f,
            indent=2,
        )
    print(f"\nSaved model → {MODELS}/version_head.{{npz,json}}  ({len(classes)} classes)")


if __name__ == "__main__":
    main()
