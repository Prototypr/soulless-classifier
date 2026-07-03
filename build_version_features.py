"""Stream version-labelled Suno audio, extract features, save vectors only.

Disk-light: each track is downloaded to a temp file, decoded, turned into a
fixed-length fingerprint (version_features.extract_features), then deleted. Only
the small feature matrix is kept.

Classes trained: suno_v2, suno_v3, suno_v3_5 (from nyuuzyou/suno, via Suno CDN
audio_urls), suno_v4 + suno_v4_5 (from sleeping-ai/Suno-Public-Playlist-small-audio
clip ids — a June-2025 scrape squarely in the v4/v4.5 era — resolved through
Suno's public clip API, which returns major_model_version per song), and
suno_v5_5 (from Kukedlc/suno-ai-music-dataset mp3s).

The 48 nyuuzyou v4 tracks are still saved separately (Xv4): originally an
open-set holdout, now an in-set sanity check — with v4 trained they should
classify as v4 instead of "uncertain". The remaining untrained version with no
public data is v5; it stays the open-set gap.

  PER_CLASS=300 ./.venv/bin/python build_version_features.py
  APPEND=1 PER_CLASS=300 ./.venv/bin/python build_version_features.py  # grow
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import librosa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_features import FEATURE_DIM, extract_features  # noqa: E402

PER_CLASS = int(os.environ.get("PER_CLASS", "300"))
WORKERS = int(os.environ.get("WORKERS", "6"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "version_features.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
random.seed(42)


def feat_from_url(url: str) -> np.ndarray | None:
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
            path = f.name
        try:
            y, sr = librosa.load(path, sr=16000, mono=True, duration=35)
        finally:
            os.remove(path)
        if len(y) < 16000:  # < 1s, unusable
            return None
        return extract_features(y, sr)
    except Exception:
        return None


CLIP_API = "https://studio-api.prod.suno.com/api/clip/"


def clip_version(clip_id: str) -> tuple[str, str] | None:
    """Resolve a Suno clip id → (major_model_version, audio_url) via the public
    clip API (no auth needed for public songs)."""
    try:
        r = requests.get(
            CLIP_API + clip_id,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        d = r.json()
        v = d.get("major_model_version") or ""
        u = d.get("audio_url") or ""
        if d.get("status") == "complete" and v and u.startswith("http"):
            return (v, u)
    except Exception:
        pass
    return None


def harvest_sleeping_ai(wanted: set[str]) -> dict[str, list[str]]:
    """Resolve the sleeping-ai playlist scrape's clip ids against the clip API
    and bucket audio_urls by version. Throttled to be polite (~4 req/s)."""
    import re
    import time

    path = hf_hub_download(
        "sleeping-ai/Suno-Public-Playlist-small-audio",
        "suno_music.txt",
        repo_type="dataset",
    )
    ids = sorted(set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", open(path).read())))
    print(f"sleeping-ai: {len(ids)} unique clip ids to resolve", flush=True)
    out: dict[str, list[str]] = {v: [] for v in wanted}
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(clip_version, cid): cid for cid in ids}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res and res[0] in wanted:
                out[res[0]].append(res[1])
            if done % 200 == 0:
                print(f"  resolved {done}/{len(ids)}: " + str({k: len(v) for k, v in out.items()}), flush=True)
            time.sleep(0.05)  # ~4 req/s per worker cap overall
    print("sleeping-ai buckets:", {k: len(v) for k, v in out.items()}, flush=True)
    return out


def collect(tasks: list[tuple[str, str]], label_note: str):
    """tasks: (url, label). Returns (X list, y list) using a thread pool."""
    X, y = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(feat_from_url, u): lab for (u, lab) in tasks}
        for fut in as_completed(futs):
            lab = futs[fut]
            f = fut.result()
            done += 1
            if f is not None:
                X.append(f)
                y.append(lab)
            if done % 50 == 0:
                print(f"  [{label_note}] {done}/{len(tasks)} fetched, {len(X)} ok", flush=True)
    return X, y


def main():
    print(f"PER_CLASS={PER_CLASS} WORKERS={WORKERS} FEATURE_DIM={FEATURE_DIM}", flush=True)

    # --- nyuuzyou/suno: v2, v3, v3.5 (train) + v4 (holdout) ---
    print("Loading nyuuzyou/suno metadata...", flush=True)
    p = hf_hub_download("nyuuzyou/suno", "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p, columns=["major_model_version", "audio_url", "status"])
    vers = t.column("major_model_version").to_pylist()
    urls = t.column("audio_url").to_pylist()
    st = t.column("status").to_pylist()
    buckets: dict[str, list[str]] = {"v2": [], "v3": [], "v3.5": [], "v4": []}
    for v, u, s in zip(vers, urls, st):
        if s != "complete" or not u or not str(u).startswith("http"):
            continue
        if v in buckets:
            buckets[v].append(u)
    print({k: len(v) for k, v in buckets.items()}, flush=True)

    tasks: list[tuple[str, str]] = []
    for v in ["v2", "v3", "v3.5"]:
        random.shuffle(buckets[v])
        lab = "suno_" + v.replace(".", "_")
        for u in buckets[v][:PER_CLASS]:
            tasks.append((u, lab))
    v4_urls = buckets["v4"][:48]

    # --- sleeping-ai playlist scrape: v4 + v4.5 (train) ---
    # SKIP_SLEEPING_AI=1 to leave these classes out (e.g. a quick v2/v3 rebuild).
    if not os.environ.get("SKIP_SLEEPING_AI"):
        sa = harvest_sleeping_ai({"v4", "v4.5"})
        for v, urls_ in sa.items():
            random.shuffle(urls_)
            lab = "suno_" + v.replace(".", "_")
            for u in urls_[:PER_CLASS]:
                tasks.append((u, lab))

    # --- Kukedlc: v5.5 mp3s (train) ---
    print("Loading Kukedlc v5.5 file list...", flush=True)
    import csv
    meta = hf_hub_download("Kukedlc/suno-ai-music-dataset", "metadata.csv", repo_type="dataset")
    v55_files = []
    with open(meta, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("major_model_version") == "v5.5" and row.get("file_name"):
                v55_files.append(row["file_name"])
    random.shuffle(v55_files)
    base = "https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset/resolve/main/"
    for fn in v55_files[:PER_CLASS]:
        tasks.append((base + fn.lstrip("/"), "suno_v5_5"))

    random.shuffle(tasks)
    print(f"Total train tasks: {len(tasks)} | v4 holdout: {len(v4_urls)}", flush=True)

    X, y = collect(tasks, "train")
    print("Collecting v4 holdout...", flush=True)
    Xv4, _ = collect([(u, "v4") for u in v4_urls], "v4")

    X = np.array(X, dtype=np.float32)
    Xv4 = np.array(Xv4, dtype=np.float32) if Xv4 else np.zeros((0, FEATURE_DIM), np.float32)
    y = np.array(y)
    # APPEND=1 merges into the existing feature set instead of overwriting — this
    # is how we EXPAND later: add a new source/version above (or bump PER_CLASS)
    # and re-run with APPEND=1 to grow the corpus without re-downloading. Classes
    # are never hard-coded downstream; the trainer + model derive them from `y`,
    # so a newly-added version becomes a class automatically on the next train.
    if os.environ.get("APPEND") and os.path.exists(OUT):
        prev = np.load(OUT, allow_pickle=True)
        if len(prev["X"]):
            X = np.concatenate([prev["X"], X]) if len(X) else prev["X"]
            y = np.concatenate([prev["y"], y]) if len(y) else prev["y"]
        if "Xv4" in prev and len(prev["Xv4"]) and not len(Xv4):
            Xv4 = prev["Xv4"]
        print(f"APPEND: merged with existing → X={X.shape}", flush=True)

    from collections import Counter
    print("TRAIN class counts:", dict(Counter(y.tolist())), flush=True)
    print(f"v4 holdout features: {len(Xv4)}", flush=True)
    np.savez_compressed(OUT, X=X, y=y, Xv4=Xv4)
    print(f"Saved -> {OUT}  (X={X.shape})", flush=True)


if __name__ == "__main__":
    main()
