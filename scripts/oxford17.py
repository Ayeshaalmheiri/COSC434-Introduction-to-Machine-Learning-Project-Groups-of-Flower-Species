"""Optional extension: the same protocol on Oxford 17 Flowers (1,360 images, 17 species).

Steps, mirroring the main pipeline:
  1. download, check, and find images that also appear in Oxford 102
  2. extract the same features
  3. label-free selection on trn1 + val1 (silhouette), frozen to results/oxford17/
  4. one evaluation on tst1 (duplicates of Oxford 102 removed), k = 17 as reference
Oxford 17 labels follow file order: images 1-80 are species 1, 81-160 species 2, and so on.
"""

import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from sklearn.metrics import silhouette_score

from p22 import cnn, config, data, evaluation, handcrafted, selection
from p22.clustering import REPRESENTATIONS, Prep, build_representation, cluster

URL = "https://thor.robots.ox.ac.uk/flowers/17/"
MD5 = {"17flowers.tgz": "b59a65d8d1a99cd66944d474e1289eab", "datasplits.mat": "4828cddfd0d803c5abbdebcb1e148a1e"}
DIR = Path(config.DATA_DIR).parent / "flowers17"
OUT = config.RESULTS_DIR / "oxford17"
FEAT = config.FEATURES_DIR / "oxford17"
N, PER_CLASS = 1360, 80
K_GRID, K_REF = list(range(5, 61, 5)), 17

# 1. Download and check
DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
for name, expected in MD5.items():
    if not (DIR / name).exists():
        urllib.request.urlretrieve(URL + name, DIR / name)
    assert data.md5(DIR / name) == expected, name
if not (DIR / "jpg").exists():
    with tarfile.open(DIR / "17flowers.tgz") as tar:
        tar.extractall(DIR, filter="data")
paths = [DIR / "jpg" / f"image_{i:04d}.jpg" for i in range(1, N + 1)]
splits = sio.loadmat(DIR / "datasplits.mat")
sel_idx = np.concatenate([splits["trn1"].ravel(), splits["val1"].ravel()]) - 1
test_idx = splits["tst1"].ravel() - 1

# 2. Features (cached)
if not (FEAT / "resnet50.npy").exists():
    for name, arr in handcrafted.extract(paths).items():
        np.save(FEAT / f"{name}.npy", arr)
    np.save(FEAT / "resnet50.npy", cnn.extract(paths))
blocks = {f.stem: np.load(f) for f in FEAT.glob("*.npy")}

# Duplicates with Oxford 102: near-identical ResNet features AND near-identical small thumbnails.
def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=np.float32).ravel()

a = blocks["resnet50"] / np.linalg.norm(blocks["resnet50"], axis=1, keepdims=True)
b = np.load(config.FEATURES_DIR / "resnet50.npy")
b /= np.linalg.norm(b, axis=1, keepdims=True)
best = np.zeros(N)
match = np.zeros(N, dtype=int)
for start in range(0, N, 256):
    sims = a[start:start + 256] @ b.T
    best[start:start + 256], match[start:start + 256] = sims.max(1), sims.argmax(1)
p102 = data.image_paths()
dups = [i for i in np.where(best > 0.95)[0]
        if np.corrcoef(thumb(paths[i]), thumb(p102[match[i]]))[0, 1] > 0.98]
pd.DataFrame({"oxford17_image": [paths[i].name for i in dups],
              "oxford102_image": [p102[match[i]].name for i in dups],
              "cosine": best[dups]}).to_csv(OUT / "duplicates_with_oxford102.csv", index=False)
print(f"{len(dups)} Oxford 17 images also appear in Oxford 102")

# 3. Label-free selection on trn1 + val1, then freeze
frozen_path = OUT / "selection_frozen.json"
if not frozen_path.exists():
    tables, choices = [], {}
    for rep in REPRESENTATIONS:
        df = selection.sweep(rep, build_representation(rep, blocks)[sel_idx], k_grid=K_GRID)
        tables.append(df)
        choices[rep] = selection.choose(df)
    pd.concat(tables).to_csv(OUT / "selection_sweep.csv", index=False)
    frozen_path.write_text(json.dumps({
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "same as Oxford 102: highest mean silhouette on trn1+val1, k in 5..60",
        "choices": choices}, indent=2))
frozen = json.loads(frozen_path.read_text())

# 4. Labels only now, logged like the main run
with open(config.LABEL_ACCESS_LOG, "a") as log:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.write(f"{stamp}\toxford17 selection_frozen.json sha256="
              f"{hashlib.sha256(frozen_path.read_bytes()).hexdigest()}\toxford17 test evaluation\n")
y = np.arange(N) // PER_CLASS
keep = np.setdiff1d(test_idx, dups)
rows = []
for rep, choice in frozen["choices"].items():
    X = build_representation(rep, blocks)
    prep = Prep(REPRESENTATIONS[rep]["scaling"], choice["pca"]).fit(X[sel_idx])
    Z, space = prep.transform(X[keep]), prep.space(X[keep])
    for method in ["kmeans", "ward", "average_cosine"]:
        for k in sorted({K_REF, choice["k"]}):
            seeds = config.SEEDS if method == "kmeans" else [config.MAIN_SEED]
            for seed in seeds:
                pred = cluster(Z, method, k, seed)
                rows.append({"representation": rep, "method": method, "k": k, "seed": seed,
                             "selected": method == choice["method"] and k == choice["k"],
                             "silhouette": silhouette_score(space, pred), **evaluation.label_scores(y[keep], pred)})
df = pd.DataFrame(rows)
summary = df.groupby(["representation", "method", "k", "selected"])[["nmi", "ari", "silhouette"]].mean().reset_index()
summary.to_csv(OUT / "final_metrics.csv", index=False)
print(f"test images after removing duplicates: {len(keep)}")
print(summary.round(3).to_string(index=False))
