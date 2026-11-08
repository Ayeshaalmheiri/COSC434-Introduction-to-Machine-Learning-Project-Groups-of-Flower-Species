"""Final evaluation on the test split (6,149 images).

This is the only module that reads species labels, and it refuses to do so
until `results/selection_frozen.json` exists. Every label access is logged
with that file's SHA-256, so the order "select first, then look at labels"
can be shown afterwards.
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score, silhouette_score

from . import config
from .clustering import METHODS, REPRESENTATIONS, Prep, build_representation, cluster


def load_frozen():
    if not config.FROZEN_SELECTION.exists():
        raise PermissionError("Selection is not frozen yet: labels stay hidden.")
    raw = config.FROZEN_SELECTION.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_labels_after_freeze(data_dir=None, reason=""):
    """Species labels (0-based, 102 classes); only after selection is frozen."""
    _, frozen_hash = load_frozen()
    with open(config.LABEL_ACCESS_LOG, "a") as log:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.write(f"{stamp}\tselection_frozen.json sha256={frozen_hash}\t{reason}\n")
    data_dir = data_dir or config.DATA_DIR
    return sio.loadmat(data_dir / "imagelabels.mat")["labels"].ravel() - 1


def label_scores(y_true, y_pred):
    return {
        "rand_index": rand_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
    }


def evaluate_test(feature_blocks, split, data_dir=None):
    """Cluster the test split with every method at k in K_REPORT plus each selected k.

    Preparation is fitted on the selection set. The frozen choice for each
    representation is flagged `selected`; the other rows show how method and
    number of groups change the result. Labels only score, never choose.
    """
    frozen, _ = load_frozen()
    y = load_labels_after_freeze(data_dir, reason="final test evaluation")[split["test"]]
    rows, partitions = [], {}
    for rep, choice in frozen["choices"].items():
        X = build_representation(rep, feature_blocks)
        prep = Prep(REPRESENTATIONS[rep]["scaling"], choice["pca"]).fit(X[split["selection"]])
        Z, space = prep.transform(X[split["test"]]), prep.space(X[split["test"]])
        for method in METHODS:
            for k in sorted(set(config.K_REPORT) | {choice["k"]}):
                seeds = config.SEEDS if method == "kmeans" else [config.MAIN_SEED]
                for seed in seeds:
                    pred = cluster(Z, method, k, seed)
                    selected = method == choice["method"] and k == choice["k"]
                    if selected and seed == seeds[0]:
                        partitions[rep] = pred
                    rows.append({
                        "representation": rep, "method": method, "pca": choice["pca"], "k": k,
                        "seed": seed, "selected": selected,
                        "silhouette": silhouette_score(space, pred), **label_scores(y, pred),
                    })
            print(f"  {rep} {method}: done")
    rng = np.random.default_rng(config.MAIN_SEED)
    for k in sorted(set(config.K_REPORT) | {c["k"] for c in frozen["choices"].values()}):
        for seed in config.SEEDS:
            pred = rng.integers(0, k, size=len(y))
            rows.append({"representation": "random", "method": "random", "pca": False, "k": k,
                         "seed": seed, "selected": False, "silhouette": np.nan, **label_scores(y, pred)})
    return pd.DataFrame(rows), partitions, y


def summarise(per_seed):
    """Mean and standard deviation over k-means seeds."""
    keys = ["representation", "method", "pca", "k", "selected"]
    metrics = ["silhouette", "rand_index", "ari", "nmi"]
    out = per_seed.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()
