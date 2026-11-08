"""Label-free selection on the selection set (train + val, 2,040 images).

Nothing in this module can see species labels: it only receives features
and the split indices. For each representation it sweeps method x PCA x k,
scores every run with silhouette and Davies-Bouldin (and, for k-means,
agreement between seeds), and picks the run with the highest silhouette.
"""

import hashlib
import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from . import config
from .clustering import METHODS, REPRESENTATIONS, Prep, build_representation, cut, hierarchy, kmeans


def seed_stability(partitions):
    """Mean pairwise adjusted Rand index between runs with different seeds (1 = identical)."""
    pairs = itertools.combinations(partitions, 2)
    return float(np.mean([adjusted_rand_score(a, b) for a, b in pairs]))


def sweep(rep_name, X_sel, k_grid=None):
    k_grid = k_grid or config.K_GRID
    rows = []
    scaling = REPRESENTATIONS[rep_name]["scaling"]
    for use_pca in (False, True):
        prep = Prep(scaling, use_pca).fit(X_sel)
        if use_pca and prep.pca is None:
            continue  # too few dimensions for PCA to matter (mean colour)
        Z, space = prep.transform(X_sel), prep.space(X_sel)
        for method in METHODS:
            tree = None if method == "kmeans" else hierarchy(Z, method)
            for k in k_grid:
                if method == "kmeans":
                    parts = [kmeans(Z, k, s) for s in config.SEEDS]
                else:
                    parts = [cut(tree, k)]  # deterministic: one run is enough
                sil = [silhouette_score(space, p) for p in parts]
                rows.append({
                    "representation": rep_name, "method": method, "pca": use_pca,
                    "pca_dims": prep.n_components, "k": k,
                    "clusters_found": len(np.unique(parts[0])),
                    "silhouette": float(np.mean(sil)), "silhouette_sd": float(np.std(sil)),
                    "davies_bouldin": float(np.mean([davies_bouldin_score(space, p) for p in parts])),
                    "largest_cluster_share": float(np.bincount(parts[0]).max() / len(parts[0])),
                    "seed_stability_ari": seed_stability(parts) if len(parts) > 1 else np.nan,
                })
            print(f"  {rep_name} pca={use_pca} {method}: done")
    return pd.DataFrame(rows)


def choose(sweep_df):
    """Highest mean silhouette; ties broken by lower Davies-Bouldin."""
    ranked = sweep_df.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    best = ranked.iloc[0]
    return {
        "method": best["method"], "pca": bool(best["pca"]), "k": int(best["k"]),
        "silhouette": float(best["silhouette"]), "davies_bouldin": float(best["davies_bouldin"]),
    }


def freeze(choices, feature_hashes, path=None):
    """Write the chosen settings. Labels can only be loaded after this file exists."""
    path = path or config.FROZEN_SELECTION
    if path.exists():
        raise FileExistsError(f"{path} already exists; selection is frozen once.")
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "per representation, highest mean silhouette on train+val over method x PCA x k",
        "k_grid": config.K_GRID,
        "k_report": config.K_REPORT,
        "seeds": config.SEEDS,
        "feature_sha256": feature_hashes,
        "choices": choices,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(feature_blocks, split):
    """Sweep every representation on the selection set and return (sweep table, choices)."""
    tables, choices = [], {}
    for rep in REPRESENTATIONS:
        X_sel = build_representation(rep, feature_blocks)[split["selection"]]
        df = sweep(rep, X_sel)
        tables.append(df)
        choices[rep] = choose(df)
    return pd.concat(tables, ignore_index=True), choices
