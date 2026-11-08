"""Step 4b: test scores for the per-method label-free choices (see DECISIONS.md, D12)."""

import json

import _path  # noqa: F401
import numpy as np
import pandas as pd

from p22 import config, data, evaluation
from p22.clustering import REPRESENTATIONS, Prep, build_representation, cluster
from sklearn.metrics import silhouette_score

split = data.load_split()
blocks = {f.stem: np.load(f) for f in config.FEATURES_DIR.glob("*.npy")}
choices = json.loads((config.RESULTS_DIR / "selection_per_method.json").read_text())["choices"]
y = evaluation.load_labels_after_freeze(reason="per-method choices (D12)")[split["test"]]

rows = []
for c in choices.values():
    X = build_representation(c["representation"], blocks)
    prep = Prep(REPRESENTATIONS[c["representation"]]["scaling"], c["pca"]).fit(X[split["selection"]])
    Z, space = prep.transform(X[split["test"]]), prep.space(X[split["test"]])
    seeds = config.SEEDS if c["method"] == "kmeans" else [config.MAIN_SEED]
    for seed in seeds:
        pred = cluster(Z, c["method"], c["k"], seed)
        rows.append({**{k: c[k] for k in ("representation", "method", "pca", "k")}, "seed": seed,
                     "largest_cluster_share": np.bincount(pred).max() / len(pred),
                     "silhouette": silhouette_score(space, pred), **evaluation.label_scores(y, pred)})
df = pd.DataFrame(rows)
out = df.groupby(["representation", "method", "pca", "k"]).agg(["mean", "std"]).drop(columns="seed")
out.columns = [f"{m}_{s}" for m, s in out.columns]
out = out.reset_index()
out.to_csv(config.RESULTS_DIR / "per_method_metrics.csv", index=False)
print(out[["representation", "method", "k", "nmi_mean", "ari_mean", "silhouette_mean",
           "largest_cluster_share_mean"]].round(3).to_string(index=False))
