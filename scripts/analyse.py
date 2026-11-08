"""Step 5: analysis figures and tables from the locked test run.

Reads results/final_metrics.csv and results/test_partitions.npz (written by
evaluate_test.py) and produces:
  figures/metrics_vs_k.png            NMI / ARI / silhouette against k, per method and representation
  figures/selection_silhouette.png    the label-free sweep on train+val
  figures/cluster_grid_<rep>.png      sample images from the purest and least pure clusters
                                      (k-means at k = 102 for every representation, so they compare)
  results/cluster_purity_<rep>.csv    per-cluster size, dominant species, purity
  results/species_spread_<rep>.csv    per-species number of clusters it is split over
  results/colour_vs_species.csv       how much clusters follow mean colour rather than species
"""

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from p22 import config, data, evaluation
from p22.clustering import REPRESENTATIONS, Prep, build_representation, cluster

R, F = config.RESULTS_DIR, config.FIGURES_DIR
REP_LABEL = {"mean_colour": "mean colour (baseline)", "hand_rgb_lbp": "RGB hist + LBP",
             "hand_hsv_lbp": "HSV hist + LBP", "resnet50": "ResNet-50"}
METHOD_STYLE = {"kmeans": "-", "ward": "--", "average_cosine": ":"}
COLOURS = {"mean_colour": "#9a9a9a", "hand_rgb_lbp": "#d08a2e", "hand_hsv_lbp": "#b5532a",
           "resnet50": "#2f6db5", "random": "#000000"}

summary = pd.read_csv(R / "final_metrics.csv")
split = data.load_split()
test_idx = split["test"]
y = evaluation.load_labels_after_freeze(reason="analysis of the locked test run")[test_idx]
paths = data.image_paths()

# 1. Metrics against k (k-means solid, Ward dashed; average-cosine is in the tables only)
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, metric in zip(axes, ["nmi", "ari", "silhouette"]):
    for rep in REP_LABEL:
        for method in ["kmeans", "ward"]:
            d = summary[(summary.representation == rep) & (summary.method == method)].sort_values("k")
            d = d[d.k >= 50]
            ax.plot(d.k, d[f"{metric}_mean"], METHOD_STYLE[method], color=COLOURS[rep], marker="o", ms=3)
    if metric != "silhouette":
        d = summary[(summary.representation == "random") & (summary.k >= 50)].sort_values("k")
        ax.plot(d.k, d[f"{metric}_mean"], color="black", lw=1)
    ax.axvline(102, color="#cccccc", lw=1, zorder=0)
    ax.set_xlabel("number of groups k (grey line: 102 species)")
    ax.set_title({"nmi": "NMI (uses labels)", "ari": "Adjusted Rand index (uses labels)",
                  "silhouette": "Silhouette (no labels)"}[metric])
    ax.spines[["top", "right"]].set_visible(False)
handles = [Line2D([], [], color=COLOURS[r], lw=3, label=REP_LABEL[r]) for r in REP_LABEL]
handles += [Line2D([], [], color="black", lw=1, label="random assignment"),
            Line2D([], [], color="grey", ls="-", label="k-means (mean of 5 seeds)"),
            Line2D([], [], color="grey", ls="--", label="Ward")]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9)
fig.suptitle("Test split, 6,149 images")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(F / "metrics_vs_k.png", dpi=120)

# 2. The label-free sweep that made the choice
sweep = pd.read_csv(R / "selection_sweep.csv")
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
for ax, rep in zip(axes, REP_LABEL):
    for (method, pca), d in sweep[sweep.representation == rep].groupby(["method", "pca"]):
        ax.plot(d.k, d.silhouette, METHOD_STYLE[method], marker="o" if pca else None, ms=3,
                label=f"{method}{' + PCA' if pca else ''}")
    ax.set_title(REP_LABEL[rep])
    ax.set_xlabel("k")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean silhouette, train+val")
axes[-1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(F / "selection_silhouette.png", dpi=120)

# 3. Per-cluster purity, species spread and image grids for each selected partition
colour = np.load(config.FEATURES_DIR / "mean_rgb.npy")[test_idx]
colour_rows = []
# k = 102 is used here only to compare representations at the same number of
# groups; it is a reference point, not a selected setting (see DECISIONS.md D8).
frozen, _ = evaluation.load_frozen()
blocks = {f.stem: np.load(f) for f in config.FEATURES_DIR.glob("*.npy")}
for rep in REP_LABEL:
    X = build_representation(rep, blocks)
    prep = Prep(REPRESENTATIONS[rep]["scaling"], frozen["choices"][rep]["pca"]).fit(X[split["selection"]])
    pred = cluster(prep.transform(X[test_idx]), "kmeans", 102, config.SEEDS[0])
    tab = pd.crosstab(pred, y)
    clusters = pd.DataFrame({
        "cluster": tab.index, "size": tab.sum(axis=1).values,
        "dominant_species": tab.idxmax(axis=1).values + 1,  # 1-based ids as in the dataset
        "purity": (tab.max(axis=1) / tab.sum(axis=1)).values,
        "n_species": (tab > 0).sum(axis=1).values,
    }).sort_values("purity", ascending=False)
    clusters.to_csv(R / f"cluster_purity_{rep}.csv", index=False)
    spread = pd.DataFrame({
        "species": tab.columns + 1, "images": tab.sum(axis=0).values,
        "clusters_used": (tab > 0).sum(axis=0).values,
        "share_in_main_cluster": (tab.max(axis=0) / tab.sum(axis=0)).values,
    }).sort_values("share_in_main_cluster")
    spread.to_csv(R / f"species_spread_{rep}.csv", index=False)

    # Do the groups follow colour more than species? Bin mean colour into as many
    # groups as there are clusters (k-means on the 3 colour values) and compare NMI.
    colour_groups = cluster(
        (colour - colour.mean(0)) / colour.std(0), "kmeans", len(np.unique(pred)))
    colour_rows.append({
        "representation": rep, "k": len(np.unique(pred)),
        "nmi_with_species": normalized_mutual_info_score(y, pred),
        "nmi_with_colour_groups": normalized_mutual_info_score(colour_groups, pred),
        "median_cluster_purity": clusters.purity.median(),
    })

    big = clusters[clusters["size"] >= 10]
    show = pd.concat([big.head(3), big.tail(3)])
    fig, axes = plt.subplots(len(show), 8, figsize=(14, 1.9 * len(show)))
    for row, (_, c) in zip(axes, show.iterrows()):
        members = test_idx[pred == c.cluster]
        pick = np.random.default_rng(0).choice(members, min(8, len(members)), replace=False)
        for ax in row:
            ax.axis("off")
        for ax, i in zip(row, pick):
            ax.imshow(data.load_image(paths[i]))
        row[0].set_title(f"cluster {int(c.cluster)}: {int(c['size'])} imgs, purity {c.purity:.0%}, "
                         f"{int(c.n_species)} species", fontsize=8, loc="left")
    fig.suptitle(f"{REP_LABEL[rep]}: k-means, k = 102: 3 purest and 3 least pure clusters (>=10 images)")
    fig.tight_layout()
    fig.savefig(F / f"cluster_grid_{rep}.png", dpi=90)
    plt.close("all")

pd.DataFrame(colour_rows).to_csv(R / "colour_vs_species.csv", index=False)
print(pd.DataFrame(colour_rows).to_string(index=False))
