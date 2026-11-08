"""Feature preparation and the three clustering methods.

Preparation (scaling and optional PCA) is always fitted on the selection
set (train + val) and then applied unchanged to the test split.
"""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from . import config

# How each representation is built from the raw feature blocks, and scaled.
#   standardise: zero mean / unit variance per feature (hand-crafted blocks
#                have very different ranges)
#   l2:          unit length per image (usual for CNN embeddings, so Euclidean
#                distance behaves like cosine distance)
REPRESENTATIONS = {
    "mean_colour": {"blocks": ["mean_rgb"], "scaling": "standardise"},
    "hand_rgb_lbp": {"blocks": ["rgb_hist", "lbp"], "scaling": "standardise"},
    "hand_hsv_lbp": {"blocks": ["hsv_hist", "lbp"], "scaling": "standardise"},
    "resnet50": {"blocks": ["resnet50"], "scaling": "l2"},
}

METHODS = ["kmeans", "ward", "average_cosine"]


def build_representation(name, feature_blocks):
    blocks = REPRESENTATIONS[name]["blocks"]
    return np.hstack([feature_blocks[b] for b in blocks]).astype(np.float64)


@dataclass
class Prep:
    """Scaling, then optional PCA. Fit on selection set, reuse on test."""
    scaling: str
    use_pca: bool

    def fit(self, X):
        self.scaler = StandardScaler().fit(X) if self.scaling == "standardise" else None
        Z = self._scale(X)
        self.pca = None
        if self.use_pca and X.shape[1] > 3:
            self.pca = PCA(n_components=config.PCA_VARIANCE, random_state=config.MAIN_SEED).fit(Z)
        return self

    def _scale(self, X):
        return self.scaler.transform(X) if self.scaler is not None else normalize(X)

    def space(self, X):
        """Representation space without PCA; silhouette is always measured here."""
        return self._scale(X)

    def transform(self, X):
        Z = self._scale(X)
        return self.pca.transform(Z) if self.pca is not None else Z

    @property
    def n_components(self):
        return None if self.pca is None else int(self.pca.n_components_)


def kmeans(Z, k, seed):
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit_predict(Z)


def hierarchy(Z, method):
    """Build the merge tree once; cut it at any k with `cut`."""
    if method == "ward":
        return linkage(Z, method="ward")
    if method == "average_cosine":
        return linkage(Z, method="average", metric="cosine")
    raise ValueError(method)


def cut(tree, k):
    return fcluster(tree, t=k, criterion="maxclust") - 1


def cluster(Z, method, k, seed=config.MAIN_SEED):
    if method == "kmeans":
        return kmeans(Z, k, seed)
    return cut(hierarchy(Z, method), k)
