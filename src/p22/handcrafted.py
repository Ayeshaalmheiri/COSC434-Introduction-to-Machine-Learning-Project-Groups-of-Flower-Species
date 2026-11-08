"""Hand-crafted colour and texture features, plus the mean-colour baseline.

Each image becomes:
  rgb_hist  96 values  (32-bin histogram per R, G, B channel)
  hsv_hist  96 values  (32-bin histogram per H, S, V channel)
  lbp       59 values  (histogram of uniform local binary pattern codes)
  mean_rgb   3 values  (average colour; the simple baseline)
Histograms are normalised to sum to 1 per channel so image size does not matter.
"""

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern

from . import config
from .data import load_image


def channel_histograms(img01, bins=config.HIST_BINS):
    """Per-channel histograms of an image with values in [0, 1]."""
    hists = [np.histogram(img01[..., c], bins=bins, range=(0.0, 1.0))[0] for c in range(3)]
    hists = [h / h.sum() for h in hists]
    return np.concatenate(hists)


def lbp_histogram(img_uint8):
    """Histogram of the 59 'non-rotation-invariant uniform' LBP codes on the grey image."""
    gray = (rgb2gray(img_uint8) * 255).astype(np.uint8)
    codes = local_binary_pattern(gray, config.LBP_POINTS, config.LBP_RADIUS, method="nri_uniform")
    n_codes = config.LBP_POINTS * (config.LBP_POINTS - 1) + 3  # 59 for P = 8
    hist = np.bincount(codes.astype(int).ravel(), minlength=n_codes)
    return hist / hist.sum()


def features_for_image(path):
    img = load_image(path)
    img01 = img.astype(np.float32) / 255.0
    return {
        "rgb_hist": channel_histograms(img01),
        "hsv_hist": channel_histograms(rgb2hsv(img01)),
        "lbp": lbp_histogram(img),
        "mean_rgb": img01.reshape(-1, 3).mean(axis=0),
    }


def extract(paths, n_jobs=-1):
    """Return a dict of (n_images, d) float32 arrays, rows in the order of `paths`."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(features_for_image)(p) for p in paths)
    return {key: np.stack([r[key] for r in rows]).astype(np.float32) for key in rows[0]}
