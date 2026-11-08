"""Download, verify and load Oxford 102 Flowers.

Labels are deliberately NOT loaded here. The only way to read them is
`evaluation.load_labels_after_freeze`, which refuses until the label-free
selection has been saved.
"""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from . import config


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir=config.DATA_DIR):
    """Fetch the three official files, check their MD5, and unpack the images."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in config.EXPECTED_MD5.items():
        path = data_dir / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(config.DOWNLOAD_URL + name, path)
        got = md5(path)
        if got != expected:
            raise RuntimeError(f"{name}: md5 {got} != expected {expected}")
        print(f"{name}: md5 ok")
    if not (data_dir / "jpg").exists():
        with tarfile.open(data_dir / "102flowers.tgz") as tar:
            tar.extractall(data_dir, filter="data")
    n = len(list((data_dir / "jpg").glob("*.jpg")))
    if n != config.N_IMAGES:
        raise RuntimeError(f"expected {config.N_IMAGES} images, found {n}")
    return data_dir


def image_paths(data_dir=config.DATA_DIR):
    """Paths ordered by image number, so row i of every feature file is image_{i+1:05d}.jpg."""
    return [Path(data_dir) / "jpg" / f"image_{i:05d}.jpg" for i in range(1, config.N_IMAGES + 1)]


def load_split(data_dir=config.DATA_DIR):
    """0-based indices of the supplied train / val / test split (no labels)."""
    m = sio.loadmat(Path(data_dir) / "setid.mat")
    split = {
        "train": m["trnid"].ravel() - 1,
        "val": m["valid"].ravel() - 1,
        "test": m["tstid"].ravel() - 1,
    }
    split["selection"] = np.concatenate([split["train"], split["val"]])
    return split


def load_image(path):
    """Shorter side to 256 px, then a centre 224x224 crop, as RGB uint8.

    Both representations see exactly these pixels.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = config.RESIZE_SHORT_SIDE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - config.CROP_SIZE) // 2, (h - config.CROP_SIZE) // 2
    img = img.crop((left, top, left + config.CROP_SIZE, top + config.CROP_SIZE))
    return np.asarray(img)


def class_counts(data_dir=config.DATA_DIR):
    """Number of images per species, for the dataset description only.

    Returns counts, not which image belongs to which species, so it cannot
    leak into clustering or selection.
    """
    labels = sio.loadmat(Path(data_dir) / "imagelabels.mat")["labels"].ravel()
    return np.bincount(labels)[1:]
