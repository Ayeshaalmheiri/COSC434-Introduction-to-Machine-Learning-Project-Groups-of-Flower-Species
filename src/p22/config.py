"""Paths, seeds and the fixed experiment grid, kept in one place.

Every script and notebook imports from here, so changing a setting
means changing it once.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw images live outside the repo by default (they must not be redistributed).
DATA_DIR = Path(os.environ.get("P22_DATA_DIR", REPO_ROOT / "data" / "flowers102"))
FEATURES_DIR = Path(os.environ.get("P22_FEATURES_DIR", REPO_ROOT / "features"))
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

FROZEN_SELECTION = RESULTS_DIR / "selection_frozen.json"
LABEL_ACCESS_LOG = RESULTS_DIR / "label_access.log"

DOWNLOAD_URL = "https://thor.robots.ox.ac.uk/flowers/102/"
# MD5 checksums published with torchvision's Flowers102 loader.
EXPECTED_MD5 = {
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
}

N_IMAGES = 8189
SEEDS = [0, 1, 2, 3, 4]  # k-means is repeated over these to measure stability
MAIN_SEED = 42

# Preprocessing shared by every representation.
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224

# Hand-crafted features.
HIST_BINS = 32
LBP_POINTS, LBP_RADIUS = 8, 1  # 'nri_uniform' LBP with P=8 gives 59 codes

# Label-free selection grid (run on train+val only).
K_GRID = list(range(10, 201, 10))
# Numbers of groups always reported on the test split, as the brief asks for
# the effect of the number of groups. k = 102 uses outside knowledge of the
# species count, so it is a reference point, never a selected setting.
K_REPORT = [50, 75, 102, 130]
PCA_VARIANCE = 0.95
