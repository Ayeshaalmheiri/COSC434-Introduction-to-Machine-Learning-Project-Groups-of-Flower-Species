"""Fast end-to-end check on a few hundred images, writing to a temporary folder.

Uses made-up labels so the real ones are never touched before the freeze.
"""

import sys
import tempfile
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from p22 import cnn, config, data, evaluation, handcrafted, selection

tmp = Path(tempfile.mkdtemp())
config.RESULTS_DIR = tmp
config.FROZEN_SELECTION = tmp / "selection_frozen.json"
config.LABEL_ACCESS_LOG = tmp / "label_access.log"
config.K_GRID, config.K_REPORT, config.SEEDS = [5, 10], [8], [0, 1]

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(config.N_IMAGES, 300, replace=False))
paths = [data.image_paths()[i] for i in idx]
blocks = handcrafted.extract(paths)
blocks["resnet50"] = np.vstack([cnn.extract(paths[:32]), rng.normal(size=(268, 2048)).astype(np.float32)])
assert blocks["rgb_hist"].shape == (300, 96) and blocks["lbp"].shape == (300, 59)
assert np.allclose(blocks["lbp"].sum(1), 1) and np.allclose(blocks["hsv_hist"].sum(1), 3)
split = {"selection": np.arange(150), "test": np.arange(150, 300)}

try:
    evaluation.load_labels_after_freeze()
    sys.exit("guard failed: labels loaded before freeze")
except PermissionError:
    print("guard ok: labels refused before freeze")

sweep_df, choices = selection.run(blocks, split)
selection.freeze(choices, {"smoke": "n/a"})
fake = tmp / "fake"
fake.mkdir()
import scipy.io as sio
sio.savemat(fake / "imagelabels.mat", {"labels": rng.integers(1, 103, size=(1, 300))})
per_seed, parts, _ = evaluation.evaluate_test(blocks, split, data_dir=fake)
print(evaluation.summarise(per_seed).head(12).to_string())
print("smoke test passed;", len(sweep_df), "selection runs; outputs in", tmp)
