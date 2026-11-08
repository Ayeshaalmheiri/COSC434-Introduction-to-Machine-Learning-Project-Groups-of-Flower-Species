"""Step 3: label-free selection on train+val, then freeze the chosen settings.

This script never reads species labels.
"""

import json

import _path  # noqa: F401
import numpy as np

from p22 import config, data, selection

split = data.load_split()
blocks = {f.stem: np.load(f) for f in config.FEATURES_DIR.glob("*.npy")}
sweep_df, choices = selection.run(blocks, split)
sweep_df.to_csv(config.RESULTS_DIR / "selection_sweep.csv", index=False)

hashes = json.loads((config.RESULTS_DIR / "features_log.json").read_text())["feature_sha256"]
frozen_hash = selection.freeze(choices, hashes)
print(json.dumps(choices, indent=2))
print(f"frozen: {config.FROZEN_SELECTION} sha256={frozen_hash}")
