"""Step 4: one locked evaluation on the test split, after selection is frozen."""

import _path  # noqa: F401
import numpy as np

from p22 import config, data, evaluation

split = data.load_split()
blocks = {f.stem: np.load(f) for f in config.FEATURES_DIR.glob("*.npy")}
per_seed, partitions, y_test = evaluation.evaluate_test(blocks, split)
per_seed.to_csv(config.RESULTS_DIR / "test_metrics_per_seed.csv", index=False)
summary = evaluation.summarise(per_seed)
summary.to_csv(config.RESULTS_DIR / "final_metrics.csv", index=False)
np.savez(config.RESULTS_DIR / "test_partitions.npz", test_idx=split["test"], **partitions)
print(summary[summary.selected].to_string(index=False))
