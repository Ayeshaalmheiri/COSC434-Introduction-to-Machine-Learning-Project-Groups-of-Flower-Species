"""Dataset figures for the slides: random sample of images and images per species."""

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p22 import config, data

config.FIGURES_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(config.MAIN_SEED)
paths = data.image_paths()
pick = rng.choice(len(paths), 24, replace=False)
fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
for ax, i in zip(axes.ravel(), pick):
    ax.imshow(data.load_image(paths[i]))
    ax.set_title(paths[i].stem, fontsize=8)
    ax.axis("off")
fig.suptitle("Oxford 102 Flowers: 24 random images after resize + centre crop")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "data_sample_grid.png", dpi=110)

counts = np.sort(data.class_counts())[::-1]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(np.arange(len(counts)), counts, color="#4a7ab5", width=0.9)
ax.set_xlabel("species (sorted by size)")
ax.set_ylabel("images")
ax.set_title(f"Images per species: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "data_class_counts.png", dpi=110)
print("saved figures")
