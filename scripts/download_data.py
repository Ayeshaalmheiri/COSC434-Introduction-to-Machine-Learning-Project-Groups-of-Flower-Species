"""Step 1: download and verify the data, write the manifest and dataset summary."""

import hashlib
import json

import _path  # noqa: F401
import numpy as np
from PIL import Image

from p22 import config, data

data_dir = data.download()
split = data.load_split()
counts = data.class_counts()

config.RESULTS_DIR.mkdir(exist_ok=True)
paths = data.image_paths()
with open(config.RESULTS_DIR / "manifest.tsv", "w") as f:
    f.write("image\tsplit\tsha256\n")
    which = np.empty(config.N_IMAGES, dtype=object)
    for name in ("train", "val", "test"):
        which[split[name]] = name
    for i, p in enumerate(paths):
        f.write(f"{p.name}\t{which[i]}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n")

sizes = np.array([Image.open(p).size for p in paths])
summary = {
    "files_md5": {k: data.md5(data_dir / k) for k in config.EXPECTED_MD5},
    "n_images": len(paths),
    "split_sizes": {k: int(len(v)) for k, v in split.items()},
    "split_disjoint_and_complete": bool(
        len(np.unique(np.concatenate([split["selection"], split["test"]]))) == config.N_IMAGES),
    "n_classes": int(len(counts)),
    "images_per_class": {"min": int(counts.min()), "median": float(np.median(counts)),
                         "max": int(counts.max())},
    "image_width": {"min": int(sizes[:, 0].min()), "max": int(sizes[:, 0].max())},
    "image_height": {"min": int(sizes[:, 1].min()), "max": int(sizes[:, 1].max())},
}
(config.RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
np.savetxt(config.RESULTS_DIR / "class_counts.csv", counts, fmt="%d", header="images_per_class",
           comments="")
print(json.dumps(summary, indent=2))
