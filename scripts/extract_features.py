"""Step 2: compute and cache every feature block (hand-crafted and ResNet-50)."""

import hashlib
import json
import platform
import time

import _path  # noqa: F401
import numpy as np
import torch

from p22 import cnn, config, data, handcrafted

config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
paths = data.image_paths()
timings = {}

t = time.time()
for name, arr in handcrafted.extract(paths).items():
    np.save(config.FEATURES_DIR / f"{name}.npy", arr)
timings["handcrafted_seconds"] = round(time.time() - t, 1)

if not (config.FEATURES_DIR / "resnet50.npy").exists():
    t = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    np.save(config.FEATURES_DIR / "resnet50.npy", cnn.extract(paths))
    timings["resnet50_seconds"] = round(time.time() - t, 1)

hashes = {f.stem: hashlib.sha256(f.read_bytes()).hexdigest()
          for f in sorted(config.FEATURES_DIR.glob("*.npy"))}
log = {"hardware": {"cpu": platform.processor() or platform.machine(),
                    "threads": torch.get_num_threads(), "gpu": torch.cuda.is_available()},
       "torch": torch.__version__, **timings, "feature_sha256": hashes}
(config.RESULTS_DIR / "features_log.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))
