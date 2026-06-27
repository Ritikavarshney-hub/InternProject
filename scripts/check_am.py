import os
import numpy as np

OCC_DIR = "results/occlusion/occlusion"

for model in ["clip", "llava", "qwen2vl", "internvl2"]:

    files = sorted([
        f for f in os.listdir(OCC_DIR)
        if f.endswith(f"_{model}_mean.npy")
        or f.endswith(f"_{model}_mean_7x7.npy")
    ])

    print("\n", model.upper())

    for f in files[:5]:
        arr = np.load(os.path.join(OCC_DIR, f))

        print(
            f,
            "shape=", arr.shape,
            "min=", arr.min(),
            "max=", arr.max(),
            "mean=", arr.mean(),
            "std=", arr.std(),
        )