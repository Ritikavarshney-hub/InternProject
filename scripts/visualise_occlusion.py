import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk

PROJECT_ROOT = "."

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "CulturalVQA"
)

OCC_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "occlusion",
    "occlusion"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "heatmaps"
)

os.makedirs(OUT_DIR, exist_ok=True)


def upsample_heatmap(scores, image_size):

    H, W = image_size

    heatmap = np.kron(
        scores,
        np.ones((
            H // scores.shape[0],
            W // scores.shape[1]
        ))
    )

    heatmap = heatmap[:H, :W]

    return heatmap


def save_visualization(image, scores, save_path):

    img = np.array(image)

    heatmap = upsample_heatmap(
        scores,
        (img.shape[0], img.shape[1])
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title("Occlusion Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main(model_name):

    ds = load_from_disk(DATASET_PATH)["test"]

    id_to_image = {}

    for row in ds:
        id_to_image[row["u_id"]] = row["image"]

    files = [
        f for f in os.listdir(OCC_DIR)
        if f"_{model_name}_" in f and "_7x7.npy" in f
    ]

    print(f"Found {len(files)} heatmaps")

    for idx, fname in enumerate(files):

        u_id = fname.split(f"_{model_name}_")[0]

        scores = np.load(
            os.path.join(OCC_DIR, fname)
        )

        image = id_to_image[u_id]

        save_path = os.path.join(
            OUT_DIR,
            fname.replace(".npy", ".png")
        )

        save_visualization(
            image,
            scores,
            save_path
        )

        if (idx + 1) % 20 == 0:
            print(idx + 1)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["clip", "llava","qwen2vl","internvl2"]
    )

    args = parser.parse_args()

    main(args.model)
