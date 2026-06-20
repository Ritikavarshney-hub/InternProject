"""
Milestone 3b — Heatmap Visualisation Utility

Renders occlusion sensitivity maps alongside the original image.
Saves PNG files to results/occlusion/ for manual inspection.

Usage:
    # Visualise a single image
    python visualise_heatmap.py --u_id <id> --model clip

    # Visualise all pilot images (reads pilot_summary.csv)
    python visualise_heatmap.py --pilot

    # Visualise specific fill variant
    python visualise_heatmap.py --u_id <id> --model clip --fill black

    # Visualise all three fill variants side-by-side for one image
    python visualise_heatmap.py --u_id <id> --model clip --compare_fills

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from datasets import load_from_disk

from occlusion import DATASET_PATH, OCCLUSION_DIR, SAMPLE_IDS_CSV, GRID_SIZE

PILOT_SUMMARY = os.path.join(OCCLUSION_DIR, "pilot_summary.csv")
VIZ_DIR       = os.path.join("results", "heatmaps")


# ---------------------------------------------------------------------------
# Core visualisation function
# ---------------------------------------------------------------------------

def show_heatmap(
    image: Image.Image,
    scores: np.ndarray,
    title: str = "",
    save_path: str = None,
    show: bool = False,
):
   
    Side-by-side: original image | occlusion heatmap overlaid on image.
    Proposal §3b: flag patches should have the highest scores for flag images.
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Upsample heatmap to image resolution for overlay
    W, H = image.size
    scores_img = np.array(
        Image.fromarray(scores.astype(np.float32)).resize((W, H), Image.NEAREST)
    )

    axes[1].imshow(image)
    # Print score range for debugging
    print(f"{title} | "f"min={scores.min():.4f} "f"max={scores.max():.4f} "f"mean={scores.mean():.4f}")

    # Diverging colour map centred at zero
    norm = TwoSlopeNorm(vmin=float(scores.min()),vcenter=0.0,vmax=float(scores.max()))
    im = axes[1].imshow(scores_img,cmap="RdBu_r",alpha=0.60,norm=norm)
    axes[1].set_title(f"Occlusion Sensitivity\n{title}", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def show_heatmap_grid(
    image: Image.Image,
    scores: np.ndarray,
    grid_size: int = GRID_SIZE,
    title: str = "",
    save_path: str = None,
):
  
    Renders explicit patch grid lines so individual patches are visible.
    Useful for verifying that the grid aligns with the model's ViT patches.
    
    W, H = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)

    # Draw grid on original
    for i in range(1, grid_size):
        axes[0].axhline(i * patch_h, color="white", linewidth=0.5, alpha=0.6)
        axes[0].axvline(i * patch_w, color="white", linewidth=0.5, alpha=0.6)
    axes[0].set_title("Original + Patch Grid", fontsize=10)
    axes[0].axis("off")

    # Raw patch-level heatmap (no interpolation)
    norm = TwoSlopeNorm(
    vmin=float(scores.min()),
    vcenter=0.0,
    vmax=float(scores.max()))

    im = axes[1].imshow(scores,cmap="RdBu_r",interpolation="nearest",norm=norm)
    axes[1].set_title(f"Patch Scores ({grid_size}×{grid_size})\n{title}", fontsize=9)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Annotate top-3 patches
    flat_ranks = np.argsort(scores.flatten())[::-1]
    for rank, flat_idx in enumerate(flat_ranks[:3]):
        pi, pj = np.unravel_index(flat_idx, scores.shape)
        axes[1].text(
            pj, pi, f"#{rank+1}",
            ha="center", va="center", fontsize=7,
            color="cyan", fontweight="bold",
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def show_fill_comparison(
    image: Image.Image,
    scores_dict: dict,       # {"mean": ndarray, "black": ndarray, "noise": ndarray}
    title: str = "",
    save_path: str = None,
):
    
    Compare occlusion heatmaps produced by different fill strategies side-by-side.
    Proposal §9: 'Report results for both grey-fill and Gaussian-noise-fill;
    if rankings are stable across both, report the average.'
    
    fills = list(scores_dict.keys())
    n = len(fills)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    W, H = image.size
    for ax, fill_name in zip(axes[1:], fills):
        scores = scores_dict[fill_name]
        scores_img = np.array(Image.fromarray(scores.astype(np.float32)).resize((W, H), Image.BILINEAR))
        norm = TwoSlopeNorm(vmin=float(scores.min()),vcenter=0.0,vmax=float(scores.max()))

        ax.imshow(image)

        ax.imshow(scores_img,cmap="RdBu_r",alpha=0.60,norm=norm,)
        ax.set_title(f"Fill: {fill_name}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_from_dataset(u_id: str) -> Image.Image:
    ds = load_from_disk(DATASET_PATH)["test"]
    for row in ds:
        if row["u_id"] == u_id:
            return row["image"]
    raise KeyError(f"u_id {u_id!r} not found in dataset.")


def load_scores(u_id: str, model: str, fill: str = "mean") -> np.ndarray:
    path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model}_{fill}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Heatmap not found: {path}\n"
            f"Run occlusion.py --model {model} --fill {fill} first."
        )
    return np.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def visualise_single(u_id: str, model: str, fill: str):
    os.makedirs(VIZ_DIR, exist_ok=True)
    image  = load_image_from_dataset(u_id)
    scores = load_scores(u_id, model, fill)
    title  = f"{u_id} | {model} | fill={fill}"

    show_heatmap(
        image, scores, title=title,
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_{fill}_overlay.png"),
    )
    show_heatmap_grid(
        image, scores, title=title,
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_{fill}_grid.png"),
    )


def visualise_pilot(model: str, fill: str):
    if not os.path.exists(PILOT_SUMMARY):
        raise FileNotFoundError(
            f"{PILOT_SUMMARY} not found. Run run_occlusion_pilot.py first."
        )
    summary = pd.read_csv(PILOT_SUMMARY)
    print(f"Visualising {len(summary)} pilot images...")
    for _, row in summary.iterrows():
        u_id = row["u_id"]
        try:
            visualise_single(u_id, model, fill)
        except FileNotFoundError as e:
            print(f"  [skip] {u_id}: {e}")


def visualise_fill_comparison(u_id: str, model: str):
    os.makedirs(VIZ_DIR, exist_ok=True)
    image       = load_image_from_dataset(u_id)
    fills       = ["mean", "black", "noise"]
    scores_dict = {}
    for fill in fills:
        try:
            scores_dict[fill] = load_scores(u_id, model, fill)
        except FileNotFoundError:
            print(f"[skip] fill={fill} not found for {u_id} — run occlusion.py --fill {fill}")

    if not scores_dict:
        print("No heatmaps found for any fill type.")
        return

    show_fill_comparison(
        image, scores_dict,
        title=f"{u_id} | {model} — fill comparison",
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_fill_comparison.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3b — Heatmap Visualisation")
    parser.add_argument("--u_id",          type=str,  default=None,    help="Image u_id to visualise.")
    parser.add_argument("--model",         type=str,  default="clip",  help="Model name (clip|llava|qwen2vl|internvl2).")
    parser.add_argument("--fill",          type=str,  default="mean",  help="Fill type (mean|black|noise).")
    parser.add_argument("--pilot",         action="store_true",        help="Visualise all 20 pilot images.")
    parser.add_argument("--compare_fills", action="store_true",        help="Side-by-side comparison of all three fill types.")
    args = parser.parse_args()

    if args.pilot:
        visualise_pilot(args.model, args.fill)
    elif args.compare_fills and args.u_id:
        visualise_fill_comparison(args.u_id, args.model)
    elif args.u_id:
        visualise_single(args.u_id, args.model, args.fill)
    else:
        parser.print_help()"""


import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image
from datasets import load_from_disk

from occlusion import DATASET_PATH, OCCLUSION_DIR, SAMPLE_IDS_CSV, GRID_SIZE

PILOT_SUMMARY = os.path.join(OCCLUSION_DIR, "pilot_summary.csv")
VIZ_DIR       = os.path.join("results", "heatmaps")

def get_norm(scores):
    smin = float(scores.min())
    smax = float(scores.max())

    # Mixed positive and negative values
    if smin < 0 < smax:
        return TwoSlopeNorm(
            vmin=smin,
            vcenter=0.0,
            vmax=smax
        )

    # All-positive or all-negative heatmaps
    return Normalize(
        vmin=smin,
        vmax=smax
    )
# ---------------------------------------------------------------------------
# Core visualisation function
# ---------------------------------------------------------------------------

def show_heatmap(
    image: Image.Image,
    scores: np.ndarray,
    title: str = "",
    save_path: str = None,
    show: bool = False,
):
    """
    Side-by-side: original image | occlusion heatmap overlaid on image.
    Proposal §3b: flag patches should have the highest scores for flag images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Upsample heatmap to image resolution for overlay
    W, H = image.size
    scores_img = np.array(
        Image.fromarray(scores.astype(np.float32)).resize((W, H), Image.NEAREST)
    )

    axes[1].imshow(image)
    # Print score range for debugging
    print(f"{title} | "f"min={scores.min():.4f} "f"max={scores.max():.4f} "f"mean={scores.mean():.4f}")

    # Diverging colour map centred at zero
    norm = get_norm(scores)
    im = axes[1].imshow(scores_img,cmap="RdBu_r",alpha=0.60,norm=norm)
    axes[1].set_title(f"Occlusion Sensitivity\n{title}", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def show_heatmap_grid(
    image: Image.Image,
    scores: np.ndarray,
    grid_size: int = GRID_SIZE,
    title: str = "",
    save_path: str = None,
):
    """
    Renders explicit patch grid lines so individual patches are visible.
    Useful for verifying that the grid aligns with the model's ViT patches.
    """
    W, H = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)

    # Draw grid on original
    for i in range(1, grid_size):
        axes[0].axhline(i * patch_h, color="white", linewidth=0.5, alpha=0.6)
        axes[0].axvline(i * patch_w, color="white", linewidth=0.5, alpha=0.6)
    axes[0].set_title("Original + Patch Grid", fontsize=10)
    axes[0].axis("off")

    # Raw patch-level heatmap (no interpolation)
    norm = get_norm(scores)

    im = axes[1].imshow(scores,cmap="RdBu_r",interpolation="nearest",norm=norm)
    axes[1].set_title(f"Patch Scores ({grid_size}×{grid_size})\n{title}", fontsize=9)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Annotate top-3 patches
    flat_ranks = np.argsort(scores.flatten())[::-1]
    for rank, flat_idx in enumerate(flat_ranks[:3]):
        pi, pj = np.unravel_index(flat_idx, scores.shape)
        axes[1].text(
            pj, pi, f"#{rank+1}",
            ha="center", va="center", fontsize=7,
            color="cyan", fontweight="bold",
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def show_fill_comparison(
    image: Image.Image,
    scores_dict: dict,       # {"mean": ndarray, "black": ndarray, "noise": ndarray}
    title: str = "",
    save_path: str = None,
):
    """
    Compare occlusion heatmaps produced by different fill strategies side-by-side.
    Proposal §9: 'Report results for both grey-fill and Gaussian-noise-fill;
    if rankings are stable across both, report the average.'
    """
    fills = list(scores_dict.keys())
    n = len(fills)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    W, H = image.size
    for ax, fill_name in zip(axes[1:], fills):
        scores = scores_dict[fill_name]
        scores_img = np.array(Image.fromarray(scores.astype(np.float32)).resize((W, H), Image.NEAREST))

        norm = get_norm(scores)
        ax.imshow(image)

        ax.imshow(scores_img,cmap="RdBu_r",alpha=0.60,norm=norm,)
        ax.set_title(f"Fill: {fill_name}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_from_dataset(u_id: str) -> Image.Image:
    ds = load_from_disk(DATASET_PATH)["test"]
    for row in ds:
        if row["u_id"] == u_id:
            return row["image"]
    raise KeyError(f"u_id {u_id!r} not found in dataset.")


def load_scores(u_id: str, model: str, fill: str = "mean") -> np.ndarray:
    path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model}_{fill}_7x7.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Heatmap not found: {path}\n"
            f"Run occlusion.py --model {model} --fill {fill} --grid-size 7 first."
        )
    return np.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def visualise_single(u_id: str, model: str, fill: str):
    os.makedirs(VIZ_DIR, exist_ok=True)
    image  = load_image_from_dataset(u_id)
    scores = load_scores(u_id, model, fill)
    title  = f"{u_id} | {model} | fill={fill}"

    show_heatmap(
        image, scores, title=title,
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_{fill}_7x7_overlay.png"),
    )
    show_heatmap_grid(
        image, scores, title=title,
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_{fill}_7x7_grid.png"),
    )


def visualise_pilot(model: str, fill: str):
    if not os.path.exists(PILOT_SUMMARY):
        raise FileNotFoundError(
            f"{PILOT_SUMMARY} not found. Run run_occlusion_pilot.py first."
        )
    summary = pd.read_csv(PILOT_SUMMARY)
    print(f"Visualising {len(summary)} pilot images...")
    for _, row in summary.iterrows():
        u_id = row["u_id"]
        try:
            visualise_single(u_id, model, fill)
        except FileNotFoundError as e:
            print(f"  [skip] {u_id}: {e}")


def visualise_fill_comparison(u_id: str, model: str):
    os.makedirs(VIZ_DIR, exist_ok=True)
    image       = load_image_from_dataset(u_id)
    fills       = ["mean", "black", "noise"]
    scores_dict = {}
    for fill in fills:
        try:
            scores_dict[fill] = load_scores(u_id, model, fill)
        except FileNotFoundError:
            print(f"[skip] fill={fill} not found for {u_id} — run occlusion.py --fill {fill}")

    if not scores_dict:
        print("No heatmaps found for any fill type.")
        return

    show_fill_comparison(
        image, scores_dict,
        title=f"{u_id} | {model} — fill comparison",
        save_path=os.path.join(VIZ_DIR, f"{u_id}_{model}_fill_comparison.png"),
    )

def visualise_all(model: str, fill: str):
    os.makedirs(VIZ_DIR, exist_ok=True)

    suffix = f"_{model}_{fill}_7x7.npy"

    files = [
        f for f in os.listdir(OCCLUSION_DIR)
        if f.endswith(suffix)
    ]

    print(f"Found {len(files)} heatmaps")

    for idx, fname in enumerate(files):
        u_id = fname[:-len(suffix)]

        try:
            visualise_single(u_id, model, fill)
        except Exception as e:
            print(f"[skip] {u_id}: {e}")

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3b — Heatmap Visualisation")
    parser.add_argument("--u_id",          type=str,  default=None,    help="Image u_id to visualise.")
    parser.add_argument("--model",         type=str,  default="clip",  help="Model name (clip|llava|qwen2vl|internvl2).")
    parser.add_argument("--fill",          type=str,  default="mean",  help="Fill type (mean|black|noise).")
    parser.add_argument("--pilot",         action="store_true",        help="Visualise all 20 pilot images.")
    parser.add_argument("--all",action="store_true",help="Visualise all available heatmaps.")
    parser.add_argument("--compare_fills", action="store_true",        help="Side-by-side comparison of all three fill types.")
    args = parser.parse_args()

    if args.all:
        visualise_all(args.model, args.fill)
    elif args.pilot:
        visualise_pilot(args.model, args.fill)
    elif args.compare_fills and args.u_id:
        visualise_fill_comparison(args.u_id, args.model)
    elif args.u_id:
        visualise_single(args.u_id, args.model, args.fill)
    else:
        parser.print_help()