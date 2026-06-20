"""
Step 2 — CLIP Zero-Shot Cue Category Detection
Research Proposal §4 + prerequisite for §6.4 (CAS) and §6.5 (Faithfulness)

For every image, computes CLIP cosine similarity against 8 visual cue category
prompts (A–H) and assigns:
  • image-level label  → which category dominates the full image
  • patch-level labels → which category dominates each of the top-3 most
                         attributed patches (from occlusion sensitivity maps)
                         Patch labels feed directly into faithfulness evaluation.

Output files (both saved to results/):
  cue_categories.csv       — image-level scores and assigned labels
  top_patch_categories.csv — category of each image's top-3 occlusion patches

Categories (A–H) from proposal §6.2:
  A — National Symbols   (flags, emblems, monuments)
  B — Clothing / Dress   (traditional garments, headwear)
  C — Architecture       (building style, rooftops, ornaments)
  D — Food / Objects     (dishes, utensils, cultural tools)
  E — Script / Text      (written language visible in image)
  F — Ritual / Festival  (ceremonies, decorations, gatherings)
  G — Natural Landscape  (terrain, vegetation, climate signals)
  H — Appearance         (skin tone, facial features — bias flag)

Usage:
    python detect_cue_categories.py
    python detect_cue_categories.py --pilot          # first 20 images only
    python detect_cue_categories.py --top_k 5        # top-5 patches instead of top-3
    python detect_cue_categories.py --model_occlusion llava  # use LLaVA maps for patches
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from datasets import load_from_disk

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
OCCLUSION_DIR = os.path.join(RESULTS_DIR,  "occlusion","occlusion")
IDS_CSV       = os.path.join(RESULTS_DIR,  "sample_ids.csv")
META_CSV      = os.path.join(RESULTS_DIR,  "sample_metadata.csv")

OUT_IMAGE = os.path.join(RESULTS_DIR, "cue_categories.csv")
# OUT_PATCH is model-specific — set after args are parsed

# ── Category taxonomy (proposal §6.2) ─────────────────────────────────────────
CATEGORIES = {
    "A_national_symbols": [
        "a photo showing a national flag or national emblem",
        "a photo of a national monument or patriotic symbol",
    ],
    "B_clothing": [
        "traditional cultural clothing or headwear",
        "a person wearing traditional ethnic garments or costume",
    ],
    "C_architecture": [
        "distinctive regional architecture or building style",
        "a temple, mosque, church, or cultural landmark building",
    ],
    "D_food": [
        "traditional food, dishes, or utensils",
        "a photo of cultural food preparation or a meal",
    ],
    "E_script": [
        "visible written script, text, or signage in a non-Latin alphabet",
        "cultural symbols, written language, or decorative script",
    ],
    "F_ritual": [
        "a cultural ceremony, festival, or ritual gathering",
        "people participating in a traditional celebration or religious event",
    ],
    "G_landscape": [
        "distinctive natural landscape, terrain, or vegetation",
        "a scenic photo showing geography or climate of a region",
    ],
    "H_appearance": [
        "people whose physical appearance or skin tone is the primary visual element",
        "a portrait where ethnicity or race of the person is most prominent",
    ],
}

CATEGORY_KEYS   = list(CATEGORIES.keys())
CATEGORY_LABELS = [k.split("_")[0] for k in CATEGORY_KEYS]   # A, B, C, …, H


def load_occlusion(u_id: str, model: str = "clip") -> np.ndarray | None:
    """Try 14×14 first, then 7×7 suffix."""
    for fname in [
        f"{u_id}_{model}_mean.npy",
        f"{u_id}_{model}_mean_7x7.npy",
    ]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def top_k_patches(scores: np.ndarray, k: int = 3):
    """Return (row, col) indices of the top-k patches by occlusion score."""
    flat_idx = np.argsort(scores.flatten())[::-1][:k]
    rows = flat_idx // scores.shape[1]
    cols = flat_idx %  scores.shape[1]
    return list(zip(rows.tolist(), cols.tolist()))


def crop_patch(image: Image.Image, row: int, col: int,
               grid_rows: int, grid_cols: int) -> Image.Image:
    """Crop one grid cell from the image."""
    W, H    = image.size
    pw      = W // grid_cols
    ph      = H // grid_rows
    x0, y0  = col * pw, row * ph
    x1, y1  = x0 + pw, y0 + ph
    return image.crop((x0, y0, x1, y1))


@torch.no_grad()
def compute_category_scores(
    clip_model,
    preprocess,
    text_feats: torch.Tensor,   # (n_categories, d)  pre-encoded
    image: Image.Image,
    device: str,
) -> np.ndarray:
    """
    Returns (n_categories,) array of softmax probabilities.
    Uses the mean of multi-prompt embeddings per category.
    """
    img_t  = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    img_f  = clip_model.encode_image(img_t)
    img_f  = img_f / img_f.norm(dim=-1, keepdim=True)
    sims   = (100.0 * img_f @ text_feats.T).squeeze(0)   # (n_categories,)
    probs  = torch.softmax(sims, dim=0).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot cue category detection")
    parser.add_argument("--pilot",            action="store_true", help="Run on first 20 images only.")
    parser.add_argument("--top_k",            type=int, default=3,  help="Number of top patches to label (default: 3).")
    parser.add_argument("--model_occlusion",  type=str, default="clip",
                        help="Which model's occlusion maps to use for patch selection (default: clip).")
    args = parser.parse_args()

    # Patch output is per-model (different models attend to different patches)
    out_patch = os.path.join(RESULTS_DIR, f"top_patch_categories_{args.model_occlusion}.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Occlusion model for patches: {args.model_occlusion}")

    # ── Load CLIP ──────────────────────────────────────────────────────────────
    print("Loading CLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # ── Pre-encode category text prompts ──────────────────────────────────────
    # For each category, encode all prompts and average → one embedding per category
    print("Encoding category prompts...")
    cat_embeddings = []
    for cat_key, prompts in CATEGORIES.items():
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cat_embeddings.append(feats.mean(dim=0))   # mean over prompts
    text_feats = torch.stack(cat_embeddings)   # (8, d)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    print(f"  {len(CATEGORY_KEYS)} categories encoded.")

    # ── Load sample ───────────────────────────────────────────────────────────
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    meta       = pd.read_csv(META_CSV).set_index("u_id")

    if args.pilot:
        sample_ids = sample_ids[:20]
        print(f"Pilot mode: {len(sample_ids)} images.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset...")
    ds     = load_from_disk(DATASET_PATH)["test"]
    id_set = set(sample_ids)
    ds     = ds.filter(lambda x: x["u_id"] in id_set)
    id_to_image = {row["u_id"]: row["image"] for row in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    image_records = []
    patch_records = []

    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image:
            print(f"[warn] {u_id} not in dataset — skipping")
            continue

        image = id_to_image[u_id]
        meta_row = meta.loc[u_id] if u_id in meta.index else None

        # ── A. Image-level category scores ────────────────────────────────────
        probs = compute_category_scores(model, preprocess, text_feats, image, device)

        rec = {
            "u_id":          u_id,
            "facet":         meta_row["facet"]   if meta_row is not None else None,
            "true_country":  meta_row["country"] if meta_row is not None else None,
        }
        # Raw scores per category
        for label, key, p in zip(CATEGORY_LABELS, CATEGORY_KEYS, probs):
            rec[f"score_{label}"] = round(float(p), 6)

        # Primary label (argmax)
        primary_idx      = int(probs.argmax())
        rec["label_primary"]    = CATEGORY_LABELS[primary_idx]
        rec["label_primary_key"]= CATEGORY_KEYS[primary_idx]
        rec["score_primary"]    = round(float(probs[primary_idx]), 6)

        # Secondary label (second-highest)
        sorted_idx = probs.argsort()[::-1]
        rec["label_secondary"]  = CATEGORY_LABELS[int(sorted_idx[1])]
        rec["score_secondary"]  = round(float(probs[sorted_idx[1]]), 6)

        # Is the primary label clearly dominant? (primary score > 2× secondary)
        rec["label_confident"] = bool(probs[primary_idx] > 2.0 * probs[int(sorted_idx[1])])

        image_records.append(rec)

        # ── B. Patch-level category scores ────────────────────────────────────
        occ = load_occlusion(u_id, args.model_occlusion)
        if occ is not None:
            grid_size = occ.shape[0]   # 14 for CLIP, 7 for LLaVA
            top_patches = top_k_patches(occ, k=args.top_k)

            for rank, (r, c) in enumerate(top_patches, 1):
                patch_img = crop_patch(image, r, c, grid_size, grid_size)
                patch_probs = compute_category_scores(
                    model, preprocess, text_feats, patch_img, device
                )
                p_idx = int(patch_probs.argmax())
                patch_rec = {
                    "u_id":          u_id,
                    "facet":         meta_row["facet"]   if meta_row is not None else None,
                    "true_country":  meta_row["country"] if meta_row is not None else None,
                    "patch_rank":    rank,
                    "patch_row":     r,
                    "patch_col":     c,
                    "occlusion_score": round(float(occ[r, c]), 6),
                    "label":         CATEGORY_LABELS[p_idx],
                    "label_key":     CATEGORY_KEYS[p_idx],
                    "label_score":   round(float(patch_probs[p_idx]), 6),
                }
                # Store all scores for full analysis
                for label, p in zip(CATEGORY_LABELS, patch_probs):
                    patch_rec[f"score_{label}"] = round(float(p), 6)
                patch_records.append(patch_rec)

        if idx % 25 == 0 or idx == len(sample_ids):
            print(f"  [{idx}/{len(sample_ids)}] {u_id} → {rec['label_primary']} "
                  f"({rec['score_primary']:.3f})  patches: {len(top_patches) if occ is not None else 'n/a'}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_images = pd.DataFrame(image_records)
    df_images.to_csv(OUT_IMAGE, index=False)
    print(f"\nImage-level labels saved → {OUT_IMAGE}  ({len(df_images)} rows)")

    if patch_records:
        df_patches = pd.DataFrame(patch_records)
        df_patches.to_csv(out_patch, index=False)
        print(f"Patch-level labels saved → {out_patch}  ({len(df_patches)} rows)")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n── Category distribution (image-level, primary label) ───────────")
    print(df_images["label_primary"].value_counts().to_string())

    if patch_records:
        print("\n── Category distribution (top-patch labels) ─────────────────────")
        print(df_patches["label"].value_counts().to_string())

    print(f"\n── Confident labels (primary > 2× secondary): "
          f"{df_images['label_confident'].mean():.1%} of images")

    print("\nDone.")


if __name__ == "__main__":
    main()
