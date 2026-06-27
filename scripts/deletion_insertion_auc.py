"""
Measures whether occlusion sensitivity maps are CAUSALLY meaningful —
i.e., the ranked patches actually drive the model's prediction.

Deletion AUC (ROAR-style):
  - Rank all patches by occlusion score (descending = most important first)
  - Progressively mask patches in ranked order; re-run model after each step
  - Record confidence in predicted country at each deletion level
  - Repeat with RANDOM order as baseline
  - Attribution quality = AUC(ranked deletion) − AUC(random deletion)
  - A large positive gap means ranked patches are genuinely responsible

Insertion AUC:
  - Start from a fully blurred image (Gaussian σ=10)
  - Progressively REVEAL patches in ranked order; re-run model after each step
  - Record confidence at each insertion level
  - Repeat with RANDOM order
  - Insertion score = AUC(ranked insertion) − AUC(random insertion)

Together they give a two-sided faithfulness check.
Both are run on CLIP by default (fastest model, 266 complete 14×14 maps).

Output files (results/analysis/):
  deletion_auc.csv     — per-image deletion AUC gap
  insertion_auc.csv    — per-image insertion AUC gap
  auc_summary.csv      — mean ± std per model, key table for paper

Usage:
    python deletion_insertion_auc.py
    python deletion_insertion_auc.py --model clip --steps 10
    python deletion_insertion_auc.py --model llava --steps 10
    python deletion_insertion_auc.py --skip_insertion    # deletion only
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageFilter
from datasets import load_from_disk

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
OCCLUSION_DIR = os.path.join(RESULTS_DIR,  "occlusion","occlusion")
ANALYSIS_DIR  = os.path.join(RESULTS_DIR,  "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

IDS_CSV  = os.path.join(RESULTS_DIR, "sample_ids.csv")
META_CSV = os.path.join(RESULTS_DIR, "sample_metadata.csv")
PRED_CSVS = {
    "clip":      os.path.join(RESULTS_DIR, "clip_predictions.csv"),
    "llava":     os.path.join(RESULTS_DIR, "llava_predictions.csv"),
    "qwen2vl":   os.path.join(RESULTS_DIR, "qwen2vl_predictions.csv"),
    "internvl2": os.path.join(RESULTS_DIR, "internvl2_predictions.csv"),
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_occlusion(u_id: str, model: str) -> np.ndarray | None:
    for fname in [
        f"{u_id}_{model}_mean.npy",
        f"{u_id}_{model}_mean_7x7.npy",
    ]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


# ── Image masking utilities ───────────────────────────────────────────────────

def mask_patches(image: Image.Image, patch_indices: list[int],
                 grid_h: int, grid_w: int,
                 fill: str = "mean") -> Image.Image:
    """
    Return a copy of image with specified patch indices replaced.
    patch_indices: flat indices into (grid_h × grid_w) grid.
    fill: 'mean' (per-image mean colour) or 'blur' (Gaussian blur of full image).
    """
    img_arr = np.array(image.convert("RGB"))
    H, W    = img_arr.shape[:2]
    ph, pw  = H // grid_h, W // grid_w

    if fill == "mean":
        fill_val = img_arr.mean(axis=(0, 1)).astype(np.uint8)
    else:   # 'blur' — used for insertion baseline starting image
        fill_val = None

    for flat_idx in patch_indices:
        r, c = flat_idx // grid_w, flat_idx % grid_w
        y0, x0 = r * ph, c * pw
        if fill_val is not None:
            img_arr[y0:y0 + ph, x0:x0 + pw] = fill_val
        else:
            # For blur: copy blurred pixels (done separately)
            pass

    return Image.fromarray(img_arr)


def make_blurred_base(image: Image.Image, sigma: float = 10.0) -> Image.Image:
    """Gaussian-blurred version of the image — starting point for insertion."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def reveal_patches(blurred: Image.Image, original: Image.Image,
                   patch_indices: list[int],
                   grid_h: int, grid_w: int) -> Image.Image:
    """
    Start from blurred image, reveal the specified patches from the original.
    Used for insertion AUC.
    """
    base_arr = np.array(blurred.convert("RGB"))
    orig_arr = np.array(original.convert("RGB"))
    H, W     = base_arr.shape[:2]
    ph, pw   = H // grid_h, W // grid_w

    for flat_idx in patch_indices:
        r, c = flat_idx // grid_w, flat_idx % grid_w
        y0, x0 = r * ph, c * pw
        base_arr[y0:y0 + ph, x0:x0 + pw] = orig_arr[y0:y0 + ph, x0:x0 + pw]

    return Image.fromarray(base_arr)


# ── CLIP confidence wrapper ───────────────────────────────────────────────────

def build_clip_conf_fn(clip_model, preprocess, text_feats: torch.Tensor,
                       target_idx: int, device: str):
    """Returns a closure: PIL.Image → float confidence for target country."""
    def fn(image: Image.Image) -> float:
        img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sims = (100.0 * feat @ text_feats.T).squeeze(0)
            probs = torch.softmax(sims, dim=0).cpu().numpy()
        return float(probs[target_idx])
    return fn


# ── AUC computation ───────────────────────────────────────────────────────────

def compute_deletion_curve(conf_fn, image: Image.Image,
                            ranked_order: list[int],
                            step_fractions: list[float],
                            grid_h: int, grid_w: int) -> list[float]:
    """
    Returns confidence at each deletion fraction level.
    step_fractions: e.g. [0.0, 0.1, 0.2, ..., 1.0]
    At fraction f: mask top (f × N) patches.
    """
    n_patches = grid_h * grid_w
    confidences = []
    for frac in step_fractions:
        k = int(round(frac * n_patches))
        masked_img = mask_patches(image, ranked_order[:k], grid_h, grid_w, fill="mean")
        confidences.append(conf_fn(masked_img))
    return confidences


def compute_insertion_curve(conf_fn, image: Image.Image,
                             ranked_order: list[int],
                             step_fractions: list[float],
                             grid_h: int, grid_w: int,
                             blurred: Image.Image) -> list[float]:
    """
    Returns confidence at each insertion fraction level.
    At fraction f: reveal top (f × N) patches from original onto blurred base.
    """
    n_patches = grid_h * grid_w
    confidences = []
    for frac in step_fractions:
        k = int(round(frac * n_patches))
        revealed_img = reveal_patches(blurred, image, ranked_order[:k], grid_h, grid_w)
        confidences.append(conf_fn(revealed_img))
    return confidences


def auc(curve: list[float]) -> float:
    """Trapezoidal AUC normalised to [0, 1] x-axis."""
    return float(np.trapezoid(curve, dx=1.0 / (len(curve) - 1)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deletion & Insertion AUC")
    parser.add_argument("--model",           default="clip",
                        choices=["clip", "llava", "qwen2vl", "internvl2"])
    parser.add_argument("--steps",           type=int, default=10,
                        help="Number of deletion/insertion steps (default: 10).")
    parser.add_argument("--skip_insertion",  action="store_true",
                        help="Skip insertion AUC (faster).")
    parser.add_argument("--pilot",           action="store_true",
                        help="Run on first 20 images only.")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed for baseline order.")
    args = parser.parse_args()

    model_name = args.model
    N_STEPS    = args.steps
    rng        = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {model_name} | Steps: {N_STEPS}")

    # Deletion/insertion fraction levels: 0%, 10%, ..., 100%
    step_fracs = [i / N_STEPS for i in range(N_STEPS + 1)]
    print(f"Step fractions: {[f'{f:.0%}' for f in step_fracs]}")

    # ── Load predictions ───────────────────────────────────────────────────────
    if not os.path.exists(PRED_CSVS[model_name]):
        raise FileNotFoundError(f"{PRED_CSVS[model_name]} not found.")
    preds = pd.read_csv(PRED_CSVS[model_name]).set_index("u_id")
    countries = sorted(pd.read_csv(META_CSV)["country"].unique().tolist())

    # ── Load CLIP for confidence extraction ────────────────────────────────────
    # CLIP is always used as the confidence evaluator since:
    # (a) it's fast (no generation needed)
    # (b) it's the primary model with complete 14×14 maps
    # (c) for cross-model comparison we use each model's OWN predicted country
    #     but CLIP's confidence function (proposal-consistent)
    print("Loading CLIP ViT-L/14 for confidence extraction...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        text_tokens = tokenizer([f"a photo from {c}" for c in countries]).to(device)
        text_feats  = clip_model.encode_text(text_tokens)
        text_feats  = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # ── Load sample IDs ────────────────────────────────────────────────────────
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    if args.pilot:
        sample_ids = sample_ids[:20]
        print(f"Pilot mode: {len(sample_ids)} images.")

    # ── Load dataset ───────────────────────────────────────────────────────────
    print("Loading dataset images...")
    ds = load_from_disk(DATASET_PATH)["test"]
    id_set = set(sample_ids)
    ds = ds.filter(lambda x: x["u_id"] in id_set)
    id_to_image = {row["u_id"]: row["image"] for row in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    del_records = []
    ins_records = []
    processed = skipped = 0

    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image:
            continue

        # Load occlusion map
        scores = load_occlusion(u_id, model_name)
        if scores is None:
            skipped += 1
            continue

        grid_h, grid_w = scores.shape
        n_patches = grid_h * grid_w
        image = id_to_image[u_id].convert("RGB")

        # Target country = what this model predicted
        if u_id not in preds.index:
            skipped += 1
            continue
        pred_country = preds.loc[u_id, "pred_country"]
        if pred_country not in countries:
            skipped += 1
            continue

        target_idx = countries.index(pred_country)
        conf_fn    = build_clip_conf_fn(clip_model, preprocess, text_feats, target_idx, device)

        # Patch ordering
        flat_scores  = scores.flatten()
        ranked_order = np.argsort(flat_scores)[::-1].tolist()  # most → least important
        random_order = rng.permutation(n_patches).tolist()

        # ── Deletion AUC ──────────────────────────────────────────────────────
        ranked_del_curve = compute_deletion_curve(
            conf_fn, image, ranked_order, step_fracs, grid_h, grid_w
        )
        random_del_curve = compute_deletion_curve(
            conf_fn, image, random_order, step_fracs, grid_h, grid_w
        )
        auc_ranked_del = auc(ranked_del_curve)
        auc_random_del = auc(random_del_curve)
        del_gap        = auc_ranked_del - auc_random_del   # negative = good (ranked drops faster)

        del_records.append({
            "u_id":            u_id,
            "model":           model_name,
            "pred_country":    pred_country,
            "correct":         bool(preds.loc[u_id, "correct"]),
            "auc_ranked":      round(auc_ranked_del, 5),
            "auc_random":      round(auc_random_del, 5),
            "auc_gap":         round(del_gap, 5),
            # Store full curve for plotting
            **{f"conf_step{i}": round(v, 5) for i, v in enumerate(ranked_del_curve)},
        })

        # ── Insertion AUC ─────────────────────────────────────────────────────
        if not args.skip_insertion:
            blurred = make_blurred_base(image, sigma=10.0)

            ranked_ins_curve = compute_insertion_curve(
                conf_fn, image, ranked_order, step_fracs, grid_h, grid_w, blurred
            )
            random_ins_curve = compute_insertion_curve(
                conf_fn, image, random_order, step_fracs, grid_h, grid_w, blurred
            )
            auc_ranked_ins = auc(ranked_ins_curve)
            auc_random_ins = auc(random_ins_curve)
            ins_gap        = auc_ranked_ins - auc_random_ins  # positive = good (ranked rises faster)

            ins_records.append({
                "u_id":         u_id,
                "model":        model_name,
                "pred_country": pred_country,
                "correct":      bool(preds.loc[u_id, "correct"]),
                "auc_ranked":   round(auc_ranked_ins, 5),
                "auc_random":   round(auc_random_ins, 5),
                "auc_gap":      round(ins_gap, 5),
            })

        processed += 1
        if idx % 25 == 0 or idx == len(sample_ids):
            g = del_records[-1]["auc_gap"] if del_records else 0
            print(f"  [{processed}/{len(sample_ids) - skipped}] {u_id} | "
                  f"del_gap={g:+.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_del = pd.DataFrame(del_records)
    del_path = os.path.join(ANALYSIS_DIR, f"deletion_auc_{model_name}.csv")
    df_del.to_csv(del_path, index=False)
    print(f"\nDeletion AUC saved → {del_path}  ({len(df_del)} rows)")

    if ins_records:
        df_ins = pd.DataFrame(ins_records)
        ins_path = os.path.join(ANALYSIS_DIR, f"insertion_auc_{model_name}.csv")
        df_ins.to_csv(ins_path, index=False)
        print(f"Insertion AUC saved → {ins_path}  ({len(df_ins)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Deletion AUC summary ({model_name}) ───────────────────────────")
    print(f"  Mean AUC (ranked)  : {df_del['auc_ranked'].mean():.4f}")
    print(f"  Mean AUC (random)  : {df_del['auc_random'].mean():.4f}")
    print(f"  Mean AUC gap       : {df_del['auc_gap'].mean():.4f}  "
          f"(negative = ranked patches drop confidence faster = GOOD)")
    print(f"  % images with gap < 0: {(df_del['auc_gap'] < 0).mean():.1%}")

    # Gap by correct vs incorrect
    for correct_val, label in [(True, "correct"), (False, "wrong")]:
        sub = df_del[df_del["correct"] == correct_val]
        if len(sub) > 0:
            print(f"  Mean gap ({label:7s}): {sub['auc_gap'].mean():.4f}  (n={len(sub)})")

    if ins_records:
        print(f"\n── Insertion AUC summary ({model_name}) ──────────────────────────")
        print(f"  Mean AUC (ranked)  : {df_ins['auc_ranked'].mean():.4f}")
        print(f"  Mean AUC (random)  : {df_ins['auc_random'].mean():.4f}")
        print(f"  Mean AUC gap       : {df_ins['auc_gap'].mean():.4f}  "
              f"(positive = ranked patches reveal confidence faster = GOOD)")
        print(f"  % images with gap > 0: {(df_ins['auc_gap'] > 0).mean():.1%}")

    # Combined summary for all models run so far
    _update_summary(model_name, df_del, df_ins if ins_records else None)

    print(f"\nDone.  Processed: {processed}  Skipped: {skipped}")


def _update_summary(model_name, df_del, df_ins):
    """Append this model's results to a combined summary table."""
    row = {
        "model":              model_name,
        "n_images":           len(df_del),
        "del_auc_ranked_mean": df_del["auc_ranked"].mean(),
        "del_auc_random_mean": df_del["auc_random"].mean(),
        "del_auc_gap_mean":    df_del["auc_gap"].mean(),
        "del_auc_gap_std":     df_del["auc_gap"].std(),
        "del_gap_pct_negative": (df_del["auc_gap"] < 0).mean(),
    }
    if df_ins is not None:
        row.update({
            "ins_auc_ranked_mean": df_ins["auc_ranked"].mean(),
            "ins_auc_random_mean": df_ins["auc_random"].mean(),
            "ins_auc_gap_mean":    df_ins["auc_gap"].mean(),
            "ins_auc_gap_std":     df_ins["auc_gap"].std(),
            "ins_gap_pct_positive":(df_ins["auc_gap"] > 0).mean(),
        })

    summary_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "analysis", "auc_summary.csv"
    )

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing["model"] != model_name]   # replace if re-run
        updated  = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.round(5).to_csv(summary_path, index=False)
    print(f"\nSummary table updated → {summary_path}")


if __name__ == "__main__":
    main()
