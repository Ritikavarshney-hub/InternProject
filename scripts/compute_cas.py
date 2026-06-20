"""
Step 4 — Causal Attribution Score (CAS) per Visual Cue Category
Research Proposal §6.4

CAS is the core finding of the paper — it answers RQ2:
"Do models disproportionately rely on stereotyped shortcut features?"

Procedure:
  1. Use cue_categories.csv (Step 2) to know which category (A–H) each image belongs to.
  2. For each image, mask its top-K% occlusion patches (grey fill = mean pixel).
  3. Re-run the model on the masked image.
  4. CAS(category C, model M, K%) =
       fraction of images in category C where masking causes M's prediction to change.

A high CAS for category A (National Symbols) means the model RELIES on flags/emblems.
A high CAS for category D (Food) or C (Architecture) means the model uses nuanced cues.
If CAS(A) >> CAS(C, D) → shortcut evidence (the central finding of the paper).

Run across multiple K values (10%, 20%, 30%) to show robustness.
CLIP is used as the re-evaluation model (fast, deterministic, complete maps).
LLaVA can be added with --model llava (uses LLaVA's occlusion maps + CLIP re-eval).

Inputs:
  results/cue_categories.csv               (from Step 2)
  results/occlusion/{u_id}_{model}_mean*.npy
  results/{model}_predictions.csv

Outputs:
  results/analysis/cas_scores.csv          — CAS per category × model × K value
  results/analysis/cas_per_image.csv       — per-image: did masking change prediction?
  results/analysis/cas_summary_plot.png    — bar chart (Figure 2 of paper)

Usage:
    python compute_cas.py
    python compute_cas.py --model clip --k_values 10 20 30
    python compute_cas.py --model llava
    python compute_cas.py --pilot
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from datasets import load_from_disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
OCCLUSION_DIR = os.path.join(RESULTS_DIR,  "occlusion","occlusion")
ANALYSIS_DIR  = os.path.join(RESULTS_DIR,  "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

IDS_CSV         = os.path.join(RESULTS_DIR, "sample_ids.csv")
META_CSV        = os.path.join(RESULTS_DIR, "sample_metadata.csv")
CUE_CSV         = os.path.join(RESULTS_DIR, "cue_categories.csv")
PRED_CSVS = {
    "clip":      os.path.join(RESULTS_DIR, "clip_predictions.csv"),
    "llava":     os.path.join(RESULTS_DIR, "llava_predictions.csv"),
    "qwen2vl":   os.path.join(RESULTS_DIR, "qwen2vl_predictions.csv"),
    "internvl2": os.path.join(RESULTS_DIR, "internvl2_predictions.csv"),
}

# Category taxonomy
CATEGORY_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H"]
CATEGORY_NAMES = {
    "A": "National Symbols",
    "B": "Clothing / Dress",
    "C": "Architecture",
    "D": "Food / Objects",
    "E": "Script / Text",
    "F": "Ritual / Festival",
    "G": "Natural Landscape",
    "H": "Appearance",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_occlusion(u_id: str, model: str) -> np.ndarray | None:
    for fname in [
        f"{u_id}_{model}_mean.npy",
        f"{u_id}_{model}_mean_7x7.npy",
    ]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def mask_top_k_patches(image: Image.Image, scores: np.ndarray,
                        k_pct: float) -> Image.Image:
    """
    Replace top-k% patches (by occlusion score) with per-image mean pixel.
    k_pct: fraction in [0, 1].
    """
    img_arr = np.array(image.convert("RGB"))
    H, W    = img_arr.shape[:2]
    gh, gw  = scores.shape
    ph, pw  = H // gh, W // gw

    mean_fill = img_arr.mean(axis=(0, 1)).astype(np.uint8)

    n_mask   = max(1, int(round(k_pct * gh * gw)))
    flat_idx = np.argsort(scores.flatten())[::-1][:n_mask]

    masked = img_arr.copy()
    for fi in flat_idx:
        r, c = fi // gw, fi % gw
        y0, x0 = r * ph, c * pw
        masked[y0:y0 + ph, x0:x0 + pw] = mean_fill

    return Image.fromarray(masked)


@torch.no_grad()
def clip_predict(clip_model, preprocess, text_feats: torch.Tensor,
                 countries: list, image: Image.Image, device: str) -> str:
    """Run CLIP on image and return the predicted country string."""
    img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    feat  = clip_model.encode_image(img_t)
    feat  = feat / feat.norm(dim=-1, keepdim=True)
    sims  = (100.0 * feat @ text_feats.T).squeeze(0)
    probs = torch.softmax(sims, dim=0).cpu().numpy()
    return countries[int(probs.argmax())]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute CAS per cue category")
    parser.add_argument("--model",    default="clip",
                        choices=["clip", "llava", "qwen2vl", "internvl2"],
                        help="Which model's occlusion maps to use.")
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 20, 30],
                        help="Top-K%% patch fractions to evaluate (default: 10 20 30).")
    parser.add_argument("--pilot",    action="store_true",
                        help="Run on first 20 images only.")
    args = parser.parse_args()

    model_name = args.model
    k_fracs    = [k / 100 for k in args.k_values]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Occlusion model: {model_name} | K values: {args.k_values}%")

    # ── Validate prerequisites ─────────────────────────────────────────────────
    if not os.path.exists(CUE_CSV):
        raise FileNotFoundError(
            f"{CUE_CSV} not found. Run detect_cue_categories.py first (Step 2)."
        )

    # ── Load cue labels ────────────────────────────────────────────────────────
    cue_df = pd.read_csv(CUE_CSV).set_index("u_id")
    print(f"Cue categories loaded: {len(cue_df)} images")
    print("  Primary label distribution:")
    print(cue_df["label_primary"].value_counts().to_string())

    # ── Load predictions ───────────────────────────────────────────────────────
    preds    = pd.read_csv(PRED_CSVS[model_name]).set_index("u_id")
    countries = sorted(pd.read_csv(META_CSV)["country"].unique().tolist())

    # ── Load CLIP ──────────────────────────────────────────────────────────────
    print("\nLoading CLIP ViT-L/14 for re-evaluation...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        text_tokens = tokenizer([f"a photo from {c}" for c in countries]).to(device)
        text_feats  = clip_model.encode_text(text_tokens)
        text_feats  = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # ── Load dataset ───────────────────────────────────────────────────────────
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    if args.pilot:
        sample_ids = sample_ids[:20]

    print("Loading dataset images...")
    ds = load_from_disk(DATASET_PATH)["test"]
    id_set = set(sample_ids)
    ds = ds.filter(lambda x: x["u_id"] in id_set)
    id_to_image = {row["u_id"]: row["image"] for row in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    per_image_records = []
    processed = skipped = 0

    for idx, u_id in enumerate(sample_ids, 1):
        # All prerequisites
        if u_id not in id_to_image:
            skipped += 1; continue
        if u_id not in cue_df.index:
            skipped += 1; continue
        if u_id not in preds.index:
            skipped += 1; continue

        scores = load_occlusion(u_id, model_name)
        if scores is None:
            skipped += 1; continue

        pred_country = preds.loc[u_id, "pred_country"]
        if pred_country not in countries:
            skipped += 1; continue

        image           = id_to_image[u_id].convert("RGB")
        category_primary = cue_df.loc[u_id, "label_primary"]        # e.g. "A"
        category_key     = cue_df.loc[u_id, "label_primary_key"]    # e.g. "A_national_symbols"

        rec = {
            "u_id":             u_id,
            "model":            model_name,
            "true_country":     cue_df.loc[u_id, "true_country"],
            "pred_country_orig":pred_country,
            "correct_orig":     bool(preds.loc[u_id, "correct"]),
            "category_primary": category_primary,
            "category_name":    CATEGORY_NAMES.get(category_primary, category_primary),
            "category_score":   float(cue_df.loc[u_id, "score_primary"]),
        }

        # For each K value: mask top-K% patches and check if prediction changes
        for k_pct, k_int in zip(k_fracs, args.k_values):
            masked_img        = mask_top_k_patches(image, scores, k_pct)
            new_pred          = clip_predict(clip_model, preprocess, text_feats,
                                            countries, masked_img, device)
            prediction_changed = (new_pred != pred_country)

            rec[f"pred_k{k_int}"]     = new_pred
            rec[f"changed_k{k_int}"]  = prediction_changed

        per_image_records.append(rec)
        processed += 1

        if idx % 25 == 0 or idx == len(sample_ids):
            # Show last record's change status
            chg = {k: rec[f"changed_k{k}"] for k in args.k_values}
            print(f"  [{processed}] {u_id} | cat={category_primary} | "
                  f"changes: { {k: '✓' if v else '✗' for k,v in chg.items()} }")

    # ── Save per-image results ─────────────────────────────────────────────────
    df_per_image = pd.DataFrame(per_image_records)
    per_image_path = os.path.join(ANALYSIS_DIR, f"cas_per_image_{model_name}.csv")
    df_per_image.to_csv(per_image_path, index=False)
    print(f"\nPer-image results saved → {per_image_path}  ({len(df_per_image)} rows)")

    # ── Compute CAS per category × K ──────────────────────────────────────────
    cas_records = []
    print("\n── CAS scores by category ────────────────────────────────────────")
    print(f"{'Category':<25} {'Name':<22}", end="")
    for k in args.k_values:
        print(f"{'K='+str(k)+'%':>8}", end="")
    print(f"  {'n':>4}")
    print("-" * 75)

    for cat in CATEGORY_KEYS:
        sub = df_per_image[df_per_image["category_primary"] == cat]
        if len(sub) == 0:
            continue

        row = {
            "category":      cat,
            "category_name": CATEGORY_NAMES.get(cat, cat),
            "model":         model_name,
            "n_images":      len(sub),
        }
        print(f"{cat:<3} {CATEGORY_NAMES.get(cat,''):<22} {CATEGORY_NAMES.get(cat,'')[:0]}", end="")
        print(f"  {cat:<3} {CATEGORY_NAMES.get(cat,''):<22}", end="")

        for k in args.k_values:
            col = f"changed_k{k}"
            cas = float(sub[col].mean()) if col in sub.columns else np.nan
            row[f"cas_k{k}"] = round(cas, 4)
            print(f"  {cas:>6.3f}", end="")

        row["n_correct_orig"] = int(sub["correct_orig"].sum())
        print(f"  {len(sub):>4}")
        cas_records.append(row)

    df_cas = pd.DataFrame(cas_records)

    # Pretty print
    print("\n── CAS table (fraction of images where masking top-K% patches "
          "changes prediction) ─")
    display_cols = ["category", "category_name", "n_images"] + \
                   [f"cas_k{k}" for k in args.k_values]
    print(df_cas[display_cols].to_string(index=False))

    cas_path = os.path.join(ANALYSIS_DIR, "cas_scores.csv")
    # Merge with existing if other models already ran
    if os.path.exists(cas_path):
        existing = pd.read_csv(cas_path)
        existing = existing[existing["model"] != model_name]
        df_cas   = pd.concat([existing, df_cas], ignore_index=True)

    df_cas.to_csv(cas_path, index=False)
    print(f"\nCAS scores saved → {cas_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_cas(df_cas, model_name, args.k_values)
    print("Done.")


def _plot_cas(df_cas: pd.DataFrame, highlight_model: str, k_values: list[int]):
    """Bar chart of CAS by category for the most representative K value (middle)."""
    k_mid  = k_values[len(k_values) // 2]
    col    = f"cas_k{k_mid}"

    sub = df_cas[df_cas["model"] == highlight_model].sort_values(col, ascending=False)
    if len(sub) == 0 or col not in sub.columns:
        return

    # Colour: red = shortcut categories (A, E, H), blue = nuanced (B, C, D, F, G)
    shortcut_cats = {"A", "E", "H"}
    colours = ["#d73027" if r["category"] in shortcut_cats else "#4575b4"
               for _, r in sub.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(sub)), sub[col], color=colours, edgecolor="white", width=0.6)

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [f"{r['category']}\n{r['category_name']}" for _, r in sub.iterrows()],
        fontsize=9
    )
    ax.set_ylabel(f"CAS at K={k_mid}%  (fraction of images where\nmasking top patches changes prediction)")
    ax.set_title(
        f"Causal Attribution Score (CAS) by visual cue category\n"
        f"Model: {highlight_model}  |  K={k_mid}%  |  "
        f"Red = shortcut categories, Blue = nuanced",
        fontsize=10
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.set_ylim(0, 1.05)

    # Annotate bars with n
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={row['n_images']}", ha="center", fontsize=7)

    ax.legend(fontsize=8)
    plt.tight_layout()

    fig_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "analysis", f"cas_plot_{highlight_model}.png"
    )
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"CAS plot saved → {fig_path}")


if __name__ == "__main__":
    main()
