"""
Milestone 3b — Pilot: Run Occlusion Sensitivity on 20 Images with CLIP

Proposal §6.3 (Pilot): Run on a small subset first, manually inspect heatmaps.
Sanity check: for a flag-dominant image, the flag patches should have the
highest occlusion scores. If heatmaps look random, fix confidence extraction
before processing all 500 images.

Saves:
    results/occlusion/{u_id}_clip_mean.npy  (one per pilot image)
    results/occlusion/pilot_summary.csv     (u_id, max_score, top_patch, true_country, pred_country, correct)

Usage:
    python run_occlusion_pilot.py
    python run_occlusion_pilot.py --n 20
    python run_occlusion_pilot.py --fill black
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from occlusion import (
    OCCLUSION_DIR,
    DATASET_PATH,
    SAMPLE_IDS_CSV,
    SAMPLE_META_CSV,
    PRED_CSVS,
    GRID_SIZE,
    load_clip_model,
    clip_confidence_fn,
    occlusion_sensitivity,
)

PILOT_SUMMARY = os.path.join(OCCLUSION_DIR, "pilot_summary.csv")


def run_pilot(n: int = 20, fill: str = "mean", grid_size: int = GRID_SIZE):
    os.makedirs(OCCLUSION_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pilot: {n} images | CLIP | fill={fill} | grid={grid_size}x{grid_size} | device={device}\n")

    # Load CLIP predictions from Milestone 2
    pred_csv = PRED_CSVS["clip"]
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(
            f"{pred_csv} not found.\n"
            f"Run predict_clip.py (Milestone 2) before this pilot."
        )
    preds = pd.read_csv(pred_csv).set_index("u_id")

    sample_ids   = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    sample_meta  = pd.read_csv(SAMPLE_META_CSV)
    countries    = sorted(sample_meta["country"].unique().tolist())

    pilot_ids = sample_ids[:n]
    id_set    = set(pilot_ids)

    ds     = load_from_disk(DATASET_PATH)["test"]
    subset = ds.filter(lambda x: x["u_id"] in id_set)

    # Build a lookup for iteration order
    subset_dict = {row["u_id"]: row for row in subset}

    model, preprocess, tokenizer = load_clip_model(device)

    records = []
    for idx, u_id in enumerate(pilot_ids):
        if u_id not in subset_dict:
            print(f"[warn] {u_id} not found in dataset — skipping")
            continue
        if u_id not in preds.index:
            print(f"[warn] {u_id} missing from predictions CSV — skipping")
            continue

        row            = subset_dict[u_id]
        target_country = preds.loc[u_id, "pred_country"]
        true_country   = row["country"]
        correct        = preds.loc[u_id, "correct"]
        image          = row["image"]

        out_path = os.path.join(OCCLUSION_DIR, f"{u_id}_clip_{fill}.npy")
        if False and os.path.exists(out_path):
    	    scores = np.load(out_path)
    	    print(f"[{idx+1}/{n}] {u_id} — loaded from cache")
        else:
            conf_fn = clip_confidence_fn(
                model, preprocess, tokenizer, countries, target_country, device
            )
            scores = occlusion_sensitivity(conf_fn, image, grid_size=grid_size, fill=fill)
            np.save(out_path, scores)

        top_flat   = scores.argmax()
        top_i, top_j = np.unravel_index(top_flat, scores.shape)
        max_score  = float(scores.max())
        mean_score = float(scores.mean())
        frac_pos   = float((scores > 0).mean())

        records.append({
            "u_id":           u_id,
            "true_country":   true_country,
            "pred_country":   target_country,
            "correct":        correct,
            "facet":          row.get("facet", ""),
            "max_score":      round(max_score, 4),
            "mean_score":     round(mean_score, 4),
            "frac_patches_positive": round(frac_pos, 3),
            "top_patch_i":    int(top_i),
            "top_patch_j":    int(top_j),
            "npy_path":       out_path,
        })

        print(
            f"[{idx+1}/{n}] {u_id} | true={true_country} | pred={target_country} | "
            f"correct={correct} | max_score={max_score:.4f} | "
            f"top_patch=({top_i},{top_j}) | frac_pos={frac_pos:.2%}"
        )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(PILOT_SUMMARY, index=False)

    # Sanity check report
    print("\n" + "=" * 60)
    print("PILOT SANITY CHECK REPORT")
    print("=" * 60)
    print(f"Images processed:       {len(records)}")
    print(f"Mean max_score:         {summary_df['max_score'].mean():.4f}")
    print(f"Mean frac_pos patches:  {summary_df['frac_patches_positive'].mean():.2%}")
    print(f"Correct predictions:    {summary_df['correct'].sum()} / {len(records)}")
    print()

    # Flag if heatmaps look suspicious (all scores near zero or negative)
    low_signal = summary_df[summary_df["max_score"] < 0.001]
    if len(low_signal) > 0:
        print(f"[WARN] {len(low_signal)} images have max_score < 0.001 — heatmaps may be flat.")
        print("       Check confidence extraction before running full 500-image pipeline.")
        print(low_signal[["u_id", "true_country", "pred_country", "max_score"]].to_string(index=False))
    else:
        print("[OK] All heatmaps show non-trivial attribution signal.")

    all_neg = summary_df[summary_df["mean_score"] < 0]
    if len(all_neg) > 0:
        print(f"\n[WARN] {len(all_neg)} images have negative mean score — fill value may be inflating confidence.")

    print(f"\nSummary saved to: {PILOT_SUMMARY}")
    print("Next step: run visualise_heatmap.py to inspect these 20 heatmaps visually.")
    print("If heatmaps look sensible, proceed with: python occlusion.py --model clip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3b — Occlusion Pilot (CLIP, 20 images)")
    parser.add_argument("--n",    type=int, default=20,    help="Number of pilot images (default: 20).")
    parser.add_argument("--fill", default="mean",          help="Patch fill: mean | black | noise (default: mean).")
    parser.add_argument("--grid", type=int, default=GRID_SIZE, help=f"Grid size (default: {GRID_SIZE}).")
    args = parser.parse_args()
    run_pilot(n=args.n, fill=args.fill, grid_size=args.grid)
