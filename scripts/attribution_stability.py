"""
Milestone 3 — Attribution Quality Metrics (Proposal §6.6, Metrics 3 & 4)

Metric 3 — Attribution Stability (test-retest reliability):
  Run occlusion sensitivity with three different fill types (mean, black, noise).
  Compute pairwise Spearman ρ of patch score rankings across fill types.
  High ρ = attribution is stable and not sensitive to masking choice.
  Proposal: 'Report mean stability per model as a methodological reliability indicator'

Metric 4 — Cross-Method Internal Agreement:
  Compute pairwise Spearman ρ between patch rankings produced by:
    occlusion sensitivity, attention rollout, Grad-CAM (and optionally LIME).
  'Images where all three methods agree on the top-K patches are high-confidence examples.'
  'Images with low cross-method agreement are flagged as uncertain — report their fraction.'

Prerequisites:
  - occlusion.py must have been run with --fill mean, black, and noise
  - attention_rollout.py and gradcam.py must have been run

Outputs:
  results/stability_metric3.csv   — per-image Spearman ρ across fill variants
  results/agreement_metric4.csv   — per-image Spearman ρ across methods
  results/stability_summary.csv   — mean ρ per model (Table to report in paper)

Usage:
    python attribution_stability.py --model clip
    python attribution_stability.py --model clip --model llava  (multiple models)
    python attribution_stability.py --all_models
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from occlusion import (
    OCCLUSION_DIR, SAMPLE_IDS_CSV, SAMPLE_META_CSV, GRID_SIZE,
)

OCCLUSION_DIR = os.path.join("results", "occlusion", "occlusion")  # Ensure this matches occlusion.py output
ROLLOUT_DIR = os.path.join("results", "attention_rollout")
GRADCAM_DIR = os.path.join("results", "gradcam")
LIME_DIR    = os.path.join("results", "lime")
RESULTS_DIR = "results"

ALL_MODELS  = ["clip", "llava", "qwen2vl", "internvl2"]
FILL_TYPES  = ["mean", "black", "noise"]

# Proposal: 'images with low cross-method agreement are flagged as uncertain'
LOW_AGREEMENT_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Metric 3: Attribution Stability across fill variants
# Proposal §6.6: 'Run occlusion sensitivity twice per image using different
#                 replacement values (grey mean, black fill, Gaussian noise fill)'
# ---------------------------------------------------------------------------

def compute_stability(model_name: str, sample_ids: list) -> pd.DataFrame:
    """
    For each image: load occlusion maps for all three fill types,
    compute all pairwise Spearman ρ, report per-image results.

    Returns a DataFrame with columns:
        u_id, rho_mean_black, rho_mean_noise, rho_black_noise, mean_rho
    """
    records = []

    for u_id in sample_ids:
        maps = {}
        for fill in FILL_TYPES:
            if fill == "mean":
                path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model_name}_mean_7x7.npy")
            else:
                path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model_name}_{fill}.npy")
            if os.path.exists(path):
                maps[fill] = np.load(path).flatten()

        if len(maps) < 2:
            # Not enough fill variants available for this image
            continue

        row = {"u_id": u_id, "model": model_name}
        rhos = []

        lengths = {k: len(v) for k, v in maps.items()}

        if len(set(lengths.values())) > 1:
            print("MISMATCH", u_id, lengths)
            continue
        for f1, f2 in combinations(maps.keys(), 2):
            print(u_id,f1, maps[f1].shape,f2, maps[f2].shape)
            rho, pval = spearmanr(maps[f1], maps[f2])
            row[f"rho_{f1}_vs_{f2}"] = round(float(rho), 4)
            row[f"pval_{f1}_vs_{f2}"] = round(float(pval), 4)
            rhos.append(float(rho))

        row["mean_rho"] = round(np.mean(rhos), 4) if rhos else np.nan
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metric 4: Cross-Method Internal Agreement
# Proposal §6.6 Metric 4: 'compute Spearman rank correlation between the
#   patch rankings produced by occlusion sensitivity, attention rollout,
#   and Grad-CAM'
# ---------------------------------------------------------------------------

def load_method_maps(u_id: str, model_name: str) -> dict[str, np.ndarray]:
    """
    Load attribution maps for one image from all available methods.
    Returns dict: method_name -> flat np.ndarray of patch scores.
    """
    maps = {}

    # Occlusion (primary, use mean-fill as canonical)
    occ_path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model_name}_mean.npy")
    if os.path.exists(occ_path):
        maps["occlusion"] = np.load(occ_path).flatten()

    # Attention Rollout
    rollout_path = os.path.join(ROLLOUT_DIR, f"{u_id}_{model_name}.npy")
    if os.path.exists(rollout_path):
        maps["attention_rollout"] = np.load(rollout_path).flatten()

    # Grad-CAM
    gradcam_path = os.path.join(GRADCAM_DIR, f"{u_id}_{model_name}.npy")
    if os.path.exists(gradcam_path):
        maps["gradcam"] = np.load(gradcam_path).flatten()

    # LIME (optional)
    lime_path = os.path.join(LIME_DIR, f"{u_id}_{model_name}_7x7.npy")
    if os.path.exists(lime_path):
        maps["lime"] = np.load(lime_path).flatten()

    return maps


def compute_cross_method_agreement(model_name: str, sample_ids: list) -> pd.DataFrame:
    """
    For each image: compute pairwise Spearman ρ between all available method maps.

    Proposal: 'flag images with low cross-method agreement as uncertain'

    Returns DataFrame with one row per image, rho for each method pair.
    """
    records = []

    for u_id in sample_ids:
        maps = load_method_maps(u_id, model_name)

        if len(maps) < 1:
            continue

        row  = {"u_id": u_id, "model": model_name, "n_methods": len(maps)}
        rhos = []

        for m1, m2 in combinations(maps.keys(), 2):
            rho, pval = spearmanr(maps[m1], maps[m2])
            col = f"rho_{m1}_vs_{m2}"
            row[col] = round(float(rho), 4)
            rhos.append(float(rho))

        row["mean_rho"]   = round(np.mean(rhos), 4) if rhos else np.nan
        row["low_agreement"] = row["mean_rho"] < LOW_AGREEMENT_THRESHOLD
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary table (for paper reporting)
# Proposal: 'Report mean stability per model'
# ---------------------------------------------------------------------------

def build_summary_table(stability_dfs: list, agreement_dfs: list) -> pd.DataFrame:
    rows = []
    for stab_df in stability_dfs:
        if stab_df.empty:
            continue
        model = stab_df["model"].iloc[0]
        rows.append({
            "model": model,
            "metric": "Stability (Metric 3)",
            "n_images": len(stab_df),
            "mean_rho": round(stab_df["mean_rho"].mean(), 4),
            "std_rho":  round(stab_df["mean_rho"].std(), 4),
            "min_rho":  round(stab_df["mean_rho"].min(), 4),
        })

    for agree_df in agreement_dfs:
        if agree_df.empty:
            continue
        model = agree_df["model"].iloc[0]
        frac_low = (agree_df["low_agreement"].sum() / len(agree_df))
        rows.append({
            "model":     model,
            "metric":    "Cross-Method Agreement (Metric 4)",
            "n_images":  len(agree_df),
            "mean_rho":  round(agree_df["mean_rho"].mean(), 4),
            "std_rho":   round(agree_df["mean_rho"].std(), 4),
            "frac_low_agreement": round(frac_low, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(models: list[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample_ids  = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    sample_meta = pd.read_csv(SAMPLE_META_CSV)

    all_stability  = []
    all_agreement  = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # --- Metric 3: Stability ---
        print(f"\n[Metric 3] Attribution Stability — {model_name}")
        stab_df = compute_stability(model_name, sample_ids)
        if not stab_df.empty:
            print(f"  Images with ≥2 fill variants: {len(stab_df)}")
            print(f"  Mean Spearman ρ across fill types: {stab_df['mean_rho'].mean():.4f}")
            print(f"  (ρ ≥ 0.7 indicates stable attribution signal)")
            all_stability.append(stab_df)
        else:
            print(f"  No occlusion maps found for ≥2 fill types. Run:")
            for fill in FILL_TYPES:
                print(f"    python occlusion.py --model {model_name} --fill {fill}")

        # --- Metric 4: Cross-Method Agreement ---
        print(f"\n[Metric 4] Cross-Method Agreement — {model_name}")
        agree_df = compute_cross_method_agreement(model_name, sample_ids)
        if not agree_df.empty:
            n_low = agree_df["low_agreement"].sum()
            print(f"  Images with ≥2 methods: {len(agree_df)}")
            print(f"  Mean cross-method Spearman ρ: {agree_df['mean_rho'].mean():.4f}")
            print(f"  Low-agreement images (ρ < {LOW_AGREEMENT_THRESHOLD}): {n_low} ({n_low/len(agree_df):.1%})")
            if n_low > 0:
                print(f"  Low-agreement u_ids (uncertain examples):")
                low_ids = agree_df[agree_df["low_agreement"]]["u_id"].tolist()
                for uid in low_ids[:10]:
                    print(f"    {uid}")
            all_agreement.append(agree_df)
        else:
            print(f"  Not enough method maps found. Run attention_rollout.py and gradcam.py first.")

    # Save per-image results
    if all_stability:
        stab_out = os.path.join(RESULTS_DIR, "stability_metric3.csv")
        pd.concat(all_stability, ignore_index=True).to_csv(stab_out, index=False)
        print(f"\nStability results saved: {stab_out}")

    if all_agreement:
        agree_out = os.path.join(RESULTS_DIR, "agreement_metric4.csv")
        pd.concat(all_agreement, ignore_index=True).to_csv(agree_out, index=False)
        print(f"Agreement results saved:  {agree_out}")

    # Build + save summary table
    summary = build_summary_table(all_stability, all_agreement)
    if not summary.empty:
        summary_out = os.path.join(RESULTS_DIR, "stability_summary.csv")
        summary.to_csv(summary_out, index=False)
        print(f"\nSummary table (paper Table):\n")
        print(summary.to_string(index=False))
        print(f"\nSaved: {summary_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 3 — Attribution Stability (Metric 3) + Cross-Method Agreement (Metric 4)"
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=ALL_MODELS,
        help="Model(s) to evaluate. Use multiple --model flags for multiple models.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run for all four models.",
    )
    args = parser.parse_args()

    if args.all_models:
        models_to_run = ALL_MODELS
    elif args.models:
        models_to_run = args.models
    else:
        parser.print_help()
        print("\nExample: python attribution_stability.py --model clip")
        print("         python attribution_stability.py --all_models")
        exit(1)

    run(models_to_run)
