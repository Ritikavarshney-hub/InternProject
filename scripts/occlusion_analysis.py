"""
Milestone 3 — Occlusion Sensitivity Interpretation
Phase1_Execution_Plan.md §3 — Occlusion Sensitivity Pipeline

Interprets ALL occlusion sensitivity maps across all 4 models and produces
complete results tables and figures as required by the execution plan.

What this script does:
  1.  Coverage report        — how many maps exist per model
  2.  Signal statistics      — max_score, mean, std, frac_positive, entropy
  3.  Map quality check      — fraction of near-zero maps (signal < threshold)
  4.  Correct vs wrong       — does attribution signal predict accuracy?
  5.  Per-facet analysis     — which facet produces strongest signal?
  6.  Per-country analysis   — which culture has most focused attribution?
  7.  Spatial distribution   — where in the image do top patches cluster?
  8.  Average heatmaps       — mean map per model / per facet / per country
  9.  Cross-method agreement — Spearman ρ(occlusion, rollout) if rollout exists
  10. Full comparison table  — all models side-by-side (Paper Table 3)

Outputs → results/analysis/milestone3/
  Tables: stats_all.csv, facet_stats.csv, country_stats.csv, correct_vs_wrong.csv,
          spatial_distribution.csv, cross_method_agreement.csv, coverage_report.csv
  Figures: fig1_signal_distribution.png, fig2_facet_heatmaps.png,
           fig3_country_signal.png, fig4_correct_vs_wrong.png,
           fig5_average_maps_model.png, fig6_average_maps_facet.png,
           fig7_top_patch_spatial.png, fig8_cross_method.png,
           fig9_full_comparison.png

Usage:
    python scripts/milestone3_occlusion_interpret.py
    python scripts/milestone3_occlusion_interpret.py --models clip llava
    python scripts/milestone3_occlusion_interpret.py --no_figs
    python scripts/milestone3_occlusion_interpret.py --signal_threshold 0.01
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
OCCLUSION_DIR = os.path.join(RESULTS_DIR, "occlusion", "occlusion")
ROLLOUT_DIR   = os.path.join(RESULTS_DIR, "attention_rollout")
OUT_DIR       = os.path.join(RESULTS_DIR, "analysis", "milestone3")
os.makedirs(OUT_DIR, exist_ok=True)

META_CSV  = os.path.join(RESULTS_DIR, "sample_metadata.csv")
IDS_CSV   = os.path.join(RESULTS_DIR, "sample_ids.csv")
PRED_CSVS = {
    "clip":      os.path.join(RESULTS_DIR, "clip_predictions.csv"),
    "llava":     os.path.join(RESULTS_DIR, "llava_predictions.csv"),
    "qwen2vl":   os.path.join(RESULTS_DIR, "qwen2vl_predictions.csv"),
    "internvl2": os.path.join(RESULTS_DIR, "internvl2_predictions.csv"),
}
ALL_MODELS    = ["clip", "llava", "qwen2vl", "internvl2"]
MODEL_LABELS  = {"clip": "CLIP", "llava": "LLaVA",
                 "qwen2vl": "Qwen2-VL", "internvl2": "InternVL2"}
MODEL_COLOURS = {"clip": "#4575b4", "llava": "#d73027",
                 "qwen2vl": "#1a9850", "internvl2": "#f46d43"}
CANONICAL     = 14   # resize all maps to 14×14 for comparison


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_map(u_id: str, model: str) -> np.ndarray | None:
    """Try 14×14 (mean.npy) then 7×7 (mean_7x7.npy)."""
    for fname in [f"{u_id}_{model}_mean.npy",
                  f"{u_id}_{model}_mean_7x7.npy"]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def load_rollout(u_id: str, model: str) -> np.ndarray | None:
    p = os.path.join(ROLLOUT_DIR, f"{u_id}_{model}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def resize_to(arr: np.ndarray, size: int = CANONICAL) -> np.ndarray:
    if arr.shape == (size, size):
        return arr
    pil = Image.fromarray(arr)
    return np.array(pil.resize((size, size), Image.BILINEAR), dtype=np.float32)


def map_entropy(arr: np.ndarray) -> float:
    pos = arr.clip(min=0)
    s   = pos.sum()
    if s == 0:
        return 0.0
    p = pos.flatten() / s
    return float(-np.sum(p * np.log(p + 1e-12)))


# ── 1. Coverage report ─────────────────────────────────────────────────────────

def coverage_report(sample_ids, models):
    print("\n── 1. Coverage Report ────────────────────────────────────────────")
    rows = []
    for model in models:
        found = sum(1 for u in sample_ids if load_map(u, model) is not None)
        rollout_found = sum(1 for u in sample_ids
                            if load_rollout(u, model) is not None)
        rows.append({
            "Model":          MODEL_LABELS[model],
            "N occlusion":    found,
            "% covered":      round(found / len(sample_ids) * 100, 1),
            "N rollout":      rollout_found,
            "% rollout":      round(rollout_found / len(sample_ids) * 100, 1),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "coverage_report.csv"), index=False)
    return df


# ── 2. Signal statistics ────────────────────────────────────────────────────────

def compute_stats(sample_ids, meta, models, preds, threshold=0.01):
    print("\n── 2. Signal Statistics ──────────────────────────────────────────")
    all_records = []

    for model in models:
        pred_df = preds.get(model, pd.DataFrame()).set_index("u_id") \
                  if model in preds else pd.DataFrame()

        for u_id in sample_ids:
            arr = load_map(u_id, model)
            if arr is None:
                continue
            meta_row = meta.loc[u_id] if u_id in meta.index else None
            pred_row = pred_df.loc[u_id] if (len(pred_df) > 0 and u_id in pred_df.index) else None

            all_records.append({
                "u_id":           u_id,
                "model":          model,
                "facet":          meta_row["facet"]   if meta_row is not None else None,
                "true_country":   meta_row["country"] if meta_row is not None else None,
                "pred_country":   pred_row["pred_country"] if pred_row is not None else None,
                "correct":        bool(pred_row["correct"]) if pred_row is not None else None,
                "max_score":      float(arr.max()),
                "mean_score":     float(arr.mean()),
                "std_score":      float(arr.std()),
                "frac_positive":  float((arr > 0).mean()),
                "entropy":        map_entropy(arr),
                "is_flat":        bool(arr.max() < threshold),
                "top_i":          int(arr.argmax() // arr.shape[1]),
                "top_j":          int(arr.argmax() %  arr.shape[1]),
            })

    df = pd.DataFrame(all_records)

    # Summary per model
    summary_rows = []
    for model in models:
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "Model":           MODEL_LABELS[model],
            "N maps":          len(sub),
            "Mean max_score":  round(sub["max_score"].mean(), 4),
            "Std max_score":   round(sub["max_score"].std(), 4),
            "Mean entropy":    round(sub["entropy"].mean(), 4),
            "Frac positive":   round(sub["frac_positive"].mean(), 4),
            f"% flat (<{threshold})": round(sub["is_flat"].mean() * 100, 1),
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "stats_all.csv"), index=False)
    df_summary.to_csv(os.path.join(OUT_DIR, "stats_summary.csv"), index=False)
    return df


# ── 3. Correct vs wrong ────────────────────────────────────────────────────────

def correct_vs_wrong(df_stats, models):
    print("\n── 3. Correct vs Wrong Predictions ──────────────────────────────")
    rows = []
    for model in models:
        sub = df_stats[(df_stats["model"] == model) & df_stats["correct"].notna()]
        if len(sub) == 0:
            continue
        cor = sub[sub["correct"] == True]["max_score"].dropna()
        wrg = sub[sub["correct"] == False]["max_score"].dropna()
        if len(cor) < 5 or len(wrg) < 5:
            continue
        t_stat, p_val = ttest_ind(cor, wrg)
        row = {
            "Model":        MODEL_LABELS[model],
            "n_correct":    len(cor),
            "n_wrong":      len(wrg),
            "mean_correct": round(cor.mean(), 4),
            "mean_wrong":   round(wrg.mean(), 4),
            "delta":        round(cor.mean() - wrg.mean(), 4),
            "t_stat":       round(t_stat, 3),
            "p_value":      round(p_val, 4),
            "significant":  "✅" if p_val < 0.05 else "—",
        }
        rows.append(row)
        print(f"  {MODEL_LABELS[model]:<12}: correct={cor.mean():.4f}  "
              f"wrong={wrg.mean():.4f}  Δ={cor.mean()-wrg.mean():+.4f}  "
              f"p={p_val:.4f} {row['significant']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "correct_vs_wrong.csv"), index=False)
    return df


# ── 4. Per-facet analysis ──────────────────────────────────────────────────────

def facet_analysis(df_stats, models):
    print("\n── 4. Per-Facet Signal Strength ──────────────────────────────────")
    facets = sorted(df_stats["facet"].dropna().unique())
    rows   = []
    for model in models:
        sub = df_stats[df_stats["model"] == model]
        for facet in facets:
            fsub = sub[sub["facet"] == facet]
            rows.append({
                "model":      model,
                "facet":      facet,
                "n":          len(fsub),
                "max_score":  round(fsub["max_score"].mean(), 4),
                "entropy":    round(fsub["entropy"].mean(), 4),
                "frac_pos":   round(fsub["frac_positive"].mean(), 4),
            })
    df = pd.DataFrame(rows)

    # Print pivot
    pivot = df.pivot_table(index="facet", columns="model",
                           values="max_score", aggfunc="mean").round(4)
    pivot.columns = [MODEL_LABELS.get(c, c) for c in pivot.columns]
    print(pivot.to_string())
    df.to_csv(os.path.join(OUT_DIR, "facet_stats.csv"), index=False)
    return df


# ── 5. Per-country analysis ────────────────────────────────────────────────────

def country_analysis(df_stats, models):
    print("\n── 5. Per-Country Signal Strength ───────────────────────────────")
    countries = sorted(df_stats["true_country"].dropna().unique())
    rows      = []
    for model in models:
        sub = df_stats[df_stats["model"] == model]
        for country in countries:
            csub = sub[sub["true_country"] == country]
            rows.append({
                "model":      model,
                "country":    country,
                "n":          len(csub),
                "max_score":  round(csub["max_score"].mean(), 4),
                "entropy":    round(csub["entropy"].mean(), 4),
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="country", columns="model",
                           values="max_score", aggfunc="mean").round(4)
    pivot.columns = [MODEL_LABELS.get(c, c) for c in pivot.columns]
    print(pivot.to_string())
    df.to_csv(os.path.join(OUT_DIR, "country_stats.csv"), index=False)
    return df


# ── 6. Cross-method agreement ──────────────────────────────────────────────────

def cross_method_agreement(sample_ids, models):
    print("\n── 6. Cross-Method Agreement (Occlusion vs Rollout) ─────────────")
    rows = []
    for model in models:
        rhos = []
        for u_id in sample_ids:
            occ  = load_map(u_id, model)
            roll = load_rollout(u_id, model)
            if occ is None or roll is None:
                continue
            occ_r  = resize_to(occ).flatten()
            roll_r = resize_to(roll).flatten()
            rho, _ = spearmanr(occ_r, roll_r)
            if not np.isnan(rho):
                rhos.append(float(rho))
        if rhos:
            rows.append({
                "Model":    MODEL_LABELS[model],
                "N pairs":  len(rhos),
                "Mean ρ":   round(np.mean(rhos), 4),
                "Std ρ":    round(np.std(rhos),  4),
                "Median ρ": round(np.median(rhos), 4),
                "% ρ>0.3":  round(np.mean(np.array(rhos) > 0.3) * 100, 1),
            })
            print(f"  {MODEL_LABELS[model]:<12}: n={len(rhos)}  "
                  f"mean ρ={np.mean(rhos):.4f} ± {np.std(rhos):.4f}")
        else:
            print(f"  {MODEL_LABELS[model]:<12}: no rollout maps found")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(df) > 0:
        df.to_csv(os.path.join(OUT_DIR, "cross_method_agreement.csv"), index=False)
    return df


# ── 7. Spatial distribution of top patches ────────────────────────────────────

def spatial_distribution(df_stats, models):
    print("\n── 7. Spatial Distribution of Top Patches ───────────────────────")
    rows = []
    for model in models:
        sub = df_stats[df_stats["model"] == model]
        if len(sub) == 0:
            continue
        mean_i = sub["top_i"].mean()
        mean_j = sub["top_j"].mean()
        # Centre of image = (CANONICAL/2, CANONICAL/2)
        centre_i = CANONICAL / 2
        centre_j = CANONICAL / 2
        dist_from_centre = np.sqrt((sub["top_i"] - centre_i)**2 +
                                   (sub["top_j"] - centre_j)**2).mean()
        rows.append({
            "Model":             MODEL_LABELS[model],
            "Mean top-patch row":  round(mean_i, 2),
            "Mean top-patch col":  round(mean_j, 2),
            "Dist from centre":  round(dist_from_centre, 2),
            "% patches at top half":   round((sub["top_i"] < CANONICAL//2).mean()*100, 1),
            "% patches at centre 50%": round(((sub["top_i"] >= CANONICAL//4) &
                                               (sub["top_i"] < 3*CANONICAL//4) &
                                               (sub["top_j"] >= CANONICAL//4) &
                                               (sub["top_j"] < 3*CANONICAL//4)).mean()*100, 1),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "spatial_distribution.csv"), index=False)
    return df


# ── Figures ────────────────────────────────────────────────────────────────────

def fig1_signal_distributions(df_stats, models):
    """Box plots of max_score per model — signal strength comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # max_score
    data   = [df_stats[df_stats["model"] == m]["max_score"].dropna().values
               for m in models]
    labels = [MODEL_LABELS[m] for m in models]
    colours = [MODEL_COLOURS[m] for m in models]
    bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True,
                         medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_ylabel("max_score (occlusion signal strength)")
    axes[0].set_title("Attribution Signal Strength per Model")
    axes[0].grid(axis="y", alpha=0.3)

    # entropy
    data2 = [df_stats[df_stats["model"] == m]["entropy"].dropna().values
              for m in models]
    bp2 = axes[1].boxplot(data2, tick_labels=labels, patch_artist=True,
                           medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp2["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[1].set_ylabel("Entropy (map diffuseness)")
    axes[1].set_title("Attribution Map Entropy per Model\n(lower = more focused)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Milestone 3 — Occlusion Sensitivity: Signal Statistics", fontsize=12)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_signal_distributions.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 1 → {p}")


def fig2_facet_heatmaps(sample_ids, meta, models):
    """Average occlusion map per facet for each model."""
    facets  = sorted(meta["facet"].dropna().unique())
    n_rows  = len(models)
    n_cols  = len(facets)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    for i, model in enumerate(models):
        for j, facet in enumerate(facets):
            uids_facet = meta[meta["facet"] == facet].index.tolist()
            maps = []
            for u_id in uids_facet:
                arr = load_map(u_id, model)
                if arr is not None:
                    maps.append(resize_to(arr))
            ax = axes[i, j]
            if maps:
                avg = np.stack(maps).mean(axis=0)
                mn, mx = avg.min(), avg.max()
                norm = TwoSlopeNorm(vmin=mn, vcenter=0.0, vmax=mx) \
                       if mn < 0 < mx else Normalize(vmin=mn, vmax=mx)
                ax.imshow(avg, cmap="RdBu_r", norm=norm, interpolation="nearest")
                ax.set_title(f"{facet}\n(n={len(maps)})", fontsize=7)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            if j == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=8)
            ax.axis("off")

    fig.suptitle("Average Occlusion Map: Model × Facet\n"
                 "(Red = attended region, Blue = unimportant)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_facet_heatmaps.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Fig 2 → {p}")


def fig3_country_signal(df_stats, models):
    """Bar chart: mean max_score by country per model."""
    countries = sorted(df_stats["true_country"].dropna().unique())
    n_models  = len(models)
    x = np.arange(len(countries))
    w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 4))
    for i, model in enumerate(models):
        sub = df_stats[df_stats["model"] == model]
        vals = [sub[sub["true_country"] == c]["max_score"].mean()
                for c in countries]
        ax.bar(x + i*w - (n_models-1)*w/2, vals, width=w,
               label=MODEL_LABELS[model],
               color=MODEL_COLOURS.get(model, "gray"),
               edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean max_score")
    ax.set_title("Attribution Signal Strength by Country\n"
                 "(higher = model attends more strongly when predicting this culture)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_country_signal.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 3 → {p}")


def fig4_correct_vs_wrong(df_stats, models):
    """Side-by-side box plots: max_score for correct vs wrong predictions."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    if n == 1: axes = [axes]

    for ax, model in zip(axes, models):
        sub = df_stats[(df_stats["model"] == model) & df_stats["correct"].notna()]
        cor = sub[sub["correct"] == True]["max_score"].dropna()
        wrg = sub[sub["correct"] == False]["max_score"].dropna()
        bp  = ax.boxplot([cor, wrg], tick_labels=["Correct", "Wrong"],
                          patch_artist=True,
                          medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#1a9850"); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#d73027"); bp["boxes"][1].set_alpha(0.7)
        ax.set_title(f"{MODEL_LABELS[model]}\n"
                     f"Δ={cor.mean()-wrg.mean():+.3f}", fontsize=9)
        ax.set_ylabel("max_score" if model == models[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Attribution Signal: Correct vs Wrong Predictions\n"
                 "(green = correct, red = wrong)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_correct_vs_wrong.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 4 → {p}")


def fig5_average_maps_model(sample_ids, meta, models):
    """Overall average heatmap per model (all images)."""
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.5))
    if len(models) == 1: axes = [axes]

    for ax, model in zip(axes, models):
        maps = []
        for u_id in sample_ids:
            arr = load_map(u_id, model)
            if arr is not None:
                maps.append(resize_to(arr))
        if maps:
            avg  = np.stack(maps).mean(axis=0)
            mn, mx = avg.min(), avg.max()
            norm = TwoSlopeNorm(vmin=mn, vcenter=0.0, vmax=mx) \
                   if mn < 0 < mx else Normalize(vmin=mn, vmax=mx)
            im = ax.imshow(avg, cmap="RdBu_r", norm=norm, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{MODEL_LABELS[model]}\n(n={len(maps)})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no maps", ha="center", va="center",
                    transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle("Mean Occlusion Sensitivity Map per Model\n"
                 "(averaged over all 266 images)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig5_average_maps_model.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 5 → {p}")


def fig6_average_maps_facet(sample_ids, meta, primary_model="clip"):
    """Average map per facet for the primary model (CLIP)."""
    facets = sorted(meta["facet"].dropna().unique())
    fig, axes = plt.subplots(1, len(facets), figsize=(3 * len(facets), 3.5))
    if len(facets) == 1: axes = [axes]

    for ax, facet in zip(axes, facets):
        uids = meta[meta["facet"] == facet].index.tolist()
        maps = [resize_to(load_map(u, primary_model))
                for u in uids if load_map(u, primary_model) is not None]
        if maps:
            avg = np.stack(maps).mean(axis=0)
            mn, mx = avg.min(), avg.max()
            norm = TwoSlopeNorm(vmin=mn, vcenter=0.0, vmax=mx) \
                   if mn < 0 < mx else Normalize(vmin=mn, vmax=mx)
            im = ax.imshow(avg, cmap="RdBu_r", norm=norm, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{facet}\n(n={len(maps)})", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Mean Occlusion Map per Facet — {MODEL_LABELS.get(primary_model, primary_model)}\n"
                 "(averaged over all images per facet)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig6_average_maps_facet.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 6 → {p}")


def fig7_top_patch_spatial(df_stats, models):
    """2D density map of where top patches cluster spatially."""
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.5))
    if len(models) == 1: axes = [axes]

    for ax, model in zip(axes, models):
        sub = df_stats[df_stats["model"] == model]
        grid = np.zeros((CANONICAL, CANONICAL))
        for _, row in sub.iterrows():
            if pd.notna(row["top_i"]) and pd.notna(row["top_j"]):
                grid[int(row["top_i"]), int(row["top_j"])] += 1

        im = ax.imshow(grid, cmap="hot", interpolation="gaussian")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=10)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")

    fig.suptitle("Spatial Distribution of Top Attribution Patch\n"
                 "(where the most-attended patch typically appears in the image)",
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig7_top_patch_spatial.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 7 → {p}")


def fig8_cross_method(sample_ids, models):
    """Histogram of per-image Spearman ρ(occlusion, rollout)."""
    has_data = False
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.5),
                              sharey=True)
    if len(models) == 1: axes = [axes]

    for ax, model in zip(axes, models):
        rhos = []
        for u_id in sample_ids:
            occ  = load_map(u_id, model)
            roll = load_rollout(u_id, model)
            if occ is None or roll is None:
                continue
            rho, _ = spearmanr(resize_to(occ).flatten(), resize_to(roll).flatten())
            if not np.isnan(rho):
                rhos.append(float(rho))
        if rhos:
            has_data = True
            ax.hist(rhos, bins=25, color=MODEL_COLOURS.get(model, "gray"),
                    alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(rhos), color="black", lw=2,
                       label=f"mean={np.mean(rhos):.3f}")
            ax.axvline(0, color="red", lw=1, ls="--")
            ax.set_title(f"{MODEL_LABELS[model]}\nn={len(rhos)}", fontsize=10)
            ax.set_xlabel("Spearman ρ")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No rollout\nmaps found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

    if has_data:
        fig.suptitle("Cross-Method Agreement: Occlusion vs Attention Rollout\n"
                     "(Spearman ρ per image — higher = methods agree on spatial attribution)",
                     fontsize=10)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "fig8_cross_method.png")
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
        print(f"  Fig 8 → {p}")
    else:
        plt.close()
        print("  Fig 8 skipped — no rollout maps available for comparison")


def fig9_full_comparison(df_stats, models):
    """Master one-page figure: 3 panels for the paper."""
    models_present = [m for m in models if df_stats[df_stats["model"]==m].shape[0] > 0]
    if not models_present:
        return

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # Panel 1: Max-score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    data   = [df_stats[df_stats["model"]==m]["max_score"].values for m in models_present]
    labels = [MODEL_LABELS[m] for m in models_present]
    cols   = [MODEL_COLOURS.get(m, "gray") for m in models_present]
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax1.set_title("Attribution Signal (max_score)", fontsize=10)
    ax1.set_ylabel("max_score"); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Facet comparison (CLIP)
    ax2 = fig.add_subplot(gs[0, 1])
    facets  = sorted(df_stats["facet"].dropna().unique())
    x       = np.arange(len(facets))
    w       = 0.8 / len(models_present)
    for i, model in enumerate(models_present):
        sub  = df_stats[df_stats["model"] == model]
        vals = [sub[sub["facet"] == f]["max_score"].mean() for f in facets]
        ax2.bar(x + i*w - (len(models_present)-1)*w/2, vals, width=w,
                label=MODEL_LABELS[model], color=MODEL_COLOURS.get(model,"gray"),
                edgecolor="white", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(facets, rotation=25, ha="right", fontsize=8)
    ax2.set_title("Signal by Facet", fontsize=10)
    ax2.set_ylabel("Mean max_score"); ax2.legend(fontsize=7); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Correct vs wrong (all models)
    ax3 = fig.add_subplot(gs[0, 2])
    model_names, cor_means, wrg_means = [], [], []
    for model in models_present:
        sub = df_stats[(df_stats["model"]==model) & df_stats["correct"].notna()]
        cor = sub[sub["correct"]==True]["max_score"].mean()
        wrg = sub[sub["correct"]==False]["max_score"].mean()
        if pd.notna(cor) and pd.notna(wrg):
            model_names.append(MODEL_LABELS[model])
            cor_means.append(cor); wrg_means.append(wrg)
    x2 = np.arange(len(model_names))
    ax3.bar(x2 - 0.2, cor_means, 0.38, label="Correct", color="#1a9850", alpha=0.8)
    ax3.bar(x2 + 0.2, wrg_means, 0.38, label="Wrong",   color="#d73027", alpha=0.8)
    ax3.set_xticks(x2); ax3.set_xticklabels(model_names, fontsize=9)
    ax3.set_title("Signal: Correct vs Wrong", fontsize=10)
    ax3.set_ylabel("Mean max_score"); ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # Panel 4–6: Average maps for top 3 models
    for idx, model in enumerate(models_present[:3]):
        ax = fig.add_subplot(gs[1, idx])
        maps = [resize_to(load_map(u, model))
                for u in df_stats[df_stats["model"]==model]["u_id"].unique()
                if load_map(u, model) is not None]
        if maps:
            avg = np.stack(maps).mean(axis=0)
            mn, mx = avg.min(), avg.max()
            norm = TwoSlopeNorm(vmin=mn, vcenter=0, vmax=mx) \
                   if mn < 0 < mx else Normalize(vmin=mn, vmax=mx)
            im = ax.imshow(avg, cmap="RdBu_r", norm=norm, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Mean map — {MODEL_LABELS[model]}\n(n={len(maps)})", fontsize=9)
        ax.axis("off")

    fig.suptitle("Milestone 3 — Occlusion Sensitivity: Full Model Comparison",
                 fontsize=13, fontweight="bold")
    p = os.path.join(OUT_DIR, "fig9_full_comparison.png")
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Fig 9 → {p}")


# ── Master comparison table (console) ─────────────────────────────────────────

def print_master_table(df_stats, models):
    print("\n" + "="*72)
    print("MILESTONE 3 MASTER COMPARISON TABLE — Occlusion Sensitivity")
    print("="*72)
    labels = [MODEL_LABELS[m] for m in models]
    print(f"\n{'Metric':<32}", end="")
    for lab in labels:
        print(f"{lab:>12}", end="")
    print()
    print("-"*72)

    metrics = [
        ("N maps",             lambda m: df_stats[df_stats["model"]==m].shape[0]),
        ("Mean max_score",     lambda m: round(df_stats[df_stats["model"]==m]["max_score"].mean(), 4)),
        ("Std max_score",      lambda m: round(df_stats[df_stats["model"]==m]["max_score"].std(), 4)),
        ("Mean entropy",       lambda m: round(df_stats[df_stats["model"]==m]["entropy"].mean(), 4)),
        ("Mean frac_positive", lambda m: round(df_stats[df_stats["model"]==m]["frac_positive"].mean(), 4)),
        ("% flat maps (<0.01)",lambda m: round(df_stats[df_stats["model"]==m]["is_flat"].mean()*100, 1)),
    ]
    for label, fn in metrics:
        print(f"  {label:<30}", end="")
        for model in models:
            try:
                val = fn(model)
                print(f"{str(val):>12}", end="")
            except Exception:
                print(f"{'—':>12}", end="")
        print()

    # Per-facet
    facets = sorted(df_stats["facet"].dropna().unique())
    print(f"\n  {'Facet accuracy (avg max_score)':<30}", end="")
    for lab in labels:
        print(f"{lab:>12}", end="")
    print()
    for facet in facets:
        print(f"    {facet:<28}", end="")
        for model in models:
            sub = df_stats[(df_stats["model"]==model) & (df_stats["facet"]==facet)]
            print(f"{round(sub['max_score'].mean(), 4):>12}", end="")
        print()

    print("="*72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",           nargs="+", default=ALL_MODELS)
    parser.add_argument("--no_figs",          action="store_true")
    parser.add_argument("--signal_threshold", type=float, default=0.01,
                        help="Threshold below which a map is 'flat' (default: 0.01)")
    args = parser.parse_args()
    models = [m for m in args.models if m in ALL_MODELS]

    # Load metadata
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    meta       = pd.read_csv(META_CSV).set_index("u_id")
    preds      = {}
    for model in models:
        if os.path.exists(PRED_CSVS.get(model, "")):
            preds[model] = pd.read_csv(PRED_CSVS[model])

    print(f"Sample: {len(sample_ids)} images | Models: {models}")
    print(f"Occlusion dir: {OCCLUSION_DIR}")
    print(f"Output dir:    {OUT_DIR}")

    # Run analyses
    coverage_report(sample_ids, models)
    df_stats = compute_stats(sample_ids, meta, models, preds, args.signal_threshold)
    correct_vs_wrong(df_stats, models)
    facet_analysis(df_stats, models)
    country_analysis(df_stats, models)
    cross_method_agreement(sample_ids, models)
    spatial_distribution(df_stats, models)
    print_master_table(df_stats, models)

    if not args.no_figs:
        print(f"\nGenerating figures → {OUT_DIR}/")
        fig1_signal_distributions(df_stats, models)
        fig2_facet_heatmaps(sample_ids, meta, models)
        fig3_country_signal(df_stats, models)
        fig4_correct_vs_wrong(df_stats, models)
        fig5_average_maps_model(sample_ids, meta, models)
        fig6_average_maps_facet(sample_ids, meta,
                                primary_model="clip" if "clip" in models else models[0])
        fig7_top_patch_spatial(df_stats, models)
        fig8_cross_method(sample_ids, models)
        fig9_full_comparison(df_stats, models)

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
