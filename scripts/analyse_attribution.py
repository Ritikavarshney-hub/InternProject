"""
Phase 1 Attribution Analysis
Research Proposal §6.6, §6.7

Produces all interpretation results from occlusion sensitivity,
attention rollout, and Grad-CAM maps.

Components:
  A. Per-image attribution stats table (all models)
  B. Cross-method agreement — Spearman ρ (occlusion / rollout / gradcam for CLIP)
  C. Per-facet average heatmaps and attribution strength
  D. Per-country attribution quality (answers RQ5: cultural disparity)
  E. Correct vs incorrect prediction comparison (does signal predict accuracy?)
  F. Cross-model occlusion agreement — Spearman ρ between 4 models
  G. Summary report

All outputs → results/analysis/

Usage:
    python analyse_attribution.py
    python analyse_attribution.py --no_figs     # skip PNG output
    python analyse_attribution.py --models clip llava   # subset of models
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_DIR  = os.path.join(RESULTS_DIR,  "analysis")
OCCLUSION_DIR = os.path.join(RESULTS_DIR,  "occlusion","occlusion")
ROLLOUT_DIR   = os.path.join(RESULTS_DIR,  "attention_rollout")
GRADCAM_DIR   = os.path.join(RESULTS_DIR,  "gradcam")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

META_CSV = os.path.join(RESULTS_DIR, "sample_metadata.csv")
IDS_CSV  = os.path.join(RESULTS_DIR, "sample_ids.csv")
PRED_CSVS = {
    "clip":      os.path.join(RESULTS_DIR, "clip_predictions.csv"),
    "llava":     os.path.join(RESULTS_DIR, "llava_predictions.csv"),
    "qwen2vl":   os.path.join(RESULTS_DIR, "qwen2vl_predictions.csv"),
    "internvl2": os.path.join(RESULTS_DIR, "internvl2_predictions.csv"),
}
ALL_MODELS    = ["clip", "llava", "qwen2vl", "internvl2"]
CANONICAL     = 14   # resize all maps to 14×14 for cross-model/method comparison
FACETS        = ["food", "traditions", "rituals", "drink", "clothing"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _resize(arr: np.ndarray, size: int = CANONICAL) -> np.ndarray:
    if arr.shape == (size, size):
        return arr
    pil = Image.fromarray(arr.astype(np.float32))
    return np.array(pil.resize((size, size), Image.BILINEAR), dtype=np.float32)


def load_occlusion(u_id: str, model: str) -> np.ndarray | None:
    """Try 14×14 name first, then 7×7 suffix."""
    for fname in [
        f"{u_id}_{model}_mean.npy",
        f"{u_id}_{model}_mean_7x7.npy",
    ]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def load_rollout(u_id: str, model: str) -> np.ndarray | None:
    p = os.path.join(ROLLOUT_DIR, f"{u_id}_{model}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def load_gradcam(u_id: str, model: str) -> np.ndarray | None:
    p = os.path.join(GRADCAM_DIR, f"{u_id}_{model}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def map_entropy(arr: np.ndarray) -> float:
    pos = arr.clip(min=0)
    s = pos.sum()
    if s == 0:
        return 0.0
    p = pos.flatten() / s
    return float(-np.sum(p * np.log(p + 1e-12)))


def norm_map(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1], return zeros for flat maps."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ── Load metadata & predictions ───────────────────────────────────────────────

def load_all(models: list[str]):
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    meta       = pd.read_csv(META_CSV).set_index("u_id")

    preds = {}
    for m in models:
        if os.path.exists(PRED_CSVS[m]):
            preds[m] = pd.read_csv(PRED_CSVS[m]).set_index("u_id")

    return sample_ids, meta, preds


# ── Component A: attribution stats table ──────────────────────────────────────

def component_A(sample_ids, meta, preds, models):
    print("\n── Component A: attribution stats table ──────────────────────────")
    records = []

    for model in models:
        pred_df = preds.get(model)
        print(f"  {model}: ", end="", flush=True)
        n_found = 0

        for u_id in sample_ids:
            arr = load_occlusion(u_id, model)
            if arr is None:
                continue
            n_found += 1

            meta_row = meta.loc[u_id] if u_id in meta.index else None
            pred_row = pred_df.loc[u_id] if (pred_df is not None and u_id in pred_df.index) else None

            rec = {
                "u_id":          u_id,
                "model":         model,
                "facet":         meta_row["facet"]    if meta_row is not None else None,
                "true_country":  meta_row["country"]  if meta_row is not None else None,
                "pred_country":  pred_row["pred_country"] if pred_row is not None else None,
                "correct":       bool(pred_row["correct"]) if pred_row is not None else None,
                "max_score":     float(arr.max()),
                "mean_score":    float(arr.mean()),
                "std_score":     float(arr.std()),
                "frac_positive": float((arr > 0).mean()),
                "entropy":       map_entropy(arr),
                "top_i":         int(arr.argmax() // arr.shape[1]),
                "top_j":         int(arr.argmax() %  arr.shape[1]),
            }
            records.append(rec)

        print(f"{n_found} maps loaded")

    df = pd.DataFrame(records)
    out = os.path.join(ANALYSIS_DIR, "attribution_stats_all.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  ({len(df)} rows)")
    return df


# ── Component B: cross-method agreement ───────────────────────────────────────

def component_B(sample_ids, save_figs):
    print("\n── Component B: cross-method agreement (CLIP) ────────────────────")
    rows = []

    for u_id in sample_ids:
        occ  = load_occlusion(u_id, "clip")
        roll = load_rollout(u_id,   "clip")
        gcam = load_gradcam(u_id,   "clip")

        if occ is None:
            continue
        occ_r = _resize(occ).flatten()

        entry = {"u_id": u_id}

        if roll is not None:
            roll_r = _resize(roll).flatten()
            rho, _ = spearmanr(occ_r, roll_r)
            entry["occ_vs_rollout"] = rho

        if gcam is not None:
            gcam_r = _resize(gcam).flatten()
            rho, _ = spearmanr(occ_r, gcam_r)
            entry["occ_vs_gradcam"] = rho

        if roll is not None and gcam is not None:
            roll_r = _resize(roll).flatten()
            gcam_r = _resize(gcam).flatten()
            rho, _ = spearmanr(roll_r, gcam_r)
            entry["rollout_vs_gradcam"] = rho

        if len(entry) > 1:
            rows.append(entry)

    if not rows:
        print("  No cross-method data yet (attention rollout / gradcam not run).")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(ANALYSIS_DIR, "cross_method_agreement.csv")
    df.to_csv(out, index=False)

    summary = df.drop(columns="u_id").agg(["mean", "std", "count"])
    print(df.drop(columns="u_id").mean().to_string())

    if save_figs and len(df) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        cols = [c for c in df.columns if c != "u_id"]
        ax.boxplot([df[c].dropna() for c in cols], tick_labels=cols, vert=True)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Cross-method spatial agreement (CLIP, per image)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        fig_path = os.path.join(ANALYSIS_DIR, "cross_method_agreement.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {fig_path}")

    print(f"  Saved → {out}  ({len(df)} images)")
    return df


# ── Component C: per-facet analysis ───────────────────────────────────────────

def component_C(stats_df, sample_ids, meta, save_figs):
    print("\n── Component C: per-facet heatmaps & attribution strength ────────")

    # Summary table: mean max_score and frac_positive by facet × model
    summary = (
        stats_df.groupby(["facet", "model"])[["max_score", "frac_positive", "entropy"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    out = os.path.join(ANALYSIS_DIR, "facet_attribution_summary.csv")
    summary.to_csv(out, index=False)
    print(f"  Saved → {out}")

    if not save_figs:
        return

    # Average heatmaps per facet for CLIP (has most complete coverage)
    clip_meta = meta.reset_index()
    fig, axes = plt.subplots(1, len(FACETS), figsize=(3 * len(FACETS), 3))
    fig.suptitle("Mean CLIP occlusion map by facet (14×14)", fontsize=11)

    for ax, facet in zip(axes, FACETS):
        uids_facet = clip_meta[clip_meta["facet"] == facet]["u_id"].tolist()
        maps = []
        for u_id in uids_facet:
            arr = load_occlusion(u_id, "clip")
            if arr is not None:
                maps.append(_resize(arr))

        if maps:
            avg = np.stack(maps).mean(axis=0)
            mn, mx = avg.min(), avg.max()
            if mx > mn:
                norm = TwoSlopeNorm(vmin=mn, vcenter=0.0, vmax=mx) if mn < 0 < mx \
                       else Normalize(vmin=mn, vmax=mx)
            else:
                norm = Normalize(vmin=0, vmax=1)
            im = ax.imshow(avg, cmap="RdBu_r", norm=norm)
            ax.set_title(f"{facet}\n(n={len(maps)})", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(facet, fontsize=9)

        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, "facet_average_heatmaps.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")

    # Bar chart: mean max_score by facet × model
    pivot = stats_df.groupby(["facet", "model"])["max_score"].mean().unstack("model")
    ax = pivot.plot(kind="bar", figsize=(9, 4), colormap="tab10", edgecolor="white")
    ax.set_title("Mean occlusion max_score by facet and model")
    ax.set_xlabel("")
    ax.set_ylabel("Mean max_score")
    ax.legend(title="Model", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, "facet_max_score_by_model.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")


# ── Component D: per-country analysis (RQ5) ───────────────────────────────────

def component_D(stats_df, save_figs):
    print("\n── Component D: per-country attribution quality (RQ5) ────────────")

    summary = (
        stats_df.groupby(["true_country", "model"])[["max_score", "frac_positive", "entropy"]]
        .mean()
        .round(4)
        .reset_index()
    )
    out = os.path.join(ANALYSIS_DIR, "country_attribution_summary.csv")
    summary.to_csv(out, index=False)
    print(f"  Saved → {out}")
    print(stats_df.groupby("true_country")["max_score"].mean().sort_values().to_string())

    if not save_figs:
        return

    # Horizontal bar chart — CLIP only, countries ranked by mean max_score
    clip_country = (
        stats_df[stats_df["model"] == "clip"]
        .groupby("true_country")["max_score"].mean()
        .sort_values()
    )
    if len(clip_country) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        clip_country.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_xlabel("Mean max_score (CLIP occlusion)")
        ax.set_title("Attribution strength by country\n(higher = model attends more strongly)")
        ax.axvline(clip_country.mean(), color="red", linestyle="--",
                   linewidth=1, label=f"mean = {clip_country.mean():.3f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig_path = os.path.join(ANALYSIS_DIR, "country_attribution_strength.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {fig_path}")


# ── Component E: correct vs incorrect ─────────────────────────────────────────

def component_E(stats_df, save_figs):
    print("\n── Component E: correct vs incorrect prediction ──────────────────")

    df = stats_df.dropna(subset=["correct"])

    summary_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        cor = sub[sub["correct"] == True]["max_score"].dropna()
        wrg = sub[sub["correct"] == False]["max_score"].dropna()

        if len(cor) < 5 or len(wrg) < 5:
            continue

        t_stat, p_val = ttest_ind(cor, wrg)
        summary_rows.append({
            "model":            model,
            "n_correct":        len(cor),
            "n_wrong":          len(wrg),
            "mean_correct":     cor.mean(),
            "mean_wrong":       wrg.mean(),
            "delta":            cor.mean() - wrg.mean(),
            "t_stat":           t_stat,
            "p_value":          p_val,
        })
        print(f"  {model}: correct={cor.mean():.4f}  wrong={wrg.mean():.4f}  "
              f"Δ={cor.mean()-wrg.mean():+.4f}  p={p_val:.4f}")

    if not summary_rows:
        print("  Not enough data for comparison.")
        return

    summary_df = pd.DataFrame(summary_rows).round(4)
    out = os.path.join(ANALYSIS_DIR, "correct_vs_incorrect.csv")
    summary_df.to_csv(out, index=False)
    print(f"  Saved → {out}")

    if not save_figs:
        return

    fig, axes = plt.subplots(1, len(summary_rows), figsize=(4 * len(summary_rows), 4), sharey=False)
    if len(summary_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, summary_rows):
        model = row["model"]
        sub   = df[df["model"] == model]
        cor   = sub[sub["correct"] == True]["max_score"].dropna()
        wrg   = sub[sub["correct"] == False]["max_score"].dropna()
        ax.boxplot([cor, wrg], tick_labels=["Correct", "Wrong"], vert=True)
        ax.set_title(f"{model}\nΔ={row['delta']:+.3f}  p={row['p_value']:.3f}", fontsize=9)
        ax.set_ylabel("max_score" if ax == axes[0] else "")

    fig.suptitle("Occlusion max_score: correct vs wrong predictions", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, "correct_vs_incorrect.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")


# ── Component F: cross-model occlusion agreement (RQ4) ────────────────────────

def component_F(sample_ids, models, save_figs):
    print("\n── Component F: cross-model occlusion agreement (RQ4) ────────────")

    # Pairwise Spearman ρ between model occlusion maps for the same image
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i+1:]]
    pair_rhos = {f"{a}_vs_{b}": [] for a, b in pairs}

    for u_id in sample_ids:
        maps = {}
        for m in models:
            arr = load_occlusion(u_id, m)
            if arr is not None:
                maps[m] = _resize(arr).flatten()

        for a, b in pairs:
            if a in maps and b in maps:
                rho, _ = spearmanr(maps[a], maps[b])
                pair_rhos[f"{a}_vs_{b}"].append(rho)

    summary_rows = []
    for pair, rhos in pair_rhos.items():
        if rhos:
            summary_rows.append({
                "pair":   pair,
                "n":      len(rhos),
                "mean_rho": np.mean(rhos),
                "std_rho":  np.std(rhos),
                "median_rho": np.median(rhos),
            })
            print(f"  {pair}: mean ρ = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}  (n={len(rhos)})")

    if not summary_rows:
        print("  Not enough cross-model data for comparison.")
        return None

    df = pd.DataFrame(summary_rows).round(4)
    out = os.path.join(ANALYSIS_DIR, "cross_model_agreement.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")

    # Build symmetric correlation matrix for models that have data
    active_models = [m for m in models if any(m in p for p in pair_rhos if pair_rhos[p])]
    if save_figs and len(active_models) >= 2:
        n = len(active_models)
        mat = np.eye(n)
        idx = {m: i for i, m in enumerate(active_models)}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            if pair_rhos.get(key):
                rho = np.mean(pair_rhos[key])
                if a in idx and b in idx:
                    mat[idx[a], idx[b]] = rho
                    mat[idx[b], idx[a]] = rho

        fig, ax = plt.subplots(figsize=(max(4, n), max(3, n)))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n)); ax.set_xticklabels(active_models, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(active_models)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if abs(mat[i,j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Mean Spearman ρ")
        ax.set_title("Cross-model spatial agreement\n(occlusion sensitivity maps)")
        plt.tight_layout()
        fig_path = os.path.join(ANALYSIS_DIR, "cross_model_agreement_matrix.png")
        plt.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {fig_path}")

    return df


# ── Component G: summary report ───────────────────────────────────────────────

def component_G(stats_df, agree_df):
    print("\n── Component G: summary report ───────────────────────────────────")
    lines = ["=" * 60, "PHASE 1 ATTRIBUTION ANALYSIS SUMMARY", "=" * 60]

    # Coverage
    lines.append("\nOcclusion map coverage:")
    for m in stats_df["model"].unique():
        n = stats_df[stats_df["model"] == m].shape[0]
        lines.append(f"  {m}: {n} images")

    # Overall signal strength per model
    lines.append("\nMean attribution signal (max_score) per model:")
    for m, g in stats_df.groupby("model"):
        lines.append(f"  {m}: {g['max_score'].mean():.4f} ± {g['max_score'].std():.4f}")

    # Hardest facet per model
    lines.append("\nWeakest facet (lowest mean max_score) per model:")
    for m in stats_df["model"].unique():
        sub = stats_df[stats_df["model"] == m]
        weak = sub.groupby("facet")["max_score"].mean().idxmin()
        val  = sub.groupby("facet")["max_score"].mean().min()
        lines.append(f"  {m}: {weak} ({val:.4f})")

    # Cross-model agreement
    if agree_df is not None and len(agree_df) > 0:
        lines.append("\nCross-model occlusion agreement (mean Spearman ρ):")
        for _, row in agree_df.iterrows():
            lines.append(f"  {row['pair']}: ρ = {row['mean_rho']:.3f}")

    lines.append("\nOutputs saved to: " + ANALYSIS_DIR)
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    out = os.path.join(ANALYSIS_DIR, "summary_report.txt")
    with open(out, "w") as f:
        f.write(report)
    print(f"\n  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Attribution Analysis")
    parser.add_argument("--no_figs",  action="store_true", help="Skip PNG output.")
    parser.add_argument("--models",   nargs="+", default=ALL_MODELS,
                        help="Models to include (default: all 4).")
    args = parser.parse_args()

    save_figs = not args.no_figs
    models    = [m for m in args.models if m in ALL_MODELS]

    print(f"Models: {models}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Analysis output: {ANALYSIS_DIR}")

    # Load metadata and predictions
    sample_ids, meta, preds = load_all(models)
    print(f"Sample IDs: {len(sample_ids)}")

    # Run components
    stats_df = component_A(sample_ids, meta, preds, models)
    agree_B  = component_B(sample_ids, save_figs)
    component_C(stats_df, sample_ids, meta, save_figs)
    component_D(stats_df, save_figs)
    component_E(stats_df, save_figs)
    agree_F  = component_F(sample_ids, models, save_figs)
    component_G(stats_df, agree_F)

    print(f"\nAll outputs in: {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
