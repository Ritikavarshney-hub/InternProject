"""
Milestone 2 Comparison Table Generator
Phase1_Execution_Plan.md §2 — Baseline Country Prediction

Generates all comparison tables and figures for the four models:
  CLIP ViT-L/14 | LLaVA-1.6 | Qwen2-VL-7B | InternVL2-8B

Tables produced:
  Table 1  — Overall accuracy per model
  Table 2  — Per-facet accuracy (model × facet)
  Table 3  — Per-country accuracy (model × country)
  Table 4  — Confidence statistics per model
  Table 5  — Top-5 most common misclassification pairs (per model)
  Table 6  — Model agreement matrix (% of images where both predict same country)
  Table 7  — Correct vs UNKNOWN predictions per model

Figures produced:
  Fig 1 — Overall accuracy bar chart
  Fig 2 — Per-facet accuracy heatmap (model × facet)
  Fig 3 — Per-country accuracy heatmap (model × country)
  Fig 4 — Confidence distribution violin plot
  Fig 5 — Confusion matrix (CLIP — most complete model)
  Fig 6 — Model agreement heatmap

All outputs → results/analysis/milestone2/

Usage:
    python scripts/milestone2_comparison.py
    python scripts/milestone2_comparison.py --no_figs   # tables only
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUT_DIR      = os.path.join(RESULTS_DIR, "analysis", "milestone2")
os.makedirs(OUT_DIR, exist_ok=True)

META_CSV = os.path.join(RESULTS_DIR, "sample_metadata.csv")
PRED_FILES = {
    "CLIP":      os.path.join(RESULTS_DIR, "clip_predictions.csv"),
    "LLaVA":     os.path.join(RESULTS_DIR, "llava_predictions.csv"),
    "Qwen2-VL":  os.path.join(RESULTS_DIR, "qwen2vl_predictions.csv"),
    "InternVL2": os.path.join(RESULTS_DIR, "internvl2_predictions.csv"),
}
MODEL_ORDER   = ["CLIP", "Qwen2-VL", "InternVL2", "LLaVA"]
MODEL_COLOURS = {"CLIP": "#4575b4", "LLaVA": "#d73027",
                 "Qwen2-VL": "#1a9850", "InternVL2": "#f46d43"}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_predictions() -> dict[str, pd.DataFrame]:
    dfs = {}
    for model, path in PRED_FILES.items():
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found — skipping {model}")
            continue
        df = pd.read_csv(path)
        # Normalise column names
        df.columns = df.columns.str.lower().str.strip()
        # Ensure 'correct' is bool
        if "correct" in df.columns:
            df["correct"] = df["correct"].astype(str).str.strip().str.lower() == "true"
        # Handle UNKNOWN predictions
        if "pred_country" in df.columns:
            df["is_unknown"] = df["pred_country"].str.upper().str.strip() == "UNKNOWN"
        dfs[model] = df
        print(f"  Loaded {model}: {len(df)} rows  acc={df['correct'].mean():.3f}")
    return dfs


# ── Table 1: Overall accuracy ──────────────────────────────────────────────────

def table1_overall(dfs: dict) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        if model not in dfs:
            continue
        df = dfs[model]
        n_total   = len(df)
        n_correct = df["correct"].sum()
        n_unknown = df.get("is_unknown", pd.Series([False]*n_total)).sum()
        mean_conf = df["confidence"].mean() if "confidence" in df.columns else None
        rows.append({
            "Model":          model,
            "Total images":   n_total,
            "Correct":        int(n_correct),
            "Wrong":          int(n_total - n_correct - n_unknown),
            "UNKNOWN":        int(n_unknown),
            "Accuracy (%)":   round(n_correct / n_total * 100, 1),
            "Mean confidence":round(mean_conf, 4) if mean_conf is not None else "—",
        })
    df_out = pd.DataFrame(rows)
    print("\n── Table 1: Overall Accuracy ─────────────────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table1_overall_accuracy.csv"), index=False)
    return df_out


# ── Table 2: Per-facet accuracy ────────────────────────────────────────────────

def table2_facet(dfs: dict) -> pd.DataFrame:
    all_facets = []
    for df in dfs.values():
        if "facet" in df.columns:
            all_facets.extend(df["facet"].unique().tolist())
    facets = sorted(set(all_facets))

    rows = {}
    for model in MODEL_ORDER:
        if model not in dfs:
            continue
        df = dfs[model]
        row = {}
        for facet in facets:
            sub = df[df["facet"] == facet]
            row[facet] = round(sub["correct"].mean() * 100, 1) if len(sub) > 0 else None
        row["Overall"] = round(df["correct"].mean() * 100, 1)
        rows[model] = row

    df_out = pd.DataFrame(rows).T
    df_out.index.name = "Model"
    df_out = df_out.reset_index()
    print("\n── Table 2: Per-Facet Accuracy (%) ──────────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table2_facet_accuracy.csv"), index=False)
    return df_out


# ── Table 3: Per-country accuracy ──────────────────────────────────────────────

def table3_country(dfs: dict) -> pd.DataFrame:
    all_countries = []
    for df in dfs.values():
        if "true_country" in df.columns:
            all_countries.extend(df["true_country"].unique().tolist())
    countries = sorted(set(all_countries))

    rows = {}
    for model in MODEL_ORDER:
        if model not in dfs:
            continue
        df = dfs[model]
        row = {}
        for country in countries:
            sub = df[df["true_country"] == country]
            row[country] = round(sub["correct"].mean() * 100, 1) if len(sub) > 0 else None
        rows[model] = row

    df_out = pd.DataFrame(rows).T
    df_out.index.name = "Model"
    df_out = df_out.reset_index()
    print("\n── Table 3: Per-Country Accuracy (%) ────────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table3_country_accuracy.csv"), index=False)
    return df_out


# ── Table 4: Confidence statistics ────────────────────────────────────────────

def table4_confidence(dfs: dict) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        if model not in dfs or "confidence" not in dfs[model].columns:
            continue
        df = dfs[model]
        cor = df[df["correct"] == True]["confidence"]
        wrg = df[df["correct"] == False]["confidence"]
        rows.append({
            "Model":          model,
            "Mean (correct)": round(cor.mean(), 4) if len(cor) > 0 else "—",
            "Mean (wrong)":   round(wrg.mean(), 4) if len(wrg) > 0 else "—",
            "Mean (all)":     round(df["confidence"].mean(), 4),
            "Std (all)":      round(df["confidence"].std(), 4),
            "Min":            round(df["confidence"].min(), 4),
            "Max":            round(df["confidence"].max(), 4),
            "% conf > 0.5":   round((df["confidence"] > 0.5).mean() * 100, 1),
        })
    df_out = pd.DataFrame(rows)
    print("\n── Table 4: Confidence Statistics ───────────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table4_confidence.csv"), index=False)
    return df_out


# ── Table 5: Top misclassification pairs ──────────────────────────────────────

def table5_misclassifications(dfs: dict, top_n: int = 5) -> pd.DataFrame:
    all_rows = []
    for model in MODEL_ORDER:
        if model not in dfs:
            continue
        df = dfs[model]
        wrong = df[(df["correct"] == False) & ~df.get("is_unknown", False)]
        if "pred_country" not in wrong.columns or "true_country" not in wrong.columns:
            continue
        pairs = wrong.groupby(["true_country", "pred_country"]).size() \
                     .reset_index(name="count") \
                     .sort_values("count", ascending=False) \
                     .head(top_n)
        for _, row in pairs.iterrows():
            all_rows.append({
                "Model":       model,
                "True":        row["true_country"],
                "Predicted as": row["pred_country"],
                "Count":       int(row["count"]),
            })

    df_out = pd.DataFrame(all_rows)
    print("\n── Table 5: Top Misclassification Pairs ─────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table5_misclassifications.csv"), index=False)
    return df_out


# ── Table 6: Model agreement matrix ───────────────────────────────────────────

def table6_agreement(dfs: dict) -> pd.DataFrame:
    models = [m for m in MODEL_ORDER if m in dfs]
    # Align on common u_ids
    common_ids = None
    for df in dfs.values():
        ids = set(df["u_id"])
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = sorted(common_ids)

    preds = {}
    for model in models:
        df = dfs[model].set_index("u_id")
        preds[model] = df.loc[common_ids, "pred_country"].values

    mat = np.zeros((len(models), len(models)))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            agree = np.mean(preds[m1] == preds[m2]) * 100
            mat[i, j] = round(agree, 1)

    df_out = pd.DataFrame(mat, index=models, columns=models)
    df_out.index.name = "Model"
    df_out = df_out.reset_index()
    print(f"\n── Table 6: Model Agreement (% same prediction, n={len(common_ids)}) ──")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table6_model_agreement.csv"), index=False)
    return df_out, models, mat


# ── Table 7: UNKNOWN / parse failures ─────────────────────────────────────────

def table7_unknown(dfs: dict) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        if model not in dfs:
            continue
        df = dfs[model]
        n_unknown = df.get("is_unknown", pd.Series([False]*len(df))).sum()
        if n_unknown > 0:
            by_facet  = df[df.get("is_unknown", False)]["facet"].value_counts().to_dict()
            by_country = df[df.get("is_unknown", False)]["true_country"].value_counts().to_dict()
        else:
            by_facet = by_country = {}
        rows.append({
            "Model":      model,
            "N unknown":  int(n_unknown),
            "% total":    round(n_unknown / len(df) * 100, 2),
            "Top facet":  max(by_facet, key=by_facet.get) if by_facet else "—",
            "Top country":max(by_country, key=by_country.get) if by_country else "—",
        })
    df_out = pd.DataFrame(rows)
    print("\n── Table 7: UNKNOWN / Parse Failures ────────────────────────────")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, "table7_unknown.csv"), index=False)
    return df_out


# ── Figures ────────────────────────────────────────────────────────────────────

def fig1_overall_accuracy(t1: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    models = t1["Model"].tolist()
    accs   = t1["Accuracy (%)"].tolist()
    colours = [MODEL_COLOURS.get(m, "steelblue") for m in models]
    bars = ax.bar(models, accs, color=colours, edgecolor="white", width=0.55)
    ax.axhline(9.1, color="gray", lw=1.2, ls="--", label="Chance (9.1%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.set_title("Milestone 2 — Overall Country Prediction Accuracy\n(266 images, 11 countries)",
                 fontsize=11)
    ax.set_ylim(0, max(accs) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_overall_accuracy.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 → {p}")


def fig2_facet_heatmap(dfs: dict):
    all_facets = sorted({f for df in dfs.values() if "facet" in df.columns
                         for f in df["facet"].unique()})
    models = [m for m in MODEL_ORDER if m in dfs]

    mat = np.zeros((len(models), len(all_facets)))
    for i, model in enumerate(models):
        df = dfs[model]
        for j, facet in enumerate(all_facets):
            sub = df[df["facet"] == facet]
            mat[i, j] = sub["correct"].mean() * 100 if len(sub) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=20, vmax=80, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_xticks(range(len(all_facets)))
    ax.set_xticklabels(all_facets, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(len(all_facets)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=10, color="black" if 30 < val < 70 else "white",
                        fontweight="bold")
    ax.set_title("Accuracy (%) by Model × Facet", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_facet_heatmap.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 → {p}")


def fig3_country_heatmap(dfs: dict):
    all_countries = sorted({c for df in dfs.values() if "true_country" in df.columns
                             for c in df["true_country"].unique()})
    models = [m for m in MODEL_ORDER if m in dfs]

    mat = np.zeros((len(models), len(all_countries)))
    for i, model in enumerate(models):
        df = dfs[model]
        for j, country in enumerate(all_countries):
            sub = df[df["true_country"] == country]
            mat[i, j] = sub["correct"].mean() * 100 if len(sub) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=10, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_xticks(range(len(all_countries)))
    ax.set_xticklabels(all_countries, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(len(all_countries)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if 30 < val < 80 else "white")
    ax.set_title("Accuracy (%) by Model × Country", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_country_heatmap.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 → {p}")


def fig4_confidence_violin(dfs: dict):
    records = []
    for model in MODEL_ORDER:
        if model not in dfs or "confidence" not in dfs[model].columns:
            continue
        df = dfs[model]
        for _, row in df.iterrows():
            records.append({
                "Model":   model,
                "confidence": row["confidence"],
                "Result":  "Correct" if row["correct"] else "Wrong",
            })
    if not records:
        return
    df_plot = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 4))
    parts = ax.violinplot(
        [df_plot[(df_plot["Model"] == m)]["confidence"].values
         for m in MODEL_ORDER if m in dfs],
        positions=range(len([m for m in MODEL_ORDER if m in dfs])),
        showmedians=True, showextrema=True,
    )
    for pc, m in zip(parts["bodies"], [m for m in MODEL_ORDER if m in dfs]):
        pc.set_facecolor(MODEL_COLOURS.get(m, "steelblue"))
        pc.set_alpha(0.7)
    ax.set_xticks(range(len([m for m in MODEL_ORDER if m in dfs])))
    ax.set_xticklabels([m for m in MODEL_ORDER if m in dfs], fontsize=10)
    ax.set_ylabel("Confidence score", fontsize=11)
    ax.set_title("Confidence Distribution per Model", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_confidence_violin.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 4 → {p}")


def fig5_confusion_matrix(dfs: dict, model: str = "CLIP"):
    if model not in dfs:
        return
    df = dfs[model]
    countries = sorted(df["true_country"].unique())
    y_true = df["true_country"].tolist()
    y_pred = df["pred_country"].apply(
        lambda x: x if x in countries else "OTHER"
    ).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=countries)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction predicted")
    ax.set_xticks(range(len(countries)))
    ax.set_yticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(countries, fontsize=9)
    ax.set_xlabel("Predicted country", fontsize=11)
    ax.set_ylabel("True country", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model} (row-normalised)", fontsize=11)
    for i in range(len(countries)):
        for j in range(len(countries)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > 0.5 else "black")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"fig5_confusion_{model.lower()}.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 5 → {p}")


def fig6_agreement_heatmap(models: list, mat: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% same prediction")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if mat[i, j] > 65 else "black")
    ax.set_title("Model Agreement Matrix\n(% of images where both predict same country)",
                 fontsize=10)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig6_model_agreement.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 6 → {p}")


def fig7_combined_summary(dfs: dict):
    """One-page summary: accuracy bars + facet lines + country bars."""
    models   = [m for m in MODEL_ORDER if m in dfs]
    facets   = sorted({f for df in dfs.values() if "facet" in df.columns
                        for f in df["facet"].unique()})
    countries = sorted({c for df in dfs.values() if "true_country" in df.columns
                         for c in df["true_country"].unique()})

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel 1: Overall accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [dfs[m]["correct"].mean()*100 for m in models]
    cols = [MODEL_COLOURS.get(m, "steelblue") for m in models]
    bars = ax1.bar(models, accs, color=cols, edgecolor="white", width=0.55)
    ax1.axhline(9.1, color="gray", lw=1, ls="--", label="Chance")
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Overall Accuracy", fontsize=11)
    ax1.set_ylim(0, max(accs) * 1.2)
    ax1.set_ylabel("%"); ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Facet accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(facets))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        df  = dfs[model]
        acc = [df[df["facet"] == f]["correct"].mean()*100 if len(df[df["facet"]==f])>0
               else 0 for f in facets]
        ax2.bar(x + i*w - (len(models)-1)*w/2, acc, width=w,
                label=model, color=MODEL_COLOURS.get(model, "gray"),
                edgecolor="white", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(facets, fontsize=9)
    ax2.set_title("Per-Facet Accuracy", fontsize=11)
    ax2.set_ylabel("%"); ax2.legend(fontsize=7); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Country accuracy (line plot)
    ax3 = fig.add_subplot(gs[1, :])
    for model in models:
        df  = dfs[model]
        acc = [df[df["true_country"] == c]["correct"].mean()*100
               if len(df[df["true_country"]==c])>0 else 0 for c in countries]
        ax3.plot(countries, acc, marker="o", lw=2, ms=6,
                 label=model, color=MODEL_COLOURS.get(model, "gray"))
    ax3.set_xticks(range(len(countries)))
    ax3.set_xticklabels(countries, rotation=30, ha="right", fontsize=9)
    ax3.set_title("Per-Country Accuracy", fontsize=11)
    ax3.set_ylabel("%"); ax3.legend(fontsize=9, ncol=2); ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 110)

    fig.suptitle("Milestone 2 — Baseline Country Prediction: All Models", fontsize=13,
                 fontweight="bold", y=1.01)
    p = os.path.join(OUT_DIR, "fig7_combined_summary.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Fig 7 → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_figs", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    print("Loading prediction CSVs...")
    dfs = load_predictions()
    if not dfs:
        print("No prediction files found. Run predict_*.py scripts first.")
        return

    print(f"\nGenerating tables → {OUT_DIR}/")
    t1 = table1_overall(dfs)
    t2 = table2_facet(dfs)
    t3 = table3_country(dfs)
    t4 = table4_confidence(dfs)
    t5 = table5_misclassifications(dfs)
    t6, models_list, mat = table6_agreement(dfs)
    t7 = table7_unknown(dfs)

    if not args.no_figs:
        print(f"\nGenerating figures → {OUT_DIR}/")
        fig1_overall_accuracy(t1)
        fig2_facet_heatmap(dfs)
        fig3_country_heatmap(dfs)
        fig4_confidence_violin(dfs)
        fig5_confusion_matrix(dfs, "CLIP")
        if len(models_list) >= 2:
            fig6_agreement_heatmap(models_list, mat)
        fig7_combined_summary(dfs)

    # ── Print the master comparison table ─────────────────────────────────────
    print("\n" + "="*70)
    print("MILESTONE 2 MASTER COMPARISON TABLE")
    print("="*70)
    models = [m for m in MODEL_ORDER if m in dfs]

    print(f"\n{'Metric':<30}", end="")
    for m in models:
        print(f"{m:>14}", end="")
    print()
    print("-"*70)

    # Overall accuracy
    print(f"{'Overall accuracy (%)':<30}", end="")
    for m in models:
        acc = dfs[m]["correct"].mean()*100
        print(f"{acc:>14.1f}", end="")
    print()

    # Per-facet
    facets = sorted({f for df in dfs.values() if "facet" in df.columns
                     for f in df["facet"].unique()})
    for facet in facets:
        print(f"  Acc — {facet:<24}", end="")
        for m in models:
            df  = dfs[m]
            sub = df[df["facet"] == facet]
            acc = sub["correct"].mean()*100 if len(sub) > 0 else float("nan")
            print(f"{acc:>14.1f}", end="")
        print()

    # Per-country
    countries = sorted({c for df in dfs.values() if "true_country" in df.columns
                         for c in df["true_country"].unique()})
    print(f"\n{'Country accuracy':<30}", end="")
    for m in models:
        print(f"{m:>14}", end="")
    print()
    for country in countries:
        print(f"  {country:<28}", end="")
        for m in models:
            df  = dfs[m]
            sub = df[df["true_country"] == country]
            acc = sub["correct"].mean()*100 if len(sub) > 0 else float("nan")
            print(f"{acc:>14.1f}", end="")
        print()

    print("\n" + "="*70)
    print(f"All outputs saved to: {OUT_DIR}/")
    print("  Tables: table1–table7 (CSV)")
    print("  Figures: fig1–fig7 (PNG)")


if __name__ == "__main__":
    main()
