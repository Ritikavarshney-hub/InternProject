import os
import numpy as np
import pandas as pd

OCC_DIR = "results/occlusion/occlusion"

META_CSV = "results/sample_metadata.csv"
CLIP_PRED = "results/clip_predictions.csv"
LLAVA_PRED = "results/llava_predictions.csv"

OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def heatmap_entropy(scores):
    x = np.abs(scores.flatten())

    if x.sum() == 0:
        return 0

    p = x / x.sum()

    return -(p * np.log(p + 1e-12)).sum()


def analyze_model(model_name):

    meta = pd.read_csv(META_CSV)

    preds = pd.read_csv(
        CLIP_PRED if model_name == "clip"
        else LLAVA_PRED
    )

    preds = preds.rename(
        columns={"pred_country": "prediction"}
    )

    rows = []

    files = [
        f for f in os.listdir(OCC_DIR)
        if f"_{model_name}_" in f
    ]

    for fname in files:

        u_id = fname.split(f"_{model_name}_")[0]

        scores = np.load(
            os.path.join(OCC_DIR, fname)
        )

        top_i, top_j = np.unravel_index(
            scores.argmax(),
            scores.shape
        )

        rows.append({
            "u_id": u_id,
            "max_score": scores.max(),
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "positive_fraction":
                (scores > 0).mean(),
            "entropy":
                heatmap_entropy(scores),
            "top_patch_row": top_i,
            "top_patch_col": top_j
        })

    stats = pd.DataFrame(rows)

    stats = stats.merge(
        meta[["u_id", "country", "facet"]],
        on="u_id",
        how="left"
    )

    stats = stats.merge(
        preds[["u_id", "prediction"]],
        on="u_id",
        how="left"
    )

    stats["correct"] = (
        stats["country"] ==
        stats["prediction"]
    )

    stats.to_csv(
        f"{OUT_DIR}/{model_name}_stats.csv",
        index=False
    )

    return stats

clip_stats = analyze_model("clip")
llava_stats = analyze_model("llava")

facet_rows = []

for model_name, df in [
    ("clip", clip_stats),
    ("llava", llava_stats)
]:

    g = (
        df.groupby("facet")
        .agg({
            "max_score":"mean",
            "entropy":"mean",
            "correct":"mean"
        })
        .reset_index()
    )

    g["model"] = model_name

    facet_rows.append(g)

facet_summary = pd.concat(facet_rows)

facet_summary.to_csv(
    "results/analysis/facet_summary.csv",
    index=False
)

country_rows = []

for model_name, df in [
    ("clip", clip_stats),
    ("llava", llava_stats)
]:

    g = (
        df.groupby("country")
        .agg({
            "max_score":"mean",
            "entropy":"mean",
            "correct":"mean"
        })
        .reset_index()
    )

    g["model"] = model_name

    country_rows.append(g)

country_summary = pd.concat(country_rows)

country_summary.to_csv(
    "results/analysis/country_summary.csv",
    index=False
)

comparison = []

for model_name, df in [
    ("clip", clip_stats),
    ("llava", llava_stats)
]:

    g = (
        df.groupby("correct")
        .agg({
            "max_score":"mean",
            "entropy":"mean"
        })
        .reset_index()
    )

    g["model"] = model_name

    comparison.append(g)

comparison = pd.concat(comparison)

comparison.to_csv(
    "results/analysis/correct_vs_wrong.csv",
    index=False
)


