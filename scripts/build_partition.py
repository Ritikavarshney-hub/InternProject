"""
Phase 2 — Step 0: Build Shortcut / Nuanced Image Partition
Phase2_Execution_Plan.md: Pre-work

Splits the 266-image sample into two semantically meaningful subsets using
Phase 1 outputs (CAS scores + CLIP cue category labels). Every Phase 2 analysis
compares these two subsets to find mechanistic differences.

Shortcut images:
  - CAS shows masking top patches CHANGES the prediction (changed_k20 == True)
  - Primary visual cue is a stereotyped shortcut: A / E / H
    A = National Symbols (flags, emblems)
    E = Script / Text    (written language)
    H = Appearance       (skin tone, facial features — bias flag)

Nuanced images:
  - CAS shows masking top patches CHANGES the prediction (changed_k20 == True)
  - Primary visual cue is a genuine cultural feature: B / C / D / F
    B = Clothing / Dress
    C = Architecture
    D = Food / Objects
    F = Ritual / Festival

Requires (from Phase 1):
  results/analysis/cas_per_image_clip.csv   ← from compute_cas.py --model clip
  results/cue_categories.csv               ← from detect_cue_categories.py

Outputs:
  results/phase2/shortcut_ids.csv           ← shortcut image IDs + metadata
  results/phase2/nuanced_ids.csv            ← nuanced image IDs + metadata
  results/phase2/partition_summary.csv      ← statistics for paper
  results/phase2/dev_set_shortcut.csv       ← top-10 high-confidence shortcut (pilot)
  results/phase2/dev_set_nuanced.csv        ← top-10 high-confidence nuanced (pilot)

Usage:
    python scripts/phase2/build_partition.py
    python scripts/phase2/build_partition.py --k 20   # use CAS at K=20% (default)
    python scripts/phase2/build_partition.py --k 30   # stricter
    python scripts/phase2/build_partition.py --n 50   # target partition size
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "cultural_vlm", "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR,  "analysis")
PHASE2_DIR   = os.path.join(RESULTS_DIR,  "phase2")
os.makedirs(PHASE2_DIR, exist_ok=True)

META_CSV = os.path.join(RESULTS_DIR, "sample_metadata.csv")
CUE_CSV  = os.path.join(RESULTS_DIR, "cue_categories.csv")

# ── Category definitions ───────────────────────────────────────────────────────
SHORTCUT_CATS = {"A", "E", "H"}   # stereotyped / shortcut cues
NUANCED_CATS  = {"B", "C", "D", "F"}   # genuine cultural cues

CAT_NAMES = {
    "A": "National Symbols",
    "B": "Clothing / Dress",
    "C": "Architecture",
    "D": "Food / Objects",
    "E": "Script / Text",
    "F": "Ritual / Festival",
    "G": "Natural Landscape",
    "H": "Appearance",
}


def main():
    parser = argparse.ArgumentParser(description="Build shortcut/nuanced partition")
    parser.add_argument("--k", type=int, default=20,
                        help="CAS deletion percentage to use (default: 20)")
    parser.add_argument("--n", type=int, default=50,
                        help="Target size per partition (default: 50). "
                             "If fewer candidates exist, uses all.")
    parser.add_argument("--require_correct", action="store_true",
                        help="Only include images where model's original prediction was correct.")
    args = parser.parse_args()

    k        = args.k
    changed_col = f"changed_k{k}"
    print(f"Building partition | K={k}% | Target n={args.n} per subset")

    # ── Load Phase 1 outputs ───────────────────────────────────────────────────
    cas_path = os.path.join("results", "analysis", "cas_per_image_clip.csv")
    if not os.path.exists(cas_path):
        raise FileNotFoundError(
            f"Missing: {cas_path}\n"
            f"Run first: python scripts/compute_cas.py --model clip --k_values 10 20 30"
        )
    if not os.path.exists(CUE_CSV):
        raise FileNotFoundError(
            f"Missing: {CUE_CSV}\n"
            f"Run first: python scripts/detect_cue_categories.py --model_occlusion clip"
        )

    cas_df = pd.read_csv(cas_path)
    cue_df = pd.read_csv(CUE_CSV).set_index("u_id")
    meta   = pd.read_csv(META_CSV).set_index("u_id")

    print(f"\nCAS data loaded:  {len(cas_df)} rows")
    print(f"Cue labels loaded: {len(cue_df)} images")

    if changed_col not in cas_df.columns:
        available = [c for c in cas_df.columns if c.startswith("changed_")]
        raise ValueError(
            f"Column '{changed_col}' not in CAS file.\n"
            f"Available: {available}\n"
            f"Re-run compute_cas.py with --k_values including {k}"
        )

    # ── Merge CAS with cue labels ──────────────────────────────────────────────
    merged = cas_df.copy()
    merged = merged.set_index("u_id")
    merged["category_primary"]  = cue_df["label_primary"]
    merged["category_name"]     = cue_df["label_primary_key"]
    merged["category_score"]    = cue_df["score_primary"]
    merged["label_confident"]   = cue_df["label_confident"]
    merged["facet"]             = meta["facet"] if "facet" in meta.columns else None
    merged["true_country"]      = meta["country"] if "country" in meta.columns else None
    merged = merged.reset_index()

    # ── Apply partition criteria ───────────────────────────────────────────────

    # Base filter: masking top-K% patches must change prediction
    changed_mask = merged[changed_col] == True

    if args.require_correct:
        changed_mask = changed_mask & (merged["correct_orig"] == True)

    # Shortcut candidates: changed prediction AND shortcut category
    shortcut_mask = changed_mask & merged["category_primary"].isin(SHORTCUT_CATS)
    nuanced_mask  = changed_mask & merged["category_primary"].isin(NUANCED_CATS)

    shortcut_pool = merged[shortcut_mask].copy()
    nuanced_pool  = merged[nuanced_mask].copy()

    print(f"\nCandidates after filtering (changed_k{k}=True):")
    print(f"  Shortcut pool (A/E/H): {len(shortcut_pool)}")
    print(f"  Nuanced pool  (B/C/D/F): {len(nuanced_pool)}")

    # ── Select top-N by category confidence (most unambiguous images first) ────
    # Sort by category_score DESC — higher score = more confidently assigned category
    shortcut_pool = shortcut_pool.sort_values("category_score", ascending=False)
    nuanced_pool  = nuanced_pool.sort_values("category_score", ascending=False)

    n_short = min(args.n, len(shortcut_pool))
    n_nuan  = min(args.n, len(nuanced_pool))

    shortcut_final = shortcut_pool.head(n_short).reset_index(drop=True)
    nuanced_final  = nuanced_pool.head(n_nuan).reset_index(drop=True)

    # ── Save full partitions ───────────────────────────────────────────────────
    cols_to_save = [
        "u_id", "true_country", "facet",
        "category_primary", "category_name", "category_score", "label_confident",
        "pred_country_orig", "correct_orig",
        f"changed_k{k}", "category_score",
    ]
    cols_to_save = [c for c in cols_to_save if c in shortcut_final.columns]

    shortcut_path = os.path.join(PHASE2_DIR, "shortcut_ids.csv")
    nuanced_path  = os.path.join(PHASE2_DIR, "nuanced_ids.csv")

    shortcut_final[cols_to_save].to_csv(shortcut_path, index=False)
    nuanced_final[cols_to_save].to_csv(nuanced_path,   index=False)

    print(f"\nPartition saved:")
    print(f"  Shortcut: {len(shortcut_final)} images → {shortcut_path}")
    print(f"  Nuanced:  {len(nuanced_final)} images → {nuanced_path}")

    # ── Development sets (top-10 per subset for piloting Phase 2 scripts) ─────
    dev_short = shortcut_final.head(10)[cols_to_save]
    dev_nuan  = nuanced_final.head(10)[cols_to_save]

    dev_short.to_csv(os.path.join(PHASE2_DIR, "dev_set_shortcut.csv"), index=False)
    dev_nuan.to_csv( os.path.join(PHASE2_DIR, "dev_set_nuanced.csv"),  index=False)
    print(f"\nDev sets saved (10 each for piloting Phase 2 scripts):")
    print(f"  {PHASE2_DIR}/dev_set_shortcut.csv")
    print(f"  {PHASE2_DIR}/dev_set_nuanced.csv")

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n── Shortcut partition breakdown ──────────────────────────────────")
    print(shortcut_final["category_primary"].map(
        lambda x: f"{x}: {CAT_NAMES.get(x,'')}"
    ).value_counts().to_string())

    print(f"\n── Nuanced partition breakdown ───────────────────────────────────")
    print(nuanced_final["category_primary"].map(
        lambda x: f"{x}: {CAT_NAMES.get(x,'')}"
    ).value_counts().to_string())

    print(f"\n── Country distribution ──────────────────────────────────────────")
    for label, df_ in [("Shortcut", shortcut_final), ("Nuanced", nuanced_final)]:
        print(f"  {label}:")
        print("   ", df_["true_country"].value_counts().to_dict())

    print(f"\n── Category confidence (mean score) ──────────────────────────────")
    print(f"  Shortcut: {shortcut_final['category_score'].mean():.3f}")
    print(f"  Nuanced:  {nuanced_final['category_score'].mean():.3f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "partition":            "shortcut",
            "categories":           "A, E, H",
            "n_candidates":         len(shortcut_pool),
            "n_selected":           len(shortcut_final),
            "mean_category_score":  shortcut_final["category_score"].mean(),
            "pct_label_confident":  shortcut_final["label_confident"].mean()
                                    if "label_confident" in shortcut_final.columns else None,
            "pct_correct_orig":     shortcut_final["correct_orig"].mean()
                                    if "correct_orig" in shortcut_final.columns else None,
        },
        {
            "partition":            "nuanced",
            "categories":           "B, C, D, F",
            "n_candidates":         len(nuanced_pool),
            "n_selected":           len(nuanced_final),
            "mean_category_score":  nuanced_final["category_score"].mean(),
            "pct_label_confident":  nuanced_final["label_confident"].mean()
                                    if "label_confident" in nuanced_final.columns else None,
            "pct_correct_orig":     nuanced_final["correct_orig"].mean()
                                    if "correct_orig" in nuanced_final.columns else None,
        },
    ])
    summary_path = os.path.join(PHASE2_DIR, "partition_summary.csv")
    summary.round(4).to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\n── Sanity checks ─────────────────────────────────────────────────")
    overlap = set(shortcut_final["u_id"]) & set(nuanced_final["u_id"])
    print(f"  Overlap between partitions: {len(overlap)}  (should be 0)")
    assert len(overlap) == 0, f"ERROR: {len(overlap)} images appear in both partitions!"

    if len(shortcut_final) < 10:
        print(f"  WARNING: Only {len(shortcut_final)} shortcut images found. "
              f"Consider lowering --k or removing --require_correct.")
    if len(nuanced_final) < 10:
        print(f"  WARNING: Only {len(nuanced_final)} nuanced images found.")

    print(f"\nDone. Partition ready for Phase 2 Steps 1–6.")
    print(f"  Phase 2 scripts read from: {PHASE2_DIR}/")


if __name__ == "__main__":
    main()
