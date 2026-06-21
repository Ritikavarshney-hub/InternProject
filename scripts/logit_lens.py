"""
Phase 2 — Step 2: Logit Lens / Prediction Commitment Tracking (RQ7)
Phase2_Execution_Plan.md: Step 2

Goal: Find the exact layer where LLaVA-1.6 "commits" to a cultural prediction,
and directly link early commitment to shortcut-category reliance (Phase 1 CAS).

Method: At each LLM layer l, take the hidden state at the last token position
(where the country name will be predicted) and project through the unembedding
matrix to get a vocabulary distribution. Track when the correct country enters
the top-5 and stays there — that is the commitment layer.

Connection to Step 1:
  Step 1 found: shortcut peak = layer 8, nuanced peak = layer 30.
  Step 2 asks: does the model also COMMIT to its answer earlier for shortcut images?
  If yes → consistent mechanistic story across two independent methods.

Connection to Phase 1:
  Correlate commitment_layer with Phase 1 deletion_auc_gap (from deletion_insertion_auc.py).
  Hypothesis: images where masking top patches causes a large confidence drop (high AUC gap)
  also commit at earlier layers (the model relies on a small number of critical patches
  that it processes quickly).

Architecture note (LLaVA-1.6 / InternLM2-7B):
  - model.model.layers[0..31]     — 32 transformer layers
  - model.model.norm               — final RMS norm
  - model.lm_head                  — unembedding matrix (vocab_size × 4096)
  - No weight tying between input embedding and lm_head in InternLM2

Procedure:
  1. Build prompt ending at the country prediction point:
       "[INST] <image>\\nWhich country is most strongly represented? Country: [/INST]"
  2. Forward pass with output_hidden_states=True (no generation needed)
  3. At each layer l: h = hidden_states[l][0, -1, :]  ← last token position
  4. logits^(l) = lm_head(norm(h))
  5. Softmax → record prob and rank of correct country's first token
  6. Commitment layer = first l where correct country is in top-5 and stays there

Requires:
  results/phase2/shortcut_ids.csv  (from Step 0)
  results/phase2/nuanced_ids.csv   (from Step 0)
  results/analysis/deletion_auc_clip.csv  (from Phase 1 Step 3)  [optional]

Outputs:
  results/phase2/commitment_layers.csv           — per-image commitment layer + metadata
  results/phase2/logit_trajectories_examples.png — shortcut vs nuanced case study
  results/phase2/commitment_layer_distribution.png — box plots (the key figure)
  results/phase2/commitment_vs_deletion_auc.png  — scatter: commitment vs AUC gap
  results/phase2/logit_lens_summary.csv          — mean commitment layer per subset

Usage:
    python scripts/phase2/logit_lens.py --pilot
    python scripts/phase2/logit_lens.py
    python scripts/phase2/logit_lens.py --top_k 3   # commit when in top-3 (stricter)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = "/DATA/bt24eee096/cultural_vlm/data/CulturalVQA"
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "cultural_vlm", "results")
PHASE2_DIR   = os.path.join(RESULTS_DIR,  "phase2")
META_CSV     = os.path.join(RESULTS_DIR,  "sample_metadata.csv")
IDS_CSV      = os.path.join(RESULTS_DIR,  "sample_ids.csv")
DEL_AUC_CSV  = os.path.join(RESULTS_DIR,  "analysis", "deletion_auc_clip.csv")
PROBING_CSV  = os.path.join(PHASE2_DIR,   "probing_accuracy_by_layer.csv")


# ── Core logit lens function ───────────────────────────────────────────────────

def logit_lens_trajectory(
    model,
    processor,
    image: Image.Image,
    target_token_ids: list[int],   # first token of each country name
    target_idx: int,               # index of the correct country
    device: str,
) -> dict:
    """
    Run one forward pass and extract the logit-lens trajectory.

    Returns:
      probs       : np.ndarray (n_layers,)  — softmax prob of correct country at each layer
      ranks       : np.ndarray (n_layers,)  — rank of correct country at each layer (0=best)
      top5_ids    : list of lists           — top-5 country indices at each layer
    """
    prompt = "[INST] <image>\nWhich country is most strongly represented in this image?\nCountry: [/INST]"

    inputs = processor(prompt, image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, d)
    # Index 0 = embedding output; 1..32 = transformer layer outputs
    hidden_states = outputs.hidden_states[1:]   # 32 tensors

    n_layers = len(hidden_states)
    probs    = np.zeros(n_layers)
    ranks    = np.zeros(n_layers, dtype=int)
    top5_ids = []

    # Unembedding components
    if hasattr(model, "language_model"):
        base_model = model.language_model
    else:
        base_model = model

    norm = base_model.model.norm
    lm_head = base_model.lm_head

    target_tok_t = torch.tensor(target_token_ids, device=device)   # (n_countries,)

    for l, h in enumerate(hidden_states):
        # h: (1, seq_len, d) — take last token position
        h_last = h[0, -1, :]                          # (d,)

        with torch.no_grad():
            h_normed = norm(h_last.unsqueeze(0))       # (1, d)
            logits   = lm_head(h_normed).squeeze(0)    # (vocab_size,)

        # Restrict to country tokens only for interpretable ranking
        cand_logits = logits[target_tok_t]             # (n_countries,)
        cand_probs  = torch.softmax(cand_logits.float(), dim=0).cpu().numpy()

        probs[l] = cand_probs[target_idx]
        ranks[l] = int((cand_probs > cand_probs[target_idx]).sum())   # 0 = highest
        top5_ids.append(np.argsort(cand_probs)[::-1][:5].tolist())

    return {"probs": probs, "ranks": ranks, "top5_ids": top5_ids}


def commitment_layer(ranks: np.ndarray, top_k: int = 5) -> int:
    """
    First layer where correct country enters top-k and STAYS there for all subsequent layers.
    Returns layer index (1-based) or n_layers+1 if never commits.
    """
    n = len(ranks)
    for l in range(n):
        if ranks[l] < top_k and np.all(ranks[l:] < top_k):
            return l + 1   # 1-based
    return n + 1   # never committed within the model


# ── Image loader ───────────────────────────────────────────────────────────────

def load_images(probe_ids: list[str]) -> dict:
    if os.path.exists(DATASET_PATH):
        print("Loading images from local dataset...")
        ds = load_from_disk(DATASET_PATH)["test"]
        ds = ds.filter(lambda x: x["u_id"] in set(probe_ids))
        return {r["u_id"]: r["image"] for r in ds}
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}.\n"
        "The logit lens requires running LLaVA on images to extract hidden states.\n"
        "Please restore the dataset or provide images via another method."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Step 2: Logit Lens")
    parser.add_argument("--pilot",   action="store_true",
                        help="Use dev sets (10+10 images).")
    parser.add_argument("--top_k",  type=int, default=5,
                        help="Commitment = stays in top-k (default: 5).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Pilot: {args.pilot} | Top-k: {args.top_k}")

    # ── Check prerequisites ────────────────────────────────────────────────────
    for f in ["shortcut_ids.csv", "nuanced_ids.csv"]:
        if not os.path.exists(os.path.join(PHASE2_DIR, f)):
            raise FileNotFoundError(f"Missing {f}. Run build_partition.py first.")

    # ── Load partition ─────────────────────────────────────────────────────────
    if args.pilot:
        short_df = pd.read_csv(os.path.join(PHASE2_DIR, "dev_set_shortcut.csv"))
        nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "dev_set_nuanced.csv"))
    else:
        short_df = pd.read_csv(os.path.join(PHASE2_DIR, "shortcut_ids.csv"))
        nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "nuanced_ids.csv"))

    meta      = pd.read_csv(META_CSV).set_index("u_id")
    all_ids   = pd.read_csv(IDS_CSV)["u_id"].tolist()
    countries = sorted(meta["country"].unique().tolist())
    short_set = set(short_df["u_id"])
    nuan_set  = set(nuan_df["u_id"])

    probe_ids = list(short_set | nuan_set) if args.pilot else all_ids
    print(f"Processing {len(probe_ids)} images "
          f"({len(short_set)} shortcut, {len(nuan_set)} nuanced)")

    # ── Load LLaVA-1.6 ────────────────────────────────────────────────────────
    print("\nLoading LLaVA-1.6...")
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("  LLaVA loaded.")

    # Pre-compute first token ID for each country (consistent with Phase 1 occlusion.py)
    target_token_ids = [
        processor.tokenizer.encode(c, add_special_tokens=False)[0]
        for c in countries
    ]
    print(f"  Country token IDs computed for {len(countries)} countries.")

    # ── Load images ────────────────────────────────────────────────────────────
    id_to_image = load_images(probe_ids)

    # ── Run logit lens per image ───────────────────────────────────────────────
    print(f"\nRunning logit lens on {len(probe_ids)} images...")
    records      = []
    trajectories = {}   # store for plotting examples

    for idx, u_id in enumerate(probe_ids, 1):
        if u_id not in id_to_image or u_id not in meta.index:
            continue

        image       = id_to_image[u_id]
        country     = meta.loc[u_id, "country"]
        target_idx  = countries.index(country)
        facet       = meta.loc[u_id, "facet"] if "facet" in meta.columns else None

        result = logit_lens_trajectory(
            model, processor, image, target_token_ids, target_idx, device
        )

        commit_l = commitment_layer(result["ranks"], top_k=args.top_k)
        n_layers = len(result["probs"])

        rec = {
            "u_id":             u_id,
            "true_country":     country,
            "facet":            facet,
            "is_shortcut":      u_id in short_set,
            "is_nuanced":       u_id in nuan_set,
            "commitment_layer": commit_l,
            "committed":        commit_l <= n_layers,
            "prob_at_commit":   float(result["probs"][commit_l - 1]) if commit_l <= n_layers else 0.0,
            "prob_at_layer_8":  float(result["probs"][7]),    # shortcut peak from Step 1
            "prob_at_layer_30": float(result["probs"][29]),   # nuanced peak from Step 1
            "prob_final":       float(result["probs"][-1]),
            "rank_final":       int(result["ranks"][-1]),
        }
        records.append(rec)
        trajectories[u_id] = result

        print(f"  [{idx:3d}/{len(probe_ids)}] {u_id[:8]}… "
              f"commit=L{commit_l:2d} | "
              f"{'shortcut' if u_id in short_set else 'nuanced' if u_id in nuan_set else 'other':8s} | "
              f"final_prob={result['probs'][-1]:.3f}", end="\r")

    print()

    df = pd.DataFrame(records)
    out_path = os.path.join(PHASE2_DIR, "commitment_layers.csv")
    df.to_csv(out_path, index=False)
    print(f"\nPer-image results → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_layers = len(list(trajectories.values())[0]["probs"])
    print(f"\n── Commitment layer summary (top-{args.top_k}) ─────────────────────")
    for label, mask_col in [("All", None), ("Shortcut", "is_shortcut"), ("Nuanced", "is_nuanced")]:
        sub = df if mask_col is None else df[df[mask_col]]
        committed = sub[sub["committed"]]
        if len(sub) == 0:
            continue
        print(f"  {label:10s}: "
              f"n={len(sub):3d}  "
              f"committed={len(committed):3d}/{len(sub)} ({len(committed)/len(sub):.0%})  "
              f"mean_commit={sub['commitment_layer'].mean():.1f}  "
              f"median={sub['commitment_layer'].median():.0f}")

    # ── Connection to Step 1 ───────────────────────────────────────────────────
    print(f"\n── Alignment with Step 1 probing peaks ──────────────────────────")
    if len(df[df["is_shortcut"]]) > 0:
        print(f"  Shortcut: probing peak = L8  | "
              f"mean commitment = L{df[df['is_shortcut']]['commitment_layer'].mean():.1f}")
    if len(df[df["is_nuanced"]]) > 0:
        print(f"  Nuanced:  probing peak = L30 | "
              f"mean commitment = L{df[df['is_nuanced']]['commitment_layer'].mean():.1f}")

    # Save summary
    summary_rows = []
    for label, mask in [("all", None), ("shortcut", "is_shortcut"), ("nuanced", "is_nuanced")]:
        sub = df if mask is None else df[df[mask]]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "subset":              label,
            "n":                   len(sub),
            "pct_committed":       round(sub["committed"].mean(), 4),
            "mean_commit_layer":   round(sub["commitment_layer"].mean(), 2),
            "median_commit_layer": round(sub["commitment_layer"].median(), 1),
            "std_commit_layer":    round(sub["commitment_layer"].std(), 2),
            "mean_final_prob":     round(sub["prob_final"].mean(), 4),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(PHASE2_DIR, "logit_lens_summary.csv"), index=False
    )

    # ── Correlation with Phase 1 Deletion AUC gap ─────────────────────────────
    if os.path.exists(DEL_AUC_CSV):
        print(f"\n── Correlation: commitment layer vs Phase 1 deletion AUC gap ────")
        del_df  = pd.read_csv(DEL_AUC_CSV).set_index("u_id")
        merged  = df.set_index("u_id").join(del_df[["auc_gap"]], how="inner")
        if len(merged) >= 10:
            rho, p = spearmanr(merged["commitment_layer"], merged["auc_gap"])
            print(f"  Spearman ρ(commit_layer, del_auc_gap) = {rho:.3f}  p={p:.4f}")
            print(f"  n = {len(merged)} images")
            if rho < -0.2:
                print("  → Earlier commitment correlates with larger AUC gap")
                print("    (model commits fast + relies on few patches = shortcut behaviour)")
            _plot_commitment_vs_auc(merged, PHASE2_DIR)
        else:
            print(f"  Only {len(merged)} overlapping images — skipping correlation.")
    else:
        print(f"\n  (Deletion AUC not found at {DEL_AUC_CSV} — skipping correlation)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_distribution(df, n_layers, args.top_k, PHASE2_DIR)
    _plot_trajectories(df, trajectories, countries, PHASE2_DIR)
    _plot_overlay_with_probing(df, PHASE2_DIR)

    print("\nDone.")


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _plot_distribution(df: pd.DataFrame, n_layers: int, top_k: int, out_dir: str):
    """Box plots of commitment layer distribution: shortcut vs nuanced vs all."""
    fig, ax = plt.subplots(figsize=(7, 4))

    groups = []
    labels = []
    colours = []

    for label, mask, colour in [
        ("All",      None,            "gray"),
        ("Shortcut\n(A/E/H)", "is_shortcut", "#d73027"),
        ("Nuanced\n(B/C/D/F)", "is_nuanced", "#4575b4"),
    ]:
        sub = df if mask is None else df[df[mask]]
        if len(sub) == 0:
            continue
        groups.append(sub["commitment_layer"].values)
        labels.append(f"{label}\n(n={len(sub)})")
        colours.append(colour)

    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Overlay Step 1 probing peaks
    ax.axhline(8,  color="#d73027", lw=1.2, ls="--", alpha=0.6, label="Probing peak shortcut (L8)")
    ax.axhline(30, color="#4575b4", lw=1.2, ls="--", alpha=0.6, label="Probing peak nuanced (L30)")
    ax.axhline(n_layers + 1, color="gray", lw=0.8, ls=":", label="Never committed")

    ax.set_ylabel("Commitment layer (top-{})".format(top_k))
    ax.set_title("Prediction Commitment Layer Distribution\n"
                 "Dashed lines = Step 1 probing peaks", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, n_layers + 3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "commitment_layer_distribution.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


def _plot_trajectories(df: pd.DataFrame, trajectories: dict,
                       countries: list, out_dir: str):
    """Side-by-side logit trajectories: one shortcut vs one nuanced example."""
    short_ids = df[df["is_shortcut"] & df["committed"]].sort_values("commitment_layer")
    nuan_ids  = df[df["is_nuanced"]  & df["committed"]].sort_values("commitment_layer")

    if len(short_ids) == 0 or len(nuan_ids) == 0:
        return

    eg_short = short_ids.iloc[0]
    eg_nuan  = nuan_ids.iloc[-1]   # latest committing nuanced for contrast

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, row, colour, label in [
        (axes[0], eg_short, "#d73027", "Shortcut"),
        (axes[1], eg_nuan,  "#4575b4", "Nuanced"),
    ]:
        u_id = row["u_id"]
        if u_id not in trajectories:
            continue
        t     = trajectories[u_id]
        probs = t["probs"]
        layers = np.arange(1, len(probs) + 1)

        ax.plot(layers, probs, color=colour, lw=2)
        ax.fill_between(layers, 0, probs, alpha=0.15, color=colour)
        ax.axvline(row["commitment_layer"], color=colour, lw=1.5, ls="--",
                   label=f"Commit: L{row['commitment_layer']:.0f}")
        ax.axhline(1/len(countries), color="gray", lw=1, ls=":",
                   label=f"Chance={1/len(countries):.3f}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(correct country)")
        ax.set_title(f"{label} — {row['true_country']} ({row.get('facet','')})\n"
                     f"Commits at L{row['commitment_layer']:.0f} | "
                     f"Final prob={row['prob_final']:.3f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(1, len(probs))
        ax.set_ylim(0, 1)

    fig.suptitle("Logit Lens: Probability of Correct Country Across Layers", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "logit_trajectories_examples.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


def _plot_commitment_vs_auc(merged: pd.DataFrame, out_dir: str):
    """Scatter: commitment layer vs Phase 1 deletion AUC gap."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colours = merged.apply(
        lambda r: "#d73027" if r.get("is_shortcut") else
                  "#4575b4" if r.get("is_nuanced") else "gray", axis=1
    )

    ax.scatter(merged["commitment_layer"], merged["auc_gap"],
               c=colours, alpha=0.6, s=30, edgecolors="none")

    # Trend line
    x = merged["commitment_layer"].values
    y = merged["auc_gap"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 5:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, p(xs), "k--", lw=1, alpha=0.6)

    rho, pval = spearmanr(merged["commitment_layer"], merged["auc_gap"])
    ax.set_xlabel("Commitment layer (logit lens)")
    ax.set_ylabel("Phase 1 Deletion AUC gap")
    ax.set_title(f"Commitment Layer vs Attribution Causal Strength\n"
                 f"Spearman ρ = {rho:.3f}  (p={pval:.3f})", fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', label='Shortcut', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575b4', label='Nuanced', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',    label='Other', markersize=8),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "commitment_vs_deletion_auc.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


def _plot_overlay_with_probing(df: pd.DataFrame, out_dir: str):
    """
    Bar chart comparing mean commitment layer vs Step 1 probing peak layer.
    Shows whether the two methods converge on the same layers.
    """
    probing_path = os.path.join(out_dir, "probing_accuracy_by_layer.csv")
    if not os.path.exists(probing_path):
        return

    prob_df = pd.read_csv(probing_path)

    fig, ax = plt.subplots(figsize=(7, 4))

    subsets = [
        ("All",      None,           "gray",    "acc_all"),
        ("Shortcut", "is_shortcut",  "#d73027", "acc_shortcut"),
        ("Nuanced",  "is_nuanced",   "#4575b4", "acc_nuanced"),
    ]

    x  = np.arange(len(subsets))
    w  = 0.35

    probing_peaks   = []
    commitment_means = []
    colours = []

    for label, mask, colour, col in subsets:
        sub = df if mask is None else df[df[mask]]

        # Commitment mean
        commitment_means.append(sub["commitment_layer"].mean() if len(sub) > 0 else np.nan)

        # Probing peak
        if col in prob_df.columns:
            probing_peaks.append(prob_df[col].idxmax() + 1)   # 1-based
        else:
            probing_peaks.append(np.nan)

        colours.append(colour)

    bars1 = ax.bar(x - w/2, probing_peaks, w, label="Probing peak layer (Step 1)",
                   color=colours, alpha=0.5, edgecolor="black")
    bars2 = ax.bar(x + w/2, commitment_means, w, label="Mean commitment layer (Step 2)",
                   color=colours, alpha=0.9, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in subsets])
    ax.set_ylabel("Layer (1–32)")
    ax.set_title("Cross-method convergence: Probing peak vs Commitment layer\n"
                 "Close values = both methods identify the same encoding depth", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 35)

    # Annotate bars
    for bar, val in list(zip(bars1, probing_peaks)) + list(zip(bars2, commitment_means)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"L{val:.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "probing_vs_commitment_overlay.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
