"""
Phase 2 — Step 5: CKA Layer Similarity + Triangulation (RQ6, RQ8)
Phase2_Execution_Plan.md: Step 5

Goal: Find where in the network the representation undergoes the LARGEST
transformation, then triangulate those transition points with:
  - Step 1 probing peaks   (shortcut L8, nuanced L30)
  - Step 2 logit lens      (commitment layers — overlay if available)

CKA (Centered Kernel Alignment) measures how similar two representation
matrices are. Between adjacent layers (l, l+1):
  - High CKA(l, l+1) → smooth, incremental change (layer does little)
  - Low  CKA(l, l+1) → large representational jump (critical layer)

Local minima in the CKA profile = "transition points" where the model
reorganises its internal representation most dramatically.

If the transition point co-localises with:
  - The shortcut probing peak (L8)     → shortcut info is processed at that transition
  - The nuanced  probing peak (L30)    → nuanced info is processed much later
  → STRONG mechanistic evidence that shortcuts take a fundamentally different
    processing path than nuanced cultural cues.

This is the KEY TRIANGULATION FIGURE of Phase 2.

Efficiency: reuses hidden_states_cache.npz from probing_analysis.py
If cache not present, re-extracts from LLaVA (requires dataset).

Reference: Kornblith et al. 2019 "Similarity of Neural Network
Representations Revisited" (linear CKA).

Inputs:
  results/phase2/hidden_states_cache.npz    ← from probing_analysis.py --cache_states
  results/phase2/shortcut_ids.csv           ← from build_partition.py
  results/phase2/nuanced_ids.csv            ← from build_partition.py
  results/phase2/probing_accuracy_by_layer.csv  ← from probing_analysis.py
  results/phase2/commitment_layers.csv          ← from logit_lens.py (optional)

Outputs:
  results/phase2/cka_profile.csv              — CKA(l,l+1) per layer pair
  results/phase2/cka_triangulation.png        — THE key figure: CKA + probing + commitment
  results/phase2/cka_shortcut_vs_nuanced.png  — separate CKA profiles per subset
  results/phase2/cka_transition_points.csv    — identified local minima (critical layers)

Usage:
    python scripts/phase2/cka_analysis.py
    python scripts/phase2/cka_analysis.py --n_samples 200  # CKA sample size
    python scripts/phase2/cka_analysis.py --n_bootstrap 5  # variance estimate
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelmin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "cultural_vlm", "results")
PHASE2_DIR   = os.path.join(RESULTS_DIR, "phase2")
DATASET_PATH = "/DATA/bt24eee096/cultural_vlm/data/CulturalVQA"
META_CSV     = os.path.join(RESULTS_DIR, "sample_metadata.csv")


# ── Linear CKA ────────────────────────────────────────────────────────────────

def centre(X: np.ndarray) -> np.ndarray:
    """Row-centre X: subtract column means."""
    return X - X.mean(axis=0, keepdims=True)


def linear_hsic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (Kornblith et al. 2019).
    X, Y: (n, d) matrices, already centred.
    Returns scalar HSIC value.
    """
    n = X.shape[0]
    XtX = X @ X.T   # (n, n) Gram matrix
    YtY = Y @ Y.T
    # Unbiased HSIC = sum of elementwise products of centred Gram matrices / (n-1)^2
    # Use the fast Frobenius-norm formula for linear kernels:
    # HSIC(X,Y) = ||X^T Y||_F^2 / (n-1)^2
    XtY = X.T @ Y   # (d_x, d_y)
    return float(np.sum(XtY ** 2)) / (n - 1) ** 2


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X (n, d_x) and Y (n, d_y).
    Returns value in [0, 1]; 1 = identical representations.
    """
    Xc = centre(X)
    Yc = centre(Y)
    hsic_xy = linear_hsic(Xc, Yc)
    hsic_xx = linear_hsic(Xc, Xc)
    hsic_yy = linear_hsic(Yc, Yc)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 1.0   # both have zero variance → trivially identical
    return float(hsic_xy / denom)


def cka_profile(reps: np.ndarray, max_samples: int = 200) -> np.ndarray:
    """
    Compute CKA(l, l+1) for all adjacent layer pairs.

    reps : (n_images, n_layers, d)
    Returns (n_layers - 1,) array of CKA values.
    """
    n, n_layers, d = reps.shape

    # Subsample for stability and speed
    if n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        reps = reps[idx]

    cka_vals = np.zeros(n_layers - 1)
    for l in range(n_layers - 1):
        X = reps[:, l,     :].astype(np.float64)
        Y = reps[:, l + 1, :].astype(np.float64)
        cka_vals[l] = cka(X, Y)

    return cka_vals


def find_transition_points(cka_vals: np.ndarray,
                           order: int = 2) -> np.ndarray:
    """
    Find local minima in the CKA profile — these are the transition points
    where the representation changes most dramatically.

    Returns 1-based layer indices of the BOUNDARIES (i.e., the gap between
    layer l and l+1 is at index l+1 in 1-based notation).
    """
    # argrelmin returns 0-based indices into cka_vals (length = n_layers-1)
    # Each index i corresponds to the boundary between layer i+1 and i+2 (1-based)
    minima = argrelmin(cka_vals, order=order)[0]
    return minima + 2   # convert to 1-based "after layer X" notation


# ── Load or extract hidden states ─────────────────────────────────────────────

def load_hidden_states(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (reps, labels, is_shortcut, is_nuanced)
    reps: (n, n_layers, d)
    """
    cache_path = os.path.join(PHASE2_DIR, "hidden_states_cache.npz")

    if os.path.exists(cache_path):
        print(f"Loading cached hidden states from {cache_path}")
        data       = np.load(cache_path, allow_pickle=True)
        reps       = data["reps"]
        labels     = data["labels"]
        is_short   = data["is_shortcut"]
        is_nuan    = data["is_nuanced"]
        print(f"  Shape: {reps.shape}  ({reps.shape[0]} images × {reps.shape[1]} layers × {reps.shape[2]}d)")
        return reps, labels, is_short, is_nuan

    # No cache — need to re-extract from LLaVA
    print("No hidden state cache found. Re-extracting from LLaVA-1.6...")
    print("Tip: run probing_analysis.py first (it builds the cache automatically).")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH} and no cache exists.\n"
            f"Run probing_analysis.py first to build the cache."
        )

    import torch
    from datasets import load_from_disk
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

    short_df = pd.read_csv(os.path.join(PHASE2_DIR, "shortcut_ids.csv"))
    nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "nuanced_ids.csv"))
    meta     = pd.read_csv(META_CSV).set_index("u_id")
    all_uids = pd.read_csv(os.path.join(RESULTS_DIR, "sample_ids.csv"))["u_id"].tolist()
    short_set = set(short_df["u_id"])
    nuan_set  = set(nuan_df["u_id"])

    ds = load_from_disk(DATASET_PATH)["test"]
    ds = ds.filter(lambda x: x["u_id"] in set(all_uids))
    id_to_image = {r["u_id"]: r["image"] for r in ds}

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb,
        torch_dtype=torch.bfloat16).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "[INST] <image>\nWhich country does this image represent? [/INST]"
    countries = sorted(meta["country"].unique().tolist())
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(countries)

    all_reps, all_labels, all_uids_out = [], [], []
    is_shortcut, is_nuanced = [], []

    for idx, u_id in enumerate(all_uids, 1):
        if u_id not in id_to_image or u_id not in meta.index:
            continue
        inputs = processor(prompt, id_to_image[u_id].convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[1:]
        rep = np.stack([h[0].float().mean(dim=0).cpu().numpy() for h in hs])
        all_reps.append(rep)
        all_labels.append(le.transform([meta.loc[u_id, "country"]])[0])
        all_uids_out.append(u_id)
        is_shortcut.append(u_id in short_set)
        is_nuanced.append(u_id in nuan_set)
        print(f"  [{idx}/{len(all_uids)}]", end="\r")

    print()
    reps = np.stack(all_reps)
    labels = np.array(all_labels)
    is_short = np.array(is_shortcut)
    is_nuan = np.array(is_nuanced)

    np.savez_compressed(cache_path, reps=reps, labels=labels,
                        uids=all_uids_out, is_shortcut=is_short, is_nuanced=is_nuan)
    print(f"Cache saved → {cache_path}")
    return reps, labels, is_short, is_nuan


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Step 5: CKA Analysis")
    parser.add_argument("--n_samples",   type=int, default=200,
                        help="Max images per CKA computation (default 200 for stability).")
    parser.add_argument("--n_bootstrap", type=int, default=0,
                        help="Bootstrap resamples for variance estimation (0=skip).")
    args = parser.parse_args()

    np.random.seed(42)

    # ── Load hidden states ─────────────────────────────────────────────────────
    reps, labels, is_short, is_nuan = load_hidden_states(args)
    n, n_layers, d = reps.shape
    print(f"\nImages: {n}  |  Layers: {n_layers}  |  d_model: {d}")
    print(f"Shortcut subset: {is_short.sum()}  |  Nuanced subset: {is_nuan.sum()}")

    # ── Compute CKA profiles ───────────────────────────────────────────────────
    print(f"\nComputing CKA profiles (n_samples={args.n_samples})...")

    print("  [ALL IMAGES]", end=" ")
    cka_all   = cka_profile(reps, args.n_samples)
    print(f"done. min={cka_all.min():.3f} at boundary L{cka_all.argmin()+1}→L{cka_all.argmin()+2}")

    cka_short = cka_nuan = None
    if is_short.sum() >= 5:
        print("  [SHORTCUT]  ", end=" ")
        cka_short = cka_profile(reps[is_short], min(args.n_samples, is_short.sum()))
        print(f"done. min={cka_short.min():.3f} at boundary L{cka_short.argmin()+1}→L{cka_short.argmin()+2}")

    if is_nuan.sum() >= 5:
        print("  [NUANCED]   ", end=" ")
        cka_nuan = cka_profile(reps[is_nuan], min(args.n_samples, is_nuan.sum()))
        print(f"done. min={cka_nuan.min():.3f} at boundary L{cka_nuan.argmin()+1}→L{cka_nuan.argmin()+2}")

    # Bootstrap variance (optional)
    cka_all_std = None
    if args.n_bootstrap > 0:
        print(f"\nBootstrap variance ({args.n_bootstrap} resamples)...")
        boot_vals = []
        rng = np.random.default_rng(42)
        for _ in range(args.n_bootstrap):
            idx = rng.choice(n, min(args.n_samples, n), replace=True)
            boot_vals.append(cka_profile(reps[idx], args.n_samples))
        cka_all_std = np.std(np.stack(boot_vals), axis=0)

    # ── Transition points ──────────────────────────────────────────────────────
    transition_all   = find_transition_points(cka_all)
    transition_short = find_transition_points(cka_short) if cka_short is not None else np.array([])
    transition_nuan  = find_transition_points(cka_nuan)  if cka_nuan  is not None else np.array([])

    print(f"\n── Transition points (local CKA minima) ─────────────────────────")
    print(f"  All images : {transition_all.tolist()}")
    print(f"  Shortcut   : {transition_short.tolist()}")
    print(f"  Nuanced    : {transition_nuan.tolist()}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    boundaries = np.arange(1, n_layers)   # boundary between layer l and l+1 (1-based)
    df = pd.DataFrame({
        "boundary":         boundaries,
        "layer_from":       boundaries,
        "layer_to":         boundaries + 1,
        "cka_all":          cka_all.round(5),
    })
    if cka_short is not None:
        df["cka_shortcut"] = cka_short.round(5)
    if cka_nuan is not None:
        df["cka_nuanced"]  = cka_nuan.round(5)
    if cka_all_std is not None:
        df["cka_all_std"]  = cka_all_std.round(5)

    csv_path = os.path.join(PHASE2_DIR, "cka_profile.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCKA profile → {csv_path}")

    # Transition points CSV
    tp_rows = []
    for tp in transition_all:
        tp_rows.append({"subset": "all", "transition_after_layer": int(tp),
                        "cka_value": float(cka_all[tp - 2]) if tp - 2 < len(cka_all) else None})
    for tp in transition_short:
        tp_rows.append({"subset": "shortcut", "transition_after_layer": int(tp),
                        "cka_value": float(cka_short[tp - 2]) if cka_short is not None and tp - 2 < len(cka_short) else None})
    for tp in transition_nuan:
        tp_rows.append({"subset": "nuanced", "transition_after_layer": int(tp),
                        "cka_value": float(cka_nuan[tp - 2]) if cka_nuan is not None and tp - 2 < len(cka_nuan) else None})
    pd.DataFrame(tp_rows).to_csv(os.path.join(PHASE2_DIR, "cka_transition_points.csv"), index=False)

    # ── Load overlay data from previous steps ──────────────────────────────────
    # Probing peaks (Step 1)
    probing_path = os.path.join(PHASE2_DIR, "probing_accuracy_by_layer.csv")
    probing_df   = pd.read_csv(probing_path) if os.path.exists(probing_path) else None

    probing_peak_all   = None
    probing_peak_short = None
    probing_peak_nuan  = None
    if probing_df is not None:
        if "acc_all"      in probing_df.columns:
            probing_peak_all   = int(probing_df["acc_all"].idxmax()) + 1
        if "acc_shortcut" in probing_df.columns:
            probing_peak_short = int(probing_df["acc_shortcut"].idxmax()) + 1
        if "acc_nuanced"  in probing_df.columns:
            probing_peak_nuan  = int(probing_df["acc_nuanced"].idxmax()) + 1
        print(f"\n── Step 1 probing peaks (from probing_accuracy_by_layer.csv) ────")
        print(f"  All:      L{probing_peak_all}")
        print(f"  Shortcut: L{probing_peak_short}")
        print(f"  Nuanced:  L{probing_peak_nuan}")

    # Commitment layers (Step 2 — optional, may not exist yet)
    commit_path = os.path.join(PHASE2_DIR, "commitment_layers.csv")
    commit_df   = None
    commit_mean_short = commit_mean_nuan = commit_mean_all = None
    if os.path.exists(commit_path):
        commit_df = pd.read_csv(commit_path)
        commit_mean_all   = commit_df["commitment_layer"].mean()
        if "is_shortcut" in commit_df.columns:
            commit_mean_short = commit_df[commit_df["is_shortcut"]]["commitment_layer"].mean()
            commit_mean_nuan  = commit_df[commit_df["is_nuanced"]]["commitment_layer"].mean()
        print(f"\n── Step 2 commitment layers (from commitment_layers.csv) ────────")
        print(f"  All:      L{commit_mean_all:.1f}")
        if commit_mean_short: print(f"  Shortcut: L{commit_mean_short:.1f}")
        if commit_mean_nuan:  print(f"  Nuanced:  L{commit_mean_nuan:.1f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_triangulation(
        cka_all, cka_short, cka_nuan, cka_all_std, n_layers,
        probing_peak_all, probing_peak_short, probing_peak_nuan,
        commit_mean_all, commit_mean_short, commit_mean_nuan,
        transition_all, PHASE2_DIR,
    )

    _plot_subset_comparison(cka_all, cka_short, cka_nuan, n_layers, PHASE2_DIR)

    # ── Convergence summary ────────────────────────────────────────────────────
    print("\n── Cross-method convergence check ───────────────────────────────")
    print("  (Do CKA transition points align with probing peaks and commitment layers?)")
    for subset, tp, prob_peak, commit_mean in [
        ("All",      transition_all,   probing_peak_all,   commit_mean_all),
        ("Shortcut", transition_short, probing_peak_short, commit_mean_short),
        ("Nuanced",  transition_nuan,  probing_peak_nuan,  commit_mean_nuan),
    ]:
        if len(tp) == 0:
            continue
        closest_tp = tp[np.argmin(np.abs(tp - (prob_peak or 0)))] if len(tp) > 0 else None
        print(f"\n  [{subset}]")
        print(f"    CKA transition points : {tp.tolist()}")
        if prob_peak:
            print(f"    Probing peak          : L{prob_peak}  (closest CKA: L{closest_tp})")
        if commit_mean:
            print(f"    Mean commitment layer : L{commit_mean:.1f}")
        if prob_peak and closest_tp:
            gap = abs(closest_tp - prob_peak)
            verdict = "✅ CONVERGE" if gap <= 3 else "⚠️  DIVERGE"
            print(f"    Gap = {gap} layers → {verdict}")

    print("\nDone.")


# ── Plot functions ─────────────────────────────────────────────────────────────

def _plot_triangulation(
    cka_all, cka_short, cka_nuan, cka_std,
    n_layers,
    probe_all, probe_short, probe_nuan,
    commit_all, commit_short, commit_nuan,
    transitions_all, out_dir,
):
    """
    THE key Phase 2 figure: CKA profile with probing peaks and commitment layers overlaid.
    Shows whether all three methods converge on the same critical layers.
    """
    boundaries = np.arange(1, n_layers)

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0])

    # ── CKA profile ────────────────────────────────────────────────────────────
    ax.plot(boundaries, cka_all, color="black", lw=2.5, label="CKA (all images)", zorder=3)
    if cka_std is not None:
        ax.fill_between(boundaries, cka_all - cka_std, cka_all + cka_std,
                        alpha=0.15, color="black")

    if cka_short is not None:
        ax.plot(boundaries, cka_short, color="#d73027", lw=1.5, ls="--",
                alpha=0.8, label="CKA (shortcut A/E/H)")
    if cka_nuan is not None:
        ax.plot(boundaries, cka_nuan, color="#4575b4", lw=1.5, ls="--",
                alpha=0.8, label="CKA (nuanced B/C/D/F)")

    # ── CKA transition points ──────────────────────────────────────────────────
    for tp in transitions_all:
        ax.axvline(tp - 0.5, color="black", lw=0.8, ls=":", alpha=0.4)
        ax.annotate(f"TP\nL{tp}", xy=(tp - 0.5, cka_all.min() + 0.01),
                    fontsize=6, ha="center", color="black", alpha=0.6)

    # ── Probing peaks (Step 1) ─────────────────────────────────────────────────
    if probe_all:
        ax.axvline(probe_all - 0.5, color="gray", lw=2, ls="-.",
                   label=f"Probing peak — all  (L{probe_all})", zorder=4)
    if probe_short:
        ax.axvline(probe_short - 0.5, color="#d73027", lw=2, ls="-.",
                   label=f"Probing peak — shortcut (L{probe_short})", zorder=4)
    if probe_nuan:
        ax.axvline(probe_nuan - 0.5, color="#4575b4", lw=2, ls="-.",
                   label=f"Probing peak — nuanced (L{probe_nuan})", zorder=4)

    # ── Commitment layers (Step 2) ─────────────────────────────────────────────
    if commit_all:
        ax.axvline(commit_all, color="purple", lw=2, ls="-",
                   label=f"Commit — all (L{commit_all:.0f})", alpha=0.7, zorder=4)
    if commit_short:
        ax.axvline(commit_short, color="#c2523c", lw=1.5, ls="-",
                   label=f"Commit — shortcut (L{commit_short:.0f})", alpha=0.7, zorder=4)
    if commit_nuan:
        ax.axvline(commit_nuan, color="#3a5fa8", lw=1.5, ls="-",
                   label=f"Commit — nuanced (L{commit_nuan:.0f})", alpha=0.7, zorder=4)

    ax.set_xlabel("Layer boundary (between layer l and l+1)", fontsize=11)
    ax.set_ylabel("Linear CKA  (1 = identical, 0 = completely different)", fontsize=11)
    ax.set_title(
        "CKA Triangulation: Representational Transitions × Probing Peaks × Commitment Layers\n"
        "Vertical lines = critical layers identified by each method independently",
        fontsize=11
    )
    ax.set_xlim(0.5, n_layers - 0.5)
    ax.set_ylim(max(0, cka_all.min() - 0.05), min(1, cka_all.max() + 0.05))
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "cka_triangulation.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


def _plot_subset_comparison(cka_all, cka_short, cka_nuan, n_layers, out_dir):
    """
    Separate CKA profiles for shortcut vs nuanced.
    If transition points differ → different processing paths.
    """
    if cka_short is None and cka_nuan is None:
        return

    boundaries = np.arange(1, n_layers)
    fig, axes  = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, cka_vals, colour, label, subset in [
        (axes[0], cka_short, "#d73027", "Shortcut (A/E/H)", "shortcut"),
        (axes[1], cka_nuan,  "#4575b4", "Nuanced  (B/C/D/F)", "nuanced"),
    ]:
        if cka_vals is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.plot(boundaries, cka_all, color="black", lw=1.5, alpha=0.4,
                label="All images (reference)")
        ax.plot(boundaries, cka_vals, color=colour, lw=2.5,
                label=label)

        # Mark transition points
        tps = find_transition_points(cka_vals)
        for tp in tps:
            ax.axvline(tp - 0.5, color=colour, lw=1.2, ls=":", alpha=0.6)
            ax.text(tp - 0.3, cka_vals.min() + 0.01,
                    f"L{tp}", fontsize=7, color=colour)

        ax.set_xlabel("Layer boundary")
        ax.set_title(f"CKA Profile: {label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlim(0.5, n_layers - 0.5)

    axes[0].set_ylabel("Linear CKA")
    fig.suptitle(
        "Stratified CKA Profiles — Do Shortcut and Nuanced Images\n"
        "Produce Transitions at Different Layer Depths?",
        fontsize=11
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "cka_shortcut_vs_nuanced.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
