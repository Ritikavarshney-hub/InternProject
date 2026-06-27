"""
Phase 2 — Step 1: Layer-wise Probing Analysis (RQ6)
Phase2_Execution_Plan.md: Step 1

Goal: Find at which transformer layer cultural information (country identity)
becomes linearly decodable from hidden representations.

At each layer l of the LLM, we train a lightweight logistic regression to
predict the country label from the mean-pooled hidden states. The "cultural
emergence curve" (accuracy vs. layer depth) reveals when the model internally
encodes cultural information — and whether shortcut images (flags, symbols)
encode it earlier than nuanced images (food, architecture).

Expected finding:
  Shortcut images → steep early rise, plateau at shallow layers
  Nuanced images  → gradual rise, peak at deeper layers
  → Mechanistic confirmation of Phase 1 CAS finding

Architecture (LLaVA-1.6):
  - Vision encoder: CLIP ViT-L/14-336  (24 blocks, 1024d)
  - MLP connector:  2-layer MLP
  - LLM:           InternLM2-7B        (32 layers, 4096d)
  We probe the 32 LLM layers + optionally the 24 vision encoder blocks.

Hidden state extraction:
  Full forward pass with output_hidden_states=True.
  Mean-pool over all sequence positions at each layer → (d_model,) vector per image.
  This captures cultural information distributed across all tokens.

Probe:
  Logistic regression (L2, C=1.0, max_iter=1000)
  5-fold stratified cross-validation (stratified on country label, 11 classes)
  Baseline: chance = 1/11 ≈ 9.1%

Requires (from Step 0):
  results/phase2/shortcut_ids.csv
  results/phase2/nuanced_ids.csv

Outputs:
  results/phase2/probing_accuracy_by_layer.csv    — per-layer accuracy (all / shortcut / nuanced)
  results/phase2/probing_emergence_curves.png     — main figure
  results/phase2/probing_country_breakdown.csv    — per-country peak layer
  results/phase2/hidden_states_cache.npz          — (optional) cached reps for reuse

Usage:
    # Pilot on dev sets (10+10 images) first
    python scripts/phase2/probing_analysis.py --pilot

    # Full run
    python scripts/phase2/probing_analysis.py

    # Full run including vision encoder layers
    python scripts/phase2/probing_analysis.py --include_vision

    # Also run CLIP as reference
    python scripts/phase2/probing_analysis.py --include_clip
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


# ── Hidden state extraction ────────────────────────────────────────────────────

def extract_llava_hidden_states(
    model, processor, image: Image.Image,
    device: str, include_vision: bool = False
) -> dict:
    """
    Run one LLaVA-1.6 forward pass and return mean-pooled hidden states
    for each LLM layer (and optionally each vision encoder block).

    Returns:
        {
          "llm":    np.ndarray shape (n_llm_layers, d_llm),    d=4096
          "vision": np.ndarray shape (n_vis_layers, d_vis),    d=1024  [if include_vision]
        }
    """
    prompt = "[INST] <image>\nWhich country does this image represent? [/INST]"

    inputs = processor(images=image.convert("RGB"),text=prompt,return_tensors="pt",)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # LLM hidden states: tuple of (n_llm_layers+1) tensors, each (1, seq_len, d)
    # Index 0 = embedding layer output; 1..32 = transformer layer outputs
    llm_hs = outputs.hidden_states[1:]   # skip embedding layer → 32 tensors

    llm_reps = np.stack([
        h[0].float().mean(dim=0).cpu().numpy()   # mean over sequence → (d,)
        for h in llm_hs
    ])   # (n_layers, d)

    result = {"llm": llm_reps}

    # Vision encoder hidden states (optional — heavier)
    if include_vision:
        vision_tower = model.vision_tower
        vis_inputs = processor.image_processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

        pixel_values = vis_inputs["pixel_values"].to(device)
        pixel_values = pixel_values.flatten(0, 1)

        with torch.no_grad():
            vis_outputs = model.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        vis_hs = vis_outputs.hidden_states[1:]   # skip embedding layer
        vis_reps = np.stack([
            h.float().mean(dim=(0, 1)).cpu().numpy()
            for h in vis_outputs.hidden_states[1:]
        ])
        result["vision"] = vis_reps

    return result


def extract_clip_hidden_states(image: Image.Image, device: str) -> np.ndarray:
    """
    Run CLIP ViT-L/14 and return mean-pooled hidden states per ViT block.
    Returns np.ndarray shape (24, 1024).
    """
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    print(model)
    model = model.to(device).eval()

    img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    block_reps = []
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            # out: (seq_len, batch, d) in open_clip
            rep = out.permute(1, 0, 2)[0].float().mean(dim=0).cpu().numpy()
            block_reps.append(rep)
        return hook

    for block in model.visual.transformer.resblocks:
        hooks.append(block.register_forward_hook(make_hook()))

    with torch.no_grad():
        model.encode_image(img_t)

    for h in hooks:
        h.remove()

    return np.stack(block_reps)   # (24, 1024)


# ── Probing ────────────────────────────────────────────────────────────────────

def build_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000,
                                      solver="lbfgs", random_state=42)),
    ])


def probe_all_layers(
    reps_all: np.ndarray,   # (n_images, n_layers, d)
    labels:   np.ndarray,   # (n_images,) integer country labels
    n_folds:  int = 5,
) -> np.ndarray:
    """
    For each layer, train a logistic regression probe with CV.
    Returns accuracy array shape (n_layers,).
    """
    n_images, n_layers, d = reps_all.shape
    from collections import Counter

    counts = Counter(labels)

    valid_classes = {c for c, n in counts.items() if n >= n_folds}

    mask = np.array([label in valid_classes for label in labels])

    reps_all = reps_all[mask]
    labels = labels[mask]

    print(f"Keeping {len(valid_classes)} classes.")
    print(f"Remaining images: {len(labels)}")

    min_class_count = min(Counter(labels).values())

    # choose a valid number of folds
    n_folds = min(n_folds, min_class_count)

    if n_folds < 2:
        raise ValueError(
            f"Need at least 2 samples per class. "
            f"Smallest class has {min_class_count} sample(s)."
    )

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )

    accs = []

    for l in range(n_layers):
        X = reps_all[:, l, :]   # (n_images, d)
        scores = cross_val_score(build_probe(), X, labels, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        accs.append(scores.mean())
        print(f"    Layer {l+1:2d}/{n_layers}  acc={scores.mean():.4f} ± {scores.std():.4f}",
              end="\r")

    print()
    return np.array(accs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Step 1: Layer-wise Probing")
    parser.add_argument("--pilot",         action="store_true",
                        help="Use dev sets (10+10 images) instead of full partition.")
    parser.add_argument("--include_vision", action="store_true",
                        help="Also probe LLaVA vision encoder layers.")
    parser.add_argument("--include_clip",  action="store_true",
                        help="Also run CLIP as a reference model.")
    parser.add_argument("--n_folds",       type=int, default=5)
    parser.add_argument("--cache_states",  action="store_true",
                        help="Save hidden states to disk for reuse (uses ~140 MB).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Pilot: {args.pilot}")

    # ── Check prerequisites ────────────────────────────────────────────────────
    for f in ["shortcut_ids.csv", "nuanced_ids.csv"]:
        p = os.path.join(PHASE2_DIR, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Run first: python scripts/phase2/build_partition.py"
            )

    # ── Load partition ─────────────────────────────────────────────────────────
    if args.pilot:
        short_df = pd.read_csv(os.path.join(PHASE2_DIR, "dev_set_shortcut.csv"))
        nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "dev_set_nuanced.csv"))
        print(f"PILOT MODE: {len(short_df)} shortcut + {len(nuan_df)} nuanced images")
    else:
        short_df = pd.read_csv(os.path.join(PHASE2_DIR, "shortcut_ids.csv"))
        nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "nuanced_ids.csv"))
        print(f"Full run: {len(short_df)} shortcut + {len(nuan_df)} nuanced images")

    meta      = pd.read_csv(META_CSV).set_index("u_id")
    all_ids   = pd.read_csv(IDS_CSV)["u_id"].tolist()
    countries = sorted(meta["country"].unique().tolist())

    # Encode country labels
    le = LabelEncoder().fit(countries)

    # Partition membership flags
    short_set = set(short_df["u_id"])
    nuan_set  = set(nuan_df["u_id"])

    # Use full sample for overall probing; partition for stratified comparison
    if args.pilot:
        probe_ids = list(short_set | nuan_set)
    else:
        probe_ids = all_ids

    # ── Load dataset images ────────────────────────────────────────────────────
    print("Loading dataset images...")
    ds = load_from_disk(DATASET_PATH)["test"]
    ds = ds.filter(lambda x: x["u_id"] in set(probe_ids))
    id_to_image = {r["u_id"]: r["image"] for r in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Load LLaVA-1.6 ────────────────────────────────────────────────────────
    print("\nLoading LLaVA-1.6...")
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    print(model)
    model.eval()
    print("  LLaVA loaded.")

    # ── Extract hidden states ──────────────────────────────────────────────────
    print(f"\nExtracting hidden states for {len(probe_ids)} images...")
    all_reps_llm    = []   # (n_images, n_layers, d)
    all_reps_vision = []
    all_labels      = []
    all_uids        = []
    is_shortcut     = []
    is_nuanced      = []

    t0 = time.time()
    for idx, u_id in enumerate(probe_ids, 1):
        if u_id not in id_to_image or u_id not in meta.index:
            continue

        image   = id_to_image[u_id]
        country = meta.loc[u_id, "country"]

        hs = extract_llava_hidden_states(
            model, processor, image, device, include_vision=args.include_vision
        )

        all_reps_llm.append(hs["llm"])
        if args.include_vision and "vision" in hs:
            all_reps_vision.append(hs["vision"])

        all_labels.append(le.transform([country])[0])
        all_uids.append(u_id)
        is_shortcut.append(u_id in short_set)
        is_nuanced.append(u_id in nuan_set)

        elapsed = time.time() - t0
        eta     = elapsed / idx * (len(probe_ids) - idx)
        print(f"  [{idx}/{len(probe_ids)}] {u_id[:8]}… | country={country} | "
              f"ETA {eta/60:.1f}min", end="\r")

    print()
    del model   # free VRAM

    reps_llm = np.stack(all_reps_llm)    # (N, n_layers, d)
    labels   = np.array(all_labels)       # (N,)
    is_short = np.array(is_shortcut)
    is_nuan  = np.array(is_nuanced)

    print(f"\nHidden states shape: {reps_llm.shape}")
    print(f"  N={reps_llm.shape[0]}  layers={reps_llm.shape[1]}  d={reps_llm.shape[2]}")

    if args.cache_states:
        cache_path = os.path.join(PHASE2_DIR, "hidden_states_cache.npz")
        np.savez_compressed(cache_path, reps=reps_llm, labels=labels,
                            uids=all_uids, is_shortcut=is_short, is_nuanced=is_nuan)
        print(f"  Cached → {cache_path}")

    # ── Probe at each layer ────────────────────────────────────────────────────
    n_layers = reps_llm.shape[1]
    chance   = 1.0 / len(countries)
    print(f"\nProbing {n_layers} LLM layers (chance baseline = {chance:.3f})")

    results = {}

    # Full sample
    print("\n  [ALL IMAGES]")
    if len(reps_llm) >= 10:
        results["all"] = probe_all_layers(reps_llm, labels, args.n_folds)
    else:
        print("    Too few images for probing — skipping.")
        results["all"] = np.full(n_layers, np.nan)

    # Shortcut subset
    if is_short.sum() >= 10:
        print(f"\n  [SHORTCUT — {is_short.sum()} images]")
        results["shortcut"] = probe_all_layers(
            reps_llm[is_short], labels[is_short], min(args.n_folds, is_short.sum())
        )
    else:
        print(f"\n  [SHORTCUT] only {is_short.sum()} images — skipping.")
        results["shortcut"] = np.full(n_layers, np.nan)

    # Nuanced subset
    if is_nuan.sum() >= 10:
        print(f"\n  [NUANCED — {is_nuan.sum()} images]")
        results["nuanced"] = probe_all_layers(
            reps_llm[is_nuan], labels[is_nuan], min(args.n_folds, is_nuan.sum())
        )
    else:
        print(f"\n  [NUANCED] only {is_nuan.sum()} images — skipping.")
        results["nuanced"] = np.full(n_layers, np.nan)

    # Vision encoder layers (optional)
    if args.include_vision and all_reps_vision:
        reps_vis = np.stack(all_reps_vision)
        print(f"\n  [VISION ENCODER — {reps_vis.shape[1]} layers]")
        results["vision"] = probe_all_layers(reps_vis, labels, args.n_folds)

    # ── CLIP reference (optional) ──────────────────────────────────────────────
    if args.include_clip:
        print(f"\nRunning CLIP reference probing...")
        clip_reps = []
        for u_id in all_uids:
            rep = extract_clip_hidden_states(id_to_image[u_id], device)
            clip_reps.append(rep)
        clip_reps = np.stack(clip_reps)
        print(f"  CLIP shape: {clip_reps.shape}")
        results["clip_all"] = probe_all_layers(clip_reps, labels, args.n_folds)

    # ── Save results ───────────────────────────────────────────────────────────
    layer_idx = np.arange(1, n_layers + 1)
    df_out = pd.DataFrame({"layer": layer_idx})
    for key, acc in results.items():
        if len(acc) == n_layers:
            df_out[f"acc_{key}"] = acc.round(5)

    # Peak layer per subset
    for key, acc in results.items():
        if not np.all(np.isnan(acc)):
            peak = int(np.nanargmax(acc)) + 1
            peak_acc = float(np.nanmax(acc))
            print(f"  Peak layer ({key:12s}): layer {peak:2d}  acc={peak_acc:.4f}")

    csv_path = os.path.join(PHASE2_DIR, "probing_accuracy_by_layer.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nAccuracy table → {csv_path}")

    # ── Per-country peak layer ─────────────────────────────────────────────────
    # ── Plot emergence curves ──────────────────────────────────────────────────
    _plot_emergence(results, n_layers, chance, PHASE2_DIR)
    print("\nDone.")


def _plot_emergence(results: dict, n_layers: int, chance: float, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(1, n_layers + 1)

    styles = {
        "all":      dict(color="black",     lw=2,   ls="-",  label="All images"),
        "shortcut": dict(color="#d73027",   lw=2,   ls="-",  label="Shortcut (A/E/H)"),
        "nuanced":  dict(color="#4575b4",   lw=2,   ls="-",  label="Nuanced (B/C/D/F)"),
        "vision":   dict(color="gray",      lw=1.5, ls="--", label="Vision encoder"),
        "clip_all": dict(color="green",     lw=1.5, ls=":",  label="CLIP (reference)"),
    }

    for key, acc in results.items():
        if np.all(np.isnan(acc)):
            continue
        s = styles.get(key, dict(lw=1.5, ls="-"))
        x = np.arange(1, len(acc) + 1)
        ax.plot(x, acc, **s)

        # Mark peak
        peak_l = int(np.nanargmax(acc)) + 1
        peak_a = float(np.nanmax(acc))
        ax.axvline(peak_l, color=s.get("color", "gray"),
                   lw=0.8, ls="--", alpha=0.4)
        ax.annotate(f"L{peak_l}", xy=(peak_l, peak_a),
                    xytext=(peak_l + 0.5, peak_a),
                    fontsize=7, color=s.get("color", "gray"))

    ax.axhline(chance, color="gray", lw=1, ls=":", label=f"Chance={chance:.3f}")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("5-fold CV accuracy (country prediction)")
    ax.set_title("Cultural Emergence Curves — LLaVA-1.6\n"
                 "At which layer does the model encode country identity?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_layers)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig_path = os.path.join(out_dir, "probing_emergence_curves.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
