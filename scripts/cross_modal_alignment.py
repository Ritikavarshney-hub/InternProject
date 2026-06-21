"""
Phase 2 — Step 6: Cross-Modal Alignment Tracking (RQ10)
Phase2_Execution_Plan.md: Step 6

Goal: At which transformer layer do visual token embeddings maximally align
with their assigned cultural text-concept embeddings (categories A–H)?

This answers RQ10: "Where does visual-cultural grounding occur in the network?"

Method:
  1. For each image i with assigned category C_i (from Phase 1 cue detection):
     - At each LLM layer l, take ALL visual token hidden states (not mean-pooled)
     - Compute cosine similarity between each token embedding and CLIP text
       embedding of category C_i
     - Take the MAXIMUM similarity across visual tokens → alignment_score(i, l)
  2. Average alignment_score over images per category → alignment_curve(C, l)
  3. Plot alignment curves per category A–H vs. layer depth

Key questions answered:
  - Which category aligns earliest?
    Expected: H (Appearance) — most visually unambiguous; already seen as
    highest CAS in Phase 1 and probing shortcut peak at L8
  - Does alignment peak BEFORE the commitment layer?
    If YES → model commits to cultural prediction before fully grounding
    visual features to text concepts → mechanistic shortcut behaviour
  - Is there a consistent "alignment layer" across categories?

Visual token identification:
  LLaVA-1.6 inserts image token embeddings at positions where the <image>
  placeholder appears in the prompt. We identify these positions from the
  input_ids before multimodal processing and extract hidden states at those
  indices. The maximum alignment across all visual tokens per layer captures
  the most culturally-relevant token at each depth.

Triangulation with previous steps:
  - Alignment curves are overlaid with:
    - Probing peaks (Step 1): L8 shortcut, L30 nuanced
    - CKA transition points (Step 5): L14, L20, L25
    - Commitment layers (Step 2): overall L25
  - If alignment peak for category H < commitment layer → shortcut mechanism
  - If alignment peak for category D/B > commitment layer → late grounding

Inputs (all from previous steps):
  results/cue_categories.csv           — image category assignments (Phase 1)
  results/phase2/shortcut_ids.csv      — shortcut partition
  results/phase2/nuanced_ids.csv       — nuanced partition
  results/phase2/probing_accuracy_by_layer.csv    — Step 1
  results/phase2/cka_profile.csv                  — Step 5
  results/phase2/commitment_layers.csv            — Step 2 (optional)

Outputs:
  results/phase2/alignment_scores.csv          — alignment_score(category, layer)
  results/phase2/alignment_curves.png          — one line per category A–H
  results/phase2/alignment_vs_commitment.png   — alignment peak vs commitment overlay
  results/phase2/alignment_summary.csv         — peak layer per category

Usage:
    python scripts/phase2/crossmodal_alignment.py --pilot   # 20 images
    python scripts/phase2/crossmodal_alignment.py           # full run
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = "/DATA/bt24eee096/cultural_vlm/data/CulturalVQA"
RESULTS_DIR  = "/DATA/bt24eee096/cultural_vlm/results"
PHASE2_DIR   = os.path.join(RESULTS_DIR, "phase2")
META_CSV     = os.path.join(RESULTS_DIR, "sample_metadata.csv")
IDS_CSV      = os.path.join(RESULTS_DIR, "sample_ids.csv")
CUE_CSV      = os.path.join(RESULTS_DIR, "cue_categories.csv")

# ── Category definitions (same as detect_cue_categories.py) ───────────────────
CATEGORIES = {
    "A": ("National Symbols",  ["a national flag, emblem, or coat of arms",
                                 "a national monument or patriotic symbol"]),
    "B": ("Clothing / Dress",  ["traditional cultural clothing or headwear",
                                 "ethnic costume or traditional dress"]),
    "C": ("Architecture",      ["distinctive regional architecture or building style",
                                 "a temple, mosque, church, or cultural landmark"]),
    "D": ("Food / Objects",    ["traditional food, dishes, or cultural utensils",
                                 "cultural tools, crafts, or everyday objects"]),
    "E": ("Script / Text",     ["written script, text, or signage in a non-Latin alphabet",
                                 "cultural symbols, calligraphy, or decorative writing"]),
    "F": ("Ritual / Festival", ["a cultural ceremony, festival, or religious ritual",
                                 "people participating in a traditional celebration"]),
    "G": ("Natural Landscape", ["distinctive natural landscape, terrain, or vegetation",
                                 "geography or scenic environment of a region"]),
    "H": ("Appearance",        ["people whose physical appearance or skin tone is prominent",
                                 "a portrait focused on ethnicity or racial features"]),
}
CAT_KEYS    = list(CATEGORIES.keys())
CAT_COLOURS = {
    "A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a", "D": "#ff7f00",
    "E": "#984ea3", "F": "#a65628", "G": "#999999", "H": "#f781bf",
}
# Shortcut vs nuanced grouping for line style
SHORTCUT_CATS = {"A", "E", "H"}


# ── CLIP text embeddings ───────────────────────────────────────────────────────

def build_llava_category_embeddings(model, processor):
    """
    Build category embeddings in the LLaVA hidden space.
    """

    if hasattr(model, "language_model"):
        base_model = model.language_model
    else:
        base_model = model

    embed_tokens = base_model.model.embed_tokens

    cat_embeds = {}

    with torch.no_grad():

        for key, (_, prompts) in CATEGORIES.items():

            prompt_embs = []

            for prompt in prompts:

                ids = processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )["input_ids"].to(embed_tokens.weight.device)

                emb = embed_tokens(ids)

                emb = emb.mean(dim=1).squeeze(0)

                emb = emb / emb.norm()

                emb = emb.float()
                prompt_embs.append(emb.cpu().numpy())

            x = np.stack(prompt_embs).mean(axis=0)

            x = x / (np.linalg.norm(x) + 1e-8)

            cat_embeds[key] = x

    return cat_embeds


# ── Visual token extraction ────────────────────────────────────────────────────

# LLaVA-1.6 uses -200 as the image token placeholder index internally.
# After multimodal processing, the actual sequence has these replaced with
# visual embeddings. We identify their positions from the raw input_ids.
LLAVA_IMAGE_TOKEN_INDEX = 32000   # <image> special token ID in the tokenizer


def extract_visual_token_hidden_states(
    model, processor, image: Image.Image, device: str
) -> np.ndarray:
    """
    Run one LLaVA-1.6 forward pass and return hidden states at VISUAL TOKEN positions.

    Returns: (n_layers, n_visual_tokens, d_model) float32 array.
    If visual token positions cannot be determined, returns mean-pooled fallback.
    """
    prompt = "[INST] <image>\nWhich country does this image represent? [/INST]"
    inputs = processor(prompt, image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Identify <image> token positions in the raw input_ids
    input_ids = inputs["input_ids"][0]   # (seq_len,)
    img_tok_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    img_positions = (input_ids == img_tok_id).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hs = outputs.hidden_states[1:]   # skip embedding layer → 32 layers

    if len(img_positions) > 0:
        # LLaVA expands one <image> token into N_img visual tokens.
        # In the processed sequence the expansion is at the same position.
        # Heuristic: take all tokens in the "image region" of the sequence.
        # For LLaVA-1.6 at 336px, N_img ≈ 576 (or 729 with tiling).
        start_pos = img_positions[0].item()
        # Find end of image region: next text token after image block
        # The sequence looks like: [text_pre] [img_tok * N] [text_post]
        # We detect this by checking where input_ids stops being <image>
        # In practice, LLaVA replaces the single <image> with many tokens
        # AFTER the processor step, so we use a heuristic:
        total_seq = hs[0].shape[1]
        n_pre     = start_pos   # text tokens before image
        # Use CLIP feature count: LLaVA-1.6 at 336px → 576 or 729 tokens
        # We estimate it from sequence length minus text tokens
        n_text_est = len(input_ids) - 1   # roughly, minus the single <image> placeholder
        n_visual   = total_seq - n_text_est
        n_visual   = max(n_visual, 1)
        end_pos    = n_pre + n_visual

        layer_reps = []
        for h in hs:
            vis_h = h[0, n_pre:end_pos, :].float().cpu().numpy()   # (n_vis, d)
            layer_reps.append(vis_h)
        return np.stack(layer_reps)   # (n_layers, n_vis, d)

    else:
        # Fallback: mean-pool all tokens
        layer_reps = []
        for h in hs:
            layer_reps.append(h[0].float().mean(dim=0, keepdim=True).cpu().numpy())
        return np.stack(layer_reps)   # (n_layers, 1, d)


def alignment_score_for_image(
    vis_hs: np.ndarray,     # (n_layers, n_vis, d)
    cat_embed: np.ndarray,  # (d,)
) -> np.ndarray:
    """
    For each layer, compute the MAXIMUM cosine similarity between any visual
    token embedding and the category text embedding.

    Returns (n_layers,) alignment scores.
    """
    n_layers = vis_hs.shape[0]
    scores   = np.zeros(n_layers)
    cat_norm = cat_embed / (np.linalg.norm(cat_embed) + 1e-8)

    for l in range(n_layers):
        H = vis_hs[l]   # (n_vis, d)
        # Normalise rows
        norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
        H_norm = H / norms
        sims = H_norm @ cat_norm   # (n_vis,)
        scores[l] = float(sims.max())   # peak alignment at this layer

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Step 6: Cross-modal Alignment")
    parser.add_argument("--pilot", action="store_true", help="Run on 20 images only.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Pilot: {args.pilot}")

    # ── Prerequisites check ────────────────────────────────────────────────────
    if not os.path.exists(CUE_CSV):
        raise FileNotFoundError(f"Missing {CUE_CSV}. Run detect_cue_categories.py first.")

    # ── Load metadata ──────────────────────────────────────────────────────────
    sample_ids = pd.read_csv(IDS_CSV)["u_id"].tolist()
    if args.pilot:
        sample_ids = sample_ids[:20]

    meta    = pd.read_csv(META_CSV).set_index("u_id")
    cue_df  = pd.read_csv(CUE_CSV).set_index("u_id")

    # ── Build CLIP category embeddings ─────────────────────────────────────────
   

    # ── Load LLaVA-1.6 ────────────────────────────────────────────────────────
    print("\nLoading LLaVA-1.6...")
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb,
        torch_dtype=torch.bfloat16).eval()
    print("  LLaVA loaded.")
    print("\nBuilding category embeddings...")
    cat_embeds = build_llava_category_embeddings(
        model,
        processor,
    )
    print(f"  {len(cat_embeds)} category embeddings built.")

    # ── Load images ────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Required for visual token extraction."
        )
    print("Loading dataset images...")
    ds = load_from_disk(DATASET_PATH)["test"]
    ds = ds.filter(lambda x: x["u_id"] in set(sample_ids))
    id_to_image = {r["u_id"]: r["image"] for r in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Main extraction loop ───────────────────────────────────────────────────
    # Store per-category alignment curves: cat → list of (n_layers,) arrays
    cat_alignment: dict[str, list] = {k: [] for k in CAT_KEYS}
    n_layers_seen = None

    print(f"\nExtracting visual token hidden states ({len(sample_ids)} images)...")
    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image or u_id not in cue_df.index:
            continue

        image    = id_to_image[u_id]
        cat_key  = cue_df.loc[u_id, "label_primary"]   # e.g. "A", "D", "H"

        if cat_key not in CATEGORIES:
            continue

        vis_hs = extract_visual_token_hidden_states(model, processor, image, device)
        # vis_hs: (n_layers, n_vis, d)

        if n_layers_seen is None:
            n_layers_seen = vis_hs.shape[0]

        # Compute alignment with this image's own category
        cat_embed = cat_embeds[cat_key]
        scores    = alignment_score_for_image(vis_hs, cat_embed)
        cat_alignment[cat_key].append(scores)

        print(f"  [{idx:3d}/{len(sample_ids)}] {u_id[:8]}… cat={cat_key} "
              f"({CATEGORIES[cat_key][0][:15]}) | peak_layer={scores.argmax()+1:2d} "
              f"peak_sim={scores.max():.3f}", end="\r")

    print()
    del model

    if n_layers_seen is None:
        print("No images processed. Check dataset and cue_categories.csv.")
        return

    # ── Aggregate per category ─────────────────────────────────────────────────
    print(f"\n── Per-category alignment peaks ─────────────────────────────────")
    summary_rows = []
    mean_curves  = {}

    for cat in CAT_KEYS:
        curves = cat_alignment[cat]
        if not curves:
            print(f"  {cat} ({CATEGORIES[cat][0]:<20}): no images")
            continue
        arr        = np.stack(curves)           # (n_images, n_layers)
        mean_curve = arr.mean(axis=0)           # (n_layers,)
        std_curve  = arr.std(axis=0)
        peak_layer = int(mean_curve.argmax()) + 1
        peak_sim   = float(mean_curve.max())
        mean_curves[cat] = (mean_curve, std_curve)

        print(f"  {cat} ({CATEGORIES[cat][0]:<22}): "
              f"n={len(curves):3d}  peak_layer=L{peak_layer:2d}  "
              f"peak_sim={peak_sim:.4f}  "
              f"{'[SHORTCUT]' if cat in SHORTCUT_CATS else '[NUANCED] '}")

        summary_rows.append({
            "category":    cat,
            "category_name": CATEGORIES[cat][0],
            "is_shortcut": cat in SHORTCUT_CATS,
            "n_images":    len(curves),
            "peak_layer":  peak_layer,
            "peak_sim":    round(peak_sim, 5),
            "mean_sim_l1": round(float(mean_curve[0]), 5),
            "mean_sim_final": round(float(mean_curve[-1]), 5),
        })

    # ── Save outputs ───────────────────────────────────────────────────────────
    # Per-layer alignment scores
    df_cols = {"layer": np.arange(1, n_layers_seen + 1)}
    for cat, (mc, _) in mean_curves.items():
        df_cols[f"align_{cat}"] = mc.round(5)
    pd.DataFrame(df_cols).to_csv(
        os.path.join(PHASE2_DIR, "alignment_scores.csv"), index=False
    )
    print(f"\nAlignment scores → {PHASE2_DIR}/alignment_scores.csv")

    # Summary
    df_summary = pd.DataFrame(summary_rows).sort_values("peak_layer")
    df_summary.to_csv(os.path.join(PHASE2_DIR, "alignment_summary.csv"), index=False)
    print(f"Summary         → {PHASE2_DIR}/alignment_summary.csv")

    # ── Load overlay data ──────────────────────────────────────────────────────
    probing_df  = _load_csv(os.path.join(PHASE2_DIR, "probing_accuracy_by_layer.csv"))
    cka_df      = _load_csv(os.path.join(PHASE2_DIR, "cka_profile.csv"))
    commit_df   = _load_csv(os.path.join(PHASE2_DIR, "commitment_layers.csv"))

    probe_peak_all   = _peak_from_csv(probing_df, "acc_all")
    probe_peak_short = _peak_from_csv(probing_df, "acc_shortcut")
    probe_peak_nuan  = _peak_from_csv(probing_df, "acc_nuanced")

    cka_transitions  = []
    if cka_df is not None and "cka_all" in cka_df.columns:
        from scipy.signal import argrelmin
        minima = argrelmin(cka_df["cka_all"].values, order=2)[0]
        cka_transitions = (minima + 2).tolist()   # 1-based

    commit_mean_all = None
    if commit_df is not None and "commitment_layer" in commit_df.columns:
        commit_mean_all = float(commit_df["commitment_layer"].mean())

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_alignment_curves(
        mean_curves, n_layers_seen,
        probe_peak_all, probe_peak_short, probe_peak_nuan,
        cka_transitions, commit_mean_all,
        PHASE2_DIR,
    )
    _plot_alignment_vs_commitment(df_summary, commit_mean_all, PHASE2_DIR)

    # ── Key interpretation ─────────────────────────────────────────────────────
    print("\n── Cross-phase interpretation ────────────────────────────────────")
    if commit_mean_all:
        print(f"  Overall commitment layer (Step 2): L{commit_mean_all:.1f}")
    for _, row in df_summary.sort_values("peak_layer").iterrows():
        cat = row["category"]
        before_commit = (commit_mean_all and row["peak_layer"] < commit_mean_all)
        status = "peaks BEFORE commitment → early grounding (shortcut)" if before_commit \
                 else "peaks AFTER  commitment → late grounding (nuanced)"
        print(f"  {cat} ({row['category_name']:<22}) peaks at L{row['peak_layer']:2d} → {status}")

    print("\nDone.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _peak_from_csv(df, col):
    if df is not None and col in df.columns:
        return int(df[col].idxmax()) + 1
    return None


def _plot_alignment_curves(
    mean_curves, n_layers,
    probe_all, probe_short, probe_nuan,
    cka_transitions, commit_all,
    out_dir,
):
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(1, n_layers + 1)

    for cat, (mc, sc) in mean_curves.items():
        name    = CATEGORIES[cat][0]
        colour  = CAT_COLOURS[cat]
        ls      = "-" if cat in SHORTCUT_CATS else "--"
        lw      = 2.2 if cat in {"H", "D"} else 1.4
        ax.plot(layers, mc, color=colour, lw=lw, ls=ls,
                label=f"{cat}: {name} (L{mc.argmax()+1})")
        ax.fill_between(layers, mc - 0.3*sc, mc + 0.3*sc, alpha=0.07, color=colour)
        # Mark peak
        pk = int(mc.argmax())
        ax.scatter(pk + 1, mc[pk], color=colour, s=40, zorder=5)

    # Overlay Step 1 probing peaks
    for lyr, lbl, col in [
        (probe_all,   f"Probing peak—all (L{probe_all})",      "black"),
        (probe_short, f"Probing peak—shortcut (L{probe_short})", "#d73027"),
        (probe_nuan,  f"Probing peak—nuanced (L{probe_nuan})",   "#4575b4"),
    ]:
        if lyr:
            ax.axvline(lyr, color=col, lw=1.5, ls="-.", alpha=0.6, label=lbl)

    # CKA transitions
    for tp in cka_transitions:
        ax.axvline(tp - 0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.text(tp - 0.3, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.97,
                f"CKA\nL{tp}", fontsize=6, ha="center", va="top", color="gray")

    # Commitment layer
    if commit_all:
        ax.axvline(commit_all, color="purple", lw=2, ls="-", alpha=0.6,
                   label=f"Commitment layer—all (L{commit_all:.0f})")

    ax.set_xlabel("Transformer layer (LLaVA-1.6 InternLM2)", fontsize=11)
    ax.set_ylabel("Max cosine similarity\n(visual token ↔ CLIP category text)", fontsize=10)
    ax.set_title(
        "Cross-Modal Alignment Curves — At Which Layer Do Visual Tokens\n"
        "Align With Their Cultural Category Concept?\n"
        "Solid = shortcut (A/E/H) | Dashed = nuanced (B/C/D/F)",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlim(0.5, n_layers + 0.5)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "alignment_curves.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


def _plot_alignment_vs_commitment(df_summary, commit_mean_all, out_dir):
    """
    Bar chart: peak alignment layer per category vs. commitment layer.
    Categories that align BEFORE commitment layer → early/shortcut grounding.
    """
    if len(df_summary) == 0:
        return

    df = df_summary.sort_values("peak_layer")
    colours = [CAT_COLOURS.get(c, "gray") for c in df["category"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(df)), df["peak_layer"], color=colours,
                  edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f"{r.category}\n{r.category_name[:12]}" for _, r in df.iterrows()],
        fontsize=8
    )
    ax.set_ylabel("Peak alignment layer")
    ax.set_title(
        "Peak Cross-Modal Alignment Layer per Category\n"
        "Categories aligning before commitment = shortcut grounding",
        fontsize=10
    )

    if commit_mean_all:
        ax.axhline(commit_mean_all, color="purple", lw=2, ls="--",
                   label=f"Commitment layer (L{commit_mean_all:.0f})")
        ax.fill_between([-0.5, len(df) - 0.5], 0, commit_mean_all,
                        alpha=0.05, color="purple")
        ax.text(len(df) - 0.4, commit_mean_all + 0.3,
                "commit", color="purple", fontsize=8)
        ax.legend(fontsize=9)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"L{row['peak_layer']}", ha="center", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "alignment_vs_commitment.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
