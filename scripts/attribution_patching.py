"""
Phase 2 — Step 3: Activation Patching / Causal Tracing (RQ8)
Phase2_Execution_Plan.md: Step 3

Goal: Identify which transformer layers CAUSALLY carry cultural information
from the image to the final prediction.

Methodology note (addresses the "where to patch" question):
  In LLaVA-1.6, visual token embeddings are injected into the input sequence
  BEFORE the 32-layer InternLM2/Mistral decoder stack. Every decoder layer
  then transforms all tokens (text + visual) together.

  We patch the OUTPUT of each decoder layer — which is exactly the
  RESIDUAL STREAM state entering the NEXT layer. In the transformer
  residual formulation:
       h_l  =  h_{l-1}  +  TransformerLayer_l( h_{l-1} )
  Patching h_l with clean values at visual positions = asking:
  "If visual tokens had clean representations at depth l, how much
  of the prediction is recovered?"

  This IS ROME-style activation patching applied to visual tokens:
  - Meng et al. 2022 cache h^(l) at subject token positions in the clean run
  - During the corrupted run they restore h^(l) at each depth l
  - We do exactly the same: cache visual token representations at each
    decoder layer output, then restore them one layer at a time

  The alternative "patch after multimodal embedding insertion" (i.e.,
  before layer 1) would trivially recover 100% (giving the decoder
  perfect visual features), which is uninformative. The causal question
  requires patching mid-network.

Layer path:
  model.language_model.model.layers[l]   ← the l-th decoder layer in LLaVA

Outputs:
  results/phase2/patching_recovery_all.csv  — per-layer mean recovery
  results/phase2/patching_per_image.csv     — per-image profiles
  results/phase2/patching_causal_profile.png
  results/phase2/patching_heatmap.png

Usage:
    python scripts/phase2/activation_patching.py --pilot   # 10+10, ~15 min
    python scripts/phase2/activation_patching.py           # 50+50, overnight
    python scripts/phase2/activation_patching.py --subset shortcut   
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH  = "/DATA/bt24eee096/cultural_vlm/data/CulturalVQA"
RESULTS_DIR   = "/DATA/bt24eee096/cultural_vlm/results"
OCCLUSION_DIR = os.path.join(RESULTS_DIR, "occlusion","occlusion")
PHASE2_DIR    = os.path.join(RESULTS_DIR, "phase2")
META_CSV      = os.path.join(RESULTS_DIR, "sample_metadata.csv")
IDS_CSV       = os.path.join(RESULTS_DIR, "sample_ids.csv")


# ── Image masking ─────────────────────────────────────────────────────────────

def load_occlusion(u_id: str) -> np.ndarray | None:
    for fname in [f"{u_id}_clip_mean.npy", f"{u_id}_clip_mean_7x7.npy"]:
        p = os.path.join(OCCLUSION_DIR, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def make_masked_image(image: Image.Image, scores: np.ndarray,
                      k_pct: float = 0.20) -> Image.Image:
    img_arr  = np.array(image.convert("RGB"))
    H, W     = img_arr.shape[:2]
    gh, gw   = scores.shape
    ph, pw   = H // gh, W // gw
    fill     = img_arr.mean(axis=(0, 1)).astype(np.uint8)
    n_mask   = max(1, int(round(k_pct * gh * gw)))
    flat_idx = np.argsort(scores.flatten())[::-1][:n_mask]
    masked   = img_arr.copy()
    for fi in flat_idx:
        r, c = fi // gw, fi % gw
        masked[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = fill
    return Image.fromarray(masked)


# ── Visual token position estimation ──────────────────────────────────────────

def find_visual_positions(processor, image: Image.Image, device: str):
    """
    Estimate (start, end) of visual token slice in the LLM input sequence.

    LLaVA replaces the single <image> placeholder token with N_img visual
    tokens. N_img depends on image size and tiling; we estimate it from
    actual processed vs raw sequence lengths.
    """
    prompt  = "[INST] <image>\nWhich country? [/INST]"
    raw_ids = processor.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    img_id  = processor.tokenizer.convert_tokens_to_ids("<image>")
    pos_raw = (raw_ids == img_id).nonzero(as_tuple=True)[0]

    inputs     = processor(prompt, image.convert("RGB"), return_tensors="pt")
    actual_len = inputs["input_ids"].shape[1]

    n_text   = len(raw_ids) - 1          # text tokens minus the <image> placeholder
    n_visual = max(1, actual_len - n_text)

    start = int(pos_raw[0].item()) if len(pos_raw) > 0 else 0
    return start, start + n_visual


# ── Probability extraction ────────────────────────────────────────────────────

@torch.no_grad()
def get_prob(model, processor, image: Image.Image,
             tgt_ids: list[int], tgt_idx: int, device: str) -> float:
    prompt = "[INST] <image>\nWhich country is most strongly represented? Country: [/INST]"
    inp    = processor(prompt, image.convert("RGB"), return_tensors="pt")
    inp    = {k: v.to(device) for k, v in inp.items()}
    out    = model(**inp, return_dict=True)
    cand   = out.logits[0, -1, :][tgt_ids]
    return float(torch.softmax(cand.float(), dim=0)[tgt_idx].item())


# ── Clean activation cache ────────────────────────────────────────────────────

def cache_clean_activations(model, processor, image: Image.Image,
                             vis_start: int, vis_end: int,
                             device: str) -> dict:
    """
    Cache visual token hidden states at every decoder layer output.
    These will be restored one-at-a-time during patched runs.

    Uses model.language_model.model.layers — the 32 InternLM2/Mistral
    decoder layers inside LlavaNextForConditionalGeneration.
    """
    prompt = "[INST] <image>\nWhich country is most strongly represented? Country: [/INST]"
    inp    = processor(prompt, image.convert("RGB"), return_tensors="pt")
    inp    = {k: v.to(device) for k, v in inp.items()}

    cache = {}

    def make_hook(layer_idx):
        def hook(module, inp_t, out_t):
            # out_t may be a tuple (hidden, cache_kv, ...) or just a tensor
            hidden = out_t[0] if isinstance(out_t, tuple) else out_t
            # Slice visual token positions; clone+detach to CPU to free VRAM
            n_vis = min(vis_end - vis_start, hidden.shape[1] - vis_start)
            cache[layer_idx] = hidden[0, vis_start:vis_start + n_vis, :].detach().cpu()
        return hook

    hooks = [
        layer.register_forward_hook(make_hook(l))
        for l, layer in enumerate(model.language_model.model.layers)
    ]

    with torch.no_grad():
        model(**inp, return_dict=True)

    for h in hooks:
        h.remove()

    return cache


# ── Patched forward pass ──────────────────────────────────────────────────────

@torch.no_grad()
def get_prob_with_patch(model, processor, masked_image: Image.Image,
                         clean_cache: dict, patch_layer: int,
                         vis_start: int, vis_end: int,
                         tgt_ids: list[int], tgt_idx: int,
                         device: str) -> float:
    """
    Forward pass on masked image, but at `patch_layer` output, restore
    visual token hidden states to their clean-run values.

    This is ROME-style causal tracing: patching the residual stream at
    depth l (= output of decoder layer l) with clean activations.
    """
    prompt     = "[INST] <image>\nWhich country is most strongly represented? Country: [/INST]"
    inp        = processor(prompt, masked_image.convert("RGB"), return_tensors="pt")
    inp        = {k: v.to(device) for k, v in inp.items()}
    clean_vis  = clean_cache.get(patch_layer)   # (n_vis, d) on CPU, may be None

    def patch_hook(module, inp_t, out_t):
        if clean_vis is None:
            return out_t
        hidden  = out_t[0] if isinstance(out_t, tuple) else out_t
        patched = hidden.clone()
        n_vis   = min(vis_end - vis_start, clean_vis.shape[0],
                      patched.shape[1] - vis_start)
        patched[0, vis_start:vis_start + n_vis, :] = (
            clean_vis[:n_vis].to(device=hidden.device, dtype=hidden.dtype)
        )
        if isinstance(out_t, tuple):
            return (patched,) + out_t[1:]
        return patched

    hook = model.language_model.model.layers[patch_layer].register_forward_hook(patch_hook)

    out    = model(**inp, return_dict=True)
    cand   = out.logits[0, -1, :][tgt_ids]
    result = float(torch.softmax(cand.float(), dim=0)[tgt_idx].item())

    hook.remove()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot",  action="store_true", help="10+10 pairs (~15 min).")
    parser.add_argument("--subset", choices=["shortcut", "nuanced", "both"], default="both")
    parser.add_argument("--k_pct",  type=float, default=0.20)
    args    = parser.parse_args()
    pilot_n = 10 if args.pilot else 50
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Pilot: {args.pilot} | Subset: {args.subset} | K={args.k_pct:.0%}")
    print("Loading LLaVA-1.6 in full bfloat16 (no quantization) for patching...")

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # n_layers via text config (works for both Mistral and LLaMA backbones)
    n_layers = model.language_model.config.num_hidden_layers
    print(f"  Loaded. Decoder layers: {n_layers}")

    meta      = pd.read_csv(META_CSV).set_index("u_id")
    countries = sorted(meta["country"].unique().tolist())
    tgt_ids   = [processor.tokenizer.encode(c, add_special_tokens=False)[0]
                 for c in countries]

    short_df = pd.read_csv(os.path.join(PHASE2_DIR, "shortcut_ids.csv"))
    nuan_df  = pd.read_csv(os.path.join(PHASE2_DIR, "nuanced_ids.csv"))

    pairs = []
    if args.subset in ("shortcut", "both"):
        pairs += [(u, "shortcut") for u in short_df["u_id"].tolist()[:pilot_n]]
    if args.subset in ("nuanced", "both"):
        pairs += [(u, "nuanced")  for u in nuan_df["u_id"].tolist()[:pilot_n]]

    print(f"  {len(pairs)} pairs × {n_layers} layers = {len(pairs)*n_layers} forward passes")
    print(f"  Estimated: {len(pairs)*n_layers*0.6/60:.0f}–{len(pairs)*n_layers*1.0/60:.0f} minutes")

    # Load images
    pair_ids = {u for u, _ in pairs}
    ds = load_from_disk(DATASET_PATH)["test"]
    ds = ds.filter(lambda x: x["u_id"] in pair_ids)
    id_to_image = {r["u_id"]: r["image"] for r in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Patching loop ──────────────────────────────────────────────────────────
    records = []
    t0      = time.time()

    for idx, (u_id, subset) in enumerate(pairs, 1):
        if u_id not in id_to_image or u_id not in meta.index:
            continue

        image   = id_to_image[u_id].convert("RGB")
        country = meta.loc[u_id, "country"]
        t_idx   = countries.index(country)
        scores  = load_occlusion(u_id)
        if scores is None:
            print(f"\n  [SKIP] {u_id}: no occlusion map")
            continue

        masked   = make_masked_image(image, scores, k_pct=args.k_pct)
        vs, ve   = find_visual_positions(processor, image, device)

        p_clean  = get_prob(model, processor, image,  tgt_ids, t_idx, device)
        p_corr   = get_prob(model, processor, masked, tgt_ids, t_idx, device)
        cache    = cache_clean_activations(model, processor, image, vs, ve, device)

        rec = {
            "u_id": u_id, "subset": subset, "true_country": country,
            "p_clean": round(p_clean, 5), "p_corrupted": round(p_corr, 5),
            "degradation": round(p_clean - p_corr, 5),
            "vis_start": vs, "vis_end": ve,
        }

        recoveries = []
        for l in range(n_layers):
            p_patch  = get_prob_with_patch(model, processor, masked, cache, l,
                                            vs, ve, tgt_ids, t_idx, device)
            rv       = round(p_patch - p_corr, 5)
            rec[f"recovery_L{l+1}"] = rv
            recoveries.append(rv)

        rv_arr    = np.array(recoveries)
        peak_l    = int(rv_arr.argmax()) + 1
        peak_v    = float(rv_arr.max())
        rec.update({"peak_recovery_layer": peak_l, "peak_recovery_value": round(peak_v, 5),
                    "mean_recovery": round(float(rv_arr.mean()), 5)})
        records.append(rec)

        elapsed = time.time() - t0
        eta     = elapsed / idx * (len(pairs) - idx)
        print(f"  [{idx:3d}/{len(pairs)}] {u_id[:8]} {subset:8s} {country:<10} | "
              f"p_clean={p_clean:.3f} p_corr={p_corr:.3f} "
              f"peak=L{peak_l}({peak_v:+.3f}) | ETA {eta/60:.1f}min", end="\r")

    print()
    if not records:
        print("No results. Check dataset/occlusion paths.")
        return

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(PHASE2_DIR, "patching_per_image.csv"), index=False)

    # ── Per-layer aggregate ────────────────────────────────────────────────────
    agg_rows = []
    for l in range(n_layers):
        col = f"recovery_L{l+1}"
        row = {"layer": l + 1,
               "recovery_all": round(float(df[col].mean()), 5),
               "recovery_std": round(float(df[col].std()),  5)}
        for sub in ["shortcut", "nuanced"]:
            v = df[df["subset"] == sub][col]
            row[f"recovery_{sub}"] = round(float(v.mean()), 5) if len(v) > 0 else None
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(PHASE2_DIR, "patching_recovery_all.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n── Recovery summary ──────────────────────────────────────────────")
    for sub in ["all", "shortcut", "nuanced"]:
        sub_df = df if sub == "all" else df[df["subset"] == sub]
        if len(sub_df) == 0:
            continue
        col  = "recovery_all" if sub == "all" else f"recovery_{sub}"
        vals = df_agg[col].dropna().values
        if len(vals) == 0:
            continue
        pk_l = int(vals.argmax()) + 1
        pk_v = float(vals.max())
        deg  = float(sub_df["degradation"].mean())
        print(f"  {sub:<10} n={len(sub_df):3d}  degradation={deg:+.4f}  "
              f"peak_recovery=L{pk_l}({pk_v:+.4f})")

    # Cross-phase triangulation
    prob_path = os.path.join(PHASE2_DIR, "probing_accuracy_by_layer.csv")
    if os.path.exists(prob_path):
        p = pd.read_csv(prob_path)
        print(f"\n── Triangulation vs Step 1 probing ──────────────────────────────")
        for sub, pcol in [("shortcut", "acc_shortcut"), ("nuanced", "acc_nuanced")]:
            rcol = f"recovery_{sub}"
            rv   = df_agg[rcol].dropna() if rcol in df_agg else pd.Series(dtype=float)
            if len(rv) == 0:
                continue
            pk_patch   = int(rv.values.argmax()) + 1
            pk_probing = int(p[pcol].idxmax()) + 1 if pcol in p.columns else None
            gap = abs(pk_patch - (pk_probing or 0))
            verdict = "✅ CONVERGE" if gap <= 3 else "⚠️  DIVERGE"
            print(f"  {sub:<10}: patching=L{pk_patch}  probing=L{pk_probing}  "
                  f"gap={gap} → {verdict}")

    _plot_causal_profile(df_agg, n_layers, PHASE2_DIR)
    _plot_heatmap(df, n_layers, PHASE2_DIR)

    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min  |  Done.")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_causal_profile(df_agg, n_layers, out_dir):
    fig, ax = plt.subplots(figsize=(11, 4))
    layers  = np.arange(1, n_layers + 1)
    styles  = {
        "recovery_all":      dict(color="black",   lw=2.5, ls="-",  label="All pairs"),
        "recovery_shortcut": dict(color="#d73027", lw=2,   ls="--", label="Shortcut (A/E/H)"),
        "recovery_nuanced":  dict(color="#4575b4", lw=2,   ls="--", label="Nuanced (B/C/D/F)"),
    }
    for col, sty in styles.items():
        if col not in df_agg or df_agg[col].isna().all():
            continue
        vals = df_agg[col].values.astype(float)
        ax.plot(layers, vals, **sty)
        ax.fill_between(layers, 0, vals.clip(min=0), alpha=0.07, color=sty["color"])
        pk = int(np.nanargmax(vals)) + 1
        ax.axvline(pk, color=sty["color"], lw=0.8, ls=":", alpha=0.5)
        ax.annotate(f"L{pk}", xy=(pk, float(vals[pk-1])),
                    fontsize=8, color=sty["color"], ha="left", va="bottom")

    ax.axhline(0, color="gray", lw=0.8, alpha=0.4)

    for path, overlays in [
        (os.path.join(out_dir, "probing_accuracy_by_layer.csv"),
         [("acc_shortcut", "#d73027", "Probing—shortcut"),
          ("acc_nuanced",  "#4575b4", "Probing—nuanced")]),
    ]:
        if os.path.exists(path):
            p = pd.read_csv(path)
            for col, c, lbl in overlays:
                if col in p.columns:
                    pk = p[col].idxmax() + 1
                    ax.axvline(pk, color=c, lw=1.5, ls="-.", alpha=0.5,
                               label=f"{lbl} (L{pk})")

    cka_path = os.path.join(out_dir, "cka_profile.csv")
    if os.path.exists(cka_path):
        from scipy.signal import argrelmin
        cka = pd.read_csv(cka_path)
        if "cka_all" in cka.columns:
            for m in argrelmin(cka["cka_all"].values, order=2)[0]:
                ax.axvline(m + 2, color="gray", lw=0.7, ls=":", alpha=0.35)
                ax.text(m + 2.1, 0, f"CKA\nL{m+2}", fontsize=6, color="gray",
                        ha="left", va="bottom")

    ax.set_xlabel("Decoder layer", fontsize=11)
    ax.set_ylabel("Mean recovery\nP(correct|patched) − P(correct|masked)", fontsize=10)
    ax.set_title("Causal Profile — Activation Patching\n"
                 "Which layer's visual representations are causally sufficient?", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlim(0.5, n_layers + 0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "patching_causal_profile.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Plot → {p}")


def _plot_heatmap(df, n_layers, out_dir):
    df_s = pd.concat([df[df["subset"] == "shortcut"],
                      df[df["subset"] == "nuanced"]]).reset_index(drop=True)
    if len(df_s) == 0:
        return
    layer_cols = [f"recovery_L{l+1}" for l in range(n_layers)
                  if f"recovery_L{l+1}" in df_s.columns]
    mat  = df_s[layer_cols].values.astype(float)
    ns   = (df_s["subset"] == "shortcut").sum()
    vmax = max(abs(mat).max(), 0.01)

    fig, ax = plt.subplots(figsize=(13, max(4, len(df_s) * 0.2 + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", origin="lower",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Recovery")
    ax.axhline(ns - 0.5, color="yellow", lw=2, ls="--")
    ax.text(1, ns / 2, "SHORTCUT", color="yellow", fontsize=9, rotation=90, va="center")
    ax.text(1, ns + (len(df_s)-ns)/2, "NUANCED",  color="white",  fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Layer"); ax.set_ylabel("Image pair")
    ax.set_xticks(np.arange(0, len(layer_cols), 4))
    ax.set_xticklabels(np.arange(1, len(layer_cols)+1, 4))
    ax.set_title("Recovery Heatmap\nRed = patching this layer restores prediction", fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, "patching_heatmap.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {p}")


if __name__ == "__main__":
    main()
