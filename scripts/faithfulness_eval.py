"""
Step 5 — Faithfulness Evaluation
Research Proposal §6.5

Answers RQ3: "When a VLM verbally explains its cultural reasoning,
does its stated justification align with what attribution analysis reveals?"

Procedure:
  1. Run Prompt 2 on each image × model:
       "Which country is most strongly represented in this image?
        After selecting, list the top three visual cues.
        Country:
        Visual Cue 1:
        Visual Cue 2:
        Visual Cue 3:"

  2. Parse the 3 stated cues.
     Map each cue phrase → category A–H using CLIP zero-shot similarity
     to the same 8 category descriptions used in Step 2.

  3. Get the top-3 patch categories from top_patch_categories_{model}.csv
     (produced by detect_cue_categories.py in Step 2).

  4. Faithfulness score per image =
       |{stated categories} ∩ {top-patch categories}| / 3

  5. Random baseline ≈ 1/8 = 0.125
     A low faithfulness score means the model CONFABULATES explanations.

Inputs (must exist before running):
  results/top_patch_categories_{model}.csv   ← from Step 2
  results/{model}_predictions.csv

Outputs:
  results/faithfulness_responses_{model}.csv   ← raw stated cues per image
  results/analysis/faithfulness_{model}.csv    ← per-image faithfulness score
  results/analysis/faithfulness_summary.csv    ← mean ± std per model (paper table)
  results/analysis/faithfulness_plot.png       ← bar chart vs random baseline

Usage:
    python faithfulness_eval.py --model llava
    python faithfulness_eval.py --model qwen2vl
    python faithfulness_eval.py --model internvl2
    python faithfulness_eval.py --model llava --pilot
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from datasets import load_from_disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

IDS_CSV  = os.path.join(RESULTS_DIR, "sample_ids.csv")
META_CSV = os.path.join(RESULTS_DIR, "sample_metadata.csv")

# ── Category taxonomy A–H (same as Steps 2, 4) ────────────────────────────────
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
CAT_KEYS = list(CATEGORIES.keys())   # ["A", "B", ..., "H"]


# ── Prompt 2 ──────────────────────────────────────────────────────────────────

def prompt2_text(country_list_str: str) -> str:
    return (
        f"Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}\n\n"
        f"After selecting the country, list exactly three visual cues "
        f"from the image that support your prediction.\n\n"
        f"Country:\n"
        f"Visual Cue 1:\n"
        f"Visual Cue 2:\n"
        f"Visual Cue 3:"
    )


# ── Response parser ────────────────────────────────────────────────────────────

def parse_response(text: str, countries: list) -> tuple[str | None, list[str]]:
    """Extract (country, [cue1, cue2, cue3]) from model output."""
    pred   = None
    cues   = []
    text_l = text.lower()

    # Country — first matching country in the output
    for c in countries:
        if re.search(r'\b' + re.escape(c.lower()) + r'\b', text_l):
            pred = c
            break

    # Visual cues — look for labelled lines
    for line in text.splitlines():
        line = line.strip()
        for prefix in ["Visual Cue 1:", "Visual Cue 2:", "Visual Cue 3:",
                        "Cue 1:", "Cue 2:", "Cue 3:", "1.", "2.", "3."]:
            if line.lower().startswith(prefix.lower()):
                cue = line[len(prefix):].strip(" :-–")
                if cue and len(cue) > 2:
                    cues.append(cue)
                break

    return pred, cues[:3]


# ── CLIP cue → category mapper ─────────────────────────────────────────────────

def build_category_features(clip_model, tokenizer, device: str) -> torch.Tensor:
    """Pre-encode category descriptions → (8, d) tensor."""
    feats = []
    for key in CAT_KEYS:
        _, prompts = CATEGORIES[key]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            f = clip_model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.mean(dim=0))
    cat_feats = torch.stack(feats)
    return cat_feats / cat_feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def cue_to_category(cue_text: str, clip_model, tokenizer,
                    cat_feats: torch.Tensor, device: str) -> str:
    """Map a free-text cue phrase to the closest category A–H."""
    tokens = tokenizer([cue_text]).to(device)
    feat   = clip_model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    sims   = (feat @ cat_feats.T).squeeze(0)
    return CAT_KEYS[int(sims.argmax())]


# ── Model runners ──────────────────────────────────────────────────────────────

def run_llava(sample_ids, id_to_image, countries):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    print("  Loading LLaVA-1.6...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16, load_in_4bit=True,
    )
    country_list_str = ", ".join(countries)
    prompt = f"[INST] <image>\n{prompt2_text(country_list_str)} [/INST]"
    responses = []

    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image:
            continue
        inputs = processor(prompt, id_to_image[u_id].convert("RGB"),
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        n   = inputs["input_ids"].shape[-1]
        txt = processor.tokenizer.decode(out[0][n:].cpu(), skip_special_tokens=True).strip()
        pred, cues = parse_response(txt, countries)
        responses.append({"u_id": u_id, "raw": txt, "stated_country": pred,
                          "cue_1": cues[0] if cues else None,
                          "cue_2": cues[1] if len(cues) > 1 else None,
                          "cue_3": cues[2] if len(cues) > 2 else None})
        if idx % 25 == 0:
            print(f"    [{idx}/{len(sample_ids)}]  parse_ok={len(cues)}/3")

    return responses


def run_qwen2vl(sample_ids, id_to_image, countries):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print("  Loading Qwen2-VL...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto").eval()

    country_list_str = ", ".join(countries)
    q_text = prompt2_text(country_list_str)
    responses = []

    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image:
            continue
        image = id_to_image[u_id].convert("RGB")
        msgs  = [{"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text": q_text}]}]
        text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = processor(text=[text], images=ii, videos=vi if vi else None,
                        return_tensors="pt", padding=True)
        # Cast float tensors to bfloat16 — required for 4-bit model with
        # bnb_4bit_compute_dtype=bfloat16. Plain .to("cuda") keeps float32
        # which causes CUBLAS_STATUS_NOT_SUPPORTED in the visual encoder.
        inp = {
            k: (v.to("cuda", dtype=torch.bfloat16)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.to("cuda")
                if isinstance(v, torch.Tensor) else v)
            for k, v in inp.items()
        }
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=20, do_sample=False)
        n   = inp["input_ids"].shape[-1]
        txt = processor.tokenizer.decode(out[0][n:].cpu(), skip_special_tokens=True).strip()
        pred, cues = parse_response(txt, countries)
        responses.append({"u_id": u_id, "raw": txt, "stated_country": pred,
                          "cue_1": cues[0] if cues else None,
                          "cue_2": cues[1] if len(cues) > 1 else None,
                          "cue_3": cues[2] if len(cues) > 2 else None})
        print(f"    [{idx}/{len(sample_ids)}]  parse_ok={len(cues)}/3")
        if idx % 25 == 0:
            print(f"    [{idx}/{len(sample_ids)}]  parse_ok={len(cues)}/3")

    return responses


def run_internvl2(sample_ids, id_to_image, countries):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    print("  Loading InternVL2-8B...")
    tok = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-8B",
                                        trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained("OpenGVLab/InternVL2-8B",
                                      torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

    tf = T.Compose([T.Lambda(lambda i: i.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    country_list_str = ", ".join(countries)
    q_text = prompt2_text(country_list_str)
    responses = []

    for idx, u_id in enumerate(sample_ids, 1):
        if u_id not in id_to_image:
            continue
        pv  = tf(id_to_image[u_id]).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            txt = model.chat(tok, pv, q_text,
                             generation_config=dict(max_new_tokens=100, do_sample=False))
        pred, cues = parse_response(txt, countries)
        responses.append({"u_id": u_id, "raw": txt, "stated_country": pred,
                          "cue_1": cues[0] if cues else None,
                          "cue_2": cues[1] if len(cues) > 1 else None,
                          "cue_3": cues[2] if len(cues) > 2 else None})
        if idx % 25 == 0:
            print(f"    [{idx}/{len(sample_ids)}]  parse_ok={len(cues)}/3")

    return responses


MODEL_RUNNERS = {"llava": run_llava, "qwen2vl": run_qwen2vl, "internvl2": run_internvl2}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=list(MODEL_RUNNERS))
    parser.add_argument("--pilot",  action="store_true")
    args   = parser.parse_args()
    model_name = args.model

    # ── Check prerequisite ────────────────────────────────────────────────────
    patch_csv = os.path.join(RESULTS_DIR, f"top_patch_categories_{model_name}.csv")
    if not os.path.exists(patch_csv):
        raise FileNotFoundError(
            f"Missing: {patch_csv}\n"
            f"Run first: python scripts/detect_cue_categories.py "
            f"--model_occlusion {model_name}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {model_name} | Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    sample_ids       = pd.read_csv(IDS_CSV)["u_id"].tolist()
    if args.pilot:
        sample_ids   = sample_ids[:20]
    meta             = pd.read_csv(META_CSV).set_index("u_id")
    countries        = sorted(meta["country"].unique().tolist())
    country_list_str = ", ".join(countries)

    # Top-3 patch categories per image from Step 2
    patch_df  = pd.read_csv(patch_csv)
    patch_cats = (patch_df[patch_df["patch_rank"] <= 3]
                  .groupby("u_id")["label"].apply(set).to_dict())

    # Images
    print("Loading dataset images...")
    ds = load_from_disk(DATASET_PATH)["test"]
    ds = ds.filter(lambda x: x["u_id"] in set(sample_ids))
    id_to_image = {r["u_id"]: r["image"] for r in ds}
    print(f"  {len(id_to_image)} images loaded.")

    # ── Run Prompt 2 ──────────────────────────────────────────────────────────
    print(f"\nRunning Prompt 2 on {len(sample_ids)} images...")
    responses = MODEL_RUNNERS[model_name](sample_ids, id_to_image, countries)

    df_resp = pd.DataFrame(responses)
    resp_path = os.path.join(RESULTS_DIR, f"faithfulness_responses_{model_name}.csv")
    df_resp.to_csv(resp_path, index=False)
    print(f"Raw responses → {resp_path}")

    # ── Load CLIP for cue → category mapping ─────────────────────────────────
    print("\nBuilding CLIP cue mapper...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model       = clip_model.to(device).eval()
    clip_tok         = open_clip.get_tokenizer("ViT-L-14")
    cat_feats        = build_category_features(clip_model, clip_tok, device)

    # ── Compute faithfulness scores ───────────────────────────────────────────
    records     = []
    parse_fails = 0

    for resp in responses:
        u_id = resp["u_id"]
        cues = [resp.get(f"cue_{i}") for i in [1, 2, 3]]
        cues = [c for c in cues if c]

        if not cues:
            parse_fails += 1

        # Map each stated cue → category A–H
        stated_cats = set()
        stated_cat_list = []
        for cue in cues:
            cat = cue_to_category(cue, clip_model, clip_tok, cat_feats, device)
            stated_cats.add(cat)
            stated_cat_list.append(cat)

        # Top-patch categories from Step 2
        top_cats = patch_cats.get(u_id, set())

        # Faithfulness = overlap / 3
        overlap      = len(stated_cats & top_cats)
        faithfulness = overlap / 3 if cues else 0.0

        meta_row = meta.loc[u_id] if u_id in meta.index else None
        records.append({
            "u_id":                  u_id,
            "model":                 model_name,
            "true_country":          meta_row["country"] if meta_row is not None else None,
            "facet":                 meta_row["facet"]   if meta_row is not None else None,
            "stated_country":        resp.get("stated_country"),
            "cue_1":                 resp.get("cue_1"),
            "cue_2":                 resp.get("cue_2"),
            "cue_3":                 resp.get("cue_3"),
            "stated_cats":           "|".join(stated_cat_list),
            "top_patch_cats":        "|".join(sorted(top_cats)),
            "overlap":               overlap,
            "faithfulness":          round(faithfulness, 4),
            "n_cues_parsed":         len(cues),
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(ANALYSIS_DIR, f"faithfulness_{model_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Per-image scores → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_f   = df["faithfulness"].mean()
    std_f    = df["faithfulness"].std()
    baseline = 1 / len(CAT_KEYS)     # 0.125

    print(f"\n── Faithfulness summary ({model_name}) ──────────────────────────")
    print(f"  Mean faithfulness  : {mean_f:.4f} ± {std_f:.4f}")
    print(f"  Random baseline    : {baseline:.4f}  (1/8 categories)")
    print(f"  Lift over baseline : {mean_f - baseline:+.4f}")
    print(f"  Parse failures     : {parse_fails}/{len(responses)}")
    print(f"  % zero faithfulness: {(df['faithfulness'] == 0).mean():.1%}")
    print(f"\n  By facet:")
    print(df.groupby("facet")["faithfulness"].mean().round(4).to_string())

    # ── Update combined summary ────────────────────────────────────────────────
    row = {
        "model":             model_name,
        "n_images":          len(df),
        "mean_faithfulness": round(mean_f, 4),
        "std_faithfulness":  round(std_f, 4),
        "random_baseline":   round(baseline, 4),
        "lift":              round(mean_f - baseline, 4),
        "pct_zero":          round((df["faithfulness"] == 0).mean(), 4),
        "parse_fail_rate":   round(parse_fails / max(len(responses), 1), 4),
    }
    summary_path = os.path.join(ANALYSIS_DIR, "faithfulness_summary.csv")
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing["model"] != model_name]
        updated  = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(summary_path, index=False)
    print(f"\nSummary table → {summary_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot(summary_path, baseline)
    print("Done.")


def _plot(summary_path: str, baseline: float):
    if not os.path.exists(summary_path):
        return
    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df["model"], df["mean_faithfulness"],
                  yerr=df["std_faithfulness"], capsize=4,
                  color="steelblue", edgecolor="white")
    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.2,
               label=f"Random baseline = {baseline:.3f}")
    ax.set_ylabel("Mean faithfulness\n(stated cue categories ∩ top-patch categories) / 3")
    ax.set_title("Do model explanations match what attribution reveals?\n"
                 "(low = confabulation)")
    ax.set_ylim(0, min(1.0, df["mean_faithfulness"].max() * 1.4 + 0.1))
    ax.legend(fontsize=9)
    for bar, val in zip(bars, df["mean_faithfulness"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(summary_path), "faithfulness_plot.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
