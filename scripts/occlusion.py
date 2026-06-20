"""
Milestone 3a — Core Occlusion Sensitivity Pipeline
Research Proposal Section 6.3 (Primary Method)

For every image × model pair, produces a patch-level attribution heatmap.
Positive value at [i,j] = masking that patch drops confidence → patch matters.

Output: results/occlusion/{u_id}_{model}_{fill}.npy          (default 14×14)
        results/occlusion/{u_id}_{model}_{fill}_7x7.npy      (when --grid 7)

Usage:
    python occlusion.py --model clip
    python occlusion.py --model llava
    python occlusion.py --model qwen2vl  --grid 7
    python occlusion.py --model internvl2 --grid 7
    python occlusion.py --model clip --fill black

Model loading mirrors the EXACT pipeline used in each predict_*.py script
so that the confidence extraction is guaranteed to be consistent with
the predictions already stored in results/*_predictions.csv.
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Paths  (relative to project root, one level above scripts/)
# ---------------------------------------------------------------------------

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH    = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
SAMPLE_IDS_CSV  = os.path.join(PROJECT_ROOT, "results", "sample_ids.csv")
SAMPLE_META_CSV = os.path.join(PROJECT_ROOT, "results", "sample_metadata.csv")
OCCLUSION_DIR   = os.path.join(PROJECT_ROOT, "results", "occlusion", "occlusion")

GRID_SIZE = 7   # default; override with --grid 7


# ---------------------------------------------------------------------------
# Core occlusion function  (model-agnostic)
# Proposal §6.3: replace patch with fill value, re-run model, measure Δconf
# ---------------------------------------------------------------------------

def occlusion_sensitivity(
    model_fn,
    image: Image.Image,
    grid_size: int = GRID_SIZE,
    fill: str = "mean",
) -> np.ndarray:
    """
    model_fn : PIL.Image -> float   confidence for the target country
    Returns (grid_size, grid_size) float32 ndarray.
    Positive value = masking that patch drops confidence (patch is important).
    """
    image   = image.convert("RGB")
    W, H    = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    img_arr = np.array(image)           # (H, W, 3) uint8

    if fill == "mean":
        fill_value = img_arr.mean(axis=(0, 1)).astype(np.uint8)
    elif fill == "black":
        fill_value = np.zeros(3, dtype=np.uint8)
    elif fill == "noise":
        rng      = np.random.default_rng(seed=42)
        noise_arr = rng.integers(0, 256, img_arr.shape, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown fill: {fill!r}.  Choose mean | black | noise")

    baseline_conf = model_fn(image)
    scores        = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            masked = img_arr.copy()
            y0, x0 = i * patch_h, j * patch_w
            if fill == "noise":
                masked[y0:y0+patch_h, x0:x0+patch_w] = \
                    noise_arr[y0:y0+patch_h, x0:x0+patch_w]
            else:
                masked[y0:y0+patch_h, x0:x0+patch_w] = fill_value
            conf = model_fn(Image.fromarray(masked))
            scores[i, j] = baseline_conf - conf

    return scores


# ===========================================================================
# CLIP  (ViT-L/14, discriminative baseline)
# Reference: predict_clip.py
# ===========================================================================

def load_clip_model(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer


def clip_confidence_fn(model, preprocess, tokenizer, countries, target_country, device):
    """Returns a closure: PIL.Image -> float (softmax probability for target_country)."""
    text_tokens = tokenizer([f"a photo from {c}" for c in countries]).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    if target_country not in countries:
        return lambda _img: 0.0

    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sims = (feat @ text_feats.T).squeeze(0).cpu().numpy()
        probs = torch.softmax(torch.tensor(sims) * 100.0, dim=0).numpy()
        return float(probs[target_idx])

    return fn


# ===========================================================================
# LLaVA-1.6 (Mistral-7B)
# Reference: predict_llava.py
# ===========================================================================

def load_llava_model():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    return model, processor


def llava_confidence_fn(model, processor, countries, target_country,
                         first_token_ids=None):
    """
    Confidence = softmax probability of target_country's first token.
    Uses Python list indexing on logits — device-safe regardless of
    where model.generate() places the output tensor.
    """
    country_list_str = ", ".join(countries)
    prompt = (
        f"[INST] <image>\n"
        f"Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}\n\n"
        f"Country: [/INST]"
    )

    # Python list (not tensor) — avoids CUDA/CPU device mismatch when indexing logits
    if first_token_ids is None:
        first_token_ids = [
            processor.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in countries
        ]

    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        inputs = processor(prompt, image.convert("RGB"),
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits      = out.scores[0][0]                  # (vocab_size,)
        cand_logits = logits[first_token_ids]            # list index — always device-safe
        probs       = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ===========================================================================
# Qwen2-VL-7B
#
# IMPORTANT — mirrors predict_qwen2vl.py (Internproject/scripts) EXACTLY:
#   • Full bfloat16, no BitsAndBytesConfig (that's how predictions were run)
#   • model.cuda() not device_map="auto"
#   • Prompt instructs model to output only a country name; no "Country:" pre-fill
#   • inputs moved to cuda via dict comprehension (not .to("cuda") on BatchEncoding)
#   • process_vision_info() for image extraction from messages
#   • Python list for first_token_ids → device-safe logit indexing
# ===========================================================================

def load_qwen2vl_model():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    # min/max_pixels caps dynamic tiling — same as predict_qwen2vl.py
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    model.eval()
    return model, processor


def qwen2vl_confidence_fn(model, processor, countries, target_country,
                           first_token_ids=None):
    """
    Confidence = softmax probability of target_country's first token,
    extracted from out.scores[0][0] — the first generated token's logits.

    The prompt instructs the model to output ONLY a country name so the
    first generated token is the country name token with high probability.
    """
    from qwen_vl_utils import process_vision_info

    country_list_str = ", ".join(countries)

    # Python list — device-safe when indexing GPU logit tensor
    if first_token_ids is None:
        first_token_ids = [
            processor.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in countries
        ]

    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        image = image.convert("RGB")

        # Prompt identical to predict_qwen2vl.py
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "You MUST answer with exactly ONE country name.\n\n"
                            f"Valid countries are:\n{country_list_str}\n\n"
                            "Choose ONLY from the list above.\n"
                            "Do not explain.\n"
                            "Do not say unknown.\n"
                            "Output only the country name."
                        ),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        )
        # Dict comprehension move — same pattern as predict_qwen2vl.py
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        logits      = out.scores[0][0]                  # first token logits (vocab_size,)
        cand_logits = logits[first_token_ids]            # list index — device-safe
        probs       = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ===========================================================================
# InternVL2-8B
#
# Architecture notes:
#   • model.generate(pixel_values=...) is a CUSTOM override in
#     modeling_internvl_chat.py that scatters visual embeddings into
#     input_embeds at positions where input_ids == img_context_token_id.
#     It then calls model.language_model.generate(inputs_embeds=...).
#
# Why we bypass model.generate(pixel_values=...):
#   • Tokenising "<IMG_CONTEXT>" * 256 as a Python string is unreliable —
#     the tokenizer can merge/split the repeated token differently, so the
#     number of img_ctx_id occurrences in input_ids ≠ N_IMG → CUDA assertion
#     "index out of bounds" in the scatter kernel.
#
# What we do instead:
#   1. Build input_ids_template with EXACT N_IMG img_ctx_id entries (torch.full)
#   2. Call model.extract_feature(pv) for visual embeddings
#   3. Scatter manually (same logic as the custom generate override)
#   4. Call model.language_model.generate(inputs_embeds=...) — standard HF path
# ===========================================================================

def load_internvl2_model():
    """
    Loads InternVL2-8B WITHOUT 4-bit quantization onto a single CUDA device.

    Why no device_map="auto":
      With device_map="auto" the model is split across 3 GPUs.  The outer
      model.generate(pixel_values=..., input_ids=...) does an in-place scatter
      of visual embeddings into input_embeds.  That scatter involves tensors
      on different CUDA devices → CUDA index-out-of-bounds assertion.
      Loading with model.cuda() (single device) avoids all cross-device issues.
      This is exactly what predict_internvl2_backup.py uses and is proven to work.

    Memory: InternVL2-8B in bfloat16 requires ~16 GB VRAM.
    """
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.eval()

    # Required for model.generate(pixel_values=..., input_ids=...) to work:
    # InternVL2's custom generate() uses this ID to find <IMG_CONTEXT> positions
    # in input_ids and replace their embeddings with visual features.
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    print(f"[internvl2] num_image_token={model.num_image_token} "
          f"img_context_token_id={model.img_context_token_id}")

    return model, tokenizer


def internvl2_confidence_fn(model, tokenizer, countries, target_country,
                             first_token_ids=None):
    """
    Confidence = softmax prob of target_country's first token from out.scores[0][0].

    Mirrors predict_internvl2_backup.py EXACTLY:
      - Full bfloat16 model on single CUDA device (no device_map="auto")
      - model.img_context_token_id set before calling generate
      - Prompt: <|im_start|>user\\n<img><IMG_CONTEXT>×N</img>\\n{question}<|im_end|>
                <|im_start|>assistant\\nCountry:
      - pixel_values on "cuda", input_ids on "cuda"
      - model.generate(pixel_values=pv, input_ids=input_ids, ..., output_scores=True)
      - first_token_ids: Python list (device-safe indexing of GPU logit tensor)
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    _transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    country_list_str = ", ".join(countries)

    # Python list — device-safe indexing regardless of which CUDA device logits are on
    if first_token_ids is None:
        first_token_ids = [
            tokenizer.encode(c, add_special_tokens=False)[0]
            for c in countries
        ]
    target_idx = countries.index(target_country)

    # Build prompt once per confidence fn — identical to predict_internvl2_backup.py
    NUM_PATCHES = 1
    N_IMG       = model.num_image_token * NUM_PATCHES   # 256

    img_block = (
        "<img>"
        + "<IMG_CONTEXT>" * N_IMG
        + "</img>"
    )
    question = (
        "Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}"
    )
    prompt = (
        f"<|im_start|>user\n{img_block}\n{question}<|im_end|>"
        f"<|im_start|>assistant\nCountry:"
    )
    # add_special_tokens=False: special tokens already in the string
    input_ids_template = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids   # (1, L) — contains exactly N_IMG occurrences of img_context_token_id

    def fn(image: Image.Image) -> float:
        pv        = _transform(image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        input_ids = input_ids_template.to("cuda")

        with torch.no_grad():
            # model.generate (outer InternVLChatModel) handles the scatter internally:
            #   1. extract_feature(pv) → vit_embeds
            #   2. embed input_ids → input_embeds
            #   3. input_embeds[selected] = vit_embeds  (scatter at IMG_CONTEXT positions)
            #   4. language_model.generate(inputs_embeds=merged, ...)
            out = model.generate(
                pixel_values=pv,
                input_ids=input_ids,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        logits      = out.scores[0][0]           # (vocab_size,) first generated token
        cand_logits = logits[first_token_ids]    # Python list index — device-safe
        probs       = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ===========================================================================
# Model registry
# ===========================================================================

MODEL_LOADERS = {
    "clip":      load_clip_model,
    "llava":     load_llava_model,
    "qwen2vl":   load_qwen2vl_model,
    "internvl2": load_internvl2_model,
}

PRED_CSVS = {
    "clip":      os.path.join(PROJECT_ROOT, "results", "clip_predictions.csv"),
    "llava":     os.path.join(PROJECT_ROOT, "results", "llava_predictions.csv"),
    "qwen2vl":   os.path.join(PROJECT_ROOT, "results", "qwen2vl_predictions.csv"),
    "internvl2": os.path.join(PROJECT_ROOT, "results", "internvl2_predictions.csv"),
}


# ===========================================================================
# Runner
# ===========================================================================

def run_occlusion(model_name: str, fill: str = "mean", grid_size: int = GRID_SIZE):
    os.makedirs(OCCLUSION_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {model_name} | fill: {fill} | grid: {grid_size}×{grid_size}")

    pred_csv = PRED_CSVS[model_name]
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"{pred_csv} not found — run prediction script first.")
    preds    = pd.read_csv(pred_csv).set_index("u_id")

    sample_ids  = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    sample_meta = pd.read_csv(SAMPLE_META_CSV)
    countries   = sorted(sample_meta["country"].unique().tolist())

    ds     = load_from_disk(DATASET_PATH)["test"]
    subset = ds.filter(lambda x: x["u_id"] in set(sample_ids))

    # ── Load model ────────────────────────────────────────────────────────────
    if model_name == "clip":
        model_obj, preprocess, tokenizer = load_clip_model(device)
    elif model_name == "llava":
        model_obj, preprocess = load_llava_model()
        tokenizer = None
    elif model_name == "qwen2vl":
        model_obj, preprocess = load_qwen2vl_model()
        tokenizer = None
    else:   # internvl2
        model_obj, tokenizer = load_internvl2_model()
        preprocess = None

    # ── Pre-compute first token IDs once as a Python list ────────────────────
    # Python list (not tensor) is always device-safe when indexing GPU logits.
    # This mirrors the first_token_confidence() helper in each predict_*.py.
    cached_first_token_ids = None
    if model_name in ("llava", "qwen2vl"):
        cached_first_token_ids = [
            preprocess.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in countries
        ]
    elif model_name == "internvl2":
        cached_first_token_ids = [
            tokenizer.encode(c, add_special_tokens=False)[0]
            for c in countries
        ]

    # ── Filename suffix for non-default grid sizes ────────────────────────────
    grid_suffix = f"_{grid_size}x{grid_size}" if grid_size != GRID_SIZE else ""

    processed = skipped = 0

    for row in subset:
        u_id     = row["u_id"]
        out_path = os.path.join(OCCLUSION_DIR,
                                f"{u_id}_{model_name}_{fill}{grid_suffix}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        if u_id not in preds.index:
            print(f"[warn] {u_id} not in predictions CSV — skipping")
            continue

        target_country = preds.loc[u_id, "pred_country"]

        if target_country not in countries:
            print(f"[skip] {u_id} — pred={target_country!r} not in countries list")
            skipped += 1
            continue

        image = row["image"]

        # Build confidence fn for this target country
        if model_name == "clip":
            conf_fn = clip_confidence_fn(
                model_obj, preprocess, tokenizer, countries, target_country, device
            )
        elif model_name == "llava":
            conf_fn = llava_confidence_fn(
                model_obj, preprocess, countries, target_country,
                first_token_ids=cached_first_token_ids,
            )
        elif model_name == "qwen2vl":
            conf_fn = qwen2vl_confidence_fn(
                model_obj, preprocess, countries, target_country,
                first_token_ids=cached_first_token_ids,
            )
        else:
            conf_fn = internvl2_confidence_fn(
                model_obj, tokenizer, countries, target_country,
                first_token_ids=cached_first_token_ids,
            )

        scores = occlusion_sensitivity(conf_fn, image, grid_size=grid_size, fill=fill)
        np.save(out_path, scores)
        processed += 1

        top_i, top_j = np.unravel_index(scores.argmax(), scores.shape)
        print(
            f"[{processed}] {u_id} | target={target_country} | "
            f"max_score={scores.max():.4f} @ patch({top_i},{top_j}) | saved"
        )

        # Periodic VRAM cleanup for large models
        if model_name != "clip" and processed % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final cleanup
    if model_name != "clip":
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished.  Saved: {processed}  |  Skipped: {skipped}")
    print(f"Output dir: {OCCLUSION_DIR}/")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3a — Occlusion Sensitivity")
    parser.add_argument(
        "--model", choices=list(MODEL_LOADERS.keys()), required=True,
        help="Model to run occlusion for.",
    )
    parser.add_argument(
        "--fill", choices=["mean", "black", "noise"], default="mean",
        help="Patch replacement strategy (default: mean).",
    )
    parser.add_argument(
        "--grid", type=int, default=GRID_SIZE,
        help=f"Grid size N for N×N patches (default: {GRID_SIZE}). Use 7 for LLMs.",
    )
    args = parser.parse_args()
    run_occlusion(args.model, fill=args.fill, grid_size=args.grid)
