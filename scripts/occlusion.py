"""
Milestone 3a — Core Occlusion Sensitivity Pipeline
Research Proposal Section 6.3 (Primary Method)

For every image × model pair, produces a 14×14 patch attribution heatmap.
Positive value at [i,j] = masking that patch hurts confidence → patch matters.

Output: results/occlusion/{u_id}_{model}_{fill}.npy

Usage:
    python occlusion.py --model clip
    python occlusion.py --model llava
    python occlusion.py --model qwen2vl
    python occlusion.py --model internvl2
    python occlusion.py --model clip --fill black
    python occlusion.py --model clip --grid 7
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
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "CulturalVQA"
)

SAMPLE_IDS_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "sample_ids.csv"
)

SAMPLE_META_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "sample_metadata.csv"
)

OCCLUSION_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "occlusion"
)

GRID_SIZE = 14

# ---------------------------------------------------------------------------
# Core occlusion function — model-agnostic
# Proposal §6.3: replace patch with fill value, re-run model, measure Δconfidence
# Proposal §9:   report results for grey-fill AND noise-fill (stability check)
# ---------------------------------------------------------------------------

def occlusion_sensitivity(
    model_fn,
    image: Image.Image,
    grid_size: int = GRID_SIZE,
    fill: str = "mean",          # "mean" | "black" | "noise"
) -> np.ndarray:
    """
    model_fn(image: PIL.Image) -> float  — confidence for the target country.

    Returns (grid_size, grid_size) float32 ndarray.
    Positive = masking that patch drops confidence (patch is important).
    """
    W, H = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    img_arr = np.array(image)

    # Compute fill colour once per image
    if fill == "mean":
        fill_value = img_arr.mean(axis=(0, 1)).astype(np.uint8)
    elif fill == "black":
        fill_value = np.zeros(img_arr.shape[2], dtype=np.uint8)
    elif fill == "noise":
        rng = np.random.default_rng(seed=42)
        # Pre-generate a noise array of full image size; crop per patch below
        noise_arr = rng.integers(0, 256, img_arr.shape, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown fill type: {fill!r}. Choose mean | black | noise")

    baseline_conf = model_fn(image)
    scores = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            masked = img_arr.copy()
            x0, y0 = j * patch_w, i * patch_h
            if fill == "noise":
                masked[y0:y0 + patch_h, x0:x0 + patch_w] = \
                    noise_arr[y0:y0 + patch_h, x0:x0 + patch_w]
            else:
                masked[y0:y0 + patch_h, x0:x0 + patch_w] = fill_value
            conf = model_fn(Image.fromarray(masked))
            scores[i, j] = baseline_conf - conf

    return scores


# ---------------------------------------------------------------------------
# CLIP confidence wrapper
# ---------------------------------------------------------------------------

def load_clip_model(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer


def clip_confidence_fn(model, preprocess, tokenizer, countries, target_country, device):
    """Returns a closure: PIL.Image -> float (similarity score for target_country)."""
    text_tokens = tokenizer([f"a photo from {c}" for c in countries]).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        img_t = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_t)
            feat /= feat.norm(dim=-1, keepdim=True)
            sims = (feat @ text_feats.T).squeeze(0).cpu().numpy()
            probs=torch.softmax(torch.tensor(sims)*100.0,dim=0).cpu().numpy()
        return float(probs[target_idx])

    return fn


# ---------------------------------------------------------------------------
# LLaVA-1.6 confidence wrapper
# ---------------------------------------------------------------------------

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


def llava_confidence_fn(model, processor, countries, target_country):
    """
    Extracts confidence as softmax probability over candidate country first-tokens.
    Proposal §6.1: use logit scores from model.generate().
    """
    country_list_str = ", ".join(countries)
    prompt = (
        f"[INST] <image>\n"
        f"Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}\n\n"
        f"Country: [/INST]"
    )

    # Tokenise the first token of every candidate country name once
    first_tokens = []
    for c in countries:
        toks = processor.tokenizer.encode(c, add_special_tokens=False)
        first_tokens.append(toks[0])
    first_tokens_t = torch.tensor(first_tokens)
    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        inputs = processor(prompt, image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits = out.scores[0][0]                           # (vocab_size,)
        cand_logits = logits[first_tokens_t]
        probs = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ---------------------------------------------------------------------------
# Qwen2-VL confidence wrapper
# ---------------------------------------------------------------------------

def load_qwen2vl_model():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    return model, processor


def qwen2vl_confidence_fn(model, processor, countries, target_country):
    country_list_str = ", ".join(countries)

    first_tokens = []
    for c in countries:
        toks = processor.tokenizer.encode(c, add_special_tokens=False)
        first_tokens.append(toks[0])
    first_tokens_t = torch.tensor(first_tokens)
    target_idx = countries.index(target_country)

    def fn(image: Image.Image) -> float:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Which country is most strongly represented in this image?\n"
                            f"Choose exactly one from: {country_list_str}\n\nCountry:"
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits = out.scores[0][0]
        cand_logits = logits[first_tokens_t]
        probs = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ---------------------------------------------------------------------------
# InternVL2-8B confidence wrapper
# ---------------------------------------------------------------------------

def load_internvl2_model():
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-8B", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def internvl2_confidence_fn(model, tokenizer, countries, target_country):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    country_list_str = ", ".join(countries)
    first_tokens = []
    for c in countries:
        toks = tokenizer.encode(c, add_special_tokens=False)
        first_tokens.append(toks[0])
    first_tokens_t = torch.tensor(first_tokens)
    target_idx = countries.index(target_country)

    question = (
        f"<image>\nWhich country is most strongly represented in this image?\n"
        f"Choose exactly one from: {country_list_str}\n\nCountry:"
    )

    def fn(image: Image.Image) -> float:
        pv = transform(image.convert("RGB")).unsqueeze(0).to(
            model.device, dtype=torch.float16
        )
        gen_cfg = dict(
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        with torch.no_grad():
            out = model.generate(
                pixel_values=pv,
                question=question,
                tokenizer=tokenizer,
                generation_config=gen_cfg,
            )
        logits = out.scores[0][0]
        cand_logits = logits[first_tokens_t]
        probs = torch.softmax(cand_logits.float(), dim=0)
        return float(probs[target_idx].item())

    return fn


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_LOADERS = {
    "clip":      load_clip_model,
    "llava":     load_llava_model,
    "qwen2vl":   load_qwen2vl_model,
    "internvl2": load_internvl2_model,
}

CONFIDENCE_FNS = {
    "clip":      clip_confidence_fn,
    "llava":     llava_confidence_fn,
    "qwen2vl":   qwen2vl_confidence_fn,
    "internvl2": internvl2_confidence_fn,
}

PRED_CSVS = {
    "clip":      os.path.join("results", "clip_predictions.csv"),
    "llava":     os.path.join("results", "llava_predictions.csv"),
    "qwen2vl":   os.path.join("results", "qwen2vl_predictions.csv"),
    "internvl2": os.path.join("results", "internvl2_predictions.csv"),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_occlusion(model_name: str, fill: str = "mean", grid_size: int = GRID_SIZE):
    """
    Run occlusion sensitivity for all sample images for one model.

    Loads predicted country from results/{model}_predictions.csv so that
    confidence is measured for the model's own prediction (not ground truth).
    Saves: results/occlusion/{u_id}_{model}_{fill}.npy
    """
    os.makedirs(OCCLUSION_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {model_name} | fill: {fill} | grid: {grid_size}x{grid_size}")

    # Load predictions from Milestone 2 CSV
    pred_csv = PRED_CSVS[model_name]
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(
            f"{pred_csv} not found. Run Milestone 2 prediction scripts first."
        )
    preds = pd.read_csv(pred_csv).set_index("u_id")

    sample_ids   = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    sample_meta  = pd.read_csv(SAMPLE_META_CSV)
    countries    = sorted(sample_meta["country"].unique().tolist())

    ds      = load_from_disk(DATASET_PATH)["test"]
    id_set  = set(sample_ids)
    subset  = ds.filter(lambda x: x["u_id"] in id_set)

    # Load model once
    if model_name == "clip":
        model_obj, preprocess, tokenizer = load_clip_model(device)
    elif model_name == "llava":
        model_obj, preprocess = load_llava_model()   # preprocess == processor
        tokenizer = None
    elif model_name == "qwen2vl":
        model_obj, preprocess = load_qwen2vl_model()
        tokenizer = None
    else:
        model_obj, preprocess = load_internvl2_model()  # preprocess == tokenizer here
        tokenizer = preprocess
        preprocess = None

    processed = 0
    skipped   = 0

    for row in subset:
        u_id = row["u_id"]
        out_path = os.path.join(OCCLUSION_DIR, f"{u_id}_{model_name}_{fill}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        if u_id not in preds.index:
            print(f"[warn] {u_id} not in predictions CSV — skipping")
            continue

        target_country = preds.loc[u_id, "pred_country"]
        image = row["image"]

        # Build confidence fn for this target country
        if model_name == "clip":
            conf_fn = clip_confidence_fn(
                model_obj, preprocess, tokenizer, countries, target_country, device
            )
        elif model_name == "llava":
            conf_fn = llava_confidence_fn(model_obj, preprocess, countries, target_country)
        elif model_name == "qwen2vl":
            conf_fn = qwen2vl_confidence_fn(model_obj, preprocess, countries, target_country)
        else:
            conf_fn = internvl2_confidence_fn(model_obj, tokenizer, countries, target_country)

        scores = occlusion_sensitivity(conf_fn, image, grid_size=grid_size, fill=fill)
        np.save(out_path, scores)
        processed += 1

        top_i, top_j = np.unravel_index(scores.argmax(), scores.shape)
        print(
            f"[{processed}] {u_id} | target={target_country} | "
            f"max_score={scores.max():.4f} @ patch({top_i},{top_j}) | saved"
        )

    # Free VRAM
    if model_name != "clip":
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished. Saved: {processed}  |  Skipped (already exist): {skipped}")
    print(f"Output dir: {OCCLUSION_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3a — Occlusion Sensitivity")
    parser.add_argument(
        "--model",
        choices=list(MODEL_LOADERS.keys()),
        required=True,
        help="Model to run occlusion for.",
    )
    parser.add_argument(
        "--fill",
        choices=["mean", "black", "noise"],
        default="mean",
        help="Patch replacement strategy (default: mean).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=GRID_SIZE,
        help=f"Grid size N (NxN patches). Proposal suggests 7 or 14 (default: {GRID_SIZE}).",
    )
    args = parser.parse_args()
    run_occlusion(args.model, fill=args.fill, grid_size=args.grid)
