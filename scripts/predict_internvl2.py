"""
Milestone 2c - InternVL2-8B Country Prediction
Model: OpenGVLab/InternVL2-8B  (4-bit, <8 GB VRAM)
Output: results/internvl2_predictions.csv
Schema: u_id, true_country, facet, pred_country, confidence, correct

InternVL2 uses its own image preprocessing pipeline (dynamic tiling).
This script uses the standard HuggingFace generate() path to keep
confidence extraction consistent with the other model scripts.


import os
import re
import torch
import pandas as pd
import torchvision.transforms as T
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "CulturalVQA"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

IDS_PATH = os.path.join(
    RESULTS_DIR,
    "sample_ids.csv"
)

META_PATH = os.path.join(
    RESULTS_DIR,
    "sample_metadata.csv"
)
OUT_PATH = os.path.join(RESULTS_DIR, "internvl2_predictions.csv")

print("\n=== PATH CHECK ===")
print("DATASET :", DATASET_PATH)
print("IDS     :", IDS_PATH)
print("META    :", META_PATH)
print("OUTPUT  :", OUT_PATH)
# ── load sample ────────────────────────────────────────────────────────────────
sample_ids = set(pd.read_csv(IDS_PATH)["u_id"].tolist())
meta       = pd.read_csv(META_PATH)
countries  = sorted(meta["country"].unique().tolist())
country_list_str = ", ".join(countries)

print(f"Sample: {len(sample_ids)} images | Countries: {len(countries)}")

# ── load model ─────────────────────────────────────────────────────────────────
MODEL_ID  = "OpenGVLab/InternVL2-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # non-quantized layers (vision encoder, embeddings)
    trust_remote_code=True,
)
model=model.cuda()
model.eval()
print(f"Model loaded | num_image_token: {model.num_image_token}")

# ── InternVL2 image preprocessing ─────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

_transform = build_transform(448)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Returns (1, 3, 448, 448) bfloat16 tensor - single tile, no dynamic tiling."""
    return _transform(image).unsqueeze(0).to(torch.bfloat16)

# ── helpers ────────────────────────────────────────────────────────────────────
def parse_country(text: str, countries: list) -> str | None:
    text_lower = text.lower()
    for c in countries:
        if re.search(r'\b' + re.escape(c.lower()) + r'\b', text_lower):
            return c
    return None
def parse_country(text: str, countries: list) -> str | None:

    text_lower = text.lower().strip()

    aliases = {
        "united states": "USA",
        "united states of america": "USA",
        "america": "USA",
    }

    for alias, canonical in aliases.items():
        if alias in text_lower:
            return canonical

    for country in countries:
        if country.lower() in text_lower:
            return country

    return None


# InternVL2 uses its own image token format - NOT "<image>" (LLaVA convention).
# model.generate() requires the input_ids to contain exactly
#   (model.num_image_token * num_patches) <IMG_CONTEXT> tokens at the image position.
# model.chat() handles this internally; we replicate it here to get output_scores.
def build_question(country_list_str):

    return (
        "Answer with exactly ONE country name.\n\n"
        f"Countries: {country_list_str}\n\n"
        "Choose ONLY from the list above.\n"
        "Do not explain.\n"
        "Do not justify.\n"
        "Output only the country name."
    )


# ── filter dataset ─────────────────────────────────────────────────────────────
ds     = load_from_disk(DATASET_PATH)["test"]
sample = ds.filter(lambda x: x["u_id"] in sample_ids)
print(f"Filtered dataset to {len(sample)} rows")

# ── predict ────────────────────────────────────────────────────────────────────
results      = []
parse_errors = 0

# num_patches=1 because preprocess_image produces a single 448×448 tile

# add_special_tokens=False: special tokens (<|im_start|> etc.) are already in the string

for i, row in enumerate(sample):
    image = row["image"].convert("RGB")

    pixel_values = preprocess_image(image).cuda()

    question = build_question(country_list_str)

    response = model.chat(tokenizer,pixel_values,question, generation_config=dict( max_new_tokens=5,do_sample=False))

    generated_text = response.strip()

    pred_country = parse_country(generated_text, countries)

    if pred_country is None:
        parse_errors += 1
        print(f"  [parse error #{parse_errors}] raw output: {repr(generated_text)}")
        pred_country = "UNKNOWN"

    confidence = -1

    results.append({
        "u_id":         row["u_id"],
        "true_country": row["country"],
        "facet":        row["facet"],
        "pred_country": pred_country,
        "confidence":   confidence,
        "correct":      row["country"] == pred_country,
    })

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(sample)} done  |  parse errors so far: {parse_errors}")

# ── save ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved → {OUT_PATH}")
print(f"Parse errors     : {parse_errors}/{len(results)}")
print(f"Overall accuracy : {df['correct'].mean():.3f}")
print("\nPer-country accuracy:")
print(df.groupby("true_country")["correct"].mean().sort_values().to_string())
print("\nPer-facet accuracy:")
print(df.groupby("facet")["correct"].mean().sort_values().to_string())
"""

"""
Milestone 2c - InternVL2-8B Country Prediction
Model: OpenGVLab/InternVL2-8B  (4-bit, <8 GB VRAM)
Output: results/internvl2_predictions.csv
Schema: u_id, true_country, facet, pred_country, confidence, correct

InternVL2 uses its own image preprocessing pipeline (dynamic tiling).
This script uses the standard HuggingFace generate() path to keep
confidence extraction consistent with the other model scripts.
"""

import os
import re
import torch
import pandas as pd
import torchvision.transforms as T
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "culturalVQA")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IDS_PATH  = os.path.join(BASE_DIR, "sample_ids.csv")
META_PATH = os.path.join(BASE_DIR, "sample_metadata.csv")
OUT_PATH  = os.path.join(RESULTS_DIR, "internvl2_predictions.csv")

for p in [IDS_PATH, META_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found - run sample.py first")

# ── load sample ────────────────────────────────────────────────────────────────
sample_ids = set(pd.read_csv(IDS_PATH)["u_id"].tolist())
meta       = pd.read_csv(META_PATH)
countries  = sorted(meta["country"].unique().tolist())
country_list_str = ", ".join(countries)

print(f"Sample: {len(sample_ids)} images | Countries: {len(countries)}")

# ── load model ─────────────────────────────────────────────────────────────────
MODEL_ID  = "OpenGVLab/InternVL2-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,   # non-quantized layers (vision encoder, embeddings)
    trust_remote_code=True,
    device_map="auto",
)
model.eval()
print(f"Model loaded | num_image_token: {model.num_image_token}")

# ── InternVL2 image preprocessing ─────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

_transform = build_transform(448)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Returns (1, 3, 448, 448) bfloat16 tensor - single tile, no dynamic tiling."""
    return _transform(image).unsqueeze(0).to(torch.bfloat16)

# ── helpers ────────────────────────────────────────────────────────────────────
def parse_country(text: str, countries: list) -> str | None:
    text_lower = text.lower()
    for c in countries:
        if re.search(r'\b' + re.escape(c.lower()) + r'\b', text_lower):
            return c
    return None


def first_token_confidence(scores_step0, tokenizer, countries: list) -> dict:
    first_ids = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in countries
    ]
    logits = scores_step0[first_ids]
    probs  = torch.softmax(logits.float(), dim=0).cpu().numpy()
    return {c: float(p) for c, p in zip(countries, probs)}


# InternVL2 uses its own image token format - NOT "<image>" (LLaVA convention).
# model.generate() requires the input_ids to contain exactly
#   (model.num_image_token * num_patches) <IMG_CONTEXT> tokens at the image position.
# model.chat() handles this internally; we replicate it here to get output_scores.
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

def build_internvl2_prompt(num_image_token: int, num_patches: int,
                            country_list_str: str) -> str:
    img_block = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * (num_image_token * num_patches)
        + IMG_END_TOKEN
    )
    question = (
        "Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}"
    )
    # InternVL2 ChatML format; "Country:" pre-fills the assistant turn
    return (
        f"<|im_start|>user\n{img_block}\n{question}<|im_end|>"
        f"<|im_start|>assistant\nCountry:"
    )

GENERATION_CONFIG = dict(
    max_new_tokens=20,
    do_sample=False,
    return_dict_in_generate=True,
    output_scores=True,
)

# ── filter dataset ─────────────────────────────────────────────────────────────
ds     = load_from_disk(DATASET_PATH)["test"]
sample = ds.filter(lambda x: x["u_id"] in sample_ids)
print(f"Filtered dataset to {len(sample)} rows")

# ── predict ────────────────────────────────────────────────────────────────────
results      = []
parse_errors = 0

# num_patches=1 because preprocess_image produces a single 448×448 tile
NUM_PATCHES = 1
prompt = build_internvl2_prompt(model.num_image_token, NUM_PATCHES, country_list_str)
# add_special_tokens=False: special tokens (<|im_start|> etc.) are already in the string
input_ids_template = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=False
).input_ids

for i, row in enumerate(sample):
    image        = row["image"].convert("RGB")
    pixel_values = preprocess_image(image).to("cuda")
    input_ids    = input_ids_template.to("cuda")

    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            **GENERATION_CONFIG,
        )

    input_len      = input_ids.shape[-1]
    generated_ids  = [out.sequences[0][input_len:]]
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    pred_country = parse_country(generated_text, countries)

    if pred_country is None:
        parse_errors += 1
        print(f"  [parse error #{parse_errors}] raw output: {repr(generated_text)}")
        pred_country = "UNKNOWN"

    conf_dict  = first_token_confidence(out.scores[0][0], tokenizer, countries)
    confidence = conf_dict.get(pred_country, 0.0)

    results.append({
        "u_id":         row["u_id"],
        "true_country": row["country"],
        "facet":        row["facet"],
        "pred_country": pred_country,
        "confidence":   confidence,
        "correct":      row["country"] == pred_country,
    })

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(sample)} done  |  parse errors so far: {parse_errors}")

# ── save ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved → {OUT_PATH}")
print(f"Parse errors     : {parse_errors}/{len(results)}")
print(f"Overall accuracy : {df['correct'].mean():.3f}")
print("\nPer-country accuracy:")
print(df.groupby("true_country")["correct"].mean().sort_values().to_string())
print("\nPer-facet accuracy:")
print(df.groupby("facet")["correct"].mean().sort_values().to_string())

