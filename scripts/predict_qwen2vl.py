"""
Milestone 2c — Qwen2-VL-7B Country Prediction
Model: Qwen/Qwen2-VL-7B-Instruct  (4-bit, <8 GB VRAM)
Output: results/qwen2vl_predictions.csv
Schema: u_id, true_country, facet, pred_country, confidence, correct

Requires: pip install qwen-vl-utils

import os
import re 
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
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

os.makedirs(RESULTS_DIR, exist_ok=True)

IDS_PATH = os.path.join(
    RESULTS_DIR,
    "sample_ids.csv"
)

META_PATH = os.path.join(
    RESULTS_DIR,
    "sample_metadata.csv"
)

OUT_PATH = os.path.join(
    RESULTS_DI  R,
    "qwen2vl_predictions.csv"
)

print("\n=== PATH CHECK ===")
print("DATASET :", DATASET_PATH)
print("IDS     :", IDS_PATH)
print("META    :", META_PATH)
print("OUTPUT  :", OUT_PATH)

for p in [IDS_PATH, META_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# ── load sample ────────────────────────────────────────────────────────────────
sample_ids = set(pd.read_csv(IDS_PATH)["u_id"].tolist())
meta       = pd.read_csv(META_PATH)
countries  = sorted(meta["country"].unique().tolist())
country_list_str = ", ".join(countries)

print(f"Sample: {len(sample_ids)} images | Countries: {len(countries)}")

# ── load model ─────────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded")
print("CUDA available:", torch.cuda.is_available())

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "CUDA device count:",
    torch.cuda.device_count()
)
# ── helpers ────────────────────────────────────────────────────────────────────
def parse_country(text: str, countries: list) -> str | None:
    text_lower = text.lower()
    for c in countries:
        if c.lower() in text_lower:
            return c
    return None


def first_token_confidence(scores_step0, tokenizer, countries: list) -> dict:
    first_ids = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in countries
    ]
    logits = scores_step0[first_ids]
    probs  = torch.softmax(logits, dim=0).cpu().float().numpy()
    return {c: float(p) for c, p in zip(countries, probs)}


def build_messages(image, country_list_str: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Which country is most strongly represented in this image?\n"
                        f"Choose exactly one from this list: {country_list_str}\n\n"
                        "Country:"
                    ),
                },
            ],
        }
    ]


# ── filter dataset ─────────────────────────────────────────────────────────────
ds     = load_from_disk(DATASET_PATH)["test"]
sample = ds.filter(lambda x: x["u_id"] in sample_ids)
print(f"Filtered dataset to {len(sample)} rows")

# ── predict ────────────────────────────────────────────────────────────────────
results      = []
parse_errors = 0

for i, row in enumerate(sample):
    image = row["image"].convert("RGB")

    messages = build_messages(image, country_list_str)
    text     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids  = out.sequences[0][inputs["input_ids"].shape[-1]:]
    print("generated_ids:", generated_ids)
    print("type:", type(generated_ids))
    print("device:",generated_ids.device)
    generated_text=processor.tokenizer.decode(
      generated_ids.cpu().tolist(),
      skip_special_tokens=True
    )
    pred_country   = parse_country(generated_text, countries)

    if pred_country is None:
        parse_errors += 1
        pred_country = "UNKNOWN"

    conf_dict  = first_token_confidence(out.scores[0][0], processor.tokenizer, countries)
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

Milestone 2c — Qwen2-VL-7B Country Prediction
Model: Qwen/Qwen2-VL-7B-Instruct
Output: results/qwen2vl_predictions.csv
Schema: u_id, true_country, facet, pred_country, confidence, correct

Requires: pip install qwen-vl-utils

Fix notes (v2):
  - Removed torch_dtype=float16 when using load_in_4bit. bitsandbytes uses
    bfloat16 for compute internally; mixing float16 caused garbage token output.
  - Replaced model.device with "cuda" — device_map="auto" across multiple GPUs
    makes model.device return "meta", putting inputs on the wrong device.
  - Changed videos=[] to videos=None to avoid silent processor failures.
  - Added min_pixels/max_pixels to processor to cap dynamic tiling.
"""

import os
import re
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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

os.makedirs(RESULTS_DIR, exist_ok=True)

IDS_PATH = os.path.join(
    RESULTS_DIR,
    "sample_ids.csv"
)

META_PATH = os.path.join(
    RESULTS_DIR,
    "sample_metadata.csv"
)

OUT_PATH = os.path.join(
    RESULTS_DIR,
    "qwen2vl_predictions.csv"
)

print("\n=== PATH CHECK ===")
print("DATASET :", DATASET_PATH)
print("IDS     :", IDS_PATH)
print("META    :", META_PATH)
print("OUTPUT  :", OUT_PATH)

for p in [IDS_PATH, META_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found — run sample.py first")

# ── load sample ────────────────────────────────────────────────────────────────
sample_ids = set(pd.read_csv(IDS_PATH)["u_id"].tolist())
meta       = pd.read_csv(META_PATH)
countries  = sorted(meta["country"].unique().tolist())
country_list_str = ", ".join(countries)

print(f"Sample: {len(sample_ids)} images | Countries: {len(countries)}")

# ── load model ─────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# min/max_pixels cap dynamic tiling so visual-token count stays predictable
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # non-quantized layers (vision encoder, embeddings)
)
model=model.cuda()
model.eval()
print(f"Model loaded")

# ── helpers ────────────────────────────────────────────────────────────────────
"""def parse_country(text: str, countries: list) -> str | None:
    text_lower = text.lower()
    for c in countries:
        if re.search(r'\b' + re.escape(c.lower()) + r'\b', text_lower):
            return c
    return None

def parse_country(text, countries):
    text_lower = text.lower()

    for country in countries:
        if country.lower() in text_lower:
            return country

    return None"""

def parse_country(text, countries):
    text_lower = text.lower()

    matches = []

    for country in countries:
        if country.lower() in text_lower:
            matches.append(country)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return matches[0]

    return None

def first_token_confidence(scores_step0, tokenizer, countries: list) -> dict:
    first_ids = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in countries
    ]
    logits = scores_step0[first_ids]
    probs  = torch.softmax(logits, dim=0).cpu().float().numpy()
    return {c: float(p) for c, p in zip(countries, probs)}


def build_messages(image, country_list_str: str) -> list:
    return [
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


# ── filter dataset ─────────────────────────────────────────────────────────────
ds     = load_from_disk(DATASET_PATH)["test"]
sample = ds.filter(lambda x: x["u_id"] in sample_ids)
print(f"Filtered dataset to {len(sample)} rows")

# ── predict ────────────────────────────────────────────────────────────────────
results      = []
parse_errors = 0

for i, row in enumerate(sample):
    image = row["image"].convert("RGB")

    messages = build_messages(image, country_list_str)
    text     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Pre-fill the assistant turn so out.scores[0] captures the first country-name
    # token, not an unconstrained assistant opener ("The", "It", etc.)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to("cuda")for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # trim input tokens; use batch_decode for consistent special-token handling
    input_len=inputs["input_ids"].shape[-1]
    generated_ids = out.sequences[0][input_len:]

    generated_text = processor.tokenizer.decode(
    generated_ids.cpu().tolist(), skip_special_tokens=True).strip()

    pred_country = parse_country(generated_text, countries)

    if pred_country is None:
        parse_errors += 1
        print(f"  [parse error #{parse_errors}] raw output: {repr(generated_text)}")
        pred_country = "UNKNOWN"

    conf_dict  = first_token_confidence(out.scores[0][0], processor.tokenizer, countries)
    confidence = conf_dict.get(pred_country, 0.0)

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
