"""
Milestone 2b — LLaVA-1.6 Country Prediction
Model: llava-hf/llava-v1.6-mistral-7b-hf  (4-bit, <8 GB VRAM)
Output: results/llava_predictions.csv
Schema: u_id, true_country, facet, pred_country, confidence, correct

Confidence method:
  Softmax over the first-token logits of each candidate country name.
  Captures the model's relative preference among candidates at the first
  decoding step. Limitation: countries sharing a first token (e.g.
  Canada / China → "C", India / Iran → "I") get equal raw logit scores;
  the parsed prediction is still unambiguous because the full decoded text
  is matched against the country list.
"""

import os
import re
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

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
    RESULTS_DIR,
    "llava_predictions.csv"
)
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
MODEL_ID  = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model     = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
model=model.cuda()
model.eval()
print("Model loaded")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

try:
    print("Model device:", next(model.parameters()).device)
except:
    pass

# ── prompt ─────────────────────────────────────────────────────────────────────
"""PROMPT_TEMPLATE = (
    "[INST] <image>\n"
    "Which country is most strongly represented in this image?\n"
    f"Choose exactly one from this list: {country_list_str}\n\n"
    "Country: [/INST]"
)"""

PROMPT_TEMPLATE = (
    "[INST] <image>\n"
    "You MUST answer with exactly ONE country name.\n\n"
    f"Valid countries are:\n{country_list_str}\n\n"
    "Choose ONLY from this list.\n"
    "Do not explain.\n"
    "Do not justify.\n"
    "Do not output any other text.\n"
    "Output only the country name.\n"
    "[/INST]"
)
# ── helpers ────────────────────────────────────────────────────────────────────
"""def parse_country(text: str, countries: list) -> str | None:
    Return the first country name found in the generated text.
    text_lower = text.lower()
    for c in countries:
        if c.lower() in text_lower:
            return c
    return None"""

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

    matches = []

    for country in countries:
        if country.lower() in text_lower:
            matches.append(country)

    if len(matches) > 0:
        return matches[0]

    return None


def first_token_confidence(scores_step0, tokenizer, countries: list) -> dict:
    """
    scores_step0: logit tensor of shape (vocab_size,) from the first decode step.
    Returns {country: probability} using softmax over candidate first tokens.
    """
    first_ids = [
        tokenizer.encode(c, add_special_tokens=False)[0]
        for c in countries
    ]
    logits = scores_step0[first_ids]
    probs  = torch.softmax(logits, dim=0).cpu().float().numpy()
    return {c: float(p) for c, p in zip(countries, probs)}


# ── filter dataset ─────────────────────────────────────────────────────────────
ds     = load_from_disk(DATASET_PATH)["test"]
sample = ds.filter(lambda x: x["u_id"] in sample_ids)
print(f"Filtered dataset to {len(sample)} rows")

# ── predict ────────────────────────────────────────────────────────────────────
results      = []
parse_errors = 0

for i, row in enumerate(sample):
    image = row["image"].convert("RGB")

    inputs = processor(
    text=PROMPT_TEMPLATE,
    images=image,
    return_tensors="pt",
    )

    inputs = {
    k: v.to("cuda")
    for k, v in inputs.items()
    }

    with torch.no_grad():
        out = model.generate( **inputs, max_new_tokens=5,do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True,)

    # decoded prediction
    generated_ids = out.sequences[0][inputs["input_ids"].shape[-1]:]
    generated_text = processor.batch_decode(
    [generated_ids.cpu()],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)[0].strip()
    pred_country = parse_country(generated_text, countries)

    if pred_country is None:
       parse_errors += 1
       print( f"  [parse error #{parse_errors}] "
       f"raw output: {repr(generated_text)}" )
       pred_country = "UNKNOWN"

    # confidence from first-token logits
    conf_dict   = first_token_confidence(out.scores[0][0], processor.tokenizer, countries)
    confidence  = conf_dict.get(pred_country, 0.0)

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
