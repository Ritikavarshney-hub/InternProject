"""
Milestone 2a — CLIP Baseline Country Prediction

Model: ViT-L-14 (OpenAI weights via open_clip)

Output:
    results/clip_predictions.csv

Schema:
    u_id,
    true_country,
    facet,
    pred_country,
    confidence,
    correct
"""

import os
import torch
import open_clip
import pandas as pd
from datasets import load_from_disk

# =============================================================================
# PATHS
# =============================================================================

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

OUT_PATH = os.path.join(
    RESULTS_DIR,
    "clip_predictions.csv"
)

print("\n=== PATH CHECK ===")
print("DATASET :", DATASET_PATH)
print("IDS     :", IDS_PATH)
print("META    :", META_PATH)
print("OUTPUT  :", OUT_PATH)

for p in [IDS_PATH, META_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# =============================================================================
# LOAD SAMPLE METADATA
# =============================================================================

sample_ids = set(
    pd.read_csv(IDS_PATH)["u_id"].tolist()
)

meta = pd.read_csv(META_PATH)

countries = sorted(
    meta["country"].unique().tolist()
)

print(
    f"\nSample contains {len(sample_ids)} images "
    f"across {len(countries)} countries"
)

# =============================================================================
# DEVICE
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# =============================================================================
# LOAD MODEL
# =============================================================================

print("\nLoading OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(
    "ViT-L-14"
)

# =============================================================================
# TEXT PROMPTS
# =============================================================================

# =============================================================================
# TEXT PROMPTS
# =============================================================================

text_prompts = [
    f"an image representing the culture of {country}"
    for country in countries
]

print("\nText prompts:")
for p in text_prompts:
    print("  ", p)

text_tokens = tokenizer(text_prompts).to(device)

with torch.no_grad():

    text_features = model.encode_text(text_tokens)

    text_features = (
        text_features
        / text_features.norm(dim=-1, keepdim=True)
    )

print("Text embeddings computed")

# =============================================================================
# LOAD DATASET
# =============================================================================

print("\nLoading CulturalVQA dataset...")

dataset = load_from_disk(
    DATASET_PATH
)["test"]

dataset = dataset.filter(
    lambda x: x["u_id"] in sample_ids
)

print(
    f"Filtered dataset size: {len(dataset)}"
)

# =============================================================================
# PREDICTION LOOP
# =============================================================================

results = []

for idx, row in enumerate(dataset):

    image = preprocess(
        row["image"].convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        image_features = model.encode_image(
            image
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        logits = 100.0 * ( image_features @ text_features.T)

        probs = torch.softmax(logits, dim=-1)

        probs = (probs.squeeze(0).cpu().numpy())

        pred_idx = probs.argmax()

        pred_country = countries[pred_idx]

        confidence = float(probs[pred_idx])

        top3_idx = probs.argsort()[-3:][::-1]

        results.append({ "u_id": row["u_id"], "true_country": row["country"], "facet": row["facet"], "pred_country": pred_country, "confidence": confidence, "top1": countries[top3_idx[0]], "top2": countries[top3_idx[1]], "top3": countries[top3_idx[2]], "correct": row["country"] == pred_country})

    if (idx + 1) % 25 == 0:

        print(
            f"{idx + 1}/{len(dataset)} processed"
        )

# =============================================================================
# SAVE RESULTS
# =============================================================================

df = pd.DataFrame(results)

df.to_csv(
    OUT_PATH,
    index=False
)

# =============================================================================
# REPORT
# =============================================================================

print("\n========================")
print("RESULTS")
print("========================")

print(
    f"Overall Accuracy: "
    f"{df['correct'].mean():.4f}"
)

print("\nPer Country Accuracy")

print(
    df.groupby(
        "true_country"
    )["correct"]
    .mean()
    .sort_values()
)

print("\nPer Facet Accuracy")

print(
    df.groupby(
        "facet"
    )["correct"]
    .mean()
    .sort_values()
)

print(f"\nSaved to:\n{OUT_PATH}")
