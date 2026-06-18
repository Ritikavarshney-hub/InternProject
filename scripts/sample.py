from datasets import load_from_disk
import pandas as pd
import os

# ======================
# Paths
# ======================

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

# ======================
# Load Dataset
# ======================

print("Loading dataset...")
print(DATASET_PATH)

ds = load_from_disk(DATASET_PATH)["test"]

df = ds.to_pandas()

print("\nColumns:")
print(df.columns.tolist())

# Remove image column
if "image" in df.columns:
    df = df.drop(columns=["image"])

print(f"\nDataset Size: {len(df)}")

# ======================
# Stratified Sampling
# ======================

sampled = []

for country in df["country"].unique():

    country_df = df[df["country"] == country]

    for facet in country_df["facet"].unique():

        facet_df = country_df[
            country_df["facet"] == facet
        ]

        n = min(5, len(facet_df))

        sampled.append(
            facet_df.sample(
                n=n,
                random_state=42
            )
        )

df_sampled = pd.concat(
    sampled,
    ignore_index=True
)

# ======================
# Statistics
# ======================

print("\nSample Size:")
print(len(df_sampled))

print("\nCountry Distribution:")
print(
    df_sampled["country"]
    .value_counts()
)

print("\nCountry x Facet:")
print(
    pd.crosstab(
        df_sampled["country"],
        df_sampled["facet"]
    )
)

# ======================
# Save
# ======================

sample_ids_path = os.path.join(
    RESULTS_DIR,
    "sample_ids.csv"
)

sample_metadata_path = os.path.join(
    RESULTS_DIR,
    "sample_metadata.csv"
)

df_sampled["u_id"].to_csv(
    sample_ids_path,
    index=False
)

df_sampled.to_csv(
    sample_metadata_path,
    index=False
)

print("\nSaved:")
print(sample_ids_path)
print(sample_metadata_path)
