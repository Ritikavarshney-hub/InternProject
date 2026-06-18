from datasets import load_from_disk
import pandas as pd
import os

# =====================================
# Paths
# =====================================

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

# =====================================
# Config
# =====================================

RANDOM_STATE = 42
MAX_PER_CELL = 5

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

ds = load_from_disk(DATASET_PATH)["test"]

df = ds.to_pandas()

if "image" in df.columns:
    df = df.drop(columns=["image"])

print(f"Rows      : {len(df)}")
print(f"Countries : {df['country'].nunique()}")
print(f"Facets    : {df['facet'].nunique()}")

# =====================================
# Stratified Sampling
# =====================================

sampled_rows = []

for country in sorted(df["country"].unique()):

    country_df = df[df["country"] == country]

    for facet in sorted(country_df["facet"].unique()):

        facet_df = country_df[
            country_df["facet"] == facet
        ]

        n = min(
            len(facet_df),
            MAX_PER_CELL
        )

        sampled = facet_df.sample(
            n=n,
            random_state=RANDOM_STATE
        )

        sampled_rows.append(sampled)

df_sampled = pd.concat(
    sampled_rows,
    ignore_index=True
)

# =====================================
# Statistics
# =====================================

print("\n========================")
print("Sample Statistics")
print("========================")

print(f"Sample Size : {len(df_sampled)}")

print("\nImages Per Country")
print(
    df_sampled["country"]
    .value_counts()
)

print("\nImages Per Facet")
print(
    df_sampled["facet"]
    .value_counts()
)

print("\nCountry x Facet")
print(
    pd.crosstab(
        df_sampled["country"],
        df_sampled["facet"]
    )
)

# =====================================
# Save
# =====================================

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
