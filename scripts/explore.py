from datasets import load_from_disk
import pandas as pd
import os

os.makedirs("../results", exist_ok=True)
dataset_path = "/DATA/bt24eee096/cultural_vlm/data/CulturalVQA/test"
ds = load_from_disk(dataset_path)

# Convert to DataFrame
df = ds.to_pandas()

print("\nColumns:")
print(df.columns)

print("\nCountries:")
print(df["country"].value_counts())

print("\nFacets:")
print(df["facet"].value_counts())

print("\nUnique countries:")
print(df["country"].nunique())

print("\nTotal images:")
print(len(df))

print("\nCountry x Facet:")
print(pd.crosstab(df["country"], df["facet"]))

facet_props = pd.crosstab(
    df["country"],
    df["facet"],
    normalize="index"
)

print("\nFacet proportions by country:")
print(facet_props)

print("\nCounts by country and facet:")
print(df.groupby(["country", "facet"]).size())