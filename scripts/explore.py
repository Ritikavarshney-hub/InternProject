from datasets import load_from_disk
import pandas as pd
import os

os.makedirs("../results", exist_ok=True)
ds = load_from_disk("../data/CulturalVQA")

print(ds)

# check splits
print(ds.keys())

df = ds["test"].to_pandas()

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
print(pd.crosstab(df["country"], ["facet"]))
facet_props = pd.crosstab(
    df["country"],
    df["facet"],
    normalize="index"
)

print(df.groupby(["country", "facet"]).size())
