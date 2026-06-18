"""
Milestone 1 — Validation
Run after sample.py to confirm sample_ids.csv and sample_metadata.csv are consistent
and ready for downstream milestones.
"""

import pandas as pd
import sys

ids  = pd.read_csv("sample_ids.csv")
meta = pd.read_csv("sample_metadata.csv")

errors = []

if list(ids.columns) != ["u_id"]:
    errors.append(f"sample_ids.csv columns wrong: {list(ids.columns)}")

required_cols = {"u_id", "country", "facet"}
missing = required_cols - set(meta.columns)
if missing:
    errors.append(f"sample_metadata.csv missing columns: {missing}")

if not ids["u_id"].equals(meta["u_id"]):
    errors.append("u_id columns do not match between the two files")

if ids["u_id"].duplicated().any():
    errors.append("Duplicate u_ids in sample_ids.csv")

if errors:
    for e in errors:
        print(f"ERROR: {e}")
    sys.exit(1)

print("=== Sample validation passed ===")
print(f"Total images  : {len(meta)}")
print(f"Countries     : {meta['country'].nunique()}")
print(f"Facets        : {meta['facet'].nunique()}")

per_country = meta["country"].value_counts()
thin = per_country[per_country < 10]
if len(thin):
    print(f"\nWARNING: {len(thin)} countries with <10 images:")
    print(thin.to_string())

print("\nAll required columns present. Ready for Milestone 2.")
