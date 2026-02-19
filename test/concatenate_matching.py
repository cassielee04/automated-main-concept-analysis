#!/usr/bin/env python
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────
# Paths
# ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" 

APHASIA_CSV  = DATA_DIR / "Matching Concept Check - Aphasia - All.csv"
CONTROLS_CSV = DATA_DIR / "Matching Concept Check - Controls - All.csv"
DEMENTIA_CSV = DATA_DIR / "Matching Concept Check - Dementia - All.csv"

OUTPUT = DATA_DIR / "Matching_Concept_All_Combined.csv"

FILES = [
    ("Aphasia", APHASIA_CSV),
    ("Control", CONTROLS_CSV),
    ("Dementia", DEMENTIA_CSV)
]

# ─────────────────────────────────────
# Load & clean function
# ─────────────────────────────────────
def load_clean(path, cohort):
    print(f"Loading {path}")

    df = pd.read_csv(path)

    # remove empty unnamed columns (excel artifacts)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # normalize column names (remove trailing spaces)
    df.columns = df.columns.str.strip()

    # add cohort label
    df["cohort"] = cohort

    # keep only relevant columns if extras exist
    keep_cols = [
        "participant",
        "utterances",
        "Ground Truth Matching?",
        "matching1",
        "matching2",
        "matching3",
        "Disagree Flag",
        "cohort"
    ]

    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    return df


# ─────────────────────────────────────
# Concatenate
# ─────────────────────────────────────
dfs = []
for cohort, path in FILES:
    dfs.append(load_clean(path, cohort))

combined = pd.concat(dfs, ignore_index=True)

# Optional: remove fully empty utterances
combined = combined[combined["utterances"].notna()]
combined = combined[combined["utterances"].astype(str).str.strip() != ""]

# Save
combined.to_csv(OUTPUT, index=False)

print("\nSaved combined dataset:")
print(OUTPUT)
print("Shape:", combined.shape)
print("\nCounts per cohort:")
print(combined["cohort"].value_counts())
