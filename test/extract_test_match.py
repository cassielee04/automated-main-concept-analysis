#!/usr/bin/env python
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. PROJECT ROOT & PYTHON PATH (makes src/ importable)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # /projectnb/skiran/Cassie/mainconcpet_analysis
sys.path.insert(0, str(PROJECT_ROOT))                          # ← this lets you do "from src.xxx import ..."

# ──────────────────────────────────────────────────────────────
# 2. IMPORTS
# ──────────────────────────────────────────────────────────────
import pandas as pd
from src.segment_utterance import segment_utterances
from src.normalize_utterances import normalize_utterances
from src.mainconcept_normalize import MainConceptAnalyzerNormalize

# ──────────────────────────────────────────────────────────────
# 3. PATHS (all based on PROJECT_ROOT → never break again)
# ──────────────────────────────────────────────────────────────
DATA_DIR         = PROJECT_ROOT / "data" / "data"
CONFIG_DIR       = PROJECT_ROOT / "config"

# Input file (add .csv if missing)
CONTROLS_CSV     = DATA_DIR / "Matching Concept Check - Dementia - All.csv"

# Config & precomputed embeddings
CONFIG_PATH      = CONFIG_DIR / "story_config.yml"
EMBEDDINGS_PATH  = CONFIG_DIR / "cinderella_mainconcept_embeddings.pkl"

# Output
OUTPUT_CSV       = DATA_DIR / "matching_mainconcept_dementia_predicted_output.csv"

# ──────────────────────────────────────────────────────────────
# 4. LOAD DATA
# ──────────────────────────────────────────────────────────────
if not CONTROLS_CSV.exists():
    raise FileNotFoundError(f"File not found:\n{CONTROLS_CSV}\nCheck the exact filename with: ls {DATA_DIR}")

df = pd.read_csv(CONTROLS_CSV)

# Adjust column name if needed (common variations)
utterance_col = None
for col in ["utterances", "Utterance", "utterance", "text", "Transcript"]:
    if col in df.columns:
        utterance_col = col
        break
if utterance_col is None:
    raise KeyError(f"Could not find utterance column. Available: {list(df.columns)}")

utterances_list = df[utterance_col].dropna().astype(str).tolist()

# ──────────────────────────────────────────────────────────────
# 5. PROCESS
# ──────────────────────────────────────────────────────────────
normalized_segmented_utterances = normalize_utterances(utterances_list)

analyzer = MainConceptAnalyzerNormalize(
    config_path=str(CONFIG_PATH),
    embeddings_file=str(EMBEDDINGS_PATH),
)

mainconcept_df = analyzer.get_mainconcept_match(
    utterances_list,
    normalized_segmented_utterances,
    return_score=True
)

mainconcept_df
# ──────────────────────────────────────────────────────────────
# 6. SAVE
# ──────────────────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)   # create folder if needed

print(f"DataFrame saved to:\n{OUTPUT_CSV}")
print(f"Shape: {mainconcept_df.shape}")