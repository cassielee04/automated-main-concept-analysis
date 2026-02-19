#!/usr/bin/env python
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. PROJECT ROOT & PYTHON PATH
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cutoff = 0.8292

# ──────────────────────────────────────────────────────────────
# 2. IMPORTS
# ──────────────────────────────────────────────────────────────
import pandas as pd
from collections import defaultdict
from src.normalize_utterances import normalize_utterances
from src.mainconcept_normalize import MainConceptAnalyzerNormalize
from src.coherence import count_errors_fold_one_participant

# ──────────────────────────────────────────────────────────────
# 3. PATHS
# ──────────────────────────────────────────────────────────────
DATA_DIR         = PROJECT_ROOT / "data"
CONFIG_DIR       = PROJECT_ROOT / "config"

INPUT_CSV        = DATA_DIR / "Matching_Concept_All_Combined.csv"
OUTPUT_CSV       = DATA_DIR / "matching_mainconcept_all_error_predicted_output.csv"

CONFIG_PATH      = CONFIG_DIR / "story_config.yml"
EMBEDDINGS_PATH  = CONFIG_DIR / "cinderella_mainconcept_embeddings.pkl"

# ──────────────────────────────────────────────────────────────
# 4. LOAD DATA
# ──────────────────────────────────────────────────────────────
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"File not found:\n{INPUT_CSV}")

print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Find utterance column
utterance_col = next(
    (c for c in ["utterances", "Utterance", "utterance", "text", "Transcript"] if c in df.columns),
    None
)
if utterance_col is None:
    raise KeyError(f"Could not find utterance column. Available: {list(df.columns)}")

# Find participant column
participant_col = next(
    (c for c in ["participant", "participant_id", "participant_code"] if c in df.columns),
    None
)

utterances_list = df[utterance_col].fillna("").astype(str).tolist()
print(f"Total utterances: {len(utterances_list)}")
if participant_col:
    print(f"Participant column found: '{participant_col}'")
else:
    print("No participant column found — treating all utterances as one group.")

# ──────────────────────────────────────────────────────────────
# 5. NORMALIZE & LOAD ANALYZER
# ──────────────────────────────────────────────────────────────
print("Normalizing utterances...")
normalized_list = normalize_utterances(utterances_list)

print("Loading analyzer...")
analyzer = MainConceptAnalyzerNormalize(
    config_path=str(CONFIG_PATH),
    embeddings_file=str(EMBEDDINGS_PATH),
    global_cutoff=cutoff
)

# ──────────────────────────────────────────────────────────────
# 6. PROCESS PER PARTICIPANT
# ──────────────────────────────────────────────────────────────

# Build participant groups preserving original row order
if participant_col:
    participant_ids = df[participant_col].fillna("__NONE__").astype(str).tolist()
else:
    participant_ids = ["__ALL__"] * len(utterances_list)

# Map each participant -> list of original row indices
idx_map = defaultdict(list)
for i, pid in enumerate(participant_ids):
    idx_map[pid].append(i)

error_types_aligned  = [None] * len(utterances_list)
matched_concepts_aligned = [None] * len(utterances_list)

for pid, indices in idx_map.items():
    print(f"  Processing participant: {pid} ({len(indices)} utterances)")

    sub_utts       = [utterances_list[i] for i in indices]
    sub_normalized = [normalized_list[i]  for i in indices]

    # Fresh analyzer per participant to reset concept/repetition tracking
    participant_analyzer = MainConceptAnalyzerNormalize(
        config_path=str(CONFIG_PATH),
        embeddings_file=str(EMBEDDINGS_PATH),
        global_cutoff=cutoff
    )
    participant_analyzer.reset_tracking()

    # Run main concept matching
    mc_df = participant_analyzer.get_mainconcept_match(
        sub_utts, sub_normalized, return_score=True
    )
    mc_df = mc_df.reset_index(drop=True)

    sub_is_match       = mc_df["is_main_concept"].tolist()
    sub_matched        = mc_df["matched_concept"].tolist()
    sub_is_repeated    = mc_df["is_repeated"].tolist()

    # Compute topic switching flags per participant
    sub_topic_switch = [False] * len(sub_utts)
    for i in range(1, len(sub_utts)):
        try:
            sub_topic_switch[i] = bool(
                participant_analyzer.is_topic_switching(sub_utts[i - 1], sub_utts[i])
            )
        except Exception:
            sub_topic_switch[i] = False

    # Classify errors
    counts, utterance_errors = count_errors_fold_one_participant(
        sub_utts, sub_is_match, sub_matched, sub_is_repeated, sub_topic_switch
    )

    # utterance_errors: list of (None, utterance, error_type, matched_concept)
    for local_idx, (_, _, err, matched) in enumerate(utterance_errors):
        global_idx = indices[local_idx]
        error_types_aligned[global_idx]    = err
        matched_concepts_aligned[global_idx] = matched

# ──────────────────────────────────────────────────────────────
# 7. BUILD OUTPUT — full column set
# ──────────────────────────────────────────────────────────────

# Collect per-row mc_df signals across all participants in original order
normalized_col_out       = [None] * len(utterances_list)
similarity_score_out     = [None] * len(utterances_list)
distance_to_centroid_out = [None] * len(utterances_list)
is_main_concept_out      = [None] * len(utterances_list)
best_idx_out             = [None] * len(utterances_list)
is_repeated_out          = [None] * len(utterances_list)

for pid, indices in idx_map.items():
    sub_utts       = [utterances_list[i] for i in indices]
    sub_normalized = [normalized_list[i]  for i in indices]

    participant_analyzer = MainConceptAnalyzerNormalize(
        config_path=str(CONFIG_PATH),
        embeddings_file=str(EMBEDDINGS_PATH),
        global_cutoff=cutoff
    )
    participant_analyzer.reset_tracking()

    mc_df = participant_analyzer.get_mainconcept_match(
        sub_utts, sub_normalized, return_score=True
    )
    mc_df = mc_df.reset_index(drop=True)

    for local_idx, global_idx in enumerate(indices):
        normalized_col_out[global_idx]       = mc_df.loc[local_idx, "normalized_utterance"]
        similarity_score_out[global_idx]     = mc_df.loc[local_idx, "similarity_score"]
        distance_to_centroid_out[global_idx] = mc_df.loc[local_idx, "distance_to_centroid"]
        is_main_concept_out[global_idx]      = mc_df.loc[local_idx, "is_main_concept"]
        best_idx_out[global_idx]             = mc_df.loc[local_idx, "best_idx"]
        is_repeated_out[global_idx]          = mc_df.loc[local_idx, "is_repeated"]

output_df = pd.DataFrame({
    "utterance":            utterances_list,
    "normalized_utterance": normalized_col_out,
    "matched_concept":      matched_concepts_aligned,
    "similarity_score":     similarity_score_out,
    "distance_to_centroid": distance_to_centroid_out,
    "is_main_concept":      is_main_concept_out,
    "best_idx":             best_idx_out,
    "is_repeated":          is_repeated_out,
    "participant":          participant_ids,
    "error_type":           error_types_aligned,
})

# ──────────────────────────────────────────────────────────────
# 8. SAVE
# ──────────────────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Shape: {output_df.shape}")
print("\nError type distribution:")
print(output_df["error_type"].value_counts().to_string())