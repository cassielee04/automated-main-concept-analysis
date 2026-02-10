# main.py
import os
import torch
import pandas as pd
import json
from lexicalrichness import LexicalRichness
from glob import glob
import argparse
from datetime import datetime

from preprocessing import load_text_from_file, clean_text, load_text_from_aws_json
from features import extract_84_features, analyze_transcript_qpa, get_stanza_pipeline, get_total_speech_duration
from pauses_fillers import count_fillers_pauses_json
from fillers import count_fillers
from ciu import calculate_cinderella_ciu
from segment_utterance import segment_utterances
from mainconcept_normalize import MainConceptAnalyzerNormalize
import sys
from pathlib import Path
from normalize_utterances import normalize_utterances
from coherence import count_errors, calculate_coherence_error_percentages
import os


def process_file(json_path, cutoff):
    # Extract participant ID
    base = os.path.splitext(os.path.basename(json_path))[0]
    participant_id = base.split('_')[0]

    # Load and clean text
    text = load_text_from_aws_json(json_path)
    cleaned_text = clean_text(text)
    total_words = len(cleaned_text.split())

    # Feature extraction
    df_features = pd.DataFrame([{}]) 


    # Fillers and pause analysis
    json_data = {}
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    df_features['participant'] = participant_id

    # # Utterance Level Features
    # # 1. Noun Verb Ratio
    
    segmented_utterances_cleaned = segment_utterances(cleaned_text)
    analyzer = MainConceptAnalyzerNormalize(global_cutoff=cutoff)
    # # Main Concept Matching
    analyzer.reset_tracking()
    df_features['total_utterances'] = len(segmented_utterances_cleaned)
    normalized_utterances = normalize_utterances(segmented_utterances_cleaned)
    main_concepts_df = analyzer.get_mainconcept_match(segmented_utterances_cleaned, normalized_utterances,  return_score=True)
    d = pd.to_numeric(
        main_concepts_df["distance_to_centroid"],
        errors="coerce"
    )
    df_features['distance_to_centroid'] =  float(d.mean())
    df_features['num_unique_main_concepts'] = analyzer.get_total_unique_mainconcepts()
    df_features['num_total_main_concepts'] = analyzer.get_total_mainconcepts()
    df_features['unique_main_concept_match_ratio'] = (df_features['num_unique_main_concepts'] / 34)
    df_features['total_main_concept_match_ratio'] = (df_features['num_total_main_concepts'] / df_features['total_utterances'])
    matched_seq, sequence_score = analyzer.score_story_sequence(segmented_utterances_cleaned, normalized_utterances, return_score=True)
    df_features['sequence_score_mainconcept'] = sequence_score


    # After calling count_errors
    participant_codes = [participant_id] * len(segmented_utterances_cleaned)
    counts, utterance_errors = count_errors(segmented_utterances_cleaned, participant_codes)
    total_utterances = int(df_features['total_utterances'].iloc[0])
    # Extract local/global percentages (as you have)
    local_pct, global_pct = calculate_coherence_error_percentages(counts, total_utterances)
    df_features['local_coherence_mainconcept'] = local_pct
    df_features['global_coherence_mainconcept'] = global_pct

    # # Define error types (from classify_error logic)
    error_types = [
        'Topic Switching', 'Missing Referent', 'Tangential Utterance',
        'Propositional Repetition', 'Filler', 'Conceptual Incongruence', 'Coherent', 
    ]

    # Extract counts and ratios for each (0 if not present)
    for error_type in error_types:
        num_col = f'num_{error_type.lower().replace(" ", "_")}'
        ratio_col = f'ratio_{error_type.lower().replace(" ", "_")}_percent'
        
        num = counts.get(error_type, 0)
        ratio = (num / total_utterances * 100) if total_utterances > 0 else 0.0
        
        df_features[num_col] = num
        df_features[ratio_col] = round(ratio, 2)  # Round to 2 decimals for cleanliness

    # Reorder columns
    cols = ['participant'] + [col for col in df_features.columns if col != 'participant']
    df_features = df_features[cols]

    return df_features


# Main routine

CUTOFFS = {
    "dementia": 0.8292,#0.8587, #0.8047,
    "aphasia": 0.8292,#0.8587, #0.8289,
    "controls": 0.8292,
    # "dementia_controls": 0.8047,
    # "aphasia_controls": 0.8289,
}


COHORTS = {
    # cohort_name: (folder_path, output_csv_name)
    "dementia": ("../../Data/transcripts/bank/dementia/aws/cinderella",
                 "3_mainconcept_features_dem_qsub.csv"),
    "aphasia": ("../../Data/transcripts/bank/aphasia/aws/cinderella",
                "3_mainconcept_features_aph_qsub.csv"),
    "controls": ("../../Data/transcripts/bank/controls/aws/cinderella",
                 "3_mainconcept_features_ctr_qsub.csv"),
    
    # "dementia_controls": ("../../Data/transcripts/bank/controls/aws/cinderella",
    #                       "dementia_controls_mainconcept_features_update_qsub.csv"),
    # "aphasia_controls": ("../../Data/transcripts/bank/controls/aws/cinderella",
    #                      "aphasia_controls_mainconcept_features_update_qsub.csv"),
}

def main(cohort: str):
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort {cohort}. Choose from: {list(COHORTS)}")
    if cohort not in CUTOFFS:
        raise ValueError(f"Missing cutoff for cohort {cohort}. Add it to CUTOFFS.")
    
    folder_rel, out_name = COHORTS[cohort]
    cutoff = CUTOFFS[cohort]

    folder_path = os.path.abspath(folder_rel)
    all_dfs = []

    for idx, text_file in enumerate(glob(os.path.join(folder_path, "*_transcribed.json"))):
        filename = os.path.basename(text_file)
        # only process wright03a file
        # if filename == "wright03a_cinderella_transcribed.json":
        base = os.path.splitext(os.path.basename(text_file))[0]
        participant_id = base.split('_')[0]
        json_path = os.path.abspath(os.path.join(folder_rel, f"{base}.json"))
        print(f"🔍 Processing files for participant: {participant_id}")
        print(f"🚀 Processing Participant: {participant_id}")
        print(f"[DEBUG] cohort={cohort} cutoff={cutoff}")
        # json_path = f"../../Data/transcripts/bank/controls/aws/cinderella/{base}.json"
        # print(f"🔍 Processing files for participant: {participant_id}")
        # print(f"\n🚀 Processing Participant: {participant_id}")

        df_features = process_file(json_path, cutoff=cutoff)
        all_dfs.append(df_features)

    # Merge all participant data into one CSV
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        out_dir = os.path.abspath("../data/classification")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)

        merged_df.to_csv(out_path, index=False)
        print(f"\nMerged features saved to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=list(COHORTS.keys()))
    args = parser.parse_args()
    main(args.cohort)