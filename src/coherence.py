import spacy

nlp = spacy.load("en_core_web_sm")
from fillers import contains_filler
from pauses_fillers import contains_empty_filler
# from mainconcept import get_mainconcept_match, is_topic_switching, is_repeated_mainconcept, find_average_similarity
from collections import Counter
from mainconcept_normalize import MainConceptAnalyzerNormalize
from missing_ref_hardcoded import ReferentChecker
import pandas as pd
from normalize_utterances import normalize_utterances, normalize_utterance
from segment_utterance import segment_utterances
from preprocessing import clean_text
import re
import os
from glob import glob

from collections import Counter

def classify_error_from_signals(
    utterances: list,
    idx: int,
    detector: ReferentChecker,
    is_mainconcept_match: bool,
    matched_concept,
    is_repeated: bool,
    is_topic_switch: bool,
) -> str:
    current_utt = utterances[idx]
    prev_utt = utterances[idx - 1] if idx > 0 else ""
    context = utterances[max(0, idx-5):idx]

    if is_mainconcept_match:
        if contains_personal_story(current_utt) and not contains_filler(current_utt):
            return "Tangential Utterance"
        elif idx > 0 and is_repeated:
            return "Propositional Repetition"
        elif idx > 0 and is_incomplete_sentence(prev_utt) and is_topic_switch:
            return "Topic Switching"
        elif detector.check_utterance(context, current_utt):
            return "Missing Referent"
        else:
            return "Coherent"
    else:
        if contains_filler(current_utt):
            return "Filler"
        elif detector.check_utterance(context, current_utt):
            return "Missing Referent"
        return "Conceptual Incongruence"


def count_errors_fold_one_participant(
    utterances: list,
    is_mainconcept_match: list[bool],
    matched_concepts: list,
    is_repeated_flags: list[bool],
    topic_switch_flags: list[bool],
) -> tuple[dict, list]:
    """
    Returns the SAME thing as count_errors() but for ONE participant and using fold signals:
      counts: dict(error_type -> count)
      utterance_errors: list of (participant_code=None, utterance, error_type, matched_concept)
    """
    assert len(utterances) == len(is_mainconcept_match) == len(matched_concepts) == len(is_repeated_flags) == len(topic_switch_flags)

    detector = ReferentChecker()
    error_counts = Counter()
    utterance_errors = []

    for idx in range(len(utterances)):
        err = classify_error_from_signals(
            utterances=utterances,
            idx=idx,
            detector=detector,
            is_mainconcept_match=bool(is_mainconcept_match[idx]),
            matched_concept=matched_concepts[idx],
            is_repeated=bool(is_repeated_flags[idx]),
            is_topic_switch=bool(topic_switch_flags[idx]),
        )
        error_counts[err] += 1
        utterance_errors.append((None, utterances[idx], err, matched_concepts[idx]))

    return dict(error_counts), utterance_errors


def is_incomplete_sentence(text):
    """Detect incomplete sentences using POS tags and dependency parsing"""
    text = text.strip()
    if not text:
        return True
    
    doc = nlp(text)

    # Check for subordinating conjunctions using POS tags
    first_token = doc[0]
    if first_token.pos_ == "SCONJ":  # Subordinating conjunction
        return True
    
    # Check for relative pronouns/adverbs at start
    if first_token.tag_ in ["WP", "WDT", "WRB"]:  # Wh- words
        return True
    
    # Check for sentence structure
    has_root_verb = any(token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"] for token in doc)
    has_subject = any(token.dep_.startswith("nsubj") or token.dep_.startswith("csubj") for token in doc)
    
    return not (has_root_verb and has_subject)

def contains_personal_story(text: str) -> bool:
    """
    Returns True if the utterance includes first-person pronouns,
    indicating a personal story (e.g., I, me, my, we, our, etc.).
    """
    first_person_pronouns = {"i", "me", "my", "mine", "myself", 
                             "we", "us", "our", "ours", "ourselves"}
    doc = nlp(text)
    
    for token in doc:
        if token.text.lower() in first_person_pronouns and token.pos_ == "PRON":
            return True
    return False

def classify_error(utterances: list, idx: int, analyzer: MainConceptAnalyzerNormalize, detector: ReferentChecker, mainconcept_result: pd.DataFrame) -> str:
    current_utt = utterances[idx]
    prev_utt = utterances[idx - 1] if idx > 0 else ""
    context = utterances[max(0, idx-5):idx]  # Previous 5 utterances
    # context = utterances[:idx]  # All previous utterances
    
    is_mainconcept_match = mainconcept_result['is_main_concept'].iloc[0]
    matched_concept = mainconcept_result['matched_concept'].iloc[0]
    is_repeated = mainconcept_result['is_repeated'].iloc[0]

    if is_mainconcept_match:
        # if matched_concept is not None:
        #         analyzer.count_repeated_mainconcept_by_idx(
        #             analyzer.concepts.index(matched_concept), add_to_set=True
        #     )
        if contains_personal_story(current_utt) and not contains_filler(current_utt):
            return "Tangential Utterance"
        elif idx > 0 and is_repeated:
            return "Propositional Repetition"
        elif idx > 0 and is_incomplete_sentence(prev_utt) and analyzer.is_topic_switching(prev_utt, current_utt):
            return "Topic Switching"
        elif detector.check_utterance(context, current_utt):
            return "Missing Referent"
        # elif detector.has_missing_referent(context, current_utt):
        #     return "Missing Referent"
        else:
            return "Coherent"

    else:
        if contains_filler(current_utt):
            return "Filler"
        # elif detector.has_missing_referent(context, current_utt):
        #     return "Missing Referent"
        elif  detector.check_utterance(context, current_utt):
            return "Missing Referent"

        return "Conceptual Incongruence"

def count_errors(utterances: list, participant_codes: list = None) -> tuple[dict, list]:
    error_counts = Counter()
    utterance_errors = []
    
    # Group utterances by participant
    groups = {}
    for idx, utt in enumerate(utterances):
        code = participant_codes[idx] if participant_codes and idx < len(participant_codes) else None
        code = code if not (pd.isna(code) or code == "") else None
        if code not in groups:
            groups[code] = []
        groups[code].append((idx, utt))  # Track original idx if needed
    
    # Process each group separately
    for code, group_utts in groups.items():
        group_utterances = [utt for _, utt in group_utts]  # Extract utts only
        normalize_group_utterances = [normalize_utterance(utt) for utt in group_utterances]
        group_codes = [code] * len(group_utterances)
        
        # Fresh analyzer per patient
        analyzer = MainConceptAnalyzerNormalize()
        detector = ReferentChecker()
        
        for g_idx in range(len(group_utterances)):
            # Compute mainconcept_result once
            normalized_utt = normalize_group_utterances[g_idx]
            mainconcept_result = analyzer.get_mainconcept_match(group_utterances[g_idx], normalized_utt)
            
            # Map back to global idx if needed, but classify on group
            error_type = classify_error(group_utterances, g_idx, analyzer, detector, mainconcept_result)
            matched_concept = mainconcept_result['matched_concept'].iloc[0]
            
            error_counts[error_type] += 1
            utterance_errors.append((code, group_utterances[g_idx], error_type, matched_concept))
    
    return dict(error_counts), utterance_errors
    
def calculate_coherence_error_percentages(error_counts: dict, total_utterances: int) -> tuple[float, float]:
    """
    Calculate percentages of local and global coherence errors.
    
    Args:
        error_counts (dict): Dictionary with error type counts.
        total_utterances (int): Total number of utterances.
    
    Returns:
        tuple: (local coherence error percentage, global coherence error percentage)
    """
    local_coherence_errors = sum(error_counts.get(error, 0) for error in ["Missing Referent", "Topic Switching"])
    global_coherence_errors = sum(error_counts.get(error, 0) for error in [
        "Tangential Utterance", "Propositional Repetition", "Filler", "Conceptual Incongruence"
    ])
    
    local_percentage = (local_coherence_errors / total_utterances) * 100 if total_utterances > 0 else 0
    global_percentage = (global_coherence_errors / total_utterances) * 100 if total_utterances > 0 else 0
    
    return local_percentage, global_percentage

def run_demo(demo_utts):
    """
    Runs error classification demo and returns results as a DataFrame.
    
    Args:
        demo_utts: List of utterances or DataFrame with 'utterance' column.
    
    Returns:
        tuple: (Error counts dict, DataFrame with index, utterance, error_type, matched_concept)
    """
    if isinstance(demo_utts, pd.DataFrame):
        utterances = demo_utts['utterance'].tolist()
    else:
        utterances = demo_utts
    
    print("length of utterances", len(utterances))
    #counts, utterance_errors = count_errors(utterances, participant_codes)
    counts, utterance_errors = count_errors(utterances)
    local_pct, global_pct = calculate_coherence_error_percentages(counts, len(utterances))
    print(f"Local Coherence Errors: {local_pct:.2f}%")
    print(f"Global Coherence Errors: {global_pct:.2f}%")
    
    # Convert utterance_errors to DataFrame
    df = pd.DataFrame(
        utterance_errors,
        columns=['Index', 'Utterance', 'Error Type', 'Matched Concept']
    )
    
    print("Error Counts:", counts)
    print("\nUtterance Error Mappings:")
    print(df.to_string(index=False))
    
    return counts, df

def extract_coherence_sanity_checks(csv_file_path) -> dict:
    """
    Extracts sanity check features from utterances.
    
    Args:
        utterances (list): List of utterances.
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        if 'utterance' not in df.columns:
            raise ValueError("CSV file must contain an 'utterance' column")
        utterances = df['utterance'].astype(str).tolist()  # Convert to string to handle any non-string entries
        participant_codes = df['participant_code'].tolist()
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_path}': {e}")
        return {}, pd.DataFrame(columns=['participant_code', 'utterance', 'Error Type', 'Matched Concept'])
    
    counts, utterance_errors = count_errors(utterances, participant_codes)
    if utterance_errors:
        local_pct, global_pct = calculate_coherence_error_percentages(counts, len(utterances))
    # print(f"Local Coherence Errors: {local_pct:.2f}%")
    # print(f"Global Coherence Errors: {global_pct:.2f}%")
    # print("Error Counts:", counts)
    
    # Convert utterance_errors to DataFrame
    result_df = pd.DataFrame(
        utterance_errors,
        columns=['participant_code', 'Utterance', 'Error Type', 'Matched Concept']
    )
    
    # Print only requested columns
    # print("\nUtterance Error Mappings:")
    # print(result_df[['participant_code','Utterance', 'Error Type', 'Matched Concept']].to_string(index=False))
    
    return counts, result_df[['participant_code','Utterance', 'Error Type', 'Matched Concept']]


# if __name__ == "__main__":


#     # folder_path = os.path.abspath("../Data/transcripts/bank/dementia/aws/cinderella/test_set")
#     # all_dfs = []
#     # for idx, text_file in enumerate(glob(os.path.join(folder_path, "*_transcribed.json"))):
#     #     base = os.path.splitext(os.path.basename(text_file))[0]
#     #     participant_id = base.split('_')[0]

#     #     json_path = f"../Data/transcripts/bank/dementia/aws/cinderella/{base}.json"
#     #     print(f"🔍 Processing files for participant: {participant_id}")
#     #     print(f"\n🚀 Processing Participant: {participant_id}")
#     #     base = os.path.splitext(os.path.basename(json_path))[0]
#     #     participant_id = base.split('_')[0]

#     #     # Load and clean text
#     #     text = load_text_from_aws_json(json_path)
#     #     cleaned_text = clean_text(text)
#     #     demo_utts = segment_utterances(cleaned_text)

#     #     # Feature extraction
#     #     df_features = pd.DataFrame([{}]) 

#     #     run_demo(demo_utts)
#     # Example usage
#     csv_file_path = "../data/coherence_controls_sample.csv"
#     # csv_file_path = "tools/sample_coherence_check.csv"
    
#     # # Run the analysis
#     counts, result_df = extract_coherence_sanity_checks(csv_file_path)
#     print("Final Error Counts:", counts)
    
#     # Optionally save the output to a new CSV file
#     result_df.to_csv("../data/sanitycheck_coherence_controls_dementia.csv", index=False)


