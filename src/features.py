# features.py

from collections import defaultdict
import stanza
import torch
import os
from datetime import timedelta
import re
from datetime import timedelta



# ---------- Set up environment and Stanza pipeline ----------
os.environ['STANZA_RESOURCES_DIR'] = '/projectnb/skiran/Cassie/resources'
USE_GPU = torch.cuda.is_available()
print(f"✅ Using GPU: {USE_GPU}")

# ✅ Download ONCE (you can comment this out after first run)
stanza.download('en', model_dir=os.environ['STANZA_RESOURCES_DIR'])

# ✅ Initialize pipeline ONCE and cache
_stanza_pipeline = stanza.Pipeline(
    lang='en',
    processors='tokenize,mwt,pos,lemma,depparse',
    use_gpu=USE_GPU,
    download_method=stanza.DownloadMethod.REUSE_RESOURCES
)

def get_stanza_pipeline():
    return _stanza_pipeline


# # Download and initialize pipeline only once
# stanza.download('en')  # comment out after first run
# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=USE_GPU)

# POS, DEP, and Morphological categories
POS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
    'SCONJ', 'SYM', 'VERB', 'X'
]

DEP_LABELS = [
    'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf',
    'compound', 'conj', 'cop', 'csubj', 'det', 'discourse', 'discolated', 'expl',
    'fixed', 'flat', 'goeswith', 'iobj', 'mark', 'nmod', 'nsubj', 'nummod', 'obj',
    'obl', 'parataxis', 'punct', 'reparandum', 'orphan', 'root', 'vocative',
    'xcomp', 'dep'
]

MORPH_FEATURES = [
    'Case', 'Gender', 'Mood', 'Number', 'Person', 'Tense', 'VerbForm', 'Voice',
    'Definite', 'Degree', 'PronType', 'NumType', 'Polarity', 'Poss', 'Reflex',
    'Aspect', 'Clusivity', 'Evident', 'Foreign', 'Polite', 'Abbr', 'ConjType',
    'Typo', 'Animacy', 'NounClass'
]

def extract_84_features(text):
    nlp = get_stanza_pipeline()
    doc = nlp(text)
    feature_counts = defaultdict(int)

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in POS_TAGS:
                feature_counts[f"POS_{word.upos}"] += 1
            if word.deprel in DEP_LABELS:
                feature_counts[f"DEP_{word.deprel}"] += 1
            if word.feats:
                morph_items = word.feats.split('|')
                for item in morph_items:
                    morph_key = item.split('=')[0]
                    if morph_key in MORPH_FEATURES:
                        feature_counts[f"MORPH_{morph_key}"] += 1

    return dict(feature_counts)

# ========== QPA (UTTERANCE METRICS + TUNIT) ==========

# def extract_tunits(doc):
#     t_units = []

#     for sent in doc.sentences:
#         # Identify main clauses (root and coordinated main verbs)
#         main_clauses = [word for word in sent.words if word.deprel in {'root', 'conj'} and word.upos == 'VERB']

#         for main in main_clauses:
#             tunit_tokens = collect_main_and_subordinate(main.id, sent.words)
#             tunit_tokens = sorted(set(tunit_tokens))
#             tunit_text = " ".join(sent.words[i-1].text for i in tunit_tokens)
#             t_units.append(tunit_text)
#     # for i, tunit in enumerate(t_units, 1):
#     #     print(f"{i}. {tunit}")

#     return t_units


def collect_main_and_subordinate(head_id, words):
    """
    Recursively collects tokens from main clause and its subordinate clauses.
    """
    tokens = [head_id]
    for word in words:
        if word.head == head_id and word.deprel not in {'conj'}:  # exclude other main clauses
            tokens.extend(collect_main_and_subordinate(word.id, words))
    return tokens

# def count_utterances(text):
#     # Normalize ellipses and long pauses (… or ...)
#     text = re.sub(r'(\.\.\.|…|\.{2,})', ' <PAUSE> ', text)

#     # Remove common filled pauses (uh, um, ah) – optional: keep if needed for analysis
#     text = re.sub(r'\b(uh|um|erm|ah|eh|hmm)\b', '', text, flags=re.IGNORECASE)

#     # Normalize repeated commas (used like pauses in disfluent text)
#     text = re.sub(r',+', ' <PAUSE> ', text)

#     # Normalize sentence-ending punctuation
#     text = re.sub(r'[!?]', '.', text)

#     # Split on either real sentence boundaries or pause-based utterance markers
#     utterances = re.split(r'[.]|<PAUSE>', text)

#     # Clean whitespace and remove empty results
#     utterances = [utt.strip() for utt in utterances if utt.strip()]

#     return len(utterances), utterances



def analyze_transcript_qpa(text, cleaned_text):
    nlp = get_stanza_pipeline()
    doc = nlp(cleaned_text)                  # 👈 Parse the text first
    total_words = sum(len([w for w in sent.words if w.upos not in {'PUNCT', 'SYM'}]) for sent in doc.sentences)
    return total_words

# def parse_timestamp(timestamp):
#     """Parses VTT timestamp to timedelta. Handles both '.' and ',' as decimal separators."""
#     timestamp = timestamp.strip().replace(",", ".")
#     hours, minutes, seconds = timestamp.split(":")
#     seconds, dot, milliseconds = seconds.partition(".")
#     milliseconds = milliseconds if dot else "0"
#     return timedelta(
#         hours=int(hours),
#         minutes=int(minutes),
#         seconds=int(seconds),
#         milliseconds=int(milliseconds.ljust(3, '0'))  # pad to 3 digits
#     )

# def get_vtt_duration(vtt_path):
#     """Get total speech time (sum of all caption durations) in seconds."""
#     total_duration = timedelta(0)

#     with open(vtt_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if "-->" in line:
#                 try:
#                     start_str, end_str = line.strip().split(" --> ")
#                     start_time = parse_timestamp(start_str)
#                     end_time = parse_timestamp(end_str)
#                     segment_duration = end_time - start_time
#                     total_duration += segment_duration
#                 except Exception as e:
#                     print(f"⚠️ Skipping bad timestamp line: {line.strip()} - {e}")
#                     continue

#     return total_duration.total_seconds()

def get_total_speech_duration(json_data):
    try:
        items = json_data['results']['items']
    except KeyError:
        print("Error: 'results' or 'items' not found in JSON data")
        return 0.0
    
    if not items:
        print("No items found in JSON data")
        return 0.0
    
    # Filter pronunciation items with valid start_time and end_time
    pronunciation_items = [
        item for item in items 
        if item.get('type') == 'pronunciation' and 'start_time' in item and 'end_time' in item
    ]
    
    if not pronunciation_items:
        print("No valid pronunciation items found")
        return 0.0
    
    try:
        # Get first start_time and last end_time
        first_start = min(float(item['start_time']) for item in pronunciation_items)
        last_end = max(float(item['end_time']) for item in pronunciation_items)
        total_speech_time = last_end - first_start
        return round(total_speech_time, 3)
    except (ValueError, TypeError) as e:
        print(f"Error processing times: {e}")
        return 0.0
