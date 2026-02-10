# preprocessing.py

import re
import json

# Step 1: Load text
def load_text_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# # Step 2: Clean disfluencies
# def clean_text(text):
#     return re.sub(r'\[(um|uh|er|ah|hm|mm)\]', '', text, flags=re.IGNORECASE).strip()

# Step 2: Extract transcript from AWS JSON
def load_text_from_aws_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Return the first full transcript
    return data['results']['transcripts'][0]['transcript']



def clean_text(text):
    # Step 1: Remove filler words (uh, um...) and remove punctuation
    text = re.sub(r'\s?\b(uh|um|er|ah|hm|mm)\b(?=[.,!?;:])', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(uh|um|er|ah|hm|mm)\b', '', text, flags=re.IGNORECASE)

    # Step 2: Remove space before punctuation (e.g., " .", " ,")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # ✅ Step 3: Fix mixed or repeated punctuation (e.g., ",.", ".,", ",,", "..")
    text = re.sub(r'([.,!?;:])([.,!?;:])', r'\2', text)  # keep the second punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)         # collapse repeated (e.g., .. → .)

    # Step 4: Collapse extra spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Step 5: Strip leading/trailing whitespace
    return text.strip()

# def clean_text(text):
#     # ============================
#     # 1. Remove fillers + neighbor punctuation
#     #    Examples removed:
#     #    "uh,"  "Um."  "ah?"  "mm;"  "uh -"
#     # ============================
#     filler_pattern = r'\b(uh|um|er|ah|hm|mm)\b[\s]*[.,!?;:-]*'
#     text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)

#     # ============================
#     # 2. Remove leftover isolated filler tokens
#     # ============================
#     text = re.sub(r'\b(uh|um|er|ah|hm|mm)\b', '', text, flags=re.IGNORECASE)

#     # ============================
#     # 3. Remove space before punctuation
#     #    e.g., " ," → ","
#     # ============================
#     text = re.sub(r'\s+([.,!?;:])', r'\1', text)

#     # ============================
#     # 4. Fix repeated or mixed punctuation
#     # ============================
#     text = re.sub(r'([.,!?;:])([.,!?;:])', r'\2', text)  # keep second
#     text = re.sub(r'([.,!?;:])\1+', r'\1', text)         # collapse repeats

#     # ============================
#     # 5. Collapse extra spaces
#     # ============================
#     text = re.sub(r'\s{2,}', ' ', text)

#     # ============================
#     # 6. Trim whitespace
#     # ============================
#     return text.strip()


