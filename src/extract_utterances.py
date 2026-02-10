import os
import re
import csv
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from segment_utterance import segment_utterances

# Root directory containing folders starting with Texts_
root_dir = "../../Data/transcripts/bank/controls"
output_csv = "utterances_controls_output.csv"


all_rows = []

# Walk through folders and find those starting with "Texts_"
for dirpath, dirnames, filenames in os.walk(root_dir):
    if os.path.basename(dirpath).startswith("Texts_Controls"):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

                # Extract patient code before first "_"
                patient_code = filename.split("_")[0]

                # Get utterances
                utterances = segment_utterances(text_content)
                print(f"Extracted {len(utterances)} utterances from {filename}")

                # Append each utterance with filename info
                for utt in utterances:
                    utt = utt.strip()
                    if utt:  # skip empty
                        all_rows.append([patient_code, utt])

# Check if file exists to handle header
file_exists = os.path.isfile(output_csv)
start_index = 0
if file_exists:
    with open(output_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        existing_rows = list(reader)
        start_index = len(existing_rows) - 1  # -1 to exclude header

# Append to CSV
with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["index", "participant_code", "utterance"])  # header

    for idx, row in enumerate(all_rows):
        writer.writerow([start_index + idx] + row)

print(f"Appended {len(all_rows)} utterances to {output_csv}")


# def extract_random_utterances_to_csv(input_csv_path, output_csv_path, num_participants=10):
#     try:
#         # Read the input CSV file
#         df = pd.read_csv(input_csv_path)

#         # Ensure the required columns exist
#         if not all(col in df.columns for col in ['participant_code', 'utterance']):
#             raise ValueError("CSV must contain 'participant_code' and 'utterance' columns")

#         # Get the first 30 unique participant codes in order of appearance
#         participant_codes = pd.unique(df['participant_code'])[:num_participants]

#         if len(participant_codes) < num_participants:
#             print(f"Warning: Only {len(participant_codes)} unique participants found, less than {num_participants} requested")

#         # Filter the DataFrame to include only the selected participants
#         filtered_df = df[df['participant_code'].isin(participant_codes)][['participant_code', 'utterance']]

#         # Save to output CSV
#         filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8')
#         print(f"Utterances for {len(participant_codes)} participants saved to {output_csv_path}")

#         return filtered_df

#     except FileNotFoundError:
#         print(f"Error: Input file '{input_csv_path}' not found")
#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     # Input CSV file (generated from your first script)
#     input_csv = "../data/utterances_dementia_output.csv"
#     # Output CSV file
#     output_csv = "../data/coherence_dementia_sample.csv"
#     extract_random_utterances_to_csv(input_csv, output_csv, num_participants=10)