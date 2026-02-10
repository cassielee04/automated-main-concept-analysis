import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import csv

#pandas error put this in terminal : export LD_LIBRARY_PATH=/projectnb/skiran/Cassie/Code/miniconda3/envs/transcribe/lib:$LD_LIBRARY_PATH

# Simulated data from the two CSV files (replace with actual file reading)
file1_path = "data/test_utterances_outcome.xlsx"   # Excel
file2_path = "data/test_mainconcept_normalize_predicted_output.csv"    # CSV

# Assume the index column is named 'ID' (replace with the actual column name)
index_column = 'ID'  # Update this based on your data


# Load Excel (pandas uses openpyxl under the hood)
df_true = pd.read_excel(file1_path)

# Load CSV
df_pred = pd.read_csv(file2_path)


# Convert to DataFrames
df1 = pd.DataFrame(df_true)
df2 = pd.DataFrame(df_pred)


df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

# Option 2: Add a new index column if needed (e.g., a sequential ID)
print("Adding a new index column 'new_id'...")
df1['new_id'] = range(1, len(df1) + 1)
df2['new_id'] = range(1, len(df2) + 1)

# Set the new index column
df1.set_index('new_id', inplace=True)
df2.set_index('new_id', inplace=True)


# Merge the DataFrames on 'utterance'
merged_df = pd.merge(df1[['utterance', 'expected_outcome']], df2[['is_main_concept']], 
                     on='new_id', how='inner')

print(f"Merged DataFrame shape: {merged_df.shape}")

# Map T/F to 1/0 and TRUE/FALSE to 1/0 for consistency
merged_df['expected_outcome'] = merged_df['expected_outcome'].map({'T': 1, 'F': 0})
merged_df['is_main_concept'] = merged_df['is_main_concept'].map({True: 1, False: 0})
merged_df.to_csv("data/merged_mainconcept_normalize_results.csv", index=False, encoding="utf-8")

# Extract ground truth and predictions
y_true = merged_df['expected_outcome']
y_pred = merged_df['is_main_concept']

# --- Find mismatches ---
mismatches = merged_df[merged_df["expected_outcome"] != merged_df["is_main_concept"]].copy()
# Split into FP / FN for clarity
false_positives  = mismatches[(mismatches["expected_outcome"] == False) & (mismatches["is_main_concept"] == True)]
false_negatives  = mismatches[(mismatches["expected_outcome"] == True)  & (mismatches["is_main_concept"] == False)]
print(len(false_positives))
print(len(false_negatives))

mismatches.to_csv("data/mainconcept_normalize_mismatches.csv", index=False, encoding="utf-8")


# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)  # default = binary, positive class is "1"
print("F1 score:", f1)
# Print results
print("Confusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.4f}")