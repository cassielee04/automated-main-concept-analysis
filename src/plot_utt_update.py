import pandas as pd
import yaml
import pathlib
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import umap  # Requires: pip install umap-learn
import torch
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns  # For better plots, optional: pip install seaborn

# Define paths and model name
config_path = "../config/story_config.yml"  # Replace with your actual config path
model_name = 'all-mpnet-base-v2'
aphasia_df = pd.read_csv('../data/utterances_aphasia_output.csv')  # aphasia
dementia_df = pd.read_csv('../data/utterances_dementia_output.csv')  # dementia
controls_df = pd.read_csv('../data/utterances_controls_output.csv')  # controls

# Add group column
aphasia_df['group'] = 'aphasia'
dementia_df['group'] = 'dementia'
controls_df['group'] = 'control'

# Combine dataframes
df = pd.concat([aphasia_df, dementia_df, controls_df], ignore_index=True)

# Group by participant and collect utterances
grouped = df.groupby(['participant_code', 'group'])['utterance'].apply(list).reset_index()

# Load the model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(model_name)

# grouped has columns: ['participant_code', 'group', 'utterance'] where utterance is a LIST of utterances
participant_labels = []
participant_codes = []
participant_mean_distances = []
participant_embeddings = []

# ---- Load concepts + centroid exactly like analyzer (mean of concept embeddings) ----
cfg = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
concepts = cfg["stories"][0]["concepts"]
assert len(concepts) == 34, f"Expected 34 concepts, got {len(concepts)}"
concept_embeds = embedder.encode(
    concepts,
    convert_to_tensor=True,
    normalize_embeddings=True,
    device=device
)
centroid = torch.mean(concept_embeds, dim=0)  # shape (D,) (no keepdim needed)

# ---- Per participant: mean of distances to centroid (analyzer-style) ----
for _, row in grouped.iterrows():
    utterances = row["utterance"]  # list[str]
    participant_code = row["participant_code"]
    group = row["group"]
    if not utterances:
        continue
    utter_embeds = embedder.encode(
        utterances,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
        batch_size=32,
        show_progress_bar=False
    )  # shape (N, D)
    # Compute mean embedding for participant
    mean_embed = utter_embeds.mean(dim=0).cpu().numpy()
    participant_embeddings.append(mean_embed)
    # cosine similarity per utterance to centroid
    sims = util.cos_sim(utter_embeds, centroid.unsqueeze(0)).squeeze(1)  # (N,)
    dists = 1 - sims  # (N,)
    mean_dist = float(dists.mean().item())
    participant_mean_distances.append(mean_dist)
    participant_labels.append(group)
    participant_codes.append(participant_code)

participant_embeddings = np.array(participant_embeddings)  # (P, D)
participant_mean_distances = np.array(participant_mean_distances)  # (P,)
centroid_np = centroid.detach().cpu().numpy().reshape(1, -1)  # (1, D)

# ---- Group stats ----
aphasia = participant_mean_distances[np.array(participant_labels) == "aphasia"]
dementia = participant_mean_distances[np.array(participant_labels) == "dementia"]
control = participant_mean_distances[np.array(participant_labels) == "control"]
print(f"Average cosine distance (analyzer-style) for aphasia: {aphasia.mean():.4f} (std: {aphasia.std():.4f})")
print(f"Average cosine distance (analyzer-style) for dementia: {dementia.mean():.4f} (std: {dementia.std():.4f})")
print(f"Average cosine distance (analyzer-style) for controls: {control.mean():.4f} (std: {control.std():.4f})")
print(
    f"Total participants: {len(participant_labels)} "
    f"(Aphasia: {len(aphasia)}, Dementia: {len(dementia)}, Controls: {len(control)})"
)

from matplotlib.lines import Line2D
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances as cosdist

# ---- Compute MDS on cosine distances ----
all_data = np.vstack([participant_embeddings, centroid_np])
dist_matrix = cosdist(all_data, all_data)
mds = MDS(
    n_components=2,
    dissimilarity='precomputed',
    random_state=42,
    n_init=4,
    max_iter=300
)
all_mds_2d = mds.fit_transform(dist_matrix)
mds_2d = all_mds_2d[:-1]
centroid_mds_2d = all_mds_2d[-1].reshape(1, -1)

# ---- Create figure with 2 subplots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---------- LEFT: Histogram of distances ----------
ax1 = axes[0]
ax1.hist(aphasia, bins=20, alpha=0.7, label='Aphasia', color='red')
ax1.hist(dementia, bins=20, alpha=0.7, label='Dementia', color='black')
ax1.hist(control, bins=20, alpha=0.7, label='Controls', color='blue')
ax1.set_xlabel('Cosine Distance to Centroid')
ax1.set_ylabel('Frequency (Participants)')
ax1.set_title('Distribution of Participant Distances to Centroid')
ax1.legend()

# ---------- RIGHT: MDS map ----------
ax2 = axes[1]
for i, (x, y) in enumerate(mds_2d):
    group = participant_labels[i]
    if group == 'aphasia':
        ax2.scatter(x, y, marker='x', color='red', s=80, linewidths=2)
    elif group == 'dementia':
        ax2.scatter(x, y, marker='x', color='black', s=80, linewidths=2)
    else:  # control
        ax2.scatter(x, y, marker='o', color='blue', edgecolor='black', s=60, linewidth=0.7)

# centroid
ax2.scatter(
    centroid_mds_2d[:, 0], centroid_mds_2d[:, 1],
    marker='*', color='green', s=180, edgecolor='black', linewidth=1
)
ax2.set_xlabel('MDS Dimension 1')
ax2.set_ylabel('MDS Dimension 2')
ax2.set_title('MDS Map of Participant Distances to the Centroid')

# Custom legend (MDS subplot)
legend_elements = [
    Line2D([0], [0], marker='x', color='red', markerfacecolor='none', markeredgecolor='red',
           markersize=10, linewidth=0, label='Aphasia'),
    Line2D([0], [0], marker='x', color='black', markerfacecolor='none', markeredgecolor='black',
           markersize=10, linewidth=0, label='Dementia'),
    Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', markeredgecolor='black',
           markersize=8, linewidth=0, label='Control'),
    Line2D([0], [0], marker='*', color='green', markerfacecolor='green', markeredgecolor='black',
           markersize=12, linewidth=0, label='Centroid')
]
ax2.legend(handles=legend_elements, loc='best')
plt.tight_layout()
plt.savefig('participant_histogram_mds_subplot_update.png', dpi=300, bbox_inches='tight')
plt.close()

# --------------------------------------------
# UMAP plot with centroid (on participant average embeddings)
# --------------------------------------------
# Uncomment and adjust if needed; note: this creates a new figure
# all_data_umap = np.vstack([participant_embeddings, centroid_np])
# reducer = umap.UMAP(
#     n_components=2,
#     n_neighbors=40,
#     min_dist=0.4,
#     metric='cosine',
#     local_connectivity=3,
#     random_state=42
# )
# all_2d = reducer.fit_transform(all_data_umap)
# embeddings_2d = all_2d[:-1]
# centroid_2d = all_2d[-1].reshape(1, -1)
# 
# plt.figure(figsize=(8, 6))
# for i, (x, y) in enumerate(embeddings_2d):
#     group = participant_labels[i]
#     if group == 'aphasia':
#         plt.scatter(x, y, marker='x', color='red', s=80, linewidths=2)
#     elif group == 'dementia':
#         plt.scatter(x, y, marker='^', color='orange', s=70, edgecolor='k', linewidth=0.7)
#     else:  # control
#         plt.scatter(x, y, marker='o', color='blue', s=60, edgecolor='k', linewidth=0.7)
# # centroid
# plt.scatter(
#     centroid_2d[:, 0], centroid_2d[:, 1],
#     marker='*', color='green', s=180, edgecolor='black', linewidth=0.9
# )
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.title('UMAP of Participant Embeddings')
# legend_elements_umap = [
#     Line2D([0], [0], marker='x', color='red', markersize=10, linewidth=0, label='Aphasia'),
#     Line2D([0], [0], marker='^', color='orange', markersize=9, markeredgecolor='k', label='Dementia'),
#     Line2D([0], [0], marker='o', color='blue', markersize=8, markeredgecolor='k', label='Control'),
#     Line2D([0], [0], marker='*', color='green', markersize=14, markeredgecolor='black', label='Centroid')
# ]
# plt.legend(handles=legend_elements_umap, loc='best')
# plt.tight_layout()
# plt.savefig('participant_embeddings_umap_shapes_aphasia_dementia_controls_center_update.png',
#             dpi=300, bbox_inches='tight')
# plt.close()

# --------------------------------------------
# OPTIONAL: MDS plot that tries to preserve distances more faithfully
# --------------------------------------------
all_data = np.vstack([participant_embeddings, centroid_np])
dist_matrix = cosdist(all_data, all_data)
mds = MDS(
    n_components=2,
    dissimilarity='precomputed',
    random_state=42,
    n_init=4,
    max_iter=300
)
all_mds_2d = mds.fit_transform(dist_matrix)
mds_2d = all_mds_2d[:-1]
centroid_mds_2d = all_mds_2d[-1].reshape(1, -1)

plt.figure(figsize=(14, 6))
# LEFT: MDS by group
plt.subplot(1, 2, 1)
for i, (x, y) in enumerate(mds_2d):
    group = participant_labels[i]
    if group == 'aphasia':
        plt.scatter(x, y, marker='x', color='red', s=80, linewidths=2)
    elif group == 'dementia':
        plt.scatter(x, y, marker='x', color='black', s=80, linewidths=2)
    else:
        plt.scatter(x, y, marker='o', color='blue', edgecolor="black", s=60, linewidth=0.7)
plt.scatter(
    centroid_mds_2d[:, 0], centroid_mds_2d[:, 1],
    marker='*', color='green', s=180, edgecolor='black', linewidth=1
)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Map of Participant Distances to the Centroid')
legend_elements = [
    Line2D([0], [0], marker='x', color='red', markerfacecolor='none', markeredgecolor='red', markersize=12, label='Aphasia'),
    Line2D([0], [0], marker='x', color='black', markersize=9, markeredgecolor='k', label='Dementia'),
    Line2D([0], [0], marker='o', color='blue', markersize=8, markeredgecolor='black', label='Control'),
    Line2D([0], [0], marker='*', color='green', markersize=14, markeredgecolor='black', label='Centroid')
]
plt.legend(handles=legend_elements, loc='best')

# RIGHT: MDS colored by distance (uncomment if desired)
# plt.subplot(1, 2, 2)
# participant_distances = participant_mean_distances
# dist_min, dist_max = min(participant_distances), max(participant_distances)
# for i, (x, y) in enumerate(mds_2d):
#     group = participant_labels[i]
#     dist = participant_distances[i]
#     if group == 'aphasia':
#         plt.scatter(x, y, c=dist, cmap='viridis',
#                     marker='x', s=90, linewidths=2, vmin=dist_min, vmax=dist_max)
#     elif group == 'dementia':
#         plt.scatter(x, y, c=dist, cmap='viridis',
#                     marker='x', s=80, linewidth=0.7,
#                     vmin=dist_min, vmax=dist_max)
#     else:
#         plt.scatter(x, y, c=dist, cmap='viridis',
#                     marker='o', s=70, edgecolor='k', linewidth=0.7,
#                     vmin=dist_min, vmax=dist_max)
# plt.scatter(
#     centroid_mds_2d[:, 0], centroid_mds_2d[:, 1],
#     marker='*', color='white', s=180, edgecolor='black', linewidth=1
# )
# plt.xlabel('MDS Dimension 1')
# plt.ylabel('MDS Dimension 2')
# plt.title('MDS Colored by Cosine Distance to Centroid')
# cbar = plt.colorbar()
# cbar.set_label('Cosine Distance to Centroid')

plt.tight_layout()
plt.savefig('participant_embeddings_mds_shapes_aphasia_dementia_controls_center_update.png',
            dpi=300, bbox_inches='tight')
plt.close()