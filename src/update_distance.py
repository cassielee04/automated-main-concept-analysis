import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import csv
from pathlib import Path
import sys
import torch
from datetime import datetime

# Set up paths relative to this script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from save_cinderella_embeddings import load_embeddings_and_centroid

# File paths
pickle_path = Path(__file__).resolve().parent.parent / "config" / "cinderella_mainconcept_embeddings.pkl"
csv_path = Path("../data/utterances_aphasia_output.csv")
stats_path = Path("../data/stats.csv")

def update_pickle_with_new_utterances(pickle_path, new_utterance_embeds=None, reset=False):
    """
    Compute cosine distances and statistics for a batch of utterance embeddings,
    matching the behavior of plot_utt.py's plot_semantic_projection function.
    Updates the pickle file with new embeddings and distances for cumulative stats.
    Args:
        pickle_path (str): Path to the pickle file.
        new_utterance_embeds (torch.Tensor): New utterance embeddings.
        reset (bool): If True, reset utterance-related data in the pickle file.
    Returns:
        dict: Updated mean_dist, std_dist, cutoff, num_new_utterances, total_utterances.
    """
    # Get number of new utterances
    num_new_utterances = new_utterance_embeds.shape[0] if new_utterance_embeds is not None else 0

    # Load existing data from pickle
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        concept_embeds = data['concept_embeds']  # Unchanged
        centroid = data['centroid']  # Unchanged
        concepts = data['concepts']  # Unchanged
        avg_concept_sim = data['avg_concept_sim']  # Unchanged
        dim = concept_embeds.shape[1]
        if 'utterance_embeds' not in data or data['utterance_embeds'].size == 0:
            utterance_embeds = np.empty((0, dim))
        else:
            utterance_embeds = data['utterance_embeds']
            if utterance_embeds.shape[1] != dim:
                raise ValueError("Saved utterance embeds dimension mismatch")
        total_utterances = data.get('total_utterances', 0)
        distances = data.get('distances', np.array([], dtype=float))
    except FileNotFoundError:
        raise FileNotFoundError("Pickle file not found. Cannot add new utterances without existing concepts.")

    # Handle reset
    if reset:
        utterance_embeds = np.empty((0, dim))
        distances = np.array([], dtype=float)
        total_utterances = 0
        mean_dist = 0.0
        std_dist = 0.0
        cutoff = 0.0
        # Save reset data back to pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'concept_embeds': concept_embeds,
                'centroid': centroid,
                'concepts': concepts,
                'avg_concept_sim': avg_concept_sim,
                'utterance_embeds': utterance_embeds,
                'mean_dist': mean_dist,
                'std_dist': std_dist,
                'cutoff': cutoff,
                'total_utterances': total_utterances,
                'distances': distances
            }, f)
        return {
            'centroid': centroid,
            'avg_concept_sim': avg_concept_sim,
            'concept_embeds': concept_embeds,
            'concepts': concepts,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'cutoff': cutoff,
            'num_new_utterances': 0,
            'total_utterances': total_utterances,
            'distances': distances
        }

    # If no new utterances, return current stats
    if new_utterance_embeds is None or num_new_utterances == 0:
        mean_dist = np.mean(distances) if len(distances) > 0 else 0.0
        std_dist = np.std(distances, ddof=1) if len(distances) > 1 else 0.0
        cutoff = mean_dist + 1.0 * std_dist
        return {
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'cutoff': cutoff,
            'num_new_utterances': 0,
            'total_utterances': total_utterances
        }

    # Validate dimensionality
    if new_utterance_embeds.shape[1] != dim:
        raise ValueError("Dimensionality of new utterance embeddings does not match concept embeddings.")

    # Move tensors to the same device
    device = new_utterance_embeds.device
    centroid_torch = torch.from_numpy(centroid).to(device)

    # Calculate cosine distances for new utterances using util.cos_sim
    cosine_dists = 1 - util.cos_sim(new_utterance_embeds, centroid_torch).squeeze()

    # Append new data to existing
    new_dists = cosine_dists.cpu().numpy()
    new_embeds_np = new_utterance_embeds.cpu().numpy()
    if len(utterance_embeds) == 0:
        utterance_embeds = new_embeds_np
    else:
        utterance_embeds = np.vstack((utterance_embeds, new_embeds_np))
    if len(distances) == 0:
        distances = new_dists
    else:
        distances = np.concatenate((distances, new_dists))
    total_utterances += num_new_utterances

    # Compute cumulative mean and standard deviation
    mean_dist = np.mean(distances) if len(distances) > 0 else 0.0
    std_dist = np.std(distances, ddof=1) if len(distances) > 1 else 0.0
    cutoff = mean_dist + 1.0 * std_dist

    # Save updated data back to pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'concept_embeds': concept_embeds,
            'centroid': centroid,
            'concepts': concepts,
            'avg_concept_sim': avg_concept_sim,
            'utterance_embeds': utterance_embeds,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'cutoff': cutoff,
            'total_utterances': total_utterances,
            'distances': distances
        }, f)

    return {
        'mean_dist': mean_dist,
        'std_dist': std_dist,
        'cutoff': cutoff,
        'num_new_utterances': num_new_utterances,
        'total_utterances': total_utterances
    }

# Example usage
if __name__ == "__main__":
    # Load existing total_utterances
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        total_utterances = data.get('total_utterances', 0)
    except FileNotFoundError:
        print("Pickle file not found. Cannot add new utterances.")
        sys.exit(1)

    # Load utterances
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    utterances = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            utterances.append(row["utterance"])

    new_utterances = utterances[total_utterances:]
    if not new_utterances:
        print("No new utterances to process.")
        #sys.exit(0)

    # Initialize the model
    model = SentenceTransformer("all-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

    # Process in batches of 20
    # batch_size = 20
    # for i in range(0, len(new_utterances), batch_size):
    #     batch_utterances = new_utterances[i:i + batch_size]
    #     batch_embeds = model.encode(batch_utterances, convert_to_tensor=True, normalize_embeddings=True, device=model.device)
    #     result = update_pickle_with_new_utterances(pickle_path, batch_embeds)

    #     print(f"Processed batch starting at utterance index {total_utterances + i}, num in batch: {len(batch_utterances)}")
    #     print(f"Total Utterances: {result['total_utterances']}")
    #     print(f"Updated Mean Distance: {result['mean_dist']:.4f}")
    #     print(f"Updated Std Distance: {result['std_dist']:.4f}")
    #     print(f"Updated Cutoff: {result['cutoff']:.4f}")

    #     # Log to stats CSV
    #     with open(stats_path, 'a', newline='', encoding='utf-8') as f:
    #         writer = csv.writer(f)
    #         if i == 0 and (not stats_path.exists() or stats_path.stat().st_size == 0):
    #             writer.writerow(['timestamp', 'total_utterances', 'mean_dist', 'std_dist', 'cutoff'])
    #         timestamp = datetime.now().isoformat()
    #         writer.writerow([timestamp, result['total_utterances'], '{:.4f}'.format(result['mean_dist']),
    #                          '{:.4f}'.format(result['std_dist']), '{:.4f}'.format(result['cutoff'])])

    # Uncomment for reset (use with caution)
    result = update_pickle_with_new_utterances(pickle_path, reset=True)
    print(f"Reset - Total Utterances: {result['total_utterances']}")
    print(f"Reset Mean Distance: {result['mean_dist']:.4f}")
    print(f"Reset Std Distance: {result['std_dist']:.4f}")
    print(f"Reset Cutoff: {result['cutoff']:.4f}")