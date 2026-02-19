import torch
from sentence_transformers import SentenceTransformer, util
import yaml
import pathlib
import os
import pickle
import csv
from pathlib import Path
import sys
from src.normalize_utterances import normalize_utterances
import pandas as pd
from sentence_transformers.util import cos_sim

def save_embeddings_and_centroid(model_name, config_path, output_path, cutoff=None, mean_dist=None, std_dist=None):
    # Load configuration
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
    concepts = cfg["stories"][0]["concepts"]
    
    # Load model
    embedder = SentenceTransformer(model_name)
    
    # Compute embeddings and centroid
    concept_embeds = embedder.encode(concepts, convert_to_tensor=True, normalize_embeddings=True)
    centroid = torch.mean(concept_embeds, dim=0, keepdim=True)
    
    # Compute average similarity
    def find_average_similarity(utterances: list, embeds: torch.Tensor) -> float:
        if len(utterances) < 2:
            return 0.0  # Not enough data

        sum_sim = 0.0
        count = 0
        for i in range(1, len(utterances)):
            sim = util.cos_sim(embeds[i - 1].unsqueeze(0), embeds[i].unsqueeze(0)).item()
            sum_sim += sim
            count += 1
            #print(f"Cosine similarity between [{i-1}] '{utterances[i - 1]}' and [{i}] '{utterances[i]}': {sim:.4f}")
        
        avg_similarity = sum_sim / count
        #print(f"\n📊 Average Neighboring Cosine Similarity: {avg_similarity:.4f}")
        return avg_similarity
    
    avg_concept_sim = find_average_similarity(concepts, concept_embeds)
    
    # Save embeddings, centroid, concepts, average similarity, and optional cutoff/mean_dist/std_dist
    save_data = {
        'concept_embeds': concept_embeds.cpu().numpy(),
        'centroid': centroid.cpu().numpy(),
        'concepts': concepts,
        'avg_concept_sim': avg_concept_sim
    }
    if cutoff is not None:
        save_data['cutoff'] = cutoff
    if mean_dist is not None:
        save_data['mean_dist'] = mean_dist
    if std_dist is not None:
        save_data['std_dist'] = std_dist
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Embeddings, centroid, and average similarity saved to {output_path}")

def load_embeddings_and_centroid(input_path):
    # Load embeddings, centroid, concepts, average similarity, cutoff, mean_dist, and std_dist if available
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    concept_embeds = torch.from_numpy(data['concept_embeds'])
    centroid = torch.from_numpy(data['centroid'])
    concepts = data['concepts']
    avg_concept_sim = data['avg_concept_sim']
    cutoff = data.get('cutoff', None)  # Default to None if not present
    mean_dist = data.get('mean_dist', None)  # Default to None if not present
    std_dist = data.get('std_dist', None)  # Default to None if not present
    return concept_embeds, centroid, concepts, avg_concept_sim, cutoff, mean_dist, std_dist

def update_cutoff(input_path, model_name, new_utterances, cutoff_multiplier=1.0, output_path=None, device='cpu', stats_csv="batch_stats_patients.csv"):
    """
    Update the cutoff, mean_dist, and std_dist using new utterances against the existing centroid.
    Processes utterances in batches of 20, records per-batch and cumulative stats (for new utterances) to a separate DataFrame/CSV.
    Computes combined historical stats for saving to pkl. Centroid and other data remain unchanged.
    Args:
    - input_path: Path to existing .pkl file with concept_embeds, centroid, etc.
    - model_name: SentenceTransformer model name (e.g., "sentence-transformers/all-mpnet-base-v2").
    - new_utterances: List of new utterances (strings) to compute distances from.
    - cutoff_multiplier: Multiplier for std_dev to determine relevance cutoff (default 1.0).
    - output_path: Path to save updated .pkl (if None, overwrites input_path).
    - device: Device for computation (e.g., 'cuda' or 'cpu').
    - stats_csv: Path to save batch stats CSV (default "batch_stats.csv").
    Returns:
    - Tuple: (concept_embeds, centroid, concepts, avg_concept_sim, cutoff, mean_dist, std_dist)
    """
    if not new_utterances:
        print("No new utterances provided; no update performed.")
        return load_embeddings_and_centroid(input_path)

    # Load existing data
    concept_embeds, centroid, concepts, avg_concept_sim, existing_cutoff, existing_mean_dist, existing_std_dist = load_embeddings_and_centroid(input_path)
    
    # Load existing num_utterances
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    existing_num_utterances = data.get('num_utterances', 0)

    # Load model and move to device
    model = SentenceTransformer(model_name)
    model.to(device)
    centroid = centroid.to(device)
    
    cleaned_utterances = normalize_utterances(new_utterances)
    
    # Prepare DataFrame to record per-batch and cumulative stats every 20 utterances
    batch_size = 20
    stats_list = []
    running_dists = []
    batch_id = 0
    
    # Process in batches of 20 utterances
    for batch_start in range(0, len(cleaned_utterances), batch_size):
        batch_utterances = cleaned_utterances[batch_start:batch_start + batch_size]
        if not batch_utterances:
            break
        
        # Encode batch utterances
        batch_embeds = model.encode(batch_utterances, convert_to_tensor=True, normalize_embeddings=True, device=device)
        
        # Compute cosine distances from batch utterances to centroid
        cosine_dists = 1 - cos_sim(batch_embeds, centroid).squeeze()
        mean_dist = cosine_dists.mean().item()
        std_dist = cosine_dists.std().item()
        cutoff = mean_dist + cutoff_multiplier * std_dist
        
        print(f"Batch {batch_id + 1} (utterances {batch_start+1}-{min(batch_start + batch_size, len(cleaned_utterances))}): "
              f"Mean distance: {mean_dist:.4f}, Std distance: {std_dist:.4f}, Cutoff: {cutoff:.4f}")
        
        # Accumulate distances for cumulative stats on new utterances
        running_dists.append(cosine_dists.detach().cpu())
        all_so_far_dists = torch.cat(running_dists, dim=0)
        cum_mean = all_so_far_dists.mean().item()
        cum_std = all_so_far_dists.std().item()
        cum_cutoff = cum_mean + cutoff_multiplier * cum_std
        
        # Record batch stats including cumulative for new so far
        stats_list.append({
            'batch_id': batch_id + 1,
            'cum_mean_dist': cum_mean,
            'cum_std_dist': cum_std,
            'cum_cutoff': cum_cutoff
        })
        batch_id += 1
    
    # Create and save batch stats DataFrame to CSV
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(stats_csv, index=False)
        print(f"Batch stats saved to {stats_csv}")
    else:
        print("No batches processed.")
    
    # Compute overall stats for all new utterances (for combining with historical)
    new_n = len(cleaned_utterances)
    if running_dists:
        all_dists = torch.cat(running_dists, dim=0)
        overall_mean_dist = all_dists.mean().item()
        overall_std_dist = all_dists.std().item()
        print(f"New utterances overall: Mean distance: {overall_mean_dist:.4f}, Std distance: {overall_std_dist:.4f}")
    else:
        overall_mean_dist, overall_std_dist = None, None
        print("No new utterances processed after cleaning.")
    
    # Combine with existing historical stats cumulatively
    if existing_num_utterances == 0:
        combined_mean_dist = overall_mean_dist
        combined_std_dist = overall_std_dist
        combined_n = new_n  # 
    else:
        old_mean = existing_mean_dist
        old_var = existing_std_dist ** 2
        new_mean = overall_mean_dist
        new_var = overall_std_dist ** 2
        old_n = existing_num_utterances
        combined_n = old_n + new_n
        combined_mean = (old_mean * old_n + new_mean * new_n) / combined_n
        
        # Combined sample variance (ddof=1)
        var_a = old_var * (old_n - 1) if old_n > 1 else 0.0
        var_b = new_var * (new_n - 1) if new_n > 1 else 0.0
        diff_var = old_n * new_n * (old_mean - new_mean) ** 2 / combined_n
        combined_var = (var_a + var_b + diff_var) / (combined_n - 1) if combined_n > 1 else 0.0
        combined_std = math.sqrt(combined_var)
        
        combined_mean_dist = combined_mean
        combined_std_dist = combined_std
    
    combined_cutoff = combined_mean_dist + cutoff_multiplier * combined_std_dist
    print(f"Combined historical: Mean distance: {combined_mean_dist:.4f}, Std distance: {combined_std_dist:.4f}, Cutoff: {combined_cutoff:.4f}")
    
    # Save updated data with combined cutoff, mean_dist, and std_dist (centroid unchanged)
    save_path = output_path or input_path
    with open(save_path, 'wb') as f:
        pickle.dump({
            'concept_embeds': concept_embeds.cpu().numpy(),
            'centroid': centroid.cpu().numpy(),
            'concepts': concepts,
            'avg_concept_sim': avg_concept_sim,
            'cutoff': combined_cutoff,
            'mean_dist': combined_mean_dist,
            'std_dist': combined_std_dist,
            'num_utterances': combined_n
        }, f)
    print(f"Combined cutoff, mean_dist, and std_dist updated and saved to {save_path}")
    
    return concept_embeds, centroid, concepts, avg_concept_sim, combined_cutoff, combined_mean_dist, combined_std_dist

# Example usage
if __name__ == "__main__":
    EMBED_ID = "sentence-transformers/all-mpnet-base-v2"
    CONFIG_PATH = "../config/story_config.yml"
    EMBEDDINGS_FILE = "../config/cinderella_mainconcept_embeddings.pkl"

    save_embeddings_and_centroid(EMBED_ID, CONFIG_PATH, EMBEDDINGS_FILE, cutoff=None, mean_dist=None, std_dist=None)

    csv_files = [
        Path("../data/utterances_aphasia_output.csv"),
        # Path("../data/utterances_controls_output.csv"),
        Path("../data/utterances_dementia_output.csv"),

    ]

    all_utterances = []   # ← put it here

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            sys.exit(1)

        print(f"Processing {csv_path.name}...")
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_utterances.append(row["utterance"])

        
    # Save embeddings if not already saved (initial save won't have cutoff/mean_dist/std_dist)
    if not os.path.exists(EMBEDDINGS_FILE):
        save_embeddings_and_centroid(EMBED_ID, CONFIG_PATH, EMBEDDINGS_FILE)
    
    
    # Update cutoff, mean_dist, and std_dist with new utterances (using GPU if available); centroid unchanged
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    updated_concept_embeds, updated_centroid, updated_concepts, updated_avg_sim, updated_cutoff, updated_mean_dist, updated_std_dist = update_cutoff(
        EMBEDDINGS_FILE, EMBED_ID, all_utterances, cutoff_multiplier=1.0, device=device
    )
    
    print(f"Updated cutoff: {updated_cutoff:.4f}, mean_dist: {updated_mean_dist:.4f}, std_dist: {updated_std_dist:.4f}")