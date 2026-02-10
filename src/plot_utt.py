import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import umap
import torch
import numpy as np
import csv
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))
from save_cinderella_embeddings import load_embeddings_and_centroid
from fillers import count_fillers, remove_fillers

# Load YAML config
config_path = Path(__file__).resolve().parent.parent / "config" / "story_config.yml"
try:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    print(f"Error: Config file not found at {config_path}")
    sys.exit(1)

ciu_nouns = cfg["stories"][0]["ciu_nouns"]
main_concepts = cfg["stories"][0]["concepts"]


if torch.cuda.is_available():
    print("device 0 name:", torch.cuda.get_device_name(0))

# Force a clear device choice
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)
model.to(device)  # ensure modules are on the target device

# # Initialize the model
# model = SentenceTransformer("all-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

def plot_semantic_projection(main_concepts, utterances, concept_embeds, centroid, ciu_nouns, batch_start, batch_end, output_file):
    try:
        if not utterances:
            print("No utterances provided for plotting.")
            return

        # Determine the device of the model
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")

        # Move loaded tensors to the same device as the model
        concept_embeds = concept_embeds.to(device)
        centroid = centroid.to(device)

        original_utterances = utterances.copy()
        cleaned_utterances = []
        print("Cleaning utterances by removing fillers:")
        for i, utt in enumerate(utterances):
            result = count_fillers(utt)
            detected = result['detected_fillers']
            cleaned = remove_fillers(utt, detected)
            cleaned_utterances.append(cleaned)
            #print(f"  Utterance {batch_start + i}: Original: \"{utt}\" -> Cleaned: \"{cleaned}\" (Removed: {detected})")

        # Use cleaned utterances for further processing
        utterances = cleaned_utterances
        # Encode new utterances
        utterance_embeds = model.encode(utterances, convert_to_tensor=True, normalize_embeddings=True, device=device)
        
        # Compute cosine distances from utterances to centroid
        cosine_dists = 1 - util.cos_sim(utterance_embeds, centroid).squeeze()
        mean_dist = cosine_dists.mean().item()
        std_dist = cosine_dists.std().item()
        cutoff = mean_dist + 1.0 * std_dist
        print(f"Mean distance: {mean_dist:.4f}, Std distance: {std_dist:.4f}, Cutoff: {cutoff:.4f}")

        # Label utterances
        labels = []
        for i, (dist, utterance) in enumerate(zip(cosine_dists, utterances)):
            if dist <= cutoff:
                labels.append("Likely Main Concept")
            else:
                labels.append("No Main Concept")
        colors = ["blue" if label == "Likely Main Concept" else "red" for label in labels]

        # Print utterance statuses
        for i, (dist, utterance) in enumerate(zip(cosine_dists, utterances)):
            status = "✅ Likely main concept" if dist <= cutoff else "❌ NOT a main concept"
            #print(f"utterance {batch_start + i} [{status}] Dist={dist:.4f} — \"{utterance}\"")

        # Stack all vectors for projection
        all_vectors = torch.cat([concept_embeds, utterance_embeds, centroid], dim=0).cpu().numpy()

        # UMAP projection
        reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
        X_2d = reducer.fit_transform(all_vectors)

        # Split the projected points
        concept_pts = X_2d[:len(main_concepts)]
        utterance_pts = X_2d[len(main_concepts):-1]
        centroid_pt = X_2d[-1]

        # Plot
        plt.figure(figsize=(10, 8))
        plt.title(f"Semantic Projection with UMAP (Utterances {batch_start}-{batch_end})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        # Plot concept points with text labels
        for i, (x, y) in enumerate(concept_pts):
            short_label = main_concepts[i][:40] + "..." if len(main_concepts[i]) > 40 else main_concepts[i]
            plt.scatter(x, y, color='gray', s=60, alpha=0.7, label='Main Concepts' if i == 0 else "")
            plt.text(x, y, short_label, fontsize=7, ha='left', va='center', color='dimgray')

        # Plot utterance points without text labels
        for i, (x, y) in enumerate(utterance_pts):
            plt.scatter(x, y, color=colors[i], s=80, label=labels[i] if i == labels.index(labels[i]) else "")

        # Plot centroid
        plt.scatter(centroid_pt[0], centroid_pt[1], color='black', marker='X', s=100, label='Centroid')

        # Finalize plot
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Close the plot to free memory
        plt.close()

    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    # Load utterances
    csv_path = Path("../data/utterances_aphasia_output.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    utterances = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            utterances.append(row["utterance"])

    # Load embeddings
    EMBEDDINGS_FILE = Path(__file__).resolve().parent.parent / "config" / "cinderella_mainconcept_embeddings.pkl"
    concept_embeds, centroid, concepts, avg_concept_sim, cutoff, mean_dist, std_dist= load_embeddings_and_centroid(EMBEDDINGS_FILE)

    # Create plots directory
    #os.makedirs("plots", exist_ok=True)
    #4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600, 6900, 7200, 7500, 7650
    # Process utterances in batches
    cumulative_ranges = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200]
    batch_size = 300
    for end in cumulative_ranges:
        batch_utterances = utterances[:end]
        output_file = f"plots/utterance_test_0_{end}_one.png"
        print(f"\nGenerating plot for utterances 0 to {end}")
        plot_semantic_projection(
            main_concepts,
            batch_utterances,
            concept_embeds,
            centroid,
            ciu_nouns,
            batch_start=0,
            batch_end=end,
            output_file=output_file
        )