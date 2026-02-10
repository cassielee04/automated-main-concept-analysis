from pathlib import Path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.normalize_utterances import normalize_utterance, normalize_utterances
from src.ciu import calculate_cinderella_ciu, count_ciu_nouns, get_ciu_nouns  # Adjust to relative if needed: from .ciu import ...
from src.features import get_stanza_pipeline
from src.segment_utterance import segment_utterances
from src.fillers import count_fillers, count_words_in_tokens
from src.save_cinderella_embeddings import load_embeddings_and_centroid
import torch
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer, util
# from .ciu import calculate_cinderella_ciu, count_ciu_nouns, get_ciu_nouns
# from .segment_utterance import segment_utterances
# from .fillers import count_fillers, count_words_in_tokens
# from .save_cinderella_embeddings import load_embeddings_and_centroid
import re
# from .normalize_utterances import normalize_utterance, normalize_utterances
from itertools import combinations

class MainConceptAnalyzerNormalize:
    
    """A class to analyze main concepts and topic switching in text using sentence embeddings."""
    
    def __init__(self, config_path="../config/story_config.yml", embeddings_file="../config/cinderella_mainconcept_embeddings.pkl", 
                 embed_id="sentence-transformers/all-mpnet-base-v2", global_cutoff=0.8289):
        """
        Initialize the analyzer with configuration, embeddings, and model.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            embeddings_file (str): Path to the pickled embeddings file.
            embed_id (str): SentenceTransformer model ID.
            global_cutoff (float): Cosine distance threshold for main concept matching.
        """
        # Set root directory and update sys.path
        self.ROOT = Path(__file__).resolve().parent
        if str(self.ROOT) not in sys.path:
            sys.path.insert(0, str(self.ROOT))
        
        # Load configuration
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load embeddings and related data
        self.concept_embeds, self.centroid, self.concepts, self.avg_concept_sim, self.cutoff, self.mean_dist, self.std_dist = load_embeddings_and_centroid(embeddings_file)
        self.concept_embeds = self.concept_embeds.to(self.device)
        self.centroid = self.centroid.to(self.device)
        self.ciu_nouns = get_ciu_nouns(narrative_type='cinderella')
        # Initialize SentenceTransformer model
        self.embedder = SentenceTransformer(embed_id, device=self.device)
        
        # Initialize tracking for main concepts
        self.unique_matched_concepts = set()
        self.total_mainconcepts = []
        self.previous_utterances = []
        
        # Store global cutoff
        self.global_cutoff = global_cutoff
    
    def get_mainconcept_match(self, utterances, normalize_utterances, return_score=False):
        """
        Check if utterances semantically match any main concept using cosine distance to centroid.
        
        Args:
            utterances: List of input sentences or a single sentence.
            return_score (bool): If True, return similarity score and matched concept.
        
        Returns:
            pd.DataFrame: DataFrame with columns for utterance, matched concept, scores, and match status.
        """
        if isinstance(utterances, str):
            utterances = [utterances]
        if isinstance(normalize_utterances, str):
            normalize_utterances = [normalize_utterances]
        # Encode utterance embeddings
        utter_embeds = self.embedder.encode(normalize_utterances, convert_to_tensor=True, normalize_embeddings=True)
        
        results = []
        for i, emb in enumerate(utter_embeds):
            # utterance = normalized_utterances[i]  # or however you're indexing the original text
            # if not utterance.strip():
            #     similarity = None  # or assign a default, e.g., 0.0, or skip it entirely
            #     is_match = False
            #     print(f"Skipping empty utterance at index {i}")
            #     continue  # Optional: skip adding it to result lists
            dist = 1 - util.cos_sim(emb.unsqueeze(0), self.centroid).squeeze().item()
            sims = util.cos_sim(emb.unsqueeze(0), self.concept_embeds)[0]
            best_idx = int(sims.argmax())
            similarity = 1 - dist
            is_match = dist <= self.global_cutoff
            
            matched_concept = None
            is_repeated = False
            utterance_lower = normalize_utterances[i].lower()
            utterance_tokens = [re.sub(r'[^\w\s]', '', token) for token in utterance_lower.split()]
            utterance_tokens = [token for token in utterance_tokens if token]
            num_ciu_nouns = count_ciu_nouns('cinderella', normalize_utterances[i])[0]
            if is_match:
                
                matched_concept = self.concepts[best_idx]
                is_repeated = self.is_repeated_utt(normalize_utterances[i], add_to_set=True)
                self.count_repeated_mainconcept_by_idx(best_idx, add_to_set=True)
            # else:
            #     if any(token in self.ciu_nouns for token in utterance_tokens) or num_ciu_nouns > 0:
            #         is_match = True
            #         matched_concept = self.concepts[best_idx]
            #         is_repeated = self.is_repeated_utt(normalize_utterances[i], add_to_set=True)
            #         self.count_repeated_mainconcept_by_idx(best_idx, add_to_set=True)

            results.append({
                "utterance": utterances[i],
                "normalized_utterance": normalize_utterances[i],
                "matched_concept": matched_concept,
                "similarity_score": round(similarity, 4) if return_score else None,
                "distance_to_centroid": round(dist, 4) if return_score else None,
                "is_main_concept": bool(matched_concept),
                "is_repeated": is_repeated
            })
        
        return pd.DataFrame(results)
    
    def count_repeated_mainconcept_by_idx(self, best_idx: int, add_to_set: bool = True) -> bool:
        """
        Track whether a main concept has been seen before.
        
        Args:
            best_idx (int): Index of the matched concept.
        
        Returns:
            bool: True if the concept is repeated, False if it's new.
        """
        mc = self.concepts[best_idx]
        is_repeated = mc in self.unique_matched_concepts
        if add_to_set:
            self.unique_matched_concepts.add(mc)
            self.total_mainconcepts.append(mc)
        #return unique_matched_concepts, get_total_mainconcepts

    
    def is_repeated_utt(self, utterance: str, add_to_set: bool = True) -> bool:
        """
        Check if the current utterance is repeated by comparing its cosine similarity
        with previous utterances. Returns True if similarity > 0.70 for any previous utterance.
        
        Args:
            utterance (str): The utterance to check.
            add_to_set (bool): If True, add the utterance to the history if not repeated.
        
        Returns:
            bool: True if the utterance is repeated, False if it's new.
        """
        # Encode the current utterance
        current_embedding = self.embedder.encode(utterance, convert_to_tensor=True, normalize_embeddings=True)
        
        # Compare with previous utterance embeddings
        for prev_utt, prev_emb in self.previous_utterances:
            similarity = util.cos_sim(current_embedding, prev_emb).item()
            # print(f"Comparing '{utterance}' with previous '{prev_utt}' | Similarity: {similarity}")
            if similarity > 0.8:
                # print("Repeated utterance detected:", utterance, " | Similarity:", similarity, " | Previous utterance:", prev_utt)
                return True
        
        # If not repeated and add_to_set is True, store the utterance and its embedding
        if add_to_set:
            self.previous_utterances.append((utterance, current_embedding))
        return False
        
    def get_unique_mainconcepts(self):
        return self.unique_matched_concepts

    def get_total_unique_mainconcepts(self) -> int:
        """Return the count of unique main concepts."""
        return len(self.unique_matched_concepts)
    
    def get_total_mainconcepts(self) -> int:
        """Return the total count of main concepts (including repeats)."""
        return len(self.total_mainconcepts)
    
    def is_topic_switching(self, prev_utt: str, curr_utt: str) -> bool:
        """
        Check if there's a topic switch between two utterances based on cosine similarity.
        
        Args:
            prev_utt (str): Previous utterance.
            curr_utt (str): Current utterance.
        
        Returns:
            bool: True if a topic switch is detected (similarity < avg_concept_sim).
        """
        cleaned_prev_utt = normalize_utterance(prev_utt)
        cleaned_curr_utt = normalize_utterance(curr_utt)
        emb_prev = self.embedder.encode(cleaned_prev_utt, convert_to_tensor=True, normalize_embeddings=True)
        emb_curr = self.embedder.encode(cleaned_curr_utt, convert_to_tensor=True, normalize_embeddings=True)
        similarity = util.cos_sim(emb_prev, emb_curr).item()
        return similarity < 0.2

    def order_difference_ratio(self, concepts_seq):
        """Compute order difference ratio for the sequence of concept indices."""
        n = len(concepts_seq)
        if n < 2:
            return 0.0

        violations = sum(
            1 for i, j in combinations(range(n), 2)
            if concepts_seq[i] > concepts_seq[j]
        )

        total_pairs = n * (n - 1) / 2
        return violations / total_pairs
    
    def reset_tracking(self):
        self.unique_matched_concepts = set()
        self.total_mainconcepts = []
        self.previous_utterances = []
 
    def score_story_sequence(self, utterances, normalize_utts=None, return_score=False):
        """
        Score the story sequence by mapping utterances to concept indices and computing order difference ratio.
        
        Args:
            utterances: List of input utterances.
            normalize_utterances: List of normalized utterances (optional; if None, uses utterances).
            return_score (bool): If True, include the order difference ratio in the output.
        
        Returns:
            tuple: (matched_concepts: list of int, score: float if return_score else None)
        """
        if normalize_utts is None:
            normalize_utts = normalize_utterances(utterances)

        # Get main concept matches
        df_matches = self.get_mainconcept_match(utterances, normalize_utts, return_score=True)

        matched_concepts = []
        for _, row in df_matches.iterrows():
            matched_concept = row['matched_concept']
            if matched_concept is not None:
                # Map concept string to 1-based index
                try:
                    idx = self.concepts.index(matched_concept) + 1
                    matched_concepts.append(idx)
                except ValueError:
                    # If not found (edge case), skip or handle
                    continue
            else:
                # No match, perhaps append 0 or skip; here we skip for sequence
                continue

        score = self.order_difference_ratio(matched_concepts) if return_score else None
        return matched_concepts, score



# Example usage
if __name__ == "__main__":
    analyzer = MainConceptAnalyzerNormalize(global_cutoff = 0.8289)
    # text = ("He tries it on their feet and their feet are way too big,.")
    # utterances = read.csv("../data/data/matching_mainconcept_dementia_predicted_output.csv")["utterance"].tolist()
    
    # test_utterances = segment_utterances(utterances)
    # normalize_utterances = normalize_utterances(test_utterances)
    # df = analyzer.get_mainconcept_match(test_utterances,normalize_utterances, return_score=True)

    df_csv = pd.read_csv("../data/data/Matching Concept Check - Controls - All.csv")
    utterances = df_csv["utterances"].dropna().astype(str).tolist()

    # --- Segment the utterances ---
    test_utterances = utterances

    # --- Normalize utterances ---
    normalized_utts = normalize_utterances(test_utterances)

    # --- Get Main Concept matching results ---
    df = analyzer.get_mainconcept_match(
        test_utterances,
        normalized_utts,
        return_score=True
    )

    df.to_csv("../data/data/base_matching_mainconcept_aphasia_controls_predicted_output.csv", index=False, encoding="utf-8")
    # print(df.to_string(index=False))
    # print(f"Average concept similarity: {analyzer.avg_concept_sim}")
    # print(f"Topic switch score: {score}")
    # print(f"Total unique main concepts: {analyzer.get_total_unique_mainconcepts()}")

    # print(f"Total main concepts (including repeats): {analyzer.get_total_mainconcepts()}")