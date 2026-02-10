import re
import spacy
from features import get_stanza_pipeline
# Load spacy model for POS tagging
# If you don't have spacy installed: pip install spacy
# If you don't have the model: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class ReferentChecker:
    def __init__(self):
        # Hashmap for entities and their aliases/pronouns
        self.entities = {
            "dad": ["he", "his", "him"],
            "father": ["he", "his", "him"],
            "cinderella": ["she", "her"],
            "stepmother": ["she", "her"],
            "stepsisters": ["they", "them", "their"],
            "stepmother and stepsisters": ["they", "their", "them"],
            "sisters": ["they", "them", "their"],
            "girl": ["she", "her"],
            "fairy godmother": ["godmother", "she", "her"],
            "godmother": ["she", "her"],
            "prince": ["he", "him", "his"],
            "princess": ["she", "her"],
            "king": ["king", "he", "him"],
            "house": ["it"],
            "cinderella and prince": ["they", "them", "their"],
            "slipper": ["it"],
            "slippers": ["them", "they", "their"],
            "shoe": ["it"],
            "shoes": ["they", "them", "their"],
            "ball": ["it"],
            "party": ["it"],
            "dress": ["it"],
            "clothes": ["they", "them", "their"],
            "princess": ["she", "her"],
            "mice": ["they", "them", "their"],
            "pumpkin": ["it"],
            "horse": ["it"],
            "horses": ["they", "them", "their"],
            "carriage": ["it"],
            "coach": ["it"]
        }
    
    def extract_pronouns_with_positions(self, utterance):
        """Extract pronouns from an utterance with their positions"""
        pronouns = ["he", "his", "him", "she", "her", "they", "them", "their", "it"]
        found_pronouns = []
        words = utterance.lower().split()
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in pronouns:
                char_pos = utterance.lower().find(clean_word)
                found_pronouns.append((clean_word, char_pos))
        
        return found_pronouns  # Add this line  `
         # Must be at module level
    def has_missing_subject(self, utterance: str) -> bool:
        """
        Return True if the sentence contains a main verb (root) but no
        grammatical subject (nsubj / nsubj:pass) *before* that verb.
        Excludes imperative clauses like "Go away!".
        """
        # Initialize Stanza pipeline (lazy-loaded as instance attribute)
        if not hasattr(self, 'stanza_nlp'):
            self.stanza_nlp = get_stanza_pipeline()

        # Process the utterance
        doc = self.stanza_nlp(utterance)
        if not doc.sentences:
            return False  # No sentences → can’t be missing-subject

        sentence = doc.sentences[0]  # Assume single sentence; extend if multi-sentence
        words = sentence.words

        # 1. Find the main verb/AUX (root)
        root_idx = next(
            (i for i, word in enumerate(words) if word.deprel == 'root' and word.upos in ['VERB', 'AUX']),
            None
        )
        if root_idx is None:
            return False  # No root verb/AUX → can’t be missing-subject

        root_word = words[root_idx]

        # 2. Is the clause imperative? (VERB with Mood=Imp)
        feats = {}
        if root_word.feats:
            feats = {kv.split('=')[0]: kv.split('=')[1] for kv in root_word.feats.split('|')}
        if root_word.upos == 'VERB' and feats.get('Mood') == 'Imp':
            return False  # Imperatives are OK

        # 3. Is there a subject (nsubj/nsubj:pass/expl) linked to the root *before* it?
        root_head = root_idx + 1  # 1-indexed
        children = [w for w in words if w.head == root_head and w.deprel in {'nsubj', 'nsubj:pass', 'expl'}]
        has_subject = any((w.id - 1) < root_idx for w in children)  # 0-indexed position < root
        return not has_subject
            
    def has_missing_referent(self, pronoun, previous_utterances, pronoun_position=None, current_utterance=None):
        """
        Check if a pronoun has a missing referent (cannot be resolved).
        
        Args:
            pronoun (str): The pronoun to check (e.g., "he", "she", "it", "they")
            previous_utterances (list): List of previous utterances (up to 5)
            pronoun_position (int): Position of pronoun in current utterance (for forward reference check)
            current_utterance (str): Current utterance text
        
        Returns:
            bool: True if referent is MISSING, False if referent found
        """
        # Only consider the last 5 utterances
        recent_utterances = previous_utterances[-5:] if len(previous_utterances) > 5 else previous_utterances
        
        # Convert pronoun to lowercase for case-insensitive matching
        pronoun_lower = pronoun.lower()
        
        # Search through recent utterances for entities that could be the referent
        for i, utterance in enumerate(recent_utterances):
            utterance_lower = utterance.lower()
            # Check each entity and see if it appears in any utterance
            for entity, pronouns in self.entities.items():
                # Check if the entity name appears in the utterance
                if entity in utterance_lower:
                    # Check if the pronoun matches any of this entity's pronouns
                    if pronoun_lower in [p.lower() for p in pronouns]:
                        # If this is the current utterance, check for forward reference
                        if i == len(recent_utterances) - 1 and current_utterance and pronoun_position is not None:
                            # Find position of entity in current utterance
                            entity_position = current_utterance.lower().find(entity)
                            if entity_position != -1 and pronoun_position < entity_position:
                                continue  # Skip - pronoun comes before entity (forward reference)
                        
                        return False  # Valid referent found, so NOT missing
        
        return True  # No referent found, so it IS missing
    
    def check_utterance(self, context, current_utterance):
        """
        Check if current utterance has any pronouns with missing referents OR missing subjects
        
        Args:
            context (list): Previous utterances (already limited to 5)
            current_utterance (str): Current utterance to check
        
        Returns:
            bool: True if ANY pronoun has missing referent OR has missing subject, False otherwise
        """
        # First check for missing subjects (verbs without subjects)
        if self.has_missing_subject(current_utterance):
            return True
        
        # Then check for missing pronoun referents
        pronouns_with_pos = self.extract_pronouns_with_positions(current_utterance)
        
        if not pronouns_with_pos:
            return False  # No pronouns to check
        
        # Include current utterance in the context for referent resolution
        all_context = context + [current_utterance]
        
        # Check each pronoun in the current utterance
        for pronoun, position in pronouns_with_pos:
            if self.has_missing_referent(pronoun, all_context, position, current_utterance):
                return True  # Found at least one missing referent
        
        return False  # All pronouns have referents and no missing subjects


# # Test with the provided utterances
# def test_utterances():
#     checker = ReferentChecker()
    
#     utterances = [
#     """OK, so Cinderella was a Um, her, her parents died.""",
#         """and she had to go live with an aunt and two stepdaughters or stepsisters.""",
#         """And they treated her rather unfair,.""",
#         """and she got the worst clothes, she always had to do all the chore, the hard chores, and uh the stepsisters.""",
#         """and the stepmother didn't treat her that nicely.""",
#         """As she got older, um, there was gonna be a big ball in the uh in the kingdom,.""",
#         """and everyone was invited.""",
#         """and she wanted to go.""",
#         """She didn't have a dress or anything to wear, so she was.""",
#         """Able to get a dress made with the help of some magic mice or magic critters.""",
#         """And she went to the ball,.""",
#         """and she had a wonderful time.""",
#         """She met the prince, the prince, you know, had feelings for her.""",
#         """and um she had to leave the ball all of a sudden.""",
#         """because her magic gown was gonna change back to, you know, frumpy clothing.""",
#         """At midnight she ran out, she lost her magic slipper, one of her slippers,.""",
#         """and she went home.""",
#         """and thought, you know, that was the end of it.""",
#         """She'd never, you know, see the prince again.""",
#         """Um, the prince took that slipper.""",
#         """and went around the kingdom trying to find out who it belonged to, and lo.""",
#         """and behold, it belonged to Cinderella, who we found.""",
#         """by going to all the different houses.""",
#         """And um she kissed him.""",
#         """and she became glamorous again,.""",
#         """and they got married.""",
#         """and they lived happily ever after.""",
#         """All the dismay of the stepmother and the stepsisters.""",
#         """The end."""
#     ]
    
#     print("Utterance Analysis:")
#     print("=" * 80)
    
#     for i, utterance in enumerate(utterances):
#         # Get context (previous 5 utterances) - matching your structure
#         context = utterances[max(0, i-5):i]
        
#         # Check if current utterance has missing referents
#         has_missing = checker.check_utterance(context, utterance)
        
#         # Extract pronouns for display
#         pronouns_with_pos = checker.extract_pronouns_with_positions(utterance)
#         pronouns = [p[0] for p in pronouns_with_pos] if pronouns_with_pos else []  # Safe handling
        
#         # Check for missing subject
#         has_missing_subject = checker.has_missing_subject(utterance)
        
#         print(f"Utterance {i+1:2}: {utterance}")
#         print(f"Pronouns: {pronouns}")
#         print(f"Has missing subject: {has_missing_subject}")
#         print(f"Has missing referent: {has_missing}")
#         print("-" * 80)

# if __name__ == "__main__":
#     test_utterances()