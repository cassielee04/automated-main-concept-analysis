# import stanza
from collections import Counter
import re
from features import get_stanza_pipeline 

# Download and initialize the English model (run once)
# stanza.download('en')

# Initialize the pipeline with dependency parsing for better context
# nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')
nlp = get_stanza_pipeline()
def is_likely_filler(word, sentence, word_idx):
    """
    Determine if a word is likely used as a filler based on context
    
    Args:
        word: Stanza word object
        sentence: Stanza sentence object
        word_idx: Index of word in sentence
    
    Returns:
        bool: True if likely a filler usage
    """
    text = word.text.lower()
    pos = word.upos
    deprel = word.deprel
    
    # Context-based rules for common filler-prone words
    
    # "Well" - filler when at sentence start or as discourse marker
    if text == "well":
        # Filler if: sentence-initial, or has discourse marker dependency
        if word_idx == 0 or deprel in ['discourse', 'intj']:
            return True
        # Not filler if it's an adverb modifying a verb/adjective
        if pos == 'ADV' and deprel in ['advmod']:
            return False
        return word_idx == 0  # Default to filler if sentence-initial
    
    # "So" - filler when used as discourse marker, not as adverb/conjunction
    if text == "so":
        if deprel in ['discourse', 'intj'] or word_idx == 0:
            return True
        # Not filler if it's modifying degree ("so tall") or conjunction
        if deprel in ['advmod', 'mark', 'cc']:
            return False
        return word_idx == 0
    
    # "Like" - filler when not used as verb or preposition
    if text == "like":
        # Not filler if it's a verb ("I like cats") or preposition ("like a bird")
        if pos in ['VERB'] or deprel in ['case', 'mark']:
            return False
        # Filler if discourse marker or interjection
        if deprel in ['discourse', 'intj']:
            return True
        # Likely filler if followed by hesitation or in middle of clause
        return True
    
    # "Okay" - usually a filler/discourse marker
    if text == "okay":
        return True
    
    # "Anyway" - usually a filler/discourse marker
    if text == "anyway":
        return True
    
    # "Yeah" - usually discourse marker/filler
    if text == "yeah":
        return True
    
    # "Simply" - filler when used as hedge/discourse marker, not as manner adverb
    if text == "simply":
        # First check for established phrases regardless of position
        next_words = [w.text.lower() for w in sentence.words[word_idx+1:word_idx+3]]
        if any(word in next_words for word in ['put', 'stated', 'said']):
            return False
        
        # Not filler if it's genuinely describing manner in other contexts
        if deprel == 'advmod' and pos == 'ADV':
            # If it's at sentence start and not in established phrase, likely filler
            if word_idx == 0:
                return True
            # Default to filler for sentence adverb usage
            return True
        return True
    
    # "Actually" - filler when used as discourse marker/hedge, not as genuine contrast
    if text == "actually":
        # Not filler if it's genuinely contrasting facts ("It's actually blue, not red")
        if deprel == 'advmod' and pos == 'ADV':
            # If sentence-initial or isolated, likely filler
            if word_idx == 0:
                return True
            # If it's in middle of sentence providing genuine contrast, less likely filler
            # Look for contradiction indicators in context
            sentence_text = ' '.join([w.text.lower() for w in sentence.words])
            contrast_indicators = ['not', "don't", "didn't", 'but', 'however', 'though']
            if any(indicator in sentence_text for indicator in contrast_indicators):
                return False
            # Default to filler for sentence adverb usage
            return True
        return True
    
    # "Basically" - often filler when used to hedge/summarize vaguely
    if text == "basically":
        # Almost always filler - rarely used in precise, non-hedging way
        # Not filler only in very specific contexts like "basically identical"
        if deprel == 'advmod' and pos == 'ADV':
            # Check if it's modifying an adjective precisely
            next_words = [w.text.lower() for w in sentence.words[word_idx+1:word_idx+2]]
            precise_modifiers = ['identical', 'equivalent', 'the same']
            if any(modifier in next_words for modifier in precise_modifiers):
                return False
            # Otherwise likely filler
            return True
        return True
    
def is_and_then_filler(text, match_start):
    """
    Determine if "and then" is used as a filler or legitimate temporal connector
    
    Args:
        text (str): The full text being analyzed
        match_start (int): Character position where "and then" starts
    
    Returns:
        bool: True if likely a filler usage
    """
    # Get context around "and then"
    context_before = text[:match_start].lower()
    context_after = text[match_start + 8:].lower()  # 8 = len("and then")
    
    # Get the sentence containing "and then"
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "and then" in sent.lower():
            current_sentence = sent.strip().lower()
            break
    
    # Hesitation indicators suggest filler usage
    hesitation_indicators = [
        'um', 'uh', 'like', 'you know', 'i mean', 'well',
        'i think', 'i guess', 'maybe', 'probably', 'sort of',
        'kind of', 'basically', 'actually'
    ]
    
    # Check for hesitation patterns
    if any(indicator in current_sentence for indicator in hesitation_indicators):
        return True
    
    # Time/sequence indicators suggest legitimate temporal use
    temporal_indicators = [
        'after', 'before', 'when', 'while', 'during', 'next',
        'later', 'finally', 'afterwards', 'subsequently', 'immediately',
        'eventually', 'suddenly', 'meanwhile'
    ]
    
    # Concrete action verbs suggest narrative sequence
    action_patterns = [
        'went', 'came', 'walked', 'ran', 'drove', 'called', 'said',
        'took', 'got', 'made', 'did', 'put', 'opened', 'closed',
        'looked', 'turned', 'moved', 'started', 'finished', 'began'
    ]
    
    # Count concrete actions in the sentence
    action_count = sum(1 for action in action_patterns if action in current_sentence)
    
    # Not filler if connecting multiple concrete actions
    if action_count >= 2:
        return False
    
    # Not filler if part of clear temporal sequence
    if any(indicator in current_sentence for indicator in temporal_indicators):
        return False
    
    # Filler if sentence is very short (likely incomplete thought)
    if len(current_sentence.split()) < 6:
        return True
    
    # Check for repetitive "and then" usage (suggests filler)
    and_then_count = text.lower().count('and then')
    if and_then_count > 3:  # Multiple uses might indicate filler habit
        return True
    
    # Default to not filler - "and then" is often legitimate
    return False


def is_first_of_all_filler(text, match_start):
    """
    Determine if "first of all" is used as a filler or legitimate enumeration
    
    Args:
        text (str): The full text being analyzed
        match_start (int): Character position where "first of all" starts
    
    Returns:
        bool: True if likely a filler usage
    """
    # Get surrounding context
    sentences = text.split('.')
    current_sentence = ""
    next_sentence = ""
    
    # Find which sentence contains "first of all"
    for i, sent in enumerate(sentences):
        if "first of all" in sent.lower():
            current_sentence = sent.strip().lower()
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip().lower()
            break
    
    # Enumeration indicators suggest legitimate use
    enumeration_indicators = [
        'second', 'secondly', 'then', 'next', 'also', 'furthermore', 
        'additionally', 'moreover', 'finally', 'lastly', 'third',
        'thirdly', 'in addition', 'another thing'
    ]
    
    # Check current and next sentence for enumeration
    combined_text = current_sentence + ' ' + next_sentence
    
    # Look for explicit enumeration patterns
    if any(indicator in combined_text for indicator in enumeration_indicators):
        return False
    
    # Look for numbered/ordered lists
    list_patterns = ['1.', '2.', '3.', 'a.', 'b.', 'c.', 'i.', 'ii.', 'iii.']
    if any(pattern in combined_text for pattern in list_patterns):
        return False
    
    # Check if next sentence starts with enumeration
    if any(next_sentence.startswith(indicator) for indicator in enumeration_indicators[:5]):
        return False
    
    # Opinion/hedge indicators suggest filler usage
    opinion_indicators = [
        'i think', 'i believe', 'i feel', 'maybe', 'perhaps',
        'i guess', 'probably', 'might', 'could be', 'in my opinion',
        'personally', 'i suppose', 'it seems', 'i would say', 'i know'
    ]
    
    # Filler if followed by hedging/opinion language
    if any(indicator in current_sentence for indicator in opinion_indicators):
        return True
    
    # Check for other common filler patterns with "first of all"
    filler_patterns = [
        'let me', 'i want to', 'i need to', 'i should', 'we should',
        'it\'s important', 'you have to understand'
    ]
    
    if any(pattern in current_sentence for pattern in filler_patterns):
        return True
    
    # Default to filler if no clear enumeration structure found
    return True


def is_i_mean_filler(text, match_start):
    """
    Determine if "i mean" is used as a filler or legitimate clarification
    
    Args:
        text (str): The full text being analyzed
        match_start (int): Character position where "i mean" starts
    
    Returns:
        bool: True if likely a filler usage
    """
    # Get the sentence containing "i mean"
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "i mean" in sent.lower():
            current_sentence = sent.strip().lower()
            break

    excluded_phrases = ["you know what i mean"]
    if any(phrase in current_sentence for phrase in excluded_phrases):
        return False
    
    # Clarification indicators suggest legitimate use
    clarification_indicators = [
        'that is', 'in other words', 'to clarify', 'specifically',
        'what i\'m saying is', 'to be clear', 'more precisely',
        'let me explain', 'actually', 'rather'
    ]
    
    # Not filler if used for genuine clarification
    if any(indicator in current_sentence for indicator in clarification_indicators):
        return False
    
    # Check if "i mean" is followed by specific examples or clarification
    # Look for concrete details after "i mean"
    mean_pos = current_sentence.find('i mean')
    if mean_pos != -1:
        after_mean = current_sentence[mean_pos + 6:].strip()  # 6 = len("i mean")
        
        # Not filler if followed by specific examples
        example_indicators = ['like', 'such as', 'for example', 'for instance']
        if any(indicator in after_mean[:20] for indicator in example_indicators):
            return False
        
        # Not filler if followed by concrete numbers/facts
        import re
        if re.search(r'\b\d+\b', after_mean[:30]):  # Numbers in first 30 chars
            return False
    
    # Filler if surrounded by other hesitation markers
    hesitation_indicators = [
        'um', 'uh', 'like', 'you know', 'well', 'so',
        'i think', 'i guess', 'maybe', 'probably', 'sort of',
        'kind of', 'basically'
    ]
    
    if sum(1 for indicator in hesitation_indicators if indicator in current_sentence) >= 2:
        return True
    
    # Filler if sentence is very short or incomplete
    if len(current_sentence.split()) < 5:
        return True
    
    # Check for repetitive "i mean" usage (suggests filler habit)
    i_mean_count = text.lower().count('i mean')
    if i_mean_count > 2:
        return True
    
    # Default to filler - "i mean" is often used as hesitation
    return True


def is_lets_see_filler(text, match_start):
    """
    Determine if "let's see" is used as a filler or legitimate thinking/checking
    
    Args:
        text (str): The full text being analyzed
        match_start (int): Character position where "let's see" starts
    
    Returns:
        bool: True if likely a filler usage
    """
    # Get the sentence containing "let's see"
    sentences = text.split('.')
    current_sentence = ""
    next_sentence = ""
    
    for i, sent in enumerate(sentences):
        if "let's see" in sent.lower():
            current_sentence = sent.strip().lower()
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip().lower()
            break
    
    # Legitimate thinking/checking indicators
    checking_indicators = [
        'here', 'there', 'this', 'that', 'where', 'what',
        'when', 'how', 'if', 'whether', 'check', 'look',
        'find', 'search', 'calculate', 'count'
    ]
    
    # Not filler if followed by actual checking/searching action
    combined_text = current_sentence + ' ' + next_sentence
    if any(indicator in combined_text for indicator in checking_indicators):
        return False
    
    # Check if "let's see" is followed by concrete information
    see_pos = current_sentence.find('let\'s see')
    if see_pos != -1:
        after_see = current_sentence[see_pos + 9:].strip()  # 9 = len("let's see")
        
        # Not filler if followed by specific details or questions
        if any(word in after_see[:20] for word in ['what', 'where', 'when', 'how', 'if']):
            return False
        
        # Not filler if followed by concrete actions
        action_words = ['we can', 'we need', 'we should', 'you can', 'i can']
        if any(action in after_see[:30] for action in action_words):
            return False
    
    # Filler if surrounded by other hesitation markers
    hesitation_indicators = [
        'um', 'uh', 'well', 'so', 'like', 'you know',
        'i think', 'i guess', 'maybe', 'probably', 'i would say', 'i know'
    ]
    
    if any(indicator in current_sentence for indicator in hesitation_indicators):
        return True
    
    # Filler if sentence is very short (likely stalling)
    if len(current_sentence.split()) < 4:
        return True
    
    # Filler if used multiple times (suggests stalling habit)
    lets_see_count = text.lower().count('let\'s see')
    if lets_see_count > 2:
        return True
    
    # Check if it's at the beginning of a response (often filler)
    if match_start < 20:  # Near the beginning of text
        return True
    
    # Default to filler - "let's see" is often used for stalling
    return True

def is_i_know_filler(text, match_start):
    """
    Determine if 'i know' is used as a filler or legitimate expression of knowledge.
    Enhanced to better catch filler usage.
    """
    # Get the sentence containing 'i know'
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "i know" in sent.lower():
            current_sentence = sent.strip().lower()
            break
    
    # Specific knowledge indicators suggest legitimate use
    specific_knowledge_indicators = [
        'the answer is', 'how to do', 'what to do', 'where to go', 'when to start',
        'who did', 'why it happened', 'the solution', 'the method', 'the process',
        'for certain', 'for sure', 'exactly', 'precisely'
    ]
    
    # Not filler if expressing specific, concrete knowledge
    if any(indicator in current_sentence for indicator in specific_knowledge_indicators):
        return False
    
    # Check if 'i know' is followed by concrete information or "that" clauses with facts
    know_pos = current_sentence.find('i know')
    if know_pos != -1:
        after_know = current_sentence[know_pos + 6:].strip()  # 6 = len('i know')
        
        # Not filler if followed by "that" + specific claim
        if after_know.startswith('that '):
            # Look for concrete facts vs opinions/hedging
            concrete_indicators = ['is', 'was', 'are', 'were', 'has', 'have', 'will', 'can', 'cannot']
            if any(indicator in after_know[:30] for indicator in concrete_indicators):
                # But still filler if it contains hedging
                hedge_indicators = ['maybe', 'probably', 'might', 'could be', 'i think', 'seems']
                if not any(hedge in after_know[:50] for hedge in hedge_indicators):
                    return False
        
        # Not filler if followed by specific examples with numbers/facts
        if re.search(r'\b\d+\b', after_know[:30]) or 'for example' in after_know[:20]:
            return False
    
    # Strong filler indicators - discourse markers and hedging
    strong_filler_indicators = [
        'um', 'uh', 'like', 'you know', 'well', 'so',
        'i think', 'i guess', 'maybe', 'probably', 'sort of',
        'kind of', 'basically', 'actually', 'i mean', 'i would say',
        'right?', 'but', 'and'  # Added common continuation patterns
    ]
    
    # Count filler indicators in sentence
    filler_count = sum(1 for indicator in strong_filler_indicators if indicator in current_sentence)
    
    # Definitely filler if multiple hesitation markers present
    if filler_count >= 2:
        return True
    
    # Filler if used in opinion/hedging context
    opinion_hedge_patterns = [
        'i think', 'i believe', 'i feel', 'in my opinion',
        'personally', 'it seems', 'i suppose', 'i would say'
    ]
    
    if any(pattern in current_sentence for pattern in opinion_hedge_patterns):
        return True
    
    # Filler if sentence is very short or incomplete (suggests hesitation)
    if len(current_sentence.split()) < 4:
        return True
    
    # Filler if at beginning of response (discourse marker usage)
    if match_start < 15:  # Near the beginning of text
        return True
    
    # Filler if used for agreement/acknowledgment without content
    agreement_patterns = ['right', 'exactly', 'yeah', 'yes', 'sure', 'okay']
    if any(pattern in current_sentence for pattern in agreement_patterns):
        return True
    
    # Check for repetitive 'i know' usage (suggests filler habit)
    i_know_count = text.lower().count('i know')
    if i_know_count > 1:  # Lowered threshold as repetition is strong indicator
        return True
    
    # Default to filler - 'i know' is very often used as a discourse marker/filler
    return True

def is_i_would_say_filler(text, match_start):
    """
    Determine if 'i would say' is used as a filler or legitimate expression.
    Enhanced to better catch filler usage.
    """
    # Get the sentence containing 'i would say'
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "i would say" in sent.lower():
            current_sentence = sent.strip().lower()
            break
    
    # Strong indicators of legitimate use (making actual estimations/judgments)
    legitimate_indicators = [
        'approximately', 'around', 'about', 'roughly', 'between',
        'at least', 'no more than', 'probably', 'likely',
        'the best', 'the worst', 'the most', 'the least',
        'definitely', 'certainly', 'absolutely'
    ]
    
    # Not filler if making specific estimates or judgments
    if any(indicator in current_sentence for indicator in legitimate_indicators):
        return False
    
    # Check what follows 'i would say'
    would_say_pos = current_sentence.find('i would say')
    if would_say_pos != -1:
        after_would_say = current_sentence[would_say_pos + 11:].strip()  # 11 = len('i would say')
        
        # Not filler if followed by specific measurements, numbers, or concrete claims
        if re.search(r'\b\d+\b', after_would_say[:30]):  # Numbers in first 30 chars
            return False
            
        # Not filler if followed by definitive statements
        definitive_starters = ['that is', 'this is', 'it is', 'they are', 'we are']
        if any(after_would_say.startswith(starter) for starter in definitive_starters):
            return False
    
    # Strong filler indicators - hedging and discourse markers
    filler_indicators = [
        'um', 'uh', 'like', 'you know', 'well', 'so',
        'i think', 'i guess', 'maybe', 'sort of',
        'kind of', 'basically', 'actually', 'i mean', 'i know',
        'probably', 'possibly', 'perhaps', 'i suppose'
    ]
    
    # Count filler/hedge indicators
    filler_count = sum(1 for indicator in filler_indicators if indicator in current_sentence)
    
    # Definitely filler if multiple hedging markers present
    if filler_count >= 2:
        return True
    
    # Filler if sentence is very short (suggests hesitation/stalling)
    if len(current_sentence.split()) < 5:
        return True
    
    # Filler if at beginning of response (discourse marker usage)
    if match_start < 15:  # Near the beginning of text
        return True
    
    # Filler if used with other opinion/hedge phrases
    opinion_patterns = [
        'in my opinion', 'personally', 'i believe', 'i feel',
        'it seems to me', 'from my perspective'
    ]
    
    if any(pattern in current_sentence for pattern in opinion_patterns):
        return True
    
    # Check for repetitive usage (suggests filler habit)
    i_would_say_count = text.lower().count('i would say')
    if i_would_say_count > 1:  # Lowered threshold
        return True
    
    # Check for vague content after "i would say"
    if would_say_pos != -1:
        after_content = current_sentence[would_say_pos + 11:].strip()
        vague_indicators = ['that', 'it', 'this', 'something', 'anything', 'everything']
        if any(after_content.startswith(indicator) for indicator in vague_indicators):
            return True
    
    # Default to filler - 'i would say' is very often used as a hedge/filler
    return True

def is_i_think_filler(text, match_start):
    """
    Determine if 'i think' is used as a filler or legitimate opinion.
    """
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "i think" in sent.lower():
            current_sentence = sent.strip().lower()
            break
    
    # Legitimate if expressing firm belief or evidence-based opinion
    legitimate_indicators = [
        'because', 'since', 'due to', 'evidence shows', 'research indicates',
        'data suggests', 'facts are', 'clearly', 'obviously', 'definitely'
    ]
    
    if any(indicator in current_sentence for indicator in legitimate_indicators):
        return False
    
    # Check what follows
    think_pos = current_sentence.find('i think')
    if think_pos != -1:
        after_think = current_sentence[think_pos + 7:].strip()  # 7 = len('i think')
        
        # Not filler if followed by concrete facts/numbers
        if re.search(r'\b\d+\b', after_think[:30]) or any(ex in after_think[:20] for ex in ['for example', 'such as']):
            return False
    
    # Filler if with other hesitations
    hesitation_indicators = [
        'um', 'uh', 'like', 'you know', 'well', 'so', 'maybe', 'probably',
        'sort of', 'kind of', 'basically', 'i mean', 'i guess'
    ]
    
    filler_count = sum(1 for ind in hesitation_indicators if ind in current_sentence)
    if filler_count >= 1:  # Even one nearby boosts it to filler
        return True
    
    if len(current_sentence.split()) < 6:
        return True
    
    if match_start < 15:  # Sentence/turn start
        return True
    
    i_think_count = text.lower().count('i think')
    if i_think_count > 2:
        return True
    
    return True  # Default to filler—it's rarely non-hedging

def is_i_remember_filler(text, match_start):
    """
    Determine if 'i remember' is used as a filler or legitimate opinion.
    """
    sentences = text.split('.')
    current_sentence = ""
    
    for sent in sentences:
        if "i remember" in sent.lower():
            current_sentence = sent.strip().lower()
            break
    
    # Legitimate if expressing firm belief or evidence-based opinion
    legitimate_indicators = [
        'because', 'since', 'due to', 'evidence shows', 'research indicates',
        'data suggests', 'facts are', 'clearly', 'obviously', 'definitely'
    ]
    
    if any(indicator in current_sentence for indicator in legitimate_indicators):
        return False
    
    # Check what follows
    remember_pos = current_sentence.find('i remember')
    if remember_pos != -1:
        after_remember = current_sentence[remember_pos + 11:].strip()  # 11 = len('i remember')
        
        # Not filler if followed by concrete facts/numbers
        if re.search(r'\b\d+\b', after_remember[:30]) or any(ex in after_remember[:20] for ex in ['for example', 'such as']):
            return False
    
    # Filler if with other hesitations
    hesitation_indicators = [
        'um', 'uh', 'like', 'you know', 'well', 'so', 'maybe', 'probably',
        'sort of', 'kind of', 'basically', 'i mean', 'i guess'
    ]
    
    filler_count = sum(1 for ind in hesitation_indicators if ind in current_sentence)
    if filler_count >= 1:  # Even one nearby boosts it to filler
        return True
    
    if len(current_sentence.split()) < 6:
        return True
    
    if match_start < 15:  # Sentence/turn start
        return True
    
    i_remember_count = text.lower().count('i remember')
    if i_remember_count > 2:
        return True
    
    return True  # Default to filler—it's rarely non-hedging




def count_fillers(text, custom_fillers=None):
    """
    Count filler words using contextual analysis with Stanza
    
    Args:
        text (str): Input text to analyze
        custom_fillers (list): Optional custom list of filler words
    
    Returns:
        dict: Dictionary with the following keys:
            - 'filler_counts' (dict): Counts of each filler word/phrase.
            - 'total_fillers' (int): Total number of fillers detected.
            - 'contextual_analysis' (list): List of dictionaries, each containing:
                - 'word' (str): The filler word or phrase.
                - 'context' (str): Contextual information such as part-of-speech (POS), dependency relation (deprel), or match position.
                - 'sentence_pos' (int): Position of the word/phrase in the sentence or text.
                - 'is_filler' (bool): Whether the word/phrase is classified as a filler.
    """
    
    # Updated filler words that need contextual analysis
    contextual_fillers = {
        'well', 'so', 'like', 'okay', 'anyway', 'yeah', 
        'simply', 'actually', 'basically'
    }
    
    # Pure filler words (almost always fillers)
    pure_empty_fillers = {
        'oh', 'um', 'uh', 'hmm', 'huh', 'ah'
    }
    
    # Multi-word filler phrases (with contextual analysis)
    contextual_multi_word_fillers = {
        'first of all', 'and then', 'i mean', "let's see", 'i would say', 'i know', 'i think', "i don't think", 'i remember', "i don't remember"
    }
    
    # Multi-word filler phrases (always counted)
    pure_multi_word_fillers = {
        'i guess', 'you know', 'ya know', 
        'let me see', 'all right', 'alright'
    }
    
    # Use custom fillers if provided
    if custom_fillers:
        all_fillers = set(custom_fillers)
        contextual_fillers = {f for f in all_fillers if f not in pure_empty_fillers}
    else:
        all_fillers = contextual_fillers | pure_empty_fillers
    
    # Process the text with Stanza
    doc = nlp(text)
    
    filler_counts = Counter()
    contextual_analysis = []
    detected_fillers = []

    # Analyze single-word fillers
    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            word_text = word.text.lower()
            
            # Count pure fillers (always counted)
            if word_text in pure_empty_fillers:
                filler_counts[word_text] += 1
                contextual_analysis.append({
                    'word': word_text,
                    'context': 'pure_filler',
                    'sentence_pos': i,
                    'is_filler': True
                })
                detected_fillers.append(word_text)
            
            elif word_text in pure_empty_fillers:
                filler_counts[word_text] += 1
                contextual_analysis.append({
                    'word': word_text,
                    'context': 'pure_empty_filler',
                    'sentence_pos': i,
                    'is_filler': True
                })
                detected_fillers.append(word_text)
            
            # Analyze contextual fillers
            elif word_text in contextual_fillers:
                is_filler = is_likely_filler(word, sentence, i)
                if is_filler:
                    filler_counts[word_text] += 1
                    detected_fillers.append(word_text)
                contextual_analysis.append({
                    'word': word_text,
                    'context': f"pos={word.upos}, deprel={word.deprel}, sent_pos={i}",
                    'sentence_pos': i,
                    'is_filler': is_filler
                })
    
    # Analyze contextual multi-word fillers
    text_lower = text.lower()
    for phrase in contextual_multi_word_fillers:
        matches = re.finditer(r'\b' + re.escape(phrase) + r'\b', text_lower)
        for match in matches:
            match_start = match.start()
            is_filler = False
            
            # Apply contextual analysis for the phrase
            if phrase == 'first of all':
                is_filler = is_first_of_all_filler(text, match_start)
            elif phrase == 'and then':
                is_filler = is_and_then_filler(text, match_start) 
            elif phrase == 'i mean':
                is_filler = is_i_mean_filler(text, match_start)  
            elif phrase == "let's see":
                is_filler = is_lets_see_filler(text, match_start)
            elif phrase == 'i know':
                is_filler = is_i_know_filler(text, match_start)  # Always consider "i know" as filler
            elif phrase == 'i would say':
                is_filler = is_i_would_say_filler(text, match_start)
            elif phrase == 'i think' or phrase == "i don't think":
                is_filler = is_i_think_filler(text, match_start)
            elif phrase == 'i remember' or phrase == "i don't remember":
                is_filler = is_i_remember_filler(text, match_start)
            # Update counts and analysis
            if is_filler:
                filler_counts[phrase] += 1
                detected_fillers.append(phrase)
                contextual_analysis.append({
                    'word': phrase,
                    'context': f"match_start={match_start}",
                    'sentence_pos': match_start,
                    'is_filler': is_filler
                })
    
    # Count pure multi-word fillers
    for phrase in pure_multi_word_fillers:
        if custom_fillers is None or phrase in custom_fillers:
            count = len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
            if count > 0:
                filler_counts[phrase] = count
                detected_fillers.extend([phrase] * count)
    
    total_fillers = sum(filler_counts.values())

    empty_fillers = sum(filler_counts[f] for f in pure_empty_fillers)
    total_fillers = sum(filler_counts.values())
    
    # The total_fillers variable represents the sum of all detected filler words, 
    # providing a quick summary of the overall filler word count in the text.
    return {
        'filler_counts': dict(filler_counts),
        'total_fillers': total_fillers,
        'contextual_analysis': contextual_analysis,
        'detected_fillers': detected_fillers
        
    }

def contains_filler(text: str) -> bool:
    """
    Returns True if the input sentence contains any filler word or phrase.
    """
    result = count_fillers(text)
    return result['total_fillers'] > 0

import re
from typing import List

def count_words_in_tokens(tokens: List[str]) -> int:
    """
    Count total number of words across a list of tokens/phrases.
    E.g., ['um', 'you know'] -> 3 words.
    """
    word_re = re.compile(r"\b[\w']+\b")  # include apostrophes like "let's"
    return sum(len(word_re.findall(t)) for t in tokens)

def analyze_text_with_context(text, show_context=True):
    """
    Analyze text with contextual filler detection
    """
    result = count_fillers(text)
    
    print(f"Filler Analysis:")
    print(f"Total fillers: {result['total_fillers']}")

def remove_fillers(utterance: str, detected_fillers: List[str]) -> str:
    """
    Remove specified fillers from the utterance, preserving structure and normalizing punctuation.
    
    Args:
        utterance (str): Input text to clean.
        detected_fillers (List[str]): List of fillers to remove, as provided by count_fillers.
    
    Returns:
        str: Text with specified fillers removed and spaces/punctuation normalized.
    """
    if not utterance or not utterance.strip() or not detected_fillers:
        return utterance.strip() if utterance else ""
    
    text_lower = utterance.lower()
    spans = []
    punct_to_remove = '.,;!?'

    # Process single-word fillers
    doc = nlp(utterance)
    filler_index = 0  # Track position in detected_fillers list
    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            word_text_lower = word.text.lower()
            # Check if current word matches the next filler in detected_fillers
            if (filler_index < len(detected_fillers) and 
                word_text_lower == detected_fillers[filler_index].lower() and
                ' ' not in detected_fillers[filler_index]):  # Ensure it's a single-word filler
                start = word.start_char
                end = word.end_char
                # Include following punctuation if present
                if i + 1 < len(sentence.words) and sentence.words[i + 1].text in punct_to_remove:
                    end = sentence.words[i + 1].end_char
                spans.append((start, end))
                filler_index += 1

    # Process multi-word fillers
    for filler in detected_fillers:
        if ' ' in filler:  # Multi-word filler
            for match in re.finditer(r'\b' + re.escape(filler.lower()) + r'\b', text_lower):
                start = match.start()
                end = match.end()
                # Include following punctuation if present
                if end < len(utterance) and utterance[end] in punct_to_remove:
                    end += 1
                spans.append((start, end))

    # Sort and merge overlapping spans
    spans.sort(key=lambda x: x[0])
    merged_spans = []
    for span in spans:
        if not merged_spans or merged_spans[-1][1] < span[0]:
            merged_spans.append(list(span))
        else:
            merged_spans[-1][1] = max(merged_spans[-1][1], span[1])

    # Build cleaned text
    if not merged_spans:
        return utterance.strip()
    
    result = []
    prev_end = 0
    for start, end in merged_spans:
        if start > prev_end:
            result.append(utterance[prev_end:start])
        prev_end = end
    result.append(utterance[prev_end:])
    cleaned = ''.join(result)

    # Normalize spacing and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[.,;!?]+', '', cleaned).strip()  # Remove leading punctuation
    cleaned = re.sub(r'[.,;!?]+$', '', cleaned).strip()  # Remove trailing punctuation
    cleaned = re.sub(r'([.,;!?])\s*([.,;!?])', r'\1', cleaned)  # Remove duplicate punctuation
    cleaned = re.sub(r'\s*([.,;!?])\s*', r'\1 ', cleaned)  # Ensure single space after punctuation

    return cleaned

    
    if result['filler_counts']:
        print("\nFiller word breakdown:")
        for filler, count in sorted(result['filler_counts'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  '{filler}': {count}")
    else:
        print("\nNo filler words detected.")
    
    if show_context and result['contextual_analysis']:
        print("\nContextual analysis:")
        for analysis in result['contextual_analysis']:
            status = "✓ FILLER" if analysis['is_filler'] else "✗ NOT FILLER"
            print(f"  '{analysis['word']}' - {status} ({analysis['context']})")
    
    return result


# Example usage and comparison
if __name__ == "__main__":
    # Test cases to show the difference
    test_cases = [
        "She was doing well in school.",          # "well" should NOT be counted
        "Well, I think we should go.",            # "well" SHOULD be counted
        "I really like this movie.",              # "like" should NOT be counted  
        "It was, like, really good.",             # "like" SHOULD be counted
        "I mean, that's a good point.",           # "i mean" SHOULD be counted
        "I guess we can try that approach.",      # "i guess" SHOULD be counted
        "You know what I mean?",                  # "you know" SHOULD be counted
        "Let's see what happens next.",           # "let's see" SHOULD be counted
        "Oh, I forgot about that.",               # "oh" SHOULD be counted
        "Okay, let's get started.",               # "okay" SHOULD be counted
        "So we went to the store and then home.", # "so" SHOULD be counted
        "Yeah, that sounds good to me.",          # "yeah" SHOULD be counted
        "Simply put, this is the answer.",        # "simply" should NOT be counted
        "Simply, I don't understand.",            # "simply" SHOULD be counted
        "It's actually blue, not red.",           # "actually" should NOT be counted
        "Actually, I think we should go.",        # "actually" SHOULD be counted
        "They are basically identical.",          # "basically" should NOT be counted
        "Basically, we need more time.",          # "basically" SHOULD be counted
        "First of all, we need to plan. Second, we execute.", # "first of all" should NOT be counted
        "First of all, I think this is hard.",    # "first of all" SHOULD be counted
        "We went to the store and then came home.", # "and then" should NOT be counted
        "I was thinking and then, um, forgot.",   # "and then" SHOULD be counted
        "I think Cinderella went to the ball",
        "I don't think that's right.",  # "i don't think" SHOULD be counted
        "I don't remember the details."  # "i don't remember" SHOULD be counted
    ]
    
    
    print("=== Contextual Filler Detection and Removal Examples ===\n")
    
    text = "Um"
    result = count_fillers(text)
    print("Original text:", text)
    print("Detected fillers:", result['detected_fillers'])
    print("Filler counts:", result['filler_counts'])
    print("Total fillers:", result['total_fillers'])
    cleaned_text = remove_fillers(text, result['detected_fillers'])
    print("Cleaned text:", cleaned_text)