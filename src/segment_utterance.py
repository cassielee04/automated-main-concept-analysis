import re
from features import get_stanza_pipeline
# from preprocessing import clean_text
# from features import get_stanza_pipeline
# from nouns_verbs_per_utterance import calculate_nouns_per_utterance, calculate_verbs_per_utterance

# Initialize Stanza pipeline for English
# stanza.download('en')
nlp = get_stanza_pipeline()

def has_main_clause(sentence):
    """Check if a sentence contains a main clause (subject + verb)."""
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_subject = False
        has_verb = False
        for word in sent.words:
            if word.deprel in ['nsubj', 'nsubj:pass']:
                has_subject = True
            if word.deprel == 'root' and word.upos in ['VERB', 'AUX']:
                has_verb = True
        if has_subject and has_verb:
            return True
    return False
 
def is_imperative(sentence):
    """Check if a sentence is an imperative (verb with no explicit subject)."""
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_verb = False
        has_subject = False
        for word in sent.words:
            if word.deprel == 'root' and word.upos == 'VERB':
                has_verb = True
            if word.deprel in ['nsubj', 'nsubj:pass']:
                has_subject = True
        if has_verb and not has_subject:
            return True
    return False

def get_conjunctions(sentence):
    """Extract conjunctions (CCONJ or SCONJ) from the sentence using Stanza."""
    doc = nlp(sentence)
    conjunctions = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['CCONJ', 'SCONJ']:
                conjunctions.append(word.text.lower())
    return list(set(conjunctions))  # Remove duplicates


# [Previous functions: has_main_clause, is_imperative, get_conjunctions remain unchanged]

def segment_utterances(text):
    """Segment text into utterances, preserving repeated conjunctions."""
    # Define utterance terminators
    terminators = r'[.!?]'
    
    # Split text into potential utterances based on terminators
    raw_sentences = re.split(f'({terminators})', text)
    utterances = []
    current_utterance = ""
    pending_conjunction = ""
    
    # Get conjunctions for the entire text
    text_conjunctions = get_conjunctions(text)
    if not text_conjunctions:
        text_conjunctions = ['and', 'but', 'although', 'or', 'so']  # Fallback list
    
    # Create regex pattern for conjunctions (single or repeated), preserving them
    conj_pattern = r'\b(' + '|'.join(text_conjunctions) + r'(?:\s+' + '|'.join(text_conjunctions) + r')*)\b'
    
    for i in range(0, len(raw_sentences) - 1, 2):
        sentence = raw_sentences[i].strip()
        terminator = raw_sentences[i + 1]
        
        if not sentence:
            continue
            
        # Split by conjunctions, keeping them in the output
        parts = re.split(conj_pattern, sentence, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]  # Remove empty strings
        
        for j, part in enumerate(parts):
            # Skip if part is empty
            if not part:  
                continue
                
            # Check if the part is a conjunction (or repeated conjunctions)
            is_conjunction = bool(re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b', part, re.IGNORECASE))
            
            # Check for main clause or imperative
            has_main = has_main_clause(part)
            is_imp = is_imperative(part)
            
            if is_conjunction and not (has_main or is_imp):
                # Store the conjunction(s) to prepend to the next valid clause
                pending_conjunction = part
                continue
                
            if has_main or is_imp:
                # If there's a pending conjunction, prepend it
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""
                
                # If it follows a conjunction and is a main clause or imperative, start new utterance
                if j > 0 and bool(re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b', parts[j-1], re.IGNORECASE)) and (has_main or is_imp):
                    if current_utterance:
                        utterances.append(current_utterance + terminator)
                    current_utterance = part
                else:
                    # Combine with current utterance
                    current_utterance = (current_utterance + " " + part).strip()
            else:
                # Append to current utterance if it's a dependent clause/phrase
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""
                current_utterance = (current_utterance + " " + part).strip()
                
        # Append the current utterance with its terminator
        if current_utterance:
            utterances.append(current_utterance + terminator)
            current_utterance = ""
    
    # Handle any remaining utterance
    if current_utterance:
        if pending_conjunction:
            current_utterance = f"{pending_conjunction} {current_utterance}".strip()
        utterances.append(current_utterance + (terminator if terminator else '.'))
    
    # Final cleanup: merge standalone conjunctions
    final_utterances = []
    i = 0
    while i < len(utterances):
        utterance = utterances[i].strip()
        # Check if the utterance is just a conjunction (or repeated conjunctions)
        if re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b[.!?]', utterance, re.IGNORECASE):
            if i + 1 < len(utterances):
                # Merge with the next utterance
                next_utterance = utterances[i + 1].strip()
                combined = f"{utterance[:-1]} {next_utterance}".strip()
                final_utterances.append(combined)
                i += 2
            else:
                # If it's the last utterance, append as is
                final_utterances.append(utterance)
                i += 1
        else:
            final_utterances.append(utterance)
            i += 1
    
    return final_utterances
# Sample dialogue
# dialogue = "All right, Cinderella's father. Uh, brings into his home and Cinderella's home, a woman and her two ugly daughters, and she becomes stepmother to Cinderella. And she favors her own daughters over Cinderella. Uh, and As they, are living together, uh. They find out that the king is going to give a ball. Uh, for the prince. And Cinderella looks through a trunk. Finding some clothing. But. She's not allowed to go to the ball, so the other two ugly stepdaughters. Uh, dress up in the pretty clothes that they find, and Cinderella needs to go out to the. Stable to take care of the animals and of course do all the other work around the house. And Cinderella is very disappointed because she won't be allowed to go to the ball, but. Um, The other stepdaughters do, and while they are gone. The fairy godmother and all the mice come in to talk to Cinderella and dress her in the fairy godmother, of course, dresses her in finery and changes uh the pumpkin into a fancy coach. And Cinderella goes to the ball and dances with the prince, but when the clock strikes 12, Cinderella has to rush out and down the palace steps to go away from the ball. And the Men who work for the king then take the glass slipper that Cinderella had lost on her flight out of the palace to see who it will fit, and when they come to Cinderella's house. It fits Cinderella, and therefore Cinderella is happily married to the prince."

# cleaned_dialogue = clean_text(dialogue)
# print("Cleaned Dialogue:", cleaned_dialogue)
# # Segment the dialogue
# segmented = segment_utterances(cleaned_dialogue)
# print("-------------------------")
# print("Segmented Utterances:", len(segmented))
# print("-------------------------")
# # print("verbs per utterance:", calculate_verbs_per_utterance(segmented, nlp))

# # Output in CHAT-like format
# for i, utterance in enumerate(segmented, 1):
#     # print(f"*PAR: {utterance} %utt{i}")
#     print(i, " ", utterance)