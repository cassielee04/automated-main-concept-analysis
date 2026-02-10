import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))
from fillers import count_fillers, remove_fillers



def normalize_utterances(utterances):
    """
    Processes a list of utterances by detecting and removing fillers from each one.
    
    Args:
    utterances (list): A list of strings representing utterances.
    
    Returns:
    list: A list of cleaned utterances with fillers removed.
    """    
    cleaned_utterances = []
    for utt in utterances:
        result = count_fillers(utt)
        detected = result['detected_fillers']
        cleaned = remove_fillers(utt, detected)
        cleaned_utterances.append(cleaned)
    return cleaned_utterances



def normalize_utterance(text):
    """
    Normalize a single utterance by removing fillers.
    
    Args:
        text (str): The input utterance string.
    
    Returns:
        str: The normalized utterance string with fillers removed.
    """
    result = count_fillers(text)
    detected = result['detected_fillers']
    cleaned = remove_fillers(text, detected)
    return cleaned.strip()
