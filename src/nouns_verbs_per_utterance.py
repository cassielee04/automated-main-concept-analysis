def calculate_nouns_per_utterance(utterances, nlp):
    """Calculate average nouns per utterance"""
    noun_count = 0
    for utterance in utterances:
        doc = nlp(utterance)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos == "NOUN" or word.pos == "PROPN":
                    noun_count += 1
    return noun_count / len(utterances)
        
def calculate_verbs_per_utterance(utterances, nlp):
    """Calculate average verbs (including verbs, copulas, and auxiliaries followed by past or present participles) per utterance"""
    verb_count = 0
    aux_count = 0
    for utterance in utterances:
        doc = nlp(utterance)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos == "VERB" or word.pos == "AUX": 
                    if word.pos == "AUX":
                        print("Checking AUX word:", word.text)
                        aux_count += 1
                    print("Counting verb:", word.text)
                    verb_count += 1

    print("Total Verbs Counted:", verb_count)
    print("Total AUX Counted:", aux_count)
    return verb_count / len(utterances)

# def main():
#     # Example usage
#     text = "She has eaten. Cinderella wanted to go to the ball."
#     npu = calculate_nouns_per_utterance(text)
#     vpu = calculate_verbs_per_utterance(text)
    
#     print("Nouns per utterance: {npu}")
#     print("Verbs per utterance: {vpu}")

# if __name__ == "__main__":
#     main()