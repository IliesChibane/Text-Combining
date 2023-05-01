import spacy
import pandas as pd
import numpy as np
import sys

from ..MLM.MLM import init_distilbert

import warnings
warnings.filterwarnings('ignore')

def similarity_analysis(masked_sentence, final_uncommon_str):
    nlp = spacy.load("en_core_web_md")
    fill_mask = init_distilbert()

    i = 0
    while "[MASK]" in masked_sentence:
        # MLM with BERT
        pred = fill_mask(masked_sentence)
        # Similarity between the masked words and the uncommon words with word embeddings
        nlp = spacy.load("en_core_web_md")  
        if type(pred[0]) == list:
            df1 = pd.DataFrame(pred[0])
        else:
            df1 = pd.DataFrame(pred) # Convert the prediction to a dataframe
        word_list = df1["token_str"].tolist() # Get the list of words from the dataframe
        # Get the list of uncommon words for the current masked word
        strings = []
        for fus in final_uncommon_str:
            strings.append(fus[i])

        # Get the similarity between the masked word and the uncommon words
        similarity = []
        for s in strings:
            similarity.append(np.mean([nlp(w).similarity(nlp(s)) for w in word_list]))

        # Select the uncommon word with the highest similarity
        selected_word = strings[np.argmax(similarity)]
        masked_sentence = masked_sentence.replace("[MASK]", selected_word, 1)
        i += 1
    
    return masked_sentence