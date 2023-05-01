import numpy as np
import nltk
import re

# Remove punctuation
def preprocess(sentences, n):
    new_sentences = []
    for sentence in sentences:
        sentence = sentence[0]
        sentence = sentence.lower()
        if n == 1:
            sentence = nltk.RegexpTokenizer(r'\w+').tokenize(sentence)
            new_sentences.append(sentence)
        else :
            sentence = re.sub(r'[^\w\s]', '', sentence)
            new_sentences.append([sentence])
    if n == 1:
        return new_sentences  
    else: 
        return np.array(new_sentences)


# Transform the sentences into vectors
def split_preprocess(sentences):
    s = []
    for sentence in sentences:
        s.append(sentence[0].split())
    return s


