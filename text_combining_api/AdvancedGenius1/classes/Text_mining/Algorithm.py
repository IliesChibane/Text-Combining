from .uncommon_parts_extraction import common_and_uncommon_extraction
from .preprocessing import preprocess

def text_mining_algorithm(sentences):
    tokenized_sentences = preprocess(sentences, 1)

    common_words, uncommon_words = common_and_uncommon_extraction(tokenized_sentences)

    return common_words, uncommon_words