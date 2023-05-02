from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.util import ngrams

# get the length of the smallest n gram
def get_gram_lentgh(uncommon_str_i):
    lens = []
    for i in range(len(uncommon_str_i[0])):
        temp = []
        for j in range(len(uncommon_str_i)):
            temp.append(len(uncommon_str_i[j][i]) if type(uncommon_str_i[j][i]) == list else 1)
        lens.append(min(temp))
    return lens

# get the original sentence in a vector form
def get_og_sentence_vector(uncommon_str, common_sentence):
    og_sentence_vector = []
    temp = common_sentence.split()
    i = 0    
    for t in temp:
        if t == "#":
            if type(uncommon_str[i]) == list:
                og_sentence_vector.extend(uncommon_str[i])
            else:
                og_sentence_vector.append(uncommon_str[i])
            i += 1
        else:
            og_sentence_vector.append(t)
    return og_sentence_vector

def init_list_of_lists(lenght):
    list_of_lists = []
    for i in range(lenght):
        list_of_lists.append([])
    return list_of_lists

# remove all the occourences of a value in a list
def remove_all(liste, value):
    while value in liste:
        liste.remove(value)
    return liste

def ngram_distribution(uncommon_str_i, common_sentence):
    # Initialize the list of lists that will contain the n-grams
    final_uncommon_str_i = init_list_of_lists(len(uncommon_str_i))

    nb_unc_str = 0

    lens = get_gram_lentgh(uncommon_str_i) # get the length of the smallest n grams

    for uncommon_str in uncommon_str_i:
        for i in range(len(uncommon_str)):
            # Make a copy of the current list of the current uncommon part for string 1
            unc_str = uncommon_str[i].copy() if type(uncommon_str[i]) == list else [uncommon_str[i]]
            og_sentence = get_og_sentence_vector(uncommon_str, common_sentence)
            temp_uncommon = uncommon_str[i].copy() if type(uncommon_str[i]) == list else [uncommon_str[i]]
            while len(unc_str) > lens[i]:
                
                bigram_measures = BigramAssocMeasures()

                # Variable containing the common words that won't allowed in the bigrams
                common_words_str = list(set(og_sentence) - set(unc_str))

                # Generate a list of all n-grams of size n for the sentence
                n_grams_str = list(ngrams(og_sentence, 2))
                
                # Use the bigram collocation finder to get the best bigrams for the sentence
                finder_str = BigramCollocationFinder.from_words(og_sentence)
                best_bigrams_str = finder_str.nbest(bigram_measures.pmi, len(n_grams_str))
                
                # Filter out bigrams that contain common words from the current list of uncommon words
                best_uncommon_ngrams_str = [ngram for ngram in best_bigrams_str if (not any(p_ngrams in ngram for p_ngrams in common_words_str))]
                
                # Generate the final list of uncommon n-grams for string 1 by filtering the filtered bigrams and remaining uncommon words
                uncommon_ngrams_str = [''] * len(unc_str)
                count1 = len(unc_str)
                count2 = 0
                # We loop through the best uncommon n-grams and check if they are in the uncommon words list
                for b in best_uncommon_ngrams_str:
                    if b[0] in unc_str and b[1] in unc_str: # if both words are in the uncommon words list
                        uncommon_ngrams_str[unc_str.index(b[0])] = " ".join(list(b)) # we add the n-gram to the final list
                        count2 += 1 # we increment the number of uncommon n-grams in the final list
                        # we remove the words of the bi-gram from the uncommon words list
                        unc_str[unc_str.index(b[0])] = '' 
                        unc_str[unc_str.index(b[1])] = ''
                        count1 -= 2 # we decrement the number of uncommon words in the uncommon words list
                    if count1 + count2 == lens[i]: # if we have the number of uncommon n-grams we want
                        break
                if unc_str != [""] * len(unc_str): # if there are still uncommon words left
                    for j in range(len(unc_str)):
                        if unc_str[j] != '':
                            uncommon_ngrams_str[j] = unc_str[j] # we add the uncommon words left to the final list
                uncommon_ngrams_str = remove_all(uncommon_ngrams_str, '') # we remove the empty strings from the final list
                unc_str = uncommon_ngrams_str.copy() # we update the current list of uncommon words
                og_sentence = unc_str.copy() # we update the current list of uncommon words
            
            final_uncommon_str_i[nb_unc_str].append(unc_str) # we add the final list of uncommon n-grams to the final list of lists
        nb_unc_str += 1 # we increment the number of uncommon parts
    return final_uncommon_str_i