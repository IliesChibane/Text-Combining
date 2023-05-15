from .n_grams_distribution import ngram_distribution

# Reduce the sequences of # into one #
def shrink(sentence):
    temp = sentence.split()
    b = False
    for i in range(len(temp)):
        if temp[i] == "#" and b:
            temp[i] = ""
        elif temp[i] == "#" and not b:
            b = True
        elif temp[i] != "#" and b:
            b = False
    while "" in temp:       
        temp.remove("")
    
    return " ".join(temp)

def flatten(final_uncommon_str):
    flatten_final_uncommon_str = []
    for i in range(len(final_uncommon_str)):
        flatten_final_uncommon_str.append([item for sublist in final_uncommon_str[i] for item in sublist])
    return flatten_final_uncommon_str

# Init the Dynamic matrix
def init_matrix(temp_sentence, sentences, lenght, l):
        # initialize the L matrix with zeros
        L = [[0] * (lenght + 1) for _ in range(len(temp_sentence) + 1)]

        # fill in the L matrix using dynamic programming
        for i in range(len(temp_sentence) + 1):
            for j in range(lenght + 1):
                # if either string is empty, the longest common substring is zero
                if i == 0 or j == 0:
                    L[i][j] = 0
                # if the characters match, add one to the length of the longest common substring
                elif temp_sentence[i - 1] == sentences[l][j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                # if the characters don't match, take the maximum length from the previous row or column
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        return L

# init list of lists
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


def common_and_uncommon_extraction(sentences):
    lens = [len(s) for s in sentences]

    # initialize the uncommon substring lists
    uncommon_str_i = init_list_of_lists(len(sentences))

    temp_sentence = sentences[0]
    for l in range(1, len(sentences)):
        # initialize the L matrix
        L = init_matrix(temp_sentence, sentences, lens[l], l)

        # calculate the index based on the length of the longer string
        index = len(temp_sentence) + lens[l]

        # initialize the common list with empty strings
        common = [""] * (index + 1)
        common[index] = ""

        # set i and j to the end of each string
        i = len(temp_sentence)
        j = lens[l]
        limit = abs(i - j)

        # trackers to follow the uncommon substrings position
        tracker_str1 = -1 
        tracker_str2 = -1
        # lists that save a sequence of uncommon substrings
        sub_uncommon_str = []
        sub_uncommon = []
        # final list that contains all the uncommon substrings
        sub_uncommon_str_i_temp = []
        sub_uncommon_str_temp = init_list_of_lists(len(sentences))

        # loop through the L matrix to find the common and uncommon substrings
        while i > 0 and j > 0:
            
            # if the characters match, add the character to the common list and move to the previous diagonal cell
            dist = abs(i - j)
            if temp_sentence[i - 1] == sentences[l][j - 1] and dist <= limit:
                common[index - 1] = temp_sentence[i - 1]
                i -= 1
                j -= 1
                index -= 1
            # if the length of the substring from the previous column is greater, add the uncommon character to uncommon_str list and move to the previous column
            elif L[i - 1][j] < L[i][j - 1]:
                if tracker_str1 == -1: # if the tracker is -1, it means that the substring is the first one
                    tracker_str1 = j - 1
                    sub_uncommon_str.append(sentences[l][j - 1]) # add the uncommon character to the list
                elif tracker_str1 == j: # if the tracker is equal to the current index, it means that the substring is part of the same sequence
                    sub_uncommon_str.append(sentences[l][j - 1]) # add the uncommon character to the sequence list
                    tracker_str1 = j - 1
                else: # if the tracker is not equal to the current index, it means that the substring is part of a different sequence
                    sub_uncommon_str.reverse() 
                    # add the sequence to the final list
                    none_index = get_last(uncommon_str_i[l], "")
                    if none_index != -1:
                        uncommon_str_i[l][none_index] = sub_uncommon_str if len(sub_uncommon_str) > 1 else sub_uncommon_str[0] # add the uncommon string to the new sequence list
                    else : uncommon_str_i[l].append(sub_uncommon_str if len(sub_uncommon_str) > 1 else sub_uncommon_str[0])
                    sub_uncommon_str = [] # reset the sequence list
                    tracker_str1 = j - 1 # reset the tracker to the first uncommon string of the new sequence
                    sub_uncommon_str.append(sentences[l][j - 1]) # add the uncommon string to the new sequence list

                j -= 1 # move to the previous column
                common[index - 1] = "#"
                index -= 1
            # if the length of the substring from the previous row is greater, add the uncommon character to uncommon_str2 list and move to the previous row
            else:
                if tracker_str2 == -1: # if the tracker is -1, it means that the substring is the first one
                    tracker_str2 = i - 1
                    sub_uncommon.append(temp_sentence[i - 1]) # add the uncommon character to the list
                elif tracker_str2 == i: # if the tracker is equal to the current index, it means that the substring is part of the same sequence
                    sub_uncommon.append(temp_sentence[i - 1]) # add the uncommon character to the sequence list
                    tracker_str2 = i - 1
                else: # if the tracker is not equal to the current index, it means that the substring is part of a different sequence
                    sub_uncommon.reverse()
                    if l == 1: # if the index point to the second string, it means we are dealing with the first string so we add the sequence to the final list 
                        uncommon_str_i[0].append(sub_uncommon if len(sub_uncommon) > 1 else sub_uncommon[0])
                    else: # else it means that we are dealing with the common sentence 
                        if '#' not in sub_uncommon: # if the sequence doesn't contain the # character, it means it is a new sequence so we add it to the final list directly
                            #sub_uncommon.reverse()
                            # we add the uncommon substring to all the uncommon parts of all the previous strings
                            for k in range(l):
                                sub_uncommon_str_temp[k].append(sub_uncommon if len(sub_uncommon) > 1 else sub_uncommon[0])
                        else: # if the sequence contains the # character, it means that it is a sequence that is part of a previous sequence so we need to update it
                            sub_uncommon_copy = sub_uncommon.copy()
                            # we add the uncommon substring to a temp list to not mess up the order of the final list
                            sub_uncommon_str_i_temp.append(sub_uncommon_copy if len(sub_uncommon_copy) > 1 else sub_uncommon_copy[0])
                            for k in range(l):
                                sub_uncommon_copy = sub_uncommon.copy()
                                uwu = 1
                                while "#" in sub_uncommon_copy and len(sub_uncommon_str_i_temp) - uwu < len(uncommon_str_i[k]): # we loop through the uncommon substring and replace the # character with the uncommon substring
                                    # we get the last uncommon substring of the previous string
                                    updated_uncommon_str = uncommon_str_i[k][len(sub_uncommon_str_i_temp) - uwu]
                                    if type(updated_uncommon_str) == list: # if the last uncommon substring is a list, it means that it is a sequence so we need to update it
                                        owo = len(updated_uncommon_str) - 1
                                        while owo >= 0: # we loop through the sequence and replace the # character with the uncommon substring
                                            if '#' in sub_uncommon_copy:
                                                ind = max(loc for loc, val in enumerate(sub_uncommon_copy) if val == '#')
                                                sub_uncommon_copy[ind] = updated_uncommon_str[owo]
                                            owo -= 1
                                    else:
                                        ind = sub_uncommon_copy.index("#")
                                        sub_uncommon_copy[ind] = updated_uncommon_str
                                    uwu -= 1
                                if "#" in sub_uncommon_copy:
                                    sub_uncommon_copy = remove_all(sub_uncommon_copy, '#') # we remove all the # characters that are left
                                sub_uncommon_str_temp[k].append(sub_uncommon_copy if len(sub_uncommon) > 1 else sub_uncommon_copy[0]) # we add the updated uncommon substring to the final list                   
                    sub_uncommon = [] # reset the sequence list
                    tracker_str2 = i - 1 # reset the tracker to the first uncommon string of the new sequence
                    sub_uncommon.append(temp_sentence[i - 1]) # add the uncommon string to the new sequence list
                    uncommon_str_i[l].append("")

                common[index - 1] = "#" # add the # character to the common substring to indicate that an uncommon substring is there
                index -= 1 # move to the previous row
                i  -= 1 # move to the next string
        
        if l == 1: # if the index point to the second string, it means we are dealing with the first string 
            if len(sub_uncommon) > 0: # if the length of the substring is greater than 0, it means that there is an uncommon substring left
                sub_uncommon.reverse()
                uncommon_str_i[0].append(sub_uncommon if len(sub_uncommon) > 1 else sub_uncommon[0]) # add the uncommon substring to the final list
        else: # else it means that we are dealing with the common sentence
            if len(sub_uncommon) > 0: # if the length of the substring is greater than 0, it means that there is an uncommon substring left
                if '#' not in sub_uncommon: # if the sequence doesn't contain the # character, it means it is a new sequence so we add it to the final list directly
                    sub_uncommon.reverse()
                    for k in range(l):
                        sub_uncommon_str_temp[k].append(sub_uncommon if len(sub_uncommon) > 1 else sub_uncommon[0])
                else: # if the sequence contains the # character, it means that it is a sequence that is part of a previous sequence so we need to update it
                    sub_uncommon.reverse()
                    for k in range(l):
                        sub_uncommon_copy = sub_uncommon.copy()
                        if len(sub_uncommon_copy) < 2: # if the length of the uncommon substring is less than 2, it means that it is a sequence of a single string so we just replace the # character with the uncommon substring
                            sub_uncommon_copy = uncommon_str_i[k][len(uncommon_str_i[k]) - 1][0] if type(uncommon_str_i[k][len(uncommon_str_i[k]) - 1]) == list else uncommon_str_i[k][len(uncommon_str_i[k]) - 1]
                        else: # if the length of the uncommon substring is greater than 2, it means that it is a sequence so we need to update it
                            uwu = 1
                            while "#" in sub_uncommon_copy and len(uncommon_str_i[k]) - uwu >= 0: # we loop through the uncommon substring and replace the # character with the uncommon substring
                                if type(uncommon_str_i[k][len(uncommon_str_i[k]) - uwu]) == list :
                                    # we loop through the terms of the sequence that needs to be updated and replace the # character with the uncommon substring
                                    for term in uncommon_str_i[k][len(uncommon_str_i[k]) - uwu]:
                                        if '#' in sub_uncommon_copy:
                                            ind = sub_uncommon_copy.index("#")
                                        sub_uncommon_copy[ind] = term
                                else: # if the last uncommon substring is not a list, it means that it is a sequence of a single string so we just replace the # character with the uncommon substring
                                    ind = sub_uncommon_copy.index("#")
                                    sub_uncommon_copy[ind] = uncommon_str_i[k][len(uncommon_str_i[k]) - 1]
                                uwu += 1
                        
                        
                        if type(uncommon_str_i[k][len(uncommon_str_i[k]) - 1][0]) == list : sub_uncommon_copy = remove_all(sub_uncommon_copy, "#") # we remove all the # characters that are left
                        sub_uncommon_str_temp[k].append(sub_uncommon_copy) # we add the updated uncommon substring to the final list
            # we add the uncommon substring to all the uncommon parts of all the previous strings
            for k in range(l):
                checking = shrink(" ".join(common)).split("#")
                nu = len(checking) - 1
                if temp_sentence[0] == "#":
                    nu += 1
                if len(sub_uncommon_str_temp[k]) < nu:
                    for q in range(0, len(uncommon_str_i[k]) - len(sub_uncommon_str_temp[k])):
                        sub_uncommon_str_temp[k].insert(0, uncommon_str_i[k][q])
                uncommon_str_i[k] = sub_uncommon_str_temp[k]
        
        if i != 0:
            temp_i = i
            sub_uncommon_str2 = [] # reset the sequence list
            while i > 0:
                sub_uncommon_str2.append(temp_sentence[i - 1])
                i -= 1
            sub_uncommon_str2.reverse()
            # add the sequence to the final list
            for k in range(l):
                if temp_i < len(temp_sentence):
                    if temp_sentence[temp_i] == "#":
                        f_unc = uncommon_str_i[k][len(uncommon_str_i[k]) - 1]
                        uncommon_str_i[k].remove(f_unc)
                        sub_uncommon_str2.extend(f_unc)
                uncommon_str_i[k].append(sub_uncommon_str2 if len(sub_uncommon_str2) > 1 else sub_uncommon_str2[0])
                uncommon_str_i[k] = remove_all(uncommon_str_i[k], "#")
            if common[0] != "#" and len(shrink(" ".join(common)).split("#")) < len(uncommon_str_i[0]):
                common.insert(0, "#")

        # we add the uncommon substring left to the current string
        if len(sub_uncommon_str) > 0:
            sub_uncommon_str.reverse()
            none_index = get_last(uncommon_str_i[l], "")
            if none_index != -1:
                uncommon_str_i[l][none_index] = sub_uncommon_str if len(sub_uncommon_str) > 1 else sub_uncommon_str[0] # add the uncommon string to the new sequence list
            else : uncommon_str_i[l].append(sub_uncommon_str if len(sub_uncommon_str) > 1 else sub_uncommon_str[0])
            if len(uncommon_str_i[l]) < len(uncommon_str_i[l - 1]):
                uncommon_str_i[l].append("")

        if j != 0:
            sub_uncommon_str = [] # reset the sequence list
            while len(uncommon_str_i[l]) + 1 > len(uncommon_str_i[l - 1]) and "" in uncommon_str_i[l]:
                uncommon_str_i[l].remove("")
            while j > 0:
                sub_uncommon_str.append(sentences[l][j - 1])
                j -= 1
            sub_uncommon_str.reverse()
            # add the sequence to the final list
            uncommon_str_i[l].append(sub_uncommon_str if len(sub_uncommon_str) > 1 else sub_uncommon_str[0])
            if common[0] != "#" and len(shrink(" ".join(common)).split("#")) < len(uncommon_str_i[0]):
                common.insert(0, "#")

        temp_sentence = remove_all(common.copy(), "") # we update the common sentence

        for rt in range(0, l):
            while len(uncommon_str_i[l]) != len(uncommon_str_i[rt]):
                if len(uncommon_str_i[l]) < len(uncommon_str_i[rt]):
                    uncommon_str_i[l].append("")
                else:
                    uncommon_str_i[rt].append("")

        if len(uncommon_str_i[l]) != len(shrink(" ".join(common)).split("#")) - 1:
            for rt in range(0, l+1):
                if len(uncommon_str_i[rt]) < len(shrink(" ".join(common)).split("#")) - 1:
                    uncommon_str_i[rt].append("")

        # N-gram distribution on the uncommon parts
        uncommon_str_i[0:l+1] = ngram_distribution(uncommon_str_i[0:l+1], shrink(" ".join(temp_sentence)))
        temp_sentence = shrink(" ".join(temp_sentence))

        # update the distribution of the uncommon parts based on the N-gram distribution
        for i in range(len(uncommon_str_i[0]), 0, -1):
            mask = "$ " * len(uncommon_str_i[0][i-1])
            temp_sentence = temp_sentence.replace("#", mask, 1)
        temp_sentence = temp_sentence.replace("$", "#")
        temp_sentence = temp_sentence.split(" ")
        temp_sentence = remove_all(temp_sentence, "")

    # join the common list into a sentence
    common_sentence = " ".join(temp_sentence)
    # replace the # character with the [MASK] token
    common_sentence = common_sentence.replace("#", "[MASK]")

    # reverse the order of the uncommon substring lists
    for i in range(len(uncommon_str_i)):
        uncommon_str_i[i].reverse()

    # return the common sentence and the lists of uncommon substrings
    return common_sentence, uncommon_str_i