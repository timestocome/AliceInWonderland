#!/python

# use this to strip f*ing unicode from text files
# $ perl -i.bak -pe 's/[^[:ascii:]]//g' BothBooks.txt 


from collections import Counter
from collections import OrderedDict

import numpy as np
import pickle



# open file and read in text
#file = open('AliceInWonderland.txt', 'r')
file = open('BothBooks.txt', encoding='utf-8')
data = file.read()


# convert all text to lower case and remove new line chars
data = data.lower()
data = data.replace('\n', ' ')

file.close()

# break text into chars
#words = data.split()
#number_of_words = len(words)
#print ("Total words in text", len(words))
number_of_chars = len(data)
print("Total chars in text", number_of_chars)



# find unique chars
chars_set = set(data)
unique_chars = len(chars_set)
print("Unique chars", len(chars_set))




# get char frequencies
# sort by most common in case we want to ditch some 
char_frequency_dictionary = Counter(data)
char_list_by_frequency = list(char_frequency_dictionary.keys())
#print(char_list_by_frequency)


# create indexed dictionary of chars ( tokenize words in text )
char_index = list(range(1, len(chars_set)))

char_dictionary = {}
for key, value in zip(char_list_by_frequency, char_index):
    char_dictionary[key] = []
    char_dictionary[key].append(value)
print(char_dictionary)



# we need this because we can only do lookups by keys so reverse and save
index_dictionary = {}
for key,value in zip(char_index, char_list_by_frequency):
    index_dictionary[key] = []
    index_dictionary[key].append(value)


# save dictionaries
with open('char_dictionary.pkl', "wb") as f:
    pickle.dump(char_dictionary, f)

with open('char_index_dictionary.pkl', "wb") as f:
    pickle.dump(index_dictionary, f)




# tokenize entire training document
tokenized_document = []
for c in data:
    token = char_dictionary.get(c)
    if token is None:
        tokenized_document.append(-1)
    else:
        tokenized_document.append(token[0])

np.save('char_tokenized_document.npy', tokenized_document)



