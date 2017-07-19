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

# not going to worry about punctuation yet
data = data.replace('--', ' ')
data = data.replace('.', '')
data = data.replace(';', '')
data = data.replace(',', '')
data = data.replace('!', '')
data = data.replace('\"', '')
data = data.replace('(', '')
data = data.replace(')', '')
data = data.replace('?', '')
data = data.replace(':', '')
#print(data)

file.close()

# break text into words
words = data.split()
number_of_words = len(words)
print ("Total words in text", len(words))


# find unique words
words_set = set(words)
unique_words = len(words_set)
#print("Unique words", len(words_set))

# get word frequencies
# sort by most common in case we want to ditch some 
word_frequency_dictionary = Counter(words)
word_list_by_frequency = list(word_frequency_dictionary.keys())
#print(word_list_by_frequency)


# create indexed dictionary of words ( tokenize words in text )
word_index = list(range(0, len(words_set)))
word_index = [i + 1 for i in word_index]        # remove 0


word_dictionary = {}
for key, value in zip(word_list_by_frequency, word_index):
    word_dictionary[key] = []
    word_dictionary[key].append(value)
#print(word_dictionary)

# we need this because we can only do lookups by keys so reverse and save
index_dictionary = {}
for key,value in zip(word_index, word_list_by_frequency):
    index_dictionary[key] = []
    index_dictionary[key].append(value)


# save dictionaries
with open('word_dictionary.pkl', "wb") as f:
    pickle.dump(word_dictionary, f)

with open('index_dictionary.pkl', "wb") as f:
    pickle.dump(index_dictionary, f)




# tokenize entire training document
tokenized_document = []
for w in words:
    token = word_dictionary.get(w)
    if token is None:
        print("no token", w)
        tokenized_document.append(-1)
    else:
        tokenized_document.append(token[0])

np.save('tokenized_document.npy', tokenized_document)
#print(tokenized_document)
print("min/max", min(tokenized_document), max(tokenized_document))



