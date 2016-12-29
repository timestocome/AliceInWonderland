#!/python

# use this to strip f*ing unicode from text files
# $ perl -i.bak -pe 's/[^[:ascii:]]//g' BothBooks.txt 


from collections import Counter
from collections import OrderedDict

import numpy as np
import pickle
import os


# open file and read in text
#file = open('AliceInWonderland.txt', 'r')
file = open('BothBooks.txt', encoding='utf-8')
data = file.read()


# convert all text to lower case and remove new line chars
data = data.lower()
data = data.replace('\n', ' ')

file.close()

# break text into chars
# words = data.split()
# number_of_words = len(words)
# print ("Total words in text", len(words))
number_of_chars = len(data)
print("Total chars in text", number_of_chars)



# find unique chars
chars_set = set(data)
unique_chars = len(chars_set)
#print("Unique chars", len(chars_set))
#print("unique chars", chars_set)



# get char frequencies
# sort by most common in case we want to ditch some 
char_frequency_dictionary = Counter(data)
char_list_by_frequency = list(char_frequency_dictionary.keys())
#print(char_list_by_frequency)


# create indexed dictionary of chars ( tokenize words in text )
char_index = list(range(0, len(chars_set)))
char_index = [i+1 for i in char_index]        # remove zeros
print(char_index)


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



############################################
# change this:
# make one file per sentence
# and stuff all the sentences into a directory 
# tokenize entire training document



# store files in data directory
current_directory = os.getcwd()
data_dir = 'char_sentences'

if not os.path.exists(data_dir):
    os.makedirs(data_dir) 
os.chdir(data_dir)

# full sentences are too long - some reach 921 chars
# going to use phrases instead
max_length = 0
min_length = 999999999
tokenized_string = []
loop_count = 0
for c in data:

    # while not end of sentence
    # ?, !, ., 
    token = char_dictionary.get(c)
    #print(c, token)
    
    tokenized_string.append(token)

    if c == '?' or c == '.' or c == '!' or c == ',':

        # save at end of sentence with unique id        
        loop_count += 1
        file_name = 'char_%s' % str(loop_count)
        flattened = [y for x in tokenized_string for y in x]
        np.save(file_name, flattened)
        if len(tokenized_string) > max_length: max_length = len(tokenized_string)
        if len(tokenized_string) < min_length: min_length = len(tokenized_string)
        tokenized_string = []

       
        #print(file_name)

print("min/max", max_length, min_length)


# check files
file_list = os.listdir(os.getcwd())
print(file_list)
count = 0
for f in file_list:    
    count += 1
    if count > 10: break
    test = np.load(f)
    print(test)
    print("**********************")
