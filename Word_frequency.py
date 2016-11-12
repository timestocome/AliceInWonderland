#!/python

# use this to strip f*ing unicode from text files
# $ perl -i.bak -pe 's/[^[:ascii:]]//g' BothBooks.txt 


from collections import Counter
from collections import OrderedDict
import operator

import numpy as np
import pickle
import csv


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
#print(data)

file.close()

# break text into words
words = data.split()
number_of_words = len(words)
print ("Total words in text", len(words))


# find unique words
words_set = set(words)
unique_words = len(words_set)
print("Unique words", len(words_set))



# count words
count = Counter(words)

with open('word_count.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in count.items()]



# word frequency
frequency = {}
for k, v in count.items():
    frequency[k] = v/number_of_words * 100.

with open('word_frequency.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in frequency.items()]


# sorted
sorted_list = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_list)    

with open('sorted_word_frequency.csv', 'w') as f:
    for i in sorted_list:
        row = "%s,%f\n" % (i[0], i[1])
        print(row)
        f.write(row)


