
# http://github.com/timestocome

# Zipf's law on words
# log frequency = log c - s log rank




# http://greenteapress.com/complexity/Zipf.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################################################################
# read in data
from collections import Counter
from collections import OrderedDict
import operator



############################################################################
# read in text, clean it up, get word frequencies and sort


# open file and read in text
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
data = data.replace(" '", ' ')
data = data.replace("' ", ' ')
data = data.replace('-', ' ')
data = data.replace('*', '')
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
    frequency[k] = v   #/number_of_words * 100.



with open('word_frequency.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in frequency.items()]


# sorted
sorted_list = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)




#############################################################
words, counts = zip(*sorted_list)

# plot histogram
'''
words = np.array(words[0:50])
counts = np.array(counts[0:50])
indexes = np.arange(len(words))


bar_width = 0.2
plt.figure(figsize=(18,18))
plt.bar(indexes, counts)
plt.title("Alice in Wonderland, Through the Looking Glass")
plt.xticks(indexes + bar_width, words, rotation=45)
plt.savefig("histogram.png")
plt.show()
'''


# plot frequency vs ranks
# should be a straight line
# log counts = log c - s log ranks

words = np.array(words)
counts = np.array(counts)
ranks = np.arange(len(words))

plt.figure(figsize=(12,12))
plt.xscale('log')
plt.yscale('log')
plt.title('Zipf Plot Alice in Wonderland and Through the Looking Glass')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xticks(ranks + .2, words[0:20], rotation=45)
plt.plot(ranks, counts, lw=4)
plt.savefig('ZipfsLaw.png')
plt.show()



