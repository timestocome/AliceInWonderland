
# http://github.com/timestocome/
# build a markov chain and use it to predict Alice In Wonderland/Through the Looking Glass text


import numpy as np
import pickle
from collections import Counter

import markovify  # https://github.com/jsvine/markovify


#######################################################################
# read in text and break into words and sentences
#####################################################################


# open file and read in text
#file = open('AliceInWonderland.txt', 'r')
file = open('BothBooks.txt', encoding='utf-8')
data = file.read()
file.close()


# create markov model
model_3 = markovify.Text(data, state_size=3)


# generate text from model
print("*******************************")
for i in range(10):
    print("__________________________")
    print(model_3.make_sentence())