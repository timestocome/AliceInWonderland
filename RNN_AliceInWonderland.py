#!/python
# https://github.com/timestocome


# Simple RNN in Python using numpy
# Doesn't learn a lot, or go very fast but it's a good network for seeing how recursive
# networks work



# starter code 
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# updated to python 3
# improved a few things ;)



import numpy as np

import theano as theano
import theano.tensor as T

import pickle
import operator
import timeit
import datetime
import sys
import random



# to do *****************************************************************************
# too slow to be useable but a good work through to gain understanding of recursive networks 
# takes about 6 minutes per epoch on my iMac

###############################################################################
# constants
unique_words = 3403             # unique words in our dictionary
rng = np.random.RandomState(42) # set up random stream
not_zero = 1e-6                 # avoid divide by zero errors



# settings to tweak
n_hidden = 128               # hidden layer number of nodes ( hidden layer width )
n_epoch = 2                  # number of times to loop through full data set
learning_rate = 0.9
decay = 0.01
print_output = 10            # print info for user how often during training epochs
n_layers = 1                 # number of gru hidden layers ( hidden layer height )
length_of_text = 8           # size of string to feed into RNN
bptt_truncate = length_of_text



###############################################################################
# misc functions
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

###############################################################################
# Alice in Wonderland
# Text read in, parsed and tokenized using ReadDataIntoWords.py
# this should be broken into proper sentences but for testing I'm splitting
#   it into 8 word strings, no punctuation

# tokenized data is 
tokenized_text = np.load('tokenized_document.npy')

# break into x, y
input = tokenized_text[0:-1]
target = tokenized_text[1:]

# break into testing, training
n_test = len(input) // 10
n_train = len(input) - n_test

train_x = input[0:n_train]
train_y = input[0:n_train]
test_x = input[n_train:]
test_y = input[n_train:]




# break into training vectors 
index = 0
x = []
y = []
for i in range(n_train):
    x.append(train_x[i:i+length_of_text])       # create a sentence
    y.append(train_y[i+1:i+1+length_of_text])   # shift sentence over one word

x_train = np.array(x)
y_train = np.array(y)


tx = []
ty = []
for i in range(n_test):
    tx.append(test_x[i:i+length_of_text])
    ty.append(test_y[i+1:i+1+length_of_text])

x_test = np.array(tx)
y_test = np.array(ty)

index_dictionary = pickle.load(open('index_dictionary.pkl', "rb"))
word_dictionary = pickle.load(open('word_dictionary.pkl', "rb"))

def index_to_word(w):
    z = index_dictionary.get(w)
    if z is None: return -1
    else: return z[0]

def word_to_index(i):
    z = word_dictionary.get(i)
    if z is None: return -1
    else: return z[0]


#print(x_train[0])
#print(y_train[0])


#############################################################################
# build RNN network

# input (unique_words)
# output (unique_words)
# s  (hidden)               # output from hidden fed into hidden in next cycle
# U (hidden, unique_words)  # input to hidden
# V (unique_words, hidden)  # hidden to output
# W (hidden, hidden)        # hidden to hidden

class RNN:


    def __init__(self, n_words=unique_words, n_hidden=n_hidden, bptt_truncate=length_of_text):

        # init constants
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.bptt_truncate = bptt_truncate

        # setup weights ( see Xavier, Yam/Chow, )
        self.U = np.random.uniform(-np.sqrt(1./n_words), np.sqrt(1./n_words), (n_hidden, n_words))
        self.V = np.random.uniform(-np.sqrt(1./n_hidden), np.sqrt(1./n_hidden), (n_words, n_hidden))
        self.W = np.random.uniform(-np.sqrt(1./n_hidden), np.sqrt(1./n_hidden), (n_hidden, n_hidden))

        

       

    def forward_propagation(self, x):

        time_steps = len(x) # same as sentence length, length_of_text

        # save all hidden output, init is zet to zero
        s = np.zeros((time_steps + 1, self.n_hidden))   
        s[-1] = np.zeros(self.n_hidden)

        # save all outputs for each time step
        o = np.zeros((time_steps, self.n_words))

        # for each word in sentence
        for t in range(time_steps):
            # [:,x[t]] is the same as a one hot vector for x
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]


    
    def predict(self, x):

        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)



    def predictions_to_words(self, p):

        text = []
        for i in p:
            w = index_to_word(i)
            text.append(w)
        return text



    # cross entropy loss
    def calculate_total_loss(self, x, y):

        loss = 0.0

        # for each example
        for i in range(len(y)):
             
             o, s = self.forward_propagation(x[i])
             correct_word_predictions = o[range(len(y[i])), y[i]] 
             loss += -1 * np.sum(np.log(correct_word_predictions))

        return loss



    # adjusted loss
    def calculate_loss(self, x, y):
        n = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / n
        



    def bptt(self, x, y):

        steps = len(y)

        # go forward
        o, s = self.forward_propagation(x)

        # accumulate gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[range(len(y)), y] = -1

        # for each output work backwards
        for t in range(steps)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            # init delta 
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] **2))

            # back propagate through time
            for bptt_step in range(max(0, steps - bptt_truncate), t+1)[::-1]:

                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t

                # update for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] **2)
                
        return [dLdU, dLdV, dLdW]



    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
    
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
    
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
 

    """
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):

        # backprop gradients
        bptt_gradients = self.bptt(x, y)

        # parameters to check 
        model_parameters = ['U', 'V', 'W']

        # check each parameter 
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)

            print("checking gradients for %s with size %d" %(pname, np.prod(parameter.shape)))

            # iterate
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])

            while not it.finished:
                ix = it.multi_index

                # save original
                original_value = parameter[ix]

                # estimate gradient (f(x+h) - f(x-h)) /(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)

                # Reset parameter to original value
                parameter[ix] = original_value
            
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
            
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print ("+h Loss: %f" % gradplus)
                    print ("-h Loss: %f" % gradminus)
                    print ("Estimated_gradient: %f" % estimated_gradient)
                    print ("Backpropagation gradient: %f" % backprop_gradient)
                    print ("Relative Error: %f" % relative_error)
                
                    return
            
                it.iternext()
            print ("Gradient check for parameter %s passed." % (pname))
            """



model = RNN()
model.sgd_step = RNN.sgd_step

"""
# test forward feed
o, s = model.forward_propagation(x_train[3])

predictions = model.predict(x_train[3])
print(predictions)    

sentence = model.predictions_to_words(predictions)
print(sentence)

# test loss function
print("Expected random loss ", np.log(unique_words))
print("Actual loss", model.calculate_loss(x_train, y_train))


# test gradients
n_checks = 1
np.random.seed(42)
model.gradient_check(x=[0,1,2,3,4], y=[1,2,3,4,5])
"""


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs

n_examples = len(y_train)
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):

    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0

    for epoch in range(nepoch):

        print(datetime.datetime.now())
        print("epoch ", epoch)

        # print a guess at a sentence
        r = random.randint(0, n_examples-1)
        predictions = model.predict(x_train[r])
        sentence = model.predictions_to_words(predictions)
        print(sentence)


        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):

            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            print ("Loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss))
           
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):

                learning_rate = learning_rate * 0.5 
                print ("Setting learning rate to %f" % learning_rate)
            
                
            sys.stdout.flush()
        
        # For each training example...
        for i in range(len(y_train)):
        
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


model = RNN()
train_with_sgd(model, x_train, y_train)


