#!/python
# https://github.com/timestocome


# Simple GRU in Python using numpy, theano


# starter code 
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
#
# updated to python 3.5 from python 2.x
# updated to Theano version 0.8.0
# adapted to read in 'Alice In Wonderland' and 'Through the Looking Glass' and generate text in same style 
# improved comments and variable names
# added regularization, then removed it. Seems to hurt more than help here
# removed sentence start/stop tokens so can more easily adapt to other types of input sequences
# lots of streamlining to make code clearer and faster
# used ReLU instead of sigmoid/tanh as activation function (much faster convergence, accuracy seems slightly better)
# removed one gru layer, the extra layer didn't help 



# to do ##########################################################################
# stop after x examples randomly trained or some loss threshold met
# adjust learning rate while training, simple or new self adjusting method?
# track loss, only overwrite saved model if loss is less than previous loss



import numpy as np

import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip

import pickle
import operator
import timeit
from datetime import datetime
import sys
import random




###############################################################################
# constants
unique_words = 4924             # unique words in our dictionary, get this from the output after running ReadDataIntoWords.py
rng = np.random.RandomState(42) # set up random stream
not_zero = 1e-6                 # avoid divide by zero errors


# network settings to tweak
n_hidden = 256               # hidden layer number of nodes ( hidden layer width )
n_epoch = 40                 # number of times to loop through full data set

learning_rate =4e-4        # limits swing in gradients
adjust_learning_rate = 1e-5 # decrease learning rate in regular steps
frequency_adjust_learning_rate = 1000  # how often to decrease learning rate



decay = 0.99                  # weight for prior information
length_of_text = 12           # size of string to feed into RNN
n_bptt_truncate = -1         # threshold back propagation through time, -1 means no early cut off


# misc settings for outputing info to user and saving data
number_of_words_to_generate = 12    # max number of words when generating sentences
dump_output = 1000
save_model = 10000

###############################################################################
# Alice in Wonderland, Through the Looking Glass
# Note: this is a small dataset to train a recursive network with
# might do better with a larger text source
# Also might do better with a more mainstream text source, lots of made up words and 
# poems and nonsense in the text
#
# Text read in, parsed and tokenized using ReadDataIntoWords.py
# this should be broken into proper sentences but for testing I'm splitting
#   it into 8 word strings, no punctuation

# tokenized data is 
tokenized_text = np.load('tokenized_document.npy')





# break into x, y
train_x = tokenized_text[0:-1]
train_y = tokenized_text[1:]

n_train = len(train_x)
print("Training examples ", n_train)            # 56,206



# break into training vectors 
index = 0
x = []
y = []
for i in range(n_train):
    x.append(train_x[i : i+length_of_text])         # create a training example
    y.append(train_y[i: i+length_of_text])          # shift training by one for targets

x_t = np.array(x)
y_t = np.array(y)



def randomize_data():
    training_range = np.arange(len(y))
    training_range = np.random.permutation(training_range)

    x_train = []
    y_train = []
    for i in training_range:
        x_train.append(x_t[i])
        y_train.append(y_t[i])

    return x_train, y_train




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

# vectorized versions of dictionary lookups so I can easily send arrays to functions
v_index_to_word = np.vectorize(index_to_word)
v_word_to_index = np.vectorize(word_to_index)

#############################################################################
# useful things
print("Expected loss for random predictions: %f" %(np.log(unique_words)))

#############################################################################
# build GRU network

# input (unique_words)
# output (unique_words)
# s (hidden)                # output from hidden fed into hidden in next cycle
# U (hidden, unique_words)  # input to hidden
# V (unique_words, hidden)  # hidden to output
# W (hidden, hidden)        # hidden to hidden

class GRU:

    def __init__(self, n_words=unique_words, n_hidden=n_hidden, bptt_truncate=n_bptt_truncate):

        # init constants
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.bptt_truncate = bptt_truncate
        n_gates = 3 # (Z: update gate, R: reset gate, C: previous hidden output)
        n_layers = 1       


         # Initialize the network parameters
         # input to hidden weights
        E = np.random.uniform(-np.sqrt(1./n_words), np.sqrt(1./n_words), (n_hidden, n_words))

         # weights for GRU ( U:inputs, W:previousHidden, V:outputs)
        U = np.random.uniform(-np.sqrt(1./n_hidden), np.sqrt(1./n_hidden), (n_gates * n_layers, n_hidden, n_hidden))
        W = np.random.uniform(-np.sqrt(1./n_hidden), np.sqrt(1./n_hidden), (n_gates * n_layers, n_hidden, n_hidden))
        V = np.random.uniform(-np.sqrt(1./n_hidden), np.sqrt(1./n_hidden), (n_words, n_hidden))
        
        # biases ( Z, R, C )
        b = np.zeros((n_gates * n_layers, n_hidden))

        # store hidden output ( memory )
        c = np.zeros(n_words)


        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        

        # SGD / rmsprop: Initialize parameters for derivatives
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

        
    def __theano_build__(self):
       
        # setup
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')


        # input, previous hidden layer 1, previous hidden layer 2
        # Z = activation(X.U + ST.W + b)
        # R = activation(X.U + ST.W + b)
        # H = activation(X.U + (ST*R).W + b)
        # ST = (1-Z) * H + Z*ST
        def forward_propagation(x_t, s_t1_prev):

            x_e = E[:, x_t]     # input layer

            # GRU layer # 1
            z_t1 = T.nnet.relu(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.relu(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
             
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            # changing softmax temp (scale) from 1 to lower (0.5) increases network's confidence
            o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]

            return [o_t, s_t1]






        # recurse through GRU layers
        [o, s1], updates = theano.scan(
            forward_propagation,
            sequences = x,
            truncate_gradient = self.bptt_truncate,
            outputs_info = [None, dict(initial=T.zeros(self.n_hidden))])


            
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost 
        cost = o_error 

       
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.predict = theano.function([x], o, allow_input_downcast=True)
        self.predict_class = theano.function([x], prediction, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)    # cost to show user
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc], allow_input_downcast=True)


        # backwards propagation
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')


      
        
        # rmsprop cache updates
        # rms frequent features get small changes, rare ones get large changes
        mE = decay * self.mE + (1 - decay) * dE **2
        mU = decay * self.mU + (1 - decay) * dU **2
        mW = decay * self.mW + (1 - decay) * dW **2
        mV = decay * self.mV + (1 - decay) * dV **2
        mb = decay * self.mb + (1 - decay) * db **2
        mc = decay * self.mc + (1 - decay) * dc **2
        
        
        # loop backwards
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + not_zero)),
                     (U, U - learning_rate * dU / T.sqrt(mU + not_zero)),
                     (W, W - learning_rate * dW / T.sqrt(mW + not_zero)),
                     (V, V - learning_rate * dV / T.sqrt(mV + not_zero)),
                     (b, b - learning_rate * db / T.sqrt(mb + not_zero)),
                     (c, c - learning_rate * dc / T.sqrt(mc + not_zero)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ], allow_input_downcast=True)
        
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    # Loss adjusted for number of unique words
    def calculate_loss(self, X, Y):
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

   


#################################################################################################
# generate text 

def generate_sentence(model, index_to_word, word_to_index):

    # We start the sentence with a random word from our vocabulary
    start_token = random.randint(0, unique_words)
    new_sentence = [start_token]
    
    # Repeat until we max words reached
    while len(new_sentence) < number_of_words_to_generate:

        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        
    sentence_str = v_index_to_word(new_sentence)
    print(" ".join(sentence_str))
    print("*********************************************************************")
    sys.stdout.flush()

    return new_sentence


################################################################################
# save and reload saved model


model_file = "saved_model"

def save_model_parameters_theano(model):

    np.savez(model_file,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    
    print ("Saved model parameters to %s. --------------------------------------------" % model_file)



def load_model_parameters_theano(path, modelClass=GRU):

    print("loading saved model")
    npzfile = np.load(path)

    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    
    print ("Building model model from %s with hidden_dim=%d word_dim=%d" % (path, n_hidden, unique_words))
    sys.stdout.flush()
    
    model = modelClass(unique_words, hidden_dim=n_hidden)

    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    
    return model 


###############################################################################################
# training

model = GRU()

def train_with_sgd(model, X_train, y_train, learning_rate=learning_rate, nepoch=n_epoch, decay=decay, callback_every=dump_output, callback=None):

    num_examples_seen = 0
    
    # For each training example...
    for i in range(0, n_train):
    
        # One SGD step
        model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
        num_examples_seen += 1
    
        # dumb, steady learning rate adjustment
        if num_examples_seen % frequency_adjust_learning_rate == 0:
            learning_rate -= adjust_learning_rate       # steadily decrease learning rate
            print ("Current learning rate %f" % learning_rate)

        # Optionally send some output to user
        if num_examples_seen % dump_output == 0:
            callback(model, num_examples_seen)      

    return model



# Save model and give user some feedback on progress
track_losses = []

def sgd_callback(model, num_examples_seen):

  dt = datetime.now()
  
  loss = model.calculate_loss(x_train[:1000], y_train[:1000])
  track_losses.append(loss)     # store for graphing later

  print("\n%s (training examples processed: %d)" % (dt, num_examples_seen))
  print("Loss: %f" % loss)
  print("Accuracy: %f" % (1. - loss / np.log(unique_words))   )
  
  generate_sentence(model, index_to_word, word_to_index)

  if num_examples_seen % save_model == 0:
      save_model_parameters_theano(model)
      np.savetxt('losses_by_run', track_losses)
  print("\n")
  sys.stdout.flush()



# main training loop
for epoch in range(n_epoch):

    x_train, y_train = randomize_data()

    train_with_sgd(model, x_train, y_train, learning_rate=learning_rate, nepoch=n_epoch, decay=decay, 
                    callback_every=dump_output, callback=sgd_callback)


