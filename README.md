# AliceInWonderland
Exploring RNNs, GRUs, and RNN-RBMs using the Lewis Carroll 'Alice in Wonderland' series as the data source

Tested a Markov chain against the text and it beat all my neural network models. 

Just for fun I added a Twitter bot script that will read in the text, generate new text with a Markov Chain and post it to Twitter. This is the skelaton for another project, just added it here as a fun toy.


Final model is RNN_Alice.py
Final data cleaning is ParseDataIntoSentences.py

This is a single layer RNN with a window size of 8
After 2000 training loops it gets ~55% accuracy on predicted sentences, using only 2836 training sentences and a vocabulary of 4041 words. There are 56207 words total in Alice in Wonderland and Through the Looking Glass. 

The accuracy and cost are still improving at 2000 epochs, testing a long run as I type this.

The network begins with stuttering the most common word 'The'
Epoch:  0 the the the the the the the the the the

Later epochs begin to sound like proper speech:

Epoch:  1910 pretend the a way of getting through into it somehow like

Epoch:  1920 the the chorus of voices went on

Epoch:  1930 sure going i humpty dumpty

Epoch:  1940 the were dinah if i might venture to ask the question

Epoch:  1950 s be do you say how dye do

Epoch:  1960 the you dont see must make king said with a melancholy air and after the his voice she frowning at the cook till if eyes just little out of sight just said the a deep voice there are tarts made of

Epoch:  1970 the him of his book rule uneasy and

Epoch:  1980 the cats eat bats i wonder

Epoch:  1990 dont the couldnt if you tried
 


RNNs do not like outliers. Data was parsed into sentences (ParseDataIntoSentences.py), punctuation, capital letters and outliers (words used once were removed). This reduced our vocabulary from 6591 to 4041. To make predictions more interesting one off words were put into a unique words array and used to randomly replace the unique word token in predicted sentences. 




