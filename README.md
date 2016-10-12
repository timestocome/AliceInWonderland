# AliceInWonderland
Exploring RNNs, GRUs using the Lewis Carroll 'Alice in Wonderland' series

Code is in Python 3.5/Theano 0.8

The training starts out well and flatlines. The regularization I added will keep the weights from blowing up so the gradient must be vanishing. Another possibility is that the network has reached a temporary plateau - I'm running this on a CPU not GPU so running longer might give the network time to move from the plateau.

Need to research this some while I'm working on the reinforcement stuff and come back to it.


