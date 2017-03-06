#
#  Multiple Layer Neural network
#

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from theano import pp  #pretty-print
from random import random
from theano import function

#Define variables:
x = T.matrix('x')

hidden_layers = 5;

w1 = theano.shared(np.random.rand(2,hidden_layers))
b1 = theano.shared(np.random.rand())
w2 = theano.shared(np.random.rand(hidden_layers))
b2 = theano.shared(np.random.rand())

# print(w2)

learning_rate = 0.01
targets = T.vector('targets')

h = T.dot(x,w1) + b1
h_a = 1 / (1 + T.exp(-h))

o = T.dot(h_a, w2) + b2
o_a = 1 / (1 + T.exp(-o))

cost = ((targets - o_a) ** 2).sum()

dw1, db1, dw2, db2 = T.grad(cost, [w1,b1,w2,b2])

train = function (
    inputs = [x, targets],
    outputs = [o_a, cost],
    updates = [
        [w1, w1-learning_rate * dw1],
        [w2, w2-learning_rate * dw2],
        [b1, b1-learning_rate * db1],
        [b2, b2-learning_rate * db1]
    ]
)


def train_iter(x, targets):
    return train(x, targets)
