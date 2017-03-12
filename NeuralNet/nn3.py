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
import time


input_units_num = 784
hidden_units_num = 5
output_unit_num = 10

rng = np.random.RandomState((np.int64)(time.time()))
learning_rate = theano.shared(0.001)
w1 = theano.shared(rng.uniform(-1.0, 1.0, (input_units_num, hidden_units_num)))
w2 = theano.shared(rng.uniform(-1.0, 1.0, (hidden_units_num,output_unit_num)))
b1 = theano.shared(rng.uniform(-0.5, 0.5, (hidden_units_num)))
b2 = theano.shared(rng.uniform(-0.5, 0.5, (output_unit_num)))


input = T.matrix('input')
targets = T.matrix('target')

h = T.dot(input, w1) + b1
h_a = 1 / (1 + T.exp(-h))
# h_a = T.maximum(T.minimum(1,h), 0)
# h_a = 0.5 * h + 0.5

o = T.dot(h_a, w2) + b2
o_a = 1 / (1 + T.exp(-o))
# o_a = T.switch(o<0, 0, 1)

cost = ((targets - o_a) ** 2).sum() / 2
# cost = -(targets*T.log(o_a) + (1-targets)*T.log(1-o_a)).sum()

dw1 = T.grad(cost, w1)
db1 = T.grad(cost, b1)
dw2 = T.grad(cost, w2)
db2 = T.grad(cost, b2)

train = function(
    inputs = [input, targets],
    outputs = [o_a, cost],
    updates = [
        [w1, w1 - learning_rate * dw1 ],
        [w2, w2 - learning_rate * dw2 ],
        [b1, b1 - learning_rate * db1 ],
        [b2, b2 - learning_rate * db2 ],
    ]
)

predict = function(
    inputs = [input],
    outputs = [o_a]
)

class NeuralNet:
    def __init__(self, input_units_num, hidden_units_num, output_unit_num, lrate):
        w1.set_value(rng.uniform(-0.5, 0.5, (input_units_num, hidden_units_num)))
        w2.set_value(rng.uniform(-0.5, 0.5, (hidden_units_num,output_unit_num)))
        b1.set_value(rng.uniform(-0.5, 0.5, (hidden_units_num)))
        b2.set_value(rng.uniform(-0.5, 0.5, (output_unit_num)))
        
        learning_rate.set_value(lrate)

    def predict_(self, x):
        return predict(x)

    def train_iter(self, x, t):
        return train(x, t)

    def train_iter_momentum(self, x, t):
        return train(x, t)

    
