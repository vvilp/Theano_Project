import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from theano import pp  #pretty-print
from random import random
from theano import function

# Basic 
x = T.vector('x')
w = theano.shared(np.array([1,1]))
b = theano.shared(-1.5)

z = T.dot(x,w)+b
a = ifelse(T.lt(z,0),0,1)

neuron = theano.function([x],a)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

for i in range(len(inputs)):
    t = inputs[i]
    out = neuron(t)
    print (out)

