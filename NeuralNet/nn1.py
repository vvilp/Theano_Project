#
#  Single Layer Neural network
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
w = theano.shared(np.array([random(),random()]))
b = theano.shared(1.0)
learning_rate = 0.01

z = T.dot(x,w)+b
a = 1/(1+T.exp(-z)) #sigmoid

targets = T.vector('targets') #Actual output
cost = -(targets*T.log(a) + (1-targets)*T.log(1-a)).sum()
# cost = ((targets - a) ** 2).sum()

dw,db = T.grad(cost,[w,b])

train = function(
    inputs = [x,targets],
    outputs = [a,cost,dw,db],
    updates = [
        [w, w-learning_rate*dw],
        [b, b-learning_rate*db]
    ]
)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [1,0,0,1]

#Iterate through all inputs and find outputs:
cost = []
for iteration in range(10):
    pred, cost_iter, dw_iter, db_iter = train(inputs, outputs)
    cost.append(cost_iter)
    print ("dw:", dw_iter)

for i in range(len(inputs)):
    print (' Input:(%d, %d), output:%.2f, target:%.2f' % (inputs[i][0],inputs[i][1],pred[i],outputs[i]))

#Plot the flow of cost:
# print '\nThe flow of cost during model run is as following:'
import matplotlib.pyplot as plt
plt.plot(cost)
plt.show()