import numpy as np
import theano.tensor as T
from theano import pp  #pretty-print
from theano import function


# # basic function
# a = T.dscalar('a')
# b = T.dscalar('b')
# c = a*b
# f = function([a,b],c)
# print (f(1.5,3))

# # return multi values
# a = T.dscalar('a')
# f = function([a],[a**2, a**3])
# print (f(3))

# # Gradient
# x = T.dscalar('x')
# y = x**3
# qy = T.grad(y,x)
# f = function([x],qy)

# print(pp(qy))




a = T.matrix('a')
b = T.matrix('b')
c = T.vector('c')
d = T.dot(a, b) + c
e = d - c
f = e.sum()
g = T.sum(e)
test = function ([a,b,c], [d,e,f,g])

a = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [1, 2],
]

b = [[1,1,1],[0, 0, 0]]
c = [1,1,1]

d, e,f,g = test(a,b,c)

print (d)
print (e)
print (f)
print (g)