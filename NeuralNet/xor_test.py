import nn3 as nn

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
targets = [[1],[0],[0],[1]]
# targets = [1,0,0,1]


cost = []
# nn.init()
for iteration in range(30000):
    out_iter, cost_iter = nn.train(inputs, targets)
    cost.append(cost_iter)
print("final loss:%.3f" % cost[len(cost)-1])

# for i in range(len(inputs)):
#     print ("Input:({0}, {1}), output:{2}, target:{3}".format(inputs[i][0],inputs[i][1],out_iter[i],targets[i]))



# cost_momentum = []
# nn.init()
# for iteration in range(30000):
#     out_iter_momentum, cost_iter_momentum = nn.train_momentum(inputs, targets)
#     cost_momentum.append(cost_iter_momentum)
# print("final loss with momentum:%.3f" % cost[len(cost_momentum)-1])

import matplotlib.pyplot as plt
plt.plot(cost)
# plt.plot(cost_momentum)
plt.show()