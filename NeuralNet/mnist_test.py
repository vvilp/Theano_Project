import nn2

import os
import struct
import numpy as np
import random
def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    return img, lbl 

def make_data(input, label):
    targets = np.zeros((len(label), 10))
    for i in range(0, len(label)):
        targets[i][label[i]] = 1
    train_data = list(zip(input, targets))
    return train_data, targets

def eval(output, label):
    correct_n = 0
    for i in range(0, len(output[0])):
        indice = np.argmax(output[0][i])
        if indice == label[i]:
            correct_n += 1
    return correct_n, correct_n / len(label)

origin_image, origin_label = read("training", "MNIST_data")
train_data, origin_targets = make_data(origin_image, origin_label)

test_image, test_label = read("testing", "MNIST_data")
test_data, test_targets = make_data(origin_image, origin_label)


epoch_num = 50
mini_batch_size = 50
import nn3 as nn
epoch_costs = []
epoch_accuracy_train = []
epoch_accuracy_test = []
net = nn.NeuralNet(len(origin_image[0]), 50, len(origin_targets[0]), 0.001)
for epoch_i in range(epoch_num):
    random.shuffle(train_data)
    mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]
    
    epoch_cost = 0
    for each_batch in mini_batches:
        train_data_input = [row[0] for row in each_batch]
        train_data_targets = [row[1] for row in each_batch]
        out_iter, cost_iter = net.train_iter(train_data_input, train_data_targets)
        mean_batch_cost = cost_iter / len(train_data_input)
        epoch_cost = epoch_cost + mean_batch_cost
    
    epoch_costs.append(epoch_cost/len(mini_batches))

    print("epoch:%d, mean_cost:%.2f" % (epoch_i, epoch_cost/len(mini_batches)))

    train_data_output = net.predict_(origin_image)
    correct_train_output,accuracy_train = eval(train_data_output, origin_label)
    test_data_output = net.predict_(test_image)
    correct_test_output,accuracy_test = eval(test_data_output, test_label)

    epoch_accuracy_train.append(accuracy_train)
    epoch_accuracy_test.append(accuracy_test)
    print ("Training Data | Correct:%d, Accurary:%.2f" % (correct_train_output, accuracy_train))
    print ("Testing Data | Correct:%d, Accurary:%.2f" % (correct_test_output, accuracy_test))


import matplotlib.pyplot as plt

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(epoch_costs)
ax2 = f2.add_subplot(111)
ax2.plot(epoch_accuracy_train)
ax2.plot(epoch_accuracy_test)
plt.show()

# plt.plot(epoch_costs)
# plt.show()

# import nn3 as nn
# net = nn.NeuralNet(train_data, targets, 10, 0.0001)
# cost = []
# for iteration in range(100):
#     out_iter, cost_iter = net.train_iter(train_data, targets)
#     mean_cost = cost_iter / len(train_label)
#     if mean_cost < 1.0:
#         cost.append(mean_cost)
#         print("{0},{1}".format(iteration, mean_cost))
#     if mean_cost < 0.25:
#         break





# for i in range(len(targets)):
#     print (["out: {0:0.2f}".format(j) for j in out_iter[i]])
#     print (["tar: {0:0.2f}".format(j) for j in targets[i]])

# import matplotlib.pyplot as plt
# plt.plot(cost)
# plt.show()