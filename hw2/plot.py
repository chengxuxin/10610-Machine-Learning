import matplotlib.pyplot as plt
import numpy as np
from decisionTree import *


train_in = "./handout/politicians_train.tsv"
test_in = "./handout/politicians_test.tsv"


max_depth = 0

train_data = read(train_in)
test_data = read(test_in)

num_attrs = train_data.shape[1] - 1

error_train_list = []
error_test_list = []
depth_list = np.arange(num_attrs+1)

for depth in depth_list:
    tree = DecisionTree(train_data, max_depth=depth)
    tree.train(tree.root)
    error_train = save(tree, train_data, None, if_write=False)
    error_test = save(tree, test_data, None, if_write=False)
    error_train_list.append(error_train)
    error_test_list.append(error_test)


plt.plot(depth_list, error_train_list)
plt.plot(depth_list, error_test_list)
plt.ylabel("Error Rate")
plt.xlabel("Maximum Depth")
plt.legend(["Train", "Test"])
plt.show()