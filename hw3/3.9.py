import numpy as np
# train a perceptron

data = np.array([[-1.939,	2.704, 1],
[-0.928,	-3.054, 1],
[-2.181,	-3.353, 1],
[-0.142,	1.44, 1],
[2.605,	-0.651, 1]])

labels = [1, -1, -1, 1, 1]
weight = np.zeros(3)

while True:
    all_correct = True
    for i, pt in enumerate(data):
        predict = np.dot(data[i], weight)
        if labels[i] == 1 and predict < 0:
            weight = weight + data[i]
            all_correct = False
            print('make mistake')
        if labels[i] == -1 and predict >= 0:
            weight = weight - data[i]
            all_correct = False
            print('make mistake')
    if all_correct:
        break
    
print(weight)
print(np.dot(data, weight))
print(np.dot(data, np.array([1.594, 5.107, 1])))
