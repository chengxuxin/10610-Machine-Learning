import sys, os
# path = os.path.abspath(os.path.join('..'))
# sys.path.append(path)
from feature import *
from lr import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

train_in = "./handout/largedata/train_data.tsv"
val_in = "./handout/largedata/valid_data.tsv"
test_in = "./handout/largedata/test_data.tsv"
formatted_train_out = "formatted_train_out.tsv"
formatted_val_out = "formatted_val_out.tsv"
formatted_test_out = "formatted_test_out.tsv"
dict_in = "./handout/dict.txt"
# extract_features1(train_in, formatted_train_out, dict_in)
# extract_features1(val_in, formatted_val_out, dict_in)
# extract_features1(test_in, formatted_test_out, dict_in)

formatted_train_in = "./formatted_train_out.tsv"
formatted_val_in = "./formatted_val_out.tsv"
formatted_test_in = "./formatted_test_out.tsv"
num_epoch = 5000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sparse_dot(x1, x2):
    x1 = np.squeeze(x1)
    x2 = np.squeeze(x2)
    prod = 0
    for x1i, x2i in zip(x1, x2):
        if x1i == 0 or x2i == 0:
            continue
        prod += x1i * x2i
    return prod

def SGD(X, Y, theta, N):
    temp = -Y + sigmoid(sparse_dot(X, theta))
    # temp = np.expand_dims(temp, axis=1)
    dJ = temp * X / N
    # dJ = np.mean(dJs, axis=0)
    return dJ

def train_plot(alpha):
    data_train = read(formatted_train_in)
    Y, X = data_train[:, 0], data_train[:, 1:]
    # X = csr_matrix(X)
    num_data = X.shape[0]
    theta = np.zeros(X.shape[1])

    data_val = read(formatted_val_in)
    Y_val, X_val = data_val[:, 0], data_val[:, 1:]

    Js_train = []
    Js_val =[]
    for ep in tqdm(range(num_epoch)):
        # print(ep, calcJ(X, Y, theta))
        for i in range(num_data):
            dJ = SGD(X[i], Y[i], theta, num_data)
            theta = np.add(theta, - np.dot(alpha, dJ))
        # for y, x in zip(Y, X):
        #     dJ = SGD(x, y, theta, num_data)
        #     theta = theta - alpha * dJ
        Js_train.append(calcJ(X, Y, theta))
        Js_val.append(calcJ(X_val, Y_val, theta))
    train_pred_labels = predict(X, theta)
    return theta, train_pred_labels, Js_train, Js_val


alpha = 0.01
theta1_001, train_pred_labels1_001, Js_train1_001, Js_val1_001 = train_plot(alpha)