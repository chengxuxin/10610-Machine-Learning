import numpy as np
import sys, os, csv

def read(filename):
    data = np.loadtxt(filename, delimiter="\t")
    num_data = data.shape[0]
    # fold bias into X
    data = np.hstack((data, np.ones((num_data, 1))))
    # print(data.shape, data[1])
    return data

def readlabels(filename):
    data = np.loadtxt(filename, delimiter="\n")
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcJ(X, Y, theta):
    probs = sigmoid(np.dot(X,theta))
    Js = Y * np.log(probs) + (1-Y) * np.log(1-probs)
    J = -np.mean(Js)
    return J

def SGD(X, Y, theta, N):
    temp = -Y + sigmoid(np.dot(X, theta))
    # temp = np.expand_dims(temp, axis=1)
    dJ = temp * X / N
    # dJ = np.mean(dJs, axis=0)
    return dJ

def predict(x, theta):
    # the bias should have been folded into x
    # x = np.hstack((x, np.ones((x.shape[0], 1))))
    prob1 = sigmoid(np.dot(x, theta))
    predicted_labels = np.where(prob1>=0.5, 1, 0)
    return predicted_labels

def train():
    alpha = 0.01
    data_train = read(formatted_train_in)
    Y, X = data_train[:, 0], data_train[:, 1:]
    num_data = X.shape[0]
    # X = np.hstack((X, np.ones((num_data, 1))))
    theta = np.zeros(X.shape[1])
    for ep in range(num_epoch):
        # print(ep, calcJ(X, Y, theta))
        for y, x in zip(Y, X):
            dJ = SGD(x, y, theta, num_data)
            theta = theta - alpha * dJ
        
    train_pred_labels = predict(X, theta)
    return theta, train_pred_labels

if __name__ == "__main__":
    formatted_train_in = sys.argv[1]
    formatted_val_in = sys.argv[2]
    formatted_test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    # formatted_train_in = "./formatted_train_out.tsv"
    # formatted_val_in = "./formatted_val_out.tsv"
    # formatted_test_in = "./formatted_test_out.tsv"
    # dict_in = "./handout/dict.txt"
    # train_out = "train_out.labels"
    # test_out = "test_out.labels"
    # metrics_out = "metrics.txt"
    # num_epoch = 500

    # training
    data_train = read(formatted_train_in)
    true_train_labels = data_train[:, 0]
    theta, train_pred_labels = train()
    np.savetxt(train_out, train_pred_labels, fmt="%d", delimiter="\n")
    error_train = np.mean(true_train_labels != train_pred_labels)
    # sanity check
    # ref_train_labels = readlabels("./handout/smalloutput/model2_train_out.labels")
    # print(np.sum(np.square(ref_train_labels - train_pred_labels)))
    
    # testing
    data_test = read(formatted_test_in)
    true_test_labels = data_test[:, 0]
    X = data_test[:, 1:]
    test_pred_labels = predict(X, theta)
    np.savetxt(test_out, test_pred_labels, fmt="%d", delimiter="\n")
    error_test = np.mean(true_test_labels != test_pred_labels)
    # ref_test_labels = readlabels("./handout/smalloutput/model1_test_out.labels")
    # print(np.sum(np.square(ref_test_labels - test_pred_labels)))

    with open(metrics_out, 'w') as f:
        f.write("error(train): {:.6f}\nerror(test): {:.6f}".format(error_train, error_test))