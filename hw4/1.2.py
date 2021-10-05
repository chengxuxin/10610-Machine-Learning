import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcJ(X, Y, theta):
    probs = sigmoid(np.dot(X,theta))
    Js = Y * np.log(probs) + (1-Y) * np.log(1-probs)
    J = -np.mean(Js)
    return J

def calc_dJ(X, Y, theta):
    temp = -Y + sigmoid(np.dot(X, theta))
    temp = np.expand_dims(temp, axis=1)
    
    dJs = np.multiply(X, temp)
    dJ = np.mean(dJs, axis=0)
    return dJ

X = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
Y = np.array([0, 1, 1, 0])
theta0 = np.array([1.5, 2, 1, 2, 3])

J = calcJ(X, Y, theta0)
dJ = calc_dJ(X, Y, theta0)
with np.printoptions(precision=4, suppress=True):
    print("J={:.4f}".format(J))
    print("dJ={}".format(dJ))
    print("new theta={}".format(theta0 - dJ))
