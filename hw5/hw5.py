import numpy as np
import array_to_latex as a2l
np.set_printoptions(precision=4)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x)
    s = np.sum(exp)
    return exp / s

def celoss(y, y_hat):
    return - np.dot(y, np.log(y_hat))

alpha_star = np.array([[1, 2, -3, 0, 1, -3], [3, 1, 2, 1, 0, 2], [2, 2, 2, 2, 2, 1], [1, 0, 2, 1, -2, 2]])
beta_star = np.array([[1, 2, -2, 1], [1, -1, 1, 2], [3, 1, -1, 1]])
alpha = np.hstack((np.ones((4, 1)), alpha_star))
beta = np.hstack((np.ones((3, 1)), beta_star))

x = np.array([1, 1, 0, 0, 1, 1])
x1 = np.array([1, 1, 1, 0, 0, 1, 1])
y = np.array([0, 1, 0])

# forward propagation
a = np.dot(alpha, x1)
print(a)

z = sigmoid(a)
print("z:{}".format(z))

z = np.hstack((1, z))
b = np.dot(beta, z)
print("b:{}".format(b))

y_hat = softmax(b)
print("y_hat:{}".format(y_hat))

loss = celoss(y, y_hat)
print("loss:{:.4f}".format(loss))

# backward propagation
dl_db = y_hat - y
dl_dbeta = dl_db[:, None] @ z[None, :]

dl_dz = dl_db @ beta_star  # no z0=1
dl_da = np.multiply(dl_dz, np.multiply(z[1:], 1-z[1:]))
dl_dalpha = dl_da[:, None] @ x1[None, :]
print("dldb", dl_db)
print("dldalpha", dl_dalpha)