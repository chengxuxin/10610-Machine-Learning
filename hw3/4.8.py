import numpy as np

x = np.array([[9, 1], [2, 1], [6, 1], [1, 1], [8, 1]])
y = np.array([[1, 0, 3, 0, 1]]).T
theta = np.array([[3, 0]]).T

alpha = 0.001
gradient = 0.4 * x.T @ (x @ theta - y)
theta_optimal = np.linalg.inv(x.T @ x) @ x.T @ y

print(gradient)
print(theta_optimal)

for i in range(5):
    theta = theta - alpha * gradient
    gradient = 0.4 * x.T @ (x @ theta - y)

print(theta)