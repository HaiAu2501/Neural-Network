import numpy as np
from matplotlib import pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [-1] * 20 + [1] * 20

# Training the SVM (using simple stochastic gradient descent)
w = np.random.randn(2)
b = 0
lr = 0.1  # Learning rate
epochs = 1000  # Number of epochs

for epoch in range(epochs):
    for i, x in enumerate(X):
        if (Y[i] * (np.dot(X[i], w) + b)) < 1:
            w = w + lr * ((X[i] * Y[i]) + (-2 * (1/epoch) * w))
            b = b + lr * Y[i]
        else:
            w = w + lr * (-2 * (1/epoch) * w)

# Visualization of the decision boundary
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')

ax.plot([-2,3], [-(w[0]*(-2) + b)/w[1], -(w[0]*3 + b)/w[1]], 'k-')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVM')
plt.show()
