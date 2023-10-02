import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(Y, Y_hat):
    m = Y.shape[1]
    return -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def logistic_regression_nn(X, Y, learning_rate, num_iterations):
    m, n = X.shape

    # Khởi tạo trọng số và bias
    W = np.random.randn(1, m) * 0.01
    b = 0

    for i in range(num_iterations):
        # Lan truyền tiến
        Z = np.dot(W, X) + b
        A = sigmoid(Z)

        # Tính toán chi phí
        cost = compute_cost(Y, A)

        # Lan truyền ngược
        dZ = A - Y
        dW = 1/n * np.dot(dZ, X.T)
        db = 1/n * np.sum(dZ)

        # Cập nhật trọng số và bias
        W -= learning_rate * dW
        b -= learning_rate * db

        if i % 1000 == 0:
            print(f"Iteration {i}, Cost: {cost}")

    return W, b

# Ví dụ về việc sử dụng mô hình:
X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])  # Dữ liệu đầu vào
Y = np.array([[0, 1, 0, 1]])  # Nhãn

W, b = logistic_regression_nn(X, Y, learning_rate=0.1, num_iterations=10000)

# Dự đoán
Z = np.dot(W, X) + b
predictions = sigmoid(Z) > 0.5
print(predictions)
