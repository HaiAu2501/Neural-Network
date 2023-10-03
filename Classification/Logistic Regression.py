import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Định nghĩa hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Định nghĩa hàm chi phí (hay hàm mất mát) dựa trên cross - entropy
def compute_cost(Y, Y_hat):
    m = Y.shape[1]
    return -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def logistic_regression_nn(X, Y, learning_rate, num_iterations):
    m, n = X.shape
    W = np.random.randn(1, m) * 0.01
    b = 0
    costs = []
    accuracies = []

    for i in range(num_iterations):
        Z = np.dot(W, X) + b
        A = sigmoid(Z)
        cost = compute_cost(Y, A)
        costs.append(cost)

        dZ = A - Y
        dW = 1/n * np.dot(dZ, X.T)
        db = 1/n * np.sum(dZ)

        # Cập nhật ma trận trọng số theo Stochastic Gradient Descent
        W -= learning_rate * dW
        b -= learning_rate * db

        accuracy = np.mean((A > 0.5) == Y)
        accuracies.append(accuracy)

        if i % 10 == 0:
            print(f"Iteration {i}, Cost: {cost}, Accuracy: {accuracy}")

    return W, b, costs, accuracies

# Đọc dữ liệu
data = pd.read_csv("C:\\Users\\admin\\Downloads\\diabetes.csv")
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Chia tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa và tiền xử lý dữ liệu, phép toán .T để lấy chuyển vị
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_T = X_train_scaled.T
y_train_T = y_train.reshape(1, y_train.shape[0])
X_test_T = X_test_scaled.T
y_test_T = y_test.reshape(1, y_test.shape[0])

# Huấn luyện mô hình với tốc độ học và số vòng lặp
W, b, costs, accuracies = logistic_regression_nn(X_train_T, y_train_T, learning_rate=0.5, num_iterations=500)

# Vẽ đồ thị chi phí và độ chính xác
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Cost over iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy over iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
