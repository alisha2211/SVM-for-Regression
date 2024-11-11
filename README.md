# SVM-for-Regression

from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Create a small dataset with a simple trend
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 3, 5, 6, 7, 8, 9, 10, 12, 13])

# Initialize SVR model with an RBF kernel
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X, y)

# Predict on the input range and plot
X_test = np.linspace(1, 10, 100).reshape(-1, 1)
y_pred = svr.predict(X_test)

# Plotting
plt.scatter(X, y, color='red', label="Data points")
plt.plot(X_test, y_pred, color='blue', label="SVR model")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("SVM Regression")
plt.legend()
plt.show()
