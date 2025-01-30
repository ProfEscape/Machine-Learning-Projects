import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to perform linear regression using Pseudoinverse
def linear_regression(X, y):
    # Add a column of ones to X for the intercept term (bias)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Compute the regression coefficients (W) using Pseudoinverse
    # W = (X^T X)^(-1) X^T y is replaced with θ = pinv(X) @ y
    W = np.linalg.pinv(X) @ y
    return W

# Function to predict target values
def predict(X, W):
    # Add a column of ones to X for the intercept term (bias)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X @ W

# Generate or load a dataset (customize here for any number of features)
data = {
    'Feature1': [9, 1, 13, 4, 2],
    'Feature2': [2, 4, 3, 8, 2],  # Linearly dependent on Feature1
    'Feature3': [1, 3, 5, 7, 9],
    'Target': [5, 7, 9, 11, 13]
}
df = pd.DataFrame(data)

# Extract features (X) and target (y)
X = df.iloc[:, :-1].values  # All columns except the last are features
y = df.iloc[:, -1].values.reshape(-1, 1)  # The last column is the target

# Perform linear regression
W = linear_regression(X, y)

# Display the computed coefficients
print("Computed coefficients (W):")
print(W)

# Predict the target values
y_pred = predict(X, W)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Visualize the actual vs. predicted values
plt.scatter(range(len(y)), y, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()


