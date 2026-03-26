import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import linear_regression.linear_regression as lr
from cost_functions.mse import mse, rmse, mae

# Step 1 - load
data = fetch_california_housing()
X = data.data[:, 0:1]
y = data.target

# Step 2 - scale and split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3 - scratch model
w_s, b_s = lr.gradient_descent_loop(X_train, y_train, iterations=10000, alpha=0.01)
y_pred_scratch = lr.predict(X_test, w_s, b_s)

# Step 4 - sklearn model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)

# Step 5 - print results
print(f"Scratch  → w: {w_s[0]:.4f}, b: {b_s:.4f}")
print(f"sklearn  → w: {model.coef_[0]:.4f}, b: {model.intercept_:.4f}")

print(f"\nScratch Metrics:")
print(f"  MSE  : {mse(y_test, y_pred_scratch):.4f}")
print(f"  RMSE : {rmse(y_test, y_pred_scratch):.4f}")
print(f"  MAE  : {mae(y_test, y_pred_scratch):.4f}")

print(f"\nSklearn Metrics:")
print(f"  MSE  : {mse(y_test, y_pred_sklearn):.4f}")
print(f"  RMSE : {rmse(y_test, y_pred_sklearn):.4f}")
print(f"  MAE  : {mae(y_test, y_pred_sklearn):.4f}")