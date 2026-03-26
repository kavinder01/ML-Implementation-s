import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sys
sys.path.append('..')
import linear_regression as lr

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)

# YOUR SCRATCH MODEL
w, b = lr.gradient_descent_loop(X, y)
print(f"Scratch  → w: {w[0]:.4f}, b: {b:.4f}")

# SKLEARN MODEL
model = LinearRegression()
model.fit(X, y)
print(f"sklearn  → w: {model.coef_[0]:.4f}, b: {model.intercept_:.4f}")

# SKLEARN RIDGE
ridge = Ridge(alpha=0.1)
ridge.fit(X, y)
print(f"Ridge SK → w: {ridge.coef_[0]:.4f}, b: {ridge.intercept_:.4f}")

# SKLEARN LASSO
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print(f"Lasso SK → w: {lasso.coef_[0]:.4f}, b: {lasso.intercept_:.4f}")