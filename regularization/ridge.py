import numpy as np
import linear_regression.linear_regression as lr

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)
lambda_ = 0.1

def compute_cost_ridge(X, y, w, b, lambda_):
    m = X.shape[0]
    j = lr.compute_cost(X, y, w, b)
    J_ridge = j + (lambda_/(2*m))*np.sum(w**2)
    return J_ridge

def compute_gradient_ridge(X, y, w, b, lambda_):
    m = X.shape[0]
    dw, db = lr.compute_gradient(X, y, w, b)
    dw_ridge = dw + (lambda_/m)*w
    db_ridge = db
    return dw_ridge, db_ridge

def gradient_descent_ridge(X, y, w=None, b=0.0,alpha=0.01,iterations=1000000,lambda_=0.1):
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        dw , db = compute_gradient_ridge(X, y, w, b, lambda_)
        w = w - alpha*dw
        b = b - alpha*db
        if i % 1000 == 0:
            cost = compute_cost_ridge(X, y, w, b, lambda_)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return w, b

w = np.zeros(1)
b = 0.0

w_r, b_r = gradient_descent_ridge(X, y, lambda_=0.1)
print(f"Ridge → Final w: {w_r}, b: {b_r:.4f}")


