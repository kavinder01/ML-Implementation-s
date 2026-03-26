import numpy as np
import linear_regression.linear_regression as lr

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)
lambda_ = 0.1


'''Lasso cost:
J_lasso = J + (lambda_/m) × Σ|w|

Lasso gradient:
dw_lasso = dw + (lambda_/m) × np.sign(w)
db stays the same'''

def compute_cost_lasso(X, y, w, b, lambda_):
    m = X.shape[0]
    j = lr.compute_cost(X, y, w, b)
    J_ridge = j + (lambda_/(2*m))*np.sum(np.abs(w**2))
    return J_ridge

def compute_gradient_lasso(X, y, w, b, lambda_):
    m = X.shape[0]
    dw, db = lr.compute_gradient(X, y, w, b)
    dw_ridge = dw + (lambda_/m)*w
    db_ridge = db
    return dw_ridge, db_ridge

def gradient_descent_lasso(X, y, w=None, b=0.0,alpha=0.01,iterations=10000,lambda_=0.1):
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        dw , db = compute_gradient_lasso(X, y, w, b, lambda_)
        w = w - alpha*dw
        b = b - alpha*db
        if i % 1000 == 0:
            cost = compute_cost_lasso(X, y, w, b, lambda_)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return w, b

w_l, b_l = gradient_descent_lasso(X, y, lambda_=0.1)
print(f"Lasso → Final w: {w_l}, b: {b_l:.4f}")