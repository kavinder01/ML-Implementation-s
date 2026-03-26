import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)



def predict(X, w, b):
    y_hat = np.dot(X, w) + b
    return y_hat

def compute_cost(X, y, w, b):
    m = X.shape[0]
    j = (1/(2*m))*np.sum((y - predict(X, w, b))**2)
    return j

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    dw = (1/m)*np.dot(X.T, y_hat - y)
    db = (1/m)*np.sum(y_hat - y)

    return dw , db



def gradient_descent_loop ( X, y, w=None, b=0.0, alpha=0.01, iterations=10000):
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        dw , db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return w, b
w, b = gradient_descent_loop(X, y)
print(f"\nFinal w: {w}")
print(f"Final b: {b}")
print(f"Prediction for X=6: {predict(np.array([[6]]), w, b)}")