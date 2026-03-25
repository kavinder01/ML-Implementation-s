import numpy as  np
import linear_regression.linear_regression as lr

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)

def batch_gradient_descent(X, y, w=None, b=0.0, alpha=0.01, iterations=10000):
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        dw , db = lr.compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = lr.compute_cost(X, y, w, b)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return w, b


def stochastic_gradient_descent(X, y, w=None, b=0.0, alpha=0.01, iterations=10000):
    m = X.shape[0]
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        new_m = np.random.randint(0, m)
        y_one = y[new_m:new_m+1]
        X_one = X[new_m:new_m+1]
        dw , db = lr.compute_gradient(X_one, y_one, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = lr.compute_cost(X_one, y_one, w, b)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return w, b



def mini_batch_gradient_descent(X, y, w=None, b=0.0, alpha=0.01, iterations=100000, batch_size=2):
    m = X.shape[0]
    start = 0
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        if start >= m:  # check FIRST
            start = 0  # reset FIRST
        end = min(start + batch_size, m)  # min prevents going past m
        X_batch = X[start:end]
        y_batch = y[start:end]
        dw , db = lr.compute_gradient(X_batch, y_batch, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = lr.compute_cost(X_batch, y_batch, w, b)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
        start += batch_size

    return w, b

print("=" * 40)
print("BATCH GRADIENT DESCENT")
print("=" * 40)
w_batch, b_batch = batch_gradient_descent(X, y)
print(f"Final w: {w_batch}")
print(f"Final b: {b_batch:.4f}")
print(f"Prediction for X=6: {lr.predict(np.array([[6]]), w_batch, b_batch)}")

print("=" * 40)
print("STOCHASTIC GRADIENT DESCENT")
print("=" * 40)
w_sgd, b_sgd = stochastic_gradient_descent(X, y)
print(f"Final w: {w_sgd}")
print(f"Final b: {b_sgd:.4f}")
print(f"Prediction for X=6: {lr.predict(np.array([[6]]), w_sgd, b_sgd)}")

print("=" * 40)
print("MINI BATCH GRADIENT DESCENT")
print("=" * 40)
w_mini, b_mini = mini_batch_gradient_descent(X, y)
print(f"Final w: {w_mini}")
print(f"Final b: {b_mini:.4f}")
print(f"Prediction for X=6: {lr.predict(np.array([[6]]), w_mini, b_mini)}")




