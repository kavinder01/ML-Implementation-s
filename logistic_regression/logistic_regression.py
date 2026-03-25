import numpy as np
import cost_functions.binary_cross_entropy as cf

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([0, 0, 0, 1, 1],     dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def compute_cost(X, y, w, b):
    m = X.shape[0]
    return cf.bce(y, predict(X, w, b))

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    dw = (1/m)*np.dot(X.T, y_hat - y)
    db = (1/m)*np.sum(y_hat - y)
    return dw, db

def gradient_descent(X, y, w=None, b=0.0, alpha=0.01, iterations=10000):
    if w is None:
        w = np.zeros(1)
    for i in range(iterations):
        dw, db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")
    return  w, b

def predict_class(X, w, b, threshold=0.5):
    '''    probs = predict(X, w, b)
    final = []
    for i in range(len(probs)):
        if probs[i] >= threshold:
            final.append(1)
        else:
            final.append(0)
    return final'''
    probs = predict(X, w, b)
    return (probs >= threshold).astype(int)


w, b = gradient_descent(X, y)
print(f"\nFinal w: {w}")
print(f"Final b: {b:.4f}")

probs = predict(X, w, b)
print(f"\nProbabilities: {np.round(probs, 4)}")

classes = predict_class(X, w, b)
print(f"Predicted classes: {classes}")
print(f"Actual classes:    {y.astype(int)}")

print(f"\nCost: {compute_cost(X, y, w, b):.6f}")
