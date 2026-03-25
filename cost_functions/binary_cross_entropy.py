import numpy as np

y = np.array([1, 0, 1, 1, 0])          # true labels
y_hat = np.array([0.9, 0.2, 0.8, 0.7, 0.3]) # predicted probabilities
y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

def bce(y, y_hat):
    m = y.shape[0]
    j = -(1/m)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
    return j

print(f'BCE : {bce(y, y_hat):.4f}')