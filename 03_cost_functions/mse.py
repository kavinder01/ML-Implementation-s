import numpy as np

y     = np.array([1.5, 3.0, 4.5, 6.0, 7.5])
y_hat = np.array([1.4, 3.2, 4.4, 6.3, 7.4])

def mse(y, y_hat):
    '''(1/m) * Σ(y_hat - y)²'''
    result_1 = np.mean((y - y_hat)**2)
    return result_1


def rmse(y, y_hat):
    '''sqrt(mse)'''
    result_2 = np.sqrt(mse(y, y_hat))
    return result_2


def mae(y, y_hat):
    '''(1/m) * Σ|y_hat - y|'''
    result_3 = np.mean(np.abs(y - y_hat))
    return result_3

print(f'mse: {mse(y, y_hat):.4f}',
      f'rmse: {rmse(y, y_hat):.4f}',
      f'mae: {mae(y, y_hat):.4f}',
      sep='\n')