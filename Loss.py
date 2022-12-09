import numpy as np

def lms(y_true, y_pred):
	return (np.power(y_true - y_pred, 2) / 2).item()

def lms_prime(y_true, y_pred):
    return (y_true - y_pred)

def mse(y_true, y_pred):
	return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)