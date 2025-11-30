# utils/metrics.py
import numpy as np
def rmse(a, b):
    return np.sqrt(((a - b)**2).mean())

def mse(a, b):
    return ((a - b)**2).mean()
