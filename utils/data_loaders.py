# utils/data_loaders.py
import numpy as np
def load_full_field(path):
    return np.load(path)

def load_sensor_map(path):
    return np.load(path)

def load_sensors(path):
    return np.load(path)
