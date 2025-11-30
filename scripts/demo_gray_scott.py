# scripts/demo_gray_scott.py
"""
Damped Gray-Scott PDE generator + tiny experiment.
Implements a simple finite-difference Gray-Scott model on a 2D grid.
"""
import numpy as np
from scipy.signal import convolve2d
from models.sindy_utils import build_library, stlsq
import torch
from models.shred import SHRED
from models.dashred import DAShred

def simulate_gray_scott(nx=64, ny=64, nt=200, Du=0.16, Dv=0.08, F=0.035, k=0.060, damp=0.001, dt=1.0):
    # 2D Gray-Scott using simple Laplacian kernel
    lap = np.array([[0.05, 0.2, 0.05],
                    [0.2, -1.0, 0.2],
                    [0.05, 0.2, 0.05]])
    U = np.ones((nx, ny))
    V = np.zeros((nx, ny))
    # seed a spot
    r = 6
    cx, cy = nx//2, ny//2
    U[cx-r:cx+r, cy-r:cy+r] = 0.50
    V[cx-r:cx+r, cy-r:cy+r] = 0.25
    data = []
    for t in range(nt):
        Lu = convolve2d(U, lap, mode='same', boundary='wrap')
        Lv = convolve2d(V, lap, mode='same', boundary='wrap')
        uvv = U * (V**2)
        U = U + (Du * Lu - uvv + F * (1 - U) - damp*U) * (dt)
        V = V + (Dv * Lv + uvv - (F + k) * V - damp*V) * (dt)
        if t % 1 == 0:
            data.append(np.stack([U, V], axis=0))  # (2, nx, ny)
    data = np.stack(data, axis=0)  # (nt, 2, nx, ny)
    return data

def run_demo():
    data = simulate_gray_scott(nx=64, ny=64, nt=200)
    # flatten each field to vector
    nt = data.shape[0]
    fields = data.reshape(nt, -1)
    # sensor map: choose p random sensors
    p = 16
    rng = np.random.default_rng(1)
    idx = rng.choice(fields.shape[1], size=p, replace=False)
    sensor_map = np.zeros((p, fields.shape[1]))
    for i, ii in enumerate(idx):
        sensor_map[i, ii] = 1.0
    np.save('data/gray_scott_sample.npy', fields)
    np.save('data/gray_scott_sensor_map.npy', sensor_map)
    # quick train SHRED
    import subprocess
    args = ['python', 'scripts/train_shred_with_sindy.py', '--sim', 'data/gray_scott_sample.npy', '--sensor_map', 'data/gray_scott_sensor_map.npy', '--window', '5', '--epochs', '8', '--latent_dim', '32', '--batch', '16', '--ckpt', 'gray_shred.pth']
    subprocess.run(args)
    model = SHRED(sensor_dim=p, latent_dim=32, enc_hidden=128, dec_out_dim=fields.shape[1])
    model.load_state_dict(torch.load('gray_shred.pth', map_location='cpu'))
    sensors = fields.dot(sensor_map.T)
    das = DAShred(model, sensor_map, window=5)
    rec = das.assimilate_sequence(sensors, steps=15, lr=1e-2)
    print('Recovered shape', rec.shape)

if __name__ == '__main__':
    run_demo()
