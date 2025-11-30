# scripts/demo_burgers.py
"""
Tiny end-to-end demo using 1D viscous Burgers equation to generate toy data,
train SHRED, and run DA + SINDy discovery in latent space.
This is minimal and intended to run quickly on CPU for demonstration.
"""
import numpy as np
from scipy.integrate import solve_ivp
import torch
from models.shred import SHRED
from scripts.train_shred_with_sindy import finite_diff
from models.sindy_utils import build_library, stlsq

def simulate_burgers(nx=128, nt=200, nu=0.01, dt=0.01):
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    u = np.sin(x)  # initial
    dx = x[1]-x[0]
    data = []
    for t in range(nt):
        # simple semi-implicit step (viscous term treated explicitly here for speed)
        uxx = np.roll(u, -1) - 2*u + np.roll(u, 1)
        u = u - dt * u * (np.roll(u, -1) - np.roll(u, 1)) / (2*dx) + nu * dt * uxx / (dx**2)
        data.append(u.copy())
    return np.stack(data)  # (nt, nx)

def run_demo():
    data = simulate_burgers(nx=128, nt=300, nu=0.01, dt=0.001)
    nt, nx = data.shape
    # sensor map: choose p sensors random
    p = 8
    rng = np.random.default_rng(0)
    sensor_idx = rng.choice(nx, size=p, replace=False)
    sensor_map = np.zeros((p, nx))
    for i, s in enumerate(sensor_idx):
        sensor_map[i, s] = 1.0
    np.save('data/burgers_sim.npy', data)
    np.save('data/burgers_sensor_map.npy', sensor_map)
    # Train quick SHRED (small epochs)
    import subprocess, sys
    args = ['python', 'scripts/train_shred_with_sindy.py', '--sim', 'data/burgers_sim.npy', '--sensor_map', 'data/burgers_sensor_map.npy', '--window', '5', '--epochs', '5', '--latent_dim', '16', '--batch', '32', '--sindy_reg', '0.0', '--ckpt', 'burgers_shred.pth']
    print('Running training (may take a minute)...')
    subprocess.run(args)
    # Run DA assimilation
    model = SHRED(sensor_dim=p, latent_dim=16, enc_hidden=64, dec_out_dim=nx)
    model.load_state_dict(torch.load('burgers_shred.pth', map_location='cpu'))
    sensors = data.dot(sensor_map.T)
    from models.dashred import DAShred
    das = DAShred(model, sensor_map, window=5)
    fields = das.assimilate_sequence(sensors, steps=20, lr=1e-2)
    print('Assimilated fields shape', fields.shape)
    # Run SINDy discovery in latent space
    # Build windows of sensors from assimilated fields
    sensors_rec = fields.dot(sensor_map.T)
    H = []
    for t in range(5, sensors_rec.shape[0]):
        H.append(sensors_rec[t-5:t])
    H = np.stack(H)
    with torch.no_grad():
        xb = torch.tensor(H, dtype=torch.float32)
        _, Z = model(xb)
    Z = Z.numpy()
    dZ = finite_diff(Z, dt=1.0)
    Z_t = Z[:-1]
    Theta, names = build_library(Z_t, poly_order=2)
    Xi = stlsq(Theta, dZ, lam=0.1, max_iter=10)
    print('SINDy Xi shape', Xi.shape)

if __name__ == '__main__':
    run_demo()
