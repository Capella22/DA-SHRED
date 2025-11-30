# models/dashred.py
import numpy as np
import torch
from torch import nn

class DAShred:
    """
    Latent-space assimilation wrapper around SHRED.
    """
    def __init__(self, shred_model, sensor_map, window, device=None):
        self.model = shred_model
        self.sensor_map = sensor_map  # (p, n_full) numpy
        self.window = window
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.mse = nn.MSELoss()

    def assimilate_sequence(self, sensors_seq, steps=10, lr=1e-2):
        H = []
        for t in range(self.window, len(sensors_seq)):
            H.append(sensors_seq[t-self.window:t])
        H = np.stack(H)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(H, dtype=torch.float32).to(self.device)
            _, z = self.model(xb)
        z_var = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([z_var], lr=lr)
        sensor_targets = []
        for t in range(self.window, len(sensors_seq)):
            sensor_targets.append(sensors_seq[t])
        sensor_targets = torch.tensor(np.stack(sensor_targets), dtype=torch.float32, device=self.device)
        sm = torch.tensor(self.sensor_map, dtype=torch.float32, device=self.device)
        for _ in range(steps):
            opt.zero_grad()
            decoded = self.model.decode(z_var)
            pred_sensors = decoded.matmul(sm.T)
            loss = self.mse(pred_sensors, sensor_targets)
            loss.backward()
            opt.step()
        with torch.no_grad():
            fields = self.model.decode(z_var).cpu().numpy()
        return fields
