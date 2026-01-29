"""
Beyond Nyquist: Time-Delayed Embeddings for Super-Nyquist Frequency Recovery

Hypothesis:
For a traveling wave sin(kx - ωt), each sensor sees temporal oscillation at frequency ω.
With multiple sensors at different positions, the PHASE DIFFERENCES encode spatial frequency k.
The LSTM seeing time histories from multiple sensors can recover spatial
frequencies beyond what static spatial sampling would allow.

Key insight: Dispersion relation ω = f(k) links temporal and spatial frequencies.
For simple traveling waves: ω = c·k, so knowing ω implies k.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy

np.random.seed(45)
torch.manual_seed(45)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# 1. Data Generation
# =============================================================================

def generate_traveling_wave_data(L=2*np.pi, N=128, T=10.0, dt=0.05,
                                  k1=2, k2=20,
                                  omega1=1, omega2=10,
                                  A1=1.0, A2=0.4,
                                  include_hf=True):
    """
    Generate traveling wave data with high spatial frequency component.

    For traveling waves: u(x,t) = A·sin(kx - ωt)
    At sensor position x_j: s_j(t) = A·sin(kx_j - ωt) = A·sin(ωt + φ_j)
    where φ_j = -kx_j encodes the spatial frequency through phase.
    """
    x = np.linspace(0, L, N, endpoint=False)
    t = np.arange(0, T, dt)
    n_steps = len(t)

    U = np.zeros((n_steps, N))

    for i, ti in enumerate(t):
        u_lf = A1 * np.sin(k1 * x - omega1 * ti)
        u_hf = A2 * np.sin(k2 * x - omega2 * ti) if include_hf else 0
        U[i] = u_lf + u_hf

    return U, x, t


def analyze_sensor_signals(U, sensor_indices, x, dt, k1, k2, omega1, omega2):
    """
    Analyze what each sensor sees in the time domain.

    For traveling wave sin(kx - ωt), sensor at x_j sees:
    - Temporal frequency: ω/(2π) Hz
    - Phase: φ_j = -k·x_j
    """
    print("\n" + "="*60)
    print("SENSOR SIGNAL ANALYSIS")
    print("="*60)

    n_sensors = len(sensor_indices)
    sensor_positions = x[sensor_indices]

    # Temporal FFT of sensor signals
    S = U[:, sensor_indices]
    temporal_fft = np.abs(np.fft.rfft(S, axis=0))
    temporal_freqs = np.fft.rfftfreq(len(U), dt)

    print(f"\nTemporal frequencies in sensor signals:")
    print(f"  Expected ω1/(2π) = {omega1/(2*np.pi):.4f} Hz")
    print(f"  Expected ω2/(2π) = {omega2/(2*np.pi):.4f} Hz")

    # Average across sensors
    avg_temporal_fft = temporal_fft.mean(axis=1)
    peak_indices = np.argsort(avg_temporal_fft)[-5:][::-1]

    print(f"\n  Detected temporal frequencies:")
    for idx in peak_indices:
        if avg_temporal_fft[idx] > 0.1 * avg_temporal_fft.max():
            print(f"    f={temporal_freqs[idx]:.4f} Hz, magnitude={avg_temporal_fft[idx]:.2f}")

    # Phase analysis
    print(f"\nPhase differences between sensors (should encode spatial k):")
    print(f"  For k={k1}: Δφ = k·Δx = {k1}·Δx")
    print(f"  For k={k2}: Δφ = k·Δx = {k2}·Δx")

    # Theoretical phase at each sensor
    phases_k1 = -k1 * sensor_positions
    phases_k2 = -k2 * sensor_positions

    print(f"\n  Sensor phases for k1={k1}:")
    for i in range(min(5, n_sensors)):
        print(f"    Sensor {i} at x={sensor_positions[i]:.3f}: φ={phases_k1[i]:.3f} rad")

    print(f"\n  Sensor phases for k2={k2}:")
    for i in range(min(5, n_sensors)):
        print(f"    Sensor {i} at x={sensor_positions[i]:.3f}: φ={phases_k2[i]:.3f} rad")

    return temporal_fft, temporal_freqs


def plot_data_diagnostic(U_sim, U_real, x, t, sensor_indices, k1, k2, num_sensors, save_path):
    """Diagnostic plots for the data."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    nyquist_k = num_sensors // 2

    # Spatial snapshots
    idx_t = len(t) // 2
    axes[0, 0].plot(x, U_sim[idx_t], 'b-', lw=2, label=f'Sim (k={k1})')
    axes[0, 0].plot(x, U_real[idx_t], 'r--', lw=1.5, label=f'Real (k={k1}+{k2})')
    axes[0, 0].scatter(x[sensor_indices], U_real[idx_t, sensor_indices],
                       c='green', s=30, zorder=5, label='Sensors')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].set_title(f'Spatial Snapshot (t={t[idx_t]:.2f})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Spatial FFT
    fft_sim = np.abs(np.fft.rfft(U_sim, axis=1)).mean(axis=0)
    fft_real = np.abs(np.fft.rfft(U_real, axis=1)).mean(axis=0)
    freqs = np.arange(len(fft_sim))

    axes[0, 1].stem(freqs, fft_sim, basefmt=' ', linefmt='b-', markerfmt='bo', label='Sim')
    axes[0, 1].stem(freqs + 0.3, fft_real, basefmt=' ', linefmt='r-', markerfmt='r^', label='Real')
    axes[0, 1].axvline(nyquist_k, color='green', linestyle='--', lw=2, label=f'Nyquist (k={nyquist_k})')
    axes[0, 1].axvline(k2, color='orange', linestyle=':', lw=2, label=f'k2={k2} (target)')
    axes[0, 1].set_xlabel('Spatial Frequency (k)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('Spatial FFT (Full Resolution)')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 30)
    axes[0, 1].grid(alpha=0.3)

    # What sensors see spatially (aliased!)
    S_real = U_real[:, sensor_indices]
    fft_sensors = np.abs(np.fft.rfft(S_real, axis=1)).mean(axis=0)
    sensor_freqs = np.arange(len(fft_sensors))

    axes[0, 2].stem(sensor_freqs, fft_sensors, basefmt=' ', linefmt='g-', markerfmt='go')
    axes[0, 2].axvline(nyquist_k, color='red', linestyle='--', lw=2, label=f'Nyquist (k={nyquist_k})')
    axes[0, 2].set_xlabel('Spatial Frequency (k)')
    axes[0, 2].set_ylabel('Magnitude')
    axes[0, 2].set_title(f'Spatial FFT at Sensors ({num_sensors} sensors)')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Time series at one sensor
    sensor_idx = 0
    axes[1, 0].plot(t, U_sim[:, sensor_indices[sensor_idx]], 'b-', lw=1.5, label='Sim')
    axes[1, 0].plot(t, U_real[:, sensor_indices[sensor_idx]], 'r--', lw=1, label='Real')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Signal')
    axes[1, 0].set_title(f'Sensor {sensor_idx} Time Series')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Temporal FFT of sensor signal
    temporal_fft_sim = np.abs(np.fft.rfft(U_sim[:, sensor_indices[sensor_idx]]))
    temporal_fft_real = np.abs(np.fft.rfft(U_real[:, sensor_indices[sensor_idx]]))
    temporal_freqs = np.fft.rfftfreq(len(t), t[1] - t[0])

    axes[1, 1].plot(temporal_freqs, temporal_fft_sim, 'b-', lw=2, label='Sim')
    axes[1, 1].plot(temporal_freqs, temporal_fft_real, 'r--', lw=1.5, label='Real')
    axes[1, 1].set_xlabel('Temporal Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Temporal FFT of Sensor Signal')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 3)
    axes[1, 1].grid(alpha=0.3)

    # Heatmaps
    axes[1, 2].imshow(U_real.T, aspect='auto', cmap='RdBu_r', origin='lower',
                      extent=[t[0], t[-1], 0, len(x)])
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('x index')
    axes[1, 2].set_title('Real Data (Full Resolution)')

    plt.suptitle(f'Beyond Nyquist Experiment: k1={k1}, k2={k2}, Nyquist={nyquist_k}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


# =============================================================================
# 2. Dataset
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        self.U = U
        self.sensor_indices = sensor_indices
        self.lags = lags
        self.S = U[:, sensor_indices]

        if scaler is None:
            self.scaler_U = MinMaxScaler()
            self.scaler_S = MinMaxScaler()
        else:
            self.scaler_U, self.scaler_S = scaler

        if fit_scaler:
            self.U_scaled = self.scaler_U.fit_transform(self.U)
            self.S_scaled = self.scaler_S.fit_transform(self.S)
        else:
            self.U_scaled = self.scaler_U.transform(self.U)
            self.S_scaled = self.scaler_S.transform(self.S)

        self.valid_indices = np.arange(lags, len(U))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        sensor_history = self.S_scaled[t - self.lags:t]
        full_state = self.U_scaled[t]
        return (torch.tensor(sensor_history, dtype=torch.float32),
                torch.tensor(full_state, dtype=torch.float32))

    def get_scalers(self):
        return (self.scaler_U, self.scaler_S)


# =============================================================================
# 3. Models
# =============================================================================

class SHRED(nn.Module):
    """LSTM encoder-decoder"""
    def __init__(self, num_sensors, lags, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_sensors, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


class HF_SHRED(nn.Module):
    """HF pathway for discovering missing frequencies"""
    def __init__(self, num_sensors, lags, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(num_sensors, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.scale = nn.Parameter(torch.tensor(0.5))

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])

    def forward(self, x):
        z = self.encode(x)
        return self.scale * self.decoder(z), z


class LatentGAN(nn.Module):
    """GAN for latent alignment"""
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        with torch.no_grad():
            self.generator[-1].weight.mul_(0.1)
            self.generator[-1].bias.zero_()

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return z + self.generator(z)


class BeyondNyquistDASHRED(nn.Module):
    """DA-SHRED for testing beyond-Nyquist frequency recovery."""

    def __init__(self, lf_shred, num_sensors, lags, hidden_size, output_size, sensor_indices):
        super().__init__()

        self.lf_lstm = copy.deepcopy(lf_shred.lstm)
        self.lf_norm = copy.deepcopy(lf_shred.norm)
        self.lf_decoder = copy.deepcopy(lf_shred.decoder)
        self.gan = LatentGAN(lf_shred.hidden_size)

        self.hf_shred = HF_SHRED(num_sensors, lags, hidden_size, output_size)

        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        self.lags = lags

    def encode_lf(self, x):
        _, (h_n, _) = self.lf_lstm(x)
        return self.lf_norm(h_n[-1])

    def decode_lf(self, z):
        return self.lf_decoder(z)

    def forward(self, sensor_history, use_gan=True):
        z_lf = self.encode_lf(sensor_history)
        if use_gan:
            z_lf = self.gan(z_lf)
        u_lf = self.decode_lf(z_lf)

        residual_history = torch.zeros_like(sensor_history)
        sensors_lf_pred = u_lf[:, self.sensor_indices]

        for lag in range(self.lags):
            residual_history[:, lag, :] = sensor_history[:, lag, :] - sensors_lf_pred.detach()

        u_hf, z_hf = self.hf_shred(residual_history)
        u_total = u_lf + u_hf

        return u_total, u_lf, u_hf, z_lf, z_hf


# =============================================================================
# 4. Sparsity Loss (NO bandlimited constraint - we want to see if it finds k>Nyquist!)
# =============================================================================

def frequency_sparsity_l1l2(signal):
    """Basic L1/L2 sparsity - no bandlimiting to allow super-Nyquist discovery."""
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    l1 = torch.sum(magnitudes, dim=-1)
    l2 = torch.sqrt(torch.sum(magnitudes ** 2, dim=-1) + 1e-8)

    return torch.mean(l1 / (l2 + 1e-8))


def frequency_sparsity_topk(signal, k=1):
    """Encourage concentration in top-k frequencies."""
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    topk_vals, _ = torch.topk(magnitudes, k, dim=-1)
    topk_energy = torch.sum(topk_vals ** 2, dim=-1)
    total_energy = torch.sum(magnitudes ** 2, dim=-1) + 1e-8

    outside_topk_ratio = 1 - topk_energy / total_energy
    return torch.mean(outside_topk_ratio)


# =============================================================================
# 5. Training Functions
# =============================================================================

def train_lf_shred(model, train_loader, epochs=200, lr=1e-3):
    print("\n=== Stage 1: Train LF-SHRED on Simulation ===")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sensors, targets in train_loader:
            sensors, targets = sensors.to(device), targets.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    return model


def train_gan(dashred, z_sim, z_real, epochs=300, lr=1e-4):
    print("\n=== Stage 2: Train GAN for LF Alignment ===")

    z_sim = torch.tensor(z_sim, dtype=torch.float32).to(device)
    z_real = torch.tensor(z_real, dtype=torch.float32).to(device)

    opt_g = optim.Adam(dashred.gan.generator.parameters(), lr=lr)
    opt_d = optim.Adam(dashred.gan.discriminator.parameters(), lr=lr)

    batch_size = 32
    n_batches = min(len(z_sim), len(z_real)) // batch_size

    for epoch in range(epochs):
        perm_sim = torch.randperm(len(z_sim))
        perm_real = torch.randperm(len(z_real))

        for i in range(n_batches):
            z_s = z_sim[perm_sim[i*batch_size:(i+1)*batch_size]]
            z_r = z_real[perm_real[i*batch_size:(i+1)*batch_size]]

            opt_d.zero_grad()
            z_fake = dashred.gan(z_s)
            d_loss = F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_r), torch.ones(len(z_r), 1, device=device)
            ) + F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_fake.detach()), torch.zeros(len(z_s), 1, device=device)
            )
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            z_fake = dashred.gan(z_s)
            g_loss = F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_fake), torch.ones(len(z_s), 1, device=device)
            )
            g_loss.backward()
            opt_g.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")


def train_hf_sparse(dashred, train_loader_real, sensor_indices, epochs=500, lr=1e-3,
                    lambda_sparse=0.1):
    """
    Train HF-SHRED WITHOUT bandlimited constraint.
    Key test: can it discover high spatial frequencies?
    """
    print(f"\n=== Stage 3: Train HF-SHRED ===")
    print(f"    Sparsity λ={lambda_sparse}")

    params = list(dashred.hf_shred.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    nyquist_k = len(sensor_indices) // 2

    history = {'sensor_loss': [], 'sparsity_loss': [], 'discovered_freqs': []}

    with torch.no_grad():
        sample_sensors, _ = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        u_lf, _, _, _, _ = dashred(sample_sensors, use_gan=True)
        sensors_current = sample_sensors[:, -1, :]
        sensors_lf = u_lf[:, sensor_idx]
        residual_scale = (sensors_current - sensors_lf).abs().mean().item()
    print(f"    Estimated residual scale: {residual_scale:.4f}")
    max_hf_magnitude = residual_scale * 3

    warmup_epochs = 100

    for epoch in range(epochs):
        dashred.train()
        epoch_sensor_loss = 0
        epoch_sparsity_loss = 0

        if epoch < warmup_epochs:
            current_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            current_lambda = lambda_sparse * min(1.0, progress * 2)

        for sensors, targets in train_loader_real:
            sensors = sensors.to(device)
            optimizer.zero_grad()

            u_total, u_lf, u_hf, _, _ = dashred(sensors, use_gan=True)

            sensors_current = sensors[:, -1, :]
            sensors_lf = u_lf[:, sensor_idx].detach()
            sensors_residual_true = sensors_current - sensors_lf
            sensors_hf = u_hf[:, sensor_idx]

            sensor_loss = F.mse_loss(sensors_hf, sensors_residual_true)
            sparsity_loss = frequency_sparsity_l1l2(u_hf) + 10.0 * frequency_sparsity_topk(u_hf, k=1)

            hf_magnitude = u_hf.abs().mean()
            magnitude_penalty = F.relu(hf_magnitude - max_hf_magnitude) ** 2

            loss = sensor_loss + current_lambda * sparsity_loss + 1.0 * magnitude_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_sensor_loss += sensor_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

        n_batches = len(train_loader_real)
        epoch_sensor_loss /= n_batches
        epoch_sparsity_loss /= n_batches

        history['sensor_loss'].append(epoch_sensor_loss)
        history['sparsity_loss'].append(epoch_sparsity_loss)

        scheduler.step(epoch_sensor_loss)

        if (epoch + 1) % 50 == 0:
            dashred.eval()
            with torch.no_grad():
                sample_sensors = next(iter(train_loader_real))[0][:1].to(device)
                _, _, u_hf_sample, _, _ = dashred(sample_sensors, use_gan=True)
                fft_mag = torch.abs(torch.fft.rfft(u_hf_sample, dim=-1)).squeeze().cpu().numpy()
                fft_mag[0] = 0  # Exclude DC
                top_freq = np.argmax(fft_mag)
                top_5_freqs = np.argsort(fft_mag)[-5:][::-1]
                hf_max = u_hf_sample.abs().max().item()
                history['discovered_freqs'].append(top_freq)

            print(f"  Epoch {epoch+1}/{epochs}, Sensor: {epoch_sensor_loss:.6f}, "
                  f"Sparsity: {epoch_sparsity_loss:.4f}, HF_max: {hf_max:.4f}, "
                  f"Top freq: k={top_freq}")
            print(f"    Top 5 frequencies: {top_5_freqs}")

    return dashred, history


# =============================================================================
# 6. Visualization
# =============================================================================

def plot_top_row(x, results, k1, k2, nyquist_k, save_path):
    """Plot 4 panels showing super-Nyquist frequency recovery."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    idx = len(results['targets']) // 2

    # Panel (a): LF Reconstruction
    axes[0].plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    axes[0].plot(x, results['lf_only'][idx], 'r--', lw=1.5, label='LF only')
    axes[0].set_title('(a) LF Reconstruction', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    # Panel (b): Full Reconstruction
    axes[1].plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    axes[1].plot(x, results['lf_hf'][idx], 'g--', lw=1.5, label='LF+HF')
    axes[1].set_title('(b) Full Reconstruction', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    # Panel (c): HF Component
    hf_true = results['targets'][idx] - results['lf_only'][idx]
    axes[2].plot(x, hf_true, 'b-', lw=2, label='True HF')
    axes[2].plot(x, results['hf'][idx], 'r--', lw=1.5, label='Predicted HF (unscaled)')
    axes[2].set_title(f'(c) HF Component (k={k2})', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)

    # Panel (d): HF Spectrum
    fft_hf = np.abs(np.fft.rfft(results['hf'], axis=1)).mean(axis=0)
    fft_hf[0] = 0
    freqs = np.arange(len(fft_hf))

    axes[3].stem(freqs, fft_hf, basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[3].axvline(nyquist_k, color='green', linestyle='--', lw=2, alpha=0.7, label=f'Sensor Nyquist')
    axes[3].axvline(k2, color='blue', linestyle=':', lw=2, label=f'Target k={k2}')
    axes[3].set_xlabel('Frequency (k)')
    axes[3].set_ylabel('Magnitude')
    axes[3].set_title('(d) Discovered HF Spectrum', fontsize=12)
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].set_xlim(0, min(40, len(fft_hf)))
    axes[3].grid(alpha=0.3)

    plt.suptitle(f'Super-Nyquist Frequency Recovery: k1={k1}, k2={k2}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


def plot_results(x, results, k1, k2, nyquist_k, history, save_path):
    """Comprehensive visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    idx = len(results['targets']) // 2

    # Row 1: Spatial reconstructions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    ax1.plot(x, results['lf_only'][idx], 'r--', lw=1.5, label='LF only')
    ax1.set_title('(a) LF Reconstruction')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    ax2.plot(x, results['lf_hf'][idx], 'g--', lw=1.5, label='LF+HF')
    ax2.set_title('(b) Full Reconstruction')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    hf_true = results['targets'][idx] - results['lf_only'][idx]
    ax3.plot(x, hf_true, 'b-', lw=2, label='True HF')
    ax3.plot(x, results['hf'][idx], 'r--', lw=1.5, label='Predicted HF')
    ax3.set_title(f'(c) HF Component (k={k2})')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # HF spectrum
    ax4 = fig.add_subplot(gs[0, 3])
    fft_hf = np.abs(np.fft.rfft(results['hf'], axis=1)).mean(axis=0)
    fft_hf[0] = 0
    freqs = np.arange(len(fft_hf))
    ax4.stem(freqs, fft_hf, basefmt=' ', linefmt='r-', markerfmt='ro')
    ax4.axvline(nyquist_k, color='green', linestyle='--', lw=2, label=f'Nyquist (k={nyquist_k})')
    ax4.axvline(k2, color='blue', linestyle=':', lw=2, label=f'Target k={k2}')
    ax4.set_xlabel('Frequency (k)')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('(d) Discovered HF Spectrum')
    ax4.legend()
    ax4.set_xlim(0, min(40, len(fft_hf)))
    ax4.grid(alpha=0.3)

    # Row 2: Heatmaps
    vmin, vmax = results['targets'].min(), results['targets'].max()

    ax5 = fig.add_subplot(gs[1, 0])
    im1 = ax5.imshow(results['targets'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax5.set_title('Ground Truth')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('x')
    plt.colorbar(im1, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 1])
    im2 = ax6.imshow(results['lf_only'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax6.set_title('LF Only')
    ax6.set_xlabel('Time')
    plt.colorbar(im2, ax=ax6)

    ax7 = fig.add_subplot(gs[1, 2])
    im3 = ax7.imshow(results['lf_hf'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax7.set_title('Full Reconstruction')
    ax7.set_xlabel('Time')
    plt.colorbar(im3, ax=ax7)

    ax8 = fig.add_subplot(gs[1, 3])
    error = np.abs(results['lf_hf'] - results['targets'])
    im4 = ax8.imshow(error.T, aspect='auto', cmap='hot', origin='lower')
    ax8.set_title('Absolute Error')
    ax8.set_xlabel('Time')
    plt.colorbar(im4, ax=ax8)

    # Row 3: Training history and summary
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(history['sensor_loss'], 'b-', lw=2, label='Sensor Loss')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss')
    ax9.set_title('Training History')
    ax9.legend()
    ax9.set_yscale('log')
    ax9.grid(alpha=0.3)

    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot(history['sparsity_loss'], 'r-', lw=2, label='Sparsity Loss')
    ax10.set_xlabel('Epoch')
    ax10.legend()
    ax10.grid(alpha=0.3)

    ax11 = fig.add_subplot(gs[2, 2])
    discovered = history['discovered_freqs']
    ax11.plot(range(0, len(discovered)*50, 50), discovered, 'go-', lw=2, markersize=8)
    ax11.axhline(k2, color='blue', linestyle='--', lw=2, label=f'Target k={k2}')
    ax11.axhline(nyquist_k, color='red', linestyle=':', lw=2, label=f'Nyquist k={nyquist_k}')
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Discovered Frequency')
    ax11.set_title('Frequency Discovery Over Training')
    ax11.legend()
    ax11.grid(alpha=0.3)

    # Summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    mse_lf = np.mean((results['lf_only'] - results['targets']) ** 2)
    mse_total = np.mean((results['lf_hf'] - results['targets']) ** 2)

    fft_hf_final = np.abs(np.fft.rfft(results['hf'], axis=1)).mean(axis=0)
    fft_hf_final[0] = 0
    discovered_k = np.argmax(fft_hf_final)

    summary = f"""
    BEYOND NYQUIST EXPERIMENT
    =========================
    
    Setup:
      LF: k1={k1} (in simulation)
      HF: k2={k2} (ABOVE Nyquist!)
      Nyquist limit: k={nyquist_k}
    
    Results:
      LF RMSE: {np.sqrt(mse_lf):.6f}
      Full RMSE: {np.sqrt(mse_total):.6f}
      Improvement: {(mse_lf - mse_total) / mse_lf * 100:.1f}%
    
    Frequency Discovery:
      Target: k={k2}
      Found: k={discovered_k}
      {'SUCCESS: Found super-Nyquist!' if discovered_k == k2 else 'FAILED' if discovered_k != k2 else ''}
      {'(But aliased!)' if discovered_k != k2 and discovered_k > 0 else ''}
    """
    ax12.text(0.05, 0.5, summary, fontsize=10, family='monospace',
              verticalalignment='center', transform=ax12.transAxes,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Beyond Nyquist: k1={k1}, k2={k2}, Nyquist={nyquist_k}', fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


# =============================================================================
# 7. Main Experiment
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SUPER-NYQUIST FREQUENCY RECOVERY EXPERIMENT")
    print("Using time-delayed embeddings to recover high spatial frequencies")
    print("="*70)

    # Parameters
    L = 2 * np.pi
    N = 128  # Full spatial resolution
    T = 20.0  # Increased for more data with high lags
    dt = 0.05

    # Frequency setup
    k1 = 2   # Low frequency (in simulation)
    k2 = 20  # High frequency (in real data)
    omega1 = 1
    omega2 = 10
    A1, A2 = 1.0, 0.4

    num_sensors = 32
    nyquist_k = num_sensors // 2  # = 16
    lags = 12  # Extended temporal history
    hidden_size = 64

    lambda_sparse = 0.1

    print(f"\n[0] Configuration:")
    print(f"    Spatial frequencies: k1={k1} (sim), k2={k2} (real)")
    print(f"    Temporal frequencies: ω1={omega1}, ω2={omega2}")
    print(f"    Sensors: {num_sensors}, Lags: {lags}")
    print(f"    dt={dt} → temporal sampling rate: {1/dt:.1f} Hz")

    # Generate data
    print("\n[1] Generating data...")
    U_sim, x, t = generate_traveling_wave_data(
        L, N, T, dt, k1, k2, omega1, omega2, A1, A2, include_hf=False
    )
    U_real, _, _ = generate_traveling_wave_data(
        L, N, T, dt, k1, k2, omega1, omega2, A1, A2, include_hf=True
    )
    print(f"    U_sim: {U_sim.shape}")
    print(f"    U_real: {U_real.shape}")

    # Analyze
    sensor_indices = np.linspace(0, N-1, num_sensors, dtype=int)
    analyze_sensor_signals(U_real, sensor_indices, x, dt, k1, k2, omega1, omega2)

    print("\n[2] Plotting data diagnostic...")
    plot_data_diagnostic(U_sim, U_real, x, t, sensor_indices, k1, k2, num_sensors,
                        'super_nyquist_data_diagnostic.png')

    # Create datasets
    print("\n[3] Creating datasets...")
    n_train = int(0.8 * len(U_sim))

    U_combined = np.vstack([U_sim, U_real])
    temp_dataset = TimeSeriesDataset(U_combined, sensor_indices, lags, fit_scaler=True)
    combined_scaler = temp_dataset.get_scalers()

    train_sim = TimeSeriesDataset(U_sim[:n_train], sensor_indices, lags, scaler=combined_scaler, fit_scaler=False)
    train_real = TimeSeriesDataset(U_real[:n_train], sensor_indices, lags, scaler=combined_scaler)
    valid_real = TimeSeriesDataset(U_real[n_train:], sensor_indices, lags, scaler=combined_scaler)

    scaler_U, _ = combined_scaler

    train_loader_sim = DataLoader(train_sim, batch_size=32, shuffle=True)
    train_loader_real = DataLoader(train_real, batch_size=32, shuffle=True)
    valid_loader_real = DataLoader(valid_real, batch_size=32)

    # Stage 1: Train LF-SHRED
    print("\n[4] Stage 1: Train LF-SHRED...")
    lf_shred = SHRED(num_sensors, lags, hidden_size, N).to(device)
    lf_shred = train_lf_shred(lf_shred, train_loader_sim, epochs=200)

    # Create DA-SHRED
    print("\n[5] Creating Beyond-Nyquist DA-SHRED...")
    dashred = BeyondNyquistDASHRED(lf_shred, num_sensors, lags, hidden_size, N, sensor_indices).to(device)

    # Stage 2: GAN
    print("\n[6] Extracting latents...")
    dashred.eval()
    Z_sim_list, Z_real_list = [], []
    with torch.no_grad():
        for sensors, _ in train_loader_sim:
            Z_sim_list.append(dashred.encode_lf(sensors.to(device)).cpu())
        for sensors, _ in train_loader_real:
            Z_real_list.append(dashred.encode_lf(sensors.to(device)).cpu())
    Z_sim = torch.cat(Z_sim_list).numpy()
    Z_real = torch.cat(Z_real_list).numpy()

    print("\n[7] Stage 2: Train GAN...")
    train_gan(dashred, Z_sim, Z_real, epochs=300)

    # Stage 3: HF without bandlimit
    print("\n[8] Stage 3: Train HF-SHRED (no bandlimit constraint)...")
    dashred, history = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=500, lambda_sparse=lambda_sparse
    )

    # Fine-tune
    print("\n[9] Fine-tuning...")
    dashred, history2 = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=200, lambda_sparse=lambda_sparse * 0.1
    )
    for key in history:
        if key in history2:
            history[key].extend(history2[key])

    # Evaluate
    print("\n[10] Evaluating...")
    dashred.eval()

    results = {'lf_only': [], 'lf_hf': [], 'hf': [], 'targets': []}

    with torch.no_grad():
        for sensors, targets in valid_loader_real:
            sensors = sensors.to(device)
            u_total, u_lf, u_hf, _, _ = dashred(sensors, use_gan=True)

            results['lf_only'].append(u_lf.cpu())
            results['lf_hf'].append(u_total.cpu())
            results['hf'].append(u_hf.cpu())
            results['targets'].append(targets)

    for k in results:
        results[k] = scaler_U.inverse_transform(torch.cat(results[k]).numpy())

    # Metrics
    mse_lf = np.mean((results['lf_only'] - results['targets']) ** 2)
    mse_total = np.mean((results['lf_hf'] - results['targets']) ** 2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  k1={k1} (simulation), k2={k2} (real)")
    print(f"\n  RMSE:")
    print(f"    LF only:  {np.sqrt(mse_lf):.6f}")
    print(f"    LF+HF:    {np.sqrt(mse_total):.6f}")
    print(f"    Improvement: {(mse_lf - mse_total) / mse_lf * 100:.1f}%")

    # Frequency analysis
    print("\n[11] Frequency Analysis...")
    fft_hf = np.abs(np.fft.rfft(results['hf'], axis=1)).mean(axis=0)
    fft_hf[0] = 0

    top_freqs = np.argsort(fft_hf)[-5:][::-1]
    print(f"\n  Top 5 discovered frequencies: {top_freqs}")
    print(f"  Their magnitudes: {fft_hf[top_freqs]}")
    print(f"\n  Target k2={k2}: magnitude={fft_hf[k2]:.4f}")

    discovered_k = np.argmax(fft_hf)
    if discovered_k == k2:
        print(f"\n  *** SUCCESS: Discovered k={discovered_k} = target! ***")
    else:
        print(f"\n  Found k={discovered_k} (target was k2={k2})")

    # Plot
    print("\n[12] Plotting results...")
    plot_top_row(x, results, k1, k2, nyquist_k, 'super_nyquist_top_row.png')
    plot_results(x, results, k1, k2, nyquist_k, history, 'super_nyquist_full_results.png')

    print("\n" + "="*70)
    print("BEYOND-NYQUIST EXPERIMENT COMPLETE")

    print("="*70)
