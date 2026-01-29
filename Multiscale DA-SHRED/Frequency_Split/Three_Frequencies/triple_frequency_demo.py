"""
Three-Frequency DA-SHRED with Sparse-Frequency HF Learning

Extension of the two-frequency demo to test whether the sparse-frequency
architecture can successfully identify and separate THREE distinct frequency modes.

Setup:
- LF mode (simulation knows): k1 (lowest frequency)
- MF mode (missing from sim): k2 (medium frequency)
- HF mode (missing from sim): k3 (highest frequency)

The HF pathway must discover BOTH k2 and k3 from sensor-only supervision.
This is a harder test of the sparsity regularization.

Loss = sensor_residual_loss + λ_sparse * frequency_sparsity_loss
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

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# 1. Data Generation - THREE FREQUENCIES
# =============================================================================

def generate_three_frequency_data(L=2*np.pi, N=128, T=10.0, dt=0.05,
                                   k1=2, k2=5, k3=11,
                                   omega1=1, omega2=3, omega3=7,
                                   A1=1.0, A2=0.4, A3=0.25,
                                   include_mf=True, include_hf=True,
                                   phase_shift=0.0):
    """Generate traveling wave data with up to three frequency components."""
    x = np.linspace(0, L, N, endpoint=False)
    t = np.arange(0, T, dt)
    n_steps = len(t)

    U = np.zeros((n_steps, N))
    for i, ti in enumerate(t):
        u_lf = A1 * np.sin(k1 * x - omega1 * ti)
        u_mf = A2 * np.sin(k2 * x - omega2 * ti + phase_shift) if include_mf else 0
        u_hf = A3 * np.sin(k3 * x - omega3 * ti + 2*phase_shift) if include_hf else 0
        U[i] = u_lf + u_mf + u_hf
    return U, x, t


def generate_standing_wave_data(L=2*np.pi, N=128, T=10.0, dt=0.05,
                                 k1=2, k2=5, k3=11,
                                 omega1=1, omega2=3, omega3=7,
                                 A1=1.0, A2=0.4, A3=0.25,
                                 include_mf=True, include_hf=True):
    """Standing wave version: sin(kx)*cos(omega*t)"""
    x = np.linspace(0, L, N, endpoint=False)
    t = np.arange(0, T, dt)
    n_steps = len(t)

    U = np.zeros((n_steps, N))
    for i, ti in enumerate(t):
        u_lf = A1 * np.sin(k1 * x) * np.cos(omega1 * ti)
        u_mf = A2 * np.sin(k2 * x) * np.cos(omega2 * ti) if include_mf else 0
        u_hf = A3 * np.sin(k3 * x) * np.cos(omega3 * ti) if include_hf else 0
        U[i] = u_lf + u_mf + u_hf
    return U, x, t


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
# 3. Frequency Sparsity Losses
# =============================================================================

def frequency_sparsity_l1(signal):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    return torch.mean(torch.sum(magnitudes, dim=-1))


def frequency_sparsity_entropy(signal):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    power = torch.abs(fft_coeffs) ** 2
    power_sum = torch.sum(power, dim=-1, keepdim=True) + 1e-8
    p = power / power_sum
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    return torch.mean(entropy)


def frequency_sparsity_normalized_l1(signal):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    l1_norm = torch.sum(magnitudes, dim=-1)
    l2_norm = torch.sqrt(torch.sum(magnitudes ** 2, dim=-1) + 1e-8)
    return torch.mean(l1_norm / (l2_norm + 1e-8))


def frequency_sparsity_topk(signal, k=3):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    topk_vals, _ = torch.topk(magnitudes, k, dim=-1)
    topk_energy = torch.sum(topk_vals ** 2, dim=-1)
    total_energy = torch.sum(magnitudes ** 2, dim=-1) + 1e-8
    outside_topk_ratio = 1 - topk_energy / total_energy
    return torch.mean(outside_topk_ratio)


def frequency_sparsity_bandlimited(signal, max_freq):
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    in_band = magnitudes[:, :max_freq+1]
    out_of_band = magnitudes[:, max_freq+1:]

    l1 = torch.sum(in_band, dim=-1)
    l2 = torch.sqrt(torch.sum(in_band ** 2, dim=-1) + 1e-8)
    sparsity_loss = l1 / (l2 + 1e-8)

    out_of_band_energy = torch.sum(out_of_band ** 2, dim=-1)
    total_energy = torch.sum(magnitudes ** 2, dim=-1) + 1e-8
    out_of_band_ratio = out_of_band_energy / total_energy

    return torch.mean(sparsity_loss) + 100.0 * torch.mean(out_of_band_ratio)


# =============================================================================
# 4. Models
# =============================================================================

class SHRED(nn.Module):
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
    def __init__(self, num_sensors, lags, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

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


class SparseFreqDASHRED(nn.Module):
    def __init__(self, lf_shred, num_sensors, lags, hidden_size, output_size, sensor_indices):
        super().__init__()

        self.lf_lstm = copy.deepcopy(lf_shred.lstm)
        self.lf_norm = copy.deepcopy(lf_shred.norm)
        self.lf_decoder = copy.deepcopy(lf_shred.decoder)
        self.gan = LatentGAN(lf_shred.hidden_size)

        self.hf_shred = HF_SHRED(num_sensors, lags, hidden_size, output_size)

        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        self.lags = lags
        self.num_sensors = num_sensors

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
                    lambda_sparse=0.1, sparsity_type='bandlimited', target_num_modes=2):
    print(f"\n=== Stage 3: Train HF-SHRED ({sparsity_type} Sparsity, λ={lambda_sparse}) ===")
    print(f"    Target: discover {target_num_modes} missing frequency modes")

    params = list(dashred.hf_shred.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    max_target_freq = len(sensor_indices) // 2
    print(f"    Max target frequency: k <= {max_target_freq}")

    sparsity_fn = {
        'l1': frequency_sparsity_l1,
        'entropy': frequency_sparsity_entropy,
        'normalized_l1': frequency_sparsity_normalized_l1,
        'topk': lambda x: frequency_sparsity_topk(x, k=target_num_modes),
        'bandlimited': lambda x: frequency_sparsity_bandlimited(x, max_target_freq),
    }[sparsity_type]

    history = {'sensor_loss': [], 'sparsity_loss': [], 'total_loss': [], 'discovered_freqs': []}

    with torch.no_grad():
        sample_sensors, _ = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        _, u_lf, _, _, _ = dashred(sample_sensors, use_gan=True)
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
            sparsity_loss = sparsity_fn(u_hf)

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
        history['total_loss'].append(epoch_sensor_loss + current_lambda * epoch_sparsity_loss)

        scheduler.step(epoch_sensor_loss)

        if (epoch + 1) % 50 == 0:
            dashred.eval()
            with torch.no_grad():
                sample_sensors = next(iter(train_loader_real))[0][:1].to(device)
                _, _, u_hf_sample, _, _ = dashred(sample_sensors, use_gan=True)
                fft_mag = torch.abs(torch.fft.rfft(u_hf_sample, dim=-1)).squeeze().cpu().numpy()
                top_freqs = np.argsort(fft_mag)[-4:][::-1]
                hf_max = u_hf_sample.abs().max().item()
                history['discovered_freqs'].append(top_freqs.tolist())

            print(f"  Epoch {epoch+1}/{epochs}, Sensor: {epoch_sensor_loss:.6f}, "
                  f"Sparsity: {epoch_sparsity_loss:.4f}, HF_max: {hf_max:.4f}, "
                  f"Top freqs: {top_freqs}")

    return dashred, history


# =============================================================================
# 6. Analysis and Visualization
# =============================================================================

def analyze_frequency_spectrum(u_hf, true_k2, true_k3, title="HF Frequency Spectrum"):
    fft_mag = np.abs(np.fft.rfft(u_hf))
    threshold = fft_mag.max() * 0.1
    peaks = []
    for i in range(1, len(fft_mag)-1):
        if fft_mag[i] > fft_mag[i-1] and fft_mag[i] > fft_mag[i+1] and fft_mag[i] > threshold:
            peaks.append((i, fft_mag[i]))
    peaks.sort(key=lambda x: -x[1])

    print(f"\n=== {title} ===")
    print(f"  True missing frequencies: k2={true_k2}, k3={true_k3}")
    print(f"  Detected peaks:")
    for i, (k, mag) in enumerate(peaks[:5]):
        match = " <-- MATCH!" if k in [true_k2, true_k3] else ""
        print(f"    Rank {i+1}: k={k}, magnitude={mag:.4f}{match}")
    return fft_mag, peaks


def compute_frequency_accuracy(fft_mag, true_freqs, tolerance=1):
    n_true = len(true_freqs)
    top_k_idx = np.argsort(fft_mag)[-n_true:][::-1]
    matches = 0
    for true_k in true_freqs:
        for discovered_k in top_k_idx:
            if abs(discovered_k - true_k) <= tolerance:
                matches += 1
                break
    return matches / n_true, top_k_idx


def plot_top_row(x, results, k1, k2, k3, save_path):
    """Plot just the top row: 4 panels showing spatial reconstructions and spectrum"""
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

    # Panel (b): Full Reconstruction (LF + HF)
    axes[1].plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    axes[1].plot(x, results['lf_hf'][idx], 'g--', lw=1.5, label='LF + HF')
    axes[1].set_title('(b) Full Reconstruction (LF + HF)', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    # Panel (c): HF Component
    hf_true = results['targets'][idx] - results['lf_only'][idx]
    hf_pred = results['lf_hf'][idx] - results['lf_only'][idx]
    axes[2].plot(x, hf_true, 'b-', lw=2, label='True HF (k2+k3)')
    axes[2].plot(x, hf_pred, 'r--', lw=1.5, label='Predicted HF')
    axes[2].set_title('(c) High Frequency Component', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)

    # Panel (d): Frequency Spectrum
    fft_mag = np.abs(np.fft.rfft(hf_pred))
    freqs = np.arange(len(fft_mag))
    axes[3].stem(freqs, fft_mag, basefmt=' ')
    axes[3].axvline(k2, color='g', linestyle='--', lw=2, label=f'True k2={k2}')
    axes[3].axvline(k3, color='m', linestyle='--', lw=2, label=f'True k3={k3}')
    axes[3].set_xlabel('Frequency (k)')
    axes[3].set_ylabel('Magnitude')
    axes[3].set_title('(d) HF Frequency Spectrum', fontsize=12)
    axes[3].legend(loc='upper right')
    axes[3].set_xlim(0, 25)
    axes[3].grid(alpha=0.3)

    plt.suptitle(f'Three-Frequency Separation: k1={k1} (LF), k2={k2} (MF), k3={k3} (HF)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


def plot_results(x, results, k1, k2, k3, history, save_path):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    idx = len(results['targets']) // 2

    # Row 1: Spatial reconstructions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    ax1.plot(x, results['lf_only'][idx], 'r--', lw=1.5, label='LF only')
    ax1.set_title('LF Reconstruction')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    ax2.plot(x, results['lf_hf'][idx], 'g--', lw=1.5, label='LF + HF')
    ax2.set_title('Full Reconstruction (LF + HF)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    hf_true = results['targets'][idx] - results['lf_only'][idx]
    hf_pred = results['lf_hf'][idx] - results['lf_only'][idx]
    ax3.plot(x, hf_true, 'b-', lw=2, label='True HF')
    ax3.plot(x, hf_pred, 'r--', lw=1.5, label='Predicted HF')
    ax3.set_title('High Frequency Component')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    fft_mag = np.abs(np.fft.rfft(hf_pred))
    freqs = np.arange(len(fft_mag))
    ax4.stem(freqs, fft_mag, basefmt=' ')
    ax4.axvline(k2, color='g', linestyle='--', lw=2, label=f'True k2={k2}')
    ax4.axvline(k3, color='m', linestyle='--', lw=2, label=f'True k3={k3}')
    ax4.set_xlabel('Frequency (k)')
    ax4.set_title('HF Frequency Spectrum')
    ax4.legend()
    ax4.set_xlim(0, 25)
    ax4.grid(alpha=0.3)

    # Row 2: Heatmaps
    vmin, vmax = results['targets'].min(), results['targets'].max()

    ax5 = fig.add_subplot(gs[1, 0])
    im1 = ax5.imshow(results['targets'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax5.set_title('Ground Truth (k1+k2+k3)')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('x')
    plt.colorbar(im1, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 1])
    im2 = ax6.imshow(results['lf_only'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax6.set_title('LF Only (missing k2, k3)')
    ax6.set_xlabel('Time')
    plt.colorbar(im2, ax=ax6)

    ax7 = fig.add_subplot(gs[1, 2])
    im3 = ax7.imshow(results['lf_hf'].T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    ax7.set_title('Reconstruction (LF + HF)')
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
    ax9.plot(history['sensor_loss'], label='Sensor Loss', lw=2)
    ax9.plot(history['sparsity_loss'], label='Sparsity Loss', lw=2)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss')
    ax9.set_title('Training History')
    ax9.legend()
    ax9.grid(alpha=0.3)
    ax9.set_yscale('log')

    ax10 = fig.add_subplot(gs[2, 1])
    if history['discovered_freqs']:
        freqs_over_time = np.array(history['discovered_freqs'])
        for i in range(min(4, freqs_over_time.shape[1])):
            epochs = np.arange(50, 50*(len(freqs_over_time)+1), 50)
            ax10.plot(epochs, freqs_over_time[:, i], 'o-', label=f'Top-{i+1}')
        ax10.axhline(k2, color='g', linestyle='--', lw=2, label=f'True k2={k2}')
        ax10.axhline(k3, color='m', linestyle='--', lw=2, label=f'True k3={k3}')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Discovered Frequency k')
        ax10.set_title('Frequency Discovery Over Training')
        ax10.legend(loc='upper right', fontsize=8)
        ax10.grid(alpha=0.3)

    # Summary
    ax11 = fig.add_subplot(gs[2, 2:])
    ax11.axis('off')

    mse_lf = np.mean((results['lf_only'] - results['targets']) ** 2)
    mse_total = np.mean((results['lf_hf'] - results['targets']) ** 2)
    improvement = (mse_lf - mse_total) / mse_lf * 100

    hf_pred_all = results['lf_hf'] - results['lf_only']
    fft_mag_avg = np.abs(np.fft.rfft(hf_pred_all, axis=1)).mean(axis=0)
    accuracy, discovered = compute_frequency_accuracy(fft_mag_avg, [k2, k3])

    summary = f"""
    THREE-FREQUENCY EXPERIMENT SUMMARY
    ===================================
    
    True Frequencies:
      - k1 = {k1} (in simulation, learned by LF)
      - k2 = {k2} (missing, to be discovered)
      - k3 = {k3} (missing, to be discovered)
    
    Discovered Top-2 Frequencies: {discovered}
    Frequency Match Accuracy: {accuracy*100:.0f}%
    
    Reconstruction Quality:
      - LF-only RMSE: {np.sqrt(mse_lf):.6f}
      - LF+HF RMSE:   {np.sqrt(mse_total):.6f}
      - Improvement:  {improvement:.1f}%
    """
    ax11.text(0.1, 0.5, summary, fontsize=11, family='monospace',
              verticalalignment='center', transform=ax11.transAxes,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Sparse-Frequency Three-Mode Separation: k1={k1}, k2={k2}, k3={k3}', fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


# =============================================================================
# 7. Main Experiment
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("THREE-FREQUENCY DA-SHRED EXPERIMENT")
    print("="*70)

    # Parameters
    L = 2 * np.pi
    N = 128
    T = 10.0
    dt = 0.05

    # Three frequencies - all within target range
    k1 = 2   # LF mode (in simulation)
    k2 = 5   # MF mode (missing from sim)
    k3 = 11  # HF mode (missing from sim)

    omega1, omega2, omega3 = 1, 3, 7
    A1, A2, A3 = 1.0, 0.4, 0.25

    num_sensors = 32  # max_k = 16, so k3=11 is within range
    lags = 20
    hidden_size = 32

    lambda_sparse = 0.1
    sparsity_type = 'bandlimited'
    wave_type = 'traveling'

    print(f"\n[0] Configuration:")
    print(f"    Wave type: {wave_type}")
    print(f"    Frequencies: k1={k1} (LF), k2={k2} (MF), k3={k3} (HF)")
    print(f"    Amplitudes: A1={A1}, A2={A2}, A3={A3}")
    print(f"    Sensors: {num_sensors} (max_k={num_sensors//2})")

    # Generate data
    print("\n[1] Generating data...")
    data_fn = generate_three_frequency_data if wave_type == 'traveling' else generate_standing_wave_data

    U_sim, x, t = data_fn(L, N, T, dt, k1, k2, k3, omega1, omega2, omega3,
                          A1, A2, A3, include_mf=False, include_hf=False)
    U_real, _, _ = data_fn(L, N, T, dt, k1, k2, k3, omega1, omega2, omega3,
                           A1, A2, A3, include_mf=True, include_hf=True)

    print(f"    U_sim: {U_sim.shape} (k1 only)")
    print(f"    U_real: {U_real.shape} (k1 + k2 + k3)")

    # Diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(U_sim.T, aspect='auto', cmap='RdBu_r', origin='lower')
    axes[0].set_title(f'Simulation (k1={k1} only)')
    axes[1].imshow(U_real.T, aspect='auto', cmap='RdBu_r', origin='lower')
    axes[1].set_title(f'Real (k1+k2+k3)')
    axes[2].imshow((U_real - U_sim).T, aspect='auto', cmap='RdBu_r', origin='lower')
    axes[2].set_title('Difference (k2 + k3)')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
    plt.tight_layout()
    plt.savefig('triple_freq_data_diagnostic.png', dpi=150)
    print("    Saved: triple_freq_data_diagnostic.png")
    plt.close()

    # Create datasets
    print("\n[2] Creating datasets...")
    sensor_indices = np.linspace(0, N-1, num_sensors, dtype=int)
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
    print("\n[3] Stage 1: Train LF-SHRED on simulation...")
    lf_shred = SHRED(num_sensors, lags, hidden_size, N).to(device)
    lf_shred = train_lf_shred(lf_shred, train_loader_sim, epochs=200)

    # Create DA-SHRED
    print("\n[4] Creating SparseFreq DA-SHRED...")
    dashred = SparseFreqDASHRED(lf_shred, num_sensors, lags, hidden_size, N, sensor_indices).to(device)

    # Stage 2: Train GAN
    print("\n[5] Extracting latents for GAN...")
    dashred.eval()
    Z_sim_list, Z_real_list = [], []
    with torch.no_grad():
        for sensors, _ in train_loader_sim:
            Z_sim_list.append(dashred.encode_lf(sensors.to(device)).cpu())
        for sensors, _ in train_loader_real:
            Z_real_list.append(dashred.encode_lf(sensors.to(device)).cpu())
    Z_sim = torch.cat(Z_sim_list).numpy()
    Z_real = torch.cat(Z_real_list).numpy()

    print("\n[6] Stage 2: Train GAN...")
    train_gan(dashred, Z_sim, Z_real, epochs=300)

    # Stage 3: Train HF with sparsity
    print(f"\n[7] Stage 3: Train HF-SHRED with {sparsity_type} sparsity...")
    dashred, history = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=500, lambda_sparse=lambda_sparse, sparsity_type=sparsity_type,
        target_num_modes=2
    )

    # Stage 4: Fine-tune
    print("\n[7.5] Stage 4: Fine-tuning with reduced sparsity...")
    dashred, history2 = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=200, lambda_sparse=lambda_sparse * 0.1, sparsity_type=sparsity_type,
        target_num_modes=2
    )

    for key in history:
        if key in history2:
            history[key].extend(history2[key])

    # Evaluate
    print("\n[8] Evaluating...")
    dashred.eval()

    results = {'lf_only': [], 'lf_hf': [], 'targets': []}

    with torch.no_grad():
        for sensors, targets in valid_loader_real:
            sensors = sensors.to(device)

            z_lf = dashred.encode_lf(sensors)
            z_lf_aligned = dashred.gan(z_lf)
            u_lf = dashred.decode_lf(z_lf_aligned)

            u_total, _, u_hf, _, _ = dashred(sensors, use_gan=True)

            results['lf_only'].append(u_lf.cpu())
            results['lf_hf'].append(u_total.cpu())
            results['targets'].append(targets)

    for k in results:
        results[k] = scaler_U.inverse_transform(torch.cat(results[k]).numpy())

    mse_lf = np.mean((results['lf_only'] - results['targets']) ** 2)
    mse_total = np.mean((results['lf_hf'] - results['targets']) ** 2)

    print(f"\n=== RESULTS ===")
    print(f"  True missing frequencies: k2={k2}, k3={k3}")
    print(f"  LF-only RMSE: {np.sqrt(mse_lf):.6f}")
    print(f"  LF+HF RMSE: {np.sqrt(mse_total):.6f}")
    print(f"  Improvement: {(mse_lf - mse_total) / mse_lf * 100:.1f}%")

    # Analyze discovered frequencies
    print("\n[9] Analyzing discovered frequencies...")
    with torch.no_grad():
        sample = next(iter(valid_loader_real))[0][:1].to(device)
        _, _, u_hf, _, _ = dashred(sample, use_gan=True)
        u_hf_np = u_hf.cpu().numpy().squeeze()

    fft_mag, peaks = analyze_frequency_spectrum(u_hf_np, k2, k3)

    hf_pred_all = results['lf_hf'] - results['lf_only']
    fft_mag_avg = np.abs(np.fft.rfft(hf_pred_all, axis=1)).mean(axis=0)
    accuracy, discovered = compute_frequency_accuracy(fft_mag_avg, [k2, k3])
    print(f"\n  Averaged over validation set:")
    print(f"    Top-2 discovered: {discovered}")
    print(f"    Frequency match accuracy: {accuracy*100:.0f}%")

    # Plot
    print("\n[10] Plotting...")
    plot_results(x, results, k1, k2, k3, history, 'triple_freq_dashred_results.png')

    # Separate top row plot (4 panels)
    plot_top_row(x, results, k1, k2, k3, 'triple_freq_top_row.png')

    # Additional spectrum plot
    fig, ax = plt.subplots(figsize=(10, 5))
    freqs = np.arange(len(fft_mag_avg))
    ax.stem(freqs, fft_mag_avg, basefmt=' ')
    ax.axvline(k2, color='g', linestyle='--', lw=2, label=f'True k2={k2}')
    ax.axvline(k3, color='m', linestyle='--', lw=2, label=f'True k3={k3}')
    ax.set_xlabel('Frequency (k)', fontsize=12)
    ax.set_ylabel('Magnitude (averaged)', fontsize=12)
    ax.set_title('Discovered HF Frequency Spectrum (Averaged)', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 25)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('triple_freq_spectrum.png', dpi=150)
    print("    Saved: triple_freq_spectrum.png")
    plt.close()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
