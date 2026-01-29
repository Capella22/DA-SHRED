"""
Two-Frequency DA-SHRED with Multiscale HF Learning

Key insight:
- We can only supervise HF at sensor locations (don't have full U_real)
- Without regularization, HF decoder could learn arbitrary patterns
- SPARSITY IN FREQUENCY DOMAIN encourages finding specific discrete modes
- Real physics often has sparse frequency content (specific modes, not noise)

Loss = sensor_residual_loss + λ_sparse * frequency_sparsity_loss

The frequency sparsity loss penalizes the number of active frequencies,
encouraging the model to explain sensor residuals with minimal frequency content.
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
# 1. Data Generation
# =============================================================================

def generate_two_frequency_data(L=2*np.pi, N=128, T=10.0, dt=0.05,
                                 k_low=2, k_high=15, omega_low=1, omega_high=8,
                                 A_low=1.0, A_high=0.3, include_hf=True):
    x = np.linspace(0, L, N, endpoint=False)
    t = np.arange(0, T, dt)
    n_steps = len(t)

    U = np.zeros((n_steps, N))
    for i, ti in enumerate(t):
        u_lf = A_low * np.sin(k_low * x) * np.cos(omega_low * ti)
        if include_hf:
            u_hf = A_high * np.sin(k_high * x) * np.cos(omega_high * ti)
            U[i] = u_lf + u_hf
        else:
            U[i] = u_lf
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
    """
    L1 sparsity on frequency magnitudes.
    Encourages few active frequencies.

    Args:
        signal: (batch, N) spatial signal
    Returns:
        scalar loss
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    return torch.mean(torch.sum(magnitudes, dim=-1))


def frequency_sparsity_l05(signal):
    """
    L0.5 pseudo-norm on frequency magnitudes.
    More aggressive sparsity than L1.
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)
    return torch.mean(torch.sum(torch.sqrt(magnitudes + 1e-8), dim=-1))


def frequency_sparsity_entropy(signal):
    """
    Entropy-based sparsity: penalizes spread of energy across frequencies.
    Minimum when energy is concentrated in few frequencies.
    Does NOT penalize total magnitude - only how spread out it is.
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    power = torch.abs(fft_coeffs) ** 2
    # Normalize to get probability distribution
    power_sum = torch.sum(power, dim=-1, keepdim=True) + 1e-8
    p = power / power_sum
    # Entropy (lower = more concentrated)
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    return torch.mean(entropy)


def frequency_sparsity_gini(signal):
    """
    Gini coefficient for frequency sparsity.
    Gini = 0 means all frequencies equal, Gini = 1 means one frequency dominates.
    We return (1 - Gini) so that lower = sparser.
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    # Sort magnitudes
    sorted_mag, _ = torch.sort(magnitudes, dim=-1)
    n = magnitudes.shape[-1]

    # Gini coefficient calculation
    cumsum = torch.cumsum(sorted_mag, dim=-1)
    total = cumsum[:, -1:] + 1e-8

    # Gini = 1 - 2 * (area under Lorenz curve)
    indices = torch.arange(1, n + 1, device=signal.device).float()
    gini = 1 - 2 * torch.sum(cumsum / total, dim=-1) / n + 1 / n

    # Return (1 - gini) so lower loss = sparser
    return torch.mean(1 - gini)


def frequency_sparsity_topk(signal, k=3):
    """
    Penalize energy outside top-k frequencies.
    Directly encourages using only k frequencies.
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    # Get top-k magnitudes
    topk_vals, _ = torch.topk(magnitudes, k, dim=-1)
    topk_energy = torch.sum(topk_vals ** 2, dim=-1)
    total_energy = torch.sum(magnitudes ** 2, dim=-1) + 1e-8

    # Penalize energy outside top-k
    outside_topk_ratio = 1 - topk_energy / total_energy
    return torch.mean(outside_topk_ratio)


def frequency_sparsity_normalized_l1(signal):
    """
    L1 sparsity normalized by total energy.
    This penalizes spread without penalizing overall magnitude.

    For a signal with energy E spread across k frequencies equally:
        normalized_l1 = k * sqrt(E/k) / E = sqrt(k/E)
    For energy concentrated in 1 frequency:
        normalized_l1 = sqrt(E) / E = 1/sqrt(E)

    Lower = sparser (fewer frequencies)
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    l1_norm = torch.sum(magnitudes, dim=-1)
    l2_norm = torch.sqrt(torch.sum(magnitudes ** 2, dim=-1) + 1e-8)

    # Ratio of L1 to L2 norm - higher means more spread out
    # For k equal components: L1/L2 = sqrt(k)
    # For 1 component: L1/L2 = 1
    sparsity_ratio = l1_norm / (l2_norm + 1e-8)

    return torch.mean(sparsity_ratio)


def frequency_sparsity_bandlimited(signal, max_freq):
    """
    Sparsity ONLY within resolvable frequency band.
    Heavily penalizes energy outside the band (unresolvable by sensors).

    Args:
        signal: (batch, N) spatial signal
        max_freq: maximum target frequency (default: num_sensors // 2)
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    # Split into resolvable and unresolvable bands
    in_band = magnitudes[:, :max_freq+1]  # k = 0 to max_freq
    out_of_band = magnitudes[:, max_freq+1:]  # k > max_freq (aliased garbage)

    # Sparsity on resolvable band only (normalized L1)
    l1 = torch.sum(in_band, dim=-1)
    l2 = torch.sqrt(torch.sum(in_band ** 2, dim=-1) + 1e-8)
    sparsity_loss = l1 / (l2 + 1e-8)

    # HEAVY penalty for ANY energy outside resolvable band
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
    """HF-SHRED with frequency-aware decoder"""
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
    """DA-SHRED with Multiscale HF learning"""

    def __init__(self, lf_shred, num_sensors, lags, hidden_size, output_size, sensor_indices):
        super().__init__()

        # LF pathway
        self.lf_lstm = copy.deepcopy(lf_shred.lstm)
        self.lf_norm = copy.deepcopy(lf_shred.norm)
        self.lf_decoder = copy.deepcopy(lf_shred.decoder)
        self.gan = LatentGAN(lf_shred.hidden_size)

        # HF pathway
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
        # LF pathway
        z_lf = self.encode_lf(sensor_history)
        if use_gan:
            z_lf = self.gan(z_lf)
        u_lf = self.decode_lf(z_lf)

        # Compute sensor residual at EACH lag timestep (not just last)
        # This gives HF_LSTM proper temporal information
        batch_size = sensor_history.shape[0]
        residual_history = torch.zeros_like(sensor_history)

        sensors_lf_pred = u_lf[:, self.sensor_indices]  # LF prediction at sensors

        for lag in range(self.lags):
            # Approximate: assume LF prediction is similar across recent lags
            # (More accurate would be to run LF_LSTM at each lag, but expensive)
            residual_history[:, lag, :] = sensor_history[:, lag, :] - sensors_lf_pred.detach()

        # HF pathway on residual history
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

            # Discriminator
            opt_d.zero_grad()
            z_fake = dashred.gan(z_s)
            d_loss = F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_r), torch.ones(len(z_r), 1, device=device)
            ) + F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_fake.detach()), torch.zeros(len(z_s), 1, device=device)
            )
            d_loss.backward()
            opt_d.step()

            # Generator
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
                    lambda_sparse=0.1, sparsity_type='bandlimited'):
    """
    Train HF-SHRED with SENSOR-ONLY supervision + frequency sparsity.

    Key: We add a MAGNITUDE CONSTRAINT to prevent HF from exploding.
    The normalized_l1 sparsity doesn't penalize magnitude, so we need this.
    """
    print(f"\n=== Stage 3: Train HF-SHRED (Sensor-Only + {sparsity_type} Sparsity) ===")
    print(f"    lambda_sparse = {lambda_sparse}")

    params = list(dashred.hf_shred.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)

    # Select sparsity function
    # For bandlimited, we need to know max resolvable frequency
    max_target_freq = len(sensor_indices) // 2
    print(f"    Max target frequency: k <= {max_target_freq}")

    sparsity_fn = {
        'l1': frequency_sparsity_l1,
        'l05': frequency_sparsity_l05,
        'entropy': frequency_sparsity_entropy,
        'gini': frequency_sparsity_gini,
        'topk': lambda x: frequency_sparsity_topk(x, k=3),
        'normalized_l1': frequency_sparsity_normalized_l1,
        'bandlimited': lambda x: frequency_sparsity_bandlimited(x, max_target_freq),
    }[sparsity_type]

    history = {'sensor_loss': [], 'sparsity_loss': [], 'total_loss': []}

    # Estimate target residual scale from first batch
    with torch.no_grad():
        sample_sensors, _ = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        _, u_lf, _, _, _ = dashred(sample_sensors, use_gan=True)
        sensors_current = sample_sensors[:, -1, :]
        sensors_lf = u_lf[:, sensor_idx]
        residual_scale = (sensors_current - sensors_lf).abs().mean().item()
    print(f"    Estimated residual scale: {residual_scale:.4f}")
    max_hf_magnitude = residual_scale * 3  # Allow 3x headroom

    warmup_epochs = 100

    for epoch in range(epochs):
        dashred.train()
        epoch_sensor_loss = 0
        epoch_sparsity_loss = 0
        epoch_mag_loss = 0

        # Gradually increase sparsity after warmup
        if epoch < warmup_epochs:
            current_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            current_lambda = lambda_sparse * min(1.0, progress * 2)

        for sensors, targets in train_loader_real:
            sensors = sensors.to(device)

            optimizer.zero_grad()

            u_total, u_lf, u_hf, _, _ = dashred(sensors, use_gan=True)

            # Loss 1: Sensor residual matching
            sensors_current = sensors[:, -1, :]
            sensors_lf = u_lf[:, sensor_idx].detach()
            sensors_residual_true = sensors_current - sensors_lf
            sensors_hf = u_hf[:, sensor_idx]

            sensor_loss = F.mse_loss(sensors_hf, sensors_residual_true)

            # Loss 2: Frequency sparsity
            sparsity_loss = sparsity_fn(u_hf)

            # Loss 3: MAGNITUDE CONSTRAINT - prevent HF from exploding
            hf_magnitude = u_hf.abs().mean()
            magnitude_penalty = F.relu(hf_magnitude - max_hf_magnitude) ** 2

            # Total loss
            loss = sensor_loss + current_lambda * sparsity_loss + 1.0 * magnitude_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_sensor_loss += sensor_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_mag_loss += magnitude_penalty.item()

        n_batches = len(train_loader_real)
        epoch_sensor_loss /= n_batches
        epoch_sparsity_loss /= n_batches
        epoch_mag_loss /= n_batches

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
                top_freqs = np.argsort(fft_mag)[-3:][::-1]
                hf_max = u_hf_sample.abs().max().item()

            print(f"  Epoch {epoch+1}/{epochs}, Sensor: {epoch_sensor_loss:.6f}, "
                  f"Sparsity: {epoch_sparsity_loss:.4f}, Mag_pen: {epoch_mag_loss:.4f}, "
                  f"HF_max: {hf_max:.4f}, Top freqs: {top_freqs}")

    return dashred, history


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    # Parameters
    L = 2 * np.pi
    N = 128
    T = 10.0
    dt = 0.05

    k_low, k_high = 2, 7  # True frequencies (k_high should be within target range)
    omega_low, omega_high = 1, 5
    A_low, A_high = 1.0, 0.3

    num_sensors = 32  # max_k = 16, so k=7 is within range
    lags = 20
    hidden_size = 32

    # Sparsity settings
    lambda_sparse = 0.1  # Sparsity weight
    sparsity_type = 'bandlimited'  # KEY: Only allow frequencies resolvable by sensors

    # Generate data
    print("[1] Generating data...")
    U_sim, x, t = generate_two_frequency_data(
        L, N, T, dt, k_low, k_high, omega_low, omega_high,
        A_low, A_high, include_hf=False
    )
    U_real, _, _ = generate_two_frequency_data(
        L, N, T, dt, k_low, k_high, omega_low, omega_high,
        A_low, A_high, include_hf=True
    )

    print(f"    U_sim: {U_sim.shape}, True LF mode: k={k_low}")
    print(f"    U_real: {U_real.shape}, True HF mode: k={k_high}")

    # Create datasets
    print("\n[2] Creating datasets...")
    sensor_indices = np.linspace(0, N-1, num_sensors, dtype=int)
    print(f"    Sensors at indices: {sensor_indices}")
    print(f"    Max target frequency: k <= {num_sensors//2}")

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

    # DIAGNOSTIC: Check sensor residual magnitude
    print("\n[6.5] DIAGNOSTIC: Checking sensor residual...")
    dashred.eval()
    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        sample_sensors, sample_targets = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        sample_targets = sample_targets.to(device)

        # Get LF prediction
        z_lf = dashred.encode_lf(sample_sensors)
        z_lf_aligned = dashred.gan(z_lf)
        u_lf = dashred.decode_lf(z_lf_aligned)

        # Sensor values
        sensors_current = sample_sensors[:, -1, :]  # Current sensor reading
        sensors_lf_pred = u_lf[:, sensor_idx]  # LF prediction at sensors
        sensors_residual = sensors_current - sensors_lf_pred

        # Full state residual (for comparison - we have it in this demo)
        full_residual = sample_targets - u_lf
        hf_at_sensors = full_residual[:, sensor_idx]

        print(f"    Sensors_current range: [{sensors_current.min():.4f}, {sensors_current.max():.4f}]")
        print(f"    Sensors_lf_pred range: [{sensors_lf_pred.min():.4f}, {sensors_lf_pred.max():.4f}]")
        print(f"    Sensor residual (current - lf_pred) range: [{sensors_residual.min():.4f}, {sensors_residual.max():.4f}]")
        print(f"    Sensor residual mean abs: {sensors_residual.abs().mean():.6f}")
        print(f"    True HF at sensors (target - lf) range: [{hf_at_sensors.min():.4f}, {hf_at_sensors.max():.4f}]")
        print(f"    True HF at sensors mean abs: {hf_at_sensors.abs().mean():.6f}")
        print(f"    Full residual (all x) mean abs: {full_residual.abs().mean():.6f}")

    # Stage 3: Train HF with sparsity
    print(f"\n[7] Stage 3: Train HF-SHRED with {sparsity_type} sparsity...")
    dashred, history = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=500, lambda_sparse=lambda_sparse, sparsity_type=sparsity_type
    )

    # Stage 4: Fine-tune with reduced sparsity to recover amplitude
    print("\n[7.5] Stage 4: Fine-tuning with reduced sparsity...")
    dashred, history2 = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=200, lambda_sparse=lambda_sparse * 0.1, sparsity_type=sparsity_type
    )

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

    print(f"\n=== Results ===")
    print(f"  True HF frequency: k = {k_high}")
    print(f"  LF-only RMSE: {np.sqrt(mse_lf):.6f}")
    print(f"  LF+HF RMSE: {np.sqrt(mse_total):.6f}")
    print(f"  Improvement: {(mse_lf - mse_total) / mse_lf * 100:.1f}%")

    # Analyze discovered frequencies
    print("\n[9] Analyzing discovered frequencies...")
    dashred.eval()
    with torch.no_grad():
        sample = next(iter(valid_loader_real))[0][:1].to(device)
        _, _, u_hf, _, _ = dashred(sample, use_gan=True)
        u_hf_np = u_hf.cpu().numpy().squeeze()

        fft_mag = np.abs(np.fft.rfft(u_hf_np))
        freqs = np.arange(len(fft_mag))

        top_3 = np.argsort(fft_mag)[-3:][::-1]
        print(f"  Top 3 frequencies in HF output: {top_3}")
        print(f"  Their magnitudes: {fft_mag[top_3]}")
        print(f"  True HF frequency: k = {k_high}")

    # Plot
    print("\n[10] Plotting...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    idx = len(results['targets']) // 2

    # Row 1: Spatial reconstructions
    axes[0, 0].plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    axes[0, 0].plot(x, results['lf_only'][idx], 'r--', lw=1.5, label='LF only')
    axes[0, 0].set_title('LF Reconstruction')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, results['targets'][idx], 'b-', lw=2, label='Ground Truth')
    axes[0, 1].plot(x, results['lf_hf'][idx], 'g--', lw=1.5, label='LF + HF')
    axes[0, 1].set_title('Full Reconstruction (LF + HF)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # HF comparison
    hf_true = results['targets'][idx] - results['lf_only'][idx]
    hf_pred = results['lf_hf'][idx] - results['lf_only'][idx]
    axes[0, 2].plot(x, hf_true, 'b-', lw=2, label='True HF')
    axes[0, 2].plot(x, hf_pred, 'r--', lw=1.5, label='Predicted HF')
    axes[0, 2].set_title('High Frequency Component')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Frequency spectrum of HF
    axes[0, 3].stem(freqs, fft_mag, basefmt=' ')
    axes[0, 3].axvline(k_high, color='g', linestyle='--', label=f'True k={k_high}')
    axes[0, 3].set_xlabel('Frequency (k)')
    axes[0, 3].set_ylabel('Magnitude')
    axes[0, 3].set_title('HF Frequency Spectrum')
    axes[0, 3].legend()
    axes[0, 3].set_xlim(0, 30)
    axes[0, 3].grid(alpha=0.3)

    # Row 2: Training history and heatmaps
    axes[1, 0].plot(history['sensor_loss'], label='Sensor Loss')
    axes[1, 0].plot(history['sparsity_loss'], label='Sparsity Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training History')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    vmin, vmax = results['targets'].min(), results['targets'].max()

    im1 = axes[1, 1].imshow(results['targets'].T, aspect='auto', cmap='RdBu_r',
                            origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1, 1])

    im2 = axes[1, 2].imshow(results['lf_hf'].T, aspect='auto', cmap='RdBu_r',
                            origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Reconstruction (LF + HF)')
    axes[1, 2].set_xlabel('Time')
    plt.colorbar(im2, ax=axes[1, 2])

    error = np.abs(results['lf_hf'] - results['targets'])
    im3 = axes[1, 3].imshow(error.T, aspect='auto', cmap='hot', origin='lower')
    axes[1, 3].set_title('Absolute Error')
    axes[1, 3].set_xlabel('Time')
    plt.colorbar(im3, ax=axes[1, 3])

    plt.suptitle(f'Multiscale HF Learning (Sensor-Only Supervision)\n'
                 f'True HF: k={k_high} | Sparsity: {sparsity_type}, λ={lambda_sparse}', fontsize=14)
    plt.tight_layout()
    plt.savefig('Multiscale_dashred_demo.png', dpi=150, bbox_inches='tight')
    print("    Saved: Multiscale_dashred_demo.png")
    plt.show()

    print("\nDone!")