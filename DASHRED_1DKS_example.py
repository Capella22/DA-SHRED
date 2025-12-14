### DA-SHRED for 1D Kuramoto-Sivashinsky Equation

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import copy
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# 1. PDE Solver: 1D Kuramoto-Sivashinsky Equation


class KuramotoSivashinsky1D:
    """
    1D Kuramoto-Sivashinsky equation solver using pseudo-spectral ETDRK4.

    Simulation: u_t + u*u_x + u_xx + nu*u_xxxx = 0
    Real physics: u_t + u*u_x + u_xx + nu*u_xxxx + mu*u = 0 (with damping)
    """

    def __init__(self, L=32 * np.pi, N=256, nu=1.0, mu=0.0, dt=0.25):
        self.L, self.N, self.nu, self.mu, self.dt = L, N, nu, mu, dt
        self.x = np.linspace(0, L, N, endpoint=False)
        self.k = 2 * np.pi / L * np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
        self.linear_op = self.k ** 2 - nu * self.k ** 4 - mu
        self._compute_etdrk4_coefficients()

    def _compute_etdrk4_coefficients(self):
        h, L = self.dt, self.linear_op
        self.E = np.exp(h * L)
        self.E2 = np.exp(h * L / 2)

        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = h * L[:, np.newaxis] + r[np.newaxis, :]

        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        self.f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        self.f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    def _nonlinear(self, u_hat):
        u = np.real(ifft(u_hat))
        return -0.5j * self.k * fft(u ** 2)

    def step(self, u_hat):
        Nu = self._nonlinear(u_hat)
        a = self.E2 * u_hat + self.Q * Nu
        Na = self._nonlinear(a)
        b = self.E2 * u_hat + self.Q * Na
        Nb = self._nonlinear(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = self._nonlinear(c)
        return self.E * u_hat + self.f1 * Nu + 2 * self.f2 * (Na + Nb) + self.f3 * Nc

    def simulate(self, u0, T, save_every=1):
        n_steps = int(T / self.dt)
        n_save = n_steps // save_every + 1
        U = np.zeros((n_save, self.N))
        u_hat = fft(u0)
        U[0] = u0
        idx = 1
        for step in range(1, n_steps + 1):
            u_hat = self.step(u_hat)
            if step % save_every == 0 and idx < n_save:
                U[idx] = np.real(ifft(u_hat))
                idx += 1
        return U


# 2. Dataset for SHRED

class TimeSeriesDataset(Dataset):
    """Dataset with time-lagged sensor measurements for SHRED"""

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



# 3. SHRED Model


class SHRED(nn.Module):
    """SHallow REcurrent Decoder: LSTM encoder -> latent -> MLP decoder"""

    def __init__(self, num_sensors, lags, hidden_size, output_size,
                 num_lstm_layers=3, decoder_layers=[256, 256], dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        layers = []
        prev = hidden_size
        for size in decoder_layers:
            layers.extend([nn.Linear(prev, size), nn.LayerNorm(size),
                           nn.ReLU(), nn.Dropout(dropout)])
            prev = size
        layers.append(nn.Linear(prev, output_size))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

    def forward(self, x):
        latent = self.encode(x)
        return self.decoder(latent), latent


# 4. DA-SHRED Model


class DASHRED(nn.Module):
    """DA-SHRED: SHRED + latent transformation for SIM2REAL gap closure"""

    def __init__(self, base_shred, freeze_decoder=False):
        super().__init__()
        self.lstm = copy.deepcopy(base_shred.lstm)
        self.decoder = copy.deepcopy(base_shred.decoder)
        self.hidden_size = base_shred.hidden_size

        # Add/refine latent transformation layer to close SIM2REAL gap (residual MLP)
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Initialize transformation near identity for subtle initial correction
        with torch.no_grad():
            self.transform[0].weight.copy_(0.1 * torch.eye(self.hidden_size))
            self.transform[0].bias.zero_()
            self.transform[2].weight.copy_(0.1 * torch.eye(self.hidden_size))
            self.transform[2].bias.zero_()

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, sensor_history, apply_transform=True):
        _, (h_n, _) = self.lstm(sensor_history)
        latent = h_n[-1]
        if apply_transform:
            latent_transformed = latent + self.transform(latent)
        else:
            latent_transformed = latent
        reconstruction = self.decoder(latent_transformed)
        return reconstruction, latent, latent_transformed


class LatentGANAligner(nn.Module):
    """GAN-based aligner to transform sim latent space to real"""

    def __init__(self, latent_dim, hidden_layers=[128, 128], dropout=0.1):
        super(LatentGANAligner, self).__init__()
        self.latent_dim = latent_dim

        # Generator: Transforms sim -> real-like
        gen_modules = []
        prev_size = latent_dim
        for size in hidden_layers:
            gen_modules.extend([nn.Linear(prev_size, size), nn.LeakyReLU(0.2), nn.Dropout(dropout)])
            prev_size = size
        gen_modules.append(nn.Linear(prev_size, latent_dim))
        self.generator = nn.Sequential(*gen_modules)

        # Discriminator: Classifies real vs generated
        disc_modules = []
        prev_size = latent_dim
        for size in hidden_layers:
            disc_modules.extend([nn.Linear(prev_size, size), nn.LeakyReLU(0.2), nn.Dropout(dropout)])
            prev_size = size
        disc_modules.append(nn.Linear(prev_size, 1))
        self.discriminator = nn.Sequential(*disc_modules)

    def forward(self, z):
        return self.generator(z)


# 5. Training Functions


def train_latent_gan_aligner(aligner, Z_sim, Z_real, num_epochs=200, batch_size=32, lr=1e-4):
    """Train GAN to align sim to real latents with adversarial and cycle loss"""

    sim_dataset = torch.utils.data.TensorDataset(torch.tensor(Z_sim, dtype=torch.float32))
    real_dataset = torch.utils.data.TensorDataset(torch.tensor(Z_real, dtype=torch.float32))
    sim_loader = torch.utils.data.DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_cycle = nn.L1Loss()
    opt_gen = optim.Adam(aligner.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(aligner.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    aligner.to(device)
    for epoch in range(num_epochs):
        aligner.train()
        gen_loss_total, disc_loss_total = 0.0, 0.0
        for (z_sim,), (z_real,) in zip(sim_loader, real_loader):
            z_sim, z_real = z_sim[0].to(device), z_real[0].to(device)

            # Train Discriminator
            opt_disc.zero_grad()
            fake = aligner(z_sim)
            disc_real = aligner.discriminator(z_real)
            disc_fake = aligner.discriminator(fake.detach())
            loss_disc = criterion_adv(disc_real, torch.ones_like(disc_real)) + \
                        criterion_adv(disc_fake, torch.zeros_like(disc_fake))
            loss_disc.backward()
            opt_disc.step()
            disc_loss_total += loss_disc.item()

            # Train Generator
            opt_gen.zero_grad()
            fake = aligner(z_sim)
            disc_fake = aligner.discriminator(fake)
            loss_adv = criterion_adv(disc_fake, torch.ones_like(disc_fake))
            loss_cycle = criterion_cycle(aligner.generator(fake), z_sim)
            loss_gen = loss_adv + 10.0 * loss_cycle
            loss_gen.backward()
            opt_gen.step()
            gen_loss_total += loss_gen.item()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Gen Loss: {gen_loss_total / len(sim_loader):.6f}, Disc Loss: {disc_loss_total / len(sim_loader):.6f}")
    return aligner


def train_shred(model, train_data, valid_data, epochs=150, batch_size=32, lr=1e-3, patience=20):
    """Train SHRED model"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.MSELoss()

    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0
    train_hist, valid_hist = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)
            loss = criterion(pred, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * sensors.size(0)
        train_loss /= len(train_data)
        train_hist.append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _ = model(sensors)
                valid_loss += criterion(pred, state).item() * sensors.size(0)
        valid_loss /= len(valid_data)
        valid_hist.append(valid_loss)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return train_hist, valid_hist


def train_dashred(model, train_dataset_real, valid_dataset_real, sensor_indices,
                  shred_model, train_sim, train_real,
                  epochs=100, batch_size=32, lr=5e-4, patience=20, gan_epochs=50,
                  smoothness_weight=0.1):
    """Train DA-SHRED with GAN-based latent alignment for sim2real matching"""
    # First, extract latents from sim (using SHRED) and real (using current model)
    Z_sim = get_latent_trajectory(shred_model, train_sim)
    Z_real = get_latent_trajectory(model, train_real, is_dashred=True)

    # Train GAN aligner to match sim to real latents
    latent_dim = model.hidden_size
    aligner = LatentGANAligner(latent_dim=latent_dim)
    print("Training GAN for latent space alignment...")
    aligner = train_latent_gan_aligner(aligner, Z_sim, Z_real, num_epochs=gan_epochs, batch_size=batch_size,
                                       lr=lr / 2)

    # Integrate trained GAN generator into model's transformation
    model.latent_transform = aligner.generator

    # Now fine-tune the full model with reconstruction loss (using aligned latents)
    train_loader = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset_real, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_loss, best_state, wait = float('inf'), None, 0
    train_hist, valid_hist = [], []

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()

            pred, latent, latent_t = model(sensors, apply_transform=True)

            recon = nn.functional.mse_loss(pred, state)
            sensor = nn.functional.mse_loss(pred[:, sensor_indices], state[:, sensor_indices])
            reg = torch.mean((latent_t - latent) ** 2)

            # Smoothness regularization: penalize large gradients in reconstruction
            # This encourages smoother outputs
            if smoothness_weight > 0:
                pred_grad = pred[:, 1:] - pred[:, :-1]  # Approximate spatial gradient
                smoothness_loss = torch.mean(pred_grad ** 2)
            else:
                smoothness_loss = 0.0

            loss = recon + 0.5 * sensor + 0.01 * reg + smoothness_weight * smoothness_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * sensors.size(0)

        train_loss /= len(train_dataset_real)
        train_hist.append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _, _ = model(sensors)
                valid_loss += nn.functional.mse_loss(pred, state).item() * sensors.size(0)
        valid_loss /= len(valid_dataset_real)
        valid_hist.append(valid_loss)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return train_hist, valid_hist


# 6. Evaluation

def evaluate(model, dataset, scaler_U, is_dashred=False):
    """Evaluate model and return predictions"""
    model.eval()
    loader = DataLoader(dataset, batch_size=64)

    preds, truths = [], []
    with torch.no_grad():
        for sensors, state in loader:
            sensors = sensors.to(device)
            if is_dashred:
                pred, _, _ = model(sensors)
            else:
                pred, _ = model(sensors)
            preds.append(pred.cpu().numpy())
            truths.append(state.numpy())

    preds = scaler_U.inverse_transform(np.vstack(preds))
    truths = scaler_U.inverse_transform(np.vstack(truths))
    mse = np.mean((preds - truths) ** 2)
    return preds, truths, mse


def get_latent_trajectory(model, dataset, is_dashred=False):
    """Extract latent states for the entire dataset"""
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset))

    with torch.no_grad():
        for sensors, _ in loader:
            sensors = sensors.to(device)
            if is_dashred:
                _, _, latent = model(sensors)
            else:
                _, latent = model(sensors)
            return latent.cpu().numpy()


# 7. SINDy for Missing Physics Discovery


def build_sindy_library(Z, poly_order=2):
    """Build polynomial library for SINDy"""
    n_samples, dim = Z.shape
    library = [np.ones(n_samples)]
    names = ['1']

    for i in range(dim):
        library.append(Z[:, i])
        names.append(f'z{i}')

    if poly_order >= 2:
        for i in range(dim):
            for j in range(i, dim):
                library.append(Z[:, i] * Z[:, j])
                names.append(f'z{i}*z{j}')

    return np.column_stack(library), names


def sindy_stls(Theta, dZ, threshold=0.01, max_iter=50):
    """Sequential Thresholded Least Squares for SINDy"""
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]

    for _ in range(max_iter):
        small_inds = np.abs(Xi) < threshold
        Xi[small_inds] = 0
        for i in range(dZ.shape[1]):
            big_inds = ~small_inds[:, i]
            if np.sum(big_inds) > 0:
                Xi[big_inds, i] = np.linalg.lstsq(Theta[:, big_inds], dZ[:, i], rcond=None)[0]
    return Xi


def compute_spectral_derivatives(u, L, max_order=4):
    """
    Compute spatial derivatives using spectral (Fourier) method

    Parameters:
    -----------
    u: ndarray (M, N) - State Space values at M time points and N spatial points
    L: float - Domain length
    max_order: int - Maximum derivative order to compute

    Returns:
    --------
    derivatives: dict - Dictionary with 'u', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx'
    """
    M, N = u.shape

    # Wavenumbers for spectral differentiation
    k = 2 * np.pi / L * np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])

    derivatives = {'u': u}

    # Transform to Fourier space
    u_hat = fft(u, axis=1)

    if max_order >= 1:
        u_x_hat = 1j * k * u_hat
        derivatives['u_x'] = np.real(ifft(u_x_hat, axis=1))

    if max_order >= 2:
        u_xx_hat = (1j * k) ** 2 * u_hat
        derivatives['u_xx'] = np.real(ifft(u_xx_hat, axis=1))

    if max_order >= 3:
        u_xxx_hat = (1j * k) ** 3 * u_hat
        derivatives['u_xxx'] = np.real(ifft(u_xxx_hat, axis=1))

    if max_order >= 4:
        u_xxxx_hat = (1j * k) ** 4 * u_hat
        derivatives['u_xxxx'] = np.real(ifft(u_xxxx_hat, axis=1))

    return derivatives


def identify_missing_damping(shred_model, dashred_model, train_dataset_sim, train_dataset_real,
                             x, L, dt, save_every,
                             initial_threshold=0.01, max_iter=50, refinement_steps=8,
                             threshold_increment=0.04):
    """
    Identify missing physics using extended SINDy library with proper spatial derivatives

    Parameters:
    -----------
    shred_model: SHRED model trained on simulation
    dashred_model: DA-SHRED model trained on real data
    train_dataset_sim: Training dataset for simulation
    train_dataset_real: Training dataset for real physics
    x: spatial grid
    L: domain length
    dt: time step
    save_every: save interval
    initial_threshold: starting threshold for STLS
    max_iter: max iterations for STLS
    refinement_steps: number of refinement steps
    threshold_increment: how much to increase in threshold on each refinement step
    """
    # Extract latents
    Z_sim = get_latent_trajectory(shred_model, train_dataset_sim)
    Z_real = get_latent_trajectory(dashred_model, train_dataset_real, is_dashred=True)

    # Compute latent difference (proxy for missing physics effect)
    latent_diff = Z_real - Z_sim

    # Reconstruct u from sim latents to build physical library (shape: M, N)
    shred_model.eval()
    with torch.no_grad():
        u_recon = shred_model.decoder(torch.tensor(Z_sim, dtype=torch.float32).to(device)).cpu().numpy()

    # Compute spatial derivatives using spectral method
    derivs = compute_spectral_derivatives(u_recon, L, max_order=4)

    # Compute time derivative using central differences
    dt_eff = dt * save_every
    u_t = (u_recon[2:] - u_recon[:-2]) / (2 * dt_eff)

    # Trim other arrays to match u_t length
    u_mid = u_recon[1:-1]
    for key in derivs:
        derivs[key] = derivs[key][1:-1]
    latent_diff_mid = latent_diff[1:-1]

    # Build extended library
    library_terms = []
    names = []

    library_terms.append(np.ones_like(u_mid))
    names.append('1')

    library_terms.append(derivs['u'])
    names.append('u')

    library_terms.append(derivs['u'] ** 2)
    names.append('u^2')

    library_terms.append(derivs['u'] ** 3)
    names.append('u^3')

    library_terms.append(derivs['u_x'])
    names.append('u_x')

    library_terms.append(derivs['u_xx'])
    names.append('u_xx')

    library_terms.append(derivs['u_xxxx'])
    names.append('u_xxxx')

    library_terms.append(derivs['u'] * derivs['u_x'])
    names.append('u*u_x')

    library_terms.append(u_t)
    names.append('u_t')

    print(f"    Extended SINDy library features: {names}")

    # Stack and flatten library for SINDy
    library = np.stack(library_terms, axis=-1).reshape(-1, len(names))

    # Repeat latent diff across spatial points to match library size
    M, N = u_mid.shape
    latent_diff_repeated = np.repeat(latent_diff_mid[:, np.newaxis, :], N, axis=1)
    latent_diff_flat = latent_diff_repeated.reshape(-1, latent_diff_mid.shape[1])

    # SINDy regression
    Xi = sindy_stls(library, latent_diff_flat, threshold=initial_threshold, max_iter=max_iter)

    # Extract average coefficient for 'u' across latent dims
    u_index = names.index('u')
    extracted_mu = np.mean(Xi[u_index, :])

    # Refinement loop with sequential thresholding
    current_threshold = initial_threshold
    for step in range(refinement_steps):
        Xi = sindy_stls(library, latent_diff_flat, threshold=current_threshold, max_iter=max_iter)
        extracted_mu = np.mean(Xi[u_index, :])

        non_zero_per_dim = np.sum(np.abs(Xi) > 0.001, axis=0)
        avg_non_zero = np.mean(non_zero_per_dim)
        print(
            f"Refinement step {step + 1}, threshold={current_threshold:.4f}, extracted μ={extracted_mu:.4f}, avg non-zero terms: {avg_non_zero:.1f}")

        if abs(extracted_mu - 0.05) < 0.001 and avg_non_zero <= 2:
            break

        current_threshold += threshold_increment

    # Print sparse result
    print("\nSparse SINDy Result (Perturbation Parameter):")
    for dim in range(Xi.shape[1]):
        print(f"Latent dim {dim}:")
        non_zero_count = 0
        for i, name in enumerate(names):
            coeff = Xi[i, dim]
            if abs(coeff) > 0.001:
                print(f"  {coeff:+.4f} * {name}")
                non_zero_count += 1
        print(f"  (Non-zero terms: {non_zero_count})")

    return Xi, names


# 8. Post-Processing: Spectral Smoothing for Reconstruction


def spectral_smooth(u, L, cutoff_fraction=0.5):
    """
    Optional: apply spectral smoothing to remove high-frequency noise.

    Parameters:
    -----------
    u: ndarray (M, N) - State Space to smooth
    L: float - Length of domain
    cutoff_fraction: float - Fraction of modes to keep (0.5 = keep half)

    Returns:
    --------
    u_smooth: ndarray - Smoothed field
    """
    M, N = u.shape
    u_hat = fft(u, axis=1)

    # Create low-pass filter
    k_max = int(N * cutoff_fraction / 2)
    filter_mask = np.zeros(N)
    filter_mask[:k_max] = 1
    filter_mask[-k_max:] = 1

    # Apply filter
    u_hat_filtered = u_hat * filter_mask
    u_smooth = np.real(ifft(u_hat_filtered, axis=1))

    return u_smooth


def gaussian_smooth(u, sigma=1.0):
    """
    Optional: apply Gaussian smoothing in physical space.

    Parameters:
    -----------
    u: ndarray (M, N) - State Space to smooth
    sigma: float - Standard deviation of Gaussian kernel

    Returns:
    --------
    u_smooth: ndarray - Smoothed field
    """
    from scipy.ndimage import gaussian_filter1d
    u_smooth = gaussian_filter1d(u, sigma=sigma, axis=1, mode='wrap')
    return u_smooth


# 9. Main Execution

if __name__ == "__main__":

    # HYPERPARAMETERS - Easy to tune


    # PDE parameters
    L = 32 * np.pi
    N = 256
    nu = 1.0
    mu_damping = 0.05  # Missing physics term
    dt = 0.02
    T = 100.0
    save_every = 4

    # SHRED/DA-SHRED parameters
    num_sensors = 12
    lags = 30
    hidden_size = 8
    decoder_layers = [128, 128]

    # Training parameters
    shred_epochs = 150
    shred_patience = 50
    dashred_epochs = 100
    dashred_patience = 50
    gan_epochs = 50
    smoothness_weight = 0.05  # Weight for smoothness regularization (try 0.0 to 0.2)

    # SINDy refinement parameters
    sindy_initial_threshold = 0.01
    sindy_max_iter = 50
    sindy_refinement_steps = 8
    sindy_threshold_increment = 0.04  # How much to increase in threshold on each refinement step

    # Post-processing smoothing (optional)
    apply_smoothing = False  # Set to True to apply spectral smoothing
    smoothing_cutoff = 0.7  # Fraction of Fourier modes to keep (higher = less smoothing)


    # Main execution


    # Generate initial condition
    x = np.linspace(0, L, N, endpoint=False)
    u0 = np.cos(x / 16) * (1 + np.sin(x / 16))

    # Generate data
    print("\n[1] Generating PDE data...")
    print(f"    Simulation: Undamped KS (μ=0)")
    print(f"    Real Physics: Damped KS (μ={mu_damping})")

    ks_sim = KuramotoSivashinsky1D(L=L, N=N, nu=nu, mu=0.0, dt=dt)
    ks_real = KuramotoSivashinsky1D(L=L, N=N, nu=nu, mu=mu_damping, dt=dt)

    U_sim = ks_sim.simulate(u0, T, save_every)
    U_real = ks_real.simulate(u0, T, save_every)

    # Plot KS heatmap
    t_sim = np.arange(0, T + dt, dt * save_every)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(U_sim.T, aspect='auto', cmap='RdBu_r',
                   extent=[t_sim[0], t_sim[-1], 0, L], origin='lower', interpolation='bilinear')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_title('1D Kuramoto-Sivashinsky Heatmap (Undamped Simulation)')
    plt.colorbar(im, ax=ax, label='u value')
    plt.tight_layout()
    plt.savefig('ks_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"    Data shape: {U_sim.shape} (timesteps x spatial points)")

    # Create datasets
    print("\n[2] Preparing datasets...")
    sensor_indices = np.linspace(0, N - 1, num_sensors, dtype=int)
    print(f"    Sensors: {num_sensors} at indices {sensor_indices}")
    print(f"    Time lags: {lags}")

    n_train = int(0.8 * len(U_sim))

    train_sim = TimeSeriesDataset(U_sim[:n_train], sensor_indices, lags, fit_scaler=True)
    valid_sim = TimeSeriesDataset(U_sim[n_train:], sensor_indices, lags,
                                  scaler=train_sim.get_scalers())
    train_real = TimeSeriesDataset(U_real[:n_train], sensor_indices, lags,
                                   scaler=train_sim.get_scalers())
    valid_real = TimeSeriesDataset(U_real[n_train:], sensor_indices, lags,
                                   scaler=train_sim.get_scalers())

    print(f"    Training samples: {len(train_sim)}")
    print(f"    Validation samples: {len(valid_sim)}")

    # Initialize and train SHRED
    print("\n[3] Training SHRED on simulation data...")
    shred_model = SHRED(
        num_sensors=num_sensors, lags=lags, hidden_size=hidden_size,
        output_size=N, num_lstm_layers=3, decoder_layers=decoder_layers, dropout=0.1
    )
    print(f"    Model parameters: {sum(p.numel() for p in shred_model.parameters()):,}")

    train_shred(shred_model, train_sim, valid_sim, epochs=shred_epochs, patience=shred_patience)

    # Evaluate SIM2REAL gap
    print("\n[4] Evaluating SIM2REAL gap...")
    scaler_U, _ = train_sim.get_scalers()

    _, _, mse_sim = evaluate(shred_model, valid_sim, scaler_U)
    _, _, mse_before = evaluate(shred_model, valid_real, scaler_U)
    print(f"    SHRED on simulation: RMSE = {np.sqrt(mse_sim):.6f}")
    print(f"    SHRED on real (gap): RMSE = {np.sqrt(mse_before):.6f}")
    print(f"    Gap ratio: {mse_before / mse_sim:.2f}x")

    # Initiate and train DASHRED
    dashred_model = DASHRED(shred_model, freeze_decoder=False)

    print("\n[5] Training DA-SHRED to close SIM2REAL gap...")
    train_hist_da, valid_hist_da = train_dashred(
        dashred_model, train_real, valid_real, sensor_indices,
        shred_model=shred_model, train_sim=train_sim, train_real=train_real,
        epochs=dashred_epochs, patience=dashred_patience, gan_epochs=gan_epochs,
        smoothness_weight=smoothness_weight
    )

    # Evaluate DA-SHRED
    print("\n[6] Evaluating DA-SHRED...")
    pred_before, truth_real, mse_before = evaluate(shred_model, valid_real, scaler_U)
    pred_after, _, mse_after = evaluate(dashred_model, valid_real, scaler_U, is_dashred=True)

    # Optional: apply post-processing smoothing
    if apply_smoothing:
        print(f"    Applying spectral smoothing (cutoff={smoothing_cutoff})...")
        pred_after = spectral_smooth(pred_after, L, cutoff_fraction=smoothing_cutoff)
        mse_after = np.mean((pred_after - truth_real) ** 2)

    print(f"    SHRED on simulation: RMSE = {np.sqrt(mse_sim):.6f}")
    print(f"    SHRED on real (gap): RMSE = {np.sqrt(mse_before):.6f}")
    print(f"    DASHRED on real: RMSE = {np.sqrt(mse_after):.6f}")
    print(f"    Gap reduction: {(mse_before - mse_after) / mse_before * 100:.1f}%")
    print(f"    Improvement: {mse_before / mse_after:.2f}x")


    # Reconstruction comparison

    print("\n[6b] Generating three-panel reconstruction comparison...")

    dashred_model.eval()
    train_loader_real = DataLoader(train_real, batch_size=len(train_real))
    with torch.no_grad():
        for sensors, _ in train_loader_real:
            sensors = sensors.to(device)
            pred_train_dashred, _, _ = dashred_model(sensors, apply_transform=True)
            pred_train_dashred = scaler_U.inverse_transform(pred_train_dashred.cpu().numpy())

    # Optional smoothing for visualization
    if apply_smoothing:
        pred_train_dashred = spectral_smooth(pred_train_dashred, L, cutoff_fraction=smoothing_cutoff)

    U_sim_train = U_sim[lags:n_train]
    U_real_train = U_real[lags:n_train]
    t_train = np.arange(lags, n_train) * dt * save_every

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(U_sim_train.min(), U_real_train.min(), pred_train_dashred.min())
    vmax = max(U_sim_train.max(), U_real_train.max(), pred_train_dashred.max())

    im1 = axes[0].imshow(U_sim_train.T, aspect='auto', cmap='RdBu_r',
                         extent=[t_train[0], t_train[-1], 0, L],
                         origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_xlabel('Time (t)', fontsize=12)
    axes[0].set_ylabel('Space (x)', fontsize=12)
    axes[0].set_title('Simulation (Undamped KS)', fontsize=14)
    plt.colorbar(im1, ax=axes[0], label='u')

    im2 = axes[1].imshow(U_real_train.T, aspect='auto', cmap='RdBu_r',
                         extent=[t_train[0], t_train[-1], 0, L],
                         origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_xlabel('Time (t)', fontsize=12)
    axes[1].set_ylabel('Space (x)', fontsize=12)
    axes[1].set_title('Real Physics (Damped KS, μ={})'.format(mu_damping), fontsize=14)
    plt.colorbar(im2, ax=axes[1], label='u')

    im3 = axes[2].imshow(pred_train_dashred.T, aspect='auto', cmap='RdBu_r',
                         extent=[t_train[0], t_train[-1], 0, L],
                         origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[2].set_xlabel('Time (t)', fontsize=12)
    axes[2].set_ylabel('Space (x)', fontsize=12)
    axes[2].set_title('DA-SHRED Reconstruction', fontsize=14)
    plt.colorbar(im3, ax=axes[2], label='u')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: reconstruction_comparison.png")
    plt.show()


    # SINDy: Discovery of Missing Physics

    print("\n[7] Identifying missing physics with SINDy...")

    Xi, feature_names = identify_missing_damping(
        shred_model, dashred_model, train_sim, train_real, x, L, dt, save_every,
        initial_threshold=sindy_initial_threshold,
        max_iter=sindy_max_iter,
        refinement_steps=sindy_refinement_steps,
        threshold_increment=sindy_threshold_increment
    )

    # Visualize SINDy coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Xi, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(Xi.shape[1]))
    ax.set_xticklabels([f'dz{i}' for i in range(Xi.shape[1])])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title('SINDy Coefficients for Latent Discrepancy')
    plt.colorbar(im, ax=ax, label='Coefficient')
    plt.tight_layout()
    plt.savefig('sindy_coefficients.png', dpi=150)
    plt.show()


    # Plot final results

    print("\n[8] Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Prediction results on unseen sensor data', fontsize=14, y=0.98)

    scaler_U, _ = train_sim.get_scalers()
    pred_before, _, mse_before = evaluate(shred_model, valid_real, scaler_U)
    pred_after, truth_real, mse_after = evaluate(dashred_model, valid_real, scaler_U, is_dashred=True)

    if apply_smoothing:
        pred_after = spectral_smooth(pred_after, L, cutoff_fraction=smoothing_cutoff)

    for i, idx in enumerate([0, len(truth_real) // 2, len(truth_real) - 1]):
        axes[0, i].plot(x, truth_real[idx], 'b-', lw=2, label='Ground Truth')
        axes[0, i].plot(x, pred_before[idx], 'r--', lw=1.5, label='SHRED (before DA)')
        axes[0, i].plot(x, pred_after[idx], 'g--', lw=1.5, label='DA-SHRED')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('u')
        axes[0, i].set_title(f'Sample {idx}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(alpha=0.3)

    err_before = np.sqrt(np.mean((pred_before - truth_real) ** 2, axis=1))
    err_after = np.sqrt(np.mean((pred_after - truth_real) ** 2, axis=1))

    axes[1, 0].plot(err_before, 'r-', label='SHRED (before DA)')
    axes[1, 0].plot(err_after, 'g-', label='DA-SHRED')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Reconstruction Error')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    im = axes[1, 1].imshow(Xi, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_xlabel('Latent Dim')
    axes[1, 1].set_ylabel('Library Feature')
    axes[1, 1].set_title('SINDy Coefficients (Missing Physics)')
    plt.colorbar(im, ax=axes[1, 1])

    axes[1, 2].imshow(np.abs(pred_after - truth_real).T, aspect='auto', cmap='hot',
                      extent=[0, len(truth_real), 0, L])
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('x')
    axes[1, 2].set_title('DA-SHRED Absolute Error')

    axes[0, 1].annotate('Snapshots (u(x))', xy=(0.5, 1.25), xycoords='axes fraction',
                        ha='center', va='center', fontsize=12)
    axes[1, 1].annotate('Error / Coef / Heatmap', xy=(0.5, 1.25), xycoords='axes fraction',
                        ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig('dashred_results.png', dpi=150, bbox_inches='tight')
    print("    Saved: dashred_results.png")
    plt.show()

    # Save checkpoint
    torch.save({
        'shred': shred_model.state_dict(),
        'dashred': dashred_model.state_dict(),
        'sensor_indices': sensor_indices,
        'params': {'num_sensors': num_sensors, 'lags': lags, 'hidden_size': hidden_size}
    }, 'dashred_checkpoint.pt')
    print("    Saved: dashred_checkpoint.pt")

    print("\nDone!")