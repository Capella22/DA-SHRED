### DA-SHRED for 2D Kuramoto-Sivashinsky Equation

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft2, ifft2
from sklearn.preprocessing import StandardScaler
import copy
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)



# 1. PDE Solver: 2D Kuramoto-Sivashinsky Equation


class KuramotoSivashinsky2D:
    """
    2D Kuramoto-Sivashinsky equation using pseudo-spectral method with ETDRK4.

    Equation (conservative form):
        u_t + (1/2)(u_x^2 + u_y^2) + ∇²u + ∇⁴u = -μ*u - ν_extra*∇²u

    Damping options:
        - Linear damping: μ > 0 adds -μ*u term
        - Diffusive damping: nu_extra > 0 adds extra diffusion -ν_extra*∇²u

    Linear part: L = -k² - k⁴ - μ - ν_extra*k²
    Nonlinear part: N(u) = -(1/2)|∇u|²
    """

    def __init__(self, Lx=16 * np.pi, Ly=16 * np.pi, Nx=64, Ny=64, mu=0.0, nu_extra=0.0, dt=0.05):
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.mu = mu
        self.nu_extra = nu_extra  # Extra diffusion for "real" physics
        self.dt = dt

        # Spatial grids
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Wavenumbers
        self.kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        self.ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')

        # k² and k⁴
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K4 = self.K2 ** 2

        # Linear operator: L = k² - k⁴ - μ - ν_extra*k²
        # The k² term is destabilizing (from -∇²), -k⁴ is stabilizing (from -∇⁴)
        # Extra diffusion adds -ν_extra*k² which is stabilizing
        self.L = self.K2 - self.K4 - mu - nu_extra * self.K2

        # Dealias mask (2/3 rule)
        kx_max = np.max(np.abs(self.kx)) * 2 / 3
        ky_max = np.max(np.abs(self.ky)) * 2 / 3
        self.dealias = (np.abs(self.KX) < kx_max) & (np.abs(self.KY) < ky_max)

        self._setup_etdrk4()

    def _setup_etdrk4(self):
        """Setup ETDRK4 coefficients."""
        h = self.dt
        L = self.L

        # Exponential factors
        self.E = np.exp(h * L)
        self.E2 = np.exp(h * L / 2)

        # Contour integral for phi functions
        M = 32
        theta = np.pi * (np.arange(1, M + 1) - 0.5) / M
        r = np.exp(1j * theta)

        # Reshape for broadcasting
        LR = h * L[:, :, np.newaxis] + r[np.newaxis, np.newaxis, :]

        # Compute phi functions via contour integral
        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=2))
        self.f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=2))
        self.f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=2))
        self.f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=2))

    def _nonlinear(self, u_hat):
        """Compute nonlinear term: N(u) = -(1/2)|∇u|² in Fourier space."""
        # Compute derivatives
        ux_hat = 1j * self.KX * u_hat
        uy_hat = 1j * self.KY * u_hat

        # Transform to physical space
        ux = np.real(ifft2(ux_hat))
        uy = np.real(ifft2(uy_hat))

        # Nonlinear term in physical space
        N_phys = -0.5 * (ux ** 2 + uy ** 2)

        # Transform back and dealias
        N_hat = fft2(N_phys)
        N_hat *= self.dealias

        return N_hat

    def step(self, u_hat):
        """Single ETDRK4 time step."""
        Nu = self._nonlinear(u_hat)
        a = self.E2 * u_hat + self.Q * Nu
        Na = self._nonlinear(a)
        b = self.E2 * u_hat + self.Q * Na
        Nb = self._nonlinear(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = self._nonlinear(c)
        return self.E * u_hat + self.f1 * Nu + 2 * self.f2 * (Na + Nb) + self.f3 * Nc

    def simulate(self, u0, T, save_every=1, verbose=True):
        """Run simulation."""
        n_steps = int(T / self.dt)
        n_save = n_steps // save_every + 1

        U = np.zeros((n_save, self.Nx, self.Ny))
        u_hat = fft2(u0)
        U[0] = u0

        save_idx = 1
        for step in range(1, n_steps + 1):
            u_hat = self.step(u_hat)

            # Check for blowup
            if np.any(np.isnan(u_hat)) or np.max(np.abs(u_hat)) > 1e10:
                print(f"  WARNING: Simulation blew up at step {step}")
                return U[:save_idx]

            if step % save_every == 0 and save_idx < n_save:
                U[save_idx] = np.real(ifft2(u_hat))
                save_idx += 1

        if verbose:
            print(f"  Simulation complete: {save_idx} snapshots, u ∈ [{U.min():.2f}, {U.max():.2f}]")

        return U



# 2. Dataset for SHRED (2D version with StandardScaler)


class TimeSeriesDataset2D(Dataset):
    """Dataset with time-lagged sensor measurements for SHRED."""

    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        """
        U: (n_timesteps, Nx, Ny)
        sensor_indices: list of (i, j) tuples
        """
        self.n_timesteps, self.Nx, self.Ny = U.shape
        self.lags = lags

        # Flatten spatial dimensions
        self.U_flat = U.reshape(self.n_timesteps, -1)  # (T, Nx*Ny)

        # Extract sensor readings
        self.sensor_indices = sensor_indices
        self.n_sensors = len(sensor_indices)
        self.S = np.zeros((self.n_timesteps, self.n_sensors))
        for idx, (i, j) in enumerate(sensor_indices):
            self.S[:, idx] = U[:, i, j]

        # Use StandardScaler (zero mean, unit variance) - more stable than MinMax
        if scaler is None:
            self.scaler_U = StandardScaler()
            self.scaler_S = StandardScaler()
        else:
            self.scaler_U, self.scaler_S = scaler

        if fit_scaler:
            self.U_scaled = self.scaler_U.fit_transform(self.U_flat)
            self.S_scaled = self.scaler_S.fit_transform(self.S)
        else:
            self.U_scaled = self.scaler_U.transform(self.U_flat)
            self.S_scaled = self.scaler_S.transform(self.S)

        self.valid_indices = np.arange(lags, self.n_timesteps)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        sensor_history = self.S_scaled[t - self.lags:t]  # (lags, n_sensors)
        full_state = self.U_scaled[t]  # (Nx*Ny,)
        return (torch.tensor(sensor_history, dtype=torch.float32),
                torch.tensor(full_state, dtype=torch.float32))

    def get_scalers(self):
        return (self.scaler_U, self.scaler_S)

    def get_flat_sensor_indices(self):
        """Convert 2D indices to flat indices."""
        return np.array([i * self.Ny + j for i, j in self.sensor_indices])



# 3. SHRED Model


class SHRED(nn.Module):
    """SHRED: LSTM encoder -> latent -> MLP decoder."""

    def __init__(self, num_sensors, lags, hidden_size, output_size,
                 num_lstm_layers=2, decoder_layers=[256, 256], dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Decoder MLP
        layers = []
        prev = hidden_size
        for size in decoder_layers:
            layers.extend([
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
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
    """DA-SHRED: SHRED with latent transformation for domain adaptation."""

    def __init__(self, base_shred, freeze_decoder=False):
        super().__init__()
        self.lstm = copy.deepcopy(base_shred.lstm)
        self.decoder = copy.deepcopy(base_shred.decoder)
        self.hidden_size = base_shred.hidden_size
        self.output_size = base_shred.output_size

        # Residual transformation
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        # Initialize small
        for m in self.transform.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x, apply_transform=True):
        _, (h_n, _) = self.lstm(x)
        latent = h_n[-1]

        if apply_transform:
            latent_t = latent + self.transform(latent)
        else:
            latent_t = latent

        return self.decoder(latent_t), latent, latent_t



# 5. Latent Aligners (MSE and GAN options)


class LatentMapper(nn.Module):
    """Simple MLP to map sim latents to real latents."""

    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        return self.net(z)


class LatentGANAligner(nn.Module):
    """GAN-based aligner for sim->real latent transformation."""

    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Generator: maps sim latent -> real latent
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Discriminator: classifies real vs fake latents
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z):
        return self.generator(z)


def train_latent_mapper(mapper, Z_sim, Z_real, epochs=200, lr=1e-3):
    """Train MSE-based mapper to align sim->real latents."""
    n = min(len(Z_sim), len(Z_real))
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(Z_sim[:n], dtype=torch.float32),
        torch.tensor(Z_real[:n], dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    criterion = nn.MSELoss()

    mapper.to(device)
    mapper.train()

    for epoch in range(epochs):
        total_loss = 0
        for z_sim, z_real in loader:
            z_sim, z_real = z_sim.to(device), z_real.to(device)
            optimizer.zero_grad()
            z_pred = mapper(z_sim)
            loss = criterion(z_pred, z_real)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"      Mapper epoch {epoch + 1}, loss: {total_loss / len(loader):.6f}")

    return mapper


def train_gan_aligner(aligner, Z_sim, Z_real, epochs=300, batch_size=32, lr=1e-4):
    """
    Train GAN-based aligner with:
    - Adversarial loss: fool discriminator
    - Reconstruction loss: mapped sim should match real
    - Cycle consistency: stability
    """
    n = min(len(Z_sim), len(Z_real))
    Z_sim_t = torch.tensor(Z_sim[:n], dtype=torch.float32)
    Z_real_t = torch.tensor(Z_real[:n], dtype=torch.float32)

    sim_dataset = torch.utils.data.TensorDataset(Z_sim_t)
    real_dataset = torch.utils.data.TensorDataset(Z_real_t)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    opt_gen = optim.Adam(aligner.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(aligner.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_recon = nn.MSELoss()

    aligner.to(device)

    for epoch in range(epochs):
        aligner.train()
        g_loss_total, d_loss_total = 0.0, 0.0

        for (z_sim_batch,), (z_real_batch,) in zip(sim_loader, real_loader):
            z_sim = z_sim_batch.to(device)
            z_real = z_real_batch.to(device)

            # Match batch sizes
            min_batch = min(z_sim.size(0), z_real.size(0))
            z_sim, z_real = z_sim[:min_batch], z_real[:min_batch]

            # === Train Discriminator ===
            opt_disc.zero_grad()

            # Real samples
            d_real = aligner.discriminator(z_real)
            real_labels = 0.9 * torch.ones_like(d_real)  # Label smoothing
            loss_real = criterion_adv(d_real, real_labels)

            # Fake samples
            z_fake = aligner.generator(z_sim)
            d_fake = aligner.discriminator(z_fake.detach())
            fake_labels = 0.1 * torch.ones_like(d_fake)  # Label smoothing
            loss_fake = criterion_adv(d_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            opt_disc.step()
            d_loss_total += loss_disc.item()

            # === Train Generator ===
            opt_gen.zero_grad()

            z_fake = aligner.generator(z_sim)
            d_fake = aligner.discriminator(z_fake)

            # Adversarial loss: fool discriminator
            loss_adv = criterion_adv(d_fake, torch.ones_like(d_fake))

            # Reconstruction loss: mapped sim should be close to real
            loss_recon = criterion_recon(z_fake, z_real)

            # Total generator loss
            loss_gen = loss_adv + 10.0 * loss_recon
            loss_gen.backward()
            opt_gen.step()
            g_loss_total += loss_gen.item()

        if (epoch + 1) % 50 == 0:
            print(f"      GAN epoch {epoch + 1}, G_loss: {g_loss_total / len(sim_loader):.4f}, "
                  f"D_loss: {d_loss_total / len(sim_loader):.4f}")

    return aligner



# 6. Training Functions


def train_shred(model, train_data, valid_data, epochs=150, batch_size=32, lr=5e-4, patience=20):
    """Train SHRED model."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(epochs):
        # Train
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
            train_loss += loss.item() * len(sensors)
        train_loss /= len(train_data)

        # Validate
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _ = model(sensors)
                valid_loss += criterion(pred, state).item() * len(sensors)
        valid_loss /= len(valid_data)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch + 1}, train: {train_loss:.6f}, valid: {valid_loss:.6f}")

    model.load_state_dict(best_state)
    return model


def train_dashred(model, train_real, valid_real, shred_model, train_sim,
                  sensor_indices_flat, epochs=200, batch_size=32, lr=1e-4, patience=30,
                  gan_epochs=300, use_gan=True):
    """Train DA-SHRED with GAN-based latent alignment."""

    # Extract latents
    Z_sim = get_latent_trajectory(shred_model, train_sim)
    Z_real = get_latent_trajectory(model, train_real, is_dashred=True)

    print(f"    Latents: sim {Z_sim.shape}, real {Z_real.shape}")

    # Train aligner
    if use_gan:
        print("    Training GAN latent aligner...")
        aligner = LatentGANAligner(model.hidden_size, hidden_dim=128)
        aligner = train_gan_aligner(aligner, Z_sim, Z_real, epochs=gan_epochs, lr=1e-4)
        # Store generator for later use
        model.latent_aligner = aligner.generator
    else:
        print("    Training MSE latent mapper...")
        mapper = LatentMapper(model.hidden_size)
        mapper = train_latent_mapper(mapper, Z_sim, Z_real, epochs=200, lr=1e-3)
        model.latent_aligner = mapper.net

    # Fine-tune full model with lower learning rate
    train_loader = DataLoader(train_real, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_real, batch_size=batch_size)

    # Differential learning rates
    optimizer = optim.Adam([
        {'params': model.lstm.parameters(), 'lr': lr * 0.1},  # Fine-tune LSTM slowly
        {'params': model.transform.parameters(), 'lr': lr},  # Train transform faster
        {'params': model.decoder.parameters(), 'lr': lr * 0.5}  # Moderate for decoder
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()

            pred, latent, latent_t = model(sensors, apply_transform=True)

            # Losses with better weighting
            recon_loss = nn.functional.mse_loss(pred, state)

            # Sensor loss - emphasize matching at sensor locations
            sensor_loss = nn.functional.mse_loss(pred[:, sensor_indices_flat], state[:, sensor_indices_flat])

            # Regularization - don't let transform deviate too much
            reg_loss = torch.mean((latent_t - latent) ** 2)

            # Smoothness loss - encourage spatial coherence
            Nx = Ny = int(np.sqrt(model.output_size))
            pred_2d = pred.view(-1, Nx, Ny)
            dx = pred_2d[:, 1:, :] - pred_2d[:, :-1, :]
            dy = pred_2d[:, :, 1:] - pred_2d[:, :, :-1]
            smooth_loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)

            loss = recon_loss + 1.0 * sensor_loss + 0.001 * reg_loss + 0.01 * smooth_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(sensors)

        train_loss /= len(train_real)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _, _ = model(sensors)
                valid_loss += nn.functional.mse_loss(pred, state).item() * len(sensors)
        valid_loss /= len(valid_real)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            curr_lr = optimizer.param_groups[1]['lr']  # Transform LR
            print(f"    Epoch {epoch + 1}, train: {train_loss:.6f}, valid: {valid_loss:.6f}, lr: {curr_lr:.2e}")

    model.load_state_dict(best_state)
    return model



# 7. Evaluation


def evaluate(model, dataset, scaler_U, is_dashred=False):
    """Evaluate and return predictions in original scale."""
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
    """Extract latents."""
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset))

    with torch.no_grad():
        for sensors, _ in loader:
            sensors = sensors.to(device)
            if is_dashred:
                _, latent, _ = model(sensors)
            else:
                _, latent = model(sensors)
            return latent.cpu().numpy()



# 8. SINDy


def sindy_stls(Theta, target, threshold=0.05, max_iter=20):
    """Sparse regression."""
    Xi = np.linalg.lstsq(Theta, target, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for j in range(target.shape[1]):
            big = ~small[:, j]
            if big.sum() > 0:
                Xi[big, j] = np.linalg.lstsq(Theta[:, big], target[:, j], rcond=None)[0]
    return Xi



# 9. Main


if __name__ == "__main__":


    # Parameters


    # PDE
    Lx, Ly = 16 * np.pi, 16 * np.pi  # Smaller domain for stability
    Nx, Ny = 64, 64
    dt = 0.05
    T = 50.0
    save_every = 5  # dt_save = 0.25

    # Damping parameters for "real" physics (choose one or combine)
    # Option A: Linear damping (u_t = ... - μ*u)
    mu_damping = 0.06  # Small gap for DA-SHRED to close

    # Option B: Diffusive damping (u_t = ... - ν_extra*∇²u), more physical
    nu_extra_damping = 0.0  # Set to e.g. 0.05 to use diffusive damping instead

    # SHRED
    num_sensors = 49  # 7x7 grid (more sensors for better reconstruction)
    lags = 20
    hidden_size = 48  # Slightly larger
    decoder_layers = [256, 256]

    # Training
    shred_epochs = 200
    shred_lr = 5e-4
    dashred_epochs = 200  # Longer training
    dashred_lr = 1e-4  # Lower learning rate
    gan_epochs = 300  # GAN aligner epochs


    # Generate Data


    print("\n" + "=" * 60)
    print("2D Kuramoto-Sivashinsky DA-SHRED")
    print("=" * 60)

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition - smooth with some structure
    u0 = np.cos(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly) + \
         0.5 * np.sin(4 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)
    u0 = u0 / np.max(np.abs(u0))  # Normalize

    print("\n[1] Generating data...")
    print(f"    Domain: [{Lx / np.pi:.1f}π, {Ly / np.pi:.1f}π], Grid: {Nx}x{Ny}")
    print(f"    Simulation: μ=0, ν_extra=0 (undamped)")
    print(f"    Real: μ={mu_damping}, ν_extra={nu_extra_damping} (damped)")

    ks_sim = KuramotoSivashinsky2D(Lx, Ly, Nx, Ny, mu=0.0, nu_extra=0.0, dt=dt)
    ks_real = KuramotoSivashinsky2D(Lx, Ly, Nx, Ny, mu=mu_damping, nu_extra=nu_extra_damping, dt=dt)

    print("  Running simulation (undamped)...")
    U_sim = ks_sim.simulate(u0, T, save_every)
    print("  Running real physics (damped)...")
    U_real = ks_real.simulate(u0, T, save_every)

    print(f"    Snapshots: {len(U_sim)}")

    # Quick plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, t_idx in enumerate([0, 40, 80]):
        if t_idx < len(U_sim):
            im = axes[i].pcolormesh(x, y, U_sim[t_idx].T, cmap='RdBu_r', shading='auto')
            axes[i].set_title(f't_idx={t_idx}')
            axes[i].set_aspect('equal')
            plt.colorbar(im, ax=axes[i])
    plt.suptitle('2D KS Simulation')
    plt.tight_layout()
    plt.savefig('ks2d_simulation.png', dpi=150)
    plt.show()


    # Create Datasets


    print("\n[2] Creating datasets...")

    # Sensor grid
    ns = int(np.sqrt(num_sensors))
    si = np.linspace(0, Nx - 1, ns, dtype=int)
    sj = np.linspace(0, Ny - 1, ns, dtype=int)
    sensor_indices = [(i, j) for i in si for j in sj]
    num_sensors = len(sensor_indices)

    print(f"    Sensors: {num_sensors} ({ns}x{ns} grid)")
    print(f"    Lags: {lags}")

    n_train = int(0.8 * len(U_sim))

    train_sim = TimeSeriesDataset2D(U_sim[:n_train], sensor_indices, lags, fit_scaler=True)
    valid_sim = TimeSeriesDataset2D(U_sim[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers())
    train_real = TimeSeriesDataset2D(U_real[:n_train], sensor_indices, lags, scaler=train_sim.get_scalers())
    valid_real = TimeSeriesDataset2D(U_real[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers())

    print(f"    Train samples: {len(train_sim)}, Valid samples: {len(valid_sim)}")

    if len(train_sim) < 10 or len(valid_sim) < 5:
        raise ValueError("Not enough samples! Increase T or reduce lags/save_every.")

    sensor_indices_flat = train_sim.get_flat_sensor_indices()


    # Train SHRED


    print("\n[3] Training SHRED...")

    output_size = Nx * Ny
    shred_model = SHRED(
        num_sensors=num_sensors, lags=lags, hidden_size=hidden_size,
        output_size=output_size, num_lstm_layers=2, decoder_layers=decoder_layers
    )
    print(f"    Parameters: {sum(p.numel() for p in shred_model.parameters()):,}")

    shred_model = train_shred(shred_model, train_sim, valid_sim, epochs=shred_epochs, lr=shred_lr)


    # Evaluate Gap


    print("\n[4] Evaluating SIM2REAL gap...")
    scaler_U, _ = train_sim.get_scalers()

    _, _, mse_sim = evaluate(shred_model, valid_sim, scaler_U)
    _, _, mse_gap = evaluate(shred_model, valid_real, scaler_U)

    print(f"    SHRED on sim: RMSE = {np.sqrt(mse_sim):.4f}")
    print(f"    SHRED on real: RMSE = {np.sqrt(mse_gap):.4f}")
    print(f"    Gap ratio: {mse_gap / mse_sim:.2f}x")


    # Train DA-SHRED


    print("\n[5] Training DA-SHRED...")

    dashred_model = DASHRED(shred_model)
    dashred_model = train_dashred(
        dashred_model, train_real, valid_real, shred_model, train_sim,
        sensor_indices_flat, epochs=dashred_epochs, lr=dashred_lr,
        gan_epochs=gan_epochs, use_gan=True
    )


    # Final Evaluation


    print("\n[6] Final evaluation...")

    pred_before, truth, mse_before = evaluate(shred_model, valid_real, scaler_U)
    pred_after, _, mse_after = evaluate(dashred_model, valid_real, scaler_U, is_dashred=True)

    print(f"    SHRED on real: RMSE = {np.sqrt(mse_before):.4f}")
    print(f"    DA-SHRED on real: RMSE = {np.sqrt(mse_after):.4f}")

    if mse_after < mse_before:
        print(f"    Improvement: {mse_before / mse_after:.2f}x")
        print(f"    Gap reduction: {(mse_before - mse_after) / mse_before * 100:.1f}%")
    else:
        print(f"    WARNING: DA-SHRED did not improve (may need tuning)")


    # Visualization


    print("\n[7] Visualization...")

    # Get training predictions
    dashred_model.eval()
    with torch.no_grad():
        loader = DataLoader(train_real, batch_size=len(train_real))
        for sensors, _ in loader:
            sensors = sensors.to(device)
            pred_flat, _, _ = dashred_model(sensors, apply_transform=True)
            pred_flat = scaler_U.inverse_transform(pred_flat.cpu().numpy())

    pred_train = pred_flat.reshape(-1, Nx, Ny)
    U_sim_train = U_sim[lags:n_train]
    U_real_train = U_real[lags:n_train]

    # Plot comparison at selected training time indices
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # These are indices into the training predictions (after lags offset)
    train_time_indices = [40, 60, 80]

    for col, t_idx in enumerate(train_time_indices):
        if t_idx >= len(pred_train):
            t_idx = len(pred_train) - 1

        vmin = min(U_sim_train[t_idx].min(), U_real_train[t_idx].min(), pred_train[t_idx].min())
        vmax = max(U_sim_train[t_idx].max(), U_real_train[t_idx].max(), pred_train[t_idx].max())

        # Original timestep = lags + t_idx
        orig_t = lags + t_idx

        im0 = axes[0, col].pcolormesh(x, y, U_sim_train[t_idx].T, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'Sim (train_idx={t_idx}, orig_t={orig_t})')
        axes[0, col].set_aspect('equal')

        im1 = axes[1, col].pcolormesh(x, y, U_real_train[t_idx].T, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Real (train_idx={t_idx}, orig_t={orig_t})')
        axes[1, col].set_aspect('equal')

        im2 = axes[2, col].pcolormesh(x, y, pred_train[t_idx].T, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        axes[2, col].set_title(f'DA-SHRED (train_idx={t_idx}, orig_t={orig_t})')
        axes[2, col].set_aspect('equal')

    axes[0, 0].set_ylabel('Simulation')
    axes[1, 0].set_ylabel('Real Physics')
    axes[2, 0].set_ylabel('DA-SHRED')

    plt.suptitle('Reconstruction Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstruction_comparison_2d.png', dpi=150)
    print("    Saved: reconstruction_comparison_2d.png")
    plt.show()

    # Error plot (Reconstruction error on yet to be seen real physics sensor measurements)
    fig, ax = plt.subplots(figsize=(8, 4))
    n_valid = pred_before.shape[0]
    rmse_before = np.sqrt(np.mean((pred_before - truth) ** 2, axis=1))
    rmse_after = np.sqrt(np.mean((pred_after - truth) ** 2, axis=1))
    ax.plot(rmse_before, 'r-', label='SHRED', linewidth=2)
    ax.plot(rmse_after, 'g-', label='DA-SHRED', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('RMSE')
    ax.set_title('Predictive Reconstruction Error on unseen Real Physics')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_comparison_2d.png', dpi=150)
    print("    Saved: error_comparison_2d.png")
    plt.show()

    # Save
    torch.save({
        'shred': shred_model.state_dict(),
        'dashred': dashred_model.state_dict(),
        'params': {'Nx': Nx, 'Ny': Ny, 'hidden_size': hidden_size}
    }, 'checkpoint_2d.pt')
    print("    Saved: checkpoint_2d.pt")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
