### DA-SHRED for 2D Gray-Scott Reaction-Diffusion System


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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



# 1. PDE Solver: 2D Gray-Scott Reaction-Diffusion System


class GrayScott2D:
    """
    2D Gray-Scott reaction-diffusion system using finite differences.

    Equations:
        ∂A/∂t = DA * ∇²A - A*B² + f*(1-A)
        ∂B/∂t = DB * ∇²B + A*B² - (f+k)*B - damping_term

    Damping options for "real" physics:
        - 'linear': -mu*B
        - 'nonlinear': -mu*B²*A

    Simulation: mu = 0 (no damping)
    Real physics: mu > 0
    """

    def __init__(self, N=256, DA=0.20, DB=0.10, f=0.0545, k=0.062,
                 mu=0.0, damping_type='linear', delta_t=1.0):
        self.N = N
        self.DA = DA
        self.DB = DB
        self.f = f
        self.k = k
        self.mu = mu  # Damping coefficient (0 for sim, >0 for real)
        self.damping_type = damping_type  # 'linear' or 'nonlinear'
        self.delta_t = delta_t

    def _apply_laplacian(self, mat):
        """Vectorized Laplacian with periodic BC."""
        return (
                np.roll(mat, 1, axis=0) +
                np.roll(mat, -1, axis=0) +
                np.roll(mat, 1, axis=1) +
                np.roll(mat, -1, axis=1) -
                4 * mat
        )

    def step(self, A, B):
        """Gray-Scott update formula."""
        # Diffusion
        diff_A = self.DA * self._apply_laplacian(A)
        diff_B = self.DB * self._apply_laplacian(B)

        # Reaction
        reaction = A * B * B
        diff_A -= reaction
        diff_B += reaction

        # Birth/death
        diff_A += self.f * (1 - A)
        diff_B -= (self.k + self.f) * B

        # Damping (for real physics)
        if self.mu > 0:
            if self.damping_type == 'linear':
                diff_B -= self.mu * B
            elif self.damping_type == 'nonlinear':
                diff_B -= self.mu * B * B * A

        return A + diff_A * self.delta_t, B + diff_B * self.delta_t

    def initialize(self, random_influence=0.2):
        """Get initial concentrations."""
        N = self.N
        A = (1 - random_influence) * np.ones((N, N))
        B = np.zeros((N, N))
        A += random_influence * np.random.random((N, N))
        B += random_influence * np.random.random((N, N))

        # Central disturbance
        N2, r = N // 2, 50
        A[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
        B[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

        return A, B

    def simulate(self, A0, B0, n_steps, save_every=100, verbose=True):
        """Run simulation (optimized for speed)."""
        n_save = n_steps // save_every + 1
        A_all = np.zeros((n_save, self.N, self.N))
        B_all = np.zeros((n_save, self.N, self.N))

        A, B = A0.copy(), B0.copy()
        A_all[0], B_all[0] = A, B

        save_idx = 1
        for step in range(1, n_steps + 1):
            A, B = self.step(A, B)
            if step % save_every == 0 and save_idx < n_save:
                A_all[save_idx], B_all[save_idx] = A, B
                save_idx += 1

        if verbose:
            print(f"  Complete: {save_idx} snapshots, B ∈ [{B_all.min():.3f}, {B_all.max():.3f}]")

        return A_all, B_all



# 2. Dataset for SHRED (2D Gray-Scott - uses V field)


class TimeSeriesDataset2D(Dataset):
    """Dataset with time-lagged sensor measurements for SHRED."""

    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        """
        U: (n_timesteps, Nx, Ny) - Note: This is actually V for Gray-Scott
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

        # Use StandardScaler (zero mean, unit variance)
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
    """Train GAN-based aligner."""
    n = min(len(Z_sim), len(Z_real))
    Z_sim_t = torch.tensor(Z_sim[:n], dtype=torch.float32)
    Z_real_t = torch.tensor(Z_real[:n], dtype=torch.float32)

    sim_dataset = torch.utils.data.TensorDataset(Z_sim_t)
    real_dataset = torch.utils.data.TensorDataset(Z_real_t)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

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

            min_batch = min(z_sim.size(0), z_real.size(0))
            z_sim, z_real = z_sim[:min_batch], z_real[:min_batch]

            # Train Discriminator
            opt_disc.zero_grad()
            d_real = aligner.discriminator(z_real)
            real_labels = 0.9 * torch.ones_like(d_real)
            loss_real = criterion_adv(d_real, real_labels)

            z_fake = aligner.generator(z_sim)
            d_fake = aligner.discriminator(z_fake.detach())
            fake_labels = 0.1 * torch.ones_like(d_fake)
            loss_fake = criterion_adv(d_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            opt_disc.step()
            d_loss_total += loss_disc.item()

            # Train Generator
            opt_gen.zero_grad()
            z_fake = aligner.generator(z_sim)
            d_fake = aligner.discriminator(z_fake)

            loss_adv = criterion_adv(d_fake, torch.ones_like(d_fake))
            loss_recon = criterion_recon(z_fake, z_real)

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
    """Train SHRED model with edge-preserving losses for sharp reconstruction."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0

    Nx = Ny = int(np.sqrt(model.output_size))

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)

            # L1 + L2 loss (L1 preserves edges better)
            l2_loss = nn.functional.mse_loss(pred, state)
            l1_loss = nn.functional.l1_loss(pred, state)

            # Reshape for gradient computation
            pred_2d = pred.view(-1, Nx, Ny)
            state_2d = state.view(-1, Nx, Ny)

            # First-order gradients
            pred_dx = pred_2d[:, 1:, :] - pred_2d[:, :-1, :]
            pred_dy = pred_2d[:, :, 1:] - pred_2d[:, :, :-1]
            state_dx = state_2d[:, 1:, :] - state_2d[:, :-1, :]
            state_dy = state_2d[:, :, 1:] - state_2d[:, :, :-1]

            # Gradient matching (L1 for sharper edges)
            grad_loss = nn.functional.l1_loss(pred_dx, state_dx) + nn.functional.l1_loss(pred_dy, state_dy)

            # Second-order gradients (Laplacian) - captures edge sharpness
            pred_lap = (torch.roll(pred_2d, 1, dims=1) + torch.roll(pred_2d, -1, dims=1) +
                        torch.roll(pred_2d, 1, dims=2) + torch.roll(pred_2d, -1, dims=2) - 4 * pred_2d)
            state_lap = (torch.roll(state_2d, 1, dims=1) + torch.roll(state_2d, -1, dims=1) +
                         torch.roll(state_2d, 1, dims=2) + torch.roll(state_2d, -1, dims=2) - 4 * state_2d)
            lap_loss = nn.functional.l1_loss(pred_lap, state_lap)

            # Combined loss
            loss = 0.5 * l2_loss + 0.5 * l1_loss + 0.2 * grad_loss + 0.1 * lap_loss

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
                valid_loss += nn.functional.mse_loss(pred, state).item() * len(sensors)
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
    """Train DA-SHRED with GAN-based latent alignment and edge-preserving loss."""

    # Extract latents
    Z_sim = get_latent_trajectory(shred_model, train_sim)
    Z_real = get_latent_trajectory(model, train_real, is_dashred=True)

    print(f"    Latents: sim {Z_sim.shape}, real {Z_real.shape}")

    # Train aligner
    if use_gan:
        print("    Training GAN latent aligner...")
        aligner = LatentGANAligner(model.hidden_size, hidden_dim=128)
        aligner = train_gan_aligner(aligner, Z_sim, Z_real, epochs=gan_epochs, lr=1e-4)
        model.latent_aligner = aligner.generator
    else:
        print("    Training MSE latent mapper...")
        mapper = LatentMapper(model.hidden_size)
        mapper = train_latent_mapper(mapper, Z_sim, Z_real, epochs=200, lr=1e-3)
        model.latent_aligner = mapper.net

    # Fine-tune full model
    train_loader = DataLoader(train_real, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_real, batch_size=batch_size)

    optimizer = optim.Adam([
        {'params': model.lstm.parameters(), 'lr': lr * 0.1},
        {'params': model.transform.parameters(), 'lr': lr},
        {'params': model.decoder.parameters(), 'lr': lr * 0.5}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0

    Nx = Ny = int(np.sqrt(model.output_size))

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()

            pred, latent, latent_t = model(sensors, apply_transform=True)

            # L1 + L2 loss (L1 preserves edges better)
            l2_loss = nn.functional.mse_loss(pred, state)
            l1_loss = nn.functional.l1_loss(pred, state)

            # Sensor matching loss
            sensor_loss = nn.functional.mse_loss(pred[:, sensor_indices_flat], state[:, sensor_indices_flat])

            # Latent regularization
            reg_loss = torch.mean((latent_t - latent) ** 2)

            # Reshape for gradient computation
            pred_2d = pred.view(-1, Nx, Ny)
            state_2d = state.view(-1, Nx, Ny)

            # First-order gradients (L1 for sharper edges)
            pred_dx = pred_2d[:, 1:, :] - pred_2d[:, :-1, :]
            pred_dy = pred_2d[:, :, 1:] - pred_2d[:, :, :-1]
            state_dx = state_2d[:, 1:, :] - state_2d[:, :-1, :]
            state_dy = state_2d[:, :, 1:] - state_2d[:, :, :-1]
            grad_loss = nn.functional.l1_loss(pred_dx, state_dx) + nn.functional.l1_loss(pred_dy, state_dy)

            # Second-order gradients (Laplacian) - captures edge sharpness
            pred_lap = (torch.roll(pred_2d, 1, dims=1) + torch.roll(pred_2d, -1, dims=1) +
                        torch.roll(pred_2d, 1, dims=2) + torch.roll(pred_2d, -1, dims=2) - 4 * pred_2d)
            state_lap = (torch.roll(state_2d, 1, dims=1) + torch.roll(state_2d, -1, dims=1) +
                         torch.roll(state_2d, 1, dims=2) + torch.roll(state_2d, -1, dims=2) - 4 * state_2d)
            lap_loss = nn.functional.l1_loss(pred_lap, state_lap)

            loss = 0.5 * l2_loss + 0.5 * l1_loss + 1.0 * sensor_loss + 0.001 * reg_loss + 0.2 * grad_loss + 0.1 * lap_loss

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
            curr_lr = optimizer.param_groups[1]['lr']
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


    # PDE - Gray-Scott (matching reference)
    N = 256  # Grid size
    DA = 0.20
    DB = 0.10
    f = 0.0545  # Feed rate (coral growth pattern)
    k = 0.062  # Kill rate

    # Damping for "real" physics
    # Choose damping type: 'linear' (-mu*B) or 'nonlinear' (-mu*B²*A)
    damping_type = 'nonlinear'

    # Separate mu values (nonlinear needs larger mu since B²*A << B typically)
    mu_linear = 0.003  # For linear damping: -mu*B
    mu_nonlinear = 0.008  # For nonlinear damping: -mu*B²*A

    # Select mu based on damping type
    mu_damping = mu_nonlinear if damping_type == 'nonlinear' else mu_linear

    # Time stepping
    delta_t = 1.0
    n_steps = 8000  # Total steps
    save_every = 80  # -> 100 snapshots

    # SHRED
    num_sensors = 49  # 7x7 grid
    lags = 15
    hidden_size = 48
    decoder_layers = [256, 256]

    # Training
    shred_epochs = 200
    shred_lr = 5e-4
    dashred_epochs = 200
    dashred_lr = 1e-4
    gan_epochs = 300


    # Generate Data


    print("\n" + "=" * 60)
    print("2D Gray-Scott DA-SHRED")
    print("=" * 60)

    print("\n[1] Generating data...")
    print(f"    Grid: {N}x{N}, Steps: {n_steps}, save_every: {save_every}")
    print(f"    Simulation: mu=0 (no damping)")
    print(f"    Real: mu={mu_damping}, type='{damping_type}'")

    # Create solvers
    gs_sim = GrayScott2D(N=N, DA=DA, DB=DB, f=f, k=k, mu=0.0, delta_t=delta_t)
    gs_real = GrayScott2D(N=N, DA=DA, DB=DB, f=f, k=k, mu=mu_damping,
                          damping_type=damping_type, delta_t=delta_t)

    # Initialize with same initial condition
    A0, B0 = gs_sim.initialize(random_influence=0.2)

    print("  Running simulation...")
    A_sim, B_sim = gs_sim.simulate(A0.copy(), B0.copy(), n_steps, save_every)
    print("  Running real physics...")
    A_real, B_real = gs_real.simulate(A0.copy(), B0.copy(), n_steps, save_every)

    print(f"    Snapshots: {len(B_sim)}")

    # Quick plot using V field
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, t_idx in enumerate([0, 40, 80]):
        if t_idx < len(B_sim):
            im = axes[i].imshow(B_sim[t_idx], cmap='Greys', vmin=0)
            axes[i].set_title(f't_idx={t_idx}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
    plt.suptitle('Gray-Scott Simulation (V field)')
    plt.tight_layout()
    plt.savefig('grayscott_simulation.png', dpi=150)
    plt.show()


    # Create Datasets (using V field)


    print("\n[2] Creating datasets...")

    # Sensor grid
    ns = int(np.sqrt(num_sensors))
    si = np.linspace(0, N - 1, ns, dtype=int)
    sj = np.linspace(0, N - 1, ns, dtype=int)
    sensor_indices = [(i, j) for i in si for j in sj]
    num_sensors = len(sensor_indices)

    print(f"    Sensors: {num_sensors} ({ns}x{ns} grid)")
    print(f"    Lags: {lags}")

    n_train = int(0.8 * len(B_sim))

    # Use B field for training (shows patterns)
    train_sim = TimeSeriesDataset2D(B_sim[:n_train], sensor_indices, lags, fit_scaler=True)
    valid_sim = TimeSeriesDataset2D(B_sim[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers())
    train_real = TimeSeriesDataset2D(B_real[:n_train], sensor_indices, lags, scaler=train_sim.get_scalers())
    valid_real = TimeSeriesDataset2D(B_real[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers())

    print(f"    Train samples: {len(train_sim)}, Valid samples: {len(valid_sim)}")

    if len(train_sim) < 10 or len(valid_sim) < 5:
        raise ValueError("Not enough samples! Increase n_steps or reduce lags/save_every.")

    sensor_indices_flat = train_sim.get_flat_sensor_indices()


    # Train SHRED


    print("\n[3] Training SHRED...")

    output_size = N * N
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

    print(f"    SHRED on sim: RMSE = {np.sqrt(mse_sim):.6f}")
    print(f"    SHRED on real: RMSE = {np.sqrt(mse_gap):.6f}")
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

    print(f"    SHRED on real: RMSE = {np.sqrt(mse_before):.6f}")
    print(f"    DA-SHRED on real: RMSE = {np.sqrt(mse_after):.6f}")

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

    pred_train = pred_flat.reshape(-1, N, N)
    B_sim_train = B_sim[lags:n_train]
    B_real_train = B_real[lags:n_train]

    # Plot comparison at selected training time indices
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    train_time_indices = [40, 60, 80]

    for col, t_idx in enumerate(train_time_indices):
        if t_idx >= len(pred_train):
            t_idx = len(pred_train) - 1

        vmin = min(B_sim_train[t_idx].min(), B_real_train[t_idx].min(), pred_train[t_idx].min())
        vmax = max(B_sim_train[t_idx].max(), B_real_train[t_idx].max(), pred_train[t_idx].max())

        orig_t = lags + t_idx

        im0 = axes[0, col].imshow(B_sim_train[t_idx], cmap='Greys', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'Sim (train_idx={t_idx}, orig_t={orig_t})')
        axes[0, col].axis('off')

        im1 = axes[1, col].imshow(B_real_train[t_idx], cmap='Greys', vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Real (train_idx={t_idx}, orig_t={orig_t})')
        axes[1, col].axis('off')

        im2 = axes[2, col].imshow(pred_train[t_idx], cmap='Greys', vmin=vmin, vmax=vmax)
        axes[2, col].set_title(f'DA-SHRED (train_idx={t_idx}, orig_t={orig_t})')
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Simulation')
    axes[1, 0].set_ylabel('Real Physics')
    axes[2, 0].set_ylabel('DA-SHRED')

    plt.suptitle('Gray-Scott Reconstruction Results (V field)', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstruction_comparison_grayscott.png', dpi=150)
    print("    Saved: reconstruction_comparison_grayscott.png")
    plt.show()

    # Error plot
    fig, ax = plt.subplots(figsize=(8, 4))
    n_valid = pred_before.shape[0]
    rmse_before = np.sqrt(np.mean((pred_before - truth) ** 2, axis=1))
    rmse_after = np.sqrt(np.mean((pred_after - truth) ** 2, axis=1))
    ax.plot(rmse_before, 'r-', label='SHRED', linewidth=2)
    ax.plot(rmse_after, 'g-', label='DA-SHRED', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('RMSE')
    ax.set_title('Predictive Reconstruction Error on unseen Real Physics (Gray-Scott)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_comparison_grayscott.png', dpi=150)
    print("    Saved: error_comparison_grayscott.png")
    plt.show()

    # Save
    torch.save({
        'shred': shred_model.state_dict(),
        'dashred': dashred_model.state_dict(),
        'params': {'N': N, 'hidden_size': hidden_size,
                   'f': f, 'k': k, 'mu_damping': mu_damping,
                   'damping_type': damping_type, 'DA': DA, 'DB': DB}
    }, 'checkpoint_grayscott.pt')
    print("    Saved: checkpoint_grayscott.pt")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)