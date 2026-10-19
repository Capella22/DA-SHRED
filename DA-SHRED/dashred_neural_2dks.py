"""DA-SHRED pipeline for 2D Kuramoto-Sivashinsky.

SHRED (LSTM encoder + POD decoder) trained on sim, DA-SHRED latent alignment (learned residual generator with sensor loss + MMD),
and forward-model SINDy for missing-physics discovery.
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.linalg import svd, solve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
DEV = torch.device('cpu')


# Config
DECODER_TYPE = 'pod'        # 'pod' | 'mlp'
SINDY_TYPE = 'weak'         # 'strong' | 'weak'
WEAK_WIDTH_SEC = 4.0
WEAK_STRIDE_SEC = 1.0
WEAK_POLY_P = 3


# ── PDE Solver ──
class KS2D:
    def __init__(self, Lx=16*np.pi, Ly=16*np.pi, Nx=64, Ny=64,
                 mu=0., nu_extra=0., dt=0.05):
        self.Nx, self.Ny, self.dt = Nx, Ny, dt
        self.Lx, self.Ly = Lx, Ly
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)
        dx, dy = Lx/Nx, Ly/Ny
        kx = np.fft.fftfreq(Nx, d=dx)*2*np.pi
        ky = np.fft.fftfreq(Ny, d=dy)*2*np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2+KY**2
        self.KX, self.KY = KX, KY
        L = K2 - K2**2 - mu - nu_extra*K2
        kxm, kym = np.max(np.abs(kx))*2/3, np.max(np.abs(ky))*2/3
        self.dealias = (np.abs(KX) < kxm) & (np.abs(KY) < kym)
        h = dt
        self.E = np.exp(h*L)
        self.E2 = np.exp(h*L/2)
        M = 32
        th = np.pi*(np.arange(1, M+1)-0.5)/M
        r = np.exp(1j*th)
        LR = h*L[:, :, None] + r[None, None, :]
        self.Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=2))
        self.f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=2))
        self.f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=2))
        self.f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=2))

    def _nl(self, uh):
        ux = np.real(ifft2(1j*self.KX*uh))
        uy = np.real(ifft2(1j*self.KY*uh))
        return fft2(-0.5*(ux**2+uy**2))*self.dealias

    def step(self, uh):
        Nu = self._nl(uh); a = self.E2*uh+self.Q*Nu; Na = self._nl(a)
        b = self.E2*uh+self.Q*Na; Nb = self._nl(b)
        c = self.E2*a+self.Q*(2*Nb-Nu); Nc = self._nl(c)
        return self.E*uh+self.f1*Nu+2*self.f2*(Na+Nb)+self.f3*Nc

    def step_physical(self, u_field, n_sub):
        uh = fft2(u_field)
        for _ in range(n_sub):
            uh = self.step(uh)
        return np.real(ifft2(uh))

    def simulate(self, u0, T, save_every=1, verbose=True):
        ns = int(T/self.dt); nsv = ns//save_every+1
        U = np.zeros((nsv, self.Nx, self.Ny))
        uh = fft2(u0); U[0] = u0; si = 1
        for s in range(1, ns+1):
            uh = self.step(uh)
            if np.any(np.isnan(uh)) or np.max(np.abs(uh)) > 1e10:
                print(f"  Blowup step {s}"); return U[:si]
            if s % save_every == 0 and si < nsv:
                U[si] = np.real(ifft2(uh)); si += 1
        if verbose:
            print(f"  {si} snaps, u in [{U[:si].min():.2f},{U[:si].max():.2f}]")
        return U[:si]


# ── Neural SHRED (LSTM encoder + decoder), trained on SIM full state ──
class PODDecoder(nn.Module):
    """Frozen linear decoder: u = Ub @ z_pod + u_mean."""
    def __init__(self, Ub, u_mean):
        super().__init__()
        self.register_buffer('Ub', torch.from_numpy(Ub.astype(np.float32)))
        self.register_buffer('u_mean', torch.from_numpy(u_mean.astype(np.float32)))
        self.rank = Ub.shape[1]
        self.state_dim = Ub.shape[0]

    def forward(self, z_pod):
        # z_pod: (B, rank), output: (B, N)
        return z_pod @ self.Ub.T + self.u_mean


class MLPDecoder(nn.Module):
    """Trainable MLP decoder: d_z -> hidden_dims -> state_dim."""
    def __init__(self, d_z, state_dim, hidden=(128, 256), dropout=0.2):
        super().__init__()
        layers = []
        prev = d_z
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, state_dim)]
        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, z):
        return self.net(z)


class SHREDNet(nn.Module):
    """LSTM encoder + (POD or MLP) decoder, switchable via decoder_type."""
    def __init__(self, n_sensors, d_z=48, n_layers=2, dropout=0.1,
                 decoder_type='pod',
                 Ub=None, u_mean=None,                      # for POD branch
                 mlp_hidden=(128, 256),                     # for MLP branch
                 head_hidden=(64,),
                 state_dim=None):
        super().__init__()
        self.decoder_type = decoder_type
        self.n_sensors = n_sensors
        self.d_z = d_z
        self.lstm = nn.LSTM(
            input_size=n_sensors, hidden_size=d_z,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d_z)

        if decoder_type == 'pod':
            assert Ub is not None and u_mean is not None, \
                "POD decoder requires Ub and u_mean"
            self.rank = Ub.shape[1]
            self.state_dim = Ub.shape[0]
            head = []
            prev = d_z
            for h in head_hidden:
                head += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(),
                         nn.Dropout(dropout)]
                prev = h
            head += [nn.Linear(prev, self.rank)]
            self.head = nn.Sequential(*head)
            self.decoder = PODDecoder(Ub, u_mean)
        elif decoder_type == 'mlp':
            assert state_dim is not None, "MLP decoder requires state_dim"
            self.state_dim = state_dim
            self.rank = None
            self.head = nn.Identity()
            self.decoder = MLPDecoder(d_z, state_dim, hidden=mlp_hidden,
                                       dropout=dropout)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type!r}")

    def encode(self, s_seq):
        out, _ = self.lstm(s_seq)
        return self.ln(out[:, -1, :])

    def to_pod(self, z):
        """Only meaningful for POD branch — returns POD coeffs."""
        if self.decoder_type != 'pod':
            raise RuntimeError("to_pod() only valid for POD decoder")
        return self.head(z)

    def decode(self, z):
        return self.decoder(self.head(z))

    def forward(self, s_seq):
        return self.decode(self.encode(s_seq))


def make_lagged_sensor_tensor(S, lags):
    """S: (T, p) -> X: (T-lags, lags, p), idx in time aligns with t=lags..T-1"""
    T, p = S.shape
    n = T - lags
    X = np.zeros((n, lags, p), dtype=np.float32)
    for i in range(n):
        X[i] = S[i:i+lags]
    return X


class SHREDTrainer:
    """Train and run inference with the SHRED model (POD or MLP decoder)."""
    def __init__(self, n_sensors, d_z=48, lags=15,
                 decoder_type='pod',
                 Ub=None, u_mean=None,                # for POD branch
                 mlp_hidden=(128, 256),               # for MLP branch
                 state_dim=None,                      # for MLP branch
                 head_hidden=(64,), dropout=0.1,
                 weight_decay=1e-5):
        self.decoder_type = decoder_type
        if decoder_type == 'pod':
            assert Ub is not None and u_mean is not None
            self.net = SHREDNet(n_sensors, d_z=d_z, dropout=dropout,
                                decoder_type='pod',
                                Ub=Ub, u_mean=u_mean,
                                head_hidden=head_hidden).to(DEV)
            self.Ub = Ub
            self.u_mean = u_mean
            self.rank = Ub.shape[1]
        else:  # mlp
            assert state_dim is not None
            self.net = SHREDNet(n_sensors, d_z=d_z, dropout=dropout,
                                decoder_type='mlp',
                                mlp_hidden=mlp_hidden,
                                state_dim=state_dim).to(DEV)
            self.Ub = None
            self.u_mean = None
            self.rank = None
        self.lags = lags
        self.sc = StandardScaler()
        self.weight_decay = weight_decay

    def fit(self, U_sim_flat, sensor_idx, n_epochs=800, lr=3e-3,
            batch_size=16, val_frac=0.15, verbose=True,
            pod_loss_weight=1.0, full_loss_weight=1.0,
            patience_max=200):
        """Train on sim full state."""
        T = U_sim_flat.shape[0]
        S = U_sim_flat[:, sensor_idx].astype(np.float32)
        self.sc.fit(S)
        Sz = self.sc.transform(S).astype(np.float32)
        X = make_lagged_sensor_tensor(Sz, self.lags)
        Y_full = U_sim_flat[self.lags:].astype(np.float32)
        n = X.shape[0]

        if self.decoder_type == 'pod':
            Y_pod = (Y_full - self.u_mean.astype(np.float32)) \
                      @ self.Ub.astype(np.float32)
        else:
            Y_pod = None

        n_tr = int((1 - val_frac) * n)
        Xtr = torch.from_numpy(X[:n_tr]); Yf_tr = torch.from_numpy(Y_full[:n_tr])
        Xva = torch.from_numpy(X[n_tr:]); Yf_va = torch.from_numpy(Y_full[n_tr:])
        if self.decoder_type == 'pod':
            Yp_tr = torch.from_numpy(Y_pod[:n_tr])
            Yp_va = torch.from_numpy(Y_pod[n_tr:])

        opt = torch.optim.Adam(self.net.parameters(), lr=lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-4)
        best_va = float('inf'); best_sd = None; patience = 0

        for ep in range(n_epochs):
            self.net.train()
            idx = torch.randperm(len(Xtr))
            tr_loss = 0.0; nb = 0
            for i in range(0, len(idx), batch_size):
                b = idx[i:i+batch_size]
                xb = Xtr[b].to(DEV); yf = Yf_tr[b].to(DEV)
                opt.zero_grad()
                z = self.net.encode(xb)

                if self.decoder_type == 'pod':
                    yp = Yp_tr[b].to(DEV)
                    zp = self.net.to_pod(z)
                    yh = self.net.decoder(zp)
                    pod_l = ((zp - yp)**2).mean()
                    full_l = ((yh - yf)**2).mean()
                    loss = pod_loss_weight*pod_l + full_loss_weight*full_l
                else:  # mlp
                    yh = self.net.decoder(self.net.head(z))
                    full_l = ((yh - yf)**2).mean()
                    loss = full_l

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item(); nb += 1
            sched.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                z = self.net.encode(Xva.to(DEV))
                if self.decoder_type == 'pod':
                    zp = self.net.to_pod(z)
                    yh = self.net.decoder(zp)
                    va_full = ((yh - Yf_va.to(DEV))**2).mean().item()
                    va_pod = ((zp - Yp_va.to(DEV))**2).mean().item()
                else:
                    yh = self.net.decoder(self.net.head(z))
                    va_full = ((yh - Yf_va.to(DEV))**2).mean().item()
                    va_pod = float('nan')

            if va_full < best_va:
                best_va = va_full
                best_sd = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if verbose and (ep+1) % 100 == 0:
                pod_str = f" va_pod={va_pod:.4f}" if self.decoder_type == 'pod' else ""
                print(f"    ep {ep+1:4d}: tr={tr_loss/max(nb,1):.4f}"
                      f"  va_full={va_full:.4f}{pod_str}"
                      f"  lr={sched.get_last_lr()[0]:.2e}")
            if patience > patience_max:
                if verbose:
                    print(f"    early stop at ep {ep+1}  (best va_full={best_va:.5f})")
                break

        if best_sd is not None:
            self.net.load_state_dict(best_sd)
        return np.sqrt(best_va)

    @torch.no_grad()
    def reconstruct(self, S, noise_std=0.0, return_latent=False):
        """S: (T, p) sensors. Returns full-state reconstruction."""
        self.net.eval()
        Sn = S.astype(np.float32)
        if noise_std > 0:
            Sn = Sn + (noise_std * np.random.randn(*Sn.shape)).astype(np.float32)
        Sz = self.sc.transform(Sn).astype(np.float32)
        X = make_lagged_sensor_tensor(Sz, self.lags)
        Xt = torch.from_numpy(X).to(DEV)
        z = self.net.encode(Xt)
        y = self.net.decode(z)
        if return_latent:
            return y.cpu().numpy(), z.cpu().numpy()
        return y.cpu().numpy()


# ── DA-SHRED residual generator with sensor-loss + MMD alignment ──
class ResidualGenerator(nn.Module):
    def __init__(self, d_z=48, hidden=64, gamma_init=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, d_z), nn.Tanh(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        with torch.no_grad():
            for m in self.net.modules():
                pass
            last_lin = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
            last_lin.weight.mul_(0.01)
            last_lin.bias.zero_()

    def forward(self, z):
        return z + self.gamma * self.net(z)


def mmd_loss(x, y):
    """Unbiased MMD with a multi-bandwidth Gaussian kernel.
    Bandwidths are set by the median pairwise-distance heuristic * {0.5,1,2}."""
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    with torch.no_grad():
        med = torch.sqrt(0.5 * torch.median(
            torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])) + 1e-8)
    bws = [0.5 * med, med, 2.0 * med]
    loss = 0.0
    for bw in bws:
        k_xx = torch.exp(-xx / (2 * bw**2))
        k_yy = torch.exp(-yy / (2 * bw**2))
        k_xy = torch.exp(-xy / (2 * bw**2))
        loss = loss + k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return loss / len(bws)


class DASHRED:
    """Sensor-loss driven alignment via learned residual generator G.
    z_tilde = z + gamma * G(z), with SHRED encoder+decoder frozen.
    Loss = sensor-MSE + lambda_mmd * MMD(z_tilde_real, z_sim)
         + lambda_reg * ||gamma * G(z)||^2
    """
    def __init__(self, trainer, sensor_idx):
        self.trainer = trainer
        self.sf = np.asarray(sensor_idx)
        self.G = ResidualGenerator(d_z=trainer.net.d_z,
                                   gamma_init=0.05).to(DEV)

    def fit(self, S_real, S_sim, noise_std=0.01,
            lambda_mmd=0.05, lambda_reg=1e-3,
            n_epochs=600, lr=1e-3, batch_size=16, verbose=True):
        """Only real sensors and sim sensors are used. No full states."""
        # Freeze SHRED
        for p in self.trainer.net.parameters():
            p.requires_grad_(False)
        self.trainer.net.eval()

        # Pre-encode sim latents (fixed target distribution)
        _, Z_sim_np = self.trainer.reconstruct(S_sim, noise_std=0.0,
                                               return_latent=True)
        Z_sim = torch.from_numpy(Z_sim_np.astype(np.float32)).to(DEV)

        # Build lagged tensor for real sensors (single noisy draw pre-fit for reproducibility)
        S_real_n = S_real.astype(np.float32) + \
            (noise_std * np.random.randn(*S_real.shape)).astype(np.float32)
        Sz = self.trainer.sc.transform(S_real_n).astype(np.float32)
        Xreal = make_lagged_sensor_tensor(Sz, self.trainer.lags)
        Xreal_t = torch.from_numpy(Xreal).to(DEV)

        Y_sens = torch.from_numpy(
            S_real_n[self.trainer.lags:].astype(np.float32)).to(DEV)  # (T-lags, p)
        sens_mean = torch.from_numpy(self.trainer.sc.mean_.astype(np.float32)).to(DEV)
        sens_scale = torch.from_numpy(self.trainer.sc.scale_.astype(np.float32)).to(DEV)

        sf_t = torch.from_numpy(self.sf).long().to(DEV)

        opt = torch.optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=5e-5)

        n = Xreal_t.shape[0]

        # Pre-compute the no-DA sensor loss for reference
        with torch.no_grad():
            z_raw_all = self.trainer.net.encode(Xreal_t)
            u_all = self.trainer.net.decode(z_raw_all)
            s_all_z = (u_all[:, sf_t] - sens_mean) / sens_scale
            y_all_z = (Y_sens - sens_mean) / sens_scale
            noda_sens = ((s_all_z - y_all_z)**2).mean().item()
        if verbose:
            print(f"    no-DA sensor loss baseline: {noda_sens:.5f}")

        best_loss = float('inf'); best_sd = None
        for ep in range(n_epochs):
            self.G.train()
            idx = torch.randperm(n)
            tr_sens = 0.0; tr_mmd = 0.0; tr_reg = 0.0; nb = 0
            for i in range(0, n, batch_size):
                b = idx[i:i+batch_size]
                xb = Xreal_t[b]; yb = Y_sens[b]
                with torch.no_grad():
                    z_raw = self.trainer.net.encode(xb)
                delta = self.G.net(z_raw)          # before gamma scaling
                z_tilde = z_raw + self.G.gamma * delta
                u_hat = self.trainer.net.decode(z_tilde)
                s_hat = u_hat[:, sf_t]
                s_hat_z = (s_hat - sens_mean) / sens_scale
                yb_z = (yb - sens_mean) / sens_scale
                sens_l = ((s_hat_z - yb_z)**2).mean()

                if lambda_mmd > 0:
                    si = torch.randint(0, len(Z_sim), (min(len(Z_sim), 128),), device=DEV)
                    mmd = mmd_loss(z_tilde, Z_sim[si])
                else:
                    mmd = torch.tensor(0.0, device=DEV)

                reg = (self.G.gamma**2) * (delta**2).mean()

                loss = sens_l + lambda_mmd*mmd + lambda_reg*reg
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
                opt.step()
                tr_sens += sens_l.item(); tr_mmd += float(mmd.item())
                tr_reg += float(reg.item()); nb += 1
            sched.step()

            self.G.eval()
            with torch.no_grad():
                z_raw_all = self.trainer.net.encode(Xreal_t)
                z_til_all = self.G(z_raw_all)
                u_all = self.trainer.net.decode(z_til_all)
                s_all_z = (u_all[:, sf_t] - sens_mean) / sens_scale
                y_all_z = (Y_sens - sens_mean) / sens_scale
                full_sens = ((s_all_z - y_all_z)**2).mean().item()
            if full_sens < best_loss:
                best_loss = full_sens
                best_sd = {k: v.detach().clone() for k, v in self.G.state_dict().items()}

            if verbose and (ep+1) % 100 == 0:
                print(f"    ep {ep+1:4d}: sens={tr_sens/max(nb,1):.4f}"
                      f"  mmd={tr_mmd/max(nb,1):.4f}"
                      f"  reg={tr_reg/max(nb,1):.5f}"
                      f"  full_sens={full_sens:.5f}  gamma={self.G.gamma.item():.3f}")
        if best_sd is not None:
            self.G.load_state_dict(best_sd)
        if verbose:
            print(f"    best full_sens={best_loss:.5f}  (vs no-DA {noda_sens:.5f})")
        return best_loss

    @torch.no_grad()
    def reconstruct(self, S, noise_std=0.01):
        """Reconstruct full state from (real) sensor trajectory."""
        self.trainer.net.eval(); self.G.eval()
        Sn = S.astype(np.float32)
        if noise_std > 0:
            Sn = Sn + (noise_std * np.random.randn(*Sn.shape)).astype(np.float32)
        Sz = self.trainer.sc.transform(Sn).astype(np.float32)
        X = make_lagged_sensor_tensor(Sz, self.trainer.lags)
        Xt = torch.from_numpy(X).to(DEV)
        z_raw = self.trainer.net.encode(Xt)
        z_til = self.G(z_raw)
        u = self.trainer.net.decode(z_til)
        return u.cpu().numpy(), z_raw.cpu().numpy(), z_til.cpu().numpy()



# ── SINDy: forward-model-based discrepancy in POD space ──

def build_sindy_basis(snapshots, rank):
    """Build a POD basis for SINDy regression (decoder-agnostic)."""
    u_mean = snapshots.mean(axis=0)
    centered = snapshots - u_mean
    Usvd, _, _ = svd(centered.T, full_matrices=False)
    return Usvd[:, :rank], u_mean


def build_pod_library(Nx, Ny, dx, dy, Ub):
    """Return list of (name, L_POD) operators. L_POD is r-by-r and acts as
    z -> L @ z, where L_POD = Ub.T @ L_phys @ Ub.
    """
    kx = np.fft.fftfreq(Nx, d=dx)*2*np.pi
    ky = np.fft.fftfreq(Ny, d=dy)*2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    r = Ub.shape[1]

    damp = -np.eye(r)                                          # -u
    Dl = np.zeros((r, r))                                      # -grad^2 u
    for j in range(r):
        uj = Ub[:, j].reshape(Nx, Ny)
        op = np.real(ifft2(K2 * fft2(uj)))
        Dl[:, j] = Ub.T @ op.flatten()
    Dh = np.zeros((r, r))                                      # -grad^4 u
    for j in range(r):
        uj = Ub[:, j].reshape(Nx, Ny)
        op = np.real(ifft2(-K2**2 * fft2(uj)))
        Dh[:, j] = Ub.T @ op.flatten()

    return [
        ('-u           (damping)',   damp),
        ('-grad^2 u    (diffusion)', Dl),
        ('-grad^4 u    (hyper-diff)',Dh),
    ]


def sindy_stls_standardized(Theta, target, thr=0.05, mi=20, al=1e-3):
    """STLS with per-column standardization so thresholding is comparable."""
    T_mean = Theta.mean(axis=0, keepdims=True)
    T_std = Theta.std(axis=0, keepdims=True) + 1e-12
    Ts = (Theta - T_mean) / T_std
    if target.ndim == 1: target = target[:, None]
    XTX = Ts.T @ Ts + al * np.eye(Ts.shape[1])
    Xi_s = solve(XTX, Ts.T @ target)
    for _ in range(mi):
        small = np.abs(Xi_s) < thr
        Xi_s[small] = 0
        for j in range(target.shape[1]):
            big = ~small[:, j]
            if big.sum() > 0:
                A = Ts[:, big].T @ Ts[:, big] + al * np.eye(big.sum())
                Xi_s[big, j] = solve(A, Ts[:, big].T @ target[:, j])
    Xi = Xi_s / T_std.T
    return Xi


def forward_model_discrepancy(U_reconstructed, pde_sim, se, dts, Ub, Um,
                              midpoint=True):
    """Compute the missing-physics rate L'(u) via a 1-step forward-model comparison, expressed in POD coords.

    For midpoint=False (Euler):
        r(t) = (u'(t+dt) - F_sim(u'(t))) / dt   approx   L'(u'(t))
    For midpoint=True (trapezoidal, higher-order in dt):
        r(t) = (u'(t+dt) - F_sim(u'(t))) / dt   approx   (L'(u(t)) + L'(u(t+dt))) / 2
    and return z_r as the time-midpoint z (average).
    """
    T, N = U_reconstructed.shape
    Nx, Ny = pde_sim.Nx, pde_sim.Ny
    disc = np.zeros((T - 1, N))
    for t in range(T - 1):
        u_now = U_reconstructed[t].reshape(Nx, Ny)
        u_adv = pde_sim.step_physical(u_now, se)
        disc[t] = U_reconstructed[t+1] - u_adv.flatten()
    rate = disc / dts
    z_rate = rate @ Ub                                       # (T-1, r)
    z_now = (U_reconstructed[:-1] - Um) @ Ub
    z_next = (U_reconstructed[1:] - Um) @ Ub
    if midpoint:
        z_r = 0.5 * (z_now + z_next)
    else:
        z_r = z_now
    return z_rate, z_r


def forward_model_ensemble(shred, dashred, S_real, pde_sim, se, dts,
                           Ub, Um, noise_std=0.01, n_draws=10, midpoint=True,
                           verbose=True):
    """Average forward-model SINDy targets over n_draws noise realizations."""
    Z_rate_list = []
    Z_r_list = []
    rng_state = np.random.get_state()
    for k in range(n_draws):
        np.random.seed(10_000 + k)
        rc, _, _ = dashred.reconstruct(S_real, noise_std=noise_std)
        zr, zz = forward_model_discrepancy(rc, pde_sim, se, dts, Ub, Um,
                                           midpoint=midpoint)
        Z_rate_list.append(zr); Z_r_list.append(zz)
        if verbose:
            print(f"    ensemble draw {k+1}/{n_draws}: "
                  f"||rate||_rms={np.sqrt(np.mean(zr**2)):.3f}")
    np.random.set_state(rng_state)
    z_rate = np.mean(Z_rate_list, axis=0)                   # (T-1, r)
    z_r    = np.mean(Z_r_list, axis=0)                      # (T-1, r)
    return z_rate, z_r


def build_sindy_system_fwd(z_rate, z_r, lib):
    """Build (Theta, target) for STRONG-FORM forward-model-based SINDy.
    """
    T, r = z_r.shape
    nl = len(lib)
    Theta = np.zeros((T * r, nl))
    target = z_rate.flatten()
    for k in range(T):
        zk = z_r[k]
        for j, (_, Op) in enumerate(lib):
            Theta[k*r:(k+1)*r, j] = Op @ zk
    return Theta, target


def make_test_functions(n_samples, width, stride, dts, p=3):
    """Compactly-supported polynomial-bump test functions for weak-form SINDy.
    """
    half = width // 2
    centers = np.arange(half, n_samples - half, stride)
    K = len(centers)
    if K < 3:
        # Degenerate case (very short trajectory); fall back to 3 functions
        centers = np.linspace(half, n_samples - half - 1, 3, dtype=int)
        K = 3
    Phi = np.zeros((n_samples, K))
    t = np.arange(n_samples)
    for k, c in enumerate(centers):
        s = (t - c) / half
        mask = np.abs(s) <= 1
        Phi[mask, k] = (1.0 - s[mask] ** 2) ** p
        Phi[:, k] /= Phi[:, k].sum() * dts   # unit time-integral
    return Phi


def build_sindy_system_weak(z_rate, z_r, lib, Phi, dts):
    """Build (Theta, target) for WEAK-FORM forward-model SINDy.
    """
    T_eff, r = z_r.shape
    K = Phi.shape[1]
    nl = len(lib)
    PhiW = Phi * dts                              # (T_eff, K)

    target_KR = PhiW.T @ z_rate                   # (K, r)
    Theta = np.zeros((K * r, nl))
    for j, (_, L_j) in enumerate(lib):
        L_zr = z_r @ L_j.T                        # (T_eff, r)
        A_j = PhiW.T @ L_zr                       # (K, r)
        Theta[:, j] = A_j.flatten()
    target = target_KR.flatten()
    return Theta, target


def run_sindy(z_rate, z_r, lib, dts, mode='strong',
              weak_width_sec=4.0, weak_stride_sec=1.0, weak_p=3,
              stls_thr=None, stls_iter=30, stls_alpha=1e-3):
    """Dispatch strong-form or weak-form forward-model SINDy."""
    if stls_thr is None:
        stls_thr = 0.02 if mode == 'strong' else 0.005
    if mode == 'strong':
        Theta, target = build_sindy_system_fwd(z_rate, z_r, lib)
    elif mode == 'weak':
        T_eff = z_r.shape[0]
        width = max(4, int(round(weak_width_sec / dts)))
        stride = max(1, int(round(weak_stride_sec / dts)))
        if T_eff <= width + 2 * stride:
            width = max(4, T_eff // 3)
            stride = max(1, width // 4)
        Phi = make_test_functions(T_eff, width, stride, dts, p=weak_p)
        Theta, target = build_sindy_system_weak(z_rate, z_r, lib, Phi, dts)
    else:
        raise ValueError(f"Unknown SINDy mode: {mode!r}")
    xi = sindy_stls_standardized(Theta, target, thr=stls_thr,
                                 mi=stls_iter, al=stls_alpha)
    return xi, Theta, target

