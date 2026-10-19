"""Shared setup for the 2D-KS experiments.

Case construction, scoring, PDE-call counter, and off-manifold diagnostic.
"""

import time
import numpy as np


# ── Anisotropic KS ──

def make_ks(KS2D, Lx, Ly, Nx, Ny, mu, dt, nu_x=0.0, nu_y=0.0):
    """KS2D with independent x/y diffusion.
    """
    ks = KS2D(Lx, Ly, Nx, Ny, mu=mu, nu_extra=0.0, dt=dt)
    if nu_x == 0.0 and nu_y == 0.0:
        return ks
    KX, KY = ks.KX, ks.KY
    K2 = KX**2 + KY**2
    L = K2 - K2**2 - mu - (nu_x * KX**2 + nu_y * KY**2)
    h = ks.dt
    ks.E, ks.E2 = np.exp(h*L), np.exp(h*L/2)
    M = 32
    th = np.pi*(np.arange(1, M+1) - 0.5)/M
    r = np.exp(1j*th)
    LR = h*L[:, :, None] + r[None, None, :]
    ks.Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=2))
    ks.f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=2))
    ks.f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=2))
    ks.f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=2))
    return ks


# ── Cost accounting ──

class PDECounter:
    """Wraps `step_physical` to count forward-model evaluations.
    """

    def __init__(self, ks):
        self.ks, self.n = ks, 0
        self._orig = ks.step_physical

    def __enter__(self):
        def counted(u, n_sub):
            self.n += 1
            return self._orig(u, n_sub)
        self.ks.step_physical = counted
        return self

    def __exit__(self, *a):
        self.ks.step_physical = self._orig
        return False


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
        return False


# ── Case construction ──

def build_case(KS2D, cfg, rng):
    """Generate sim and real trajectories, sensors, and observations.
    """
    Lx, Ly, Nx, Ny = cfg["Lx"], cfg["Ly"], cfg["Nx"], cfg["Ny"]
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = (np.cos(2*np.pi*X/Lx)*np.cos(2*np.pi*Y/Ly)
          + 0.5*np.sin(4*np.pi*X/Lx)*np.sin(2*np.pi*Y/Ly))
    u0 /= np.abs(u0).max()
    u0_real = u0.copy()
    if cfg.get("real_ic_jitter", 0.0) > 0:
        u0_real = u0 + cfg["real_ic_jitter"]*rng.standard_normal(u0.shape)

    ks_sim = make_ks(KS2D, Lx, Ly, Nx, Ny, cfg["mu"], cfg["dt"])
    ks_real = make_ks(KS2D, Lx, Ly, Nx, Ny, cfg["mu"], cfg["dt"],
                      nu_x=cfg.get("nu_x", 0.0), nu_y=cfg.get("nu_y", 0.0))

    U_sim_tr = ks_sim.simulate(u0, cfg["T_train"], cfg["se"], verbose=False)
    U_sim = ks_sim.simulate(u0, cfg["T"], cfg["se"], verbose=False)
    U_real = ks_real.simulate(u0_real, cfg["T"], cfg["se"], verbose=False)
    n = min(len(U_sim), len(U_real))
    Usf, Urf = U_sim[:n].reshape(n, -1), U_real[:n].reshape(n, -1)
    Utr = U_sim_tr.reshape(len(U_sim_tr), -1)

    ns = cfg["n_sensors_side"]
    si = np.linspace(0, Nx-1, ns, dtype=int)
    sj = np.linspace(0, Ny-1, ns, dtype=int)
    sf = np.array([i*Ny + j for i in si for j in sj])
    ys = Urf[:, sf] + rng.normal(0, cfg["noise"], (n, len(sf)))

    return dict(ks_sim=ks_sim, ks_real=ks_real, Usf=Usf, Urf=Urf, Utrain=Utr,
                sf=sf, ys=ys, n=n, u0=u0,
                gap=float(np.sqrt(np.mean((Usf - Urf)**2))),
                rms_real=float(np.sqrt(np.mean(Urf**2))))


def score(rec, truth, lags):
    a, b = rec[-(len(truth)-lags):], truth[lags:]
    m = min(len(a), len(b))
    v = np.all(np.isfinite(a[-m:]), axis=1)
    if v.sum() < 2:
        return float("inf")
    return float(np.sqrt(np.mean((a[-m:][v] - b[-m:][v])**2)))


# ── Off-manifold diagnostic ──

def laplacian(field, ks):
    K2 = ks.KX**2 + ks.KY**2
    return np.real(np.fft.ifft2(-K2*np.fft.fft2(field)))


def biharmonic(field, ks):
    K2 = ks.KX**2 + ks.KY**2
    return np.real(np.fft.ifft2(K2**2*np.fft.fft2(field)))


def offmanifold(basis, mean, fields):
    """||(I - P)f|| / ||f|| for each field, P = basis projector.
    """
    out = {}
    for name, F in fields.items():
        Fc = F - (mean if name == "u" else 0.0)
        P = (Fc @ basis) @ basis.T
        out[name] = float(np.linalg.norm(Fc - P) / (np.linalg.norm(Fc) + 1e-30))
    return out


def derivative_fields(Uflat, ks, Nx, Ny):
    n = len(Uflat)
    lap = np.empty_like(Uflat); bih = np.empty_like(Uflat)
    for i in range(n):
        f = Uflat[i].reshape(Nx, Ny)
        lap[i] = laplacian(f, ks).ravel()
        bih[i] = biharmonic(f, ks).ravel()
    return {"u": Uflat, "grad2_u": lap, "grad4_u": bih}
