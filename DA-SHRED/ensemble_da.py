"""Ensemble data assimilation: stochastic EnKF with optional localization.

State ensemble is (n, k): n state variables, k members.
"""

from __future__ import annotations

import numpy as np


# ── Localization ──

def gaspari_cohn(r, c):
    """Gaspari-Cohn (1999) 5th-order piecewise rational function.

    Compact support on |r| <= 2c, equals 1 at r=0 and 5/24 at r=c.
    """
    r = np.abs(np.asarray(r, float)) / max(float(c), 1e-30)
    out = np.zeros_like(r)
    m = r <= 1
    z = r[m]
    out[m] = -0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3 - 5.0/3.0 * z**2 + 1.0
    m = (r > 1) & (r <= 2)
    z = np.maximum(r[m], 1e-12)
    out[m] = (z**5 / 12.0 - 0.5 * z**4 + 0.625 * z**3 + 5.0/3.0 * z**2
              - 5.0 * z + 4.0 - 2.0 / (3.0 * z))
    return np.clip(out, 0.0, 1.0)


def torus_distance_1d(n, obs_idx, L):
    x = np.arange(n) * (L / n)
    xo = x[np.asarray(obs_idx)]
    d = np.abs(x[:, None] - xo[None, :])
    return np.minimum(d, L - d)


def torus_distance_2d(nx, ny, obs_idx, Lx, Ly):
    """(nx*ny, n_obs) distances on a periodic 2D grid, C-ordered flattening.
    """
    ii, jj = np.divmod(np.arange(nx * ny), ny)
    x, y = ii * (Lx / nx), jj * (Ly / ny)
    si, sj = np.divmod(np.asarray(obs_idx), ny)
    sx, sy = si * (Lx / nx), sj * (Ly / ny)
    dx = np.abs(x[:, None] - sx[None, :]); dx = np.minimum(dx, Lx - dx)
    dy = np.abs(y[:, None] - sy[None, :]); dy = np.minimum(dy, Ly - dy)
    return np.hypot(dx, dy)


# ── Initialization ──

def climatological_ensemble(snapshots, k, rng, mean=None, scale=1.0):
    """Initial ensemble drawn from model climatology.
    """
    X = np.asarray(snapshots).reshape(len(snapshots), -1)
    m = X.mean(0) if mean is None else np.asarray(mean).ravel()
    A = X - X.mean(0)
    idx = rng.integers(0, len(A), size=k)
    return m[:, None] + scale * A[idx].T



# ── Diagnostics ──

class Diagnostics:
    """Per-step RMSE, ensemble spread, and normalized innovation.
    """

    def __init__(self):
        self.rmse, self.spread, self.innov_norm, self.diverged = [], [], [], False

    def record(self, ens, truth=None, innovation=None, R_diag=None, HPHt=None):
        k = ens.shape[1]
        if truth is not None:
            self.rmse.append(float(np.sqrt(np.mean((ens.mean(1) - truth) ** 2))))
        self.spread.append(float(np.sqrt(np.mean(ens.var(1, ddof=1)))))
        if innovation is not None and R_diag is not None:
            denom = R_diag + (HPHt if HPHt is not None else 0.0)
            self.innov_norm.append(float(np.mean(innovation ** 2 / denom)))

    def summary(self):
        r = np.array(self.rmse); s = np.array(self.spread)
        out = dict(rmse=float(np.sqrt(np.mean(r ** 2))) if len(r) else np.nan,
                   rmse_final=float(r[-1]) if len(r) else np.nan,
                   spread=float(np.mean(s)) if len(s) else np.nan,
                   diverged=bool(self.diverged))
        if len(r) and len(s):
            out["spread_skill_ratio"] = float(np.mean(s) / (np.sqrt(np.mean(r**2)) + 1e-30))
        if self.innov_norm:
            out["innovation_variance"] = float(np.mean(self.innov_norm))
        return out


# ── Filters ──

class _Base:
    def __init__(self, forecast, obs_idx, sigma_obs, inflation=1.0,
                 rtps=0.0, rng=None):
        self.forecast_fn = forecast
        self.obs = np.asarray(obs_idx)
        self.R_diag = np.full(len(self.obs), float(sigma_obs) ** 2)
        self.inflation = float(inflation)
        self.rtps = float(rtps)
        self.rng = rng or np.random.default_rng(0)
        self.diag = Diagnostics()

    def forecast(self, ens, t, dt):
        out = self.forecast_fn(ens, t, dt)
        if not np.all(np.isfinite(out)) or np.abs(out).max() > 1e10:
            self.diag.diverged = True
        return out

    def _inflate(self, ens):
        if self.inflation == 1.0:
            return ens
        m = ens.mean(1, keepdims=True)
        return m + self.inflation * (ens - m)

    def _relax(self, ens_a, sd_b):
        """Relaxation to prior spread (Whitaker & Hamill 2012).
        """
        if self.rtps <= 0:
            return ens_a
        m = ens_a.mean(1, keepdims=True)
        sd_a = ens_a.std(1, ddof=1, keepdims=True)
        f = 1.0 + self.rtps * (sd_b - sd_a) / (sd_a + 1e-30)
        return m + f * (ens_a - m)


class StochasticEnKF(_Base):
    """Perturbed-observation EnKF (Evensen 1994; Burgers et al. 1998).

    Optionally localized by tapering the sample covariances with a Gaspari-Cohn function of distance.
    """

    def __init__(self, *a, loc_state_obs=None, loc_obs_obs=None, **kw):
        super().__init__(*a, **kw)
        self.Lso, self.Loo = loc_state_obs, loc_obs_obs

    def analyze(self, ens, y):
        k = ens.shape[1]
        sd_b = ens.std(1, ddof=1, keepdims=True)
        ens = self._inflate(ens)
        Y = ens[self.obs, :]
        xm, ym = ens.mean(1, keepdims=True), Y.mean(1, keepdims=True)
        A, D = ens - xm, Y - ym
        Pxy = A @ D.T / (k - 1)
        Pyy = D @ D.T / (k - 1)
        if self.Lso is not None:
            Pxy = Pxy * self.Lso
            Pyy = Pyy * self.Loo
        S = Pyy + np.diag(self.R_diag)
        innov = y - ym.ravel()
        self.diag.record(ens, innovation=innov, R_diag=self.R_diag,
                         HPHt=np.diag(Pyy))
        K = Pxy @ np.linalg.pinv(S)
        pert = self.rng.normal(0.0, np.sqrt(self.R_diag)[:, None], size=(len(self.obs), k))
        ens_a = ens + K @ (y[:, None] + pert - Y)
        return self._relax(ens_a, sd_b)


# ── Driver ──

def run_filter(filt, ens0, ys, t0, dt, truth=None):
    """Sequential forecast/analysis loop. Returns (analysis_means, diagnostics)."""
    ens = ens0
    rec = np.empty((len(ys), ens.shape[0]))
    for step, y in enumerate(ys):
        if step:
            ens = filt.forecast(ens, t0 + (step - 1) * dt, dt)
            if filt.diag.diverged:
                rec[step:] = np.nan
                break
        ens = filt.analyze(ens, y)
        if not np.all(np.isfinite(ens)):
            filt.diag.diverged = True
            rec[step:] = np.nan
            break
        rec[step] = ens.mean(1)
        if truth is not None:
            filt.diag.rmse.append(float(np.sqrt(np.mean((rec[step] - truth[step]) ** 2))))
    return rec, filt.diag
