"""Spectral forward-model SINDy for Fourier-diagonal libraries.

Drop-in alternative to POD-projected regression.
"""

import numpy as np
from numpy.fft import fft2, ifft2

# Fourier multipliers of the standard three-term library, in the same order
# and with the same signs as build_pod_library:
#   ('-u  (damping)', '-grad^2 u  (diffusion)', '-grad^4 u  (hyper-diff)')
# so the recovered nu is -xi[1], exactly as in the POD path.
LIB_NAMES = ('-u           (damping)',
             '-grad^2 u    (diffusion)',
             '-grad^4 u    (hyper-diff)')


def _multipliers(K2):
    return (-np.ones_like(K2), K2, -K2**2)


def forward_residual(U, ks_sim, se, dts, midpoint=True):
    """Same forward-model discrepancy as forward_model_discrepancy, but kept in physical space instead of being projected onto a POD basis.

    r(t) = (u(t+dt) - F_sim(u(t))) / dt   ~   L'(u)
    """
    Nx, Ny = ks_sim.Nx, ks_sim.Ny
    T = len(U)
    r = np.empty((T-1, Nx*Ny))
    for t in range(T-1):
        adv = ks_sim.step_physical(U[t].reshape(Nx, Ny), se)
        r[t] = U[t+1] - adv.ravel()
    r /= dts
    umid = 0.5*(U[:-1] + U[1:]) if midpoint else U[:-1]
    return r, umid


def make_test_functions(n, width, stride, dts, p=3):
    half = max(2, width//2)
    centers = np.arange(half, n-half, max(1, stride))
    if len(centers) < 3:
        centers = np.linspace(half, max(half+1, n-half-1), 3, dtype=int)
    Phi = np.zeros((n, len(centers)))
    t = np.arange(n)
    for k, c in enumerate(centers):
        s = (t-c)/half
        m = np.abs(s) <= 1
        Phi[m, k] = (1.0 - s[m]**2)**p
        Phi[:, k] /= (Phi[:, k].sum()*dts)
    return Phi


def build_spectral_system(r, umid, ks_sim, dts, kcut=np.inf, weak=True,
                          weak_width_sec=4.0, weak_stride_sec=1.0, weak_p=3):
    """Assemble (Theta, target) for the three-term library in Fourier space.

    One row per (retained wavenumber, time or test function), with real and imaginary parts stacked. No projection, no truncation beyond `kcut`.
    """
    Nx, Ny = ks_sim.Nx, ks_sim.Ny
    K2 = ks_sim.KX**2 + ks_sim.KY**2
    keep = (np.sqrt(K2) <= kcut).ravel()
    mults = [m.ravel()[keep] for m in _multipliers(K2)]

    T = len(r)
    Rh = np.empty((T, int(keep.sum())), complex)
    Uh = np.empty_like(Rh)
    for t in range(T):
        Rh[t] = fft2(r[t].reshape(Nx, Ny)).ravel()[keep]
        Uh[t] = fft2(umid[t].reshape(Nx, Ny)).ravel()[keep]

    if weak:
        w = max(4, int(round(weak_width_sec/dts)))
        st = max(1, int(round(weak_stride_sec/dts)))
        Phi = make_test_functions(T, w, st, dts, p=weak_p)*dts
        Rh, Uh = Phi.T @ Rh, Phi.T @ Uh          # integrate in time

    target = Rh.ravel()
    Theta = np.stack([(Uh*m[None, :]).ravel() for m in mults], axis=1)
    return (np.vstack([Theta.real, Theta.imag]),
            np.concatenate([target.real, target.imag]))


def stls_standardized(Theta, target, thr=0.005, mi=30, al=1e-3):
    sc = np.linalg.norm(Theta, axis=0)
    sc[sc == 0] = 1.0
    A = Theta/sc
    n = A.shape[1]
    xi = np.linalg.solve(A.T@A + al*np.eye(n), A.T@target)
    for _ in range(mi):
        small = np.abs(xi) < thr
        if small.all() or not small.any():
            break
        xi[small] = 0.0
        big = ~small
        xi[big] = np.linalg.solve(A[:, big].T@A[:, big] + al*np.eye(big.sum()),
                                  A[:, big].T@target)
    return xi/sc


def run_sindy_spectral(U, ks_sim, se, dts, kcut=np.inf, weak=True,
                       stls_thr=0.005, **kw):
    """Returns (xi, Theta, target). Recovered nu is -xi[1], as in the POD path."""
    r, umid = forward_residual(U, ks_sim, se, dts)
    Theta, target = build_spectral_system(r, umid, ks_sim, dts, kcut=kcut,
                                          weak=weak, **kw)
    return stls_standardized(Theta, target, thr=stls_thr), Theta, target


