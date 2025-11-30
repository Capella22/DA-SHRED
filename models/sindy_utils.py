# models/sindy_utils.py
# SINDy utility functions: library builder + enhanced STLSQ (threshold scheduling)
import numpy as np

def build_library(X, poly_order=2, include_sine=False, include_cos=False):
    """
    Build candidate library Theta from data X.
    X: (N, n_features)
    Returns Theta (N, n_lib), names list
    """
    N, n = X.shape
    Theta = [np.ones((N,1))]
    names = ['1']
    # linear
    for i in range(n):
        Theta.append(X[:, i:i+1])
        names.append(f'x{i}')
    # poly terms (combinatorial; limited to keep library moderate)
    if poly_order >= 2:
        for i in range(n):
            for j in range(i, n):
                Theta.append((X[:, i] * X[:, j])[:, None])
                names.append(f'x{i}*x{j}')
    if poly_order >= 3:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Theta.append((X[:, i] * X[:, j] * X[:, k])[:, None])
                    names.append(f'x{i}*x{j}*x{k}')
    if include_sine:
        for i in range(n):
            Theta.append(np.sin(X[:, i])[:, None])
            names.append(f'sin(x{i})')
    if include_cos:
        for i in range(n):
            Theta.append(np.cos(X[:, i])[:, None])
            names.append(f'cos(x{i})')
    Theta = np.concatenate(Theta, axis=1)
    return Theta, names

def stlsq(Theta, dXdt, lam=0.1, max_iter=10, thresh_schedule=None, verbose=False):
    """
    Enhanced STLSQ algorithm with optional threshold scheduling.
    Theta: (N, n_lib)
    dXdt: (N, n_targets)
    lam: final threshold value
    thresh_schedule: list/tuple of thresholds to apply sequentially (overrides lam if provided)
    Returns Xi: (n_lib, n_targets)
    """
    if thresh_schedule is None:
        thresh_schedule = [lam]
    # initial least squares solution
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    for tidx, cur_thresh in enumerate(thresh_schedule):
        if verbose:
            print(f"[STLSQ] schedule step {tidx+1}/{len(thresh_schedule)} threshold={cur_thresh}")
        small = np.abs(Xi) < cur_thresh
        Xi[small] = 0
        # refine coefficients on active set for each target column
        for j in range(dXdt.shape[1]):
            big_idx = ~small[:, j]
            if big_idx.sum() == 0:
                Xi[:, j] = 0
                continue
            # solve least squares using only active columns
            Xi[big_idx, j] = np.linalg.lstsq(Theta[:, big_idx], dXdt[:, j], rcond=None)[0]
    return Xi
