"""Noise sweep: sensor noise sensitivity (sigma in {0, 0.01, 0.05, 0.10}).

32 sensors, 3 placements, se=5, nu=0.15.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runlog import start_run
from common2dks import build_case, score, PDECounter, Timer
from ensemble_da import StochasticEnKF, run_filter, climatological_ensemble
import spectral_sindy as sp

# Config

CFG = dict(Lx=16*np.pi, Ly=16*np.pi, Nx=64, Ny=64, dt=0.05,
           T=20.0, T_train=20.0, se=5, mu=0.02,
           nu_x=0.15, nu_y=0.15,
           n_sensors_side=7, noise=0.01, real_ic_jitter=0.0)

NOISE_VALUES     = [0.0, 0.01, 0.05, 0.10]
N_SENSORS        = 32
PLACEMENT_SEEDS  = [2000, 3000, 4000]

LAGS, RANK, D_Z, NE = 15, 30, 48, 50
INFLATION    = 1.05
SHRED_EPOCHS, DA_EPOCHS = 1200, 600
WEAK = dict(weak_width_sec=4.0, weak_stride_sec=1.0, weak_p=3)
KCUT = 1.0
SEED = 0


def random_sensors(nx, ny, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nx * ny, size=n, replace=False))


def ms(v):
    a = np.array([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def save_visualization(run, Urf, Usf, rc_da, rec_e, Nx, Ny, LAGS,
                       sigma, placement_idx):
    """Save reconstruction snapshot for one sigma value."""
    n = len(Urf)
    t_vis = LAGS + (n - LAGS) // 2
    real_field  = Urf[t_vis].reshape(Nx, Ny)
    sim_field   = Usf[t_vis].reshape(Nx, Ny)
    t_da = t_vis - LAGS
    if t_da < 0 or t_da >= len(rc_da):
        t_da = len(rc_da) // 2
    da_field = rc_da[t_da].reshape(Nx, Ny)
    enkf_field = rec_e[t_vis].reshape(Nx, Ny)

    vmin, vmax = real_field.min(), real_field.max()
    emax = max(abs(real_field - da_field).max(),
               abs(real_field - enkf_field).max(),
               abs(sim_field - real_field).max())

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    titles_top = ["Real (ground truth)", "DA-SHRED reconstruction",
                  "EnKF reconstruction"]
    fields_top = [real_field, da_field, enkf_field]
    for ax, title, f in zip(axes[0], titles_top, fields_top):
        im = ax.imshow(f.T, origin="lower", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0, -1], shrink=0.8, label="u")

    titles_bot = ["|Sim - Real|", "|DA-SHRED - Real|", "|EnKF - Real|"]
    errors_bot = [np.abs(sim_field - real_field),
                  np.abs(da_field - real_field),
                  np.abs(enkf_field - real_field)]
    for ax, title, e in zip(axes[1], titles_bot, errors_bot):
        im = ax.imshow(e.T, origin="lower", cmap="hot", vmin=0, vmax=emax)
        ax.set_title(f"{title}\nRMSE={np.sqrt(np.mean(e**2)):.4f}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, -1], shrink=0.8, label="|error|")

    fig.suptitle(f"Noise sweep  σ = {sigma:.3f}  "
                 f"(placement {placement_idx}, "
                 f"t = {t_vis * CFG['dt'] * CFG['se']:.1f}s)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"visualization_sigma_{sigma:.3f}.png"
    path = run.file(fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {fname}")


def main():
    full_cfg = {**CFG, "NOISE_VALUES": NOISE_VALUES, "N_SENSORS": N_SENSORS,
                "PLACEMENT_SEEDS": PLACEMENT_SEEDS,
                "LAGS": LAGS, "RANK": RANK, "D_Z": D_Z, "NE": NE, "SEED": SEED}

    with start_run("noise", config=full_cfg) as run:
        import dashred_neural_2dks as mod
        Nx, Ny, se = CFG["Nx"], CFG["Ny"], CFG["se"]
        dts = CFG["dt"] * se
        np.random.seed(SEED)

        # ── generate case (once; noise varies below) ──
        print("[1] Build case (sim-only, reused across sigma and placements)")
        case = build_case(mod.KS2D, CFG, np.random.default_rng(SEED))
        Usf, Urf, ks_sim = case["Usf"], case["Urf"], case["ks_sim"]
        gap, n = case["gap"], case["n"]
        print(f"    {n} frames, SIM2REAL gap {gap:.4f}, {N_SENSORS} sensors")

        # ── POD / SINDy basis ──
        nt = int(0.85 * len(case["Utrain"]))
        Utr = case["Utrain"][:nt]
        Um = Utr.mean(0)
        Ub = np.linalg.svd((Utr - Um).T, full_matrices=False)[0][:, :RANK]
        Ub_s, Um_s = mod.build_sindy_basis(Utr, RANK)
        lib = mod.build_pod_library(Nx, Ny, CFG["Lx"]/Nx, CFG["Ly"]/Ny,
                                    Ub_s)
        dif_j = next(i for i, (nm, _) in enumerate(lib) if "diffusion" in nm)

        # ── Pre-train one SHRED per placement (noise-independent) ──
        print("[2] Pre-training SHRED models for each placement")
        shred_list, sf_list = [], []
        for pi, pseed in enumerate(PLACEMENT_SEEDS):
            sf = random_sensors(Nx, Ny, N_SENSORS, seed=pseed)
            sf_list.append(sf)
            with Timer() as tm_tr:
                shred = mod.SHREDTrainer(n_sensors=N_SENSORS, d_z=D_Z,
                                         lags=LAGS, decoder_type="pod",
                                         Ub=Ub, u_mean=Um,
                                         head_hidden=(64,), dropout=0.1)
                shred.fit(Utr, sf, n_epochs=SHRED_EPOCHS, lr=3e-3,
                          batch_size=8, pod_loss_weight=1.0,
                          full_loss_weight=1.0, verbose=False)
            shred_list.append(shred)
            print(f"    placement {pi} (seed {pseed}): "
                  f"SHRED trained in {tm_tr.seconds:.0f}s")

        rows = []
        for sigma in NOISE_VALUES:
            print("\n" + "=" * 74)
            print(f"  sigma = {sigma:.3f}")
            print("=" * 74)

            # Floor for EnKF's R matrix
            sigma_enkf = max(sigma, 1e-6)

            per = {k: [] for k in
                   ("enkf", "nda", "da", "pde", "enkf_s", "train_s", "infer_s",
                    "nu_gt_w", "nu_da_w", "nu_en_w",
                    "nu_gt_sp", "nu_da_sp", "nu_en_sp",
                    "det_da_w", "det_en_w", "det_da_sp", "det_en_sp")}

            saved_vis = False

            for p, pseed in enumerate(PLACEMENT_SEEDS):
                sf = sf_list[p]
                shred = shred_list[p]
                S_real, S_sim = Urf[:, sf], Usf[:, sf]

                rng = np.random.default_rng(SEED)
                ys = Urf[:, sf] + rng.normal(0, sigma_enkf, (n, len(sf)))

                # ---- EnKF ----
                def fcast(ens, t, dt, _ks=ks_sim):
                    out = np.empty_like(ens)
                    for i in range(ens.shape[1]):
                        out[:, i] = _ks.step_physical(
                            ens[:, i].reshape(Nx, Ny), se).ravel()
                    return out
                f = StochasticEnKF(fcast, sf, sigma_enkf,
                                   inflation=INFLATION,
                                   rng=np.random.default_rng(SEED))
                with PDECounter(ks_sim) as pc, Timer() as tm_e:
                    rec_e, dg = run_filter(
                        f, climatological_ensemble(Utr, NE, rng),
                        ys, 0.0, dts, truth=Urf)
                r_e = score(rec_e, Urf, LAGS)

                # ---- DA-SHRED ----
                rc_nda = shred.reconstruct(S_real, noise_std=sigma)
                dashred = mod.DASHRED(shred, sf)
                with Timer() as tm_da:
                    dashred.fit(S_real, S_sim, noise_std=sigma,
                                lambda_mmd=0.05, lambda_reg=1e-3,
                                n_epochs=DA_EPOCHS, lr=1e-3, batch_size=16,
                                verbose=False)
                with Timer() as tm_inf:
                    rc_da, _, _ = dashred.reconstruct(S_real,
                                                       noise_std=sigma)
                r_nda, r_da = score(rc_nda, Urf, LAGS), score(rc_da, Urf, LAGS)

                per["enkf"].append(r_e); per["nda"].append(r_nda)
                per["da"].append(r_da); per["pde"].append(pc.n)
                per["enkf_s"].append(tm_e.seconds)
                per["train_s"].append(tm_da.seconds)
                per["infer_s"].append(tm_inf.seconds)

                # ---- SINDy ----
                for key, U in (("gt", Urf[LAGS:]), ("da", rc_da),
                               ("en", rec_e[LAGS:])):
                    zr, z = mod.forward_model_discrepancy(
                        U, ks_sim, se, dts, Ub_s, Um_s, midpoint=True)
                    xi = mod.run_sindy(zr, z, lib, dts, mode="weak",
                                       **WEAK)[0]
                    per[f"nu_{key}_w"].append(-float(xi[dif_j, 0]))
                    xs = sp.run_sindy_spectral(U, ks_sim, se, dts, kcut=KCUT,
                                               weak=True, **WEAK)[0]
                    per[f"nu_{key}_sp"].append(-float(xs[1]))

                # detection rates
                per["det_da_w"].append(
                    abs(per["nu_da_w"][-1]) > 1e-12)
                per["det_en_w"].append(
                    abs(per["nu_en_w"][-1]) > 1e-12)
                per["det_da_sp"].append(
                    abs(per["nu_da_sp"][-1]) > 1e-12)
                per["det_en_sp"].append(
                    abs(per["nu_en_sp"][-1]) > 1e-12)

                print(f"    placement {p} (seed {pseed}): "
                      f"EnKF {r_e:.4f}  DA-SHRED {r_da:.4f}  "
                      f"nu weak: GT {per['nu_gt_w'][-1]:+.4f} "
                      f"DA {per['nu_da_w'][-1]:+.4f} "
                      f"EnKF {per['nu_en_w'][-1]:+.4f}", flush=True)

                # ---- visualization for first placement per sigma ----
                if not saved_vis:
                    save_visualization(run, Urf, Usf, rc_da, rec_e,
                                       Nx, Ny, LAGS, sigma, p)
                    saved_vis = True

            agg = {k: ms(v) for k, v in per.items()}
            rows.append(dict(sigma=sigma, gap=gap, **agg))
            print(f"    mean+-sd  EnKF {agg['enkf'][0]:.4f}"
                  f"+-{agg['enkf'][1]:.4f}"
                  f"   DA-SHRED {agg['da'][0]:.4f}+-{agg['da'][1]:.4f}")
            print(f"    detection DA(w)={agg['det_da_w'][0]:.0%}  "
                  f"EnKF(w)={agg['det_en_w'][0]:.0%}")

        run.write_json("results.json", rows)

        # ── Summary ──
        TRUE_NU = CFG["nu_x"]
        print("\n" + "=" * 74)
        print(f"  NOISE SWEEP   nu = {TRUE_NU}, {N_SENSORS} sensors")
        print(f"  {len(PLACEMENT_SEEDS)} placements "
              f"(seeds {PLACEMENT_SEEDS})")
        print(f"  SIM2REAL gap = {gap:.4f}")
        print("=" * 74)

        print(f"\n  RECONSTRUCTION")
        print(f"  {'sigma':>8s} {'EnKF':>17s} {'SHRED no-DA':>17s} "
              f"{'DA-SHRED':>17s}")
        for r in rows:
            def cell(k):
                m, s = r[k]
                return f"{m:.4f}+-{s:.4f}{'*' if m > gap else ' '}"
            print(f"  {r['sigma']:8.3f} {cell('enkf'):>17s} "
                  f"{cell('nda'):>17s} {cell('da'):>17s}")
        print("  * = worse than ignoring the sensors (gap)")

        print(f"\n  DISCOVERY — recovered nu   (true = {TRUE_NU:+.2f})")
        print(f"  {'sigma':>8s} {'GT-fed (w)':>11s} {'DA (weak)':>11s} "
              f"{'EnKF (w)':>11s} {'GT (spec)':>11s} {'DA (spec)':>11s} "
              f"{'EnKF (sp)':>11s}")
        for r in rows:
            print(f"  {r['sigma']:8.3f} "
                  f"{r['nu_gt_w'][0]:+11.4f} {r['nu_da_w'][0]:+11.4f} "
                  f"{r['nu_en_w'][0]:+11.4f} {r['nu_gt_sp'][0]:+11.4f} "
                  f"{r['nu_da_sp'][0]:+11.4f} {r['nu_en_sp'][0]:+11.4f}")

        print(f"\n  DETECTION RATE (STLS retained diffusion term?)")
        print(f"  {'sigma':>8s} {'DA (w)':>8s} {'EnKF (w)':>8s} "
              f"{'DA (sp)':>8s} {'EnKF (sp)':>8s}")
        for r in rows:
            print(f"  {r['sigma']:8.3f} "
                  f"{r['det_da_w'][0]:8.0%} {r['det_en_w'][0]:8.0%} "
                  f"{r['det_da_sp'][0]:8.0%} {r['det_en_sp'][0]:8.0%}")

        print(f"\n  COST")
        print(f"  {'sigma':>8s} {'EnKF PDE':>10s} {'EnKF s':>8s} "
              f"{'DA fit s':>9s} {'infer s':>9s}")
        for r in rows:
            print(f"  {r['sigma']:8.3f} {int(r['pde'][0]):10d} "
                  f"{r['enkf_s'][0]:8.1f} {r['train_s'][0]:9.1f} "
                  f"{r['infer_s'][0]:9.3f}")


if __name__ == "__main__":
    main()
