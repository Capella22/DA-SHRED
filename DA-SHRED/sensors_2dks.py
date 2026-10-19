"""Sensor sparsity ablation (Table 3).

Sweeps over sensor counts with repeated random placements,
comparing DA-SHRED reconstruction and SINDy discovery against EnKF.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from runlog import start_run
from common2dks import build_case, score, PDECounter, Timer
from ensemble_da import StochasticEnKF, run_filter, climatological_ensemble
import spectral_sindy as sp

# Config

CFG = dict(Lx=16*np.pi, Ly=16*np.pi, Nx=64, Ny=64, dt=0.05,
           T=20.0, T_train=20.0, se=5, mu=0.02,
           nu_x=0.15, nu_y=0.15,               # isotropic nu = 0.15
           n_sensors_side=7, noise=0.01, real_ic_jitter=0.0)

SENSOR_COUNTS = [4, 8, 12, 20, 32, 48]
N_PLACEMENTS = 8

LAGS, RANK, D_Z, NE = 15, 30, 48, 50
INFLATION = 1.05
SHRED_EPOCHS, DA_EPOCHS = 1200, 600
RUN_DISCOVERY = True
KCUT = 1.0
SEED = 0



def random_sensors(nx, ny, n, seed=0):
    """Uniform random placement without replacement."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nx*ny, size=n, replace=False))


def ms(v):
    a = np.array([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    with start_run("sensors_2dks",
                   config={**CFG, "SENSOR_COUNTS": SENSOR_COUNTS,
                           "N_PLACEMENTS": N_PLACEMENTS, "LAGS": LAGS,
                           "RANK": RANK, "NE": NE, "SEED": SEED}) as run:
        import dashred_neural_2dks as mod
        Nx, Ny, se = CFG["Nx"], CFG["Ny"], CFG["se"]
        dts = CFG["dt"]*se
        rng0 = np.random.default_rng(SEED)
        np.random.seed(SEED)

        print("[1] Case (generated once; sensors vary below)")
        case = build_case(mod.KS2D, CFG, rng0)
        Usf, Urf, ks_sim = case["Usf"], case["Urf"], case["ks_sim"]
        gap, n = case["gap"], case["n"]
        nt = int(0.85*len(case["Utrain"]))
        Utr = case["Utrain"][:nt]
        Um = Utr.mean(0)
        Ub = np.linalg.svd((Utr - Um).T, full_matrices=False)[0][:, :RANK]
        Ub_s, Um_s = mod.build_sindy_basis(Utr, RANK)
        lib = mod.build_pod_library(Nx, Ny, CFG["Lx"]/Nx, CFG["Ly"]/Ny, Ub_s)
        dif_j = next(i for i, (nm, _) in enumerate(lib) if "diffusion" in nm)
        print(f"    {n} frames, SIM2REAL gap {gap:.4f}  <- the bar to beat")
        print(f"    attractor: rank {RANK} POD captures 100% of sim energy")

        rows = []
        for ns in SENSOR_COUNTS:
            print("\n" + "="*74)
            print(f"  {ns} sensors  ({100*ns/(Nx*Ny):.2f}% of the field)   "
                  f"EnKF sees {ns}/analysis, SHRED sees {ns*LAGS}/reconstruction")
            print("="*74)
            per = {k: [] for k in ("enkf", "nda", "da", "nu_da", "nu_enkf",
                                   "nu_gt", "nu_da_sp", "nu_enkf_sp",
                                   "nu_gt_sp", "pde", "pde_shred",
                                   "enkf_s", "train_s", "infer_s")}
            for p in range(N_PLACEMENTS):
                sf = random_sensors(Nx, Ny, ns, seed=1000 + p)
                rng = np.random.default_rng(SEED + p)
                ys = Urf[:, sf] + rng.normal(0, CFG["noise"], (n, len(sf)))
                S_real, S_sim = Urf[:, sf], Usf[:, sf]

                def fcast(ens, t, dt, _ks=ks_sim):
                    out = np.empty_like(ens)
                    for i in range(ens.shape[1]):
                        out[:, i] = _ks.step_physical(
                            ens[:, i].reshape(Nx, Ny), se).ravel()
                    return out
                f = StochasticEnKF(fcast, sf, CFG["noise"],
                                   inflation=INFLATION,
                                   rng=np.random.default_rng(SEED + p))
                with PDECounter(ks_sim) as pc, Timer() as tm_e:
                    rec_e, dg = run_filter(f, climatological_ensemble(Utr, NE, rng),
                                           ys, 0.0, dts, truth=Urf)
                r_e = score(rec_e, Urf, LAGS)

                with PDECounter(ks_sim) as pc_s:
                    shred = mod.SHREDTrainer(n_sensors=ns, d_z=D_Z, lags=LAGS,
                                             decoder_type="pod", Ub=Ub, u_mean=Um,
                                             head_hidden=(64,), dropout=0.1)
                    with Timer() as tm_tr:
                        shred.fit(Utr, sf, n_epochs=SHRED_EPOCHS, lr=3e-3,
                                  batch_size=8, pod_loss_weight=1.0,
                                  full_loss_weight=1.0, verbose=False)
                    rc_nda = shred.reconstruct(S_real, noise_std=CFG["noise"])
                    dashred = mod.DASHRED(shred, sf)
                    with Timer() as tm_da:
                        dashred.fit(S_real, S_sim, noise_std=CFG["noise"],
                                    lambda_mmd=0.05, lambda_reg=1e-3,
                                    n_epochs=DA_EPOCHS, lr=1e-3, batch_size=16,
                                    verbose=False)
                    with Timer() as tm_inf:
                        rc_da, _, _ = dashred.reconstruct(S_real,
                                                          noise_std=CFG["noise"])
                r_nda, r_da = score(rc_nda, Urf, LAGS), score(rc_da, Urf, LAGS)

                per["enkf"].append(r_e); per["nda"].append(r_nda)
                per["da"].append(r_da); per["pde"].append(pc.n)
                per["pde_shred"].append(pc_s.n)
                per["enkf_s"].append(tm_e.seconds)
                per["train_s"].append(tm_tr.seconds + tm_da.seconds)
                per["infer_s"].append(tm_inf.seconds)

                if RUN_DISCOVERY:
                    for key, U in (("nu_gt", Urf[LAGS:]),
                                   ("nu_da", rc_da),
                                   ("nu_enkf", rec_e[LAGS:])):
                        zr, z = mod.forward_model_discrepancy(
                            U, ks_sim, se, dts, Ub_s, Um_s, midpoint=True)
                        xi = mod.run_sindy(zr, z, lib, dts, mode="weak",
                                           weak_width_sec=4.0,
                                           weak_stride_sec=1.0, weak_p=3)[0]
                        per[key].append(-float(xi[dif_j, 0]))
                        xs = sp.run_sindy_spectral(
                            U, ks_sim, se, dts, kcut=KCUT, weak=True,
                            weak_width_sec=4.0, weak_stride_sec=1.0,
                            weak_p=3)[0]
                        per[key + "_sp"].append(-float(xs[1]))

                print(f"    placement {p}: EnKF {r_e:.4f} ({tm_e.seconds:.1f}s, "
                      f"{pc.n} solves)   no-DA {r_nda:.4f}   "
                      f"DA-SHRED {r_da:.4f} (train {tm_tr.seconds + tm_da.seconds:.0f}s, "
                      f"infer {tm_inf.seconds:.2f}s, {pc_s.n} solves)", flush=True)

            agg = {k: ms(v) for k, v in per.items()}
            rows.append(dict(n_sensors=ns, shred_budget=ns*LAGS,
                             gap=gap, **{k: v for k, v in agg.items()}))
            print(f"    mean+-sd   EnKF {agg['enkf'][0]:.4f}+-{agg['enkf'][1]:.4f}"
                  f"   no-DA {agg['nda'][0]:.4f}+-{agg['nda'][1]:.4f}"
                  f"   DA-SHRED {agg['da'][0]:.4f}+-{agg['da'][1]:.4f}")
            print(f"    cost       EnKF {agg['enkf_s'][0]:.1f}s + "
                  f"{int(agg['pde'][0])} solves EVERY deployment   |   "
                  f"DA-SHRED train {agg['train_s'][0]:.0f}s ONCE, "
                  f"infer {agg['infer_s'][0]:.3f}s, "
                  f"{int(agg['pde_shred'][0])} solves")
            if RUN_DISCOVERY:
                print(f"    discovered nu: GT-fed {agg['nu_gt'][0]:+.4f}   "
                      f"DA-SHRED {agg['nu_da'][0]:+.4f}   "
                      f"EnKF {agg['nu_enkf'][0]:+.4f}   (true +0.150)")

        run.write_json("results.json", rows)

        # ---------------- summary ------------------------------------------
        print("\n" + "="*74)
        print("  RECONSTRUCTION vs SENSOR COUNT   (mean +- sd over "
              f"{N_PLACEMENTS} placements)")
        print("="*74)
        print(f"  {'sensors':>7s} {'SHRED budget':>13s} {'EnKF':>16s} "
              f"{'SHRED no-DA':>16s} {'DA-SHRED':>16s}")
        for r in rows:
            def cell(k):
                m, s = r[k]
                flag = "*" if m > r["gap"] else " "
                return f"{m:.4f}+-{s:.4f}{flag}"
            print(f"  {r['n_sensors']:7d} {r['shred_budget']:13d} "
                  f"{cell('enkf'):>16s} {cell('nda'):>16s} {cell('da'):>16s}")
        print(f"\n  * = worse than the SIM2REAL gap ({gap:.4f})")

        # ---------------- cost ---------------------------------------------
        print("\n" + "="*74)
        print("  COST   (solves are the structural metric; seconds are secondary)")
        print("="*74)
        print(f"  {'sensors':>7s} {'EnKF solves':>12s} {'EnKF s':>9s} "
              f"{'SHRED solves':>13s} {'train s':>9s} {'infer s':>9s}")
        for r in rows:
            print(f"  {r['n_sensors']:7d} {int(r['pde'][0]):12d} "
                  f"{r['enkf_s'][0]:9.1f} {int(r['pde_shred'][0]):13d} "
                  f"{r['train_s'][0]:9.0f} {r['infer_s'][0]:9.3f}")

        # ---------------- discovery ----------------------------------------
        if RUN_DISCOVERY:
            print("\n" + "="*74)
            print("  DISCOVERY vs SENSOR COUNT   (inflation over the GT-fed "
                  "reference)")
            print("="*74)
            for tag, lab in (("", "POD-30 regression (current)"),
                             ("_sp", f"spectral regression, k <= {KCUT:g}")):
                print(f"\n  --- {lab} ---")
                print(f"  {'sensors':>7s} {'GT-fed':>9s} {'DA-SHRED':>18s} "
                      f"{'infl':>8s} {'EnKF':>18s} {'infl':>8s}")
                for r in rows:
                    g = r["nu_gt"+tag][0]
                    print(f"  {r['n_sensors']:7d} {g:+9.4f} "
                          f"{r['nu_da'+tag][0]:+11.4f}"
                          f"+-{r['nu_da'+tag][1]:.4f} "
                          f"{r['nu_da'+tag][0]-g:+8.4f} "
                          f"{r['nu_enkf'+tag][0]:+11.4f}"
                          f"+-{r['nu_enkf'+tag][1]:.4f} "
                          f"{r['nu_enkf'+tag][0]-g:+8.4f}")


if __name__ == "__main__":
    main()
