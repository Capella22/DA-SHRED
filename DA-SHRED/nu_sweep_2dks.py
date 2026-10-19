"""Viscosity (nu_extra) sweep (Table 4).

Sweeps nu_extra over [0.05, 1.00], comparing DA-SHRED and EnKF on reconstruction, weak/strong/spectral SINDy discovery, and tracking slope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from runlog import start_run
from common2dks import (build_case, score, PDECounter, Timer,
                        offmanifold, derivative_fields)
from ensemble_da import StochasticEnKF, run_filter, climatological_ensemble
import spectral_sindy as sp

# Config

CFG = dict(Lx=16*np.pi, Ly=16*np.pi, Nx=64, Ny=64, dt=0.05,
           T=20.0, T_train=20.0, se=5, mu=0.02,
           nu_x=0.15, nu_y=0.15,               # overwritten per sweep point
           n_sensors_side=7, noise=0.01, real_ic_jitter=0.0)

NU_VALUES = [0.05, 0.15, 0.25, 0.50, 0.75, 1.00]
N_SENSORS = 8
PLACEMENT_SEED = 1000
N_SEEDS = 3

LAGS, RANK, D_Z, NE = 15, 30, 48, 50
INFLATION = 1.05
SHRED_EPOCHS, DA_EPOCHS = 1200, 600
WEAK = dict(weak_width_sec=4.0, weak_stride_sec=1.0, weak_p=3)
KCUT = 1.0
SEED = 0



def random_sensors(nx, ny, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nx*ny, size=n, replace=False))


def ms(v):
    a = np.array([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    with start_run("nu_sweep_2dks",
                   config={**CFG, "NU_VALUES": NU_VALUES,
                           "N_SENSORS": N_SENSORS,
                           "PLACEMENT_SEED": PLACEMENT_SEED,
                           "N_SEEDS": N_SEEDS, "LAGS": LAGS, "RANK": RANK,
                           "D_Z": D_Z, "NE": NE, "SEED": SEED}) as run:
        import dashred_neural_2dks as mod
        Nx, Ny, se = CFG["Nx"], CFG["Ny"], CFG["se"]
        dts = CFG["dt"]*se
        np.random.seed(SEED)
        sf = random_sensors(Nx, Ny, N_SENSORS, seed=PLACEMENT_SEED)

        print("[1] SHRED (sim-only; trained once, reused at every nu)")
        case0 = build_case(mod.KS2D, {**CFG, "nu_x": NU_VALUES[0],
                                      "nu_y": NU_VALUES[0]},
                           np.random.default_rng(SEED))
        nt = int(0.85*len(case0["Utrain"]))
        Utr = case0["Utrain"][:nt]
        Um = Utr.mean(0)
        Ub = np.linalg.svd((Utr - Um).T, full_matrices=False)[0][:, :RANK]
        Ub_s, Um_s = mod.build_sindy_basis(Utr, RANK)
        lib = mod.build_pod_library(Nx, Ny, CFG["Lx"]/Nx, CFG["Ly"]/Ny, Ub_s)
        dif_j = next(i for i, (nm, _) in enumerate(lib) if "diffusion" in nm)

        with Timer() as tm_tr:
            shred = mod.SHREDTrainer(n_sensors=N_SENSORS, d_z=D_Z, lags=LAGS,
                                     decoder_type="pod", Ub=Ub, u_mean=Um,
                                     head_hidden=(64,), dropout=0.1)
            shred.fit(Utr, sf, n_epochs=SHRED_EPOCHS, lr=3e-3, batch_size=8,
                      pod_loss_weight=1.0, full_loss_weight=1.0, verbose=False)
        print(f"    trained in {tm_tr.seconds:.0f}s on {N_SENSORS} sensors "
              f"(placement seed {PLACEMENT_SEED})")

        rows = []
        for nu in NU_VALUES:
            print("\n" + "="*74)
            print(f"  nu_extra = {nu:.2f}")
            print("="*74)
            cfg_nu = {**CFG, "nu_x": nu, "nu_y": nu}
            case = build_case(mod.KS2D, cfg_nu, np.random.default_rng(SEED))
            assert np.allclose(case["Utrain"], case0["Utrain"]), \
                "sim trajectory changed with nu — SHRED reuse is invalid"
            Usf, Urf, ks_sim = case["Usf"], case["Urf"], case["ks_sim"]
            gap, n = case["gap"], case["n"]
            S_real, S_sim = Urf[:, sf], Usf[:, sf]

            tvar = float(np.mean(np.var(Urf, axis=0)))
            tvar_sim = float(np.mean(np.var(Usf, axis=0)))
            relmag = float(np.sqrt(np.mean((Usf - Urf)**2))
                           / (np.sqrt(np.mean(Urf**2)) + 1e-30))
            alive = tvar > 0.05*tvar_sim
            print(f"    SIM2REAL gap {gap:.4f}   relmag {relmag:.4f}   "
                  f"temporal var real/sim {tvar/tvar_sim:.3f}"
                  f"{'' if alive else '   [DYNAMICS DEAD — do not interpret]'}")

            om = offmanifold(Ub_s, Um_s,
                             derivative_fields(Urf, ks_sim, Nx, Ny))
            print(f"    off-manifold vs sim basis:  u {100*om['u']:5.1f}%   "
                  f"grad^2 u {100*om['grad2_u']:5.1f}%   "
                  f"grad^4 u {100*om['grad4_u']:5.1f}%")

            per = {k: [] for k in
                   ("enkf", "nda", "da", "enkf_s", "train_s", "infer_s", "pde",
                    "gt_w", "da_w", "en_w", "gt_s", "da_s", "en_s",
                    "gt_sp", "da_sp", "en_sp",
                    "det_da_w", "det_da_s", "det_en_w")}

            for p in range(N_SEEDS):
                rng = np.random.default_rng(SEED + p)
                ys = Urf[:, sf] + rng.normal(0, CFG["noise"], (n, len(sf)))

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
                    rec_e, dg = run_filter(
                        f, climatological_ensemble(Utr, NE, rng),
                        ys, 0.0, dts, truth=Urf)
                r_e = score(rec_e, Urf, LAGS)

                rc_nda = shred.reconstruct(S_real, noise_std=CFG["noise"])
                dashred = mod.DASHRED(shred, sf)
                with Timer() as tm_da:
                    dashred.fit(S_real, S_sim, noise_std=CFG["noise"],
                                lambda_mmd=0.05, lambda_reg=1e-3,
                                n_epochs=DA_EPOCHS, lr=1e-3, batch_size=16,
                                verbose=False)
                with Timer() as tm_i:
                    rc_da, _, _ = dashred.reconstruct(S_real,
                                                      noise_std=CFG["noise"])
                r_nda, r_da = score(rc_nda, Urf, LAGS), score(rc_da, Urf, LAGS)

                # ---- discovery: same (zr, z) solved BOTH ways -------------
                out = {}
                for key, U in (("gt", Urf[LAGS:]), ("da", rc_da),
                               ("en", rec_e[LAGS:])):
                    zr, z = mod.forward_model_discrepancy(
                        U, ks_sim, se, dts, Ub_s, Um_s, midpoint=True)
                    for mode, tag in (("weak", "w"), ("strong", "s")):
                        xi = mod.run_sindy(zr, z, lib, dts, mode=mode,
                                           **WEAK)[0]
                        out[f"{key}_{tag}"] = -float(xi[dif_j, 0])
                    xs = sp.run_sindy_spectral(U, ks_sim, se, dts, kcut=KCUT,
                                               weak=True, **WEAK)[0]
                    out[f"{key}_sp"] = -float(xs[1])
                for k, v in out.items():
                    per[k].append(v)
                # detection = did STLS retain the term at all?
                per["det_da_w"].append(abs(out["da_w"]) > 1e-12)
                per["det_da_s"].append(abs(out["da_s"]) > 1e-12)
                per["det_en_w"].append(abs(out["en_w"]) > 1e-12)

                per["enkf"].append(r_e); per["nda"].append(r_nda)
                per["da"].append(r_da); per["pde"].append(pc.n)
                per["enkf_s"].append(tm_e.seconds)
                per["train_s"].append(tm_da.seconds)
                per["infer_s"].append(tm_i.seconds)
                print(f"    seed {p}: EnKF {r_e:.4f} ({tm_e.seconds:.1f}s)   "
                      f"no-DA {r_nda:.4f}   DA-SHRED {r_da:.4f}   |   "
                      f"nu weak: GT {out['gt_w']:+.4f}  DA {out['da_w']:+.4f}  "
                      f"EnKF {out['en_w']:+.4f}", flush=True)

            agg = {k: ms([float(x) for x in v]) for k, v in per.items()}
            rows.append(dict(nu=nu, gap=gap, relmag=relmag, alive=bool(alive),
                             offman={k: float(v) for k, v in om.items()},
                             **{k: v for k, v in agg.items()}))
            print(f"    mean+-sd  EnKF {agg['enkf'][0]:.4f}+-{agg['enkf'][1]:.4f}"
                  f"   DA-SHRED {agg['da'][0]:.4f}+-{agg['da'][1]:.4f}"
                  f"   (gap {gap:.4f})")
            print(f"    nu weak   GT-fed {agg['gt_w'][0]:+.4f}   "
                  f"DA-SHRED {agg['da_w'][0]:+.4f}   EnKF {agg['en_w'][0]:+.4f}"
                  f"   (true {nu:+.2f})")

        run.write_json("results.json", rows)

        # ── Summary ──
        print("\n" + "="*74)
        print("  RECONSTRUCTION vs nu     * = worse than ignoring the sensors")
        print("="*74)
        print(f"  {'nu':>5s} {'gap':>8s} {'EnKF':>17s} {'SHRED no-DA':>17s} "
              f"{'DA-SHRED':>17s}")
        for r in rows:
            def cell(k):
                m, s = r[k]
                return f"{m:.4f}+-{s:.4f}{'*' if m > r['gap'] else ' '}"
            print(f"  {r['nu']:5.2f} {r['gap']:8.4f} {cell('enkf'):>17s} "
                  f"{cell('nda'):>17s} {cell('da'):>17s}")

        print("\n" + "="*74)
        print("  BASIS COLLAPSE — the mechanism variable")
        print("="*74)
        print(f"  {'nu':>5s} {'off-man u':>11s} {'grad^2 u':>11s} "
              f"{'grad^4 u':>11s} {'GT-fed nu':>11s}")
        for r in rows:
            o = r["offman"]
            print(f"  {r['nu']:5.2f} {100*o['u']:10.1f}% {100*o['grad2_u']:10.1f}%"
                  f" {100*o['grad4_u']:10.1f}% {r['gt_w'][0]:+11.4f}")
        print("\n" + "="*74)
        print("  DISCOVERY — does the estimate TRACK the truth?")
        print("="*74)
        print(f"  {'nu':>5s} {'GT-fed':>9s} {'DA weak':>10s} {'DA strong':>10s} "
              f"{'EnKF weak':>10s} {'infl(weak)':>11s} {'detect':>8s}")
        for r in rows:
            det = ("DA:" + ("y" if r["det_da_w"][0] > 0.5 else "n")
                   + " En:" + ("y" if r["det_en_w"][0] > 0.5 else "n"))
            flag = "" if r["alive"] else "  [DEAD]"
            print(f"  {r['nu']:5.2f} {r['gt_w'][0]:+9.4f} {r['da_w'][0]:+10.4f} "
                  f"{r['da_s'][0]:+10.4f} {r['en_w'][0]:+10.4f} "
                  f"{r['da_w'][0]-r['gt_w'][0]:+11.4f} {det:>8s}{flag}")

        print("\n" + "="*74)
        print(f"  SPECTRAL REGRESSION, k <= {KCUT:g}   (same reconstructions)")
        print("="*74)
        print(f"  {'nu':>5s} {'GT-fed':>10s} {'DA-SHRED':>11s} {'EnKF':>11s} "
              f"{'infl':>9s} {'DA rel err':>11s}")
        for r in rows:
            g = r["gt_sp"][0]
            rel = 100*(r["da_sp"][0]-r["nu"])/r["nu"]
            print(f"  {r['nu']:5.2f} {g:+10.4f} {r['da_sp'][0]:+11.4f} "
                  f"{r['en_sp'][0]:+11.4f} {r['da_sp'][0]-g:+9.4f} "
                  f"{rel:+10.1f}%")
        print("\n  Tracking slope d(recovered)/d(true)   ideal +1.00")
        print(f"  {'interval':>14s} {'GT-fed':>9s} {'DA weak':>9s} "
              f"{'DA strong':>10s} {'EnKF weak':>10s} {'GT spec':>9s} "
              f"{'DA spec':>9s}")
        for a, b in zip(rows[:-1], rows[1:]):
            d = b["nu"] - a["nu"]
            sl = {k: (b[k][0] - a[k][0])/d for k in ("gt_w", "da_w", "da_s",
                                                     "en_w", "gt_sp",
                                                     "da_sp")}
            print(f"  {a['nu']:5.2f} ->{b['nu']:5.2f} {sl['gt_w']:+9.2f} "
                  f"{sl['da_w']:+9.2f} {sl['da_s']:+10.2f} {sl['en_w']:+10.2f}"
                  f" {sl['gt_sp']:+9.2f} {sl['da_sp']:+9.2f}")

        print("\n" + "="*74)
        print("  COST   (EnKF pays per deployment; SHRED trained once, reused)")
        print("="*74)
        print(f"  {'nu':>5s} {'EnKF solves':>12s} {'EnKF s':>9s} "
              f"{'DA fit s':>9s} {'infer s':>9s}")
        for r in rows:
            print(f"  {r['nu']:5.2f} {int(r['pde'][0]):12d} "
                  f"{r['enkf_s'][0]:9.1f} {r['train_s'][0]:9.1f} "
                  f"{r['infer_s'][0]:9.3f}")
        print(f"  + one-time SHRED training: {tm_tr.seconds:.0f}s, shared by all"
              f" {len(NU_VALUES)} sweep points.")


if __name__ == "__main__":
    main()
