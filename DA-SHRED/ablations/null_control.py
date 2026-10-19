"""Null control with calibration (nu=0, gamma=0).

Two bias-correction approaches: X-to-Y calibration and DA-minus-noDA subtraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from runlog import start_run
from common2dks import build_case, score, PDECounter, Timer
from ensemble_da import StochasticEnKF, run_filter, climatological_ensemble
import spectral_sindy as sp

# Config

CFG = dict(Lx=16*np.pi, Ly=16*np.pi, Nx=64, Ny=64, dt=0.05,
           T=20.0, T_train=20.0, se=5, mu=0.02,
           nu_x=0.0, nu_y=0.0,               # NULL: no model error
           n_sensors_side=7, noise=0.01,
           real_ic_jitter=0.01)

N_SENSORS       = 20
PLACEMENT_SEED  = 1000
N_SEEDS         = 3
LAGS, RANK, D_Z, NE = 15, 30, 48, 50
INFLATION       = 1.05
SHRED_EPOCHS, DA_EPOCHS = 1200, 600
WEAK = dict(weak_width_sec=4.0, weak_stride_sec=1.0, weak_p=3)
KCUT = 1.0
SEED = 0

# Calibration config
N_CAL          = 3          # number of X→Y calibration trajectories
CAL_SEED_BASE  = 9000       # seeds for calibration trajectory generation


def random_sensors(nx, ny, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nx * ny, size=n, replace=False))


def ms(v):
    a = np.array([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def do_sindy(U, ks_sim, se, dts, Ub_s, Um_s, lib, damp_j, dif_j, bih_j,
             kcut, weak_kw):
    """Run POD-weak and spectral SINDy on field U.

    Returns dict with keys:
        damp_w, nu_w, bih_w   (POD weak, sign-corrected)
        damp_sp, nu_sp, bih_sp (spectral, sign-corrected)
    """
    zr, z = __import__('dashred_neural_2dks').forward_model_discrepancy(
        U, ks_sim, se, dts, Ub_s, Um_s, midpoint=True)

    # POD-SINDy weak form
    xi = __import__('dashred_neural_2dks').run_sindy(
        zr, z, lib, dts, mode="weak", **weak_kw)[0]

    # Spectral SINDy
    xs = sp.run_sindy_spectral(U, ks_sim, se, dts, kcut=kcut,
                               weak=True, **weak_kw)[0]

    return dict(
        damp_w=float(xi[damp_j, 0]),     nu_w=-float(xi[dif_j, 0]),
        bih_w=float(xi[bih_j, 0]),
        damp_sp=float(xs[0]),            nu_sp=-float(xs[1]),
        bih_sp=float(xs[2]),
    )


SINDY_KEYS = ("damp_w", "nu_w", "bih_w", "damp_sp", "nu_sp", "bih_sp")


def main():
    full_cfg = {**CFG, "N_SENSORS": N_SENSORS,
                "PLACEMENT_SEED": PLACEMENT_SEED, "N_SEEDS": N_SEEDS,
                "LAGS": LAGS, "RANK": RANK, "D_Z": D_Z, "NE": NE,
                "SEED": SEED, "KCUT": KCUT,
                "N_CAL": N_CAL, "CAL_SEED_BASE": CAL_SEED_BASE}

    with start_run("null_control", config=full_cfg) as run:
        import dashred_neural_2dks as mod
        Nx, Ny, se = CFG["Nx"], CFG["Ny"], CFG["se"]
        dts = CFG["dt"] * se
        np.random.seed(SEED)
        sf = random_sensors(Nx, Ny, N_SENSORS, seed=PLACEMENT_SEED)

        # ── [1] generate main case ────────────────────────────────
        print("[1] Build case: null control (nu = 0, gamma = 0, "
              "different IC)")
        case = build_case(mod.KS2D, CFG, np.random.default_rng(SEED))
        Usf, Urf, ks_sim = case["Usf"], case["Urf"], case["ks_sim"]
        gap, n = case["gap"], case["n"]
        S_real, S_sim = Urf[:, sf], Usf[:, sf]
        print(f"    {n} frames, SIM2REAL gap {gap:.4f}")
        print(f"    {N_SENSORS} sensors (placement seed {PLACEMENT_SEED})")

        # ── POD / SINDy basis ─────────────────────────────────────
        nt = int(0.85 * len(case["Utrain"]))
        Utr = case["Utrain"][:nt]
        Um = Utr.mean(0)
        Ub = np.linalg.svd((Utr - Um).T, full_matrices=False)[0][:, :RANK]
        Ub_s, Um_s = mod.build_sindy_basis(Utr, RANK)
        lib = mod.build_pod_library(Nx, Ny, CFG["Lx"]/Nx, CFG["Ly"]/Ny,
                                    Ub_s)
        damp_j = next(i for i, (nm, _) in enumerate(lib) if "damping" in nm)
        dif_j  = next(i for i, (nm, _) in enumerate(lib) if "diffusion" in nm)
        bih_j  = next(i for i, (nm, _) in enumerate(lib) if "hyper" in nm)

        sindy_args = dict(ks_sim=ks_sim, se=se, dts=dts, Ub_s=Ub_s,
                          Um_s=Um_s, lib=lib, damp_j=damp_j, dif_j=dif_j,
                          bih_j=bih_j, kcut=KCUT, weak_kw=WEAK)

        # ── [2] Train SHRED (sim-only, reused everywhere) ────────
        print(f"\n[2] Train SHRED (sim-only, reused across all runs)")
        shred = mod.SHREDTrainer(n_sensors=N_SENSORS, d_z=D_Z, lags=LAGS,
                                 decoder_type="pod", Ub=Ub, u_mean=Um,
                                 head_hidden=(64,), dropout=0.1)
        with Timer() as tm_tr:
            shred.fit(Utr, sf, n_epochs=SHRED_EPOCHS, lr=3e-3,
                      batch_size=8, pod_loss_weight=1.0,
                      full_loss_weight=1.0, verbose=False)
        print(f"    SHRED trained in {tm_tr.seconds:.0f}s")

        # ── [3] Pre-compute no-DA reconstruction + SINDy ──────────
        print(f"\n[3] Pre-compute no-DA reconstruction and SINDy")
        rc_nda = shred.reconstruct(S_real, noise_std=CFG["noise"])
        sd_nda = do_sindy(rc_nda, **sindy_args)
        r_nda = score(rc_nda, Urf, LAGS)
        print(f"    no-DA recon error: {r_nda:.4f}")
        print(f"    no-DA SINDy: dif_w={sd_nda['nu_w']:+.4f}  "
              f"damp_w={sd_nda['damp_w']:+.4f}  bih_w={sd_nda['bih_w']:+.4f}")

        # ── [4] Approach A: X→Y calibration ───────────────────────
        #    Generate N_CAL trajectory pairs from sim PDE (same physics,
        #    different ICs).  Run full DA-SHRED + SINDy on each.
        print(f"\n[4] Approach A: X→Y calibration ({N_CAL} trajectories)")

        cal_biases = {k: [] for k in SINDY_KEYS}

        for ci in range(N_CAL):
            cal_seed = CAL_SEED_BASE + ci
            print(f"\n  cal trajectory {ci} (seed {cal_seed})")

            # Build a calibration case: same PDE, different IC pair
            cal_case = build_case(mod.KS2D, CFG,
                                  np.random.default_rng(cal_seed))
            Usf_c = cal_case["Usf"]
            Urf_c = cal_case["Urf"]
            S_real_c = Urf_c[:, sf]
            S_sim_c  = Usf_c[:, sf]
            n_c = cal_case["n"]
            gap_c = cal_case["gap"]
            print(f"    {n_c} frames, gap {gap_c:.4f}")

            # No-DA on calibration trajectory
            rc_nda_c = shred.reconstruct(S_real_c, noise_std=CFG["noise"])

            # DA-SHRED on calibration trajectory
            dashred_c = mod.DASHRED(shred, sf)
            dashred_c.fit(S_real_c, S_sim_c, noise_std=CFG["noise"],
                          lambda_mmd=0.05, lambda_reg=1e-3,
                          n_epochs=DA_EPOCHS, lr=1e-3, batch_size=16,
                          verbose=False)
            rc_da_c, _, _ = dashred_c.reconstruct(S_real_c,
                                                   noise_std=CFG["noise"])

            sd_da_c = do_sindy(rc_da_c, **sindy_args)
            # Also compute DA−noDA for this calibration pair
            sd_nda_c = do_sindy(rc_nda_c, **sindy_args)

            for k in SINDY_KEYS:
                cal_biases[k].append(sd_da_c[k])

            print(f"    DA SINDy: dif_w={sd_da_c['nu_w']:+.4f}  "
                  f"damp_w={sd_da_c['damp_w']:+.4f}  "
                  f"bih_w={sd_da_c['bih_w']:+.4f}")
            print(f"    noDA SINDy: dif_w={sd_nda_c['nu_w']:+.4f}  "
                  f"damp_w={sd_nda_c['damp_w']:+.4f}  "
                  f"bih_w={sd_nda_c['bih_w']:+.4f}")

        # Mean calibration bias
        cal_mean = {k: float(np.mean(cal_biases[k])) for k in SINDY_KEYS}
        cal_sd   = {k: float(np.std(cal_biases[k], ddof=1))
                    if len(cal_biases[k]) > 1 else 0.0
                    for k in SINDY_KEYS}
        print(f"\n  Calibration bias (mean over {N_CAL}):")
        print(f"    dif_w={cal_mean['nu_w']:+.4f} +- {cal_sd['nu_w']:.4f}  "
              f"damp_w={cal_mean['damp_w']:+.4f} +- {cal_sd['damp_w']:.4f}  "
              f"bih_w={cal_mean['bih_w']:+.4f} +- {cal_sd['bih_w']:.4f}")

        # ── [5] Main experiment: DA-SHRED + EnKF ──────────────────
        print(f"\n[5] Main experiment ({N_SEEDS} seeds)")

        per = {k: [] for k in
               ("enkf", "nda", "da", "pde", "enkf_s", "train_s", "infer_s",
                # raw DA-SHRED SINDy
                "damp_da_w", "nu_da_w", "bih_da_w",
                "damp_da_sp", "nu_da_sp", "bih_da_sp",
                # GT-fed SINDy
                "damp_gt_w", "nu_gt_w", "bih_gt_w",
                "damp_gt_sp", "nu_gt_sp", "bih_gt_sp",
                # EnKF-fed SINDy
                "damp_en_w", "nu_en_w", "bih_en_w",
                "damp_en_sp", "nu_en_sp", "bih_en_sp",
                # no-DA SINDy (per-seed for Approach 2)
                "damp_nda_w", "nu_nda_w", "bih_nda_w",
                "damp_nda_sp", "nu_nda_sp", "bih_nda_sp",
                # Approach A corrected: DA − cal_mean
                "damp_calA_w", "nu_calA_w", "bih_calA_w",
                "damp_calA_sp", "nu_calA_sp", "bih_calA_sp",
                # Approach 2 corrected: DA − noDA
                "damp_app2_w", "nu_app2_w", "bih_app2_w",
                "damp_app2_sp", "nu_app2_sp", "bih_app2_sp")}

        for p in range(N_SEEDS):
            rng = np.random.default_rng(SEED + p)
            ys = Urf[:, sf] + rng.normal(0, CFG["noise"], (n, len(sf)))

            print(f"\n  seed {p}")

            # ---- EnKF ------------------------------------------------
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

            # ---- DA-SHRED ---------------------------------------------
            dashred = mod.DASHRED(shred, sf)
            with Timer() as tm_da:
                dashred.fit(S_real, S_sim, noise_std=CFG["noise"],
                            lambda_mmd=0.05, lambda_reg=1e-3,
                            n_epochs=DA_EPOCHS, lr=1e-3, batch_size=16,
                            verbose=False)
            with Timer() as tm_inf:
                rc_da, _, _ = dashred.reconstruct(S_real,
                                                   noise_std=CFG["noise"])
            r_da = score(rc_da, Urf, LAGS)

            per["enkf"].append(r_e); per["nda"].append(r_nda)
            per["da"].append(r_da); per["pde"].append(pc.n)
            per["enkf_s"].append(tm_e.seconds)
            per["train_s"].append(tm_da.seconds)
            per["infer_s"].append(tm_inf.seconds)

            # ---- SINDy on GT, DA, noDA, EnKF ─────────────────────
            sd_gt = do_sindy(Urf[LAGS:], **sindy_args)
            sd_da = do_sindy(rc_da, **sindy_args)
            sd_en = do_sindy(rec_e[LAGS:], **sindy_args)
            # noDA per-seed (rc_nda already computed but same for all seeds)
            sd_nda_seed = do_sindy(rc_nda, **sindy_args)

            for k in SINDY_KEYS:
                col, m = k.rsplit("_", 1)  # e.g. ("damp", "w") or ("nu", "sp")
                per[f"{col}_gt_{m}"].append(sd_gt[k])
                per[f"{col}_da_{m}"].append(sd_da[k])
                per[f"{col}_en_{m}"].append(sd_en[k])
                per[f"{col}_nda_{m}"].append(sd_nda_seed[k])

                # Approach A: DA − calibration mean
                per[f"{col}_calA_{m}"].append(sd_da[k] - cal_mean[k])

                # Approach 2: DA − noDA
                per[f"{col}_app2_{m}"].append(sd_da[k] - sd_nda_seed[k])

            print(f"    EnKF {r_e:.4f}  no-DA {r_nda:.4f}  "
                  f"DA-SHRED {r_da:.4f}")
            print(f"    raw DA  dif_w={sd_da['nu_w']:+.4f}  "
                  f"damp_w={sd_da['damp_w']:+.4f}  "
                  f"bih_w={sd_da['bih_w']:+.4f}")
            print(f"    calA    dif_w={sd_da['nu_w']-cal_mean['nu_w']:+.4f}  "
                  f"damp_w={sd_da['damp_w']-cal_mean['damp_w']:+.4f}  "
                  f"bih_w={sd_da['bih_w']-cal_mean['bih_w']:+.4f}")
            print(f"    app2    dif_w={sd_da['nu_w']-sd_nda_seed['nu_w']:+.4f}  "
                  f"damp_w={sd_da['damp_w']-sd_nda_seed['damp_w']:+.4f}  "
                  f"bih_w={sd_da['bih_w']-sd_nda_seed['bih_w']:+.4f}")

        # ── Aggregate and save ──
        agg = {k: ms(v) for k, v in per.items()}
        run.write_json("results.json",
                       dict(agg=agg,
                            per={k: [float(x) for x in v]
                                 for k, v in per.items()},
                            cal_mean=cal_mean,
                            cal_sd=cal_sd,
                            cal_per=cal_biases,
                            nda_sindy=sd_nda,
                            gap=gap))

        # ── Summary ──
        print("\n" + "=" * 74)
        print("  NULL CONTROL WITH CALIBRATION")
        print(f"  nu = 0.00, gamma = 0.00  (same physics, different IC)")
        print(f"  {N_SENSORS} sensors, {N_SEEDS} seeds, "
              f"{N_CAL} calibration trajectories")
        print("=" * 74)

        print(f"\n  SIM2REAL gap (from IC divergence): {gap:.4f}")

        print(f"\n  RECONSTRUCTION   (mean +- sd over {N_SEEDS} seeds)")
        print(f"    EnKF        {agg['enkf'][0]:.4f} "
              f"+- {agg['enkf'][1]:.4f}")
        print(f"    SHRED no-DA {agg['nda'][0]:.4f} "
              f"+- {agg['nda'][1]:.4f}")
        print(f"    DA-SHRED    {agg['da'][0]:.4f} "
              f"+- {agg['da'][1]:.4f}")

        print(f"\n  DISCOVERY — all should be ~0 (true L' = 0)")
        hdr = f"    {'':12s} {'GT':>8s} {'DA raw':>8s} {'CalA':>8s} " \
              f"{'App2':>8s} {'EnKF':>8s}"
        print(hdr)
        for col_label, col_key in [("damp POD", "damp"),
                                    ("damp spec", "damp"),
                                    ("dif  POD", "nu"),
                                    ("dif  spec", "nu"),
                                    ("bih  POD", "bih"),
                                    ("bih  spec", "bih")]:
            m = "w" if "POD" in col_label else "sp"
            gt  = agg[f"{col_key}_gt_{m}"][0]
            da  = agg[f"{col_key}_da_{m}"][0]
            calA = agg[f"{col_key}_calA_{m}"][0]
            app2 = agg[f"{col_key}_app2_{m}"][0]
            en  = agg[f"{col_key}_en_{m}"][0]
            print(f"    {col_label:12s} {gt:+8.4f} {da:+8.4f} "
                  f"{calA:+8.4f} {app2:+8.4f} {en:+8.4f}")

        # Reduction summary
        print(f"\n  REDUCTION (|raw DA| → |corrected|)")
        for col_label, col_key in [("dif POD", "nu"), ("dif spec", "nu"),
                                    ("damp POD", "damp"), ("damp spec", "damp"),
                                    ("bih POD", "bih"), ("bih spec", "bih")]:
            m = "w" if "POD" in col_label else "sp"
            raw = abs(agg[f"{col_key}_da_{m}"][0])
            calA = abs(agg[f"{col_key}_calA_{m}"][0])
            app2 = abs(agg[f"{col_key}_app2_{m}"][0])
            if raw > 1e-6:
                r_calA = (1 - calA / raw) * 100
                r_app2 = (1 - app2 / raw) * 100
                print(f"    {col_label:12s}  "
                      f"CalA {r_calA:+6.1f}%  App2 {r_app2:+6.1f}%")

        print(f"\n  CALIBRATION BIAS (from {N_CAL} X→Y trajectories)")
        for k in SINDY_KEYS:
            col, m = k.rsplit("_", 1)
            label = f"{col} {'POD' if m=='w' else 'spec'}"
            print(f"    {label:12s}  {cal_mean[k]:+.4f} +- {cal_sd[k]:.4f}")

        print(f"\n  NO-DA SINDY BIAS (single trajectory)")
        for k in SINDY_KEYS:
            col, m = k.rsplit("_", 1)
            label = f"{col} {'POD' if m=='w' else 'spec'}"
            print(f"    {label:12s}  {sd_nda[k]:+.4f}")

        print(f"\n  COST")
        print(f"    EnKF PDE solves (mean): {int(agg['pde'][0])}")
        print(f"    EnKF wall-clock (mean): {agg['enkf_s'][0]:.1f}s")
        print(f"    DA-SHRED train  (mean): {agg['train_s'][0]:.1f}s")
        print(f"    DA-SHRED infer  (mean): {agg['infer_s'][0]:.3f}s")
        print(f"    Calibration runs:       {N_CAL}")


if __name__ == "__main__":
    main()
