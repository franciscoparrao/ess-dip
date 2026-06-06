"""
Exp 27 (reviewer A1) — empirical validation of the effective-sample-size
calibration for the dip functional. The calibration n_eff = n_C / a borrows the
integral-range effective size, derived for the variance of the sample MEAN
(Cressie 1993), and applies it to the dip, a functional of the empirical
DISTRIBUTION. We have no closed-form effective size for the dip; instead we show
empirically that the calibration as implemented (a = R^2, the within-class
1/e-range proxy) controls the false-split level at or below alpha and keeps it
stable as n_eff varies over two orders of magnitude, while retaining power.

For each (autocorrelation sigma, patch size) cell we run a SINGLE homogeneity
decision (root node only, isolating the per-node calibration) on:
  - H0 fields (one Gaussian-RF class)  -> false-split rate, three calibrations:
      * ESS + 2-means direction      (the method as applied)
      * ESS + fixed random direction (isolates calibration from selection)
      * nominal-n + 2-means          (the uncalibrated baseline it fixes)
  - H1 fields (two classes, sep=3)     -> power (ESS + 2-means).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
import diptest
import methods as M

ALPHA, NU, TAU = 0.05, 12, 0.25
SIGMAS = (2.0, 4.0, 6.0, 8.0)
SIDES = (32, 48, 64, 80)
REPS = 160


def _range_area(work):
    h, w, _ = work.shape
    R = M.estimate_range_local(work, tile=min(24, h // 2, w // 2))
    return R * R                      # a = R^2, exactly ess_dip_local's calibration


def _dip_dir(X, rng, twomeans):
    if twomeans:
        lab = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(X)
        d = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    else:
        d = rng.standard_normal(X.shape[1])
    d = d / (np.linalg.norm(d) + 1e-12)
    return diptest.dipstat(X @ d)


def decide(img, rng):
    """Returns (ess_2m, ess_fixed, nominal, n_eff). 1 = test splits (rejects H0)."""
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    a = _range_area(work)
    n = work.shape[0] * work.shape[1]
    n_eff = n / a
    X = work.reshape(-1, work.shape[-1])
    D_2m = _dip_dir(X, rng, True)
    D_fx = _dip_dir(X, rng, False)
    if n_eff < NU:                    # algorithm floor: too little information -> leaf
        ess_2m = ess_fx = 0
    else:
        ess_2m = int(M._ess_pval(D_2m, n_eff, rng) < ALPHA)
        ess_fx = int(M._ess_pval(D_fx, n_eff, rng) < ALPHA)
    nominal = int(M._ess_pval(D_2m, n, rng) < ALPHA)   # calibrate at nominal n
    return ess_2m, ess_fx, nominal, n_eff


def cell(args):
    si, sigma, L = args
    h0 = {"ess": [], "fix": [], "nom": [], "neff": []}
    pw = []
    for r in range(REPS):
        rng = np.random.default_rng(27000 + si * 1000 + r)
        img, _, _ = M.make_world("null", rng, H=L, W=L, field_sigma=sigma)
        e2, ef, nm, ne = decide(img, rng)
        h0["ess"].append(e2); h0["fix"].append(ef); h0["nom"].append(nm)
        h0["neff"].append(ne)
        img1, _, _ = M.make_world("struct", rng, H=L, W=L, k_true=2, sep=3.0,
                                  field_sigma=sigma)
        e2h, _, _, _ = decide(img1, rng)
        pw.append(e2h)
    return dict(sigma=sigma, L=L, n=L * L,
                n_eff=float(np.mean(h0["neff"])),
                fsr_ess=float(np.mean(h0["ess"])),
                fsr_fixed=float(np.mean(h0["fix"])),
                fsr_nominal=float(np.mean(h0["nom"])),
                power=float(np.mean(pw)))


def main():
    os.makedirs("experiments/results", exist_ok=True)
    jobs = [(i, s, L) for i, (s, L) in enumerate(
        [(s, L) for s in SIGMAS for L in SIDES])]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(cell, j) for j in jobs]):
            rows.append(fut.result())
    rows.sort(key=lambda d: d["n_eff"])
    print(f"  alpha = {ALPHA};  REPS = {REPS};  H0 false-split rate (FSR), H1 power")
    print(f"  {'sigma':>5} {'n':>6} {'n_eff':>7} | {'FSR ESS':>8} {'FSR fix':>8} "
          f"{'FSR nom':>8} | {'power':>6}")
    for d in rows:
        print(f"  {d['sigma']:>5.0f} {d['n']:>6} {d['n_eff']:>7.1f} | "
              f"{d['fsr_ess']:>8.3f} {d['fsr_fixed']:>8.3f} {d['fsr_nominal']:>8.3f} | "
              f"{d['power']:>6.2f}")
    print(f"\n  ESS FSR: mean {np.mean([d['fsr_ess'] for d in rows]):.3f}, "
          f"max {max(d['fsr_ess'] for d in rows):.3f} "
          f"(over n_eff {rows[0]['n_eff']:.0f}-{rows[-1]['n_eff']:.0f})")
    with open("experiments/results/level_neff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved -> experiments/results/level_neff.csv")


if __name__ == "__main__":
    main()
