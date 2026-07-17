"""
Exp 36 (reviewer R2-10) — quantify the degradation of minority-class detection
under class imbalance. Two-class scenes with controlled minority proportion
pi: a GRF thresholded at its (1-pi) quantile gives a contiguous minority
region occupying ~pi of the scene; class means drawn as in make_world
(N(0, sep^2) per band). Sweep pi over {0.5,...,0.02} x sep in {3,4}, 15 reps,
and record k_hat for ESS-Dip (default), ESS-Dip-R (Gaussian null), the gap
statistic and the silhouette. The paired design regenerates the SAME scene
(world seed keyed by pi,sep,rep) for every method.
Output: results/imbalance_sweep.csv (method, pi, sep, rep, k_hat).
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

PIS = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02)
SEPS = (3.0, 4.0)
N_REALIZ = 15
KCAP, ALPHA, MIN_NEFF, TAU, B_ENS = 8, 0.05, 12, 0.25, 5
_GN = {}


def make_imbalanced(rng, pi, sep, H=96, W=96, field_sigma=6.0,
                    noise_sigma=2.0):
    """Two contiguous classes, minority proportion ~pi (true K = 2)."""
    f = M._grf(H, W, field_sigma, rng)
    labels = (f > np.quantile(f, 1.0 - pi)).astype(int)
    means = rng.standard_normal((2, M.B)) * sep
    img = means[labels] + np.stack([M._grf(H, W, noise_sigma, rng)
                                    for _ in range(M.B)], -1)
    return img, labels


def gpval(D, n_eff, rng):
    key = max(12, min(int(round(n_eff)), 2000))
    if key not in _GN:
        _GN[key] = np.sort([diptest.dipstat(rng.standard_normal(key))
                            for _ in range(300)])
    return float(np.mean(_GN[key] >= D))


def k_hat_R(img, rng, kcap=KCAP):
    """ESS-Dip-R exactly as in exp 35 (Gaussian null, local range, ensemble)."""
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    area = M.estimate_range_local(
        work, tile=min(24, img.shape[0] // 2, img.shape[1] // 2)) ** 2
    X = work.reshape(-1, work.shape[-1])

    def one(seed):
        rg = np.random.default_rng(seed); stack, leaves = [X], 0
        while stack:
            if leaves + len(stack) >= kcap:
                return kcap
            C = stack.pop(); n_eff = len(C) / area
            if n_eff < MIN_NEFF:
                leaves += 1; continue
            lab = KMeans(2, n_init=5,
                         random_state=int(rg.integers(1e9))).fit_predict(C)
            m0, m1 = lab == 0, lab == 1
            if m0.sum() < area or m1.sum() < area:
                leaves += 1; continue
            d = C[m1].mean(0) - C[m0].mean(0); d = d / (np.linalg.norm(d) + 1e-12)
            if gpval(diptest.dipstat(C @ d), n_eff, rg) < ALPHA:
                stack.extend([C[m0], C[m1]])
            else:
                leaves += 1
        return leaves
    return int(np.median([one(int(rng.integers(1e9))) for _ in range(B_ENS)]))


METHODS = {
    "ess_dip":    lambda img, rng: M.ess_dip_local(img, rng),
    "ess_dip_R":  lambda img, rng: k_hat_R(img, rng),
    "gap":        lambda img, rng: M.classical_gap(img, rng),
    "silhouette": lambda img, rng: M.silhouette(img, rng),
}


def job(args):
    method, pi, sep, rep = args
    # world seed depends only on (pi, sep, rep): identical scene per method
    wseed = 36_000 + PIS.index(pi) * 1000 + SEPS.index(sep) * 100 + rep
    img, _ = make_imbalanced(np.random.default_rng(wseed), pi, sep)
    mrng = np.random.default_rng(wseed * 7 + hash(method) % 1000)
    return method, pi, sep, rep, int(METHODS[method](img, mrng))


def main():
    jobs = [(m, pi, sep, r) for m in METHODS
            for pi in PIS for sep in SEPS for r in range(N_REALIZ)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 60 == 0:
                print(f"[exp36] {i}/{len(jobs)}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/imbalance_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "pi", "sep", "rep", "k_hat"])
        w.writerows(sorted(rows))
    # console summary: detection rate P(k_hat >= 2) and mean k_hat
    arr = {}
    for m, pi, sep, r, k in rows:
        arr.setdefault((m, pi), []).append(k)
    print(f"\n{'method':>12} | pi    | P(K>=2) | mean K")
    for (m, pi), ks in sorted(arr.items()):
        ks = np.array(ks)
        print(f"{m:>12} | {pi:.2f}  |  {np.mean(ks >= 2):.2f}   | {ks.mean():.2f}")


if __name__ == "__main__":
    main()
