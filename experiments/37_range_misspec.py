"""
Exp 37 (reviewer R2-11) — robustness of the calibration to misestimation of
the autocorrelation range. The estimated local range ell is perturbed by a
multiplicative factor f in {1/4, 1/2, 3/4, 1, 3/2, 2, 4} (the decorrelation
area scales as f^2), and the deployed pipeline (adaptive detrend + local range
+ B=5 ensemble) is run with the perturbed range on the null world
(specificity: P(k_hat > 1)) and the structured world with K=4 (recovery:
k_hat). The paired design reuses the SAME 15 scenes per world across factors.
Output: results/range_misspec.csv (world, factor, rep, k_hat, truth).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M

FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
N_REALIZ = 15
TAU, B_ENS = 0.25, 5
WORLDS = {
    "null":   (dict(kind="null", field_sigma=6.0), 1),
    "struct": (dict(kind="struct", k_true=4, sep=3.0, field_sigma=6.0), 4),
}


def k_hat_scaled(img, rng, factor):
    """Deployed pipeline with the estimated range multiplied by `factor`."""
    H, W, _ = img.shape
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    R = M.estimate_range_local(work, tile=min(24, H // 2, W // 2))
    ks = [M.ess_dip(work, rng, range_val=R * factor) for _ in range(B_ENS)]
    return int(np.median(ks))


def job(args):
    world, factor, rep = args
    cell, truth = WORLDS[world]
    # world seed depends only on (world, rep): same scene across factors
    wseed = 37_000 + list(WORLDS).index(world) * 1000 + rep
    img, _, _ = M.make_world(rng=np.random.default_rng(wseed), **cell)
    mrng = np.random.default_rng(wseed * 11 + int(factor * 100))
    return world, factor, rep, k_hat_scaled(img, mrng, factor), truth


def main():
    jobs = [(w, f, r) for w in WORLDS for f in FACTORS for r in range(N_REALIZ)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 30 == 0:
                print(f"[exp37] {i}/{len(jobs)}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/range_misspec.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "factor", "rep", "k_hat", "truth"])
        w.writerows(sorted(rows))
    # console summary
    arr = {}
    for w_, f_, r_, k_, t_ in rows:
        arr.setdefault((w_, f_), []).append(k_)
    print(f"\n{'world':>8} | factor | false-split / mean K")
    for (w_, f_), ks in sorted(arr.items()):
        ks = np.array(ks)
        if w_ == "null":
            print(f"{w_:>8} | {f_:4.2f}   | P(K>1) = {np.mean(ks > 1):.2f}")
        else:
            print(f"{w_:>8} | {f_:4.2f}   | mean K = {ks.mean():.2f}  "
                  f"P(K=4) = {np.mean(ks == 4):.2f}")


if __name__ == "__main__":
    main()
