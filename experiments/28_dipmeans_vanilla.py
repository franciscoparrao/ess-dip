"""
Exp 28 (reviewer C1) — vanilla dip-means baseline on the benchmark scenes.
Standard dip-means: recursive 2-means splitting, splitting whenever Hartigan's
dip test rejects unimodality at the NOMINAL sample size (diptest's own p-value),
with no effective-sample-size calibration, no local range, and no trend removal.
This isolates what the spatial machinery of ESS-Dip buys over plain dip-means.
Run on the same 33 cells x 15 realisations and seeds as exp 08 (bench.csv).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
import diptest
import methods as M

N_REALIZ, KCAP, ALPHA, MIN_N = 15, 8, 0.05, 20


def build_cells():
    cells = [dict(kind="null", field_sigma=fs) for fs in (3.0, 6.0, 9.0)]
    cells += [dict(kind="trend", trend_amp=ta) for ta in (1.5, 3.0)]
    cells += [dict(kind="struct", k_true=k, sep=sep, field_sigma=fs)
              for k in (2, 3, 4, 5) for sep in (2.0, 3.0, 4.0) for fs in (6.0, 9.0)]
    cells += [dict(kind="mixed", k_true=k, sep=3.0, field_sigma=6.0, trend_amp=ta)
              for k in (3, 4) for ta in (1.5, 3.0)]
    return cells


def dipmeans_vanilla(img, rng):
    """Recursive dip-means at the nominal sample size (no spatial calibration)."""
    X = img.reshape(-1, img.shape[-1])
    stack, leaves = [X], 0
    while stack:
        if leaves + len(stack) >= KCAP:
            return KCAP
        C = stack.pop()
        if len(C) < MIN_N:
            leaves += 1; continue
        lab = KMeans(2, n_init=5,
                     random_state=int(rng.integers(1e9))).fit_predict(C)
        m0, m1 = lab == 0, lab == 1
        if m0.sum() < 2 or m1.sum() < 2:
            leaves += 1; continue
        d = C[m1].mean(0) - C[m0].mean(0)
        d = d / (np.linalg.norm(d) + 1e-12)
        _, pval = diptest.diptest(C @ d)        # p-value at the nominal size
        if pval < ALPHA:
            stack.extend([C[m0], C[m1]])
        else:
            leaves += 1
    return leaves


def job(args):
    ci, cell, r = args
    rng = np.random.default_rng(10_000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    return (cell["kind"], truth, dipmeans_vanilla(img, rng))


def main():
    cells = build_cells()
    jobs = [(i, c, r) for i, c in enumerate(cells) for r in range(N_REALIZ)]
    res = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(job, a) for a in jobs]):
            res.append(fut.result())
    res = np.array([(k, t, kh) for k, t, kh in res], dtype=object)
    kind = res[:, 0]; truth = res[:, 1].astype(int); kh = res[:, 2].astype(int)

    cf = np.isin(kind, ["null", "trend"]); st = kind == "struct"; mx = kind == "mixed"
    spec = np.mean(kh[cf] == 1)
    sacc = np.mean(kh[st] == truth[st]); smae = np.mean(np.abs(kh[st] - truth[st]))
    macc = np.mean(kh[mx] == truth[mx])
    bal = np.mean([spec, sacc, macc])
    print("vanilla dip-means (nominal-size dip test, no spatial calibration):")
    print(f"  specificity (k=1 on {cf.sum()} class-free)  = {spec:.2f}  "
          f"(mean k_hat = {kh[cf].mean():.2f})")
    print(f"  structured  acc = {sacc:.2f}   MAE = {smae:.2f}")
    print(f"  mixed       acc = {macc:.2f}")
    print(f"  BALANCED score  = {bal:.3f}")
    print(f"\n  Table-1 row:  vanilla dip-means & {spec:.2f} & {sacc:.2f} & "
          f"{smae:.2f} & {macc:.2f} & {bal:.3f}")


if __name__ == "__main__":
    main()
