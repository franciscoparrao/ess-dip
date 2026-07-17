"""
Exp 38 (reviewer R2-14) — the simplest spatial adaptation of an existing
index: the gap statistic with an autocorrelation-matched reference. Instead of
uniform references over the bounding box (Tibshirani), each reference dataset
is a Gaussian random field per band whose 1/e range matches the scene estimate
(local tile-median), scaled to each band's standard deviation. Everything else
(log W_k, Tibshirani one-SE rule, kmax) is exactly the classical gap. Run on
the same 33-cell x 15-rep factorial as Table 1 for direct comparability.
Output: results/spatial_gap.csv (kind, truth, k_hat) + console summary.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M

N_REALIZ, KMAX, N_REF = 15, 8, 8


def spatial_gap(img, rng, n_ref=N_REF, kmax=KMAX):
    """Gap statistic with GRF references matched to the estimated local range.

    For the Gaussian-kernel generator _grf, the field correlogram is
    rho(h) = exp(-h^2 / (4 sigma^2)), so the 1/e crossing sits at h = 2 sigma;
    sigma = ell_hat / 2 reproduces the scene's estimated range.
    """
    H, W, B_ = img.shape
    ell = M.estimate_range_local(img, tile=min(24, H // 2, W // 2))
    sigma = max(ell / 2.0, 0.5)
    X = img.reshape(-1, B_)
    sd = X.std(0) + 1e-12

    logW = np.array([M._logW(X, k, rng) for k in range(1, kmax + 1)])
    ref = np.empty((n_ref, kmax))
    for b in range(n_ref):
        Xr = np.stack([M._grf(H, W, sigma, rng) for _ in range(B_)],
                      -1).reshape(-1, B_) * sd
        ref[b] = [M._logW(Xr, k, rng) for k in range(1, kmax + 1)]
    gap = ref.mean(0) - logW
    se = ref.std(0) * np.sqrt(1 + 1 / n_ref)
    for k in range(kmax - 1):                     # Tibshirani one-SE rule
        if gap[k] >= gap[k + 1] - se[k + 1]:
            return k + 1
    return kmax


def build_cells():
    c = [dict(kind="null", field_sigma=fs) for fs in (3., 6., 9.)]
    c += [dict(kind="trend", trend_amp=ta) for ta in (1.5, 3.)]
    c += [dict(kind="struct", k_true=k, sep=s, field_sigma=fs)
          for k in (2, 3, 4, 5) for s in (2., 3., 4.) for fs in (6., 9.)]
    c += [dict(kind="mixed", k_true=k, sep=3., field_sigma=6., trend_amp=ta)
          for k in (3, 4) for ta in (1.5, 3.)]
    return c


def job(args):
    ci, cell, r = args
    rng = np.random.default_rng(10_000 + ci * 100 + r)  # same worlds as exp35
    img, truth, _ = M.make_world(rng=rng, **cell)
    return cell["kind"], truth, spatial_gap(img, np.random.default_rng(
        38_000 + ci * 100 + r))


def main():
    cells = build_cells()
    jobs = [(ci, c, r) for ci, c in enumerate(cells) for r in range(N_REALIZ)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        futs = {ex.submit(job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 60 == 0:
                print(f"[exp38] {i}/{len(jobs)}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/spatial_gap.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "truth", "k_hat"]); w.writerows(rows)
    arr = {}
    for kind, truth, k in rows:
        arr.setdefault(kind, []).append((truth, k))
    spec = np.mean([k == 1 for kind in ("null", "trend")
                    for _, k in arr.get(kind, [])])
    acc_s = np.mean([k == t for t, k in arr.get("struct", [])])
    mae_s = np.mean([abs(k - t) for t, k in arr.get("struct", [])])
    acc_m = np.mean([k == t for t, k in arr.get("mixed", [])])
    bal = np.mean([spec, acc_s, acc_m])
    print(f"\n[spatial gap] specificity {spec:.2f} | struct acc {acc_s:.2f} "
          f"(MAE {mae_s:.2f}) | mixed acc {acc_m:.2f} | balanced {bal:.3f}")
    for kind in ("null", "trend", "struct", "mixed"):
        ks = np.array([k for _, k in arr.get(kind, [])])
        print(f"  {kind:7s}: mean K {ks.mean():.2f}")


if __name__ == "__main__":
    main()
