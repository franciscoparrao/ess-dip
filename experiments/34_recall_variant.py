"""
Exp 34 (lever 2) — a recall-oriented variant of ESS-Dip. The conservative first
pass under-detects for two reasons: the uniform (least-favourable) null is
over-conservative when the within-class null is Gaussian, and the single 2-means
split direction is governed by the majority class, hiding minority modes. We test
two specificity-preserving changes against the baseline:
  V1 (gauss): calibrate the dip against the Gaussian null (Assumption ass:null)
              rather than the uniform least-favourable null.
  V2 (multi): test the dip along several candidate directions (k-means k=2,3,4),
              splitting if any is significant after a Bonferroni correction.
  V3 = V1 + V2.
Same local-range area, detrending and ensemble as ESS-Dip. Benchmarked on the
33-cell synthetic grid; the split is always the binary 2-means partition.
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

N_REALIZ, KCAP, ALPHA, MIN_NEFF, TAU, B_ENS = 8, 8, 0.05, 12, 0.25, 3
_NULL = {}


def null_quantile_pval(D, n_eff, rng, gauss):
    key = (gauss, max(12, min(int(round(n_eff)), 2000)))
    if key not in _NULL:
        m = key[1]
        gen = (rng.standard_normal if gauss else rng.uniform)
        _NULL[key] = np.sort([diptest.dipstat(gen(size=m)) for _ in range(300)])
    return float(np.mean(_NULL[key] >= D))


def build_cells():
    c = [dict(kind="null", field_sigma=fs) for fs in (3., 6., 9.)]
    c += [dict(kind="trend", trend_amp=ta) for ta in (1.5, 3.)]
    c += [dict(kind="struct", k_true=k, sep=s, field_sigma=fs)
          for k in (2, 3, 4, 5) for s in (2., 3., 4.) for fs in (6., 9.)]
    c += [dict(kind="mixed", k_true=k, sep=3., field_sigma=6., trend_amp=ta)
          for k in (3, 4) for ta in (1.5, 3.)]
    return c


def candidate_dirs(C, rng, multi):
    lab = KMeans(2, n_init=3, random_state=int(rng.integers(1e9))).fit_predict(C)
    m0, m1 = lab == 0, lab == 1
    dirs = []
    if m0.sum() and m1.sum():
        d = C[m1].mean(0) - C[m0].mean(0)
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    if multi:
        for k in (3, 4):
            if len(C) >= 8 * k:
                cen = KMeans(k, n_init=3,
                             random_state=int(rng.integers(1e9))).fit(C).cluster_centers_
                best, bi, bj = -1, 0, 1
                for i in range(k):
                    for j in range(i + 1, k):
                        dd = np.linalg.norm(cen[i] - cen[j])
                        if dd > best:
                            best, bi, bj = dd, i, j
                d = cen[bi] - cen[bj]
                dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return dirs, m0, m1


def k_hat(img, area, rng, gauss, multi):
    X = img.reshape(-1, img.shape[-1])
    stack, leaves = [X], 0
    while stack:
        if leaves + len(stack) >= KCAP:
            return KCAP
        C = stack.pop()
        n_eff = len(C) / area
        if n_eff < MIN_NEFF:
            leaves += 1; continue
        dirs, m0, m1 = candidate_dirs(C, rng, multi)
        if not dirs or m0.sum() < area or m1.sum() < area:
            leaves += 1; continue
        pvals = [null_quantile_pval(diptest.dipstat(C @ d), n_eff, rng, gauss)
                 for d in dirs]
        if min(pvals) < ALPHA / len(dirs):           # Bonferroni over directions
            stack.extend([C[m0], C[m1]])
        else:
            leaves += 1
    return leaves


VARIANTS = {"ESS-Dip (base)": (False, False), "V1 gauss": (True, False),
            "V2 multi": (False, True), "V3 gauss+multi": (True, True)}


def job(args):
    ci, cell, r = args
    rng = np.random.default_rng(34000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    area = M.estimate_range_local(work, tile=min(24, img.shape[0] // 2)) ** 2
    out = {"kind": cell["kind"], "truth": truth}
    for name, (g, mu) in VARIANTS.items():
        ks = [k_hat(work, area, rng, g, mu) for _ in range(B_ENS)]
        out[name] = int(np.median(ks))
    return out


def main():
    cells = build_cells()
    jobs = [(i, c, r) for i, c in enumerate(cells) for r in range(N_REALIZ)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(job, a) for a in jobs]):
            rows.append(fut.result())
    kind = np.array([r["kind"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    cf = np.isin(kind, ["null", "trend"]); st = kind == "struct"; mx = kind == "mixed"
    print(f"  {'variant':16s} {'spec':>5s} {'struct':>7s} {'mixed':>6s} {'bal':>6s}")
    for name in VARIANTS:
        kh = np.array([r[name] for r in rows])
        spec = np.mean(kh[cf] == 1)
        sa = np.mean(kh[st] == truth[st]); ma = np.mean(kh[mx] == truth[mx])
        print(f"  {name:16s} {spec:>5.2f} {sa:>7.2f} {ma:>6.2f} "
              f"{np.mean([spec, sa, ma]):>6.3f}")


if __name__ == "__main__":
    main()
