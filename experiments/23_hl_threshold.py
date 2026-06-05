"""
Exp 23 — sensitivity of the Hennig & Lin comparison to the significance
threshold (reviewer m2). We record max_K V(K) and arg max per scene once, then
apply thresholds V > {1.64, 1.96, 2.33} post hoc, to show the qualitative
conclusion (i.i.d. null over-detects; spatial null helps specificity but
collapses on mixed) is stable.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import methods as M

spec = importlib.util.spec_from_file_location("hl", "experiments/16_hennig_lin.py")
HL = importlib.util.module_from_spec(spec); spec.loader.exec_module(HL)

N_REALIZ, M_SUB, B_BOOT = 6, 1000, 12
THRESHOLDS = (1.64, 1.96, 2.33)


def hl_maxv(img, rng, null):
    H, W, p = img.shape
    X = img.reshape(-1, p)
    idx = rng.choice(len(X), min(M_SUB, len(X)), replace=False)
    asw_d = HL.asw_curve(X[idx], rng)
    nullmat = {k: [] for k in range(2, HL.KMAX + 1)}
    for _ in range(B_BOOT):
        Xb = (rng.multivariate_normal(X.mean(0),
              np.cov(X, rowvar=False) + 1e-6 * np.eye(p), size=len(idx))
              if null == "iid" else HL.phase_surrogate(img, rng)[idx])
        for k, v in HL.asw_curve(Xb, rng).items():
            nullmat[k].append(v)
    V = {k: (asw_d[k] - np.mean(nullmat[k])) / (np.std(nullmat[k]) + 1e-9)
         for k in range(2, HL.KMAX + 1)}
    bk = max(V, key=V.get)
    return bk, V[bk]


def cells():
    c = [dict(kind="null", field_sigma=fs) for fs in (3., 6., 9.)]
    c += [dict(kind="trend", trend_amp=ta) for ta in (1.5, 3.)]
    c += [dict(kind="struct", k_true=k, sep=s, field_sigma=fs)
          for k in (2, 3, 4, 5) for s in (2., 3., 4.) for fs in (6., 9.)]
    c += [dict(kind="mixed", k_true=k, sep=3., field_sigma=6., trend_amp=ta)
          for k in (3, 4) for ta in (1.5, 3.)]
    return c


def job(args):
    ci, cell, r = args
    rng = np.random.default_rng(60000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    out = []
    for null in ("iid", "grf"):
        bk, mv = hl_maxv(img, rng, null)
        out.append((cell["kind"], truth, null, bk, mv))
    return out


def main():
    cs = cells()
    jobs = [(i, c, r) for i, c in enumerate(cs) for r in range(N_REALIZ)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(job, a) for a in jobs]):
            rows.extend(fut.result())
    rows = np.array(rows, dtype=object)
    kind = rows[:, 0]; truth = rows[:, 1].astype(int); null = rows[:, 2]
    bk = rows[:, 3].astype(int); mv = rows[:, 4].astype(float)

    print("HL threshold sensitivity (specificity | struct acc | mixed acc):")
    for nl in ("iid", "grf"):
        print(f"  null = {nl}")
        for t in THRESHOLDS:
            khat = np.where(mv > t, bk, 1)
            sel = null == nl
            cf = sel & np.isin(kind, ["null", "trend"])
            st = sel & (kind == "struct"); mx = sel & (kind == "mixed")
            spec = np.mean(khat[cf] == 1)
            sa = np.mean(khat[st] == truth[st]); ma = np.mean(khat[mx] == truth[mx])
            print(f"    V>{t:.2f}:  spec {spec:.2f} | struct {sa:.2f} | mixed {ma:.2f}")


if __name__ == "__main__":
    main()
