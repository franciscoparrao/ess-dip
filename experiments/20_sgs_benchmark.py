"""
Exp 20 — recursive SGS estimator over the full synthetic grid (reviewer M1).
Produces specificity / structured / mixed rates comparable to Table 1, so the
Gaussian-simulation reference is reported as a full method, not a one-shot test.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M
import geostat as G

N_REALIZ = 15
N_NULL, N_RUNS = 40, 3
OUT = "experiments/results"


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
    rng = np.random.default_rng(30000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    try:
        k = int(G.sgs_dip(img, rng, n_null=N_NULL, n_runs=N_RUNS))
    except Exception:
        k = -1
    return dict(kind=cell["kind"], truth_k=truth, method="SGS-Dip", k_hat=k)


def main():
    cs = cells()
    jobs = [(i, c, r) for i, c in enumerate(cs) for r in range(N_REALIZ)]
    print(f"{len(jobs)} scenes, recursive SGS")
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(job, a) for a in jobs]):
            rows.append(fut.result()); done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    with open(os.path.join(OUT, "sgs_bench.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["kind", "truth_k", "method", "k_hat"])
        wr.writeheader(); wr.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows); df["correct"] = df.k_hat == df.truth_k
    s = df[df.kind.isin(["null", "trend"])].correct.mean()
    a = df[df.kind == "struct"].correct.mean()
    mae = (df[df.kind == "struct"].k_hat - df[df.kind == "struct"].truth_k).abs().mean()
    c = df[df.kind == "mixed"].correct.mean()
    print(f"\nSGS-Dip: spec {s:.2f} | struct {a:.2f} (MAE {mae:.2f}) | "
          f"mixed {c:.2f} | balanced {np.mean([s,a,c]):.3f}")
    print("  (ESS-Dip: spec 1.00 | struct 0.81 | mixed 0.50 | bal 0.771)")


if __name__ == "__main__":
    main()
