"""
Exp 16b — Hennig & Lin over the full benchmark grid, for regime-rate
comparison with Table 1 (specificity / structured / mixed).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import methods as M

spec = importlib.util.spec_from_file_location("hl", "experiments/16_hennig_lin.py")
HL = importlib.util.module_from_spec(spec); spec.loader.exec_module(HL)

N_REALIZ = 15
M_SUB, B_BOOT = 1000, 15
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
    rng = np.random.default_rng(20000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    rows = []
    for name, nl in [("HL_iid", "iid"), ("HL_grf", "grf")]:
        try:
            k = int(HL.hennig_lin(img, rng, nl, m=M_SUB, b=B_BOOT))
        except Exception:
            k = -1
        rows.append(dict(kind=cell["kind"], truth_k=truth, method=name, k_hat=k))
    return rows


def main():
    cs = cells()
    jobs = [(i, c, r) for i, c in enumerate(cs) for r in range(N_REALIZ)]
    print(f"{len(jobs)} scenes x 2 HL variants")
    rows, done = [], 0
    nproc = max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for fut in as_completed([ex.submit(job, a) for a in jobs]):
            rows.extend(fut.result()); done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    with open(os.path.join(OUT, "hl_bench.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["kind", "truth_k", "method", "k_hat"])
        wr.writeheader(); wr.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows); df["correct"] = df.k_hat == df.truth_k
    fp = df[df.kind.isin(["null", "trend"])]
    st = df[df.kind == "struct"]; mx = df[df.kind == "mixed"]
    print(f"\n{'method':8s} {'spec':>6s} {'struct':>7s} {'mixed':>6s} {'balanced':>9s}")
    for m in ("HL_iid", "HL_grf"):
        s = fp[fp.method == m].correct.mean()
        a = st[st.method == m].correct.mean()
        c = mx[mx.method == m].correct.mean()
        print(f"{m:8s} {s:>6.2f} {a:>7.2f} {c:>6.2f} {np.mean([s,a,c]):>9.3f}")
    print("  (compare ESS-Dip: spec 1.00 | struct 0.81 | mixed 0.50 | bal 0.771)")


if __name__ == "__main__":
    main()
