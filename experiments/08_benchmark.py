"""
Exp 08 — quantitative benchmark (the paper's central results).

Grid of synthetic conditions x N random realizations; every method run on the
SAME image per realization (paired comparison). Worlds include null and trend
(truth k=1) to measure FALSE-POSITIVE behaviour — the contribution — plus
'mixed' (classes + global trend), the realistic remote-sensing case.

Run:  OMP_NUM_THREADS handled below; just `.venv/bin/python experiments/08_benchmark.py`
Outputs: experiments/results/bench.csv, summary printed, figs/08_benchmark.png
"""
import os
# single-threaded BLAS BEFORE numpy import; forked workers inherit this
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M

N_REALIZ = 15
OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("experiments/figs", exist_ok=True)


def build_cells():
    cells = []
    for fs in (3.0, 6.0, 9.0):
        cells.append(dict(kind="null", field_sigma=fs))
    for ta in (1.5, 3.0):
        cells.append(dict(kind="trend", trend_amp=ta))
    for k in (2, 3, 4, 5):
        for sep in (2.0, 3.0, 4.0):
            for fs in (6.0, 9.0):
                cells.append(dict(kind="struct", k_true=k, sep=sep,
                                  field_sigma=fs))
    for k in (3, 4):
        for ta in (1.5, 3.0):
            cells.append(dict(kind="mixed", k_true=k, sep=3.0, field_sigma=6.0,
                              trend_amp=ta))
    return cells


def job(args):
    cell_idx, cell, r = args
    rng = np.random.default_rng(10_000 + cell_idx * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    rows = []
    for mname, fn in M.METHODS.items():
        try:
            k_hat = int(fn(img, rng))
        except Exception:
            k_hat = -1
        rows.append({**cell, "cell_idx": cell_idx, "realiz": r,
                     "truth_k": truth, "method": mname, "k_hat": k_hat})
    return rows


def main():
    cells = build_cells()
    jobs = [(i, c, r) for i, c in enumerate(cells) for r in range(N_REALIZ)]
    print(f"{len(cells)} cells x {N_REALIZ} realizations = {len(jobs)} jobs, "
          f"{len(M.METHODS)} methods each")

    fields = ["kind", "k_true", "sep", "field_sigma", "trend_amp", "noise_sigma",
              "cell_idx", "realiz", "truth_k", "method", "k_hat"]
    path = os.path.join(OUT_DIR, "bench.csv")
    n_done = 0
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        nproc = max(2, (os.cpu_count() or 4) - 2)
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(job, a) for a in jobs]
            for fut in as_completed(futs):
                for row in fut.result():
                    wr.writerow(row)
                n_done += 1
                if n_done % 25 == 0:
                    fh.flush()
                    print(f"  {n_done}/{len(jobs)} realizations done", flush=True)
    print(f"results -> {path}")
    summarize(path)


def summarize(path):
    import pandas as pd
    df = pd.read_csv(path)
    df["correct"] = df.k_hat == df.truth_k
    df["abserr"] = (df.k_hat - df.truth_k).abs()

    print("\n==================  SUMMARY  ==================")
    # (A) false-positive control on null+trend (truth k=1): want k_hat==1
    fp = df[df.kind.isin(["null", "trend"])]
    print("\n[A] Specificity on NULL+TREND (truth k=1) — fraction correctly k=1:")
    for m, g in fp.groupby("method"):
        print(f"   {m:18s} k=1 rate = {g.correct.mean():.2f}   "
              f"mean k_hat = {g.k_hat.mean():.2f}")

    # (B) exact-k accuracy on struct (truth>1)
    st = df[df.kind == "struct"]
    print("\n[B] Exact-k accuracy on STRUCT (truth=k):")
    for m, g in st.groupby("method"):
        print(f"   {m:18s} acc = {g.correct.mean():.2f}   MAE = {g.abserr.mean():.2f}")

    # (C) mixed (classes + trend): the realistic case
    mx = df[df.kind == "mixed"]
    print("\n[C] Exact-k accuracy on MIXED (classes + trend):")
    for m, g in mx.groupby("method"):
        print(f"   {m:18s} acc = {g.correct.mean():.2f}   MAE = {g.abserr.mean():.2f}")

    # (D) overall balanced score = mean(specificity, struct-acc, mixed-acc)
    print("\n[D] Balanced score = mean(NULL/TREND k=1 rate, STRUCT acc, MIXED acc):")
    rows = []
    for m in df.method.unique():
        s = fp[fp.method == m].correct.mean()
        a = st[st.method == m].correct.mean()
        c = mx[mx.method == m].correct.mean()
        rows.append((m, np.mean([s, a, c]), s, a, c))
    rows.sort(key=lambda x: -x[1])
    for m, bal, s, a, c in rows:
        print(f"   {m:18s} balanced = {bal:.3f}   (spec {s:.2f} | struct {a:.2f} | mixed {c:.2f})")

    make_figure(df, fp, st, mx, rows)


def make_figure(df, fp, st, mx, ranked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [r[0] for r in ranked]
    spec = {m: fp[fp.method == m].correct.mean() for m in order}
    sacc = {m: st[st.method == m].correct.mean() for m in order}
    macc = {m: mx[mx.method == m].correct.mean() for m in order}

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(order)); w = 0.27
    ax[0].bar(x - w, [spec[m] for m in order], w, label="NULL+TREND k=1 rate")
    ax[0].bar(x, [sacc[m] for m in order], w, label="STRUCT exact-k acc")
    ax[0].bar(x + w, [macc[m] for m in order], w, label="MIXED exact-k acc")
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=30, ha="right")
    ax[0].set_ylabel("rate"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Specificity vs accuracy by method"); ax[0].legend(fontsize=8)

    # accuracy vs k_true (struct), per method
    for m in order:
        g = st[st.method == m]
        acc_by_k = g.groupby("k_true").correct.mean()
        ax[1].plot(acc_by_k.index, acc_by_k.values, "o-", label=m)
    ax[1].set_xlabel("true number of classes"); ax[1].set_ylabel("exact-k accuracy")
    ax[1].set_title("STRUCT: accuracy vs k_true"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = "experiments/figs/08_benchmark.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
