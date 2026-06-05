"""
Exp 12 — analysis on real Sentinel-2 multispectral data (the user's domain:
typical satellite bands). Mirrors exp 10 (hyperspectral) on a Po Valley scene
with ESA WorldCover ground truth.

(A) Specificity on single-class windows (truth k=1).
(B) Full-scene k estimation (truth = number of WorldCover classes present).

10 Sentinel-2 bands (B02..B12), z-scored per band.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M

WIN, STRIDE, PURITY, MAXWIN = 16, 8, 0.90, 60
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
      60: "bare", 80: "water", 90: "wetland"}

cube = np.load("data/s2_cube.npy")            # (H,W,10)
labels = np.load("data/s2_labels.npy")        # (H,W)
H, W, Bb = cube.shape
# z-score per band (robust to band scale differences)
flat = cube.reshape(-1, Bb)
cube = ((flat - flat.mean(0)) / (flat.std(0) + 1e-9)).reshape(H, W, Bb)

vals, cnt = np.unique(labels, return_counts=True)
present = [(int(v), int(c)) for v, c in zip(vals, cnt) if c / labels.size >= 0.01]
truth_k = len(present)
print(f"scene {cube.shape}; WorldCover classes >=1%: "
      + ", ".join(f"{WC.get(v,v)}({100*c/labels.size:.0f}%)" for v, c in present)
      + f"  -> truth_k={truth_k}")


def homog_windows():
    out = []
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            blk = labels[y:y + WIN, x:x + WIN].ravel()
            v, c = np.unique(blk, return_counts=True)
            dom, p = int(v[c.argmax()]), c.max() / blk.size
            if p >= PURITY:
                out.append((y, x, dom, float(p)))
    out.sort(key=lambda t: -t[3])
    return out[:MAXWIN]


def spec_job(args):
    gid, win, dom = args
    rng = np.random.default_rng(8000 + gid)
    rows = []
    for m, fn in M.METHODS.items():
        try:
            k = int(fn(win, rng, kmax=8) if m in
                    ("classical_gap", "silhouette", "calinski_harabasz",
                     "davies_bouldin") else fn(win, rng))
        except Exception:
            k = -1
        rows.append(dict(window=gid, dom_class=WC.get(dom, dom), method=m, k_hat=k))
    return rows


def run_specificity():
    wins = homog_windows()
    doms = {}
    for _, _, d, _ in wins:
        doms[WC.get(d, d)] = doms.get(WC.get(d, d), 0) + 1
    print(f"\n[A] single-class windows: {len(wins)} "
          f"(by class: {doms})  win={WIN} purity>={PURITY}")
    jobs = [(i, cube[y:y + WIN, x:x + WIN], d)
            for i, (y, x, d, p) in enumerate(wins)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(spec_job, j) for j in jobs]):
            rows.extend(fut.result())
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/s2_specificity.csv", index=False)
    print("\n[A] SENTINEL-2 SPECIFICITY on single-class windows (truth k=1):")
    print(f"    {'method':18s} {'k=1 rate':>9s} {'mean k_hat':>11s}")
    for m in M.METHODS:
        g = df[df.method == m]
        print(f"    {m:18s} {np.mean(g.k_hat == 1):>9.2f} {g.k_hat.mean():>11.2f}")


def run_fullscene(kmax=12):
    print(f"\n[B] FULL-SCENE k estimation (truth_k={truth_k}):")
    rng = np.random.default_rng(99)
    print(f"    {'method':18s} {'k_hat':>6s}")
    for m, fn in M.METHODS.items():
        try:
            k = int(fn(cube, rng, kmax=kmax) if m in
                    ("classical_gap", "silhouette", "calinski_harabasz",
                     "davies_bouldin") else fn(cube, rng, kcap=kmax))
        except Exception:
            k = -1
        flag = "  <-- truth" if k == truth_k else ""
        print(f"    {m:18s} {k:>6d}{flag}")


if __name__ == "__main__":
    os.makedirs("experiments/results", exist_ok=True)
    run_specificity()
    run_fullscene()
