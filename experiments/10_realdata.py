"""
Exp 10 — real hyperspectral data (Indian Pines, Salinas).

Two experiments:

(A) REAL-DATA SPECIFICITY (the contribution, on real data).
    Extract windows of the scene dominated by a SINGLE labeled class
    (purity >= P). Such a window is one real land-cover type: spatially
    autocorrelated and spectrally variable (illumination/soil gradients),
    but truth k = 1. We measure how often each method correctly returns
    k=1 vs invents spurious sub-clusters. Real analogue of NULL/TREND.

(B) FULL-SCENE k ESTIMATION (truth = 16 labeled classes).
    Run every method on the whole PCA-reduced cube; report k_hat vs 16.

Cubes are z-scored per band and PCA-reduced (standard HSI preprocessing).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import methods as M

N_PCA = 10
WIN, STRIDE, PURITY, MAXWIN = 16, 8, 0.90, 50
OUT = "experiments/results"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "IndianPines": ("data/Indian_pines_corrected.mat", "data/Indian_pines_gt.mat"),
    "Salinas": ("data/Salinas_corrected.mat", "data/Salinas_gt.mat"),
}


def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_pca(imgf, gtf, n_pca=N_PCA):
    md, mg = loadmat(imgf), loadmat(gtf)
    cube = md[_key(md)].astype(float)
    gt = mg[_key(mg)].astype(int)
    H, W, Bb = cube.shape
    Xs = StandardScaler().fit_transform(cube.reshape(-1, Bb))
    Xp = PCA(n_pca, random_state=0).fit_transform(Xs)
    ev = PCA(n_pca, random_state=0).fit(Xs).explained_variance_ratio_.sum()
    return Xp.reshape(H, W, n_pca), gt, ev


def homogeneous_windows(gt):
    H, W = gt.shape
    out = []
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            blk = gt[y:y + WIN, x:x + WIN].ravel()
            vals, cnt = np.unique(blk, return_counts=True)
            dom, p = vals[cnt.argmax()], cnt.max() / blk.size
            if dom != 0 and p >= PURITY:
                out.append((y, x, int(dom), float(p)))
    out.sort(key=lambda t: -t[3])
    return out[:MAXWIN]


def specificity_job(args):
    ds, idx, gid, win, dom = args
    rng = np.random.default_rng(7000 + gid)
    rows = []
    for mname, fn in M.METHODS.items():
        try:
            k = int(fn(win, rng, kmax=8) if mname in
                    ("classical_gap", "silhouette", "calinski_harabasz",
                     "davies_bouldin") else fn(win, rng))
        except Exception:
            k = -1
        rows.append(dict(dataset=ds, window=idx, dom_class=dom,
                         method=mname, k_hat=k))
    return rows


def run_specificity(cubes):
    jobs = []
    meta = {}
    gid = 0
    for ds, (cube, gt, _) in cubes.items():
        wins = homogeneous_windows(gt)
        meta[ds] = len(wins)
        for i, (y, x, dom, p) in enumerate(wins):
            jobs.append((ds, f"{ds}:{i}", gid, cube[y:y + WIN, x:x + WIN], dom))
            gid += 1
    print(f"[A] homogeneous single-class windows: "
          + ", ".join(f"{k}={v}" for k, v in meta.items())
          + f"  (win={WIN}, purity>={PURITY})")

    path = os.path.join(OUT, "real_specificity.csv")
    allrows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(specificity_job, j) for j in jobs]):
            allrows.extend(fut.result())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["dataset", "window", "dom_class",
                                            "method", "k_hat"])
        wr.writeheader(); wr.writerows(allrows)

    import pandas as pd
    df = pd.DataFrame(allrows)
    print("\n[A] REAL-DATA SPECIFICITY on single-class windows (truth k=1):")
    print(f"    {'method':18s} {'k=1 rate':>9s} {'mean k_hat':>11s}")
    for m in M.METHODS:
        g = df[df.method == m]
        print(f"    {m:18s} {np.mean(g.k_hat == 1):>9.2f} {g.k_hat.mean():>11.2f}")
    return df


def run_fullscene(cubes, kmax=20):
    print("\n[B] FULL-SCENE k estimation (truth = 16 labeled classes):")
    print(f"    {'method':18s} " + "  ".join(f"{d:>11s}" for d in cubes))
    rng = np.random.default_rng(99)
    res = {m: {} for m in M.METHODS}
    for ds, (cube, gt, _) in cubes.items():
        for m, fn in M.METHODS.items():
            try:
                res[m][ds] = int(fn(cube, rng, kmax=kmax) if m in
                                 ("classical_gap", "silhouette",
                                  "calinski_harabasz", "davies_bouldin")
                                 else fn(cube, rng, kcap=kmax))
            except Exception as e:
                res[m][ds] = -1
    for m in M.METHODS:
        print(f"    {m:18s} " + "  ".join(f"{res[m][d]:>11d}" for d in cubes))


def main():
    cubes = {}
    for ds, (imgf, gtf) in DATASETS.items():
        cube, gt, ev = load_pca(imgf, gtf)
        cubes[ds] = (cube, gt, ev)
        print(f"{ds}: cube->{cube.shape}  PCA({N_PCA}) explains {ev:.3f} variance")
    run_specificity(cubes)
    run_fullscene(cubes)


if __name__ == "__main__":
    main()
