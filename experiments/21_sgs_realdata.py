"""
Exp 21 — recursive SGS estimator on real data (reviewer M1).
Single-class-window specificity (hyperspectral + Sentinel-2) and full-scene K,
to report the exact reference on the same benchmarks as ESS-Dip.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import geostat as G

WIN, STRIDE, PURITY, MAXWIN = 16, 8, 0.90, 50


def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_hsi(imgf, gtf, n=10):
    md, mg = loadmat(imgf), loadmat(gtf)
    cube = md[_key(md)].astype(float); gt = mg[_key(mg)].astype(int)
    H, W, B = cube.shape
    Xp = PCA(n, random_state=0).fit_transform(
        StandardScaler().fit_transform(cube.reshape(-1, B)))
    return Xp.reshape(H, W, n), gt


def windows(gt):
    H, W = gt.shape; out = []
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            blk = gt[y:y + WIN, x:x + WIN].ravel()
            v, c = np.unique(blk, return_counts=True)
            if v[c.argmax()] != 0 and c.max() / blk.size >= PURITY:
                out.append((y, x))
    return out[:MAXWIN]


def win_job(args):
    gid, sub = args
    return int(G.sgs_dip(sub, np.random.default_rng(40000 + gid),
                         alpha=0.05 / 8, n_null=160, n_runs=3, min_n=40))


def main():
    cubes = {
        "IndianPines": load_hsi("data/Indian_pines_corrected.mat",
                                "data/Indian_pines_gt.mat"),
        "Salinas": load_hsi("data/Salinas_corrected.mat",
                            "data/Salinas_gt.mat"),
    }
    s2 = np.load("data/s2_cube.npy"); s2l = np.load("data/s2_labels.npy")
    H, W, Bb = s2.shape
    s2 = ((s2.reshape(-1, Bb) - s2.reshape(-1, Bb).mean(0)) /
          (s2.reshape(-1, Bb).std(0) + 1e-9)).reshape(H, W, Bb)

    # ---- (A) specificity on single-class windows ----
    jobs, meta = [], {}
    gid = 0
    for name, (cube, gt) in cubes.items():
        ws = windows(gt); meta.setdefault("hsi", 0)
        for (y, x) in ws:
            jobs.append((gid, cube[y:y + WIN, x:x + WIN])); gid += 1
        meta["hsi"] += len(ws)
    s2w = windows(s2l); meta["s2"] = len(s2w)
    s2_start = gid
    for (y, x) in s2w:
        jobs.append((gid, s2[y:y + WIN, x:x + WIN])); gid += 1

    res = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        futs = {ex.submit(win_job, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res[futs[fut]] = fut.result()
    res = np.array(res)
    print("[A] SGS-Dip specificity on single-class windows (truth k=1):")
    print(f"   hyperspectral ({meta['hsi']} win): k=1 rate = "
          f"{np.mean(res[:s2_start] == 1):.2f}")
    print(f"   Sentinel-2   ({meta['s2']} win): k=1 rate = "
          f"{np.mean(res[s2_start:] == 1):.2f}")

    # ---- (B) full-scene K ----
    print("\n[B] SGS-Dip (corrected) full-scene K (nominal classes in parentheses):")
    rng = np.random.default_rng(7)
    for name, (cube, gt) in cubes.items():
        k = int(G.sgs_dip(cube, rng, alpha=0.05 / 20, n_null=120, n_runs=2,
                          kcap=20, min_n=80))
        print(f"   {name:12s} ({len(np.unique(gt))-1}): K={k}")
    k = int(G.sgs_dip(s2, rng, alpha=0.05 / 12, n_null=120, n_runs=2,
                      kcap=12, min_n=80))
    print(f"   {'Sentinel-2':12s} (4): K={k}")


if __name__ == "__main__":
    main()
