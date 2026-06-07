"""
Exp 33 (blind review C1) — run the missing baselines on the REAL single-class
windows: the Hennig-Lin calibrated validity index (spatial null, the method
ESS-Dip extends) and plain dip-means (nominal-size dip test). The existing
real-data specificity table reports only the standard indices and ESS-Dip;
these two baselines were evaluated on synthetic data only. Here we carry them to
the real windows, on the same pure single-class regions (true K=1), to isolate
the contribution on real imagery. Adds rows to Table tab:realspec.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
import diptest
import importlib.util

spec = importlib.util.spec_from_file_location("hl", "experiments/16_hennig_lin.py")
HL = importlib.util.module_from_spec(spec); spec.loader.exec_module(HL)

N_PCA, WIN, STRIDE, PURITY = 10, 16, 8, 0.90
KCAP, ALPHA, MIN_N = 8, 0.05, 20


def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_hsi(imgf, gtf):
    md, mg = loadmat(imgf), loadmat(gtf)
    cube = md[_key(md)].astype(float); gt = mg[_key(mg)].astype(int)
    H, W, B = cube.shape
    Xp = PCA(N_PCA, random_state=0).fit_transform(
        StandardScaler().fit_transform(cube.reshape(-1, B)))
    return Xp.reshape(H, W, N_PCA), gt


def load_s2():
    raw = np.load("data/s2_cube.npy"); gt = np.load("data/s2_labels.npy")
    H, W, B = raw.shape
    f = raw.reshape(-1, B)
    return ((f - f.mean(0)) / (f.std(0) + 1e-9)).reshape(H, W, B), gt


def windows(cube, gt, maxw):
    H, W = gt.shape
    out = []
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            blk = gt[y:y + WIN, x:x + WIN].ravel()
            v, c = np.unique(blk, return_counts=True)
            dom, p = v[c.argmax()], c.max() / blk.size
            if dom != 0 and p >= PURITY:
                out.append((p, cube[y:y + WIN, x:x + WIN]))
    out.sort(key=lambda t: -t[0])
    return [w for _, w in out[:maxw]]


def dipmeans_vanilla(img, rng):
    X = img.reshape(-1, img.shape[-1]); stack, leaves = [X], 0
    while stack:
        if leaves + len(stack) >= KCAP:
            return KCAP
        C = stack.pop()
        if len(C) < MIN_N:
            leaves += 1; continue
        lab = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(C)
        m0, m1 = lab == 0, lab == 1
        if m0.sum() < 2 or m1.sum() < 2:
            leaves += 1; continue
        d = C[m1].mean(0) - C[m0].mean(0); d = d / (np.linalg.norm(d) + 1e-12)
        if diptest.diptest(C @ d)[1] < ALPHA:
            stack.extend([C[m0], C[m1]])
        else:
            leaves += 1
    return leaves


def job(args):
    gid, win = args
    rng = np.random.default_rng(33000 + gid)
    try:
        k_hl = int(HL.hennig_lin(win, rng, null="grf"))
    except Exception:
        k_hl = -1
    k_dm = int(dipmeans_vanilla(win, rng))
    return k_hl, k_dm


def run(name, wins):
    jobs = list(enumerate(wins))
    res = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            res.append(fut.result())
    hl = np.array([r[0] for r in res]); dm = np.array([r[1] for r in res])
    print(f"\n  {name} ({len(wins)} windows, true K=1):")
    print(f"    {'method':22s} {'k=1 rate':>9s} {'mean k_hat':>11s}")
    print(f"    {'Hennig-Lin (spatial)':22s} {np.mean(hl==1):>9.2f} {hl.mean():>11.2f}")
    print(f"    {'Dip-means (nominal)':22s} {np.mean(dm==1):>9.2f} {dm.mean():>11.2f}")
    return hl, dm


def main():
    ip = load_hsi("data/Indian_pines_corrected.mat", "data/Indian_pines_gt.mat")
    sa = load_hsi("data/Salinas_corrected.mat", "data/Salinas_gt.mat")
    s2 = load_s2()
    hsi_wins = windows(*ip, 50) + windows(*sa, 50)
    s2_wins = windows(*s2, 60)
    run("Hyperspectral (IP+Salinas)", hsi_wins)
    run("Sentinel-2", s2_wins)


if __name__ == "__main__":
    main()
