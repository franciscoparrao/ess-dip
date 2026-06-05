"""
Exp 18 — sensitivity of the real-data specificity result (reviewer item C6).

The single-class-window specificity of Table 3 depends on preprocessing
choices: the number of PCA components (hyperspectral), the window size, and the
purity threshold. We sweep each and report ESS-Dip's k=1 rate to show the
result is not an artefact of those choices.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import methods as M


def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_hsi(imgf, gtf, n_pca):
    md, mg = loadmat(imgf), loadmat(gtf)
    cube = md[_key(md)].astype(float); gt = mg[_key(mg)].astype(int)
    H, W, B = cube.shape
    Xp = PCA(n_pca, random_state=0).fit_transform(
        StandardScaler().fit_transform(cube.reshape(-1, B)))
    return Xp.reshape(H, W, n_pca), gt


def windows(gt, win, stride, purity, maxw=40):
    H, W = gt.shape; out = []
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            blk = gt[y:y + win, x:x + win].ravel()
            v, c = np.unique(blk, return_counts=True)
            if v[c.argmax()] != 0 and c.max() / blk.size >= purity:
                out.append((y, x))
    return out[:maxw]


def spec_rate(cube, gt, win, purity):
    ws = windows(gt, win, win // 2, purity)
    if not ws:
        return float("nan"), 0
    ks = [M.ess_dip_local(cube[y:y + win, x:x + win], np.random.default_rng(i))
          for i, (y, x) in enumerate(ws)]
    return float(np.mean(np.array(ks) == 1)), len(ws)


def main():
    IP = ("data/Indian_pines_corrected.mat", "data/Indian_pines_gt.mat")

    print("[C6a] PCA-component sensitivity (Indian Pines, win=16, purity=0.90)")
    print(f"{'n_pca':>5} {'ESS-Dip k=1 rate':>16} {'(#win)':>8}")
    for npca in (5, 10, 15, 30):
        cube, gt = load_hsi(*IP, npca)
        r, n = spec_rate(cube, gt, 16, 0.90)
        print(f"{npca:>5} {r:>16.2f} {n:>8d}")

    print("\n[C6b] Window-size and purity sensitivity (Indian Pines, n_pca=10)")
    cube, gt = load_hsi(*IP, 10)
    print(f"{'win':>5} | " + " ".join(f"p>={p:>4}" for p in (0.85, 0.90, 0.95)))
    for win in (12, 16, 24):
        cells = []
        for purity in (0.85, 0.90, 0.95):
            r, n = spec_rate(cube, gt, win, purity)
            cells.append(f"{r:.2f}({n})")
        print(f"{win:>5} | " + " ".join(f"{c:>9}" for c in cells))


if __name__ == "__main__":
    main()
