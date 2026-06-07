"""
Exp 32 (blind review A2) — within-class variograms on the real scenes.
A geostatistician needs to see that the within-class autocorrelation the
calibration relies on is real and estimable on actual imagery, that local tile
estimation captures the within-class scale (not the between-class mosaic), and
that most tiles fall within a single class. For each scene we:
  (1) average the radial empirical semivariogram of the first PC over pure
      single-class windows -> the within-class variogram, fit a model, report
      the range, the 1/e range, the field integral range a_int and the
      indicator integral range a_I;
  (2) compare the local (tile-median) range with the scene-wide range, the
      conflation the method avoids;
  (3) report the fraction of tiles that are single-class.
Saves the empirical/fitted curves for figures/make_fig_variogram.R.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import methods as M

N_PCA, WV, MAXW = 10, 16, 200
LMAX = 9                                     # variogram lags to fit/show


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


def first_pc(block):
    X = block.reshape(-1, block.shape[-1])
    u = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    return ((X @ u) - (X @ u).mean()).reshape(block.shape[:2])


def radial_corr(pc, nlag):
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real)
    ac /= ac.max()
    h, w = pc.shape; cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.round(np.hypot(yy - cy, xx - cx)).astype(int)
    return np.array([ac[r == k].mean() for k in range(min(nlag, cy))])


def pure_windows(gt, H, W):
    for pur in (0.90, 0.85, 0.80):
        out = []
        for y in range(0, H - WV + 1, WV // 2):
            for x in range(0, W - WV + 1, WV // 2):
                blk = gt[y:y + WV, x:x + WV].ravel()
                v, c = np.unique(blk, return_counts=True)
                if v[c.argmax()] != 0 and c.max() / blk.size >= pur:
                    out.append((y, x))
        if len(out) >= 10:
            return out[:MAXW], pur
    return out[:MAXW], pur


def integral_ranges(rho):
    rho = np.clip(rho, 0.0, 0.999999); rr = np.arange(len(rho))
    a_int = float(np.sum(rho * 2 * np.pi * rr))
    a_I = float(np.sum((2 / np.pi) * np.arcsin(rho) * 2 * np.pi * rr))
    below = np.where(rho < 1 / np.e)[0]
    ell = float(below[0]) if below.size else float(len(rho))
    return ell, a_I, a_int


def fit_models(rho):
    r = np.arange(1, len(rho)); y = rho[1:]            # skip lag 0 (nugget)
    models = {
        "exponential": lambda r, b: np.exp(-r / b),
        "gaussian":    lambda r, b: np.exp(-(r / b) ** 2),
        "spherical":   lambda r, b: np.where(r < b,
                       1 - 1.5 * (r / b) + 0.5 * (r / b) ** 3, 0.0)}
    best = None
    for name, fn in models.items():
        try:
            p, _ = curve_fit(fn, r, y, p0=[3.0], bounds=(0.5, 50), maxfev=5000)
            sse = float(np.sum((y - fn(r, *p)) ** 2))
            if best is None or sse < best[2]:
                best = (name, float(p[0]), sse, fn(np.arange(len(rho)), *p))
        except Exception:
            pass
    return best


def main():
    os.makedirs("experiments/results", exist_ok=True)
    scenes = {"Indian Pines": load_hsi("data/Indian_pines_corrected.mat",
                                       "data/Indian_pines_gt.mat"),
              "Salinas": load_hsi("data/Salinas_corrected.mat",
                                  "data/Salinas_gt.mat"),
              "Sentinel-2": load_s2()}
    curves = []
    print(f"  {'scene':13} {'model':11} {'range':>5} {'l/e':>4} "
          f"{'aI':>5} {'aint':>5} {'aint/aI':>7} | {'loc rng':>7} {'glob rng':>8} "
          f"{'pure tiles':>10}")
    for name, (cube, gt) in scenes.items():
        H, W = gt.shape
        wins, pur = pure_windows(gt, H, W)
        rs = [radial_corr(first_pc(cube[y:y + WV, x:x + WV]), LMAX) for y, x in wins]
        L = min(len(r) for r in rs)
        rho = np.mean([r[:L] for r in rs], axis=0)
        model, rng_, sse, fit = fit_models(rho)
        ell, a_I, a_int = integral_ranges(rho)
        # local vs global range on the full scene
        loc = M.estimate_range_local(cube, tile=WV)
        glob = M.estimate_range(cube)
        # pure-tile fraction
        nt = pure = 0
        for y in range(0, H - WV + 1, WV):
            for x in range(0, W - WV + 1, WV):
                blk = gt[y:y + WV, x:x + WV].ravel()
                v, c = np.unique(blk, return_counts=True)
                nt += 1; pure += int(c.max() / blk.size >= pur)
        print(f"  {name:13} {model:11} {rng_:>5.1f} {ell:>4.1f} {a_I:>5.0f} "
              f"{a_int:>5.0f} {a_int/max(a_I,1):>7.2f} | {loc:>7.1f} {glob:>8.1f} "
              f"{pure/nt:>9.2f} ({len(wins)} win)")
        for h, (g, f) in enumerate(zip(1 - rho, 1 - fit)):
            curves.append(dict(scene=name, lag=h, gamma=round(float(g), 4),
                               fit=round(float(f), 4), model=model,
                               range=round(rng_, 2)))
    with open("experiments/results/real_variograms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "lag", "gamma", "fit",
                                          "model", "range"])
        w.writeheader(); w.writerows(curves)
    print("saved -> experiments/results/real_variograms.csv")


if __name__ == "__main__":
    main()
