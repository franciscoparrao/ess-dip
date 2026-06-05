"""
Exp 07 — ESS-calibrated dip (the refinement).

Exp 06 (declustering + dip-means) solved the trend confound but lost power to
recover many classes, because declustering DISCARDS pixels: with a large
autocorrelation range only ~30 independent samples survive, too few for the
dip test to resolve 4-5 classes.

Refinement: do NOT discard pixels. Compute the dip statistic on ALL pixels of
a cluster (stable estimate, full detection power), but calibrate its p-value
against the null dip distribution at the EFFECTIVE sample size
    n_eff = n_pixels / decorrelation_area,
where decorrelation_area ~ R^2 and R is the spatial autocorrelation range
(cf. Cressie / Dutilleul et al. 1993 effective sample size). Autocorrelation
then only RAISES the significance threshold (preventing false splits on smooth
trends) without throwing away data.

Recursive splitter (projected dip-means with ESS calibration): project a
cluster onto PC1, test unimodality with the ESS-calibrated dip; split via
2-means while significant.

Target: NULL->1, TREND->1, STRUCT->4, and recover higher k than exp 06.
"""

import numpy as np
import diptest
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RNG = np.random.default_rng(13)
H = W = 96
B = 6
ALPHA = 0.05
KCAP = 12
N_NULL = 300          # null replicates per n_eff bucket
MIN_NEFF = 12         # below this, treat as too little info -> unimodal


# ---- worlds -------------------------------------------------------------
def grf(sigma, rng):
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def null_image(rng):
    return np.stack([grf(4.0, rng) for _ in range(B)], -1)


def trend_image(rng):
    yy, xx = np.mgrid[0:H, 0:W] / H
    bands = []
    for _ in range(B):
        a, b, c = rng.standard_normal(3)
        p = a * xx + b * yy + c * xx * yy
        p = (p - p.mean()) / (p.std() + 1e-12)
        bands.append(3.0 * p + 0.4 * grf(3.0, rng))
    return np.stack(bands, -1)


def struct_image(rng, k_true=4, sep=2.0):
    fields = np.stack([grf(9.0, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1)
    means = rng.standard_normal((k_true, B)) * sep
    img = means[labels] + np.stack([grf(3.0, rng) for _ in range(B)], -1)
    return img, labels


# ---- spatial autocorrelation range / effective sample size --------------
def estimate_range(field2d):
    pc = (field2d - field2d.mean()) / (field2d.std() + 1e-12)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real
    ac /= ac[0, 0]
    prof = 0.5 * (ac[0, :W // 2] + ac[:H // 2, 0])
    below = np.where(prof < 1 / np.e)[0]
    return max(2.0, float(below[0]) if below.size else W / 6)


# ---- ESS-calibrated dip null (cached by rounded n_eff) ------------------
_null_cache = {}


def null_dip_threshold_p(D, n_eff, rng):
    """p-value of dip D against the null dip distribution at size n_eff."""
    key = int(round(n_eff))
    key = max(MIN_NEFF, min(key, 2000))
    if key not in _null_cache:
        dips = np.array([diptest.dipstat(rng.uniform(size=key))
                         for _ in range(N_NULL)])
        _null_cache[key] = np.sort(dips)
    nd = _null_cache[key]
    return float(np.mean(nd >= D))


# ---- recursive ESS-dip splitter -----------------------------------------
def ess_dipmeans(X, pos, R, rng):
    """
    X   : (n, B) pixel features
    pos : (n, 2) pixel coordinates (for effective sample size)
    R   : spatial autocorrelation range (pixels)
    """
    area = R * R                       # pixels per independent patch
    stack = [(X, pos)]
    leaves = 0
    while stack:
        if leaves + len(stack) >= KCAP:
            return KCAP
        C, P = stack.pop()
        n = len(C)
        n_eff = n / area
        if n_eff < MIN_NEFF:
            leaves += 1
            continue
        # propose a 2-way split, then test the dip ALONG the split direction
        # (centroid-difference axis) -- far more sensitive than PC1 and
        # directly tests whether the proposed cut is a real gap.
        lab = KMeans(2, n_init=5,
                     random_state=int(rng.integers(1e9))).fit_predict(C)
        m0, m1 = lab == 0, lab == 1
        d = C[m1].mean(0) - C[m0].mean(0)
        d = d / (np.linalg.norm(d) + 1e-12)
        proj = C @ d
        D = diptest.dipstat(proj)
        p = null_dip_threshold_p(D, n_eff, rng)
        if p < ALPHA and m0.sum() >= area and m1.sum() >= area:
            stack.append((C[m0], P[m0]))
            stack.append((C[m1], P[m1]))
        else:
            leaves += 1
    return leaves


def run(name, img, k_true, rng):
    flat = img.reshape(-1, B)
    pc1 = PCA(1).fit_transform(flat).reshape(H, W)
    R = estimate_range(pc1)
    yy, xx = np.mgrid[0:H, 0:W]
    pos = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
    k = ess_dipmeans(flat, pos, R, rng)
    print(f"=== {name} (truth {k_true}): R={R:.1f}px  n_eff(full)="
          f"{flat.shape[0] / R**2:.0f}  -> k={k}")
    return k


def main():
    rN = run("NULL", null_image(RNG), 1, RNG)
    rT = run("TREND", trend_image(RNG), 1, RNG)
    rS = run("STRUCT", struct_image(RNG)[0], 4, RNG)
    print(f"\nSUMMARY  NULL={rN}  TREND={rT}  STRUCT={rS}   (target 1,1,4)\n")

    print("STRUCT sweep (k_true x sep):")
    for k_true in (3, 4, 5):
        row = []
        for sep in (2.0, 3.0, 4.0):
            img, _ = struct_image(RNG, k_true=k_true, sep=sep)
            row.append(run(f"  k={k_true} sep={sep}", img, k_true, RNG))
        print(f"  k_true={k_true}: {row}   (target all {k_true})")


if __name__ == "__main__":
    main()
