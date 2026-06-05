"""
Exp 06 — synthesis method: spatial declustering + projected dip-means.

Two ingredients identified across exp 01-05:
  (1) spatial declustering -> ~i.i.d. sample (fixes effective sample size)
  (2) a MODALITY criterion (Hartigan dip test) instead of a 2nd-order null
      -> only split a cluster if its projection is significantly MULTIMODAL.
A smooth trend is unimodal -> never split -> k=1. Discrete classes are
multimodal -> split until each piece is unimodal -> k=K.

Projected dip-means (cf. Kalogeratos & Likas 2012): recursively project a
cluster onto PC1, run the dip test; if multimodal (p<alpha) split via 2-means.

Target: NULL->1, TREND->1, STRUCT->4.
"""

import numpy as np
import diptest
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RNG = np.random.default_rng(13)
H = W = 96
B = 6
N_OFFSETS = 6
ALPHA = 0.05
MIN_TEST = 25        # min points to bother testing a split
MIN_CHILD = 12       # reject a split producing a tinier child
KCAP = 12            # safety cap


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


def estimate_range(img):
    pc = PCA(1).fit_transform(img.reshape(-1, B)).reshape(H, W)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real
    ac /= ac[0, 0]
    prof = 0.5 * (ac[0, :W // 2] + ac[:H // 2, 0])
    below = np.where(prof < 1 / np.e)[0]
    rng_pix = int(below[0]) if below.size else W // 6
    return max(2, min(rng_pix, W // 6))


def dipmeans(X, rng):
    """Return number of clusters via recursive projected dip splitting."""
    stack = [X]
    final = 0
    while stack:
        if final + len(stack) >= KCAP:        # safety
            return KCAP
        C = stack.pop()
        if len(C) < MIN_TEST:
            final += 1
            continue
        proj = PCA(1).fit_transform(C).ravel()
        _, pval = diptest.diptest(proj)
        if pval < ALPHA:
            lab = KMeans(2, n_init=5,
                         random_state=int(rng.integers(1e9))).fit_predict(C)
            c0, c1 = C[lab == 0], C[lab == 1]
            if len(c0) >= MIN_CHILD and len(c1) >= MIN_CHILD:
                stack.extend([c0, c1])
            else:
                final += 1
        else:
            final += 1
    return final


def run(name, img, k_true, rng):
    stride = estimate_range(img)
    ks = []
    for _ in range(N_OFFSETS):
        oy, ox = rng.integers(stride), rng.integers(stride)
        sub = img[oy::stride, ox::stride].reshape(-1, B)
        if len(sub) >= MIN_TEST:
            ks.append(dipmeans(sub, rng))
    kmed = int(np.median(ks)) if ks else 1
    print(f"\n=== {name}  (truth k={k_true}) ===")
    print(f"  range={stride}px  k per offset = {ks}  -> median k={kmed}")
    return kmed


def main():
    rN = run("NULL", null_image(RNG), 1, RNG)
    rT = run("TREND", trend_image(RNG), 1, RNG)
    rS = run("STRUCT", struct_image(RNG)[0], 4, RNG)
    print(f"\nSUMMARY  NULL={rN}  TREND={rT}  STRUCT={rS}   (target 1,1,4)")


if __name__ == "__main__":
    main()
