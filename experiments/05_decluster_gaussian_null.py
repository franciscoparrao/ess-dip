"""
Exp 05 — synthesis: spatial declustering + UNIMODAL (Gaussian) null.

Exp 04 showed declustering fixes NULL and STRUCT but NOT TREND: a smooth
gradient is a continuous, unimodal-but-structured manifold in feature space,
and the classical gap (uniform null = "no structure at all") reads any
non-uniform spread as clustering.

Fix: keep declustering (restore ~i.i.d. samples), but replace the uniform
reference with a single multivariate GAUSSIAN matched to the data mean and
covariance — the maximum-entropy UNIMODAL null. The gap then measures
clustering BEYOND a single unimodal blob:
  - TREND  : gradient ~ one elongated unimodal cloud -> gap flat -> k=1.
  - STRUCT : K discrete modes -> gap peaks at K.
  - NULL   : one Gaussian cloud -> k=1.

Target: NULL->1, TREND->1, STRUCT->4.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

RNG = np.random.default_rng(13)
H = W = 96
B = 6
KMAX = 8
N_REF = 10
N_OFFSETS = 6


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


def gaussian_ref(X, rng):
    """Single multivariate Gaussian matched to data mean+cov (unimodal null)."""
    cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(B)
    return rng.multivariate_normal(X.mean(0), cov, size=len(X))


def gap_select(X, rng):
    ks = np.arange(1, KMAX + 1)

    def logW(Z, k):
        if k == 1:
            return np.log(((Z - Z.mean(0)) ** 2).sum() + 1e-12)
        return np.log(MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=512,
                     random_state=int(rng.integers(1e9))).fit(Z).inertia_ + 1e-12)

    lw = np.array([logW(X, k) for k in ks])
    ref = np.array([[logW(gaussian_ref(X, rng), k) for k in ks]
                    for _ in range(N_REF)])
    gap = ref.mean(0) - lw
    s = ref.std(0) * np.sqrt(1 + 1.0 / N_REF)
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def run(name, img, k_true, rng):
    stride = estimate_range(img)
    ksel = []
    for _ in range(N_OFFSETS):
        oy, ox = rng.integers(stride), rng.integers(stride)
        sub = img[oy::stride, ox::stride].reshape(-1, B)
        if len(sub) >= 30:
            ksel.append(gap_select(sub, rng))
    kmed = int(np.median(ksel)) if ksel else 1
    print(f"\n=== {name}  (truth k={k_true}) ===")
    print(f"  range={stride}px  k per offset = {ksel}  -> median k={kmed}")
    return kmed


def main():
    rN = run("NULL", null_image(RNG), 1, RNG)
    rT = run("TREND", trend_image(RNG), 1, RNG)
    rS = run("STRUCT", struct_image(RNG)[0], 4, RNG)
    print(f"\nSUMMARY  NULL={rN}  TREND={rT}  STRUCT={rS}   (target 1,1,4)")


if __name__ == "__main__":
    main()
