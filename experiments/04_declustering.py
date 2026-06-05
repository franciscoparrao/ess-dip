"""
Exp 04 — spatial declustering.

Simplest attack on the trend-vs-class confound diagnosed in exp 03: the
over-detection comes from autocorrelation duplicating information (effective
sample size << n). So thin the pixels to ~independence (stride > autocorr
range), then run the CLASSIC Tibshirani gap (uniform null) on the thinned,
~i.i.d. sample.

Intuition for why this should separate trend from clusters:
  - TREND  : thinned samples still lie on a smooth gradient -> unimodal in
             feature space -> gap ~ uniform null -> k=1.
  - STRUCT : thinned samples retain the K discrete modes -> gap peaks at K.
  - NULL   : thinned samples ~ unimodal Gaussian -> k=1.

Range is estimated from the decay of the first-PC spatial autocorrelation.
Average k over several grid offsets for stability.

Target across the three worlds: NULL->1, TREND->1, STRUCT->4.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

RNG = np.random.default_rng(13)
H = W = 96          # a bit larger so thinned samples are still plenty
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


# ---- autocorrelation range from first PC --------------------------------
def estimate_range(img):
    """Lag (in pixels) where radial autocorrelation of PC1 drops below 1/e."""
    flat = img.reshape(-1, B)
    pc = PCA(1).fit_transform(flat).reshape(H, W)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    F = np.fft.fft2(pc)
    ac = np.fft.ifft2(np.abs(F) ** 2).real / pc.size      # autocovariance
    ac = ac / ac[0, 0]
    # radial profile of the (wrap) autocorrelation
    prof = 0.5 * (ac[0, :W // 2] + ac[:H // 2, 0])
    below = np.where(prof < 1 / np.e)[0]
    rng_pix = int(below[0]) if below.size else W // 6
    return max(2, min(rng_pix, W // 6))


def gap_select(X, rng):
    ks = np.arange(1, KMAX + 1)

    def logW(Z, k):
        if k == 1:
            return np.log(((Z - Z.mean(0)) ** 2).sum() + 1e-12)
        return np.log(MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=512,
                     random_state=int(rng.integers(1e9))).fit(Z).inertia_ + 1e-12)

    lw = np.array([logW(X, k) for k in ks])
    ref = np.zeros((N_REF, len(ks)))
    for r in range(N_REF):
        Xr = rng.uniform(X.min(0), X.max(0), size=X.shape)
        ref[r] = [logW(Xr, k) for k in ks]
    gap = ref.mean(0) - lw
    s = ref.std(0) * np.sqrt(1 + 1.0 / N_REF)
    for i in range(len(ks) - 1):           # Tibshirani 1-SE rule
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def run(name, img, k_true, rng):
    stride = estimate_range(img)
    ksel = []
    for o in range(N_OFFSETS):
        oy, ox = rng.integers(stride), rng.integers(stride)
        sub = img[oy::stride, ox::stride].reshape(-1, B)
        if len(sub) >= 30:
            ksel.append(gap_select(sub, rng))
    # too few independent samples => dominated by large-scale structure
    # (trend): no evidence for discrete clustering -> k=1
    kmed = int(np.median(ksel)) if ksel else 1
    print(f"\n=== {name}  (truth k={k_true}) ===")
    print(f"  est. autocorr range = {stride} px  ->  "
          f"~{(H // stride) * (W // stride)} indep. samples/offset")
    print(f"  declustered gap k per offset = {ksel}")
    print(f"  declustered gap (median)     -> k={kmed}   [ours]")
    return kmed


def main():
    Xn = null_image(RNG)
    Xt = trend_image(RNG)
    Xs, _ = struct_image(RNG)
    rN = run("NULL", Xn, 1, RNG)
    rT = run("TREND", Xt, 1, RNG)
    rS = run("STRUCT", Xs, 4, RNG)
    print(f"\nSUMMARY  NULL={rN}  TREND={rT}  STRUCT={rS}   (target 1,1,4)")


if __name__ == "__main__":
    main()
