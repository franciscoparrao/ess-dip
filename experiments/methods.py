"""
Shared module for the benchmark: synthetic world generators and all
k-selection methods under comparison. Consolidates validated code from
exp 02 / 06 / 07.

All methods take a multiband image `img` of shape (H, W, B) and return an
estimated number of clusters k_hat (integer >= 1).
"""

import numpy as np
import diptest
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

KMAX = 8
B = 6


# =========================================================================
# world generators
# =========================================================================
def _grf(H, W, sigma, rng):
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def _trend_field(H, W, rng):
    yy, xx = np.mgrid[0:H, 0:W] / H
    a, b, c = rng.standard_normal(3)
    p = a * xx + b * yy + c * xx * yy
    return (p - p.mean()) / (p.std() + 1e-12)


def make_world(kind, rng, H=96, W=96, k_true=4, sep=3.0,
               field_sigma=6.0, noise_sigma=2.0, trend_amp=2.0):
    """
    kind in {'null', 'trend', 'struct', 'mixed'}.
    Returns (img (H,W,B), truth_k, labels or None).
    """
    if kind == "null":
        img = np.stack([_grf(H, W, field_sigma, rng) for _ in range(B)], -1)
        return img, 1, None

    if kind == "trend":
        bands = [trend_amp * _trend_field(H, W, rng) + 0.4 * _grf(H, W, 3.0, rng)
                 for _ in range(B)]
        return np.stack(bands, -1), 1, None

    # discrete contiguous patches
    fields = np.stack([_grf(H, W, field_sigma, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1)
    means = rng.standard_normal((k_true, B)) * sep
    img = means[labels] + np.stack([_grf(H, W, noise_sigma, rng)
                                    for _ in range(B)], -1)
    if kind == "mixed":                       # classes + global trend
        tr = np.stack([_trend_field(H, W, rng) for _ in range(B)], -1)
        img = img + trend_amp * tr
    return img, k_true, labels


# =========================================================================
# helpers
# =========================================================================
def _mbk(X, k, rng):
    return MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=256,
                           random_state=int(rng.integers(1e9))).fit(X)


def _logW(X, k, rng):
    if k == 1:
        return np.log(((X - X.mean(0)) ** 2).sum() + 1e-12)
    return np.log(_mbk(X, k, rng).inertia_ + 1e-12)


def estimate_range(img, clamp=None):
    H, W, _ = img.shape
    pc = PCA(1).fit_transform(img.reshape(-1, img.shape[-1])).reshape(H, W)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real
    ac /= ac[0, 0]
    def first_below(p):
        w = np.where(p < 1 / np.e)[0]
        return float(w[0]) if w.size else float(len(p))
    Rx = first_below(ac[0, :max(2, W // 2)])      # horizontal
    Ry = first_below(ac[:max(2, H // 2), 0])      # vertical (handles H != W)
    R = max(2.0, 0.5 * (Rx + Ry))
    return min(R, clamp) if clamp else R


def poly_r2(img, degree=2):
    """Fraction of variance explained by a low-order polynomial surface
    (averaged over bands) — a clean detector of a smooth global trend."""
    H, W, Bb = img.shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    terms = [np.ones_like(xx, float)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((xx ** (d - i)) * (yy ** i))
    A = np.stack([t.ravel() for t in terms], 1)
    r2 = []
    for b in range(Bb):
        y = img[..., b].ravel()
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2.append(1 - (y - A @ coef).var() / (y.var() + 1e-12))
    return float(np.mean(r2))


# =========================================================================
# methods   (each returns k_hat)
# =========================================================================
def classical_gap(img, rng, n_ref=8, kmax=KMAX):
    X = img.reshape(-1, img.shape[-1])
    ks = np.arange(1, kmax + 1)
    lw = np.array([_logW(X, k, rng) for k in ks])
    ref = np.array([[_logW(rng.uniform(X.min(0), X.max(0), size=X.shape), k, rng)
                     for k in ks] for _ in range(n_ref)])
    gap = ref.mean(0) - lw
    s = ref.std(0) * np.sqrt(1 + 1.0 / n_ref)
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def _internal_index(img, rng, which, kmax=KMAX):
    X = img.reshape(-1, img.shape[-1])
    idx = rng.choice(len(X), min(1500, len(X)), replace=False)
    Xs = X[idx]
    scores = {}
    for k in range(2, kmax + 1):
        lab = _mbk(Xs, k, rng).predict(Xs)
        if which == "sil":
            scores[k] = silhouette_score(Xs, lab)
        elif which == "ch":
            scores[k] = calinski_harabasz_score(Xs, lab)
        else:
            scores[k] = -davies_bouldin_score(Xs, lab)   # lower is better
    return int(max(scores, key=scores.get))


def silhouette(img, rng, kmax=KMAX): return _internal_index(img, rng, "sil", kmax)
def calinski_harabasz(img, rng, kmax=KMAX): return _internal_index(img, rng, "ch", kmax)
def davies_bouldin(img, rng, kmax=KMAX): return _internal_index(img, rng, "db", kmax)


def decluster_dipmeans(img, rng, n_offsets=6, alpha=0.05,
                       min_test=25, min_child=12, kcap=KMAX):
    """exp 06: spatial declustering + projected dip-means (median over offsets)."""
    H, W, _ = img.shape
    stride = int(max(2, min(round(estimate_range(img)), W // 6)))
    ks = []
    for _ in range(n_offsets):
        oy, ox = rng.integers(stride), rng.integers(stride)
        sub = img[oy::stride, ox::stride].reshape(-1, img.shape[-1])
        if len(sub) < min_test:
            continue
        stack, leaves = [sub], 0
        while stack:
            if leaves + len(stack) >= kcap:
                leaves = kcap; break
            C = stack.pop()
            if len(C) < min_test:
                leaves += 1; continue
            proj = PCA(1).fit_transform(C).ravel()
            _, p = diptest.diptest(proj)
            if p < alpha:
                lab = KMeans(2, n_init=5,
                             random_state=int(rng.integers(1e9))).fit_predict(C)
                c0, c1 = C[lab == 0], C[lab == 1]
                if len(c0) >= min_child and len(c1) >= min_child:
                    stack.extend([c0, c1])
                else:
                    leaves += 1
            else:
                leaves += 1
        ks.append(leaves)
    return int(np.median(ks)) if ks else 1


_ESS_NULL = {}


def _ess_pval(D, n_eff, rng, n_null=300):
    key = max(12, min(int(round(n_eff)), 2000))
    if key not in _ESS_NULL:
        _ESS_NULL[key] = np.sort([diptest.dipstat(rng.uniform(size=key))
                                  for _ in range(n_null)])
    return float(np.mean(_ESS_NULL[key] >= D))


def estimate_range_local(img, tile=24, clamp=None):
    """Median autocorrelation range over small tiles. Tiles fall mostly inside
    a single class, so this estimates the WITHIN-class autocorrelation length
    (the nuisance) rather than the between-class patch scale — the quantity
    the effective sample size should use."""
    H, W, _ = img.shape
    Rs = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            Rs.append(estimate_range(img[y:y + tile, x:x + tile]))
    R = float(np.median(Rs)) if Rs else estimate_range(img)
    return min(R, clamp) if clamp else R


def ess_dip(img, rng, alpha=0.05, min_neff=12, kcap=KMAX, range_clamp=None,
            range_val=None, area_val=None):
    """exp 07: dip on all pixels, p-value calibrated to effective sample size.
    area_val (decorrelation area, e.g. an integral range) overrides the
    ell^2 shortcut when supplied (reviewer item R1)."""
    X = img.reshape(-1, img.shape[-1])
    if area_val is not None:
        area = area_val
    else:
        R = range_val if range_val is not None else estimate_range(img, clamp=range_clamp)
        area = R * R
    stack, leaves = [X], 0
    while stack:
        if leaves + len(stack) >= kcap:
            return kcap
        C = stack.pop()
        n_eff = len(C) / area
        if n_eff < min_neff:
            leaves += 1; continue
        lab = KMeans(2, n_init=5,
                     random_state=int(rng.integers(1e9))).fit_predict(C)
        m0, m1 = lab == 0, lab == 1
        d = C[m1].mean(0) - C[m0].mean(0)
        d = d / (np.linalg.norm(d) + 1e-12)
        D = diptest.dipstat(C @ d)
        if _ess_pval(D, n_eff, rng) < alpha and m0.sum() >= area and m1.sum() >= area:
            stack.extend([C[m0], C[m1]])
        else:
            leaves += 1
    return leaves


def detrend_poly(img, degree=2):
    """Remove a smooth low-order polynomial surface per band (illumination /
    topographic / atmospheric gradients) while leaving discrete patches —
    which are not polynomial — largely intact."""
    H, W, Bb = img.shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    terms = [np.ones_like(xx, float)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((xx ** (d - i)) * (yy ** i))
    A = np.stack([t.ravel() for t in terms], 1)
    out = np.empty_like(img, float)
    for b in range(Bb):
        y = img[..., b].ravel()
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[..., b] = (y - A @ coef).reshape(H, W)
    return out


def ess_dip_detrend(img, rng, **kw):
    """exp 07 + polynomial detrending: handles classes + global trend together."""
    return ess_dip(detrend_poly(img), rng, **kw)


def ess_dip_adaptive(img, rng, tau=0.25, n_runs=5, kcap=KMAX):
    """
    exp 09: adaptive detrend + robust range + ensemble.
      - detrend ONLY if a smooth global trend is detected (poly_r2 > tau),
        so pure-class scenes keep their power and trend scenes are corrected;
      - clamp the autocorrelation range (avoids power-killing range outliers);
      - median over n_runs to absorb 2-means initialisation variance.
    """
    H, W, _ = img.shape
    work = detrend_poly(img) if poly_r2(img) > tau else img
    clamp = W / 8.0
    ks = [ess_dip(work, rng, range_clamp=clamp, kcap=kcap) for _ in range(n_runs)]
    return int(np.median(ks))


def ess_dip_local(img, rng, tau=0.25, n_runs=5, tile=24, kcap=KMAX):
    """exp 13: adaptive detrend + LOCAL (within-class) range + ensemble.
    Local range estimation gives a realistic effective sample size on
    multi-class scenes, restoring recall without sacrificing specificity."""
    H, W, _ = img.shape
    work = detrend_poly(img) if poly_r2(img) > tau else img
    R = estimate_range_local(work, tile=min(tile, H // 2, W // 2))
    ks = [ess_dip(work, rng, range_val=R, kcap=kcap) for _ in range(n_runs)]
    return int(np.median(ks))


METHODS = {
    "classical_gap": classical_gap,
    "silhouette": silhouette,
    "calinski_harabasz": calinski_harabasz,
    "davies_bouldin": davies_bouldin,
    "decluster_dip": decluster_dipmeans,
    "ess_dip": ess_dip,
    "ess_dip_detrend": ess_dip_detrend,
    "ess_dip_adaptive": ess_dip_adaptive,
    "ess_dip_local": ess_dip_local,
}
