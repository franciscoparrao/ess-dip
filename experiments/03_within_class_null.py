"""
Exp 03 — within-class null.

The phase surrogate of the RAW image over-corrects because it preserves the
low-frequency power produced by discrete class patches. Fix: estimate the
spatial structure from the residuals of an over-segmentation (class means
removed) so only WITHIN-class autocorrelation remains, and build the null
from that.

WC-gap null construction:
  1. over-segment (k = K_OVER) -> class means -> residual = X - mean[label]
  2. phase-randomize the residual image (within-class spectrum, no discrete
     structure), shared phase across bands
  3. rescale per band to the DATA total variance, impose data band
     cross-covariance, add data mean

Target result across the three worlds: NULL->1, TREND->1, STRUCT->4.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans

RNG = np.random.default_rng(11)
H = W = 64
B = 6
KMAX = 8
N_REF = 8
K_OVER = 12


# ---- worlds (same generators as exp 02) ---------------------------------
def grf(sigma, rng):
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def null_image(rng):
    return np.stack([grf(4.0, rng) for _ in range(B)], -1).reshape(-1, B)


def trend_image(rng):
    yy, xx = np.mgrid[0:H, 0:W] / H
    bands = []
    for _ in range(B):
        a, b, c = rng.standard_normal(3)
        p = a * xx + b * yy + c * xx * yy
        p = (p - p.mean()) / (p.std() + 1e-12)
        bands.append(3.0 * p + 0.4 * grf(3.0, rng))
    return np.stack(bands, -1).reshape(-1, B)


def struct_image(rng, k_true=4, sep=2.0):
    fields = np.stack([grf(6.0, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1).ravel()
    means = rng.standard_normal((k_true, B)) * sep
    img = means[labels].reshape(H, W, B).astype(float)
    img += np.stack([grf(2.0, rng) for _ in range(B)], -1)
    return img.reshape(-1, B), labels


# ---- nulls --------------------------------------------------------------
def km(X, k, rng):
    return MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024,
                           random_state=int(rng.integers(1e9))).fit(X)


def uniform_ref(X, rng):
    return rng.uniform(X.min(0), X.max(0), size=X.shape)


def _phase_surrogate(field_img, rng):
    """Phase-randomize a (H,W,B) image with shared phase; return (N,B)."""
    phi = np.angle(fft2(rng.standard_normal((H, W))))
    out = np.empty_like(field_img)
    for b in range(B):
        out[..., b] = np.real(ifft2(np.abs(fft2(field_img[..., b])) *
                                    np.exp(1j * phi)))
    return out.reshape(-1, B)


def raw_phase_ref(X, rng):
    return _phase_surrogate(X.reshape(H, W, B), rng)


def make_within_class_ref(X, rng):
    """Return a closure that draws within-class-null references for X."""
    lab = km(X, K_OVER, rng).predict(X)
    means = np.stack([X[lab == j].mean(0) if (lab == j).any() else X.mean(0)
                      for j in range(K_OVER)])
    resid = (X - means[lab]).reshape(H, W, B)
    data_mean = X.mean(0)
    data_std = X.std(0)
    # data band cross-covariance (correlation, scaled later by std)
    Xc = (X - data_mean) / (data_std + 1e-12)
    L = np.linalg.cholesky(np.corrcoef(Xc, rowvar=False) +
                           1e-6 * np.eye(B))

    def ref(_X, rng):
        s = _phase_surrogate(resid, rng)              # within-class spectrum
        s = (s - s.mean(0)) / (s.std(0) + 1e-12)      # standardize
        s = s @ L.T                                   # impose cross-cov
        s = (s - s.mean(0)) / (s.std(0) + 1e-12)
        return s * data_std + data_mean               # match data variance+mean
    return ref


# ---- gap ----------------------------------------------------------------
def logW(X, k, rng):
    if k == 1:
        return np.log(((X - X.mean(0)) ** 2).sum() + 1e-12)
    return np.log(km(X, k, rng).inertia_ + 1e-12)


def gap_curve(X, ref_fn, rng):
    ks = np.arange(1, KMAX + 1)
    lw = np.array([logW(X, k, rng) for k in ks])
    ref = np.array([[logW(ref_fn(X, rng), k, rng) for k in ks]
                    for _ in range(N_REF)])
    return ks, ref.mean(0) - lw


def k_peak(ks, g):
    return int(ks[int(np.argmax(g))])


def run(name, X, k_true, rng):
    ks, gu = gap_curve(X, uniform_ref, rng)
    _, gp = gap_curve(X, raw_phase_ref, rng)
    _, gw = gap_curve(X, make_within_class_ref(X, rng), rng)
    print(f"\n=== {name}  (truth k={k_true}) ===")
    print(f"  uniform-null gap   -> k={k_peak(ks, gu)}")
    print(f"  raw-phase gap      -> k={k_peak(ks, gp)}")
    print(f"  within-class gap   -> k={k_peak(ks, gw)}   [ours]")
    return dict(ks=ks, gu=gu, gp=gp, gw=gw, k_true=k_true)


def main():
    Xn = null_image(RNG)
    Xt = trend_image(RNG)
    Xs, _ = struct_image(RNG)
    res = [run("NULL", Xn, 1, RNG),
           run("TREND", Xt, 1, RNG),
           run("STRUCT", Xs, 4, RNG)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = ["NULL", "TREND", "STRUCT"]
    for a, r, n in zip(ax, res, names):
        a.plot(r["ks"], r["gu"], "o-", label="uniform")
        a.plot(r["ks"], r["gp"], "s-", label="raw-phase")
        a.plot(r["ks"], r["gw"], "^-", lw=2, label="within-class (ours)")
        a.axvline(r["k_true"], color="k", ls="--", lw=1)
        a.set_title(f"{n} (truth k={r['k_true']})")
        a.set_xlabel("k"); a.set_ylabel("Gap(k)"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("experiments/figs/03_within_class.png", dpi=130)
    print("\nfigure -> experiments/figs/03_within_class.png")


if __name__ == "__main__":
    main()
