"""
De-risking v2 — faster, with corrected selector and the regime that
actually breaks the classical gap: spatial TREND/gradient.

Findings to confirm:
  (1) silhouette / Davies-Bouldin / Calinski-Harabasz over-detect under
      spatial autocorrelation.
  (2) VG-gap (phase-surrogate null) selected by argmax/peak recovers the
      truth where the classical uniform-null gap fails.

Worlds (all multiband rasters):
  NULL   : stationary autocorrelated GRF, no classes        -> truth k=1
  TREND  : smooth large-scale gradient, no discrete classes -> truth k=1
  STRUCT : K contiguous patches + autocorrelated noise      -> truth k=4
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)

RNG = np.random.default_rng(7)
H = W = 64
B = 6
KMAX = 8
N_REF = 10
SIL_SAMPLE = 1500


def grf(sigma, rng):
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def null_image(rng):
    img = np.stack([grf(4.0, rng) for _ in range(B)], -1)
    return img.reshape(-1, B), np.zeros(H * W, int)


def trend_image(rng):
    """Smooth large-scale gradients (planar + low-freq), no discrete classes."""
    yy, xx = np.mgrid[0:H, 0:W] / H
    bands = []
    for _ in range(B):
        a, b, c = rng.standard_normal(3)
        plane = a * xx + b * yy + c * xx * yy
        plane = (plane - plane.mean()) / (plane.std() + 1e-12)
        bands.append(3.0 * plane + 0.4 * grf(3.0, rng))   # trend-dominated
    return np.stack(bands, -1).reshape(-1, B), np.zeros(H * W, int)


def struct_image(rng, k_true=4, sep=2.0):
    fields = np.stack([grf(6.0, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1).ravel()
    means = rng.standard_normal((k_true, B)) * sep
    img = means[labels].reshape(H, W, B).astype(float)
    img += np.stack([grf(2.0, rng) for _ in range(B)], -1)
    return img.reshape(-1, B), labels


def uniform_ref(X, rng):
    return rng.uniform(X.min(0), X.max(0), size=X.shape)


def phase_ref(X, rng):
    bands = X.reshape(H, W, B)
    phi = np.angle(fft2(rng.standard_normal((H, W))))
    out = np.empty_like(bands)
    for b in range(B):
        out[..., b] = np.real(ifft2(np.abs(fft2(bands[..., b])) *
                                    np.exp(1j * phi)))
    return out.reshape(-1, B)


def logW(X, k, rng):
    if k == 1:
        return np.log(((X - X.mean(0)) ** 2).sum() + 1e-12)
    km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024,
                         random_state=int(rng.integers(1e9)))
    km.fit(X)
    return np.log(km.inertia_ + 1e-12)


def gap(X, ref_fn, rng):
    ks = np.arange(1, KMAX + 1)
    lw = np.array([logW(X, k, rng) for k in ks])
    ref = np.array([[logW(ref_fn(X, rng), k, rng) for k in ks]
                    for _ in range(N_REF)])
    g = ref.mean(0) - lw
    s = ref.std(0) * np.sqrt(1 + 1.0 / N_REF)
    return ks, g, s


def k_tibshirani(ks, g, s):
    for i in range(len(ks) - 1):
        if g[i] >= g[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def k_peak(ks, g):
    """argmax, but if curve is monotone-decreasing from k=1 -> k=1."""
    return int(ks[int(np.argmax(g))])


def internal_indices(X, rng):
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    sil, ch, db = {}, {}, {}
    for k in range(2, KMAX + 1):
        lab = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024,
                              random_state=int(rng.integers(1e9))).fit_predict(Xs)
        sil[k] = silhouette_score(Xs, lab)
        ch[k] = calinski_harabasz_score(Xs, lab)
        db[k] = davies_bouldin_score(Xs, lab)
    return (max(sil, key=sil.get), max(ch, key=ch.get), min(db, key=db.get))


def run(name, X, truth, k_true, rng):
    ks, gu, su = gap(X, uniform_ref, rng)
    _, gv, sv = gap(X, phase_ref, rng)
    ksil, kch, kdb = internal_indices(X, rng)
    print(f"\n=== {name}  (truth k={k_true}) ===")
    print(f"  silhouette                 -> k={ksil}")
    print(f"  Calinski-Harabasz          -> k={kch}")
    print(f"  Davies-Bouldin             -> k={kdb}")
    print(f"  classical gap  (Tibshirani)-> k={k_tibshirani(ks, gu, su)}")
    print(f"  classical gap  (peak)      -> k={k_peak(ks, gu)}")
    print(f"  VG-gap         (peak)      -> k={k_peak(ks, gv)}   [ours]")
    return dict(ks=ks, gu=gu, gv=gv, k_true=k_true)


def main():
    worlds = [("NULL", *null_image(RNG), 1),
              ("TREND", *trend_image(RNG), 1),
              ("STRUCT", *struct_image(RNG), 4)]
    res = [run(n, X, t, k, RNG) for n, X, t, k in worlds]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, r, (n, *_), in zip(ax, res, worlds):
        a.plot(r["ks"], r["gu"], "o-", label="classical gap (uniform)")
        a.plot(r["ks"], r["gv"], "s-", label="VG-gap (phase)")
        a.axvline(r["k_true"], color="k", ls="--", lw=1, label="true k")
        a.set_title(f"{n}  (truth k={r['k_true']})")
        a.set_xlabel("k"); a.set_ylabel("Gap(k)"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("experiments/figs/02_derisk_v2.png", dpi=130)
    print("\nfigure -> experiments/figs/02_derisk_v2.png")


if __name__ == "__main__":
    main()
