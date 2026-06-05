"""
De-risking experiment for the spatial cluster-number selection paper.

Question under test
-------------------
On spatially autocorrelated multiband rasters, do classical k-selection
criteria (Tibshirani gap statistic with a uniform null, and the average
silhouette) OVER-DETECT the number of clusters? And does replacing the
null with Fourier phase-randomized surrogates (which preserve spatial
autocorrelation but destroy discrete structure) fix it?

If classical methods do NOT over-detect here, there is no paper. This
script is the make-or-break test.

Two synthetic multiband "images":
  (A) NULL world  : pure spatially-autocorrelated Gaussian fields, NO
                    discrete land-cover classes. Correct answer: k = 1.
  (B) STRUCTURED  : K contiguous land-cover patches, each with its own
                    mean spectrum, embedded in autocorrelated noise.
                    Correct answer: k = K_true.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

RNG = np.random.default_rng(7)
H = W = 80          # image size  (6400 pixels)
B = 6              # spectral bands (Sentinel-2-like)
KMAX = 8
N_REF = 12         # reference replicates for each gap null
SIL_SAMPLE = 2000


# --------------------------------------------------------------------------
# field generators
# --------------------------------------------------------------------------
def grf(h, w, sigma, rng):
    """One spatially-autocorrelated Gaussian field (smoothed white noise)."""
    f = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def null_image(sigma=4.0, rng=RNG):
    """B autocorrelated bands, NO discrete classes. Truth: k = 1."""
    img = np.stack([grf(H, W, sigma, rng) for _ in range(B)], axis=-1)
    return img.reshape(-1, B), np.zeros(H * W, dtype=int)


def structured_image(k_true=4, sigma_field=6.0, sigma_noise=2.0,
                     sep=2.0, rng=RNG):
    """K contiguous patches + autocorrelated noise. Truth: k = k_true."""
    # contiguous label map: argmax over k_true smooth fields
    fields = np.stack([grf(H, W, sigma_field, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1).ravel()
    # per-class mean spectra, separated by `sep` in B-space
    means = rng.standard_normal((k_true, B)) * sep
    img = means[labels].reshape(H, W, B).astype(float)
    # add spatially autocorrelated noise (the realistic confound)
    noise = np.stack([grf(H, W, sigma_noise, rng) for _ in range(B)], -1)
    img += noise
    return img.reshape(-1, B), labels


# --------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------
def uniform_reference(X, rng):
    """Tibshirani gap null: uniform over the feature bounding box."""
    lo, hi = X.min(0), X.max(0)
    return rng.uniform(lo, hi, size=X.shape)


def phase_surrogate(X, rng):
    """
    AAFT-style surrogate of a multiband image: preserve each band's power
    spectrum (=> same autocorrelation / variogram) and the inter-band phase
    alignment (=> same cross-spectra), but destroy discrete structure.
    """
    bands = X.reshape(H, W, B)
    # one shared random phase field (Hermitian-symmetric => real output)
    phi = np.angle(fft2(rng.standard_normal((H, W))))
    out = np.empty_like(bands)
    for b in range(B):
        amp = np.abs(fft2(bands[..., b]))
        out[..., b] = np.real(ifft2(amp * np.exp(1j * phi)))
    return out.reshape(-1, B)


# --------------------------------------------------------------------------
# gap statistic
# --------------------------------------------------------------------------
def pooled_log_W(X, k, rng):
    if k == 1:
        return np.log(((X - X.mean(0)) ** 2).sum())
    km = KMeans(n_clusters=k, n_init=4, random_state=int(rng.integers(1e9)))
    lab = km.fit_predict(X)
    w = sum(((X[lab == j] - X[lab == j].mean(0)) ** 2).sum()
            for j in range(k))
    return np.log(w + 1e-12)


def gap_curve(X, reference_fn, rng):
    ks = np.arange(1, KMAX + 1)
    logW = np.array([pooled_log_W(X, k, rng) for k in ks])
    ref = np.zeros((N_REF, len(ks)))
    for r in range(N_REF):
        Xr = reference_fn(X, rng)
        ref[r] = [pooled_log_W(Xr, k, rng) for k in ks]
    gap = ref.mean(0) - logW
    sk = ref.std(0) * np.sqrt(1 + 1.0 / N_REF)
    return ks, gap, sk


def select_k_gap(ks, gap, sk):
    """Tibshirani rule: smallest k with gap(k) >= gap(k+1) - s(k+1)."""
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def select_k_silhouette(X, rng):
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    best_k, best_s = 2, -1
    for k in range(2, KMAX + 1):
        lab = KMeans(n_clusters=k, n_init=4,
                     random_state=int(rng.integers(1e9))).fit_predict(Xs)
        s = silhouette_score(Xs, lab)
        if s > best_s:
            best_s, best_k = s, k
    return best_k  # silhouette structurally cannot return k=1


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def evaluate(name, X, truth, k_true, rng):
    ks, g_uni, s_uni = gap_curve(X, uniform_reference, rng)
    _, g_vg, s_vg = gap_curve(X, phase_surrogate, rng)
    k_uni = select_k_gap(ks, g_uni, s_uni)
    k_vg = select_k_gap(ks, g_vg, s_vg)
    k_sil = select_k_silhouette(X, rng)

    print(f"\n=== {name}  (truth k = {k_true}) ===")
    print(f"  classical gap (uniform null) -> k = {k_uni}")
    print(f"  silhouette                   -> k = {k_sil}")
    print(f"  VG-gap (phase surrogate)     -> k = {k_vg}   [ours]")
    if k_true > 1:
        for k, lab_name in [(k_uni, "uni"), (k_vg, "vg")]:
            km = KMeans(n_clusters=max(k, 1), n_init=4,
                        random_state=0).fit_predict(X)
            print(f"    ARI vs truth @k={k} ({lab_name}) = "
                  f"{adjusted_rand_score(truth, km):.3f}")
    return dict(ks=ks, g_uni=g_uni, g_vg=g_vg,
                k_uni=k_uni, k_vg=k_vg, k_sil=k_sil, k_true=k_true)


def main():
    Xn, tn = null_image(rng=RNG)
    Xs, ts = structured_image(k_true=4, rng=RNG)
    rA = evaluate("NULL world", Xn, tn, 1, RNG)
    rB = evaluate("STRUCTURED world", Xs, ts, 4, RNG)

    # money figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, r, ttl in [(ax[0], rA, "NULL world (truth k=1)"),
                      (ax[1], rB, "STRUCTURED world (truth k=4)")]:
        a.plot(r["ks"], r["g_uni"], "o-", label="classical gap (uniform)")
        a.plot(r["ks"], r["g_vg"], "s-", label="VG-gap (phase surrogate)")
        a.axvline(r["k_true"], color="k", ls="--", lw=1, label="true k")
        a.set_title(ttl)
        a.set_xlabel("k")
        a.set_ylabel("Gap(k)")
        a.legend(fontsize=8)
    fig.tight_layout()
    out = "experiments/figs/01_derisk.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
