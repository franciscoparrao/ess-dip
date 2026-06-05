"""
Exp 16 — comparison against Hennig & Lin (2015), the method ESS-Dip extends
(reviewer item R2, second half).

Hennig & Lin calibrate a cluster-validity index against its distribution under
a problem-specific null model (parametric bootstrap) and use the calibrated
index to test homogeneity and choose K. We implement their framework with the
average silhouette width (ASW) and two null models:

  HL(iid) : a single multivariate Gaussian -- the naive, autocorrelation-blind
            application;
  HL(grf) : a Fourier phase-randomized surrogate preserving the per-band
            variogram -- their autocorrelation-aware variant, adapted to the
            continuous-field setting.

The calibrated index is V(K) = (ASW_data(K) - mean_b ASW_null(K)) / sd_b; the
homogeneity test rejects K=1 when max_K V(K) exceeds a threshold.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from numpy.fft import fft2, ifft2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import methods as M

KMAX, MSUB, BBOOT, THRESH = 8, 1200, 20, 1.96


def asw_curve(X, rng):
    out = {}
    for k in range(2, KMAX + 1):
        lab = MiniBatchKMeans(k, n_init=3, batch_size=512,
                              random_state=int(rng.integers(1e9))).fit_predict(X)
        out[k] = silhouette_score(X, lab)
    return out


def phase_surrogate(img, rng):
    """AAFT surrogate: preserve each band's power spectrum (variogram) with a
    shared random phase across bands; returns (H*W, p)."""
    H, W, p = img.shape
    phi = np.angle(fft2(rng.standard_normal((H, W))))
    out = np.empty((H, W, p))
    for b in range(p):
        out[..., b] = np.real(ifft2(np.abs(fft2(img[..., b])) * np.exp(1j * phi)))
    return out.reshape(-1, p)


def hennig_lin(img, rng, null="grf", m=MSUB, b=BBOOT):
    H, W, p = img.shape
    X = img.reshape(-1, p)
    idx = rng.choice(len(X), min(m, len(X)), replace=False)
    asw_d = asw_curve(X[idx], rng)
    mean, cov = X.mean(0), np.cov(X, rowvar=False) + 1e-6 * np.eye(p)
    nullmat = {k: [] for k in range(2, KMAX + 1)}
    for _ in range(b):
        if null == "iid":
            Xb = rng.multivariate_normal(mean, cov, size=len(idx))
        else:
            Xb = phase_surrogate(img, rng)[idx]
        for k, v in asw_curve(Xb, rng).items():
            nullmat[k].append(v)
    V = {k: (asw_d[k] - np.mean(nullmat[k])) / (np.std(nullmat[k]) + 1e-9)
         for k in range(2, KMAX + 1)}
    khat = max(V, key=V.get)
    return khat if V[khat] > THRESH else 1


def main():
    worlds = [("null", {}, 1), ("trend", {"trend_amp": 3.0}, 1),
              ("struct", {"k_true": 4, "sep": 3.0}, 4),
              ("struct", {"k_true": 3, "sep": 3.0}, 3),
              ("mixed", {"k_true": 3, "sep": 3.0, "trend_amp": 1.5}, 3)]
    print(f"{'world':18s} {'truth':>5s} | {'HL(iid)':>7s} {'HL(grf)':>7s} {'ESS-Dip':>7s}")
    for name, kw, truth in worlds:
        hi, hg, ed = [], [], []
        for s in range(6):
            rng = np.random.default_rng(800 + s)
            img, t, _ = M.make_world(kind=name, rng=rng, **kw)
            hi.append(hennig_lin(img, rng, "iid"))
            hg.append(hennig_lin(img, rng, "grf"))
            ed.append(M.ess_dip_local(img, rng))
        lbl = f"{name} k={truth}"
        print(f"{lbl:18s} {truth:>5d} | {int(np.median(hi)):>7d} "
              f"{int(np.median(hg)):>7d} {int(np.median(ed)):>7d}")


if __name__ == "__main__":
    main()
