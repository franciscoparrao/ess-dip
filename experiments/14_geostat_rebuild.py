"""
Exp 14 — reviewer items R1 + R2 (Mathematical Geosciences).

R1  Effective sample size from the variogram INTEGRAL RANGE (Cressie), not the
    ad-hoc ell^2. Method `ess_dip_vario` = ESS-Dip whose decorrelation area is
    the median tile integral range from a fitted variogram model.

R2  A Gaussian-simulation (SGS) null: the leading homogeneity test of each
    scene is calibrated against spectral GRF realisations matching the fitted
    variogram of the split projection -- the geostatistical neutral model --
    instead of (or alongside) the ESS shortcut.

We compare, on the four synthetic worlds, the ell^2 shortcut, the integral-range
ESS, and the SGS-null homogeneity decision.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import diptest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import methods as M
import geostat as G


# ---- R1: ESS-Dip with integral-range decorrelation area -----------------
def ess_dip_vario(img, rng, tau=0.25, n_runs=5, tile=24, kcap=M.KMAX):
    work = M.detrend_poly(img) if M.poly_r2(img) > tau else img
    area = G.decorrelation_area_local(work, tile=min(tile, work.shape[0] // 2))
    ks = [M.ess_dip(work, rng, area_val=area, kcap=kcap) for _ in range(n_runs)]
    return int(np.median(ks)), area


# ---- R2: SGS-null homogeneity test on the whole scene -------------------
def sgs_homogeneity(img, rng, alpha=0.05, n_null=120, tau=0.25):
    """Test K>=2 vs K=1 by calibrating the dip of the leading 2-means split
    projection against GRF simulations matching its variogram."""
    work = M.detrend_poly(img) if M.poly_r2(img) > tau else img
    H, W, B = work.shape
    X = work.reshape(-1, B)
    lab = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(X)
    d = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    yimg = (X @ d).reshape(H, W)
    D = diptest.dipstat(yimg.ravel())
    model = G.fit_variogram(yimg)
    null = G.sgs_dip_null(yimg, model, rng, n_null=n_null)
    p = float(np.mean(null >= D))
    return p, model


def main():
    worlds = [("NULL", {}, 1), ("TREND", {"trend_amp": 3.0}, 1),
              ("STRUCT", {"k_true": 4, "sep": 3.0}, 4),
              ("MIXED", {"k_true": 3, "sep": 3.0, "trend_amp": 1.5}, 3)]

    print("R1  integral-range ESS vs ell^2 shortcut, and R2 SGS-null homogeneity")
    print(f"{'world':7s} {'truth':>5s} | {'K(ell^2)':>8s} {'K(vario)':>8s} "
          f"{'decorr_area':>11s} | {'SGS p':>7s} {'SGS decision':>13s}")
    for name, kw, truth in worlds:
        ks_short, ks_vario, areas, ps = [], [], [], []
        for s in range(6):
            rng = np.random.default_rng(700 + s)
            img, t, _ = M.make_world(kind=name.lower(), rng=rng, **kw)
            ks_short.append(M.ess_dip_local(img, rng))
            kv, area = ess_dip_vario(img, rng)
            ks_vario.append(kv); areas.append(area)
            p, _ = sgs_homogeneity(img, rng)
            ps.append(p)
        kshort = int(np.median(ks_short)); kvario = int(np.median(ks_vario))
        psmed = float(np.median(ps))
        # SGS decision: reject homogeneity (structure present) if median p < 0.05
        decision = "K>=2 (reject)" if psmed < 0.05 else "K=1 (no rej.)"
        ok = "OK" if ((truth == 1) == (psmed >= 0.05)) else "  "
        print(f"{name:7s} {truth:>5d} | {kshort:>8d} {kvario:>8d} "
              f"{np.median(areas):>11.0f} | {psmed:>7.3f} {decision:>13s} {ok}")


if __name__ == "__main__":
    main()
