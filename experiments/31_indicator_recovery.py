"""
Exp 31 (blind review, central concern) — show the decorrelation-area spectrum
a_int (field integral range) > a_I (indicator integral range) > ell^2 (1/e proxy)
translates into a recovery spectrum: the mean-based field integral range
over-corrects (under-detects), the indicator integral range recovers more, and
the ell^2 proxy recovers the truth. All three are estimated LOCALLY (tile
median), matching the method. This makes the calibration geostatistically
load-bearing: the right object for a distributional functional is the indicator
integral range, not the field integral range.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import methods as M

T, REPS = 24, 9
WORLDS = [("Null",   dict(kind="null", field_sigma=6.0), 1),
          ("Trend",  dict(kind="trend", trend_amp=3.0), 1),
          ("Struct", dict(kind="struct", k_true=4, sep=3.0, field_sigma=6.0), 4),
          ("Mixed",  dict(kind="mixed", k_true=3, sep=3.0, field_sigma=6.0,
                          trend_amp=3.0), 3)]


def tile_areas(field):
    """Local (tile-median) ell^2, indicator integral range, field integral range."""
    H, W, _ = field.shape
    L, AI, AINT = [], [], []
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            f = field[y:y + T, x:x + T]
            pc = f.reshape(-1, f.shape[-1])
            pc = pc @ np.linalg.svd(pc - pc.mean(0), full_matrices=False)[2][0]
            pc = (pc - pc.mean()).reshape(T, T)
            ac = np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real
            ac = np.fft.fftshift(ac); ac /= ac.max()
            cy = cx = T // 2
            yy, xx = np.mgrid[0:T, 0:T]
            r = np.round(np.hypot(yy - cy, xx - cx)).astype(int)
            prof = np.array([ac[r == k].mean() for k in range(cy)])
            rho = np.clip(prof, 0.0, 0.999999); rr = np.arange(len(rho))
            a_int = float(np.sum(rho * 2 * np.pi * rr))
            a_I = float(np.sum((2 / np.pi) * np.arcsin(rho) * 2 * np.pi * rr))
            below = np.where(prof < 1 / np.e)[0]
            ell = float(below[0]) if below.size else float(len(prof))
            L.append(ell * ell); AI.append(a_I); AINT.append(a_int)
    return (max(1.0, np.median(L)), max(1.0, np.median(AI)),
            max(1.0, np.median(AINT)))


def main():
    print(f"  {'World':7} {'trueK':>5} | {'K(a_int)':>8} {'K(a_I)':>7} "
          f"{'K(l^2)':>7} | areas l2/aI/aint")
    for name, cell, tk in WORLDS:
        kii, kI, kl, ratios = [], [], [], []
        for r in range(REPS):
            rng = np.random.default_rng(31000 + r)
            img, _, _ = M.make_world(rng=rng, **cell)
            work = M.detrend_poly(img) if M.poly_r2(img) > 0.25 else img
            a_l, a_I, a_int = tile_areas(work)
            ratios.append((a_l, a_I, a_int))
            kii.append(M.ess_dip(work, rng, area_val=a_int, kcap=8))
            kI.append(M.ess_dip(work, rng, area_val=a_I, kcap=8))
            kl.append(M.ess_dip(work, rng, area_val=a_l, kcap=8))
        a_l, a_I, a_int = np.median([x[0] for x in ratios]), \
            np.median([x[1] for x in ratios]), np.median([x[2] for x in ratios])
        print(f"  {name:7} {tk:>5} | {int(np.median(kii)):>8} "
              f"{int(np.median(kI)):>7} {int(np.median(kl)):>7} | "
              f"{a_l:.0f}/{a_I:.0f}/{a_int:.0f}")


if __name__ == "__main__":
    main()
