"""
Exp 30 (blind review, central concern) — make the effective-sample-size
calibration geostatistically load-bearing. For a functional of the empirical
DISTRIBUTION (the dip), the relevant decorrelation area is the integral range of
the level-crossing (indicator) process, not of the field. For a standardised
Gaussian field with correlogram rho, the median-level indicator correlogram is
the Gaussian arcsin law rho_I(h) = (2/pi) arcsin rho(h) <= rho(h), so the
indicator integral range a_I = int rho_I <= a_int = int rho: the field integral
range (derived for the sample MEAN) over-corrects. We check numerically that
(i) a_I < a_int and (ii) the 1/e proxy ell^2 approximates a_I, justifying it.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import methods as M

SIGMAS = (2.0, 3.0, 4.0, 6.0, 8.0)
REPS, H, W = 40, 96, 96


def radial_ac(field):
    f = (field - field.mean()) / (field.std() + 1e-12)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(f)) ** 2).real
    ac = np.fft.fftshift(ac); ac /= ac.max()
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.round(np.hypot(yy - cy, xx - cx)).astype(int)
    rmax = min(cy, cx)
    prof = np.array([ac[r == k].mean() for k in range(rmax)])
    return prof


def areas(prof):
    rho = np.clip(prof, 0.0, 0.999999)
    r = np.arange(len(rho))
    # field integral range  a_int = int rho dh = sum_r rho(r) 2*pi*r
    a_int = float(np.sum(rho * 2 * np.pi * r))
    # indicator integral range via Gaussian arcsin law
    rho_I = (2.0 / np.pi) * np.arcsin(rho)
    a_I = float(np.sum(rho_I * 2 * np.pi * r))
    # 1/e range
    below = np.where(prof < 1.0 / np.e)[0]
    ell = float(below[0]) if below.size else float(len(prof))
    return a_int, a_I, ell * ell


def main():
    print(f"  {'sigma':>5} {'ell^2':>8} {'a_I (ind.)':>11} {'a_int (field)':>13} "
          f"{'a_int/a_I':>9} {'ell^2/a_I':>9}")
    rows = []
    for s in SIGMAS:
        ai, aI, l2 = [], [], []
        for r in range(REPS):
            rng = np.random.default_rng(30000 + int(s * 100) + r)
            field = M._grf(H, W, s, rng)
            a_int, a_I, ell2 = areas(radial_ac(field))
            ai.append(a_int); aI.append(a_I); l2.append(ell2)
        mi, mI, ml = np.median(ai), np.median(aI), np.median(l2)
        rows.append((s, ml, mI, mi))
        print(f"  {s:>5.0f} {ml:>8.1f} {mI:>11.1f} {mi:>13.1f} "
              f"{mi/mI:>9.2f} {ml/mI:>9.2f}")
    print("\n  Interpretation:")
    print("   a_int/a_I > 1  : the field integral range over-corrects (too large)")
    print("   ell^2/a_I ~ 1  : the 1/e proxy approximates the indicator integral range")


if __name__ == "__main__":
    main()
