"""
Exp 17 — robustness to anisotropy (reviewer item C5).

The decorrelation area is estimated from a radially averaged (isotropic)
variogram. Real autocorrelation is often directional. We test whether ESS-Dip
retains specificity and recall when the field is anisotropic, varying the
anisotropy ratio (range_y / range_x) of the within-class structure.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter
import methods as M

H, P, SEEDS = 96, 6, 8


def grf_a(sy, sx, rng):
    f = gaussian_filter(rng.standard_normal((H, H)), sigma=(sy, sx), mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def null_aniso(ratio, rng, base=3.0):
    return np.stack([grf_a(base * ratio, base, rng) for _ in range(P)], -1)


def struct_aniso(ratio, rng, k=4, sep=3.0, base=2.0):
    fields = np.stack([grf_a(6, 6, rng) for _ in range(k)], -1)   # isotropic patches
    labels = fields.argmax(-1)
    means = rng.standard_normal((k, P)) * sep
    noise = np.stack([grf_a(base * ratio, base, rng) for _ in range(P)], -1)
    return means[labels] + noise


def main():
    print("Anisotropy robustness (ratio = range_y / range_x; ratio=1 isotropic)")
    print(f"{'ratio':>5} | {'NULL spec (k=1)':>16} | {'STRUCT acc (k=4)':>17} "
          f"| {'gap NULL spec':>13}")
    for ratio in (1, 2, 4, 8):
        spec, acc, gap_spec = [], [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(900 + s)
            spec.append(M.ess_dip_local(null_aniso(ratio, rng), rng) == 1)
            acc.append(M.ess_dip_local(struct_aniso(ratio, rng), rng) == 4)
            gap_spec.append(M.classical_gap(null_aniso(ratio, rng), rng) == 1)
        print(f"{ratio:>5} | {np.mean(spec):>16.2f} | {np.mean(acc):>17.2f} "
              f"| {np.mean(gap_spec):>13.2f}")


if __name__ == "__main__":
    main()
