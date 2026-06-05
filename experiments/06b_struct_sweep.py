"""
Exp 06b — confirm STRUCT=3 in exp 06 was a separability artifact, not a
structural failure of declustering + dip-means. Sweep class separation `sep`
and number of true classes; check recovered k.
"""

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dm", "experiments/06_dipmeans_decluster.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

RNG = np.random.default_rng(21)


def struct(k_true, sep, rng):
    fields = np.stack([dm.grf(9.0, rng) for _ in range(k_true)], -1)
    labels = fields.argmax(-1)
    means = rng.standard_normal((k_true, dm.B)) * sep
    img = means[labels] + np.stack([dm.grf(3.0, rng) for _ in range(dm.B)], -1)
    return img


print("STRUCT sweep (declustering + dip-means):")
print(f"{'k_true':>6} {'sep':>5}   recovered k (median over offsets)")
for k_true in (3, 4, 5):
    for sep in (2.0, 3.0, 4.0):
        img = struct(k_true, sep, RNG)
        k = dm.run(f"k={k_true} sep={sep}", img, k_true, RNG)
        marker = "OK" if k == k_true else ""
        print(f"{k_true:>6} {sep:>5}   -> {k}   {marker}")
