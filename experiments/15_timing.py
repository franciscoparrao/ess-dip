"""
Exp 15 — computational performance (reviewer item C2).

Wall-clock time and peak memory of ESS-Dip versus image size N (pixels) and
band count p, against the gap statistic, with the empirical scaling exponent.
Substantiates the near-linear claim of Section 3.10 and the higher cost of the
exact Gaussian-simulation reference.

Single-threaded (OMP=1) for clean scaling.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import time, csv, tracemalloc
import numpy as np
from scipy.ndimage import gaussian_filter
import methods as M
import geostat as G

REPS = 3
OUT = "experiments/results"
os.makedirs(OUT, exist_ok=True)


def grf(H, sigma, rng):
    f = gaussian_filter(rng.standard_normal((H, H)), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def scene(H, p, rng, k=4, sep=3.0):
    fields = np.stack([grf(H, 6.0, rng) for _ in range(k)], -1)
    labels = fields.argmax(-1)
    means = rng.standard_normal((k, p)) * sep
    return means[labels] + np.stack([grf(H, 2.0, rng) for _ in range(p)], -1)


def timed(fn, img, rng, reps=REPS):
    ts = []
    for _ in range(reps):
        r = np.random.default_rng(int(rng.integers(1e9)))
        t0 = time.perf_counter(); fn(img, r); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(5)

    print("=== Scaling in N (pixels), p=6 bands ===")
    rows = []
    sizes = [48, 64, 96, 128, 160, 192, 256]
    print(f"{'H':>5} {'N':>8} {'ESS-Dip(s)':>11} {'gap(s)':>9} {'speedup':>8}")
    for H in sizes:
        img = scene(H, 6, rng)
        t_ess = timed(M.ess_dip_local, img, rng)
        t_gap = timed(M.classical_gap, img, rng) if H <= 192 else float("nan")
        rows.append(dict(axis="N", H=H, N=H * H, p=6,
                         ess_dip=t_ess, gap=t_gap))
        sp = (t_gap / t_ess) if t_gap == t_gap else float("nan")
        print(f"{H:>5} {H*H:>8} {t_ess:>11.3f} {t_gap:>9.3f} {sp:>8.1f}")

    # fitted scaling exponent for ESS-Dip
    N = np.array([r["N"] for r in rows], float)
    te = np.array([r["ess_dip"] for r in rows], float)
    slope = np.polyfit(np.log(N), np.log(te), 1)[0]
    print(f"  ESS-Dip empirical exponent  d log t / d log N = {slope:.2f}  "
          f"(1.0 = linear)")

    print("\n=== Scaling in p (bands), H=128 ===")
    print(f"{'p':>5} {'ESS-Dip(s)':>11}")
    for p in (3, 6, 10, 20, 40):
        img = scene(128, p, rng)
        t_ess = timed(M.ess_dip_local, img, rng)
        rows.append(dict(axis="p", H=128, N=128 * 128, p=p, ess_dip=t_ess,
                         gap=float("nan")))
        print(f"{p:>5} {t_ess:>11.3f}")

    print("\n=== Exact SGS reference cost (homogeneity test, B=120) ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "e14", "experiments/14_geostat_rebuild.py")
    e14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e14)
    print(f"{'H':>5} {'SGS(s)':>9}")
    for H in (64, 96, 128):
        img = scene(H, 6, rng)
        t = timed(lambda im, r: e14.sgs_homogeneity(im, r), img, rng, reps=2)
        rows.append(dict(axis="sgs", H=H, N=H * H, p=6, ess_dip=float("nan"),
                         gap=float("nan"), sgs=t))
        print(f"{H:>5} {t:>9.3f}")

    print("\n=== Peak memory, ESS-Dip @ 256x256x6 ===")
    img = scene(256, 6, rng)
    tracemalloc.start()
    M.ess_dip_local(img, np.random.default_rng(1))
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  peak Python allocation = {peak/1e6:.1f} MB  "
          f"(input cube {img.nbytes/1e6:.1f} MB)")

    with open(os.path.join(OUT, "timing.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["axis", "H", "N", "p", "ess_dip",
                                            "gap", "sgs"], extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print(f"\nslope={slope:.2f}; results -> {OUT}/timing.csv")


if __name__ == "__main__":
    main()
