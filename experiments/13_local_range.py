"""
Exp 13 — local (within-class) range to restore full-scene recall.

Tests that ess_dip_local (adaptive detrend + LOCAL range + ensemble):
  (1) recovers more classes on FULL multi-class scenes than the global-range
      version (which collapsed to k=1);
  (2) does NOT break specificity on single-class windows;
  (3) does NOT break the synthetic null/trend/struct behaviour.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import importlib.util
import methods as M

spec = importlib.util.spec_from_file_location("rd", "experiments/10_realdata.py")
rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)


def load_s2():
    cube = np.load("data/s2_cube.npy"); lab = np.load("data/s2_labels.npy")
    H, W, Bb = cube.shape
    f = cube.reshape(-1, Bb)
    return ((f - f.mean(0)) / (f.std(0) + 1e-9)).reshape(H, W, Bb), lab


print("=" * 64)
print("(1) FULL-SCENE recall: global-range vs LOCAL-range")
print("=" * 64)
rng = np.random.default_rng(99)
scenes = []
ipc, ipg, _ = rd.load_pca(*rd.DATASETS["IndianPines"]); scenes.append(("IndianPines", ipc, 16))
spc, spg, _ = rd.load_pca(*rd.DATASETS["Salinas"]); scenes.append(("Salinas", spc, 16))
s2c, s2l = load_s2(); scenes.append(("Sentinel2", s2c, 4))

print(f"{'scene':12s} {'truth':>5s} {'R_glob':>7s} {'R_loc':>6s} "
      f"{'global k':>9s} {'LOCAL k':>8s}")
for name, cube, truth in scenes:
    Rg = M.estimate_range(cube)
    Rl = M.estimate_range_local(cube)
    kg = M.ess_dip_adaptive(cube, rng, kcap=20)
    kl = M.ess_dip_local(cube, rng, kcap=20)
    print(f"{name:12s} {truth:>5d} {Rg:>7.1f} {Rl:>6.1f} {kg:>9d} {kl:>8d}")

print("\n" + "=" * 64)
print("(2) SPECIFICITY preserved? Sentinel-2 single-crop windows (truth 1)")
print("=" * 64)
WIN = 16
wins = []
H, W = s2l.shape
for y in range(0, H - WIN + 1, 8):
    for x in range(0, W - WIN + 1, 8):
        blk = s2l[y:y + WIN, x:x + WIN].ravel()
        v, c = np.unique(blk, return_counts=True)
        if c.max() / blk.size >= 0.9:
            wins.append((y, x))
wins = wins[:30]
ks = [M.ess_dip_local(s2c[y:y + WIN, x:x + WIN], np.random.default_rng(i))
      for i, (y, x) in enumerate(wins)]
print(f"  {len(wins)} windows: k=1 rate = {np.mean(np.array(ks) == 1):.2f}  "
      f"mean k = {np.mean(ks):.2f}")

print("\n" + "=" * 64)
print("(3) SYNTHETIC behaviour preserved? (8 seeds each)")
print("=" * 64)
for kind, kw, truth in [("null", {}, 1), ("trend", {"trend_amp": 3.0}, 1),
                        ("struct", {"k_true": 4, "sep": 3.0}, 4),
                        ("mixed", {"k_true": 3, "sep": 3.0, "trend_amp": 1.5}, 3)]:
    ks = []
    for s in range(8):
        r = np.random.default_rng(500 + s)
        img, t, _ = M.make_world(kind=kind, rng=r, **kw)
        ks.append(M.ess_dip_local(img, r))
    print(f"  {kind:7s} truth={truth}: k={ks}  acc={np.mean(np.array(ks)==truth):.2f}")
