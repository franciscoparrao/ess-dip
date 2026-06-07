"""
Exp 35 (lever 2) — full evaluation of the recall variant ESS-Dip-R (Gaussian
null, the H0 of Assumption ass:null, in place of the uniform least-favourable
null). Synthetic benchmark (495 scenes, B=5 ensemble) and real data (single-class
window specificity + full-scene K), to produce the numbers for the paper.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
import diptest
import methods as M

N_PCA, KCAP, ALPHA, MIN_NEFF, TAU, B_ENS = 10, 8, 0.05, 12, 0.25, 5
_GN = {}


def gpval(D, n_eff, rng):
    key = max(12, min(int(round(n_eff)), 2000))
    if key not in _GN:
        _GN[key] = np.sort([diptest.dipstat(rng.standard_normal(key))
                            for _ in range(300)])
    return float(np.mean(_GN[key] >= D))


def k_hat_R(img, rng, kcap=KCAP):
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    area = M.estimate_range_local(
        work, tile=min(24, img.shape[0] // 2, img.shape[1] // 2)) ** 2
    X = work.reshape(-1, work.shape[-1])

    def one(seed):
        rg = np.random.default_rng(seed); stack, leaves = [X], 0
        while stack:
            if leaves + len(stack) >= kcap:
                return kcap
            C = stack.pop(); n_eff = len(C) / area
            if n_eff < MIN_NEFF:
                leaves += 1; continue
            lab = KMeans(2, n_init=5,
                         random_state=int(rg.integers(1e9))).fit_predict(C)
            m0, m1 = lab == 0, lab == 1
            if m0.sum() < area or m1.sum() < area:
                leaves += 1; continue
            d = C[m1].mean(0) - C[m0].mean(0); d = d / (np.linalg.norm(d) + 1e-12)
            if gpval(diptest.dipstat(C @ d), n_eff, rg) < ALPHA:
                stack.extend([C[m0], C[m1]])
            else:
                leaves += 1
        return leaves
    return int(np.median([one(int(rng.integers(1e9))) for _ in range(B_ENS)]))


# ---- synthetic ----
def build_cells():
    c = [dict(kind="null", field_sigma=fs) for fs in (3., 6., 9.)]
    c += [dict(kind="trend", trend_amp=ta) for ta in (1.5, 3.)]
    c += [dict(kind="struct", k_true=k, sep=s, field_sigma=fs)
          for k in (2, 3, 4, 5) for s in (2., 3., 4.) for fs in (6., 9.)]
    c += [dict(kind="mixed", k_true=k, sep=3., field_sigma=6., trend_amp=ta)
          for k in (3, 4) for ta in (1.5, 3.)]
    return c


def sjob(args):
    ci, cell, r = args
    rng = np.random.default_rng(10_000 + ci * 100 + r)
    img, truth, _ = M.make_world(rng=rng, **cell)
    return cell["kind"], truth, k_hat_R(img, rng)


def synthetic():
    cells = build_cells()
    jobs = [(i, c, r) for i, c in enumerate(cells) for r in range(15)]
    res = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
        for fut in as_completed([ex.submit(sjob, a) for a in jobs]):
            res.append(fut.result())
    kind = np.array([x[0] for x in res]); tr = np.array([x[1] for x in res])
    kh = np.array([x[2] for x in res])
    cf = np.isin(kind, ["null", "trend"]); st = kind == "struct"; mx = kind == "mixed"
    spec, sa = np.mean(kh[cf] == 1), np.mean(kh[st] == tr[st])
    smae = np.mean(np.abs(kh[st] - tr[st])); ma = np.mean(kh[mx] == tr[mx])
    print("[synthetic] ESS-Dip-R (Gaussian null):")
    print(f"  spec {spec:.2f} | struct {sa:.2f} (MAE {smae:.2f}) | mixed {ma:.2f} "
          f"| balanced {np.mean([spec, sa, ma]):.3f}")
    print(f"  Table-1 row:  ESS-Dip-R & {spec:.2f} & {sa:.2f} & {smae:.2f} & "
          f"{ma:.2f} & {np.mean([spec,sa,ma]):.3f}")


# ---- real ----
def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_hsi(f, g):
    md, mg = loadmat(f), loadmat(g)
    cube = md[_key(md)].astype(float); gt = mg[_key(mg)].astype(int)
    H, W, B = cube.shape
    Xp = PCA(N_PCA, random_state=0).fit_transform(
        StandardScaler().fit_transform(cube.reshape(-1, B)))
    return Xp.reshape(H, W, N_PCA), gt


def load_s2():
    raw = np.load("data/s2_cube.npy"); gt = np.load("data/s2_labels.npy")
    H, W, B = raw.shape; f = raw.reshape(-1, B)
    return ((f - f.mean(0)) / (f.std(0) + 1e-9)).reshape(H, W, B), gt


def windows(cube, gt, maxw):
    H, W = gt.shape; out = []
    for y in range(0, H - 16 + 1, 8):
        for x in range(0, W - 16 + 1, 8):
            blk = gt[y:y + 16, x:x + 16].ravel()
            v, c = np.unique(blk, return_counts=True)
            if v[c.argmax()] != 0 and c.max() / blk.size >= 0.90:
                out.append((c.max() / blk.size, cube[y:y + 16, x:x + 16]))
    out.sort(key=lambda t: -t[0]); return [w for _, w in out[:maxw]]


def wjob(args):
    gid, win = args
    return k_hat_R(win, np.random.default_rng(35000 + gid))


def real():
    ip = load_hsi("data/Indian_pines_corrected.mat", "data/Indian_pines_gt.mat")
    sa = load_hsi("data/Salinas_corrected.mat", "data/Salinas_gt.mat")
    s2 = load_s2()
    hsi = windows(*ip, 50) + windows(*sa, 50); s2w = windows(*s2, 60)
    for name, wins in [("Hyperspectral", hsi), ("Sentinel-2", s2w)]:
        with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as ex:
            ks = list(ex.map(wjob, list(enumerate(wins))))
        ks = np.array(ks)
        print(f"[real specificity] {name} ({len(wins)} win): "
              f"k=1 rate {np.mean(ks==1):.2f}, mean k_hat {ks.mean():.2f}")
    rng = np.random.default_rng(99)
    print("[real full-scene K] ESS-Dip-R:")
    for nm, (cube, gt) in [("IndianPines", ip), ("Salinas", sa), ("Sentinel-2", s2)]:
        print(f"  {nm:12s} {k_hat_R(cube, rng, kcap=20)}")


if __name__ == "__main__":
    synthetic()
    real()
