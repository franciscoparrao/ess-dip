"""
Exp 29 (reviewer C2) — partition-quality metrics on the real scenes, beyond the
number of classes K. For each scene we produce per-pixel labels from ESS-Dip
(recursive tracking) and from the gap statistic and silhouette (k-means at the K
each selects, from Table tab:realfull), and report ARI and NMI against the
ground-truth land-cover map on labelled pixels. These are expected to be low:
no unsupervised criterion recovers a 16-class partition, and ESS-Dip's
conservatism (it under-detects) shows up directly here, consistent with its
positioning as a precision-oriented homogeneity test rather than a classifier.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import diptest
import methods as M

N_PCA, TAU, KCAP, ALPHA, MIN_NEFF = 10, 0.25, 20, 0.05, 12
# K selected by the standard criteria on each full scene (Table tab:realfull)
KSEL = {"IndianPines": {"gap": 3, "sil": 2},
        "Salinas":     {"gap": 9, "sil": 2},
        "Sentinel-2":  {"gap": 5, "sil": 2}}


def _key(d): return [k for k in d if not k.startswith("__")][0]


def load_hsi(imgf, gtf):
    md, mg = loadmat(imgf), loadmat(gtf)
    cube = md[_key(md)].astype(float); gt = mg[_key(mg)].astype(int)
    H, W, B = cube.shape
    Xs = StandardScaler().fit_transform(cube.reshape(-1, B))
    Xp = PCA(N_PCA, random_state=0).fit_transform(Xs)
    return Xp.reshape(H, W, N_PCA), gt


def load_s2():
    raw = np.load("data/s2_cube.npy"); gt = np.load("data/s2_labels.npy")
    H, W, B = raw.shape
    f = raw.reshape(-1, B)
    cube = ((f - f.mean(0)) / (f.std(0) + 1e-9)).reshape(H, W, B)
    return cube, gt


def essdip_labels(img, seed):
    p = img.shape[-1]
    work = M.detrend_poly(img) if M.poly_r2(img) > TAU else img
    h, w, _ = img.shape
    a = M.estimate_range_local(work, tile=min(24, h // 2, w // 2)) ** 2
    X = work.reshape(-1, p); rg = np.random.default_rng(seed)
    stack, leaves = [np.arange(len(X))], []
    while stack:
        if len(leaves) + len(stack) >= KCAP:
            leaves.extend(stack); break
        ids = stack.pop(); C = X[ids]; n_eff = len(C) / a
        if n_eff < MIN_NEFF:
            leaves.append(ids); continue
        lab = KMeans(2, n_init=5, random_state=int(rg.integers(1e9))).fit_predict(C)
        m0, m1 = lab == 0, lab == 1
        d = C[m1].mean(0) - C[m0].mean(0); d = d / (np.linalg.norm(d) + 1e-12)
        if M._ess_pval(diptest.dipstat(C @ d), n_eff, rg) < ALPHA \
           and m0.sum() >= a and m1.sum() >= a:
            stack.extend([ids[m0], ids[m1]])
        else:
            leaves.append(ids)
    lm = np.full(len(X), -1, int)
    for k, ids in enumerate(leaves):
        lm[ids] = k
    return lm.reshape(h, w), len(leaves)


def kmeans_labels(img, k):
    return KMeans(k, n_init=5, random_state=0).fit_predict(
        img.reshape(-1, img.shape[-1]))


def evaluate(name, cube, gt):
    flatgt = gt.ravel()
    mask = flatgt != 0 if name != "Sentinel-2" else np.ones(flatgt.shape, bool)
    ess, k_ess = essdip_labels(cube, 99)
    preds = {"ESS-Dip": (ess.ravel(), k_ess),
             "gap": (kmeans_labels(cube, KSEL[name]["gap"]), KSEL[name]["gap"]),
             "silhouette": (kmeans_labels(cube, KSEL[name]["sil"]), KSEL[name]["sil"])}
    rows = []
    for m, (lab, k) in preds.items():
        ari = adjusted_rand_score(flatgt[mask], lab[mask])
        nmi = normalized_mutual_info_score(flatgt[mask], lab[mask])
        rows.append(dict(scene=name, method=m, k=k,
                         ari=round(ari, 3), nmi=round(nmi, 3)))
    return rows


def main():
    os.makedirs("experiments/results", exist_ok=True)
    scenes = {}
    scenes["IndianPines"] = load_hsi("data/Indian_pines_corrected.mat",
                                     "data/Indian_pines_gt.mat")
    scenes["Salinas"] = load_hsi("data/Salinas_corrected.mat",
                                 "data/Salinas_gt.mat")
    scenes["Sentinel-2"] = load_s2()
    allrows = []
    print(f"  {'scene':12s} {'method':11s} {'K':>3s} {'ARI':>7s} {'NMI':>7s}")
    for name, (cube, gt) in scenes.items():
        for r in evaluate(name, cube, gt):
            allrows.append(r)
            print(f"  {r['scene']:12s} {r['method']:11s} {r['k']:>3d} "
                  f"{r['ari']:>7.3f} {r['nmi']:>7.3f}")
    with open("experiments/results/real_ari.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "method", "k", "ari", "nmi"])
        w.writeheader(); w.writerows(allrows)
    print("saved -> experiments/results/real_ari.csv")


if __name__ == "__main__":
    main()
