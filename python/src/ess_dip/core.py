"""ESS-Dip: effective-sample-size calibrated dip test for the number of classes.

Parra, F. (2026). Effective-sample-size calibration of a modality test for cluster
homogeneity and the number of classes in spatially autocorrelated data.
Mathematical Geosciences.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import diptest
import numpy as np
from sklearn.cluster import KMeans

from ._null import p_value
from ._range import local_range
from ._trend import detrend, poly_r2

# Above half, the median of the tile ranges sits at the tile's ceiling (tile // 2): the
# estimate is then only a lower bound. In the paper's own simulations up to ~75 % of the
# tiles were censored and specificity still held, because the uniform null absorbs a range
# underestimated by up to a factor of four (Table 3); the warning informs, it does not alarm.
CENSORED_WARNING = 0.50


def _as_cube(X) -> tuple[np.ndarray, np.ndarray | None]:
    """Accept (H, W) or (H, W, p) arrays; NaN marks pixels outside the region."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3:
        raise ValueError("X must have shape (H, W) or (H, W, p)")
    valid = np.isfinite(X).all(axis=-1)
    if valid.all():
        return X, None
    if not valid.any():
        raise ValueError("X has no valid pixel")
    X = np.where(valid[..., None], X, 0.0)
    return X, valid


@dataclass
class Node:
    """One homogeneity decision of the recursion."""
    n: int
    n_eff: float
    dip: float | None
    p_value: float | None
    split: bool
    reason: str


@dataclass
class HomogeneityResult:
    """Outcome of a single homogeneity test (the root node of ESS-Dip)."""
    homogeneous: bool
    dip: float
    p_value: float
    n_eff: float
    range_: float
    detrended: bool
    note: str = field(default="")

    def __repr__(self) -> str:
        verdict = "no evidence of more than one class" if self.homogeneous else "multimodal beyond autocorrelation"
        return (f"HomogeneityResult({verdict}; dip={self.dip:.4f}, p={self.p_value:.3f}, "
                f"n_eff={self.n_eff:.1f}, range={self.range_:.2f} px)")


class ESSDip:
    """Estimate the number of classes that a spatial field supports beyond its autocorrelation.

    The defaults reproduce the published method: adaptive detrending (tau = 0.25), a
    within-class range from 24-pixel tiles, a uniform (least-favourable) dip null,
    alpha = 0.05 and an ensemble of five runs whose median is reported.

    Parameters
    ----------
    alpha : float
        Level of each split test.
    null : {"uniform", "gaussian"}
        Reference law of the dip. ``"uniform"`` is the conservative, distribution-free
        ESS-Dip; ``"gaussian"`` is the recall-oriented ESS-Dip-R.
    detrend : {"auto", True, False}
        ``"auto"`` removes a quadratic surface when it explains more than ``tau`` of the
        variance. Turn it off for coarse inputs (e.g. patch embeddings), where the
        variation is smooth at the scale of the scene and the rule can remove real
        boundaries.
    tau : float
        Threshold of the adaptive detrending rule.
    range_ : float or None
        Within-class 1/e range in pixels. ``None`` estimates it from tiles; pass a value
        when the range is longer than about half the tile (the estimate is then censored).
    tile : int
        Tile side in pixels for the range estimate.
    min_neff : float
        Smallest effective sample size at which a node is tested.
    max_classes : int
        Cap on the number of classes.
    n_runs : int
        Ensemble size; the median number of classes is reported.
    random_state : int or None

    Attributes
    ----------
    n_classes_ : int
        Estimated number of classes, a conservative estimate: in the paper's simulation it
        never exceeded the true number.
    labels_ : ndarray of shape (H, W)
        Leaf partition of one run that returned ``n_classes_`` (-1 outside the region).
    range_estimated_ : float
        Range used, in pixels.
    range_censored_fraction_ : float
        Share of tiles whose autocorrelation never fell below 1/e.
    area_ : float
        Decorrelation area, ``range ** 2`` pixels.
    n_eff_ : float
        Effective sample size of the whole region.
    detrended_ : bool
        Whether the trend was removed.
    runs_ : list of int
        Number of classes in each ensemble run.
    nodes_ : list of Node
        Decisions of the reported run.
    """

    def __init__(self, alpha=0.05, null="uniform", detrend="auto", tau=0.25, range_=None,
                 tile=24, min_neff=12, max_classes=8, n_runs=5, random_state=None):
        self.alpha = alpha
        self.null = null
        self.detrend = detrend
        self.tau = tau
        self.range_ = range_
        self.tile = tile
        self.min_neff = min_neff
        self.max_classes = max_classes
        self.n_runs = n_runs
        self.random_state = random_state

    # ------------------------------------------------------------------ preparation
    def _prepare(self, X):
        cube, valid = _as_cube(X)
        if self.detrend == "auto":
            do = poly_r2(cube, valid) > self.tau
        else:
            do = bool(self.detrend)
        work = detrend(cube, valid) if do else cube
        if self.range_ is None:
            r, cens, _ = local_range(work, valid, self.tile)
            if cens > CENSORED_WARNING:
                warnings.warn(
                    f"{100 * cens:.0f}% of the range tiles are censored, so the estimated range "
                    f"({r:g} px) is only a lower bound and the effective sample size may be "
                    "overstated. The uniform null tolerates a moderate underestimate; if the "
                    "range may be several times longer, use a larger tile or pass range_.",
                    stacklevel=3)
        else:
            r, cens = float(self.range_), float("nan")
        return work, valid, do, r, cens

    # ------------------------------------------------------------------ recursion
    def _one_run(self, points, area, rng):
        stack = [np.arange(len(points))]
        leaves, nodes = [], []
        while stack:
            if len(leaves) + len(stack) >= self.max_classes:
                leaves.extend(stack)
                nodes.append(Node(0, 0.0, None, None, False, "max_classes reached"))
                break
            idx = stack.pop()
            n_eff = len(idx) / area
            if n_eff < self.min_neff:
                leaves.append(idx)
                nodes.append(Node(len(idx), n_eff, None, None, False, "n_eff below min_neff"))
                continue
            c = points[idx]
            lab = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(c)
            m0, m1 = lab == 0, lab == 1
            d = c[m1].mean(0) - c[m0].mean(0)
            d = d / (np.linalg.norm(d) + 1e-12)
            dip = float(diptest.dipstat(c @ d))
            p = p_value(dip, n_eff, rng, self.null)
            if p < self.alpha and m0.sum() >= area and m1.sum() >= area:
                stack.extend([idx[m0], idx[m1]])
                nodes.append(Node(len(idx), n_eff, dip, p, True, "multimodal"))
            else:
                leaves.append(idx)
                why = "not significant" if p >= self.alpha else "child smaller than one decorrelation patch"
                nodes.append(Node(len(idx), n_eff, dip, p, False, why))
        return leaves, nodes

    def fit(self, X, y=None):
        """Estimate the number of classes of X, an array (H, W) or (H, W, p); NaN = outside."""
        rng = np.random.default_rng(self.random_state)
        work, valid, did, r, cens = self._prepare(X)
        h, w, _ = work.shape
        flat_valid = np.ones(h * w, bool) if valid is None else valid.ravel()
        points = work.reshape(-1, work.shape[-1])[flat_valid]
        area = r * r
        runs = [self._one_run(points, area, rng) for _ in range(self.n_runs)]
        ks = [len(leaves) for leaves, _ in runs]
        k = int(np.median(ks))
        leaves, nodes = next((run for run in runs if len(run[0]) == k), runs[0])
        lab = np.full(len(points), -1)
        for j, idx in enumerate(leaves):
            lab[idx] = j
        full = np.full(h * w, -1)
        full[flat_valid] = lab
        self.n_classes_ = k
        self.labels_ = full.reshape(h, w)
        self.range_estimated_ = r
        self.range_censored_fraction_ = cens
        self.area_ = area
        self.n_eff_ = len(points) / area
        self.detrended_ = did
        self.runs_ = ks
        self.nodes_ = nodes
        return self

    def fit_predict(self, X, y=None):
        """Fit and return the leaf partition."""
        return self.fit(X).labels_


def homogeneity_test(X, alpha=0.05, null="uniform", detrend="auto", tau=0.25, range_=None,
                     tile=24, random_state=None) -> HomogeneityResult:
    """Test whether a region holds multimodality beyond its spatial autocorrelation.

    A single decision at the root of ESS-Dip. Failing to reject is not evidence of
    homogeneity: it only means that the data do not contradict it at the effective
    sample size.
    """
    est = ESSDip(alpha=alpha, null=null, detrend=detrend, tau=tau, range_=range_, tile=tile,
                 random_state=random_state)
    rng = np.random.default_rng(random_state)
    work, valid, did, r, _ = est._prepare(X)
    h, w, _ = work.shape
    keep = np.ones(h * w, bool) if valid is None else valid.ravel()
    c = work.reshape(-1, work.shape[-1])[keep]
    n_eff = len(c) / (r * r)
    note = ""
    if n_eff < 12:
        note = "n_eff below 12: too little independent information for a reliable test."
    lab = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(c)
    d = c[lab == 1].mean(0) - c[lab == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    dip = float(diptest.dipstat(c @ d))
    p = p_value(dip, n_eff, rng, null)
    return HomogeneityResult(homogeneous=p >= alpha, dip=dip, p_value=p, n_eff=n_eff,
                             range_=r, detrended=did, note=note)
