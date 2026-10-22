"""Adaptive trend–residual separation (Section 3.6 of the paper)."""
from __future__ import annotations

import numpy as np


def _design(h: int, w: int, degree: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    terms = [np.ones_like(xx, float)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((xx ** (d - i)) * (yy ** i))
    return np.stack([t.ravel() for t in terms], 1)


def poly_r2(img: np.ndarray, valid: np.ndarray | None = None, degree: int = 2) -> float:
    """Mean fraction of each band's variance explained by a polynomial surface."""
    h, w, nb = img.shape
    a = _design(h, w, degree)
    keep = np.ones(h * w, bool) if valid is None else valid.ravel()
    r2 = []
    for b in range(nb):
        y = img[..., b].ravel()[keep]
        coef, *_ = np.linalg.lstsq(a[keep], y, rcond=None)
        r2.append(1 - (y - a[keep] @ coef).var() / (y.var() + 1e-12))
    return float(np.mean(r2))


def detrend(img: np.ndarray, valid: np.ndarray | None = None, degree: int = 2) -> np.ndarray:
    """Subtract a least-squares polynomial surface from each band (fitted on valid pixels)."""
    h, w, nb = img.shape
    a = _design(h, w, degree)
    keep = np.ones(h * w, bool) if valid is None else valid.ravel()
    out = np.array(img, dtype=float, copy=True)
    for b in range(nb):
        y = img[..., b].ravel()
        coef, *_ = np.linalg.lstsq(a[keep], y[keep], rcond=None)
        out[..., b] = (y - a @ coef).reshape(h, w)
    return out
