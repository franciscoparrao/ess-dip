"""Within-class decorrelation range (Section 3.5 of the paper).

The range is the lag at which the autocorrelation function of the first principal
component first falls below 1/e. It is estimated on small tiles and summarised by the
median, so that it measures the within-class scale rather than the between-class mosaic.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _first_below(profile: np.ndarray) -> tuple[float, bool]:
    """First lag below 1/e, and whether the profile never got there (censored)."""
    w = np.where(profile < 1 / np.e)[0]
    return (float(w[0]), False) if w.size else (float(len(profile)), True)


def tile_range(img: np.ndarray) -> tuple[float, bool]:
    """1/e range of one complete tile, in pixels, and whether it is censored.

    Identical to ``estimate_range`` of the paper's code, including its floor of 2 pixels.
    """
    h, w, b = img.shape
    pc = PCA(1).fit_transform(img.reshape(-1, b)).reshape(h, w)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    ac = np.fft.ifft2(np.abs(np.fft.fft2(pc)) ** 2).real
    ac /= ac[0, 0]
    rx, cx = _first_below(ac[0, :max(2, w // 2)])
    ry, cy = _first_below(ac[:max(2, h // 2), 0])
    return max(2.0, 0.5 * (rx + ry)), (cx or cy)


def local_range(img: np.ndarray, valid: np.ndarray | None = None, tile: int = 24
                ) -> tuple[float, float, int]:
    """Median 1/e range over the complete tiles of a scene.

    Returns ``(range_px, censored_fraction, n_tiles)``. Only tiles whose pixels are all
    valid are used. If none fits, the whole scene is used as a single tile, which the
    paper's code also does.
    """
    h, w, _ = img.shape
    tile = int(min(tile, h // 2, w // 2))
    ranges, censored = [], []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            if valid is not None and not valid[y:y + tile, x:x + tile].all():
                continue
            r, c = tile_range(img[y:y + tile, x:x + tile])
            ranges.append(r)
            censored.append(c)
    if not ranges:
        if valid is not None and not valid.all():
            raise ValueError("No complete tile fits in the valid area; pass range_ explicitly.")
        r, c = tile_range(img)
        return r, float(c), 1
    return float(np.median(ranges)), float(np.mean(censored)), len(ranges)
