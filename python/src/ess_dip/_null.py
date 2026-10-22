"""Monte Carlo reference distribution of the dip at the effective sample size.

``uniform`` is the least-favourable unimodal law (default ESS-Dip, distribution-free);
``gaussian`` is the model's own null (the recall variant ESS-Dip-R, Section 3.4).
"""
from __future__ import annotations

import diptest
import numpy as np

M_MIN, M_MAX, N_SIM = 12, 2000, 300
_CACHE: dict[tuple[str, int], np.ndarray] = {}


def null_size(n_eff: float) -> int:
    """Round the effective sample size and clamp it to [12, 2000], as in the paper."""
    return max(M_MIN, min(int(round(n_eff)), M_MAX))


def p_value(dip: float, n_eff: float, rng: np.random.Generator, null: str = "uniform") -> float:
    """Share of simulated null dips at least as large as the observed one."""
    if null not in ("uniform", "gaussian"):
        raise ValueError("null must be 'uniform' or 'gaussian'")
    m = null_size(n_eff)
    key = (null, m)
    if key not in _CACHE:
        draw = rng.uniform if null == "uniform" else rng.standard_normal
        _CACHE[key] = np.sort([diptest.dipstat(draw(size=m)) for _ in range(N_SIM)])
    return float(np.mean(_CACHE[key] >= dip))
