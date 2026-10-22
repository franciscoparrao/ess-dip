"""ESS-Dip: how many classes does a spatial field support beyond its autocorrelation?

>>> from ess_dip import ESSDip, homogeneity_test
>>> model = ESSDip(random_state=0).fit(image)     # image: (H, W, bands), NaN = outside
>>> model.n_classes_, model.labels_
"""
from ._range import local_range, tile_range
from ._trend import detrend, poly_r2
from .core import ESSDip, HomogeneityResult, Node, homogeneity_test

__all__ = ["ESSDip", "homogeneity_test", "HomogeneityResult", "Node",
           "local_range", "tile_range", "detrend", "poly_r2"]
__version__ = "0.1.0"
