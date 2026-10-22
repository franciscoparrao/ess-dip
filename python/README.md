# ess-dip

**How many classes does a spatial field support beyond its autocorrelation?**

Standard criteria for choosing the number of clusters (the gap statistic, silhouette,
Calinski–Harabasz, Davies–Bouldin) take no account of the dependence between
neighbouring pixels. On autocorrelated fields simulated with no classes at all, they
report more than one class every time. ESS-Dip tests instead for *multimodality in
excess of spatial autocorrelation*: it evaluates the Hartigan dip along a 2-means split
direction and calibrates it at the effective sample size given by a within-class
autocorrelation range. Applied recursively, it returns a conservative number of classes:
in the simulation study of the paper it never exceeded the true number.

Use it as a homogeneity guard before an unsupervised classification, as a stopping rule
for a divisive or region-growing procedure, or to audit the clusters of any other
algorithm.

## Install

```bash
pip install ess-dip
```

Requires Python ≥ 3.9, NumPy, scikit-learn and `diptest`.

## Use

```python
import numpy as np
from ess_dip import ESSDip, homogeneity_test

# image: array (H, W, bands); NaN marks pixels outside the region
model = ESSDip(random_state=0).fit(image)
model.n_classes_          # number of classes supported beyond autocorrelation
model.labels_             # (H, W) leaf partition, -1 outside the region
model.range_estimated_    # within-class 1/e range, in pixels
model.n_eff_              # effective sample size of the region

# one decision only: is this region a single class?
homogeneity_test(image, random_state=0)
```

Standardise the bands first, and reduce strongly collinear cubes with principal
components: the split direction is chosen in Euclidean feature space.

## Options that matter

| Option | Default | When to change it |
|---|---|---|
| `null` | `"uniform"` | `"gaussian"` gives the recall variant ESS-Dip-R: more classes at the same measured specificity, but it may split a skewed or heavy-tailed class. |
| `detrend` | `"auto"` | Removes a quadratic surface when it explains more than `tau` of the variance. Set `False` for coarse inputs such as patch embeddings, whose variation is smooth at the scene scale: there the rule can remove real boundaries. |
| `range_` | `None` | Pass the within-class 1/e range (pixels) when you know it, or when the package warns that many range tiles are censored. |
| `tile` | `24` | Tile side for the range estimate; it resolves ranges up to about half its size. |

## Reading the result

`n_classes_` is best read as a lower bound: the data support *at least* that many classes
beyond their autocorrelation. A result of 1 is not proof of homogeneity; it means that
the data do not contradict it at the effective sample size. The method is conservative by
design and loses power against weakly separated classes and against minorities below
roughly a twentieth of the region.

## Citation

If you use this package, please cite:

> Parra, F. (2026). Effective-sample-size calibration of a modality test for cluster
> homogeneity and the number of classes in spatially autocorrelated data.
> *Mathematical Geosciences*.

The repository <https://github.com/franciscoparrao/ess-dip> reproduces every result of
the paper.

## Licence

MIT.
