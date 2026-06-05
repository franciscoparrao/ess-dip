"""
Geostatistical machinery for the ESS-Dip method (reviewer items R1, R2).

R1  proper variogram modelling + effective sample size from the *integral
    range* (Cressie 1993), replacing the ad-hoc n_eff = n / ell^2;
R2  a Gaussian-simulation null (spectral GRF matching the fitted variogram),
    the geostatistical neutral model used in place of / alongside the
    effective-sample-size shortcut.

Empirical variograms are computed radially from the FFT autocovariance, so the
whole pipeline stays near-linear in the number of pixels.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.optimize import curve_fit


# ---------------------------------------------------------------- variogram
def radial_variogram(field, max_lag=None):
    """Radially averaged standardised semivariogram gamma(h) and correlation
    rho(h) of a 2-D field, h = 0..max_lag (pixels)."""
    H, W = field.shape
    f = (field - field.mean()) / (field.std() + 1e-12)
    ac = np.fft.fftshift(ifft2(np.abs(fft2(f)) ** 2).real) / f.size
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - cy, xx - cx)
    L = max_lag or min(H, W) // 3
    rho = np.array([ac[(r >= k - 0.5) & (r < k + 0.5)].mean() for k in range(L + 1)])
    rho = rho / rho[0]
    return np.arange(L + 1), 1.0 - rho, rho


def _exp(h, c0, c, a):   return c0 + c * (1 - np.exp(-h / a))
def _gau(h, c0, c, a):   return c0 + c * (1 - np.exp(-(h / a) ** 2))
def _sph(h, c0, c, a):
    t = np.minimum(h / a, 1.0)
    return c0 + c * (1.5 * t - 0.5 * t ** 3)

# decorrelation-area constant kappa: integral range = kappa * a^2 * (c/(c0+c))
#   integral range = 2*pi * integral_0^inf h * rho_struct(h) dh
_MODELS = {
    "exponential": (_exp, 2 * np.pi),          # int h e^{-h/a} dh = a^2
    "gaussian":    (_gau, np.pi),              # int h e^{-(h/a)^2} dh = a^2/2
    "spherical":   (_sph, 0.2 * np.pi),        # int_0^a h (1-1.5t+0.5t^3) = 0.1 a^2
}


def fit_variogram(field, max_lag=None):
    """Fit exponential/gaussian/spherical models; return the best by SSE as
    dict(model, nugget c0, partial sill c, range a, decorrelation_area)."""
    h, g, _ = radial_variogram(field, max_lag)
    best = None
    for name, (fn, kappa) in _MODELS.items():
        try:
            p0 = [0.1, max(g[-1], 0.5), max(2.0, len(h) / 3)]
            bounds = ([0, 1e-3, 1.0], [1.0, 2.0, 3.0 * len(h)])
            popt, _ = curve_fit(fn, h, g, p0=p0, bounds=bounds, maxfev=8000)
            sse = float(np.sum((fn(h, *popt) - g) ** 2))
            c0, c, a = popt
            frac = c / (c0 + c + 1e-12)               # structured fraction
            area = max(1.0, kappa * a * a * frac)     # integral range (pixels^2)
            cand = dict(model=name, c0=c0, c=c, a=a, sse=sse,
                        decorrelation_area=area, kappa=kappa, frac=frac)
            if best is None or sse < best["sse"]:
                best = cand
        except Exception:
            continue
    if best is None:                                   # fallback: 1/e range
        below = np.where((1 - g) < 1 / np.e)[0]
        a = float(below[0]) if below.size else len(h) / 3
        best = dict(model="fallback", c0=0.0, c=1.0, a=a, sse=np.inf,
                    decorrelation_area=max(1.0, a * a), kappa=1.0, frac=1.0)
    return best


# ---------------------------------------------- effective sample size (R1)
def decorrelation_area_local(img, tile=24):
    """Median integral-range decorrelation area over tiles (within-class
    scale). Operates on the first principal component of the cube."""
    from sklearn.decomposition import PCA
    H, W, B = img.shape
    pc = PCA(1).fit_transform(img.reshape(-1, B)).reshape(H, W)
    areas = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            areas.append(fit_variogram(pc[y:y + tile, x:x + tile],
                                       max_lag=tile // 2)["decorrelation_area"])
    return float(np.median(areas)) if areas else fit_variogram(pc)["decorrelation_area"]


# ---------------------------------------------- Gaussian simulation null (R2)
def grf_field(shape, model, rng):
    """Spectral (FFT) unconditional Gaussian simulation on a grid with the
    covariance C(h) = c0*delta(h) + c*corr_model(h) of a fitted variogram.
    Returns a standardised field (a geostatistical neutral-model realisation)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(np.minimum(yy, H - yy), np.minimum(xx, W - xx))   # wrap distance
    c0, c, a, name = model["c0"], model["c"], model["a"], model["model"]
    if name == "gaussian":
        corr = np.exp(-(r / a) ** 2)
    elif name == "spherical":
        t = np.minimum(r / a, 1.0); corr = 1.5 * t - 0.5 * t ** 3; corr = 1 - corr
    else:                                                          # exponential / fallback
        corr = np.exp(-r / a)
    cov = c * corr
    cov[0, 0] += c0                                                # nugget at lag 0
    S = np.real(fft2(cov))
    S = np.clip(S, 0, None)                                        # circulant embedding
    field = np.real(ifft2(np.sqrt(S) * fft2(rng.standard_normal((H, W)))))
    return (field - field.mean()) / (field.std() + 1e-12)


def sgs_dip_null(y_proj, model, rng, n_null=120):
    """Null distribution of the dip for the projected field y_proj under the
    fitted variogram, via spectral Gaussian simulation on its bounding grid."""
    import diptest
    H, W = y_proj.shape
    dips = np.empty(n_null)
    for i in range(n_null):
        dips[i] = diptest.dipstat(grf_field((H, W), model, rng).ravel())
    return np.sort(dips)


def _box(mask, yvals):
    """Place standardised node values into the node's bounding box (zero-fill
    outside); return the box field and the in-box node sub-mask."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    box = np.zeros((y1 - y0, x1 - x0))
    box[sub] = (yvals - yvals.mean()) / (yvals.std() + 1e-12)
    return box, sub


def sgs_dip(img, rng, alpha=0.05, n_null=60, min_n=50, kcap=8, tau=0.25, n_runs=3):
    """Recursive K-estimator calibrated by the EXACT Gaussian-simulation null:
    at each node the dip along the 2-means split direction is compared against
    spectral GRF realisations matching the node-projection variogram on the
    node's spatial support. Adaptive trend removal as in ESS-Dip; median over
    n_runs ensemble members."""
    import diptest, importlib
    from sklearn.cluster import KMeans
    M = importlib.import_module("methods")   # reuse poly_r2 / detrend_poly
    work = M.detrend_poly(img) if M.poly_r2(img) > tau else img
    H, W, p = work.shape
    flat = work.reshape(-1, p)

    def one_run(seed):
        r = np.random.default_rng(seed)
        stack = [np.ones((H, W), bool)]
        leaves = 0
        while stack:
            if leaves + len(stack) >= kcap:
                return kcap
            m = stack.pop()
            idx = np.where(m.ravel())[0]
            pix = flat[idx]
            if len(pix) < min_n:
                leaves += 1; continue
            lab = KMeans(2, n_init=5,
                         random_state=int(r.integers(1e9))).fit_predict(pix)
            if lab.sum() < min_n or (1 - lab).sum() < min_n:
                leaves += 1; continue
            d = pix[lab == 1].mean(0) - pix[lab == 0].mean(0)
            d = d / (np.linalg.norm(d) + 1e-12)
            y = pix @ d
            D = diptest.dipstat(y)
            box, sub = _box(m, y)
            model = fit_variogram(box, max_lag=max(2, min(box.shape) // 2))
            null = np.array([diptest.dipstat(grf_field(box.shape, model, r)[sub])
                             for _ in range(n_null)])
            if np.mean(null >= D) < alpha:
                m0 = np.zeros(H * W, bool); m1 = np.zeros(H * W, bool)
                m0[idx[lab == 0]] = True; m1[idx[lab == 1]] = True
                stack.extend([m0.reshape(H, W), m1.reshape(H, W)])
            else:
                leaves += 1
        return leaves

    return int(np.median([one_run(int(rng.integers(1e9))) for _ in range(n_runs)]))
