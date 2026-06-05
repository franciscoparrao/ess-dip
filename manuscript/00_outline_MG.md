# Manuscript outline — calibrated to *Mathematical Geosciences* (MG)

Target: **Mathematical Geosciences** (Springer / IAMG). IF 3.6, subscription/hybrid
(no page charges), Major Revision default, ~7-day triage. EIC Dimitrakopoulos.

## Calibration principles (MG-specific)

1. **Mathematical rigor FIRST, geo-application second.** MG's outline is explicit:
   "mathematical rigor primero, geo-application secundario." → Front-load a formal
   *Mathematical formulation* section (definitions, null model, ESS calibration,
   algorithm, a stated specificity property). The satellite-imagery application is the
   *real-life application* that demonstrates the method, not the headline.
2. **Speak geostatistics.** Use the vocabulary the board lives in: **variogram /
   autocorrelation range, declustering, effective sample size, stationarity, trend +
   residual decomposition, Gaussian random field null**. This reframes our "local
   range / ESS / detrend" components as members of the geostatistical canon.
3. **Citation strategy targets the board** (cite their lineage, honestly):
   - **Cressie (1993)** *Statistics for Spatial Data* — effective sample size. *(Cressie is on the MG board.)*
   - **Deutsch & Journel** (GSLIB), **Isaaks & Srivastava (1989)** — declustering. *(Deutsch is on the board.)*
   - **Matheron (1963)** — variogram / regionalized variables (IAMG origin lineage).
   - **Dutilleul et al. (1993)** — ESS-corrected inference under spatial autocorrelation.
   - **Hennig & Lin (2015)**, *Statistics and Computing* — closest prior art: cluster-number
     selection with an autocorrelation-aware null. **Build on it, do not compete.**
   - **Tibshirani, Walther & Hastie (2001)** — gap statistic (the criterion we correct).
   - **Hartigan & Hartigan (1985)** — dip test; **Kalogeratos & Likas (2012)** — dip-means.
   - **Goovaerts & Jacquez (2004)** — geostatistical spatial neutral models, over-detection.
   - **Wagner & Dray (2015)** — Moran spectral randomization (null-model machinery).
4. **Reproducibility is a positive signal at MG** ("code + datasets welcome as
   supplementary material"). Ship a public repo + Zenodo DOI; say so explicitly.
5. **Figures in R/ggplot2** (publication-quality, per house default), double-column Springer.

## Risks to neutralize (anticipating MG reviewers)

- **"Heuristic, not rigorous."** Our method is empirically validated. MG wants math first.
  → Add a formal null model + a *stated* false-split-control property (Proposition), even if
  the strong result is empirical. Justify the ESS relation and the dip calibration formally.
- **"This is a remote-sensing / CS application, not geomathematics."** → Keep the math/geostat
  core central; present RS as application. Frame as a *spatial-data* method (general), imagery
  as the instance.
- **"Incremental over Hennig & Lin."** → State the precise delta: continuous-field (raster)
  instantiation, within-class range via declustering, adaptive trend/residual separation,
  effective-sample-size calibration of a modality test. Their work is point-pattern/discrete.
- **Scope inflation.** Avoid "first". Claim the specific, defensible novelty.

---

## Title (options, method-first per MG)

1. **"Effective-sample-size calibration of a modality test for selecting the number of
   clusters in spatially autocorrelated data"** *(method-first; recommended)*
2. "Declustered modality testing for the number of classes in unsupervised classification
   of spatial fields"
3. "How many classes? A geostatistically calibrated test for cluster-number selection under
   spatial autocorrelation"

## Abstract (draft v0 — ~230 words, MG tone)

> Selecting the number of clusters *K* is a core problem in unsupervised classification.
> Internal validity criteria — the gap statistic, average silhouette, Calinski–Harabasz,
> Davies–Bouldin — assume independent observations. Spatial data violate this assumption:
> neighbouring locations are autocorrelated, so the effective sample size is far below the
> nominal count, and smooth large-scale trends induce apparent multimodality in feature
> space. We show that, as a consequence, standard criteria over-detect structure
> systematically: on autocorrelated fields with no discrete classes they report spurious
> clusters with probability close to one.
>
> We propose a cluster-number selection procedure that treats clustering as multimodality of
> the feature distribution *in excess of* spatial autocorrelation. The procedure (i) tests the
> Hartigan dip statistic along the locally optimal split direction, (ii) calibrates its null
> distribution against the **effective sample size** derived from the **within-class
> autocorrelation range**, estimated by a declustering-style local estimator, and (iii) removes
> a low-order spatial trend adaptively to separate continuous variation from discrete classes.
> A recursive partitioning yields *K*.
>
> On a controlled simulation study with known ground truth the method controls the
> false-positive rate at ~0 while remaining competitive in exact-*K* recovery, and attains the
> best balanced score among nine criteria. On three real scenes — the Indian Pines and Salinas
> hyperspectral images and a Sentinel-2 multispectral scene with ESA WorldCover labels — it
> correctly identifies single land-cover regions as homogeneous where standard criteria
> fragment them. Code and data are released.

---

## Section skeleton

### 1. Introduction
- The number-of-clusters problem in unsupervised classification of spatial/imagery data.
- Standard criteria assume i.i.d.; spatial autocorrelation (Tobler) breaks this → effective
  sample size collapse + trend-induced apparent multimodality.
- Empirical hook: standard criteria over-detect (forward-reference the 100% false-positive result).
- The gap in the literature: autocorrelation-aware *K*-selection exists for discrete/point data
  (Hennig & Lin 2015) but **not** as a continuous-field method for gridded data, and **no**
  method separates smooth trend from discrete class structure in *K*-selection.
- Contributions (bulleted): (1) formal modality-based, ESS-calibrated test; (2) within-class
  range via local declustering; (3) adaptive trend/residual separation; (4) controlled
  benchmark + 3 real scenes; (5) open code/data.

### 2. Background and related work
- Internal validity indices and the gap statistic (Tibshirani 2001); their i.i.d. assumption.
- Cluster-number selection under dependence: Hennig & Lin (2015) — the conceptual basis.
- Geostatistical toolkit: variogram & range (Matheron; Cressie 1993), **effective sample
  size** (Cressie 1993; Dutilleul 1993), **declustering** (Isaaks & Srivastava; Deutsch).
- Spatial neutral / null models: Goovaerts & Jacquez (2004); Moran spectral randomization
  (Wagner & Dray 2015); sequential Gaussian simulation.
- Modality testing: Hartigan dip (1985); dip-means (Kalogeratos & Likas 2012).
- Delineation from **spatially-constrained clustering / regionalization** (SKATER, ClustGeo,
  max-p): those *impose* contiguity; we *correct* the validity criterion. Different problem.

### 3. Mathematical formulation  *(the MG core — math first)*
- 3.1 Setup & notation: lattice D ⊂ Z², feature field z(s) ∈ R^p, partition, target K.
- 3.2 Clustering as excess multimodality; the unimodal Gaussian-random-field null.
- 3.3 The dip statistic along the 2-means split direction (rotation-targeted projection).
- 3.4 **Effective-sample-size calibration**: n_eff = |A| / a, decorrelation area a = ℓ²,
  null p-value P(D_{n_eff} ≥ D_obs). Formal statement + justification (Cressie ESS).
- 3.5 **Local within-class range** ℓ: median tile-wise autocorrelation range (declustering
  rationale — tiles fall within a class, so ℓ estimates within-class, not between-class, scale).
- 3.6 **Adaptive trend removal**: low-order polynomial; remove iff R²_poly > τ (trend/residual
  decomposition; universal-kriging mindset).
- 3.7 Recursive partitioning algorithm (pseudocode box) + ensemble aggregation → K.
- 3.8 **Proposition (false-split control)**: under a unimodal GRF null with stationary
  autocorrelation, the calibrated test controls the false-split rate at level α as n_eff→∞.
  (State assumptions; empirical confirmation in §4.)
- 3.9 Computational complexity / scalability (FFT range estimate, mini-batch k-means).

### 4. Simulation study  *(controlled ground truth — MG values this)*
- 4.1 Design: null / trend / structured / mixed worlds; parameters (K, separation, range, SNR).
- 4.2 Methods compared (9): gap, silhouette, CH, DB, and our variants.
- 4.3 Metrics: specificity (truth K=1), exact-K accuracy, MAE, balanced score.
- 4.4 Results: main table + figures (specificity vs accuracy; accuracy vs K).
- 4.5 Ablations: local vs global range; adaptive detrend; ensemble.

### 5. Application: unsupervised land-cover classification
- 5.1 Data: Indian Pines, Salinas (hyperspectral); Sentinel-2 + ESA WorldCover (multispectral).
- 5.2 Real-data specificity on single-class windows (the headline real result).
- 5.3 Full-scene behaviour; honest limits (many overlapping classes; class imbalance).

### 6. Discussion
- What the method controls (false positives) at what cost (power); operating envelope.
- Relation to geostatistical simulation nulls; extension to spatio-temporal / multiscale.
- Limitations: extreme K, strong imbalance, single global α.

### 7. Conclusions

### Code and data availability
- GitHub repo + Zenodo DOI; synthetic generators + real-data pipeline + all methods; seeds.

---

## Immediate writing order (proposed)
1. **§3 Mathematical formulation** first — it anchors the MG framing; everything else references it.
2. **§4 Simulation** (results already in hand: `experiments/08_benchmark.py`).
3. **§5 Application** (results in `experiments/10`, `12`).
4. **§2 Background** + **§1 Introduction** (write intro last, once the contribution is crisp).
5. Abstract + title finalize.
6. Figures in R from `experiments/results/*.csv`.
