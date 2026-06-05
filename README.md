# ESS-Dip

**Effective-sample-size calibration of a modality test for selecting the number
of clusters in spatially autocorrelated data.**

Standard criteria for choosing the number of clusters *K* (the gap statistic,
silhouette, Calinski–Harabasz, Davies–Bouldin) assume independent observations.
On spatial data this fails: neighbouring pixels are autocorrelated, the
effective sample size is far below the nominal one, and a smooth large-scale
trend mimics discrete structure — so these criteria **over-detect** clusters.
ESS-Dip recasts *K*-selection as a test for **multimodality in excess of spatial
autocorrelation**: it evaluates the Hartigan dip along the locally optimal split
direction, calibrates it at the **effective sample size** derived from a
**within-class autocorrelation range** (local declustering), and removes a
low-order **trend** adaptively. A recursive partitioning returns *K*.

This repository reproduces every result in the accompanying paper (under review
at *Mathematical Geosciences*).

## Repository layout

```
experiments/
  methods.py                 # generators + all methods (ESS-Dip, ablations, baselines)
  08_benchmark.py            # synthetic benchmark -> results/bench.csv
  10_realdata.py             # Indian Pines + Salinas (hyperspectral)
  11_fetch_sentinel2.py      # fetch Sentinel-2 + ESA WorldCover (Planetary Computer)
  12_sentinel2_analysis.py   # Sentinel-2 (multispectral)
  13_local_range.py          # local-range validation
  results/                   # CSV outputs
manuscript/                  # LaTeX source, figures (R), refs
  figures/make_fig1.R        # Figure 1 (ggplot2)
docs/research-log.md         # development log
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# synthetic benchmark (Table 1, Figure 1 data)
python experiments/08_benchmark.py

# Figure 1 (requires R with ggplot2, dplyr, tidyr, patchwork, readr)
cd manuscript/figures && Rscript make_fig1.R
```

A convenience script that runs the full pipeline end to end is provided:

```bash
bash reproduce.sh
```

## Data

- **Indian Pines** and **Salinas** hyperspectral scenes — standard public
  benchmarks; `reproduce.sh` downloads the corrected cubes and ground truth.
- **Sentinel-2** Level-2A imagery — Copernicus programme, accessed through the
  Microsoft Planetary Computer (`11_fetch_sentinel2.py`; needs the optional
  dependencies in `requirements.txt`). Land-cover labels: ESA WorldCover.
- **Synthetic scenes** — generated deterministically by `methods.make_world`
  from fixed seeds; no download required.

## Method parameters

ESS-Dip is run with fixed defaults throughout (no per-scene tuning): level
`α = 0.05`, trend threshold `τ = 0.25`, tile side `t = 24`, minimum effective
size `ν = 12`, ensemble size `B = 5`. See `experiments/methods.py`
(`ess_dip_local`).

## Citing

If you use this code, please cite the paper and the software (see
`CITATION.cff`).

## License

MIT — see `LICENSE`.
