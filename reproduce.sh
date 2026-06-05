#!/usr/bin/env bash
# Reproduce the results of the ESS-Dip paper end to end.
# Usage: bash reproduce.sh [--with-sentinel2]
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[setup] creating virtual environment"
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

echo "[1/4] synthetic benchmark -> experiments/results/bench.csv"
$PY experiments/08_benchmark.py

echo "[2/4] downloading Indian Pines + Salinas"
mkdir -p data
dl() { [ -f "data/$1" ] || curl -fsSL -C - --retry 5 -o "data/$1" "$2"; }
dl Indian_pines_corrected.mat https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
dl Indian_pines_gt.mat        https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat
dl Salinas_corrected.mat      https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat
dl Salinas_gt.mat             https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat

echo "[3/4] hyperspectral application -> experiments/results/real_specificity.csv"
$PY experiments/10_realdata.py

if [ "${1:-}" = "--with-sentinel2" ]; then
  echo "[opt] Sentinel-2 fetch + analysis (needs optional deps)"
  $PY experiments/11_fetch_sentinel2.py
  $PY experiments/12_sentinel2_analysis.py
fi

echo "[4/4] Figure 1 (requires R + ggplot2)"
( cd manuscript/figures && Rscript make_fig1.R )

echo "done. Outputs in experiments/results/ and manuscript/figures/."
