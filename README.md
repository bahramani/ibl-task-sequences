# SeqProject2026

Brain-wide sequence analysis for IBL Neuropixels sessions, with a cache-first pipeline for interactive dashboards.

## Overview
This repository computes neuron-level timing and coupling metrics from task, passive, and whisking-aligned data, then serves those results through dashboards for exploration.

Recommended workflow: **run `03_calc_dashboard.py` first, then launch dashboards**.

## Required packages
`environment.yml` provides most dependencies. For the full analysis + dashboard workflow, ensure these are installed:

- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `one-api`
- `brainbox`
- `iblatlas`
- `ibllib`
- `rastermap`
- `plotly`
- `plotly-resampler`
- `streamlit`

## Quick start
From repository root:

```bash
conda env create -f environment.yml
conda activate Seq2026
pip install streamlit plotly plotly-resampler rastermap
```

## Pipeline
### 1) Run 03 calc (build/update cache)
Set processing scope in `notebooks/03_calc_dashboard.py` (`COMPUTE_ALL`, `PIDS`, `SUBJECT`, `REGIONS`, `TAG`) and run:

```bash
python notebooks/03_calc_dashboard.py
```

Primary outputs:

- `data/dashboard_cache/<pid>.pkl`
- `data/processed/<pid>_delay_results_dashboard.csv`

### 2) Launch dashboards
Full Streamlit dashboard:

```bash
streamlit run notebooks/04_dashboard.py
```

Lightweight local web dashboard:

```bash
python web_dashboard/server.py
```

Open `http://127.0.0.1:8000` for the lightweight dashboard.

## Data and outputs
- Input data are fetched through ONE/IBL and cached under `data/raw`.
- Analysis caches for dashboard use are written to `data/dashboard_cache`.
- Processed tables are written to `data/processed`.
- Figures and exports are stored under `results`.

## Scientific note
- The coupling analysis in this project is based on spike-triggered population coupling with split-half reliability estimates across behavioral contexts (spontaneous, task, ITI).
- Coupling method reference (as requested): https://www.biorxiv.org/content/10.64898/2025.12.20.695676v2

## Project structure
- `notebooks/03_calc_dashboard.py`: main cache computation pipeline
- `notebooks/04_dashboard.py`: full Streamlit dashboard
- `web_dashboard/server.py`: lightweight HTTP dashboard
- `utils/`: IO, analysis, and plotting utilities
- `data/`: raw cache, processed outputs, dashboard cache
- `results/`: figures and derived outputs

## License
Released under the MIT License. See `LICENSE`.
