# SeqProject2026

Sequence analysis for IBL Neuropixels sessions, with a cache-first workflow for fast dashboards.

## What this does
- Computes per-neuron delay and coupling metrics from IBL task/passive/whisking data.
- Saves per-PID cache files in `data/dashboard_cache/*.pkl`.
- Uses those cache files to drive interactive dashboards.

Main flow: **run `03_calc_dashboard.py` first, then open dashboards**.

## Required packages
This repo includes an `environment.yml` with core analysis dependencies.  
For the full `03 calc -> dashboards` flow, make sure these are available:

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

## Setup
From the repo root:

```bash
conda env create -f environment.yml
conda activate Seq2026
pip install streamlit plotly plotly-resampler rastermap
```

## Run order
### 1) Compute cache (`03 calc`)
Edit options in `notebooks/03_calc_dashboard.py` (`COMPUTE_ALL`, `PIDS`, `SUBJECT`, `REGIONS`, `TAG`), then run:

```bash
python notebooks/03_calc_dashboard.py
```

This creates/updates:
- `data/dashboard_cache/<pid>.pkl`
- `data/processed/<pid>_delay_results_dashboard.csv`

### 2) Open dashboard(s)
Option A (full Streamlit dashboard):

```bash
streamlit run notebooks/04_dashboard.py
```

Option B (lightweight local web dashboard):

```bash
python web_dashboard/server.py
```

Then open: `http://127.0.0.1:8000`

## Repository layout
- `notebooks/03_calc_dashboard.py`: cache builder
- `notebooks/04_dashboard.py`: Streamlit dashboard
- `web_dashboard/server.py`: lightweight local dashboard server
- `utils/`: analysis/plotting/io helpers
- `data/`: cache, raw, and processed outputs
