# %% Imports
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.express as px

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

import utils.plotting_plotly as plotting_utils

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _set_plotly_renderer(preferred=None):
    if preferred:
        pio.renderers.default = preferred
        return
    if _in_notebook():
        try:
            import nbformat  # noqa: F401

            pio.renderers.default = "notebook_connected"
            return
        except Exception:
            pass
    pio.renderers.default = "browser"


def _is_dark_template(fig):
    try:
        template_name = None
        if hasattr(fig, "layout") and hasattr(fig.layout, "template"):
            template_name = getattr(fig.layout.template, "name", None)
        if not template_name:
            template_name = pio.templates.default
        if not template_name:
            template_name = getattr(plotting_utils, "DEFAULT_TEMPLATE", None)
        return "dark" in str(template_name).lower()
    except Exception:
        return False


def show_fig(fig, renderer=None):
    # Make HTML background match dark mode so the browser tab isn't white.
    if _is_dark_template(fig):
        fig.update_layout(paper_bgcolor="#0f0f10", plot_bgcolor="#0f0f10")
    if renderer:
        pio.renderers.default = renderer
    try:
        fig.show()
    except ValueError as exc:
        if "nbformat" in str(exc).lower():
            pio.renderers.default = "browser"
            fig.show()
        else:
            raise


PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"
_set_plotly_renderer(PLOTLY_RENDERER)


# %% Cache helpers
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"


def list_pids(cache_dir=CACHE_DIR):
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.pkl")])


def load_cache(pid, cache_dir=CACHE_DIR):
    path = cache_dir / f"{pid}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_label_array(clusters):
    if clusters is None:
        return None
    metrics = getattr(clusters, "metrics", None)
    if isinstance(metrics, pd.DataFrame) and "label" in metrics.columns:
        return np.asarray(metrics["label"])
    if hasattr(clusters, "label"):
        return np.asarray(clusters.label)
    if isinstance(clusters, dict) and "label" in clusters:
        return np.asarray(clusters.get("label"))
    return None


# %% Vector building helpers
def _guess_region_col(df):
    for col in ["region", "acronym", "area", "atlas_region"]:
        if col in df.columns:
            return col
    return None


def _available_regions(cache, vector_specs):
    regions = set()
    for spec in vector_specs.values():
        df = cache.get(spec["df"])
        if isinstance(df, pd.DataFrame):
            col = _guess_region_col(df)
            if col is None:
                continue
            regions.update(df[col].dropna().astype(str).tolist())
    return sorted([r for r in regions if r not in ("root", "void")])


def _good_cluster_ids(cache):
    labels = _get_label_array(cache.get("clusters"))
    cluster_ids_local = cache.get("cluster_ids")
    if labels is None or cluster_ids_local is None:
        return None
    return set(np.asarray(cluster_ids_local)[labels == 1].tolist())


def _vector_table_from_cache(cache, spec, region=None, only_good=False, key="cluster_id"):
    df = cache.get(spec["df"])
    if df is None or spec["col"] not in df.columns or key not in df.columns:
        return None

    df = df[[key, spec["col"]]].copy()

    if region is not None:
        region_col = _guess_region_col(cache.get(spec["df"]))
        if region_col and region_col in cache.get(spec["df"]).columns:
            mask = cache.get(spec["df"])[region_col].astype(str).str.startswith(str(region))
            df = cache.get(spec["df"]).loc[mask, [key, spec["col"]]].copy()
        else:
            print(f"Warning: region filter requested, but no region column in {spec['df']}.")

    if only_good:
        good_ids = _good_cluster_ids(cache)
        if good_ids is not None:
            df = df[df[key].isin(good_ids)]

    df = df.dropna(subset=[spec["col"]])
    df = df.groupby(key, as_index=False)[spec["col"]].mean()
    return df


def _align_vectors(vector_tables, key="cluster_id"):
    aligned = None
    for name, table in vector_tables.items():
        if table is None:
            return None, None
        col = table.columns[1]
        table = table.rename(columns={col: name})
        aligned = table if aligned is None else aligned.merge(table, on=key, how="inner")
    if aligned is None or aligned.empty:
        return None, None
    aligned = aligned.dropna(subset=list(vector_tables.keys()))
    vectors = {name: aligned[name].to_numpy(dtype=float) for name in vector_tables.keys()}
    return aligned, vectors


def _pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _fisher_z(r):
    if not np.isfinite(r):
        return np.nan
    r = np.clip(r, -0.999999, 0.999999)
    return float(np.arctanh(r))


def compute_similarity_from_vectors(a1, a2, b1, b2):
    # STEP 2: within-category reliabilities
    r_xx = _pearsonr(a1, a2)
    r_yy = _pearsonr(b1, b2)
    if np.isfinite(r_xx) and np.isfinite(r_yy) and r_xx > 0 and r_yy > 0:
        total_reliability = float(np.sqrt(r_xx * r_yy))
    else:
        total_reliability = np.nan

    # STEP 3: cross-validated raw correlation
    r_cross_1 = _pearsonr(a1, b2)
    r_cross_2 = _pearsonr(a2, b1)
    if np.isfinite(r_cross_1) and np.isfinite(r_cross_2):
        z1 = _fisher_z(r_cross_1)
        z2 = _fisher_z(r_cross_2)
        raw_corr = float(np.tanh(np.nanmean([z1, z2])))
    else:
        raw_corr = np.nan

    # STEP 4: noise-corrected similarity
    if np.isfinite(total_reliability) and total_reliability > 0:
        similarity = raw_corr / total_reliability
    else:
        similarity = np.nan

    return {
        "r_xx": r_xx,
        "r_yy": r_yy,
        "total_reliability": total_reliability,
        "r_cross_1": r_cross_1,
        "r_cross_2": r_cross_2,
        "raw_correlation": raw_corr,
        "similarity": similarity,
    }


def _vectors_for_pid_region(cache, vector_specs, region=None, only_good=False):
    vector_tables = {}
    for name, spec in vector_specs.items():
        vector_tables[name] = _vector_table_from_cache(
            cache, spec, region=region, only_good=only_good
        )
    aligned, vectors = _align_vectors(vector_tables)
    if aligned is None:
        return None
    return vectors


# %% User-editable settings
ANALYSIS_RUN = True
ANALYSIS_LABEL = "spont_delay_vs_stim_delay"
ANALYSIS_EVENT = "stimOn_times"  # used in df_reliability column names

# Which PIDs to include
ANALYSIS_PIDS = list_pids()  # e.g., ["pid1", "pid2"]

# Region filtering / grouping
ANALYSIS_REGION = None  # set a single region (e.g. "VISp") or keep None
ANALYSIS_MIN_N = 5  # min neurons required per region after combining PIDs
ANALYSIS_ONLY_GOOD = True  # True -> use only good units (label==1)

# Plot theme toggle
use_dark_theme = True  # True -> plotly_dark
if use_dark_theme:
    plotting_utils.DEFAULT_TEMPLATE = "plotly_dark"
    pio.templates.default = "plotly_dark"
else:
    plotting_utils.DEFAULT_TEMPLATE = "plotly_white"
    pio.templates.default = "plotly_white"

# Vector definitions (edit here to swap measures)
ANALYSIS_VECTOR_SPECS = {
    # Spontaneous delays (first/second half)
    "A1": {"df": "df_coupling", "col": "coupling_delay_ms_h1"},
    "A2": {"df": "df_coupling", "col": "coupling_delay_ms_h2"},
    # Evoked delays (event-specific split-halves)
    "B1": {"df": "df_reliability", "col": f"delay_h1_{ANALYSIS_EVENT}"},
    "B2": {"df": "df_reliability", "col": f"delay_h2_{ANALYSIS_EVENT}"},
}

# Example swap (strengths instead of delays):
# ANALYSIS_VECTOR_SPECS = {
#     "A1": {"df": "df_coupling", "col": "coupling_strength_h1"},
#     "A2": {"df": "df_coupling", "col": "coupling_strength_h2"},
#     "B1": {"df": "df_coupling_task", "col": "coupling_strength_h1"},
#     "B2": {"df": "df_coupling_task", "col": "coupling_strength_h2"},
# }


# %% Run region-combined analysis
if ANALYSIS_RUN:
    if ANALYSIS_REGION is not None:
        region_list = [ANALYSIS_REGION]
    else:
        # Union of regions across all PIDs and vector DataFrames
        region_set = set()
        for pid in ANALYSIS_PIDS:
            cache = load_cache(pid)
            region_set.update(_available_regions(cache, ANALYSIS_VECTOR_SPECS))
        region_list = sorted(region_set)

    results = []
    for region in tqdm(region_list, desc="Regions"):
        combined = {name: [] for name in ANALYSIS_VECTOR_SPECS.keys()}
        for pid in tqdm(ANALYSIS_PIDS, desc="PIDs", leave=False):
            cache = load_cache(pid)
            vectors = _vectors_for_pid_region(
                cache, ANALYSIS_VECTOR_SPECS, region=region, only_good=ANALYSIS_ONLY_GOOD
            )
            if vectors is None:
                continue
            for name, arr in vectors.items():
                combined[name].append(arr)

        if any(len(v) == 0 for v in combined.values()):
            continue

        # Concatenate across PIDs (combine all neurons for this region)
        concat = {name: np.concatenate(vals) for name, vals in combined.items()}
        n_units = len(concat["A1"])
        if n_units < ANALYSIS_MIN_N:
            continue

        stats = compute_similarity_from_vectors(
            concat["A1"], concat["A2"], concat["B1"], concat["B2"]
        )
        stats.update(
            {
                "region": region,
                "analysis": ANALYSIS_LABEL,
                "n_units": n_units,
            }
        )
        results.append(stats)

    results_df = pd.DataFrame(results)
    display(results_df)

    required_cols = ["total_reliability", "raw_correlation", "similarity", "region"]
    if not all(col in results_df.columns for col in required_cols):
        print("Missing expected columns for plotting. Check vector specs and inputs.")
    else:
        plot_df = results_df.dropna(subset=required_cols)
        regions_for_colors = sorted(plot_df["region"].astype(str).unique().tolist())
        region_colors = None
        if hasattr(plotting_utils, "_region_color_map"):
            region_colors = plotting_utils._region_color_map(regions_for_colors)

        # Plot 1: Raw correlation vs total reliability
        fig_raw = px.scatter(
            plot_df,
            x="total_reliability",
            y="raw_correlation",
            color="region",
            category_orders={"region": regions_for_colors},
            color_discrete_map=region_colors,
            hover_data=["n_units"],
            title="Raw Correlation vs Total Reliability (Regions Combined Across PIDs)",
        )
        fig_raw.update_layout(
            width=900,
            height=650,
            margin=dict(l=70, r=40, t=80, b=60),
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        )
        show_fig(fig_raw)

        # Plot 2: Similarity vs total reliability
        fig_sim = px.scatter(
            plot_df,
            x="total_reliability",
            y="similarity",
            color="region",
            category_orders={"region": regions_for_colors},
            color_discrete_map=region_colors,
            hover_data=["n_units"],
            title="Similarity vs Total Reliability (Regions Combined Across PIDs)",
        )
        fig_sim.update_layout(
            width=900,
            height=650,
            margin=dict(l=70, r=40, t=80, b=60),
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        )
        show_fig(fig_sim)
