# %% Imports
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

import plotly.io as pio
import plotly.express as px

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.plotting_plotly import (  # noqa: E402
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
    plot_population_coupling_heatmap_plotly,
    plot_coupling_strength_summary_plotly,
    plot_coupling_delay_summary_plotly,
    plot_single_neuron_plotly,
    plot_single_neuron_conditioned_event_plotly,
    plot_stpr_curve_halves_plotly,
)
import utils.plotting_plotly as plotting_utils


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
    # Make the exported HTML background match dark mode so the browser tab isn't white.
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


# %% Helpers
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"


def list_pids(cache_dir=CACHE_DIR):
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.pkl")])


def choose_pid(pid_list, default_index=0):
    if not pid_list:
        raise RuntimeError("No PIDs found in data/dashboard_cache.")
    print("Available PIDs:")
    for idx, pid_val in enumerate(pid_list):
        print(f"  [{idx}] {pid_val}")
    prompt = f"Enter PID or index (press Enter for {pid_list[default_index]}): "
    selection = input(prompt).strip()
    if selection == "":
        return pid_list[default_index]
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(pid_list):
            return pid_list[idx]
        raise ValueError(f"Index {idx} out of range.")
    if selection in pid_list:
        return selection
    raise ValueError(f"PID '{selection}' not found.")


def load_cache(pid, cache_dir=CACHE_DIR):
    path = cache_dir / f"{pid}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_array(obj, key):
    if obj is None:
        return None
    if hasattr(obj, key):
        return np.asarray(getattr(obj, key))
    if isinstance(obj, dict) and key in obj:
        return np.asarray(obj[key])
    return None


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


def _format_seconds(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    return f"{val:.2f}s"


def _spont_interval_text(interval):
    if interval is None:
        return "NA"
    start, end = interval
    if start is None or end is None:
        return "NA"
    return f"{start:.2f}-{end:.2f}s"


def build_region_table(cluster_acronyms, labels):
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    all_counts = pd.Series(cluster_acronyms).value_counts().sort_index()
    if labels is not None:
        good_counts = pd.Series(cluster_acronyms[labels == 1]).value_counts().sort_index()
    else:
        good_counts = pd.Series(dtype=int)
    region_table = (
        pd.DataFrame({"region": all_counts.index, "all": all_counts.values})
        .merge(
            pd.DataFrame({"region": good_counts.index, "good": good_counts.values}),
            on="region",
            how="left",
        )
        .fillna(0)
    )
    region_table["all"] = region_table["all"].astype(int)
    region_table["good"] = region_table["good"].astype(int)
    return region_table


def _df_overview(df, name):
    if df is None:
        return pd.DataFrame(
            [{"name": name, "type": "None", "shape": None, "columns": None}]
        )
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(
            [{"name": name, "type": type(df).__name__, "shape": None, "columns": None}]
        )
    cols = list(df.columns)
    if len(cols) > 12:
        col_text = ", ".join(cols[:12]) + ", ..."
    else:
        col_text = ", ".join(cols)
    return pd.DataFrame(
        [
            {
                "name": name,
                "type": "DataFrame",
                "shape": df.shape,
                "columns": col_text,
            }
        ]
    )


def summarize_df(df, name, head_rows=5):
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"{name}: None")
        return
    print(f"{name}: shape={df.shape}")
    display(df.head(head_rows))
    def _normalize_value(val):
        if isinstance(val, np.ndarray):
            return tuple(val.tolist())
        if isinstance(val, (list, tuple)):
            return tuple(_normalize_value(v) for v in val)
        if isinstance(val, dict):
            return tuple(sorted((k, _normalize_value(v)) for k, v in val.items()))
        return val

    def _safe_nunique(frame):
        counts = {}
        for col in frame.columns:
            series = frame[col]
            if series.dtype == "object":
                try:
                    counts[col] = series.nunique(dropna=True)
                except TypeError:
                    counts[col] = series.map(_normalize_value).nunique(dropna=True)
            else:
                counts[col] = series.nunique(dropna=True)
        return pd.Series(counts)

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_null": df.isna().sum(),
            "n_unique": _safe_nunique(df),
        }
    )
    display(summary)


def describe_spikes(spikes):
    times = _as_array(spikes, "times")
    clusters = _as_array(spikes, "clusters")
    if times is None or clusters is None:
        print("spikes: missing times/clusters arrays")
        return
    print(
        "spikes:",
        f"n_spikes={len(times)}",
        f"n_units={len(np.unique(clusters))}",
        f"time_range=({np.nanmin(times):.2f}, {np.nanmax(times):.2f})",
    )


def describe_clusters(clusters):
    if clusters is None:
        print("clusters: None")
        return
    keys = []
    if isinstance(clusters, dict):
        keys = list(clusters.keys())
    else:
        keys = [k for k in ["acronym", "depths", "metrics", "label"] if hasattr(clusters, k)]
    print(f"clusters: fields={keys}")
    metrics = getattr(clusters, "metrics", None)
    if isinstance(metrics, pd.DataFrame):
        print(f"clusters.metrics: shape={metrics.shape}, columns={list(metrics.columns)[:12]}")


def _get_plot_config(data, plot_only_good):
    config_plot = dict(data.get("config_plot", {}))
    config_calc = data.get("config_calc", {})
    config_plot["PLOT_ONLY_GOOD_UNITS"] = plot_only_good
    config_plot["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
    config_plot["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
    config_plot["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
    config_plot["PLOTLY_TEMPLATE"] = "plotly_white"
    return config_plot, config_calc


def _filter_by_good(data, plot_only_good):
    df_coupling = data.get("df_coupling")
    df_coupling_task = data.get("df_coupling_task")
    df_comparison = data.get("df_comparison")
    if not plot_only_good:
        return df_coupling, df_coupling_task, df_comparison
    cluster_ids = data.get("cluster_ids")
    labels = _get_label_array(data.get("clusters"))
    if labels is None or cluster_ids is None:
        return df_coupling, df_coupling_task, df_comparison
    good_cluster_ids = np.asarray(cluster_ids)[labels == 1]
    if df_coupling is not None:
        df_coupling = df_coupling[df_coupling["cluster_id"].isin(good_cluster_ids)]
    if df_coupling_task is not None:
        df_coupling_task = df_coupling_task[
            df_coupling_task["cluster_id"].isin(good_cluster_ids)
        ]
    if df_comparison is not None:
        df_comparison = df_comparison[
            df_comparison["cluster_id"].isin(good_cluster_ids)
        ]
    return df_coupling, df_coupling_task, df_comparison


def _choose_trial_row(trials, trial_idx=None):
    if trials is None or trials.empty:
        return None
    if trial_idx is None:
        return trials.iloc[0]
    if "trial_idx" in trials.columns and trial_idx in trials["trial_idx"].values:
        return trials.loc[trials["trial_idx"] == trial_idx].iloc[0]
    return trials.iloc[0]


def _pick_default_cluster_id(cluster_ids):
    if cluster_ids is None:
        return None
    if len(cluster_ids) == 0:
        return None
    return int(np.asarray(cluster_ids)[0])


# %% Select PID and load cache
pid_list = list_pids()
if not pid_list:
    raise RuntimeError("No cached sessions found in data/dashboard_cache.")

# Option A: Set PID directly
PID = None  # Example: "c9664185-d3fd-4e0e-89cf-77c402038938"

if PID is None:
    pid = choose_pid(pid_list, default_index=0)
else:
    if PID not in pid_list:
        raise ValueError(f"PID '{PID}' not found in cache list.")
    pid = PID

print(f"Selected PID: {pid}")

data = load_cache(pid)


# %% Quick cache summary
cache_keys = sorted(list(data.keys()))
print(f"Cache keys ({len(cache_keys)}): {cache_keys}")

overview = pd.concat(
    [
        _df_overview(data.get("trials"), "trials"),
        _df_overview(data.get("df_res"), "df_res"),
        _df_overview(data.get("df_reliability"), "df_reliability"),
        _df_overview(data.get("df_coupling"), "df_coupling"),
        _df_overview(data.get("df_coupling_task"), "df_coupling_task"),
        _df_overview(data.get("df_comparison"), "df_comparison"),
    ],
    ignore_index=True,
)
display(overview)

describe_spikes(data.get("spikes"))
describe_clusters(data.get("clusters"))

meta = data.get("meta", {})
info = {
    "Lab": meta.get("lab"),
    "Num trials": meta.get("num_trials"),
    "PID": meta.get("pid"),
    "EID": meta.get("eid"),
    "PIDs in session": meta.get("num_other_pids"),
    "Date": meta.get("date"),
    "Recording length": _format_seconds(meta.get("recording_length_s")),
    "Spont length": _format_seconds(meta.get("spont_length_s")),
    "Subject": meta.get("subject"),
    "Spont interval": _spont_interval_text(meta.get("spont_interval")),
}
info_df = pd.DataFrame(info, index=[0]).T
info_df.columns = ["Value"]
display(info_df.astype(str))


# %% DataFrames: quick structure
summarize_df(data.get("trials"), "trials")
summarize_df(data.get("df_res"), "df_res")
summarize_df(data.get("df_reliability"), "df_reliability")
summarize_df(data.get("df_coupling"), "df_coupling")
summarize_df(data.get("df_coupling_task"), "df_coupling_task")
summarize_df(data.get("df_comparison"), "df_comparison")


# %% Region counts (all vs good)
cluster_acronyms = data.get("cluster_acronyms_plot")
labels = _get_label_array(data.get("clusters"))
if cluster_acronyms is not None:
    region_table = build_region_table(cluster_acronyms, labels)
    display(region_table)


# %% Plot configuration
plot_only_good = True
variability_metric = "fano"  # "fano" or "cv"
sorting_metric = "depth"  # "depth", "stim", "feedback", "move", "spont", "task"
population_sort_mode = "delay"  # "delay", "spont", "task", "depth"
plot_regions = None  # Example: ["VISp", "MOp"] or None for all
trial_idx = 234  # Set an int to override the default trial selection
selected_cluster_id = None  # Set an int to override the default cluster selection

use_dark_theme = False  # True -> plotly_dark for all plots

plot_config, config_calc = _get_plot_config(data, plot_only_good)
if use_dark_theme:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_dark"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_dark"
    pio.templates.default = "plotly_dark"
else:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_white"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_white"
    pio.templates.default = "plotly_white"
df_coupling_plot, df_coupling_task_plot, df_comparison_plot = _filter_by_good(
    data, plot_only_good
)

spikes = data.get("spikes")
clusters = data.get("clusters")
cluster_ids = data.get("cluster_ids")
cluster_acronyms = data.get("cluster_acronyms_plot")
session = data.get("session")
trials = data.get("trials")


# %% General raster (time window)
times = _as_array(spikes, "times")
if times is not None and len(times) > 0:
    min_time = float(np.nanmin(times))
    max_time = float(np.nanmax(times))
    t_start = 960.0  # min_time
    t_end = 966.0  # max_time

    fig_general = plot_time_window_raster_plotly(
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        session,
        plot_config,
        t_start,
        t_end,
        sorting_metric=sorting_metric,
        variability_metric=variability_metric,
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
        pupil_features=data.get("pupil_features"),
        pupil_times=data.get("pupil_times"),
        region_colors=None,
    )
    show_fig(fig_general)
else:
    print("No spike times available for general raster.")


# %% Trial inspector
trial_row = _choose_trial_row(trials, trial_idx=trial_idx)
if trial_row is None:
    print("Trials not available in cache.")
else:
    if trial_idx is None:
        trial_idx = int(trial_row["trial_idx"])
    trial_table = pd.DataFrame(
        {
            "Trial": [trial_idx],
            "Contrast": [trial_row.get("contrast")],
            "Reaction Time": [trial_row.get("reaction_time")],
            "Correct Response": [trial_row.get("correct_response")],
            "Subject Response": [trial_row.get("subject_response")],
        }
    )
    display(trial_table)

    fig_trial = plot_trial_raster_plotly(
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        session,
        plot_config,
        trial_idx,
        variability_metric=variability_metric,
        sorting_metric=sorting_metric,
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
        pupil_features=data.get("pupil_features"),
        pupil_times=data.get("pupil_times"),
        region_colors=None,
    )
    show_fig(fig_trial)


# %% Population analysis
if trials is None:
    print("No trials in cache. Skip population analysis.")
else:
    for event_name in ["stimOn_times", "firstMovement_times", "feedback_times"]:
        cfg = dict(plot_config)
        cfg["PLOT_EVENT"] = event_name
        fig_pop = plot_population_sorted_plotly(
            session,
            spikes,
            clusters,
            cluster_ids,
            cluster_acronyms,
            data.get("df_res"),
            cfg,
            df_coupling=df_coupling_plot,
            df_coupling_task=df_coupling_task_plot,
            region_acronyms=plot_regions,
            sort_mode=population_sort_mode,
        )
        show_fig(fig_pop)


# %% Coupling heatmaps
if df_coupling_plot is None or df_coupling_task_plot is None:
    print("Coupling data missing. Re-run 03_calc_dashboard.py with CALC_SPONT=True.")
else:
    fig_spont = plot_population_coupling_heatmap_plotly(
        df_coupling_plot, plot_config, config_calc, region_acronyms=plot_regions
    )
    fig_task = plot_population_coupling_heatmap_plotly(
        df_coupling_task_plot, plot_config, config_calc, region_acronyms=plot_regions
    )
    show_fig(fig_spont)
    show_fig(fig_task)


# %% stPR comparison (spont vs task)
if df_comparison_plot is None or df_comparison_plot.empty:
    print("df_comparison missing or empty. Skipping stPR comparison plots.")
else:
    region_order = (
        df_comparison_plot["region"].unique().tolist()
        if "region" in df_comparison_plot.columns
        else None
    )
    fig_strength = plot_coupling_strength_summary_plotly(
        df_comparison_plot,
        region_order=region_order,
        region_colors=None,
        template=plot_config["PLOTLY_TEMPLATE"],
        highlight_cluster_id=None,
    )
    fig_delay = plot_coupling_delay_summary_plotly(
        df_comparison_plot,
        region_order=region_order,
        region_colors=None,
        template=plot_config["PLOTLY_TEMPLATE"],
        highlight_cluster_id=None,
    )
    show_fig(fig_strength)
    show_fig(fig_delay)


# %% Single neuron plots
cluster_id = selected_cluster_id or _pick_default_cluster_id(cluster_ids)
cluster_id = 357
if cluster_id is None:
    print("No cluster IDs available.")
else:
    fig_single = plot_single_neuron_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        cluster_id,
    )
    show_fig(fig_single)

    fig_move = plot_single_neuron_conditioned_event_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        cluster_id,
        event_name="firstMovement_times",
        condition_type="choice",
        title="First Movement Response",
    )
    show_fig(fig_move)

    fig_feedback = plot_single_neuron_conditioned_event_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        cluster_id,
        event_name="feedback_times",
        condition_type="feedback",
        title="Feedback Response",
    )
    show_fig(fig_feedback)

    if df_coupling_plot is not None:
        fig_spont_curve = plot_stpr_curve_halves_plotly(
            df_coupling_plot,
            config_calc,
            cluster_id,
            title="Spont stPR Curve (First vs Second Half)",
            template=plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_spont_curve)
    if df_coupling_task_plot is not None:
        fig_task_curve = plot_stpr_curve_halves_plotly(
            df_coupling_task_plot,
            config_calc,
            cluster_id,
            title="Task stPR Curve (First vs Second Half)",
            template=plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_task_curve)


# %% Noise-corrected similarity analysis (generic, easy to swap vectors)
# This section implements the 7-step workflow you described.
# It is written to be re-used for any two paired measures by editing the vector specs.

# -----------------------
# USER-EDITABLE SETTINGS
# -----------------------
ANALYSIS_RUN = True
ANALYSIS_LABEL = "spont_delay_vs_stim_delay"
ANALYSIS_EVENT = "stimOn_times"  # used to pick delay_h1_{event}, delay_h2_{event}
ANALYSIS_PIDS = [pid]  # change to pid_list for all sessions
ANALYSIS_REGION = None  # e.g. "VISp" or "MOp"; None = all regions
ANALYSIS_GROUP_BY_REGION = True  # True -> one point per region (legend shows regions)
ANALYSIS_MIN_N = 20  # minimum neurons required to compute metrics
ANALYSIS_ONLY_GOOD = False  # True -> use only good units (label==1)

# Vector definitions:
# - df: which cached DataFrame to read from
# - col: the column name containing the vector values
# You can swap these to compare any pair of measures.
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
    regions = sorted([r for r in regions if r not in ("root", "void")])
    return regions


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
    r_xx = _pearsonr(a1, a2)  # A reliability
    r_yy = _pearsonr(b1, b2)  # B reliability
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


def compute_similarity_for_cache(cache, vector_specs, region=None, only_good=False, min_n=10):
    # STEP 1: build and align A1/A2/B1/B2
    vector_tables = {}
    for name, spec in vector_specs.items():
        vector_tables[name] = _vector_table_from_cache(
            cache, spec, region=region, only_good=only_good
        )

    aligned, vectors = _align_vectors(vector_tables)
    if aligned is None:
        return None

    n = len(aligned)
    if n < min_n:
        return {"n_units": n}

    stats = compute_similarity_from_vectors(
        vectors["A1"], vectors["A2"], vectors["B1"], vectors["B2"]
    )
    stats["n_units"] = n
    return stats


if ANALYSIS_RUN:
    # STEP 5: repeat across sessions (or regions)
    results = []
    for pid_i in ANALYSIS_PIDS:
        cache_i = load_cache(pid_i)
        if ANALYSIS_REGION is not None:
            region_list = [ANALYSIS_REGION]
        elif ANALYSIS_GROUP_BY_REGION:
            region_list = _available_regions(cache_i, ANALYSIS_VECTOR_SPECS)
        else:
            region_list = [None]

        if not region_list:
            region_list = [None]

        for region in region_list:
            stats = compute_similarity_for_cache(
                cache_i,
                ANALYSIS_VECTOR_SPECS,
                region=region,
                only_good=ANALYSIS_ONLY_GOOD,
                min_n=ANALYSIS_MIN_N,
            )
            if stats is None:
                continue
            stats.update(
                {
                    "session_id": pid_i,
                    "region": region or "ALL",
                    "analysis": ANALYSIS_LABEL,
                }
            )
            results.append(stats)

    results_df = pd.DataFrame(results)
    display(results_df)

    required_cols = ["total_reliability", "raw_correlation", "similarity"]
    if not all(col in results_df.columns for col in required_cols):
        print(
            "Missing expected columns for plotting. "
            "Check ANALYSIS_VECTOR_SPECS and that the source DataFrames "
            "contain the requested columns."
        )
        plot_df = pd.DataFrame()
    else:
        plot_df = results_df.dropna(subset=required_cols)

    if not plot_df.empty:
        regions_for_colors = sorted(plot_df["region"].astype(str).unique().tolist())
        region_colors = None
        if hasattr(plotting_utils, "_region_color_map"):
            region_colors = plotting_utils._region_color_map(regions_for_colors)

        # STEP 6: Plot raw correlation vs total reliability
        fig_raw = px.scatter(
            plot_df,
            x="total_reliability",
            y="raw_correlation",
            color="region",
            category_orders={"region": regions_for_colors},
            color_discrete_map=region_colors,
            hover_data=["session_id", "n_units"],
            title="Raw Correlation vs Total Reliability",
        )
        fig_raw.update_layout(
            width=900,
            height=650,
            margin=dict(l=70, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        show_fig(fig_raw)

        # STEP 7: Plot similarity vs total reliability
        fig_sim = px.scatter(
            plot_df,
            x="total_reliability",
            y="similarity",
            color="region",
            category_orders={"region": regions_for_colors},
            color_discrete_map=region_colors,
            hover_data=["session_id", "n_units"],
            title="Similarity vs Total Reliability",
        )
        # Add y=x reference line (dashed red)
        x_vals = plot_df["total_reliability"].to_numpy(dtype=float)
        y_vals = plot_df["similarity"].to_numpy(dtype=float)
        if np.isfinite(x_vals).any() and np.isfinite(y_vals).any():
            min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
            max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
            fig_sim.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(color="red", dash="dash"),
            )
        fig_sim.update_layout(
            width=900,
            height=650,
            margin=dict(l=70, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        show_fig(fig_sim)
    else:
        print("Not enough data points to plot similarity results.")
