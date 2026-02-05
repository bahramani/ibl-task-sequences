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
import plotly.graph_objects as go
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

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
from utils.io import setup_paths, init_one, load_session_data


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
LOAD_RAW_DATA = True
LOAD_RAW_WHEEL = False
LOAD_RAW_POSE = False
ALLOW_REMOTE_METADATA = True


def _load_raw_session(pid, load_wheel=False, load_pose=False, mode="local"):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one = init_one(ibl_cache, mode=mode)
    ssl, spikes, clusters, sl = load_session_data(
        pid,
        one,
        load_wheel=load_wheel,
        load_pose=load_pose,
    )
    return spikes, clusters, sl, ssl


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


def _get_label_array(clusters, cluster_ids=None):
    if clusters is None:
        return None
    if cluster_ids is not None:
        values = plotting_utils._get_label_values(clusters, cluster_ids)
        if values is not None:
            return values
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
    label_levels = ["0.00", "0.30-0.40", "0.60-0.70", "1.00"]

    df = pd.DataFrame({"region": cluster_acronyms})
    if labels is None:
        df["label"] = np.nan
    else:
        df["label"] = np.asarray(labels, dtype=float)

    regions = pd.Series(cluster_acronyms).value_counts().sort_index()
    rows = []
    for region in regions.index:
        region_df = df[df["region"] == region]
        total = len(region_df)
        row = {"region": region}
        for lvl in label_levels:
            if lvl == "0.00":
                mask = np.isclose(region_df["label"], 0.0)
            elif lvl == "1.00":
                mask = np.isclose(region_df["label"], 1.0)
            elif lvl == "0.30-0.40":
                mask = (region_df["label"] >= 0.3) & (region_df["label"] <= 0.4)
            else:
                mask = (region_df["label"] >= 0.6) & (region_df["label"] <= 0.7)
            count = int(mask.sum())
            pct = (count / total * 100.0) if total > 0 else 0.0
            row[lvl] = f"{count} ({pct:.1f}%)"
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        total_all = len(df)
        all_row = {"region": "ALL"}
        for lvl in label_levels:
            if lvl == "0.00":
                mask = np.isclose(df["label"], 0.0)
            elif lvl == "1.00":
                mask = np.isclose(df["label"], 1.0)
            elif lvl == "0.30-0.40":
                mask = (df["label"] >= 0.3) & (df["label"] <= 0.4)
            else:
                mask = (df["label"] >= 0.6) & (df["label"] <= 0.7)
            count = int(mask.sum())
            pct = (count / total_all * 100.0) if total_all > 0 else 0.0
            all_row[lvl] = f"{count} ({pct:.1f}%)"
        table = pd.concat([table, pd.DataFrame([all_row])], ignore_index=True)
    return table


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


def _get_plot_config(data, plot_label_min):
    config_plot = dict(data.get("config_plot", {}))
    config_calc = data.get("config_calc", {})
    config_plot["PLOT_ONLY_GOOD_UNITS"] = False
    config_plot["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
    config_plot["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
    config_plot["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
    config_plot["PLOT_LABEL_MIN"] = plot_label_min
    config_plot["PLOTLY_TEMPLATE"] = "plotly_white"
    config_plot["DELAY_UNITS"] = config_calc.get("DELAY_UNITS", "s")
    return config_plot, config_calc


def _filter_by_label_min(data, plot_label_min):
    df_coupling = data.get("df_coupling")
    df_coupling_task = data.get("df_coupling_task")
    df_comparison = data.get("df_comparison")
    if plot_label_min is None:
        return df_coupling, df_coupling_task, df_comparison
    cluster_ids = data.get("cluster_ids")
    labels = _get_label_array(data.get("clusters"), cluster_ids)
    if labels is None or cluster_ids is None:
        return df_coupling, df_coupling_task, df_comparison
    try:
        labels_float = labels.astype(float)
        plot_cluster_ids = np.asarray(cluster_ids)[labels_float >= float(plot_label_min)]
    except (TypeError, ValueError):
        plot_cluster_ids = np.asarray(cluster_ids)[labels == 1]
    if df_coupling is not None:
        df_coupling = df_coupling[df_coupling["cluster_id"].isin(plot_cluster_ids)]
    if df_coupling_task is not None:
        df_coupling_task = df_coupling_task[
            df_coupling_task["cluster_id"].isin(plot_cluster_ids)
        ]
    if df_comparison is not None:
        df_comparison = df_comparison[
            df_comparison["cluster_id"].isin(plot_cluster_ids)
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


# %% Select PID and load cache.
pid_list = list_pids()
if not pid_list:
    raise RuntimeError("No cached sessions found in data/dashboard_cache.")

# Option A: Set PID directly
PID = "c9664185-d3fd-4e0e-89cf-77c402038938"# None  # Example: "c9664185-d3fd-4e0e-89cf-77c402038938"

if PID is None:
    pid = choose_pid(pid_list, default_index=0)
else:
    if PID not in pid_list:
        raise ValueError(f"PID '{PID}' not found in cache list.")
    pid = PID

print(f"Selected PID: {pid}")

data = load_cache(pid)
spikes = data.get("spikes")
clusters = data.get("clusters")
session = data.get("session")
raw_error = None
raw_error_remote = None
if LOAD_RAW_DATA or spikes is None or clusters is None or session is None:
    try:
        spikes, clusters, session, _ssl = _load_raw_session(
            pid, load_wheel=LOAD_RAW_WHEEL, load_pose=LOAD_RAW_POSE, mode="local"
        )
    except Exception as exc:
        raw_error = exc
        if ALLOW_REMOTE_METADATA:
            try:
                spikes, clusters, session, _ssl = _load_raw_session(
                    pid, load_wheel=LOAD_RAW_WHEEL, load_pose=LOAD_RAW_POSE, mode="remote"
                )
            except Exception as exc_remote:
                raw_error_remote = exc_remote

    if raw_error is not None:
        if spikes is None or clusters is None or session is None:
            msg = (
                "Raw session data not available in data/raw. "
                "Run 03_calc_dashboard.py once with remote access to populate the cache."
            )
            if raw_error_remote is not None:
                msg = (
                    f"{msg}\nLocal error: {type(raw_error).__name__}: {raw_error}\n"
                    f"Remote error: {type(raw_error_remote).__name__}: {raw_error_remote}"
                )
            raise RuntimeError(msg) from raw_error
        if raw_error_remote is None:
            print(f"Raw load failed, using cached blobs: {raw_error}")
        else:
            print(f"Local load failed; using remote metadata lookup: {raw_error}")
    else:
        data["spikes"] = spikes
        data["clusters"] = clusters
        data["session"] = session


# %% Quick cache summary
cache_keys = sorted(list(data.keys()))
print(f"Cache keys ({len(cache_keys)}): {cache_keys}")

overview = pd.concat(
    [
        _df_overview(data.get("trials"), "trials"),
        _df_overview(data.get("df_res"), "df_res"),
        _df_overview(data.get("df_coupling"), "df_coupling"),
        _df_overview(data.get("df_coupling_task"), "df_coupling_task"),
        _df_overview(data.get("df_coupling_task_tf"), "df_coupling_task_tf"),
        _df_overview(data.get("df_coupling_iti"), "df_coupling_iti"),
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
summarize_df(data.get("df_coupling"), "df_coupling")
summarize_df(data.get("df_coupling_task"), "df_coupling_task")
summarize_df(data.get("df_coupling_task_tf"), "df_coupling_task_tf")
summarize_df(data.get("df_coupling_iti"), "df_coupling_iti")
summarize_df(data.get("df_comparison"), "df_comparison")


# %% Region counts (all vs good)
cluster_acronyms = data.get("cluster_acronyms_plot")
labels = _get_label_array(data.get("clusters"), data.get("cluster_ids"))
if cluster_acronyms is not None:
    region_table = build_region_table(cluster_acronyms, labels)
    display(region_table)


# %% Plot configuration
plot_label_min = 0.5
variability_metric = "fano"  # "fano" or "cv"
sorting_metric = "depth"  # "depth", "stim", "feedback", "move", "spont", "task"
population_sort_mode = "delay"  # "delay", "spont", "task", "depth"
plot_regions = None  # Example: ["VISp", "MOp"] or None for all
trial_idx = 234  # Set an int to override the default trial selection
selected_cluster_id = None  # Set an int to override the default cluster selection

use_dark_theme = False  # True -> plotly_dark for all plots

plot_config, config_calc = _get_plot_config(data, plot_label_min)
if use_dark_theme:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_dark"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_dark"
    pio.templates.default = "plotly_dark"
else:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_white"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_white"
    pio.templates.default = "plotly_white"
df_coupling_plot, df_coupling_task_plot, df_comparison_plot = _filter_by_label_min(
    data, plot_label_min
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
            title="Task stPR Curve (Odd vs Even Trials)",
            template=plot_config["PLOTLY_TEMPLATE"],
            split_suffixes=("odd", "even"),
            split_labels=("Odd trials", "Even trials"),
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
ANALYSIS_LABEL_MIN = 0.5  # None -> all units

# Vector definitions:
# - df: which cached DataFrame to read from
# - col: the column name containing the vector values
# You can swap these to compare any pair of measures.
ANALYSIS_VECTOR_SPECS = {
    # Spontaneous delays (first/second half)
    "A1": {"df": "df_coupling", "col": "coupling_delay_ms_h1"},
    "A2": {"df": "df_coupling", "col": "coupling_delay_ms_h2"},
    # Evoked delays (event-specific split-halves)
    "B1": {"df": "df_res", "col": f"delay_{ANALYSIS_EVENT}_odd"},
    "B2": {"df": "df_res", "col": f"delay_{ANALYSIS_EVENT}_even"},
}

# Example swap (strengths instead of delays):
# ANALYSIS_VECTOR_SPECS = {
#     "A1": {"df": "df_coupling", "col": "coupling_strength_h1"},
#     "A2": {"df": "df_coupling", "col": "coupling_strength_h2"},
#     "B1": {"df": "df_coupling_task", "col": "coupling_strength_odd"},
#     "B2": {"df": "df_coupling_task", "col": "coupling_strength_even"},
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


def _label_cluster_ids(cache, label_min=None):
    labels = _get_label_array(cache.get("clusters"), cache.get("cluster_ids"))
    cluster_ids_local = cache.get("cluster_ids")
    if labels is None or cluster_ids_local is None:
        return None
    if label_min is None:
        return set(np.asarray(cluster_ids_local).tolist())
    try:
        labels_float = labels.astype(float)
        return set(
            np.asarray(cluster_ids_local)[labels_float >= float(label_min)].tolist()
        )
    except (TypeError, ValueError):
        return set(np.asarray(cluster_ids_local)[labels == 1].tolist())


def _vector_table_from_cache(cache, spec, region=None, label_min=None, key="cluster_id"):
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

    if label_min is not None:
        good_ids = _label_cluster_ids(cache, label_min)
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


def compute_similarity_for_cache(cache, vector_specs, region=None, label_min=None, min_n=10):
    # STEP 1: build and align A1/A2/B1/B2
    vector_tables = {}
    for name, spec in vector_specs.items():
        vector_tables[name] = _vector_table_from_cache(
            cache, spec, region=region, label_min=label_min
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
                label_min=ANALYSIS_LABEL_MIN,
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


# %% Reliability + pairwise correlations (per region)
CORR_RUN = True
CORR_LABEL_MIN = plot_label_min
CORR_MIN_N = 2

# Variables to include in reliability + pairwise correlation analysis.
CORR_VARIABLES = [
    {
        "name": "Delay (Stim On)",
        "df": "df_res",
        "v1": "delay_stimOn_times_odd",
        "v2": "delay_stimOn_times_even",
    },
    {
        "name": "Delay (First Move)",
        "df": "df_res",
        "v1": "delay_firstMovement_times_odd",
        "v2": "delay_firstMovement_times_even",
    },
    {
        "name": "Delay (Response)",
        "df": "df_res",
        "v1": "delay_response_times_odd",
        "v2": "delay_response_times_even",
    },
    {
        "name": "Delay (Feedback)",
        "df": "df_res",
        "v1": "delay_feedback_times_odd",
        "v2": "delay_feedback_times_even",
    },
    {
        "name": "stPR Delay (Spont)",
        "df": "df_coupling",
        "v1": "coupling_delay_ms_h1",
        "v2": "coupling_delay_ms_h2",
    },
    {
        "name": "stPR Delay (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "name": "stPR Delay (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "name": "stPR Strength (Spont)",
        "df": "df_coupling",
        "v1": "coupling_strength_h1",
        "v2": "coupling_strength_h2",
    },
    {
        "name": "stPR Strength (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "name": "stPR Strength (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "name": "stPR Max (Spont)",
        "df": "df_coupling",
        "v1": "coupling_max_h1",
        "v2": "coupling_max_h2",
    },
    {
        "name": "stPR Max (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_max_odd",
        "v2": "coupling_max_even",
    },
    {
        "name": "stPR Max (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_max_odd",
        "v2": "coupling_max_even",
    },
]


def _pearsonr_with_n(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < CORR_MIN_N:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


def _spearmanr_with_n(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < CORR_MIN_N:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if spearmanr is not None:
        res = spearmanr(x, y)
        return float(res.correlation), n
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan, n
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), n


def _build_region_lookup(cache, label_min=None):
    cluster_ids_local = cache.get("cluster_ids")
    cluster_acronyms_local = cache.get("cluster_acronyms_plot")
    if cluster_ids_local is None or cluster_acronyms_local is None:
        return pd.DataFrame(columns=["cluster_id", "region"])
    region_df = pd.DataFrame(
        {
            "cluster_id": np.asarray(cluster_ids_local),
            "region": np.asarray(cluster_acronyms_local).astype(str),
        }
    )
    if label_min is not None:
        labels = _get_label_array(cache.get("clusters"), cache.get("cluster_ids"))
        if labels is not None:
            try:
                labels_float = labels.astype(float)
                good_ids = np.asarray(cluster_ids_local)[
                    labels_float >= float(label_min)
                ]
            except (TypeError, ValueError):
                good_ids = np.asarray(cluster_ids_local)[labels == 1]
            region_df = region_df[region_df["cluster_id"].isin(good_ids)]
    region_df = region_df[~region_df["region"].isin(["root", "void"])]
    return region_df.reset_index(drop=True)


def _build_variable_table(cache, spec, region_lookup):
    df = cache.get(spec["df"])
    if df is None:
        return None
    if spec["v1"] not in df.columns or spec["v2"] not in df.columns:
        return None
    df_var = df[["cluster_id", spec["v1"], spec["v2"]]].copy()
    df_var = df_var.groupby("cluster_id", as_index=False).mean(numeric_only=True)
    df_var = df_var.merge(region_lookup, on="cluster_id", how="inner")
    v1 = df_var[spec["v1"]].to_numpy(dtype=float)
    v2 = df_var[spec["v2"]].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(v1) & np.isfinite(v2)
    mean_vals[valid] = (v1[valid] + v2[valid]) / 2.0
    df_var["mean"] = mean_vals
    return df_var


if CORR_RUN:
    region_lookup = _build_region_lookup(data, CORR_LABEL_MIN)
    if region_lookup.empty:
        print("No regions available for correlation analysis.")
    else:
        available_specs = []
        missing_vars = []
        for spec in CORR_VARIABLES:
            df_src = data.get(spec["df"])
            if df_src is None:
                missing_vars.append(spec["name"])
                continue
            if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
                missing_vars.append(spec["name"])
                continue
            available_specs.append(spec)

        if missing_vars:
            print("Skipping missing variables:", ", ".join(missing_vars))

        if not available_specs:
            print("No variables available for correlation analysis.")
        else:
            spec_by_name = {spec["name"]: spec for spec in available_specs}
            regions_all = sorted(region_lookup["region"].unique().tolist())
            if plot_regions:
                regions_filtered = []
                for region in regions_all:
                    if any(str(region).startswith(str(r)) for r in plot_regions):
                        regions_filtered.append(region)
                regions_all = regions_filtered

            for region in regions_all:
                region_ids = region_lookup.loc[
                    region_lookup["region"] == region, "cluster_id"
                ].to_numpy()
                n_total = int(len(region_ids))
                if n_total == 0:
                    continue

                var_tables = {}
                for spec in available_specs:
                    df_var = _build_variable_table(data, spec, region_lookup)
                    if df_var is None:
                        continue
                    df_var = df_var[df_var["region"] == region]
                    if df_var.empty:
                        continue
                    var_tables[spec["name"]] = df_var

                if not var_tables:
                    continue

                names = [spec["name"] for spec in available_specs if spec["name"] in var_tables]
                if len(names) == 0:
                    continue

                reliability = {}
                reliability_n = {}
                for spec in available_specs:
                    name = spec["name"]
                    df_var = var_tables.get(name)
                    if df_var is None:
                        reliability[name] = np.nan
                        reliability_n[name] = 0
                        continue
                    r, n = _pearsonr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                    reliability[name] = r
                    reliability_n[name] = n

                mean_wide = pd.DataFrame({"cluster_id": region_ids})
                for spec in available_specs:
                    name = spec["name"]
                    df_var = var_tables.get(name)
                    if df_var is None:
                        mean_wide[name] = np.nan
                        continue
                    mean_wide = mean_wide.merge(
                        df_var[["cluster_id", "mean"]].rename(columns={"mean": name}),
                        on="cluster_id",
                        how="left",
                    )

                n_vars = len(names)
                corr_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
                n_mat = np.zeros((n_vars, n_vars), dtype=int)
                text_mat = np.empty((n_vars, n_vars), dtype=object)
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        if i == j:
                            r_val = reliability.get(name_i, np.nan)
                            n_val = reliability_n.get(name_i, 0)
                            corr_mat[i, j] = r_val
                            n_mat[i, j] = n_val
                            text_mat[i, j] = (
                                f"rel={r_val:.2f}<br>(n={n_val})"
                                if np.isfinite(r_val)
                                else f"rel=nan<br>(n={n_val})"
                            )
                        else:
                            r_val, n_val = _pearsonr_with_n(
                                mean_wide[name_i], mean_wide[name_j]
                            )
                            corr_mat[i, j] = r_val
                            n_mat[i, j] = n_val
                            text_mat[i, j] = (
                                f"r={r_val:.2f}<br>(n={n_val})"
                                if np.isfinite(r_val)
                                else f"r=nan<br>(n={n_val})"
                            )

                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_mat,
                        x=names,
                        y=names,
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        reversescale=True,
                        text=text_mat,
                        texttemplate="%{text}",
                        hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title=(
                        f"Reliability (diag) + Pairwise Pearson (off-diag) | "
                        f"Region {region} | N total (label>= {CORR_LABEL_MIN}): {n_total}"
                    ),
                    width=1200,
                    height=1000,
                    template=plot_config["PLOTLY_TEMPLATE"],
                    margin=dict(l=90, r=30, t=90, b=90),
                )
                fig.update_xaxes(tickangle=45)
                show_fig(fig)

                spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
                spearman_n = np.zeros((n_vars, n_vars), dtype=int)
                spearman_text = np.empty((n_vars, n_vars), dtype=object)
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        if i == j:
                            spec = spec_by_name.get(name_i)
                            if spec is None:
                                r_val, n_val = np.nan, 0
                            else:
                                r_val, n_val = _spearmanr_with_n(
                                    var_tables[name_i][spec["v1"]],
                                    var_tables[name_i][spec["v2"]],
                                )
                            spearman_mat[i, j] = r_val
                            spearman_n[i, j] = n_val
                            spearman_text[i, j] = (
                                f"rel={r_val:.2f}<br>(n={n_val})"
                                if np.isfinite(r_val)
                                else f"rel=nan<br>(n={n_val})"
                            )
                        else:
                            r_val, n_val = _spearmanr_with_n(
                                mean_wide[name_i], mean_wide[name_j]
                            )
                            spearman_mat[i, j] = r_val
                            spearman_n[i, j] = n_val
                            spearman_text[i, j] = (
                                f"rho={r_val:.2f}<br>(n={n_val})"
                                if np.isfinite(r_val)
                                else f"rho=nan<br>(n={n_val})"
                            )

                fig_s = go.Figure(
                    data=go.Heatmap(
                        z=spearman_mat,
                        x=names,
                        y=names,
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        reversescale=True,
                        text=spearman_text,
                        texttemplate="%{text}",
                        hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
                    )
                )
                fig_s.update_layout(
                    title=(
                        f"Reliability (diag) + Pairwise Spearman (off-diag) | "
                        f"Region {region} | N total (label>= {CORR_LABEL_MIN}): {n_total}"
                    ),
                    width=1200,
                    height=1000,
                    template=plot_config["PLOTLY_TEMPLATE"],
                    margin=dict(l=90, r=30, t=90, b=90),
                )
                fig_s.update_xaxes(tickangle=45)
                show_fig(fig_s)
