# %% Imports
from pathlib import Path
import fnmatch
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
from plotly.subplots import make_subplots
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None
try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None

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
import utils.analysis as ana_utils
from utils.io import (
    setup_paths,
    init_one,
    load_session_data,
    build_cluster_id_map,
    load_task_replay_datasets,
    build_passive_event_times,
)


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
LOAD_RAW_MOTION_ENERGY = True
LOAD_RAW_PUPIL = True
MOTION_ENERGY_VIEWS = ("left", "right")
ALLOW_REMOTE_METADATA = True


def _load_raw_session(
    pid,
    load_wheel=False,
    load_pose=False,
    load_motion_energy=False,
    load_pupil=False,
    motion_energy_views=None,
    mode="local",
):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one = init_one(ibl_cache, mode="remote")
    ssl, spikes, clusters, sl = load_session_data(
        pid,
        one,
        load_wheel=load_wheel,
        load_pose=load_pose,
        load_motion_energy=load_motion_energy,
        load_pupil=load_pupil,
        motion_energy_views=motion_energy_views,
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


def _get_session_attr(session_obj, name):
    if session_obj is None:
        return None
    if isinstance(session_obj, dict):
        return session_obj.get(name)
    return getattr(session_obj, name, None)


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


def _label_values_for_clusters(cluster_ids, clusters, labels):
    values = plotting_utils._get_label_values(clusters, cluster_ids)
    if values is not None:
        return values
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.shape[0] == len(cluster_ids):
        return labels.astype(float)
    return None


def _get_cluster_firing_rate(clusters, cluster_ids=None):
    if clusters is None:
        return None
    rate = None
    if hasattr(clusters, "firing_rate"):
        rate = np.asarray(clusters.firing_rate)
    elif isinstance(clusters, dict) and "firing_rate" in clusters:
        rate = np.asarray(clusters.get("firing_rate"))
    elif hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "firing_rate" in clusters.metrics.columns:
            rate = np.asarray(clusters.metrics["firing_rate"])
    if rate is None:
        return None
    if cluster_ids is None:
        return rate
    cluster_ids = np.asarray(cluster_ids)
    if len(rate) == len(cluster_ids):
        return rate
    cluster_id_all = None
    if hasattr(clusters, "cluster_id"):
        cluster_id_all = np.asarray(clusters.cluster_id)
    elif isinstance(clusters, dict) and "cluster_id" in clusters:
        cluster_id_all = np.asarray(clusters.get("cluster_id"))
    if cluster_id_all is None or len(cluster_id_all) != len(rate):
        return None
    rate_map = dict(zip(cluster_id_all, rate))
    return np.asarray([rate_map.get(cid, np.nan) for cid in cluster_ids])


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


def _has_spont_interval(meta):
    if not meta:
        return False
    interval = meta.get("spont_interval")
    if interval is None:
        return False
    try:
        start, end = interval
    except (TypeError, ValueError):
        return False
    if start is None or end is None:
        return False
    try:
        start_val = float(start)
        end_val = float(end)
    except (TypeError, ValueError):
        return False
    return np.isfinite(start_val) and np.isfinite(end_val) and end_val > start_val


def _build_region_colors(acronyms):
    if BrainRegions is None:
        return None
    colors = {}
    br = BrainRegions()
    unique_regions = pd.Series(acronyms).astype(str).unique().tolist()
    for region in unique_regions:
        try:
            idx = br.acronym2index(region)[1][0][0]
            rgb = br.rgb[idx]
            colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        except Exception:
            continue
    return colors


def _build_stpr_lags(config_calc, curve_len=None):
    bin_size_ms = config_calc.get("STPR_BIN_SIZE", 0.001) * 1000
    if bin_size_ms <= 0:
        bin_size_ms = 1.0
    window_ms = config_calc.get("STPR_WINDOW_MS", 80)
    window_bins = int(round(window_ms / bin_size_ms)) if bin_size_ms > 0 else 0
    if curve_len is None:
        return np.arange(-window_bins, window_bins + 1) * bin_size_ms
    expected_len = window_bins * 2 + 1
    if curve_len == expected_len:
        return np.arange(-window_bins, window_bins + 1) * bin_size_ms
    half = (curve_len - 1) / 2.0
    return (np.arange(curve_len) - half) * bin_size_ms


def _plot_stpr_mean_comparison(
    df_spont, df_task, df_iti, config_calc, cluster_id, template
):
    fig = go.Figure()
    curve_specs = [
        ("Spont", df_spont, "#1f77b4"),
        ("Task", df_task, "#ff7f0e"),
        ("ITI", df_iti, "#2ca02c"),
    ]
    added = False
    delays = []
    for label, df_src, color in curve_specs:
        if df_src is None or len(df_src) == 0:
            continue
        row = df_src.loc[df_src["cluster_id"] == cluster_id]
        if row.empty:
            continue
        curve = np.asarray(row.iloc[0].get("stpr_curve", []), dtype=float)
        delay = row.iloc[0].get("coupling_delay_ms", np.nan)
        if curve.size == 0:
            continue
        lags = _build_stpr_lags(config_calc, curve.size)
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=curve,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{label} Mean",
            )
        )
        delays.append((label, delay, color))
        added = True

    if not added:
        fig.add_annotation(text="No stPR mean curves available", showarrow=False)
    else:
        for label, delay_val, color in delays:
            if np.isfinite(delay_val):
                fig.add_vline(x=delay_val, line=dict(color=color, dash="dot"))

    fig.update_layout(
        title="stPR Mean Curves (Task vs Spont vs ITI)",
        xaxis_title="Lag (ms)",
        yaxis_title="stPR (z)",
        template=template,
        font=dict(size=13),
        width=900,
        height=550,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


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


def _infer_time_col(df):
    for col in ("times", "time", "timestamp", "timestamps"):
        if col in df.columns:
            return col
    return None


def _numeric_cols(df, exclude=None):
    exclude_set = set(exclude or [])
    cols = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _downsample_df(df, max_points=5000):
    if max_points is None or len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].reset_index(drop=True)


def _plot_df_signal(df, title, max_points=5000):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print(f"{title}: empty or not a DataFrame")
        return None
    time_col = _infer_time_col(df)
    value_cols = _numeric_cols(df, exclude=[time_col] if time_col else None)
    if not value_cols:
        print(f"{title}: no numeric columns to plot")
        return None
    cols = ([time_col] if time_col else []) + value_cols
    df_plot = _downsample_df(df[cols].copy(), max_points=max_points)
    x_vals = df_plot[time_col] if time_col else np.arange(len(df_plot))
    fig = go.Figure()
    for col in value_cols:
        fig.add_trace(go.Scatter(x=x_vals, y=df_plot[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)" if time_col else "Sample",
        yaxis_title="Value",
        width=900,
        height=350,
    )
    return fig


def _extract_motion_energy_series(df, max_points=5000):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None, None
    time_col = _infer_time_col(df)
    value_col = None
    for candidate in ("whiskerMotionEnergy", "motionEnergy", "energy"):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_cols = _numeric_cols(df, exclude=[time_col] if time_col else None)
        if numeric_cols:
            value_col = numeric_cols[0]
    if value_col is None:
        return None, None, None
    cols = [value_col] if time_col is None else [time_col, value_col]
    df_plot = _downsample_df(df[cols].copy(), max_points=max_points)
    x_vals = df_plot[time_col] if time_col else np.arange(len(df_plot))
    y_vals = df_plot[value_col]
    return x_vals, y_vals, value_col


def _plot_motion_energy_lr(motion_energy, max_points=5000):
    if not isinstance(motion_energy, dict):
        print("motion_energy not available to plot left/right cameras.")
        return None
    left_df = motion_energy.get("leftCamera")
    right_df = motion_energy.get("rightCamera")
    x_left, y_left, col_left = _extract_motion_energy_series(left_df, max_points=max_points)
    x_right, y_right, col_right = _extract_motion_energy_series(
        right_df, max_points=max_points
    )
    if x_left is None or x_right is None:
        print("Motion energy columns not found for one or both cameras.")
        return None
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Left camera ({col_left})",
            f"Right camera ({col_right})",
        ),
    )
    fig.add_trace(go.Scatter(x=x_left, y=y_left, mode="lines", name="Left"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_right, y=y_right, mode="lines", name="Right"), row=2, col=1
    )
    fig.update_yaxes(title_text="Motion energy", row=1, col=1)
    fig.update_yaxes(title_text="Motion energy", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        title="Motion energy (left vs right cameras)",
        width=900,
        height=600,
        showlegend=False,
    )
    return fig


def _preview_motion_energy(motion_energy, max_points=5000):
    if motion_energy is None:
        print("motion_energy: None")
        return
    if isinstance(motion_energy, pd.DataFrame):
        print(
            f"motion_energy: DataFrame shape={motion_energy.shape}, "
            f"columns={list(motion_energy.columns)}"
        )
        display(motion_energy.head(5))
        fig = _plot_df_signal(motion_energy, "Motion energy", max_points=max_points)
        if fig is not None:
            show_fig(fig)
        return
    if isinstance(motion_energy, dict):
        print(f"motion_energy keys: {list(motion_energy.keys())}")
        for view, df in motion_energy.items():
            if df is None:
                print(f"motion_energy[{view}]: None")
                continue
            if not isinstance(df, pd.DataFrame):
                print(f"motion_energy[{view}]: {type(df).__name__}")
                continue
            print(
                f"motion_energy[{view}]: shape={df.shape}, columns={list(df.columns)}"
            )
            if "times" in df.columns:
                print(f"{view} times head: {df['times'].head(5).to_numpy()}")
            if "whiskerMotionEnergy" in df.columns:
                print(
                    f"{view} whiskerMotionEnergy head: "
                    f"{df['whiskerMotionEnergy'].head(5).to_numpy()}"
                )
            display(df.head(5))
            fig = _plot_df_signal(df, f"Motion energy ({view})", max_points=max_points)
            if fig is not None:
                show_fig(fig)
        return
    print(f"motion_energy: {type(motion_energy).__name__}")


def _preview_pupil(pupil, max_points=5000):
    if pupil is None:
        print("pupil: None")
        return
    if isinstance(pupil, pd.DataFrame):
        print(f"pupil: DataFrame shape={pupil.shape}, columns={list(pupil.columns)}")
        display(pupil.head(5))
        fig = _plot_df_signal(pupil, "Pupil", max_points=max_points)
        if fig is not None:
            show_fig(fig)
        return
    if isinstance(pupil, dict):
        print(f"pupil keys: {list(pupil.keys())}")
        for key, df in pupil.items():
            if df is None:
                print(f"pupil[{key}]: None")
                continue
            if not isinstance(df, pd.DataFrame):
                print(f"pupil[{key}]: {type(df).__name__}")
                continue
            print(f"pupil[{key}]: shape={df.shape}, columns={list(df.columns)}")
            display(df.head(5))
            fig = _plot_df_signal(df, f"Pupil ({key})", max_points=max_points)
            if fig is not None:
                show_fig(fig)
        return
    print(f"pupil: {type(pupil).__name__}")


def _availability_label(flag):
    if flag is None:
        return "Unknown"
    return "Yes" if flag else "No"


def _merge_availability(*flags):
    if any(flag is True for flag in flags):
        return True
    if any(flag is None for flag in flags):
        return None
    return False


def _session_has_data(session_obj, name):
    if session_obj is None:
        return None
    obj = _get_session_attr(session_obj, name)
    if obj is None:
        return False
    if isinstance(obj, pd.DataFrame):
        return not obj.empty
    if isinstance(obj, dict):
        return len(obj) > 0
    try:
        return len(obj) > 0
    except TypeError:
        return True


def _normalize_dataset_list(dsets):
    if dsets is None:
        return None
    if isinstance(dsets, pd.DataFrame):
        cols = [
            col
            for col in (
                "rel_path",
                "path",
                "file_name",
                "filename",
                "dataset",
                "name",
                "dataset_type",
            )
            if col in dsets.columns
        ]
        if cols:
            return dsets[cols].astype(str).agg(" ".join, axis=1).tolist()
        return dsets.astype(str).agg(" ".join, axis=1).tolist()
    if isinstance(dsets, (list, tuple, np.ndarray, pd.Index)):
        return [str(item) for item in dsets]
    return [str(dsets)]


def _list_datasets_for_eid(eid_val, allow_remote=True):
    if eid_val is None:
        return None
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    try:
        one_local = init_one(ibl_cache, mode="local")
        dsets = one_local.list_datasets(eid_val, details=True)
        return _normalize_dataset_list(dsets)
    except Exception:
        if not allow_remote:
            return None
        try:
            one_remote = init_one(ibl_cache, mode="remote")
            dsets = one_remote.list_datasets(eid_val, details=True)
            return _normalize_dataset_list(dsets)
        except Exception:
            return None


def _has_dataset_pattern(dsets, patterns):
    if dsets is None:
        return None
    for pattern in patterns:
        for item in dsets:
            if fnmatch.fnmatch(item, pattern) or pattern.strip("*") in item:
                return True
    return False


def _check_passive_rfmap(eid_val, allow_remote=True):
    if eid_val is None:
        return None
    try:
        from brainbox.io.one import load_passive_rfmap
    except Exception:
        return None
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    try:
        one_local = init_one(ibl_cache, mode="local")
        rfmap = load_passive_rfmap(eid_val, one=one_local)
        return rfmap is not None
    except Exception:
        if not allow_remote:
            return False
        try:
            one_remote = init_one(ibl_cache, mode="remote")
            rfmap = load_passive_rfmap(eid_val, one=one_remote)
            return rfmap is not None
        except Exception:
            return False


def _get_plot_config(data, plot_label_min):
    config_plot = dict(data.get("config_plot", {}))
    config_calc = data.get("config_calc", {})
    config_plot["PLOT_ONLY_GOOD_UNITS"] = False
    config_plot["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
    config_plot["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
    config_plot["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
    config_plot["SINGLE_NEURON_SMOOTH_SIGMA"] = 0.5
    config_plot["SINGLE_NEURON_BIN_SIZE"] = 0.03
    config_plot["PLOT_LABEL_MIN"] = plot_label_min
    config_plot["PLOTLY_TEMPLATE"] = "plotly_white"
    config_plot["DELAY_UNITS"] = config_calc.get("DELAY_UNITS", "s")
    return config_plot, config_calc


def _filter_by_label_min(data, plot_label_min):
    df_coupling = data.get("df_coupling")
    df_coupling_task = data.get("df_coupling_task")
    df_coupling_iti = data.get("df_coupling_iti")
    df_comparison = data.get("df_comparison")
    if plot_label_min is None:
        return df_coupling, df_coupling_task, df_coupling_iti, df_comparison
    cluster_ids = data.get("cluster_ids")
    labels = _get_label_array(data.get("clusters"), cluster_ids)
    if labels is None or cluster_ids is None:
        return df_coupling, df_coupling_task, df_coupling_iti, df_comparison
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
    if df_coupling_iti is not None:
        df_coupling_iti = df_coupling_iti[
            df_coupling_iti["cluster_id"].isin(plot_cluster_ids)
        ]
    if df_comparison is not None:
        df_comparison = df_comparison[
            df_comparison["cluster_id"].isin(plot_cluster_ids)
        ]
    return df_coupling, df_coupling_task, df_coupling_iti, df_comparison


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

# PID = "c9664185-d3fd-4e0e-89cf-77c402038938" # The first VISp and ENTm
# PID = "27bac116-ea57-4512-ad35-714a62d259cd" # VISp with passive

## AUD PIDs for Passive Delay
PID = "3282a590-8688-44fc-9811-cdf8b80d9a80" # AUDp responsive to both stim and feedback
# PID = "100433fa-2c59-4432-8295-aa27657fe3fb" # AUDp responsive to both stim and feedback, not as great
# PID = "3282a590-8688-44fc-9811-cdf8b80d9a80" # AUDp responsive to both stim and feedback, not as great
# PID = "1df6ebb3-9d16-4c0f-96cc-e1b1596d2006" # AUDp strongly responsive to first move! Check the sequence here
# PID = "b2ea68e2-c732-4d17-8166-1a8595fff225" # AUDp responsive to both stim and feedback with sustained responses

if PID is None:
    pid = choose_pid(pid_list, default_index=0)
else:
    if PID not in pid_list:
        raise ValueError(f"PID '{PID}' not found in cache list.")
    pid = PID

print(f"Selected PID: {pid}")

data = load_cache(pid)
raw_error = None
raw_error_remote = None
raw_source = None
raw_spikes = None
raw_clusters = None
raw_session = None

spikes = data.get("spikes")
clusters = data.get("clusters")
session = data.get("session")

if LOAD_RAW_DATA or spikes is None or clusters is None or session is None:
    try:
        raw_spikes, raw_clusters, raw_session, _ssl = _load_raw_session(
            pid,
            load_wheel=LOAD_RAW_WHEEL,
            load_pose=LOAD_RAW_POSE,
            load_motion_energy=LOAD_RAW_MOTION_ENERGY,
            load_pupil=LOAD_RAW_PUPIL,
            motion_energy_views=MOTION_ENERGY_VIEWS,
            mode="local",
        )
        raw_source = "local"
    except Exception as exc:
        raw_error = exc
        if ALLOW_REMOTE_METADATA:
            try:
                raw_spikes, raw_clusters, raw_session, _ssl = _load_raw_session(
                    pid,
                    load_wheel=LOAD_RAW_WHEEL,
                    load_pose=LOAD_RAW_POSE,
                    load_motion_energy=LOAD_RAW_MOTION_ENERGY,
                    load_pupil=LOAD_RAW_PUPIL,
                    motion_energy_views=MOTION_ENERGY_VIEWS,
                    mode="remote",
                )
                raw_source = "remote"
            except Exception as exc_remote:
                raw_error_remote = exc_remote

spikes = raw_spikes if raw_spikes is not None else spikes
clusters = raw_clusters if raw_clusters is not None else clusters
session = raw_session if raw_session is not None else session

if raw_spikes is not None:
    data["spikes"] = spikes
if raw_clusters is not None:
    data["clusters"] = clusters
if raw_session is not None:
    data["session"] = session

motion_energy = data.get("motion_energy")
pupil = data.get("pupil")
if motion_energy is None:
    motion_energy = _get_session_attr(raw_session, "motion_energy")
    if motion_energy is None:
        motion_energy = _get_session_attr(session, "motion_energy")
if pupil is None:
    pupil = _get_session_attr(raw_session, "pupil")
    if pupil is None:
        pupil = _get_session_attr(session, "pupil")
if motion_energy is not None:
    data["motion_energy"] = motion_energy
if pupil is not None:
    data["pupil"] = pupil

if (spikes is None or clusters is None or session is None) and raw_error is not None:
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
if raw_error is not None and raw_source is None:
    print(f"Raw load failed, using cached blobs: {raw_error}")
elif raw_error is not None and raw_source is not None:
    print(f"Local load failed; using {raw_source} metadata lookup: {raw_error}")

# Keep cluster metadata aligned with the live clusters object (same behavior as 04_dashboard).
cluster_ids = data.get("cluster_ids")
cluster_acronyms = data.get("cluster_acronyms_plot")
if cluster_ids is None and clusters is not None:
    cluster_ids, _ = build_cluster_id_map(clusters)
if cluster_acronyms is None and clusters is not None:
    if hasattr(clusters, "acronym"):
        cluster_acronyms = np.asarray(clusters.acronym)
    elif isinstance(clusters, dict) and "acronym" in clusters:
        cluster_acronyms = np.asarray(clusters.get("acronym"))
data["clusters"] = clusters
data["cluster_ids"] = cluster_ids
data["cluster_acronyms_plot"] = cluster_acronyms


# %% Quick cache summary
cache_keys = sorted(list(data.keys()))
print(f"Cache keys ({len(cache_keys)}): {cache_keys}")

overview = pd.concat(
    [
        _df_overview(data.get("trials"), "trials"),
        _df_overview(data.get("motion_energy"), "motion_energy"),
        _df_overview(data.get("pupil"), "pupil"),
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
eid = meta.get("eid") if isinstance(meta, dict) else None
if eid is None:
    eid = data.get("eid")

dsets = _list_datasets_for_eid(eid, allow_remote=ALLOW_REMOTE_METADATA)
wheel_available = _merge_availability(
    _session_has_data(session, "wheel"),
    _has_dataset_pattern(dsets, ["*wheel*", "*_ibl_wheel*"]),
)
pose_available = _merge_availability(
    _session_has_data(session, "pose"),
    _has_dataset_pattern(
        dsets,
        [
            "*camera.dlc*",
            "*Camera.dlc*",
            "*Camera*dlc*",
            "*camera*dlc*",
            "*dlc.pqt*",
        ],
    ),
)
motion_energy_available = _merge_availability(
    _session_has_data(session, "motion_energy"),
    _has_dataset_pattern(dsets, ["*motionEnergy*", "*motion_energy*", "*motionenergy*"]),
)
pupil_available = _merge_availability(
    _session_has_data(session, "pupil"),
    _has_dataset_pattern(
        dsets, ["*pupil*", "*Pupil*", "*pupilDiameter*", "*pupil_diameter*"]
    ),
)
task_replay_visual = _has_dataset_pattern(dsets, ["*passiveGabor*"])
task_replay_auditory = _has_dataset_pattern(dsets, ["*passiveStims*"])
rfmap_available = _check_passive_rfmap(eid, allow_remote=ALLOW_REMOTE_METADATA)
passive_event_times = {}
visual_TR = None
auditory_TR = None
passive_events_available = (task_replay_visual is not False) or (
    task_replay_auditory is not False
)
if eid is not None and passive_events_available:
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one_local = init_one(ibl_cache, mode="local")
    one_remote = init_one(ibl_cache, mode="remote") if ALLOW_REMOTE_METADATA else None
    visual_TR, auditory_TR = load_task_replay_datasets(
        eid,
        one_local,
        one_remote,
        allow_remote=ALLOW_REMOTE_METADATA,
    )
    passive_event_times = build_passive_event_times(visual_TR, auditory_TR)

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
    "Wheel data": _availability_label(wheel_available),
    "Pose data": _availability_label(pose_available),
    "Motion energy data": _availability_label(motion_energy_available),
    "Pupil data": _availability_label(pupil_available),
    "Task replay (visual)": _availability_label(task_replay_visual),
    "Task replay (auditory)": _availability_label(task_replay_auditory),
    "Passive RF map": _availability_label(rfmap_available),
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


# %% Motion energy left/right quick access
motion_energy = data.get("motion_energy")
if not isinstance(motion_energy, dict):
    print("motion_energy not available to extract left/right camera tables.")
else:
    a = motion_energy.get("leftCamera")
    b = motion_energy.get("rightCamera")

    if isinstance(a, pd.DataFrame):
        print(f"leftCamera columns: {list(a.columns)}")
        if "times" in a.columns:
            display(a["times"].head(5))
        if "whiskerMotionEnergy" in a.columns:
            display(a["whiskerMotionEnergy"].head(5))
    else:
        print(f"leftCamera: {type(a).__name__}")

    if isinstance(b, pd.DataFrame):
        print(f"rightCamera columns: {list(b.columns)}")
        if "times" in b.columns:
            display(b["times"].head(5))
        if "whiskerMotionEnergy" in b.columns:
            display(b["whiskerMotionEnergy"].head(5))
    else:
        print(f"rightCamera: {type(b).__name__}")


# %% Motion energy left/right plot
fig_motion_lr = _plot_motion_energy_lr(data.get("motion_energy"))
if fig_motion_lr is not None:
    show_fig(fig_motion_lr)


# %% Region counts (all vs good)
labels = _get_label_array(clusters, cluster_ids)
plot_cluster_ids = np.asarray(cluster_ids) if cluster_ids is not None else None
plot_cluster_acronyms = cluster_acronyms
if cluster_acronyms is not None and cluster_ids is not None:
    all_counts = pd.Series(cluster_acronyms).value_counts().sort_index()
    if labels is not None:
        good_counts = pd.Series(cluster_acronyms[labels == 1]).value_counts().sort_index()
    else:
        good_counts = pd.Series(dtype=int)
    region_table = pd.DataFrame(
        {"All Neurons": all_counts, "Good Neurons": good_counts}
    ).fillna(0)
    region_table = region_table.astype(int)
    display(region_table)


# %% Plot configuration
config_calc = data.get("config_calc", {})
calc_label_min = config_calc.get("CALC_LABEL_MIN", None)
if calc_label_min is None and config_calc.get("CALC_ONLY_GOOD_UNITS", False):
    calc_label_min = 1.0
calc_label = "All neurons" if calc_label_min is None else f"Label >= {calc_label_min}"
print(f"Calculations: {calc_label}")

plot_label_min = float(calc_label_min if calc_label_min is not None else 0.5)
variability_metric = "fano"  # "fano" or "cv"
plot_regions = None  # Example: ["VISp", "MOp"] or None for all
trial_idx = 234  # Set an int to override the default trial selection
selected_cluster_id = None  # Set an int to override the default cluster selection

calc_label_min = 1.0 ####################
plot_label_min = 1.0 ####################

calc_label = "All neurons" if calc_label_min is None else f"Label >= {calc_label_min}"
print(f"Calculations: {calc_label}")

use_dark_theme = False  # True -> plotly_dark for all plots
use_good_stpr = True  # True -> use good-neuron stPR from cache if available

plot_config, config_calc = _get_plot_config(data, plot_label_min)
if plot_regions is not None:
    plot_config["PLOT_REGIONS"] = plot_regions

if use_dark_theme:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_dark"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_dark"
    pio.templates.default = "plotly_dark"
else:
    plot_config["PLOTLY_TEMPLATE"] = "plotly_white"
    plotting_utils.DEFAULT_TEMPLATE = "plotly_white"
    pio.templates.default = "plotly_white"

region_colors = (
    _build_region_colors(cluster_acronyms) if cluster_acronyms is not None else None
)

sort_map = {
    "Default (Depth)": "depth",
    "Delay to Stim On": "stim",
    "Delay to First Move": "move",
    "Delay to Response": "response",
    "Delay to Feedback": "feedback",
    "Task stPR Delay": "task",
    "Task stPR Strength": "task_strength",
    "Task stPR Max": "task_max",
    "ITI stPR Delay": "iti",
    "ITI stPR Strength": "iti_strength",
    "ITI stPR Max": "iti_max",
    "Spont stPR Delay": "spont",
    "Spont stPR Strength": "spont_strength",
    "Spont stPR Max": "spont_max",
    "Firing rate": "firing_rate",
}
plot_sort_map = {
    "Default (Depth)": "depth",
    "Own Event Delay": "delay",
    "Task stPR Delay": "task",
    "Task stPR Strength": "task_strength",
    "Task stPR Max": "task_max",
    "ITI stPR Delay": "iti",
    "ITI stPR Strength": "iti_strength",
    "ITI stPR Max": "iti_max",
    "Spont stPR Delay": "spont",
    "Spont stPR Strength": "spont_strength",
    "Spont stPR Max": "spont_max",
    "Firing rate": "firing_rate",
}

general_sort = "Task stPR Delay"  # keys in sort_map
trial_sort = "Task stPR Delay"  # keys in sort_map
population_sort = "Own Event Delay"  # keys in plot_sort_map
sorting_metric = sort_map.get(general_sort, general_sort)
trial_sorting_metric = sort_map.get(trial_sort, trial_sort)
population_sort_mode = plot_sort_map.get(population_sort, population_sort)

df_coupling_good = data.get("df_coupling_good")
df_coupling_task_good = data.get("df_coupling_task_good")
df_coupling_iti_good = data.get("df_coupling_iti_good")
if use_good_stpr:
    missing = []
    if df_coupling_good is None:
        missing.append("Spont")
    if df_coupling_task_good is None:
        missing.append("Task")
    if df_coupling_iti_good is None:
        missing.append("ITI")
    if len(missing) == 3:
        print(
            "Good-neuron stPR not available in cache; using all neurons for stPR metrics."
        )
        use_good_stpr = False
    elif missing:
        print(
            "Good-neuron stPR missing for: "
            + ", ".join(missing)
            + ". Using all neurons for those contexts."
        )

df_coupling_plot = (
    df_coupling_good if use_good_stpr and df_coupling_good is not None else data.get("df_coupling")
)
df_coupling_task_plot = (
    df_coupling_task_good
    if use_good_stpr and df_coupling_task_good is not None
    else data.get("df_coupling_task")
)
df_coupling_iti_plot = (
    df_coupling_iti_good
    if use_good_stpr and df_coupling_iti_good is not None
    else data.get("df_coupling_iti")
)
df_comparison_plot = data.get("df_comparison")

plot_label_values = (
    _label_values_for_clusters(plot_cluster_ids, clusters, labels)
    if plot_cluster_ids is not None
    else None
)
if plot_label_values is not None and plot_label_min is not None:
    plot_mask = plot_label_values >= float(plot_label_min)
    if plot_cluster_ids is not None:
        plot_cluster_ids = plot_cluster_ids[plot_mask]
    if plot_cluster_acronyms is not None:
        plot_cluster_acronyms = np.asarray(plot_cluster_acronyms)[plot_mask]
    plot_label_values = plot_label_values[plot_mask]
    if df_coupling_plot is not None:
        df_coupling_plot = df_coupling_plot[
            df_coupling_plot["cluster_id"].isin(plot_cluster_ids)
        ]
    if df_coupling_task_plot is not None:
        df_coupling_task_plot = df_coupling_task_plot[
            df_coupling_task_plot["cluster_id"].isin(plot_cluster_ids)
        ]
    if df_coupling_iti_plot is not None:
        df_coupling_iti_plot = df_coupling_iti_plot[
            df_coupling_iti_plot["cluster_id"].isin(plot_cluster_ids)
        ]
    if df_comparison_plot is not None:
        df_comparison_plot = df_comparison_plot[
            df_comparison_plot["cluster_id"].isin(plot_cluster_ids)
        ]

if plot_cluster_ids is None:
    plot_cluster_ids = cluster_ids
if plot_cluster_acronyms is None:
    plot_cluster_acronyms = cluster_acronyms

cluster_firing_rate = data.get("cluster_firing_rate")
if cluster_firing_rate is None and clusters is not None:
    cluster_firing_rate = _get_cluster_firing_rate(clusters, cluster_ids)
if cluster_firing_rate is not None:
    cluster_firing_rate = np.asarray(cluster_firing_rate, dtype=float)
    if cluster_ids is not None and len(cluster_firing_rate) != len(cluster_ids):
        cluster_firing_rate = None

df_firing_rate = None
if cluster_firing_rate is not None and cluster_ids is not None:
    df_firing_rate = pd.DataFrame(
        {
            "cluster_id": np.asarray(cluster_ids),
            "firing_rate_h1": cluster_firing_rate,
            "firing_rate_h2": cluster_firing_rate,
        }
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
    t_start = float(min_time)
    t_end = float(min(min_time + 10.0, max_time))

    passive_event_styles = {
        "passive_visual": ("Passive Visual", "#17becf", "dot"),
        "passive_tone": ("Passive Tone", "#bcbd22", "dash"),
        "passive_noise": ("Passive Noise", "#8c564b", "dashdot"),
    }
    fig_general = plot_time_window_raster_plotly(
        spikes,
        clusters,
        plot_cluster_ids,
        plot_cluster_acronyms,
        session,
        plot_config,
        t_start,
        t_end,
        sorting_metric=sorting_metric,
        variability_metric=variability_metric,
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
        df_coupling_iti=df_coupling_iti_plot,
        df_firing_rate=df_firing_rate,
        pupil_features=data.get("pupil_features"),
        pupil_times=data.get("pupil_times"),
        region_colors=region_colors,
        extra_event_times=passive_event_times or None,
        extra_event_styles=passive_event_styles,
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
        plot_cluster_ids,
        plot_cluster_acronyms,
        session,
        plot_config,
        trial_idx,
        variability_metric=variability_metric,
        sorting_metric=trial_sorting_metric,
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
        df_coupling_iti=df_coupling_iti_plot,
        df_firing_rate=df_firing_rate,
        pupil_features=data.get("pupil_features"),
        pupil_times=data.get("pupil_times"),
        region_colors=region_colors,
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
            plot_cluster_ids,
            plot_cluster_acronyms,
            data.get("df_res"),
            cfg,
            df_coupling=df_coupling_plot,
            df_coupling_task=df_coupling_task_plot,
            df_coupling_iti=df_coupling_iti_plot,
            df_firing_rate=df_firing_rate,
            region_acronyms=plot_regions,
            sort_mode=population_sort_mode,
        )
        show_fig(fig_pop)


# %% Coupling heatmaps
if df_coupling_plot is None and df_coupling_task_plot is None and df_coupling_iti_plot is None:
    print("Coupling data missing. Re-run 03_calc_dashboard.py with CALC_SPONT=True.")
else:
    def _clamp_coupling_colorbar(fig, zmin=-2, zmax=2):
        if fig is None:
            return fig
        fig.update_traces(zmin=zmin, zmax=zmax, selector=dict(type="heatmap"))
        return fig

    def _hide_heatmap_colorbars(fig):
        if fig is None:
            return fig
        fig.update_traces(showscale=False, selector=dict(type="heatmap"))
        return fig

    def _set_coupling_title(fig, label):
        if fig is None:
            return fig
        fig.update_layout(title=f"Spike-triggered Population Rate ({label})")
        return fig

    fig_spont = None
    fig_task = None
    fig_iti = None
    if df_coupling_plot is not None:
        fig_spont = plot_population_coupling_heatmap_plotly(
            df_coupling_plot,
            plot_config,
            config_calc,
            region_acronyms=plot_regions,
            zscore_by_region=True,
            colorbar_mode="per_row",
        )
        fig_spont = _set_coupling_title(
            _hide_heatmap_colorbars(_clamp_coupling_colorbar(fig_spont)), "Spont"
        )
    if df_coupling_task_plot is not None:
        fig_task = plot_population_coupling_heatmap_plotly(
            df_coupling_task_plot,
            plot_config,
            config_calc,
            region_acronyms=plot_regions,
            zscore_by_region=True,
            colorbar_mode="per_row",
        )
        fig_task = _set_coupling_title(
            _hide_heatmap_colorbars(_clamp_coupling_colorbar(fig_task)), "Task"
        )
    if df_coupling_iti_plot is not None:
        fig_iti = plot_population_coupling_heatmap_plotly(
            df_coupling_iti_plot,
            plot_config,
            config_calc,
            region_acronyms=plot_regions,
            zscore_by_region=True,
            colorbar_mode="per_row",
        )
        fig_iti = _set_coupling_title(
            _hide_heatmap_colorbars(_clamp_coupling_colorbar(fig_iti)), "ITI"
        )

    if fig_spont is not None:
        show_fig(fig_spont)
    if fig_task is not None:
        show_fig(fig_task)
    if fig_iti is not None:
        show_fig(fig_iti)


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
        region_colors=region_colors,
        template=plot_config["PLOTLY_TEMPLATE"],
        highlight_cluster_id=None,
    )
    fig_delay = plot_coupling_delay_summary_plotly(
        df_comparison_plot,
        region_order=region_order,
        region_colors=region_colors,
        template=plot_config["PLOTLY_TEMPLATE"],
        highlight_cluster_id=None,
    )
    show_fig(fig_strength)
    show_fig(fig_delay)


# %% Single neuron plots
cluster_id = selected_cluster_id or _pick_default_cluster_id(cluster_ids)
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
    if df_coupling_iti_plot is not None:
        fig_iti_curve = plot_stpr_curve_halves_plotly(
            df_coupling_iti_plot,
            config_calc,
            cluster_id,
            title="ITI stPR Curve (Odd vs Even Trials)",
            template=plot_config["PLOTLY_TEMPLATE"],
            split_suffixes=("odd", "even"),
            split_labels=("Odd trials", "Even trials"),
        )
        show_fig(fig_iti_curve)

    if df_coupling_plot is not None or df_coupling_task_plot is not None or df_coupling_iti_plot is not None:
        fig_mean = _plot_stpr_mean_comparison(
            df_coupling_plot,
            df_coupling_task_plot,
            df_coupling_iti_plot,
            config_calc,
            cluster_id,
            plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_mean)


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


# %% Binned reliability + thresholded correlation (per region)
BIN_RUN = True
BIN_LABEL_MIN = plot_label_min
BIN_REGION = None  # e.g. "VISp" (prefix ok), or "ALL" for all regions

# Variable to bin by (defaults to stPR task delay).
BIN_VARIABLE_NAME = "stPR Delay (Task)"

# Quantile boundaries for equal-N bins.
BIN_QUANTILES = (1 / 3, 2 / 3)

# Variable to correlate using the same thresholds (defaults to stPR spont delay).
BIN_CORR_VARIABLE_NAME = "stPR Delay (Spont)"


def _format_range_text(cut1, cut2, label):
    if not (np.isfinite(cut1) and np.isfinite(cut2)):
        return "NA"
    if label == "low":
        return f"<= {cut1:.3f}"
    if label == "mid":
        return f"({cut1:.3f}, {cut2:.3f}]"
    return f"> {cut2:.3f}"


def _add_unity_line(fig, x_vals, y_vals):
    if len(x_vals) == 0 or len(y_vals) == 0:
        return
    min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
    max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        return
    if min_val == max_val:
        return
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash"),
    )


if BIN_RUN:
    region_lookup = _build_region_lookup(data, BIN_LABEL_MIN)
    if region_lookup.empty:
        print("No regions available for binned analysis.")
    else:
        regions_all = sorted(region_lookup["region"].unique().tolist())
        print("Available regions:", ", ".join(regions_all))

        if BIN_REGION is None or str(BIN_REGION).upper() == "ALL":
            region_lookup_sel = region_lookup.copy()
            region_label = "All regions"
        else:
            region_prefix = str(BIN_REGION)
            region_mask = region_lookup["region"].astype(str).str.startswith(region_prefix)
            region_lookup_sel = region_lookup[region_mask].copy()
            region_label = f"Region {region_prefix}"

        if region_lookup_sel.empty:
            print(f"No units found for region filter: {BIN_REGION}")
        else:
            available_specs = []
            for spec in CORR_VARIABLES:
                df_src = data.get(spec["df"])
                if df_src is None:
                    continue
                if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
                    continue
                available_specs.append(spec)

            spec_by_name = {spec["name"]: spec for spec in available_specs}
            if not spec_by_name:
                print("No variables available for binned analysis.")
            elif BIN_VARIABLE_NAME not in spec_by_name:
                print("BIN_VARIABLE_NAME not found. Available options:", ", ".join(spec_by_name))
            else:
                spec_a = spec_by_name[BIN_VARIABLE_NAME]
                df_a = _build_variable_table(data, spec_a, region_lookup_sel)
                if df_a is None or df_a.empty:
                    print(f"No data for {BIN_VARIABLE_NAME} in {region_label}.")
                else:
                    v1_a = df_a[spec_a["v1"]].to_numpy(dtype=float)
                    v2_a = df_a[spec_a["v2"]].to_numpy(dtype=float)
                    mean_a = df_a["mean"].to_numpy(dtype=float)
                    valid_a = np.isfinite(v1_a) & np.isfinite(v2_a)
                    v1_a = v1_a[valid_a]
                    v2_a = v2_a[valid_a]
                    mean_a = mean_a[valid_a]

                    if mean_a.size == 0:
                        print(f"No finite values for {BIN_VARIABLE_NAME} in {region_label}.")
                    else:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(
                            go.Histogram(
                                x=v1_a,
                                name=f"{spec_a['v1']} (half 1)",
                                opacity=0.65,
                            )
                        )
                        fig_hist.add_trace(
                            go.Histogram(
                                x=v2_a,
                                name=f"{spec_a['v2']} (half 2)",
                                opacity=0.65,
                            )
                        )
                        fig_hist.update_layout(
                            title=f"{BIN_VARIABLE_NAME} histogram (halves) | {region_label}",
                            barmode="overlay",
                            width=900,
                            height=520,
                            margin=dict(l=70, r=40, t=80, b=60),
                            template=plot_config["PLOTLY_TEMPLATE"],
                        )

                        q1, q2 = BIN_QUANTILES
                        if not (0 < q1 < q2 < 1):
                            raise ValueError("BIN_QUANTILES must satisfy 0 < q1 < q2 < 1.")
                        cut1, cut2 = np.nanquantile(mean_a, [q1, q2])
                        cut1 = float(cut1)
                        cut2 = float(cut2)
                        print(
                            f"Quantile thresholds for {BIN_VARIABLE_NAME}: "
                            f"{cut1:.3f} (q={q1:.2f}), {cut2:.3f} (q={q2:.2f})"
                        )

                        if np.isfinite(cut1) and np.isfinite(cut2) and cut1 > cut2:
                            cut1, cut2 = cut2, cut1

                        if not (np.isfinite(cut1) and np.isfinite(cut2)):
                            print("Thresholds are not finite; skipping bin analysis.")
                        else:
                            fig_hist.add_vline(
                                x=cut1,
                                line=dict(color="gray", dash="dot"),
                            )
                            fig_hist.add_vline(
                                x=cut2,
                                line=dict(color="gray", dash="dot"),
                            )
                            show_fig(fig_hist)

                            bins_a = [
                                ("low", mean_a <= cut1),
                                ("mid", (mean_a > cut1) & (mean_a <= cut2)),
                                ("high", mean_a > cut2),
                            ]

                            rows = []
                            for label, mask in bins_a:
                                r_val, n_val = _pearsonr_with_n(v1_a[mask], v2_a[mask])
                                rho_val, n_s = _spearmanr_with_n(v1_a[mask], v2_a[mask])
                                rows.append(
                                    {
                                        "bin": label,
                                        "range": _format_range_text(cut1, cut2, label),
                                        "n": n_val,
                                        "pearson_r": r_val,
                                        "spearman_rho": rho_val,
                                    }
                                )
                            display(pd.DataFrame(rows))

                            if BIN_CORR_VARIABLE_NAME not in spec_by_name:
                                print(
                                    "BIN_CORR_VARIABLE_NAME not found. Available options: "
                                    + ", ".join(spec_by_name)
                                )
                            else:
                                spec_b = spec_by_name[BIN_CORR_VARIABLE_NAME]
                                df_b = _build_variable_table(data, spec_b, region_lookup_sel)
                                if df_b is None or df_b.empty:
                                    print(
                                        f"No data for {BIN_CORR_VARIABLE_NAME} in {region_label}."
                                    )
                                else:
                                    merged = (
                                        df_a[["cluster_id", "mean"]]
                                        .rename(columns={"mean": "mean_a"})
                                        .merge(
                                            df_b[["cluster_id", "mean"]].rename(
                                                columns={"mean": "mean_b"}
                                            ),
                                            on="cluster_id",
                                            how="inner",
                                        )
                                    )
                                    if merged.empty:
                                        print(
                                            "No overlapping units between "
                                            f"{BIN_VARIABLE_NAME} and {BIN_CORR_VARIABLE_NAME}."
                                        )
                                    else:
                                        x_vals = merged["mean_a"].to_numpy(dtype=float)
                                        y_vals = merged["mean_b"].to_numpy(dtype=float)
                                        valid_xy = np.isfinite(x_vals) & np.isfinite(y_vals)
                                        x_vals = x_vals[valid_xy]
                                        y_vals = y_vals[valid_xy]

                                        if x_vals.size == 0:
                                            print("No finite mean values to plot correlation.")
                                        else:
                                            bins_b = [
                                                ("low", y_vals <= cut1),
                                                ("mid", (y_vals > cut1) & (y_vals <= cut2)),
                                                ("high", y_vals > cut2),
                                            ]
                                            colors = {
                                                "low": "#1f77b4",
                                                "mid": "#ff7f0e",
                                                "high": "#2ca02c",
                                            }

                                            corr_rows = []
                                            for label, mask in bins_b:
                                                r_val, n_val = _pearsonr_with_n(
                                                    x_vals[mask], y_vals[mask]
                                                )
                                                rho_val, n_s = _spearmanr_with_n(
                                                    x_vals[mask], y_vals[mask]
                                                )
                                                corr_rows.append(
                                                    {
                                                        "bin": label,
                                                        "range": _format_range_text(
                                                            cut1, cut2, label
                                                        ),
                                                        "n": n_val,
                                                        "pearson_r": r_val,
                                                        "spearman_rho": rho_val,
                                                    }
                                                )
                                            display(pd.DataFrame(corr_rows))

                                            r_all, n_all = _pearsonr_with_n(x_vals, y_vals)
                                            rho_all, n_all_s = _spearmanr_with_n(
                                                x_vals, y_vals
                                            )

                                            fig_corr = go.Figure()
                                            for label, mask in bins_b:
                                                if not np.any(mask):
                                                    continue
                                                fig_corr.add_trace(
                                                    go.Scatter(
                                                        x=x_vals[mask],
                                                        y=y_vals[mask],
                                                        mode="markers",
                                                        name=f"{label} bin",
                                                        marker=dict(
                                                            size=7,
                                                            opacity=0.7,
                                                            color=colors.get(label),
                                                        ),
                                                    )
                                                )

                                            if np.isfinite(cut1):
                                                fig_corr.add_hline(
                                                    y=cut1,
                                                    line=dict(color="gray", dash="dot"),
                                                )
                                            if np.isfinite(cut2):
                                                fig_corr.add_hline(
                                                    y=cut2,
                                                    line=dict(color="gray", dash="dot"),
                                                )

                                            _add_unity_line(fig_corr, x_vals, y_vals)
                                            fig_corr.update_layout(
                                                title=(
                                                    f"{BIN_VARIABLE_NAME} vs {BIN_CORR_VARIABLE_NAME} | "
                                                    f"Pearson r={r_all:.2f} (n={n_all}) | "
                                                    f"Spearman rho={rho_all:.2f} (n={n_all_s}) | "
                                                    f"{region_label}"
                                                ),
                                                xaxis_title=f"{BIN_VARIABLE_NAME} mean",
                                                yaxis_title=f"{BIN_CORR_VARIABLE_NAME} mean",
                                                width=900,
                                                height=620,
                                                margin=dict(l=70, r=40, t=90, b=70),
                                                template=plot_config["PLOTLY_TEMPLATE"],
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="left",
                                                    x=0,
                                                ),
                                            )
                                            show_fig(fig_corr)


# %% Task replay (passive) datasets
eid = meta.get("eid") if isinstance(meta, dict) else None
if eid is None:
    eid = data.get("eid")

if eid is None:
    print("EID not found in cache metadata. Cannot load task replay datasets.")
else:
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one_local = init_one(ibl_cache, mode="local")
    one_remote = init_one(ibl_cache, mode="remote") if ALLOW_REMOTE_METADATA else None
    if visual_TR is None or auditory_TR is None:
        visual_TR, auditory_TR = load_task_replay_datasets(
            eid,
            one_local,
            one_remote,
            allow_remote=ALLOW_REMOTE_METADATA,
        )

    def _describe_tr_data(obj, label):
        if obj is None:
            print(f"{label}: None")
            return
        print(f"{label}: type={type(obj)}")
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"{label}: keys={keys}")
            for key in keys:
                try:
                    arr = np.asarray(obj[key])
                    print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
                except Exception:
                    print(f"  {key}: type={type(obj[key])}")
            return
        if isinstance(obj, (list, tuple)):
            print(f"{label}: len={len(obj)}")
            for idx, item in enumerate(obj[:5]):
                try:
                    arr = np.asarray(item)
                    print(f"  [{idx}]: shape={arr.shape} dtype={arr.dtype}")
                except Exception:
                    print(f"  [{idx}]: type={type(item)}")
            if len(obj) > 5:
                print("  ...")
            return
        try:
            arr = np.asarray(obj)
            print(f"{label}: shape={arr.shape} dtype={arr.dtype}")
        except Exception:
            pass

    _describe_tr_data(visual_TR, "visual_TR")
    _describe_tr_data(auditory_TR, "auditory_TR")


# %% Task replay (passive) population analysis
PASSIVE_SOURCE = "visual_TR"  # or "auditory_TR"
passive_population_sort = "Spont stPR Delay" # "Own Event Delay"  # keys in plot_sort_map
passive_population_sort_mode = plot_sort_map.get(
    passive_population_sort, passive_population_sort
)
visual_TR = globals().get("visual_TR", None)
auditory_TR = globals().get("auditory_TR", None)
df_res_noise = None
noise_event_col = None
noise_delay_units = None


def _extract_tr_field(tr_obj, keys, suffixes=None):
    if tr_obj is None:
        return None
    if hasattr(tr_obj, "keys"):
        key_list = list(tr_obj.keys())
        for key in keys:
            if key in tr_obj:
                return np.asarray(tr_obj[key])
        if suffixes:
            for key in key_list:
                key_str = str(key)
                for suffix in suffixes:
                    if key_str.endswith(suffix):
                        return np.asarray(tr_obj[key])
    for key in keys:
        if hasattr(tr_obj, key):
            return np.asarray(getattr(tr_obj, key))
    if suffixes:
        for suffix in suffixes:
            if hasattr(tr_obj, suffix):
                return np.asarray(getattr(tr_obj, suffix))
    return None


def _extract_passive_times_and_contrast(tr_obj):
    if tr_obj is None:
        return None, None

    base_obj = tr_obj
    if isinstance(tr_obj, dict) and "table" in tr_obj:
        base_obj = tr_obj["table"]
    elif isinstance(tr_obj, (list, tuple)):
        if len(tr_obj) == 1:
            base_obj = tr_obj[0]
        else:
            base_obj = None
            for item in tr_obj:
                if isinstance(item, pd.DataFrame) and {"start", "contrast"}.issubset(
                    item.columns
                ):
                    base_obj = item
                    break
                if hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    if {"start", "contrast"}.issubset(set(item.dtype.names)):
                        base_obj = item
                        break
            if base_obj is None and len(tr_obj) > 0:
                base_obj = tr_obj[0]

    if isinstance(base_obj, pd.DataFrame):
        if "start" in base_obj.columns:
            times = base_obj["start"].to_numpy(dtype=float)
            contrasts = (
                base_obj["contrast"].to_numpy(dtype=float)
                if "contrast" in base_obj.columns
                else np.ones_like(times, dtype=float)
            )
            return times, contrasts
    if hasattr(base_obj, "dtype") and getattr(base_obj.dtype, "names", None):
        names = set(base_obj.dtype.names)
        if "start" in names:
            times = np.asarray(base_obj["start"], dtype=float)
            if "contrast" in names:
                contrasts = np.asarray(base_obj["contrast"], dtype=float)
            else:
                contrasts = np.ones_like(times, dtype=float)
            return times, contrasts

    arr_candidate = np.asarray(base_obj)
    if arr_candidate.ndim == 2 and arr_candidate.shape[1] >= 5:
        times = np.asarray(arr_candidate[:, 0], dtype=float)
        contrasts = np.asarray(arr_candidate[:, 3], dtype=float)
        return times, contrasts

    times = _extract_tr_field(
        base_obj,
        keys=("stimOn_times", "times", "stimOn", "onset_times", "event_times", "start"),
        suffixes=(".times", "times", ".start", "start"),
    )
    if times is None and isinstance(base_obj, (list, tuple, np.ndarray)):
        arr = np.asarray(base_obj)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            times = arr

    contrasts = _extract_tr_field(
        base_obj,
        keys=("contrast", "contrasts", "stimContrast"),
        suffixes=(".contrast", "contrast"),
    )
    if contrasts is None:
        contrast_left = _extract_tr_field(
            base_obj,
            keys=("contrastLeft",),
            suffixes=(".contrastLeft", "contrastLeft"),
        )
        contrast_right = _extract_tr_field(
            base_obj,
            keys=("contrastRight",),
            suffixes=(".contrastRight", "contrastRight"),
        )
        if contrast_left is not None or contrast_right is not None:
            if contrast_left is None:
                contrast_left = np.full_like(contrast_right, np.nan, dtype=float)
            if contrast_right is None:
                contrast_right = np.full_like(contrast_left, np.nan, dtype=float)
            contrasts = np.nanmax(
                np.vstack([np.abs(contrast_left), np.abs(contrast_right)]), axis=0
            )

    if times is None:
        return None, None

    times = np.asarray(times, dtype=float)
    if contrasts is None:
        contrasts = np.ones_like(times, dtype=float)
    else:
        contrasts = np.asarray(contrasts, dtype=float)
        if contrasts.shape[0] != times.shape[0]:
            if contrasts.size == 1:
                contrasts = np.full_like(times, float(contrasts.ravel()[0]))
            else:
                contrasts = np.ones_like(times, dtype=float)

    finite_mask = np.isfinite(times)
    if not np.all(finite_mask):
        times = times[finite_mask]
        contrasts = contrasts[finite_mask]

    return times, contrasts


passive_source_obj = visual_TR if PASSIVE_SOURCE == "visual_TR" else auditory_TR
passive_times, passive_contrasts = _extract_passive_times_and_contrast(passive_source_obj)

if passive_times is None or len(passive_times) == 0:
    print("Passive replay times not found; skipping passive population analysis.")
else:
    if passive_contrasts is not None:
        contrasts_arr = np.asarray(passive_contrasts, dtype=float)
        times_arr = np.asarray(passive_times, dtype=float)
        valid_mask = np.isfinite(times_arr) & np.isfinite(contrasts_arr) & (contrasts_arr > 0)
        passive_times = times_arr[valid_mask]
        passive_contrasts = contrasts_arr[valid_mask]

    if passive_times is None or len(passive_times) == 0:
        print("Passive replay times not found after filtering 0% contrast; skipping.")
        skip_passive = True
    else:
        skip_passive = False
        print(f"Passive replay events ({PASSIVE_SOURCE}, contrast>0): n={len(passive_times)}")

        passive_trials = {"stimOn_times": passive_times}
        passive_session = {"trials": passive_trials}
        events_by_name = {"stimOn_times": passive_times}
        contrasts_by_name = {"stimOn_times": passive_contrasts}

    if not skip_passive:
        _, cid_to_idx = build_cluster_id_map(clusters)
        passive_calc_config = dict(config_calc) if isinstance(config_calc, dict) else {}
        passive_calc_config.setdefault("DELAY_METHOD", "center_of_mass")
        passive_calc_config.setdefault("DELAY_UNITS", plot_config.get("DELAY_UNITS", "s"))
        passive_calc_config.setdefault("FULL_CONTRAST_VALUES", (1.0, 100.0))
        passive_calc_config.setdefault("BIN_SIZE", 0.005)
        passive_calc_config.setdefault("BASELINE_PRE", 0.2)
        passive_calc_config.setdefault("PSTH_WINDOW_START", -1.0)
        passive_calc_config.setdefault("PSTH_WINDOW_END", 1.0)
        passive_calc_config.setdefault("RESPONSIVE_WINDOW_START", 0.02)
        passive_calc_config.setdefault("RESPONSIVE_WINDOW_END", 0.35)
        passive_calc_config.setdefault("SMOOTH_SIGMA", 1)
        passive_calc_config.setdefault("MIN_TRIALS", 50)
        passive_calc_config.setdefault("MIN_TRIALS_SPLIT", 25)
        passive_calc_config.setdefault("CALC_LABEL_MIN", None)
        passive_calc_config.setdefault("CALC_ONLY_GOOD_UNITS", False)
        passive_calc_config["EVENT_NAMES"] = ["stimOn_times"]
        if not isinstance(passive_calc_config.get("DELAY_WINDOWS"), dict):
            passive_calc_config["DELAY_WINDOWS"] = {}
        if "stimOn_times" not in passive_calc_config["DELAY_WINDOWS"]:
            passive_calc_config["DELAY_WINDOWS"]["stimOn_times"] = (0.02, 0.35)

        _, _, path_data_processed, _ = setup_paths(BASE_PATH)
        passive_out_dir = path_data_processed / "passive_replay"
        passive_out_dir.mkdir(parents=True, exist_ok=True)

        df_res_passive = ana_utils.calculate_delays(
            spikes,
            clusters,
            cluster_acronyms,
            events_by_name,
            contrasts_by_name,
            passive_calc_config,
            passive_out_dir,
            pid,
            cid_to_idx,
        )

        if df_res_passive is not None and not df_res_passive.empty:
            if str(passive_calc_config.get("DELAY_UNITS", "s")).lower().startswith("ms"):
                delay_cols = [
                    ana_utils.delay_column_name("stimOn_times"),
                    ana_utils.delay_split_column_name("stimOn_times", "odd"),
                    ana_utils.delay_split_column_name("stimOn_times", "even"),
                ]
                for col in delay_cols:
                    if col in df_res_passive.columns:
                        df_res_passive[col] = df_res_passive[col].astype(float) * 1000.0

        if df_res_passive is None or df_res_passive.empty:
            print("Passive delay results empty; skipping passive population plot.")
        else:
            cfg_passive = dict(plot_config)
            cfg_passive["PLOT_EVENT"] = "stimOn_times"
            fig_passive = plot_population_sorted_plotly(
                passive_session,
                spikes,
                clusters,
                plot_cluster_ids,
                plot_cluster_acronyms,
                df_res_passive,
                cfg_passive,
                df_coupling=df_coupling_plot,
                df_coupling_task=df_coupling_task_plot,
                df_coupling_iti=df_coupling_iti_plot,
                df_firing_rate=df_firing_rate,
                region_acronyms=plot_regions,
                sort_mode=passive_population_sort_mode,
            )
            fig_passive.update_layout(
                title=f"Passive Replay Population PSTH Heatmaps ({PASSIVE_SOURCE}) | Align: Stim On"
            )
            show_fig(fig_passive)


# %% Task replay (passive) tones/noise population analysis
auditory_TR = globals().get("auditory_TR", None)


def _coerce_passive_stims_table(tr_obj):
    if tr_obj is None:
        return None
    base_obj = tr_obj
    if isinstance(tr_obj, dict) and "table" in tr_obj:
        base_obj = tr_obj["table"]
    elif isinstance(tr_obj, (list, tuple)):
        if len(tr_obj) == 1:
            base_obj = tr_obj[0]
        else:
            base_obj = None
            for item in tr_obj:
                if isinstance(item, pd.DataFrame) and {"toneOn", "noiseOn"}.issubset(
                    item.columns
                ):
                    base_obj = item
                    break
                if hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    if {"toneOn", "noiseOn"}.issubset(set(item.dtype.names)):
                        base_obj = item
                        break
            if base_obj is None and len(tr_obj) > 0:
                base_obj = tr_obj[0]

    if isinstance(base_obj, pd.DataFrame):
        return base_obj.copy()

    if isinstance(base_obj, dict):
        try:
            return pd.DataFrame(base_obj)
        except Exception:
            return None

    if hasattr(base_obj, "dtype") and getattr(base_obj.dtype, "names", None):
        try:
            return pd.DataFrame(
                {name: np.asarray(base_obj[name]) for name in base_obj.dtype.names}
            )
        except Exception:
            return None

    arr = np.asarray(base_obj)
    if arr.ndim == 2 and arr.shape[1] >= 6:
        columns = ["valveOn", "valveOff", "toneOn", "toneOff", "noiseOn", "noiseOff"]
        return pd.DataFrame(arr[:, :6], columns=columns)
    return None


def _extract_passive_stim_times(table, column_name):
    if table is None or not hasattr(table, "columns"):
        return None, None, None
    if column_name not in table.columns:
        match = None
        for col in table.columns:
            if str(col).endswith(column_name):
                match = col
                break
        if match is None:
            return None, None, None
        column_name = match
    times_all = np.asarray(table[column_name], dtype=float)
    valid_mask = np.isfinite(times_all)
    return times_all[valid_mask], np.nonzero(valid_mask)[0], column_name


passive_stims_table = _coerce_passive_stims_table(auditory_TR)

if passive_stims_table is None:
    print("Passive stims table not found in auditory_TR; skipping tone/noise analysis.")
else:
    stim_specs = [("Tone", "toneOn"), ("Noise", "noiseOn")]
    for stim_label, stim_col in stim_specs:
        stim_times, stim_trial_idx, stim_col = _extract_passive_stim_times(
            passive_stims_table, stim_col
        )
        if stim_times is None or len(stim_times) == 0:
            print(f"{stim_label}: no valid {stim_col} times found; skipping.")
            continue

        print(f"{stim_label} events: n={len(stim_times)}")

        stim_trials = {stim_col: stim_times}
        stim_session = {"trials": stim_trials}
        events_by_name = {stim_col: stim_times}
        contrasts_by_name = {stim_col: np.ones_like(stim_times, dtype=float)}
        trial_idx_by_name = {stim_col: stim_trial_idx}

        _, cid_to_idx = build_cluster_id_map(clusters)
        stim_calc_config = dict(config_calc) if isinstance(config_calc, dict) else {}
        # Force auditory-specific delay settings (override cached defaults).
        stim_calc_config["DELAY_METHOD"] = "psth_peak"
        stim_calc_config["DELAY_UNITS"] = plot_config.get("DELAY_UNITS", "s")
        stim_calc_config["FULL_CONTRAST_VALUES"] = (1.0, 100.0)
        stim_calc_config["BIN_SIZE"] = 0.005
        stim_calc_config["BASELINE_PRE"] = 0.2
        stim_calc_config["PSTH_WINDOW_START"] = -1.0
        stim_calc_config["PSTH_WINDOW_END"] = 1.0
        stim_calc_config["RESPONSIVE_WINDOW_START"] = 0.0
        stim_calc_config["RESPONSIVE_WINDOW_END"] = 0.15
        stim_calc_config["SMOOTH_SIGMA"] = 1
        stim_calc_config["MIN_TRIALS"] = 20
        stim_calc_config["MIN_TRIALS_SPLIT"] = 10
        stim_calc_config["CALC_LABEL_MIN"] = None
        stim_calc_config["CALC_ONLY_GOOD_UNITS"] = False
        stim_calc_config["EVENT_NAMES"] = [stim_col]
        if not isinstance(stim_calc_config.get("DELAY_WINDOWS"), dict):
            stim_calc_config["DELAY_WINDOWS"] = {}
        if stim_col not in stim_calc_config["DELAY_WINDOWS"]:
            stim_calc_config["DELAY_WINDOWS"][stim_col] = (0.0, 0.15)

        _, _, path_data_processed, _ = setup_paths(BASE_PATH)
        passive_out_dir = path_data_processed / "passive_stims"
        passive_out_dir.mkdir(parents=True, exist_ok=True)

        df_res_stim = ana_utils.calculate_delays(
            spikes,
            clusters,
            cluster_acronyms,
            events_by_name,
            contrasts_by_name,
            stim_calc_config,
            passive_out_dir,
            pid,
            cid_to_idx,
            trial_idx_by_name=trial_idx_by_name,
        )

        delay_units_label = (
            "ms"
            if str(stim_calc_config.get("DELAY_UNITS", "s")).lower().startswith("ms")
            else "s"
        )
        if df_res_stim is not None and not df_res_stim.empty:
            if delay_units_label == "ms":
                delay_cols = [
                    ana_utils.delay_column_name(stim_col),
                    ana_utils.delay_split_column_name(stim_col, "odd"),
                    ana_utils.delay_split_column_name(stim_col, "even"),
                ]
                for col in delay_cols:
                    if col in df_res_stim.columns:
                        df_res_stim[col] = df_res_stim[col].astype(float) * 1000.0

        if df_res_stim is None or df_res_stim.empty:
            print(f"{stim_label} delay results empty; skipping population plot.")
            continue

        if stim_label == "Noise":
            df_res_noise = df_res_stim.copy()
            noise_event_col = stim_col
            noise_delay_units = delay_units_label

        cfg_stim = dict(plot_config)
        cfg_stim["PLOT_EVENT"] = stim_col
        fig_stim = plot_population_sorted_plotly(
            stim_session,
            spikes,
            clusters,
            plot_cluster_ids,
            plot_cluster_acronyms,
            df_res_stim,
            cfg_stim,
            df_coupling=df_coupling_plot,
            df_coupling_task=df_coupling_task_plot,
            df_coupling_iti=df_coupling_iti_plot,
            df_firing_rate=df_firing_rate,
            region_acronyms=plot_regions,
            sort_mode=passive_population_sort_mode,
        )
        fig_stim.update_layout(
            title=f"Passive Stim Population PSTH Heatmaps ({stim_label}) | Align: {stim_col}"
        )
        show_fig(fig_stim)


# %% stPR Spont delay vs noise delay correlation (AUDp/AUDd/AUDv)
target_regions = {"AUDp", "AUDd", "AUDv"}

_corr_min_n_local = globals().get("CORR_MIN_N", 2)


def _pearsonr_with_n_local(x, y, min_n=_corr_min_n_local):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


def _spearmanr_with_n_local(x, y, min_n=_corr_min_n_local):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if "spearmanr" in globals() and spearmanr is not None:
        res = spearmanr(x, y)
        return float(res.correlation), n
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan, n
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), n


def _add_unity_line_local(fig, x_vals, y_vals):
    if len(x_vals) == 0 or len(y_vals) == 0:
        return
    min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
    max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        return
    if min_val == max_val:
        return
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash"),
    )


if df_res_noise is None or df_res_noise.empty:
    print("Noise delay results not available; skipping correlation plot.")
elif df_coupling_plot is None or df_coupling_plot.empty:
    print("stPR coupling table not available; skipping correlation plot.")
else:
    _, cid_to_idx = build_cluster_id_map(clusters)
    if cluster_acronyms is None or len(cluster_acronyms) == 0:
        print("Cluster acronyms not available; cannot filter regions.")
    else:
        cluster_acronyms_arr = np.asarray(cluster_acronyms).astype(str)
        region_lookup = {
            int(cid): cluster_acronyms_arr[idx]
            for cid, idx in cid_to_idx.items()
            if idx < len(cluster_acronyms_arr)
        }

        event_col = noise_event_col or "noiseOn"
        delay_odd_col = ana_utils.delay_split_column_name(event_col, "odd")
        delay_even_col = ana_utils.delay_split_column_name(event_col, "even")
        base_cols = ["cluster_id"]
        if delay_odd_col in df_res_noise.columns:
            base_cols.append(delay_odd_col)
        if delay_even_col in df_res_noise.columns:
            base_cols.append(delay_even_col)
        noise_df = df_res_noise[base_cols].copy()

        if delay_odd_col in noise_df.columns and delay_even_col in noise_df.columns:
            noise_df["noise_delay"] = np.nanmean(
                noise_df[[delay_odd_col, delay_even_col]].to_numpy(dtype=float),
                axis=1,
            )
        else:
            fallback_col = ana_utils.delay_column_name(event_col)
            if fallback_col in df_res_noise.columns:
                noise_df["noise_delay"] = df_res_noise[fallback_col].astype(float)
                print(
                    "Warning: odd/even noise delay columns missing; using overall delay."
                )
            else:
                print("Noise delay columns missing; skipping correlation plot.")
                noise_df["noise_delay"] = np.nan

        if str(noise_delay_units or "s").lower().startswith("ms"):
            noise_df["noise_delay_ms"] = noise_df["noise_delay"]
        else:
            noise_df["noise_delay_ms"] = noise_df["noise_delay"] * 1000.0

        stpr_df = df_coupling_plot[["cluster_id", "coupling_delay_ms"]].copy()
        merged = stpr_df.merge(
            noise_df[["cluster_id", "noise_delay_ms"]], on="cluster_id", how="inner"
        )
        merged["region"] = merged["cluster_id"].map(region_lookup)
        merged = merged[merged["region"].isin(target_regions)]

        x_vals = merged["coupling_delay_ms"].to_numpy(dtype=float)
        y_vals = merged["noise_delay_ms"].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]

        r_val, n_val = _pearsonr_with_n_local(x_vals, y_vals)
        rho_val, n_s = _spearmanr_with_n_local(x_vals, y_vals)

        def _fmt_corr(val):
            return "NA" if not np.isfinite(val) else f"{val:.2f}"

        fig_corr = go.Figure()
        fig_corr.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=7, opacity=0.75, color="#1f77b4"),
                name="Units",
            )
        )
        _add_unity_line_local(fig_corr, x_vals, y_vals)
        fig_corr.update_layout(
            title=(
                "Spont stPR Delay vs Noise Delay (AUDp/AUDd/AUDv) | "
                f"Pearson r={_fmt_corr(r_val)} (n={n_val}) | "
                f"Spearman rho={_fmt_corr(rho_val)} (n={n_s})"
            ),
            xaxis_title="stPR Spont Delay (ms)",
            yaxis_title="Noise Delay (ms)",
            width=800,
            height=600,
            margin=dict(l=70, r=40, t=90, b=70),
            template=plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_corr)

# %%
a = 2