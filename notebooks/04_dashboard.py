# %% 
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.plotting_plotly import (
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
    plot_population_coupling_heatmap_plotly,
    plot_single_neuron_plotly,
    plot_single_neuron_conditioned_event_plotly,
    plot_stpr_curve_halves_plotly,
)
import utils.plotting_plotly as plotting_utils
from utils.io import setup_paths, init_one, load_session_data, build_cluster_id_map

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None


st.set_page_config(page_title="Neuron Dashboard", layout="wide")

BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"


def _list_pids(cache_dir):
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.pkl")])


@st.cache_data(show_spinner=False)
def _load_cache(pid):
    path = CACHE_DIR / f"{pid}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def _get_one(mode):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    return init_one(ibl_cache, mode=mode)


@st.cache_resource(show_spinner=False)
def _load_raw_session(pid, load_wheel, load_pose, mode):
    one = _get_one(mode)
    ssl, spikes, clusters, sl = load_session_data(
        pid,
        one,
        load_wheel=load_wheel,
        load_pose=load_pose,
    )
    return spikes, clusters, sl, ssl


def _get_label_array(clusters, cluster_ids=None):
    if cluster_ids is not None:
        values = plotting_utils._get_label_values(clusters, cluster_ids)
        if values is not None:
            return values
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            return np.asarray(clusters.metrics.label)
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


def _plot_stpr_mean_comparison(df_spont, df_task, df_iti, config_calc, cluster_id, template):
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
            go.Scatter(x=lags, y=curve, mode="lines", line=dict(color=color, width=2), name=label)
        )
        delays.append((delay, color))
        added = True

    if not added:
        fig.add_annotation(text="No stPR mean curves available", showarrow=False)

    for delay, color in delays:
        if np.isfinite(delay):
            fig.add_vline(x=delay, line=dict(color=color, dash="dash"))

    fig.add_vline(x=0, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title="stPR Mean Curves (Task vs Spont vs ITI)",
        xaxis_title="Lag (ms)",
        yaxis_title="stPR (z)",
        template=template,
        width=900,
        height=550,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _get_cached_value(state_key, key, builder):
    cache_key = f"{state_key}_key"
    value_key = f"{state_key}_value"
    if st.session_state.get(cache_key) != key:
        st.session_state[cache_key] = key
        st.session_state[value_key] = builder()
    return st.session_state.get(value_key)


CORR_MIN_N = 2
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


def _pearsonr_with_n(x, y, min_n=CORR_MIN_N):
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


def _spearmanr_with_n(x, y, min_n=CORR_MIN_N):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
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


def _build_region_lookup(cluster_ids, cluster_acronyms, labels, label_min=None):
    if cluster_ids is None or cluster_acronyms is None:
        return pd.DataFrame(columns=["cluster_id", "region"])
    region_df = pd.DataFrame(
        {
            "cluster_id": np.asarray(cluster_ids),
            "region": np.asarray(cluster_acronyms).astype(str),
        }
    )
    if label_min is not None and labels is not None:
        try:
            labels_float = np.asarray(labels, dtype=float)
            good_ids = np.asarray(cluster_ids)[labels_float >= float(label_min)]
        except (TypeError, ValueError):
            good_ids = np.asarray(cluster_ids)[np.asarray(labels) == 1]
        region_df = region_df[region_df["cluster_id"].isin(good_ids)]
    region_df = region_df[~region_df["region"].isin(["root", "void"])]
    return region_df.reset_index(drop=True)


def _build_variable_table(df, spec, region_lookup):
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


def _format_corr_value(val):
    if np.isfinite(val):
        return f"{val:.2f}"
    return "nan"


def _resolve_region_colors(regions, region_colors):
    regions = [str(r) for r in sorted(pd.unique(regions))]
    resolved = {}
    missing = []
    if region_colors:
        for region in regions:
            color = region_colors.get(region)
            if color:
                resolved[region] = color
            else:
                missing.append(region)
    else:
        missing = regions

    if missing:
        palette = qualitative.Plotly
        for idx, region in enumerate(missing):
            resolved[region] = palette[idx % len(palette)]
    return resolved


def _add_unity_line(fig, x_vals, y_vals):
    if len(x_vals) <= 1 or len(y_vals) <= 1:
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


def _scatter_by_region(
    x_vals,
    y_vals,
    regions,
    cluster_ids,
    region_colors,
    title,
    x_label,
    y_label,
    template,
):
    fig = go.Figure()
    color_map = None
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
    if regions is None:
        hovertemplate = "Cluster %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
        customdata = cluster_ids if cluster_ids is not None else None
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                customdata=customdata,
                hovertemplate=hovertemplate if customdata is not None else None,
                marker=dict(size=6, opacity=0.65),
            )
        )
    else:
        regions = np.asarray(regions).astype(str)
        color_map = _resolve_region_colors(regions, region_colors)
        for region in sorted(pd.unique(regions)):
            mask = regions == region
            if not np.any(mask):
                continue
            color = color_map.get(region) if color_map else None
            marker = dict(size=6, opacity=0.65)
            if color:
                marker["color"] = color
            customdata = cluster_ids[mask] if cluster_ids is not None else None
            hovertemplate = (
                f"Region: {region}<br>Cluster "
                + "%{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
                if customdata is not None
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(x_vals)[mask],
                    y=np.asarray(y_vals)[mask],
                    mode="markers",
                    name=str(region),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    marker=marker,
                )
            )
    _add_unity_line(fig, x_vals, y_vals)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=template,
        height=620,
        width=620,
        margin=dict(l=70, r=40, t=90, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig, color_map


def _build_pairwise_corr_plot(
    data,
    region_lookup,
    spec_x,
    spec_y,
    template,
    region_label,
    region_colors=None,
    highlight_cluster_id=None,
):
    df_x = _build_variable_table(data.get(spec_x["df"]), spec_x, region_lookup)
    if df_x is None or df_x.empty:
        return None, f"No data for {spec_x['name']} in {region_label}."

    if spec_x["name"] == spec_y["name"]:
        x_vals = df_x[spec_x["v1"]].to_numpy(dtype=float)
        y_vals = df_x[spec_x["v2"]].to_numpy(dtype=float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        regions = df_x["region"].to_numpy(dtype=str)[mask] if "region" in df_x else None
        cluster_ids = (
            df_x["cluster_id"].to_numpy(dtype=int)[mask]
            if "cluster_id" in df_x
            else None
        )
        if len(x_plot) == 0:
            return None, f"No valid pairs for {spec_x['name']} in {region_label}."
        r_val, n_r = _pearsonr_with_n(x_vals, y_vals)
        rho_val, n_s = _spearmanr_with_n(x_vals, y_vals)
        title = (
            f"{spec_x['name']} reliability | "
            f"Pearson r={_format_corr_value(r_val)} (n={n_r}) | "
            f"Spearman rho={_format_corr_value(rho_val)} (n={n_s}) | "
            f"{region_label}"
        )
        fig, color_map = _scatter_by_region(
            x_plot,
            y_plot,
            regions,
            cluster_ids,
            region_colors,
            title,
            spec_x["v1"],
            spec_x["v2"],
            template,
        )
        if highlight_cluster_id is not None and "cluster_id" in df_x.columns:
            row = df_x.loc[df_x["cluster_id"] == highlight_cluster_id]
            if not row.empty:
                hx = float(row.iloc[0][spec_x["v1"]])
                hy = float(row.iloc[0][spec_x["v2"]])
                if np.isfinite(hx) and np.isfinite(hy):
                    h_region = None
                    if "region" in row.columns:
                        h_region = str(row.iloc[0]["region"])
                    h_color = None
                    if color_map and h_region in color_map:
                        h_color = color_map[h_region]
                    outline = "white" if "dark" in str(template).lower() else "black"
                    marker = dict(size=14, opacity=0.9, line=dict(width=3, color=outline))
                    if h_color:
                        marker["color"] = h_color
                    fig.add_trace(
                        go.Scatter(
                            x=[hx],
                            y=[hy],
                            mode="markers",
                            marker=marker,
                            name="Selected neuron",
                            showlegend=False,
                        )
                    )
        return fig, None

    df_y = _build_variable_table(data.get(spec_y["df"]), spec_y, region_lookup)
    if df_y is None or df_y.empty:
        return None, f"No data for {spec_y['name']} in {region_label}."

    merged = df_x[["cluster_id", "mean"]].merge(
        df_y[["cluster_id", "mean"]],
        on="cluster_id",
        how="inner",
        suffixes=("_x", "_y"),
    )
    if merged.empty:
        return None, f"No overlapping units for {spec_x['name']} and {spec_y['name']}."

    if "region" in df_x.columns:
        merged = merged.merge(
            df_x[["cluster_id", "region"]].drop_duplicates("cluster_id"),
            on="cluster_id",
            how="left",
        )

    x_vals = merged["mean_x"].to_numpy(dtype=float)
    y_vals = merged["mean_y"].to_numpy(dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_plot = x_vals[mask]
    y_plot = y_vals[mask]
    regions = merged["region"].to_numpy(dtype=str)[mask] if "region" in merged else None
    cluster_ids = (
        merged["cluster_id"].to_numpy(dtype=int)[mask]
        if "cluster_id" in merged
        else None
    )
    if len(x_plot) == 0:
        return None, f"No valid pairs for {spec_x['name']} vs {spec_y['name']}."

    r_val, n_r = _pearsonr_with_n(x_vals, y_vals)
    rho_val, n_s = _spearmanr_with_n(x_vals, y_vals)
    title = (
        f"{spec_x['name']} vs {spec_y['name']} | "
        f"Pearson r={_format_corr_value(r_val)} (n={n_r}) | "
        f"Spearman rho={_format_corr_value(rho_val)} (n={n_s}) | "
        f"{region_label}"
    )
    fig, color_map = _scatter_by_region(
        x_plot,
        y_plot,
        regions,
        cluster_ids,
        region_colors,
        title,
        spec_x["name"],
        spec_y["name"],
        template,
    )
    if highlight_cluster_id is not None and "cluster_id" in merged.columns:
        row = merged.loc[merged["cluster_id"] == highlight_cluster_id]
        if not row.empty:
            hx = float(row.iloc[0]["mean_x"])
            hy = float(row.iloc[0]["mean_y"])
            if np.isfinite(hx) and np.isfinite(hy):
                h_region = None
                if "region" in row.columns:
                    h_region = str(row.iloc[0]["region"])
                h_color = None
                if color_map and h_region in color_map:
                    h_color = color_map[h_region]
                outline = "white" if "dark" in str(template).lower() else "black"
                marker = dict(size=14, opacity=0.9, line=dict(width=3, color=outline))
                if h_color:
                    marker["color"] = h_color
                fig.add_trace(
                    go.Scatter(
                        x=[hx],
                        y=[hy],
                        mode="markers",
                        marker=marker,
                        name="Selected neuron",
                        showlegend=False,
                    )
                )
    return fig, None


st.title("Neuron Session Dashboard")

pid_list = _list_pids(CACHE_DIR)
if not pid_list:
    st.warning("No cached sessions found in data/dashboard_cache.")
    st.stop()

pid = st.sidebar.selectbox("Select PID", pid_list)
st.sidebar.subheader("Raw data")
load_wheel = st.sidebar.toggle("Load wheel data", value=False)
load_pose = st.sidebar.toggle("Load pose data", value=False)
allow_remote = st.sidebar.toggle("Allow Alyx lookup (online)", value=True)

with st.spinner("Loading session cache..."):
    data = _load_cache(pid)

raw_error = None
raw_error_remote = None
raw_source = None
raw_spikes = None
raw_clusters = None
raw_session = None
with st.spinner("Loading raw session data (data/raw)..."):
    try:
        raw_spikes, raw_clusters, raw_session, _ssl = _load_raw_session(
            pid, load_wheel=load_wheel, load_pose=load_pose, mode="local"
        )
        raw_source = "local"
    except Exception as exc:
        raw_error = exc
        if allow_remote:
            try:
                raw_spikes, raw_clusters, raw_session, _ssl = _load_raw_session(
                    pid, load_wheel=load_wheel, load_pose=load_pose, mode="remote"
                )
                raw_source = "remote"
            except Exception as exc_remote:
                raw_error_remote = exc_remote

spikes = raw_spikes if raw_spikes is not None else data.get("spikes")
clusters = raw_clusters if raw_clusters is not None else data.get("clusters")
session = raw_session if raw_session is not None else data.get("session")

if (spikes is None or clusters is None or session is None) and raw_error is not None:
    st.error(
        "Raw session data not available in data/raw. "
        "Run 03_calc_dashboard.py once with remote access to populate the cache."
    )
    st.caption(f"Local load error: {type(raw_error).__name__}: {raw_error}")
    if raw_error_remote is not None:
        st.caption(f"Remote load error: {type(raw_error_remote).__name__}: {raw_error_remote}")
    st.stop()
elif raw_error is not None and raw_source is None:
    st.warning(f"Raw load failed, using cached blobs. Details: {raw_error}")
elif raw_error is not None and raw_source is not None:
    st.warning(f"Local load failed; using {raw_source} metadata lookup. Details: {raw_error}")

meta = data.get("meta", {})
cluster_ids = data.get("cluster_ids")
cluster_acronyms = data.get("cluster_acronyms_plot")

if cluster_ids is None and clusters is not None:
    cluster_ids, _ = build_cluster_id_map(clusters)
if cluster_acronyms is None and clusters is not None:
    if hasattr(clusters, "acronym"):
        cluster_acronyms = np.asarray(clusters.acronym)
    elif isinstance(clusters, dict) and "acronym" in clusters:
        cluster_acronyms = np.asarray(clusters.get("acronym"))
if cluster_ids is None or cluster_acronyms is None:
    st.error("Cluster IDs or acronyms missing. Rebuild cache or verify raw data.")
    st.stop()
trials = data.get("trials")
config_plot = data.get("config_plot", {})
config_calc = data.get("config_calc", {})

if session is None:
    st.warning("Session data missing. Ensure data/raw is populated for this PID.")

if trials is None:
    st.warning("Trial data missing in cache.")
    st.stop()

st.subheader("Session Info")
info = {
    "Lab": meta.get("lab"),
    "Num trials": meta.get("num_trials"),
    "PID": meta.get("pid"),
    "EID": meta.get("eid"),
    "PIDs Numbers in this session": meta.get("num_other_pids"),
    "Date": meta.get("date"),
    "Recording length": _format_seconds(meta.get("recording_length_s")),
    "Spont length": _format_seconds(meta.get("spont_length_s")),
    "Subject": meta.get("subject"),
    "Spont interval": _spont_interval_text(meta.get("spont_interval")),
}
info_df = pd.DataFrame(info, index=[0]).T
info_df.columns = ["Value"]
st.table(info_df.astype(str))

st.subheader("Region Table")
cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
all_counts = pd.Series(cluster_acronyms).value_counts().sort_index()
labels = _get_label_array(clusters, cluster_ids)
if labels is not None:
    good_counts = pd.Series(cluster_acronyms[labels == 1]).value_counts().sort_index()
    good_cluster_ids = np.asarray(cluster_ids)[labels == 1]
else:
    good_counts = pd.Series(dtype=int)
    good_cluster_ids = None

plot_cluster_ids = np.asarray(cluster_ids)
plot_cluster_acronyms = cluster_acronyms
plot_label_values = _label_values_for_clusters(plot_cluster_ids, clusters, labels)
region_table = pd.DataFrame({"All Neurons": all_counts, "Good Neurons": good_counts}).fillna(0)
region_table = region_table.astype(int)
st.dataframe(region_table, width="stretch")

calc_label_min = config_calc.get("CALC_LABEL_MIN", None)
if calc_label_min is None and config_calc.get("CALC_ONLY_GOOD_UNITS", False):
    calc_label_min = 1.0
calc_label = "All neurons" if calc_label_min is None else f"Label >= {calc_label_min}"
st.caption(f"Calculations: {calc_label}")

plot_label_min = st.number_input(
    "Plot label min",
    min_value=0.0,
    max_value=1.0,
    value=float(calc_label_min if calc_label_min is not None else 0.5),
    step=0.1,
)

plot_config = dict(config_plot)
plot_config["PLOT_ONLY_GOOD_UNITS"] = False
plot_config["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
plot_config["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
plot_config["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
plot_config["SINGLE_NEURON_SMOOTH_SIGMA"] = 0.5
plot_config["SINGLE_NEURON_BIN_SIZE"] = 0.03
plot_config["DELAY_UNITS"] = config_calc.get("DELAY_UNITS", "s")
plotly_dark_mode = st.toggle("Plotly dark mode", value=False)
plot_config["PLOTLY_TEMPLATE"] = "plotly_dark" if plotly_dark_mode else "plotly_white"
plotting_utils.DEFAULT_TEMPLATE = plot_config["PLOTLY_TEMPLATE"]
pio.templates.default = plot_config["PLOTLY_TEMPLATE"]
plot_config["PLOT_LABEL_MIN"] = plot_label_min
region_colors = _build_region_colors(cluster_acronyms)

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
}

df_coupling_plot = data.get("df_coupling")
df_coupling_task_plot = data.get("df_coupling_task")
df_coupling_iti_plot = data.get("df_coupling_iti")
df_comparison_plot = data.get("df_comparison")
if plot_label_values is not None and plot_label_min is not None:
    plot_mask = plot_label_values >= float(plot_label_min)
    plot_cluster_ids = plot_cluster_ids[plot_mask]
    plot_cluster_acronyms = plot_cluster_acronyms[plot_mask]
    plot_label_values = plot_label_values[plot_mask]
    if df_coupling_plot is not None:
        df_coupling_plot = df_coupling_plot[df_coupling_plot["cluster_id"].isin(plot_cluster_ids)]
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

st.subheader("General Raster")
variability_choice = st.radio(
    "PSTH variability metric",
    ["Fano Factor", "CV"],
    horizontal=True,
    key="psth_variability_metric",
)
variability_metric = "fano" if variability_choice == "Fano Factor" else "cv"
min_time = float(np.nanmin(spikes["times"]))
max_time = float(np.nanmax(spikes["times"]))
if "general_t_start" not in st.session_state:
    st.session_state.general_t_start = float(min_time)
if "general_t_end" not in st.session_state:
    st.session_state.general_t_end = float(min(min_time + 10.0, max_time))

col_a, col_b, col_shift = st.columns([1, 1, 0.5])
with col_shift:
    shift_seconds = st.number_input(
        "Shift (s)",
        key="general_shift_seconds",
        value=1.0,
        min_value=0.0,
        step=0.5,
    )
    shift_window = st.button("Shift +", use_container_width=True)
if shift_window:
    window = st.session_state.general_t_end - st.session_state.general_t_start
    total_range = max_time - min_time
    if window <= 0:
        window = min(10.0, total_range)
    if total_range > 0 and window > total_range:
        window = total_range
    shift_val = float(shift_seconds)
    new_start = st.session_state.general_t_start + shift_val
    new_end = st.session_state.general_t_end + shift_val
    if new_end > max_time:
        new_end = max_time
        new_start = max_time - window
    if new_start < min_time:
        new_start = min_time
        new_end = min_time + window
    st.session_state.general_t_start = float(new_start)
    st.session_state.general_t_end = float(new_end)
with col_a:
    t_start = st.number_input(
        "Start time (s)",
        key="general_t_start",
        min_value=min_time,
        max_value=max_time,
    )
with col_b:
    t_end = st.number_input(
        "End time (s)",
        key="general_t_end",
        min_value=min_time,
        max_value=max_time,
    )

general_sort = st.selectbox(
    "General raster sorting",
    [
        "Default (Depth)",
        "Delay to Stim On",
        "Delay to First Move",
        "Delay to Response",
        "Delay to Feedback",
        "Task stPR Delay",
        "Task stPR Strength",
        "Task stPR Max",
        "ITI stPR Delay",
        "ITI stPR Strength",
        "ITI stPR Max",
        "Spont stPR Delay",
        "Spont stPR Strength",
        "Spont stPR Max",
    ],
    key="general_sort",
)

if t_end <= t_start:
    st.warning("End time must be greater than start time.")
else:
    fig_session = plot_time_window_raster_plotly(
        spikes,
        clusters,
        plot_cluster_ids,
        plot_cluster_acronyms,
        session,
        plot_config,
        t_start,
        t_end,
        variability_metric=variability_metric,
        sorting_metric=sort_map[general_sort],
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
        df_coupling_iti=df_coupling_iti_plot,
        pupil_features=data.get("pupil_features"),
        pupil_times=data.get("pupil_times"),
        region_colors=region_colors,
    )
    st.plotly_chart(fig_session, width="stretch")

st.subheader("Trial Inspector")
trial_idx = st.selectbox("Trial Number", trials["trial_idx"].tolist())
trial_row = trials.loc[trials["trial_idx"] == trial_idx].iloc[0]
trial_table = pd.DataFrame(
    {
        "Contrast": [trial_row["contrast"]],
        "Reaction Time": [trial_row["reaction_time"]],
        "Response Type": [trial_row["correct_response"]],
        "Subject Response": [trial_row["subject_response"]],
    }
)
st.table(trial_table)

sort_choice = st.selectbox(
    "Sorting", 
    [
        "Default (Depth)",
        "Delay to Stim On",
        "Delay to First Move",
        "Delay to Response",
        "Delay to Feedback",
        "Task stPR Delay",
        "Task stPR Strength",
        "Task stPR Max",
        "ITI stPR Delay",
        "ITI stPR Strength",
        "ITI stPR Max",
        "Spont stPR Delay",
        "Spont stPR Strength",
        "Spont stPR Max",
    ],
)

fig_trial = plot_trial_raster_plotly(
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    session,
    plot_config,
    trial_idx,
    variability_metric=variability_metric,
    sorting_metric=sort_map[sort_choice],
    df_res=data.get("df_res"),
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    df_coupling_iti=df_coupling_iti_plot,
    pupil_features=data.get("pupil_features"),
    pupil_times=data.get("pupil_times"),
    region_colors=region_colors,
)
st.plotly_chart(fig_trial, width="stretch")

st.subheader("Response Analysis")
plot_sort = st.selectbox(
    "Population sort", 
    [
        "Default (Depth)",
        "Own Event Delay",
        "Task stPR Delay",
        "Task stPR Strength",
        "Task stPR Max",
        "ITI stPR Delay",
        "ITI stPR Strength",
        "ITI stPR Max",
        "Spont stPR Delay",
        "Spont stPR Strength",
        "Spont stPR Max",
    ],
    index=1,
)
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
}
pop_key = (
    pid,
    plot_label_min,
    plot_sort,
    plot_config["PLOTLY_TEMPLATE"],
    tuple(config_plot.get("PLOT_REGIONS") or []),
    config_plot.get("POP_BIN_SIZE"),
    config_plot.get("POP_SMOOTH_SIGMA"),
    config_plot.get("POP_CMAP_NAME"),
    config_plot.get("POP_NORMALIZE"),
)

def _build_population_figs():
    figs = []
    for event_name in ["stimOn_times", "firstMovement_times", "feedback_times"]:
        cfg = dict(config_plot)
        cfg["PLOT_EVENT"] = event_name
        cfg["PLOT_ONLY_GOOD_UNITS"] = False
        cfg["PLOTLY_TEMPLATE"] = plot_config["PLOTLY_TEMPLATE"]
        figs.append(
            plot_population_sorted_plotly(
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
                region_acronyms=cfg.get("PLOT_REGIONS"),
                sort_mode=plot_sort_map[plot_sort],
            )
        )
    return figs

pop_figs = _get_cached_value("population_analysis_figs", pop_key, _build_population_figs)
cols = st.columns(3)
for fig_pop, col in zip(pop_figs or [], cols):
    col.plotly_chart(fig_pop, width="stretch")

st.subheader("Coupling")
coupling_key = (
    pid,
    plot_label_min,
    plot_config["PLOTLY_TEMPLATE"],
    tuple(config_plot.get("PLOT_REGIONS") or []),
    config_calc.get("STPR_BIN_SIZE"),
    config_calc.get("STPR_WINDOW_MS"),
    plot_config.get("POP_CMAP_NAME"),
    True,
    "per_row_all",
)

def _build_coupling_figs():
    region_acronyms = config_plot.get("PLOT_REGIONS")
    fig_spont = plot_population_coupling_heatmap_plotly(
        df_coupling_plot,
        plot_config,
        config_calc,
        region_acronyms=region_acronyms,
        zscore_by_region=True,
        colorbar_mode="per_row",
    )
    fig_task = plot_population_coupling_heatmap_plotly(
        df_coupling_task_plot,
        plot_config,
        config_calc,
        region_acronyms=region_acronyms,
        zscore_by_region=True,
        colorbar_mode="per_row",
    )
    fig_iti = plot_population_coupling_heatmap_plotly(
        df_coupling_iti_plot,
        plot_config,
        config_calc,
        region_acronyms=region_acronyms,
        zscore_by_region=True,
        colorbar_mode="per_row",
    )
    return fig_spont, fig_task, fig_iti

fig_spont, fig_task, fig_iti = _get_cached_value(
    "coupling_figs", coupling_key, _build_coupling_figs
)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Spont Coupling**")
    st.plotly_chart(fig_spont, width="stretch")
with col2:
    st.markdown("**Task Coupling**")
    st.plotly_chart(fig_task, width="stretch")
with col3:
    st.markdown("**ITI Coupling**")
    st.plotly_chart(fig_iti, width="stretch")

st.subheader("Correlation Matrices")
corr_key = (
    pid,
    plot_label_min,
    plot_config["PLOTLY_TEMPLATE"],
    tuple(config_plot.get("PLOT_REGIONS") or []),
)

def _build_corr_figs():
    region_lookup = _build_region_lookup(
        cluster_ids,
        cluster_acronyms,
        labels,
        label_min=plot_label_min,
    )
    if region_lookup.empty:
        return []

    available_specs = []
    for spec in CORR_VARIABLES:
        df_src = data.get(spec["df"])
        if df_src is None:
            continue
        if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
            continue
        available_specs.append(spec)

    if not available_specs:
        return []

    spec_by_name = {spec["name"]: spec for spec in available_specs}
    var_tables_all = {}
    for spec in available_specs:
        df_var = _build_variable_table(data.get(spec["df"]), spec, region_lookup)
        if df_var is None or df_var.empty:
            continue
        var_tables_all[spec["name"]] = df_var

    regions_all = sorted(region_lookup["region"].unique().tolist())
    region_filters = config_plot.get("PLOT_REGIONS")
    if region_filters:
        filtered = []
        for region in regions_all:
            if any(str(region).startswith(str(r)) for r in region_filters):
                filtered.append(region)
        regions_all = filtered

    results = []
    for region in regions_all:
        region_ids = region_lookup.loc[
            region_lookup["region"] == region, "cluster_id"
        ].to_numpy()
        n_total = int(len(region_ids))
        if n_total == 0:
            continue

        var_tables = {}
        for spec in available_specs:
            name = spec["name"]
            df_var = var_tables_all.get(name)
            if df_var is None:
                continue
            df_region = df_var[df_var["region"] == region]
            if df_region.empty:
                continue
            var_tables[name] = df_region

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
            r_val, n_val = _pearsonr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
            reliability[name] = r_val
            reliability_n[name] = n_val

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
        text_mat = np.empty((n_vars, n_vars), dtype=object)
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    r_val = reliability.get(name_i, np.nan)
                    n_val = reliability_n.get(name_i, 0)
                    corr_mat[i, j] = r_val
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
                "Reliability (diag) + Pairwise Pearson (off-diag) | "
                f"Region {region} | N total (label>= {plot_label_min}): {n_total}"
            ),
            height=min(1000, max(500, 40 * n_vars + 200)),
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig.update_xaxes(tickangle=45)

        spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
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
                "Reliability (diag) + Pairwise Spearman (off-diag) | "
                f"Region {region} | N total (label>= {plot_label_min}): {n_total}"
            ),
            height=min(1000, max(500, 40 * n_vars + 200)),
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig_s.update_xaxes(tickangle=45)

        results.append(
            {
                "region": region,
                "n_total": n_total,
                "pearson": fig,
                "spearman": fig_s,
            }
        )

    return results

corr_results = _get_cached_value("corr_matrix_figs", corr_key, _build_corr_figs)
if not corr_results:
    st.info("No correlation matrices available for the current filters.")
else:
    st.caption(
        "Diagonal entries show within-variable reliability; off-diagonal entries show "
        "pairwise correlations of mean values."
    )
    for idx, entry in enumerate(corr_results):
        region = entry["region"]
        n_total = entry["n_total"]
        with st.expander(f"Region {region} (N={n_total})", expanded=idx == 0):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Pearson**")
                st.plotly_chart(entry["pearson"], width="stretch")
            with col_b:
                st.markdown("**Spearman**")
                st.plotly_chart(entry["spearman"], width="stretch")

units_df = pd.DataFrame(
    {
        "cluster_id": np.asarray(cluster_ids),
        "region": np.asarray(cluster_acronyms).astype(str),
    }
)
label_vals_full = _label_values_for_clusters(np.asarray(cluster_ids), clusters, labels)
if label_vals_full is not None:
    units_df["label_value"] = label_vals_full
else:
    units_df["label_value"] = np.nan

if plot_label_min is not None and plot_label_values is not None:
    units_df = units_df[units_df["cluster_id"].isin(plot_cluster_ids)]

units_df = units_df.sort_values(["region", "cluster_id"]).reset_index(drop=True)
label_map = {}
units_df_empty = units_df.empty
if units_df_empty:
    selected_cluster_id = None
else:
    for _, row in units_df.iterrows():
        label_val = row.get("label_value", np.nan)
        label_text = "NA" if pd.isna(label_val) else f"label={label_val:.2f}"
        label_map[row["cluster_id"]] = f"{row['cluster_id']} | {row['region']} | {label_text}"

    default_cluster_id = int(units_df["cluster_id"].iloc[0])
    selected_cluster_id = st.session_state.get("single_neuron_select", default_cluster_id)
    if selected_cluster_id not in units_df["cluster_id"].values:
        selected_cluster_id = default_cluster_id
        st.session_state["single_neuron_select"] = selected_cluster_id

st.subheader("Variable Correlation")
region_lookup_plot = _build_region_lookup(
    cluster_ids,
    cluster_acronyms,
    labels,
    label_min=plot_label_min,
)
available_specs = []
for spec in CORR_VARIABLES:
    df_src = data.get(spec["df"])
    if df_src is None:
        continue
    if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
        continue
    available_specs.append(spec)

if region_lookup_plot.empty or not available_specs:
    st.info("No variables available for the correlation plot.")
else:
    spec_by_name = {spec["name"]: spec for spec in available_specs}
    var_names = [spec["name"] for spec in available_specs]
    regions_all = sorted(region_lookup_plot["region"].unique().tolist())
    region_filters = config_plot.get("PLOT_REGIONS")
    if region_filters:
        filtered = []
        for region in regions_all:
            if any(str(region).startswith(str(r)) for r in region_filters):
                filtered.append(region)
        regions_all = filtered

    if not regions_all:
        st.info("No regions available for the current filters.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            var_x = st.selectbox(
                "Variable X",
                var_names,
                key="corr_var_x",
            )
        with col_b:
            default_idx = 1 if len(var_names) > 1 else 0
            var_y = st.selectbox(
                "Variable Y",
                var_names,
                index=default_idx,
                key="corr_var_y",
            )
        with col_c:
            region_choice = st.selectbox(
                "Region",
                ["ALL"] + regions_all,
                key="corr_region",
            )

        if var_x not in spec_by_name or var_y not in spec_by_name:
            st.info("Selected variables are not available.")
        else:
            if region_choice == "ALL":
                region_label = "All regions"
                region_lookup_sel = region_lookup_plot
            else:
                region_label = f"Region {region_choice}"
                region_lookup_sel = region_lookup_plot[
                    region_lookup_plot["region"] == region_choice
                ]

            fig_corr, corr_msg = _build_pairwise_corr_plot(
                data,
                region_lookup_sel,
                spec_by_name[var_x],
                spec_by_name[var_y],
                plot_config["PLOTLY_TEMPLATE"],
                region_label,
                region_colors=region_colors,
                highlight_cluster_id=selected_cluster_id,
            )
            if fig_corr is None:
                st.info(corr_msg or "Not enough data to plot correlations.")
            else:
                st.plotly_chart(fig_corr, width="stretch")

st.subheader("Single Neuron")
if units_df_empty or selected_cluster_id is None:
    st.info("No neurons available for selection with current filters.")
else:
    selected_idx = int(
        np.where(units_df["cluster_id"].values == selected_cluster_id)[0][0]
    )
    selected_cluster_id = st.selectbox(
        "Select neuron",
        units_df["cluster_id"].tolist(),
        index=selected_idx,
        format_func=lambda cid: label_map.get(cid, str(cid)),
        key="single_neuron_select",
    )

    selected_row = units_df.loc[units_df["cluster_id"] == selected_cluster_id].iloc[0]
    label_val = selected_row.get("label_value", np.nan)
    quality_text = "NA" if pd.isna(label_val) else f"{label_val:.2f}"

    info_cols = st.columns(3)
    info_cols[0].metric("Cluster ID", selected_cluster_id)
    info_cols[1].metric("Region", selected_row["region"])
    info_cols[2].metric("Label", quality_text)

    fig_single = plot_single_neuron_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        selected_cluster_id,
    )
    st.plotly_chart(fig_single, width="stretch")

    st.markdown("**First Movement (Left vs Right)**")
    fig_move = plot_single_neuron_conditioned_event_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        selected_cluster_id,
        event_name="firstMovement_times",
        condition_type="choice",
        title="First Movement Response",
    )
    st.plotly_chart(fig_move, width="stretch")

    st.markdown("**Feedback (Correct vs Incorrect)**")
    fig_feedback = plot_single_neuron_conditioned_event_plotly(
        session,
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        data.get("df_res"),
        plot_config,
        selected_cluster_id,
        event_name="feedback_times",
        condition_type="feedback",
        title="Feedback Response",
    )
    st.plotly_chart(fig_feedback, width="stretch")

    col_task, col_spont, col_iti = st.columns(3)
    with col_task:
        st.markdown("**Task stPR (Odd vs Even Trials)**")
        fig_task_curve = plot_stpr_curve_halves_plotly(
            df_coupling_task_plot,
            config_calc,
            selected_cluster_id,
            title="Task stPR Curve (Odd vs Even Trials)",
            template=plot_config["PLOTLY_TEMPLATE"],
            split_suffixes=("odd", "even"),
            split_labels=("Odd trials", "Even trials"),
        )
        st.plotly_chart(fig_task_curve, width="stretch")
    with col_spont:
        st.markdown("**Spont stPR (First vs Second Half)**")
        fig_spont_curve = plot_stpr_curve_halves_plotly(
            df_coupling_plot,
            config_calc,
            selected_cluster_id,
            title="Spont stPR Curve (First vs Second Half)",
            template=plot_config["PLOTLY_TEMPLATE"],
        )
        st.plotly_chart(fig_spont_curve, width="stretch")
    with col_iti:
        st.markdown("**ITI stPR (Odd vs Even Trials)**")
        fig_iti_curve = plot_stpr_curve_halves_plotly(
            df_coupling_iti_plot,
            config_calc,
            selected_cluster_id,
            title="ITI stPR Curve (Odd vs Even Trials)",
            template=plot_config["PLOTLY_TEMPLATE"],
            split_suffixes=("odd", "even"),
            split_labels=("Odd trials", "Even trials"),
        )
        st.plotly_chart(fig_iti_curve, width="stretch")

    st.markdown("**stPR Mean Curves (Task vs Spont vs ITI)**")
    fig_stpr_mean = _plot_stpr_mean_comparison(
        df_coupling_plot,
        df_coupling_task_plot,
        df_coupling_iti_plot,
        config_calc,
        selected_cluster_id,
        plot_config["PLOTLY_TEMPLATE"],
    )
    st.plotly_chart(fig_stpr_mean, width="stretch")
