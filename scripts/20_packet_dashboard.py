# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))

from utils.io import setup_paths, init_one, load_session_data
from utils.plotting_plotly import plot_time_window_raster_plotly, build_whisk_raster_overlay_inputs
import utils.plotting_plotly as plotting_utils
from utils.packet_dashboard import (
    BASE_CACHE_DIR,
    PACKET_CACHE_DIR,
    add_packet_cluster_markers_to_raster,
    build_cluster_heatmap_figure,
    build_feature_correlation_figure,
    build_neuron_scatter,
    build_packet_psth_figure,
    get_region_raster_y_bounds,
    get_neuron_scatter_options,
    load_base_cache,
    load_packet_cache,
    list_available_pids,
)


st.set_page_config(page_title="Packet Dashboard", layout="wide")

DEFAULT_CLUSTER_METHODS = [
    "raw_kmeans",
    "normalized_kmeans",
    "residual_kmeans",
    "raw_pca_kmeans",
    "normalized_pca_kmeans",
    "residual_pca_kmeans",
]


@st.cache_data(show_spinner=False)
def _load_packet_cache(pid):
    return load_packet_cache(pid, PACKET_CACHE_DIR)


@st.cache_data(show_spinner=False)
def _load_base_cache_data(pid):
    return load_base_cache(pid, BASE_CACHE_DIR)


@st.cache_resource(show_spinner=False)
def _get_one(mode):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    return init_one(ibl_cache, mode=mode)


@st.cache_resource(show_spinner=False)
def _load_raw_session(pid):
    last_exc = None
    for mode in ("local", "remote"):
        try:
            one = _get_one(mode)
            ssl, spikes, clusters, sl = load_session_data(
                pid,
                one,
                load_wheel=True,
                load_pose=True,
                load_motion_energy=False,
            )
            return spikes, clusters, sl, ssl
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Raw session load failed for {pid}: {last_exc}")


def _meta_table(packet_cache):
    meta = dict(packet_cache.get("meta", {}))
    spont_interval = meta.get("spont_interval")
    spont_text = "None"
    try:
        arr = np.asarray(spont_interval, dtype=float)
        if arr.size == 2 and np.isfinite(arr).all():
            spont_text = f"{arr[0]:.2f} to {arr[1]:.2f} s"
    except Exception:
        spont_text = "None"
    rows = [
        ("PID", meta.get("pid", packet_cache.get("pid"))),
        ("EID", meta.get("eid")),
        ("Subject", meta.get("subject")),
        ("Lab", meta.get("lab")),
        ("Date", meta.get("date")),
        ("Recording Length (s)", meta.get("recording_length_s")),
        ("Spont Length (s)", meta.get("spont_length_s")),
        ("Spont Interval", spont_text),
        ("Template Source", packet_cache.get("packet_config", {}).get("TEMPLATE_SOURCE")),
        ("Packet Dashboard Version", packet_cache.get("packet_dashboard_version")),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _default_window(region_bundle, recording_length_s):
    task_intervals = np.asarray(region_bundle["packet_dataset"].get("task_intervals", np.empty((0, 2))), dtype=float)
    if task_intervals.size > 0 and task_intervals.ndim == 2 and task_intervals.shape[1] == 2:
        valid = np.isfinite(task_intervals).all(axis=1) & (task_intervals[:, 1] > task_intervals[:, 0])
        task_intervals = task_intervals[valid]
        if task_intervals.size > 0:
            task_end = float(np.nanmax(task_intervals[:, 1]))
            start = max(0.0, task_end - 10.0)
            end_cap = float(recording_length_s) if recording_length_s is not None and np.isfinite(recording_length_s) else task_end
            end = min(end_cap, task_end)
            if end > start:
                return float(start), float(end)
    packet_times = np.asarray(
        pd.to_numeric(region_bundle["packet_dataset"]["packet_times"], errors="coerce"),
        dtype=float,
    )
    if packet_times.size == 0:
        return 0.0, min(20.0, float(recording_length_s) if recording_length_s is not None else 20.0)
    start = max(0.0, float(np.nanmin(packet_times)) - 5.0)
    end_cap = float(recording_length_s) if recording_length_s is not None and np.isfinite(recording_length_s) else float(np.nanmax(packet_times) + 5.0)
    end = min(end_cap, float(np.nanmax(packet_times)) + 5.0)
    if end <= start:
        end = start + 20.0
    return float(start), float(end)


st.title("Packet Dashboard")

packet_pid_list = list_available_pids(PACKET_CACHE_DIR)
if not packet_pid_list:
    st.error(f"No packet caches found in `{PACKET_CACHE_DIR}`. Run `19_packet_comp.py` first.")
    st.stop()

with st.sidebar:
    pid = st.selectbox("PID", packet_pid_list, index=0)
    packet_cache = _load_packet_cache(pid)
    base_cache = _load_base_cache_data(pid)
    region_summary_df = packet_cache.get("region_summary_df", pd.DataFrame())
    region_options = region_summary_df["region"].astype(str).tolist() if not region_summary_df.empty else []
    if not region_options:
        st.error("No packet regions available in this cache.")
        st.stop()
    region = st.selectbox("Region", region_options, index=0)
    region_bundle_sidebar = packet_cache["region_results"][region]
    cluster_methods = list(packet_cache.get("packet_config", {}).get("CLUSTER_METHODS", DEFAULT_CLUSTER_METHODS))
    cluster_methods_available = [
        method
        for method in cluster_methods
        if region_bundle_sidebar.get("cluster_results", {}).get(method) is not None
    ]
    if not cluster_methods_available:
        cluster_methods_available = cluster_methods
    cluster_method = st.selectbox("Clustering Method", cluster_methods_available, index=0)

region_bundle = packet_cache["region_results"][region]
cluster_result = region_bundle.get("cluster_results", {}).get(cluster_method)
if cluster_result is None:
    st.error(f"No clustering result available for region `{region}` with clustering method `{cluster_method}`.")
    st.stop()

recording_length_s = packet_cache.get("meta", {}).get("recording_length_s")
default_t_start, default_t_end = _default_window(region_bundle, recording_length_s)
window_sig = f"{pid}|{region}|{cluster_method}"
if st.session_state.get("packet_window_sig") != window_sig:
    st.session_state.packet_window_sig = window_sig
    st.session_state.packet_t_start = float(default_t_start)
    st.session_state.packet_t_end = float(default_t_end)
with st.sidebar:
    shift_seconds = st.number_input(
        "Shift (s)",
        key="packet_shift_seconds",
        value=1.0,
        min_value=0.0,
        step=0.5,
    )
    shift_window = st.button("Shift +", use_container_width=True)
    if shift_window:
        max_time = float(recording_length_s) if recording_length_s is not None and np.isfinite(recording_length_s) else float(default_t_end)
        min_time = 0.0
        window = float(st.session_state.packet_t_end) - float(st.session_state.packet_t_start)
        if window <= 0:
            window = 10.0
        total_range = max_time - min_time
        if total_range > 0 and window > total_range:
            window = total_range
        shift_val = float(shift_seconds)
        new_start = float(st.session_state.packet_t_start) + shift_val
        new_end = float(st.session_state.packet_t_end) + shift_val
        if new_end > max_time:
            new_end = max_time
            new_start = max_time - window
        if new_start < min_time:
            new_start = min_time
            new_end = min_time + window
        st.session_state.packet_t_start = float(new_start)
        st.session_state.packet_t_end = float(new_end)
    t_start = st.number_input("Window Start (s)", key="packet_t_start", step=1.0)
    t_end = st.number_input("Window End (s)", key="packet_t_end", step=1.0)
    if t_end <= t_start:
        st.warning("Window end must be greater than start. Using start + 20 s.")
        t_end = t_start + 20.0

plot_template = "plotly_white"
plotting_utils.DEFAULT_TEMPLATE = plot_template
pio.templates.default = plot_template

meta_col, region_col = st.columns([1.0, 1.4])
with meta_col:
    st.subheader("Session Info")
    st.dataframe(_meta_table(packet_cache), use_container_width=True, hide_index=True)
with region_col:
    st.subheader("Region Packet Summary")
    st.dataframe(region_summary_df, use_container_width=True, hide_index=True)

st.subheader("General Raster")
try:
    spikes, clusters, sl, _ssl = _load_raw_session(pid)
    raster_cluster_ids = np.asarray(region_bundle["packet_dataset"].get("cluster_ids", np.array([], dtype=int)), dtype=int)
    raster_cluster_acronyms = np.asarray(
        region_bundle["packet_dataset"].get("unit_table", pd.DataFrame()).get("acronym", pd.Series(dtype=str)),
        dtype=str,
    )
    raster_config_plot = dict(packet_cache.get("config_plot", {}))
    raster_config_plot["PLOT_ONLY_GOOD_UNITS"] = False
    raster_config_plot["PLOT_LABEL_MIN"] = None
    raster_config_plot["AVG_PSTH_ONLY_GOOD"] = False
    raster_sort_metric = "spont"

    whisk_inputs = build_whisk_raster_overlay_inputs(
        df_wh=base_cache.get("df_wh"),
        wh_detect=base_cache.get("wh_detect"),
        wh_event_base=base_cache.get("wh_event_base"),
        wh_events_by_period=base_cache.get("wh_events_by_period"),
    )
    fig_raster = plot_time_window_raster_plotly(
        spikes,
        clusters,
        raster_cluster_ids,
        raster_cluster_acronyms,
        sl,
        raster_config_plot,
        float(t_start),
        float(t_end),
        region_acronyms=[region],
        sorting_metric=raster_sort_metric,
        df_res=base_cache.get("df_res"),
        df_coupling=base_cache.get("df_coupling"),
        df_coupling_task=base_cache.get("df_coupling_task"),
        df_coupling_iti=base_cache.get("df_coupling_iti"),
        motion_mean_df=whisk_inputs.get("motion_mean_df"),
        extra_event_times=whisk_inputs.get("extra_event_times"),
        extra_event_styles=whisk_inputs.get("extra_event_styles"),
        extra_event_spans=whisk_inputs.get("extra_event_spans"),
        extra_event_span_styles=whisk_inputs.get("extra_event_span_styles"),
        packet_score_times=region_bundle["packet_dataset"].get("detection_bin_centers"),
        packet_score_values=region_bundle["packet_dataset"].get("region_score_plot"),
        packet_score_threshold=packet_cache.get("packet_config", {}).get("PACKET_THRESHOLD", 2.0),
        avg_psth_region_acronyms=[region],
        avg_psth_subplot_title="Population PSTH",
    )
    region_y0, region_y1 = get_region_raster_y_bounds(
        region,
        clusters,
        raster_cluster_ids,
        raster_cluster_acronyms,
        raster_config_plot,
        raster_sort_metric,
        df_res=base_cache.get("df_res"),
        df_coupling=base_cache.get("df_coupling"),
        df_coupling_task=base_cache.get("df_coupling_task"),
        df_coupling_iti=base_cache.get("df_coupling_iti"),
    )
    fig_raster = add_packet_cluster_markers_to_raster(
        fig_raster,
        cluster_result["packet_plot_df"],
        float(t_start),
        float(t_end),
        packet_window_s=region_bundle["packet_dataset"].get("packet_window_s"),
        region_y0=region_y0,
        region_y1=region_y1,
    )
    if region_y0 is not None and region_y1 is not None:
        region_height = max(1.0, float(region_y1) - float(region_y0))
        region_pad = max(2.0, 0.08 * region_height)
        fig_raster.update_yaxes(
            range=[float(region_y0) - region_pad, float(region_y1) + region_pad],
            row=1,
            col=1,
        )
    st.plotly_chart(fig_raster, use_container_width=True)
except Exception as exc:
    st.warning(f"General raster unavailable: {type(exc).__name__}: {exc}")

st.subheader("Template and Cluster Heatmaps")
heat_col1, heat_col2 = st.columns(2)
with heat_col1:
    st.plotly_chart(
        build_cluster_heatmap_figure(region_bundle, cluster_method=cluster_method, normalized_space=False, template=plot_template),
        use_container_width=True,
    )
with heat_col2:
    st.plotly_chart(
        build_cluster_heatmap_figure(region_bundle, cluster_method=cluster_method, normalized_space=True, template=plot_template),
        use_container_width=True,
    )

st.subheader("Main Packet Features")
st.plotly_chart(build_feature_correlation_figure(region_bundle, template=plot_template), use_container_width=True)
st.dataframe(region_bundle["feature_correlation_df"], use_container_width=True, hide_index=True)

st.subheader("Neuron Scatter")
neuron_x_col, neuron_y_col = st.columns(2)
neuron_options = get_neuron_scatter_options()
with neuron_x_col:
    neuron_x = st.selectbox("Neuron X", neuron_options, index=1, key="neuron_x")
with neuron_y_col:
    neuron_y = st.selectbox("Neuron Y", neuron_options, index=4 if len(neuron_options) > 4 else 0, key="neuron_y")
st.plotly_chart(build_neuron_scatter(region_bundle, neuron_x, neuron_y, template=plot_template), use_container_width=True)

st.subheader("Packet PSTH")
trials = getattr(sl, "trials", None) if "sl" in locals() else None
if trials is None:
    trials = base_cache.get("trials")
fig_psth = build_packet_psth_figure(
    np.asarray(region_bundle["packet_dataset"]["packet_times"], dtype=float),
    np.asarray(cluster_result["cluster_labels"], dtype=int),
    trials,
    packet_cache.get("wh_events_by_period", {}),
    packet_cache.get("packet_config", {}).get("WHISK_EVENT_CONTEXT", "all"),
    packet_cache.get("config_plot", {}),
    title=f"Region {region} | {cluster_method} | Packet Cluster PSTHs",
    cluster_name_prefix="Cluster",
    template=plot_template,
    precomputed=cluster_result.get("packet_psth"),
)
st.plotly_chart(fig_psth, use_container_width=True)
