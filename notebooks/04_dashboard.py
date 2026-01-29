# %%
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.plotting_plotly import (
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
    plot_population_coupling_heatmap_plotly,
    plot_coupling_strength_summary_plotly,
    plot_coupling_delay_summary_plotly,
    plot_coupling_sorting_summary_plotly,
)

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


def _get_label_array(clusters):
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            return np.asarray(clusters.metrics.label)
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




st.title("Neuron Session Dashboard")

pid_list = _list_pids(CACHE_DIR)
if not pid_list:
    st.warning("No cached sessions found in data/dashboard_cache.")
    st.stop()

pid = st.sidebar.selectbox("Select PID", pid_list)

with st.spinner("Loading session cache..."):
    data = _load_cache(pid)

meta = data.get("meta", {})
cluster_ids = data.get("cluster_ids")
cluster_acronyms = data.get("cluster_acronyms_plot")
clusters = data.get("clusters")
spikes = data.get("spikes")
session = data.get("session")
trials = data.get("trials")
config_plot = data.get("config_plot", {})
config_calc = data.get("config_calc", {})
labels = _get_label_array(clusters)
if labels is not None:
    good_cluster_ids = np.asarray(cluster_ids)[labels == 1]
else:
    good_cluster_ids = None

if session is None:
    st.warning("Session data missing in cache. Re-run 03_calc_dashboard.py to rebuild.")

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

calc_only_good = config_calc.get("CALC_ONLY_GOOD_UNITS", None)
calc_label = "Good neurons only" if calc_only_good else "All neurons"
st.caption(f"Calculations: {calc_label}")
plot_only_good = st.toggle(
    "Plot only good neurons",
    value=config_plot.get("PLOT_ONLY_GOOD_UNITS", True),
)
avg_psth_only_good = st.toggle(
    "Avg PSTH uses good neurons only",
    value=True,
)
if calc_only_good and not plot_only_good:
    st.warning(
        "Calculations were done only for good neurons; delay/coupling metrics for other neurons will be missing."
    )

st.subheader("Region Table")
cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
all_counts = pd.Series(cluster_acronyms).value_counts().sort_index()
if labels is not None:
    good_counts = pd.Series(cluster_acronyms[labels == 1]).value_counts().sort_index()
else:
    good_counts = pd.Series(dtype=int)
region_table = pd.DataFrame({"All Neurons": all_counts, "Good Neurons": good_counts}).fillna(0)
region_table = region_table.astype(int)
st.dataframe(region_table, width="stretch")

plot_config = dict(config_plot)
plot_config["PLOT_ONLY_GOOD_UNITS"] = plot_only_good
plot_config["AVG_PSTH_ONLY_GOOD"] = avg_psth_only_good
plot_config["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
plot_config["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
plot_config["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
plot_config["PLOTLY_TEMPLATE"] = "plotly_white"
region_colors = _build_region_colors(cluster_acronyms)

sort_map = {
    "Default (Depth)": "depth",
    "Delay to Stim": "stim",
    "Delay to Feedback": "feedback",
    "Delay to Move": "move",
    "Coupling Spont": "spont",
    "Coupling Task": "task",
}

df_coupling_plot = data.get("df_coupling")
df_coupling_task_plot = data.get("df_coupling_task")
df_comparison_plot = data.get("df_comparison")
if plot_only_good and good_cluster_ids is not None:
    if df_coupling_plot is not None:
        df_coupling_plot = df_coupling_plot[df_coupling_plot["cluster_id"].isin(good_cluster_ids)]
    if df_coupling_task_plot is not None:
        df_coupling_task_plot = df_coupling_task_plot[
            df_coupling_task_plot["cluster_id"].isin(good_cluster_ids)
        ]
    if df_comparison_plot is not None:
        df_comparison_plot = df_comparison_plot[
            df_comparison_plot["cluster_id"].isin(good_cluster_ids)
        ]

st.subheader("General Raster")
min_time = float(np.nanmin(spikes["times"]))
max_time = float(np.nanmax(spikes["times"]))
col_a, col_b = st.columns(2)
with col_a:
    t_start = st.number_input("Start time (s)", value=min_time, min_value=min_time, max_value=max_time)
with col_b:
    t_end = st.number_input("End time (s)", value=min(min_time + 10.0, max_time), min_value=min_time, max_value=max_time)

general_sort = st.selectbox(
    "General raster sorting",
    ["Default (Depth)", "Delay to Stim", "Delay to Feedback", "Delay to Move", "Coupling Spont", "Coupling Task"],
    key="general_sort",
)

if t_end <= t_start:
    st.warning("End time must be greater than start time.")
else:
    fig_session = plot_time_window_raster_plotly(
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        session,
        plot_config,
        t_start,
        t_end,
        sorting_metric=sort_map[general_sort],
        df_res=data.get("df_res"),
        df_coupling=df_coupling_plot,
        df_coupling_task=df_coupling_task_plot,
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
        "Correct Response": [trial_row["correct_response"]],
        "Subject Response": [trial_row["subject_response"]],
    }
)
st.table(trial_table)

sort_choice = st.selectbox(
    "Sorting", 
    ["Default (Depth)", "Delay to Stim", "Delay to Feedback", "Delay to Move", "Coupling Spont", "Coupling Task"],
)

fig_trial = plot_trial_raster_plotly(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    session,
    plot_config,
    trial_idx,
    sorting_metric=sort_map[sort_choice],
    df_res=data.get("df_res"),
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    pupil_features=data.get("pupil_features"),
    pupil_times=data.get("pupil_times"),
    region_colors=region_colors,
)
st.plotly_chart(fig_trial, width="stretch")

st.subheader("Population Analysis")
plot_sort = st.selectbox(
    "Population sort", 
    ["Own event delays", "Coupling Spont", "Coupling Task", "Default"],
)
plot_sort_map = {
    "Own event delays": "delay",
    "Coupling Spont": "spont",
    "Coupling Task": "task",
    "Default": "depth",
}

cols = st.columns(3)
for event_name, col in zip(["stimOn_times", "firstMovement_times", "feedback_times"], cols):
    cfg = dict(config_plot)
    cfg["PLOT_EVENT"] = event_name
    cfg["PLOT_ONLY_GOOD_UNITS"] = plot_only_good
    cfg["PLOTLY_TEMPLATE"] = plot_config["PLOTLY_TEMPLATE"]
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
        region_acronyms=cfg.get("PLOT_REGIONS"),
        sort_mode=plot_sort_map[plot_sort],
    )
    col.plotly_chart(fig_pop, width="stretch")

st.subheader("Coupling")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Spont Coupling**")
    fig_spont = plot_population_coupling_heatmap_plotly(
        df_coupling_plot,
        plot_config,
        config_calc,
        region_acronyms=config_plot.get("PLOT_REGIONS"),
    )
    st.plotly_chart(fig_spont, width="stretch")
with col2:
    st.markdown("**Task Coupling**")
    fig_task = plot_population_coupling_heatmap_plotly(
        df_coupling_task_plot,
        plot_config,
        config_calc,
        region_acronyms=config_plot.get("PLOT_REGIONS"),
    )
    st.plotly_chart(fig_task, width="stretch")

st.subheader("stPR Comparison")
region_order = None
if df_comparison_plot is not None and "region" in df_comparison_plot.columns:
    region_order = df_comparison_plot["region"].unique().tolist()

fig_strength = plot_coupling_strength_summary_plotly(
    df_comparison_plot,
    region_order,
    region_colors=region_colors,
    template=plot_config["PLOTLY_TEMPLATE"],
)
fig_delay = plot_coupling_delay_summary_plotly(
    df_comparison_plot,
    region_order,
    region_colors=region_colors,
    template=plot_config["PLOTLY_TEMPLATE"],
)

cols = st.columns(2)
cols[0].plotly_chart(fig_strength, width="stretch")
cols[1].plotly_chart(fig_delay, width="stretch")
