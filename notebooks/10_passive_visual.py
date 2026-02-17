# %% Imports
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

import utils.analysis as ana_utils
import utils.plotting_plotly as plotting_utils
from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    build_cluster_id_map,
    get_cluster_labels_array,
    load_task_replay_datasets,
    extract_passive_times_and_contrast,
)
from utils.plotting_plotly import (
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
)


# %% Helpers
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"


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


def show_fig(fig, renderer=None):
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


def list_cached_pids(cache_dir=CACHE_DIR):
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


def _init_one_with_fallback(ibl_cache, preferred_mode="local", allow_remote=True):
    modes = []
    if preferred_mode:
        modes.append(preferred_mode)
    if "local" not in modes:
        modes.append("local")
    if allow_remote and "remote" not in modes:
        modes.append("remote")
    last_error = None
    for mode in modes:
        try:
            one = init_one(ibl_cache, mode=mode)
            return one, mode
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize ONE. Last error: {last_error}")


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
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
    return np.asarray([rate_map.get(cid, np.nan) for cid in cluster_ids], dtype=float)


def _select_cluster_ids_by_label(cluster_ids, clusters, label_min=None):
    cluster_ids = np.asarray(cluster_ids)
    if label_min is None:
        return cluster_ids
    labels = get_cluster_labels_array(clusters)
    if labels is None:
        return cluster_ids
    labels = np.asarray(labels)
    if labels.shape[0] != cluster_ids.shape[0]:
        return cluster_ids
    try:
        labels_float = labels.astype(float)
        mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return cluster_ids[mask]


def _build_trial_table(trials, trial_idx):
    contrast_left = np.asarray(trials["contrastLeft"], dtype=float)
    contrast_right = np.asarray(trials["contrastRight"], dtype=float)
    trial_contrasts = np.nanmax(np.vstack([np.abs(contrast_left), np.abs(contrast_right)]), axis=0)
    trial_contrasts = np.where(np.isnan(trial_contrasts), 0, trial_contrasts)
    reaction_time = (
        np.asarray(trials["response_times"], dtype=float)
        - np.asarray(trials["stimOn_times"], dtype=float)
    )
    choice_map = {1: "Left", -1: "Right", 0: "NoGo"}
    subject_response = choice_map.get(int(np.asarray(trials["choice"])[trial_idx]), "NA")
    correct_response = bool(np.asarray(trials["feedbackType"])[trial_idx] == 1)
    return pd.DataFrame(
        {
            "Trial": [int(trial_idx)],
            "Contrast": [float(trial_contrasts[trial_idx])],
            "Reaction Time (s)": [float(reaction_time[trial_idx])],
            "Correct Response": [correct_response],
            "Subject Response": [subject_response],
        }
    )


def _build_delay_event_inputs(sl, passive_visual_times, passive_visual_contrasts):
    trials = sl.trials
    trial_contrasts = ana_utils.get_trial_contrasts(sl)

    events_by_name = {}
    contrasts_by_name = {}
    trial_idx_by_name = {}

    stim_times_all = np.asarray(trials["stimOn_times"], dtype=float)
    stim_valid = np.isfinite(stim_times_all)
    stim_nonzero = np.isfinite(trial_contrasts) & (trial_contrasts > 0)
    stim_mask = stim_valid & stim_nonzero
    stim_idx = np.nonzero(stim_mask)[0]
    events_by_name["stimOn_times"] = stim_times_all[stim_mask]
    contrasts_by_name["stimOn_times"] = trial_contrasts[stim_idx]
    trial_idx_by_name["stimOn_times"] = stim_idx

    passive_times = np.asarray(passive_visual_times, dtype=float).reshape(-1)
    passive_contrasts = np.asarray(passive_visual_contrasts, dtype=float).reshape(-1)
    if passive_contrasts.shape[0] != passive_times.shape[0]:
        if (
            passive_contrasts.size == 1
            and passive_times.size > 0
            and np.isfinite(float(passive_contrasts.ravel()[0]))
        ):
            passive_contrasts = np.full(
                passive_times.shape[0],
                float(passive_contrasts.ravel()[0]),
                dtype=float,
            )
        else:
            passive_contrasts = np.ones_like(passive_times, dtype=float)

    passive_mask = np.isfinite(passive_times) & np.isfinite(passive_contrasts) & (passive_contrasts > 0)
    passive_times = passive_times[passive_mask]
    passive_contrasts = passive_contrasts[passive_mask]
    if passive_times.size > 0:
        order = np.argsort(passive_times)
        passive_times = passive_times[order]
        passive_contrasts = passive_contrasts[order]

    events_by_name["passive_visual_times"] = passive_times
    contrasts_by_name["passive_visual_times"] = passive_contrasts
    trial_idx_by_name["passive_visual_times"] = np.arange(len(passive_times), dtype=int)

    return events_by_name, contrasts_by_name, trial_idx_by_name


def _passive_visual_event_counts(passive_visual_times, passive_visual_contrasts):
    times = np.asarray(passive_visual_times, dtype=float).reshape(-1)
    contrasts = np.asarray(passive_visual_contrasts, dtype=float).reshape(-1)
    if contrasts.shape[0] != times.shape[0]:
        contrasts = np.ones_like(times, dtype=float)
    valid = np.isfinite(times) & np.isfinite(contrasts) & (contrasts > 0)
    return {"visual_nonzero": int(valid.sum())}


def _has_passive_visual_part(passive_visual_times, passive_visual_contrasts):
    counts = _passive_visual_event_counts(passive_visual_times, passive_visual_contrasts)
    return counts["visual_nonzero"] > 0, counts


def _build_event_session(event_times, event_name):
    return {"trials": {event_name: np.asarray(event_times, dtype=float)}}


def _pearsonr_with_n(x, y, min_n=2):
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


def _spearmanr_with_n(x, y, min_n=2):
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


def _build_corr_heatmap_fig(corr_mat, n_mat, names, title, template):
    text_mat = np.empty(corr_mat.shape, dtype=object)
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            val = corr_mat[i, j]
            n_val = int(n_mat[i, j])
            if np.isfinite(val):
                text_mat[i, j] = f"{val:.2f}<br>(n={n_val})"
            else:
                text_mat[i, j] = f"nan<br>(n={n_val})"
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
        title=title,
        width=1150,
        height=900,
        template=template,
        margin=dict(l=90, r=30, t=90, b=90),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def _build_multi_event_population_panel(
    event_specs,
    event_sessions,
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    df_res,
    plot_config,
    sort_mode,
    region_name,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
):
    n_events = len(event_specs)
    n_cols = 3
    n_rows = int(np.ceil(n_events / n_cols))
    fig_panel = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[label for label, _event in event_specs],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
    )

    for idx, (_event_label, event_name) in enumerate(event_specs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        event_session = event_sessions.get(event_name)
        if event_session is None:
            continue

        cfg = dict(plot_config)
        cfg["PLOT_EVENT"] = event_name
        pop_window_pre = float(cfg.get("POP_WINDOW_PRE", 0.05))
        pop_window_post = float(cfg.get("POP_WINDOW_POST", 0.15))
        cfg["POP_WINDOW_PRE"] = pop_window_pre
        cfg["POP_WINDOW_POST"] = pop_window_post
        sort_mode_event = sort_mode
        if event_name == "passive_visual_times":
            # Force passive visual panel to use the exact Stim On neuron order.
            sort_mode_event = "delay:stimOn_times"
        fig_event = plot_population_sorted_plotly(
            event_session,
            spikes,
            clusters,
            plot_cluster_ids,
            plot_cluster_acronyms,
            df_res,
            cfg,
            df_coupling=df_coupling,
            df_coupling_task=df_coupling_task,
            df_coupling_iti=df_coupling_iti,
            df_firing_rate=df_firing_rate,
            region_acronyms=[region_name],
            sort_mode=sort_mode_event,
        )
        if fig_event is None or len(fig_event.data) == 0:
            continue

        for trace in fig_event.data:
            trace_copy = go.Figure(data=[trace]).data[0]
            if isinstance(trace_copy, go.Heatmap):
                trace_copy.showscale = idx == 0
                if idx == 0:
                    trace_copy.colorbar = dict(title="Norm FR", len=0.8, y=0.5)
            else:
                trace_copy.showlegend = False
            fig_panel.add_trace(trace_copy, row=row, col=col)

        fig_panel.add_vline(
            x=0,
            row=row,
            col=col,
            line=dict(color="black", dash="dash"),
        )
        fig_panel.update_xaxes(range=[-pop_window_pre, pop_window_post], row=row, col=col)
        fig_panel.update_yaxes(autorange="reversed", row=row, col=col)

    fig_panel.update_layout(
        title=f"Response Analysis (Region {region_name})",
        width=1500,
        height=350 * n_rows + 140,
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        margin=dict(l=70, r=40, t=90, b=70),
    )
    fig_panel.update_xaxes(title_text="Time from event (s)", row=n_rows, col=1)
    return fig_panel


_set_plotly_renderer(PLOTLY_RENDERER)


# %% PID and ONE session loading
PID = "27bac116-ea57-4512-ad35-714a62d259cd"  # None for selection
TARGET_REGION = "VISp"
ONE_PREFERRED_MODE = "remote"  # "local" or "remote"
ALLOW_REMOTE_FALLBACK = True

LOAD_WHEEL = False
LOAD_POSE = False
LOAD_MOTION_ENERGY = False
LOAD_PUPIL = False

path_data, _path_fig, path_data_processed, ibl_cache = setup_paths(BASE_PATH)
one, one_mode = _init_one_with_fallback(
    ibl_cache,
    preferred_mode=ONE_PREFERRED_MODE,
    allow_remote=ALLOW_REMOTE_FALLBACK,
)
print(f"Using ONE mode: {one_mode}")

if PID is None:
    pid_list = list_cached_pids()
    if not pid_list:
        raise RuntimeError(
            "Set PID manually. No cached PID list found in data/dashboard_cache."
        )
    pid = choose_pid(pid_list, default_index=0)
else:
    pid = PID

print(f"Selected PID: {pid}")

ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)
ssl, spikes, clusters, sl = load_session_data(
    pid,
    one,
    ba=ba,
    load_wheel=LOAD_WHEEL,
    load_pose=LOAD_POSE,
    load_motion_energy=LOAD_MOTION_ENERGY,
    load_pupil=LOAD_PUPIL,
)
eid = getattr(ssl, "eid", None)
if eid is None:
    eid, _ = one.pid2eid(pid)
print(f"EID: {eid}")

cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
cluster_acronyms_calc = map_acronyms(clusters, br, "Beryl")
cluster_acronyms_plot = np.asarray(cluster_acronyms_calc).astype(str)
target_region_mask = np.asarray(
    [str(acr).startswith(str(TARGET_REGION)) for acr in cluster_acronyms_plot],
    dtype=bool,
)
target_region_cluster_ids = np.asarray(cluster_ids)[target_region_mask]
target_region_cluster_id_set = set(int(cid) for cid in target_region_cluster_ids.tolist())
target_region_cid_to_idx = {
    int(cid): cid_to_idx[int(cid)]
    for cid in target_region_cluster_ids
    if int(cid) in cid_to_idx
}
if target_region_cluster_ids.size == 0:
    raise RuntimeError(
        f"No clusters from region '{TARGET_REGION}' were found for PID {pid} (EID {eid})."
    )

try:
    one_local = init_one(ibl_cache, mode="local")
except Exception:
    one_local = one
try:
    one_remote = init_one(ibl_cache, mode="remote") if ALLOW_REMOTE_FALLBACK else None
except Exception:
    one_remote = None

visual_TR, _ = load_task_replay_datasets(
    eid,
    one_local,
    one_remote,
    allow_remote=ALLOW_REMOTE_FALLBACK,
)
passive_visual_times_raw, passive_visual_contrasts_raw = extract_passive_times_and_contrast(visual_TR)
has_passive_visual, passive_visual_counts = _has_passive_visual_part(
    passive_visual_times_raw,
    passive_visual_contrasts_raw,
)
print(
    "Passive visual events (contrast>0):",
    passive_visual_counts,
)
if not has_passive_visual:
    raise RuntimeError(
        f"Selected PID {pid} (EID {eid}) has no non-zero passive visual events. "
        "Choose a PID that includes passiveGabor data."
    )

passive_visual_times = np.asarray(passive_visual_times_raw, dtype=float).reshape(-1)
passive_visual_contrasts = np.asarray(passive_visual_contrasts_raw, dtype=float).reshape(-1)
if passive_visual_contrasts.shape[0] != passive_visual_times.shape[0]:
    passive_visual_contrasts = np.ones_like(passive_visual_times, dtype=float)
passive_visual_mask = (
    np.isfinite(passive_visual_times)
    & np.isfinite(passive_visual_contrasts)
    & (passive_visual_contrasts > 0)
)
passive_visual_times = passive_visual_times[passive_visual_mask]
passive_visual_contrasts = passive_visual_contrasts[passive_visual_mask]
if passive_visual_times.size > 0:
    _passive_order = np.argsort(passive_visual_times)
    passive_visual_times = passive_visual_times[_passive_order]
    passive_visual_contrasts = passive_visual_contrasts[_passive_order]

passive_event_times = {"passive_visual": passive_visual_times}


# %% CONFIG (analysis + plotting)
DELAY_EVENT_NAMES = [
    "stimOn_times",
    "passive_visual_times",
]

CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_LABEL_MIN": 0.9,  # requested default for calculations
    "CALC_SPONT": True,
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    "DELAY_METHOD": "center_of_mass",  # "center_of_mass" or "psth_peak"
    "DELAY_UNITS": "ms",
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    "DELAY_WINDOWS": {
        "stimOn_times": (0.01, 0.2),
        "passive_visual_times": (0.01, 0.2),
    },
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -1.0,
    "PSTH_WINDOW_END": 1.0,
    "RESPONSIVE_WINDOW_START": 0.01,
    "RESPONSIVE_WINDOW_END": 0.2,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 10,
    "MIN_TRIALS_SPLIT": 5,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_LOW_PASS_ORDER": 3,
    "STPR_POP_USE_GOOD_UNITS": False,
    "TASK_POST_EVENT_S": 1.0,
    "ITI_SKIP_FIRST_LAST": True,
}

CONFIG_PLOT = {
    "ATLAS_MAPPING": "Beryl",
    "PLOT_ONLY_GOOD_UNITS": False,
    "PLOT_EVENT": "stimOn_times",
    "PLOT_REGIONS": [TARGET_REGION],
    "RASTER_WINDOW_PRE": 1,
    "RASTER_WINDOW_POST": 2,
    "RASTER_ALIGN_TO_EVENT": True,
    "RASTER_ALIGN_TO_STIM_ON": True,
    "SINGLE_NEURON_RASTER_PRE": 0.5,
    "SINGLE_NEURON_RASTER_POST": 1.0,
    "SINGLE_NEURON_BIN_SIZE": 0.05,
    "SINGLE_NEURON_SMOOTH_SIGMA": 1,
    "SEQUENCE_WINDOW_PRE": 0.5,
    "SEQUENCE_WINDOW_POST": 1.0,
    "SEQUENCE_ALIGN_TO_EVENT": True,
    "SEQUENCE_ALIGN_TO_STIM": True,
    "POP_WINDOW_PRE": 0.05,
    "POP_WINDOW_POST": 0.15,
    "POP_BIN_SIZE": 0.005,
    "POP_SMOOTH_SIGMA": 2,
    "POP_CMAP_NAME": "bwr",
    "POP_NORMALIZE": True,
    "SORT_BY_SPONT": True,
}

PLOT_DARK_THEME = False
PLOT_LABEL_MIN = float(CONFIG_CALC.get("CALC_LABEL_MIN", 0.5))
CORR_MIN_N = 2

TRIAL_INDEX = 293 
TRIAL_RASTER_SORT = "Spont stPR Delay"

GENERAL_RASTER_START = 4543  # defaults to first spike time
GENERAL_RASTER_END = 4549  # defaults to +10s from start
GENERAL_RASTER_SORT = "Spont stPR Delay"

HEATMAP_SORT = "Own Event Delay" # "Own Event Delay"

RASTER_SORT_MAP = {
    "Default (Depth)": "depth",
    "Delay to Stim On": "delay:stimOn_times",
    "Delay to Passive Visual": "delay:passive_visual_times",
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

HEATMAP_SORT_MAP = {
    "Own Event Delay": "delay",
    "Default (Depth)": "depth",
    "Delay to Stim On": "delay:stimOn_times",
    "Delay to Passive Visual": "delay:passive_visual_times",
    "Task stPR Delay": "task",
    "Task stPR Strength": "task_strength",
    "ITI stPR Delay": "iti",
    "ITI stPR Strength": "iti_strength",
    "Spont stPR Delay": "spont",
    "Spont stPR Strength": "spont_strength",
    "Firing rate": "firing_rate",
}

plot_config = dict(CONFIG_PLOT)
plot_config["PSTH_WINDOW_START"] = CONFIG_CALC.get("PSTH_WINDOW_START", -0.2)
plot_config["PSTH_WINDOW_END"] = CONFIG_CALC.get("PSTH_WINDOW_END", 0.35)
plot_config["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
plot_config["SINGLE_NEURON_SMOOTH_SIGMA"] = 0.5
plot_config["SINGLE_NEURON_BIN_SIZE"] = 0.01
plot_config["PLOT_LABEL_MIN"] = PLOT_LABEL_MIN
plot_config["DELAY_UNITS"] = CONFIG_CALC.get("DELAY_UNITS", "s")
plot_config["PLOTLY_TEMPLATE"] = "plotly_dark" if PLOT_DARK_THEME else "plotly_white"
plotting_utils.DEFAULT_TEMPLATE = plot_config["PLOTLY_TEMPLATE"]
pio.templates.default = plot_config["PLOTLY_TEMPLATE"]


# %% Calculations (delays + stPR). No dashboard cache loading.
events_by_name, contrasts_by_name, trial_idx_by_name = _build_delay_event_inputs(
    sl,
    passive_visual_times,
    passive_visual_contrasts,
)
print("Delay event counts:", {k: len(v) for k, v in events_by_name.items()})

delay_config = dict(CONFIG_CALC)
delay_config["EVENT_NAMES"] = list(DELAY_EVENT_NAMES)
df_res = ana_utils.calculate_event_delays(
    spikes,
    clusters,
    cluster_acronyms_calc,
    events_by_name,
    delay_config,
    target_region_cid_to_idx,
    contrasts_by_name=contrasts_by_name,
    trial_idx_by_name=trial_idx_by_name,
    include_splits=True,
    output_path=path_data_processed / f"{pid}_passive_visual_delay_results.csv",
)

passive_delay_cols = [
    ana_utils.delay_column_name("passive_visual_times"),
]
for col in passive_delay_cols:
    if col not in df_res.columns:
        df_res[col] = np.nan
passive_arr = df_res[passive_delay_cols].to_numpy(dtype=float)
with np.errstate(invalid="ignore"):
    df_res["delay_passive_all_times"] = np.nanmean(passive_arr, axis=1)

calc_label_min = CONFIG_CALC.get("CALC_LABEL_MIN", None)
if calc_label_min is None and CONFIG_CALC.get("CALC_ONLY_GOOD_UNITS", False):
    calc_label_min = 1.0
calc_cluster_ids = _select_cluster_ids_by_label(
    cluster_ids,
    clusters,
    label_min=calc_label_min,
)
calc_cluster_ids = np.asarray(
    [cid for cid in calc_cluster_ids if int(cid) in target_region_cluster_id_set],
    dtype=np.asarray(cluster_ids).dtype,
)

spont_intervals = _load_spontaneous_intervals(one, eid) if CONFIG_CALC.get("CALC_SPONT", True) else None
spont_interval_list = []
if spont_intervals is not None:
    spont_intervals = np.asarray(spont_intervals, dtype=float)
    spont_interval_list = [tuple(row) for row in spont_intervals]

df_coupling = None
if CONFIG_CALC.get("CALC_SPONT", True) and spont_interval_list:
    spikes_spont = ana_utils.slice_spikes_by_intervals(spikes, spont_interval_list)
    df_coupling = ana_utils.compute_population_coupling(
        spikes_spont,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=True,
        intervals=spont_interval_list,
        context_label="Spont",
    )

task_windows = ana_utils.build_task_window_table(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
if not task_windows.empty:
    task_odd_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy()
    task_even_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy()
else:
    task_odd_intervals = np.empty((0, 2))
    task_even_intervals = np.empty((0, 2))

df_task_odd = None
df_task_even = None
if len(task_odd_intervals) > 0:
    spikes_task_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_task_odd = ana_utils.compute_population_coupling(
        spikes_task_odd,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=task_odd_intervals,
        context_label="Task odd",
    )
if len(task_even_intervals) > 0:
    spikes_task_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_task_even = ana_utils.compute_population_coupling(
        spikes_task_even,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=task_even_intervals,
        context_label="Task even",
    )
if df_task_odd is not None and df_task_odd.empty:
    df_task_odd = None
if df_task_even is not None and df_task_even.empty:
    df_task_even = None
df_coupling_task = ana_utils.merge_stpr_splits(
    df_task_odd,
    df_task_even,
    CONFIG_CALC,
    split_a="odd",
    split_b="even",
) if (df_task_odd is not None or df_task_even is not None) else None

trial_end_times = ana_utils.compute_trial_end_times(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
stim_on_times = np.asarray(sl.trials["stimOn_times"], dtype=float)
iti_windows = ana_utils.build_iti_windows(
    trial_end_times,
    stim_on_times,
    skip_first_last=CONFIG_CALC["ITI_SKIP_FIRST_LAST"],
)
if not iti_windows.empty:
    iti_odd_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy()
    iti_even_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy()
else:
    iti_odd_intervals = np.empty((0, 2))
    iti_even_intervals = np.empty((0, 2))

df_iti_odd = None
df_iti_even = None
if len(iti_odd_intervals) > 0:
    spikes_iti_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_iti_odd = ana_utils.compute_population_coupling(
        spikes_iti_odd,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=iti_odd_intervals,
        context_label="ITI odd",
    )
if len(iti_even_intervals) > 0:
    spikes_iti_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_iti_even = ana_utils.compute_population_coupling(
        spikes_iti_even,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=iti_even_intervals,
        context_label="ITI even",
    )
if df_iti_odd is not None and df_iti_odd.empty:
    df_iti_odd = None
if df_iti_even is not None and df_iti_even.empty:
    df_iti_even = None
df_coupling_iti = ana_utils.merge_stpr_splits(
    df_iti_odd,
    df_iti_even,
    CONFIG_CALC,
    split_a="odd",
    split_b="even",
) if (df_iti_odd is not None or df_iti_even is not None) else None

cluster_firing_rate = _get_cluster_firing_rate(clusters, cluster_ids)
df_firing_rate = None
if cluster_firing_rate is not None and cluster_ids is not None:
    df_firing_rate = pd.DataFrame(
        {
            "cluster_id": np.asarray(cluster_ids),
            "firing_rate_h1": np.asarray(cluster_firing_rate, dtype=float),
            "firing_rate_h2": np.asarray(cluster_firing_rate, dtype=float),
        }
    )

plot_cluster_ids = _select_cluster_ids_by_label(
    cluster_ids,
    clusters,
    label_min=PLOT_LABEL_MIN,
)
plot_cluster_ids = np.asarray(
    [cid for cid in plot_cluster_ids if int(cid) in target_region_cluster_id_set],
    dtype=np.asarray(cluster_ids).dtype,
)
plot_cluster_acronyms = np.asarray(
    [cluster_acronyms_plot[cid_to_idx[int(cid)]] for cid in plot_cluster_ids],
    dtype=str,
)

df_res_plot = df_res[df_res["cluster_id"].isin(plot_cluster_ids)].copy()
df_coupling_plot = (
    df_coupling[df_coupling["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling is not None
    else None
)
df_coupling_task_plot = (
    df_coupling_task[df_coupling_task["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling_task is not None
    else None
)
df_coupling_iti_plot = (
    df_coupling_iti[df_coupling_iti["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling_iti is not None
    else None
)
if df_firing_rate is not None:
    df_firing_rate = df_firing_rate[df_firing_rate["cluster_id"].isin(plot_cluster_ids)].copy()

print(f"Delay table shape: {df_res_plot.shape}")
if df_coupling_plot is not None:
    print(f"Spont stPR shape: {df_coupling_plot.shape}")
if df_coupling_task_plot is not None:
    print(f"Task stPR shape: {df_coupling_task_plot.shape}")
if df_coupling_iti_plot is not None:
    print(f"ITI stPR shape: {df_coupling_iti_plot.shape}")


# %% Trial raster (specific trial, 04-style)
trial_sort_metric = RASTER_SORT_MAP.get(TRIAL_RASTER_SORT, "depth")
n_trials = len(sl.trials["stimOn_times"])
trial_idx = 328 # int(np.clip(int(TRIAL_INDEX), 0, max(0, n_trials - 1)))
display(_build_trial_table(sl.trials, trial_idx))

fig_trial = plot_trial_raster_plotly(
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    sl,
    plot_config,
    trial_idx,
    sorting_metric=trial_sort_metric,
    variability_metric="fano",
    df_res=df_res_plot,
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    df_coupling_iti=df_coupling_iti_plot,
    df_firing_rate=df_firing_rate,
)
show_fig(fig_trial)


# %% General raster (time window, include passive events)
general_sort_metric = RASTER_SORT_MAP.get(GENERAL_RASTER_SORT, "depth")
spike_times = np.asarray(spikes["times"], dtype=float)
if spike_times.size == 0:
    raise RuntimeError("No spikes found for this PID.")

min_t = float(np.nanmin(spike_times))
max_t = float(np.nanmax(spike_times))
t_start = min_t if GENERAL_RASTER_START is None else float(GENERAL_RASTER_START)
t_end = min(min_t + 10.0, max_t) if GENERAL_RASTER_END is None else float(GENERAL_RASTER_END)
t_start = max(min_t, t_start)
t_end = min(max_t, t_end)

passive_event_styles = {
    "passive_visual": ("Passive Visual", "#17becf", "dot"),
}

fig_general = plot_time_window_raster_plotly(
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    sl,
    plot_config,
    t_start,
    t_end,
    sorting_metric=general_sort_metric,
    variability_metric="fano",
    df_res=df_res_plot,
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    df_coupling_iti=df_coupling_iti_plot,
    df_firing_rate=df_firing_rate,
    extra_event_times=passive_event_times or None,
    extra_event_styles=passive_event_styles,
)
show_fig(fig_general)


# %% Response heatmaps (task + passive visual)
heatmap_sort_mode = HEATMAP_SORT_MAP.get(HEATMAP_SORT, "delay")
heatmap_plot_config = dict(plot_config)
heatmap_plot_config["POP_WINDOW_PRE"] = 0.1
heatmap_plot_config["POP_WINDOW_POST"] = 0.2
heatmap_event_specs = [
    ("Stim On", "stimOn_times"),
    ("Passive Visual (Stim On sort)", "passive_visual_times"),
]

event_sessions = {}
for _label, event_name in heatmap_event_specs:
    event_sessions[event_name] = _build_event_session(
        events_by_name.get(event_name, np.array([])),
        event_name,
    )

unique_regions = sorted(pd.Series(plot_cluster_acronyms).astype(str).unique().tolist())
plot_regions = plot_config.get("PLOT_REGIONS")
if plot_regions:
    selected_regions = []
    for region_prefix in plot_regions:
        matches = [reg for reg in unique_regions if str(reg).startswith(str(region_prefix))]
        if matches:
            selected_regions.extend(matches)
    selected_regions = list(dict.fromkeys(selected_regions))
else:
    selected_regions = unique_regions

if not selected_regions:
    print("No plotting regions matched current filters.")
else:
    for region_name in selected_regions:
        fig_panel = _build_multi_event_population_panel(
            heatmap_event_specs,
            event_sessions,
            spikes,
            clusters,
            plot_cluster_ids,
            plot_cluster_acronyms,
            df_res_plot,
            heatmap_plot_config,
            sort_mode=heatmap_sort_mode,
            region_name=region_name,
            df_coupling=df_coupling_plot,
            df_coupling_task=df_coupling_task_plot,
            df_coupling_iti=df_coupling_iti_plot,
            df_firing_rate=df_firing_rate,
        )
        show_fig(fig_panel)

# %% Correlation matrices (Pearson + Spearman) per selected region
CORR_VARIABLE_SPECS = [
    {
        "key": "delay_stim",
        "name": "Delay (Stim On)",
        "df": "df_res",
        "v1": "delay_stimOn_times_odd",
        "v2": "delay_stimOn_times_even",
    },
    {
        "key": "delay_passive_visual",
        "name": "Delay (Passive Visual)",
        "df": "df_res",
        "v1": "delay_passive_visual_times_odd",
        "v2": "delay_passive_visual_times_even",
    },
    {
        "key": "stpr_delay_spont",
        "name": "stPR Delay (Spont)",
        "df": "df_coupling",
        "v1": "coupling_delay_ms_h1",
        "v2": "coupling_delay_ms_h2",
    },
    {
        "key": "stpr_delay_task",
        "name": "stPR Delay (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "key": "stpr_delay_iti",
        "name": "stPR Delay (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "key": "stpr_strength_spont",
        "name": "stPR Strength (Spont)",
        "df": "df_coupling",
        "v1": "coupling_strength_h1",
        "v2": "coupling_strength_h2",
    },
    {
        "key": "stpr_strength_task",
        "name": "stPR Strength (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "key": "stpr_strength_iti",
        "name": "stPR Strength (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
]

source_tables = {
    "df_res": df_res_plot,
    "df_coupling": df_coupling_plot,
    "df_coupling_task": df_coupling_task_plot,
    "df_coupling_iti": df_coupling_iti_plot,
}

region_lookup = pd.DataFrame(
    {
        "cluster_id": plot_cluster_ids,
        "region": plot_cluster_acronyms,
    }
)
region_lookup = region_lookup[~region_lookup["region"].isin(["root", "void"])].copy()


def _corr_half_cols(spec):
    return f"__{spec['key']}_h1", f"__{spec['key']}_h2"


def _build_corr_variable_table(df_src, spec, region_df):
    if df_src is None:
        return None
    v1 = spec["v1"]
    v2 = spec["v2"]
    if v1 not in df_src.columns or v2 not in df_src.columns:
        return None
    df_var = df_src[["cluster_id", v1, v2]].copy()
    df_var = df_var.groupby("cluster_id", as_index=False).mean(numeric_only=True)
    if region_df is not None:
        df_var = df_var.merge(region_df[["cluster_id", "region"]], on="cluster_id", how="inner")
    vals_1 = df_var[v1].to_numpy(dtype=float)
    vals_2 = df_var[v2].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(vals_1) & np.isfinite(vals_2)
    mean_vals[valid] = (vals_1[valid] + vals_2[valid]) / 2.0
    df_var = df_var.rename(columns={v1: "half_1", v2: "half_2"})
    df_var["mean"] = mean_vals
    return df_var


if region_lookup.empty:
    print("No neurons available for correlation analysis.")
else:
    var_tables_all = {}
    for spec in CORR_VARIABLE_SPECS:
        df_var_all = _build_corr_variable_table(
            source_tables.get(spec["df"]),
            spec,
            region_lookup,
        )
        if df_var_all is not None and not df_var_all.empty:
            var_tables_all[spec["name"]] = df_var_all

    for region_name in selected_regions:
        region_ids = region_lookup.loc[
            region_lookup["region"] == region_name, "cluster_id"
        ].to_numpy()
        if len(region_ids) == 0:
            continue

        var_tables_region = {}
        for spec in CORR_VARIABLE_SPECS:
            name = spec["name"]
            df_var_all = var_tables_all.get(name)
            if df_var_all is None:
                continue
            df_var_reg = df_var_all[df_var_all["region"] == region_name].copy()
            if not df_var_reg.empty:
                var_tables_region[name] = df_var_reg

        available_specs = [spec for spec in CORR_VARIABLE_SPECS if spec["name"] in var_tables_region]
        available_names = [spec["name"] for spec in available_specs]
        if len(available_names) < 2:
            print(f"Region {region_name}: not enough variables with finite data.")
            continue

        rel_pearson = {}
        rel_spearman = {}
        rel_n_pearson = {}
        rel_n_spearman = {}
        for spec in available_specs:
            name = spec["name"]
            df_var = var_tables_region[name]
            r_p, n_p = _pearsonr_with_n(df_var["half_1"], df_var["half_2"], min_n=CORR_MIN_N)
            r_s, n_s = _spearmanr_with_n(df_var["half_1"], df_var["half_2"], min_n=CORR_MIN_N)
            rel_pearson[name] = r_p
            rel_spearman[name] = r_s
            rel_n_pearson[name] = n_p
            rel_n_spearman[name] = n_s

        mean_wide = pd.DataFrame({"cluster_id": region_ids})
        for spec in available_specs:
            name = spec["name"]
            mean_wide = mean_wide.merge(
                var_tables_region[name][["cluster_id", "mean"]].rename(columns={"mean": name}),
                on="cluster_id",
                how="left",
            )

        n_vars = len(available_names)
        pearson_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
        pearson_text = np.empty((n_vars, n_vars), dtype=object)
        n_mat_p = np.zeros((n_vars, n_vars), dtype=int)
        spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
        spearman_text = np.empty((n_vars, n_vars), dtype=object)
        n_mat_s = np.zeros((n_vars, n_vars), dtype=int)

        for i, name_i in enumerate(available_names):
            for j, name_j in enumerate(available_names):
                if i == j:
                    r_p = rel_pearson.get(name_i, np.nan)
                    n_p = rel_n_pearson.get(name_i, 0)
                    r_s = rel_spearman.get(name_i, np.nan)
                    n_s = rel_n_spearman.get(name_i, 0)
                    pearson_mat[i, j] = r_p
                    n_mat_p[i, j] = int(n_p)
                    pearson_text[i, j] = (
                        f"rel={r_p:.2f}<br>(n={n_p})"
                        if np.isfinite(r_p)
                        else f"rel=nan<br>(n={n_p})"
                    )
                    spearman_mat[i, j] = r_s
                    n_mat_s[i, j] = int(n_s)
                    spearman_text[i, j] = (
                        f"rel={r_s:.2f}<br>(n={n_s})"
                        if np.isfinite(r_s)
                        else f"rel=nan<br>(n={n_s})"
                    )
                else:
                    x = mean_wide[name_i].to_numpy(dtype=float)
                    y = mean_wide[name_j].to_numpy(dtype=float)
                    r_p, n_p = _pearsonr_with_n(x, y, min_n=CORR_MIN_N)
                    r_s, n_s = _spearmanr_with_n(x, y, min_n=CORR_MIN_N)
                    pearson_mat[i, j] = r_p
                    n_mat_p[i, j] = int(n_p)
                    pearson_text[i, j] = (
                        f"r={r_p:.2f}<br>(n={n_p})"
                        if np.isfinite(r_p)
                        else f"r=nan<br>(n={n_p})"
                    )
                    spearman_mat[i, j] = r_s
                    n_mat_s[i, j] = int(n_s)
                    spearman_text[i, j] = (
                        f"rho={r_s:.2f}<br>(n={n_s})"
                        if np.isfinite(r_s)
                        else f"rho=nan<br>(n={n_s})"
                    )

        fig_p = go.Figure(
            data=go.Heatmap(
                z=pearson_mat,
                x=available_names,
                y=available_names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=pearson_text,
                texttemplate="%{text}",
                hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
            )
        )
        fig_p.update_layout(
            title=(
                "Reliability (diag) + Pairwise Pearson (off-diag) | "
                f"Region {region_name} | N units={len(region_ids)} | min_n={CORR_MIN_N}"
            ),
            width=1150,
            height=900,
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig_p.update_xaxes(tickangle=45)
        show_fig(fig_p)

        fig_s = go.Figure(
            data=go.Heatmap(
                z=spearman_mat,
                x=available_names,
                y=available_names,
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
                f"Region {region_name} | N units={len(region_ids)} | min_n={CORR_MIN_N}"
            ),
            width=1150,
            height=900,
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig_s.update_xaxes(tickangle=45)
        show_fig(fig_s)



# %% Variable correlation scatter (interactive variable choice, 04-dashboard style)
def _choose_option(options, prompt, default_value):
    options = list(options or [])
    if not options:
        raise ValueError("No options available.")
    if default_value not in options:
        default_value = options[0]
    default_idx = options.index(default_value)
    print(prompt)
    for idx, name in enumerate(options):
        print(f"  [{idx}] {name}")
    try:
        selection = input(
            f"Choose index or exact name (Enter for [{default_idx}] {default_value}): "
        ).strip()
    except Exception:
        return default_value
    if selection == "":
        return default_value
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(options):
            return options[idx]
        print(f"Index {idx} out of range. Using default: {default_value}")
        return default_value
    if selection in options:
        return selection
    lower_map = {str(opt).lower(): opt for opt in options}
    if selection.lower() in lower_map:
        return lower_map[selection.lower()]
    print(f"Unknown selection '{selection}'. Using default: {default_value}")
    return default_value


def _format_corr_value(val):
    return "nan" if not np.isfinite(val) else f"{val:.3f}"


if region_lookup.empty:
    print("No neurons available for variable correlation scatter.")
else:
    scatter_regions = (
        selected_regions
        if "selected_regions" in globals() and len(selected_regions) > 0
        else sorted(region_lookup["region"].unique().tolist())
    )
    region_lookup_scatter = region_lookup[region_lookup["region"].isin(scatter_regions)].copy()
    merged_scatter = region_lookup_scatter[["cluster_id", "region"]].drop_duplicates().copy()

    available_specs_scatter = []
    for spec in CORR_VARIABLE_SPECS:
        df_var = _build_corr_variable_table(
            source_tables.get(spec["df"]),
            spec,
            region_lookup_scatter,
        )
        if df_var is None or df_var.empty:
            continue
        name = spec["name"]
        tmp = df_var[["cluster_id", "mean"]].rename(columns={"mean": name})
        merged_scatter = merged_scatter.merge(tmp, on="cluster_id", how="left")
        if np.isfinite(tmp[name].to_numpy(dtype=float)).any():
            available_specs_scatter.append(spec)

    available_var_names = [spec["name"] for spec in available_specs_scatter if spec["name"] in merged_scatter.columns]

    if len(available_var_names) < 2:
        print("Not enough variables with finite data to draw a scatter plot.")
    else:
        default_x = "Delay (Stim On)"
        default_y = "stPR Delay (Task)"
        if default_x not in available_var_names:
            default_x = available_var_names[0]
        if default_y not in available_var_names:
            default_y = available_var_names[1] if len(available_var_names) > 1 else available_var_names[0]

        var_x_name = _choose_option(
            available_var_names,
            "Variable X options:",
            default_x,
        )
        var_y_name = _choose_option(
            available_var_names,
            "Variable Y options:",
            default_y,
        )
        if var_y_name == var_x_name:
            fallbacks = [name for name in available_var_names if name != var_x_name]
            if fallbacks:
                var_y_name = fallbacks[0]
                print(f"Variable Y matched X; using '{var_y_name}' for Y.")

        plot_df = merged_scatter[["cluster_id", "region", var_x_name, var_y_name]].copy()
        x_vals = plot_df[var_x_name].to_numpy(dtype=float)
        y_vals = plot_df[var_y_name].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        plot_df = plot_df.loc[valid_mask].reset_index(drop=True)

        if plot_df.empty:
            print(f"No overlapping finite values for '{var_x_name}' and '{var_y_name}'.")
        else:
            x_vals = plot_df[var_x_name].to_numpy(dtype=float)
            y_vals = plot_df[var_y_name].to_numpy(dtype=float)
            r_p, n_p = _pearsonr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)
            r_s, n_s = _spearmanr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)
            region_names = sorted(plot_df["region"].astype(str).unique().tolist())
            region_color_map = plotting_utils._region_color_map(region_names)

            fig_corr = go.Figure()
            for reg_name in region_names:
                reg_df = plot_df.loc[plot_df["region"].astype(str) == str(reg_name)].copy()
                if reg_df.empty:
                    continue
                fig_corr.add_trace(
                    go.Scatter(
                        x=reg_df[var_x_name].to_numpy(dtype=float),
                        y=reg_df[var_y_name].to_numpy(dtype=float),
                        mode="markers",
                        name=str(reg_name),
                        customdata=reg_df["cluster_id"].to_numpy(dtype=int),
                        marker=dict(
                            size=7,
                            opacity=0.7,
                            color=region_color_map.get(str(reg_name), "gray"),
                        ),
                        hovertemplate=(
                            "Region: %{fullData.name}<br>"
                            "Cluster ID: %{customdata}<br>"
                            f"{var_x_name}: %{{x:.3f}}<br>"
                            f"{var_y_name}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
            max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
            if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                fig_corr.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Unity line (y=x)",
                        line=dict(color="red", dash="dash", width=2),
                        hovertemplate=(
                            "Unity line<br>"
                            "x=%{x:.3f}<br>"
                            "y=%{y:.3f}<extra></extra>"
                        ),
                    )
                )

            fit_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_fit = x_vals[fit_mask]
            y_fit = y_vals[fit_mask]
            if x_fit.size >= 2 and np.nanstd(x_fit) > 0:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                x_line = np.array([float(np.nanmin(x_fit)), float(np.nanmax(x_fit))], dtype=float)
                y_line = slope * x_line + intercept
                fig_corr.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"Linear fit (y={slope:.2f}x+{intercept:.2f})",
                        line=dict(color="black", width=2),
                        hovertemplate=(
                            "Linear fit<br>"
                            "x=%{x:.3f}<br>"
                            "y=%{y:.3f}<extra></extra>"
                        ),
                    )
                )

            fig_corr.update_layout(
                title=(
                    "Variable Correlation | "
                    f"{var_x_name} vs {var_y_name}"
                ),
                template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                width=900,
                height=700,
                margin=dict(l=70, r=40, t=95, b=140),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                ),
            )
            fig_corr.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="left",
                bordercolor="rgba(120,120,120,0.6)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                text=(
                    f"Pearson r={_format_corr_value(r_p)} (n={n_p})<br>"
                    f"Spearman rho={_format_corr_value(r_s)} (n={n_s})"
                ),
            )
            fig_corr.update_xaxes(title_text=var_x_name)
            fig_corr.update_yaxes(title_text=var_y_name)
            show_fig(fig_corr)


# %% All PIDs with VISp
# %% All PIDs with VISp: discover PIDs using the 03-style region search
ALL_PID_TARGET_REGION = TARGET_REGION
ALL_PID_TAG = "2025_Q3_IBL_et_al_BWM"
_all_pid_label_min_raw = CONFIG_CALC.get("CALC_LABEL_MIN", 0.5)
ALL_PID_LABEL_MIN = 0.5 if _all_pid_label_min_raw is None else float(_all_pid_label_min_raw)
ALL_PID_OVERRIDE_PIDS = None  # Optional explicit PID list to skip search


def _get_pids_for_regions_like_03(one_client, regions, tag):
    if not regions:
        return []
    all_pids = []
    for region in regions:
        try:
            eids, _session_dicts = one_client.search(
                tag=tag,
                details=True,
                query_type="remote",
                atlas_acronym=region,
            )
        except Exception:
            eids, _session_dicts = one_client.search(
                tag=tag,
                details=True,
                atlas_acronym=region,
            )
        for eid in eids:
            try:
                insertions = one_client.alyx.rest("insertions", "list", session=eid)
            except Exception:
                insertions = []
            all_pids.extend(
                [ins["id"] for ins in insertions if isinstance(ins, dict) and "id" in ins]
            )
    return list(dict.fromkeys(all_pids))


def _pid_has_passive_visual_part(pid_value):
    eid_pid = None
    last_pid2eid_error = None
    for one_client in (one, one_remote, one_local):
        if one_client is None:
            continue
        try:
            eid_pid, _ = one_client.pid2eid(pid_value)
            if eid_pid is not None:
                break
        except Exception as exc:
            last_pid2eid_error = exc
    if eid_pid is None:
        return (
            False,
            {"visual_nonzero": 0},
            f"pid2eid failed: {last_pid2eid_error}",
        )

    try:
        visual_tr_pid, _ = load_task_replay_datasets(
            eid_pid,
            one_local,
            one_remote,
            allow_remote=ALLOW_REMOTE_FALLBACK,
        )
        passive_visual_times_pid, passive_visual_contrasts_pid = extract_passive_times_and_contrast(
            visual_tr_pid
        )
        has_passive, counts = _has_passive_visual_part(
            passive_visual_times_pid,
            passive_visual_contrasts_pid,
        )
        return has_passive, counts, None
    except Exception as exc:
        return False, {"visual_nonzero": 0}, str(exc)


pid_query_one = one_remote if "one_remote" in globals() and one_remote is not None else one
if ALL_PID_OVERRIDE_PIDS:
    all_region_pids = list(dict.fromkeys(ALL_PID_OVERRIDE_PIDS))
else:
    all_region_pids = _get_pids_for_regions_like_03(
        pid_query_one,
        [ALL_PID_TARGET_REGION],
        ALL_PID_TAG,
    )
print(
    f"Found {len(all_region_pids)} PIDs from region search for '{ALL_PID_TARGET_REGION}' "
    f"(tag='{ALL_PID_TAG}')."
)
if len(all_region_pids) > 0:
    print("First 10 PIDs:", all_region_pids[:10])

candidate_region_pids = list(all_region_pids)
all_region_pids = []
pids_without_passive = []
passive_check_failures = []
for pid_value in candidate_region_pids:
    has_passive, passive_counts, err_msg = _pid_has_passive_visual_part(pid_value)
    if err_msg is not None:
        passive_check_failures.append({"pid": pid_value, "error": err_msg})
        continue
    if has_passive:
        all_region_pids.append(pid_value)
    else:
        pids_without_passive.append(
            {
                "pid": pid_value,
                "visual_nonzero_events": passive_counts.get("visual_nonzero", 0),
            }
        )
print(
    f"PIDs with passive visual data: {len(all_region_pids)}/{len(candidate_region_pids)}."
)
if pids_without_passive:
    print(f"Skipped PIDs without passive visual events: {len(pids_without_passive)}")
    display(pd.DataFrame(pids_without_passive).head(20))
if passive_check_failures:
    print(f"PIDs skipped due to passive-data check errors: {len(passive_check_failures)}")
    display(pd.DataFrame(passive_check_failures).head(20))


# %% All PIDs with VISp: per-PID calculations, region filter, and merge neurons
def _compute_region_units_for_pid(pid_value, region_prefix, label_min):
    ssl_pid, spikes_pid, clusters_pid, sl_pid = load_session_data(
        pid_value,
        one,
        ba=ba,
        load_wheel=LOAD_WHEEL,
        load_pose=LOAD_POSE,
        load_motion_energy=LOAD_MOTION_ENERGY,
        load_pupil=LOAD_PUPIL,
    )
    eid_pid = getattr(ssl_pid, "eid", None)
    if eid_pid is None:
        eid_pid, _ = one.pid2eid(pid_value)

    visual_tr_pid, _ = load_task_replay_datasets(
        eid_pid,
        one_local,
        one_remote,
        allow_remote=ALLOW_REMOTE_FALLBACK,
    )
    passive_visual_times_pid, passive_visual_contrasts_pid = extract_passive_times_and_contrast(
        visual_tr_pid
    )
    has_passive_pid, _passive_counts_pid = _has_passive_visual_part(
        passive_visual_times_pid,
        passive_visual_contrasts_pid,
    )
    if not has_passive_pid:
        return pd.DataFrame()
    events_by_name_pid, contrasts_by_name_pid, trial_idx_by_name_pid = _build_delay_event_inputs(
        sl_pid,
        passive_visual_times_pid,
        passive_visual_contrasts_pid,
    )

    cluster_ids_pid, cid_to_idx_pid = build_cluster_id_map(clusters_pid)
    if cluster_ids_pid is None or len(cluster_ids_pid) == 0:
        return pd.DataFrame()
    cluster_ids_pid = np.asarray(cluster_ids_pid)

    cluster_acronyms_pid = np.asarray(map_acronyms(clusters_pid, br, "Beryl")).astype(str)
    region_by_cluster = np.asarray(
        [cluster_acronyms_pid[cid_to_idx_pid[int(cid)]] for cid in cluster_ids_pid],
        dtype=str,
    )

    label_cluster_ids = _select_cluster_ids_by_label(
        cluster_ids_pid,
        clusters_pid,
        label_min=label_min,
    )
    label_cluster_id_set = set(int(cid) for cid in np.asarray(label_cluster_ids).tolist())
    region_mask = np.asarray(
        [str(acr).startswith(str(region_prefix)) for acr in region_by_cluster],
        dtype=bool,
    )
    region_cluster_ids = np.asarray(
        [
            cid
            for cid, keep in zip(cluster_ids_pid, region_mask)
            if keep and int(cid) in label_cluster_id_set
        ],
        dtype=cluster_ids_pid.dtype,
    )
    if region_cluster_ids.size == 0:
        return pd.DataFrame()

    delay_config_pid = dict(CONFIG_CALC)
    delay_config_pid["EVENT_NAMES"] = list(DELAY_EVENT_NAMES)
    region_cid_to_idx_pid = {
        int(cid): cid_to_idx_pid[int(cid)]
        for cid in region_cluster_ids
        if int(cid) in cid_to_idx_pid
    }
    df_res_pid = ana_utils.calculate_event_delays(
        spikes_pid,
        clusters_pid,
        cluster_acronyms_pid,
        events_by_name_pid,
        delay_config_pid,
        region_cid_to_idx_pid,
        contrasts_by_name=contrasts_by_name_pid,
        trial_idx_by_name=trial_idx_by_name_pid,
        include_splits=True,
        output_path=path_data_processed / f"{pid_value}_passive_visual_delay_results.csv",
    )
    passive_delay_cols_pid = [
        ana_utils.delay_column_name("passive_visual_times"),
    ]
    for col in passive_delay_cols_pid:
        if col not in df_res_pid.columns:
            df_res_pid[col] = np.nan
    passive_arr_pid = df_res_pid[passive_delay_cols_pid].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        df_res_pid["delay_passive_all_times"] = np.nanmean(passive_arr_pid, axis=1)
    df_res_pid = df_res_pid[df_res_pid["cluster_id"].isin(region_cluster_ids)].copy()

    spont_intervals_pid = (
        _load_spontaneous_intervals(one, eid_pid) if CONFIG_CALC.get("CALC_SPONT", True) else None
    )
    spont_interval_list_pid = []
    if spont_intervals_pid is not None:
        spont_intervals_pid = np.asarray(spont_intervals_pid, dtype=float)
        spont_interval_list_pid = [tuple(row) for row in spont_intervals_pid]

    df_coupling_pid = None
    if CONFIG_CALC.get("CALC_SPONT", True) and spont_interval_list_pid:
        spikes_spont_pid = ana_utils.slice_spikes_by_intervals(spikes_pid, spont_interval_list_pid)
        df_coupling_pid = ana_utils.compute_population_coupling(
            spikes_spont_pid,
            clusters_pid,
            cluster_acronyms_pid,
            CONFIG_CALC,
            cluster_ids=region_cluster_ids,
            split_halves=True,
            intervals=spont_interval_list_pid,
            context_label="Spont",
        )

    task_windows_pid = ana_utils.build_task_window_table(
        sl_pid.trials,
        CONFIG_CALC["EVENT_NAMES"],
        post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
    )
    if not task_windows_pid.empty:
        task_odd_intervals_pid = task_windows_pid.loc[task_windows_pid["odd"], ["start", "end"]].to_numpy()
        task_even_intervals_pid = task_windows_pid.loc[
            ~task_windows_pid["odd"], ["start", "end"]
        ].to_numpy()
    else:
        task_odd_intervals_pid = np.empty((0, 2))
        task_even_intervals_pid = np.empty((0, 2))

    df_task_odd_pid = None
    df_task_even_pid = None
    if len(task_odd_intervals_pid) > 0:
        spikes_task_odd_pid = ana_utils.slice_spikes_by_intervals(
            spikes_pid,
            task_odd_intervals_pid,
            exclude_intervals=spont_interval_list_pid,
        )
        df_task_odd_pid = ana_utils.compute_population_coupling(
            spikes_task_odd_pid,
            clusters_pid,
            cluster_acronyms_pid,
            CONFIG_CALC,
            cluster_ids=region_cluster_ids,
            split_halves=False,
            intervals=task_odd_intervals_pid,
            context_label="Task odd",
        )
    if len(task_even_intervals_pid) > 0:
        spikes_task_even_pid = ana_utils.slice_spikes_by_intervals(
            spikes_pid,
            task_even_intervals_pid,
            exclude_intervals=spont_interval_list_pid,
        )
        df_task_even_pid = ana_utils.compute_population_coupling(
            spikes_task_even_pid,
            clusters_pid,
            cluster_acronyms_pid,
            CONFIG_CALC,
            cluster_ids=region_cluster_ids,
            split_halves=False,
            intervals=task_even_intervals_pid,
            context_label="Task even",
        )
    if df_task_odd_pid is not None and df_task_odd_pid.empty:
        df_task_odd_pid = None
    if df_task_even_pid is not None and df_task_even_pid.empty:
        df_task_even_pid = None
    df_coupling_task_pid = (
        ana_utils.merge_stpr_splits(
            df_task_odd_pid,
            df_task_even_pid,
            CONFIG_CALC,
            split_a="odd",
            split_b="even",
        )
        if (df_task_odd_pid is not None or df_task_even_pid is not None)
        else None
    )

    trial_end_times_pid = ana_utils.compute_trial_end_times(
        sl_pid.trials,
        CONFIG_CALC["EVENT_NAMES"],
        post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
    )
    stim_on_times_pid = np.asarray(sl_pid.trials["stimOn_times"], dtype=float)
    iti_windows_pid = ana_utils.build_iti_windows(
        trial_end_times_pid,
        stim_on_times_pid,
        skip_first_last=CONFIG_CALC["ITI_SKIP_FIRST_LAST"],
    )
    if not iti_windows_pid.empty:
        iti_odd_intervals_pid = iti_windows_pid.loc[iti_windows_pid["odd"], ["start", "end"]].to_numpy()
        iti_even_intervals_pid = iti_windows_pid.loc[
            ~iti_windows_pid["odd"], ["start", "end"]
        ].to_numpy()
    else:
        iti_odd_intervals_pid = np.empty((0, 2))
        iti_even_intervals_pid = np.empty((0, 2))

    df_iti_odd_pid = None
    df_iti_even_pid = None
    if len(iti_odd_intervals_pid) > 0:
        spikes_iti_odd_pid = ana_utils.slice_spikes_by_intervals(
            spikes_pid,
            iti_odd_intervals_pid,
            exclude_intervals=spont_interval_list_pid,
        )
        df_iti_odd_pid = ana_utils.compute_population_coupling(
            spikes_iti_odd_pid,
            clusters_pid,
            cluster_acronyms_pid,
            CONFIG_CALC,
            cluster_ids=region_cluster_ids,
            split_halves=False,
            intervals=iti_odd_intervals_pid,
            context_label="ITI odd",
        )
    if len(iti_even_intervals_pid) > 0:
        spikes_iti_even_pid = ana_utils.slice_spikes_by_intervals(
            spikes_pid,
            iti_even_intervals_pid,
            exclude_intervals=spont_interval_list_pid,
        )
        df_iti_even_pid = ana_utils.compute_population_coupling(
            spikes_iti_even_pid,
            clusters_pid,
            cluster_acronyms_pid,
            CONFIG_CALC,
            cluster_ids=region_cluster_ids,
            split_halves=False,
            intervals=iti_even_intervals_pid,
            context_label="ITI even",
        )
    if df_iti_odd_pid is not None and df_iti_odd_pid.empty:
        df_iti_odd_pid = None
    if df_iti_even_pid is not None and df_iti_even_pid.empty:
        df_iti_even_pid = None
    df_coupling_iti_pid = (
        ana_utils.merge_stpr_splits(
            df_iti_odd_pid,
            df_iti_even_pid,
            CONFIG_CALC,
            split_a="odd",
            split_b="even",
        )
        if (df_iti_odd_pid is not None or df_iti_even_pid is not None)
        else None
    )

    source_tables_pid = {
        "df_res": df_res_pid,
        "df_coupling": df_coupling_pid,
        "df_coupling_task": df_coupling_task_pid,
        "df_coupling_iti": df_coupling_iti_pid,
    }
    merged_pid = pd.DataFrame({"cluster_id": region_cluster_ids})
    for spec in CORR_VARIABLE_SPECS:
        src_df = source_tables_pid.get(spec["df"])
        if src_df is None or spec["v1"] not in src_df.columns or spec["v2"] not in src_df.columns:
            continue
        h1_col, h2_col = _corr_half_cols(spec)
        tmp = src_df[["cluster_id", spec["v1"], spec["v2"]]].copy()
        tmp = tmp.groupby("cluster_id", as_index=False).mean(numeric_only=True)
        tmp = tmp.rename(columns={spec["v1"]: h1_col, spec["v2"]: h2_col})
        h1_vals = tmp[h1_col].to_numpy(dtype=float)
        h2_vals = tmp[h2_col].to_numpy(dtype=float)
        mean_vals = np.full(len(tmp), np.nan, dtype=float)
        valid = np.isfinite(h1_vals) & np.isfinite(h2_vals)
        mean_vals[valid] = (h1_vals[valid] + h2_vals[valid]) / 2.0
        tmp[spec["name"]] = mean_vals
        merged_pid = merged_pid.merge(
            tmp[["cluster_id", h1_col, h2_col, spec["name"]]],
            on="cluster_id",
            how="left",
        )

    region_map = {int(cid): str(reg) for cid, reg in zip(cluster_ids_pid, region_by_cluster)}
    merged_pid["pid"] = str(pid_value)
    merged_pid["region"] = merged_pid["cluster_id"].map(
        lambda cid: region_map.get(int(cid), str(region_prefix))
    )
    merged_pid["unit_id"] = merged_pid["pid"].astype(str) + ":" + merged_pid["cluster_id"].astype(str)
    return merged_pid


all_pid_unit_tables = []
all_pid_failures = []

for idx, pid_value in enumerate(all_region_pids, start=1):
    print(f"[{idx}/{len(all_region_pids)}] Processing PID {pid_value}")
    try:
        pid_units = _compute_region_units_for_pid(
            pid_value,
            region_prefix=ALL_PID_TARGET_REGION,
            label_min=ALL_PID_LABEL_MIN,
        )
    except Exception as exc:
        all_pid_failures.append({"pid": pid_value, "error": str(exc)})
        print(f"  Failed: {exc}")
        continue

    if pid_units is None or pid_units.empty:
        print("  No neurons passed region+label filters.")
        continue

    all_pid_unit_tables.append(pid_units)
    print(f"  Kept {len(pid_units)} neurons.")

if all_pid_unit_tables:
    all_region_units = pd.concat(all_pid_unit_tables, ignore_index=True)
    fixed_cols = ["unit_id", "pid", "cluster_id", "region"]
    metric_cols = [col for col in all_region_units.columns if col not in fixed_cols]
    all_region_units = all_region_units[fixed_cols + metric_cols]
    print(
        f"Combined neurons: {len(all_region_units)} | "
        f"PIDs contributing neurons: {all_region_units['pid'].nunique()}"
    )
    display(all_region_units.head())
else:
    all_region_units = pd.DataFrame(columns=["unit_id", "pid", "cluster_id", "region"])
    print("No neurons were collected across the selected PIDs.")

if all_pid_failures:
    fail_df = pd.DataFrame(all_pid_failures)
    print(f"Failed PIDs: {len(fail_df)}")
    display(fail_df.head(20))


# %% All PIDs with VISp: correlation matrices for all combined neurons
if all_region_units.empty:
    print("No combined neurons available for correlation matrices.")
else:
    combined_available_specs = []
    for spec in CORR_VARIABLE_SPECS:
        name = spec["name"]
        h1_col, h2_col = _corr_half_cols(spec)
        if name not in all_region_units.columns or h1_col not in all_region_units.columns or h2_col not in all_region_units.columns:
            continue
        vals = all_region_units[name].to_numpy(dtype=float)
        if np.isfinite(vals).any():
            combined_available_specs.append(spec)

    combined_available_names = [spec["name"] for spec in combined_available_specs]
    if len(combined_available_names) < 2:
        print("Not enough variables with finite data for combined correlation matrices.")
    else:
        rel_pearson_all = {}
        rel_spearman_all = {}
        rel_n_pearson_all = {}
        rel_n_spearman_all = {}
        for spec in combined_available_specs:
            name = spec["name"]
            h1_col, h2_col = _corr_half_cols(spec)
            r_p, n_p = _pearsonr_with_n(
                all_region_units[h1_col].to_numpy(dtype=float),
                all_region_units[h2_col].to_numpy(dtype=float),
                min_n=CORR_MIN_N,
            )
            r_s, n_s = _spearmanr_with_n(
                all_region_units[h1_col].to_numpy(dtype=float),
                all_region_units[h2_col].to_numpy(dtype=float),
                min_n=CORR_MIN_N,
            )
            rel_pearson_all[name] = r_p
            rel_spearman_all[name] = r_s
            rel_n_pearson_all[name] = n_p
            rel_n_spearman_all[name] = n_s

        n_vars = len(combined_available_names)
        pearson_mat_all = np.full((n_vars, n_vars), np.nan, dtype=float)
        pearson_text_all = np.empty((n_vars, n_vars), dtype=object)
        spearman_mat_all = np.full((n_vars, n_vars), np.nan, dtype=float)
        spearman_text_all = np.empty((n_vars, n_vars), dtype=object)

        for i, name_i in enumerate(combined_available_names):
            for j, name_j in enumerate(combined_available_names):
                if i == j:
                    r_p = rel_pearson_all.get(name_i, np.nan)
                    n_p = rel_n_pearson_all.get(name_i, 0)
                    r_s = rel_spearman_all.get(name_i, np.nan)
                    n_s = rel_n_spearman_all.get(name_i, 0)
                    pearson_mat_all[i, j] = r_p
                    pearson_text_all[i, j] = (
                        f"rel={r_p:.2f}<br>(n={n_p})"
                        if np.isfinite(r_p)
                        else f"rel=nan<br>(n={n_p})"
                    )
                    spearman_mat_all[i, j] = r_s
                    spearman_text_all[i, j] = (
                        f"rel={r_s:.2f}<br>(n={n_s})"
                        if np.isfinite(r_s)
                        else f"rel=nan<br>(n={n_s})"
                    )
                else:
                    x = all_region_units[name_i].to_numpy(dtype=float)
                    y = all_region_units[name_j].to_numpy(dtype=float)
                    r_p, n_p = _pearsonr_with_n(x, y, min_n=CORR_MIN_N)
                    r_s, n_s = _spearmanr_with_n(x, y, min_n=CORR_MIN_N)
                    pearson_mat_all[i, j] = r_p
                    pearson_text_all[i, j] = (
                        f"r={r_p:.2f}<br>(n={n_p})"
                        if np.isfinite(r_p)
                        else f"r=nan<br>(n={n_p})"
                    )
                    spearman_mat_all[i, j] = r_s
                    spearman_text_all[i, j] = (
                        f"rho={r_s:.2f}<br>(n={n_s})"
                        if np.isfinite(r_s)
                        else f"rho=nan<br>(n={n_s})"
                    )

        title_suffix = (
            f"Region {ALL_PID_TARGET_REGION} | PIDs={all_region_units['pid'].nunique()} | "
            f"N neurons={len(all_region_units)} | label_min={ALL_PID_LABEL_MIN} | min_n={CORR_MIN_N}"
        )
        fig_p_all = go.Figure(
            data=go.Heatmap(
                z=pearson_mat_all,
                x=combined_available_names,
                y=combined_available_names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=pearson_text_all,
                texttemplate="%{text}",
                hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
            )
        )
        fig_p_all.update_layout(
            title=f"Reliability (diag) + Pairwise Pearson (off-diag) | {title_suffix}",
            width=1150,
            height=900,
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig_p_all.update_xaxes(tickangle=45)
        show_fig(fig_p_all)

        fig_s_all = go.Figure(
            data=go.Heatmap(
                z=spearman_mat_all,
                x=combined_available_names,
                y=combined_available_names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=spearman_text_all,
                texttemplate="%{text}",
                hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
            )
        )
        fig_s_all.update_layout(
            title=f"Reliability (diag) + Pairwise Spearman (off-diag) | {title_suffix}",
            width=1150,
            height=900,
            template=plot_config["PLOTLY_TEMPLATE"],
            margin=dict(l=90, r=30, t=90, b=90),
        )
        fig_s_all.update_xaxes(tickangle=45)
        show_fig(fig_s_all)


# %% All PIDs with VISp: variable correlation scatter for all combined neurons
if all_region_units.empty:
    print("No combined neurons available for variable correlation scatter.")
else:
    available_var_names_all = []
    for spec in CORR_VARIABLE_SPECS:
        name = spec["name"]
        if name not in all_region_units.columns:
            continue
        vals = all_region_units[name].to_numpy(dtype=float)
        if np.isfinite(vals).any():
            available_var_names_all.append(name)

    if len(available_var_names_all) < 2:
        print("Not enough variables with finite data to draw the combined scatter.")
    else:
        default_x_all = "Delay (Stim On)" if "Delay (Stim On)" in available_var_names_all else available_var_names_all[0]
        default_y_all = (
            "stPR Delay (Task)"
            if "stPR Delay (Task)" in available_var_names_all
            else available_var_names_all[1]
        )

        var_x_name_all = _choose_option(
            available_var_names_all,
            "Combined scatter variable X options:",
            default_x_all,
        )
        var_y_name_all = _choose_option(
            available_var_names_all,
            "Combined scatter variable Y options:",
            default_y_all,
        )
        if var_x_name_all == var_y_name_all:
            fallback_vars = [name for name in available_var_names_all if name != var_x_name_all]
            if fallback_vars:
                var_y_name_all = fallback_vars[0]
                print(f"Variable Y matched X; using '{var_y_name_all}' for Y.")

        plot_df_all = all_region_units[
            ["unit_id", "pid", "cluster_id", "region", var_x_name_all, var_y_name_all]
        ].copy()
        x_vals_all = plot_df_all[var_x_name_all].to_numpy(dtype=float)
        y_vals_all = plot_df_all[var_y_name_all].to_numpy(dtype=float)
        valid_mask_all = np.isfinite(x_vals_all) & np.isfinite(y_vals_all)
        plot_df_all = plot_df_all.loc[valid_mask_all].reset_index(drop=True)

        if plot_df_all.empty:
            print(f"No overlapping finite values for '{var_x_name_all}' and '{var_y_name_all}'.")
        else:
            from plotly.colors import qualitative

            x_vals_all = plot_df_all[var_x_name_all].to_numpy(dtype=float)
            y_vals_all = plot_df_all[var_y_name_all].to_numpy(dtype=float)
            r_p_all, n_p_all = _pearsonr_with_n(x_vals_all, y_vals_all, min_n=CORR_MIN_N)
            r_s_all, n_s_all = _spearmanr_with_n(x_vals_all, y_vals_all, min_n=CORR_MIN_N)

            pid_names_all = sorted(plot_df_all["pid"].astype(str).unique().tolist())
            palette_all = qualitative.Plotly + qualitative.D3 + qualitative.Set3
            pid_color_map_all = {
                pid_name: palette_all[idx % len(palette_all)]
                for idx, pid_name in enumerate(pid_names_all)
            }

            fig_corr_all = go.Figure()
            for pid_name in pid_names_all:
                pid_df = plot_df_all.loc[plot_df_all["pid"].astype(str) == str(pid_name)].copy()
                if pid_df.empty:
                    continue
                customdata = np.column_stack(
                    [
                        pid_df["unit_id"].astype(str).to_numpy(),
                        pid_df["region"].astype(str).to_numpy(),
                        pid_df["cluster_id"].to_numpy(dtype=int),
                    ]
                )
                fig_corr_all.add_trace(
                    go.Scatter(
                        x=pid_df[var_x_name_all].to_numpy(dtype=float),
                        y=pid_df[var_y_name_all].to_numpy(dtype=float),
                        mode="markers",
                        name=str(pid_name),
                        customdata=customdata,
                        marker=dict(
                            size=7,
                            opacity=0.7,
                            color=pid_color_map_all.get(str(pid_name), "gray"),
                        ),
                        hovertemplate=(
                            "PID: %{fullData.name}<br>"
                            "Unit: %{customdata[0]}<br>"
                            "Region: %{customdata[1]}<br>"
                            "Cluster ID: %{customdata[2]}<br>"
                            f"{var_x_name_all}: %{{x:.3f}}<br>"
                            f"{var_y_name_all}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            min_val_all = float(np.nanmin([np.nanmin(x_vals_all), np.nanmin(y_vals_all)]))
            max_val_all = float(np.nanmax([np.nanmax(x_vals_all), np.nanmax(y_vals_all)]))
            if np.isfinite(min_val_all) and np.isfinite(max_val_all) and max_val_all > min_val_all:
                fig_corr_all.add_trace(
                    go.Scatter(
                        x=[min_val_all, max_val_all],
                        y=[min_val_all, max_val_all],
                        mode="lines",
                        name="Unity line (y=x)",
                        line=dict(color="red", dash="dash", width=2),
                        hovertemplate="Unity line<extra></extra>",
                    )
                )

            fit_mask_all = np.isfinite(x_vals_all) & np.isfinite(y_vals_all)
            x_fit_all = x_vals_all[fit_mask_all]
            y_fit_all = y_vals_all[fit_mask_all]
            if x_fit_all.size >= 2 and np.nanstd(x_fit_all) > 0:
                slope_all, intercept_all = np.polyfit(x_fit_all, y_fit_all, 1)
                x_line_all = np.array(
                    [float(np.nanmin(x_fit_all)), float(np.nanmax(x_fit_all))],
                    dtype=float,
                )
                y_line_all = slope_all * x_line_all + intercept_all
                fig_corr_all.add_trace(
                    go.Scatter(
                        x=x_line_all,
                        y=y_line_all,
                        mode="lines",
                        name=f"Linear fit (y={slope_all:.2f}x+{intercept_all:.2f})",
                        line=dict(color="black", width=2),
                        hovertemplate="Linear fit<extra></extra>",
                    )
                )

            fig_corr_all.update_layout(
                title=(
                    f"All Neurons Correlation | Region {ALL_PID_TARGET_REGION} | "
                    f"{var_x_name_all} vs {var_y_name_all}"
                ),
                template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                width=1000,
                height=760,
                margin=dict(l=70, r=40, t=100, b=155),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                ),
            )
            fig_corr_all.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="left",
                bordercolor="rgba(120,120,120,0.6)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                text=(
                    f"Pearson r={_format_corr_value(r_p_all)} (n={n_p_all})<br>"
                    f"Spearman rho={_format_corr_value(r_s_all)} (n={n_s_all})<br>"
                    f"PIDs={plot_df_all['pid'].nunique()} | Neurons={len(plot_df_all)}"
                ),
            )
            fig_corr_all.update_xaxes(title_text=var_x_name_all)
            fig_corr_all.update_yaxes(title_text=var_y_name_all)
            show_fig(fig_corr_all)


# %% All PIDs with VISp: most variable visual-delay neurons vs stPR delay sign
if all_region_units.empty:
    print("No combined neurons available for visual variability analysis.")
else:
    delay_cols_active = [
        "Delay (Stim On)",
    ]
    delay_cols_passive = [
        "Delay (Passive Visual)",
    ]
    delay_cols_all = delay_cols_active + delay_cols_passive
    stpr_delay_cols = [
        "stPR Delay (Task)",
        "stPR Delay (ITI)",
        "stPR Delay (Spont)",
    ]

    available_delay_cols = [col for col in delay_cols_all if col in all_region_units.columns]
    available_active_cols = [col for col in delay_cols_active if col in all_region_units.columns]
    available_passive_cols = [col for col in delay_cols_passive if col in all_region_units.columns]
    available_stpr_cols = [col for col in stpr_delay_cols if col in all_region_units.columns]

    if len(available_delay_cols) < 2:
        print("Need at least 2 delay columns to compute per-neuron variability.")
    elif not available_stpr_cols:
        print("No stPR delay columns available for comparison.")
    else:
        variability_cols = [
            "unit_id",
            "pid",
            "cluster_id",
            "region",
            *available_delay_cols,
            *available_stpr_cols,
        ]
        var_df = all_region_units[variability_cols].copy()

        def _rowwise_variability_stats(df_values, min_valid=2):
            arr = df_values.to_numpy(dtype=float)
            n_rows = arr.shape[0]
            std_vals = np.full(n_rows, np.nan, dtype=float)
            range_vals = np.full(n_rows, np.nan, dtype=float)
            n_valid = np.zeros(n_rows, dtype=int)
            for i in range(n_rows):
                finite_vals = arr[i, np.isfinite(arr[i])]
                n_valid[i] = finite_vals.size
                if finite_vals.size >= min_valid:
                    std_vals[i] = float(np.std(finite_vals))
                    range_vals[i] = float(np.max(finite_vals) - np.min(finite_vals))
            return std_vals, range_vals, n_valid

        all_std, all_range, all_n = _rowwise_variability_stats(
            var_df[available_delay_cols],
            min_valid=2,
        )
        var_df["delay_variability_std"] = all_std
        var_df["delay_variability_range"] = all_range
        var_df["delay_variability_ncols"] = all_n

        if len(available_active_cols) >= 2:
            active_std, _, _ = _rowwise_variability_stats(
                var_df[available_active_cols],
                min_valid=2,
            )
            var_df["delay_variability_std_active"] = active_std
        else:
            var_df["delay_variability_std_active"] = np.nan

        if len(available_passive_cols) >= 2:
            passive_std, _, _ = _rowwise_variability_stats(
                var_df[available_passive_cols],
                min_valid=2,
            )
            var_df["delay_variability_std_passive"] = passive_std
        else:
            var_df["delay_variability_std_passive"] = np.nan

        valid_var_mask = np.isfinite(var_df["delay_variability_std"].to_numpy(dtype=float))
        n_valid_var = int(valid_var_mask.sum())
        if n_valid_var < CORR_MIN_N:
            print(
                "Not enough neurons with finite delay-variability values "
                f"(n={n_valid_var}, min_n={CORR_MIN_N})."
            )
        else:
            TOP_QUANTILE = 0.75
            threshold = float(
                np.nanquantile(var_df.loc[valid_var_mask, "delay_variability_std"], TOP_QUANTILE)
            )
            var_df["is_top_variable"] = (
                valid_var_mask
                & (var_df["delay_variability_std"].to_numpy(dtype=float) >= threshold)
            )
            top_mask = var_df["is_top_variable"].to_numpy(dtype=bool)
            non_top_mask = valid_var_mask & ~top_mask
            top_pct = int(round((1.0 - TOP_QUANTILE) * 100))

            print(
                "Delay columns used for visual variability: "
                + ", ".join(available_delay_cols)
            )
            print(
                f"Top-variable neurons: top {top_pct}% by delay_variability_std "
                f"(threshold={threshold:.3f}) | "
                f"n={int(top_mask.sum())}/{n_valid_var} neurons."
            )

            top_cols = [
                "unit_id",
                "pid",
                "cluster_id",
                "region",
                "delay_variability_std",
                "delay_variability_std_active",
                "delay_variability_std_passive",
                "delay_variability_range",
                *available_stpr_cols,
            ]
            top_preview = (
                var_df.loc[top_mask, top_cols]
                .sort_values("delay_variability_std", ascending=False)
                .head(30)
                .reset_index(drop=True)
            )
            display(top_preview)

            summary_rows = []
            for col in available_stpr_cols:
                x_vals = var_df["delay_variability_std"].to_numpy(dtype=float)
                y_vals = var_df[col].to_numpy(dtype=float)
                r_p, n_p = _pearsonr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)
                r_s, n_s = _spearmanr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)

                top_vals = var_df.loc[top_mask, col].to_numpy(dtype=float)
                top_vals = top_vals[np.isfinite(top_vals)]
                non_top_vals = var_df.loc[non_top_mask, col].to_numpy(dtype=float)
                non_top_vals = non_top_vals[np.isfinite(non_top_vals)]

                summary_rows.append(
                    {
                        "stPR_delay_metric": col,
                        "pearson_r_vs_variability": r_p,
                        "pearson_n": int(n_p),
                        "spearman_rho_vs_variability": r_s,
                        "spearman_n": int(n_s),
                        "median_top_variable": (
                            float(np.median(top_vals)) if top_vals.size > 0 else np.nan
                        ),
                        "median_remaining": (
                            float(np.median(non_top_vals))
                            if non_top_vals.size > 0
                            else np.nan
                        ),
                        "positive_frac_top_variable": (
                            float(np.mean(top_vals > 0)) if top_vals.size > 0 else np.nan
                        ),
                        "positive_frac_remaining": (
                            float(np.mean(non_top_vals > 0))
                            if non_top_vals.size > 0
                            else np.nan
                        ),
                        "n_top_valid": int(top_vals.size),
                        "n_remaining_valid": int(non_top_vals.size),
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            display(summary_df)

            if not summary_df.empty:
                fig_pos_frac = go.Figure()
                fig_pos_frac.add_trace(
                    go.Bar(
                        x=summary_df["stPR_delay_metric"].tolist(),
                        y=summary_df["positive_frac_top_variable"].to_numpy(dtype=float),
                        name=f"Top {top_pct}% variable",
                        marker_color="#d62728",
                    )
                )
                fig_pos_frac.add_trace(
                    go.Bar(
                        x=summary_df["stPR_delay_metric"].tolist(),
                        y=summary_df["positive_frac_remaining"].to_numpy(dtype=float),
                        name="Remaining neurons",
                        marker_color="#7f7f7f",
                    )
                )
                fig_pos_frac.update_layout(
                    title=(
                        f"Positive stPR Delay Fraction | Most Variable Visual-Delay Neurons "
                        f"(Region {ALL_PID_TARGET_REGION})"
                    ),
                    barmode="group",
                    template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                    width=980,
                    height=520,
                    margin=dict(l=70, r=30, t=90, b=70),
                    yaxis=dict(
                        title="Fraction with positive stPR delay",
                        range=[0, 1],
                        tickformat=".0%",
                    ),
                )
                fig_pos_frac.update_xaxes(title_text="stPR delay metric")
                show_fig(fig_pos_frac)

                fig_scatter = make_subplots(
                    rows=1,
                    cols=len(available_stpr_cols),
                    subplot_titles=available_stpr_cols,
                    horizontal_spacing=0.06 if len(available_stpr_cols) > 1 else 0.1,
                )
                for idx, col in enumerate(available_stpr_cols, start=1):
                    scatter_mask = np.isfinite(
                        var_df["delay_variability_std"].to_numpy(dtype=float)
                    ) & np.isfinite(var_df[col].to_numpy(dtype=float))
                    scatter_df = var_df.loc[
                        scatter_mask,
                        [
                            "unit_id",
                            "pid",
                            "cluster_id",
                            "region",
                            "delay_variability_std",
                            col,
                            "is_top_variable",
                        ],
                    ].copy()
                    if scatter_df.empty:
                        continue

                    group_specs = [
                        ("Remaining neurons", ~scatter_df["is_top_variable"], "#7f7f7f"),
                        (f"Top {top_pct}% variable", scatter_df["is_top_variable"], "#d62728"),
                    ]
                    for group_name, group_mask, color in group_specs:
                        group_df = scatter_df.loc[group_mask].copy()
                        if group_df.empty:
                            continue
                        customdata = np.column_stack(
                            [
                                group_df["unit_id"].astype(str).to_numpy(),
                                group_df["pid"].astype(str).to_numpy(),
                                group_df["region"].astype(str).to_numpy(),
                                group_df["cluster_id"].astype(str).to_numpy(),
                            ]
                        )
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=group_df["delay_variability_std"].to_numpy(dtype=float),
                                y=group_df[col].to_numpy(dtype=float),
                                mode="markers",
                                name=group_name,
                                showlegend=(idx == 1),
                                customdata=customdata,
                                marker=dict(size=6, opacity=0.75, color=color),
                                hovertemplate=(
                                    "Group: %{fullData.name}<br>"
                                    "Unit: %{customdata[0]}<br>"
                                    "PID: %{customdata[1]}<br>"
                                    "Region: %{customdata[2]}<br>"
                                    "Cluster ID: %{customdata[3]}<br>"
                                    "Delay variability (std): %{x:.3f}<br>"
                                    f"{col}: %{{y:.3f}}<extra></extra>"
                                ),
                            ),
                            row=1,
                            col=idx,
                        )

                    fig_scatter.update_xaxes(
                        title_text="Auditory-delay variability (std)",
                        row=1,
                        col=idx,
                    )
                    fig_scatter.update_yaxes(
                        title_text="stPR delay",
                        zeroline=True,
                        zerolinecolor="black",
                        row=1,
                        col=idx,
                    )

                fig_scatter.update_layout(
                    title=(
                        "Visual-Delay Variability vs stPR Delay "
                        f"| Region {ALL_PID_TARGET_REGION}"
                    ),
                    template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                    width=max(980, 380 * len(available_stpr_cols)),
                    height=520,
                    margin=dict(l=70, r=30, t=90, b=70),
                    legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="left", x=0),
                )
                show_fig(fig_scatter)


# %% All PIDs with VISp: most variable visual-delay neurons vs stPR strength sign
if all_region_units.empty:
    print("No combined neurons available for visual variability vs stPR strength analysis.")
else:
    delay_cols_active_strength = [
        "Delay (Stim On)",
    ]
    delay_cols_passive_strength = [
        "Delay (Passive Visual)",
    ]
    delay_cols_all_strength = delay_cols_active_strength + delay_cols_passive_strength
    stpr_strength_cols = [
        "stPR Strength (Task)",
        "stPR Strength (ITI)",
        "stPR Strength (Spont)",
    ]

    available_delay_cols_strength = [
        col for col in delay_cols_all_strength if col in all_region_units.columns
    ]
    available_active_cols_strength = [
        col for col in delay_cols_active_strength if col in all_region_units.columns
    ]
    available_passive_cols_strength = [
        col for col in delay_cols_passive_strength if col in all_region_units.columns
    ]
    available_strength_cols = [col for col in stpr_strength_cols if col in all_region_units.columns]

    if len(available_delay_cols_strength) < 2:
        print("Need at least 2 delay columns to compute per-neuron variability.")
    elif not available_strength_cols:
        print("No stPR strength columns available for comparison.")
    else:
        variability_cols_strength = [
            "unit_id",
            "pid",
            "cluster_id",
            "region",
            *available_delay_cols_strength,
            *available_strength_cols,
        ]
        var_df_strength = all_region_units[variability_cols_strength].copy()

        def _rowwise_variability_stats_strength(df_values, min_valid=2):
            arr = df_values.to_numpy(dtype=float)
            n_rows = arr.shape[0]
            std_vals = np.full(n_rows, np.nan, dtype=float)
            range_vals = np.full(n_rows, np.nan, dtype=float)
            n_valid = np.zeros(n_rows, dtype=int)
            for i in range(n_rows):
                finite_vals = arr[i, np.isfinite(arr[i])]
                n_valid[i] = finite_vals.size
                if finite_vals.size >= min_valid:
                    std_vals[i] = float(np.std(finite_vals))
                    range_vals[i] = float(np.max(finite_vals) - np.min(finite_vals))
            return std_vals, range_vals, n_valid

        all_std_strength, all_range_strength, all_n_strength = _rowwise_variability_stats_strength(
            var_df_strength[available_delay_cols_strength],
            min_valid=2,
        )
        var_df_strength["delay_variability_std"] = all_std_strength
        var_df_strength["delay_variability_range"] = all_range_strength
        var_df_strength["delay_variability_ncols"] = all_n_strength

        if len(available_active_cols_strength) >= 2:
            active_std_strength, _, _ = _rowwise_variability_stats_strength(
                var_df_strength[available_active_cols_strength],
                min_valid=2,
            )
            var_df_strength["delay_variability_std_active"] = active_std_strength
        else:
            var_df_strength["delay_variability_std_active"] = np.nan

        if len(available_passive_cols_strength) >= 2:
            passive_std_strength, _, _ = _rowwise_variability_stats_strength(
                var_df_strength[available_passive_cols_strength],
                min_valid=2,
            )
            var_df_strength["delay_variability_std_passive"] = passive_std_strength
        else:
            var_df_strength["delay_variability_std_passive"] = np.nan

        valid_var_mask_strength = np.isfinite(
            var_df_strength["delay_variability_std"].to_numpy(dtype=float)
        )
        n_valid_var_strength = int(valid_var_mask_strength.sum())
        if n_valid_var_strength < CORR_MIN_N:
            print(
                "Not enough neurons with finite delay-variability values "
                f"(n={n_valid_var_strength}, min_n={CORR_MIN_N})."
            )
        else:
            TOP_QUANTILE_STRENGTH = 0.75
            threshold_strength = float(
                np.nanquantile(
                    var_df_strength.loc[valid_var_mask_strength, "delay_variability_std"],
                    TOP_QUANTILE_STRENGTH,
                )
            )
            var_df_strength["is_top_variable"] = (
                valid_var_mask_strength
                & (
                    var_df_strength["delay_variability_std"].to_numpy(dtype=float)
                    >= threshold_strength
                )
            )
            top_mask_strength = var_df_strength["is_top_variable"].to_numpy(dtype=bool)
            non_top_mask_strength = valid_var_mask_strength & ~top_mask_strength
            top_pct_strength = int(round((1.0 - TOP_QUANTILE_STRENGTH) * 100))

            print(
                "Delay columns used for visual variability: "
                + ", ".join(available_delay_cols_strength)
            )
            print(
                f"Top-variable neurons: top {top_pct_strength}% by delay_variability_std "
                f"(threshold={threshold_strength:.3f}) | "
                f"n={int(top_mask_strength.sum())}/{n_valid_var_strength} neurons."
            )

            top_cols_strength = [
                "unit_id",
                "pid",
                "cluster_id",
                "region",
                "delay_variability_std",
                "delay_variability_std_active",
                "delay_variability_std_passive",
                "delay_variability_range",
                *available_strength_cols,
            ]
            top_preview_strength = (
                var_df_strength.loc[top_mask_strength, top_cols_strength]
                .sort_values("delay_variability_std", ascending=False)
                .head(30)
                .reset_index(drop=True)
            )
            display(top_preview_strength)

            summary_rows_strength = []
            for col in available_strength_cols:
                x_vals_strength = var_df_strength["delay_variability_std"].to_numpy(dtype=float)
                y_vals_strength = var_df_strength[col].to_numpy(dtype=float)
                r_p_strength, n_p_strength = _pearsonr_with_n(
                    x_vals_strength, y_vals_strength, min_n=CORR_MIN_N
                )
                r_s_strength, n_s_strength = _spearmanr_with_n(
                    x_vals_strength, y_vals_strength, min_n=CORR_MIN_N
                )

                top_vals_strength = var_df_strength.loc[top_mask_strength, col].to_numpy(dtype=float)
                top_vals_strength = top_vals_strength[np.isfinite(top_vals_strength)]
                non_top_vals_strength = var_df_strength.loc[
                    non_top_mask_strength, col
                ].to_numpy(dtype=float)
                non_top_vals_strength = non_top_vals_strength[np.isfinite(non_top_vals_strength)]

                summary_rows_strength.append(
                    {
                        "stPR_strength_metric": col,
                        "pearson_r_vs_variability": r_p_strength,
                        "pearson_n": int(n_p_strength),
                        "spearman_rho_vs_variability": r_s_strength,
                        "spearman_n": int(n_s_strength),
                        "median_top_variable": (
                            float(np.median(top_vals_strength))
                            if top_vals_strength.size > 0
                            else np.nan
                        ),
                        "median_remaining": (
                            float(np.median(non_top_vals_strength))
                            if non_top_vals_strength.size > 0
                            else np.nan
                        ),
                        "positive_frac_top_variable": (
                            float(np.mean(top_vals_strength > 0))
                            if top_vals_strength.size > 0
                            else np.nan
                        ),
                        "positive_frac_remaining": (
                            float(np.mean(non_top_vals_strength > 0))
                            if non_top_vals_strength.size > 0
                            else np.nan
                        ),
                        "n_top_valid": int(top_vals_strength.size),
                        "n_remaining_valid": int(non_top_vals_strength.size),
                    }
                )

            summary_df_strength = pd.DataFrame(summary_rows_strength)
            display(summary_df_strength)

            if not summary_df_strength.empty:
                fig_pos_frac_strength = go.Figure()
                fig_pos_frac_strength.add_trace(
                    go.Bar(
                        x=summary_df_strength["stPR_strength_metric"].tolist(),
                        y=summary_df_strength["positive_frac_top_variable"].to_numpy(dtype=float),
                        name=f"Top {top_pct_strength}% variable",
                        marker_color="#d62728",
                    )
                )
                fig_pos_frac_strength.add_trace(
                    go.Bar(
                        x=summary_df_strength["stPR_strength_metric"].tolist(),
                        y=summary_df_strength["positive_frac_remaining"].to_numpy(dtype=float),
                        name="Remaining neurons",
                        marker_color="#7f7f7f",
                    )
                )
                fig_pos_frac_strength.update_layout(
                    title=(
                        "Positive stPR Strength Fraction | Most Variable Visual-Delay Neurons "
                        f"(Region {ALL_PID_TARGET_REGION})"
                    ),
                    barmode="group",
                    template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                    width=980,
                    height=520,
                    margin=dict(l=70, r=30, t=90, b=70),
                    yaxis=dict(
                        title="Fraction with positive stPR strength",
                        range=[0, 1],
                        tickformat=".0%",
                    ),
                )
                fig_pos_frac_strength.update_xaxes(title_text="stPR strength metric")
                show_fig(fig_pos_frac_strength)

                fig_scatter_strength = make_subplots(
                    rows=1,
                    cols=len(available_strength_cols),
                    subplot_titles=available_strength_cols,
                    horizontal_spacing=0.06 if len(available_strength_cols) > 1 else 0.1,
                )
                for idx, col in enumerate(available_strength_cols, start=1):
                    scatter_mask_strength = np.isfinite(
                        var_df_strength["delay_variability_std"].to_numpy(dtype=float)
                    ) & np.isfinite(var_df_strength[col].to_numpy(dtype=float))
                    scatter_df_strength = var_df_strength.loc[
                        scatter_mask_strength,
                        [
                            "unit_id",
                            "pid",
                            "cluster_id",
                            "region",
                            "delay_variability_std",
                            col,
                            "is_top_variable",
                        ],
                    ].copy()
                    if scatter_df_strength.empty:
                        continue

                    group_specs_strength = [
                        (
                            "Remaining neurons",
                            ~scatter_df_strength["is_top_variable"],
                            "#7f7f7f",
                        ),
                        (
                            f"Top {top_pct_strength}% variable",
                            scatter_df_strength["is_top_variable"],
                            "#d62728",
                        ),
                    ]
                    for group_name, group_mask, color in group_specs_strength:
                        group_df_strength = scatter_df_strength.loc[group_mask].copy()
                        if group_df_strength.empty:
                            continue
                        customdata_strength = np.column_stack(
                            [
                                group_df_strength["unit_id"].astype(str).to_numpy(),
                                group_df_strength["pid"].astype(str).to_numpy(),
                                group_df_strength["region"].astype(str).to_numpy(),
                                group_df_strength["cluster_id"].astype(str).to_numpy(),
                            ]
                        )
                        fig_scatter_strength.add_trace(
                            go.Scatter(
                                x=group_df_strength["delay_variability_std"].to_numpy(dtype=float),
                                y=group_df_strength[col].to_numpy(dtype=float),
                                mode="markers",
                                name=group_name,
                                showlegend=(idx == 1),
                                customdata=customdata_strength,
                                marker=dict(size=6, opacity=0.75, color=color),
                                hovertemplate=(
                                    "Group: %{fullData.name}<br>"
                                    "Unit: %{customdata[0]}<br>"
                                    "PID: %{customdata[1]}<br>"
                                    "Region: %{customdata[2]}<br>"
                                    "Cluster ID: %{customdata[3]}<br>"
                                    "Delay variability (std): %{x:.3f}<br>"
                                    f"{col}: %{{y:.3f}}<extra></extra>"
                                ),
                            ),
                            row=1,
                            col=idx,
                        )

                    fig_scatter_strength.update_xaxes(
                        title_text="Auditory-delay variability (std)",
                        row=1,
                        col=idx,
                    )
                    fig_scatter_strength.update_yaxes(
                        title_text="stPR strength",
                        zeroline=True,
                        zerolinecolor="black",
                        row=1,
                        col=idx,
                    )

                fig_scatter_strength.update_layout(
                    title=(
                        "Visual-Delay Variability vs stPR Strength "
                        f"| Region {ALL_PID_TARGET_REGION}"
                    ),
                    template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                    width=max(980, 380 * len(available_strength_cols)),
                    height=520,
                    margin=dict(l=70, r=30, t=90, b=70),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.17,
                        xanchor="left",
                        x=0,
                    ),
                )
                show_fig(fig_scatter_strength)


# %% End of notebook

# %% Soloist/Choister hypothesis: split-safe role setup
SOLOIST_Q_LOW = 0.33
SOLOIST_Q_HIGH = 0.67
ROLE_ORDER = ["Soloist", "Intermediate", "Choister"]


def _quantile_role_labels(values, q_low=SOLOIST_Q_LOW, q_high=SOLOIST_Q_HIGH):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=object)
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() == 0:
        return out

    finite_vals = arr[finite_mask]
    ql = float(np.nanquantile(finite_vals, q_low))
    qh = float(np.nanquantile(finite_vals, q_high))

    role_vals = np.full(finite_vals.shape, "Intermediate", dtype=object)
    if np.isfinite(ql) and np.isfinite(qh) and qh > ql:
        role_vals[finite_vals <= ql] = "Soloist"
        role_vals[finite_vals >= qh] = "Choister"
    else:
        ranks = pd.Series(finite_vals).rank(method="average", pct=True).to_numpy(dtype=float)
        role_vals[ranks <= q_low] = "Soloist"
        role_vals[ranks >= q_high] = "Choister"

    out[finite_mask] = role_vals
    return out


ROLE_COLS = {
    "spont_h1": "__stpr_strength_spont_h1",
    "spont_h2": "__stpr_strength_spont_h2",
    "task_h1": "__stpr_strength_task_h1",
    "task_h2": "__stpr_strength_task_h2",
    "iti_h1": "__stpr_strength_iti_h1",
    "iti_h2": "__stpr_strength_iti_h2",
}

if all_region_units.empty:
    print("No combined neurons available for soloist/choister analysis.")
else:
    missing_role_cols = [col for col in ROLE_COLS.values() if col not in all_region_units.columns]
    if missing_role_cols:
        print("Missing split-half stPR strength columns:", missing_role_cols)
    else:
        keep_cols = ["unit_id", "pid", "cluster_id", "region", *ROLE_COLS.values()]
        df_roles = all_region_units[keep_cols].copy()

        # Role assignment from spontaneous H1 (defining split), then evaluate on H2.
        df_roles["role_spont_h1"] = _quantile_role_labels(df_roles[ROLE_COLS["spont_h1"]])
        df_roles["role_spont_h2"] = _quantile_role_labels(df_roles[ROLE_COLS["spont_h2"]])
        df_roles["role_task_h2"] = _quantile_role_labels(df_roles[ROLE_COLS["task_h2"]])
        df_roles["role_iti_h2"] = _quantile_role_labels(df_roles[ROLE_COLS["iti_h2"]])

        role_to_idx = {"Soloist": 0, "Intermediate": 1, "Choister": 2}
        src_idx = np.asarray([role_to_idx.get(v, np.nan) for v in df_roles["role_spont_h1"]], dtype=float)
        task_idx = np.asarray([role_to_idx.get(v, np.nan) for v in df_roles["role_task_h2"]], dtype=float)
        iti_idx = np.asarray([role_to_idx.get(v, np.nan) for v in df_roles["role_iti_h2"]], dtype=float)
        df_roles["role_shift_task_h2_vs_spont_h1"] = task_idx - src_idx
        df_roles["role_shift_iti_h2_vs_spont_h1"] = iti_idx - src_idx

        print(
            "Role quantile cutoffs (Spont H1): "
            f"q{int(SOLOIST_Q_LOW * 100)} / q{int(SOLOIST_Q_HIGH * 100)} "
            f"| n={len(df_roles)} neurons | PIDs={df_roles['pid'].nunique()}"
        )
        print("Role counts from Spont H1:")
        print(df_roles["role_spont_h1"].value_counts(dropna=False))
        display(
            df_roles[
                [
                    "unit_id",
                    "pid",
                    "region",
                    ROLE_COLS["spont_h1"],
                    ROLE_COLS["spont_h2"],
                    ROLE_COLS["task_h2"],
                    ROLE_COLS["iti_h2"],
                    "role_spont_h1",
                    "role_task_h2",
                    "role_iti_h2",
                    "role_shift_task_h2_vs_spont_h1",
                    "role_shift_iti_h2_vs_spont_h1",
                ]
            ]
            .head(20)
            .reset_index(drop=True)
        )


# %% Soloist/Choister transitions: Spont role (H1) -> Task/ITI role (H2)
def _transition_count_and_fraction(df, src_col, dst_col, order):
    tmp = df[[src_col, dst_col]].dropna().copy()
    if tmp.empty:
        return None, None
    ct = pd.crosstab(tmp[src_col], tmp[dst_col]).reindex(index=order, columns=order, fill_value=0)
    row_sums = ct.sum(axis=1).replace(0, np.nan)
    frac = ct.div(row_sums, axis=0)
    return ct, frac


if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for transition analysis.")
else:
    transitions = [
        ("Task", "role_task_h2"),
        ("ITI", "role_iti_h2"),
    ]
    fig_trans = make_subplots(
        rows=1,
        cols=len(transitions),
        subplot_titles=[
            "Spont H1 -> Task H2 role transition",
            "Spont H1 -> ITI H2 role transition",
        ],
        horizontal_spacing=0.12,
    )

    for col_idx, (label, dst_col) in enumerate(transitions, start=1):
        ct, frac = _transition_count_and_fraction(df_roles, "role_spont_h1", dst_col, ROLE_ORDER)
        if ct is None or frac is None:
            continue

        z = frac.to_numpy(dtype=float)
        text = np.empty(z.shape, dtype=object)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                pct_val = z[i, j]
                n_cell = int(ct.iat[i, j])
                pct_text = "nan" if not np.isfinite(pct_val) else f"{100.0 * pct_val:.1f}%"
                text[i, j] = f"{pct_text}<br>(n={n_cell})"

        fig_trans.add_trace(
            go.Heatmap(
                z=z,
                x=ROLE_ORDER,
                y=ROLE_ORDER,
                zmin=0,
                zmax=1,
                colorscale="YlGnBu",
                text=text,
                texttemplate="%{text}",
                hovertemplate="From=%{y}<br>To=%{x}<br>%{text}<extra></extra>",
                showscale=(col_idx == len(transitions)),
                colorbar=dict(title="Row fraction", tickformat=".0%") if col_idx == len(transitions) else None,
            ),
            row=1,
            col=col_idx,
        )
        fig_trans.update_xaxes(title_text=f"{label} role (H2)", row=1, col=col_idx)
        fig_trans.update_yaxes(title_text="Spont role (H1)", row=1, col=col_idx)

        print(f"{label} role transition counts (rows: Spont H1, cols: {label} H2):")
        display(ct)

    fig_trans.update_layout(
        title=(
            f"Soloist/Choister Transitions | Region {ALL_PID_TARGET_REGION} | "
            "Split-safe (roles from Spont H1, evaluated on H2)"
        ),
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        width=max(980, 520 * len(transitions)),
        height=500,
        margin=dict(l=70, r=40, t=90, b=70),
    )
    show_fig(fig_trans)


# %% Soloist/Choister trend plots: quantile trend + grouped distributions
def _finite_quantile(vals, q):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for trend plots.")
else:
    role_color_map = {
        "Soloist": "#1f77b4",
        "Intermediate": "#7f7f7f",
        "Choister": "#d62728",
    }
    metric_specs = [
        (ROLE_COLS["spont_h2"], "Spont strength (H2)", "#2ca02c"),
        (ROLE_COLS["task_h2"], "Task strength (H2)", "#1f77b4"),
        (ROLE_COLS["iti_h2"], "ITI strength (H2)", "#ff7f0e"),
    ]

    trend_df = df_roles[
        [
            "unit_id",
            "pid",
            "role_spont_h1",
            ROLE_COLS["spont_h1"],
            ROLE_COLS["spont_h2"],
            ROLE_COLS["task_h2"],
            ROLE_COLS["iti_h2"],
        ]
    ].copy()
    trend_df = trend_df[np.isfinite(trend_df[ROLE_COLS["spont_h1"]].to_numpy(dtype=float))].copy()

    unique_x = np.unique(np.round(trend_df[ROLE_COLS["spont_h1"]].to_numpy(dtype=float), 12))
    n_bins = int(np.clip(min(10, len(unique_x)), 3, 10))
    trend_df["spont_h1_bin"] = pd.qcut(
        trend_df[ROLE_COLS["spont_h1"]],
        q=n_bins,
        duplicates="drop",
    )

    summary = (
        trend_df.groupby("spont_h1_bin", observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "x_center": float(np.nanmedian(g[ROLE_COLS["spont_h1"]].to_numpy(dtype=float))),
                    "n": int(len(g)),
                    "spont_med": float(np.nanmedian(g[ROLE_COLS["spont_h2"]].to_numpy(dtype=float))),
                    "spont_q25": _finite_quantile(g[ROLE_COLS["spont_h2"]], 0.25),
                    "spont_q75": _finite_quantile(g[ROLE_COLS["spont_h2"]], 0.75),
                    "task_med": float(np.nanmedian(g[ROLE_COLS["task_h2"]].to_numpy(dtype=float))),
                    "task_q25": _finite_quantile(g[ROLE_COLS["task_h2"]], 0.25),
                    "task_q75": _finite_quantile(g[ROLE_COLS["task_h2"]], 0.75),
                    "iti_med": float(np.nanmedian(g[ROLE_COLS["iti_h2"]].to_numpy(dtype=float))),
                    "iti_q25": _finite_quantile(g[ROLE_COLS["iti_h2"]], 0.25),
                    "iti_q75": _finite_quantile(g[ROLE_COLS["iti_h2"]], 0.75),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("x_center")
        .reset_index(drop=True)
    )
    display(summary)

    fig_trend = go.Figure()
    for col_key, label, color in metric_specs:
        key = (
            "spont"
            if col_key == ROLE_COLS["spont_h2"]
            else ("task" if col_key == ROLE_COLS["task_h2"] else "iti")
        )
        x_vals = summary["x_center"].to_numpy(dtype=float)
        y_med = summary[f"{key}_med"].to_numpy(dtype=float)
        y_q25 = summary[f"{key}_q25"].to_numpy(dtype=float)
        y_q75 = summary[f"{key}_q75"].to_numpy(dtype=float)
        n_vals = summary["n"].to_numpy(dtype=int)
        valid = np.isfinite(x_vals) & np.isfinite(y_med)
        if not np.any(valid):
            continue
        fig_trend.add_trace(
            go.Scatter(
                x=x_vals[valid],
                y=y_med[valid],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    visible=True,
                    array=(y_q75 - y_med)[valid],
                    arrayminus=(y_med - y_q25)[valid],
                ),
                customdata=n_vals[valid].reshape(-1, 1),
                hovertemplate=(
                    "Spont H1 bin center: %{x:.3f}<br>"
                    "Median strength: %{y:.3f}<br>"
                    "Bin n: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig_trend.update_layout(
        title=(
            f"Strength Trends Across Spont-H1 Quantiles | Region {ALL_PID_TARGET_REGION} "
            "| median with IQR"
        ),
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        width=980,
        height=520,
        margin=dict(l=70, r=40, t=90, b=70),
        xaxis_title="Spont stPR strength (H1) quantile-bin center",
        yaxis_title="stPR strength (H2)",
    )
    show_fig(fig_trend)

    fig_box = go.Figure()
    for col_key, label, color in metric_specs:
        box_df = df_roles[["role_spont_h1", col_key]].copy()
        box_df = box_df[
            box_df["role_spont_h1"].isin(ROLE_ORDER)
            & np.isfinite(box_df[col_key].to_numpy(dtype=float))
        ].copy()
        if box_df.empty:
            continue
        fig_box.add_trace(
            go.Box(
                x=box_df["role_spont_h1"],
                y=box_df[col_key].to_numpy(dtype=float),
                name=label,
                marker_color=color,
                boxmean="sd",
                jitter=0.2,
                pointpos=0,
                marker=dict(size=4, opacity=0.35),
                hovertemplate=(
                    "Spont-H1 role: %{x}<br>"
                    "Strength: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig_box.update_layout(
        title=(
            f"Group Distributions by Spont-H1 Role | Region {ALL_PID_TARGET_REGION} "
            "| Split-safe evaluation on H2"
        ),
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        width=980,
        height=520,
        margin=dict(l=70, r=40, t=90, b=70),
        xaxis=dict(
            title="Role from Spont H1",
            categoryorder="array",
            categoryarray=ROLE_ORDER,
        ),
        yaxis=dict(title="stPR strength (H2)"),
    )
    show_fig(fig_box)


# %% Nonlinearity test: linear vs monotonic (isotonic) fits
def _aggregate_unique_xy(x, y):
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    x_unique, inv = np.unique(x_sorted, return_inverse=True)
    y_sum = np.zeros(len(x_unique), dtype=float)
    w_sum = np.zeros(len(x_unique), dtype=float)
    np.add.at(y_sum, inv, y_sorted)
    np.add.at(w_sum, inv, 1.0)
    y_mean = y_sum / np.maximum(w_sum, 1.0)
    return x_unique, y_mean, w_sum


def _pav_increasing(y, w):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.size == 0:
        return y

    levels = []
    weights = []
    lengths = []
    for idx in range(y.size):
        levels.append(float(y[idx]))
        weights.append(float(w[idx]))
        lengths.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            new_weight = weights[-2] + weights[-1]
            new_level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / new_weight
            levels[-2] = float(new_level)
            weights[-2] = float(new_weight)
            lengths[-2] = lengths[-2] + lengths[-1]
            levels.pop()
            weights.pop()
            lengths.pop()

    y_iso = np.empty(y.size, dtype=float)
    start = 0
    for level_val, length_val in zip(levels, lengths):
        end = start + int(length_val)
        y_iso[start:end] = float(level_val)
        start = end
    return y_iso


def _fit_isotonic_increasing(x, y):
    x_unique, y_mean, w_sum = _aggregate_unique_xy(x, y)
    y_iso = _pav_increasing(y_mean, w_sum)
    return x_unique, y_iso


def _predict_isotonic(x_new, x_unique, y_iso):
    if len(x_unique) == 0:
        return np.full_like(np.asarray(x_new, dtype=float), np.nan, dtype=float)
    return np.interp(
        np.asarray(x_new, dtype=float),
        np.asarray(x_unique, dtype=float),
        np.asarray(y_iso, dtype=float),
        left=float(y_iso[0]),
        right=float(y_iso[-1]),
    )


def _r2_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def _mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 1:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _cv_linear_vs_isotonic(x, y, n_splits=5, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 12:
        return pd.DataFrame()

    n_splits = int(min(max(3, n_splits), n))
    rng = np.random.default_rng(seed)
    idx_perm = rng.permutation(n)
    folds = np.array_split(idx_perm, n_splits)

    rows = []
    for fold_idx in range(len(folds)):
        test_idx = folds[fold_idx]
        if len(test_idx) == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        if int(train_mask.sum()) < 3:
            continue

        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[test_idx]
        y_test = y[test_idx]
        if np.nanstd(x_train) <= 0:
            continue

        slope, intercept = np.polyfit(x_train, y_train, 1)
        y_pred_lin = slope * x_test + intercept

        x_unique, y_iso = _fit_isotonic_increasing(x_train, y_train)
        y_pred_iso = _predict_isotonic(x_test, x_unique, y_iso)

        rows.append(
            {
                "fold": fold_idx,
                "n_train": int(train_mask.sum()),
                "n_test": int(len(test_idx)),
                "r2_linear": _r2_np(y_test, y_pred_lin),
                "r2_isotonic": _r2_np(y_test, y_pred_iso),
                "mae_linear": _mae_np(y_test, y_pred_lin),
                "mae_isotonic": _mae_np(y_test, y_pred_iso),
            }
        )
    return pd.DataFrame(rows)


if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for nonlinear-fit analysis.")
else:
    x_col = ROLE_COLS["spont_h1"]
    fit_specs = [
        ("Task", ROLE_COLS["task_h2"]),
        ("ITI", ROLE_COLS["iti_h2"]),
    ]
    role_color_map = {
        "Soloist": "#1f77b4",
        "Intermediate": "#7f7f7f",
        "Choister": "#d62728",
    }

    fit_summary_rows = []
    fig_fit = make_subplots(
        rows=1,
        cols=len(fit_specs),
        subplot_titles=[f"Spont H1 vs {label} H2" for label, _ in fit_specs],
        horizontal_spacing=0.1,
    )

    for col_idx, (label, y_col) in enumerate(fit_specs, start=1):
        pair_df = df_roles[[x_col, y_col, "role_spont_h1"]].copy()
        finite_mask = np.isfinite(pair_df[x_col].to_numpy(dtype=float)) & np.isfinite(
            pair_df[y_col].to_numpy(dtype=float)
        )
        pair_df = pair_df.loc[finite_mask].reset_index(drop=True)
        if pair_df.empty:
            continue

        x_vals = pair_df[x_col].to_numpy(dtype=float)
        y_vals = pair_df[y_col].to_numpy(dtype=float)
        r_p, n_p = _pearsonr_with_n(x_vals, y_vals, min_n=3)
        r_s, n_s = _spearmanr_with_n(x_vals, y_vals, min_n=3)

        cv_df = _cv_linear_vs_isotonic(x_vals, y_vals, n_splits=5, seed=41 + col_idx)
        cv_r2_lin = float(np.nanmean(cv_df["r2_linear"])) if not cv_df.empty else np.nan
        cv_r2_iso = float(np.nanmean(cv_df["r2_isotonic"])) if not cv_df.empty else np.nan
        cv_mae_lin = float(np.nanmean(cv_df["mae_linear"])) if not cv_df.empty else np.nan
        cv_mae_iso = float(np.nanmean(cv_df["mae_isotonic"])) if not cv_df.empty else np.nan

        fit_summary_rows.append(
            {
                "target": f"{label} H2",
                "n": int(len(pair_df)),
                "pearson_r": r_p,
                "spearman_rho": r_s,
                "cv_r2_linear": cv_r2_lin,
                "cv_r2_isotonic": cv_r2_iso,
                "cv_r2_gain_iso_minus_linear": cv_r2_iso - cv_r2_lin
                if np.isfinite(cv_r2_lin) and np.isfinite(cv_r2_iso)
                else np.nan,
                "cv_mae_linear": cv_mae_lin,
                "cv_mae_isotonic": cv_mae_iso,
                "cv_mae_gain_linear_minus_iso": cv_mae_lin - cv_mae_iso
                if np.isfinite(cv_mae_lin) and np.isfinite(cv_mae_iso)
                else np.nan,
            }
        )

        for role_name in ROLE_ORDER:
            role_df = pair_df[pair_df["role_spont_h1"] == role_name].copy()
            if role_df.empty:
                continue
            fig_fit.add_trace(
                go.Scatter(
                    x=role_df[x_col].to_numpy(dtype=float),
                    y=role_df[y_col].to_numpy(dtype=float),
                    mode="markers",
                    name=role_name,
                    legendgroup=role_name,
                    showlegend=(col_idx == 1),
                    marker=dict(
                        size=6,
                        opacity=0.45,
                        color=role_color_map.get(role_name, "gray"),
                    ),
                    hovertemplate=(
                        "Role (Spont H1): %{fullData.name}<br>"
                        "Spont H1 strength: %{x:.3f}<br>"
                        f"{label} H2 strength: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        x_line = np.linspace(float(np.nanmin(x_vals)), float(np.nanmax(x_vals)), 300)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        y_line_lin = slope * x_line + intercept
        x_unique, y_iso = _fit_isotonic_increasing(x_vals, y_vals)
        y_line_iso = _predict_isotonic(x_line, x_unique, y_iso)

        fig_fit.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line_lin,
                mode="lines",
                name="Linear fit",
                legendgroup=f"fit_{label}",
                showlegend=(col_idx == 1),
                line=dict(color="black", width=2),
                hovertemplate="Linear fit<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig_fit.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line_iso,
                mode="lines",
                name="Monotonic fit (isotonic)",
                legendgroup=f"fit_{label}",
                showlegend=(col_idx == 1),
                line=dict(color="#d62728", width=2, dash="dash"),
                hovertemplate="Monotonic (isotonic) fit<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        x_ref = "x domain" if col_idx == 1 else f"x{col_idx} domain"
        y_ref = "y domain" if col_idx == 1 else f"y{col_idx} domain"
        fig_fit.add_annotation(
            x=0.99,
            y=0.99,
            xref=x_ref,
            yref=y_ref,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            align="left",
            bordercolor="rgba(120,120,120,0.6)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            text=(
                f"Pearson r={_format_corr_value(r_p)} (n={n_p})<br>"
                f"Spearman rho={_format_corr_value(r_s)} (n={n_s})<br>"
                f"CV R2: lin={_format_corr_value(cv_r2_lin)}, iso={_format_corr_value(cv_r2_iso)}"
            ),
        )
        fig_fit.update_xaxes(title_text="Spont stPR strength (H1)", row=1, col=col_idx)
        fig_fit.update_yaxes(title_text=f"{label} stPR strength (H2)", row=1, col=col_idx)

    fit_summary_df = pd.DataFrame(fit_summary_rows)
    if fit_summary_df.empty:
        print("Not enough data to compute nonlinear-fit comparison.")
    else:
        print("Linear vs monotonic fit summary (cross-validated):")
        display(fit_summary_df)

    fig_fit.update_layout(
        title=(
            f"Nonlinearity Check (Split-safe) | Region {ALL_PID_TARGET_REGION} "
            "| Spont H1 predicts Task/ITI H2"
        ),
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        width=max(980, 560 * len(fit_specs)),
        height=560,
        margin=dict(l=70, r=40, t=90, b=90),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
        ),
    )
    show_fig(fig_fit)


# %% Soloist/Choister within-group correlations (H2 evaluation)
if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for within-group correlation analysis.")
else:
    within_rows = []
    for role_name in ROLE_ORDER:
        role_df = df_roles[df_roles["role_spont_h1"] == role_name].copy()
        if role_df.empty:
            continue

        x_eval = role_df[ROLE_COLS["spont_h2"]].to_numpy(dtype=float)
        y_task_eval = role_df[ROLE_COLS["task_h2"]].to_numpy(dtype=float)
        y_iti_eval = role_df[ROLE_COLS["iti_h2"]].to_numpy(dtype=float)

        r_p_task, n_p_task = _pearsonr_with_n(x_eval, y_task_eval, min_n=3)
        r_s_task, n_s_task = _spearmanr_with_n(x_eval, y_task_eval, min_n=3)
        r_p_iti, n_p_iti = _pearsonr_with_n(x_eval, y_iti_eval, min_n=3)
        r_s_iti, n_s_iti = _spearmanr_with_n(x_eval, y_iti_eval, min_n=3)

        within_rows.append(
            {
                "role_from_spont_h1": role_name,
                "n_role_units": int(len(role_df)),
                "Task_H2_vs_Spont_H2_pearson": r_p_task,
                "Task_H2_vs_Spont_H2_spearman": r_s_task,
                "Task_pair_n_pearson": int(n_p_task),
                "Task_pair_n_spearman": int(n_s_task),
                "ITI_H2_vs_Spont_H2_pearson": r_p_iti,
                "ITI_H2_vs_Spont_H2_spearman": r_s_iti,
                "ITI_pair_n_pearson": int(n_p_iti),
                "ITI_pair_n_spearman": int(n_s_iti),
                "median_role_shift_task_h2_vs_spont_h1": float(
                    np.nanmedian(role_df["role_shift_task_h2_vs_spont_h1"].to_numpy(dtype=float))
                ),
                "median_role_shift_iti_h2_vs_spont_h1": float(
                    np.nanmedian(role_df["role_shift_iti_h2_vs_spont_h1"].to_numpy(dtype=float))
                ),
            }
        )

    within_df = pd.DataFrame(within_rows)
    if within_df.empty:
        print("No within-group summary available.")
    else:
        display(within_df)


# %% Conclusion plot candidate: Task vs Spont by role (group-wise linear fits + unity)
if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for Task-vs-Spont group plot.")
else:
    x_col_plot = ROLE_COLS["spont_h2"]
    y_col_plot = ROLE_COLS["task_h2"]
    role_color_map = {
        "Soloist": "#1f77b4",
        "Intermediate": "#7f7f7f",
        "Choister": "#d62728",
    }

    plot_df_role = df_roles[
        [
            "unit_id",
            "pid",
            "cluster_id",
            "region",
            "role_spont_h1",
            x_col_plot,
            y_col_plot,
        ]
    ].copy()
    finite_mask_role = np.isfinite(plot_df_role[x_col_plot].to_numpy(dtype=float)) & np.isfinite(
        plot_df_role[y_col_plot].to_numpy(dtype=float)
    )
    plot_df_role = plot_df_role.loc[finite_mask_role].copy()
    plot_df_role = plot_df_role[plot_df_role["role_spont_h1"].isin(ROLE_ORDER)].copy()

    if plot_df_role.empty:
        print("No finite Task-vs-Spont values available for group-wise plot.")
    else:
        fig_group = go.Figure()
        stats_rows = []

        x_all = plot_df_role[x_col_plot].to_numpy(dtype=float)
        y_all = plot_df_role[y_col_plot].to_numpy(dtype=float)
        min_ref = float(np.nanmin([np.nanmin(x_all), np.nanmin(y_all)]))
        max_ref = float(np.nanmax([np.nanmax(x_all), np.nanmax(y_all)]))

        for role_name in ROLE_ORDER:
            role_df = plot_df_role[plot_df_role["role_spont_h1"] == role_name].copy()
            if role_df.empty:
                continue

            x_vals = role_df[x_col_plot].to_numpy(dtype=float)
            y_vals = role_df[y_col_plot].to_numpy(dtype=float)
            r_p, n_p = _pearsonr_with_n(x_vals, y_vals, min_n=3)
            r_s, n_s = _spearmanr_with_n(x_vals, y_vals, min_n=3)

            stats_rows.append(
                {
                    "role_from_spont_h1": role_name,
                    "n_units": int(len(role_df)),
                    "pearson_r_taskH2_vs_spontH2": r_p,
                    "pearson_n": int(n_p),
                    "spearman_rho_taskH2_vs_spontH2": r_s,
                    "spearman_n": int(n_s),
                }
            )

            customdata_role = np.column_stack(
                [
                    role_df["unit_id"].astype(str).to_numpy(),
                    role_df["pid"].astype(str).to_numpy(),
                    role_df["cluster_id"].astype(str).to_numpy(),
                    role_df["region"].astype(str).to_numpy(),
                ]
            )
            fig_group.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=role_name,
                    legendgroup=role_name,
                    marker=dict(
                        size=6,
                        opacity=0.55,
                        color=role_color_map.get(role_name, "gray"),
                    ),
                    customdata=customdata_role,
                    hovertemplate=(
                        "Role: %{fullData.name}<br>"
                        "Unit: %{customdata[0]}<br>"
                        "PID: %{customdata[1]}<br>"
                        "Cluster ID: %{customdata[2]}<br>"
                        "Region: %{customdata[3]}<br>"
                        "Spont H2 strength: %{x:.3f}<br>"
                        "Task H2 strength: %{y:.3f}<extra></extra>"
                    ),
                )
            )

            if len(role_df) >= 3 and np.nanstd(x_vals) > 0:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.array([float(np.nanmin(x_vals)), float(np.nanmax(x_vals))], dtype=float)
                y_line = slope * x_line + intercept
                fig_group.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"{role_name} fit (y={slope:.2f}x+{intercept:.2f})",
                        legendgroup=role_name,
                        line=dict(
                            color=role_color_map.get(role_name, "gray"),
                            width=3,
                        ),
                        hovertemplate=f"{role_name} linear fit<extra></extra>",
                    )
                )

        if np.isfinite(min_ref) and np.isfinite(max_ref) and max_ref > min_ref:
            fig_group.add_trace(
                go.Scatter(
                    x=[min_ref, max_ref],
                    y=[min_ref, max_ref],
                    mode="lines",
                    name="Unity line (y=x)",
                    line=dict(color="black", width=2, dash="dash"),
                    hovertemplate="Unity line (y=x)<extra></extra>",
                )
            )

        stats_df = pd.DataFrame(stats_rows)
        if not stats_df.empty:
            display(stats_df)
            annotation_lines = []
            for _, row in stats_df.iterrows():
                annotation_lines.append(
                    f"{row['role_from_spont_h1']}: "
                    f"r={_format_corr_value(row['pearson_r_taskH2_vs_spontH2'])}, "
                    f"rho={_format_corr_value(row['spearman_rho_taskH2_vs_spontH2'])}, "
                    f"n={int(row['n_units'])}"
                )
            fig_group.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="left",
                bordercolor="rgba(120,120,120,0.6)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                text="<br>".join(annotation_lines),
            )

        fig_group.update_layout(
            title=(
                f"Task vs Spont Strength by Spont-H1 Role | Region {ALL_PID_TARGET_REGION} "
                "| Split-safe evaluation on H2"
            ),
            template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
            width=980,
            height=620,
            margin=dict(l=70, r=40, t=90, b=90),
            xaxis_title="Spont stPR strength (H2)",
            yaxis_title="Task stPR strength (H2)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="left",
                x=0,
            ),
        )
        show_fig(fig_group)


# %% Role-wise shift plot: histogram of Task(H2) - Spont(H2)
if "df_roles" not in globals() or df_roles.empty:
    print("No role table available for Task-Spont shift histogram.")
else:
    x_col_shift = ROLE_COLS["spont_h2"]
    y_col_shift = ROLE_COLS["task_h2"]
    role_color_map = {
        "Soloist": "#1f77b4",
        "Intermediate": "#7f7f7f",
        "Choister": "#d62728",
    }

    shift_df = df_roles[
        [
            "unit_id",
            "pid",
            "cluster_id",
            "region",
            "role_spont_h1",
            x_col_shift,
            y_col_shift,
        ]
    ].copy()
    finite_mask_shift = np.isfinite(shift_df[x_col_shift].to_numpy(dtype=float)) & np.isfinite(
        shift_df[y_col_shift].to_numpy(dtype=float)
    )
    shift_df = shift_df.loc[finite_mask_shift].copy()
    shift_df = shift_df[shift_df["role_spont_h1"].isin(ROLE_ORDER)].copy()

    if shift_df.empty:
        print("No finite Task-Spont shift values available.")
    else:
        shift_df["task_minus_spont_h2"] = (
            shift_df[y_col_shift].to_numpy(dtype=float) - shift_df[x_col_shift].to_numpy(dtype=float)
        )

        # Common bins across roles for direct visual comparison.
        shift_vals_all = shift_df["task_minus_spont_h2"].to_numpy(dtype=float)
        lo = float(np.nanquantile(shift_vals_all, 0.005))
        hi = float(np.nanquantile(shift_vals_all, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(shift_vals_all))
            hi = float(np.nanmax(shift_vals_all))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        bins = np.linspace(lo, hi, 40)

        fig_shift = go.Figure()
        stats_rows_shift = []
        for role_name in ROLE_ORDER:
            role_df = shift_df[shift_df["role_spont_h1"] == role_name].copy()
            if role_df.empty:
                continue

            vals = role_df["task_minus_spont_h2"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            stats_rows_shift.append(
                {
                    "role_from_spont_h1": role_name,
                    "n_units": int(vals.size),
                    "mean_task_minus_spont_h2": float(np.mean(vals)),
                    "median_task_minus_spont_h2": float(np.median(vals)),
                    "std_task_minus_spont_h2": float(np.std(vals)),
                    "q25_task_minus_spont_h2": float(np.quantile(vals, 0.25)),
                    "q75_task_minus_spont_h2": float(np.quantile(vals, 0.75)),
                    "positive_fraction_task_minus_spont_h2": float(np.mean(vals > 0)),
                }
            )

            fig_shift.add_trace(
                go.Histogram(
                    x=vals,
                    xbins=dict(start=float(bins[0]), end=float(bins[-1]), size=float(bins[1] - bins[0])),
                    histnorm="probability density",
                    name=f"{role_name} (n={len(vals)})",
                    marker_color=role_color_map.get(role_name, "gray"),
                    opacity=0.45,
                    hovertemplate=(
                        f"Role: {role_name}<br>"
                        "Task-Spont (H2): %{x:.3f}<br>"
                        "Density: %{y:.4f}<extra></extra>"
                    ),
                )
            )

        stats_shift_df = pd.DataFrame(stats_rows_shift)
        if not stats_shift_df.empty:
            display(stats_shift_df)

        fig_shift.add_vline(
            x=0.0,
            line=dict(color="black", width=2, dash="dash"),
            annotation_text="No shift (Task=Spont)",
            annotation_position="top right",
        )

        # Add median markers for each role to emphasize direction and magnitude of shift.
        for _, row in stats_shift_df.iterrows():
            role_name = str(row["role_from_spont_h1"])
            med_val = float(row["median_task_minus_spont_h2"])
            fig_shift.add_vline(
                x=med_val,
                line=dict(color=role_color_map.get(role_name, "gray"), width=2),
                opacity=0.75,
            )

        fig_shift.update_layout(
            title=(
                f"Task-Spont Strength Shift by Spont-H1 Role | Region {ALL_PID_TARGET_REGION} "
                "| Shift = Task(H2) - Spont(H2)"
            ),
            template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
            width=980,
            height=560,
            margin=dict(l=70, r=40, t=90, b=90),
            barmode="overlay",
            xaxis_title="Task(H2) - Spont(H2) stPR strength",
            yaxis_title="Density",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="left",
                x=0,
            ),
        )
        show_fig(fig_shift)
