# %% Imports
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

try:
    from plotly_resampler import FigureResampler
except Exception:  # pragma: no cover
    FigureResampler = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))  # if script is in /scripts/

from utils.io import setup_paths, init_one, load_session_data, build_cluster_id_map
import utils.plotting_plotly as plotting_utils
import utils.analysis as ana_utils
from utils.analysis import event_label, compute_psth_for_clusters
from types import SimpleNamespace


# %% Config
# ---- Config ----
# PID = "c9664185-d3fd-4e0e-89cf-77c402038938"
PID = "f967a527-257f-404a-871d-b91575dca3b4"

# Data loading
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
LOAD_RAW_DATA = True
LOAD_RAW_WHEEL = False
LOAD_RAW_POSE = False
LOAD_RAW_MOTION_ENERGY = True
ALLOW_REMOTE_METADATA = True

# Template source and cross-validation
TEMPLATE_SOURCE = "spont"  # "spont", "task", "iti"
USE_SPLIT_TEMPLATES = True
SPONT_SPLIT_SEGMENTS = 10

# Thresholds and weighting
STPR_STRENGTH_MIN = 0.01
WEIGHT_BY_COUPLING = False
RECTIFY_MATCH_SCORES = True  # set negative matches to 0

# Signal processing
DETECTION_BIN_SIZE = 0.005  # seconds
SMOOTH_SIGMA_S = 0.005  # seconds
TEMPLATE_TIME_SCALE = 1.0  # stretch (>1) / squish (<1)

# Packet probability normalization + detection
PACKET_SCORE_ZSCORE = True
PACKET_THRESHOLD = 2.0  # z-score if PACKET_SCORE_ZSCORE else raw score
MIN_PACKET_GAP_S = 0.1

# Raster options
LABEL_MIN = 0.9
REGIONS = "PO"  # e.g., ["VISp", "MOp"] or None for all
REGION_PREFIX_MATCH = False
SORT_CHOICE = "Spont stPR Delay"
TRIAL_IDX = 829
TIME_WINDOW_START = 4100.0
TIME_WINDOW_END = 4120.0
PACKET_PSTH_REGION = None  # default: first available region
PACKET_WHISK_EVENT_CONTEXT = "all"  # "all", "task", "iti", "spont"
PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"


# %% Helpers
# ---- Helpers ----
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
    fig.show()


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


def _load_raw_session(pid, load_wheel=False, load_pose=False, load_motion_energy=False, mode="local"):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one = init_one(ibl_cache, mode=mode)
    ssl, spikes, clusters, sl = load_session_data(
        pid,
        one,
        load_wheel=load_wheel,
        load_pose=load_pose,
        load_motion_energy=load_motion_energy,
    )
    return spikes, clusters, sl, ssl


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


def gaussian_kernel(sigma_bins, truncate=3.0):
    if sigma_bins is None or sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = int(round(truncate * sigma_bins))
    if radius < 1:
        return np.array([1.0], dtype=float)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / float(sigma_bins)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_signal(rate, kernel):
    if kernel.size <= 1:
        return rate
    return np.convolve(rate, kernel, mode="same")


def resample_template(curve, orig_bin_s, target_bin_s, time_scale=1.0):
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return curve
    if orig_bin_s <= 0 or target_bin_s <= 0:
        return curve
    scale = (orig_bin_s / target_bin_s) * float(time_scale)
    new_len = int(round(curve.size * scale))
    new_len = max(new_len, 3)
    if new_len == curve.size:
        return curve.copy()
    x_old = np.linspace(0, curve.size - 1, curve.size)
    x_new = np.linspace(0, curve.size - 1, new_len)
    return np.interp(x_new, x_old, curve)


def normalized_xcorr(signal, template):
    signal = np.asarray(signal, dtype=float)
    template = np.asarray(template, dtype=float)
    if signal.size == 0 or template.size < 3:
        return np.full(signal.shape, np.nan, dtype=float)

    tpl = template - np.nanmean(template)
    tpl_std = np.nanstd(tpl)
    if not np.isfinite(tpl_std) or tpl_std == 0:
        return np.full(signal.shape, np.nan, dtype=float)

    tpl_z = tpl / tpl_std
    L = tpl_z.size
    kernel = np.ones(L, dtype=float) / float(L)
    mu = np.convolve(signal, kernel, mode="same")
    mu2 = np.convolve(signal * signal, kernel, mode="same")
    var = mu2 - mu * mu
    var[var < 0] = 0
    sigma = np.sqrt(var)

    dot = np.convolve(signal, tpl_z[::-1], mode="same")
    denom = sigma * float(L)
    score = np.full(signal.shape, np.nan, dtype=float)
    valid = denom > 0
    score[valid] = dot[valid] / denom[valid]

    half = L // 2
    if half > 0:
        score[:half] = np.nan
        score[-half:] = np.nan
    return score


def robust_zscore(values):
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if np.isfinite(mad) and mad > 0:
        return 0.6745 * (values - med) / mad
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - mean) / std


def detect_peaks(score, times, threshold, min_gap_s):
    score = np.asarray(score, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(score)
    safe_score = np.where(valid, score, -np.inf)
    above = safe_score >= threshold
    if not np.any(above):
        return []

    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]

    peaks = []
    for s, e in zip(starts, ends):
        idx = s + int(np.nanargmax(safe_score[s:e]))
        peaks.append(idx)

    peaks = sorted(peaks, key=lambda i: times[i])
    if min_gap_s is None or min_gap_s <= 0 or len(peaks) <= 1:
        return peaks

    filtered = [peaks[0]]
    for idx in peaks[1:]:
        if times[idx] - times[filtered[-1]] < min_gap_s:
            if safe_score[idx] > safe_score[filtered[-1]]:
                filtered[-1] = idx
        else:
            filtered.append(idx)
    return filtered


# %% Load Session
_set_plotly_renderer(PLOTLY_RENDERER)

pid_list = list_pids()
if not pid_list:
    raise RuntimeError("No cached sessions found in data/dashboard_cache.")

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
            pid,
            load_wheel=LOAD_RAW_WHEEL,
            load_pose=LOAD_RAW_POSE,
            load_motion_energy=LOAD_RAW_MOTION_ENERGY,
            mode="local",
        )
    except Exception as exc:
        raw_error = exc
        if ALLOW_REMOTE_METADATA:
            try:
                spikes, clusters, session, _ssl = _load_raw_session(
                    pid,
                    load_wheel=LOAD_RAW_WHEEL,
                    load_pose=LOAD_RAW_POSE,
                    load_motion_energy=LOAD_RAW_MOTION_ENERGY,
                    mode="remote",
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


# %% Template Table
# ---- Prepare template table ----
config_calc = data.get("config_calc", {})
config_plot = dict(data.get("config_plot", {}))
config_plot["PLOTLY_TEMPLATE"] = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
config_plot["PLOT_LABEL_MIN"] = LABEL_MIN
plotting_utils.DEFAULT_TEMPLATE = config_plot["PLOTLY_TEMPLATE"]
pio.templates.default = config_plot["PLOTLY_TEMPLATE"]

trials = None
if session is not None:
    if hasattr(session, "trials"):
        trials = session.trials
    elif isinstance(session, dict):
        trials = session.get("trials")


def _filter_units_by_region(df_units_in):
    if df_units_in is None or df_units_in.empty or REGIONS is None:
        return df_units_in.copy() if df_units_in is not None else df_units_in
    df_units_local = df_units_in.copy()
    regions = REGIONS if isinstance(REGIONS, (list, tuple)) else [REGIONS]
    if REGION_PREFIX_MATCH:
        region_mask = np.zeros(len(df_units_local), dtype=bool)
        for reg in regions:
            region_mask |= df_units_local["acronym"].astype(str).str.startswith(str(reg))
        return df_units_local.loc[region_mask].copy()
    return df_units_local[
        df_units_local["acronym"].isin([str(region) for region in regions])
    ].copy()


def _prepare_filtered_units_df(only_good):
    df_units_local, _quality_mask = plotting_utils._prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        only_good,
        label_min=LABEL_MIN,
    )
    return _filter_units_by_region(df_units_local)

df_coupling_spont = data.get("df_coupling")
df_coupling_task = data.get("df_coupling_task")
df_coupling_iti = data.get("df_coupling_iti")

template_source = str(TEMPLATE_SOURCE).strip().lower()
if template_source not in {"spont", "task", "iti"}:
    raise ValueError(f"Unsupported TEMPLATE_SOURCE '{TEMPLATE_SOURCE}'.")
template_source_label = {"spont": "Spont", "task": "Task", "iti": "ITI"}[template_source]

stpr_bin_s = float(config_calc.get("STPR_BIN_SIZE", 0.001))
use_split = bool(USE_SPLIT_TEMPLATES)


def _interval_array_local(intervals):
    if intervals is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(intervals, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, 2)
    if arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    valid = np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])
    return np.asarray(arr[valid], dtype=float)


def _split_intervals_equal_chunks(intervals, n_chunks):
    arr = _interval_array_local(intervals)
    if arr.size == 0 or int(n_chunks) <= 0:
        return np.empty((0, 2), dtype=float)
    chunks = []
    for start_s, end_s in arr:
        edges = np.linspace(float(start_s), float(end_s), int(n_chunks) + 1)
        for idx in range(int(n_chunks)):
            if edges[idx + 1] > edges[idx]:
                chunks.append((edges[idx], edges[idx + 1]))
    return np.asarray(chunks, dtype=float) if chunks else np.empty((0, 2), dtype=float)


def _split_spont_intervals_alternating(intervals, n_chunks=10):
    chunks = _split_intervals_equal_chunks(intervals, n_chunks)
    if chunks.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    return chunks[::2].copy(), chunks[1::2].copy()


def _template_cluster_ids():
    df_units_tpl = _prepare_filtered_units_df(config_plot.get("PLOT_ONLY_GOOD_UNITS", False))
    if df_units_tpl is None or df_units_tpl.empty:
        return np.array([], dtype=int)
    return df_units_tpl["cluster_id"].to_numpy(dtype=int)


def _build_behavior_period_windows():
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    spont_intervals = _interval_array_local(meta.get("spont_interval") if isinstance(meta, dict) else None)

    if trials is None:
        return spont_intervals, pd.DataFrame(columns=["trial_idx", "start", "end", "correct", "odd"]), pd.DataFrame(columns=["trial_idx", "start", "end", "odd"])

    event_names = list(
        config_calc.get("EVENT_NAMES", ["stimOn_times", "firstMovement_times", "feedback_times"])
    )
    task_post_event_s = float(config_calc.get("TASK_POST_EVENT_S", 1.0))

    task_windows = ana_utils.build_task_window_table(
        trials,
        event_names,
        post_event_s=task_post_event_s,
    )
    if "stimOn_times" not in trials.keys():
        stim_on_times = np.array([], dtype=float)
    else:
        stim_on_times = np.asarray(trials["stimOn_times"], dtype=float).reshape(-1)
    trial_end_times = ana_utils.compute_trial_end_times(
        trials,
        event_names,
        post_event_s=task_post_event_s,
    )
    iti_windows = ana_utils.build_iti_windows(
        trial_end_times,
        stim_on_times,
        skip_first_last=bool(config_calc.get("ITI_SKIP_FIRST_LAST", True)),
    )
    return spont_intervals, task_windows, iti_windows


def _context_split_intervals(source):
    spont_intervals, task_windows, iti_windows = _build_behavior_period_windows()
    spont_exclude = [tuple(row) for row in spont_intervals.tolist()] if spont_intervals.size > 0 else None

    if source == "spont":
        split_a_intervals, split_b_intervals = _split_spont_intervals_alternating(
            spont_intervals,
            n_chunks=SPONT_SPLIT_SEGMENTS,
        )
        return split_a_intervals, split_b_intervals, None

    if source == "task":
        if task_windows.empty:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), spont_exclude
        split_a_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        split_b_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        return split_a_intervals, split_b_intervals, spont_exclude

    if source == "iti":
        if iti_windows.empty:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), spont_exclude
        split_a_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        split_b_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        return split_a_intervals, split_b_intervals, spont_exclude

    raise ValueError(f"Unhandled template source '{source}'.")


def _compute_stpr_split_df(intervals, exclude_intervals, context_label, cluster_ids_use):
    intervals = _interval_array_local(intervals)
    if intervals.size == 0 or len(cluster_ids_use) == 0:
        return None
    spikes_ctx = ana_utils.slice_spikes_by_intervals(
        spikes,
        intervals,
        exclude_intervals=exclude_intervals,
    )
    df_ctx = ana_utils.compute_population_coupling(
        spikes_ctx,
        clusters,
        cluster_acronyms,
        config_calc,
        cluster_ids=cluster_ids_use,
        split_halves=False,
        intervals=intervals,
        context_label=context_label,
    )
    if df_ctx is not None and not df_ctx.empty:
        return df_ctx
    return None


def _compute_template_source_table(source):
    cluster_ids_use = _template_cluster_ids()
    if cluster_ids_use.size == 0:
        raise RuntimeError("No units available to compute source templates after region filtering.")

    split_a_intervals, split_b_intervals, exclude_intervals = _context_split_intervals(source)
    df_a = _compute_stpr_split_df(
        split_a_intervals,
        exclude_intervals,
        f"{source.capitalize()} odd",
        cluster_ids_use,
    )
    df_b = _compute_stpr_split_df(
        split_b_intervals,
        exclude_intervals,
        f"{source.capitalize()} even",
        cluster_ids_use,
    )
    if df_a is None and df_b is None:
        raise RuntimeError(f"No split stPR results were computed for TEMPLATE_SOURCE='{source}'.")
    return ana_utils.merge_stpr_splits(
        df_a,
        df_b,
        config_calc,
        split_a="odd",
        split_b="even",
    )


print(
    f"Computing template source '{template_source}' with split stPRs "
    f"(odd/even); spont uses {SPONT_SPLIT_SEGMENTS} alternating chunks."
)
df_tpl = _compute_template_source_table(template_source)
if df_tpl is None or df_tpl.empty or "cluster_id" not in df_tpl.columns:
    raise RuntimeError(f"Template source '{template_source}' did not produce a valid coupling table.")
df_tpl = df_tpl.set_index("cluster_id", drop=False)
df_tpl_for_merge = df_tpl.reset_index(drop=True).copy()

if template_source == "spont":
    df_coupling_spont = df_tpl_for_merge.copy()
elif template_source == "task":
    df_coupling_task = df_tpl_for_merge.copy()
elif template_source == "iti":
    df_coupling_iti = df_tpl_for_merge.copy()


def _extract_template_and_strength(row, split=None):
    if split:
        curve = row.get(f"stpr_curve_{split}", [])
        strength = row.get(f"coupling_strength_{split}", np.nan)
    else:
        curve = row.get("stpr_curve", [])
        strength = row.get("coupling_strength", np.nan)
    curve = np.asarray(curve, dtype=float)
    return curve, strength


templates_a = {}
templates_b = {}
strength_a = {}
strength_b = {}
for cid, row in df_tpl.iterrows():
    curve_a, s_a = _extract_template_and_strength(row, split="odd")
    curve_b, s_b = _extract_template_and_strength(row, split="even")
    if curve_a.size > 0:
        templates_a[cid] = resample_template(
            curve_a, stpr_bin_s, DETECTION_BIN_SIZE, TEMPLATE_TIME_SCALE
        )
        strength_a[cid] = s_a
    if curve_b.size > 0:
        templates_b[cid] = resample_template(
            curve_b, stpr_bin_s, DETECTION_BIN_SIZE, TEMPLATE_TIME_SCALE
        )
        strength_b[cid] = s_b
    if not use_split:
        curve_mean, s_mean = _extract_template_and_strength(row, split=None)
        if curve_mean.size > 0:
            templates_a[cid] = resample_template(
                curve_mean, stpr_bin_s, DETECTION_BIN_SIZE, TEMPLATE_TIME_SCALE
            )
            strength_a[cid] = s_mean


def _mean_strength(cid):
    s_a = strength_a.get(cid, np.nan)
    s_b = strength_b.get(cid, np.nan)
    if np.isfinite(s_a) and np.isfinite(s_b):
        return float((s_a + s_b) / 2.0)
    if np.isfinite(s_a):
        return float(s_a)
    if np.isfinite(s_b):
        return float(s_b)
    return np.nan


# %% Unit Table
# ---- Build unit table / region selection ----
df_units = _prepare_filtered_units_df(config_plot.get("PLOT_ONLY_GOOD_UNITS", False))
avg_psth_only_good = config_plot.get(
    "AVG_PSTH_ONLY_GOOD", config_plot.get("PLOT_ONLY_GOOD_UNITS", False)
)
df_units_psth = _prepare_filtered_units_df(avg_psth_only_good)

if df_units.empty:
    raise RuntimeError("No units remain after label and region filtering.")

region_order = df_units["acronym"].dropna().unique().tolist()
print(f"Regions: {', '.join(region_order)}")


# %% Template Reliability
# ---- Plot split reliability for selected region(s) ----
def _pearsonr_with_n(x_vals, y_vals):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        return np.nan, n_valid
    return float(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1]), n_valid


df_tpl_rel = df_tpl.reset_index(drop=True).merge(
    df_units[["cluster_id", "acronym"]],
    on="cluster_id",
    how="inner",
).drop_duplicates(subset=["cluster_id"])

if template_source == "spont":
    split_label_a = "Spont split 1"
    split_label_b = "Spont split 2"
else:
    split_label_a = "Odd"
    split_label_b = "Even"

delay_col_a = "coupling_delay_ms_odd"
delay_col_b = "coupling_delay_ms_even"
strength_col_a = "coupling_strength_odd"
strength_col_b = "coupling_strength_even"
reliability_cols = {delay_col_a, delay_col_b, strength_col_a, strength_col_b}
if not reliability_cols.issubset(set(df_tpl_rel.columns)):
    print(
        f"Split reliability plot skipped: required columns missing for template source {template_source_label}."
    )
else:
    rel_template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
    rel_base_color = plotting_utils._template_base_color(rel_template)
    rel_region_colors = plotting_utils._region_color_map(region_order)
    subplot_titles = []
    for region in region_order:
        subplot_titles.extend(
            [
                f"{region} delay reliability",
                f"{region} strength reliability",
            ]
        )
    fig_rel = make_subplots(
        rows=max(1, len(region_order)),
        cols=2,
        subplot_titles=tuple(subplot_titles),
        horizontal_spacing=0.12,
        vertical_spacing=0.10,
    )

    for row_idx, region in enumerate(region_order, start=1):
        df_region_rel = df_tpl_rel.loc[df_tpl_rel["acronym"] == region].copy()
        region_color = rel_region_colors.get(region)

        x_delay = pd.to_numeric(df_region_rel.get(delay_col_a), errors="coerce").to_numpy(dtype=float)
        y_delay = pd.to_numeric(df_region_rel.get(delay_col_b), errors="coerce").to_numpy(dtype=float)
        delay_r, delay_n = _pearsonr_with_n(x_delay, y_delay)
        fig_rel.add_trace(
            go.Scatter(
                x=x_delay,
                y=y_delay,
                mode="markers",
                marker=dict(color=region_color, size=8, opacity=0.85),
                showlegend=False,
                hovertemplate=(
                    f"{region}<br>{split_label_a}: %{{x:.2f}} ms<br>{split_label_b}: %{{y:.2f}} ms<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )
        delay_valid = np.isfinite(x_delay) & np.isfinite(y_delay)
        if np.any(delay_valid):
            delay_min = float(np.nanmin(np.r_[x_delay[delay_valid], y_delay[delay_valid]]))
            delay_max = float(np.nanmax(np.r_[x_delay[delay_valid], y_delay[delay_valid]]))
            fig_rel.add_trace(
                go.Scatter(
                    x=[delay_min, delay_max],
                    y=[delay_min, delay_max],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )
        fig_rel.update_xaxes(
            title_text=f"Delay {split_label_a} (ms)<br>r={delay_r:.2f}, n={delay_n}" if np.isfinite(delay_r) else f"Delay {split_label_a} (ms)<br>n={delay_n}",
            row=row_idx,
            col=1,
        )
        fig_rel.update_yaxes(title_text=f"Delay {split_label_b} (ms)", row=row_idx, col=1)

        x_strength = pd.to_numeric(df_region_rel.get(strength_col_a), errors="coerce").to_numpy(dtype=float)
        y_strength = pd.to_numeric(df_region_rel.get(strength_col_b), errors="coerce").to_numpy(dtype=float)
        strength_r, strength_n = _pearsonr_with_n(x_strength, y_strength)
        fig_rel.add_trace(
            go.Scatter(
                x=x_strength,
                y=y_strength,
                mode="markers",
                marker=dict(color=region_color, size=8, opacity=0.85),
                showlegend=False,
                hovertemplate=(
                    f"{region}<br>{split_label_a}: %{{x:.3f}}<br>{split_label_b}: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=2,
        )
        strength_valid = np.isfinite(x_strength) & np.isfinite(y_strength)
        if np.any(strength_valid):
            strength_min = float(np.nanmin(np.r_[x_strength[strength_valid], y_strength[strength_valid]]))
            strength_max = float(np.nanmax(np.r_[x_strength[strength_valid], y_strength[strength_valid]]))
            fig_rel.add_trace(
                go.Scatter(
                    x=[strength_min, strength_max],
                    y=[strength_min, strength_max],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=2,
            )
        fig_rel.update_xaxes(
            title_text=(
                f"Strength {split_label_a}<br>r={strength_r:.2f}, n={strength_n}"
                if np.isfinite(strength_r)
                else f"Strength {split_label_a}<br>n={strength_n}"
            ),
            row=row_idx,
            col=2,
        )
        fig_rel.update_yaxes(title_text=f"Strength {split_label_b}", row=row_idx, col=2)

    fig_rel.update_layout(
        title=f"{template_source_label} stPR split reliability",
        template=rel_template,
        font=dict(color=rel_base_color),
        height=380 * max(1, len(region_order)),
        margin=dict(l=70, r=40, t=90, b=60),
    )
    show_fig(fig_rel, renderer=PLOTLY_RENDERER)


# %% Whisk Events
# ---- Whisking event detection (same helper/config as 03_calc_dashboard.py) ----
def _empty_whisk_outputs():
    return {
        "wh_detect": {
            "all_bouts": np.empty((0, 2), dtype=float),
            "brief_bouts": np.empty((0, 2), dtype=float),
            "long_bouts": np.empty((0, 2), dtype=float),
            "all_onsets": np.array([], dtype=float),
            "brief_onsets": np.array([], dtype=float),
            "long_onsets": np.array([], dtype=float),
            "all_durations": np.array([], dtype=float),
            "brief_durations": np.array([], dtype=float),
            "long_durations": np.array([], dtype=float),
        },
        "wh_event_base": {
            "wh_brief_times": np.array([], dtype=float),
            "wh_long_times": np.array([], dtype=float),
            "wh_all_times": np.array([], dtype=float),
            "wh_all_times_loco": np.array([], dtype=float),
            "wh_all_times_non_loco": np.array([], dtype=float),
        },
        "wh_events_by_period": {},
        "wh_long_offset_times_spont": np.array([], dtype=float),
    }


spont_intervals, task_windows, iti_windows = _build_behavior_period_windows()
df_wh = data.get("df_wh")
if not isinstance(df_wh, pd.DataFrame):
    df_wh = pd.DataFrame(
        columns=[
            "bin_idx",
            "bin_start_s",
            "bin_center_s",
            "bin_end_s",
            "wh_norm",
            "n_views",
        ]
    )

whisk_bundle = None
whisk_source = None
wheel_obj = getattr(session, "wheel", None) if session is not None else None

if not df_wh.empty:
    try:
        whisk_bundle = ana_utils.build_whisk_events(
            df_wh,
            config_calc,
            spont_intervals=spont_intervals,
            task_windows=task_windows,
            iti_windows=iti_windows,
            wheel=wheel_obj,
        )
        whisk_source = "cached df_wh"
    except Exception as exc:
        print(f"Whisk detection rebuild from cached df_wh failed: {type(exc).__name__}: {exc}")

if whisk_bundle is None:
    motion_energy = getattr(session, "motion_energy", None) if session is not None else None
    if motion_energy is not None:
        df_me_raw = ana_utils.extract_motion_energy_trace(
            motion_energy,
            max_interp_gap_frames=3,
            ensure_positive_motion=True,
        )
        if not df_me_raw.empty:
            df_wh_from_me = ana_utils.build_whisk_trace(df_me_raw, config_calc)
            if df_wh_from_me is not None and not df_wh_from_me.empty:
                df_wh = df_wh_from_me.copy()
                whisk_bundle = ana_utils.build_whisk_events(
                    df_wh,
                    config_calc,
                    spont_intervals=spont_intervals,
                    task_windows=task_windows,
                    iti_windows=iti_windows,
                    wheel=wheel_obj,
                )
                whisk_source = "session motion energy"

if whisk_bundle is None:
    cached_wh_detect = data.get("wh_detect")
    cached_wh_event_base = data.get("wh_event_base")
    cached_wh_events_by_period = data.get("wh_events_by_period")
    if isinstance(cached_wh_event_base, dict) and isinstance(cached_wh_events_by_period, dict):
        whisk_bundle = {
            "wh_detect": cached_wh_detect if isinstance(cached_wh_detect, dict) else {},
            "wh_event_base": cached_wh_event_base,
            "wh_events_by_period": cached_wh_events_by_period,
            "wh_long_offset_times_spont": np.asarray(
                cached_wh_events_by_period.get("wh_long_offset_times_spont", np.array([])),
                dtype=float,
            ),
        }
        whisk_source = "cached whisk bundle"

if whisk_bundle is None:
    whisk_bundle = _empty_whisk_outputs()
    whisk_source = "empty"

wh_detect = whisk_bundle.get("wh_detect", {})
wh_event_base = whisk_bundle.get("wh_event_base", {})
wh_events_by_period = whisk_bundle.get("wh_events_by_period", {})
data["df_wh"] = df_wh
data["wh_detect"] = wh_detect
data["wh_event_base"] = wh_event_base
data["wh_events_by_period"] = wh_events_by_period

print(f"Whisk events source: {whisk_source}")
for wh_key in (
    "wh_brief_times",
    "wh_long_times",
    "wh_all_times",
    "wh_brief_times_task",
    "wh_long_times_task",
    "wh_brief_times_iti",
    "wh_long_times_iti",
    "wh_brief_times_spont",
    "wh_long_times_spont",
):
    wh_arr = np.asarray(wh_events_by_period.get(wh_key, np.array([])), dtype=float)
    if wh_arr.size > 0:
        print(f"  {wh_key}: {len(wh_arr)}")


# %% Spike Index
# ---- Build spike index (fast lookup) ----
spike_times = np.asarray(spikes["times"])
spike_clusters = np.asarray(spikes["clusters"])

selected_cluster_ids = df_units["cluster_id"].to_numpy()
selected_mask = np.isin(spike_clusters, selected_cluster_ids)
spike_times_sel = spike_times[selected_mask]
spike_clusters_sel = spike_clusters[selected_mask]

order = np.argsort(spike_clusters_sel)
clusters_sorted = spike_clusters_sel[order]
times_sorted = spike_times_sel[order]

spike_times_by_cluster = {}
if len(clusters_sorted) > 0:
    unique_cids, start_idx = np.unique(clusters_sorted, return_index=True)
    for i, cid in enumerate(unique_cids):
        start = start_idx[i]
        end = start_idx[i + 1] if i + 1 < len(start_idx) else len(clusters_sorted)
        spike_times_by_cluster[int(cid)] = times_sorted[start:end]


# %% Packet Probability
# ---- Compute packet probability per region ----
t_start = float(np.nanmin(spike_times))
t_end = float(np.nanmax(spike_times))

bin_edges = np.arange(t_start, t_end + DETECTION_BIN_SIZE, DETECTION_BIN_SIZE)
n_bins = len(bin_edges) - 1
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

sigma_bins = SMOOTH_SIGMA_S / DETECTION_BIN_SIZE if DETECTION_BIN_SIZE > 0 else 0
kernel = gaussian_kernel(sigma_bins)

region_scores_raw = {}
region_scores_plot = {}
region_event_indices = {}
region_event_times = {}
region_event_scores = {}
region_event_windows = {}
region_template_duration_s = {}
region_used_cluster_ids = {}
used_cluster_ids = set()

region_to_cids = {
    region: df_units.loc[df_units["acronym"] == region, "cluster_id"].to_numpy()
    for region in region_order
}

for region, cids in region_to_cids.items():
    if len(cids) == 0:
        continue

    score_sum = np.zeros(n_bins, dtype=float)
    weight_sum = np.zeros(n_bins, dtype=float)
    used_cids_region = []
    used_template_durations_s = []
    kept = 0
    for cid in tqdm(cids, desc=f"Region {region}", unit="unit"):
        cid = int(cid)
        strength_val = _mean_strength(cid)
        if not np.isfinite(strength_val) or strength_val < STPR_STRENGTH_MIN:
            continue

        tpl_a = templates_a.get(cid)
        tpl_b = templates_b.get(cid) if use_split else None
        if tpl_a is None and tpl_b is None:
            continue

        spikes_c = spike_times_by_cluster.get(cid, np.array([]))
        if spikes_c.size == 0:
            continue

        bin_idx = ((spikes_c - t_start) / DETECTION_BIN_SIZE).astype(int)
        mask = (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(mask):
            continue
        counts = np.bincount(bin_idx[mask], minlength=n_bins).astype(float)
        rate = counts / DETECTION_BIN_SIZE
        rate_smooth = smooth_signal(rate, kernel)

        scores = []
        template_lengths_bins = []
        if tpl_a is not None and tpl_a.size >= 3:
            scores.append(normalized_xcorr(rate_smooth, tpl_a))
            template_lengths_bins.append(tpl_a.size)
        if tpl_b is not None and tpl_b.size >= 3:
            scores.append(normalized_xcorr(rate_smooth, tpl_b))
            template_lengths_bins.append(tpl_b.size)
        if not scores:
            continue

        if len(scores) == 1:
            score = scores[0]
        else:
            score = np.nanmean(np.vstack(scores), axis=0)

        if RECTIFY_MATCH_SCORES:
            score = np.maximum(0.0, score)

        weight = strength_val if WEIGHT_BY_COUPLING else 1.0
        valid = np.isfinite(score)
        if np.any(valid):
            score_sum[valid] += weight * score[valid]
            weight_sum[valid] += weight
        if template_lengths_bins:
            used_template_durations_s.append(
                float(np.mean(template_lengths_bins)) * DETECTION_BIN_SIZE
            )
        used_cids_region.append(cid)
        used_cluster_ids.add(cid)
        kept += 1

    region_used_cluster_ids[region] = np.asarray(used_cids_region, dtype=int)
    if used_template_durations_s:
        region_template_duration_s[region] = float(np.nanmedian(used_template_durations_s))
    else:
        region_template_duration_s[region] = np.nan

    if not np.any(weight_sum > 0):
        region_scores_raw[region] = np.full(n_bins, np.nan, dtype=float)
        continue

    region_score = np.full(n_bins, np.nan, dtype=float)
    valid_bins = weight_sum > 0
    region_score[valid_bins] = score_sum[valid_bins] / weight_sum[valid_bins]
    region_scores_raw[region] = region_score
    print(f"Region {region}: used {kept}/{len(cids)} units")

    if PACKET_SCORE_ZSCORE:
        region_scores_plot[region] = robust_zscore(region_score)
    else:
        region_scores_plot[region] = region_score.copy()

    peaks = detect_peaks(
        region_scores_plot[region], bin_centers, PACKET_THRESHOLD, MIN_PACKET_GAP_S
    )
    region_event_indices[region] = peaks
    peak_times = bin_centers[peaks] if peaks else np.array([])
    region_event_times[region] = peak_times
    region_event_scores[region] = region_scores_plot[region][peaks] if peaks else np.array([])

    template_duration_s = region_template_duration_s.get(region, np.nan)
    if peak_times.size > 0 and np.isfinite(template_duration_s) and template_duration_s > 0:
        half_window_s = template_duration_s / 2.0
        region_event_windows[region] = np.column_stack(
            [peak_times - half_window_s, peak_times + half_window_s]
        )
    else:
        region_event_windows[region] = np.empty((0, 2), dtype=float)


# %% Raster Setup
# ---- Build raster (sorted) ----
df_units_plot = df_units[df_units["cluster_id"].isin(list(used_cluster_ids))].copy()
df_units_psth_plot = df_units_psth[df_units_psth["cluster_id"].isin(list(used_cluster_ids))].copy()
if df_units_plot.empty:
    raise RuntimeError("No units contributed to packet detection for raster plotting.")

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

sorting_metric = sort_map.get(SORT_CHOICE, "depth")
trials = None
if session is not None:
    if hasattr(session, "trials"):
        trials = session.trials
    elif isinstance(session, dict):
        trials = session.get("trials")

if trials is None:
    raise RuntimeError("Session trials not available for trial raster.")

def _get_trial_event(trials_obj, key, trial_idx):
    if trials_obj is None or key not in trials_obj.keys():
        return np.nan
    events = np.asarray(trials_obj[key])
    if trial_idx < 0 or trial_idx >= len(events):
        return np.nan
    return events[trial_idx]

def _n_trials(trials_obj):
    if trials_obj is None:
        return 0
    shape = getattr(trials_obj, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    if isinstance(trials_obj, dict):
        for values in trials_obj.values():
            arr = np.asarray(values)
            if arr.ndim > 0:
                return int(len(arr))
        return 0
    if hasattr(trials_obj, "__len__"):
        return int(len(trials_obj))
    return 0

def _safe_trial_idx(trials_obj, trial_idx):
    n_trials = _n_trials(trials_obj)
    if trial_idx is None or trial_idx < 0 or trial_idx >= n_trials:
        return 0
    return int(trial_idx)

df_units_sorted, sort_label = plotting_utils._merge_metric(
    df_units_plot,
    sorting_metric,
    df_res=data.get("df_res"),
    df_coupling=df_coupling_spont,
    df_coupling_task=df_coupling_task,
    df_coupling_iti=df_coupling_iti,
    df_firing_rate=None,
)
df_units_sorted, region_order_sorted, sort_label = plotting_utils._sort_within_regions(
    df_units_sorted, sort_label
)

cluster_index_map = dict(
    zip(df_units_sorted["cluster_id"].values, df_units_sorted.index.values)
)
cluster_region_map = dict(
    zip(df_units_sorted["cluster_id"].values, df_units_sorted["acronym"].values)
)
region_row_bounds = {}
for acronym in region_order_sorted:
    group = df_units_sorted[df_units_sorted["acronym"] == acronym]
    if group.empty:
        continue
    region_row_bounds[acronym] = (
        float(group.index.min()) - 0.5,
        float(group.index.max()) + 0.5,
    )

template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
base_color = plotting_utils._template_base_color(template)
region_colors = plotting_utils._region_color_map(region_order_sorted)
packet_rows = (1, 2, 3)

EVENT_STYLE_MAP = {
    "stimOn_times": ("Stim On", "blue"),
    "firstMovement_times": ("First Move", "green"),
    "response_times": ("Response", "purple"),
    "feedback_times": ("Feedback", "red"),
}


def _sanitize_window_bounds(window_start, window_end):
    window_start = max(float(window_start), float(t_start))
    window_end = min(float(window_end), float(t_end))
    if not np.isfinite(window_start) or not np.isfinite(window_end) or window_end <= window_start:
        raise ValueError(f"Invalid window bounds: {window_start} to {window_end}")
    return window_start, window_end


def _build_packet_figure(raster_title="Raster"):
    fig_base = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(raster_title, "Packet Probability", "Population PSTH"),
    )
    return FigureResampler(fig_base) if FigureResampler is not None else fig_base


def _extract_raster_window_spikes(window_start, window_end, t_offset=0.0):
    window_mask = (spike_times >= window_start) & (spike_times <= window_end)
    window_spike_times = spike_times[window_mask] - t_offset
    window_spike_clusters = spike_clusters[window_mask]

    spike_mask = np.isin(window_spike_clusters, df_units_sorted["cluster_id"].values)
    window_spike_times = window_spike_times[spike_mask]
    window_spike_clusters = window_spike_clusters[spike_mask]

    spike_y = pd.Series(window_spike_clusters).map(cluster_index_map).to_numpy()
    spike_regions = pd.Series(window_spike_clusters).map(cluster_region_map).to_numpy()
    return window_spike_times, window_spike_clusters, spike_y, spike_regions


def _add_raster_trace(fig, x_vals, y_vals, spike_clusters_vals, spike_regions_vals):
    raster_trace = go.Scattergl(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(color=base_color, size=3, symbol="line-ns-open"),
        customdata=np.column_stack([spike_clusters_vals, spike_regions_vals]),
        hovertemplate=(
            "Time: %{x:.3f}s<br>Unit: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
        ),
        name="Spikes",
    )
    if FigureResampler is not None and len(x_vals) > 0:
        fig.add_trace(
            raster_trace,
            max_n_samples=len(x_vals),
            hf_x=x_vals,
            hf_y=y_vals,
            row=1,
            col=1,
        )
    else:
        fig.add_trace(raster_trace, row=1, col=1)


def _add_region_background(fig, x_start, x_end):
    for acronym in region_order_sorted:
        bounds = region_row_bounds.get(acronym)
        if bounds is None:
            continue
        y0, y1 = bounds
        fill_color = plotting_utils._color_to_rgba(region_colors.get(acronym), alpha=0.18)
        fig.add_shape(
            type="rect",
            x0=x_start,
            x1=x_end,
            y0=y0,
            y1=y1,
            line=dict(width=0),
            fillcolor=fill_color,
            layer="below",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x_end,
            y=(y0 + y1) / 2,
            xanchor="left",
            yanchor="middle",
            text=acronym,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xshift=10,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=region_colors.get(acronym), size=8),
                name=acronym,
                showlegend=True,
            ),
            row=1,
            col=1,
        )


def _add_packet_boxes(fig, window_start, window_end, t_offset=0.0, align_to_event=False):
    packet_box_color = "rgba(220, 20, 60, 0.95)"
    for region in region_order_sorted:
        bounds = region_row_bounds.get(region)
        if bounds is None:
            continue
        packet_windows = np.asarray(
            region_event_windows.get(region, np.empty((0, 2), dtype=float)),
            dtype=float,
        )
        if packet_windows.size == 0:
            continue
        y0, y1 = bounds
        for box_start, box_end in packet_windows:
            if not np.isfinite(box_start) or not np.isfinite(box_end):
                continue
            if box_end < window_start or box_start > window_end:
                continue
            x0 = max(float(box_start), float(window_start))
            x1 = min(float(box_end), float(window_end))
            if x1 <= x0:
                continue
            if align_to_event:
                x0 -= float(t_offset)
                x1 -= float(t_offset)
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color=packet_box_color, width=2, dash="dash"),
                fillcolor="rgba(0, 0, 0, 0)",
                layer="above",
                row=1,
                col=1,
            )


def _add_packet_probability_panel(fig, row, window_start, window_end, t_offset=0.0, align_to_event=False):
    for region in region_order_sorted:
        score_plot = region_scores_plot.get(region)
        if score_plot is None:
            continue
        color = region_colors.get(region)
        mask = (bin_centers >= window_start) & (bin_centers <= window_end)
        x_vals = bin_centers[mask] - t_offset if align_to_event else bin_centers[mask]
        y_vals = score_plot[mask]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=color),
                name=f"{region} prob",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        ev_times = np.asarray(region_event_times.get(region, np.array([])), dtype=float)
        ev_scores = np.asarray(region_event_scores.get(region, np.array([])), dtype=float)
        if ev_times.size == 0:
            continue
        keep = (ev_times >= window_start) & (ev_times <= window_end)
        if not np.any(keep):
            continue
        ev_x = ev_times[keep] - t_offset if align_to_event else ev_times[keep]
        ev_y = ev_scores[keep]
        fig.add_trace(
            go.Scatter(
                x=ev_x,
                y=ev_y,
                mode="markers",
                marker=dict(symbol="star", size=9, color=color),
                name=f"{region} packets",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    fig.add_hline(
        y=PACKET_THRESHOLD,
        line=dict(color="black", width=1.5, dash="dash"),
        row=row,
        col=1,
    )


def _add_population_psth_panel(fig, row, window_start, window_end, t_offset=0.0, align_to_event=False):
    if df_units_psth_plot.empty:
        return
    bin_size = float(config_plot.get("POP_BIN_SIZE", 0.005))
    if not np.isfinite(bin_size) or bin_size <= 0:
        bin_size = 0.005
    smooth_window_s = 0.05
    smooth_bins = max(1, int(round(smooth_window_s / bin_size)))

    window_mask = (spike_times >= window_start) & (spike_times <= window_end)
    window_spike_times_all = spike_times[window_mask]
    window_spike_clusters_all = spike_clusters[window_mask]

    if align_to_event:
        psth_spike_times_all = window_spike_times_all - t_offset
        psth_start = window_start - t_offset
        psth_end = window_end - t_offset
    else:
        psth_spike_times_all = window_spike_times_all
        psth_start = window_start
        psth_end = window_end

    bins = np.arange(psth_start, psth_end + bin_size, bin_size)
    if bins.size < 2 or bins[-1] < psth_end:
        bins = np.r_[bins, psth_end]
    if bins.size < 2:
        return

    bin_centers_local = (bins[:-1] + bins[1:]) / 2
    for acronym in region_order_sorted:
        region_ids = df_units_psth_plot.loc[
            df_units_psth_plot["acronym"] == acronym, "cluster_id"
        ].to_numpy()
        if region_ids.size == 0:
            continue
        region_mask = np.isin(window_spike_clusters_all, region_ids)
        region_spike_times = psth_spike_times_all[region_mask]
        counts, _ = np.histogram(region_spike_times, bins=bins)
        rate = counts / (len(region_ids) * bin_size)
        rate_smoothed = plotting_utils._moving_mean(rate, smooth_bins)
        fig.add_trace(
            go.Scatter(
                x=bin_centers_local,
                y=rate_smoothed,
                mode="lines",
                line=dict(color=region_colors.get(acronym)),
                name=f"{acronym} PSTH",
                showlegend=False,
            ),
            row=row,
            col=1,
        )


def _add_event_lines(fig, event_times_map, window_start, window_end, rows, t_offset=0.0, align_to_event=False):
    for event_name, (label, color) in EVENT_STYLE_MAP.items():
        event_times = event_times_map.get(event_name)
        if event_times is None:
            continue
        event_times = np.atleast_1d(np.asarray(event_times, dtype=float))
        keep = np.isfinite(event_times) & (event_times >= window_start) & (event_times <= window_end)
        event_times = event_times[keep]
        if event_times.size == 0:
            continue
        x_vals = event_times - t_offset if align_to_event else event_times
        for row in rows:
            for x_event in x_vals:
                fig.add_vline(x=float(x_event), line=dict(color=color, width=1.5), row=row, col=1)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )


# %% Raster Views
# ---- Plot (trial view) ----
def plot_trial_view(trial_idx):
    trial_idx = _safe_trial_idx(trials, trial_idx)

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in trials.keys():
        align_event = "stimOn_times"
    t_align = _get_trial_event(trials, align_event, trial_idx)

    t_stim_on = _get_trial_event(trials, "stimOn_times", trial_idx)
    t_first_move = _get_trial_event(trials, "firstMovement_times", trial_idx)
    t_response = _get_trial_event(trials, "response_times", trial_idx)
    t_feedback = _get_trial_event(trials, "feedback_times", trial_idx)

    if np.isnan(t_align):
        t_align = t_stim_on

    raster_pre = config_plot.get("RASTER_WINDOW_PRE", 1.0)
    raster_post = config_plot.get("RASTER_WINDOW_POST", 2.0)
    use_event_window = config_plot.get("TRIAL_RASTER_USE_EVENT_WINDOW", True)
    if use_event_window:
        win_start = config_plot.get("PSTH_WINDOW_START", -raster_pre)
        win_end = config_plot.get("PSTH_WINDOW_END", raster_post)
        valid_events = [t for t in [t_stim_on, t_first_move, t_feedback] if np.isfinite(t)]
        if valid_events:
            t_start_plot = min(valid_events) + win_start
            t_end_plot = max(valid_events) + win_end
        else:
            t_start_plot = t_align - raster_pre
            t_end_plot = t_align + raster_post
    else:
        t_start_plot = t_align - raster_pre
        t_end_plot = t_align + raster_post

    align_to_event = config_plot.get(
        "RASTER_ALIGN_TO_EVENT", config_plot.get("RASTER_ALIGN_TO_STIM_ON", True)
    )
    if align_to_event:
        t_offset = t_align
        xlabel_text = f"Time from {event_label(align_event)} (s)"
    else:
        t_offset = 0.0
        xlabel_text = "Time in session (s)"

    cont_l = trials["contrastLeft"][trial_idx] if "contrastLeft" in trials.keys() else np.nan
    cont_r = trials["contrastRight"][trial_idx] if "contrastRight" in trials.keys() else np.nan
    if not np.isnan(cont_l):
        contrast_val = cont_l
        stim_side = "Left"
    elif not np.isnan(cont_r):
        contrast_val = cont_r
        stim_side = "Right"
    else:
        contrast_val = 0
        stim_side = "Zero"

    choice_map = {1: "Left", -1: "Right", 0: "NoGo"}
    response_str = choice_map.get(trials["choice"][trial_idx], "NA") if "choice" in trials.keys() else "NA"
    fb_val = trials["feedbackType"][trial_idx] if "feedbackType" in trials.keys() else np.nan
    outcome_str = "Correct" if fb_val == 1 else "Incorrect"
    plot_title = (
        f"Trial {trial_idx} | Contrast: {contrast_val} ({stim_side}) | "
        f"Response: {response_str} | {outcome_str}"
    )

    x_start = t_start_plot - t_offset
    x_end = t_end_plot - t_offset
    window_spike_times, window_spike_clusters, spike_y, spike_regions = _extract_raster_window_spikes(
        t_start_plot, t_end_plot, t_offset=t_offset
    )

    fig = _build_packet_figure()
    _add_raster_trace(fig, window_spike_times, spike_y, window_spike_clusters, spike_regions)
    _add_region_background(fig, x_start, x_end)
    _add_packet_boxes(
        fig, t_start_plot, t_end_plot, t_offset=t_offset, align_to_event=align_to_event
    )
    _add_packet_probability_panel(
        fig, row=2, window_start=t_start_plot, window_end=t_end_plot, t_offset=t_offset, align_to_event=align_to_event
    )
    _add_population_psth_panel(
        fig, row=3, window_start=t_start_plot, window_end=t_end_plot, t_offset=t_offset, align_to_event=align_to_event
    )

    event_times_map = {
        "stimOn_times": t_stim_on,
        "firstMovement_times": t_first_move,
        "response_times": t_response,
        "feedback_times": t_feedback,
    }
    _add_event_lines(
        fig,
        event_times_map,
        t_start_plot,
        t_end_plot,
        rows=packet_rows,
        t_offset=t_offset,
        align_to_event=align_to_event,
    )

    ylabel_text = (
        f"Good Units (n={len(df_units_sorted)})"
        if config_plot.get("PLOT_ONLY_GOOD_UNITS", False)
        else f"All Units (n={len(df_units_sorted)})"
    )

    fig.update_yaxes(
        title_text=ylabel_text,
        row=1,
        col=1,
        showticklabels=False,
        range=[-0.5, len(df_units_sorted) - 0.5],
    )
    fig.update_yaxes(
        title_text="Packet score (z)" if PACKET_SCORE_ZSCORE else "Packet score",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=3, col=1)
    fig.update_xaxes(title_text=xlabel_text, row=3, col=1)

    fig.update_layout(
        title=f"{plot_title} | Sort: {sort_label}",
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    for row in packet_rows:
        fig.update_xaxes(range=[x_start, x_end], row=row, col=1)

    return fig


def plot_time_window_view(window_start=TIME_WINDOW_START, window_end=TIME_WINDOW_END):
    window_start, window_end = _sanitize_window_bounds(window_start, window_end)

    window_spike_times, window_spike_clusters, spike_y, spike_regions = _extract_raster_window_spikes(
        window_start, window_end, t_offset=0.0
    )

    fig = _build_packet_figure(raster_title="Raster")
    _add_raster_trace(fig, window_spike_times, spike_y, window_spike_clusters, spike_regions)
    _add_region_background(fig, window_start, window_end)
    _add_packet_boxes(fig, window_start, window_end, t_offset=0.0, align_to_event=False)
    _add_packet_probability_panel(
        fig, row=2, window_start=window_start, window_end=window_end, t_offset=0.0, align_to_event=False
    )
    _add_population_psth_panel(
        fig, row=3, window_start=window_start, window_end=window_end, t_offset=0.0, align_to_event=False
    )

    event_times_map = {
        event_name: np.asarray(trials[event_name]) if event_name in trials.keys() else None
        for event_name in EVENT_STYLE_MAP
    }
    _add_event_lines(
        fig,
        event_times_map,
        window_start,
        window_end,
        rows=packet_rows,
        t_offset=0.0,
        align_to_event=False,
    )

    ylabel_text = (
        f"Good Units (n={len(df_units_sorted)})"
        if config_plot.get("PLOT_ONLY_GOOD_UNITS", False)
        else f"All Units (n={len(df_units_sorted)})"
    )
    fig.update_yaxes(
        title_text=ylabel_text,
        row=1,
        col=1,
        showticklabels=False,
        range=[-0.5, len(df_units_sorted) - 0.5],
    )
    fig.update_yaxes(
        title_text="Packet score (z)" if PACKET_SCORE_ZSCORE else "Packet score",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=3, col=1)
    fig.update_xaxes(title_text="Time in session (s)", row=3, col=1)

    fig.update_layout(
        title=f"Session window {window_start:.2f}-{window_end:.2f}s | Sort: {sort_label}",
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    for row in packet_rows:
        fig.update_xaxes(range=[window_start, window_end], row=row, col=1)

    return fig


# %% Trial Raster
fig_trial = plot_trial_view(TRIAL_IDX)
show_fig(fig_trial, renderer=PLOTLY_RENDERER)


# %% Time Window Raster
fig_time_window = plot_time_window_view(TIME_WINDOW_START, TIME_WINDOW_END)
show_fig(fig_time_window, renderer=PLOTLY_RENDERER)


# %% Summary
# ---- Summary ----
print("Detected packet counts per region:")
for region in region_order:
    n_events = len(region_event_times.get(region, []))
    print(f"  {region}: {n_events}")


# %% Packet PSTH
# ---- Packet PSTH (stim-response style) ----
packet_region = PACKET_PSTH_REGION or (region_order[0] if region_order else None)
if packet_region is None or packet_region not in region_event_times:
    print("Packet PSTH: no valid region selected.")
else:
    packet_times = np.asarray(region_event_times.get(packet_region, []), dtype=float)
    if packet_times.size == 0:
        print(f"Packet PSTH: no packet events for region {packet_region}.")
    else:
        align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
        if align_event not in trials.keys():
            align_event = "stimOn_times"

        event_series = np.asarray(trials[align_event])
        if event_series is None or len(event_series) == 0:
            print("Packet PSTH: no event series available.")
        else:
            packet_spikes = SimpleNamespace(
                times=packet_times,
                clusters=np.zeros(len(packet_times), dtype=int),
            )

            raster_pre = config_plot.get("SINGLE_NEURON_RASTER_PRE", 0.5)
            raster_post = config_plot.get("SINGLE_NEURON_RASTER_POST", 1.0)
            bin_size = config_plot.get("SINGLE_NEURON_BIN_SIZE", 0.05)
            smooth_sigma = config_plot.get("SINGLE_NEURON_SMOOTH_SIGMA", 1)

            contrasts_to_plot = [1.0, 0.25, 0.125, 0.0625, 0.0]
            contrast_colors = {
                1.0: "rgba(0,0,0,1.0)",
                0.5: "rgba(51,51,51,1.0)",
                0.25: "rgba(102,102,102,1.0)",
                0.125: "rgba(153,153,153,1.0)",
                0.0625: "rgba(191,191,191,1.0)",
                0.0: "rgba(217,217,217,1.0)",
            }

            fig_psth = make_subplots(
                rows=3,
                cols=2,
                specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
                subplot_titles=(
                    "Left Stimuli PSTH (Packet Events)",
                    "Right Stimuli PSTH (Packet Events)",
                    "Left Stimuli Raster",
                    "Right Stimuli Raster",
                    "Global PSTH (All Trials)",
                ),
                row_heights=[0.25, 0.45, 0.3],
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
            )

            sides = [("Left", "contrastLeft"), ("Right", "contrastRight")]
            global_curves = []
            global_bin_centers = None

            for col_idx, (_side_label, contrast_key) in enumerate(sides, start=1):
                contrast_arr = np.asarray(trials[contrast_key]) if contrast_key in trials.keys() else None
                if contrast_arr is None:
                    continue

                current_raster_y = 0
                raster_x = []
                raster_y = []
                raster_colors = []

                for cont in contrasts_to_plot:
                    mask = (contrast_arr == cont) & (~np.isnan(event_series))
                    events = event_series[mask]
                    if len(events) == 0:
                        continue

                    psth_by_cluster, bin_centers = compute_psth_for_clusters(
                        packet_spikes,
                        [0],
                        events,
                        -raster_pre,
                        raster_post,
                        bin_size,
                        smooth_sigma,
                        show_progress=False,
                    )
                    psth_entry = psth_by_cluster.get(0)
                    if psth_entry and bin_centers is not None:
                        firing_rate = psth_entry["fr_smooth"]
                    else:
                        firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

                    color_val = contrast_colors.get(cont, "rgba(0,0,0,1.0)")
                    label_text = f"{cont * 100:.0f}%" if cont > 0 else "0%"
                    fig_psth.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=firing_rate,
                            mode="lines",
                            line=dict(color=color_val, width=2),
                            name=label_text,
                            showlegend=(col_idx == 1),
                            legendgroup=label_text,
                        ),
                        row=1,
                        col=col_idx,
                    )

                    if bin_centers is not None and len(firing_rate) > 0:
                        global_curves.append(np.asarray(firing_rate, dtype=float))
                        if global_bin_centers is None:
                            global_bin_centers = np.asarray(bin_centers, dtype=float)

                    for event_t in events:
                        t_start_evt = event_t - raster_pre
                        t_end_evt = event_t + raster_post
                        pkt = packet_times[(packet_times >= t_start_evt) & (packet_times <= t_end_evt)]
                        aligned = pkt - event_t
                        if len(aligned) > 0:
                            raster_x.extend(aligned.tolist())
                            raster_y.extend([current_raster_y] * len(aligned))
                            raster_colors.extend([color_val] * len(aligned))
                        current_raster_y += 1

                    current_raster_y += 2

                fig_psth.add_trace(
                    go.Scattergl(
                        x=raster_x,
                        y=raster_y,
                        mode="markers",
                        marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
                        showlegend=False,
                    ),
                    row=2,
                    col=col_idx,
                )

                fig_psth.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=col_idx)
                fig_psth.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=col_idx)
                fig_psth.update_xaxes(range=[-raster_pre, raster_post], row=1, col=col_idx)
                fig_psth.update_xaxes(range=[-raster_pre, raster_post], row=2, col=col_idx)
                fig_psth.update_yaxes(title_text="Packet rate (Hz)", row=1, col=col_idx)
                fig_psth.update_yaxes(title_text="Trials", showticklabels=False, row=2, col=col_idx)

            if global_curves and global_bin_centers is not None:
                min_len = min(len(curve) for curve in global_curves)
                curves = [curve[:min_len] for curve in global_curves]
                global_curve = np.nanmean(np.vstack(curves), axis=0)
                x_vals = global_bin_centers[:min_len]
                fig_psth.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=global_curve,
                        mode="lines",
                        line=dict(color=base_color, width=2),
                        name="Global PSTH",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
                fig_psth.add_vline(
                    x=0,
                    line=dict(color="blue", dash="dash"),
                    row=3,
                    col=1,
                )
                fig_psth.update_xaxes(
                    title_text=f"Time from {event_label(align_event)} (s)",
                    row=3,
                    col=1,
                )
                fig_psth.update_yaxes(title_text="Packet rate (Hz)", row=3, col=1)
            else:
                fig_psth.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="No valid trials found",
                    showarrow=False,
                    row=3,
                    col=1,
                )

            fig_psth.update_layout(
                title=f"Packet Events Response | Region {packet_region}",
                height=900,
                margin=dict(l=70, r=40, t=80, b=60),
            )
            fig_psth.update_layout(template=template, font=dict(color=base_color))

            show_fig(fig_psth, renderer=PLOTLY_RENDERER)

        def _plot_packet_conditioned_event(event_name, condition_type, title, pre=None, post=None):
            if event_name not in trials.keys():
                print(f"Packet PSTH: event '{event_name}' not in trials.")
                return
            event_series = np.asarray(trials[event_name])
            if event_series is None or len(event_series) == 0:
                print(f"Packet PSTH: no events for {event_name}.")
                return

            if condition_type == "choice":
                choice_arr = np.asarray(trials["choice"]) if "choice" in trials.keys() else None
                if choice_arr is None:
                    print("Packet PSTH: choice array missing.")
                    return
                conditions = [
                    ("Left", choice_arr == 1, "#1f77b4"),
                    ("Right", choice_arr == -1, "#ff7f0e"),
                ]
                sort_label = "Left/Right"
            elif condition_type == "feedback":
                feedback_arr = (
                    np.asarray(trials["feedbackType"]) if "feedbackType" in trials.keys() else None
                )
                if feedback_arr is None:
                    print("Packet PSTH: feedback array missing.")
                    return
                conditions = [
                    ("Correct", feedback_arr == 1, "#2ca02c"),
                    ("Incorrect", feedback_arr == -1, "#d62728"),
                ]
                sort_label = "Correct/Incorrect"
            else:
                print(f"Packet PSTH: unknown condition type {condition_type}.")
                return

            raster_pre_use = raster_pre if pre is None else float(pre)
            raster_post_use = raster_post if post is None else float(post)

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f"{event_label(event_name)} PSTH ({sort_label})",
                    f"{event_label(event_name)} Raster ({sort_label})",
                    f"{event_label(event_name)} Global PSTH",
                ),
            )

            global_curves = []
            global_bin_centers = None
            raster_x = []
            raster_y = []
            raster_colors = []
            current_raster_y = 0

            for label, cond_mask, color in conditions:
                if cond_mask is None:
                    continue
                mask = cond_mask & (~np.isnan(event_series))
                events = event_series[mask]
                if len(events) == 0:
                    continue

                psth_by_cluster, bin_centers = compute_psth_for_clusters(
                    packet_spikes,
                    [0],
                    events,
                    -raster_pre_use,
                    raster_post_use,
                    bin_size,
                    smooth_sigma,
                    show_progress=False,
                )
                psth_entry = psth_by_cluster.get(0)
                if psth_entry and bin_centers is not None:
                    firing_rate = psth_entry["fr_smooth"]
                else:
                    firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=firing_rate,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=label,
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

                if bin_centers is not None and len(firing_rate) > 0:
                    global_curves.append(np.asarray(firing_rate, dtype=float))
                    if global_bin_centers is None:
                        global_bin_centers = np.asarray(bin_centers, dtype=float)

                for event_t in events:
                    t_start_evt = event_t - raster_pre_use
                    t_end_evt = event_t + raster_post_use
                    pkt = packet_times[(packet_times >= t_start_evt) & (packet_times <= t_end_evt)]
                    aligned = pkt - event_t
                    if len(aligned) > 0:
                        raster_x.extend(aligned.tolist())
                        raster_y.extend([current_raster_y] * len(aligned))
                        raster_colors.extend([color] * len(aligned))
                    current_raster_y += 1

                current_raster_y += 2

            fig.add_trace(
                go.Scattergl(
                    x=raster_x,
                    y=raster_y,
                    mode="markers",
                    marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=1)
            fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=1)
            fig.update_xaxes(range=[-raster_pre_use, raster_post_use], row=1, col=1)
            fig.update_xaxes(range=[-raster_pre_use, raster_post_use], row=2, col=1)
            fig.update_yaxes(title_text="Packet rate (Hz)", row=1, col=1)
            fig.update_yaxes(title_text="Trials", showticklabels=False, row=2, col=1)

            if global_curves and global_bin_centers is not None:
                min_len = min(len(curve) for curve in global_curves)
                curves = [curve[:min_len] for curve in global_curves]
                global_curve = np.nanmean(np.vstack(curves), axis=0)
                x_vals = global_bin_centers[:min_len]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=global_curve,
                        mode="lines",
                        line=dict(color=base_color, width=2),
                        name="Global PSTH",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
                fig.add_vline(
                    x=0,
                    line=dict(color="blue", dash="dash"),
                    row=3,
                    col=1,
                )
                fig.update_xaxes(
                    title_text=f"Time from {event_label(event_name)} (s)",
                    row=3,
                    col=1,
                )
                fig.update_yaxes(title_text="Packet rate (Hz)", row=3, col=1)
            else:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="No valid trials found",
                    showarrow=False,
                    row=3,
                    col=1,
                )

            fig.update_layout(
                title=title,
                height=700,
                margin=dict(l=70, r=40, t=80, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_layout(template=template, font=dict(color=base_color))
            show_fig(fig, renderer=PLOTLY_RENDERER)

        def _plot_packet_events_from_series(event_specs, title, pre=None, post=None, xaxis_title="Time from event (s)"):
            raster_pre_use = raster_pre if pre is None else float(pre)
            raster_post_use = raster_post if post is None else float(post)

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Event-aligned packet PSTH",
                    "Event-aligned packet raster",
                    "Global PSTH",
                ),
            )

            global_curves = []
            global_bin_centers = None
            raster_x = []
            raster_y = []
            raster_colors = []
            current_raster_y = 0

            for label, event_times_arr, color in event_specs:
                event_times_arr = np.asarray(event_times_arr, dtype=float)
                event_times_arr = event_times_arr[np.isfinite(event_times_arr)]
                if event_times_arr.size == 0:
                    continue

                psth_by_cluster, bin_centers = compute_psth_for_clusters(
                    packet_spikes,
                    [0],
                    event_times_arr,
                    -raster_pre_use,
                    raster_post_use,
                    bin_size,
                    smooth_sigma,
                    show_progress=False,
                )
                psth_entry = psth_by_cluster.get(0)
                if psth_entry and bin_centers is not None:
                    firing_rate = psth_entry["fr_smooth"]
                else:
                    firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=firing_rate,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"{label} (n={len(event_times_arr)})",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

                if bin_centers is not None and len(firing_rate) > 0:
                    global_curves.append(np.asarray(firing_rate, dtype=float))
                    if global_bin_centers is None:
                        global_bin_centers = np.asarray(bin_centers, dtype=float)

                for event_t in event_times_arr:
                    t_start_evt = event_t - raster_pre_use
                    t_end_evt = event_t + raster_post_use
                    pkt = packet_times[(packet_times >= t_start_evt) & (packet_times <= t_end_evt)]
                    aligned = pkt - event_t
                    if len(aligned) > 0:
                        raster_x.extend(aligned.tolist())
                        raster_y.extend([current_raster_y] * len(aligned))
                        raster_colors.extend([color] * len(aligned))
                    current_raster_y += 1

                current_raster_y += 2

            fig.add_trace(
                go.Scattergl(
                    x=raster_x,
                    y=raster_y,
                    mode="markers",
                    marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=1)
            fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=1)
            fig.update_xaxes(range=[-raster_pre_use, raster_post_use], row=1, col=1)
            fig.update_xaxes(range=[-raster_pre_use, raster_post_use], row=2, col=1)
            fig.update_yaxes(title_text="Packet rate (Hz)", row=1, col=1)
            fig.update_yaxes(title_text="Events", showticklabels=False, row=2, col=1)

            if global_curves and global_bin_centers is not None:
                min_len = min(len(curve) for curve in global_curves)
                curves = [curve[:min_len] for curve in global_curves]
                global_curve = np.nanmean(np.vstack(curves), axis=0)
                x_vals = global_bin_centers[:min_len]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=global_curve,
                        mode="lines",
                        line=dict(color=base_color, width=2),
                        name="Global PSTH",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
                fig.add_vline(
                    x=0,
                    line=dict(color="blue", dash="dash"),
                    row=3,
                    col=1,
                )
                fig.update_xaxes(title_text=xaxis_title, row=3, col=1)
                fig.update_yaxes(title_text="Packet rate (Hz)", row=3, col=1)
            else:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="No valid events found",
                    showarrow=False,
                    row=3,
                    col=1,
                )

            fig.update_layout(
                title=title,
                height=700,
                margin=dict(l=70, r=40, t=80, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_layout(template=template, font=dict(color=base_color))
            show_fig(fig, renderer=PLOTLY_RENDERER)

        _plot_packet_conditioned_event(
            "firstMovement_times",
            "choice",
            f"Packet Events | First Movement | Region {packet_region}",
        )
        _plot_packet_conditioned_event(
            "feedback_times",
            "feedback",
            f"Packet Events | Feedback | Region {packet_region}",
            pre=0.0,
            post=6.0,
        )

        whisk_context_mode = str(PACKET_WHISK_EVENT_CONTEXT).strip().lower()
        whisk_context_suffix_map = {
            "all": "",
            "task": "_task",
            "iti": "_iti",
            "spont": "_spont",
        }
        if whisk_context_mode not in whisk_context_suffix_map:
            print(
                f"Packet PSTH: unsupported PACKET_WHISK_EVENT_CONTEXT='{PACKET_WHISK_EVENT_CONTEXT}'."
            )
        else:
            whisk_suffix = whisk_context_suffix_map[whisk_context_mode]
            whisk_event_specs = [
                (
                    "Brief whisk",
                    np.asarray(
                        wh_events_by_period.get(f"wh_brief_times{whisk_suffix}", np.array([])),
                        dtype=float,
                    ),
                    "#1f77b4",
                ),
                (
                    "Long whisk",
                    np.asarray(
                        wh_events_by_period.get(f"wh_long_times{whisk_suffix}", np.array([])),
                        dtype=float,
                    ),
                    "#d62728",
                ),
            ]
            if any(np.asarray(events, dtype=float).size > 0 for _label, events, _color in whisk_event_specs):
                _plot_packet_events_from_series(
                    whisk_event_specs,
                    title=(
                        f"Packet Events | Whisking ({whisk_context_mode}) | "
                        f"Region {packet_region}"
                    ),
                    xaxis_title="Time from whisk onset (s)",
                )
            else:
                print(
                    f"Packet PSTH: no whisk events found for context '{whisk_context_mode}'."
                )


# %% Packet Analysis Config
# ---- Clean packet-feature presentation config ----
PRESENT_PACKET_REGION = None  # defaults to packet_region, then first region with packets
PRESENT_CONTEXT_MODE = "all"  # "all", "task", "spont", "iti"
PRESENT_SHAPE_MODE = "raw"  # "raw", "normalized", "residual"
PRESENT_N_CLUSTERS = 4
PRESENT_PCA_COMPONENTS = 6
PRESENT_UMAP_PRE_PCA_COMPONENTS = 10
PRESENT_UMAP_NEIGHBORS = 20
PRESENT_UMAP_MIN_DIST = 0.2
PRESENT_UMAP_RANDOM_STATE = 0
PRESENT_TOP_COUPLING_FEATURES = 8
PRESENT_TOP_COUPLING_SCATTERS = 1
PRESENT_PACKET_ONSET_FRACTION = 0.25
PRESENT_PACKET_PERIOD_MIN_OVERLAP_S = 0.05
PRESENT_CLUSTER_FEATURES = [
    "packet_fraction",
    "peak_rate",
    "template_dot",
    "spike_com",
    "relative_rank_order",
    "temporal_width",
]
PRESENT_FEATURE_PAIR_SPECS = [
    ("spike_com", "peak_rate"),
]

# Compatibility with packet extraction helpers.
PACKET_EMBED_REGION = PRESENT_PACKET_REGION
PACKET_CONTEXT_MODE = PRESENT_CONTEXT_MODE
PACKET_PCA_COMPONENTS = PRESENT_PCA_COMPONENTS
PACKET_CLUSTER_K = PRESENT_N_CLUSTERS
PACKET_CLUSTER_PCS = 4
PACKET_KMEANS_N_INIT = 20
PACKET_KMEANS_MAX_ITER = 100


# %% Packet PCA Helpers
# ---- Packet PCA + clustering helpers ----
def _resample_vector_to_len(values, target_len):
    values = np.asarray(values, dtype=float)
    target_len = int(target_len)
    if target_len <= 0:
        return np.array([], dtype=float)
    if values.size == 0:
        return np.zeros(target_len, dtype=float)
    if values.size == target_len:
        return values.copy()
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values)


def _row_zscore_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix.copy()
    matrix_z = np.zeros_like(matrix, dtype=float)
    for row_idx, row in enumerate(matrix):
        row_mean = np.nanmean(row)
        row_std = np.nanstd(row)
        if np.isfinite(row_std) and row_std > 0:
            matrix_z[row_idx] = (row - row_mean) / row_std
        else:
            matrix_z[row_idx] = row - row_mean
    return matrix_z


def _coerce_interval_array(intervals):
    if intervals is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(intervals, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, 2)
    if arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    valid = np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])
    return np.asarray(arr[valid], dtype=float)


def _mask_times_in_intervals(times, intervals):
    times = np.asarray(times, dtype=float)
    intervals = _coerce_interval_array(intervals)
    mask = np.zeros(times.shape[0], dtype=bool)
    for t0, t1 in intervals:
        mask |= (times >= float(t0)) & (times <= float(t1))
    return mask


def _classify_packet_context(packet_times):
    packet_times = np.asarray(packet_times, dtype=float)
    spont_intervals, task_windows_local, iti_windows_local = _build_behavior_period_windows()
    task_intervals = (
        task_windows_local.loc[:, ["start", "end"]].to_numpy(dtype=float)
        if isinstance(task_windows_local, pd.DataFrame) and not task_windows_local.empty
        else np.empty((0, 2), dtype=float)
    )
    iti_intervals = (
        iti_windows_local.loc[:, ["start", "end"]].to_numpy(dtype=float)
        if isinstance(iti_windows_local, pd.DataFrame) and not iti_windows_local.empty
        else np.empty((0, 2), dtype=float)
    )
    task_mask = _mask_times_in_intervals(packet_times, task_intervals)
    spont_mask = _mask_times_in_intervals(packet_times, spont_intervals)
    iti_mask = _mask_times_in_intervals(packet_times, iti_intervals)

    overlap_mask = (
        task_mask.astype(int) + spont_mask.astype(int) + iti_mask.astype(int)
    ) > 1
    labels = np.full(packet_times.shape[0], "other", dtype=object)
    labels[task_mask & ~spont_mask] = "task"
    labels[spont_mask & ~task_mask & ~iti_mask] = "spont"
    labels[iti_mask & ~task_mask & ~spont_mask] = "iti"
    labels[overlap_mask] = "overlap"
    return labels, task_intervals, spont_intervals, iti_intervals


def _choose_packet_embed_region():
    preferred_region = PACKET_EMBED_REGION
    if preferred_region is None:
        preferred_region = globals().get("packet_region", None)
    if preferred_region in region_event_times and len(region_event_times.get(preferred_region, [])) > 0:
        return preferred_region
    for region in region_order_sorted:
        if len(region_event_times.get(region, [])) > 0:
            return region
    return None


def _packet_region_units_df(region):
    used_ids = np.asarray(region_used_cluster_ids.get(region, []), dtype=int)
    if used_ids.size == 0:
        return df_units_sorted.iloc[0:0].copy()
    mask = (df_units_sorted["acronym"] == region) & (df_units_sorted["cluster_id"].isin(used_ids))
    return df_units_sorted.loc[mask].copy()


def _packet_region_template_matrix(cluster_ids_region, target_len):
    rows = []
    for cid in np.asarray(cluster_ids_region, dtype=int):
        tpl_rows = []
        tpl_a = templates_a.get(int(cid))
        tpl_b = templates_b.get(int(cid)) if use_split else None
        if tpl_a is not None and np.asarray(tpl_a).size >= 3:
            tpl_rows.append(_resample_vector_to_len(tpl_a, target_len))
        if tpl_b is not None and np.asarray(tpl_b).size >= 3:
            tpl_rows.append(_resample_vector_to_len(tpl_b, target_len))
        if tpl_rows:
            rows.append(np.nanmean(np.vstack(tpl_rows), axis=0))
        else:
            rows.append(np.zeros(int(target_len), dtype=float))
    if not rows:
        return np.zeros((0, int(target_len)), dtype=float)
    return np.vstack(rows)


def _extract_region_packet_dataset(region):
    packet_times = np.asarray(region_event_times.get(region, []), dtype=float)
    packet_scores = np.asarray(region_event_scores.get(region, []), dtype=float)
    if packet_times.size == 0:
        raise RuntimeError(f"No detected packets available for region {region}.")

    region_units_df = _packet_region_units_df(region)
    if region_units_df.empty:
        raise RuntimeError(f"No units available for packet extraction in region {region}.")

    template_duration_s = float(region_template_duration_s.get(region, np.nan))
    packet_windows = np.asarray(
        region_event_windows.get(region, np.empty((0, 2), dtype=float)),
        dtype=float,
    )
    if (not np.isfinite(template_duration_s) or template_duration_s <= 0) and packet_windows.size > 0:
        template_duration_s = float(np.nanmedian(packet_windows[:, 1] - packet_windows[:, 0]))
    if not np.isfinite(template_duration_s) or template_duration_s <= 0:
        raise RuntimeError(f"Template duration is invalid for region {region}.")

    n_timebins = max(3, int(round(template_duration_s / DETECTION_BIN_SIZE)))
    packet_window_s = n_timebins * DETECTION_BIN_SIZE
    half_window_s = packet_window_s / 2.0
    rel_bin_edges = np.linspace(-half_window_s, half_window_s, n_timebins + 1)
    rel_bin_centers = (rel_bin_edges[:-1] + rel_bin_edges[1:]) / 2.0

    if packet_windows.shape != (packet_times.size, 2):
        packet_windows = np.column_stack([packet_times - half_window_s, packet_times + half_window_s])

    cluster_ids_region = region_units_df["cluster_id"].to_numpy(dtype=int)
    packet_tensor = np.zeros((packet_times.size, cluster_ids_region.size, n_timebins), dtype=float)
    for pkt_idx, peak_time in enumerate(
        tqdm(packet_times, desc=f"Extract packets {region}", unit="packet")
    ):
        for unit_idx, cid in enumerate(cluster_ids_region):
            spikes_rel = np.asarray(spike_times_by_cluster.get(int(cid), np.array([])), dtype=float) - peak_time
            mask = (spikes_rel >= rel_bin_edges[0]) & (spikes_rel <= rel_bin_edges[-1])
            if np.any(mask):
                counts, _ = np.histogram(spikes_rel[mask], bins=rel_bin_edges)
                rate = counts.astype(float) / DETECTION_BIN_SIZE
            else:
                rate = np.zeros(n_timebins, dtype=float)
            packet_tensor[pkt_idx, unit_idx] = smooth_signal(rate, kernel)

    packet_context, task_intervals, spont_intervals, iti_intervals = _classify_packet_context(packet_times)
    detection_template_matrix = _packet_region_template_matrix(cluster_ids_region, n_timebins)
    return dict(
        region=region,
        packet_times=packet_times,
        packet_scores=packet_scores,
        packet_windows=packet_windows,
        packet_context=packet_context,
        packet_tensor=packet_tensor,
        cluster_ids=cluster_ids_region,
        unit_table=region_units_df,
        template_matrix=detection_template_matrix,
        rel_bin_centers=rel_bin_centers,
        rel_bin_edges=rel_bin_edges,
        packet_window_s=packet_window_s,
        task_intervals=task_intervals,
        spont_intervals=spont_intervals,
        iti_intervals=iti_intervals,
        context_mode="all",
    )


def _compute_packet_pca(packet_tensor, n_components=6):
    packet_tensor = np.asarray(packet_tensor, dtype=float)
    if packet_tensor.ndim != 3 or packet_tensor.shape[0] == 0:
        raise ValueError("packet_tensor must have shape (n_packets, n_units, n_timebins).")
    flat_packets = packet_tensor.reshape(packet_tensor.shape[0], -1)
    feature_mean = np.nanmean(flat_packets, axis=0)
    flat_centered = np.nan_to_num(flat_packets - feature_mean, nan=0.0)

    u, s, vt = np.linalg.svd(flat_centered, full_matrices=False)
    max_components = min(int(n_components), flat_centered.shape[0], flat_centered.shape[1])
    if max_components < 1:
        raise RuntimeError("Not enough packet samples for PCA.")

    scores = u[:, :max_components] * s[:max_components]
    components = vt[:max_components]
    total_var = np.sum(s ** 2)
    if total_var > 0:
        explained_ratio = (s[:max_components] ** 2) / total_var
    else:
        explained_ratio = np.zeros(max_components, dtype=float)

    return dict(
        flat_packets=flat_packets,
        flat_centered=flat_centered,
        feature_mean=feature_mean,
        scores=scores,
        components=components,
        explained_ratio=explained_ratio,
    )


def _kmeans_numpy(x, n_clusters, n_init=20, max_iter=100, random_state=0):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must have shape (n_samples, n_features).")

    n_samples = x.shape[0]
    n_clusters = max(1, min(int(n_clusters), n_samples))
    if n_clusters == 1:
        centroid = np.nanmean(x, axis=0, keepdims=True)
        return np.zeros(n_samples, dtype=int), centroid

    rng = np.random.default_rng(int(random_state))
    best_labels = None
    best_centroids = None
    best_inertia = None

    for _ in range(int(n_init)):
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = x[init_idx].copy()
        labels = np.full(n_samples, -1, dtype=int)

        for _iter in range(int(max_iter)):
            dist2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dist2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            updated_centroids = []
            for cluster_idx in range(n_clusters):
                members = x[labels == cluster_idx]
                if members.size == 0:
                    updated_centroids.append(x[rng.integers(0, n_samples)])
                else:
                    updated_centroids.append(np.nanmean(members, axis=0))
            centroids = np.vstack(updated_centroids)

        inertia = float(np.sum((x - centroids[labels]) ** 2))
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    sort_order = np.argsort(best_centroids[:, 0])
    relabel_map = {int(old): int(new) for new, old in enumerate(sort_order)}
    labels_sorted = np.asarray([relabel_map[int(lbl)] for lbl in best_labels], dtype=int)
    centroids_sorted = best_centroids[sort_order]
    return labels_sorted, centroids_sorted


def _filter_packet_dataset_by_context(packet_dataset, context_mode):
    context_mode = str(context_mode).strip().lower()
    if context_mode not in {"all", "task", "spont", "iti", "other"}:
        raise ValueError(f"Unsupported PACKET_CONTEXT_MODE: {context_mode}")

    packet_context = np.asarray(packet_dataset.get("packet_context", []), dtype=object)
    if context_mode == "all":
        keep_mask = np.ones(packet_context.shape[0], dtype=bool)
    else:
        keep_mask = packet_context == context_mode

    filtered = dict(packet_dataset)
    filtered["packet_times"] = np.asarray(packet_dataset["packet_times"], dtype=float)[keep_mask]
    filtered["packet_scores"] = np.asarray(packet_dataset["packet_scores"], dtype=float)[keep_mask]
    filtered["packet_windows"] = np.asarray(packet_dataset["packet_windows"], dtype=float)[keep_mask]
    filtered["packet_context"] = packet_context[keep_mask]
    filtered["packet_tensor"] = np.asarray(packet_dataset["packet_tensor"], dtype=float)[keep_mask]
    filtered["context_mode"] = context_mode
    filtered["selection_mask"] = keep_mask
    return filtered


def _plot_packet_pca_scatter(packet_dataset, pca_result):
    scores = np.asarray(pca_result["scores"], dtype=float)
    explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
    x_vals = scores[:, 0] if scores.shape[1] >= 1 else np.zeros(scores.shape[0], dtype=float)
    y_vals = scores[:, 1] if scores.shape[1] >= 2 else np.zeros(scores.shape[0], dtype=float)
    evr1 = 100.0 * explained_ratio[0] if explained_ratio.size >= 1 else 0.0
    evr2 = 100.0 * explained_ratio[1] if explained_ratio.size >= 2 else 0.0
    customdata = np.column_stack(
        [
            np.arange(1, len(packet_dataset["packet_times"]) + 1, dtype=int),
            packet_dataset["packet_times"],
            packet_dataset["packet_scores"],
        ]
    )

    fig = go.Figure()
    context_mode = str(packet_dataset.get("context_mode", "all")).lower()
    packet_context = np.asarray(packet_dataset.get("packet_context", []), dtype=object)
    if context_mode == "all":
        context_style = {
            "task": ("Task", "#1f77b4"),
            "spont": ("Spont", "#ff7f0e"),
            "iti": ("ITI", "#2ca02c"),
            "other": ("Other", "#7f7f7f"),
            "overlap": ("Overlap", "#9467bd"),
        }
        for context_key in ("task", "spont", "iti", "other", "overlap"):
            mask = packet_context == context_key
            if not np.any(mask):
                continue
            label, color = context_style[context_key]
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask],
                    y=y_vals[mask],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0.9),
                    customdata=customdata[mask],
                    name=f"{label} (n={int(np.sum(mask))})",
                    hovertemplate=(
                        "Packet %{customdata[0]}<br>"
                        "Peak time: %{customdata[1]:.3f}s<br>"
                        "Peak score: %{customdata[2]:.2f}<br>"
                        "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=10, color=base_color, opacity=0.85),
                customdata=customdata,
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Peak score: %{customdata[2]:.2f}<br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=(
            f"Packet PCA | Region {packet_dataset['region']} | "
            f"Context: {context_mode.capitalize()}"
        ),
        template=template,
        font=dict(color=base_color),
        height=550,
        margin=dict(l=70, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=f"PC1 ({evr1:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({evr2:.1f}% var)")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_packet_cluster_scatter(packet_dataset, pca_result, cluster_labels):
    scores = np.asarray(pca_result["scores"], dtype=float)
    explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
    x_vals = scores[:, 0] if scores.shape[1] >= 1 else np.zeros(scores.shape[0], dtype=float)
    y_vals = scores[:, 1] if scores.shape[1] >= 2 else np.zeros(scores.shape[0], dtype=float)
    evr1 = 100.0 * explained_ratio[0] if explained_ratio.size >= 1 else 0.0
    evr2 = 100.0 * explained_ratio[1] if explained_ratio.size >= 2 else 0.0
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig = go.Figure()
    for cluster_idx in np.unique(cluster_labels):
        mask = np.asarray(cluster_labels) == cluster_idx
        customdata = np.column_stack(
            [
                np.arange(1, len(packet_dataset["packet_times"]) + 1, dtype=int)[mask],
                packet_dataset["packet_times"][mask],
                packet_dataset["packet_scores"][mask],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    size=10,
                    color=palette[int(cluster_idx) % len(palette)],
                    opacity=0.9,
                ),
                name=f"Cluster {int(cluster_idx) + 1} (n={int(np.sum(mask))})",
                customdata=customdata,
                hovertemplate=(
                    "Cluster "
                    + str(int(cluster_idx) + 1)
                    + "<br>Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Peak score: %{customdata[2]:.2f}<br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"Packet PCA + k-means | Region {packet_dataset['region']} | "
            f"Context: {str(packet_dataset.get('context_mode', 'all')).capitalize()}"
        ),
        template=template,
        font=dict(color=base_color),
        height=550,
        margin=dict(l=70, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=f"PC1 ({evr1:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({evr2:.1f}% var)")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_packet_cluster_heatmaps(packet_dataset, cluster_labels):
    packet_tensor = np.asarray(packet_dataset["packet_tensor"], dtype=float)
    template_matrix = np.asarray(packet_dataset["template_matrix"], dtype=float)
    rel_bin_centers_ms = np.asarray(packet_dataset["rel_bin_centers"], dtype=float) * 1000.0
    unique_clusters = np.unique(cluster_labels)

    subplot_titles = ["Detection template matrix"]
    matrices = [_row_zscore_matrix(template_matrix)]
    for cluster_idx in unique_clusters:
        members = packet_tensor[np.asarray(cluster_labels) == cluster_idx]
        cluster_mean = np.nanmean(members, axis=0)
        matrices.append(_row_zscore_matrix(cluster_mean))
        subplot_titles.append(f"Cluster {int(cluster_idx) + 1} mean packet (n={members.shape[0]})")

    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices if mat.size > 0])
    cmax = np.nanpercentile(np.abs(finite_vals), 98) if finite_vals.size > 0 else 1.0
    cmax = max(float(cmax), 1.0)

    n_rows = len(matrices)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=tuple(subplot_titles),
    )
    for row_idx, matrix in enumerate(matrices, start=1):
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=rel_bin_centers_ms,
                y=np.arange(matrix.shape[0]),
                colorscale="RdBu_r",
                zmin=-cmax,
                zmax=cmax,
                colorbar=dict(title="z", len=0.9) if row_idx == 1 else None,
                showscale=row_idx == 1,
                hovertemplate=(
                    "Time: %{x:.1f} ms<br>"
                    "Unit row: %{y}<br>"
                    "Value: %{z:.2f}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="Units", showticklabels=False, row=row_idx, col=1)

    fig.update_xaxes(title_text="Time from packet peak (ms)", row=n_rows, col=1)
    fig.update_layout(
        title=(
            f"Packet cluster heatmaps | Region {packet_dataset['region']} | "
            f"Context: {str(packet_dataset.get('context_mode', 'all')).capitalize()} | "
            "rows z-scored across time"
        ),
        template=template,
        font=dict(color=base_color),
        height=280 * n_rows + 120,
        margin=dict(l=70, r=40, t=90, b=60),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


# %% Packet Feature Overview
# ---- Packet feature analysis below replaces the older exploratory packet PCA run ----


# %% Packet Feature Config
# ---- Packet feature extraction config ----
PACKET_FEATURE_REGION = PRESENT_PACKET_REGION
PACKET_FEATURE_CONTEXT_MODE = PRESENT_CONTEXT_MODE
PACKET_FEATURE_ONSET_FRACTION = PRESENT_PACKET_ONSET_FRACTION
PACKET_FEATURE_TOP_SCATTERS = PRESENT_TOP_COUPLING_SCATTERS
PACKET_FEATURE_EXCLUDE_TARGET_LINKED_FROM_RANKING = True


# %% Packet Feature Extraction
# ---- Per-packet / per-neuron feature extraction ----
PACKET_CONTRIBUTION_FEATURES = [
    "spike_count",
    "binary_participation",
    "packet_fraction",
    "baseline_normalized_count",
    "excess_count",
    "rate_auc",
    "peak_rate",
    "template_dot",
    "template_cosine",
]
PACKET_TIMING_FEATURES = [
    "t_first_spike",
    "t_median_spike",
    "t_com",
    "t_peak_rate",
    "t_onset_threshold",
    "relative_rank_order",
    "temporal_width",
]
PACKET_FEATURE_ORDER = PACKET_CONTRIBUTION_FEATURES + PACKET_TIMING_FEATURES
PACKET_TARGET_LINKED_FEATURES = {}
PACKET_FEATURE_LABELS = {
    "spike_count": "Spike count",
    "binary_participation": "Binary participation",
    "packet_fraction": "Packet fraction",
    "baseline_normalized_count": "Baseline-normalized count",
    "excess_count": "Excess count",
    "rate_auc": "Rate AUC",
    "peak_rate": "Peak rate",
    "template_dot": "Template dot",
    "template_cosine": "Template cosine",
    "t_first_spike": "Time to first spike",
    "t_median_spike": "Median spike time",
    "t_com": "Spike COM",
    "t_peak_rate": "Peak-rate time",
    "t_onset_threshold": "Onset threshold time",
    "relative_rank_order": "Relative rank order",
    "temporal_width": "Temporal width",
}
PACKET_FEATURE_ALIASES = {
    "spike_com": "t_com",
    "spike com": "t_com",
    "median_spike_time": "t_median_spike",
    "median spike time": "t_median_spike",
    "peak_rate_time": "t_peak_rate",
    "peak rate time": "t_peak_rate",
    "time_to_first_spike": "t_first_spike",
    "time to first spike": "t_first_spike",
}
PACKET_TIME_FEATURES = {
    "t_first_spike",
    "t_median_spike",
    "t_com",
    "t_peak_rate",
    "t_onset_threshold",
    "temporal_width",
}
PACKET_SIZE_SCALED_FEATURES = {
    "spike_count",
    "baseline_normalized_count",
    "excess_count",
    "rate_auc",
    "peak_rate",
    "template_dot",
}
PACKET_FEATURE_GROUPS = {
    **{name: "Contribution" for name in PACKET_CONTRIBUTION_FEATURES},
    **{name: "Timing" for name in PACKET_TIMING_FEATURES},
}


def _resolve_feature_name(feature_name):
    feature_key = str(feature_name).strip().lower().replace("-", "_")
    feature_key = feature_key.replace("__", "_")
    return PACKET_FEATURE_ALIASES.get(feature_key, feature_key)


def _feature_label(feature_name, normalized=False):
    feat = _resolve_feature_name(feature_name)
    label = PACKET_FEATURE_LABELS.get(feat, feat)
    if normalized and feat in PACKET_SIZE_SCALED_FEATURES:
        return f"{label} / packet size"
    return label


def _is_time_feature(feature_name):
    feat = _resolve_feature_name(feature_name)
    return feat in PACKET_TIME_FEATURES


def _display_feature_label(feature_name, stat="mean", normalized=False):
    feat = _resolve_feature_name(feature_name)
    label = _feature_label(feat, normalized=normalized)
    stat_norm = _normalize_feature_stat(stat)
    if _is_time_feature(feat):
        if stat_norm == "mean":
            return f"{label} (ms)"
        return f"{label} variance (ms^2)"
    if stat_norm == "var":
        return f"{label} variance"
    return label


def _display_feature_values(feature_name, values, stat="mean"):
    feat = _resolve_feature_name(feature_name)
    stat_norm = _normalize_feature_stat(stat)
    arr = np.asarray(values, dtype=float)
    if not _is_time_feature(feat):
        return arr
    if stat_norm == "mean":
        return arr * 1000.0
    return arr * (1000.0 ** 2)


def _normalize_feature_stat(stat):
    stat_norm = str(stat).strip().lower()
    if stat_norm not in {"mean", "var"}:
        raise ValueError(f"Unsupported feature summary stat: {stat}")
    return stat_norm


def _feature_summary_col(feature_name, normalized=False, stat="mean"):
    feat = _resolve_feature_name(feature_name)
    stat_norm = _normalize_feature_stat(stat)
    suffix = "norm" if normalized else "raw"
    if stat_norm == "mean":
        return f"{feat}__packet_{suffix}"
    return f"{feat}__packet_{suffix}_{stat_norm}"


def _feature_neuron_summary_col(feature_name, stat="mean"):
    feat = _resolve_feature_name(feature_name)
    stat_norm = _normalize_feature_stat(stat)
    if stat_norm == "mean":
        return feat
    return f"{feat}__{stat_norm}"


def _feature_stat_label(feature_name, stat="mean", normalized=False):
    base_label = _display_feature_label(feature_name, stat=stat, normalized=normalized)
    stat_norm = _normalize_feature_stat(stat)
    if stat_norm == "mean":
        return base_label
    return base_label


def _choose_packet_feature_region():
    preferred_region = PACKET_FEATURE_REGION
    if preferred_region is None:
        preferred_region = globals().get("packet_region", None)
    if preferred_region in region_event_times and len(region_event_times.get(preferred_region, [])) > 0:
        return preferred_region
    return _choose_packet_embed_region()


def _resolve_packet_feature_context_mode():
    return str(PACKET_FEATURE_CONTEXT_MODE).strip().lower()


def _count_spikes_in_intervals_local(spike_times_arr, intervals):
    spike_times_arr = np.asarray(spike_times_arr, dtype=float)
    intervals = _coerce_interval_array(intervals)
    if spike_times_arr.size == 0 or intervals.size == 0:
        return 0
    total = 0
    for t0, t1 in intervals:
        left = np.searchsorted(spike_times_arr, float(t0), side="left")
        right = np.searchsorted(spike_times_arr, float(t1), side="right")
        total += int(max(0, right - left))
    return int(total)


def _packet_context_intervals(packet_dataset, context_mode):
    context_mode = str(context_mode).strip().lower()
    if context_mode == "task":
        return packet_dataset.get("task_intervals", np.empty((0, 2), dtype=float))
    if context_mode == "spont":
        return packet_dataset.get("spont_intervals", np.empty((0, 2), dtype=float))
    if context_mode == "iti":
        return packet_dataset.get("iti_intervals", np.empty((0, 2), dtype=float))
    return None


def _estimate_baseline_rates(cluster_ids_region, packet_dataset, context_mode):
    intervals = _packet_context_intervals(packet_dataset, context_mode)
    if intervals is None:
        duration_s = float(t_end - t_start)
    else:
        intervals = _coerce_interval_array(intervals)
        duration_s = float(np.sum(intervals[:, 1] - intervals[:, 0])) if intervals.size > 0 else 0.0

    baseline_rates = {}
    for cid in np.asarray(cluster_ids_region, dtype=int):
        spikes_c = np.asarray(spike_times_by_cluster.get(int(cid), np.array([])), dtype=float)
        if duration_s <= 0 or spikes_c.size == 0:
            baseline_rates[int(cid)] = 0.0
            continue
        if intervals is None:
            n_spikes_ctx = int(np.sum((spikes_c >= t_start) & (spikes_c <= t_end)))
        else:
            n_spikes_ctx = _count_spikes_in_intervals_local(spikes_c, intervals)
        baseline_rates[int(cid)] = float(n_spikes_ctx) / float(duration_s)
    return baseline_rates, float(duration_s)


def _safe_cosine_similarity(x_vals, y_vals):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.shape != y_arr.shape:
        return np.nan
    x_norm = float(np.linalg.norm(x_arr))
    y_norm = float(np.linalg.norm(y_arr))
    if x_norm <= 0 or y_norm <= 0:
        return np.nan
    return float(np.dot(x_arr, y_arr) / (x_norm * y_norm))


def _compute_packet_neuron_feature_tables(packet_dataset, context_mode, onset_fraction=0.25):
    packet_times_local = np.asarray(packet_dataset["packet_times"], dtype=float)
    packet_scores_local = np.asarray(packet_dataset["packet_scores"], dtype=float)
    packet_context_local = np.asarray(packet_dataset["packet_context"], dtype=object)
    packet_tensor_local = np.asarray(packet_dataset["packet_tensor"], dtype=float)
    template_matrix_local = np.asarray(packet_dataset["template_matrix"], dtype=float)
    rel_bin_centers_local = np.asarray(packet_dataset["rel_bin_centers"], dtype=float)
    rel_bin_edges_local = np.asarray(packet_dataset["rel_bin_edges"], dtype=float)
    cluster_ids_region = np.asarray(packet_dataset["cluster_ids"], dtype=int)
    unit_table_local = packet_dataset["unit_table"][["cluster_id", "acronym"]].copy()

    baseline_rates_hz, baseline_duration_s = _estimate_baseline_rates(
        cluster_ids_region,
        packet_dataset,
        context_mode,
    )
    packet_window_s = float(packet_dataset["packet_window_s"])
    feature_rows = []

    coupling_lookup = df_tpl_for_merge[
        ["cluster_id", "coupling_strength", "coupling_delay_ms"]
    ].drop_duplicates(subset=["cluster_id"])
    unit_table_local = unit_table_local.merge(coupling_lookup, on="cluster_id", how="left")
    region_by_cid = dict(zip(unit_table_local["cluster_id"], unit_table_local["acronym"]))
    strength_by_cid = dict(zip(unit_table_local["cluster_id"], unit_table_local["coupling_strength"]))
    delay_by_cid = dict(zip(unit_table_local["cluster_id"], unit_table_local["coupling_delay_ms"]))

    for pkt_idx, peak_time in enumerate(
        tqdm(packet_times_local, desc=f"Packet features {packet_dataset['region']}", unit="packet")
    ):
        n_units = cluster_ids_region.size
        spike_count = np.zeros(n_units, dtype=float)
        binary_participation = np.zeros(n_units, dtype=float)
        packet_fraction = np.zeros(n_units, dtype=float)
        baseline_normalized_count = np.zeros(n_units, dtype=float)
        excess_count = np.zeros(n_units, dtype=float)
        rate_auc = np.zeros(n_units, dtype=float)
        peak_rate = np.zeros(n_units, dtype=float)
        template_dot = np.zeros(n_units, dtype=float)
        template_cosine = np.zeros(n_units, dtype=float)

        t_first_spike = np.full(n_units, np.nan, dtype=float)
        t_median_spike = np.full(n_units, np.nan, dtype=float)
        t_com = np.full(n_units, np.nan, dtype=float)
        t_peak_rate = np.full(n_units, np.nan, dtype=float)
        t_onset_threshold = np.full(n_units, np.nan, dtype=float)
        relative_rank_order = np.full(n_units, np.nan, dtype=float)
        temporal_width = np.full(n_units, np.nan, dtype=float)

        for unit_idx, cid in enumerate(cluster_ids_region):
            cid = int(cid)
            rate_trace = np.nan_to_num(packet_tensor_local[pkt_idx, unit_idx], nan=0.0)
            peak_rate_val = float(np.nanmax(rate_trace)) if rate_trace.size > 0 else 0.0
            peak_rate[unit_idx] = max(0.0, peak_rate_val) if np.isfinite(peak_rate_val) else 0.0
            rate_auc[unit_idx] = float(np.nansum(rate_trace) * DETECTION_BIN_SIZE)

            template_trace = (
                np.nan_to_num(template_matrix_local[unit_idx], nan=0.0)
                if template_matrix_local.shape[0] > unit_idx
                else np.zeros_like(rate_trace)
            )

            spikes_rel = np.asarray(spike_times_by_cluster.get(cid, np.array([])), dtype=float) - peak_time
            keep = (spikes_rel >= rel_bin_edges_local[0]) & (spikes_rel <= rel_bin_edges_local[-1])
            spikes_rel = np.asarray(spikes_rel[keep], dtype=float)
            n_spikes = int(spikes_rel.size)

            if n_spikes == 0:
                continue

            expected_count = float(baseline_rates_hz.get(cid, 0.0)) * packet_window_s
            spike_count[unit_idx] = float(n_spikes)
            binary_participation[unit_idx] = 1.0
            baseline_normalized_count[unit_idx] = (
                float(n_spikes) / expected_count if expected_count > 0 else np.nan
            )
            excess_count[unit_idx] = float(n_spikes) - expected_count
            template_dot[unit_idx] = float(np.dot(rate_trace, template_trace) * DETECTION_BIN_SIZE)
            template_cosine_val = _safe_cosine_similarity(rate_trace, template_trace)
            template_cosine[unit_idx] = (
                template_cosine_val if np.isfinite(template_cosine_val) else 0.0
            )

            t_first_spike[unit_idx] = float(np.min(spikes_rel))
            t_median_spike[unit_idx] = float(np.median(spikes_rel))
            t_com[unit_idx] = float(np.mean(spikes_rel))
            temporal_width[unit_idx] = float(np.std(spikes_rel)) if n_spikes > 1 else 0.0

            if peak_rate[unit_idx] > 0 and rate_trace.size > 0:
                peak_idx = int(np.nanargmax(rate_trace))
                t_peak_rate[unit_idx] = float(rel_bin_centers_local[peak_idx])
                onset_threshold_val = float(onset_fraction) * peak_rate[unit_idx]
                onset_mask = rate_trace >= onset_threshold_val
                if np.any(onset_mask):
                    t_onset_threshold[unit_idx] = float(rel_bin_centers_local[np.argmax(onset_mask)])

        total_packet_spikes = float(np.sum(spike_count))
        if total_packet_spikes > 0:
            packet_fraction = spike_count / total_packet_spikes

        active_units = np.where(binary_participation > 0)[0]
        if active_units.size == 1:
            relative_rank_order[active_units[0]] = 0.5
        elif active_units.size > 1:
            active_order = active_units[np.argsort(t_first_spike[active_units], kind="mergesort")]
            for rank_idx, unit_idx in enumerate(active_order):
                relative_rank_order[unit_idx] = float(rank_idx) / float(active_units.size - 1)

        for unit_idx, cid in enumerate(cluster_ids_region):
            feature_rows.append(
                {
                    "packet_idx": int(pkt_idx),
                    "packet_peak_time_s": float(packet_times_local[pkt_idx]),
                    "packet_score": float(packet_scores_local[pkt_idx]),
                    "packet_context": str(packet_context_local[pkt_idx]),
                    "cluster_id": int(cid),
                    "region": str(region_by_cid.get(int(cid), packet_dataset["region"])),
                    "coupling_strength": float(strength_by_cid.get(int(cid), np.nan)),
                    "coupling_delay_ms": float(delay_by_cid.get(int(cid), np.nan)),
                    "spike_count": float(spike_count[unit_idx]),
                    "binary_participation": float(binary_participation[unit_idx]),
                    "packet_fraction": float(packet_fraction[unit_idx]),
                    "baseline_normalized_count": float(baseline_normalized_count[unit_idx]),
                    "excess_count": float(excess_count[unit_idx]),
                    "rate_auc": float(rate_auc[unit_idx]),
                    "peak_rate": float(peak_rate[unit_idx]),
                    "template_dot": float(template_dot[unit_idx]),
                    "template_cosine": float(template_cosine[unit_idx]),
                    "t_first_spike": float(t_first_spike[unit_idx]),
                    "t_median_spike": float(t_median_spike[unit_idx]),
                    "t_com": float(t_com[unit_idx]),
                    "t_peak_rate": float(t_peak_rate[unit_idx]),
                    "t_onset_threshold": float(t_onset_threshold[unit_idx]),
                    "relative_rank_order": float(relative_rank_order[unit_idx]),
                    "temporal_width": float(temporal_width[unit_idx]),
                }
            )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        return feature_df, pd.DataFrame()

    summary = (
        feature_df.groupby(["cluster_id", "region"], as_index=False)[PACKET_FEATURE_ORDER]
        .mean()
        .copy()
    )
    var_rows = []
    for (cluster_id, region), group_df in feature_df.groupby(["cluster_id", "region"], sort=False):
        row = {"cluster_id": int(cluster_id), "region": str(region)}
        for feat in PACKET_FEATURE_ORDER:
            values = pd.to_numeric(group_df[feat], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            row[_feature_neuron_summary_col(feat, stat="var")] = (
                float(np.nanvar(values[valid])) if np.any(valid) else np.nan
            )
        var_rows.append(row)
    summary = summary.merge(pd.DataFrame(var_rows), on=["cluster_id", "region"], how="left")
    summary = summary.merge(
        unit_table_local[["cluster_id", "coupling_strength", "coupling_delay_ms"]].drop_duplicates(
            subset=["cluster_id"]
        ),
        on="cluster_id",
        how="left",
    )
    summary["n_packets"] = (
        feature_df.groupby("cluster_id").size().reindex(summary["cluster_id"]).to_numpy(dtype=int)
    )
    summary["n_active_packets"] = (
        feature_df.groupby("cluster_id")["binary_participation"]
        .sum()
        .reindex(summary["cluster_id"])
        .to_numpy(dtype=float)
    )
    for feat in PACKET_TIMING_FEATURES:
        summary[f"{feat}_valid_n"] = (
            feature_df.groupby("cluster_id")[feat]
            .apply(lambda s: int(np.sum(np.isfinite(np.asarray(s, dtype=float)))))
            .reindex(summary["cluster_id"])
            .to_numpy(dtype=int)
        )

    summary["context_mode"] = str(context_mode)
    summary["baseline_rate_hz"] = summary["cluster_id"].map(
        lambda cid: float(baseline_rates_hz.get(int(cid), 0.0))
    )
    summary["baseline_duration_s"] = float(baseline_duration_s)
    return feature_df, summary


def _packet_size_from_feature_df(feature_df, packet_dataset):
    packet_ids = np.arange(len(packet_dataset["packet_times"]), dtype=int)
    return (
        feature_df.groupby("packet_idx")["spike_count"]
        .sum()
        .reindex(packet_ids, fill_value=0.0)
        .to_numpy(dtype=float)
    )


def _transform_packet_size(packet_size):
    packet_size = np.asarray(packet_size, dtype=float)
    packet_size = np.where(np.isfinite(packet_size), np.maximum(packet_size, 0.0), 0.0)
    return np.log1p(packet_size)


def _add_normalized_feature_columns(feature_df, packet_size_by_idx):
    feature_df = feature_df.copy()
    packet_size_series = pd.Series(packet_size_by_idx, index=np.arange(len(packet_size_by_idx), dtype=int))
    size_for_row = feature_df["packet_idx"].map(packet_size_series).to_numpy(dtype=float)
    size_for_row = np.where(np.isfinite(size_for_row), size_for_row, 0.0)

    for feat in PACKET_FEATURE_ORDER:
        norm_col = f"{feat}__norm"
        values = pd.to_numeric(feature_df[feat], errors="coerce").to_numpy(dtype=float)
        if feat in PACKET_SIZE_SCALED_FEATURES:
            norm_vals = np.full(values.shape, np.nan, dtype=float)
            valid = np.isfinite(values) & np.isfinite(size_for_row) & (size_for_row > 0)
            norm_vals[valid] = values[valid] / size_for_row[valid]
            zero_mask = (size_for_row <= 0) & np.isfinite(values) & (values == 0)
            norm_vals[zero_mask] = 0.0
            feature_df[norm_col] = norm_vals
        else:
            feature_df[norm_col] = values
    return feature_df


def _weighted_variance_local(values, weights=None):
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    if weights is None:
        if not np.any(valid):
            return np.nan
        return float(np.nanvar(vals[valid]))
    w = np.asarray(weights, dtype=float)
    valid &= np.isfinite(w) & (w >= 0)
    if not np.any(valid):
        return np.nan
    vals = vals[valid]
    w = w[valid]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float(np.nanvar(vals))
    mean_val = float(np.sum(w * vals) / w_sum)
    return float(np.sum(w * (vals - mean_val) ** 2) / w_sum)


def _aggregate_packet_feature(group_df, feature_name, normalized=False, stat="mean"):
    feat = _resolve_feature_name(feature_name)
    stat_norm = _normalize_feature_stat(stat)
    col_name = f"{feat}__norm" if normalized else feat
    values = pd.to_numeric(group_df[col_name], errors="coerce").to_numpy(dtype=float)
    participation = pd.to_numeric(group_df["binary_participation"], errors="coerce").to_numpy(dtype=float)
    active_mask = participation > 0

    if feat in PACKET_TIMING_FEATURES:
        valid = np.isfinite(values) & active_mask
        if not np.any(valid):
            return np.nan
        weights = pd.to_numeric(group_df["packet_fraction"], errors="coerce").to_numpy(dtype=float)[valid]
        weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
        if stat_norm == "var":
            return _weighted_variance_local(values[valid], weights=weights)
        if float(np.sum(weights)) > 0:
            return float(np.average(values[valid], weights=weights))
        return float(np.mean(values[valid]))

    if feat == "binary_participation":
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.nan if stat_norm == "var" else 0.0
        if stat_norm == "var":
            return float(np.nanvar(values[valid]))
        return float(np.mean(values[valid]))

    valid = np.isfinite(values) & active_mask
    if not np.any(valid):
        valid = np.isfinite(values)
    if not np.any(valid):
        return np.nan if stat_norm == "var" else 0.0
    if stat_norm == "var":
        return float(np.nanvar(values[valid]))
    return float(np.mean(values[valid]))


def _aggregate_packet_coupling(group_df, target_col):
    values = pd.to_numeric(group_df[target_col], errors="coerce").to_numpy(dtype=float)
    participation = pd.to_numeric(group_df["binary_participation"], errors="coerce").to_numpy(dtype=float)
    active_mask = participation > 0
    valid = np.isfinite(values) & active_mask
    if not np.any(valid):
        return np.nan
    weights = pd.to_numeric(group_df["packet_fraction"], errors="coerce").to_numpy(dtype=float)[valid]
    weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
    if float(np.sum(weights)) > 0:
        return float(np.average(values[valid], weights=weights))
    return float(np.mean(values[valid]))


def _build_packet_summary_table(feature_df, packet_dataset):
    packet_df = pd.DataFrame(
        {
            "packet_idx": np.arange(len(packet_dataset["packet_times"]), dtype=int),
            "packet_peak_time_s": np.asarray(packet_dataset["packet_times"], dtype=float),
            "packet_score": np.asarray(packet_dataset["packet_scores"], dtype=float),
            "packet_context": np.asarray(packet_dataset["packet_context"], dtype=object),
        }
    )
    packet_size = _packet_size_from_feature_df(feature_df, packet_dataset)
    packet_df["packet_size"] = packet_size
    packet_df["packet_size_log"] = _transform_packet_size(packet_size)

    grouped = feature_df.groupby("packet_idx", sort=True)
    packet_df["coupling_strength__packet"] = [
        _aggregate_packet_coupling(grouped.get_group(pkt_idx), "coupling_strength")
        if pkt_idx in grouped.groups
        else np.nan
        for pkt_idx in packet_df["packet_idx"]
    ]
    packet_df["coupling_delay_ms__packet"] = [
        _aggregate_packet_coupling(grouped.get_group(pkt_idx), "coupling_delay_ms")
        if pkt_idx in grouped.groups
        else np.nan
        for pkt_idx in packet_df["packet_idx"]
    ]
    for feat in PACKET_FEATURE_ORDER:
        packet_df[_feature_summary_col(feat, normalized=False)] = [
            _aggregate_packet_feature(grouped.get_group(pkt_idx), feat, normalized=False, stat="mean")
            if pkt_idx in grouped.groups
            else (np.nan if feat in PACKET_TIMING_FEATURES else 0.0)
            for pkt_idx in packet_df["packet_idx"]
        ]
        packet_df[_feature_summary_col(feat, normalized=True)] = [
            _aggregate_packet_feature(grouped.get_group(pkt_idx), feat, normalized=True, stat="mean")
            if pkt_idx in grouped.groups
            else (np.nan if feat in PACKET_TIMING_FEATURES else 0.0)
            for pkt_idx in packet_df["packet_idx"]
        ]
        packet_df[_feature_summary_col(feat, normalized=False, stat="var")] = [
            _aggregate_packet_feature(grouped.get_group(pkt_idx), feat, normalized=False, stat="var")
            if pkt_idx in grouped.groups
            else np.nan
            for pkt_idx in packet_df["packet_idx"]
        ]
        packet_df[_feature_summary_col(feat, normalized=True, stat="var")] = [
            _aggregate_packet_feature(grouped.get_group(pkt_idx), feat, normalized=True, stat="var")
            if pkt_idx in grouped.groups
            else np.nan
            for pkt_idx in packet_df["packet_idx"]
        ]
    return packet_df


packet_feature_region = _choose_packet_feature_region()
packet_feature_context = _resolve_packet_feature_context_mode()
packet_feature_results = None
packet_feature_df = None
packet_feature_summary = None
packet_summary_df = None
if packet_feature_region is None:
    print("Packet features: no region with detected packets is available.")
else:
    packet_feature_dataset_all = _extract_region_packet_dataset(packet_feature_region)
    packet_feature_dataset = _filter_packet_dataset_by_context(
        packet_feature_dataset_all,
        packet_feature_context,
    )
    if packet_feature_dataset["packet_tensor"].shape[0] == 0:
        print(
            f"Packet features: no packets found for region {packet_feature_region} "
            f"and context '{packet_feature_context}'."
        )
    else:
        packet_feature_df, packet_feature_summary = _compute_packet_neuron_feature_tables(
            packet_feature_dataset,
            context_mode=packet_feature_context,
            onset_fraction=PACKET_FEATURE_ONSET_FRACTION,
        )
        packet_size = _packet_size_from_feature_df(packet_feature_df, packet_feature_dataset)
        packet_feature_df = _add_normalized_feature_columns(packet_feature_df, packet_size)
        packet_summary_df = _build_packet_summary_table(packet_feature_df, packet_feature_dataset)
        packet_feature_results = {
            "region": packet_feature_region,
            "context_mode": packet_feature_context,
            "packet_dataset": packet_feature_dataset,
            "packet_feature_df": packet_feature_df,
            "packet_feature_summary": packet_feature_summary,
            "packet_summary_df": packet_summary_df,
        }
        print(
            f"Packet features | Region {packet_feature_region} | "
            f"context={packet_feature_context} | packets={packet_feature_dataset['packet_tensor'].shape[0]} | "
            f"units={packet_feature_dataset['packet_tensor'].shape[1]} | rows={len(packet_feature_df)}"
        )
        print("Contribution features:", ", ".join(PACKET_CONTRIBUTION_FEATURES))
        print("Timing features:", ", ".join(PACKET_TIMING_FEATURES))


# %% Packet Feature Correlations
# ---- Compare packet-derived neuron summaries to coupling strength / delay ----
def _spearmanr_with_n(x_vals, y_vals):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        return np.nan, n_valid
    x_rank = pd.Series(x_arr[valid]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y_arr[valid]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), n_valid


def _feature_target_correlation_table(summary_df):
    rows = []
    target_specs = [
        ("coupling_strength", "Coupling strength"),
        ("coupling_delay_ms", "Coupling delay (ms)"),
    ]
    for feature_name in PACKET_FEATURE_ORDER:
        for target_col, target_label in target_specs:
            pearson_r, n_valid = _pearsonr_with_n(summary_df[feature_name], summary_df[target_col])
            spearman_r, _ = _spearmanr_with_n(summary_df[feature_name], summary_df[target_col])
            rows.append(
                {
                    "feature": feature_name,
                    "feature_label": _display_feature_label(feature_name, stat="mean"),
                    "group": PACKET_FEATURE_GROUPS.get(feature_name, "Other"),
                    "target": target_col,
                    "target_label": target_label,
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "abs_pearson_r": abs(pearson_r) if np.isfinite(pearson_r) else np.nan,
                    "abs_spearman_r": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                    "n_valid": int(n_valid),
                    "target_linked_to": PACKET_TARGET_LINKED_FEATURES.get(feature_name),
                }
            )
    return pd.DataFrame(rows)


def _plot_feature_correlation_heatmaps(corr_df, title_prefix):
    pearson_mat = corr_df.pivot(index="feature_label", columns="target_label", values="pearson_r")
    spearman_mat = corr_df.pivot(index="feature_label", columns="target_label", values="spearman_r")
    feature_order_labels = [_display_feature_label(name, stat="mean") for name in PACKET_FEATURE_ORDER]
    pearson_mat = pearson_mat.reindex(feature_order_labels)
    spearman_mat = spearman_mat.reindex(feature_order_labels)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pearson correlation", "Spearman correlation"),
        horizontal_spacing=0.16,
    )
    for col_idx, (mat, label) in enumerate(((pearson_mat, "Pearson"), (spearman_mat, "Spearman")), start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat.to_numpy(dtype=float),
                x=list(mat.columns),
                y=list(mat.index),
                colorscale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="r", len=0.9) if col_idx == 2 else None,
                showscale=(col_idx == 2),
                hovertemplate="Feature: %{y}<br>Target: %{x}<br>r=%{z:.2f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
    fig.update_layout(
        title=f"{title_prefix} | Packet feature vs coupling correlations",
        template=template,
        font=dict(color=base_color),
        height=max(520, 28 * len(feature_order_labels) + 180),
        margin=dict(l=120, r=60, t=90, b=60),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_feature_ranking_bars(corr_df, title_prefix):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Best features for coupling strength", "Best features for coupling delay"),
        horizontal_spacing=0.16,
    )
    target_specs = [
        ("coupling_strength", 1, "#1f77b4"),
        ("coupling_delay_ms", 2, "#d62728"),
    ]
    for target_col, col_idx, color in target_specs:
        sub = corr_df.loc[corr_df["target"] == target_col].copy()
        if PACKET_FEATURE_EXCLUDE_TARGET_LINKED_FROM_RANKING:
            sub = sub.loc[
                (sub["target_linked_to"].isna()) | (sub["target_linked_to"] != target_col)
            ].copy()
        sub = sub.sort_values("abs_spearman_r", ascending=True)
        fig.add_trace(
            go.Bar(
                x=sub["abs_spearman_r"],
                y=sub["feature_label"],
                orientation="h",
                marker=dict(color=color),
                customdata=np.column_stack([sub["spearman_r"], sub["n_valid"]]),
                hovertemplate=(
                    "Feature: %{y}<br>|Spearman|: %{x:.2f}<br>"
                    "Spearman: %{customdata[0]:.2f}<br>n=%{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Absolute Spearman r", row=1, col=col_idx)
    fig.update_layout(
        title=f"{title_prefix} | Feature ranking by absolute Spearman correlation",
        template=template,
        font=dict(color=base_color),
        height=max(520, 28 * len(PACKET_FEATURE_ORDER) + 180),
        margin=dict(l=150, r=40, t=90, b=60),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_feature_target_scatter_grid(summary_df, corr_df, title_prefix, top_k=3):
    top_k = max(1, int(top_k))
    target_specs = [
        ("coupling_strength", "Coupling strength"),
        ("coupling_delay_ms", "Coupling delay (ms)"),
    ]
    chosen = []
    for target_col, target_label in target_specs:
        sub = corr_df.loc[corr_df["target"] == target_col].copy()
        if PACKET_FEATURE_EXCLUDE_TARGET_LINKED_FROM_RANKING:
            sub = sub.loc[
                (sub["target_linked_to"].isna()) | (sub["target_linked_to"] != target_col)
            ].copy()
        sub = sub.sort_values("abs_spearman_r", ascending=False).head(top_k)
        chosen.append((target_col, target_label, sub))

    n_rows = max(len(sub) for _target_col, _target_label, sub in chosen)
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=tuple(
            title
            for _row in range(n_rows)
            for title in ("Strength feature scatter", "Delay feature scatter")
        ),
        horizontal_spacing=0.14,
        vertical_spacing=0.10,
    )
    color_map = {"coupling_strength": "#1f77b4", "coupling_delay_ms": "#d62728"}

    for col_idx, (target_col, target_label, sub) in enumerate(chosen, start=1):
        for row_idx in range(n_rows):
            if row_idx >= len(sub):
                continue
            row = sub.iloc[row_idx]
            feat = row["feature"]
            x_vals = pd.to_numeric(summary_df[feat], errors="coerce").to_numpy(dtype=float)
            x_vals = _display_feature_values(feat, x_vals, stat="mean")
            y_vals = pd.to_numeric(summary_df[target_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_vals) & np.isfinite(y_vals)
            fig.add_trace(
                go.Scatter(
                    x=x_vals[valid],
                    y=y_vals[valid],
                    mode="markers",
                    marker=dict(color=color_map[target_col], size=8, opacity=0.85),
                    customdata=summary_df.loc[valid, ["cluster_id", "region"]].to_numpy(),
                    hovertemplate=(
                        "Cluster %{customdata[0]}<br>"
                        "Region %{customdata[1]}<br>"
                        f"{_display_feature_label(feat, stat='mean')}: %{{x:.3f}}<br>"
                        f"{target_label}: %{{y:.3f}}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx + 1,
                col=col_idx,
            )
            if np.sum(valid) >= 2:
                x_fit = x_vals[valid]
                y_fit = y_vals[valid]
                x_min = float(np.nanmin(x_fit))
                x_max = float(np.nanmax(x_fit))
                if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                    slope, intercept = np.polyfit(x_fit, y_fit, 1)
                    x_line = np.array([x_min, x_max], dtype=float)
                    y_line = slope * x_line + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row_idx + 1,
                        col=col_idx,
                    )
            fig.update_xaxes(
                title_text=(
                    f"{_display_feature_label(feat, stat='mean')}<br>"
                    f"Spearman={row['spearman_r']:.2f}, n={int(row['n_valid'])}"
                ),
                row=row_idx + 1,
                col=col_idx,
            )
            fig.update_yaxes(title_text=target_label, row=row_idx + 1, col=col_idx)

    fig.update_layout(
        title=f"{title_prefix} | Top packet-feature scatter plots",
        template=template,
        font=dict(color=base_color),
        height=340 * n_rows + 120,
        margin=dict(l=80, r=40, t=90, b=60),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


packet_feature_correlations = None
if packet_feature_summary is None or packet_feature_summary.empty:
    print("Packet feature correlations: no summary table available.")
else:
    packet_feature_correlations = _feature_target_correlation_table(packet_feature_summary)
    feature_title_prefix = (
        f"Region {packet_feature_results['region']} | "
        f"context={packet_feature_results['context_mode']}"
    )
    print("Top features for coupling strength (by absolute Spearman):")
    strength_rank = packet_feature_correlations.loc[
        packet_feature_correlations["target"] == "coupling_strength",
        ["feature_label", "group", "pearson_r", "spearman_r", "n_valid"],
    ].sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    print(strength_rank.head(10).to_string(index=False))
    print("Top features for coupling delay (by absolute Spearman):")
    delay_rank = packet_feature_correlations.loc[
        packet_feature_correlations["target"] == "coupling_delay_ms",
        ["feature_label", "group", "pearson_r", "spearman_r", "n_valid"],
    ].sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    print(delay_rank.head(10).to_string(index=False))

    _plot_feature_ranking_bars(packet_feature_correlations, feature_title_prefix)
    _plot_feature_target_scatter_grid(
        packet_feature_summary,
        packet_feature_correlations,
        feature_title_prefix,
        top_k=PACKET_FEATURE_TOP_SCATTERS,
    )


# %% Custom Scatter Config
# ---- Interactive-style scatter selection via config ----
# Options:
# PACKET_CUSTOM_SCATTER_LEVEL: "neuron", "packet"
# PACKET_CUSTOM_SCATTER_X / PACKET_CUSTOM_SCATTER_Y:
#   "coupling_strength", "coupling_delay_ms",
#   "spike_count", "binary_participation", "packet_fraction",
#   "baseline_normalized_count", "excess_count", "rate_auc", "peak_rate",
#   "template_dot", "template_cosine",
#   "t_first_spike", "t_median_spike", "t_com", "t_peak_rate",
#   "t_onset_threshold", "relative_rank_order", "temporal_width"
# Aliases:
#   "spike_com" -> "t_com"
#   "median_spike_time" -> "t_median_spike"
#   "peak_rate_time" -> "t_peak_rate"
#   "time_to_first_spike" -> "t_first_spike"
# PACKET_CUSTOM_SCATTER_FEATURE_STAT: "mean", "var"
#   only affects packet/neuron features; coupling strength/delay are unchanged
# PACKET_CUSTOM_SCATTER_USE_NORMALIZED_PACKET_FEATURES: False, True
#   only affects packet-level feature variables
# PACKET_CUSTOM_SCATTER_ADD_FIT: False, True
PACKET_CUSTOM_SCATTER_LEVEL = "neuron"  # "neuron" or "packet"
PACKET_CUSTOM_SCATTER_X = "coupling_delay_ms"
PACKET_CUSTOM_SCATTER_Y = "t_median_spike"
PACKET_CUSTOM_SCATTER_FEATURE_STAT = "mean"
PACKET_CUSTOM_SCATTER_USE_NORMALIZED_PACKET_FEATURES = False
PACKET_CUSTOM_SCATTER_ADD_FIT = True


# %% Custom Scatter Helpers
def _resolve_scatter_variable_name(variable_name):
    var = str(variable_name).strip().lower().replace("-", "_")
    var = var.replace("__", "_")
    special = {
        "coupling_delay": "coupling_delay_ms",
        "delay": "coupling_delay_ms",
        "coupling strength": "coupling_strength",
        "coupling delay": "coupling_delay_ms",
        "strength": "coupling_strength",
    }
    return special.get(var, _resolve_feature_name(var))


def _available_scatter_variables(level):
    base_vars = ["coupling_strength", "coupling_delay_ms"]
    if str(level).strip().lower() == "packet":
        return base_vars + PACKET_FEATURE_ORDER
    return base_vars + PACKET_FEATURE_ORDER


def _resolve_scatter_column(level, variable_name, feature_stat="mean", normalized_packet_features=False):
    level = str(level).strip().lower()
    var = _resolve_scatter_variable_name(variable_name)
    stat_norm = _normalize_feature_stat(feature_stat)
    if var == "coupling_strength":
        if level == "packet":
            return "coupling_strength__packet", "Packet-weighted coupling strength"
        return "coupling_strength", "Coupling strength"
    if var == "coupling_delay_ms":
        if level == "packet":
            return "coupling_delay_ms__packet", "Packet-weighted coupling delay (ms)"
        return "coupling_delay_ms", "Coupling delay (ms)"
    if var not in PACKET_FEATURE_ORDER:
        raise ValueError(f"Unknown scatter variable: {variable_name}")
    if level == "packet":
        return _feature_summary_col(
            var,
            normalized=normalized_packet_features,
            stat=stat_norm,
        ), _display_feature_label(
            var,
            stat=stat_norm,
            normalized=normalized_packet_features,
        )
    return _feature_neuron_summary_col(var, stat=stat_norm), _display_feature_label(var, stat=stat_norm)


def _plot_custom_scatter(
    level,
    x_variable,
    y_variable,
    feature_stat="mean",
    normalized_packet_features=False,
    add_fit=True,
):
    level = str(level).strip().lower()
    if level == "neuron":
        if packet_feature_summary is None or packet_feature_summary.empty:
            print("Custom scatter: neuron-level summary table is not available.")
            return
        df_plot = packet_feature_summary.copy()
        hover_cols = ["cluster_id", "region"]
        hover_template = (
            "Cluster %{customdata[0]}<br>"
            "Region %{customdata[1]}<br>"
        )
    elif level == "packet":
        if packet_summary_df is None or packet_summary_df.empty:
            print("Custom scatter: packet-level summary table is not available.")
            return
        df_plot = packet_summary_df.copy()
        hover_cols = ["packet_idx", "packet_peak_time_s", "packet_context"]
        hover_template = (
            "Packet %{customdata[0]}<br>"
            "Peak time: %{customdata[1]:.3f}s<br>"
            "Context: %{customdata[2]}<br>"
        )
    else:
        raise ValueError(f"Unsupported PACKET_CUSTOM_SCATTER_LEVEL: {level}")

    x_col, x_label = _resolve_scatter_column(
        level,
        x_variable,
        feature_stat=feature_stat,
        normalized_packet_features=normalized_packet_features,
    )
    y_col, y_label = _resolve_scatter_column(
        level,
        y_variable,
        feature_stat=feature_stat,
        normalized_packet_features=normalized_packet_features,
    )
    x_vals = pd.to_numeric(df_plot[x_col], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df_plot[y_col], errors="coerce").to_numpy(dtype=float)
    x_var = _resolve_scatter_variable_name(x_variable)
    y_var = _resolve_scatter_variable_name(y_variable)
    if x_var in PACKET_FEATURE_ORDER:
        x_vals = _display_feature_values(x_var, x_vals, stat=feature_stat)
    if y_var in PACKET_FEATURE_ORDER:
        y_vals = _display_feature_values(y_var, y_vals, stat=feature_stat)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        print(f"Custom scatter: no valid rows for x={x_variable}, y={y_variable}, level={level}.")
        return

    pearson_r, _ = _pearsonr_with_n(x_vals, y_vals)
    spearman_r, _ = _spearmanr_with_n(x_vals, y_vals)
    customdata = df_plot.loc[valid, hover_cols].to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals[valid],
            y=y_vals[valid],
            mode="markers",
            marker=dict(size=8, color=base_color, opacity=0.82),
            customdata=customdata,
            hovertemplate=(
                hover_template
                + f"{x_label}: %{{x:.3f}}<br>"
                + f"{y_label}: %{{y:.3f}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    if add_fit and n_valid >= 2:
        x_fit = x_vals[valid]
        y_fit = y_vals[valid]
        x_min = float(np.nanmin(x_fit))
        x_max = float(np.nanmax(x_fit))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            x_line = np.array([x_min, x_max], dtype=float)
            y_line = slope * x_line + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=(
            f"Custom scatter | level={level} | feature_stat={_normalize_feature_stat(feature_stat)} | "
            f"Pearson={pearson_r:.2f}, Spearman={spearman_r:.2f}, n={n_valid}"
        ),
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    show_fig(fig, renderer=PLOTLY_RENDERER)


# %% Custom Scatter
print("Custom scatter available variables (neuron level):")
print(", ".join(_available_scatter_variables("neuron")))
print("Custom scatter available variables (packet level):")
print(", ".join(_available_scatter_variables("packet")))
_plot_custom_scatter(
    PACKET_CUSTOM_SCATTER_LEVEL,
    PACKET_CUSTOM_SCATTER_X,
    PACKET_CUSTOM_SCATTER_Y,
    feature_stat=PACKET_CUSTOM_SCATTER_FEATURE_STAT,
    normalized_packet_features=PACKET_CUSTOM_SCATTER_USE_NORMALIZED_PACKET_FEATURES,
    add_fit=PACKET_CUSTOM_SCATTER_ADD_FIT,
)


# %% Packet Clustering Helpers
def _build_packet_feature_matrix(feature_df, packet_dataset, feature_names, normalized=False):
    packet_ids = np.arange(len(packet_dataset["packet_times"]), dtype=int)
    cluster_ids = np.asarray(packet_dataset["cluster_ids"], dtype=int)
    matrices = []
    column_labels = []
    for feature_name in feature_names:
        feat = _resolve_feature_name(feature_name)
        col_name = f"{feat}__norm" if normalized else feat
        pivot = feature_df.pivot(index="packet_idx", columns="cluster_id", values=col_name)
        pivot = pivot.reindex(index=packet_ids, columns=cluster_ids)
        matrices.append(pivot.to_numpy(dtype=float))
        column_labels.extend(
            [f"{_feature_label(feat, normalized=normalized)} | unit {int(cid)}" for cid in cluster_ids]
        )
    if not matrices:
        return np.zeros((len(packet_ids), 0), dtype=float), []
    return np.concatenate(matrices, axis=1), column_labels


def _standardize_packet_feature_matrix(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("Feature matrix must be 2D.")
    if x.shape[1] == 0:
        return x, np.zeros(0, dtype=bool)

    valid_counts = np.sum(np.isfinite(x), axis=0).astype(float)
    safe_counts = np.maximum(valid_counts, 1.0)
    x_filled = np.where(np.isfinite(x), x, 0.0)
    col_mean = np.sum(x_filled, axis=0) / safe_counts
    centered = x - col_mean
    centered = np.where(np.isfinite(centered), centered, 0.0)
    col_std = np.sqrt(np.sum(centered ** 2, axis=0) / safe_counts)
    keep_mask = np.isfinite(col_std) & (col_std > 0)
    if not np.any(keep_mask):
        return np.zeros((x.shape[0], 0), dtype=float), keep_mask
    standardized = centered[:, keep_mask] / col_std[keep_mask]
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    return standardized, keep_mask


def _compute_matrix_pca(matrix, n_components=6):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("matrix must have shape (n_samples, n_features) with non-zero size.")
    feature_mean = np.nanmean(x, axis=0)
    x_centered = np.nan_to_num(x - feature_mean, nan=0.0)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    max_components = min(int(n_components), x_centered.shape[0], x_centered.shape[1])
    if max_components < 1:
        raise RuntimeError("Not enough samples/features for PCA.")
    scores = u[:, :max_components] * s[:max_components]
    total_var = np.sum(s ** 2)
    explained_ratio = (
        (s[:max_components] ** 2) / total_var if total_var > 0 else np.zeros(max_components, dtype=float)
    )
    return dict(scores=scores, components=vt[:max_components], explained_ratio=explained_ratio)


def _compute_matrix_umap(
    matrix,
    n_components=2,
    pre_pca_components=10,
    n_neighbors=20,
    min_dist=0.2,
    random_state=0,
):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        return None
    try:
        import umap
    except Exception:
        return None

    umap_input = x
    if pre_pca_components is not None and int(pre_pca_components) > 0 and x.shape[1] > int(pre_pca_components):
        pca_result = _compute_matrix_pca(
            x,
            n_components=min(int(pre_pca_components), x.shape[0], x.shape[1]),
        )
        umap_input = np.asarray(pca_result["scores"], dtype=float)

    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    return dict(
        scores=np.asarray(reducer.fit_transform(umap_input), dtype=float),
        input_matrix=np.asarray(umap_input, dtype=float),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        pre_pca_components=int(pre_pca_components) if pre_pca_components is not None else None,
    )


def _residualize_matrix_against_scalar(matrix, scalar):
    x = np.asarray(matrix, dtype=float)
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != scalar.shape[0]:
        raise ValueError("matrix and scalar must agree on the number of rows.")
    design = np.column_stack([np.ones(scalar.shape[0], dtype=float), scalar])
    coef, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def _prepare_feature_cluster_matrix(feature_df, packet_dataset, feature_names, shape_mode):
    shape_mode = str(shape_mode).strip().lower()
    raw_matrix, raw_columns = _build_packet_feature_matrix(
        feature_df,
        packet_dataset,
        feature_names,
        normalized=False,
    )
    normalized_matrix, normalized_columns = _build_packet_feature_matrix(
        feature_df,
        packet_dataset,
        feature_names,
        normalized=True,
    )
    raw_std, raw_keep = _standardize_packet_feature_matrix(raw_matrix)
    normalized_std, normalized_keep = _standardize_packet_feature_matrix(normalized_matrix)
    packet_size = _packet_size_from_feature_df(feature_df, packet_dataset)
    size_covariate = _transform_packet_size(packet_size)
    residual_std = _residualize_matrix_against_scalar(raw_std, size_covariate)
    residual_std, residual_keep = _standardize_packet_feature_matrix(residual_std)

    if shape_mode == "raw":
        cluster_matrix = raw_std
        column_labels = [label for label, keep in zip(raw_columns, raw_keep) if keep]
    elif shape_mode == "normalized":
        cluster_matrix = normalized_std
        column_labels = [label for label, keep in zip(normalized_columns, normalized_keep) if keep]
    elif shape_mode == "residual":
        cluster_matrix = residual_std
        column_labels = [
            f"{label} | size residual"
            for label, keep in zip(raw_columns, raw_keep)
            if keep
        ]
    else:
        raise ValueError(f"Unsupported PRESENT_SHAPE_MODE: {shape_mode}")

    return dict(
        shape_mode=shape_mode,
        packet_size=packet_size,
        size_covariate=size_covariate,
        cluster_matrix=cluster_matrix,
        column_labels=column_labels,
    )


def _cluster_palette():
    return [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def _event_windows_from_times(event_times, start_offset_s, end_offset_s):
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    valid = np.isfinite(event_times)
    if not np.any(valid):
        return np.empty((0, 2), dtype=float)
    event_times = event_times[valid]
    windows = np.column_stack([event_times + float(start_offset_s), event_times + float(end_offset_s)])
    valid_windows = np.isfinite(windows).all(axis=1) & (windows[:, 1] > windows[:, 0])
    return np.asarray(windows[valid_windows], dtype=float)


def _interval_overlap_exceeds(interval_a, interval_b, min_overlap_s):
    start = max(float(interval_a[0]), float(interval_b[0]))
    end = min(float(interval_a[1]), float(interval_b[1]))
    return (end - start) > float(min_overlap_s)


def _packet_overlap_mask(packet_windows, intervals, min_overlap_s):
    packet_windows = _coerce_interval_array(packet_windows)
    intervals = _coerce_interval_array(intervals)
    mask = np.zeros(packet_windows.shape[0], dtype=bool)
    if packet_windows.size == 0 or intervals.size == 0:
        return mask
    for pkt_idx, pkt_window in enumerate(packet_windows):
        for interval in intervals:
            if _interval_overlap_exceeds(pkt_window, interval, min_overlap_s):
                mask[pkt_idx] = True
                break
    return mask


def _packet_period_annotations(packet_dataset, min_overlap_s=0.05):
    packet_windows = np.asarray(packet_dataset.get("packet_windows", np.empty((0, 2))), dtype=float)
    packet_context = np.asarray(packet_dataset.get("packet_context", []), dtype=object)
    n_packets = packet_windows.shape[0]

    def _trial_times_local(key):
        if hasattr(trials, "keys") and key in trials.keys():
            return np.asarray(trials[key], dtype=float)
        return np.array([], dtype=float)

    whisk_bouts = _coerce_interval_array(wh_detect.get("all_bouts", np.empty((0, 2), dtype=float)))
    stim_windows = _event_windows_from_times(_trial_times_local("stimOn_times"), 0.0, 0.2)
    first_move_windows = _event_windows_from_times(_trial_times_local("firstMovement_times"), -0.1, 0.2)
    feedback_windows = _event_windows_from_times(_trial_times_local("feedback_times"), 0.0, 0.4)

    whisk_mask = _packet_overlap_mask(packet_windows, whisk_bouts, min_overlap_s)
    stim_mask = _packet_overlap_mask(packet_windows, stim_windows, min_overlap_s)
    first_move_mask = _packet_overlap_mask(packet_windows, first_move_windows, min_overlap_s)
    feedback_mask = _packet_overlap_mask(packet_windows, feedback_windows, min_overlap_s)

    whisk_only_mask = whisk_mask & ~(stim_mask | first_move_mask | feedback_mask)
    non_whisking_mask = ~whisk_mask

    context_split = packet_context.astype(object).copy()
    spont_mask = packet_context == "spont"
    context_split[spont_mask & whisk_mask] = "spont_whisking"
    context_split[spont_mask & ~whisk_mask] = "spont_non_whisking"

    event_period_order = ["whisking", "stim_on", "first_move", "feedback"]
    event_period_masks = {
        "whisking": whisk_mask,
        "stim_on": stim_mask,
        "first_move": first_move_mask,
        "feedback": feedback_mask,
    }

    exclusive_labels = np.full(n_packets, "non_whisking", dtype=object)
    exclusive_labels[whisk_only_mask] = "whisking"
    exclusive_labels[first_move_mask] = "first_move"
    exclusive_labels[feedback_mask] = "feedback"
    exclusive_labels[stim_mask] = "stim_on"

    return {
        "whisking_mask": whisk_mask,
        "whisk_only_mask": whisk_only_mask,
        "non_whisking_mask": non_whisking_mask,
        "stim_on_mask": stim_mask,
        "first_move_mask": first_move_mask,
        "feedback_mask": feedback_mask,
        "context_split": context_split,
        "event_period_order": event_period_order,
        "event_period_masks": event_period_masks,
        "exclusive_labels": exclusive_labels,
    }


def _plot_feature_cluster_embedding(packet_summary_df, pca_result, cluster_labels, title_prefix):
    scores = np.asarray(pca_result["scores"], dtype=float)
    explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
    palette = _cluster_palette()
    fig = go.Figure()
    for cluster_idx in np.unique(cluster_labels):
        mask = np.asarray(cluster_labels) == cluster_idx
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=palette[int(cluster_idx) % len(palette)], size=9, opacity=0.9),
                name=f"Cluster {int(cluster_idx) + 1} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{title_prefix} | Feature-space packet embedding",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    evr1 = 100.0 * explained_ratio[0] if explained_ratio.size >= 1 else 0.0
    evr2 = 100.0 * explained_ratio[1] if explained_ratio.size >= 2 else 0.0
    fig.update_xaxes(title_text=f"PC1 ({evr1:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({evr2:.1f}% var)")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_feature_embedding_by_context(packet_summary_df, pca_result, title_prefix):
    scores = np.asarray(pca_result["scores"], dtype=float)
    explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
    context_style = {
        "task": ("Task", "#1f77b4"),
        "spont": ("Spont", "#ff7f0e"),
        "iti": ("ITI", "#2ca02c"),
        "other": ("Other", "#7f7f7f"),
        "overlap": ("Overlap", "#9467bd"),
    }
    fig = go.Figure()
    context_vals = np.asarray(packet_summary_df["packet_context"], dtype=object)
    for context_key in ("task", "spont", "iti", "other", "overlap"):
        mask = context_vals == context_key
        if not np.any(mask):
            continue
        label, color = context_style[context_key]
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.86),
                name=f"{label} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    evr1 = 100.0 * explained_ratio[0] if explained_ratio.size >= 1 else 0.0
    evr2 = 100.0 * explained_ratio[1] if explained_ratio.size >= 2 else 0.0
    fig.update_layout(
        title=f"{title_prefix} | Feature-space packet embedding by context",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=f"PC1 ({evr1:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({evr2:.1f}% var)")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_feature_embedding_by_period(packet_summary_df, pca_result, title_prefix):
    scores = np.asarray(pca_result["scores"], dtype=float)
    explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
    period_style = {
        "stim_on": ("Stim On", "#1f77b4"),
        "feedback": ("Feedback", "#d62728"),
        "first_move": ("First Move", "#2ca02c"),
        "whisking": ("Whisking", "#ff7f0e"),
        "non_whisking": ("Non-whisking", "#7f7f7f"),
    }
    labels = np.asarray(packet_summary_df["period_exclusive_label"], dtype=object)
    fig = go.Figure()
    for period_key in ("stim_on", "feedback", "first_move", "whisking", "non_whisking"):
        mask = labels == period_key
        if not np.any(mask):
            continue
        label, color = period_style[period_key]
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.86),
                name=f"{label} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[
                    mask,
                    ["packet_idx", "packet_peak_time_s", "packet_context", "period_exclusive_label"],
                ].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "Period: %{customdata[3]}<br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    evr1 = 100.0 * explained_ratio[0] if explained_ratio.size >= 1 else 0.0
    evr2 = 100.0 * explained_ratio[1] if explained_ratio.size >= 2 else 0.0
    fig.update_layout(
        title=f"{title_prefix} | Feature-space packet embedding by event period",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=f"PC1 ({evr1:.1f}% var)")
    fig.update_yaxes(title_text=f"PC2 ({evr2:.1f}% var)")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_umap_cluster_embedding(packet_summary_df, umap_result, cluster_labels, title_prefix):
    scores = np.asarray(umap_result["scores"], dtype=float)
    palette = _cluster_palette()
    fig = go.Figure()
    for cluster_idx in np.unique(cluster_labels):
        mask = np.asarray(cluster_labels) == cluster_idx
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=palette[int(cluster_idx) % len(palette)], size=9, opacity=0.9),
                name=f"Cluster {int(cluster_idx) + 1} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{title_prefix} | Feature-space UMAP",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_umap_by_context(packet_summary_df, umap_result, title_prefix):
    scores = np.asarray(umap_result["scores"], dtype=float)
    context_style = {
        "task": ("Task", "#1f77b4"),
        "spont": ("Spont", "#ff7f0e"),
        "iti": ("ITI", "#2ca02c"),
        "other": ("Other", "#7f7f7f"),
        "overlap": ("Overlap", "#9467bd"),
    }
    fig = go.Figure()
    context_vals = np.asarray(packet_summary_df["packet_context"], dtype=object)
    for context_key in ("task", "spont", "iti", "other", "overlap"):
        mask = context_vals == context_key
        if not np.any(mask):
            continue
        label, color = context_style[context_key]
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.86),
                name=f"{label} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{title_prefix} | Feature-space UMAP by context",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_umap_by_period(packet_summary_df, umap_result, title_prefix):
    scores = np.asarray(umap_result["scores"], dtype=float)
    period_style = {
        "stim_on": ("Stim On", "#1f77b4"),
        "feedback": ("Feedback", "#d62728"),
        "first_move": ("First Move", "#2ca02c"),
        "whisking": ("Whisking", "#ff7f0e"),
        "non_whisking": ("Non-whisking", "#7f7f7f"),
    }
    labels = np.asarray(packet_summary_df["period_exclusive_label"], dtype=object)
    fig = go.Figure()
    for period_key in ("stim_on", "feedback", "first_move", "whisking", "non_whisking"):
        mask = labels == period_key
        if not np.any(mask):
            continue
        label, color = period_style[period_key]
        fig.add_trace(
            go.Scatter(
                x=scores[mask, 0] if scores.shape[1] >= 1 else np.zeros(int(np.sum(mask))),
                y=scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(int(np.sum(mask))),
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.86),
                name=f"{label} (n={int(np.sum(mask))})",
                customdata=packet_summary_df.loc[
                    mask,
                    ["packet_idx", "packet_peak_time_s", "packet_context", "period_exclusive_label"],
                ].to_numpy(),
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Peak time: %{customdata[1]:.3f}s<br>"
                    "Context: %{customdata[2]}<br>"
                    "Period: %{customdata[3]}<br>"
                    "UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{title_prefix} | Feature-space UMAP by event period",
        template=template,
        font=dict(color=base_color),
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2")
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_packet_feature_pair_scatters(packet_summary_df, cluster_labels, title_prefix):
    if not PRESENT_FEATURE_PAIR_SPECS:
        return
    palette = _cluster_palette()
    n_rows = len(PRESENT_FEATURE_PAIR_SPECS)
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=tuple(
            title
            for x_feat, y_feat in PRESENT_FEATURE_PAIR_SPECS
            for title in (
                f"Raw: {_display_feature_label(x_feat, stat='mean')} vs {_display_feature_label(y_feat, stat='mean')}",
                f"Normalized: {_display_feature_label(x_feat, stat='mean', normalized=True)} vs {_display_feature_label(y_feat, stat='mean', normalized=True)}",
            )
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.10,
    )

    for row_idx, (x_feat_name, y_feat_name) in enumerate(PRESENT_FEATURE_PAIR_SPECS, start=1):
        x_feat = _resolve_feature_name(x_feat_name)
        y_feat = _resolve_feature_name(y_feat_name)
        raw_cols = (_feature_summary_col(x_feat, False), _feature_summary_col(y_feat, False))
        norm_cols = (_feature_summary_col(x_feat, True), _feature_summary_col(y_feat, True))
        for col_idx, (x_col, y_col, normalized) in enumerate(
            ((raw_cols[0], raw_cols[1], False), (norm_cols[0], norm_cols[1], True)),
            start=1,
        ):
            for cluster_idx in np.unique(cluster_labels):
                mask = np.asarray(cluster_labels) == cluster_idx
                x_vals = pd.to_numeric(packet_summary_df.loc[mask, x_col], errors="coerce").to_numpy(dtype=float)
                y_vals = pd.to_numeric(packet_summary_df.loc[mask, y_col], errors="coerce").to_numpy(dtype=float)
                x_vals = _display_feature_values(x_feat, x_vals, stat="mean")
                y_vals = _display_feature_values(y_feat, y_vals, stat="mean")
                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                if not np.any(valid):
                    continue
                customdata = packet_summary_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy()[valid]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals[valid],
                        y=y_vals[valid],
                        mode="markers",
                        marker=dict(color=palette[int(cluster_idx) % len(palette)], size=8, opacity=0.82),
                        name=f"Cluster {int(cluster_idx) + 1}" if row_idx == 1 and col_idx == 1 else None,
                        legendgroup=f"cluster-{int(cluster_idx)}",
                        showlegend=(row_idx == 1 and col_idx == 1),
                        customdata=customdata,
                        hovertemplate=(
                            "Packet %{customdata[0]}<br>"
                            "Peak time: %{customdata[1]:.3f}s<br>"
                            "Context: %{customdata[2]}<br>"
                            f"{_display_feature_label(x_feat, stat='mean', normalized=normalized)}: %{{x:.3f}}<br>"
                            f"{_display_feature_label(y_feat, stat='mean', normalized=normalized)}: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            fig.update_xaxes(
                title_text=_display_feature_label(x_feat, stat="mean", normalized=normalized),
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=_display_feature_label(y_feat, stat="mean", normalized=normalized),
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title=f"{title_prefix} | Packet feature scatter pairs",
        template=template,
        font=dict(color=base_color),
        height=420 * n_rows + 120,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_cluster_usage(packet_summary_df, title_prefix):
    palette = _cluster_palette()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cluster counts by event period", "Cluster counts by context"),
        horizontal_spacing=0.16,
    )
    period_order = ["whisking", "stim_on", "first_move", "feedback"]
    period_labels = {
        "whisking": "Whisking",
        "stim_on": "Stim On",
        "first_move": "First Move",
        "feedback": "Feedback",
    }
    period_count_table = pd.DataFrame(index=period_order)
    for cluster_idx in sorted(packet_summary_df["cluster_label"].dropna().astype(int).unique()):
        period_count_table[cluster_idx] = [
            int(np.sum(packet_summary_df[f"period_{period_key}"].to_numpy(dtype=bool) & (packet_summary_df["cluster_label"] == cluster_idx).to_numpy(dtype=bool)))
            for period_key in period_order
        ]
        fig.add_trace(
            go.Bar(
                x=[period_labels[key] for key in period_order],
                y=period_count_table[cluster_idx].to_numpy(dtype=float),
                marker=dict(color=palette[int(cluster_idx) % len(palette)]),
                name=f"Cluster {int(cluster_idx) + 1}",
                legendgroup=f"cluster-{int(cluster_idx)}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    context_order = [
        key
        for key in ["task", "iti", "other", "overlap", "spont_whisking", "spont_non_whisking"]
        if key in set(packet_summary_df["context_split"].astype(str))
    ]
    context_labels = {
        "task": "Task",
        "iti": "ITI",
        "other": "Other",
        "overlap": "Overlap",
        "spont_whisking": "Spont Whisking",
        "spont_non_whisking": "Spont Non-whisking",
    }
    count_table = (
        packet_summary_df.groupby(["context_split", "cluster_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(context_order, fill_value=0)
    )
    for cluster_idx in count_table.columns:
        fig.add_trace(
            go.Bar(
                x=[context_labels.get(key, key) for key in count_table.index.astype(str)],
                y=count_table[cluster_idx].to_numpy(dtype=float),
                marker=dict(color=palette[int(cluster_idx) % len(palette)]),
                name=f"Cluster {int(cluster_idx) + 1}",
                legendgroup=f"cluster-{int(cluster_idx)}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Event period", row=1, col=1)
    fig.update_yaxes(title_text="Packet count", row=1, col=1)
    fig.update_xaxes(title_text="Packet context", row=1, col=2)
    fig.update_yaxes(title_text="Packet count", row=1, col=2)
    fig.update_layout(
        title=f"{title_prefix} | Packet cluster usage",
        template=template,
        font=dict(color=base_color),
        barmode="stack",
        height=520,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


def _plot_cluster_heatmaps(packet_dataset, packet_tensor_space, cluster_labels, title_text):
    packet_tensor_space = np.asarray(packet_tensor_space, dtype=float)
    template_matrix = np.asarray(packet_dataset["template_matrix"], dtype=float)
    rel_bin_centers_ms = np.asarray(packet_dataset["rel_bin_centers"], dtype=float) * 1000.0
    unique_clusters = np.unique(cluster_labels)

    matrices = [_row_zscore_matrix(template_matrix)]
    subplot_titles = ["Detection template"]
    for cluster_idx in unique_clusters:
        members = packet_tensor_space[np.asarray(cluster_labels) == cluster_idx]
        cluster_mean = np.nanmean(members, axis=0) if members.size > 0 else np.zeros_like(template_matrix)
        matrices.append(_row_zscore_matrix(cluster_mean))
        subplot_titles.append(f"Cluster {int(cluster_idx) + 1} mean packet (n={members.shape[0]})")

    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices if mat.size > 0])
    cmax = np.nanpercentile(np.abs(finite_vals), 98) if finite_vals.size > 0 else 1.0
    cmax = max(float(cmax), 1.0)

    fig = make_subplots(
        rows=len(matrices),
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.06,
    )
    for row_idx, matrix in enumerate(matrices, start=1):
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=rel_bin_centers_ms,
                y=np.arange(matrix.shape[0]),
                colorscale="RdBu_r",
                zmin=-cmax,
                zmax=cmax,
                colorbar=dict(title="z", len=0.9) if row_idx == len(matrices) else None,
                showscale=(row_idx == len(matrices)),
                hovertemplate="Unit %{y}<br>Time %{x:.1f} ms<br>z=%{z:.2f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="Units", row=row_idx, col=1)

    fig.update_xaxes(title_text="Time from packet peak (ms)", row=len(matrices), col=1)
    fig.update_layout(
        title=title_text,
        template=template,
        font=dict(color=base_color),
        height=280 * len(matrices) + 120,
        margin=dict(l=70, r=40, t=90, b=60),
    )
    show_fig(fig, renderer=PLOTLY_RENDERER)


# %% Packet Feature Clustering
packet_cluster_results = None
if packet_feature_results is None or packet_feature_df is None or packet_summary_df is None:
    print("Packet feature clustering: no packet feature dataset is available.")
else:
    selected_features = []
    for feature_name in PRESENT_CLUSTER_FEATURES:
        feat = _resolve_feature_name(feature_name)
        if feat not in PACKET_FEATURE_ORDER:
            raise ValueError(f"Unknown packet feature: {feature_name}")
        if feat not in selected_features:
            selected_features.append(feat)

    cluster_input = _prepare_feature_cluster_matrix(
        packet_feature_df,
        packet_feature_results["packet_dataset"],
        selected_features,
        PRESENT_SHAPE_MODE,
    )
    cluster_matrix = np.asarray(cluster_input["cluster_matrix"], dtype=float)
    if cluster_matrix.shape[0] < 2 or cluster_matrix.shape[1] == 0:
        print("Packet feature clustering: not enough packets/features after preprocessing.")
    else:
        cluster_labels, cluster_centroids = _kmeans_numpy(
            cluster_matrix,
            n_clusters=PRESENT_N_CLUSTERS,
            n_init=PACKET_KMEANS_N_INIT,
            max_iter=PACKET_KMEANS_MAX_ITER,
            random_state=0,
        )
        cluster_pca = _compute_matrix_pca(cluster_matrix, n_components=PRESENT_PCA_COMPONENTS)
        packet_summary_plot = packet_summary_df.copy()
        packet_summary_plot["cluster_label"] = cluster_labels
        packet_period_info = _packet_period_annotations(
            packet_feature_results["packet_dataset"],
            min_overlap_s=PRESENT_PACKET_PERIOD_MIN_OVERLAP_S,
        )
        packet_summary_plot["context_split"] = packet_period_info["context_split"]
        packet_summary_plot["period_exclusive_label"] = packet_period_info["exclusive_labels"]
        packet_summary_plot["period_whisking"] = packet_period_info["whisking_mask"]
        packet_summary_plot["period_stim_on"] = packet_period_info["stim_on_mask"]
        packet_summary_plot["period_first_move"] = packet_period_info["first_move_mask"]
        packet_summary_plot["period_feedback"] = packet_period_info["feedback_mask"]

        packet_tensor_raw = np.asarray(packet_feature_results["packet_dataset"]["packet_tensor"], dtype=float)
        packet_size = np.asarray(cluster_input["packet_size"], dtype=float)
        denom = np.where(packet_size > 0, packet_size, 1.0)
        packet_tensor_normalized = packet_tensor_raw / denom[:, None, None]

        packet_cluster_results = {
            "cluster_input": cluster_input,
            "cluster_labels": cluster_labels,
            "cluster_centroids": cluster_centroids,
            "cluster_pca": cluster_pca,
            "packet_summary_df": packet_summary_plot,
            "packet_tensor_normalized": packet_tensor_normalized,
            "packet_period_info": packet_period_info,
        }
        print(
            f"Packet feature clustering | shape_mode={PRESENT_SHAPE_MODE} | "
            f"clusters={PRESENT_N_CLUSTERS} | matrix={cluster_matrix.shape[0]}x{cluster_matrix.shape[1]}"
        )


# %% Packet Presentation Figures
if packet_cluster_results is None:
    print("Packet presentation figures: feature clustering results are not available.")
else:
    plot_title_prefix = (
        f"Region {packet_feature_results['region']} | "
        f"context={packet_feature_results['context_mode']} | "
        f"shape={PRESENT_SHAPE_MODE}"
    )
    _plot_feature_cluster_embedding(
        packet_cluster_results["packet_summary_df"],
        packet_cluster_results["cluster_pca"],
        packet_cluster_results["cluster_labels"],
        plot_title_prefix,
    )
    _plot_feature_embedding_by_context(
        packet_cluster_results["packet_summary_df"],
        packet_cluster_results["cluster_pca"],
        plot_title_prefix,
    )
    _plot_feature_embedding_by_period(
        packet_cluster_results["packet_summary_df"],
        packet_cluster_results["cluster_pca"],
        plot_title_prefix,
    )
    _plot_cluster_usage(
        packet_cluster_results["packet_summary_df"],
        plot_title_prefix,
    )
    _plot_cluster_heatmaps(
        packet_feature_results["packet_dataset"],
        packet_feature_results["packet_dataset"]["packet_tensor"],
        packet_cluster_results["cluster_labels"],
        f"{plot_title_prefix} | Raw packet heatmaps",
    )
    _plot_cluster_heatmaps(
        packet_feature_results["packet_dataset"],
        packet_cluster_results["packet_tensor_normalized"],
        packet_cluster_results["cluster_labels"],
        f"{plot_title_prefix} | Size-normalized packet heatmaps",
    )


# %% Packet UMAP Figures
packet_umap_result = None
if packet_cluster_results is None:
    print("Packet UMAP figures: feature clustering results are not available.")
else:
    plot_title_prefix = (
        f"Region {packet_feature_results['region']} | "
        f"context={packet_feature_results['context_mode']} | "
        f"shape={PRESENT_SHAPE_MODE}"
    )
    cluster_matrix = np.asarray(packet_cluster_results["cluster_input"]["cluster_matrix"], dtype=float)
    packet_umap = _compute_matrix_umap(
        cluster_matrix,
        n_components=2,
        pre_pca_components=PRESENT_UMAP_PRE_PCA_COMPONENTS,
        n_neighbors=PRESENT_UMAP_NEIGHBORS,
        min_dist=PRESENT_UMAP_MIN_DIST,
        random_state=PRESENT_UMAP_RANDOM_STATE,
    )
    if packet_umap is None:
        print("Packet UMAP figures: `umap-learn` is not available or the packet matrix is too small.")
    else:
        packet_umap_result = packet_umap
        _plot_umap_cluster_embedding(
            packet_cluster_results["packet_summary_df"],
            packet_umap,
            packet_cluster_results["cluster_labels"],
            plot_title_prefix,
        )
        _plot_umap_by_context(
            packet_cluster_results["packet_summary_df"],
            packet_umap,
            plot_title_prefix,
        )
        _plot_umap_by_period(
            packet_cluster_results["packet_summary_df"],
            packet_umap,
            plot_title_prefix,
        )


# %% Single-Neuron Packet Feature Scatter Matrix
PACKET_SINGLE_NEURON_FEATURES = list(PRESENT_CLUSTER_FEATURES)
PACKET_SINGLE_NEURON_CLUSTER_ID = None  # None -> neuron with highest combined variance across selected features

if packet_cluster_results is None or packet_feature_df is None:
    print("Single-neuron packet feature matrix: packet feature clustering results are not available.")
else:
    selected_features = []
    for feature_name in PACKET_SINGLE_NEURON_FEATURES:
        feat = _resolve_feature_name(feature_name)
        if feat in PACKET_FEATURE_ORDER and feat not in selected_features:
            selected_features.append(feat)
    if not selected_features:
        print("Single-neuron packet feature matrix: no valid packet features were selected.")
    else:
        packet_cluster_lookup = packet_cluster_results["packet_summary_df"][
            ["packet_idx", "cluster_label"]
        ].drop_duplicates(subset=["packet_idx"])
        feature_df_local = packet_feature_df.merge(packet_cluster_lookup, on="packet_idx", how="left")

        chosen_cluster_id = PACKET_SINGLE_NEURON_CLUSTER_ID
        if chosen_cluster_id is None:
            variance_rows = []
            for cluster_id, group_df in feature_df_local.groupby("cluster_id", sort=False):
                row = {"cluster_id": int(cluster_id)}
                for feat in selected_features:
                    values = pd.to_numeric(group_df[feat], errors="coerce").to_numpy(dtype=float)
                    values = _display_feature_values(feat, values, stat="mean")
                    valid = np.isfinite(values)
                    row[feat] = float(np.nanvar(values[valid])) if np.any(valid) else np.nan
                variance_rows.append(row)

            variance_df = pd.DataFrame(variance_rows)
            score_terms = []
            for feat in selected_features:
                values = pd.to_numeric(variance_df[feat], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values)
                feat_score = np.full(values.shape, np.nan, dtype=float)
                if np.any(valid):
                    spread = float(np.nanstd(values[valid]))
                    if spread > 0:
                        feat_score[valid] = (values[valid] - float(np.nanmean(values[valid]))) / spread
                    else:
                        feat_score[valid] = 0.0
                score_terms.append(feat_score)
            combined_score = np.nanmean(np.column_stack(score_terms), axis=1)
            if not np.any(np.isfinite(combined_score)):
                print("Single-neuron packet feature matrix: unable to determine a default neuron.")
                chosen_cluster_id = None
            else:
                best_idx = int(np.nanargmax(combined_score))
                chosen_cluster_id = int(variance_df.iloc[best_idx]["cluster_id"])

        if chosen_cluster_id is None:
            pass
        else:
            neuron_df = (
                feature_df_local[feature_df_local["cluster_id"] == int(chosen_cluster_id)]
                .copy()
                .sort_values("packet_idx")
            )
            if neuron_df.empty:
                print(
                    f"Single-neuron packet feature matrix: cluster_id {int(chosen_cluster_id)} "
                    "is not available in the current packet-feature table."
                )
            else:
                palette = _cluster_palette()
                n_features = len(selected_features)
                display_col_map = {}
                display_labels = {}
                for feat in selected_features:
                    display_col = f"{feat}__display"
                    values = pd.to_numeric(neuron_df[feat], errors="coerce").to_numpy(dtype=float)
                    neuron_df[display_col] = _display_feature_values(feat, values, stat="mean")
                    display_col_map[feat] = display_col
                    display_labels[feat] = _display_feature_label(feat, stat="mean")

                unique_packet_clusters = sorted(
                    neuron_df["cluster_label"].dropna().astype(int).unique().tolist()
                )
                fig_matrix = make_subplots(
                    rows=n_features,
                    cols=n_features,
                    shared_xaxes=False,
                    shared_yaxes=False,
                    horizontal_spacing=0.02,
                    vertical_spacing=0.02,
                )

                for row_idx, y_feat in enumerate(selected_features, start=1):
                    for col_idx, x_feat in enumerate(selected_features, start=1):
                        if row_idx == col_idx:
                            for cluster_idx in unique_packet_clusters:
                                mask = neuron_df["cluster_label"].to_numpy(dtype=float) == float(cluster_idx)
                                x_vals = pd.to_numeric(
                                    neuron_df.loc[mask, display_col_map[x_feat]],
                                    errors="coerce",
                                ).to_numpy(dtype=float)
                                valid = np.isfinite(x_vals)
                                if not np.any(valid):
                                    continue
                                fig_matrix.add_trace(
                                    go.Histogram(
                                        x=x_vals[valid],
                                        marker=dict(color=palette[int(cluster_idx) % len(palette)]),
                                        opacity=0.55,
                                        nbinsx=18,
                                        name=f"Packet Cluster {int(cluster_idx) + 1}",
                                        legendgroup=f"packet-cluster-{int(cluster_idx)}",
                                        showlegend=(row_idx == 1 and col_idx == 1),
                                        hovertemplate=(
                                            f"{display_labels[x_feat]}<br>"
                                            "Value: %{x:.3f}<br>"
                                            "Count: %{y}<extra></extra>"
                                        ),
                                    ),
                                    row=row_idx,
                                    col=col_idx,
                                )
                        else:
                            for cluster_idx in unique_packet_clusters:
                                mask = neuron_df["cluster_label"].to_numpy(dtype=float) == float(cluster_idx)
                                x_vals = pd.to_numeric(
                                    neuron_df.loc[mask, display_col_map[x_feat]],
                                    errors="coerce",
                                ).to_numpy(dtype=float)
                                y_vals = pd.to_numeric(
                                    neuron_df.loc[mask, display_col_map[y_feat]],
                                    errors="coerce",
                                ).to_numpy(dtype=float)
                                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                                if not np.any(valid):
                                    continue
                                customdata = neuron_df.loc[
                                    mask,
                                    ["packet_idx", "packet_peak_time_s", "packet_context"],
                                ].to_numpy()[valid]
                                fig_matrix.add_trace(
                                    go.Scatter(
                                        x=x_vals[valid],
                                        y=y_vals[valid],
                                        mode="markers",
                                        marker=dict(
                                            color=palette[int(cluster_idx) % len(palette)],
                                            size=7,
                                            opacity=0.78,
                                        ),
                                        name=f"Packet Cluster {int(cluster_idx) + 1}",
                                        legendgroup=f"packet-cluster-{int(cluster_idx)}",
                                        showlegend=False,
                                        customdata=customdata,
                                        hovertemplate=(
                                            "Packet %{customdata[0]}<br>"
                                            "Peak time: %{customdata[1]:.3f}s<br>"
                                            "Context: %{customdata[2]}<br>"
                                            f"{display_labels[x_feat]}: %{{x:.3f}}<br>"
                                            f"{display_labels[y_feat]}: %{{y:.3f}}<extra></extra>"
                                        ),
                                    ),
                                    row=row_idx,
                                    col=col_idx,
                                )

                for feat_idx, feat in enumerate(selected_features, start=1):
                    fig_matrix.update_xaxes(title_text=display_labels[feat], row=n_features, col=feat_idx)
                    fig_matrix.update_yaxes(title_text=display_labels[feat], row=feat_idx, col=1)

                fig_matrix.update_layout(
                    title=(
                        f"Single-neuron packet feature matrix | cluster_id {int(chosen_cluster_id)} | "
                        f"region {packet_feature_results['region']} | packets={len(neuron_df)}"
                    ),
                    template=template,
                    font=dict(color=base_color),
                    barmode="overlay",
                    height=max(900, 180 * n_features + 120),
                    width=max(900, 180 * n_features + 120),
                    margin=dict(l=90, r=40, t=90, b=90),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                show_fig(fig_matrix, renderer=PLOTLY_RENDERER)
                print(
                    f"Single-neuron packet feature matrix | cluster_id={int(chosen_cluster_id)} | "
                    f"n_packets={len(neuron_df)}"
                )


# %% Packet Cluster PSTH
if packet_cluster_results is None or packet_feature_results is None:
    print("Packet cluster PSTH: clustering results are not available.")
else:
    packet_times_cluster = np.asarray(
        packet_feature_results["packet_dataset"]["packet_times"],
        dtype=float,
    )
    cluster_labels = np.asarray(packet_cluster_results["cluster_labels"], dtype=int)
    if packet_times_cluster.size == 0 or cluster_labels.size == 0:
        print("Packet cluster PSTH: no clustered packet events are available.")
    elif packet_times_cluster.size != cluster_labels.size:
        print("Packet cluster PSTH: packet times and cluster labels have inconsistent lengths.")
    else:
        packet_spikes_cluster = SimpleNamespace(
            times=packet_times_cluster,
            clusters=cluster_labels,
        )
        bin_size = config_plot.get("SINGLE_NEURON_BIN_SIZE", 0.05)
        smooth_sigma = config_plot.get("SINGLE_NEURON_SMOOTH_SIGMA", 1)
        default_pre = float(config_plot.get("SINGLE_NEURON_RASTER_PRE", 0.5))
        default_post = float(config_plot.get("SINGLE_NEURON_RASTER_POST", 1.0))
        event_specs = [
            ("stimOn_times", event_label("stimOn_times"), default_pre, default_post, f"Time from {event_label('stimOn_times')} (s)"),
            (
                "firstMovement_times",
                event_label("firstMovement_times"),
                default_pre,
                default_post,
                f"Time from {event_label('firstMovement_times')} (s)",
            ),
            (
                "feedback_times",
                event_label("feedback_times"),
                0.5,
                5.0,
                f"Time from {event_label('feedback_times')} (s)",
            ),
        ]
        valid_specs = []
        for event_name, display_label, pre_s, post_s, xaxis_title in event_specs:
            if hasattr(trials, "keys") and event_name in trials.keys():
                events = np.asarray(trials[event_name], dtype=float).reshape(-1)
                events = events[np.isfinite(events)]
                if events.size > 0:
                    valid_specs.append(
                        (event_name, display_label, events, float(pre_s), float(post_s), xaxis_title)
                    )

        whisk_context_mode = str(PACKET_WHISK_EVENT_CONTEXT).strip().lower()
        whisk_context_suffix_map = {
            "all": "",
            "task": "_task",
            "iti": "_iti",
            "spont": "_spont",
        }
        whisk_suffix = whisk_context_suffix_map.get(whisk_context_mode, "")
        whisk_brief = np.asarray(
            wh_events_by_period.get(f"wh_brief_times{whisk_suffix}", np.array([])),
            dtype=float,
        )
        whisk_long = np.asarray(
            wh_events_by_period.get(f"wh_long_times{whisk_suffix}", np.array([])),
            dtype=float,
        )
        whisk_events = np.concatenate([whisk_brief[np.isfinite(whisk_brief)], whisk_long[np.isfinite(whisk_long)]])
        if whisk_events.size > 0:
            whisk_events = np.unique(np.sort(whisk_events))
            valid_specs.append(
                (
                    "whisk_combined",
                    f"Whisk Onset ({whisk_context_mode})",
                    whisk_events,
                    default_pre,
                    default_post,
                    "Time from whisk onset (s)",
                )
            )

        if not valid_specs:
            print("Packet cluster PSTH: no valid task/whisk event series are available.")
        else:
            palette = _cluster_palette()
            unique_clusters = np.unique(cluster_labels)
            n_panels = len(valid_specs)
            n_rows = 2
            n_cols = 2
            fig_cluster_psth = make_subplots(
                rows=n_rows,
                cols=n_cols,
                shared_yaxes=True,
                horizontal_spacing=0.08,
                vertical_spacing=0.14,
                subplot_titles=tuple(
                    f"{display_label} (n={len(events)})"
                    for _event_name, display_label, events, _pre_s, _post_s, _xaxis_title in valid_specs
                ),
            )

            for panel_idx, (_event_name, display_label, events, pre_s, post_s, xaxis_title) in enumerate(
                valid_specs,
                start=0,
            ):
                row_idx = (panel_idx // n_cols) + 1
                col_idx = (panel_idx % n_cols) + 1
                psth_by_cluster, bin_centers = compute_psth_for_clusters(
                    packet_spikes_cluster,
                    [int(cluster_idx) for cluster_idx in unique_clusters],
                    events,
                    -float(pre_s),
                    float(post_s),
                    bin_size,
                    smooth_sigma,
                    show_progress=False,
                )
                for cluster_idx in unique_clusters:
                    psth_entry = psth_by_cluster.get(int(cluster_idx))
                    if psth_entry and bin_centers is not None:
                        firing_rate = psth_entry["fr_smooth"]
                    else:
                        firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)
                    fig_cluster_psth.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=firing_rate,
                            mode="lines",
                            line=dict(
                                color=palette[int(cluster_idx) % len(palette)],
                                width=2,
                            ),
                            name=f"Cluster {int(cluster_idx) + 1} (n={int(np.sum(cluster_labels == cluster_idx))})",
                            legendgroup=f"cluster-{int(cluster_idx)}",
                            showlegend=(panel_idx == 0),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                fig_cluster_psth.add_vline(
                    x=0,
                    line=dict(color="black", dash="dash"),
                    row=row_idx,
                    col=col_idx,
                )
                fig_cluster_psth.update_xaxes(
                    title_text=xaxis_title,
                    range=[-float(pre_s), float(post_s)],
                    row=row_idx,
                    col=col_idx,
                )
                if col_idx == 1:
                    fig_cluster_psth.update_yaxes(title_text="Packet rate (Hz)", row=row_idx, col=col_idx)

            fig_cluster_psth.update_layout(
                title=(
                    f"Packet Cluster PSTHs | Region {packet_feature_results['region']} | "
                    f"context={packet_feature_results['context_mode']}"
                ),
                height=760,
                margin=dict(l=70, r=40, t=90, b=130),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig_cluster_psth.update_layout(template=template, font=dict(color=base_color))
            show_fig(fig_cluster_psth, renderer=PLOTLY_RENDERER)


# %% Packet UMAP Clustering
packet_umap_cluster_results = None
if packet_cluster_results is None or packet_feature_results is None:
    print("Packet UMAP clustering: feature clustering results are not available.")
else:
    plot_title_prefix = (
        f"Region {packet_feature_results['region']} | "
        f"context={packet_feature_results['context_mode']} | "
        f"shape={PRESENT_SHAPE_MODE}"
    )
    if packet_umap_result is None:
        cluster_matrix = np.asarray(packet_cluster_results["cluster_input"]["cluster_matrix"], dtype=float)
        packet_umap_result = _compute_matrix_umap(
            cluster_matrix,
            n_components=2,
            pre_pca_components=PRESENT_UMAP_PRE_PCA_COMPONENTS,
            n_neighbors=PRESENT_UMAP_NEIGHBORS,
            min_dist=PRESENT_UMAP_MIN_DIST,
            random_state=PRESENT_UMAP_RANDOM_STATE,
        )
    if packet_umap_result is None:
        print("Packet UMAP clustering: `umap-learn` is not available or the packet matrix is too small.")
    else:
        umap_scores = np.asarray(packet_umap_result["scores"], dtype=float)
        if umap_scores.ndim != 2 or umap_scores.shape[0] < 2 or umap_scores.shape[1] < 2:
            print("Packet UMAP clustering: UMAP embedding is too small for clustering.")
        else:
            umap_cluster_labels, umap_cluster_centroids = _kmeans_numpy(
                umap_scores,
                n_clusters=PRESENT_N_CLUSTERS,
                n_init=PACKET_KMEANS_N_INIT,
                max_iter=PACKET_KMEANS_MAX_ITER,
                random_state=0,
            )
            packet_summary_umap = packet_cluster_results["packet_summary_df"].copy()
            packet_summary_umap["umap_cluster_label"] = umap_cluster_labels
            _plot_umap_cluster_embedding(
                packet_summary_umap,
                packet_umap_result,
                umap_cluster_labels,
                f"{plot_title_prefix} | UMAP k-means",
            )
            packet_umap_cluster_results = {
                "umap_result": packet_umap_result,
                "cluster_labels": umap_cluster_labels,
                "cluster_centroids": umap_cluster_centroids,
                "packet_summary_df": packet_summary_umap,
            }
            print(
                f"Packet UMAP clustering | shape_mode={PRESENT_SHAPE_MODE} | "
                f"clusters={PRESENT_N_CLUSTERS} | matrix={umap_scores.shape[0]}x{umap_scores.shape[1]}"
            )


# %% Packet UMAP Cluster PSTH
if packet_umap_cluster_results is None or packet_feature_results is None:
    print("Packet UMAP cluster PSTH: UMAP clustering results are not available.")
else:
    packet_times_cluster = np.asarray(
        packet_feature_results["packet_dataset"]["packet_times"],
        dtype=float,
    )
    cluster_labels = np.asarray(packet_umap_cluster_results["cluster_labels"], dtype=int)
    if packet_times_cluster.size == 0 or cluster_labels.size == 0:
        print("Packet UMAP cluster PSTH: no clustered packet events are available.")
    elif packet_times_cluster.size != cluster_labels.size:
        print("Packet UMAP cluster PSTH: packet times and cluster labels have inconsistent lengths.")
    else:
        packet_spikes_cluster = SimpleNamespace(
            times=packet_times_cluster,
            clusters=cluster_labels,
        )
        bin_size = config_plot.get("SINGLE_NEURON_BIN_SIZE", 0.05)
        smooth_sigma = config_plot.get("SINGLE_NEURON_SMOOTH_SIGMA", 1)
        default_pre = float(config_plot.get("SINGLE_NEURON_RASTER_PRE", 0.5))
        default_post = float(config_plot.get("SINGLE_NEURON_RASTER_POST", 1.0))
        event_specs = [
            ("stimOn_times", event_label("stimOn_times"), default_pre, default_post, f"Time from {event_label('stimOn_times')} (s)"),
            (
                "firstMovement_times",
                event_label("firstMovement_times"),
                default_pre,
                default_post,
                f"Time from {event_label('firstMovement_times')} (s)",
            ),
            (
                "feedback_times",
                event_label("feedback_times"),
                0.5,
                5.0,
                f"Time from {event_label('feedback_times')} (s)",
            ),
        ]
        valid_specs = []
        for event_name, display_label, pre_s, post_s, xaxis_title in event_specs:
            if hasattr(trials, "keys") and event_name in trials.keys():
                events = np.asarray(trials[event_name], dtype=float).reshape(-1)
                events = events[np.isfinite(events)]
                if events.size > 0:
                    valid_specs.append(
                        (event_name, display_label, events, float(pre_s), float(post_s), xaxis_title)
                    )

        whisk_context_mode = str(PACKET_WHISK_EVENT_CONTEXT).strip().lower()
        whisk_context_suffix_map = {
            "all": "",
            "task": "_task",
            "iti": "_iti",
            "spont": "_spont",
        }
        whisk_suffix = whisk_context_suffix_map.get(whisk_context_mode, "")
        whisk_brief = np.asarray(
            wh_events_by_period.get(f"wh_brief_times{whisk_suffix}", np.array([])),
            dtype=float,
        )
        whisk_long = np.asarray(
            wh_events_by_period.get(f"wh_long_times{whisk_suffix}", np.array([])),
            dtype=float,
        )
        whisk_events = np.concatenate([whisk_brief[np.isfinite(whisk_brief)], whisk_long[np.isfinite(whisk_long)]])
        if whisk_events.size > 0:
            whisk_events = np.unique(np.sort(whisk_events))
            valid_specs.append(
                (
                    "whisk_combined",
                    f"Whisk Onset ({whisk_context_mode})",
                    whisk_events,
                    default_pre,
                    default_post,
                    "Time from whisk onset (s)",
                )
            )

        if not valid_specs:
            print("Packet UMAP cluster PSTH: no valid task/whisk event series are available.")
        else:
            palette = _cluster_palette()
            unique_clusters = np.unique(cluster_labels)
            n_rows = 2
            n_cols = 2
            fig_cluster_psth = make_subplots(
                rows=n_rows,
                cols=n_cols,
                shared_yaxes=True,
                horizontal_spacing=0.08,
                vertical_spacing=0.14,
                subplot_titles=tuple(
                    f"{display_label} (n={len(events)})"
                    for _event_name, display_label, events, _pre_s, _post_s, _xaxis_title in valid_specs
                ),
            )

            for panel_idx, (_event_name, display_label, events, pre_s, post_s, xaxis_title) in enumerate(
                valid_specs,
                start=0,
            ):
                row_idx = (panel_idx // n_cols) + 1
                col_idx = (panel_idx % n_cols) + 1
                psth_by_cluster, bin_centers = compute_psth_for_clusters(
                    packet_spikes_cluster,
                    [int(cluster_idx) for cluster_idx in unique_clusters],
                    events,
                    -float(pre_s),
                    float(post_s),
                    bin_size,
                    smooth_sigma,
                    show_progress=False,
                )
                for cluster_idx in unique_clusters:
                    psth_entry = psth_by_cluster.get(int(cluster_idx))
                    if psth_entry and bin_centers is not None:
                        firing_rate = psth_entry["fr_smooth"]
                    else:
                        firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)
                    fig_cluster_psth.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=firing_rate,
                            mode="lines",
                            line=dict(
                                color=palette[int(cluster_idx) % len(palette)],
                                width=2,
                            ),
                            name=f"UMAP Cluster {int(cluster_idx) + 1} (n={int(np.sum(cluster_labels == cluster_idx))})",
                            legendgroup=f"umap-cluster-{int(cluster_idx)}",
                            showlegend=(panel_idx == 0),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                fig_cluster_psth.add_vline(
                    x=0,
                    line=dict(color="black", dash="dash"),
                    row=row_idx,
                    col=col_idx,
                )
                fig_cluster_psth.update_xaxes(
                    title_text=xaxis_title,
                    range=[-float(pre_s), float(post_s)],
                    row=row_idx,
                    col=col_idx,
                )
                if col_idx == 1:
                    fig_cluster_psth.update_yaxes(title_text="Packet rate (Hz)", row=row_idx, col=col_idx)

            fig_cluster_psth.update_layout(
                title=(
                    f"Packet UMAP Cluster PSTHs | Region {packet_feature_results['region']} | "
                    f"context={packet_feature_results['context_mode']}"
                ),
                height=760,
                margin=dict(l=70, r=40, t=90, b=130),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig_cluster_psth.update_layout(template=template, font=dict(color=base_color))
            show_fig(fig_cluster_psth, renderer=PLOTLY_RENDERER)


