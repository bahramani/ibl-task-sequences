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
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.io import setup_paths, init_one, load_session_data, build_cluster_id_map
import utils.plotting_plotly as plotting_utils
import utils.analysis as ana_utils
from utils.analysis import event_label, compute_psth_for_clusters
from types import SimpleNamespace


# %% Config
# ---- Config ----
# PID =  "c9664185-d3fd-4e0e-89cf-77c402038938"
PID = "f967a527-257f-404a-871d-b91575dca3b4"

# Data loading
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
LOAD_RAW_DATA = True
LOAD_RAW_WHEEL = False
LOAD_RAW_POSE = False
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
LABEL_MIN = 0.5
REGIONS = "SSp-ul"  # e.g., ["VISp", "MOp"] or None for all
REGION_PREFIX_MATCH = False
SORT_CHOICE = "Spont stPR Delay"
TRIAL_IDX = 829
TIME_WINDOW_START = 4100.0
TIME_WINDOW_END = 4120.0
PACKET_PSTH_REGION = None  # default: first available region
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


def _context_split_intervals(source):
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    spont_intervals = _interval_array_local(meta.get("spont_interval") if isinstance(meta, dict) else None)
    spont_exclude = [tuple(row) for row in spont_intervals.tolist()] if spont_intervals.size > 0 else None

    if source == "spont":
        split_a_intervals, split_b_intervals = _split_spont_intervals_alternating(
            spont_intervals,
            n_chunks=SPONT_SPLIT_SEGMENTS,
        )
        return split_a_intervals, split_b_intervals, None

    if trials is None:
        raise RuntimeError(f"Trials are required for TEMPLATE_SOURCE='{source}'.")

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


# %% Packet PCA Config
# ---- Packet PCA + clustering config ----
PACKET_EMBED_REGION = None  # defaults to packet_region, then first region with packets
PACKET_CONTEXT_MODE = "spont"  # "all", "task", "spont"
PACKET_PCA_COMPONENTS = 6
PACKET_CLUSTER_K = 3
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


def _build_packet_task_intervals():
    if trials is None or not hasattr(trials, "keys") or "stimOn_times" not in trials.keys():
        return np.empty((0, 2), dtype=float)

    event_names = list(
        config_calc.get("EVENT_NAMES", ["stimOn_times", "firstMovement_times", "feedback_times"])
    )
    post_event_s = float(config_calc.get("TASK_POST_EVENT_S", 1.0))
    stim_on = np.asarray(trials["stimOn_times"], dtype=float).reshape(-1)
    if stim_on.size == 0:
        return np.empty((0, 2), dtype=float)

    end_times = np.full(stim_on.shape, np.nan, dtype=float)
    for event_name in event_names:
        if event_name not in trials.keys():
            continue
        event_times = np.asarray(trials[event_name], dtype=float).reshape(-1)
        if event_times.shape[0] != stim_on.shape[0]:
            continue
        valid = np.isfinite(event_times)
        candidate_end = event_times + post_event_s
        update_mask = valid & (
            (~np.isfinite(end_times)) | (candidate_end > end_times)
        )
        end_times[update_mask] = candidate_end[update_mask]

    valid_rows = np.isfinite(stim_on) & np.isfinite(end_times) & (end_times > stim_on)
    if not np.any(valid_rows):
        return np.empty((0, 2), dtype=float)
    return np.column_stack([stim_on[valid_rows], end_times[valid_rows]])


def _classify_packet_context(packet_times):
    packet_times = np.asarray(packet_times, dtype=float)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    spont_intervals = _coerce_interval_array(meta.get("spont_interval") if isinstance(meta, dict) else None)
    task_intervals = _build_packet_task_intervals()

    task_mask = _mask_times_in_intervals(packet_times, task_intervals)
    spont_mask = _mask_times_in_intervals(packet_times, spont_intervals)

    labels = np.full(packet_times.shape[0], "other", dtype=object)
    labels[task_mask & ~spont_mask] = "task"
    labels[spont_mask & ~task_mask] = "spont"
    labels[task_mask & spont_mask] = "overlap"
    return labels, task_intervals, spont_intervals


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

    packet_context, task_intervals, spont_intervals = _classify_packet_context(packet_times)
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
    if context_mode not in {"all", "task", "spont"}:
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
            "other": ("Other / ITI", "#7f7f7f"),
            "overlap": ("Overlap", "#9467bd"),
        }
        for context_key in ("task", "spont", "other", "overlap"):
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
    rel_bin_centers = np.asarray(packet_dataset["rel_bin_centers"], dtype=float)
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
                x=rel_bin_centers,
                y=np.arange(matrix.shape[0]),
                colorscale="RdBu_r",
                zmin=-cmax,
                zmax=cmax,
                colorbar=dict(title="z", len=0.9) if row_idx == 1 else None,
                showscale=row_idx == 1,
                hovertemplate=(
                    "Time: %{x:.3f}s<br>"
                    "Unit row: %{y}<br>"
                    "Value: %{z:.2f}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="Units", showticklabels=False, row=row_idx, col=1)

    fig.update_xaxes(title_text="Time from packet peak (s)", row=n_rows, col=1)
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


# %% Packet PCA Run
# ---- Packet PCA + clustering ----
packet_embed_region = _choose_packet_embed_region()
packet_embedding_results = None
if packet_embed_region is None:
    print("Packet PCA: no region with detected packets is available.")
else:
    packet_dataset_all = _extract_region_packet_dataset(packet_embed_region)
    context_counts = pd.Series(packet_dataset_all["packet_context"], dtype="object").value_counts()
    if not context_counts.empty:
        count_msg = ", ".join(f"{ctx}={int(count)}" for ctx, count in context_counts.items())
        print(f"Packet PCA context counts | Region {packet_embed_region} | {count_msg}")

    packet_dataset = _filter_packet_dataset_by_context(
        packet_dataset_all,
        PACKET_CONTEXT_MODE,
    )
    n_packets = packet_dataset["packet_tensor"].shape[0]
    n_units_region = packet_dataset["packet_tensor"].shape[1]
    n_timebins = packet_dataset["packet_tensor"].shape[2]
    print(
        f"Packet PCA dataset | Region {packet_embed_region} | "
        f"context={PACKET_CONTEXT_MODE}, packets={n_packets}, "
        f"units={n_units_region}, timebins={n_timebins}"
    )

    if n_packets == 0:
        print(f"Packet PCA: no packets found for context '{PACKET_CONTEXT_MODE}'.")
    elif n_packets < 2:
        print("Packet PCA: need at least 2 packets for PCA scatter and clustering.")
    else:
        pca_result = _compute_packet_pca(
            packet_dataset["packet_tensor"],
            n_components=PACKET_PCA_COMPONENTS,
        )
        explained_ratio = np.asarray(pca_result["explained_ratio"], dtype=float)
        if explained_ratio.size >= 2:
            print(
                f"Packet PCA explained variance: "
                f"PC1={100.0 * explained_ratio[0]:.1f}%, "
                f"PC2={100.0 * explained_ratio[1]:.1f}%"
            )

        n_cluster_features = max(
            1,
            min(
                int(PACKET_CLUSTER_PCS),
                pca_result["scores"].shape[1],
            ),
        )
        cluster_labels, cluster_centroids = _kmeans_numpy(
            pca_result["scores"][:, :n_cluster_features],
            n_clusters=PACKET_CLUSTER_K,
            n_init=PACKET_KMEANS_N_INIT,
            max_iter=PACKET_KMEANS_MAX_ITER,
            random_state=0,
        )

        packet_embedding_results = dict(
            region=packet_embed_region,
            packet_dataset_all=packet_dataset_all,
            packet_dataset=packet_dataset,
            pca_result=pca_result,
            cluster_labels=cluster_labels,
            cluster_centroids=cluster_centroids,
        )

        _plot_packet_pca_scatter(packet_dataset, pca_result)
        _plot_packet_cluster_scatter(packet_dataset, pca_result, cluster_labels)
        _plot_packet_cluster_heatmaps(packet_dataset, cluster_labels)
