# %%
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
from utils.analysis import event_label, compute_psth_for_clusters
from types import SimpleNamespace


# %%
# ---- Config ----
PID =  "c9664185-d3fd-4e0e-89cf-77c402038938"

# Data loading
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
LOAD_RAW_DATA = True
LOAD_RAW_WHEEL = False
LOAD_RAW_POSE = False
ALLOW_REMOTE_METADATA = True

# Template source and cross-validation
TEMPLATE_SOURCE = "task"  # "task" -> df_coupling_task
USE_SPLIT_TEMPLATES = True
SPLIT_A = "odd"
SPLIT_B = "even"

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
PACKET_THRESHOLD = 2.5  # z-score if PACKET_SCORE_ZSCORE else raw score
MIN_PACKET_GAP_S = 0.1

# Raster options
LABEL_MIN = 0.5
REGIONS = "VISp"  # e.g., ["VISp", "MOp"] or None for all
REGION_PREFIX_MATCH = False
SORT_CHOICE = "Task stPR Delay"
TRIAL_IDX = 829
PACKET_PSTH_REGION = None  # default: first available region
PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"


# %%
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


# %%
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


# %%
# ---- Prepare template table ----
config_calc = data.get("config_calc", {})
config_plot = dict(data.get("config_plot", {}))
config_plot["PLOTLY_TEMPLATE"] = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
config_plot["PLOT_LABEL_MIN"] = LABEL_MIN
plotting_utils.DEFAULT_TEMPLATE = config_plot["PLOTLY_TEMPLATE"]
pio.templates.default = config_plot["PLOTLY_TEMPLATE"]

df_coupling_task = data.get("df_coupling_task")
if df_coupling_task is None or len(df_coupling_task) == 0:
    raise RuntimeError("df_coupling_task is missing from cache.")

df_tpl = df_coupling_task.copy()
if "cluster_id" not in df_tpl.columns:
    raise RuntimeError("df_coupling_task missing cluster_id column.")

df_tpl = df_tpl.set_index("cluster_id", drop=False)

use_split = USE_SPLIT_TEMPLATES
split_cols = (f"stpr_curve_{SPLIT_A}", f"stpr_curve_{SPLIT_B}")
if use_split and not (split_cols[0] in df_tpl.columns and split_cols[1] in df_tpl.columns):
    print("Split stPR curves not found; falling back to mean stPR template.")
    use_split = False

stpr_bin_s = float(config_calc.get("STPR_BIN_SIZE", 0.001))


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
    if use_split:
        curve_a, s_a = _extract_template_and_strength(row, split=SPLIT_A)
        curve_b, s_b = _extract_template_and_strength(row, split=SPLIT_B)
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
    else:
        curve, s = _extract_template_and_strength(row, split=None)
        if curve.size > 0:
            templates_a[cid] = resample_template(
                curve, stpr_bin_s, DETECTION_BIN_SIZE, TEMPLATE_TIME_SCALE
            )
            strength_a[cid] = s


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


# %%
# ---- Build unit table / region selection ----
df_units, _quality_mask = plotting_utils._prepare_units_df(
    cluster_ids,
    cluster_acronyms,
    clusters,
    config_plot.get("PLOT_ONLY_GOOD_UNITS", False),
    label_min=LABEL_MIN,
)

if df_units.empty:
    raise RuntimeError("No units available after label filtering.")

if REGIONS is not None:
    regions = REGIONS if isinstance(REGIONS, (list, tuple)) else [REGIONS]
    if REGION_PREFIX_MATCH:
        region_mask = np.zeros(len(df_units), dtype=bool)
        for reg in regions:
            region_mask |= df_units["acronym"].astype(str).str.startswith(str(reg))
        df_units = df_units.loc[region_mask].copy()
    else:
        df_units = df_units[df_units["acronym"].isin([str(r) for r in regions])].copy()

if df_units.empty:
    raise RuntimeError("No units remain after region filtering.")

region_order = df_units["acronym"].dropna().unique().tolist()
print(f"Regions: {', '.join(region_order)}")


# %%
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


# %%
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

region_to_cids = {
    region: df_units.loc[df_units["acronym"] == region, "cluster_id"].to_numpy()
    for region in region_order
}

for region, cids in region_to_cids.items():
    if len(cids) == 0:
        continue

    score_sum = np.zeros(n_bins, dtype=float)
    weight_sum = np.zeros(n_bins, dtype=float)
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
        if tpl_a is not None and tpl_a.size >= 3:
            scores.append(normalized_xcorr(rate_smooth, tpl_a))
        if tpl_b is not None and tpl_b.size >= 3:
            scores.append(normalized_xcorr(rate_smooth, tpl_b))
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
        kept += 1

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
    region_event_times[region] = bin_centers[peaks] if peaks else np.array([])
    region_event_scores[region] = region_scores_plot[region][peaks] if peaks else np.array([])


# %%
# ---- Build raster (sorted) ----
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

def _safe_trial_idx(trials_obj, trial_idx):
    n_trials = len(trials_obj) if hasattr(trials_obj, "__len__") else 0
    if trial_idx is None or trial_idx < 0 or trial_idx >= n_trials:
        return 0
    return int(trial_idx)

df_units_sorted, sort_label = plotting_utils._merge_metric(
    df_units,
    sorting_metric,
    df_res=data.get("df_res"),
    df_coupling=data.get("df_coupling"),
    df_coupling_task=df_coupling_task,
    df_coupling_iti=data.get("df_coupling_iti"),
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

template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
base_color = plotting_utils._template_base_color(template)
region_colors = plotting_utils._region_color_map(region_order_sorted)


# %%
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

    window_mask = (spike_times >= t_start_plot) & (spike_times <= t_end_plot)
    window_spike_times = spike_times[window_mask] - t_offset
    window_spike_clusters = spike_clusters[window_mask]

    spike_mask = np.isin(window_spike_clusters, df_units_sorted["cluster_id"].values)
    window_spike_times = window_spike_times[spike_mask]
    window_spike_clusters = window_spike_clusters[spike_mask]

    spike_y = pd.Series(window_spike_clusters).map(cluster_index_map).to_numpy()
    spike_regions = pd.Series(window_spike_clusters).map(cluster_region_map).to_numpy()

    fig_base = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Raster", "Packet Probability"),
    )
    fig = FigureResampler(fig_base) if FigureResampler is not None else fig_base

    raster_trace = go.Scattergl(
        x=window_spike_times,
        y=spike_y,
        mode="markers",
        marker=dict(color=base_color, size=3, symbol="line-ns-open"),
        customdata=np.column_stack([window_spike_clusters, spike_regions]),
        hovertemplate=(
            "Time: %{x:.3f}s<br>Unit: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
        ),
        name="Spikes",
    )

    if FigureResampler is not None:
        fig.add_trace(
            raster_trace,
            max_n_samples=len(window_spike_times),
            hf_x=window_spike_times,
            hf_y=spike_y,
            row=1,
            col=1,
        )
    else:
        fig.add_trace(raster_trace, row=1, col=1)

    for acronym in region_order_sorted:
        group = df_units_sorted[df_units_sorted["acronym"] == acronym]
        if group.empty:
            continue
        y0 = group.index.min() - 0.5
        y1 = group.index.max() + 0.5
        fill_color = plotting_utils._color_to_rgba(region_colors.get(acronym), alpha=0.18)
        fig.add_shape(
            type="rect",
            x0=t_start_plot - t_offset,
            x1=t_end_plot - t_offset,
            y0=y0,
            y1=y1,
            line=dict(width=0),
            fillcolor=fill_color,
            layer="below",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=t_end_plot - t_offset,
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

    for region in region_order:
        score_plot = region_scores_plot.get(region)
        if score_plot is None:
            continue
        color = region_colors.get(region)
        mask = (bin_centers >= t_start_plot) & (bin_centers <= t_end_plot)
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
            row=2,
            col=1,
        )
        ev_times = region_event_times.get(region, np.array([]))
        ev_scores = region_event_scores.get(region, np.array([]))
        if ev_times.size > 0:
            keep = (ev_times >= t_start_plot) & (ev_times <= t_end_plot)
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
                row=2,
                col=1,
            )

    event_style_map = {
        "stimOn_times": ("Stim On", "blue"),
        "firstMovement_times": ("First Move", "green"),
        "response_times": ("Response", "purple"),
        "feedback_times": ("Feedback", "red"),
    }
    event_times_map = {
        "stimOn_times": t_stim_on,
        "firstMovement_times": t_first_move,
        "response_times": t_response,
        "feedback_times": t_feedback,
    }
    for event_name, (label, color) in event_style_map.items():
        t_event = event_times_map.get(event_name, np.nan)
        if not np.isfinite(t_event):
            continue
        if t_event < t_start_plot or t_event > t_end_plot:
            continue
        x_event = t_event - t_offset if align_to_event else t_event
        fig.add_vline(x=x_event, line=dict(color=color, width=1.5), row=1, col=1)
        fig.add_vline(x=x_event, line=dict(color=color, width=1.5), row=2, col=1)
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
    fig.update_xaxes(title_text=xlabel_text, row=2, col=1)

    fig.update_layout(
        title=f"{plot_title} | Sort: {sort_label}",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    fig.update_xaxes(
        range=[t_start_plot - t_offset, t_end_plot - t_offset],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[t_start_plot - t_offset, t_end_plot - t_offset],
        row=2,
        col=1,
    )

    return fig


fig = plot_trial_view(TRIAL_IDX)
show_fig(fig, renderer=PLOTLY_RENDERER)


# %%
# ---- Summary ----
print("Detected packet counts per region:")
for region in region_order:
    n_events = len(region_event_times.get(region, []))
    print(f"  {region}: {n_events}")


# %%
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


# %%
a = 2