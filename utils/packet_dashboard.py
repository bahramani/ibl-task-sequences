from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

from utils.io import setup_paths, init_one, load_session_data, build_cluster_id_map
from utils.analysis import event_label, compute_psth_for_clusters
import utils.analysis as ana_utils
import utils.plotting_plotly as plotting_utils


BASE_PATH = Path(__file__).resolve().parents[1]
BASE_CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
PACKET_CACHE_DIR = BASE_PATH / "data" / "packet_dashboard_cache"

PACKET_DASHBOARD_VERSION = "packet-dashboard-v3.0"

DEFAULT_PACKET_CONFIG = {
    "TEMPLATE_SOURCE": "spont",
    "USE_SPLIT_TEMPLATES": True,
    "SPONT_SPLIT_SEGMENTS": 10,
    "STPR_STRENGTH_MIN": 0.01,
    "WEIGHT_BY_COUPLING": False,
    "RECTIFY_MATCH_SCORES": True,
    "DETECTION_BIN_SIZE": 0.005,
    "SMOOTH_SIGMA_S": 0.005,
    "TEMPLATE_TIME_SCALE": 1.0,
    "PACKET_SCORE_ZSCORE": True,
    "PACKET_THRESHOLD": 2.0,
    "MIN_PACKET_GAP_S": 0.1,
    "LABEL_MIN": 0.5,
    "TARGET_REGION": None,
    "SORT_METRIC_KEY": "spont",
    "PACKET_PERIOD_MIN_OVERLAP_S": 0.05,
    "FEATURES_MAIN": [
        "packet_fraction",
        "peak_rate",
        "template_dot",
        "t_com",
        "relative_rank_order",
        "temporal_width",
        "other_unit_mean_activity_at_t_com",
        "packet_rate_over_recording_rate",
    ],
    "CLUSTER_METHODS": [
        "raw_kmeans",
        "normalized_kmeans",
        "residual_kmeans",
        "raw_pca_kmeans",
        "normalized_pca_kmeans",
        "residual_pca_kmeans",
    ],
    "CLUSTER_MIN_K": 2,
    "CLUSTER_MAX_K": 6,
    "KMEANS_N_INIT": 20,
    "KMEANS_MAX_ITER": 100,
    "PCA_COMPONENTS": 6,
    "WHISK_EVENT_CONTEXT": "all",
}

MAIN_PACKET_FEATURES = list(DEFAULT_PACKET_CONFIG["FEATURES_MAIN"])
PACKET_TIME_FEATURES = {"t_com", "temporal_width"}
PACKET_SIZE_SCALED_FEATURES = {"peak_rate", "template_dot"}
PACKET_FEATURE_LABELS = {
    "packet_fraction": "Packet fraction",
    "peak_rate": "Peak rate",
    "template_dot": "Template dot",
    "t_com": "Spike COM",
    "relative_rank_order": "Relative rank order",
    "temporal_width": "Temporal width",
    "other_unit_mean_activity_at_t_com": "Other-unit mean activity at own COM",
    "packet_rate_over_recording_rate": "Packet mean rate / recording rate",
    "coupling_strength": "Coupling strength",
    "coupling_delay_ms": "Coupling delay (ms)",
    "packet_score": "Packet score (z)",
    "packet_size": "Packet size",
    "pc1": "PC1",
    "pc2": "PC2",
}
PACKET_FEATURE_ALIASES = {
    "spike_com": "t_com",
    "spike com": "t_com",
    "temporal width": "temporal_width",
    "peak rate": "peak_rate",
    "template dot": "template_dot",
    "packet fraction": "packet_fraction",
    "relative rank order": "relative_rank_order",
    "other unit mean activity at own com": "other_unit_mean_activity_at_t_com",
    "packet mean rate / recording rate": "packet_rate_over_recording_rate",
    "packet rate over recording rate": "packet_rate_over_recording_rate",
    "coupling delay": "coupling_delay_ms",
    "coupling strength": "coupling_strength",
}

PACKET_CLUSTER_METHOD_SPECS = {
    "raw_kmeans": {"shape_mode": "raw", "use_pca": False},
    "normalized_kmeans": {"shape_mode": "normalized", "use_pca": False},
    "residual_kmeans": {"shape_mode": "residual", "use_pca": False},
    "raw_pca_kmeans": {"shape_mode": "raw", "use_pca": True},
    "normalized_pca_kmeans": {"shape_mode": "normalized", "use_pca": True},
    "residual_pca_kmeans": {"shape_mode": "residual", "use_pca": True},
}
PACKET_CLUSTER_METHOD_ALIASES = {
    "pca_kmeans": "raw_pca_kmeans",
}
DEFAULT_PACKET_CLUSTER_METHOD = DEFAULT_PACKET_CONFIG["CLUSTER_METHODS"][0]


def list_available_pids(cache_dir=BASE_CACHE_DIR):
    if not Path(cache_dir).exists():
        return []
    return sorted([path.stem for path in Path(cache_dir).glob("*.pkl")])


def load_base_cache(pid, cache_dir=BASE_CACHE_DIR):
    path = Path(cache_dir) / f"{pid}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Base cache not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_packet_cache(pid, cache_dir=PACKET_CACHE_DIR):
    path = Path(cache_dir) / f"{pid}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Packet cache not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_packet_cache(pid, cache, cache_dir=PACKET_CACHE_DIR):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pid}.pkl"
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    return path


def load_raw_session(
    pid,
    load_wheel=False,
    load_pose=False,
    load_motion_energy=False,
    mode="local",
):
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
    new_len = max(int(round(curve.size * scale)), 3)
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
    length = tpl_z.size
    kernel = np.ones(length, dtype=float) / float(length)
    mu = np.convolve(signal, kernel, mode="same")
    mu2 = np.convolve(signal * signal, kernel, mode="same")
    var = mu2 - mu * mu
    var[var < 0] = 0
    sigma = np.sqrt(var)

    dot = np.convolve(signal, tpl_z[::-1], mode="same")
    denom = sigma * float(length)
    score = np.full(signal.shape, np.nan, dtype=float)
    valid = denom > 0
    score[valid] = dot[valid] / denom[valid]

    half = length // 2
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
        return np.zeros_like(values, dtype=float)
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
    for start_idx, end_idx in zip(starts, ends):
        peak_idx = start_idx + int(np.nanargmax(safe_score[start_idx:end_idx]))
        peaks.append(peak_idx)

    peaks = sorted(peaks, key=lambda idx: times[idx])
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


def _mean_strength(cid, strength_a, strength_b):
    s_a = strength_a.get(int(cid), np.nan)
    s_b = strength_b.get(int(cid), np.nan)
    if np.isfinite(s_a) and np.isfinite(s_b):
        return float((s_a + s_b) / 2.0)
    if np.isfinite(s_a):
        return float(s_a)
    if np.isfinite(s_b):
        return float(s_b)
    return np.nan


def _prepare_units_df(cluster_ids, cluster_acronyms, clusters, label_min):
    df_units, _ = plotting_utils._prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        only_good=False,
        label_min=label_min,
    )
    return df_units.copy()


def _sort_units_df(df_units, df_res, df_coupling, df_coupling_task, df_coupling_iti, metric_key):
    df_units_sorted, sort_label = plotting_utils._merge_metric(
        df_units.copy(),
        metric_key,
        df_res=df_res,
        df_coupling=df_coupling,
        df_coupling_task=df_coupling_task,
        df_coupling_iti=df_coupling_iti,
        df_firing_rate=None,
    )
    df_units_sorted, region_order_sorted, sort_label = plotting_utils._sort_within_regions(
        df_units_sorted,
        sort_label,
        metric_key=metric_key,
    )
    return df_units_sorted, region_order_sorted, sort_label


def _split_intervals_equal_chunks(intervals, n_chunks):
    arr = _coerce_interval_array(intervals)
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


def _build_behavior_period_windows(trials, config_calc, meta):
    spont_intervals = _coerce_interval_array(meta.get("spont_interval") if isinstance(meta, dict) else None)
    if trials is None:
        empty_cols = ["trial_idx", "start", "end", "correct", "odd"]
        return spont_intervals, pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=["trial_idx", "start", "end", "odd"])

    event_names = list(config_calc.get("EVENT_NAMES", ["stimOn_times", "firstMovement_times", "feedback_times"]))
    task_post_event_s = float(config_calc.get("TASK_POST_EVENT_S", 1.0))
    task_windows = ana_utils.build_task_window_table(trials, event_names, post_event_s=task_post_event_s)
    stim_on_times = (
        np.asarray(trials["stimOn_times"], dtype=float).reshape(-1)
        if hasattr(trials, "keys") and "stimOn_times" in trials.keys()
        else np.array([], dtype=float)
    )
    trial_end_times = ana_utils.compute_trial_end_times(trials, event_names, post_event_s=task_post_event_s)
    iti_windows = ana_utils.build_iti_windows(
        trial_end_times,
        stim_on_times,
        skip_first_last=bool(config_calc.get("ITI_SKIP_FIRST_LAST", True)),
    )
    return spont_intervals, task_windows, iti_windows


def _context_split_intervals(source, spont_intervals, task_windows, iti_windows, packet_config):
    source = str(source).strip().lower()
    spont_exclude = [tuple(row) for row in spont_intervals.tolist()] if spont_intervals.size > 0 else None
    if source == "spont":
        split_a, split_b = _split_spont_intervals_alternating(
            spont_intervals,
            n_chunks=int(packet_config["SPONT_SPLIT_SEGMENTS"]),
        )
        return split_a, split_b, None
    if source == "task":
        if task_windows.empty:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), spont_exclude
        split_a = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        split_b = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        return split_a, split_b, spont_exclude
    if source == "iti":
        if iti_windows.empty:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), spont_exclude
        split_a = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        split_b = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float)
        return split_a, split_b, spont_exclude
    raise ValueError(f"Unsupported template source: {source}")


def _compute_stpr_split_df(spikes, clusters, cluster_acronyms, config_calc, intervals, exclude_intervals, context_label, cluster_ids_use):
    intervals = _coerce_interval_array(intervals)
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


def build_template_source_table(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    config_calc,
    packet_config,
    meta,
    trials,
):
    df_units = _prepare_units_df(cluster_ids, cluster_acronyms, clusters, packet_config["LABEL_MIN"])
    cluster_ids_use = df_units["cluster_id"].to_numpy(dtype=int)
    if cluster_ids_use.size == 0:
        raise RuntimeError("No units available for packet template construction.")

    source = str(packet_config["TEMPLATE_SOURCE"]).strip().lower()
    spont_intervals, task_windows, iti_windows = _build_behavior_period_windows(trials, config_calc, meta)
    split_a, split_b, exclude_intervals = _context_split_intervals(
        source,
        spont_intervals,
        task_windows,
        iti_windows,
        packet_config,
    )
    df_a = _compute_stpr_split_df(
        spikes,
        clusters,
        cluster_acronyms,
        config_calc,
        split_a,
        exclude_intervals,
        f"{source.capitalize()} odd",
        cluster_ids_use,
    )
    df_b = _compute_stpr_split_df(
        spikes,
        clusters,
        cluster_acronyms,
        config_calc,
        split_b,
        exclude_intervals,
        f"{source.capitalize()} even",
        cluster_ids_use,
    )
    if df_a is None and df_b is None:
        raise RuntimeError(f"No split stPR tables were computed for template source {source}.")
    df_tpl = ana_utils.merge_stpr_splits(
        df_a,
        df_b,
        config_calc,
        split_a="odd",
        split_b="even",
    )
    if df_tpl is None or df_tpl.empty:
        raise RuntimeError(f"Template source {source} did not produce a valid table.")
    return df_tpl.reset_index(drop=True).copy(), spont_intervals, task_windows, iti_windows


def _extract_template_and_strength(row, split=None):
    if split:
        curve = row.get(f"stpr_curve_{split}", [])
        strength = row.get(f"coupling_strength_{split}", np.nan)
    else:
        curve = row.get("stpr_curve", [])
        strength = row.get("coupling_strength", np.nan)
    return np.asarray(curve, dtype=float), strength


def build_template_lookup(df_tpl, stpr_bin_s, detection_bin_s, time_scale=1.0, use_split=True):
    templates_a = {}
    templates_b = {}
    strength_a = {}
    strength_b = {}
    if "cluster_id" not in df_tpl.columns:
        raise ValueError("df_tpl must contain cluster_id.")
    for _row_idx, row in df_tpl.iterrows():
        cid = int(row["cluster_id"])
        curve_a, s_a = _extract_template_and_strength(row, split="odd")
        curve_b, s_b = _extract_template_and_strength(row, split="even")
        if curve_a.size > 0:
            templates_a[cid] = resample_template(curve_a, stpr_bin_s, detection_bin_s, time_scale)
            strength_a[cid] = s_a
        if curve_b.size > 0:
            templates_b[cid] = resample_template(curve_b, stpr_bin_s, detection_bin_s, time_scale)
            strength_b[cid] = s_b
        if not use_split:
            curve_mean, s_mean = _extract_template_and_strength(row, split=None)
            if curve_mean.size > 0:
                templates_a[cid] = resample_template(curve_mean, stpr_bin_s, detection_bin_s, time_scale)
                strength_a[cid] = s_mean
    return templates_a, templates_b, strength_a, strength_b


def _empty_whisk_outputs():
    return {
        "df_wh": pd.DataFrame(columns=["bin_idx", "bin_start_s", "bin_center_s", "bin_end_s", "wh_norm", "n_views"]),
        "wh_detect": {
            "all_bouts": np.empty((0, 2), dtype=float),
            "brief_bouts": np.empty((0, 2), dtype=float),
            "long_bouts": np.empty((0, 2), dtype=float),
            "all_onsets": np.array([], dtype=float),
            "brief_onsets": np.array([], dtype=float),
            "long_onsets": np.array([], dtype=float),
        },
        "wh_event_base": {
            "wh_brief_times": np.array([], dtype=float),
            "wh_long_times": np.array([], dtype=float),
            "wh_all_times": np.array([], dtype=float),
        },
        "wh_events_by_period": {},
    }


def build_whisk_bundle(base_cache, config_calc, session, spont_intervals, task_windows, iti_windows):
    df_wh = base_cache.get("df_wh")
    if not isinstance(df_wh, pd.DataFrame):
        df_wh = pd.DataFrame(columns=["bin_idx", "bin_start_s", "bin_center_s", "bin_end_s", "wh_norm", "n_views"])

    whisk_bundle = None
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
        except Exception:
            whisk_bundle = None

    if whisk_bundle is None:
        cached_wh_detect = base_cache.get("wh_detect")
        cached_wh_event_base = base_cache.get("wh_event_base")
        cached_wh_events_by_period = base_cache.get("wh_events_by_period")
        if isinstance(cached_wh_event_base, dict) and isinstance(cached_wh_events_by_period, dict):
            whisk_bundle = {
                "wh_detect": cached_wh_detect if isinstance(cached_wh_detect, dict) else {},
                "wh_event_base": cached_wh_event_base,
                "wh_events_by_period": cached_wh_events_by_period,
            }

    if whisk_bundle is None:
        return _empty_whisk_outputs()

    return {
        "df_wh": df_wh,
        "wh_detect": whisk_bundle.get("wh_detect", {}),
        "wh_event_base": whisk_bundle.get("wh_event_base", {}),
        "wh_events_by_period": whisk_bundle.get("wh_events_by_period", {}),
    }


def build_spike_index(spikes, selected_cluster_ids):
    spike_times = np.asarray(spikes["times"] if isinstance(spikes, dict) else spikes.times, dtype=float)
    spike_clusters = np.asarray(spikes["clusters"] if isinstance(spikes, dict) else spikes.clusters, dtype=int)
    selected_mask = np.isin(spike_clusters, np.asarray(selected_cluster_ids, dtype=int))
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
            spike_times_by_cluster[int(cid)] = np.asarray(times_sorted[start:end], dtype=float)
    return spike_times_by_cluster, spike_times, spike_clusters


def _classify_packet_context(packet_times, task_intervals, spont_intervals, iti_intervals):
    packet_times = np.asarray(packet_times, dtype=float)
    task_intervals = _coerce_interval_array(task_intervals)
    spont_intervals = _coerce_interval_array(spont_intervals)
    iti_intervals = _coerce_interval_array(iti_intervals)

    def _contains(intervals):
        if intervals.size == 0:
            return np.zeros(packet_times.shape[0], dtype=bool)
        starts = intervals[:, 0][None, :]
        ends = intervals[:, 1][None, :]
        pts = packet_times[:, None]
        return np.any((pts >= starts) & (pts <= ends), axis=1)

    task_mask = _contains(task_intervals)
    spont_mask = _contains(spont_intervals)
    iti_mask = _contains(iti_intervals)
    overlap_mask = (task_mask.astype(int) + spont_mask.astype(int) + iti_mask.astype(int)) > 1
    labels = np.full(packet_times.shape[0], "other", dtype=object)
    labels[task_mask & ~spont_mask] = "task"
    labels[spont_mask & ~task_mask & ~iti_mask] = "spont"
    labels[iti_mask & ~task_mask & ~spont_mask] = "iti"
    labels[overlap_mask] = "overlap"
    return labels


def detect_packets_by_region(
    df_units,
    region_order,
    spike_times_by_cluster,
    templates_a,
    templates_b,
    strength_a,
    strength_b,
    packet_config,
):
    spike_times_all = np.concatenate([arr for arr in spike_times_by_cluster.values()]) if spike_times_by_cluster else np.array([], dtype=float)
    if spike_times_all.size == 0:
        raise RuntimeError("No spikes available after unit filtering.")

    t_start = float(np.nanmin(spike_times_all))
    t_end = float(np.nanmax(spike_times_all))
    bin_edges = np.arange(t_start, t_end + packet_config["DETECTION_BIN_SIZE"], packet_config["DETECTION_BIN_SIZE"])
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    sigma_bins = packet_config["SMOOTH_SIGMA_S"] / packet_config["DETECTION_BIN_SIZE"]
    kernel = gaussian_kernel(sigma_bins)

    region_detection = {}
    for region in region_order:
        cids = df_units.loc[df_units["acronym"] == region, "cluster_id"].to_numpy(dtype=int)
        if cids.size == 0:
            continue
        score_sum = np.zeros(n_bins, dtype=float)
        weight_sum = np.zeros(n_bins, dtype=float)
        population_rate_sum = np.zeros(n_bins, dtype=float)
        population_count = 0
        used_cids_region = []
        template_durations_s = []
        for cid in tqdm(cids, desc=f"Packet detect {region}", unit="unit"):
            strength_val = _mean_strength(cid, strength_a, strength_b)
            if not np.isfinite(strength_val) or strength_val < packet_config["STPR_STRENGTH_MIN"]:
                continue
            tpl_a = templates_a.get(int(cid))
            tpl_b = templates_b.get(int(cid)) if packet_config["USE_SPLIT_TEMPLATES"] else None
            if tpl_a is None and tpl_b is None:
                continue
            spikes_c = np.asarray(spike_times_by_cluster.get(int(cid), np.array([])), dtype=float)
            if spikes_c.size == 0:
                continue
            bin_idx = ((spikes_c - t_start) / packet_config["DETECTION_BIN_SIZE"]).astype(int)
            mask = (bin_idx >= 0) & (bin_idx < n_bins)
            if not np.any(mask):
                continue
            counts = np.bincount(bin_idx[mask], minlength=n_bins).astype(float)
            rate = counts / packet_config["DETECTION_BIN_SIZE"]
            rate_smooth = smooth_signal(rate, kernel)
            population_rate_sum += rate_smooth
            population_count += 1

            scores = []
            template_lengths = []
            if tpl_a is not None and np.asarray(tpl_a).size >= 3:
                scores.append(normalized_xcorr(rate_smooth, tpl_a))
                template_lengths.append(len(tpl_a))
            if tpl_b is not None and np.asarray(tpl_b).size >= 3:
                scores.append(normalized_xcorr(rate_smooth, tpl_b))
                template_lengths.append(len(tpl_b))
            if not scores:
                continue
            score = scores[0] if len(scores) == 1 else np.nanmean(np.vstack(scores), axis=0)
            if packet_config["RECTIFY_MATCH_SCORES"]:
                score = np.maximum(0.0, score)
            weight = float(strength_val) if packet_config["WEIGHT_BY_COUPLING"] else 1.0
            valid = np.isfinite(score)
            if np.any(valid):
                score_sum[valid] += weight * score[valid]
                weight_sum[valid] += weight
            if template_lengths:
                template_durations_s.append(float(np.mean(template_lengths)) * packet_config["DETECTION_BIN_SIZE"])
            used_cids_region.append(int(cid))

        if not np.any(weight_sum > 0):
            continue

        region_score_raw = np.full(n_bins, np.nan, dtype=float)
        valid_bins = weight_sum > 0
        region_score_raw[valid_bins] = score_sum[valid_bins] / weight_sum[valid_bins]
        region_score_plot = robust_zscore(region_score_raw) if packet_config["PACKET_SCORE_ZSCORE"] else region_score_raw.copy()
        peaks = detect_peaks(region_score_plot, bin_centers, packet_config["PACKET_THRESHOLD"], packet_config["MIN_PACKET_GAP_S"])
        packet_times = bin_centers[peaks] if peaks else np.array([], dtype=float)
        packet_scores = region_score_plot[peaks] if peaks else np.array([], dtype=float)
        template_duration_s = float(np.nanmedian(template_durations_s)) if template_durations_s else np.nan
        if packet_times.size > 0 and np.isfinite(template_duration_s) and template_duration_s > 0:
            half_window_s = template_duration_s / 2.0
            packet_windows = np.column_stack([packet_times - half_window_s, packet_times + half_window_s])
        else:
            packet_windows = np.empty((0, 2), dtype=float)

        region_detection[region] = {
            "bin_centers": bin_centers.astype(np.float32),
            "region_score_raw": region_score_raw.astype(np.float32),
            "region_score_plot": region_score_plot.astype(np.float32),
            "population_rate_hz": (
                (population_rate_sum / max(population_count, 1)).astype(np.float32)
                if population_count > 0
                else np.zeros(n_bins, dtype=np.float32)
            ),
            "packet_times": packet_times.astype(np.float32),
            "packet_scores": packet_scores.astype(np.float32),
            "packet_windows": packet_windows.astype(np.float32),
            "template_duration_s": template_duration_s,
            "used_cluster_ids": np.asarray(used_cids_region, dtype=int),
        }
    return region_detection, kernel


def _resample_vector_to_len(values, target_len):
    values = np.asarray(values, dtype=float)
    target_len = int(target_len)
    if values.size == 0:
        return np.zeros(target_len, dtype=float)
    if values.size == target_len:
        return values.copy()
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values)


def _packet_region_template_matrix(cluster_ids_region, target_len, templates_a, templates_b, use_split):
    rows = []
    for cid in np.asarray(cluster_ids_region, dtype=int):
        tpl_rows = []
        tpl_a = templates_a.get(int(cid))
        tpl_b = templates_b.get(int(cid)) if use_split else None
        if tpl_a is not None and np.asarray(tpl_a).size >= 3:
            tpl_rows.append(_resample_vector_to_len(tpl_a, target_len))
        if tpl_b is not None and np.asarray(tpl_b).size >= 3:
            tpl_rows.append(_resample_vector_to_len(tpl_b, target_len))
        rows.append(np.nanmean(np.vstack(tpl_rows), axis=0) if tpl_rows else np.zeros(int(target_len), dtype=float))
    return np.vstack(rows) if rows else np.zeros((0, int(target_len)), dtype=float)


def extract_region_packet_dataset(
    region,
    region_detection,
    df_units_sorted,
    spike_times_by_cluster,
    kernel,
    templates_a,
    templates_b,
    packet_config,
    task_windows,
    spont_intervals,
    iti_windows,
):
    det = region_detection.get(region)
    if det is None:
        raise RuntimeError(f"No detection result for region {region}.")
    packet_times = np.asarray(det["packet_times"], dtype=float)
    if packet_times.size == 0:
        raise RuntimeError(f"No packets detected for region {region}.")

    used_ids = np.asarray(det["used_cluster_ids"], dtype=int)
    region_units_df = df_units_sorted.loc[
        (df_units_sorted["acronym"] == region) & (df_units_sorted["cluster_id"].isin(used_ids))
    ].copy()
    if region_units_df.empty:
        raise RuntimeError(f"No unit rows available for region {region}.")

    template_duration_s = float(det["template_duration_s"])
    if not np.isfinite(template_duration_s) or template_duration_s <= 0:
        raise RuntimeError(f"Invalid template duration for region {region}.")
    n_timebins = max(3, int(round(template_duration_s / packet_config["DETECTION_BIN_SIZE"])))
    packet_window_s = n_timebins * packet_config["DETECTION_BIN_SIZE"]
    half_window_s = packet_window_s / 2.0
    rel_bin_edges = np.linspace(-half_window_s, half_window_s, n_timebins + 1)
    rel_bin_centers = (rel_bin_edges[:-1] + rel_bin_edges[1:]) / 2.0
    packet_windows = np.asarray(det["packet_windows"], dtype=float)
    if packet_windows.shape != (packet_times.size, 2):
        packet_windows = np.column_stack([packet_times - half_window_s, packet_times + half_window_s])

    cluster_ids_region = region_units_df["cluster_id"].to_numpy(dtype=int)
    packet_tensor = np.zeros((packet_times.size, cluster_ids_region.size, n_timebins), dtype=np.float32)
    for pkt_idx, peak_time in enumerate(tqdm(packet_times, desc=f"Extract packets {region}", unit="packet")):
        for unit_idx, cid in enumerate(cluster_ids_region):
            spikes_rel = np.asarray(spike_times_by_cluster.get(int(cid), np.array([])), dtype=float) - float(peak_time)
            mask = (spikes_rel >= rel_bin_edges[0]) & (spikes_rel <= rel_bin_edges[-1])
            if np.any(mask):
                counts, _ = np.histogram(spikes_rel[mask], bins=rel_bin_edges)
                rate = counts.astype(float) / packet_config["DETECTION_BIN_SIZE"]
            else:
                rate = np.zeros(n_timebins, dtype=float)
            packet_tensor[pkt_idx, unit_idx] = smooth_signal(rate, kernel)

    task_intervals = task_windows.loc[:, ["start", "end"]].to_numpy(dtype=float) if not task_windows.empty else np.empty((0, 2), dtype=float)
    iti_intervals = iti_windows.loc[:, ["start", "end"]].to_numpy(dtype=float) if not iti_windows.empty else np.empty((0, 2), dtype=float)
    packet_context = _classify_packet_context(packet_times, task_intervals, spont_intervals, iti_intervals)
    template_matrix = _packet_region_template_matrix(
        cluster_ids_region,
        n_timebins,
        templates_a,
        templates_b,
        packet_config["USE_SPLIT_TEMPLATES"],
    )
    return {
        "region": region,
        "packet_times": packet_times.astype(np.float32),
        "packet_scores": np.asarray(det["packet_scores"], dtype=np.float32),
        "packet_windows": packet_windows.astype(np.float32),
        "packet_context": packet_context,
        "packet_tensor": packet_tensor.astype(np.float32),
        "cluster_ids": cluster_ids_region,
        "unit_table": region_units_df.reset_index(drop=True).copy(),
        "template_matrix": template_matrix.astype(np.float32),
        "rel_bin_centers": rel_bin_centers.astype(np.float32),
        "rel_bin_edges": rel_bin_edges.astype(np.float32),
        "packet_window_s": float(packet_window_s),
        "task_intervals": task_intervals.astype(np.float32),
        "spont_intervals": spont_intervals.astype(np.float32),
        "iti_intervals": iti_intervals.astype(np.float32),
        "detection_bin_centers": np.asarray(det["bin_centers"], dtype=np.float32),
        "region_score_plot": np.asarray(det["region_score_plot"], dtype=np.float32),
        "region_score_raw": np.asarray(det["region_score_raw"], dtype=np.float32),
        "population_rate_hz": np.asarray(det["population_rate_hz"], dtype=np.float32),
    }


def _estimate_baseline_rates(cluster_ids_region, packet_dataset, context_mode, spike_times_by_cluster, t_start, t_end):
    context_mode = str(context_mode).strip().lower()
    if context_mode == "task":
        intervals = packet_dataset.get("task_intervals", np.empty((0, 2), dtype=float))
    elif context_mode == "spont":
        intervals = packet_dataset.get("spont_intervals", np.empty((0, 2), dtype=float))
    elif context_mode == "iti":
        intervals = packet_dataset.get("iti_intervals", np.empty((0, 2), dtype=float))
    else:
        intervals = None

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
            n_spikes_ctx = 0
            for interval in intervals:
                left = np.searchsorted(spikes_c, float(interval[0]), side="left")
                right = np.searchsorted(spikes_c, float(interval[1]), side="right")
                n_spikes_ctx += int(max(0, right - left))
        baseline_rates[int(cid)] = float(n_spikes_ctx) / float(duration_s)
    return baseline_rates


def _safe_weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    if not np.any(valid):
        return np.nan
    vals = vals[valid]
    w = w[valid]
    if float(np.sum(w)) > 0:
        return float(np.average(vals, weights=w))
    return float(np.mean(vals))


def _weighted_variance(values, weights):
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    if not np.any(valid):
        return np.nan
    vals = vals[valid]
    w = w[valid]
    if float(np.sum(w)) <= 0:
        return float(np.nanvar(vals))
    mean_val = float(np.average(vals, weights=w))
    return float(np.sum(w * (vals - mean_val) ** 2) / np.sum(w))


def compute_main_packet_feature_tables(packet_dataset, df_tpl_for_merge, spike_times_by_cluster, context_mode):
    packet_times = np.asarray(packet_dataset["packet_times"], dtype=float)
    packet_scores = np.asarray(packet_dataset["packet_scores"], dtype=float)
    packet_context = np.asarray(packet_dataset["packet_context"], dtype=object)
    packet_tensor = np.asarray(packet_dataset["packet_tensor"], dtype=float)
    template_matrix = np.asarray(packet_dataset["template_matrix"], dtype=float)
    rel_bin_centers = np.asarray(packet_dataset["rel_bin_centers"], dtype=float)
    rel_bin_edges = np.asarray(packet_dataset["rel_bin_edges"], dtype=float)
    packet_window_s = float(packet_dataset.get("packet_window_s", np.nan))
    cluster_ids_region = np.asarray(packet_dataset["cluster_ids"], dtype=int)
    unit_table = packet_dataset["unit_table"][["cluster_id", "acronym"]].copy()

    all_spikes = np.concatenate([np.asarray(v, dtype=float) for v in spike_times_by_cluster.values()]) if spike_times_by_cluster else np.array([], dtype=float)
    t_start = float(np.nanmin(all_spikes)) if all_spikes.size > 0 else 0.0
    t_end = float(np.nanmax(all_spikes)) if all_spikes.size > 0 else 0.0
    baseline_rates_hz = _estimate_baseline_rates(cluster_ids_region, packet_dataset, context_mode, spike_times_by_cluster, t_start, t_end)

    coupling_lookup = df_tpl_for_merge[["cluster_id", "coupling_strength", "coupling_delay_ms"]].drop_duplicates(subset=["cluster_id"])
    unit_table = unit_table.merge(coupling_lookup, on="cluster_id", how="left")
    region_by_cid = dict(zip(unit_table["cluster_id"], unit_table["acronym"]))
    strength_by_cid = dict(zip(unit_table["cluster_id"], unit_table["coupling_strength"]))
    delay_by_cid = dict(zip(unit_table["cluster_id"], unit_table["coupling_delay_ms"]))

    feature_rows = []
    dt_bin = float(rel_bin_edges[1] - rel_bin_edges[0]) if rel_bin_edges.size >= 2 else 0.0
    for pkt_idx, peak_time in enumerate(tqdm(packet_times, desc=f"Packet features {packet_dataset['region']}", unit="packet")):
        n_units = cluster_ids_region.size
        spike_count = np.zeros(n_units, dtype=float)
        binary_participation = np.zeros(n_units, dtype=float)
        packet_fraction = np.zeros(n_units, dtype=float)
        peak_rate = np.zeros(n_units, dtype=float)
        template_dot = np.zeros(n_units, dtype=float)
        t_com = np.full(n_units, np.nan, dtype=float)
        t_first_spike = np.full(n_units, np.nan, dtype=float)
        relative_rank_order = np.full(n_units, np.nan, dtype=float)
        temporal_width = np.full(n_units, np.nan, dtype=float)
        other_unit_mean_activity_at_t_com = np.full(n_units, np.nan, dtype=float)
        packet_rate_over_recording_rate = np.full(n_units, np.nan, dtype=float)

        for unit_idx, cid in enumerate(cluster_ids_region):
            cid = int(cid)
            rate_trace = np.nan_to_num(packet_tensor[pkt_idx, unit_idx], nan=0.0)
            peak_rate_val = float(np.nanmax(rate_trace)) if rate_trace.size > 0 else 0.0
            peak_rate[unit_idx] = max(0.0, peak_rate_val) if np.isfinite(peak_rate_val) else 0.0
            template_trace = (
                np.nan_to_num(template_matrix[unit_idx], nan=0.0)
                if template_matrix.shape[0] > unit_idx
                else np.zeros_like(rate_trace)
            )
            spikes_rel = np.asarray(spike_times_by_cluster.get(cid, np.array([])), dtype=float) - float(peak_time)
            keep = (spikes_rel >= rel_bin_edges[0]) & (spikes_rel <= rel_bin_edges[-1])
            spikes_rel = np.asarray(spikes_rel[keep], dtype=float)
            n_spikes = int(spikes_rel.size)
            if n_spikes == 0:
                continue

            spike_count[unit_idx] = float(n_spikes)
            binary_participation[unit_idx] = 1.0
            template_dot[unit_idx] = float(np.dot(rate_trace, template_trace) * dt_bin)
            t_first_spike[unit_idx] = float(np.min(spikes_rel))
            t_com[unit_idx] = float(np.mean(spikes_rel))
            temporal_width[unit_idx] = float(np.std(spikes_rel)) if n_spikes > 1 else 0.0

            baseline_rate_hz = float(baseline_rates_hz.get(cid, np.nan))
            if np.isfinite(baseline_rate_hz) and baseline_rate_hz > 0 and np.isfinite(packet_window_s) and packet_window_s > 0:
                packet_mean_rate_hz = float(n_spikes) / float(packet_window_s)
                packet_rate_over_recording_rate[unit_idx] = packet_mean_rate_hz / baseline_rate_hz

        if n_units >= 2 and rel_bin_centers.size > 0:
            for unit_idx in range(n_units):
                if not np.isfinite(t_com[unit_idx]):
                    continue
                t_com_val = float(np.clip(float(t_com[unit_idx]), float(rel_bin_centers[0]), float(rel_bin_centers[-1])))
                interp_vals = np.array(
                    [
                        float(np.interp(t_com_val, rel_bin_centers, np.nan_to_num(packet_tensor[pkt_idx, other_idx], nan=0.0)))
                        for other_idx in range(n_units)
                    ],
                    dtype=float,
                )
                other_mask = np.ones(n_units, dtype=bool)
                other_mask[unit_idx] = False
                other_vals = interp_vals[other_mask]
                valid_other = np.isfinite(other_vals)
                if np.any(valid_other):
                    other_unit_mean_activity_at_t_com[unit_idx] = float(np.mean(other_vals[valid_other]))

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
                    "packet_peak_time_s": float(packet_times[pkt_idx]),
                    "packet_score": float(packet_scores[pkt_idx]),
                    "packet_context": str(packet_context[pkt_idx]),
                    "cluster_id": int(cid),
                    "region": str(region_by_cid.get(int(cid), packet_dataset["region"])),
                    "coupling_strength": float(strength_by_cid.get(int(cid), np.nan)),
                    "coupling_delay_ms": float(delay_by_cid.get(int(cid), np.nan)),
                    "packet_fraction": float(packet_fraction[unit_idx]),
                    "peak_rate": float(peak_rate[unit_idx]),
                    "template_dot": float(template_dot[unit_idx]),
                    "t_com": float(t_com[unit_idx]),
                    "relative_rank_order": float(relative_rank_order[unit_idx]),
                    "temporal_width": float(temporal_width[unit_idx]),
                    "other_unit_mean_activity_at_t_com": float(other_unit_mean_activity_at_t_com[unit_idx]),
                    "packet_rate_over_recording_rate": float(packet_rate_over_recording_rate[unit_idx]),
                }
            )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        return feature_df, pd.DataFrame(), pd.DataFrame()

    true_packet_size = np.zeros(len(packet_times), dtype=float)
    for pkt_idx, group_df in feature_df.groupby("packet_idx", sort=False):
        fractions = pd.to_numeric(group_df["packet_fraction"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(fractions) & (fractions > 0)
        if np.any(valid):
            true_packet_size[int(pkt_idx)] = float(np.nanmedian(1.0 / fractions[valid]))
    true_packet_size = np.where(np.isfinite(true_packet_size), true_packet_size, 0.0)
    size_series = pd.Series(true_packet_size, index=np.arange(len(true_packet_size), dtype=int))
    for feat in MAIN_PACKET_FEATURES:
        norm_col = f"{feat}__norm"
        values = pd.to_numeric(feature_df[feat], errors="coerce").to_numpy(dtype=float)
        if feat in PACKET_SIZE_SCALED_FEATURES:
            packet_sizes = feature_df["packet_idx"].map(size_series).to_numpy(dtype=float)
            norm_vals = np.full(values.shape, np.nan, dtype=float)
            valid = np.isfinite(values) & np.isfinite(packet_sizes) & (packet_sizes > 0)
            norm_vals[valid] = values[valid] / packet_sizes[valid]
            zero_mask = (packet_sizes <= 0) & np.isfinite(values) & (values == 0)
            norm_vals[zero_mask] = 0.0
            feature_df[norm_col] = norm_vals
        else:
            feature_df[norm_col] = values

    neuron_summary = feature_df.groupby(["cluster_id", "region"], as_index=False)[MAIN_PACKET_FEATURES].mean().copy()
    var_rows = []
    for (cluster_id, region), group_df in feature_df.groupby(["cluster_id", "region"], sort=False):
        row = {"cluster_id": int(cluster_id), "region": str(region)}
        for feat in MAIN_PACKET_FEATURES:
            values = pd.to_numeric(group_df[feat], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            row[f"{feat}__var"] = float(np.nanvar(values[valid])) if np.any(valid) else np.nan
        var_rows.append(row)
    neuron_summary = neuron_summary.merge(pd.DataFrame(var_rows), on=["cluster_id", "region"], how="left")
    neuron_summary = neuron_summary.merge(
        unit_table[["cluster_id", "coupling_strength", "coupling_delay_ms"]].drop_duplicates(subset=["cluster_id"]),
        on="cluster_id",
        how="left",
    )

    packet_rows = []
    packet_coupling_rows = []
    for pkt_idx, group_df in feature_df.groupby("packet_idx", sort=True):
        row = {
            "packet_idx": int(pkt_idx),
            "packet_peak_time_s": float(packet_times[int(pkt_idx)]),
            "packet_score": float(packet_scores[int(pkt_idx)]),
            "packet_context": str(packet_context[int(pkt_idx)]),
            "packet_size": float(true_packet_size[int(pkt_idx)]),
        }
        weights = pd.to_numeric(group_df["packet_fraction"], errors="coerce").to_numpy(dtype=float)
        for feat in MAIN_PACKET_FEATURES:
            raw_values = pd.to_numeric(group_df[feat], errors="coerce").to_numpy(dtype=float)
            norm_values = pd.to_numeric(group_df[f"{feat}__norm"], errors="coerce").to_numpy(dtype=float)
            if feat in PACKET_TIME_FEATURES:
                active = pd.to_numeric(group_df["packet_fraction"], errors="coerce").to_numpy(dtype=float) > 0
                row[f"{feat}__packet_raw"] = _safe_weighted_mean(raw_values[active], weights[active])
                row[f"{feat}__packet_raw_var"] = _weighted_variance(raw_values[active], weights[active])
                row[f"{feat}__packet_norm"] = _safe_weighted_mean(norm_values[active], weights[active])
                row[f"{feat}__packet_norm_var"] = _weighted_variance(norm_values[active], weights[active])
            else:
                valid_raw = np.isfinite(raw_values)
                valid_norm = np.isfinite(norm_values)
                row[f"{feat}__packet_raw"] = float(np.mean(raw_values[valid_raw])) if np.any(valid_raw) else np.nan
                row[f"{feat}__packet_raw_var"] = float(np.var(raw_values[valid_raw])) if np.any(valid_raw) else np.nan
                row[f"{feat}__packet_norm"] = float(np.mean(norm_values[valid_norm])) if np.any(valid_norm) else np.nan
                row[f"{feat}__packet_norm_var"] = float(np.var(norm_values[valid_norm])) if np.any(valid_norm) else np.nan
        packet_rows.append(row)

        coupling_strength = pd.to_numeric(group_df["coupling_strength"], errors="coerce").to_numpy(dtype=float)
        coupling_delay = pd.to_numeric(group_df["coupling_delay_ms"], errors="coerce").to_numpy(dtype=float)
        valid_w = np.isfinite(weights) & (weights >= 0)
        packet_coupling_rows.append(
            {
                "packet_idx": int(pkt_idx),
                "coupling_strength__packet": _safe_weighted_mean(coupling_strength[valid_w], weights[valid_w]),
                "coupling_delay_ms__packet": _safe_weighted_mean(coupling_delay[valid_w], weights[valid_w]),
            }
        )

    packet_summary = pd.DataFrame(packet_rows).merge(pd.DataFrame(packet_coupling_rows), on="packet_idx", how="left")
    return feature_df, neuron_summary, packet_summary


def _resolve_feature_name(feature_name):
    key = str(feature_name).strip().lower().replace("-", "_")
    key = key.replace("__", "_")
    return PACKET_FEATURE_ALIASES.get(key, key)


def _feature_label(feature_name, normalized=False):
    feat = _resolve_feature_name(feature_name)
    label = PACKET_FEATURE_LABELS.get(feat, feat)
    if normalized and feat in PACKET_SIZE_SCALED_FEATURES:
        return f"{label} / packet size"
    return label


def _display_feature_label(feature_name, normalized=False):
    feat = _resolve_feature_name(feature_name)
    label = _feature_label(feat, normalized=normalized)
    if feat in PACKET_TIME_FEATURES:
        return f"{label} (ms)"
    return label


def _display_feature_values(feature_name, values):
    feat = _resolve_feature_name(feature_name)
    arr = np.asarray(values, dtype=float)
    if feat in PACKET_TIME_FEATURES:
        return arr * 1000.0
    return arr


def _feature_summary_col(feature_name, normalized=False):
    feat = _resolve_feature_name(feature_name)
    return f"{feat}__packet_norm" if normalized else f"{feat}__packet_raw"


def _feature_neuron_var_col(feature_name):
    feat = _resolve_feature_name(feature_name)
    return f"{feat}__var"


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
        column_labels.extend([f"{_feature_label(feat, normalized=normalized)} | unit {int(cid)}" for cid in cluster_ids])
    if not matrices:
        return np.zeros((len(packet_ids), 0), dtype=float), []
    return np.concatenate(matrices, axis=1), column_labels


def _standardize_matrix(matrix):
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
    scores = u[:, :max_components] * s[:max_components]
    total_var = np.sum(s ** 2)
    explained_ratio = (s[:max_components] ** 2) / total_var if total_var > 0 else np.zeros(max_components, dtype=float)
    return {
        "scores": scores.astype(np.float32),
        "components": vt[:max_components].astype(np.float32),
        "explained_ratio": explained_ratio.astype(np.float32),
    }


def _residualize_matrix_against_scalar(matrix, scalar):
    x = np.asarray(matrix, dtype=float)
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    design = np.column_stack([np.ones(scalar.shape[0], dtype=float), scalar])
    coef, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def _transform_packet_size(packet_size):
    packet_size = np.asarray(packet_size, dtype=float)
    packet_size = np.where(np.isfinite(packet_size), np.maximum(packet_size, 0.0), 0.0)
    return np.log1p(packet_size)


def _filter_column_labels(column_labels, keep_mask):
    return [label for label, keep in zip(column_labels, np.asarray(keep_mask, dtype=bool)) if keep]


def prepare_feature_cluster_inputs(feature_df, packet_summary_df, packet_dataset, feature_names):
    raw_matrix, raw_columns = _build_packet_feature_matrix(feature_df, packet_dataset, feature_names, normalized=False)
    normalized_matrix, normalized_columns = _build_packet_feature_matrix(feature_df, packet_dataset, feature_names, normalized=True)
    raw_std, raw_keep = _standardize_matrix(raw_matrix)
    normalized_std, normalized_keep = _standardize_matrix(normalized_matrix)
    packet_size = pd.to_numeric(packet_summary_df["packet_size"], errors="coerce").to_numpy(dtype=float)
    size_covariate = _transform_packet_size(packet_size)

    raw_labels = _filter_column_labels(raw_columns, raw_keep)
    normalized_labels = _filter_column_labels(normalized_columns, normalized_keep)

    residual_base = _residualize_matrix_against_scalar(raw_std, size_covariate)
    residual_std, residual_keep = _standardize_matrix(residual_base)
    residual_labels = [
        f"{label} | size residual"
        for label, keep in zip(raw_labels, np.asarray(residual_keep, dtype=bool))
        if keep
    ]

    by_shape = {
        "raw": {
            "shape_mode": "raw",
            "cluster_matrix": raw_std.astype(np.float32),
            "column_labels": raw_labels,
            "feature_space": "raw_standardized",
        },
        "normalized": {
            "shape_mode": "normalized",
            "cluster_matrix": normalized_std.astype(np.float32),
            "column_labels": normalized_labels,
            "feature_space": "normalized_standardized",
        },
        "residual": {
            "shape_mode": "residual",
            "cluster_matrix": residual_std.astype(np.float32),
            "column_labels": residual_labels,
            "feature_space": "raw_size_residual_standardized",
        },
    }
    return {
        "cluster_matrix": by_shape["raw"]["cluster_matrix"],
        "column_labels": list(by_shape["raw"]["column_labels"]),
        "packet_size": packet_size.astype(np.float32),
        "size_covariate": size_covariate.astype(np.float32),
        "feature_space": by_shape["raw"]["feature_space"],
        "by_shape": by_shape,
    }


def _resolve_cluster_method_spec(cluster_method):
    method_key = str(cluster_method).strip().lower()
    method_key = PACKET_CLUSTER_METHOD_ALIASES.get(method_key, method_key)
    spec = PACKET_CLUSTER_METHOD_SPECS.get(method_key)
    if spec is None:
        raise ValueError(f"Unsupported packet clustering method: {cluster_method}")
    return method_key, dict(spec)


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


def _euclidean_distance_matrix(x):
    sq = np.sum(x ** 2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (x @ x.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2, dtype=float)


def _silhouette_score_numpy(x, labels):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=int)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= x.shape[0]:
        return np.nan
    dist = _euclidean_distance_matrix(x)
    s_vals = np.full(x.shape[0], np.nan, dtype=float)
    for idx in range(x.shape[0]):
        same_mask = labels == labels[idx]
        same_mask[idx] = False
        a_val = float(np.mean(dist[idx, same_mask])) if np.any(same_mask) else 0.0
        b_val = np.nan
        for other in unique_labels:
            if other == labels[idx]:
                continue
            other_mask = labels == other
            if np.any(other_mask):
                other_mean = float(np.mean(dist[idx, other_mask]))
                b_val = other_mean if not np.isfinite(b_val) else min(b_val, other_mean)
        denom = max(a_val, b_val) if np.isfinite(b_val) else 0.0
        s_vals[idx] = (b_val - a_val) / denom if denom > 0 else 0.0
    valid = np.isfinite(s_vals)
    return float(np.mean(s_vals[valid])) if np.any(valid) else np.nan


def choose_cluster_count(cluster_matrix, k_min=2, k_max=6, n_init=20, max_iter=100, random_state=0):
    x = np.asarray(cluster_matrix, dtype=float)
    n_samples = x.shape[0]
    if n_samples < 2 or x.shape[1] == 0:
        return None, pd.DataFrame(columns=["k", "silhouette"])
    if n_samples == 2:
        return 2, pd.DataFrame([{"k": 2, "silhouette": np.nan}])
    k_upper = min(int(k_max), n_samples - 1)
    k_lower = max(2, int(k_min))
    if k_upper < k_lower:
        return None, pd.DataFrame(columns=["k", "silhouette"])
    eval_rows = []
    rng = np.random.default_rng(int(random_state))
    if n_samples > 400:
        eval_idx = np.sort(rng.choice(n_samples, size=400, replace=False))
        x_eval = x[eval_idx]
    else:
        x_eval = x
    for k in range(k_lower, k_upper + 1):
        labels, _centroids = _kmeans_numpy(x_eval, k, n_init=n_init, max_iter=max_iter, random_state=random_state)
        silhouette = _silhouette_score_numpy(x_eval, labels)
        eval_rows.append({"k": int(k), "silhouette": float(silhouette) if np.isfinite(silhouette) else np.nan})
    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        return None, eval_df
    eval_df["score_for_pick"] = eval_df["silhouette"].fillna(-np.inf)
    best_row = eval_df.sort_values(["score_for_pick", "k"], ascending=[False, True]).iloc[0]
    return int(best_row["k"]), eval_df.drop(columns=["score_for_pick"])


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


def _resolve_target_regions(region_names, target_region):
    ordered_regions = [str(region).strip() for region in region_names if str(region).strip()]
    ordered_regions = list(dict.fromkeys(ordered_regions))
    target_region = None if target_region is None else str(target_region).strip()
    if not target_region:
        return ordered_regions

    exact_matches = [region for region in ordered_regions if region == target_region]
    if exact_matches:
        return exact_matches

    prefix_matches = [region for region in ordered_regions if region.startswith(target_region)]
    if prefix_matches:
        return prefix_matches

    raise RuntimeError(
        f"Requested TARGET_REGION `{target_region}` was not found in packet regions: {ordered_regions}"
    )


def _packet_context_color_map():
    return {
        "task": "#1f77b4",
        "spont": "#ff7f0e",
        "iti": "#2ca02c",
        "other": "#7f7f7f",
        "overlap": "#9467bd",
        "spont_whisking": "#bcbd22",
        "spont_non_whisking": "#8c564b",
    }


def _show_region_pca_selection_figures(region, packet_summary_df, feature_embeddings_pca, shape_modes):
    import matplotlib.pyplot as plt

    context_values = packet_summary_df.get("context_split", packet_summary_df.get("packet_context", pd.Series(dtype=object)))
    context_values = np.asarray(pd.Series(context_values).astype(str), dtype=object)
    color_map = _packet_context_color_map()
    figures = {}

    for shape_mode in shape_modes:
        pca_result = feature_embeddings_pca.get(shape_mode)
        if pca_result is None:
            continue
        scores = np.asarray(pca_result["scores"], dtype=float)
        explained = np.asarray(pca_result["explained_ratio"], dtype=float)
        if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] == 0:
            continue

        x_vals = scores[:, 0]
        y_vals = scores[:, 1] if scores.shape[1] >= 2 else np.zeros(scores.shape[0], dtype=float)
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        unique_contexts = [ctx for ctx in pd.unique(context_values).tolist() if isinstance(ctx, str) and ctx]
        if unique_contexts:
            for context_label in unique_contexts:
                mask = context_values == context_label
                if not np.any(mask):
                    continue
                ax.scatter(
                    x_vals[mask],
                    y_vals[mask],
                    s=26,
                    alpha=0.8,
                    label=context_label,
                    color=color_map.get(context_label, "#7f7f7f"),
                    edgecolors="none",
                )
            ax.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax.scatter(x_vals, y_vals, s=26, alpha=0.8, color="#1f77b4", edgecolors="none")

        pc1_label = f"PC1 ({100.0 * explained[0]:.1f}% var)" if explained.size >= 1 else "PC1"
        pc2_label = f"PC2 ({100.0 * explained[1]:.1f}% var)" if explained.size >= 2 else "PC2"
        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.set_title(f"Region {region} | {shape_mode} PCA feature space | packets={scores.shape[0]}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        figures[shape_mode] = fig

    if figures:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
    return plt, figures


def _prompt_manual_pca_cluster_counts(region, packet_summary_df, feature_embeddings_pca, packet_config, requested_methods):
    mode = str(packet_config.get("PCA_CLUSTER_K_SELECTION", "auto")).strip().lower()
    if mode not in {"prompt", "interactive", "manual_prompt"}:
        return {}

    requested_methods = {
        PACKET_CLUSTER_METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
        for method in requested_methods
    }
    shape_modes = [
        shape_mode
        for shape_mode in ("raw", "normalized", "residual")
        if f"{shape_mode}_pca_kmeans" in requested_methods
    ]
    if not shape_modes:
        return {}

    plt, figures = _show_region_pca_selection_figures(
        region,
        packet_summary_df,
        feature_embeddings_pca,
        shape_modes,
    )
    manual_counts = {}
    try:
        for shape_mode in shape_modes:
            pca_result = feature_embeddings_pca.get(shape_mode)
            if pca_result is None:
                continue
            scores = np.asarray(pca_result["scores"], dtype=float)
            if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] == 0:
                continue

            k_upper = min(int(packet_config.get("CLUSTER_MAX_K", 6)), int(scores.shape[0]))
            k_lower = max(1, int(packet_config.get("CLUSTER_MIN_K", 2)))
            k_lower = min(k_lower, k_upper)
            auto_k, _k_eval_df = choose_cluster_count(
                scores,
                k_min=max(2, k_lower),
                k_max=max(2, k_upper),
                n_init=int(packet_config.get("KMEANS_N_INIT", 20)),
                max_iter=int(packet_config.get("KMEANS_MAX_ITER", 100)),
                random_state=0,
            )
            default_k = int(auto_k) if auto_k is not None else int(k_lower)

            while True:
                response = input(
                    f"Region {region} | {shape_mode} PCA clusters [{k_lower}-{k_upper}, Enter={default_k}]: "
                ).strip()
                if response == "":
                    chosen_k = default_k
                    break
                if response.isdigit():
                    chosen_k = int(response)
                    if k_lower <= chosen_k <= k_upper:
                        break
                print(f"Please enter an integer between {k_lower} and {k_upper}.")

            manual_counts[f"{shape_mode}_pca_kmeans"] = int(chosen_k)
            print(f"Using {chosen_k} clusters for region {region} | {shape_mode} PCA clustering.")
    finally:
        if plt is not None:
            for fig in figures.values():
                plt.close(fig)

    return manual_counts


def annotate_packet_periods(packet_dataset, trials, wh_detect, min_overlap_s=0.05):
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
    context_split = packet_context.astype(object).copy()
    spont_mask = packet_context == "spont"
    context_split[spont_mask & whisk_mask] = "spont_whisking"
    context_split[spont_mask & ~whisk_mask] = "spont_non_whisking"

    exclusive_labels = np.full(n_packets, "non_whisking", dtype=object)
    exclusive_labels[whisk_only_mask] = "whisking"
    exclusive_labels[first_move_mask] = "first_move"
    exclusive_labels[feedback_mask] = "feedback"
    exclusive_labels[stim_mask] = "stim_on"

    return {
        "context_split": context_split,
        "exclusive_labels": exclusive_labels,
        "whisking_mask": whisk_mask,
        "stim_on_mask": stim_mask,
        "first_move_mask": first_move_mask,
        "feedback_mask": feedback_mask,
    }


def compute_feature_correlations(neuron_summary_df):
    rows = []
    for feat in MAIN_PACKET_FEATURES:
        feat_vals = pd.to_numeric(neuron_summary_df[feat], errors="coerce").to_numpy(dtype=float)
        for target in ("coupling_strength", "coupling_delay_ms"):
            target_vals = pd.to_numeric(neuron_summary_df[target], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(feat_vals) & np.isfinite(target_vals)
            if int(np.sum(valid)) < 2:
                pearson_r = np.nan
                spearman_r = np.nan
            else:
                pearson_r = float(np.corrcoef(feat_vals[valid], target_vals[valid])[0, 1])
                x_rank = pd.Series(feat_vals[valid]).rank(method="average").to_numpy(dtype=float)
                y_rank = pd.Series(target_vals[valid]).rank(method="average").to_numpy(dtype=float)
                spearman_r = float(np.corrcoef(x_rank, y_rank)[0, 1])
            rows.append(
                {
                    "feature": feat,
                    "feature_label": _display_feature_label(feat),
                    "target": target,
                    "target_label": PACKET_FEATURE_LABELS.get(target, target),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "n_valid": int(np.sum(valid)),
                }
            )
    return pd.DataFrame(rows)


def compute_region_packet_bundle(
    region,
    packet_dataset,
    df_tpl_for_merge,
    spike_times_by_cluster,
    packet_config,
    trials,
    wh_detect,
    wh_events_by_period,
    config_plot,
):
    feature_df, neuron_summary_df, packet_summary_df = compute_main_packet_feature_tables(
        packet_dataset,
        df_tpl_for_merge,
        spike_times_by_cluster,
        context_mode="all",
    )
    if feature_df.empty or neuron_summary_df.empty or packet_summary_df.empty:
        raise RuntimeError(f"No packet feature tables could be built for region {region}.")

    period_info = annotate_packet_periods(
        packet_dataset,
        trials,
        wh_detect,
        min_overlap_s=float(packet_config["PACKET_PERIOD_MIN_OVERLAP_S"]),
    )
    packet_summary_df = packet_summary_df.copy()
    packet_summary_df["context_split"] = period_info["context_split"]
    packet_summary_df["period_exclusive_label"] = period_info["exclusive_labels"]
    packet_summary_df["period_whisking"] = period_info["whisking_mask"]
    packet_summary_df["period_stim_on"] = period_info["stim_on_mask"]
    packet_summary_df["period_first_move"] = period_info["first_move_mask"]
    packet_summary_df["period_feedback"] = period_info["feedback_mask"]

    packet_size = pd.to_numeric(packet_summary_df["packet_size"], errors="coerce").to_numpy(dtype=float)
    packet_tensor_raw = np.asarray(packet_dataset["packet_tensor"], dtype=float)
    denom = np.where(packet_size > 0, packet_size, 1.0)
    packet_tensor_normalized = (packet_tensor_raw / denom[:, None, None]).astype(np.float32)

    cluster_inputs = prepare_feature_cluster_inputs(
        feature_df,
        packet_summary_df,
        packet_dataset,
        MAIN_PACKET_FEATURES,
    )
    feature_embeddings_pca = {}
    for shape_mode, shape_input in cluster_inputs["by_shape"].items():
        shape_matrix = np.asarray(shape_input["cluster_matrix"], dtype=float)
        if shape_matrix.shape[0] > 0 and shape_matrix.shape[1] > 0:
            feature_embeddings_pca[shape_mode] = _compute_matrix_pca(
                shape_matrix,
                n_components=int(packet_config["PCA_COMPONENTS"]),
            )
        else:
            feature_embeddings_pca[shape_mode] = None

    requested_cluster_methods = list(packet_config.get("CLUSTER_METHODS", DEFAULT_PACKET_CONFIG["CLUSTER_METHODS"]))
    manual_pca_cluster_counts = _prompt_manual_pca_cluster_counts(
        region,
        packet_summary_df,
        feature_embeddings_pca,
        packet_config,
        requested_cluster_methods,
    )

    cluster_results = {}
    summary_row = {"region": region, "n_packets": int(len(packet_dataset["packet_times"]))}
    for cluster_method in requested_cluster_methods:
        method_key, method_spec = _resolve_cluster_method_spec(cluster_method)
        cluster_results[method_key] = None
        summary_row[f"n_clusters_{method_key}"] = 0

        shape_mode = method_spec["shape_mode"]
        shape_input = cluster_inputs["by_shape"].get(shape_mode)
        if shape_input is None:
            continue
        shape_matrix = np.asarray(shape_input["cluster_matrix"], dtype=float)
        if shape_matrix.shape[0] < 2 or shape_matrix.shape[1] == 0:
            continue

        shape_pca = feature_embeddings_pca.get(shape_mode)
        if bool(method_spec["use_pca"]):
            if shape_pca is None:
                continue
            cluster_space = np.asarray(shape_pca["scores"], dtype=float)
            cluster_space_name = "PCA"
        else:
            cluster_space = shape_matrix
            cluster_space_name = "All features"
        if cluster_space.ndim != 2 or cluster_space.shape[0] < 2 or cluster_space.shape[1] == 0:
            continue

        auto_k, k_eval_df = choose_cluster_count(
            cluster_space,
            k_min=int(packet_config["CLUSTER_MIN_K"]),
            k_max=int(packet_config["CLUSTER_MAX_K"]),
            n_init=int(packet_config["KMEANS_N_INIT"]),
            max_iter=int(packet_config["KMEANS_MAX_ITER"]),
            random_state=0,
        )
        if auto_k is None:
            continue

        manual_k = manual_pca_cluster_counts.get(method_key)
        chosen_k = int(manual_k) if manual_k is not None else int(auto_k)

        cluster_labels, cluster_centroids = _kmeans_numpy(
            cluster_space,
            n_clusters=chosen_k,
            n_init=int(packet_config["KMEANS_N_INIT"]),
            max_iter=int(packet_config["KMEANS_MAX_ITER"]),
            random_state=0,
        )

        packet_plot_df = packet_summary_df.copy()
        packet_plot_df["cluster_label"] = cluster_labels
        if shape_pca is not None:
            pca_scores = np.asarray(shape_pca["scores"], dtype=float)
            packet_plot_df["pc1"] = (
                pca_scores[:, 0] if pca_scores.shape[1] >= 1 else np.full(cluster_space.shape[0], np.nan, dtype=float)
            )
            packet_plot_df["pc2"] = (
                pca_scores[:, 1] if pca_scores.shape[1] >= 2 else np.full(cluster_space.shape[0], np.nan, dtype=float)
            )
        else:
            packet_plot_df["pc1"] = np.full(cluster_space.shape[0], np.nan, dtype=float)
            packet_plot_df["pc2"] = np.full(cluster_space.shape[0], np.nan, dtype=float)

        cluster_results[method_key] = {
            "method": method_key,
            "shape_mode": shape_mode,
            "cluster_space_name": cluster_space_name,
            "n_clusters": int(chosen_k),
            "noise_count": 0,
            "k_selection": k_eval_df.copy(),
            "cluster_count_source": "manual" if manual_k is not None else "auto",
            "suggested_k": int(auto_k),
            "cluster_labels": np.asarray(cluster_labels, dtype=int),
            "cluster_centroids": np.asarray(cluster_centroids, dtype=np.float32),
            "embedding": (
                {
                    "name": "PCA",
                    "scores": np.asarray(shape_pca["scores"], dtype=np.float32),
                    "explained_ratio": np.asarray(shape_pca["explained_ratio"], dtype=np.float32),
                }
                if shape_pca is not None
                else None
            ),
            "packet_plot_df": packet_plot_df,
            "packet_psth": precompute_packet_psth_data(
                np.asarray(packet_dataset["packet_times"], dtype=float),
                np.asarray(cluster_labels, dtype=int),
                trials,
                wh_events_by_period,
                packet_config.get("WHISK_EVENT_CONTEXT", "all"),
                config_plot,
            ),
        }
        summary_row[f"n_clusters_{method_key}"] = int(chosen_k)

    return {
        "packet_dataset": packet_dataset,
        "feature_df": feature_df,
        "neuron_summary_df": neuron_summary_df,
        "packet_summary_df": packet_summary_df,
        "feature_correlation_df": compute_feature_correlations(neuron_summary_df),
        "cluster_input": {
            "cluster_matrix": cluster_inputs["cluster_matrix"].astype(np.float32),
            "column_labels": list(cluster_inputs["column_labels"]),
            "packet_size": cluster_inputs["packet_size"].astype(np.float32),
            "size_covariate": cluster_inputs["size_covariate"].astype(np.float32),
            "feature_space": cluster_inputs["feature_space"],
        },
        "cluster_inputs": cluster_inputs["by_shape"],
        "feature_embedding_pca": feature_embeddings_pca.get("raw"),
        "feature_embeddings_pca": feature_embeddings_pca,
        "packet_tensor_normalized": packet_tensor_normalized,
        "cluster_results": cluster_results,
        "summary_row": summary_row,
    }


def compute_packet_dashboard_cache(pid, packet_config=None, packet_cache_dir=PACKET_CACHE_DIR):
    packet_config = {
        **DEFAULT_PACKET_CONFIG,
        **({} if packet_config is None else dict(packet_config)),
    }
    base_cache = load_base_cache(pid)

    spikes = None
    clusters = None
    session = None
    raw_errors = []
    for mode in ("local", "remote"):
        try:
            spikes, clusters, session, _ssl = load_raw_session(
                pid,
                load_wheel=False,
                load_pose=False,
                load_motion_energy=False,
                mode=mode,
            )
            break
        except Exception as exc:
            raw_errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    if spikes is None or clusters is None or session is None:
        raise RuntimeError("Raw session load failed.\n" + "\n".join(raw_errors))

    cluster_ids = base_cache.get("cluster_ids")
    cluster_acronyms = base_cache.get("cluster_acronyms_plot")
    if cluster_ids is None and clusters is not None:
        cluster_ids, _ = build_cluster_id_map(clusters)
    if cluster_acronyms is None and clusters is not None:
        if hasattr(clusters, "acronym"):
            cluster_acronyms = np.asarray(clusters.acronym)
        elif isinstance(clusters, dict) and "acronym" in clusters:
            cluster_acronyms = np.asarray(clusters.get("acronym"))

    config_calc = dict(base_cache.get("config_calc", {}))
    config_plot = dict(base_cache.get("config_plot", {}))
    meta = dict(base_cache.get("meta", {}))
    trials = getattr(session, "trials", None) if session is not None else None
    if trials is None:
        trials = base_cache.get("trials")
    if trials is None:
        raise RuntimeError("Trials are not available for packet computation.")

    df_units = _prepare_units_df(cluster_ids, cluster_acronyms, clusters, packet_config["LABEL_MIN"])
    if df_units.empty:
        raise RuntimeError("No units available after label filtering.")
    df_units_sorted, region_order_sorted, sort_label = _sort_units_df(
        df_units,
        base_cache.get("df_res"),
        base_cache.get("df_coupling"),
        base_cache.get("df_coupling_task"),
        base_cache.get("df_coupling_iti"),
        packet_config["SORT_METRIC_KEY"],
    )
    region_order = df_units["acronym"].dropna().astype(str).unique().tolist()
    selected_regions = _resolve_target_regions(
        list(region_order_sorted) + list(region_order),
        packet_config.get("TARGET_REGION"),
    )
    selected_region_set = set(selected_regions)
    region_order_sorted = [region for region in region_order_sorted if str(region) in selected_region_set]
    region_order = [region for region in region_order if str(region) in selected_region_set]
    if not region_order or not region_order_sorted:
        raise RuntimeError("No packet regions remain after applying TARGET_REGION filtering.")

    df_tpl_for_merge, spont_intervals, task_windows, iti_windows = build_template_source_table(
        spikes,
        clusters,
        cluster_ids,
        cluster_acronyms,
        config_calc,
        packet_config,
        meta,
        trials,
    )
    templates_a, templates_b, strength_a, strength_b = build_template_lookup(
        df_tpl_for_merge,
        float(config_calc.get("STPR_BIN_SIZE", 0.001)),
        float(packet_config["DETECTION_BIN_SIZE"]),
        time_scale=float(packet_config["TEMPLATE_TIME_SCALE"]),
        use_split=bool(packet_config["USE_SPLIT_TEMPLATES"]),
    )

    whisk_bundle = build_whisk_bundle(base_cache, config_calc, session, spont_intervals, task_windows, iti_windows)
    wh_detect = whisk_bundle["wh_detect"]
    spike_times_by_cluster, _spike_times_all, _spike_clusters_all = build_spike_index(
        spikes,
        df_units["cluster_id"].to_numpy(dtype=int),
    )

    region_detection, kernel = detect_packets_by_region(
        df_units,
        region_order,
        spike_times_by_cluster,
        templates_a,
        templates_b,
        strength_a,
        strength_b,
        packet_config,
    )

    region_results = {}
    region_summary_rows = []
    for region in region_order_sorted:
        if region not in region_detection or len(region_detection[region]["packet_times"]) == 0:
            continue
        packet_dataset = extract_region_packet_dataset(
            region,
            region_detection,
            df_units_sorted,
            spike_times_by_cluster,
            kernel,
            templates_a,
            templates_b,
            packet_config,
            task_windows,
            spont_intervals,
            iti_windows,
        )
        region_bundle = compute_region_packet_bundle(
            region,
            packet_dataset,
            df_tpl_for_merge,
            spike_times_by_cluster,
            packet_config,
            trials,
            wh_detect,
            whisk_bundle["wh_events_by_period"],
            config_plot,
        )
        region_results[region] = region_bundle
        region_summary_rows.append(region_bundle["summary_row"])

    region_summary_df = pd.DataFrame(region_summary_rows).sort_values("region").reset_index(drop=True)
    packet_cache = {
        "packet_dashboard_version": PACKET_DASHBOARD_VERSION,
        "pid": pid,
        "meta": meta,
        "config_calc": config_calc,
        "config_plot": config_plot,
        "packet_config": packet_config,
        "cluster_ids": np.asarray(cluster_ids, dtype=int),
        "cluster_acronyms_plot": np.asarray(cluster_acronyms).astype(str),
        "sort_label": sort_label,
        "region_order": list(region_order_sorted),
        "region_summary_df": region_summary_df,
        "template_source_df": df_tpl_for_merge,
        "wh_detect": whisk_bundle["wh_detect"],
        "wh_event_base": whisk_bundle["wh_event_base"],
        "wh_events_by_period": whisk_bundle["wh_events_by_period"],
        "region_results": region_results,
    }
    save_packet_cache(pid, packet_cache, cache_dir=packet_cache_dir)
    return packet_cache


def cluster_palette():
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


def _cluster_color(cluster_idx):
    if int(cluster_idx) < 0:
        return "#7f7f7f"
    palette = cluster_palette()
    return palette[int(cluster_idx) % len(palette)]


def _cluster_display_name(cluster_idx, prefix="Cluster"):
    cluster_idx = int(cluster_idx)
    if cluster_idx < 0:
        return "Noise"
    return f"{prefix} {cluster_idx + 1}"


def _sorted_cluster_labels(labels):
    labels = [int(lbl) for lbl in np.unique(np.asarray(labels, dtype=int))]
    non_noise = sorted(lbl for lbl in labels if lbl >= 0)
    noise = [lbl for lbl in labels if lbl < 0]
    return non_noise + noise


def get_cluster_result(region_bundle, cluster_method=DEFAULT_PACKET_CLUSTER_METHOD):
    cluster_results = region_bundle.get("cluster_results", {})
    if not isinstance(cluster_results, dict) or not cluster_results:
        return None
    if cluster_method is None:
        for method_key in DEFAULT_PACKET_CONFIG["CLUSTER_METHODS"]:
            result = cluster_results.get(method_key)
            if result is not None:
                return result
        for result in cluster_results.values():
            if result is not None:
                return result
        return None
    method_key = PACKET_CLUSTER_METHOD_ALIASES.get(str(cluster_method).strip().lower(), str(cluster_method).strip().lower())
    return cluster_results.get(method_key)


def get_region_raster_y_bounds(
    region,
    clusters,
    cluster_ids,
    cluster_acronyms,
    config_plot,
    sort_metric_key,
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
):
    df_units = plotting_utils._prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        config_plot.get("PLOT_ONLY_GOOD_UNITS", False),
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )[0]
    if df_units.empty:
        return None, None
    df_units, sort_label = plotting_utils._merge_metric(
        df_units,
        sort_metric_key,
        df_res=df_res,
        df_coupling=df_coupling,
        df_coupling_task=df_coupling_task,
        df_coupling_iti=df_coupling_iti,
        df_firing_rate=None,
    )
    df_units, _region_order, _sort_label = plotting_utils._sort_within_regions(
        df_units,
        sort_label,
        metric_key=sort_metric_key,
    )
    group = df_units.loc[df_units["acronym"].astype(str) == str(region)]
    if group.empty:
        return None, None
    return float(group.index.min() - 0.5), float(group.index.max() + 0.5)


def add_packet_cluster_markers_to_raster(
    fig,
    packet_plot_df,
    t_start,
    t_end,
    cluster_col="cluster_label",
    packet_window_s=None,
    region_y0=None,
    region_y1=None,
):
    if fig is None or packet_plot_df is None or packet_plot_df.empty:
        return fig

    palette = cluster_palette()
    df_plot = packet_plot_df.copy()
    time_vals = pd.to_numeric(df_plot["packet_peak_time_s"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time_vals) & (time_vals >= float(t_start)) & (time_vals <= float(t_end))
    if not np.any(keep):
        return fig
    df_plot = df_plot.loc[keep].copy()
    if "packet_start_s" not in df_plot.columns or "packet_end_s" not in df_plot.columns:
        if packet_window_s is not None and np.isfinite(float(packet_window_s)) and float(packet_window_s) > 0:
            half_window_s = float(packet_window_s) / 2.0
            df_plot["packet_start_s"] = pd.to_numeric(df_plot["packet_peak_time_s"], errors="coerce") - half_window_s
            df_plot["packet_end_s"] = pd.to_numeric(df_plot["packet_peak_time_s"], errors="coerce") + half_window_s

    if region_y0 is None or region_y1 is None:
        try:
            yaxis = fig.layout.yaxis
            if getattr(yaxis, "range", None) is not None:
                region_y0 = min(float(yaxis.range[0]), float(yaxis.range[1]))
                region_y1 = max(float(yaxis.range[0]), float(yaxis.range[1]))
            else:
                raise ValueError("No y range")
        except Exception:
            region_y0 = -0.5
            region_y1 = float(len(df_plot)) - 0.5

    y_top = float(region_y1) - 0.2
    if "packet_start_s" in df_plot.columns and "packet_end_s" in df_plot.columns:
        for row in df_plot.itertuples(index=False):
            start_s = float(getattr(row, "packet_start_s", np.nan))
            end_s = float(getattr(row, "packet_end_s", np.nan))
            cluster_val = getattr(row, cluster_col, np.nan)
            if not (np.isfinite(start_s) and np.isfinite(end_s) and np.isfinite(cluster_val)):
                continue
            color = _cluster_color(int(cluster_val))
            fig.add_shape(
                type="rect",
                x0=start_s,
                x1=end_s,
                y0=float(region_y0),
                y1=float(region_y1),
                line=dict(color=color, width=2, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
                row=1,
                col=1,
            )
    for cluster_idx in _sorted_cluster_labels(df_plot[cluster_col].dropna().astype(int).to_numpy(dtype=int)):
        mask = pd.to_numeric(df_plot[cluster_col], errors="coerce").to_numpy(dtype=float) == float(cluster_idx)
        x_vals = pd.to_numeric(df_plot.loc[mask, "packet_peak_time_s"], errors="coerce").to_numpy(dtype=float)
        if x_vals.size == 0:
            continue
        customdata = df_plot.loc[mask, ["packet_idx", "packet_score", "packet_context"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.full(x_vals.shape, y_top, dtype=float),
                mode="markers",
                marker=dict(
                    color=_cluster_color(int(cluster_idx)),
                    symbol="triangle-down-open",
                    size=13,
                    line=dict(width=2),
                ),
                name=_cluster_display_name(int(cluster_idx), prefix="Packet Cluster"),
                legendgroup=f"packet-cluster-{int(cluster_idx)}",
                showlegend=True,
                customdata=customdata,
                hovertemplate=(
                    "Packet %{customdata[0]}<br>"
                    "Score: %{customdata[1]:.2f}<br>"
                    "Context: %{customdata[2]}<br>"
                    "Time: %{x:.3f}s<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    return fig


def build_packet_score_figure(region_bundle, t_start, t_end, cluster_method=DEFAULT_PACKET_CLUSTER_METHOD, template="plotly_white"):
    dataset = region_bundle["packet_dataset"]
    cluster_result = get_cluster_result(region_bundle, cluster_method)
    packet_plot_df = cluster_result["packet_plot_df"] if cluster_result is not None else region_bundle["packet_summary_df"]
    time_vals = np.asarray(dataset["detection_bin_centers"], dtype=float)
    score_vals = np.asarray(dataset["region_score_plot"], dtype=float)
    rate_vals = np.asarray(dataset["population_rate_hz"], dtype=float)
    keep = np.isfinite(time_vals) & (time_vals >= float(t_start)) & (time_vals <= float(t_end))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Packet Probability", "Population PSTH"),
    )
    fig.add_trace(
        go.Scatter(x=time_vals[keep], y=score_vals[keep], mode="lines", line=dict(color="#13866d", width=2), name="Packet score"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vals[keep], y=rate_vals[keep], mode="lines", line=dict(color="#13866d", width=2), name="Population PSTH", showlegend=False),
        row=2,
        col=1,
    )
    if packet_plot_df is not None and not packet_plot_df.empty:
        pkt_keep = (
            pd.to_numeric(packet_plot_df["packet_peak_time_s"], errors="coerce").to_numpy(dtype=float) >= float(t_start)
        ) & (
            pd.to_numeric(packet_plot_df["packet_peak_time_s"], errors="coerce").to_numpy(dtype=float) <= float(t_end)
        )
        pkt_df = packet_plot_df.loc[pkt_keep].copy()
        for cluster_idx in _sorted_cluster_labels(pkt_df["cluster_label"].dropna().astype(int).to_numpy(dtype=int)):
            mask = pd.to_numeric(pkt_df["cluster_label"], errors="coerce").to_numpy(dtype=float) == float(cluster_idx)
            x_vals = pd.to_numeric(pkt_df.loc[mask, "packet_peak_time_s"], errors="coerce").to_numpy(dtype=float)
            y_vals = pd.to_numeric(pkt_df.loc[mask, "packet_score"], errors="coerce").to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    marker=dict(color=_cluster_color(int(cluster_idx)), size=10, symbol="diamond"),
                    name=_cluster_display_name(int(cluster_idx)),
                    legendgroup=f"cluster-{int(cluster_idx)}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    fig.update_yaxes(title_text="Packet score (z)", row=1, col=1)
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time in session (s)", row=2, col=1)
    fig.update_layout(
        title=f"Region {dataset['region']} | {cluster_method} | Packet probability and population PSTH",
        template=template,
        height=620,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _row_zscore_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    std = np.nanstd(matrix, axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (matrix - mean) / std


def _shared_row_zscore_matrices(matrices):
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices if np.asarray(matrix).ndim == 2]
    if not matrices:
        return []
    combined = np.concatenate(matrices, axis=1)
    mean = np.nanmean(combined, axis=1, keepdims=True)
    std = np.nanstd(combined, axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return [(matrix - mean) / std for matrix in matrices]


def build_cluster_heatmap_figure(region_bundle, cluster_method=DEFAULT_PACKET_CLUSTER_METHOD, normalized_space=False, template="plotly_white"):
    dataset = region_bundle["packet_dataset"]
    cluster_result = get_cluster_result(region_bundle, cluster_method)
    if cluster_result is None:
        return go.Figure()
    cluster_labels = np.asarray(cluster_result["cluster_labels"], dtype=int)
    packet_tensor_space = (
        np.asarray(region_bundle["packet_tensor_normalized"], dtype=float)
        if normalized_space
        else np.asarray(dataset["packet_tensor"], dtype=float)
    )
    template_matrix = np.asarray(dataset["template_matrix"], dtype=float)
    cluster_ids = np.asarray(dataset["cluster_ids"], dtype=int)
    delay_lookup = (
        region_bundle["neuron_summary_df"][["cluster_id", "coupling_delay_ms"]]
        .drop_duplicates(subset=["cluster_id"])
        .set_index("cluster_id")["coupling_delay_ms"]
        .to_dict()
    )
    order_df = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(int),
            "orig_idx": np.arange(cluster_ids.size, dtype=int),
            "coupling_delay_ms": [delay_lookup.get(int(cid), np.nan) for cid in cluster_ids],
        }
    ).sort_values(["coupling_delay_ms", "orig_idx"], na_position="last")
    order_idx = order_df["orig_idx"].to_numpy(dtype=int)
    template_matrix = template_matrix[order_idx]
    packet_tensor_space = packet_tensor_space[:, order_idx, :]
    rel_bin_centers_ms = np.asarray(dataset["rel_bin_centers"], dtype=float) * 1000.0
    unique_clusters = _sorted_cluster_labels(cluster_labels)

    matrices = [_row_zscore_matrix(template_matrix)]
    subplot_titles = ["Detection template"]
    cluster_means = []
    cluster_sizes = []
    for cluster_idx in unique_clusters:
        members = packet_tensor_space[cluster_labels == cluster_idx]
        cluster_mean = np.nanmean(members, axis=0) if members.size > 0 else np.zeros_like(template_matrix)
        cluster_means.append(cluster_mean)
        cluster_sizes.append(int(members.shape[0]))
    for cluster_idx, cluster_mean_z, n_members in zip(
        unique_clusters,
        _shared_row_zscore_matrices(cluster_means),
        cluster_sizes,
    ):
        matrices.append(cluster_mean_z)
        subplot_titles.append(f"{_cluster_display_name(int(cluster_idx))} mean packet (n={n_members})")

    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices if mat.size > 0])
    cmax = np.nanpercentile(np.abs(finite_vals), 98) if finite_vals.size > 0 else 1.0
    cmax = max(float(cmax), 1.0)
    fig = make_subplots(
        rows=len(matrices),
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.05,
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
        title=(
            f"Region {dataset['region']} | {cluster_method} clustering | "
            f"{'Size-normalized' if normalized_space else 'Raw'} packet heatmaps"
        ),
        template=template,
        height=280 * len(matrices) + 120,
        margin=dict(l=70, r=40, t=90, b=60),
    )
    return fig


def build_feature_correlation_figure(region_bundle, template="plotly_white"):
    corr_df = region_bundle["feature_correlation_df"].copy()
    if corr_df.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coupling strength", "Coupling delay"),
        horizontal_spacing=0.18,
    )
    for col_idx, target in enumerate(("coupling_strength", "coupling_delay_ms"), start=1):
        sub = corr_df.loc[corr_df["target"] == target].sort_values("spearman_r", key=lambda s: s.abs(), ascending=True)
        fig.add_trace(
            go.Bar(
                x=sub["spearman_r"].to_numpy(dtype=float),
                y=sub["feature_label"].astype(str).tolist(),
                orientation="h",
                marker=dict(color="#13866d"),
                customdata=sub[["pearson_r", "n_valid"]].to_numpy(),
                hovertemplate="Spearman=%{x:.2f}<br>Pearson=%{customdata[0]:.2f}<br>n=%{customdata[1]}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Spearman r", row=1, col=col_idx)
    fig.update_layout(
        title=f"Region {region_bundle['packet_dataset']['region']} | Main packet features vs coupling",
        template=template,
        height=520,
        margin=dict(l=90, r=40, t=90, b=60),
    )
    return fig


def _embedding_color_map(mode):
    if mode == "context":
        return {
            "task": ("Task", "#1f77b4"),
            "spont": ("Spont", "#ff7f0e"),
            "iti": ("ITI", "#2ca02c"),
            "other": ("Other", "#7f7f7f"),
            "overlap": ("Overlap", "#9467bd"),
            "spont_whisking": ("Spont Whisking", "#ff7f0e"),
            "spont_non_whisking": ("Spont Non-whisking", "#8c564b"),
        }
    return {
        "stim_on": ("Stim On", "#1f77b4"),
        "feedback": ("Feedback", "#d62728"),
        "first_move": ("First Move", "#2ca02c"),
        "whisking": ("Whisking", "#ff7f0e"),
        "non_whisking": ("Non-whisking", "#7f7f7f"),
    }


def build_embedding_figure(packet_plot_df, x_vals, y_vals, color_by="cluster", title="", x_label="PC1", y_label="PC2", template="plotly_white"):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    fig = go.Figure()
    if color_by == "cluster":
        for cluster_idx in _sorted_cluster_labels(packet_plot_df["cluster_label"].dropna().astype(int).to_numpy(dtype=int)):
            mask = pd.to_numeric(packet_plot_df["cluster_label"], errors="coerce").to_numpy(dtype=float) == float(cluster_idx)
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask],
                    y=y_vals[mask],
                    mode="markers",
                    marker=dict(color=_cluster_color(int(cluster_idx)), size=9, opacity=0.88),
                    name=f"{_cluster_display_name(int(cluster_idx))} (n={int(np.sum(mask))})",
                    customdata=packet_plot_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy(),
                    hovertemplate="Packet %{customdata[0]}<br>Peak time %{customdata[1]:.3f}s<br>Context %{customdata[2]}<br>X %{x:.2f}<br>Y %{y:.2f}<extra></extra>",
                )
            )
    else:
        col_name = "packet_context" if color_by == "context" else "period_exclusive_label"
        style_map = _embedding_color_map("context" if color_by == "context" else "period")
        labels = packet_plot_df[col_name].astype(str).to_numpy()
        for key, (label, color) in style_map.items():
            mask = labels == key
            if not np.any(mask):
                continue
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask],
                    y=y_vals[mask],
                    mode="markers",
                    marker=dict(color=color, size=9, opacity=0.86),
                    name=f"{label} (n={int(np.sum(mask))})",
                    customdata=packet_plot_df.loc[mask, ["packet_idx", "packet_peak_time_s", col_name]].to_numpy(),
                    hovertemplate="Packet %{customdata[0]}<br>Peak time %{customdata[1]:.3f}s<br>Label %{customdata[2]}<br>X %{x:.2f}<br>Y %{y:.2f}<extra></extra>",
                )
            )
    fig.update_layout(
        title=title,
        template=template,
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def get_neuron_scatter_options():
    return ["coupling_strength", "coupling_delay_ms"] + list(MAIN_PACKET_FEATURES)


def get_packet_scatter_options():
    return ["pc1", "pc2", "packet_score", "packet_size"] + list(MAIN_PACKET_FEATURES)


def build_neuron_scatter(region_bundle, x_var, y_var, template="plotly_white"):
    df_plot = region_bundle["neuron_summary_df"].copy()
    x_var = _resolve_feature_name(x_var)
    y_var = _resolve_feature_name(y_var)
    x_vals = pd.to_numeric(df_plot[x_var], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df_plot[y_var], errors="coerce").to_numpy(dtype=float)
    if x_var in MAIN_PACKET_FEATURES:
        x_vals = _display_feature_values(x_var, x_vals)
    if y_var in MAIN_PACKET_FEATURES:
        y_vals = _display_feature_values(y_var, y_vals)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals[valid],
            y=y_vals[valid],
            mode="markers",
            marker=dict(color="#13866d", size=8, opacity=0.82),
            customdata=df_plot.loc[valid, ["cluster_id", "region"]].to_numpy(),
            hovertemplate="Cluster %{customdata[0]}<br>Region %{customdata[1]}<br>X %{x:.3f}<br>Y %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Region {region_bundle['packet_dataset']['region']} | Neuron scatter",
        template=template,
        height=560,
        margin=dict(l=70, r=40, t=90, b=60),
    )
    fig.update_xaxes(title_text=_display_feature_label(x_var))
    fig.update_yaxes(title_text=_display_feature_label(y_var))
    return fig


def build_packet_scatter(region_bundle, cluster_method, x_var, y_var, color_by="cluster", template="plotly_white"):
    cluster_result = get_cluster_result(region_bundle, cluster_method)
    if cluster_result is None:
        return go.Figure()
    df_plot = cluster_result["packet_plot_df"].copy()
    x_var = _resolve_feature_name(x_var)
    y_var = _resolve_feature_name(y_var)
    x_vals = pd.to_numeric(df_plot[x_var], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df_plot[y_var], errors="coerce").to_numpy(dtype=float)
    if x_var in MAIN_PACKET_FEATURES:
        x_vals = _display_feature_values(x_var, x_vals)
    if y_var in MAIN_PACKET_FEATURES:
        y_vals = _display_feature_values(y_var, y_vals)
    title = f"Region {region_bundle['packet_dataset']['region']} | {cluster_method} | Packet scatter"
    return build_embedding_figure(
        df_plot,
        x_vals,
        y_vals,
        color_by=color_by,
        title=title,
        x_label=_display_feature_label(x_var),
        y_label=_display_feature_label(y_var),
        template=template,
    )


def choose_default_neuron_id(region_bundle):
    feature_df = region_bundle["feature_df"].copy()
    variance_rows = []
    for cluster_id, group_df in feature_df.groupby("cluster_id", sort=False):
        row = {"cluster_id": int(cluster_id)}
        for feat in MAIN_PACKET_FEATURES:
            values = _display_feature_values(feat, pd.to_numeric(group_df[feat], errors="coerce").to_numpy(dtype=float))
            valid = np.isfinite(values)
            row[feat] = float(np.nanvar(values[valid])) if np.any(valid) else np.nan
        variance_rows.append(row)
    variance_df = pd.DataFrame(variance_rows)
    if variance_df.empty:
        return None
    score_terms = []
    for feat in MAIN_PACKET_FEATURES:
        vals = pd.to_numeric(variance_df[feat], errors="coerce").to_numpy(dtype=float)
        feat_score = np.full(vals.shape, np.nan, dtype=float)
        valid = np.isfinite(vals)
        if np.any(valid):
            spread = float(np.nanstd(vals[valid]))
            if spread > 0:
                feat_score[valid] = (vals[valid] - float(np.nanmean(vals[valid]))) / spread
            else:
                feat_score[valid] = 0.0
        score_terms.append(feat_score)
    combined = np.nanmean(np.column_stack(score_terms), axis=1)
    if not np.any(np.isfinite(combined)):
        return None
    best_idx = int(np.nanargmax(combined))
    return int(variance_df.iloc[best_idx]["cluster_id"])


def build_single_neuron_scatter_matrix(region_bundle, cluster_method, cluster_id=None, template="plotly_white"):
    cluster_result = get_cluster_result(region_bundle, cluster_method)
    if cluster_result is None:
        return go.Figure(), None
    feature_df = region_bundle["feature_df"].copy()
    packet_cluster_lookup = cluster_result["packet_plot_df"][["packet_idx", "cluster_label"]].drop_duplicates(subset=["packet_idx"])
    feature_df = feature_df.merge(packet_cluster_lookup, on="packet_idx", how="left")
    if cluster_id is None:
        cluster_id = choose_default_neuron_id(region_bundle)
    if cluster_id is None:
        return go.Figure(), None

    neuron_df = feature_df.loc[feature_df["cluster_id"] == int(cluster_id)].copy().sort_values("packet_idx")
    if neuron_df.empty:
        return go.Figure(), int(cluster_id)
    n_features = len(MAIN_PACKET_FEATURES)
    display_col_map = {}
    display_labels = {}
    for feat in MAIN_PACKET_FEATURES:
        display_col = f"{feat}__display"
        neuron_df[display_col] = _display_feature_values(feat, pd.to_numeric(neuron_df[feat], errors="coerce").to_numpy(dtype=float))
        display_col_map[feat] = display_col
        display_labels[feat] = _display_feature_label(feat)

    unique_packet_clusters = _sorted_cluster_labels(neuron_df["cluster_label"].dropna().astype(int).to_numpy(dtype=int))
    fig = make_subplots(
        rows=n_features,
        cols=n_features,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )
    for row_idx, y_feat in enumerate(MAIN_PACKET_FEATURES, start=1):
        for col_idx, x_feat in enumerate(MAIN_PACKET_FEATURES, start=1):
            if row_idx == col_idx:
                for cluster_idx in unique_packet_clusters:
                    mask = neuron_df["cluster_label"].to_numpy(dtype=float) == float(cluster_idx)
                    x_vals = pd.to_numeric(neuron_df.loc[mask, display_col_map[x_feat]], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(x_vals)
                    if not np.any(valid):
                        continue
                    fig.add_trace(
                        go.Histogram(
                            x=x_vals[valid],
                            marker=dict(color=_cluster_color(int(cluster_idx))),
                            opacity=0.55,
                            nbinsx=18,
                            name=_cluster_display_name(int(cluster_idx), prefix="Packet Cluster"),
                            legendgroup=f"packet-cluster-{int(cluster_idx)}",
                            showlegend=(row_idx == 1 and col_idx == 1),
                            hovertemplate=f"{display_labels[x_feat]}<br>Value: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
            else:
                for cluster_idx in unique_packet_clusters:
                    mask = neuron_df["cluster_label"].to_numpy(dtype=float) == float(cluster_idx)
                    x_vals = pd.to_numeric(neuron_df.loc[mask, display_col_map[x_feat]], errors="coerce").to_numpy(dtype=float)
                    y_vals = pd.to_numeric(neuron_df.loc[mask, display_col_map[y_feat]], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if not np.any(valid):
                        continue
                    customdata = neuron_df.loc[mask, ["packet_idx", "packet_peak_time_s", "packet_context"]].to_numpy()[valid]
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals[valid],
                            y=y_vals[valid],
                            mode="markers",
                            marker=dict(color=_cluster_color(int(cluster_idx)), size=7, opacity=0.78),
                            name=_cluster_display_name(int(cluster_idx), prefix="Packet Cluster"),
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
    for feat_idx, feat in enumerate(MAIN_PACKET_FEATURES, start=1):
        fig.update_xaxes(title_text=display_labels[feat], row=n_features, col=feat_idx)
        fig.update_yaxes(title_text=display_labels[feat], row=feat_idx, col=1)
    fig.update_layout(
        title=(
            f"Single-neuron packet feature matrix | cluster_id {int(cluster_id)} | "
            f"region {region_bundle['packet_dataset']['region']} | packets={len(neuron_df)}"
        ),
        template=template,
        barmode="overlay",
        height=max(900, 180 * n_features + 120),
        width=max(900, 180 * n_features + 120),
        margin=dict(l=90, r=40, t=90, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig, int(cluster_id)


def _trial_numeric_array(trials, key):
    try:
        if hasattr(trials, "keys") and key in trials.keys():
            return np.asarray(trials[key], dtype=float).reshape(-1)
    except Exception:
        pass
    return np.array([], dtype=float)


def _subset_event_times(event_times, mask):
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(event_times.size, mask.size)
    if n <= 0:
        return np.array([], dtype=float)
    sub = event_times[:n][mask[:n]]
    sub = sub[np.isfinite(sub)]
    return np.sort(sub.astype(float))


def _merge_sorted_unique(*arrays):
    parts = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.unique(np.sort(np.concatenate(parts)))


def _collect_packet_psth_specs(trials, wh_events_by_period, whisk_context_mode, config_plot):
    default_pre = float(config_plot.get("SINGLE_NEURON_RASTER_PRE", 0.5))
    default_post = float(config_plot.get("SINGLE_NEURON_RASTER_POST", 1.0))

    stim_times = _trial_numeric_array(trials, "stimOn_times")
    first_move_times = _trial_numeric_array(trials, "firstMovement_times")
    feedback_times = _trial_numeric_array(trials, "feedback_times")
    contrast_left = np.abs(_trial_numeric_array(trials, "contrastLeft"))
    contrast_right = np.abs(_trial_numeric_array(trials, "contrastRight"))
    choice = _trial_numeric_array(trials, "choice")
    feedback_type = _trial_numeric_array(trials, "feedbackType")

    n_stim = min(stim_times.size, contrast_left.size, contrast_right.size)
    left_stim_times = np.array([], dtype=float)
    right_stim_times = np.array([], dtype=float)
    stim_correct_times = np.array([], dtype=float)
    stim_incorrect_times = np.array([], dtype=float)
    if n_stim > 0:
        stim_arr = stim_times[:n_stim]
        left_abs = contrast_left[:n_stim]
        right_abs = contrast_right[:n_stim]
        left_pos = np.isfinite(left_abs) & (left_abs > 0)
        right_pos = np.isfinite(right_abs) & (right_abs > 0)
        both_pos = left_pos & right_pos
        left_mask = (left_pos & ~right_pos) | (both_pos & (left_abs > right_abs))
        right_mask = (right_pos & ~left_pos) | (both_pos & (right_abs > left_abs))
        left_stim_times = _subset_event_times(stim_arr, left_mask)
        right_stim_times = _subset_event_times(stim_arr, right_mask)
        n_fb_stim = min(n_stim, feedback_type.size)
        if n_fb_stim > 0:
            stim_fb = stim_arr[:n_fb_stim]
            fb_vals = feedback_type[:n_fb_stim]
            stim_correct_times = _subset_event_times(stim_fb, np.isfinite(fb_vals) & (fb_vals == 1))
            stim_incorrect_times = _subset_event_times(stim_fb, np.isfinite(fb_vals) & (fb_vals == -1))

    n_move = min(first_move_times.size, choice.size)
    first_move_left = np.array([], dtype=float)
    first_move_right = np.array([], dtype=float)
    if n_move > 0:
        move_arr = first_move_times[:n_move]
        move_choice = choice[:n_move]
        first_move_left = _subset_event_times(move_arr, np.isfinite(move_choice) & (move_choice == 1))
        first_move_right = _subset_event_times(move_arr, np.isfinite(move_choice) & (move_choice == -1))

    n_feedback = min(feedback_times.size, feedback_type.size)
    feedback_correct = np.array([], dtype=float)
    feedback_incorrect = np.array([], dtype=float)
    if n_feedback > 0:
        feedback_arr = feedback_times[:n_feedback]
        feedback_vals = feedback_type[:n_feedback]
        feedback_correct = _subset_event_times(feedback_arr, np.isfinite(feedback_vals) & (feedback_vals == 1))
        feedback_incorrect = _subset_event_times(feedback_arr, np.isfinite(feedback_vals) & (feedback_vals == -1))

    whisk_task_iti = _merge_sorted_unique(
        wh_events_by_period.get("wh_brief_times_task", np.array([])),
        wh_events_by_period.get("wh_long_times_task", np.array([])),
        wh_events_by_period.get("wh_brief_times_iti", np.array([])),
        wh_events_by_period.get("wh_long_times_iti", np.array([])),
    )
    whisk_spont = _merge_sorted_unique(
        wh_events_by_period.get("wh_brief_times_spont", np.array([])),
        wh_events_by_period.get("wh_long_times_spont", np.array([])),
    )

    return [
        {
            "name": "stim_left",
            "display_label": "Stim On Left",
            "events": left_stim_times,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from Stim On (s)",
        },
        {
            "name": "stim_right",
            "display_label": "Stim On Right",
            "events": right_stim_times,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from Stim On (s)",
        },
        {
            "name": "first_move_left",
            "display_label": "First Move Left",
            "events": first_move_left,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from First Move (s)",
        },
        {
            "name": "first_move_right",
            "display_label": "First Move Right",
            "events": first_move_right,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from First Move (s)",
        },
        {
            "name": "feedback_correct",
            "display_label": "Feedback Correct",
            "events": feedback_correct,
            "pre_s": 0.5,
            "post_s": 5.0,
            "xaxis_title": "Time from Feedback (s)",
        },
        {
            "name": "feedback_incorrect",
            "display_label": "Feedback Incorrect",
            "events": feedback_incorrect,
            "pre_s": 0.5,
            "post_s": 5.0,
            "xaxis_title": "Time from Feedback (s)",
        },
        {
            "name": "stim_correct",
            "display_label": "Stim On Correct",
            "events": stim_correct_times,
            "pre_s": 2.0,
            "post_s": 1.0,
            "xaxis_title": "Time from Stim On (s)",
        },
        {
            "name": "stim_incorrect",
            "display_label": "Stim On Incorrect",
            "events": stim_incorrect_times,
            "pre_s": 2.0,
            "post_s": 1.0,
            "xaxis_title": "Time from Stim On (s)",
        },
        {
            "name": "whisk_task_iti",
            "display_label": "Whisking Task/ITI",
            "events": whisk_task_iti,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from whisk onset (s)",
        },
        {
            "name": "whisk_spont",
            "display_label": "Whisking Spont",
            "events": whisk_spont,
            "pre_s": float(default_pre),
            "post_s": float(default_post),
            "xaxis_title": "Time from whisk onset (s)",
        },
    ]


def precompute_packet_psth_data(packet_times, cluster_labels, trials, wh_events_by_period, whisk_context_mode, config_plot):
    packet_times = np.asarray(packet_times, dtype=float)
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    if packet_times.size == 0 or cluster_labels.size == 0 or packet_times.size != cluster_labels.size:
        return None
    packet_spikes_cluster = SimpleNamespace(times=packet_times, clusters=cluster_labels)
    bin_size = config_plot.get("SINGLE_NEURON_BIN_SIZE", 0.05)
    smooth_sigma = config_plot.get("SINGLE_NEURON_SMOOTH_SIGMA", 1)
    valid_specs = _collect_packet_psth_specs(trials, wh_events_by_period, whisk_context_mode, config_plot)
    unique_clusters = _sorted_cluster_labels(cluster_labels)
    cluster_ids_int = [int(cluster_idx) for cluster_idx in unique_clusters]
    cluster_counts = {int(cluster_idx): int(np.sum(cluster_labels == cluster_idx)) for cluster_idx in unique_clusters}
    panels = []
    for spec in valid_specs:
        bin_edges = np.arange(-float(spec["pre_s"]), float(spec["post_s"]) + bin_size, bin_size)
        bin_centers_default = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            packet_spikes_cluster,
            cluster_ids_int,
            spec["events"],
            -float(spec["pre_s"]),
            float(spec["post_s"]),
            bin_size,
            smooth_sigma,
            show_progress=False,
        )
        if bin_centers is None:
            bin_centers = bin_centers_default
        cluster_curves = {}
        for cluster_idx in cluster_ids_int:
            psth_entry = psth_by_cluster.get(int(cluster_idx))
            n_bins = len(bin_centers) if bin_centers is not None else 0
            fr_raw = psth_entry["fr_raw"] if psth_entry and bin_centers is not None else np.zeros(n_bins, dtype=float)
            fr_smooth = psth_entry["fr_smooth"] if psth_entry and bin_centers is not None else np.zeros(n_bins, dtype=float)
            cluster_curves[int(cluster_idx)] = {
                "fr_raw": np.asarray(fr_raw, dtype=np.float32),
                "fr_smooth": np.asarray(fr_smooth, dtype=np.float32),
            }
        panels.append(
            {
                "name": spec["name"],
                "display_label": spec["display_label"],
                "event_count": int(len(spec["events"])),
                "pre_s": float(spec["pre_s"]),
                "post_s": float(spec["post_s"]),
                "xaxis_title": spec["xaxis_title"],
                "bin_centers": np.asarray(bin_centers if bin_centers is not None else np.array([], dtype=float), dtype=np.float32),
                "cluster_curves": cluster_curves,
            }
        )
    return {
        "cluster_ids": cluster_ids_int,
        "cluster_counts": cluster_counts,
        "panels": panels,
    }


def build_packet_psth_figure(packet_times, cluster_labels, trials, wh_events_by_period, whisk_context_mode, config_plot, title, cluster_name_prefix="Cluster", template="plotly_white", precomputed=None):
    psth_data = precomputed
    if psth_data is not None:
        panels = psth_data.get("panels", [])
        if len(panels) != 10:
            psth_data = None
    if psth_data is None:
        psth_data = precompute_packet_psth_data(
            packet_times,
            cluster_labels,
            trials,
            wh_events_by_period,
            whisk_context_mode,
            config_plot,
        )
    if not psth_data or not psth_data.get("panels"):
        return go.Figure()

    fig = make_subplots(
        rows=5,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
        subplot_titles=tuple(f"{panel['display_label']} (n={panel['event_count']})" for panel in psth_data["panels"]),
    )
    legend_drawn = False
    for panel_idx, panel in enumerate(psth_data["panels"], start=0):
        row_idx = (panel_idx // 2) + 1
        col_idx = (panel_idx % 2) + 1
        bin_centers = np.asarray(panel["bin_centers"], dtype=float)
        if panel["event_count"] <= 0:
            fig.add_annotation(
                text="No events",
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(title_text=panel["xaxis_title"], range=[-float(panel["pre_s"]), float(panel["post_s"])], row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text="Packet rate (Hz)", row=row_idx, col=col_idx)
            continue
        for cluster_idx in psth_data["cluster_ids"]:
            psth_entry = panel["cluster_curves"].get(int(cluster_idx), {})
            firing_rate = np.asarray(psth_entry.get("fr_smooth", np.zeros(bin_centers.size, dtype=float)), dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=firing_rate,
                    mode="lines",
                    line=dict(color=_cluster_color(int(cluster_idx)), width=2),
                    name=f"{_cluster_display_name(int(cluster_idx), prefix=cluster_name_prefix)} (n={int(psth_data['cluster_counts'].get(int(cluster_idx), 0))})",
                    legendgroup=f"{cluster_name_prefix}-{int(cluster_idx)}",
                    showlegend=(not legend_drawn),
                ),
                row=row_idx,
                col=col_idx,
            )
            legend_drawn = True
        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=row_idx, col=col_idx)
        fig.update_xaxes(title_text=panel["xaxis_title"], range=[-float(panel["pre_s"]), float(panel["post_s"])], row=row_idx, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Packet rate (Hz)", row=row_idx, col=col_idx)
    fig.update_layout(
        title=title,
        template=template,
        height=1700,
        margin=dict(l=70, r=40, t=90, b=150),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )
    return fig
