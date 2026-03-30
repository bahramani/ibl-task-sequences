# %%
from pathlib import Path
import pickle
import traceback
import sys

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # if notebook is in /notebooks/

from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    load_task_replay_datasets,
    build_passive_event_wrappers,
    select_task_stim_events_by_side,
    build_cluster_id_map,
    get_cluster_labels_array,
)
import utils.analysis as ana_utils

CALC_VERSION = "whisking-v1.1"

CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_LABEL_MIN": 0.9,
    "CALC_LABEL_STRICT_GT": True,
    "CALC_SPONT": True,
    # Used for task/ITI windows; delay event names are built dynamically below.
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "feedback_times"],
    "DELAY_METHOD": "com_signed",
    "WH_DELAY_METHOD": "com_signed",
    "LATENZY_USE_DUR": None,
    "LATENZY_MIN_SCORE": None,
    "DELAY_UNITS": "ms",
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    "DELAY_WINDOWS": {
        "stimOn_times": (0.02, 0.35),
        "firstMovement_times": (-0.1, 0.2),
        "feedback_times": (-0.1, 0.2),
    },
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -1.0,
    "PSTH_WINDOW_END": 1.0,
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.35,
    "RESPONSIVE_USE_ZSCORE": True,
    "RESPONSIVE_ZSCORE_SOURCE": "smooth",
    "COM_USE_THRESHOLD": True,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 10,
    "MIN_TRIALS_SPLIT": 5,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_LOW_PASS_ORDER": 3,
    "STPR_POP_USE_GOOD_UNITS": False,
    "COUPLING_PARALLEL_WORKERS": 0,  # 0 => auto (cpu_count-1 capped by #jobs)
    "COUPLING_PARALLEL_PREFER_PROCESS": True,
    "TASK_POST_EVENT_S": 1.0,
    "ITI_SKIP_FIRST_LAST": True,
    # Passive defaults from notebook 09/10.
    "PASSIVE_VISUAL_DELAY_WINDOW": (0.01, 0.2),
    "PASSIVE_AUDITORY_DELAY_WINDOW": (0.0, 0.1),
    # Whisk defaults from notebook 11.
    "WH_BIN_SIGNAL": False,
    "WH_BIN_S": 0.3,
    "WH_NORM_PCTL": 1.0,
    "WH_NORM_TOP_PCTL": 100.0,
    "WH_BIN_REDUCE": "mean",
    "WH_NORMALIZE_AFTER_BIN": True,
    "WH_TARGET_MEAN": 0.06,
    "WH_START_THR": 0.10,
    "WH_END_THR": 0.04,
    "WH_END_QUIET_WINDOW_S": 0.5,
    "WH_MERGE_GAP_S": 0.3,
    "WH_BRIEF_RANGE_S": (0.25, 2.0),
    "WH_LONG_MIN_S": 2.0,
    "WH_LOCO_LOOKAHEAD_S": 6.0,
    "WH_LOCO_SPEED_THR_CM_S": 1.0,
    "WH_WHEEL_RADIUS_CM": 3.1,
    "WH_DELAY_WINDOW": (0.0, 0.4),
    "WH_SPLIT_MODE": "odd_even",
    "AROUSAL_GROUP_MODE": "response_sign",  # "corr" or "response_sign"
    "AROUSAL_SIGN_EVENT": "wh_brief_times_spont",
    "AROUSAL_POS_THR": 0.05,
    "AROUSAL_NEG_THR": -0.05,
    "AROUSAL_MIN_FR_HZ": 0.5,
    "AROUSAL_MIN_BRIEF_EVENTS": 5,
    "AROUSAL_MIN_LONG_EVENTS": 5,
    "AROUSAL_BIN_S": 0.3,
    "AROUSAL_SMOOTH_SIGMA": 5,
    "AROUSAL_MIN_CORR_BINS": 10,
    "AROUSAL_USE_EVENT_BASELINE_ZSCORE": True,
    "AROUSAL_BASELINE_PRE": 0.2,
    "AROUSAL_MIN_BASELINE_BINS": 3,
    "AROUSAL_REQUIRE_SPLIT_HALF": True,
}

CONFIG_PLOT = {
    "ATLAS_MAPPING": "Beryl",
    "PLOT_ONLY_GOOD_UNITS": False,
    "PLOT_EVENT": "stimOn_times",
    "PLOT_REGIONS": None,
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
    "POP_WINDOW_PRE": 0.5,
    "POP_WINDOW_POST": 0.5,
    "POP_BIN_SIZE": 0.005,
    "POP_SMOOTH_SIGMA": 2,
    "POP_CMAP_NAME": "bwr",
    "POP_NORMALIZE": True,
    "SORT_BY_SPONT": True,
}

AUDITORY_FEEDBACK_ACRONYMS = frozenset({"AUDp", "AUDv", "AUDd"})
EVENT_RESPONSE_SPECS = (
    {
        "event_name": "stimOn_times",
        "response_window": (0.02, 0.35),
        "baseline_window": (-0.2, 0.0),
    },
    {
        "event_name": "firstMovement_times",
        "response_window": (-0.1, 0.2),
        "baseline_window": (-0.3, -0.1),
    },
    {
        "event_name": "feedback_times",
        "response_window": (-0.1, 0.2),
        "baseline_window": (-0.3, -0.1),
    },
    {
        "event_name": "wh_brief_times_spont",
        "response_window": (0.0, 0.4),
        "baseline_window": (-0.2, 0.0),
    },
)
AUDITORY_FEEDBACK_EVENT_SPECS = (
    {
        "event_name": "feedback_correct_times",
        "response_window": (0.0, 0.1),
        "baseline_window": (-0.2, 0.0),
    },
    {
        "event_name": "feedback_incorrect_times",
        "response_window": (0.0, 0.1),
        "baseline_window": (-0.2, 0.0),
    },
)

# Update this list with the PIDs you want to process, or leave as None to query by subject.
COMPUTE_ALL = True  # If True, ignore PIDS/SUBJECT/REGIONS and process all insertions for TAG.
PIDS = [
    # "afe87fbb-3a17-461f-b333-e22903f1d70d",
    "49c2ea3d-2b50-4e8a-b124-9e190960784e",
    # "eebcaf65-7fa4-4118-869d-a084e84530e2",
    # "c9664185-d3fd-4e0e-89cf-77c402038938"
]
SUBJECT = None  # "CSH_ZAD_029"
REGIONS = None # ["VISp", "MOp", "PO", "CA1", "AUDp", "CP", "VPM"]
TAG = "2025_Q3_IBL_et_al_BWM"


def _tag_search_kwargs(tag):
    # IBL docs: tags are attached to datasets, and insertion queries should use
    # django='datasets__tags__name,<tag>' when restricting by a data-release tag.
    if not tag:
        return {}
    return {"django": f"datasets__tags__name,{tag}"}


def _fetch_session_metadata(one, eid):
    meta = {"lab": None, "date": None, "subject": None}
    try:
        sess = one.alyx.rest("sessions", "read", id=eid)
        if isinstance(sess, dict):
            meta["lab"] = sess.get("lab") or sess.get("lab_name") or sess.get("location")
            meta["subject"] = sess.get("subject") or sess.get("subject_nickname")
            meta["date"] = sess.get("start_time") or sess.get("date")
    except Exception:
        pass
    return meta


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


def _get_pids_for_subject(one, subject, tag):
    return list(
        dict.fromkeys(
            one.search_insertions(
                subject=subject,
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    )


def _get_pids_for_regions(one, regions, tag):
    if not regions:
        return []
    all_pids = []
    for region in regions:
        all_pids.extend(
            one.search_insertions(
                atlas_acronym=region,
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    return list(dict.fromkeys(all_pids))


def _get_all_tag_pids(one, tag):
    return list(
        dict.fromkeys(
            one.search_insertions(
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    )


def _load_cache_if_exists(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


HEAVY_CACHE_KEYS = (
    "spikes",
    "clusters",
    "session",
    "pupil_features",
    "pupil_times",
)


def _strip_heavy_cache(cache):
    if cache is None:
        return cache, False
    removed = False
    for key in HEAVY_CACHE_KEYS:
        if key in cache:
            cache.pop(key, None)
            removed = True
    return cache, removed


def _cache_needs_update(cache, config_calc, calc_version):
    if cache is None:
        return True
    if cache.get("calc_version") != calc_version:
        return True
    if cache.get("config_calc") != config_calc:
        return True
    return False


def _empty_whisk_bundle():
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
        "wh_events_by_period": {
            "wh_brief_times": np.array([], dtype=float),
            "wh_long_times": np.array([], dtype=float),
            "wh_all_times": np.array([], dtype=float),
            "wh_brief_times_spont": np.array([], dtype=float),
            "wh_long_times_spont": np.array([], dtype=float),
            "wh_all_times_spont": np.array([], dtype=float),
            "wh_long_offset_times_spont": np.array([], dtype=float),
        },
        "wh_long_offset_times_spont": np.array([], dtype=float),
        "wh_loco": {
            "all_onsets": np.array([], dtype=float),
            "loco_flags": np.array([], dtype=bool),
            "loco_onsets": np.array([], dtype=float),
            "non_loco_onsets": np.array([], dtype=float),
            "wheel_times": np.array([], dtype=float),
            "wheel_speed_cm_s": np.array([], dtype=float),
        },
    }


def _strict_good_cluster_ids(cluster_ids, labels, label_min, strict_gt=True):
    if labels is None or label_min is None:
        return np.asarray(cluster_ids)
    labels = np.asarray(labels)
    if labels.shape[0] != len(cluster_ids):
        return np.asarray(cluster_ids)
    try:
        labels_float = labels.astype(float)
        if strict_gt:
            mask = labels_float > float(label_min)
        else:
            mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return np.asarray(cluster_ids)[mask]


def _event_response_full_col(event_name):
    return f"response_zmean_{event_name}"


def _event_response_split_col(event_name, split_name):
    return f"{_event_response_full_col(event_name)}_{split_name}"


def _build_event_payloads_from_trial_df(trial_df, wh_events_by_period):
    payloads = {}
    if isinstance(trial_df, pd.DataFrame) and not trial_df.empty:
        trial_idx = pd.to_numeric(trial_df.get("trial_idx"), errors="coerce").to_numpy(dtype=float)
        trial_idx = np.where(np.isfinite(trial_idx), trial_idx, np.arange(len(trial_df), dtype=float)).astype(int)
        contrast = None
        if "contrast" in trial_df.columns:
            contrast = pd.to_numeric(trial_df["contrast"], errors="coerce").to_numpy(dtype=float)

        if {"stimOn_times", "contrast"}.issubset(trial_df.columns):
            stim = pd.to_numeric(trial_df["stimOn_times"], errors="coerce").to_numpy(dtype=float)
            stim_contrast = np.asarray(contrast, dtype=float)
            mask = np.isfinite(stim) & np.isfinite(stim_contrast) & (stim_contrast > 0)
            payloads["stimOn_times"] = {
                "events": np.asarray(stim[mask], dtype=float),
                "split_index": trial_idx[mask].astype(int),
                "contrasts": np.asarray(stim_contrast[mask], dtype=float),
            }

        for event_name in ("firstMovement_times", "feedback_times"):
            if event_name not in trial_df.columns:
                continue
            values = pd.to_numeric(trial_df[event_name], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            payloads[event_name] = {
                "events": np.asarray(values[mask], dtype=float),
                "split_index": trial_idx[mask].astype(int),
                "contrasts": (
                    np.asarray(contrast[mask], dtype=float)
                    if contrast is not None
                    else np.ones(int(np.sum(mask)), dtype=float)
                ),
            }

        if {"feedback_times", "correct_response"}.issubset(trial_df.columns):
            feedback = pd.to_numeric(trial_df["feedback_times"], errors="coerce").to_numpy(dtype=float)
            correct_response = trial_df["correct_response"]
            correct_mask = np.isfinite(feedback) & correct_response.eq(True).to_numpy(dtype=bool)
            incorrect_mask = np.isfinite(feedback) & correct_response.eq(False).to_numpy(dtype=bool)
            for event_name, mask in (
                ("feedback_correct_times", correct_mask),
                ("feedback_incorrect_times", incorrect_mask),
            ):
                payloads[event_name] = {
                    "events": np.asarray(feedback[mask], dtype=float),
                    "split_index": trial_idx[mask].astype(int),
                    "contrasts": (
                        np.asarray(contrast[mask], dtype=float)
                        if contrast is not None
                        else np.ones(int(np.sum(mask)), dtype=float)
                    ),
                }

    wh_events = np.asarray((wh_events_by_period or {}).get("wh_brief_times_spont", np.array([])), dtype=float)
    wh_events = np.sort(wh_events[np.isfinite(wh_events)])
    payloads["wh_brief_times_spont"] = {
        "events": wh_events.astype(float),
        "split_index": np.arange(wh_events.size, dtype=int),
        "contrasts": np.ones(wh_events.size, dtype=float),
    }
    return payloads


def _zscore_trace(arr, baseline_mask):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    baseline_mask = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    if arr.size == 0 or baseline_mask.size != arr.size:
        return np.full(arr.shape, np.nan, dtype=float)
    ref = arr[baseline_mask]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return (arr - mu) / sd


def _response_zmean_from_psth(psth_entry, bin_centers, baseline_window, response_window, zscore_source):
    if psth_entry is None or bin_centers is None:
        return np.nan
    if zscore_source == "raw":
        trace = np.asarray(psth_entry.get("fr_raw", np.array([])), dtype=float)
    else:
        smooth = psth_entry.get("fr_smooth", None)
        trace = np.asarray(smooth, dtype=float) if smooth is not None else np.asarray(psth_entry.get("fr_raw", np.array([])), dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    if trace.size == 0 or centers.size != trace.size:
        return np.nan
    baseline_mask = (centers >= baseline_window[0]) & (centers < baseline_window[1])
    response_mask = (centers >= response_window[0]) & (centers <= response_window[1])
    if not np.any(response_mask):
        return np.nan
    ztrace = _zscore_trace(trace, baseline_mask=baseline_mask)
    response_vals = ztrace[response_mask]
    response_vals = response_vals[np.isfinite(response_vals)]
    if response_vals.size == 0:
        return np.nan
    return float(np.mean(response_vals))


def _compute_event_response_metrics(spikes, cluster_ids, events, split_index, config, spec, zscore_source):
    n_units = int(cluster_ids.size)
    full_vals = np.full(n_units, np.nan, dtype=float)
    odd_vals = np.full(n_units, np.nan, dtype=float)
    even_vals = np.full(n_units, np.nan, dtype=float)
    min_trials = int(config.get("MIN_TRIALS", 10))
    min_trials_split = int(config.get("MIN_TRIALS_SPLIT", 5))
    if events.size < min_trials or n_units == 0:
        return full_vals, odd_vals, even_vals

    psth_kwargs = {
        "window_start": float(config.get("PSTH_WINDOW_START", -1.0)),
        "window_end": float(config.get("PSTH_WINDOW_END", 1.0)),
        "bin_size": float(config.get("BIN_SIZE", 0.005)),
        "smooth_sigma": float(config.get("SMOOTH_SIGMA", 1)),
        "show_progress": False,
        "desc": f"PSTH {spec['event_name']}",
    }
    psth_full, bin_centers_full = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events,
        **psth_kwargs,
    )
    for row_idx, cid in enumerate(cluster_ids):
        full_vals[row_idx] = _response_zmean_from_psth(
            psth_full.get(int(cid)),
            bin_centers_full,
            spec["baseline_window"],
            spec["response_window"],
            zscore_source=zscore_source,
        )

    odd_mask = (split_index % 2) == 1
    even_mask = ~odd_mask
    events_odd = events[odd_mask]
    events_even = events[even_mask]
    if events_odd.size < min_trials_split or events_even.size < min_trials_split:
        return full_vals, odd_vals, even_vals

    psth_odd, bin_centers_odd = ana_utils.compute_psth_for_clusters(spikes, cluster_ids, events_odd, **psth_kwargs)
    psth_even, bin_centers_even = ana_utils.compute_psth_for_clusters(spikes, cluster_ids, events_even, **psth_kwargs)
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        odd_vals[row_idx] = _response_zmean_from_psth(
            psth_odd.get(cid_int),
            bin_centers_odd,
            spec["baseline_window"],
            spec["response_window"],
            zscore_source=zscore_source,
        )
        even_vals[row_idx] = _response_zmean_from_psth(
            psth_even.get(cid_int),
            bin_centers_even,
            spec["baseline_window"],
            spec["response_window"],
            zscore_source=zscore_source,
        )
    return full_vals, odd_vals, even_vals


def _apply_delay_units(arr, config):
    out = np.asarray(arr, dtype=float).copy()
    if str(config.get("DELAY_UNITS", "s")).lower().startswith("ms"):
        out *= 1000.0
    return out


def _compute_event_split_delay_metrics(
    spikes,
    cluster_ids,
    spike_times_by_cluster,
    events,
    split_index,
    contrasts,
    config,
    event_name,
):
    n_units = int(cluster_ids.size)
    full_vals = np.full(n_units, np.nan, dtype=float)
    odd_vals = np.full(n_units, np.nan, dtype=float)
    even_vals = np.full(n_units, np.nan, dtype=float)
    min_trials = int(config.get("MIN_TRIALS", 10))
    min_trials_split = int(config.get("MIN_TRIALS_SPLIT", 5))
    if events.size < min_trials or n_units == 0:
        return full_vals, odd_vals, even_vals

    win_start, win_end = ana_utils.get_event_delay_window(config, event_name)
    event_config = {
        **config,
        "RESPONSIVE_WINDOW_START": float(win_start),
        "RESPONSIVE_WINDOW_END": float(win_end),
    }
    event_method = ana_utils.get_event_delay_method(config, event_name)
    psth_kwargs = {
        "window_start": float(config.get("PSTH_WINDOW_START", -1.0)),
        "window_end": float(config.get("PSTH_WINDOW_END", 1.0)),
        "bin_size": float(config.get("BIN_SIZE", 0.005)),
        "smooth_sigma": float(config.get("SMOOTH_SIGMA", 1)),
        "show_progress": False,
        "desc": f"PSTH delay {event_name}",
    }
    psth_full, bin_centers_full = ana_utils.compute_psth_for_clusters(spikes, cluster_ids, events, **psth_kwargs)
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        psth_entry = psth_full.get(cid_int)
        delay, is_responsive = ana_utils.calculate_delay(
            psth_entry.get("fr_raw") if psth_entry else None,
            psth_entry.get("fr_smooth") if psth_entry else None,
            bin_centers_full,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events,
            trial_contrasts=contrasts,
            return_sign=False,
        )
        if is_responsive:
            full_vals[row_idx] = delay

    odd_mask = (split_index % 2) == 1
    even_mask = ~odd_mask
    events_odd = events[odd_mask]
    events_even = events[even_mask]
    if events_odd.size < min_trials_split or events_even.size < min_trials_split:
        return _apply_delay_units(full_vals, config), odd_vals, even_vals

    contrasts_odd = contrasts[odd_mask]
    contrasts_even = contrasts[even_mask]
    psth_odd, bin_centers_odd = ana_utils.compute_psth_for_clusters(spikes, cluster_ids, events_odd, **psth_kwargs)
    psth_even, bin_centers_even = ana_utils.compute_psth_for_clusters(spikes, cluster_ids, events_even, **psth_kwargs)
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        odd_entry = psth_odd.get(cid_int)
        delay_odd, resp_odd = ana_utils.calculate_delay(
            odd_entry.get("fr_raw") if odd_entry else None,
            odd_entry.get("fr_smooth") if odd_entry else None,
            bin_centers_odd,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events_odd,
            trial_contrasts=contrasts_odd,
            return_sign=False,
        )
        if resp_odd:
            odd_vals[row_idx] = delay_odd

        even_entry = psth_even.get(cid_int)
        delay_even, resp_even = ana_utils.calculate_delay(
            even_entry.get("fr_raw") if even_entry else None,
            even_entry.get("fr_smooth") if even_entry else None,
            bin_centers_even,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events_even,
            trial_contrasts=contrasts_even,
            return_sign=False,
        )
        if resp_even:
            even_vals[row_idx] = delay_even

    return (
        _apply_delay_units(full_vals, config),
        _apply_delay_units(odd_vals, config),
        _apply_delay_units(even_vals, config),
    )


def _augment_df_res_with_event_response_metrics(df_res, spikes, trial_df, wh_events_by_period, config_calc):
    if df_res is None or df_res.empty or "cluster_id" not in df_res.columns or "acronym" not in df_res.columns:
        return df_res

    df_res = df_res.copy()
    cluster_ids = pd.to_numeric(df_res["cluster_id"], errors="coerce").to_numpy(dtype=float)
    valid_cluster_mask = np.isfinite(cluster_ids)
    df_res = df_res.loc[valid_cluster_mask].copy()
    cluster_ids = cluster_ids[valid_cluster_mask].astype(int)
    if cluster_ids.size == 0:
        return df_res

    config = dict(config_calc or {})
    delay_windows = dict(config.get("DELAY_WINDOWS", {}) or {})
    delay_windows.setdefault("feedback_correct_times", (0.0, 0.1))
    delay_windows.setdefault("feedback_incorrect_times", (0.0, 0.1))
    config["DELAY_WINDOWS"] = delay_windows

    zscore_source = str(config.get("RESPONSIVE_ZSCORE_SOURCE", "smooth")).strip().lower()
    if zscore_source not in {"raw", "smooth"}:
        zscore_source = "smooth"

    event_payloads = _build_event_payloads_from_trial_df(trial_df, wh_events_by_period)
    for spec in EVENT_RESPONSE_SPECS:
        payload = event_payloads.get(
            spec["event_name"],
            {"events": np.array([], dtype=float), "split_index": np.array([], dtype=int), "contrasts": np.array([], dtype=float)},
        )
        full_vals, odd_vals, even_vals = _compute_event_response_metrics(
            spikes=spikes,
            cluster_ids=cluster_ids,
            events=np.asarray(payload["events"], dtype=float),
            split_index=np.asarray(payload["split_index"], dtype=int),
            config=config,
            spec=spec,
            zscore_source=zscore_source,
        )
        df_res[_event_response_full_col(spec["event_name"])] = full_vals
        df_res[_event_response_split_col(spec["event_name"], "odd")] = odd_vals
        df_res[_event_response_split_col(spec["event_name"], "even")] = even_vals

    auditory_mask = df_res["acronym"].astype(str).isin(AUDITORY_FEEDBACK_ACRONYMS).to_numpy(dtype=bool)
    for spec in AUDITORY_FEEDBACK_EVENT_SPECS:
        df_res[_event_response_full_col(spec["event_name"])] = np.nan
        df_res[_event_response_split_col(spec["event_name"], "odd")] = np.nan
        df_res[_event_response_split_col(spec["event_name"], "even")] = np.nan
        delay_col = ana_utils.delay_column_name(spec["event_name"])
        delay_odd_col = ana_utils.delay_split_column_name(spec["event_name"], "odd")
        delay_even_col = ana_utils.delay_split_column_name(spec["event_name"], "even")
        df_res[delay_col] = np.nan
        df_res[delay_odd_col] = np.nan
        df_res[delay_even_col] = np.nan

        if not np.any(auditory_mask):
            continue

        payload = event_payloads.get(
            spec["event_name"],
            {"events": np.array([], dtype=float), "split_index": np.array([], dtype=int), "contrasts": np.array([], dtype=float)},
        )
        auditory_cluster_ids = cluster_ids[auditory_mask]
        spike_times_by_cluster = {
            int(cid): spikes["times"][spikes["clusters"] == int(cid)]
            for cid in auditory_cluster_ids
        }
        full_vals, odd_vals, even_vals = _compute_event_response_metrics(
            spikes=spikes,
            cluster_ids=auditory_cluster_ids,
            events=np.asarray(payload["events"], dtype=float),
            split_index=np.asarray(payload["split_index"], dtype=int),
            config=config,
            spec=spec,
            zscore_source=zscore_source,
        )
        df_res.loc[auditory_mask, _event_response_full_col(spec["event_name"])] = full_vals
        df_res.loc[auditory_mask, _event_response_split_col(spec["event_name"], "odd")] = odd_vals
        df_res.loc[auditory_mask, _event_response_split_col(spec["event_name"], "even")] = even_vals

        delay_full, delay_odd, delay_even = _compute_event_split_delay_metrics(
            spikes=spikes,
            cluster_ids=auditory_cluster_ids,
            spike_times_by_cluster=spike_times_by_cluster,
            events=np.asarray(payload["events"], dtype=float),
            split_index=np.asarray(payload["split_index"], dtype=int),
            contrasts=np.asarray(payload.get("contrasts", np.ones(len(payload["events"]), dtype=float)), dtype=float),
            config=config,
            event_name=spec["event_name"],
        )
        df_res.loc[auditory_mask, delay_col] = delay_full
        df_res.loc[auditory_mask, delay_odd_col] = delay_odd
        df_res.loc[auditory_mask, delay_even_col] = delay_even

    return df_res.sort_values("cluster_id").reset_index(drop=True)


def main():
    base_path = Path(__file__).resolve().parents[1]
    path_data, _path_fig, path_data_processed, ibl_cache = setup_paths(base_path)
    cache_dir = path_data / "dashboard_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    one = init_one(ibl_cache)
    ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)

    if COMPUTE_ALL:
        pids_to_process = _get_all_tag_pids(one, TAG)
    elif PIDS:
        pids_to_process = list(PIDS)
    elif SUBJECT is not None:
        pids_to_process = _get_pids_for_subject(one, SUBJECT, TAG)
    else:
        pids_to_process = _get_pids_for_regions(one, REGIONS, TAG)
    if not pids_to_process:
        raise RuntimeError("No PIDs found. Set COMPUTE_ALL, PIDS, SUBJECT, or REGIONS.")

    for pid in tqdm(pids_to_process, desc="Processing PIDs"):
        cache_path = cache_dir / f"{pid}.pkl"
        cache_existing = _load_cache_if_exists(cache_path)
        cache_existing, removed_heavy = _strip_heavy_cache(cache_existing)
        if not _cache_needs_update(cache_existing, CONFIG_CALC, CALC_VERSION):
            updated = False
            if cache_existing is not None and cache_existing.get("config_plot") != CONFIG_PLOT:
                cache_existing["config_plot"] = CONFIG_PLOT
                cache_existing["config_calc"] = CONFIG_CALC
                cache_existing["calc_version"] = CALC_VERSION
                updated = True
            if removed_heavy:
                updated = True
            if updated and cache_existing is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_existing, f)
                print(f"Updated cache metadata: {cache_path}")
            else:
                print(f"Using cached results for {pid}")
            continue

        step = "load_session_data"
        try:
            ssl, spikes, clusters, sl = load_session_data(
                pid,
                one,
                ba,
                load_wheel=True,
                load_pose=False,
                load_motion_energy=True,
            )
            eid = ssl.eid
            eid2pid = one.eid2pid(eid)
            if isinstance(eid2pid, tuple):
                eid2pid = eid2pid[0]
            eid2pid = [str(p) for p in eid2pid]

            step = "cluster_setup"
            cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
            cluster_ids = np.asarray(cluster_ids)
            cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
            cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

            step = "cluster_firing_rates"
            cluster_firing_rate = None
            if hasattr(clusters, "firing_rate"):
                cluster_firing_rate = np.asarray(clusters.firing_rate)
            elif isinstance(clusters, dict) and "firing_rate" in clusters:
                cluster_firing_rate = np.asarray(clusters["firing_rate"])
            elif hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
                if "firing_rate" in clusters.metrics.columns:
                    cluster_firing_rate = np.asarray(clusters.metrics["firing_rate"])
            if cluster_firing_rate is not None and len(cluster_firing_rate) != len(cluster_ids):
                cluster_firing_rate = None

            step = "task_windows"
            task_windows = ana_utils.build_task_window_table(
                sl.trials,
                CONFIG_CALC["EVENT_NAMES"],
                post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
            )
            stim_on_times = np.asarray(sl.trials["stimOn_times"])
            trial_end_times = ana_utils.compute_trial_end_times(
                sl.trials,
                CONFIG_CALC["EVENT_NAMES"],
                post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
            )
            iti_windows = ana_utils.build_iti_windows(
                trial_end_times,
                stim_on_times,
                skip_first_last=CONFIG_CALC["ITI_SKIP_FIRST_LAST"],
            )

            step = "load_spont_intervals"
            spont_intervals = None
            if CONFIG_CALC["CALC_SPONT"]:
                spont_intervals = _load_spontaneous_intervals(one, eid)

            step = "passive_data"
            visual_tr, auditory_tr = load_task_replay_datasets(
                eid,
                one,
                one_remote=one,
                allow_remote=True,
            )
            task_stim_subsets = select_task_stim_events_by_side(sl, top_n=2, round_decimals=6)
            passive_wrappers = build_passive_event_wrappers(
                visual_tr,
                auditory_tr,
                top_n=2,
                round_decimals=6,
            )

            step = "whisk_data"
            df_me_raw = ana_utils.extract_motion_energy_trace(
                getattr(sl, "motion_energy", None),
                max_interp_gap_frames=3,
                ensure_positive_motion=True,
            )
            motion_energy_available = not df_me_raw.empty
            if motion_energy_available:
                df_wh = ana_utils.build_whisk_trace(df_me_raw, CONFIG_CALC)
                if df_wh is None or df_wh.empty:
                    whisk_bundle = _empty_whisk_bundle()
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
                else:
                    whisk_bundle = ana_utils.build_whisk_events(
                        df_wh,
                        CONFIG_CALC,
                        spont_intervals=spont_intervals,
                        task_windows=task_windows,
                        iti_windows=iti_windows,
                        wheel=getattr(sl, "wheel", None),
                    )
            else:
                whisk_bundle = _empty_whisk_bundle()
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
            wh_detect = whisk_bundle.get("wh_detect", {})
            wh_event_base = whisk_bundle.get("wh_event_base", {})
            wh_events_by_period = whisk_bundle.get("wh_events_by_period", {})

            step = "build_event_inputs"
            events_by_name, contrasts_by_name, trial_idx_by_name = (
                ana_utils.build_dashboard_delay_event_inputs(
                    sl,
                    task_stim_subsets=task_stim_subsets,
                    passive_events=passive_wrappers,
                    whisk_events=wh_events_by_period,
                )
            )

            delay_event_names = [
                "stimOn_times",
                "firstMovement_times",
                "feedback_times",
                "stimOn_times_task_zero_lr",
                "passive_tone_times",
                "passive_valve_times",
                "passive_noise_times",
                "passive_visual_times",
                "passive_visual_top2_left_times",
                "wh_brief_times_spont",
                "wh_long_times_spont",
                "wh_all_times_spont",
                "wh_long_offset_times_spont",
            ]
            delay_windows = dict(CONFIG_CALC.get("DELAY_WINDOWS", {}))
            delay_windows["stimOn_times_task_zero_lr"] = tuple(
                CONFIG_CALC.get("DELAY_WINDOWS", {}).get("stimOn_times", (0.02, 0.35))
            )
            delay_windows["passive_visual_times"] = tuple(
                CONFIG_CALC.get("PASSIVE_VISUAL_DELAY_WINDOW", (0.01, 0.2))
            )
            delay_windows["passive_visual_top2_left_times"] = tuple(
                CONFIG_CALC.get("PASSIVE_VISUAL_DELAY_WINDOW", (0.01, 0.2))
            )
            for nm in ("passive_tone_times", "passive_valve_times", "passive_noise_times"):
                delay_windows[nm] = tuple(
                    CONFIG_CALC.get("PASSIVE_AUDITORY_DELAY_WINDOW", (0.0, 0.1))
                )
            for nm in (
                "wh_brief_times_spont",
                "wh_long_times_spont",
                "wh_all_times_spont",
                "wh_long_offset_times_spont",
            ):
                delay_windows[nm] = tuple(CONFIG_CALC.get("WH_DELAY_WINDOW", (0.0, 0.4)))

            delay_methods_by_event = {}
            for nm in delay_event_names:
                if nm.startswith("wh_"):
                    delay_methods_by_event[nm] = str(
                        CONFIG_CALC.get("WH_DELAY_METHOD", CONFIG_CALC.get("DELAY_METHOD", "com"))
                    )
                else:
                    delay_methods_by_event[nm] = str(CONFIG_CALC.get("DELAY_METHOD", "com"))

            delay_config = dict(CONFIG_CALC)
            delay_config["EVENT_NAMES"] = list(delay_event_names)
            delay_config["DELAY_WINDOWS"] = delay_windows
            delay_config["DELAY_METHODS_BY_EVENT"] = delay_methods_by_event

            step = "calculate_delays"
            split_delay_events = [
                "stimOn_times",
                "firstMovement_times",
                "feedback_times",
                "wh_brief_times_spont",
                "passive_visual_times",
                "passive_tone_times",
                "passive_valve_times",
                "passive_noise_times",
            ]
            split_delay_events = [nm for nm in split_delay_events if nm in delay_event_names]
            full_only_delay_events = [nm for nm in delay_event_names if nm not in set(split_delay_events)]

            df_res_split = pd.DataFrame()
            if len(split_delay_events) > 0:
                delay_config_split = dict(delay_config)
                delay_config_split["EVENT_NAMES"] = list(split_delay_events)
                df_res_split = ana_utils.calculate_event_delays(
                    spikes,
                    clusters,
                    cluster_acronyms_calc,
                    events_by_name,
                    delay_config_split,
                    cid_to_idx,
                    contrasts_by_name=contrasts_by_name,
                    trial_idx_by_name=trial_idx_by_name,
                    include_splits=True,
                    include_splits_events=split_delay_events,
                    output_path=None,
                )

            df_res_full = pd.DataFrame()
            if len(full_only_delay_events) > 0:
                delay_config_full = dict(delay_config)
                delay_config_full["EVENT_NAMES"] = list(full_only_delay_events)
                df_res_full = ana_utils.calculate_event_delays(
                    spikes,
                    clusters,
                    cluster_acronyms_calc,
                    events_by_name,
                    delay_config_full,
                    cid_to_idx,
                    contrasts_by_name=contrasts_by_name,
                    trial_idx_by_name=trial_idx_by_name,
                    include_splits=False,
                    output_path=None,
                )

            if df_res_split is None or df_res_split.empty:
                df_res = df_res_full.copy()
            elif df_res_full is None or df_res_full.empty:
                df_res = df_res_split.copy()
            else:
                drop_dupes = [col for col in ("acronym", "label") if col in df_res_full.columns]
                df_res = df_res_split.merge(
                    df_res_full.drop(columns=drop_dupes),
                    on="cluster_id",
                    how="outer",
                )
            if df_res is None:
                df_res = pd.DataFrame()
            elif not df_res.empty and "cluster_id" in df_res.columns:
                df_res = df_res.sort_values("cluster_id").reset_index(drop=True)
            (path_data_processed / f"{pid}_delay_results_dashboard.csv").parent.mkdir(parents=True, exist_ok=True)
            if isinstance(df_res, pd.DataFrame):
                df_res.to_csv(path_data_processed / f"{pid}_delay_results_dashboard.csv", index=False)

            step = "arousal_groups"
            if df_res is None or df_res.empty:
                df_res = pd.DataFrame(columns=["cluster_id", "acronym", "label"])
            selected_ids = (
                df_res["cluster_id"].to_numpy(dtype=int)
                if "cluster_id" in df_res.columns and len(df_res) > 0
                else np.array([], dtype=int)
            )
            selected_acronyms = np.asarray(
                [cluster_acronyms_calc[cid_to_idx[int(cid)]] for cid in selected_ids],
                dtype=str,
            )
            selected_cid_to_idx = {int(cid): cid_to_idx[int(cid)] for cid in selected_ids}

            if motion_energy_available and not df_wh.empty and selected_ids.size > 0:
                df_arousal = ana_utils.compute_arousal_groups_from_whisk(
                    spikes,
                    clusters,
                    selected_acronyms,
                    selected_cid_to_idx,
                    selected_ids,
                    wh_events_by_period.get("wh_brief_times_spont", np.array([])),
                    CONFIG_CALC,
                    whisk_times=df_wh["bin_center_s"].to_numpy(dtype=float),
                    whisk_values=df_wh["wh_norm"].to_numpy(dtype=float),
                    spont_intervals=spont_intervals,
                )
            else:
                df_arousal = None

            if df_arousal is not None and not df_arousal.empty and not df_res.empty:
                arousal_cols = [
                    "cluster_id",
                    "arousal_corr",
                    "arousal_corr_h1",
                    "arousal_corr_h2",
                    "arousal_corr_abs",
                    "arousal_mod",
                    "arousal_fr_hz",
                    "arousal_group",
                    "arousal_n_events",
                    "arousal_n_bins",
                ]
                df_res = df_res.merge(df_arousal[arousal_cols], on="cluster_id", how="left")
            else:
                df_res["arousal_corr"] = np.nan
                df_res["arousal_corr_h1"] = np.nan
                df_res["arousal_corr_h2"] = np.nan
                df_res["arousal_corr_abs"] = np.nan
                df_res["arousal_mod"] = np.nan
                df_res["arousal_fr_hz"] = np.nan
                df_res["arousal_group"] = "neutral"
                df_res["arousal_n_events"] = 0
                df_res["arousal_n_bins"] = 0

            arousal_group_mode = str(CONFIG_CALC.get("AROUSAL_GROUP_MODE", "corr")).strip().lower()
            if arousal_group_mode == "response_sign":
                arousal_sign_event = str(CONFIG_CALC.get("AROUSAL_SIGN_EVENT", "wh_brief_times_spont")).strip()
                arousal_sign_col = ana_utils.response_sign_column_name(arousal_sign_event)
                sign_to_group = {"exc": "arousal_plus", "inh": "arousal_minus", "none": "neutral"}
                if arousal_sign_col in df_res.columns:
                    sign_vals = df_res[arousal_sign_col].astype(str).str.lower()
                    df_res["arousal_group"] = sign_vals.map(sign_to_group).fillna("neutral")
            df_res["arousal_group"] = df_res["arousal_group"].fillna("neutral")

            wh_sort_delay_cols = [
                ana_utils.delay_column_name("wh_brief_times_spont"),
                ana_utils.delay_column_name("wh_long_times_spont"),
                ana_utils.delay_column_name("wh_all_times_spont"),
            ]
            df_res = ana_utils.add_wh_delay_sorting(
                df_res,
                wh_sort_delay_cols,
                group_col="arousal_group",
            )

            step = "labels_and_clusters"
            labels = get_cluster_labels_array(clusters)
            label_min = CONFIG_CALC.get("CALC_LABEL_MIN", None)
            strict_gt = bool(CONFIG_CALC.get("CALC_LABEL_STRICT_GT", False))
            coupling_cluster_ids = _strict_good_cluster_ids(
                cluster_ids,
                labels,
                label_min,
                strict_gt=strict_gt,
            )

            labels_float = None
            if labels is not None:
                try:
                    labels_float = np.asarray(labels, dtype=float)
                except (TypeError, ValueError):
                    labels_float = None
            if labels_float is not None and len(labels_float) == len(cluster_ids):
                good_only_cluster_ids = cluster_ids[np.isclose(labels_float, 1.0)]
            elif labels is not None and len(np.asarray(labels)) == len(cluster_ids):
                good_only_cluster_ids = cluster_ids[np.asarray(labels) == 1]
            else:
                good_only_cluster_ids = cluster_ids

            spont_interval_list = []
            if spont_intervals is not None:
                spont_intervals = np.asarray(spont_intervals, dtype=float)
                spont_interval_list = [tuple(row) for row in spont_intervals]

            step = "compute_spont_coupling"
            spont_start = None
            spont_end = None
            df_coupling = None
            df_coupling_task = None
            df_coupling_iti = None
            df_coupling_good = None
            df_coupling_task_good = None
            df_coupling_iti_good = None

            if CONFIG_CALC["CALC_SPONT"] and spont_interval_list:
                spont_start = float(spont_intervals[0][0])
                spont_end = float(spont_intervals[0][1])
                spikes_spont = ana_utils.slice_spikes_by_intervals(spikes, spont_interval_list)
                df_coupling = ana_utils.compute_population_coupling(
                    spikes_spont,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=True,
                    intervals=spont_interval_list,
                    context_label="Spont",
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    df_coupling_good = ana_utils.compute_population_coupling(
                        spikes_spont,
                        clusters,
                        cluster_acronyms_calc,
                        CONFIG_CALC,
                        cluster_ids=good_only_cluster_ids,
                        split_halves=True,
                        intervals=spont_interval_list,
                        context_label="Spont good",
                    )

            if not task_windows.empty:
                task_odd_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy()
                task_even_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy()
            else:
                task_odd_intervals = np.empty((0, 2))
                task_even_intervals = np.empty((0, 2))

            if not iti_windows.empty:
                iti_odd_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy()
                iti_even_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy()
            else:
                iti_odd_intervals = np.empty((0, 2))
                iti_even_intervals = np.empty((0, 2))

            step = "slice_interval_spikes"
            spikes_by_context = {}
            if len(task_odd_intervals) > 0:
                spikes_by_context["task_odd"] = ana_utils.slice_spikes_by_intervals(
                    spikes,
                    task_odd_intervals,
                    exclude_intervals=spont_interval_list,
                )
            if len(task_even_intervals) > 0:
                spikes_by_context["task_even"] = ana_utils.slice_spikes_by_intervals(
                    spikes,
                    task_even_intervals,
                    exclude_intervals=spont_interval_list,
                )
            if len(iti_odd_intervals) > 0:
                spikes_by_context["iti_odd"] = ana_utils.slice_spikes_by_intervals(
                    spikes,
                    iti_odd_intervals,
                    exclude_intervals=spont_interval_list,
                )
            if len(iti_even_intervals) > 0:
                spikes_by_context["iti_even"] = ana_utils.slice_spikes_by_intervals(
                    spikes,
                    iti_even_intervals,
                    exclude_intervals=spont_interval_list,
                )

            step = "task_iti_coupling_parallel"
            coupling_jobs = []

            def _add_context_jobs(ctx_key, intervals_arr, label_base):
                if ctx_key not in spikes_by_context:
                    return
                coupling_jobs.append(
                    {
                        "key": ctx_key,
                        "spikes": spikes_by_context[ctx_key],
                        "clusters": clusters,
                        "cluster_acronyms": cluster_acronyms_calc,
                        "config": CONFIG_CALC,
                        "cluster_ids": coupling_cluster_ids,
                        "split_halves": False,
                        "intervals": intervals_arr,
                        "context_label": label_base,
                    }
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    coupling_jobs.append(
                        {
                            "key": f"{ctx_key}_good",
                            "spikes": spikes_by_context[ctx_key],
                            "clusters": clusters,
                            "cluster_acronyms": cluster_acronyms_calc,
                            "config": CONFIG_CALC,
                            "cluster_ids": good_only_cluster_ids,
                            "split_halves": False,
                            "intervals": intervals_arr,
                            "context_label": f"{label_base} good",
                        }
                    )

            _add_context_jobs("task_odd", task_odd_intervals, "Task odd")
            _add_context_jobs("task_even", task_even_intervals, "Task even")
            _add_context_jobs("iti_odd", iti_odd_intervals, "ITI odd")
            _add_context_jobs("iti_even", iti_even_intervals, "ITI even")

            coupling_parallel_workers = int(CONFIG_CALC.get("COUPLING_PARALLEL_WORKERS", 0))
            coupling_parallel_prefer_process = bool(
                CONFIG_CALC.get("COUPLING_PARALLEL_PREFER_PROCESS", True)
            )
            coupling_outputs = ana_utils.run_population_coupling_jobs(
                coupling_jobs,
                max_workers=coupling_parallel_workers,
                prefer_processes=coupling_parallel_prefer_process,
            )

            df_task_odd = coupling_outputs.get("task_odd")
            df_task_even = coupling_outputs.get("task_even")
            df_task_odd_good = coupling_outputs.get("task_odd_good")
            df_task_even_good = coupling_outputs.get("task_even_good")
            df_iti_odd = coupling_outputs.get("iti_odd")
            df_iti_even = coupling_outputs.get("iti_even")
            df_iti_odd_good = coupling_outputs.get("iti_odd_good")
            df_iti_even_good = coupling_outputs.get("iti_even_good")

            if df_task_odd is not None and df_task_odd.empty:
                df_task_odd = None
            if df_task_even is not None and df_task_even.empty:
                df_task_even = None
            if df_task_odd_good is not None and df_task_odd_good.empty:
                df_task_odd_good = None
            if df_task_even_good is not None and df_task_even_good.empty:
                df_task_even_good = None
            if df_task_odd is not None or df_task_even is not None:
                df_coupling_task = ana_utils.merge_stpr_splits(
                    df_task_odd,
                    df_task_even,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )
            if df_task_odd_good is not None or df_task_even_good is not None:
                df_coupling_task_good = ana_utils.merge_stpr_splits(
                    df_task_odd_good,
                    df_task_even_good,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )

            step = "iti_coupling"
            if df_iti_odd is not None and df_iti_odd.empty:
                df_iti_odd = None
            if df_iti_even is not None and df_iti_even.empty:
                df_iti_even = None
            if df_iti_odd_good is not None and df_iti_odd_good.empty:
                df_iti_odd_good = None
            if df_iti_even_good is not None and df_iti_even_good.empty:
                df_iti_even_good = None
            if df_iti_odd is not None or df_iti_even is not None:
                df_coupling_iti = ana_utils.merge_stpr_splits(
                    df_iti_odd,
                    df_iti_even,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )
            if df_iti_odd_good is not None or df_iti_even_good is not None:
                df_coupling_iti_good = ana_utils.merge_stpr_splits(
                    df_iti_odd_good,
                    df_iti_even_good,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )

            step = "trial_df"
            trial_contrasts = ana_utils.get_trial_contrasts(sl)
            choice_map = {1: "Left", -1: "Right", 0: "NoGo"}
            choices = [choice_map.get(val, "NA") for val in sl.trials["choice"]]
            trial_df = pd.DataFrame(
                {
                    "trial_idx": np.arange(len(sl.trials)),
                    "contrast": trial_contrasts,
                    "reaction_time": sl.trials["response_times"].values
                    - sl.trials["stimOn_times"].values,
                    "correct_response": sl.trials["feedbackType"].values == 1,
                    "subject_response": choices,
                    "stimOn_times": sl.trials["stimOn_times"].values,
                    "firstMovement_times": sl.trials["firstMovement_times"].values,
                    "feedback_times": sl.trials["feedback_times"].values,
                    "response_times": sl.trials["response_times"].values,
                }
            )

            step = "event_response_metrics"
            df_res = _augment_df_res_with_event_response_metrics(
                df_res=df_res,
                spikes=spikes,
                trial_df=trial_df,
                wh_events_by_period=wh_events_by_period,
                config_calc=CONFIG_CALC,
            )
            if isinstance(df_res, pd.DataFrame):
                df_res.to_csv(path_data_processed / f"{pid}_delay_results_dashboard.csv", index=False)

            step = "meta"
            meta = _fetch_session_metadata(one, eid)
            recording_len = float(np.nanmax(spikes["times"]) - np.nanmin(spikes["times"]))
            spont_length = None
            if spont_start is not None and spont_end is not None:
                spont_length = float(spont_end - spont_start)

            meta.update(
                {
                    "pid": pid,
                    "eid": eid,
                    "num_trials": int(len(sl.trials)),
                    "recording_length_s": recording_len,
                    "spont_interval": (spont_start, spont_end),
                    "spont_length_s": spont_length,
                    "num_other_pids": int(len(eid2pid)),
                }
            )

            step = "save_cache"
            cache = {
                "cluster_ids": cluster_ids,
                "cluster_firing_rate": cluster_firing_rate,
                "cluster_acronyms_plot": cluster_acronyms_plot,
                "trials": trial_df,
                "df_res": df_res,
                "df_coupling": df_coupling,
                "df_coupling_task": df_coupling_task,
                "df_coupling_iti": df_coupling_iti,
                "df_coupling_good": df_coupling_good,
                "df_coupling_task_good": df_coupling_task_good,
                "df_coupling_iti_good": df_coupling_iti_good,
                "task_stim_subsets": task_stim_subsets,
                "passive_events": passive_wrappers,
                "df_wh": df_wh,
                "wh_detect": wh_detect,
                "wh_event_base": wh_event_base,
                "wh_events_by_period": wh_events_by_period,
                "meta": meta,
                "config_calc": CONFIG_CALC,
                "config_plot": CONFIG_PLOT,
                "calc_version": CALC_VERSION,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"Saved cache: {cache_path}")

        except Exception as exc:  # pragma: no cover
            print(f"Failed {pid} at step '{step}': {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
