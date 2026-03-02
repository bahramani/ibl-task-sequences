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

CALC_VERSION = "whisking-v1.0"

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
    eids, _session_dicts = one.search(
        tag=tag,
        subject=subject,
        details=True,
        query_type="remote",
    )
    pids = []
    for eid in eids:
        insertions = one.alyx.rest("insertions", "list", session=eid)
        pids.extend([ins["id"] for ins in insertions])
    return pids


def _get_pids_for_regions(one, regions, tag):
    if not regions:
        return []
    all_pids = []
    for region in regions:
        eids, _session_dicts = one.search(
            tag=tag,
            details=True,
            query_type="remote",
            atlas_acronym=region,
        )
        for eid in eids:
            insertions = one.alyx.rest("insertions", "list", session=eid)
            all_pids.extend([ins["id"] for ins in insertions])
    return list(dict.fromkeys(all_pids))


def _get_all_tag_pids(one, tag):
    eids, _session_dicts = one.search(
        tag=tag,
        details=True,
        query_type="remote",
    )
    pids = []
    for eid in eids:
        insertions = one.alyx.rest("insertions", "list", session=eid)
        pids.extend([ins["id"] for ins in insertions if ins.get("id")])
    return list(dict.fromkeys(pids))


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
