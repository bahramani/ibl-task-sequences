# %%
from pathlib import Path
import pickle
import traceback

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    build_cluster_id_map,
    get_cluster_labels_array,
)
import utils.analysis as ana_utils

CALC_VERSION = "2026-02-06-v5"

CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_LABEL_MIN": 0.5,
    "CALC_SPONT": True,
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    "DELAY_METHOD": "com",
    "DELAY_UNITS": "ms",
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    "DELAY_WINDOWS": {
        "stimOn_times": (0.02, 0.35),
        "firstMovement_times": (-0.1, 0.2),
        "response_times": (-0.1, 0.2),
        "feedback_times": (-0.1, 0.2),
    },
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -1,
    "PSTH_WINDOW_END": 1,
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.35,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 50,
    "MIN_TRIALS_SPLIT": 25,
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
    "PLOT_REGIONS": None, ############
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
PIDS = ['afe87fbb-3a17-461f-b333-e22903f1d70d', '3eb6e6e0-8a57-49d6-b7c9-f39d5834e682',
        'eebcaf65-7fa4-4118-869d-a084e84530e2']
SUBJECT = None # "CSH_ZAD_029"
# If SUBJECT is None, use REGIONS to find all PIDs from sessions with those atlas acronyms.
REGIONS = None # ["VISp", "AUDpo", "MOs", "GRN", "ZI", "SCm"] 
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
            return np.array([[spont[0], spont[1]]])
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
    # De-duplicate, keep stable order
    return list(dict.fromkeys(all_pids))


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


def main():
    base_path = Path(__file__).resolve().parents[1]
    path_data, _path_fig, path_data_processed, ibl_cache = setup_paths(base_path)
    cache_dir = path_data / "dashboard_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    one = init_one(ibl_cache)
    ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)

    if PIDS:
        pids_to_process = list(PIDS)
    elif SUBJECT is not None:
        pids_to_process = _get_pids_for_subject(one, SUBJECT, TAG)
    else:
        pids_to_process = _get_pids_for_regions(one, REGIONS, TAG)
    if not pids_to_process:
        raise RuntimeError("No PIDs found. Set PIDS, SUBJECT, or REGIONS.")

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
                load_wheel=False,
                load_pose=False,
            )
            eid = ssl.eid
            eid2pid = one.eid2pid(eid)
            if isinstance(eid2pid, tuple):
                eid2pid = eid2pid[0]
            eid2pid = [str(p) for p in eid2pid]

            step = "cluster_setup"
            cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
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

            step = "build_event_dicts"
            events_by_name, contrasts_by_name, trial_idx_by_name = ana_utils.build_event_dicts(
                sl,
                CONFIG_CALC["EVENT_NAMES"],
                CONFIG_CALC["MIN_TRIALS"],
                return_trial_idx=True,
            )

            step = "calculate_delays"
            df_res = ana_utils.calculate_delays(
                spikes,
                clusters,
                cluster_acronyms_calc,
                events_by_name,
                contrasts_by_name,
                CONFIG_CALC,
                path_data_processed,
                pid,
                cid_to_idx,
                trial_idx_by_name=trial_idx_by_name,
            )
            if CONFIG_CALC.get("DELAY_UNITS", "s").lower().startswith("ms") and df_res is not None:
                delay_cols = []
                for event_name in CONFIG_CALC.get("EVENT_NAMES", []):
                    delay_cols.append(ana_utils.delay_column_name(event_name))
                    delay_cols.append(ana_utils.delay_split_column_name(event_name, "odd"))
                    delay_cols.append(ana_utils.delay_split_column_name(event_name, "even"))
                for col in delay_cols:
                    if col in df_res.columns:
                        df_res[col] = df_res[col].astype(float) * 1000.0

            spont_intervals = None
            spont_start = None
            spont_end = None
            df_coupling = None
            df_coupling_task = None
            df_coupling_task_tf = None
            df_coupling_iti = None
            df_coupling_good = None
            df_coupling_task_good = None
            df_coupling_iti_good = None
            df_comparison = None

            step = "load_spont_intervals"
            if CONFIG_CALC["CALC_SPONT"]:
                spont_intervals = _load_spontaneous_intervals(one, eid)

            step = "labels_and_clusters"
            labels = get_cluster_labels_array(clusters)
            label_min = CONFIG_CALC.get("CALC_LABEL_MIN", None)
            if label_min is None and CONFIG_CALC.get("CALC_ONLY_GOOD_UNITS", False):
                label_min = 1.0
            labels_float = None
            if labels is not None:
                try:
                    labels_float = labels.astype(float)
                except (TypeError, ValueError):
                    labels_float = None
            if labels_float is not None and label_min is not None:
                good_cluster_ids = np.asarray(cluster_ids)[labels_float >= float(label_min)]
            elif labels is not None and label_min is not None:
                good_cluster_ids = np.asarray(cluster_ids)[labels == 1]
            else:
                good_cluster_ids = np.asarray(cluster_ids)

            if labels_float is not None:
                good_only_cluster_ids = np.asarray(cluster_ids)[np.isclose(labels_float, 1.0)]
            elif labels is not None:
                good_only_cluster_ids = np.asarray(cluster_ids)[labels == 1]
            else:
                good_only_cluster_ids = np.asarray(cluster_ids)

            coupling_cluster_ids = good_cluster_ids if label_min is not None else cluster_ids

            spont_interval_list = []
            if spont_intervals is not None:
                spont_intervals = np.asarray(spont_intervals, dtype=float)
                spont_interval_list = [tuple(row) for row in spont_intervals]

            step = "compute_spont_stpr"
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

            step = "task_windows"
            task_windows = ana_utils.build_task_window_table(
                sl.trials,
                CONFIG_CALC["EVENT_NAMES"],
                post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
            )
            if not task_windows.empty:
                task_odd_intervals = task_windows.loc[
                    task_windows["odd"], ["start", "end"]
                ].to_numpy()
                task_even_intervals = task_windows.loc[
                    ~task_windows["odd"], ["start", "end"]
                ].to_numpy()
                task_true_intervals = task_windows.loc[
                    task_windows["correct"], ["start", "end"]
                ].to_numpy()
                task_false_intervals = task_windows.loc[
                    ~task_windows["correct"], ["start", "end"]
                ].to_numpy()
            else:
                task_odd_intervals = np.empty((0, 2))
                task_even_intervals = np.empty((0, 2))
                task_true_intervals = np.empty((0, 2))
                task_false_intervals = np.empty((0, 2))

            step = "task_odd_even_stpr"
            df_task_odd = None
            df_task_even = None
            df_task_odd_good = None
            df_task_even_good = None
            if len(task_odd_intervals) > 0:
                spikes_task_odd = ana_utils.slice_spikes_by_intervals(
                    spikes, task_odd_intervals, exclude_intervals=spont_interval_list
                )
                df_task_odd = ana_utils.compute_population_coupling(
                    spikes_task_odd,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=task_odd_intervals,
                    context_label="Task odd",
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    df_task_odd_good = ana_utils.compute_population_coupling(
                        spikes_task_odd,
                        clusters,
                        cluster_acronyms_calc,
                        CONFIG_CALC,
                        cluster_ids=good_only_cluster_ids,
                        split_halves=False,
                        intervals=task_odd_intervals,
                        context_label="Task odd good",
                    )
            if len(task_even_intervals) > 0:
                spikes_task_even = ana_utils.slice_spikes_by_intervals(
                    spikes, task_even_intervals, exclude_intervals=spont_interval_list
                )
                df_task_even = ana_utils.compute_population_coupling(
                    spikes_task_even,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=task_even_intervals,
                    context_label="Task even",
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    df_task_even_good = ana_utils.compute_population_coupling(
                        spikes_task_even,
                        clusters,
                        cluster_acronyms_calc,
                        CONFIG_CALC,
                        cluster_ids=good_only_cluster_ids,
                        split_halves=False,
                        intervals=task_even_intervals,
                        context_label="Task even good",
                    )
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
                    df_task_odd, df_task_even, CONFIG_CALC, split_a="odd", split_b="even"
                )
            if df_task_odd_good is not None or df_task_even_good is not None:
                df_coupling_task_good = ana_utils.merge_stpr_splits(
                    df_task_odd_good,
                    df_task_even_good,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )

            step = "task_true_false_stpr"
            df_task_true = None
            df_task_false = None
            if len(task_true_intervals) > 0:
                spikes_task_true = ana_utils.slice_spikes_by_intervals(
                    spikes, task_true_intervals, exclude_intervals=spont_interval_list
                )
                df_task_true = ana_utils.compute_population_coupling(
                    spikes_task_true,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=task_true_intervals,
                    context_label="Task true",
                )
            if len(task_false_intervals) > 0:
                spikes_task_false = ana_utils.slice_spikes_by_intervals(
                    spikes, task_false_intervals, exclude_intervals=spont_interval_list
                )
                df_task_false = ana_utils.compute_population_coupling(
                    spikes_task_false,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=task_false_intervals,
                    context_label="Task false",
                )
            if df_task_true is not None and df_task_true.empty:
                df_task_true = None
            if df_task_false is not None and df_task_false.empty:
                df_task_false = None
            if df_task_true is not None or df_task_false is not None:
                df_coupling_task_tf = ana_utils.merge_stpr_splits(
                    df_task_true, df_task_false, CONFIG_CALC, split_a="true", split_b="false"
                )

            step = "iti_windows"
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
            if not iti_windows.empty:
                iti_odd_intervals = iti_windows.loc[
                    iti_windows["odd"], ["start", "end"]
                ].to_numpy()
                iti_even_intervals = iti_windows.loc[
                    ~iti_windows["odd"], ["start", "end"]
                ].to_numpy()
            else:
                iti_odd_intervals = np.empty((0, 2))
                iti_even_intervals = np.empty((0, 2))

            step = "iti_stpr"
            df_iti_odd = None
            df_iti_even = None
            df_iti_odd_good = None
            df_iti_even_good = None
            if len(iti_odd_intervals) > 0:
                spikes_iti_odd = ana_utils.slice_spikes_by_intervals(
                    spikes, iti_odd_intervals, exclude_intervals=spont_interval_list
                )
                df_iti_odd = ana_utils.compute_population_coupling(
                    spikes_iti_odd,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=iti_odd_intervals,
                    context_label="ITI odd",
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    df_iti_odd_good = ana_utils.compute_population_coupling(
                        spikes_iti_odd,
                        clusters,
                        cluster_acronyms_calc,
                        CONFIG_CALC,
                        cluster_ids=good_only_cluster_ids,
                        split_halves=False,
                        intervals=iti_odd_intervals,
                        context_label="ITI odd good",
                    )
            if len(iti_even_intervals) > 0:
                spikes_iti_even = ana_utils.slice_spikes_by_intervals(
                    spikes, iti_even_intervals, exclude_intervals=spont_interval_list
                )
                df_iti_even = ana_utils.compute_population_coupling(
                    spikes_iti_even,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=False,
                    intervals=iti_even_intervals,
                    context_label="ITI even",
                )
                if good_only_cluster_ids is not None and len(good_only_cluster_ids) > 0:
                    df_iti_even_good = ana_utils.compute_population_coupling(
                        spikes_iti_even,
                        clusters,
                        cluster_acronyms_calc,
                        CONFIG_CALC,
                        cluster_ids=good_only_cluster_ids,
                        split_halves=False,
                        intervals=iti_even_intervals,
                        context_label="ITI even good",
                    )
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
                    df_iti_odd, df_iti_even, CONFIG_CALC, split_a="odd", split_b="even"
                )
            if df_iti_odd_good is not None or df_iti_even_good is not None:
                df_coupling_iti_good = ana_utils.merge_stpr_splits(
                    df_iti_odd_good,
                    df_iti_even_good,
                    CONFIG_CALC,
                    split_a="odd",
                    split_b="even",
                )

            step = "comparison_table"
            if df_coupling is not None and df_coupling_task is not None:
                df_comparison = df_coupling.merge(
                    df_coupling_task[
                        [
                            "cluster_id",
                            "coupling_strength",
                            "coupling_delay_ms",
                            "sorting_number",
                        ]
                    ],
                    on="cluster_id",
                    suffixes=("_spont", "_task"),
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
                "pid": pid,
                "eid": eid,
                "pname": getattr(ssl, "pname", None),
                "cluster_ids": cluster_ids,
                "cluster_firing_rate": cluster_firing_rate,
                "cluster_acronyms_plot": cluster_acronyms_plot,
                "cluster_acronyms_calc": cluster_acronyms_calc,
                "trials": trial_df,
                "df_res": df_res,
                "df_coupling": df_coupling,
                "df_coupling_task": df_coupling_task,
                "df_coupling_task_tf": df_coupling_task_tf,
                "df_coupling_iti": df_coupling_iti,
                "df_coupling_good": df_coupling_good,
                "df_coupling_task_good": df_coupling_task_good,
                "df_coupling_iti_good": df_coupling_iti_good,
                "df_comparison": df_comparison,
                "eid2pid": eid2pid,
                "spont_intervals": spont_intervals,
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
