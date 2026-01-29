from pathlib import Path
import pickle
import sys

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    load_pupil_data,
    build_cluster_id_map,
)
import utils.analysis as ana_utils


CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_ONLY_GOOD_UNITS": True,
    "CALC_SPONT": True,
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    "DELAY_METHOD": "center_of_mass",
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -0.2,
    "PSTH_WINDOW_END": 0.35,
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.35,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 50,
    "RELIABILITY_WINDOW_START": 0.01,
    "RELIABILITY_WINDOW_END": 0.15,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 100,
    "STPR_SMOOTH_SIGMA_MS": 5,
    "STPR_POP_USE_GOOD_UNITS": False,
    "COMBINE_PIDS": False,
}

CONFIG_PLOT = {
    "ATLAS_MAPPING": "Beryl",
    "PLOT_ONLY_GOOD_UNITS": True,
    "PLOT_EVENT": "stimOn_times",
    "PLOT_REGIONS": ["VISp", "ENTm"],
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

# Update this list with the PIDs you want to process.
PIDS = [
    "c9664185-d3fd-4e0e-89cf-77c402038938",
]


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


def _get_label_array(clusters):
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            return np.asarray(clusters.metrics.label)
    if hasattr(clusters, "label"):
        return np.asarray(clusters.label)
    if isinstance(clusters, dict) and "label" in clusters:
        return np.asarray(clusters.get("label"))
    return None


def main():
    base_path = Path(__file__).resolve().parents[1]
    path_data, _path_fig, path_data_processed, ibl_cache = setup_paths(base_path)
    cache_dir = path_data / "dashboard_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    one = init_one(ibl_cache)
    ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)

    for pid in tqdm(PIDS, desc="Processing PIDs"):
        try:
            ssl, spikes, clusters, sl = load_session_data(pid, one, ba)
            pupil_features, pupil_times = load_pupil_data(sl)
            eid = ssl.eid
            eid2pid = one.eid2pid(eid)
            if isinstance(eid2pid, tuple):
                eid2pid = eid2pid[0]
            eid2pid = [str(p) for p in eid2pid]

            cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
            cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
            cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

            events_by_name, contrasts_by_name = ana_utils.build_event_dicts(
                sl, CONFIG_CALC["EVENT_NAMES"], CONFIG_CALC["MIN_TRIALS"]
            )

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
            )
            df_reliability = ana_utils.calculate_delay_reliability(
                spikes,
                clusters,
                cluster_acronyms_calc,
                events_by_name,
                contrasts_by_name,
                CONFIG_CALC,
                path_data_processed,
                pid,
                cid_to_idx,
                df_res=df_res,
            )

            spont_intervals = None
            spont_start = None
            spont_end = None
            df_coupling = None
            df_coupling_task = None
            df_comparison = None

            if CONFIG_CALC["CALC_SPONT"]:
                spont_intervals = _load_spontaneous_intervals(one, eid)

            labels = _get_label_array(clusters)
            if labels is not None:
                good_cluster_ids = np.asarray(cluster_ids)[labels == 1]
            else:
                good_cluster_ids = np.asarray(cluster_ids)

            if CONFIG_CALC["CALC_SPONT"] and spont_intervals is not None:
                spont_start = float(spont_intervals[0][0])
                spont_end = float(spont_intervals[0][1])

                valid_time_mask = np.zeros(len(spikes["times"]), dtype=bool)
                for start, end in spont_intervals:
                    valid_time_mask |= (spikes["times"] >= start) & (spikes["times"] <= end)
                spikes_spont = {key: val[valid_time_mask] for key, val in spikes.items()}

                coupling_cluster_ids = (
                    good_cluster_ids if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"] else cluster_ids
                )
                df_coupling = ana_utils.compute_population_coupling(
                    spikes_spont,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=True,
                )

                task_time_mask = spikes["times"] < spont_start
                spikes_task = {key: val[task_time_mask] for key, val in spikes.items()}
                df_coupling_task = ana_utils.compute_population_coupling(
                    spikes_task,
                    clusters,
                    cluster_acronyms_calc,
                    CONFIG_CALC,
                    cluster_ids=coupling_cluster_ids,
                    split_halves=True,
                )

                if df_coupling is not None and df_coupling_task is not None:
                    df_comparison = df_coupling.merge(
                        df_coupling_task[[
                            "cluster_id",
                            "coupling_strength",
                            "coupling_delay_ms",
                            "sorting_number",
                        ]],
                        on="cluster_id",
                        suffixes=("_spont", "_task"),
                    )

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

            session = {
                "trials": sl.trials,
                "wheel": sl.wheel,
                "pose": sl.pose,
            }

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

            cache = {
                "pid": pid,
                "eid": eid,
                "pname": getattr(ssl, "pname", None),
                "cluster_ids": cluster_ids,
                "cluster_acronyms_plot": cluster_acronyms_plot,
                "cluster_acronyms_calc": cluster_acronyms_calc,
                "clusters": clusters,
                "spikes": spikes,
                "session": session,
                "pupil_features": pupil_features,
                "pupil_times": pupil_times,
                "trials": trial_df,
                "df_res": df_res,
                "df_reliability": df_reliability,
                "df_coupling": df_coupling,
                "df_coupling_task": df_coupling_task,
                "df_comparison": df_comparison,
                "eid2pid": eid2pid,
                "spont_intervals": spont_intervals,
                "meta": meta,
                "config_calc": CONFIG_CALC,
                "config_plot": CONFIG_PLOT,
            }

            cache_path = cache_dir / f"{pid}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"Saved cache: {cache_path}")

        except Exception as exc:  # pragma: no cover
            print(f"Failed {pid}: {exc}")


if __name__ == "__main__":
    main()
