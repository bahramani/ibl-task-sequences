import os
from pathlib import Path
import pandas as pd
import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

def setup_paths(base_path):
    """Create the main project folders and return their paths."""
    path_data = base_path / "data"
    path_fig = base_path / "results" / "figures"
    path_data_processed = path_data / "processed"
    ibl_cache = path_data / "raw"
    for p in [ibl_cache, path_fig, path_data_processed]:
        p.mkdir(exist_ok=True, parents=True)
    return path_data, path_fig, path_data_processed, ibl_cache


def init_one(ibl_cache, mode="remote", base_url="https://openalyx.internationalbrainlab.org", silent=True):
    """Initialize the ONE API with a custom cache directory."""
    try:
        one = ONE(
            base_url=base_url,
            password="international",
            silent=silent,
            mode=mode,
            cache_dir=ibl_cache,
        )
        print(f"ONE API initialized (mode={mode}).")
        return one
    except Exception as exc:
        raise RuntimeError(f"Error initializing ONE: {exc}")


def prepare_region_dirs(path_data):
    """Create per-region output folders for Beryl acronyms."""
    ba = AllenAtlas()
    br = BrainRegions()
    beryl_indices = np.unique(br.mappings["Beryl"])
    beryl_acronyms = br.acronym[beryl_indices]
    beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["void"]))
    beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["root"]))
    for region in beryl_acronyms:
        os.makedirs(f"{path_data}/session_plots_integrated/{region}/", exist_ok=True)
    
    hier_file = os.path.join(path_data, "hierarchy_summary_CreConf_all_regions.csv")
    hier_df = pd.read_csv(hier_file)
    area_to_hier_score = dict(zip(hier_df["areas"], hier_df["CC+TC+CT iterated"]))
    hier_scores = np.array([area_to_hier_score.get(region, np.nan) for region in beryl_acronyms])

    return ba, br, beryl_acronyms, hier_scores

def map_acronyms(clusters, br, mapping):
    """Map cluster acronyms to the requested atlas (Beryl or Allen)."""
    if mapping == "Beryl":
        return br.acronym2acronym(clusters.acronym, mapping="Beryl")
    return clusters.acronym


def load_session_data(
    pid,
    one,
    ba=None,
    load_trials=True,
    load_wheel=True,
    load_pose=True,
    load_motion_energy=False,
    load_pupil=False,
    pose_views=("left", "right"),
    motion_energy_views=("left", "right"),
):
    """Load spikes, clusters, and session data for a given probe insertion."""
    if ba is None:
        ba = AllenAtlas()
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    eid, _ = one.pid2eid(pid)
    print(f"Session ID (EID): {eid}")
    print(f"Probe Name: {ssl.pname}")

    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    print(f"Spikes loaded: {spikes.times.shape[0]} spikes")
    if "acronym" in clusters:
        print(f"Cluster regions found: {set(clusters.acronym)}")

    sl = SessionLoader(
        eid=eid,
        one=one,
        trials=load_trials,
        wheel=load_wheel,
        pose=load_pose,
        motion_energy=load_motion_energy,
        pupil=load_pupil,
    )
    if load_trials:
        sl.load_trials()
        print(f"Trials loaded. Found keys: {list(sl.trials.keys())}")
    else:
        sl.trials = None
    if load_wheel:
        sl.load_wheel()
        print(f"Wheel data loaded. Found keys: {list(sl.wheel.keys())}")
    else:
        sl.wheel = None
    if load_pose:
        views = list(pose_views) if pose_views is not None else None
        if views is None:
            sl.load_pose()
        else:
            sl.load_pose(views=views)
        print(f"Pose data loaded. Found keys: {list(sl.pose.keys())}")
    else:
        sl.pose = None

    if load_motion_energy:
        try:
            views = list(motion_energy_views) if motion_energy_views is not None else None
            if views is None:
                sl.load_motion_energy()
            else:
                sl.load_motion_energy(views=views)
            print(
                "Motion energy data loaded. "
                f"Found keys: {list(getattr(sl, 'motion_energy', {}).keys())}"
            )
        except Exception as exc:
            sl.motion_energy = None
            print(f"Motion energy data not available: {exc}")
    else:
        sl.motion_energy = None

    if load_pupil:
        try:
            sl.load_pupil()
            pupil_cols = None
            if isinstance(sl.pupil, pd.DataFrame):
                pupil_cols = list(sl.pupil.columns)
            print(
                "Pupil data loaded. "
                f"Columns: {pupil_cols if pupil_cols is not None else 'NA'}"
            )
        except Exception as exc:
            sl.pupil = None
            print(f"Pupil data not available: {exc}")
    else:
        sl.pupil = None

    return ssl, spikes, clusters, sl


def load_pupil_data(sl):
    """Load pupil features and times from the left camera, if available."""
    try:
        left_camera = sl.one.load_object(sl.eid, "leftCamera", collection="alf")
        if hasattr(left_camera, "features") and hasattr(left_camera, "times"):
            print("Camera data loaded successfully.")
            return left_camera.features, left_camera.times
        print("Camera object loaded but 'features' or 'times' attribute missing.")
    except Exception as exc:
        print(f"Could not load pupil data: {exc}")
    return None, None

def build_cluster_id_map(clusters):
    """Return (cluster_ids, cid_to_idx) for safe cluster-id indexing."""
    if hasattr(clusters, "cluster_id"):
        cluster_ids = np.asarray(clusters.cluster_id)
    elif "cluster_id" in clusters:
        cluster_ids = np.asarray(clusters["cluster_id"])
    else:
        cluster_ids = np.arange(len(clusters.acronym))
    cid_to_idx = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
    return cluster_ids, cid_to_idx


def get_cluster_label(clusters, idx):
    """Fetch a unit quality label using a safe cluster index."""
    if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
        return clusters.metrics.label[idx]
    if hasattr(clusters, "label"):
        return clusters.label[idx]
    return 1


def get_cluster_labels_array(clusters):
    """Return the full label array if available, else None."""
    if clusters is None:
        return None
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            return np.asarray(clusters.metrics.label)
    if hasattr(clusters, "label"):
        return np.asarray(clusters.label)
    if isinstance(clusters, dict) and "label" in clusters:
        return np.asarray(clusters.get("label"))
    return None


def _extract_tr_field(tr_obj, keys, suffixes=None):
    if tr_obj is None:
        return None
    if hasattr(tr_obj, "keys"):
        key_list = list(tr_obj.keys())
        for key in keys:
            if key in tr_obj:
                return np.asarray(tr_obj[key])
        if suffixes:
            for key in key_list:
                key_str = str(key)
                for suffix in suffixes:
                    if key_str.endswith(suffix):
                        return np.asarray(tr_obj[key])
    for key in keys:
        if hasattr(tr_obj, key):
            return np.asarray(getattr(tr_obj, key))
    if suffixes:
        for suffix in suffixes:
            if hasattr(tr_obj, suffix):
                return np.asarray(getattr(tr_obj, suffix))
    return None


def extract_passive_times_and_contrast(tr_obj):
    if tr_obj is None:
        return None, None

    base_obj = tr_obj
    if isinstance(tr_obj, dict) and "table" in tr_obj:
        base_obj = tr_obj["table"]
    elif isinstance(tr_obj, (list, tuple)):
        if len(tr_obj) == 1:
            base_obj = tr_obj[0]
        else:
            base_obj = None
            for item in tr_obj:
                if isinstance(item, pd.DataFrame) and {"start", "contrast"}.issubset(
                    item.columns
                ):
                    base_obj = item
                    break
                if hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    if {"start", "contrast"}.issubset(set(item.dtype.names)):
                        base_obj = item
                        break
            if base_obj is None and len(tr_obj) > 0:
                base_obj = tr_obj[0]

    if isinstance(base_obj, pd.DataFrame):
        if "start" in base_obj.columns:
            times = base_obj["start"].to_numpy(dtype=float)
            contrasts = (
                base_obj["contrast"].to_numpy(dtype=float)
                if "contrast" in base_obj.columns
                else np.ones_like(times, dtype=float)
            )
            return times, contrasts
    if hasattr(base_obj, "dtype") and getattr(base_obj.dtype, "names", None):
        names = set(base_obj.dtype.names)
        if "start" in names:
            times = np.asarray(base_obj["start"], dtype=float)
            if "contrast" in names:
                contrasts = np.asarray(base_obj["contrast"], dtype=float)
            else:
                contrasts = np.ones_like(times, dtype=float)
            return times, contrasts

    arr_candidate = np.asarray(base_obj)
    if arr_candidate.ndim == 2 and arr_candidate.shape[1] >= 5:
        times = np.asarray(arr_candidate[:, 0], dtype=float)
        contrasts = np.asarray(arr_candidate[:, 3], dtype=float)
        return times, contrasts

    times = _extract_tr_field(
        base_obj,
        keys=("stimOn_times", "times", "stimOn", "onset_times", "event_times", "start"),
        suffixes=(".times", "times", ".start", "start"),
    )
    if times is None and isinstance(base_obj, (list, tuple, np.ndarray)):
        arr = np.asarray(base_obj)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            times = arr

    contrasts = _extract_tr_field(
        base_obj,
        keys=("contrast", "contrasts", "stimContrast"),
        suffixes=(".contrast", "contrast"),
    )
    if contrasts is None:
        contrast_left = _extract_tr_field(
            base_obj,
            keys=("contrastLeft",),
            suffixes=(".contrastLeft", "contrastLeft"),
        )
        contrast_right = _extract_tr_field(
            base_obj,
            keys=("contrastRight",),
            suffixes=(".contrastRight", "contrastRight"),
        )
        if contrast_left is not None or contrast_right is not None:
            if contrast_left is None:
                contrast_left = np.full_like(contrast_right, np.nan, dtype=float)
            if contrast_right is None:
                contrast_right = np.full_like(contrast_left, np.nan, dtype=float)
            contrasts = np.nanmax(
                np.vstack([np.abs(contrast_left), np.abs(contrast_right)]), axis=0
            )

    if times is None:
        return None, None

    times = np.asarray(times, dtype=float)
    if contrasts is None:
        contrasts = np.ones_like(times, dtype=float)
    else:
        contrasts = np.asarray(contrasts, dtype=float)
        if contrasts.shape[0] != times.shape[0]:
            if contrasts.size == 1:
                contrasts = np.full_like(times, float(contrasts.ravel()[0]))
            else:
                contrasts = np.ones_like(times, dtype=float)

    finite_mask = np.isfinite(times)
    if not np.all(finite_mask):
        times = times[finite_mask]
        contrasts = contrasts[finite_mask]

    return times, contrasts


def coerce_passive_stims_table(tr_obj):
    if tr_obj is None:
        return None
    base_obj = tr_obj
    if isinstance(tr_obj, dict) and "table" in tr_obj:
        base_obj = tr_obj["table"]
    elif isinstance(tr_obj, (list, tuple)):
        if len(tr_obj) == 1:
            base_obj = tr_obj[0]
        else:
            base_obj = None
            for item in tr_obj:
                if isinstance(item, pd.DataFrame) and {"toneOn", "noiseOn"}.issubset(
                    item.columns
                ):
                    base_obj = item
                    break
                if hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    if {"toneOn", "noiseOn"}.issubset(set(item.dtype.names)):
                        base_obj = item
                        break
            if base_obj is None and len(tr_obj) > 0:
                base_obj = tr_obj[0]

    if isinstance(base_obj, pd.DataFrame):
        return base_obj.copy()

    if isinstance(base_obj, dict):
        try:
            return pd.DataFrame(base_obj)
        except Exception:
            return None

    if hasattr(base_obj, "dtype") and getattr(base_obj.dtype, "names", None):
        try:
            return pd.DataFrame(
                {name: np.asarray(base_obj[name]) for name in base_obj.dtype.names}
            )
        except Exception:
            return None

    arr = np.asarray(base_obj)
    if arr.ndim == 2 and arr.shape[1] >= 6:
        columns = ["valveOn", "valveOff", "toneOn", "toneOff", "noiseOn", "noiseOff"]
        return pd.DataFrame(arr[:, :6], columns=columns)
    return None


def extract_passive_stim_times(table, column_name):
    if table is None or not hasattr(table, "columns"):
        return None, None, None
    if column_name not in table.columns:
        match = None
        for col in table.columns:
            if str(col).endswith(column_name):
                match = col
                break
        if match is None:
            return None, None, None
        column_name = match
    times_all = np.asarray(table[column_name], dtype=float)
    valid_mask = np.isfinite(times_all)
    return times_all[valid_mask], np.nonzero(valid_mask)[0], column_name


def load_task_replay_datasets(eid_val, one_local, one_remote=None, allow_remote=True):
    if eid_val is None or one_local is None:
        return None, None

    def _load(pattern):
        try:
            return one_local.load_dataset(eid_val, pattern, collection="alf")
        except Exception:
            if not allow_remote or one_remote is None:
                return None
            try:
                return one_remote.load_dataset(eid_val, pattern, collection="alf")
            except Exception:
                return None

    visual_tr = _load("*passiveGabor*")
    auditory_tr = _load("*passiveStims*")
    return visual_tr, auditory_tr


def build_passive_visual_contrast_events(
    visual_tr=None,
    include_zero_contrast=False,
    preferred_contrasts=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.0),
    round_decimals=6,
):
    """Return passive visual onset times grouped by contrast."""
    visual_times, visual_contrasts = extract_passive_times_and_contrast(visual_tr)
    if visual_times is None or visual_contrasts is None:
        return {}

    times_arr = np.asarray(visual_times, dtype=float)
    contrasts_arr = np.asarray(visual_contrasts, dtype=float)
    valid_mask = np.isfinite(times_arr) & np.isfinite(contrasts_arr)
    if not include_zero_contrast:
        valid_mask &= contrasts_arr > 0
    times_arr = times_arr[valid_mask]
    contrasts_arr = contrasts_arr[valid_mask]
    if len(times_arr) == 0:
        return {}

    if round_decimals is not None:
        contrasts_arr = np.round(contrasts_arr, int(round_decimals))
        tol = 10.0 ** (-int(round_decimals))
    else:
        tol = 1e-9

    unique_contrasts = np.unique(contrasts_arr.astype(float))
    unique_contrasts = np.asarray(unique_contrasts, dtype=float)
    ordered_contrasts = []
    for preferred in preferred_contrasts or []:
        if np.any(np.isclose(unique_contrasts, float(preferred), atol=tol)):
            ordered_contrasts.append(float(preferred))
    for contrast in np.sort(unique_contrasts)[::-1]:
        if not np.any(np.isclose(contrast, np.asarray(ordered_contrasts), atol=tol)):
            ordered_contrasts.append(float(contrast))

    events_by_contrast = {}
    for contrast in ordered_contrasts:
        mask = np.isclose(contrasts_arr, contrast, atol=tol)
        contrast_times = np.asarray(times_arr[mask], dtype=float)
        if len(contrast_times) == 0:
            continue
        events_by_contrast[float(contrast)] = np.sort(contrast_times)
    return events_by_contrast


def build_passive_auditory_event_times(auditory_tr=None):
    """Return passive auditory onset times for valve/tone/noise."""
    events = {}
    passive_table = coerce_passive_stims_table(auditory_tr)
    for stim_key, stim_col in (
        ("valve", "valveOn"),
        ("tone", "toneOn"),
        ("noise", "noiseOn"),
    ):
        stim_times, _stim_idx, _stim_col = extract_passive_stim_times(
            passive_table, stim_col
        )
        if stim_times is not None and len(stim_times) > 0:
            events[stim_key] = np.sort(np.asarray(stim_times, dtype=float))
    return events


def build_passive_event_times(visual_tr=None, auditory_tr=None):
    events = {}

    visual_by_contrast = build_passive_visual_contrast_events(
        visual_tr, include_zero_contrast=False
    )
    if visual_by_contrast:
        visual_times = np.concatenate(
            [np.asarray(v, dtype=float) for v in visual_by_contrast.values()]
        )
        events["passive_visual"] = np.sort(visual_times)

    auditory_events = build_passive_auditory_event_times(auditory_tr)
    if "valve" in auditory_events:
        events["passive_valve"] = np.asarray(auditory_events["valve"], dtype=float)
    if "tone" in auditory_events:
        events["passive_tone"] = np.asarray(auditory_events["tone"], dtype=float)
    if "noise" in auditory_events:
        events["passive_noise"] = np.asarray(auditory_events["noise"], dtype=float)

    return events


def _coerce_passive_base_obj(tr_obj):
    if tr_obj is None:
        return None
    base_obj = tr_obj
    if isinstance(tr_obj, dict) and "table" in tr_obj:
        base_obj = tr_obj["table"]
    elif isinstance(tr_obj, (list, tuple)):
        if len(tr_obj) == 1:
            base_obj = tr_obj[0]
        elif len(tr_obj) > 1:
            preferred = None
            for item in tr_obj:
                if isinstance(item, pd.DataFrame) and "start" in item.columns:
                    preferred = item
                    break
                if hasattr(item, "dtype") and getattr(item.dtype, "names", None):
                    if "start" in set(item.dtype.names):
                        preferred = item
                        break
            base_obj = preferred if preferred is not None else tr_obj[0]
    return base_obj


def extract_passive_visual_times_contrast_side(tr_obj):
    """
    Return passive visual event times with absolute contrast and inferred side.

    Output side values are one of: ``left``, ``right``, ``unknown``.
    """
    times, contrasts = extract_passive_times_and_contrast(tr_obj)
    if times is None or contrasts is None:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=object),
        )

    times = np.asarray(times, dtype=float).reshape(-1)
    contrasts = np.asarray(contrasts, dtype=float).reshape(-1)
    if contrasts.shape[0] != times.shape[0]:
        if contrasts.size == 1 and times.size > 0:
            contrasts = np.full(times.shape[0], float(contrasts.ravel()[0]), dtype=float)
        else:
            contrasts = np.ones_like(times, dtype=float)

    side = np.full(times.shape[0], "unknown", dtype=object)
    base_obj = _coerce_passive_base_obj(tr_obj)
    contrast_left = _extract_tr_field(
        base_obj,
        keys=("contrastLeft",),
        suffixes=(".contrastLeft", "contrastLeft"),
    )
    contrast_right = _extract_tr_field(
        base_obj,
        keys=("contrastRight",),
        suffixes=(".contrastRight", "contrastRight"),
    )

    left_arr = None
    right_arr = None
    if contrast_left is not None:
        left_arr = np.asarray(contrast_left, dtype=float).reshape(-1)
    if contrast_right is not None:
        right_arr = np.asarray(contrast_right, dtype=float).reshape(-1)

    if left_arr is not None or right_arr is not None:
        if left_arr is None and right_arr is not None:
            left_arr = np.full(right_arr.shape[0], np.nan, dtype=float)
        if right_arr is None and left_arr is not None:
            right_arr = np.full(left_arr.shape[0], np.nan, dtype=float)
        if (
            left_arr is not None
            and right_arr is not None
            and left_arr.shape[0] == times.shape[0]
        ):
            left_abs = np.abs(left_arr)
            right_abs = np.abs(right_arr)
            left_on = np.isfinite(left_abs) & (left_abs > 0)
            right_on = np.isfinite(right_abs) & (right_abs > 0)
            side[left_on & ~right_on] = "left"
            side[right_on & ~left_on] = "right"
            both_on = left_on & right_on
            side[both_on & (left_abs > right_abs)] = "left"
            side[both_on & (right_abs > left_abs)] = "right"

    if not np.any((side == "left") | (side == "right")):
        side_hint = _extract_tr_field(
            base_obj,
            keys=(
                "position",
                "stim_position",
                "stimPos",
                "azimuth",
                "side",
                "stim_side",
                "stimSide",
                "is_right",
                "isRight",
            ),
            suffixes=(
                ".position",
                "position",
                ".azimuth",
                "azimuth",
                ".side",
                "side",
                ".stim_side",
                "stim_side",
            ),
        )
        if side_hint is not None:
            side_hint = np.asarray(side_hint).reshape(-1)
            if side_hint.shape[0] == times.shape[0]:
                if np.issubdtype(side_hint.dtype, np.number):
                    side_hint_num = side_hint.astype(float)
                    side[side_hint_num < 0] = "left"
                    side[side_hint_num > 0] = "right"
                else:
                    side_hint_str = pd.Series(side_hint).astype(str).str.lower().to_numpy()
                    right_mask = (
                        (side_hint_str == "right")
                        | (side_hint_str == "r")
                        | (side_hint_str == "1")
                        | (side_hint_str == "true")
                    )
                    left_mask = (
                        (side_hint_str == "left")
                        | (side_hint_str == "l")
                        | (side_hint_str == "-1")
                        | (side_hint_str == "false")
                    )
                    side[left_mask] = "left"
                    side[right_mask] = "right"

    if not np.any((side == "left") | (side == "right")):
        signed_mask = np.isfinite(contrasts)
        if np.any(contrasts[signed_mask] < 0) and np.any(contrasts[signed_mask] > 0):
            side[contrasts < 0] = "left"
            side[contrasts > 0] = "right"

    contrast_abs = np.abs(contrasts)
    finite_mask = np.isfinite(times) & np.isfinite(contrast_abs)
    return times[finite_mask], contrast_abs[finite_mask], side[finite_mask]


def top_nonzero_contrasts(values, top_n=2, round_decimals=6):
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.array([], dtype=float)
    vals = np.round(vals, int(round_decimals))
    unique_vals = np.unique(vals.astype(float))
    unique_vals = np.sort(unique_vals)[::-1]
    return unique_vals[: int(top_n)]


def select_task_stim_events_by_side(sl, top_n=2, round_decimals=6):
    """
    Build task Stim-On subsets used in dashboard heatmaps.

    Returns keys:
    - ``task_all_right_times`` / ``task_all_left_times``
    - ``task_top2_right_times`` / ``task_top2_left_times``
    - ``task_zero_lr_times``
    and matching ``*_values`` arrays for contrast summaries.
    """
    trials = getattr(sl, "trials", None)
    if trials is None:
        return {
            "task_all_right_times": np.array([], dtype=float),
            "task_all_left_times": np.array([], dtype=float),
            "task_top2_right_times": np.array([], dtype=float),
            "task_top2_left_times": np.array([], dtype=float),
            "task_zero_lr_times": np.array([], dtype=float),
            "task_all_right_values": np.array([], dtype=float),
            "task_all_left_values": np.array([], dtype=float),
            "task_top2_right_values": np.array([], dtype=float),
            "task_top2_left_values": np.array([], dtype=float),
        }

    stim_times = np.asarray(trials["stimOn_times"], dtype=float).reshape(-1)
    contrast_left = np.asarray(trials["contrastLeft"], dtype=float).reshape(-1)
    contrast_right = np.asarray(trials["contrastRight"], dtype=float).reshape(-1)
    if (
        stim_times.shape[0] != contrast_left.shape[0]
        or stim_times.shape[0] != contrast_right.shape[0]
    ):
        return {
            "task_all_right_times": np.array([], dtype=float),
            "task_all_left_times": np.array([], dtype=float),
            "task_top2_right_times": np.array([], dtype=float),
            "task_top2_left_times": np.array([], dtype=float),
            "task_zero_lr_times": np.array([], dtype=float),
            "task_all_right_values": np.array([], dtype=float),
            "task_all_left_values": np.array([], dtype=float),
            "task_top2_right_values": np.array([], dtype=float),
            "task_top2_left_values": np.array([], dtype=float),
        }

    left_abs = np.abs(contrast_left)
    right_abs = np.abs(contrast_right)
    stim_valid = np.isfinite(stim_times)
    left_pos = np.isfinite(left_abs) & (left_abs > 0)
    right_pos = np.isfinite(right_abs) & (right_abs > 0)
    left_only = left_pos & (~right_pos)
    right_only = right_pos & (~left_pos)
    both_pos = left_pos & right_pos
    left_only = left_only | (both_pos & (left_abs > right_abs))
    right_only = right_only | (both_pos & (right_abs > left_abs))

    tol = 10.0 ** (-int(round_decimals))
    left_zero = np.isfinite(left_abs) & np.isclose(left_abs, 0.0, atol=tol)
    right_zero = np.isfinite(right_abs) & np.isclose(right_abs, 0.0, atol=tol)
    left_missing = ~np.isfinite(contrast_left)
    right_missing = ~np.isfinite(contrast_right)
    all_right_mask = right_only & stim_valid
    all_left_mask = left_only & stim_valid
    all_right_vals = np.unique(np.round(right_abs[all_right_mask], int(round_decimals)))
    all_left_vals = np.unique(np.round(left_abs[all_left_mask], int(round_decimals)))
    all_right_vals = np.sort(all_right_vals)[::-1]
    all_left_vals = np.sort(all_left_vals)[::-1]

    right_top_vals = top_nonzero_contrasts(
        right_abs[right_only],
        top_n=top_n,
        round_decimals=round_decimals,
    )
    left_top_vals = top_nonzero_contrasts(
        left_abs[left_only],
        top_n=top_n,
        round_decimals=round_decimals,
    )

    right_top_mask = np.zeros_like(stim_valid, dtype=bool)
    for val in right_top_vals:
        right_top_mask |= np.isclose(right_abs, float(val), atol=tol)
    right_top_mask &= right_only & stim_valid

    left_top_mask = np.zeros_like(stim_valid, dtype=bool)
    for val in left_top_vals:
        left_top_mask |= np.isclose(left_abs, float(val), atol=tol)
    left_top_mask &= left_only & stim_valid

    zero_left_trials = left_zero & (right_missing | right_zero)
    zero_right_trials = right_zero & (left_missing | left_zero)
    zero_mask = stim_valid & (zero_left_trials | zero_right_trials)

    return {
        "task_all_right_times": np.sort(stim_times[all_right_mask]),
        "task_all_left_times": np.sort(stim_times[all_left_mask]),
        "task_top2_right_times": np.sort(stim_times[right_top_mask]),
        "task_top2_left_times": np.sort(stim_times[left_top_mask]),
        "task_zero_lr_times": np.sort(stim_times[zero_mask]),
        "task_all_right_values": all_right_vals,
        "task_all_left_values": all_left_vals,
        "task_top2_right_values": right_top_vals,
        "task_top2_left_values": left_top_vals,
    }


def select_passive_visual_events_by_side(visual_tr, top_n=2, round_decimals=6):
    """
    Select passive visual top-contrast events split by side.
    """
    times, contrasts, sides = extract_passive_visual_times_contrast_side(visual_tr)
    if times.size == 0:
        return {
            "passive_top2_right_times": np.array([], dtype=float),
            "passive_top2_left_times": np.array([], dtype=float),
            "passive_top2_right_values": np.array([], dtype=float),
            "passive_top2_left_values": np.array([], dtype=float),
        }

    finite_nonzero = np.isfinite(times) & np.isfinite(contrasts) & (contrasts > 0)
    times = times[finite_nonzero]
    contrasts = contrasts[finite_nonzero]
    sides = np.asarray(sides, dtype=object)[finite_nonzero]

    right_mask = sides == "right"
    left_mask = sides == "left"
    right_top_vals = top_nonzero_contrasts(
        contrasts[right_mask],
        top_n=top_n,
        round_decimals=round_decimals,
    )
    left_top_vals = top_nonzero_contrasts(
        contrasts[left_mask],
        top_n=top_n,
        round_decimals=round_decimals,
    )

    tol = 10.0 ** (-int(round_decimals))
    right_top_mask = np.zeros_like(right_mask, dtype=bool)
    for val in right_top_vals:
        right_top_mask |= np.isclose(contrasts, float(val), atol=tol)
    right_top_mask &= right_mask

    left_top_mask = np.zeros_like(left_mask, dtype=bool)
    for val in left_top_vals:
        left_top_mask |= np.isclose(contrasts, float(val), atol=tol)
    left_top_mask &= left_mask

    return {
        "passive_top2_right_times": np.sort(times[right_top_mask]),
        "passive_top2_left_times": np.sort(times[left_top_mask]),
        "passive_top2_right_values": right_top_vals,
        "passive_top2_left_values": left_top_vals,
    }


def build_passive_event_wrappers(visual_tr=None, auditory_tr=None, top_n=2, round_decimals=6):
    """
    Return consolidated passive event groups for dashboard calculation/rendering.
    """
    visual_top = select_passive_visual_events_by_side(
        visual_tr,
        top_n=top_n,
        round_decimals=round_decimals,
    )
    auditory = build_passive_auditory_event_times(auditory_tr)
    out = {
        "passive_visual_top2_right_times": np.asarray(
            visual_top.get("passive_top2_right_times", np.array([])),
            dtype=float,
        ),
        "passive_visual_top2_left_times": np.asarray(
            visual_top.get("passive_top2_left_times", np.array([])),
            dtype=float,
        ),
        "passive_tone_times": np.asarray(auditory.get("tone", np.array([])), dtype=float),
        "passive_valve_times": np.asarray(auditory.get("valve", np.array([])), dtype=float),
        "passive_noise_times": np.asarray(auditory.get("noise", np.array([])), dtype=float),
    }
    return out
