# %% Imports

import os
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.ephys_plots import image_fr_plot
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.flatmaps import FlatMap
from iblatlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice, plot_swanson_vector
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D
from one.api import ONE
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
try:
    from tqdm.auto import tqdm
except ImportError:  # Graceful fallback if tqdm is unavailable.
    def tqdm(iterable, **kwargs):
        return iterable

# %% Init Functions #################################################################

def setup_paths(base_path):
    """Create the main project folders and return their paths."""
    path_data = base_path / "data"
    path_fig = base_path / "results" / "figures"
    path_data_processed = path_data / "processed"
    ibl_cache = path_data / "raw"
    for p in [ibl_cache, path_fig, path_data_processed]:
        p.mkdir(exist_ok=True, parents=True)
    return path_data, path_fig, path_data_processed, ibl_cache


def init_one(ibl_cache):
    """Initialize the ONE API with a custom cache directory."""
    try:
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            silent=True,
            cache_dir=ibl_cache,
        )
        print("ONE API initialized.")
        return one
    except Exception as exc:
        raise RuntimeError(f"Error initializing ONE: {exc}")


def prepare_region_dirs(path_data, beryl_acronyms):
    """Create per-region output folders for Beryl acronyms."""
    for region in beryl_acronyms:
        os.makedirs(f"{path_data}/session_plots_integrated/{region}/", exist_ok=True)


def map_acronyms(clusters, br, mapping):
    """Map cluster acronyms to the requested atlas (Beryl or Allen)."""
    if mapping == "Beryl":
        return br.acronym2acronym(clusters.acronym, mapping="Beryl")
    return clusters.acronym


def load_session_data(pid, one, ba):
    """Load spikes, clusters, and session data for a given probe insertion."""
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    print(f"Session ID (EID): {ssl.eid}")
    print(f"Probe Name: {ssl.pname}")

    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    print(f"Spikes loaded: {spikes.times.shape[0]} spikes")
    if "acronym" in clusters:
        print(f"Cluster regions found: {set(clusters.acronym)}")

    sl = SessionLoader(eid=ssl.eid, one=one)
    sl.load_trials()
    print(f"Trials loaded. Found keys: {list(sl.trials.keys())}")
    sl.load_wheel()
    print(f"Wheel data loaded. Found keys: {list(sl.wheel.keys())}")
    sl.load_pose(views=["left", "right"])
    print(f"Pose data loaded. Found keys: {list(sl.pose.keys())}")

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


def get_trial_contrasts(sl):
    """Return per-trial contrasts (abs max of left/right), NaNs -> 0."""
    contrast_left = np.abs(sl.trials.contrastLeft)
    contrast_right = np.abs(sl.trials.contrastRight)
    trial_contrasts = np.nanmax(np.vstack([contrast_left, contrast_right]), axis=0)
    return np.where(np.isnan(trial_contrasts), 0, trial_contrasts)


def build_event_dicts(sl, event_names, min_trials):
    """Build per-event arrays of valid event times and aligned contrasts."""
    events_by_name = {}
    contrasts_by_name = {}
    trial_contrasts = get_trial_contrasts(sl)
    for event_name in event_names:
        if event_name not in sl.trials.keys():
            print(f"Warning: Event '{event_name}' not found in trials.")
            events_by_name[event_name] = np.array([])
            contrasts_by_name[event_name] = np.array([])
            continue
        events_all = np.asarray(sl.trials[event_name])
        valid_mask = ~np.isnan(events_all)
        events = events_all[valid_mask]
        contrasts = trial_contrasts[valid_mask]
        if len(events) < min_trials:
            print(
                f"Warning: {event_name} has only {len(events)} trials "
                f"(min {min_trials})."
            )
        events_by_name[event_name] = events
        contrasts_by_name[event_name] = contrasts
    return events_by_name, contrasts_by_name


def get_event_time(sl, event_name, trial_idx):
    """Return a single-trial event time (NaN if missing)."""
    if event_name not in sl.trials.keys():
        return np.nan
    events = np.asarray(sl.trials[event_name])
    if trial_idx < 0 or trial_idx >= len(events):
        return np.nan
    return events[trial_idx]


def event_label(event_name):
    """Human-readable labels for event names."""
    label_map = {
        "stimOn_times": "Stim On",
        "firstMovement_times": "First Move",
        "response_times": "Response",
        "feedback_times": "Feedback",
    }
    return label_map.get(event_name, event_name)


def delay_column_name(event_name):
    return f"delay_{event_name}"


def responsive_column_name(event_name):
    return f"responsive_{event_name}"


def reliability_column_names(event_name):
    return f"delay_h1_{event_name}", f"delay_h2_{event_name}"


def compute_psth_for_clusters(
    spikes,
    cluster_ids,
    event_times,
    window_start,
    window_end,
    bin_size,
    smooth_sigma,
    show_progress=False,
    desc="PSTH",
):
    """Compute raw/smoothed PSTHs for multiple clusters and event times."""
    if event_times is None or len(event_times) == 0 or len(cluster_ids) == 0:
        return {}, None

    bins = np.arange(window_start, window_end + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    s_min = event_times.min() + window_start
    s_max = event_times.max() + window_end
    mask = (spikes.times >= s_min) & (spikes.times <= s_max)
    times_window = spikes.times[mask]
    clusters_window = spikes.clusters[mask]

    psth_by_cluster = {}
    iterator = tqdm(cluster_ids, desc=desc, unit="cluster") if show_progress else cluster_ids
    for cid in iterator:
        neuron_spikes = times_window[clusters_window == cid]
        if len(neuron_spikes) == 0:
            continue

        relative_spikes = []
        for t_ev in event_times:
            t0 = t_ev + window_start
            t1 = t_ev + window_end
            in_trial = neuron_spikes[(neuron_spikes >= t0) & (neuron_spikes <= t1)]
            if len(in_trial) > 0:
                relative_spikes.append(in_trial - t_ev)

        if len(relative_spikes) == 0:
            fr_raw = np.zeros(len(bin_centers))
        else:
            all_rel_spikes = np.concatenate(relative_spikes)
            counts, _ = np.histogram(all_rel_spikes, bins=bins)
            fr_raw = counts / len(event_times) / bin_size

        if smooth_sigma and smooth_sigma > 0:
            fr_smooth = gaussian_filter1d(fr_raw, sigma=smooth_sigma)
        else:
            fr_smooth = fr_raw.copy()

        psth_by_cluster[cid] = {"fr_raw": fr_raw, "fr_smooth": fr_smooth}

    return psth_by_cluster, bin_centers


def calculate_delay(
    fr_raw,
    fr_smooth,
    bin_centers,
    config,
    method=None,
    neuron_spikes=None,
    event_times=None,
    trial_contrasts=None,
):
    """
    Compute delay within a responsive window and responsiveness.

    Supported methods:
    - "center_of_mass": PSTH center of mass within the responsive window.
    - "psth_peak": peak time of the PSTH within the responsive window.
    - "tfs": time to first spike after event onset (100% contrast trials only).
    """
    # Default to the configured delay method if none is provided explicitly.
    method = method or config.get("DELAY_METHOD", "center_of_mass")

    if method == "tfs":
        # TFS relies on single-trial spike times, not the PSTH.
        if neuron_spikes is None or event_times is None or trial_contrasts is None:
            return np.nan, False
        if len(neuron_spikes) == 0 or len(event_times) == 0:
            return np.nan, False

        # Identify 100% contrast trials. Accept 1.0 or 100.0 by default.
        full_contrast_values = config.get("FULL_CONTRAST_VALUES", (1.0, 100.0))
        full_mask = np.zeros(len(trial_contrasts), dtype=bool)
        for val in full_contrast_values:
            full_mask |= np.isclose(trial_contrasts, val)
        if not np.any(full_mask):
            return np.nan, False

        # For each full-contrast trial, take the first spike in the responsive window.
        first_spike_offsets = []
        window_start = config["RESPONSIVE_WINDOW_START"]
        window_end = config["RESPONSIVE_WINDOW_END"]
        for t_ev in event_times[full_mask]:
            t0 = t_ev + window_start
            t1 = t_ev + window_end
            idx0 = np.searchsorted(neuron_spikes, t0, side="left")
            idx1 = np.searchsorted(neuron_spikes, t1, side="right")
            if idx0 < idx1:
                first_spike_offsets.append(neuron_spikes[idx0] - t_ev)

        if len(first_spike_offsets) == 0:
            return np.nan, False
        return float(np.mean(first_spike_offsets)), True

    # From here on we rely on the PSTH representation.
    if fr_raw is None or bin_centers is None:
        return np.nan, False

    # Compute a baseline threshold from the unsmoothed PSTH to avoid smoothing bias.
    idx_baseline = (bin_centers >= -config["BASELINE_PRE"]) & (bin_centers < 0)
    baseline_fr = fr_raw[idx_baseline]
    if len(baseline_fr) == 0:
        return np.nan, False

    threshold = np.mean(baseline_fr) + 2 * np.std(baseline_fr)
    idx_responsive = (bin_centers >= config["RESPONSIVE_WINDOW_START"]) & (
        bin_centers <= config["RESPONSIVE_WINDOW_END"]
    )

    responsive_mask = idx_responsive & (fr_raw > threshold)
    if not np.any(responsive_mask):
        return np.nan, False

    resp_fr = fr_smooth[responsive_mask] if fr_smooth is not None else fr_raw[responsive_mask]
    resp_time = bin_centers[responsive_mask]
    if method == "psth_peak":
        # Peak time is the bin center at maximum firing rate within the window.
        peak_idx = int(np.argmax(resp_fr))
        return float(resp_time[peak_idx]), True

    # Default: center of mass (previous behavior).
    sum_fr = np.sum(resp_fr)
    if sum_fr <= 0:
        return np.nan, False
    delay = np.sum(resp_fr * resp_time) / sum_fr
    return delay, True

# %% Main Calculation Functions ##################################################################

def calculate_delays(
    spikes,
    clusters,
    cluster_acronyms,
    events_by_name,
    contrasts_by_name,
    config,
    path_data_processed,
    pid,
    cid_to_idx,
):
    """Compute delays for all clusters across multiple events."""
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for delay calculation.")
        return pd.DataFrame()

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]
    print(f"Found {len(cluster_ids)} clusters.")

    results = []
    selected_cluster_ids = []
    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = get_cluster_label(clusters, idx)
        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
            continue
        results.append(
            {
                "cluster_id": cid,
                "acronym": cluster_acronyms[idx],
                "label": label,
            }
        )
        selected_cluster_ids.append(cid)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No clusters met the selection criteria.")
        return df_res

    spike_times_by_cluster = {
        cid: spikes.times[spikes.clusters == cid] for cid in selected_cluster_ids
    }

    for event_name in event_names:
        events = events_by_name.get(event_name, np.array([]))
        contrasts = contrasts_by_name.get(event_name, np.array([]))
        delay_col = delay_column_name(event_name)
        resp_col = responsive_column_name(event_name)

        if events is None or len(events) < config["MIN_TRIALS"]:
            df_res[delay_col] = np.nan
            df_res[resp_col] = False
            continue

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH ({event_name})",
        )

        delays = []
        responsive_flags = []
        for cid in tqdm(selected_cluster_ids, desc=f"Delays ({event_name})", unit="cluster"):
            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry = psth_by_cluster.get(cid)
            fr_raw = psth_entry["fr_raw"] if psth_entry else None
            fr_smooth = psth_entry["fr_smooth"] if psth_entry else None

            delay, is_responsive = calculate_delay(
                fr_raw,
                fr_smooth,
                bin_centers,
                config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events,
                trial_contrasts=contrasts,
            )

            delays.append(delay)
            responsive_flags.append(is_responsive)

        df_res[delay_col] = delays
        df_res[resp_col] = responsive_flags

    output_path = path_data_processed / f"{pid}_delay_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Computed delays for {len(df_res)} neurons. Saved to {output_path}.")
    return df_res


def calculate_delay_reliability(
    spikes,
    clusters,
    cluster_acronyms,
    events_by_name,
    contrasts_by_name,
    config,
    path_data_processed,
    pid,
    cid_to_idx,
    df_res=None,
):
    """Compute split-half delay reliability across multiple events."""
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for reliability calculation.")
        return pd.DataFrame()

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]

    results = []
    selected_cluster_ids = []
    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = get_cluster_label(clusters, idx)
        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
            continue
        results.append({"cluster_id": cid, "acronym": cluster_acronyms[idx]})
        selected_cluster_ids.append(cid)

    df_reliability = pd.DataFrame(results)
    if df_reliability.empty:
        print("No clusters met the selection criteria.")
        return df_reliability

    spike_times_by_cluster = {
        cid: spikes.times[spikes.clusters == cid] for cid in selected_cluster_ids
    }

    for event_name in event_names:
        events = events_by_name.get(event_name, np.array([]))
        contrasts = contrasts_by_name.get(event_name, np.array([]))
        col_h1, col_h2 = reliability_column_names(event_name)
        df_reliability[col_h1] = np.nan
        df_reliability[col_h2] = np.nan

        if events is None or len(events) < config["MIN_TRIALS"]:
            continue

        mid_idx = len(events) // 2
        events_h1 = events[:mid_idx]
        events_h2 = events[mid_idx:]
        contrasts_h1 = contrasts[:mid_idx]
        contrasts_h2 = contrasts[mid_idx:]

        psth_h1, bin_centers_h1 = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_h1,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH H1 ({event_name})",
        )
        psth_h2, bin_centers_h2 = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_h2,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH H2 ({event_name})",
        )

        if df_res is not None and responsive_column_name(event_name) in df_res.columns:
            resp_lookup = dict(
                zip(df_res["cluster_id"].values, df_res[responsive_column_name(event_name)].values)
            )
            responsive_mask = np.array(
                [resp_lookup.get(cid, False) for cid in selected_cluster_ids], dtype=bool
            )
        else:
            responsive_mask = np.ones(len(selected_cluster_ids), dtype=bool)

        delays_h1 = []
        delays_h2 = []
        rel_config = {
            **config,
            "RESPONSIVE_WINDOW_START": config["RELIABILITY_WINDOW_START"],
            "RESPONSIVE_WINDOW_END": config["RELIABILITY_WINDOW_END"],
        }

        for cid, is_resp in tqdm(
            list(zip(selected_cluster_ids, responsive_mask)),
            desc=f"Reliability ({event_name})",
            unit="cluster",
        ):
            if not is_resp:
                delays_h1.append(np.nan)
                delays_h2.append(np.nan)
                continue

            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry_h1 = psth_h1.get(cid)
            psth_entry_h2 = psth_h2.get(cid)
            fr_raw_h1 = psth_entry_h1["fr_raw"] if psth_entry_h1 else None
            fr_smooth_h1 = psth_entry_h1["fr_smooth"] if psth_entry_h1 else None
            fr_raw_h2 = psth_entry_h2["fr_raw"] if psth_entry_h2 else None
            fr_smooth_h2 = psth_entry_h2["fr_smooth"] if psth_entry_h2 else None

            delay_h1, _ = calculate_delay(
                fr_raw_h1,
                fr_smooth_h1,
                bin_centers_h1,
                rel_config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events_h1,
                trial_contrasts=contrasts_h1,
            )
            delay_h2, _ = calculate_delay(
                fr_raw_h2,
                fr_smooth_h2,
                bin_centers_h2,
                rel_config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events_h2,
                trial_contrasts=contrasts_h2,
            )

            delays_h1.append(delay_h1)
            delays_h2.append(delay_h2)

        df_reliability[col_h1] = delays_h1
        df_reliability[col_h2] = delays_h2

    output_path = path_data_processed / f"{pid}_delay_reliability.csv"
    df_reliability.to_csv(output_path, index=False)
    print(
        f"Found {len(df_reliability)} responsive neurons (both halves). Saved to {output_path}."
    )
    return df_reliability

# %% Main Plotting Functions ##################################################################

def plot_trial_raster(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    sl,
    pupil_features,
    pupil_times,
    config_plot,
    pid,
    path_fig,
    save_figure,
    trial_idx,
):
    """Plot a trial-aligned raster with wheel, paw, and pupil traces."""
    t_stim_on = get_event_time(sl, "stimOn_times", trial_idx)
    t_first_move = get_event_time(sl, "firstMovement_times", trial_idx)
    t_feedback = get_event_time(sl, "feedback_times", trial_idx)

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in sl.trials.keys():
        print(f"Warning: Event '{align_event}' not found. Falling back to stimOn_times.")
        align_event = "stimOn_times"
    t_align = get_event_time(sl, align_event, trial_idx)
    if np.isnan(t_align):
        t_align = t_stim_on

    t_start = t_align - config_plot["RASTER_WINDOW_PRE"]
    t_end = t_align + config_plot["RASTER_WINDOW_POST"]

    align_to_event = config_plot.get(
        "RASTER_ALIGN_TO_EVENT", config_plot.get("RASTER_ALIGN_TO_STIM_ON", True)
    )
    if align_to_event:
        t_offset = t_align
        xlabel_text = f"Time from {event_label(align_event)} (s)"
    else:
        t_offset = 0
        xlabel_text = "Time in session (s)"

    cont_l = sl.trials["contrastLeft"][trial_idx]
    cont_r = sl.trials["contrastRight"][trial_idx]

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
    choice_val = sl.trials["choice"][trial_idx]
    response_str = choice_map.get(choice_val, str(choice_val))

    fb_val = sl.trials["feedbackType"][trial_idx]
    outcome_str = "Correct" if fb_val == 1 else "Incorrect"

    plot_title = (
        f"Trial {trial_idx} | Contrast: {contrast_val} ({stim_side}) | "
        f"Response: {response_str} | {outcome_str}"
    )

    if config_plot["PLOT_ONLY_GOOD_UNITS"]:
        if hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        elif hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        else:
            quality_mask = np.ones(len(cluster_ids), dtype=bool)
        ylabel_text = f"Good Units (n={np.sum(quality_mask)})"
    else:
        quality_mask = np.ones(len(cluster_ids), dtype=bool)
        ylabel_text = f"All Units (n={np.sum(quality_mask)})"

    df_units = pd.DataFrame(
        {
            "cluster_id": cluster_ids[quality_mask],
            "acronym": cluster_acronyms[quality_mask],
            "depth": clusters.depths[quality_mask],
        }
    )
    df_units = df_units.sort_values(by="depth", ascending=True).reset_index(drop=True)

    unique_regions = df_units["acronym"].unique()
    region_colors = {reg: plt.cm.tab20(i % 20) for i, reg in enumerate(unique_regions)}
    region_colors["void"] = "black"

    mask_window = (spikes.times >= t_start) & (spikes.times <= t_end)
    window_spike_times = spikes.times[mask_window]
    window_spike_clusters = spikes.clusters[mask_window]

    mask_wheel = (sl.wheel["times"] >= t_start) & (sl.wheel["times"] <= t_end)
    wheel_t = sl.wheel["times"][mask_wheel]
    wheel_pos = sl.wheel["position"][mask_wheel]

    pose_t = None
    paw_speed = None
    if hasattr(sl, "pose") and "leftCamera" in sl.pose:
        pose_df = sl.pose["leftCamera"]
        if "times" in pose_df.columns:
            pose_timestamps = pose_df["times"].values
        else:
            pose_timestamps = pose_df.index.values

        mask_pose = (pose_timestamps >= t_start) & (pose_timestamps <= t_end)
        pose_t = pose_timestamps[mask_pose]

        paw_key = "paw_r" if "paw_r_x" in pose_df.columns else "paw_l"
        if f"{paw_key}_x" in pose_df.columns:
            dx = np.gradient(pose_df[f"{paw_key}_x"].values[mask_pose])
            dy = np.gradient(pose_df[f"{paw_key}_y"].values[mask_pose])
            dt = np.gradient(pose_t)
            dt[dt == 0] = np.nan
            speed_raw = np.sqrt(dx**2 + dy**2) / dt
            paw_speed = (
                pd.Series(speed_raw).fillna(0).rolling(window=5, center=True).mean().values
            )

    pupil_t = None
    pupil_diam = None
    if pupil_features is not None and pupil_times is not None:
        diam_col = "pupilDiameter_raw"
        if diam_col in pupil_features.columns:
            n_frames = min(len(pupil_times), len(pupil_features))
            pt = pupil_times[:n_frames]
            pd_vals = pupil_features[diam_col].values[:n_frames]
            mask_pupil = (pt >= t_start) & (pt <= t_end)
            pupil_t = pt[mask_pupil]
            pupil_diam = pd_vals[mask_pupil]

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(
        4,
        2,
        width_ratios=[20, 1],
        height_ratios=[10, 1, 1, 1],
        wspace=0.05,
        hspace=0.1,
    )

    ax_raster = fig.add_subplot(gs[0, 0])
    for y_idx, row in df_units.iterrows():
        unit_spike_times = window_spike_times[window_spike_clusters == row["cluster_id"]]
        if len(unit_spike_times) > 0:
            ax_raster.vlines(
                unit_spike_times - t_offset,
                y_idx - 0.45,
                y_idx + 0.45,
                color="k",
                linewidth=0.8,
            )

    for ax in [ax_raster]:
        ax.axvline(t_stim_on - t_offset, color="blue", linestyle="-", linewidth=2)
        ax.axvline(t_first_move - t_offset, color="green", linestyle="-", linewidth=2)
        ax.axvline(t_feedback - t_offset, color="red", linestyle="-", linewidth=2)

    ax_raster.set_xlim(t_start - t_offset, t_end - t_offset)
    ax_raster.set_ylim(-1, len(df_units))
    ax_raster.set_ylabel(ylabel_text)
    ax_raster.set_title(plot_title)
    ax_raster.tick_params(labelbottom=False)

    ax_regions = fig.add_subplot(gs[0, 1])
    y_min = 0
    for acronym, group in df_units.groupby("acronym", sort=False):
        count = len(group)
        color = region_colors.get(acronym, "gray")
        ax_regions.add_patch(plt.Rectangle((0, y_min), 1, count, color=color))
        ax_regions.text(
            1.2,
            y_min + count / 2,
            acronym,
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        y_min += count

    ax_regions.set_ylim(0, len(df_units))
    ax_regions.axis("off")

    ax_wheel = fig.add_subplot(gs[1, 0], sharex=ax_raster)
    ax_wheel.plot(wheel_t - t_offset, wheel_pos, color="black")
    ax_wheel.set_ylabel("Wheel (rad)")
    ax_wheel.axvline(t_stim_on - t_offset, color="blue", linewidth=1.5)
    ax_wheel.axvline(t_first_move - t_offset, color="green", linewidth=1.5)
    ax_wheel.axvline(t_feedback - t_offset, color="red", linewidth=1.5)
    ax_wheel.tick_params(labelbottom=False)

    ax_paw = fig.add_subplot(gs[2, 0], sharex=ax_raster)
    if paw_speed is not None:
        ax_paw.plot(pose_t - t_offset, paw_speed, color="black")
    else:
        ax_paw.text(0.5, 0.5, "Paw data not available", ha="center", transform=ax_paw.transAxes)

    ax_paw.set_ylabel("Paw (px/s)")
    ax_paw.axvline(t_stim_on - t_offset, color="blue", linewidth=1.5)
    ax_paw.axvline(t_first_move - t_offset, color="green", linewidth=1.5)
    ax_paw.axvline(t_feedback - t_offset, color="red", linewidth=1.5)
    ax_paw.tick_params(labelbottom=False)

    ax_pupil = fig.add_subplot(gs[3, 0], sharex=ax_raster)
    if pupil_diam is not None:
        ax_pupil.plot(pupil_t - t_offset, pupil_diam, color="black")
    else:
        ax_pupil.text(
            0.5, 0.5, "Pupil data not available", ha="center", transform=ax_pupil.transAxes
        )

    ax_pupil.set_ylabel("Pupil (mm)")
    ax_pupil.set_xlabel(xlabel_text)
    ax_pupil.axvline(t_stim_on - t_offset, color="blue", linewidth=1.5)
    ax_pupil.axvline(t_first_move - t_offset, color="green", linewidth=1.5)
    ax_pupil.axvline(t_feedback - t_offset, color="red", linewidth=1.5)
    ax_pupil.tick_params(labelbottom=True)

    lines = [
        plt.Line2D([0], [0], color="blue", linewidth=2),
        plt.Line2D([0], [0], color="green", linewidth=2),
        plt.Line2D([0], [0], color="red", linewidth=2),
    ]
    ax_raster.legend(
        lines,
        ["Stim On", "First Move", "Feedback"],
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0, 1.15),
        ncol=3,
    )

    if save_figure:
        filename = f"{pid}_{trial_idx}_Raster.png"
        save_path = path_fig / filename
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_delay_histogram(df_res, config_calc, config_plot, save_flag, path_fig, pid):
    """Plot delay histogram stacked by region."""
    event_name = config_plot.get("PLOT_EVENT", "stimOn_times")
    delay_col = delay_column_name(event_name)
    if delay_col not in df_res.columns:
        print(f"Delay column '{delay_col}' not found.")
        return
    df_plot = df_res.dropna(subset=[delay_col]).copy()

    title_suffix = ""
    if config_plot["PLOT_ONLY_GOOD_UNITS"]:
        df_plot = df_plot[df_plot["label"] == 1]
        title_suffix = "(Good Units Only)"
    else:
        title_suffix = "(All Units)"

    if len(df_plot) == 0:
        print("No units met criteria for delay histogram.")
        return

    top_n = 8
    region_counts = df_plot["acronym"].value_counts()
    top_regions = region_counts.nlargest(top_n).index.tolist()

    df_plot["plot_region"] = df_plot["acronym"].apply(
        lambda x: x if x in top_regions else "Other"
    )

    unique_regions = df_plot["plot_region"].unique()
    unique_regions = sorted([r for r in unique_regions if r != "Other"]) + (
        ["Other"] if "Other" in unique_regions else []
    )

    data_to_plot = [
        df_plot[df_plot["plot_region"] == r][delay_col].values for r in unique_regions
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    hist_bins = np.linspace(
        config_calc["RESPONSIVE_WINDOW_START"], config_calc["RESPONSIVE_WINDOW_END"], 30
    )

    ax.hist(
        data_to_plot,
        bins=hist_bins,
        stacked=True,
        label=unique_regions,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xlabel("Response Delay (s)")
    ax.set_ylabel("Number of Neurons")
    ax.set_title(
        f"Distribution of Response Delays ({event_label(event_name)}) {title_suffix}\n"
        f"(Window: {config_calc['RESPONSIVE_WINDOW_START']}-{config_calc['RESPONSIVE_WINDOW_END']}s)"
    )
    ax.legend(title="Region")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_flag:
        save_path = path_fig / f"{pid}_delay_histogram.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Delay histogram saved to: {save_path}")

    plt.show()

    print("\nSummary Statistics by Region:")
    print(df_plot.groupby("plot_region")[delay_col].describe()[["count", "mean", "std"]])


def plot_delay_reliability(df_reliability, config_calc, config_plot, save_flag, path_fig, pid):
    """Plot split-half delay reliability scatter by region."""
    if len(df_reliability) == 0:
        print("No neurons met the reliability criteria.")
        return
    event_name = config_plot.get("PLOT_EVENT", "stimOn_times")
    col_h1, col_h2 = reliability_column_names(event_name)
    if col_h1 not in df_reliability.columns or col_h2 not in df_reliability.columns:
        print(f"Reliability columns '{col_h1}'/'{col_h2}' not found.")
        return

    top_n = 9
    region_counts = df_reliability["acronym"].value_counts()
    top_regions = region_counts.nlargest(top_n).index.tolist()
    if "VISp" in df_reliability["acronym"].values and "VISp" not in top_regions:
        top_regions.append("VISp")

    df_reliability = df_reliability.copy()
    df_reliability["plot_region"] = df_reliability["acronym"].apply(
        lambda x: x if x in top_regions else "Other"
    )

    unique_plot_regions = sorted(df_reliability["plot_region"].unique())
    default_colors = sns.color_palette("tab20", n_colors=len(unique_plot_regions))
    palette_dict = dict(zip(unique_plot_regions, default_colors))
    if "VISp" in palette_dict:
        palette_dict["VISp"] = "blue"
    if "Other" in palette_dict:
        palette_dict["Other"] = "gray"

    r_val, p_val = stats.pearsonr(df_reliability[col_h1], df_reliability[col_h2])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=df_reliability,
        x=col_h1,
        y=col_h2,
        hue="plot_region",
        palette=palette_dict,
        marker="o",
        s=50,
        alpha=1.0,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )

    lims = [
        min(df_reliability[col_h1].min(), df_reliability[col_h2].min()),
        max(df_reliability[col_h1].max(), df_reliability[col_h2].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", alpha=0.5, label="Identity")

    unit_type = "Good Units" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "All Units"
    ax.set_xlabel(f"Delay (s) - First Half ({event_label(event_name)})")
    ax.set_ylabel(f"Delay (s) - Second Half ({event_label(event_name)})")
    ax.set_title(
        f"Reliability of Response Delays ({unit_type})\n"
        f"Pearson r = {r_val:.2f}, p = {p_val:.2e}\n"
        f"n={len(df_reliability)}"
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Region")
    ax.axis("square")
    plt.tight_layout()

    if save_flag:
        save_path = path_fig / f"{pid}_delay_reliability.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reliability plot saved to: {save_path}")

    plt.show()


def plot_single_neuron(
    sl,
    spikes,
    clusters,
    cluster_acronyms,
    cid_to_idx,
    df_res,
    config_plot,
    save_flag,
    path_fig,
    pid,
    cluster_id,
):
    """Plot PSTHs and rasters for a single neuron, plus global PSTH."""
    target_cluster_id = cluster_id
    if target_cluster_id not in cid_to_idx:
        fallback_id = next(iter(cid_to_idx), None)
        print(f"Error: Cluster ID {target_cluster_id} not found. Using {fallback_id}.")
        if fallback_id is None:
            return
        target_cluster_id = fallback_id

    target_idx = cid_to_idx[target_cluster_id]
    target_acronym = cluster_acronyms[target_idx]
    print(f"Analyzing Cluster ID: {target_cluster_id} | Region: {target_acronym}")

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    delay_col = delay_column_name(align_event)
    unit_delay = np.nan
    if df_res is not None:
        match = df_res[df_res["cluster_id"] == target_cluster_id]
        if not match.empty:
            if delay_col in match.columns:
                unit_delay = match.iloc[0][delay_col]
            print(f"Global Delay found: {unit_delay:.4f} s")
        else:
            print(f"Cluster {target_cluster_id} not found in df_res.")

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], hspace=0.4)

    ax_psth_L = fig.add_subplot(gs[0, 0])
    ax_psth_R = fig.add_subplot(gs[0, 1], sharey=ax_psth_L)
    ax_rast_L = fig.add_subplot(gs[1, 0], sharex=ax_psth_L)
    ax_rast_R = fig.add_subplot(gs[1, 1], sharex=ax_psth_R)
    ax_global = fig.add_subplot(gs[2, :])

    axes_split = [[ax_psth_L, ax_psth_R], [ax_rast_L, ax_rast_R]]
    neuron_spikes = spikes.times[spikes.clusters == target_cluster_id]

    contrasts_to_plot = [1.0, 0.25, 0.125, 0.0625, 0.0]
    contrast_colors = {
        1.0: (0.0, 0.0, 0.0, 1.0),
        0.5: (0.2, 0.2, 0.2, 1.0),
        0.25: (0.4, 0.4, 0.4, 1.0),
        0.125: (0.6, 0.6, 0.6, 1.0),
        0.0625: (0.75, 0.75, 0.75, 1.0),
        0.0: (0.85, 0.85, 0.85, 1.0),
    }

    sides = ["Left", "Right"]
    for col_idx, side in enumerate(sides):
        ax_psth = axes_split[0][col_idx]
        ax_rast = axes_split[1][col_idx]
        current_raster_y = 0

        for cont in contrasts_to_plot:
            if side == "Left":
                contrast_arr = sl.trials.contrastLeft
            else:
                contrast_arr = sl.trials.contrastRight

            event_series = np.asarray(sl.trials[align_event])
            mask = (contrast_arr == cont) & (~np.isnan(event_series))
            events = event_series[mask]

            if len(events) == 0:
                continue

            psth_by_cluster, bin_centers = compute_psth_for_clusters(
                spikes,
                [target_cluster_id],
                events,
                -config_plot["SINGLE_NEURON_RASTER_PRE"],
                config_plot["SINGLE_NEURON_RASTER_POST"],
                config_plot["SINGLE_NEURON_BIN_SIZE"],
                config_plot["SINGLE_NEURON_SMOOTH_SIGMA"],
                show_progress=False,
            )
            psth_entry = psth_by_cluster.get(target_cluster_id)
            if psth_entry and bin_centers is not None:
                firing_rate = psth_entry["fr_smooth"]
            else:
                firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

            color_val = contrast_colors.get(cont, (0, 0, 0, 1))
            label_text = f"{cont * 100:.0f}%" if cont > 0 else "0%"
            ax_psth.plot(bin_centers, firing_rate, color=color_val, linewidth=2, label=label_text)

            block_start_y = current_raster_y
            for event_t in events:
                t_start = event_t - config_plot["SINGLE_NEURON_RASTER_PRE"]
                t_end = event_t + config_plot["SINGLE_NEURON_RASTER_POST"]
                trial_spikes = neuron_spikes[(neuron_spikes >= t_start) & (neuron_spikes <= t_end)]
                aligned_spikes = trial_spikes - event_t

                ax_rast.vlines(
                    aligned_spikes,
                    current_raster_y - 0.4,
                    current_raster_y + 0.4,
                    colors="black",
                    linewidth=0.8,
                )
                current_raster_y += 1

            bar_x = config_plot["SINGLE_NEURON_RASTER_POST"] + 0.05
            rect = patches.Rectangle(
                (bar_x, block_start_y),
                0.05,
                current_raster_y - block_start_y,
                linewidth=0,
                facecolor=color_val,
                clip_on=False,
            )
            ax_rast.add_patch(rect)
            current_raster_y += 2

        ax_psth.set_title(f"{side} Stimuli")
        ax_psth.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_rast.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_psth.legend(frameon=False, title="Contrast")

        for ax in [ax_psth, ax_rast]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_rast.set_yticks([])
        ax_rast.set_xlim(
            -config_plot["SINGLE_NEURON_RASTER_PRE"],
            config_plot["SINGLE_NEURON_RASTER_POST"] + 0.15,
        )
        ax_rast.set_ylim(-1, current_raster_y)

        if col_idx == 0:
            ax_psth.set_ylabel("Firing Rate (Hz)")
            ax_rast.set_ylabel("Trials (Sorted)")

    all_event_series = np.asarray(sl.trials[align_event])
    all_events = all_event_series[~np.isnan(all_event_series)]
    if len(all_events) > 0:
        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            [target_cluster_id],
            all_events,
            -config_plot["SINGLE_NEURON_RASTER_PRE"],
            config_plot["SINGLE_NEURON_RASTER_POST"],
            config_plot["SINGLE_NEURON_BIN_SIZE"],
            config_plot["SINGLE_NEURON_SMOOTH_SIGMA"],
            show_progress=False,
        )

        psth_entry = psth_by_cluster.get(target_cluster_id)
        if psth_entry and bin_centers is not None:
            fr_global = psth_entry["fr_raw"]
            fr_global_smooth = psth_entry["fr_smooth"]
            ax_global.plot(
                bin_centers, fr_global_smooth, color="k", linewidth=2, label="Global Response"
            )
            ax_global.fill_between(bin_centers, fr_global, color="gray", alpha=0.2)

            if not np.isnan(unit_delay):
                ax_global.axvline(
                    unit_delay,
                    color="red",
                    linestyle="-",
                    linewidth=2,
                    label=f"Delay: {unit_delay * 1000:.1f} ms",
                )
                ymax = ax_global.get_ylim()[1]
                if ymax > 0:
                    ax_global.text(
                        unit_delay + 0.02,
                        ymax * 0.9,
                        f"{unit_delay * 1000:.0f}ms",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                    )
        else:
            ax_global.text(0.5, 0.5, "No spikes in window", ha="center")

        ax_global.axvline(
            0,
            color="blue",
            linestyle="--",
            linewidth=1,
            label=event_label(align_event),
        )
        ax_global.set_ylabel("Firing Rate (Hz)")
        ax_global.set_xlabel(f"Time from {event_label(align_event)} (s)")
        ax_global.set_title("Global PSTH (All Trials)")
        ax_global.legend(loc="upper right", frameon=False)
        ax_global.spines["top"].set_visible(False)
        ax_global.spines["right"].set_visible(False)
        ax_global.set_xlim(
            -config_plot["SINGLE_NEURON_RASTER_PRE"],
            config_plot["SINGLE_NEURON_RASTER_POST"],
        )
    else:
        ax_global.text(0.5, 0.5, "No valid trials found", ha="center")

    plt.suptitle(
        f"Cluster #{target_cluster_id} ({target_acronym}) Response Analysis", fontsize=16
    )

    if save_flag:
        save_path = path_fig / f"{pid}_cluster_{target_cluster_id}_response.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Single neuron figure saved to: {save_path}")

    plt.show()


def plot_sequence_raster(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    save_flag,
    path_fig,
    pid,
    trial_idx,
    region_acronyms=None,
):
    """Plot region rasters sorted by delay for a single trial."""
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS", ["VISp"])
    elif isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]
    else:
        region_acronyms = list(region_acronyms)

    if len(region_acronyms) == 0:
        print("No regions provided for plot_sequence_raster.")
        return

    window_pre = config_plot["SEQUENCE_WINDOW_PRE"]
    window_post = config_plot["SEQUENCE_WINDOW_POST"]
    align_to_event = config_plot.get(
        "SEQUENCE_ALIGN_TO_EVENT", config_plot.get("SEQUENCE_ALIGN_TO_STIM", True)
    )

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in sl.trials.keys():
        print(f"Warning: Event '{align_event}' not found. Falling back to stimOn_times.")
        align_event = "stimOn_times"
    delay_col = delay_column_name(align_event)

    try:
        if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        elif hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        else:
            quality_mask = np.ones(len(clusters.acronym), dtype=bool)
    except AttributeError:
        print("Error: clusters data incomplete.")
        quality_mask = np.ones(len(cluster_acronyms), dtype=bool)

    cluster_acronyms_str = cluster_acronyms.astype(str)

    t_stim = get_event_time(sl, "stimOn_times", trial_idx)
    t_move = get_event_time(sl, "firstMovement_times", trial_idx)
    t_feed = get_event_time(sl, "feedback_times", trial_idx)
    t_align = get_event_time(sl, align_event, trial_idx)
    if np.isnan(t_align):
        t_align = t_stim

    t_start = t_align - window_pre
    t_end = t_align + window_post
    t_offset = t_align if align_to_event else 0

    fig, axes = plt.subplots(
        len(region_acronyms), 1, figsize=(10, 6 * len(region_acronyms)), sharex=True
    )
    if len(region_acronyms) == 1:
        axes = [axes]

    label_text = "Good Neurons" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "Neurons"

    for ax_raster, region in zip(axes, region_acronyms):
        try:
            region_mask = np.char.startswith(cluster_acronyms_str, region)
            if config_plot["PLOT_ONLY_GOOD_UNITS"]:
                final_mask = region_mask & quality_mask
            else:
                final_mask = region_mask
            region_cluster_ids = cluster_ids[final_mask]
            print(f"Found {len(region_cluster_ids)} {label_text} in {region}.")
        except AttributeError:
            print(f"Error: clusters data incomplete for {region}.")
            region_cluster_ids = []

        if len(region_cluster_ids) > 0:
            df_region = pd.DataFrame(
                {
                    "cluster_id": region_cluster_ids,
                    "acronym": cluster_acronyms[final_mask],
                    "depth": clusters.depths[final_mask],
                }
            )
        else:
            df_region = pd.DataFrame(columns=["cluster_id", "acronym", "depth"])

        if df_res is not None and len(df_region) > 0:
            if delay_col in df_res.columns:
                df_region = df_region.merge(
                    df_res[["cluster_id", delay_col]].rename(columns={delay_col: "delay"}),
                    on="cluster_id",
                    how="left",
                )
            else:
                df_region["delay"] = np.nan
        else:
            df_region["delay"] = np.nan

        df_sorted = df_region.sort_values(
            by="delay", ascending=True, na_position="last"
        ).reset_index(drop=True)

        n_responsive = df_sorted["delay"].notna().sum()
        n_nan = df_sorted["delay"].isna().sum()
        print(
            f"Sorting Complete ({region}): {n_responsive} sequenced, {n_nan} unresponsive (bottom)."
        )

        mask_spikes = (np.isin(spikes.clusters, region_cluster_ids)) & (
            spikes.times >= t_start
        ) & (spikes.times <= t_end)

        relevant_spikes = spikes.times[mask_spikes]
        relevant_clusters = spikes.clusters[mask_spikes]

        if len(df_sorted) == 0:
            ax_raster.text(
                0.5,
                0.5,
                f"No units found for {region}",
                transform=ax_raster.transAxes,
                ha="center",
                va="center",
            )
            ax_raster.set_xlim(t_start - t_offset, t_end - t_offset)
            ax_raster.set_ylim(0, 1)
        else:
            for y_idx, row in df_sorted.iterrows():
                cid = row["cluster_id"]
                delay = row["delay"]
                unit_spikes = relevant_spikes[relevant_clusters == cid]

                if len(unit_spikes) > 0:
                    spike_times_aligned = unit_spikes - t_offset
                    if pd.isna(delay):
                        color = "lightgray"
                        lw = 0.5
                    else:
                        color = "black"
                        lw = 0.8

                    ax_raster.vlines(
                        spike_times_aligned,
                        y_idx - 0.45,
                        y_idx + 0.45,
                        color=color,
                        linewidth=lw,
                    )

            ax_raster.set_xlim(t_start - t_offset, t_end - t_offset)
            ax_raster.set_ylim(len(df_sorted), -1)

            if n_nan > 0:
                sep_y = n_responsive
                ax_raster.axhline(sep_y, color="red", linestyle="--", linewidth=1, alpha=0.7)
                ax_raster.text(
                    (t_start - t_offset) + 0.05,
                    sep_y + n_nan / 2,
                    "Unresponsive / Untuned",
                    color="gray",
                    fontsize=10,
                    va="center",
                    fontweight="bold",
                )


        ax_raster.set_ylabel(
            f"{region} {label_text} (n={len(df_sorted)})\nSorted by Delay (NaN bottom)",
            fontsize=12,
        )
        ax_raster.set_title(
            f"Trial {trial_idx} | {region} Sequence ({label_text})",
            fontsize=14,
        )

        ax_raster.axvline(0, color="blue", linewidth=2, label=event_label(align_event))
        if not np.isnan(t_move):
            ax_raster.axvline(
                t_move - t_offset,
                color="green",
                linewidth=2,
                linestyle="--",
                label="Move",
            )
        if not np.isnan(t_feed):
            ax_raster.axvline(
                t_feed - t_offset,
                color="red",
                linewidth=2,
                linestyle="--",
                label="Feedback",
            )

        ax_raster.legend(loc="upper left", frameon=False, ncol=3)
        ax_raster.spines["top"].set_visible(False)
        ax_raster.spines["right"].set_visible(False)

    axes[-1].set_xlabel(f"Time from {event_label(align_event)} (s)", fontsize=12)

    plt.tight_layout()

    if save_flag:
        if len(region_acronyms) == 1:
            file_name = f"{pid}_trial_{trial_idx}_sequence.png"
        else:
            region_tag = "_".join(region_acronyms)
            file_name = f"{pid}_trial_{trial_idx}_sequence_{region_tag}.png"
        save_path = path_fig / file_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sequence raster saved to: {save_path}")

    plt.show()


def plot_population_sorted(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    save_flag,
    path_fig,
    pid,
    region_acronyms=None,
):
    """Plot population heatmaps sorted by delay with delay markers."""
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS", ["VISp"])
    elif isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]
    else:
        region_acronyms = list(region_acronyms)

    if len(region_acronyms) == 0:
        print("No regions provided for plot_population_sorted.")
        return

    window_pre = config_plot["POP_WINDOW_PRE"]
    window_post = config_plot["POP_WINDOW_POST"]
    bin_size = config_plot["POP_BIN_SIZE"]
    smooth_sigma = config_plot["POP_SMOOTH_SIGMA"]
    cmap_name = config_plot["POP_CMAP_NAME"]
    normalize = config_plot["POP_NORMALIZE"]

    try:
        if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        elif hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        else:
            quality_mask = np.ones(len(clusters.acronym), dtype=bool)
    except AttributeError:
        print("Error: Cluster data incomplete.")
        quality_mask = np.ones(len(cluster_acronyms), dtype=bool)

    cluster_acronyms_str = cluster_acronyms.astype(str)
    label_text = "Good Neurons" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "Neurons"
    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in sl.trials.keys():
        print(f"Warning: Event '{align_event}' not found. Falling back to stimOn_times.")
        align_event = "stimOn_times"
    delay_col = delay_column_name(align_event)
    event_series = np.asarray(sl.trials[align_event])
    stim_times = event_series[~np.isnan(event_series)]

    fig, axes = plt.subplots(
        len(region_acronyms), 1, figsize=(10, 6 * len(region_acronyms)), sharex=True
    )
    if len(region_acronyms) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_acronyms):
        try:
            region_mask = np.char.startswith(cluster_acronyms_str, region)
            if config_plot["PLOT_ONLY_GOOD_UNITS"]:
                final_mask = region_mask & quality_mask
            else:
                final_mask = region_mask
            region_ids = cluster_ids[final_mask]
            print(f"Found {len(region_ids)} {label_text} in {region}.")
        except AttributeError:
            region_ids = []
            print(f"Error: Cluster data incomplete for {region}.")

        df_region = pd.DataFrame({"cluster_id": region_ids})

        if df_res is not None and len(df_region) > 0:
            if delay_col in df_res.columns:
                df_region = df_region.merge(
                    df_res[["cluster_id", delay_col]].rename(columns={delay_col: "delay"}),
                    on="cluster_id",
                    how="left",
                )
            else:
                print(f"Warning: {delay_col} not found. Latencies will be NaN.")
                df_region["delay"] = np.nan
        else:
            if df_res is None:
                print("Warning: df_res not found. Latencies will be NaN.")
            df_region["delay"] = np.nan

        df_sorted = df_region.sort_values(
            by="delay", ascending=True, na_position="last"
        ).reset_index(drop=True)

        n_neurons = len(df_sorted)
        if n_neurons == 0:
            ax.text(
                0.5,
                0.5,
                f"No units found for {region}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-window_pre, window_post)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Neurons (Sorted by Latency)\nTotal: 0", fontsize=12)
            title_str = "Normalized " if normalize else "Raw "
            ax.set_title(
                f"{title_str}Average Response (PSTH) Heatmap | {region} Units", fontsize=14
            )
            continue

        if len(stim_times) == 0:
            ax.text(
                0.5,
                0.5,
                f"No valid {align_event} events for {region}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-window_pre, window_post)
            ax.set_ylim(0, 1)
            continue

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            df_sorted["cluster_id"].values,
            stim_times,
            -window_pre,
            window_post,
            bin_size,
            smooth_sigma,
            show_progress=True,
            desc=f"PSTH ({region})",
        )
        n_bins = len(bin_centers) if bin_centers is not None else 0
        psth_matrix = np.zeros((n_neurons, n_bins))

        for row_idx, row in df_sorted.iterrows():
            cid = row["cluster_id"]
            psth_entry = psth_by_cluster.get(cid)
            if not psth_entry:
                continue
            fr_smooth = psth_entry["fr_smooth"]
            if normalize:
                peak = np.max(fr_smooth)
                if peak > 0:
                    fr_smooth = fr_smooth / peak
            psth_matrix[row_idx, :] = fr_smooth

        im = ax.imshow(
            psth_matrix,
            aspect="auto",
            origin="lower",
            extent=[-window_pre, window_post, 0, n_neurons],
            cmap=cmap_name,
            interpolation="nearest",
        )

        valid_delays = df_sorted.dropna(subset=["delay"])
        y_positions = valid_delays.index + 0.5
        x_positions = valid_delays["delay"]
        ax.scatter(x_positions, y_positions, color="black", s=10, marker="o", label="Delay")

        ax.axvline(0, color="black", linestyle="--", linewidth=1, label=event_label(align_event))
        ax.set_ylabel(f"Neurons (Sorted by Latency)\nTotal: {n_neurons}", fontsize=12)
        title_str = "Normalized " if normalize else "Raw "
        ax.set_title(
            f"{title_str}Average Response (PSTH) Heatmap | {region} Units", fontsize=14
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            "Normalized Firing Rate" if normalize else "Firing Rate (Hz)",
            rotation=270,
            labelpad=15,
        )

        ax.set_xlim(-window_pre, window_post)
        ax.set_ylim(n_neurons, 0)

    axes[-1].set_xlabel(f"Time from {event_label(align_event)} (s)", fontsize=12)

    plt.tight_layout()

    if save_flag:
        if len(region_acronyms) == 1:
            file_name = f"{pid}_population_sorted.png"
        else:
            region_tag = "_".join(region_acronyms)
            file_name = f"{pid}_population_sorted_{region_tag}.png"
        save_path = path_fig / file_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Population heatmap saved to: {save_path}")

    plt.show()

def plot_population_PSTH_sorted(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    save_flag,
    path_fig,
    pid,
    region_acronyms=None,
):
    """Plot stacked PSTHs sorted by delay for each neuron."""
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS", ["VISp"])
    elif isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]
    else:
        region_acronyms = list(region_acronyms)

    if len(region_acronyms) == 0:
        print("No regions provided for plot_population_PSTH_sorted.")
        return

    window_pre = config_plot["POP_WINDOW_PRE"]
    window_post = config_plot["POP_WINDOW_POST"]
    bin_size = config_plot["POP_BIN_SIZE"]
    smooth_sigma = config_plot["POP_SMOOTH_SIGMA"]
    normalize = config_plot["POP_NORMALIZE"]

    try:
        if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        elif hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        else:
            quality_mask = np.ones(len(clusters.acronym), dtype=bool)
    except AttributeError:
        print("Error: Cluster data incomplete.")
        quality_mask = np.ones(len(cluster_acronyms), dtype=bool)

    cluster_acronyms_str = cluster_acronyms.astype(str)
    label_text = "Good Neurons" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "Neurons"
    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in sl.trials.keys():
        print(f"Warning: Event '{align_event}' not found. Falling back to stimOn_times.")
        align_event = "stimOn_times"
    delay_col = delay_column_name(align_event)
    event_series = np.asarray(sl.trials[align_event])
    stim_times = event_series[~np.isnan(event_series)]

    bins = np.arange(-window_pre, window_post + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(
        len(region_acronyms), 1, figsize=(10, 6 * len(region_acronyms)), sharex=True
    )
    if len(region_acronyms) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_acronyms):
        try:
            region_mask = np.char.startswith(cluster_acronyms_str, region)
            if config_plot["PLOT_ONLY_GOOD_UNITS"]:
                final_mask = region_mask & quality_mask
            else:
                final_mask = region_mask
            region_ids = cluster_ids[final_mask]
            print(f"Found {len(region_ids)} {label_text} in {region}.")
        except AttributeError:
            region_ids = []
            print(f"Error: Cluster data incomplete for {region}.")

        df_region = pd.DataFrame({"cluster_id": region_ids})

        if df_res is not None and len(df_region) > 0:
            if delay_col in df_res.columns:
                df_region = df_region.merge(
                    df_res[["cluster_id", delay_col]].rename(columns={delay_col: "delay"}),
                    on="cluster_id",
                    how="left",
                )
            else:
                print(f"Warning: {delay_col} not found. Latencies will be NaN.")
                df_region["delay"] = np.nan
        else:
            if df_res is None:
                print("Warning: df_res not found. Latencies will be NaN.")
            df_region["delay"] = np.nan

        df_sorted = df_region.sort_values(
            by="delay", ascending=True, na_position="last"
        ).reset_index(drop=True)

        n_neurons = len(df_sorted)
        if n_neurons == 0:
            ax.text(
                0.5,
                0.5,
                f"No units found for {region}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-window_pre, window_post)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Neurons (Sorted by Latency)\nTotal: 0", fontsize=12)
            title_str = "Normalized " if normalize else "Raw "
            ax.set_title(
                f"{title_str}Average PSTHs | {region} Units", fontsize=14
            )
            continue

        if len(stim_times) == 0:
            ax.text(
                0.5,
                0.5,
                f"No valid {align_event} events for {region}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-window_pre, window_post)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Neurons (Sorted by Latency)\nTotal: 0", fontsize=12)
            continue

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            df_sorted["cluster_id"].values,
            stim_times,
            -window_pre,
            window_post,
            bin_size,
            smooth_sigma,
            show_progress=True,
            desc=f"PSTH ({region})",
        )

        psth_list = []
        for _, row in df_sorted.iterrows():
            cid = row["cluster_id"]
            psth_entry = psth_by_cluster.get(cid)
            if not psth_entry:
                psth_list.append(None)
                continue
            fr_smooth = psth_entry["fr_smooth"]
            if normalize:
                peak = np.max(fr_smooth)
                if peak > 0:
                    fr_smooth = fr_smooth / peak
            psth_list.append(fr_smooth)

        valid_psths = [p for p in psth_list if p is not None and len(p) > 0]
        if len(valid_psths) == 0:
            ax.text(
                0.5,
                0.5,
                f"No PSTHs available for {region}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-window_pre, window_post)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Neurons (Sorted by Latency)\nTotal: 0", fontsize=12)
            title_str = "Normalized " if normalize else "Raw "
            ax.set_title(
                f"{title_str}Average PSTHs | {region} Units", fontsize=14
            )
            continue

        if normalize:
            scale = 0.9
        else:
            global_peak = max(np.max(psth) for psth in valid_psths)
            scale = 0.9 / global_peak if global_peak > 0 else 1.0

        for row_idx, psth in enumerate(psth_list):
            if psth is None:
                continue
            y_vals = psth * scale + row_idx
            ax.plot(bin_centers, y_vals, color="black", linewidth=0.8, alpha=0.6)

        valid_delays = df_sorted.dropna(subset=["delay"])
        y_positions = valid_delays.index + 0.5
        x_positions = valid_delays["delay"]
        ax.scatter(x_positions, y_positions, color="black", s=10, marker="o", label="Delay")

        ax.axvline(0, color="black", linestyle="--", linewidth=1, label=event_label(align_event))
        ax.set_ylabel(f"Neurons (Sorted by Latency)\nTotal: {n_neurons}", fontsize=12)
        title_str = "Normalized " if normalize else "Raw "
        ax.set_title(
            f"{title_str}Average PSTHs | {region} Units", fontsize=14
        )
        ax.set_xlim(-window_pre, window_post)
        ax.set_ylim(n_neurons, 0)

    axes[-1].set_xlabel(f"Time from {event_label(align_event)} (s)", fontsize=12)

    plt.tight_layout()

    if save_flag:
        if len(region_acronyms) == 1:
            file_name = f"{pid}_population_psth_sorted.png"
        else:
            region_tag = "_".join(region_acronyms)
            file_name = f"{pid}_population_psth_sorted_{region_tag}.png"
        save_path = path_fig / file_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Population PSTH plot saved to: {save_path}")

    plt.show()


# %% Parameters ##############################################################################

CONFIG_CALC = {
    # Which atlas to use when assigning brain regions ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Run calculations only on good units (label == 1) or on all units
    "CALC_ONLY_GOOD_UNITS": True,
    # Events to compute delays for
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    # Delay calculation method: "center_of_mass", "psth_peak", or "tfs"
    "DELAY_METHOD": "psth_peak",
    # Values treated as 100% contrast for the TFS method
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    # PSTH bin width (seconds)
    "BIN_SIZE": 0.005,
    # Baseline window duration before the event (seconds)
    "BASELINE_PRE": 0.2,
    # PSTH window start/end relative to the chosen event (seconds)
    "PSTH_WINDOW_START": -0.2,
    "PSTH_WINDOW_END": 0.35,
    # Responsive window for delay relative to the event (seconds)
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.15,
    # Gaussian smoothing sigma for PSTH (in bins)
    "SMOOTH_SIGMA": 1,
    # Minimum trials required to include a unit
    "MIN_TRIALS": 50,
    # Reliability window for split-half delay (seconds)
    "RELIABILITY_WINDOW_START": 0.01,
    "RELIABILITY_WINDOW_END": 0.15,
}

CONFIG_PLOT = {
    # Which atlas to use for plots ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Plot only good units (label == 1) or all units
    "PLOT_ONLY_GOOD_UNITS": True,
    # Event to use for alignment and sorting
    "PLOT_EVENT": "stimOn_times",
    # Regions to plot when region_acronyms is not provided
    "PLOT_REGIONS": ["VISp", 'ENTm'],
    # Raster plot window around trial events (seconds)
    "RASTER_WINDOW_PRE": 1,
    "RASTER_WINDOW_POST": 2,
    "RASTER_ALIGN_TO_EVENT": True,
    "RASTER_ALIGN_TO_STIM_ON": True,
    # Single-neuron plot windows and PSTH bin size (seconds)
    "SINGLE_NEURON_RASTER_PRE": 0.5,
    "SINGLE_NEURON_RASTER_POST": 1.0,
    "SINGLE_NEURON_BIN_SIZE": 0.05,
    "SINGLE_NEURON_SMOOTH_SIGMA": 1,
    # Sequence raster window (seconds)
    "SEQUENCE_WINDOW_PRE": 0.5,
    "SEQUENCE_WINDOW_POST": 1.0,
    "SEQUENCE_ALIGN_TO_EVENT": True,
    "SEQUENCE_ALIGN_TO_STIM": True,
    # Population heatmap window and style
    "POP_WINDOW_PRE": 0.5,
    "POP_WINDOW_POST": 0.5,
    "POP_BIN_SIZE": 0.005,
    "POP_SMOOTH_SIGMA": 2,
    # Heatmap colormap (examples: "Greys", "viridis", "RdGy", "magma", "cividis")
    "POP_CMAP_NAME": "bwr",
    "POP_NORMALIZE": True,
}

# %% Loading Data ##############################################################################

base_path = Path(r"C:/Users/Experiment/Documents/Amirreza/SeqProject2026")
path_data, path_fig, path_data_processed, ibl_cache = setup_paths(base_path)
print(f"Directories ready. Cache: {ibl_cache}")

one = init_one(ibl_cache)
ba = AllenAtlas()

br = BrainRegions()
beryl_indices = np.unique(br.mappings["Beryl"])
beryl_acronyms = br.acronym[beryl_indices]
beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["void"]))
beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["root"]))
prepare_region_dirs(path_data, beryl_acronyms)

hier_file = os.path.join(path_data, "hierarchy_summary_CreConf_all_regions.csv")
hier_df = pd.read_csv(hier_file)
area_to_hier_score = dict(zip(hier_df["areas"], hier_df["CC+TC+CT iterated"]))
hier_scores = np.array([area_to_hier_score.get(region, np.nan) for region in beryl_acronyms])

# Great for CP and MOp: 
# pid = '26118c10-35dd-4ab1-9f0f-b9a89a1da070'

pid = '3d3d5a5e-df26-43ee-80b6-2d72d85668a5' # "c9664185-d3fd-4e0e-89cf-77c402038938"
print(f"\nProcessing PID: {pid}")

ssl, spikes, clusters, sl = load_session_data(pid, one, ba)
pupil_features, pupil_times = load_pupil_data(sl)

# Resolve cluster IDs for safe indexing.
cluster_ids, cid_to_idx = build_cluster_id_map(clusters)

# Map acronyms once for calculations and for plots (can be different atlas choices)
cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

# Build event-aligned arrays for each requested event.
events_by_name, contrasts_by_name = build_event_dicts(
    sl, CONFIG_CALC["EVENT_NAMES"], CONFIG_CALC["MIN_TRIALS"]
)

# %% Calculations ###########################################################################

df_res = calculate_delays(
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
df_reliability = calculate_delay_reliability(
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

# %% Select Trial and Unit to Plot ###########################################################

trial_idx = 79
single_neuron_id = 559

# %% Plot Single Trial Raster ##############################################################################

plot_trial_raster(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms_plot,
    sl,
    pupil_features,
    pupil_times,
    CONFIG_PLOT,
    pid,
    path_fig,
    save_figure=True,
    trial_idx=trial_idx,
)

# %% Plot delay histogram and reliability ############################################################

plot_delay_histogram(
    df_res, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

plot_delay_reliability(
    df_reliability, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

# %% Single Neuron Plots ###########################################################################

# single_neuron_id = 725

plot_single_neuron(
    sl,
    spikes,
    clusters,
    cluster_acronyms_plot,
    cid_to_idx,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    cluster_id=single_neuron_id,
)

# %% Sequence Plots ###########################################################################

# trial_idx = 843

CONFIG_PLOT.update(
    {
        "PLOT_REGIONS": ['MOp', 'CP'],
        'PLOT_ONLY_GOOD_UNITS': True,
    })

regions_to_plot = CONFIG_PLOT["PLOT_REGIONS"]

plot_sequence_raster(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    trial_idx=trial_idx,
    region_acronyms=regions_to_plot,
)


plot_population_sorted(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    region_acronyms=regions_to_plot,
)

plot_population_PSTH_sorted(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    region_acronyms=regions_to_plot,
)

