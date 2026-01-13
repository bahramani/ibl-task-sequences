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

def get_psth_data(neuron_spikes, event_times, config):
    """Compute a smoothed PSTH for a neuron given event times."""
    # Guard against missing inputs early to keep downstream logic simple.
    if len(neuron_spikes) == 0 or len(event_times) == 0:
        return None, None

    # Define the PSTH bins around stimulus onset.
    bins = np.arange(
        config["PSTH_WINDOW_START"],
        config["PSTH_WINDOW_END"] + config["BIN_SIZE"],
        config["BIN_SIZE"],
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Limit spikes to the global min/max window across all trials for efficiency.
    s_min = event_times.min() + config["PSTH_WINDOW_START"]
    s_max = event_times.max() + config["PSTH_WINDOW_END"]
    subset_spikes = neuron_spikes[(neuron_spikes >= s_min) & (neuron_spikes <= s_max)]
    if len(subset_spikes) == 0:
        return None, None

    # Collect trial-aligned spikes so we can compute the pooled PSTH.
    relative_spikes = []
    for t_ev in event_times:
        t0 = t_ev + config["PSTH_WINDOW_START"]
        t1 = t_ev + config["PSTH_WINDOW_END"]
        in_trial = subset_spikes[(subset_spikes >= t0) & (subset_spikes <= t1)]
        relative_spikes.append(in_trial - t_ev)

    if len(relative_spikes) == 0:
        return None, None

    # Histogram and smooth to obtain the PSTH.
    all_rel_spikes = np.concatenate(relative_spikes)
    counts, _ = np.histogram(all_rel_spikes, bins=bins)
    fr = counts / len(event_times) / config["BIN_SIZE"]
    fr_smooth = gaussian_filter1d(fr, sigma=config["SMOOTH_SIGMA"])
    return fr_smooth, bin_centers


def calculate_delay(
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
    - "tfs": time to first spike after stimulus onset (100% contrast trials only).
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
    if fr_smooth is None or bin_centers is None:
        return np.nan, False

    # Compute a baseline threshold to define responsiveness.
    idx_baseline = (bin_centers >= -config["BASELINE_PRE"]) & (bin_centers < 0)
    baseline_fr = fr_smooth[idx_baseline]
    if len(baseline_fr) == 0:
        return np.nan, False

    threshold = np.mean(baseline_fr) + 2 * np.std(baseline_fr)
    idx_responsive = (bin_centers >= config["RESPONSIVE_WINDOW_START"]) & (
        bin_centers <= config["RESPONSIVE_WINDOW_END"]
    )

    responsive_mask = idx_responsive & (fr_smooth > threshold)
    if not np.any(responsive_mask):
        return np.nan, False

    resp_fr = fr_smooth[responsive_mask]
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


def calculate_delays(
    spikes,
    clusters,
    cluster_acronyms,
    events,
    trial_contrasts,
    config,
    path_data_processed,
    pid,
):
    """Compute delays for all clusters and save results to processed data."""
    n_trials = len(events)
    print(f"Processing {n_trials} trials...")
    if n_trials < config["MIN_TRIALS"]:
        print("Not enough trials for delay calculation.")
        return pd.DataFrame()

    # Loop over each unit, compute its PSTH, and extract a delay metric.
    cluster_ids = np.unique(spikes.clusters)
    print(f"Found {len(cluster_ids)} clusters.")

    results = []
    for cid in cluster_ids:
        neuron_spikes = spikes.times[spikes.clusters == cid]
        if len(neuron_spikes) == 0:
            continue

        # PSTH-based delay methods use the smoothed firing rate.
        fr_smooth, bin_centers = get_psth_data(neuron_spikes, events, config)
        delay, is_responsive = calculate_delay(
            fr_smooth,
            bin_centers,
            config,
            method=config.get("DELAY_METHOD"),
            neuron_spikes=neuron_spikes,
            event_times=events,
            trial_contrasts=trial_contrasts,
        )

        # Map the cluster to its region acronym for downstream grouping.
        try:
            acronym = cluster_acronyms[cid]
        except Exception:
            acronym = "Unknown"

        # Determine if the unit is "good" based on available labels.
        try:
            if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
                label = clusters.metrics.label[cid]
            elif hasattr(clusters, "label"):
                label = clusters.label[cid]
            else:
                label = 1
        except Exception:
            label = 1

        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
            continue

        results.append(
            {
                "cluster_id": cid,
                "acronym": acronym,
                "label": label,
                "delay": delay,
                "responsive": is_responsive,
            }
        )

    df_res = pd.DataFrame(results)
    output_path = path_data_processed / f"{pid}_delay_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Computed delays for {len(df_res)} neurons. Saved to {output_path}.")
    return df_res


def calculate_delay_reliability(
    spikes,
    clusters,
    cluster_acronyms,
    events,
    trial_contrasts,
    config,
    path_data_processed,
    pid,
):
    """Compute split-half delay reliability and save results to processed data."""
    n_total = len(events)
    if n_total < config["MIN_TRIALS"]:
        print("Not enough trials for reliability calculation.")
        return pd.DataFrame()

    # Split trials into two halves to compute reliability across conditions.
    mid_idx = n_total // 2
    events_h1 = events[:mid_idx]
    events_h2 = events[mid_idx:]
    contrasts_h1 = trial_contrasts[:mid_idx]
    contrasts_h2 = trial_contrasts[mid_idx:]

    cluster_ids = np.unique(spikes.clusters)
    results = []

    for cid in cluster_ids:
        try:
            acronym = cluster_acronyms[cid]
            if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
                label = clusters.metrics.label[cid]
            elif hasattr(clusters, "label"):
                label = clusters.label[cid]
            else:
                label = 1
        except Exception:
            acronym = "Unknown"
            label = 1

        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
            continue

        neuron_spikes = spikes.times[spikes.clusters == cid]
        # Only retain units that are responsive in the full dataset.
        fr_all, bins_all = get_psth_data(neuron_spikes, events, config)
        _, is_responsive = calculate_delay(
            fr_all,
            bins_all,
            config,
            method=config.get("DELAY_METHOD"),
            neuron_spikes=neuron_spikes,
            event_times=events,
            trial_contrasts=trial_contrasts,
        )
        if not is_responsive:
            continue

        fr_h1, bins_h1 = get_psth_data(neuron_spikes, events_h1, config)
        delay_h1, _ = calculate_delay(
            fr_h1,
            bins_h1,
            {
                **config,
                "RESPONSIVE_WINDOW_START": config["RELIABILITY_WINDOW_START"],
                "RESPONSIVE_WINDOW_END": config["RELIABILITY_WINDOW_END"],
            },
            method=config.get("DELAY_METHOD"),
            neuron_spikes=neuron_spikes,
            event_times=events_h1,
            trial_contrasts=contrasts_h1,
        )

        fr_h2, bins_h2 = get_psth_data(neuron_spikes, events_h2, config)
        delay_h2, _ = calculate_delay(
            fr_h2,
            bins_h2,
            {
                **config,
                "RESPONSIVE_WINDOW_START": config["RELIABILITY_WINDOW_START"],
                "RESPONSIVE_WINDOW_END": config["RELIABILITY_WINDOW_END"],
            },
            method=config.get("DELAY_METHOD"),
            neuron_spikes=neuron_spikes,
            event_times=events_h2,
            trial_contrasts=contrasts_h2,
        )

        if not np.isnan(delay_h1) and not np.isnan(delay_h2):
            results.append(
                {
                    "cluster_id": cid,
                    "acronym": acronym,
                    "delay_h1": delay_h1,
                    "delay_h2": delay_h2,
                }
            )

    df_reliability = pd.DataFrame(results)
    output_path = path_data_processed / f"{pid}_delay_reliability.csv"
    df_reliability.to_csv(output_path, index=False)
    print(
        f"Found {len(df_reliability)} responsive neurons (both halves). Saved to {output_path}."
    )
    return df_reliability


def plot_trial_raster(
    spikes,
    clusters,
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
    t_stim_on = sl.trials["stimOn_times"][trial_idx]
    t_first_move = sl.trials["firstMovement_times"][trial_idx]
    t_feedback = sl.trials["feedback_times"][trial_idx]

    t_start = t_stim_on - config_plot["RASTER_WINDOW_PRE"]
    t_end = t_feedback + config_plot["RASTER_WINDOW_POST"]

    if config_plot["RASTER_ALIGN_TO_STIM_ON"]:
        t_offset = t_stim_on
        xlabel_text = "Time from Stim On (s)"
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
        target_indices = np.where(clusters.label == 1)[0]
        ylabel_text = f"Good Units (n={len(target_indices)})"
    else:
        target_indices = np.arange(len(clusters.label))
        ylabel_text = f"All Units (n={len(target_indices)})"

    df_units = pd.DataFrame(
        {
            "id": target_indices,
            "acronym": cluster_acronyms[target_indices],
            "depth": clusters.depths[target_indices],
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
        unit_spike_times = window_spike_times[window_spike_clusters == row["id"]]
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
    df_plot = df_res.dropna(subset=["delay"]).copy()

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

    data_to_plot = [df_plot[df_plot["plot_region"] == r]["delay"].values for r in unique_regions]

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
        f"Distribution of Response Delays {title_suffix}\n"
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
    print(df_plot.groupby("plot_region")["delay"].describe()[["count", "mean", "std"]])


def plot_delay_reliability(df_reliability, config_calc, config_plot, save_flag, path_fig, pid):
    """Plot split-half delay reliability scatter by region."""
    if len(df_reliability) == 0:
        print("No neurons met the reliability criteria.")
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

    r_val, p_val = stats.pearsonr(df_reliability["delay_h1"], df_reliability["delay_h2"])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=df_reliability,
        x="delay_h1",
        y="delay_h2",
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
        min(df_reliability["delay_h1"].min(), df_reliability["delay_h2"].min()),
        max(df_reliability["delay_h1"].max(), df_reliability["delay_h2"].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", alpha=0.5, label="Identity")

    unit_type = "Good Units" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "All Units"
    ax.set_xlabel("Delay (s) - First Half")
    ax.set_ylabel("Delay (s) - Second Half")
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
    sl, spikes, clusters, cluster_acronyms, df_res, config_plot, save_flag, path_fig, pid, cluster_id
):
    """Plot PSTHs and rasters for a single neuron, plus global PSTH."""
    target_cluster_id = cluster_id
    if target_cluster_id >= len(clusters.acronym):
        print(
            f"Error: Cluster ID {target_cluster_id} is out of bounds "
            f"(Max ID: {len(clusters.acronym) - 1})."
        )
        target_cluster_id = 0

    target_acronym = cluster_acronyms[target_cluster_id]
    print(f"Analyzing Cluster ID: {target_cluster_id} | Region: {target_acronym}")

    unit_delay = np.nan
    if df_res is not None:
        match = df_res[df_res["cluster_id"] == target_cluster_id]
        if not match.empty:
            unit_delay = match.iloc[0]["delay"]
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

            mask = (contrast_arr == cont) & (~np.isnan(sl.trials.stimOn_times))
            events = sl.trials.stimOn_times[mask]

            if len(events) == 0:
                continue

            psth_spikes_concat = []
            for event_t in events:
                t_start = event_t - config_plot["SINGLE_NEURON_RASTER_PRE"]
                t_end = event_t + config_plot["SINGLE_NEURON_RASTER_POST"]
                in_window = neuron_spikes[(neuron_spikes >= t_start) & (neuron_spikes <= t_end)]
                psth_spikes_concat.append(in_window - event_t)

            psth_bins = np.arange(
                -config_plot["SINGLE_NEURON_RASTER_PRE"],
                config_plot["SINGLE_NEURON_RASTER_POST"] + config_plot["SINGLE_NEURON_BIN_SIZE"],
                config_plot["SINGLE_NEURON_BIN_SIZE"],
            )

            if len(psth_spikes_concat) > 0:
                all_spikes = np.concatenate(psth_spikes_concat)
                counts, _ = np.histogram(all_spikes, bins=psth_bins)
                firing_rate = counts / len(events) / config_plot["SINGLE_NEURON_BIN_SIZE"]
            else:
                firing_rate = np.zeros(len(psth_bins) - 1)

            color_val = contrast_colors.get(cont, (0, 0, 0, 1))
            label_text = f"{cont * 100:.0f}%" if cont > 0 else "0%"
            ax_psth.plot(psth_bins[:-1], firing_rate, color=color_val, linewidth=2, label=label_text)

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

    all_events = sl.trials.stimOn_times[~np.isnan(sl.trials.stimOn_times)]
    if len(all_events) > 0:
        psth_spikes_global = []
        bins_global = np.arange(
            -config_plot["SINGLE_NEURON_RASTER_PRE"],
            config_plot["SINGLE_NEURON_RASTER_POST"] + config_plot["SINGLE_NEURON_BIN_SIZE"],
            0.01,
        )

        s_min = all_events.min() - config_plot["SINGLE_NEURON_RASTER_PRE"]
        s_max = all_events.max() + config_plot["SINGLE_NEURON_RASTER_POST"]
        subset_spikes = neuron_spikes[(neuron_spikes >= s_min) & (neuron_spikes <= s_max)]

        for t_ev in all_events:
            t0 = t_ev - config_plot["SINGLE_NEURON_RASTER_PRE"]
            t1 = t_ev + config_plot["SINGLE_NEURON_RASTER_POST"]
            in_trial = subset_spikes[(subset_spikes >= t0) & (subset_spikes <= t1)]
            psth_spikes_global.append(in_trial - t_ev)

        if len(psth_spikes_global) > 0:
            all_spikes_global = np.concatenate(psth_spikes_global)
            counts_g, _ = np.histogram(all_spikes_global, bins=bins_global)
            fr_global = counts_g / len(all_events) / 0.01
            fr_global_smooth = gaussian_filter1d(fr_global, sigma=2)

            bin_centers = (bins_global[:-1] + bins_global[1:]) / 2
            ax_global.plot(bin_centers, fr_global, color="k", linewidth=2, label="Global Response")
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

        ax_global.axvline(0, color="blue", linestyle="--", linewidth=1, label="Stim On")
        ax_global.set_ylabel("Firing Rate (Hz)")
        ax_global.set_xlabel("Time from Stimulus Onset (s)")
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
    sl, spikes, clusters, cluster_acronyms, df_res, config_plot, save_flag, path_fig, pid, trial_idx
):
    """Plot a VISp raster sorted by delay for a single trial."""
    window_pre = config_plot["SEQUENCE_WINDOW_PRE"]
    window_post = config_plot["SEQUENCE_WINDOW_POST"]
    align_to_stim = config_plot["SEQUENCE_ALIGN_TO_STIM"]

    try:
        visp_mask = np.char.startswith(cluster_acronyms.astype(str), "VISp")
        if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        elif hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        else:
            quality_mask = np.ones(len(clusters.acronym), dtype=bool)

        if config_plot["PLOT_ONLY_GOOD_UNITS"]:
            final_mask = visp_mask & quality_mask
        else:
            final_mask = visp_mask
        visp_cluster_ids = np.where(final_mask)[0]
        label_text = "Good Neurons" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "Neurons"
        print(f"Found {len(visp_cluster_ids)} {label_text} in VISp layers.")
    except AttributeError:
        print("Error: clusters data incomplete.")
        visp_cluster_ids = []

    df_visp = pd.DataFrame(
        {
            "cluster_id": visp_cluster_ids,
            "acronym": cluster_acronyms[visp_cluster_ids],
            "depth": clusters.depths[visp_cluster_ids],
        }
    )

    if df_res is not None:
        df_visp = df_visp.merge(df_res[["cluster_id", "delay"]], on="cluster_id", how="left")
    else:
        print("Warning: df_res not found. Sorting by depth instead.")
        df_visp["delay"] = np.nan

    df_sorted = df_visp.sort_values(by="delay", ascending=True, na_position="first").reset_index(
        drop=True
    )

    n_responsive = df_sorted["delay"].notna().sum()
    n_nan = df_sorted["delay"].isna().sum()
    print(f"Sorting Complete: {n_responsive} sequenced, {n_nan} unresponsive (bottom).")

    t_stim = sl.trials["stimOn_times"][trial_idx]
    t_move = sl.trials["firstMovement_times"][trial_idx]
    t_feed = sl.trials["feedback_times"][trial_idx]

    t_start = t_stim - window_pre
    t_end = t_stim + window_post
    t_offset = t_stim if align_to_stim else 0

    mask_spikes = (np.isin(spikes.clusters, visp_cluster_ids)) & (
        spikes.times >= t_start
    ) & (spikes.times <= t_end)

    relevant_spikes = spikes.times[mask_spikes]
    relevant_clusters = spikes.clusters[mask_spikes]

    fig, ax_raster = plt.subplots(figsize=(10, 8))
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
    ax_raster.set_ylim(-1, len(df_sorted))
    ax_raster.set_ylabel(
        f"VISp Neurons (n={len(df_sorted)})\nSorted by Delay (NaN bottom)",
        fontsize=12,
    )
    ax_raster.set_xlabel("Time from Stimulus Onset (s)", fontsize=12)
    ax_raster.set_title(
        f"Trial {trial_idx} | VISp Sequence (Good Units Only)", fontsize=14
    )

    if n_nan > 0:
        ax_raster.axhline(n_nan, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax_raster.text(
            (t_start - t_offset) + 0.05,
            n_nan / 2,
            "Unresponsive / Untuned",
            color="gray",
            fontsize=10,
            va="center",
            fontweight="bold",
        )
        ax_raster.text(
            (t_start - t_offset) + 0.05,
            n_nan + (n_responsive / 2),
            "Sequenced Activity",
            color="black",
            fontsize=10,
            va="center",
            fontweight="bold",
        )

    ax_raster.axvline(0, color="blue", linewidth=2, label="Stim On")
    if not np.isnan(t_move):
        ax_raster.axvline(
            t_move - t_offset, color="green", linewidth=2, linestyle="--", label="Move"
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

    plt.tight_layout()

    if save_flag:
        save_path = path_fig / f"{pid}_trial_{trial_idx}_sequence.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sequence raster saved to: {save_path}")

    plt.show()


def plot_population_sorted(
    sl, spikes, clusters, cluster_acronyms, df_res, config_plot, save_flag, path_fig, pid
):
    """Plot a population heatmap sorted by delay with delay markers."""
    window_pre = config_plot["POP_WINDOW_PRE"]
    window_post = config_plot["POP_WINDOW_POST"]
    bin_size = config_plot["POP_BIN_SIZE"]
    smooth_sigma = config_plot["POP_SMOOTH_SIGMA"]
    cmap_name = config_plot["POP_CMAP_NAME"]
    normalize = config_plot["POP_NORMALIZE"]

    try:
        visp_mask = np.char.startswith(cluster_acronyms.astype(str), "VISp")
        if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
            quality_mask = clusters.metrics.label == 1
        elif hasattr(clusters, "label"):
            quality_mask = clusters.label == 1
        else:
            quality_mask = np.ones(len(clusters.acronym), dtype=bool)

        if config_plot["PLOT_ONLY_GOOD_UNITS"]:
            final_mask = visp_mask & quality_mask
        else:
            final_mask = visp_mask
        visp_ids = np.where(final_mask)[0]
        label_text = "Good Neurons" if config_plot["PLOT_ONLY_GOOD_UNITS"] else "Neurons"
        print(f"Found {len(visp_ids)} {label_text} in VISp.")
    except AttributeError:
        visp_ids = []
        print("Error: Cluster data incomplete.")

    df_visp = pd.DataFrame({"cluster_id": visp_ids})

    if df_res is not None:
        df_visp = df_visp.merge(df_res[["cluster_id", "delay"]], on="cluster_id", how="left")
    else:
        print("Warning: df_res not found. Latencies will be NaN.")
        df_visp["delay"] = np.nan

    df_sorted = df_visp.sort_values(by="delay", ascending=True, na_position="first").reset_index(
        drop=True
    )

    bins = np.arange(-window_pre, window_post + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(bin_centers)
    n_neurons = len(df_sorted)

    psth_matrix = np.zeros((n_neurons, n_bins))
    stim_times = sl.trials["stimOn_times"][~np.isnan(sl.trials["stimOn_times"])]

    for row_idx, row in df_sorted.iterrows():
        cid = row["cluster_id"]
        unit_spikes = spikes.times[spikes.clusters == cid]
        if len(unit_spikes) == 0:
            continue

        all_rel_spikes = []
        t_min = stim_times.min() - window_pre
        t_max = stim_times.max() + window_post
        subset = unit_spikes[(unit_spikes >= t_min) & (unit_spikes <= t_max)]

        for t_stim in stim_times:
            t0 = t_stim - window_pre
            t1 = t_stim + window_post
            in_window = subset[(subset >= t0) & (subset <= t1)]
            all_rel_spikes.append(in_window - t_stim)

        if len(all_rel_spikes) > 0:
            flat_spikes = np.concatenate(all_rel_spikes)
            counts, _ = np.histogram(flat_spikes, bins=bins)
            fr = counts / len(stim_times) / bin_size
            fr_smooth = gaussian_filter1d(fr, sigma=smooth_sigma)
            if normalize:
                peak = np.max(fr_smooth)
                if peak > 0:
                    fr_smooth = fr_smooth / peak
            psth_matrix[row_idx, :] = fr_smooth

    fig, ax = plt.subplots(figsize=(10, 10))
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

    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Stim On")
    ax.set_xlabel("Time from Stimulus Onset (s)", fontsize=12)
    ax.set_ylabel(f"Neurons (Sorted by Latency)\nTotal: {n_neurons}", fontsize=12)
    title_str = "Normalized " if normalize else "Raw "
    ax.set_title(f"{title_str}Average Response (PSTH) Heatmap | VISp Units", fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Normalized Firing Rate" if normalize else "Firing Rate (Hz)",
        rotation=270,
        labelpad=15,
    )

    ax.set_xlim(-window_pre, window_post)
    ax.set_ylim(0, n_neurons)

    plt.tight_layout()

    if save_flag:
        save_path = path_fig / f"{pid}_population_sorted.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Population heatmap saved to: {save_path}")

    plt.show()

    CONFIG_CALC = {
    # Which atlas to use when assigning brain regions ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Run calculations only on good units (label == 1) or on all units
    "CALC_ONLY_GOOD_UNITS": False,
    # Delay calculation method: "center_of_mass", "psth_peak", or "tfs"
    "DELAY_METHOD": "psth_peak",
    # Values treated as 100% contrast for the TFS method
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    # PSTH bin width (seconds)
    "BIN_SIZE": 0.005,
    # Baseline window duration before stimulus onset (seconds)
    "BASELINE_PRE": 0.2,
    # PSTH window start/end relative to stimulus onset (seconds)
    "PSTH_WINDOW_START": -0.2,
    "PSTH_WINDOW_END": 0.35,
    # Responsive window for delay (center of mass) after stimulus onset (seconds)
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.350,
    # Gaussian smoothing sigma for PSTH (in bins)
    "SMOOTH_SIGMA": 1,
    # Minimum trials required to include a unit
    "MIN_TRIALS": 50,
    # Reliability window for split-half delay (seconds)
    "RELIABILITY_WINDOW_START": 0.01,
    "RELIABILITY_WINDOW_END": 0.3,
}

CONFIG_PLOT = {
    # Which atlas to use for plots ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Plot only good units (label == 1) or all units
    "PLOT_ONLY_GOOD_UNITS": False,
    # Raster plot window around trial events (seconds)
    "RASTER_WINDOW_PRE": 1,
    "RASTER_WINDOW_POST": 2,
    "RASTER_ALIGN_TO_STIM_ON": True,
    # Single-neuron plot windows and PSTH bin size (seconds)
    "SINGLE_NEURON_RASTER_PRE": 0.5,
    "SINGLE_NEURON_RASTER_POST": 1.0,
    "SINGLE_NEURON_BIN_SIZE": 0.05,
    "SINGLE_NEURON_SMOOTH_SIGMA": 1,
    # Sequence raster window (seconds)
    "SEQUENCE_WINDOW_PRE": 0.5,
    "SEQUENCE_WINDOW_POST": 1.0,
    "SEQUENCE_ALIGN_TO_STIM": True,
    # Population heatmap window and style
    "POP_WINDOW_PRE": 1,
    "POP_WINDOW_POST": 1,
    "POP_BIN_SIZE": 0.005,
    "POP_SMOOTH_SIGMA": 2,
    # Heatmap colormap (examples: "Greys", "viridis", "plasma", "magma", "cividis")
    "POP_CMAP_NAME": "bwr",
    "POP_NORMALIZE": True,
}

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

pid = "c9664185-d3fd-4e0e-89cf-77c402038938"
print(f"\nProcessing PID: {pid}")

ssl, spikes, clusters, sl = load_session_data(pid, one, ba)
pupil_features, pupil_times = load_pupil_data(sl)

# Align trial-level arrays by filtering to valid stimulus onset times.
valid_trial_mask = ~np.isnan(sl.trials.stimOn_times)
events = sl.trials.stimOn_times[valid_trial_mask]
contrast_left = np.abs(sl.trials.contrastLeft[valid_trial_mask])
contrast_right = np.abs(sl.trials.contrastRight[valid_trial_mask])
trial_contrasts = np.nanmax(np.vstack([contrast_left, contrast_right]), axis=0)
trial_contrasts = np.where(np.isnan(trial_contrasts), 0, trial_contrasts)

# Map acronyms once for calculations and for plots (can be different atlas choices)
cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

df_res = calculate_delays(
    spikes,
    clusters,
    cluster_acronyms_calc,
    events,
    trial_contrasts,
    CONFIG_CALC,
    path_data_processed,
    pid,
)
df_reliability = calculate_delay_reliability(
    spikes,
    clusters,
    cluster_acronyms_calc,
    events,
    trial_contrasts,
    CONFIG_CALC,
    path_data_processed,
    pid,
)

trial_idx = 210
single_neuron_id = 656

plot_trial_raster(
    spikes,
    clusters,
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

plot_delay_histogram(
    df_res, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

plot_delay_reliability(
    df_reliability, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

plot_single_neuron(
    sl,
    spikes,
    clusters,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    cluster_id=single_neuron_id,
)

plot_sequence_raster(
    sl,
    spikes,
    clusters,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    trial_idx=trial_idx,
)


plot_population_sorted(
    sl,
    spikes,
    clusters,
    cluster_acronyms_plot,
    df_res,
    CONFIG_PLOT,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
)