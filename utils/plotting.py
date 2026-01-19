import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .analysis import (
    compute_psth_for_clusters, 
    get_event_time, 
    event_label, 
    delay_column_name, 
    reliability_column_names
)


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

    # Store sorted cluster information for each region
    region_sorted_info = {}

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

        # Store sorted cluster information for this region
        region_sorted_info[region] = df_sorted

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

    # Print sorted cluster IDs for each region
    print("\n" + "="*80)
    print("SORTED CLUSTER IDs BY REGION (sorted by delay)")
    print("="*80)
    for region, df_sorted in region_sorted_info.items():
        print(f"\n{region} ({len(df_sorted)} neurons):")
        print("-" * 80)
        if len(df_sorted) > 0:
            for idx, row in df_sorted.iterrows():
                delay_val = row["delay"]
                if pd.notna(delay_val):
                    print(f"  Position {idx:3d}: Cluster ID {int(row['cluster_id']):5d} | Delay: {delay_val*1000:6.2f} ms")
                else:
                    print(f"  Position {idx:3d}: Cluster ID {int(row['cluster_id']):5d} | Delay: N/A (unresponsive)")
        else:
            print("  No neurons found")
    print("="*80 + "\n")

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