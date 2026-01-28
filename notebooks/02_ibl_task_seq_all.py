# %% Imports
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

from utils.io import setup_paths, init_one, prepare_region_dirs, map_acronyms, load_session_data, build_cluster_id_map, load_pupil_data
import utils.analysis as ana_utils
import utils.plotting as plot_utils

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions



# %% Parameters ##############################################################################

CONFIG_CALC = {
    # Which atlas to use when assigning brain regions ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Run calculations only on good units (label == 1) or on all units
    "CALC_ONLY_GOOD_UNITS": True,
    # Load spontaneous data
    "CALC_SPONT": True,
    # Events to compute delays for
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    # Delay calculation method: "center_of_mass", "psth_peak", or "tfs"
    "DELAY_METHOD": "center_of_mass",
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
    "RESPONSIVE_WINDOW_END": 0.35,
    # Gaussian smoothing sigma for PSTH (in bins)
    "SMOOTH_SIGMA": 1,
    # Minimum trials required to include a unit
    "MIN_TRIALS": 50,
    # Reliability window for split-half delay (seconds)
    "RELIABILITY_WINDOW_START": 0.01,
    "RELIABILITY_WINDOW_END": 0.15,
    # Spike-triggered population coupling settings
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 100,
    "STPR_SMOOTH_SIGMA_MS": 5,
    # Use only good units when building population rate for stPR
    "STPR_POP_USE_GOOD_UNITS": False,
    # Combine spikes/clusters across all PIDs in the same EID (negative = current behavior)
    "COMBINE_PIDS": False,
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
    "SORT_BY_SPONT": True
}

# %% Loading Data ##############################################################################

base_path = Path(r"C:/Users/Experiment/Documents/Amirreza/SeqProject2026")
path_data, path_fig, path_data_processed, ibl_cache = setup_paths(base_path)
print(f"Directories ready. Cache: {ibl_cache}")

one = init_one(ibl_cache)

ba, br, beryl_acronyms, hier_scores = prepare_region_dirs(path_data)

# Great for double response after feedback
# pid = '9e069684-a4be-4b70-b9e6-446309f977d4'

# Great for CP and MOp:
# pid = '26118c10-35dd-4ab1-9f0f-b9a89a1da070'
# pid = 'f475ae14-9415-453e-b800-1480ea1c868d'

# Main one for VISp and EnTm:
pid = "c9664185-d3fd-4e0e-89cf-77c402038938"

# Good for MOs, seems responsive to first movement
# pid = 'acf04c3f-650a-4de0-b0d0-edf695dd2025'

# The one that is not working well:
# pid = # '3d3d5a5e-df26-43ee-80b6-2d72d85668a5'
print(f"\nProcessing PID: {pid}")

ssl, spikes, clusters, sl = load_session_data(pid, one, ba)
pupil_features, pupil_times = load_pupil_data(sl)

eid = one.pid2eid(pid)[0]

# Optionally combine spikes/clusters across all other PIDs for this EID.
if CONFIG_CALC["COMBINE_PIDS"]:
    all_pids = one.eid2pid(eid)
    if isinstance(all_pids, tuple):
        all_pids = all_pids[0]
    pid_str = str(pid)
    all_pids = [str(p) for p in all_pids]
    all_pids = list(dict.fromkeys(all_pids))
    other_pids = [p for p in all_pids if p != pid_str]
    region_scope = "good units" if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"] else "all units"

    cluster_acronyms_calc_orig = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
    region_labels = np.asarray(cluster_acronyms_calc_orig)
    if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"] and "label" in clusters:
        region_labels = region_labels[np.asarray(clusters["label"]) == 1]
    if len(region_labels) > 0:
        region_counts = pd.Series(region_labels).value_counts().sort_index()
        region_summary = ", ".join(
            [f"{region} ({count})" for region, count in region_counts.items()]
        )
    else:
        region_summary = "None"
    print(f"Regions in original PID ({region_scope}): {region_summary}")

    if len(other_pids) == 0:
        print("No other PIDs found for this EID.")
    else:
        next_cluster_offset = int(np.max(np.asarray(clusters["cluster_id"]))) + 1
        for other_pid in other_pids:
            ssl_other, spikes_other, clusters_other, _sl_other = load_session_data(
                other_pid, one, ba
            )

            cluster_acronyms_other = map_acronyms(
                clusters_other, br, CONFIG_CALC["ATLAS_MAPPING"]
            )
            other_region_labels = np.asarray(cluster_acronyms_other)
            if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"] and "label" in clusters_other:
                other_region_labels = other_region_labels[
                    np.asarray(clusters_other["label"]) == 1
                ]
            if len(other_region_labels) > 0:
                other_counts = pd.Series(other_region_labels).value_counts().sort_index()
                other_summary = ", ".join(
                    [f"{region} ({count})" for region, count in other_counts.items()]
                )
            else:
                other_summary = "None"
            print(f"Regions in other PID {other_pid} ({region_scope}): {other_summary}")

            # Offset cluster IDs to avoid collisions across probes.
            clusters_other_ids = np.asarray(clusters_other["cluster_id"])
            clusters_other["cluster_id"] = clusters_other_ids + next_cluster_offset
            spikes_other["clusters"] = (
                np.asarray(spikes_other["clusters"]) + next_cluster_offset
            )

            # Combine spikes.
            for key in spikes.keys():
                if key in spikes_other:
                    spikes[key] = np.concatenate(
                        [np.asarray(spikes[key]), np.asarray(spikes_other[key])]
                    )

            # Combine clusters (handle pandas DataFrame metrics if present).
            cluster_keys = set(list(clusters.keys()) + list(clusters_other.keys()))
            for key in cluster_keys:
                if key not in clusters:
                    clusters[key] = clusters_other[key]
                    continue
                if key not in clusters_other:
                    continue
                base_val = clusters[key]
                other_val = clusters_other[key]
                if isinstance(base_val, pd.DataFrame) or isinstance(other_val, pd.DataFrame):
                    base_df = base_val if isinstance(base_val, pd.DataFrame) else pd.DataFrame(base_val)
                    other_df = other_val if isinstance(other_val, pd.DataFrame) else pd.DataFrame(other_val)
                    clusters[key] = pd.concat([base_df, other_df], ignore_index=True)
                else:
                    clusters[key] = np.concatenate(
                        [np.asarray(base_val), np.asarray(other_val)]
                    )

            next_cluster_offset = int(np.max(np.asarray(clusters["cluster_id"]))) + 1

# Resolve cluster IDs for safe indexing.
cluster_ids, cid_to_idx = build_cluster_id_map(clusters)

# Map acronyms once for calculations and for plots (can be different atlas choices)
cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

# Build event-aligned arrays for each requested event.
events_by_name, contrasts_by_name = ana_utils.build_event_dicts(
    sl, CONFIG_CALC["EVENT_NAMES"], CONFIG_CALC["MIN_TRIALS"]
)

# %% Load spontaneous activity periods
good_cluster_ids = None
if CONFIG_CALC["CALC_SPONT"]:
    print("\nLoading spontaneous activity period...")
    try:
        passive_times = one.load_dataset(eid, '*passivePeriods*', collection='alf')
        spont_intervals = np.array([[passive_times['spontaneousActivity'][0],
                                     passive_times['spontaneousActivity'][1]]])
        spont_duration_sec = spont_intervals[0][1] - spont_intervals[0][0]
        spont_duration_min = spont_duration_sec / 60
        print(f"Spontaneous interval: {spont_intervals[0][0]:.2f}s to {spont_intervals[0][1]:.2f}s "
              f"(duration: {spont_duration_sec:.2f}s = {spont_duration_min:.2f} min)")

        last_feedback_time = sl.trials['feedback_times'].iloc[-1]
        print(f"Last feedback event time: {last_feedback_time:.2f}s")

        # Filter spikes to only include spontaneous period
        valid_time_mask = np.zeros(len(spikes['times']), dtype=bool)
        for start, end in spont_intervals:
            valid_time_mask |= ((spikes['times'] >= start) & (spikes['times'] <= end))

        spikes_spont = {key: val[valid_time_mask] for key, val in spikes.items()}
        print(f"Spontaneous spikes: {len(spikes_spont['times'])} / {len(spikes['times'])} total spikes")

        # Filter for good units if specified
        if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"]:
            good_cluster_ids = clusters['cluster_id'][clusters['label'] == 1]
            good_spk_mask = np.isin(spikes_spont['clusters'], good_cluster_ids)
            spikes_spont_good = {key: val[good_spk_mask] for key, val in spikes_spont.items()}

            cluster_good_mask = np.isin(clusters['cluster_id'], good_cluster_ids)
            clusters_good = pd.DataFrame({key: val[cluster_good_mask] for key, val in clusters.items()})

            print(f"Good units in spontaneous period: {len(np.unique(spikes_spont_good['clusters']))} units, "
                  f"{len(spikes_spont_good['times'])} spikes")
        else:
            spikes_spont_good = spikes_spont
            clusters_good = clusters

    except Exception as e:
        print(f"Could not load spontaneous period: {e}")
        spont_intervals = None
        spikes_spont = None
        spikes_spont_good = None


# %% Calculations ###########################################################################

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
# %% ################################################
#####################################################

# Get the start of the spontaneous interval
spont_start = spont_intervals[0][0]
print(f"Spontaneous period starts at: {spont_start:.2f}s")

# Filter spikes for task period (everything before spontaneous interval)
task_time_mask = spikes['times'] < spont_start
spikes_task = {key: val[task_time_mask] for key, val in spikes.items()}

if CONFIG_CALC["CALC_SPONT"] and spikes_spont is not None:
    coupling_cluster_ids = (
        good_cluster_ids if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"] else cluster_ids
    )
    df_coupling = ana_utils.compute_population_coupling(
        spikes_spont,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=coupling_cluster_ids,
    )

if CONFIG_CALC["CALC_SPONT"] and spont_intervals is not None:
    print("\n=== Computing stPR for spontaneous vs task periods ===")

    # Select regions to analyze and plot (set to None to use all regions)
    regions_to_plot_comparison = None # Or specify like: ['VISp', 'ENTm']

    # Get the start of the spontaneous interval
    spont_start = spont_intervals[0][0]
    print(f"Spontaneous period starts at: {spont_start:.2f}s")

    # Filter spikes for task period (everything before spontaneous interval)
    task_time_mask = spikes['times'] < spont_start
    spikes_task = {key: val[task_time_mask] for key, val in spikes.items()}
    print(f"Task period spikes: {len(spikes_task['times'])} / {len(spikes['times'])} total spikes")

    # Filter for good units if specified
    if CONFIG_CALC["CALC_ONLY_GOOD_UNITS"]:
        good_task_mask = np.isin(spikes_task['clusters'], good_cluster_ids)
        spikes_task_good = {key: val[good_task_mask] for key, val in spikes_task.items()}
        print(f"Good units in task period: {len(np.unique(spikes_task_good['clusters']))} units, "
              f"{len(spikes_task_good['times'])} spikes")
    else:
        spikes_task_good = spikes_task

    # Calculate stPR for task period
    print("\nCalculating stPR for task period...")
    df_coupling_task = ana_utils.compute_population_coupling(
        spikes_task,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=coupling_cluster_ids,
    )

    # Merge spontaneous and task stPR dataframes
    df_comparison = df_coupling.merge(
        df_coupling_task[['cluster_id', 'coupling_strength', 'coupling_delay_ms', 'sorting_number']],
        on='cluster_id',
        suffixes=('_spont', '_task')
    )

    print(f"\nFound {len(df_comparison)} neurons with stPR in both periods")

# %%

# rastermap_params = {
#     "n_clusters": 100,
#     "n_PCs": 64,
#     "locality": 0.5,
#     "time_lag_window": 15,
#     "grid_upsample": 0,
# }
#
# df_rastermap = ana_utils.compute_rastermap_sorting(
#     spikes,
#     cluster_ids,
#     cluster_acronyms_calc,
#     bin_size=0.01,
#     rastermap_params=rastermap_params,
#     separate_by_region=True,
#     region_acronyms=None,
# )

# %% Select Trial and Unit to Plot ###########################################################
trial_idx = 385 # 385 in Zador is great
single_neuron_id = 559

# %% Plot Single Trial Raster ##############################################################################
CONFIG_PLOT.update(
    {"PLOT_ONLY_GOOD_UNITS": True}
)

plot_utils.plot_trial_raster(
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

plot_utils.plot_delay_histogram(
    df_res, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

plot_utils.plot_delay_reliability(
    df_reliability, CONFIG_CALC, CONFIG_PLOT, save_flag=True, path_fig=path_fig, pid=pid
)

# %% Single Neuron Plots ###########################################################################

single_neuron_id = 201

plot_utils.plot_single_neuron(
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
        "PLOT_REGIONS": ['VISp', 'ENTm'],
        'PLOT_ONLY_GOOD_UNITS': 1,
        'SORT_BY_SPONT': 1,
        'SORT_BY_RASTERMAP': False,
    })

regions_to_plot = CONFIG_PLOT["PLOT_REGIONS"]

# plot_utils.plot_sequence_raster(
#     sl,
#     spikes,
#     clusters,
#     cluster_ids,
#     cluster_acronyms_plot,
#     df_res,
#     CONFIG_PLOT,
#     save_flag=True,
#     path_fig=path_fig,
#     pid=pid,
#     trial_idx=trial_idx,
#     region_acronyms=regions_to_plot,
# )

plot_utils.plot_population_sorted(
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
    df_coupling=df_coupling,
    #df_rastermap=df_rastermap,
    region_acronyms=regions_to_plot,
)

# %%
plot_utils.plot_population_coupling_heatmap(
    df_coupling,
    CONFIG_PLOT,
    CONFIG_CALC,
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
    coupling_strength_thr=0.1,
    region_acronyms=regions_to_plot,
)

# %%
CONFIG_PLOT.update(
    {
        'PLOT_ONLY_GOOD_UNITS': False,
        'PLOT_EVENT': 'stimOn_times'
    })

plot_utils.plot_time_window_raster(
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
    t_start=1565,
    t_end=1567,
    region_acronyms=["VISp", "ENTm"],
    df_res=df_res,
    df_coupling=df_coupling_task,
    # df_rastermap=df_rastermap,
    sort_mode="default"
    )
# dealy spont default rastermap

# %% Calculate stPR for spontaneous vs task periods (Pearson + Spearman, 3 plots only)

# -------------------------
# Filter by regions if specified
# -------------------------
if regions_to_plot_comparison is not None:
    region_mask = df_comparison['region'].isin(regions_to_plot_comparison)
    df_comparison = df_comparison[region_mask]
    print(f"Filtered to {len(df_comparison)} neurons in regions: {regions_to_plot_comparison}")
    unique_regions = regions_to_plot_comparison
else:
    unique_regions = df_comparison['region'].unique()
    print(f"Using all regions: {list(unique_regions)}")

# -------------------------
# Filter for valid data
# -------------------------
valid_strength_mask = (
    df_comparison['coupling_strength_spont'].notna() &
    df_comparison['coupling_strength_task'].notna()
)
df_strength = df_comparison[valid_strength_mask]
print(f"\nNeurons with valid coupling strength: {len(df_strength)}")

valid_delay_mask = (
    df_comparison['coupling_delay_ms_spont'].notna() &
    df_comparison['coupling_delay_ms_task'].notna()
)
df_delay = df_comparison[valid_delay_mask]
print(f"Neurons with valid coupling delay: {len(df_delay)}")

valid_sorting_mask = (
    df_comparison['sorting_number_spont'].notna() &
    df_comparison['sorting_number_task'].notna()
)
df_sorting = df_comparison[valid_sorting_mask]
print(f"Neurons with valid sorting number: {len(df_sorting)}")


def compute_corrs(x, y):
    """Return Pearson r and Spearman rho (NaN-safe)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, int(mask.sum())
    rp = pearsonr(x[mask], y[mask]).statistic
    rs = spearmanr(x[mask], y[mask]).statistic
    return rp, rs, int(mask.sum())


# -------------------------
# Compute correlations
# -------------------------
correlation_strength_p, correlation_strength_s, n_strength = (np.nan, np.nan, 0)
correlation_delay_p, correlation_delay_s, n_delay = (np.nan, np.nan, 0)
correlation_sorting_p, correlation_sorting_s, n_sorting = (np.nan, np.nan, 0)

if len(df_strength) > 1:
    correlation_strength_p, correlation_strength_s, n_strength = compute_corrs(
        df_strength['coupling_strength_spont'],
        df_strength['coupling_strength_task']
    )

if len(df_delay) > 1:
    correlation_delay_p, correlation_delay_s, n_delay = compute_corrs(
        df_delay['coupling_delay_ms_spont'],
        df_delay['coupling_delay_ms_task']
    )

if len(df_sorting) > 1:
    correlation_sorting_p, correlation_sorting_s, n_sorting = compute_corrs(
        df_sorting['sorting_number_spont'],
        df_sorting['sorting_number_task']
    )


# -------------------------
# Color map for regions
# -------------------------
region_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
region_to_color = dict(zip(unique_regions, region_colors))


def scatter_with_unity(ax, df, xcol, ycol, xlabel, ylabel, title, region_order):
    """Scatter colored by region + unity line."""
    for region in region_order:
        m = df['region'] == region
        if m.sum() > 0:
            ax.scatter(
                df.loc[m, xcol],
                df.loc[m, ycol],
                c=[region_to_color[region]],
                alpha=0.6,
                s=50,
                edgecolors='k',
                linewidths=0.5,
                label=region
            )

    min_val = min(df[xcol].min(), df[ycol].min())
    max_val = max(df[xcol].max(), df[ycol].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', alpha=0.5, linewidth=2, label='Unity')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal', adjustable='box')


# -------------------------
# Plot 1: Coupling Strength
# -------------------------
if len(df_strength) > 0:
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    fig_path_strength = path_fig / f"{pid}_stPR_strength_spontaneous_vs_task.png"

    title = (
        "Coupling Strength: Spontaneous vs Task\n"
        f"(Pearson r = {correlation_strength_p:.3f}, "
        f"Spearman ρ = {correlation_strength_s:.3f}, "
        f"n = {n_strength})"
    )

    scatter_with_unity(
        ax1, df_strength,
        'coupling_strength_task',
        'coupling_strength_spont',
        'Task stPR (coupling strength)',
        'Spontaneous stPR (coupling strength)',
        title,
        unique_regions
    )

    plt.tight_layout()
    plt.savefig(fig_path_strength, dpi=300, bbox_inches='tight')
    print(f"\nSaved coupling strength figure to: {fig_path_strength}")
    plt.show()


# -------------------------
# Plot 2: Coupling Delay
# -------------------------
if len(df_delay) > 0:
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig_path_delay = path_fig / f"{pid}_stPR_delay_spontaneous_vs_task.png"

    title = (
        "Coupling Delay: Spontaneous vs Task\n"
        f"(Pearson r = {correlation_delay_p:.3f}, "
        f"Spearman ρ = {correlation_delay_s:.3f}, "
        f"n = {n_delay})"
    )

    scatter_with_unity(
        ax2, df_delay,
        'coupling_delay_ms_task',
        'coupling_delay_ms_spont',
        'Task Coupling Delay (ms)',
        'Spontaneous Coupling Delay (ms)',
        title,
        unique_regions
    )

    plt.tight_layout()
    plt.savefig(fig_path_delay, dpi=300, bbox_inches='tight')
    print(f"Saved coupling delay figure to: {fig_path_delay}")
    plt.show()


# -------------------------
# Plot 3: Sorting Number
# -------------------------
if len(df_sorting) > 0:
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    fig_path_sorting = path_fig / f"{pid}_stPR_sorting_spontaneous_vs_task.png"

    title = (
        "Sorting Number: Spontaneous vs Task\n"
        f"(Pearson r = {correlation_sorting_p:.3f}, "
        f"Spearman ρ = {correlation_sorting_s:.3f}, "
        f"n = {n_sorting})"
    )

    scatter_with_unity(
        ax3, df_sorting,
        'sorting_number_task',
        'sorting_number_spont',
        'Task Sorting Number',
        'Spontaneous Sorting Number',
        title,
        unique_regions
    )

    plt.tight_layout()
    plt.savefig(fig_path_sorting, dpi=300, bbox_inches='tight')
    print(f"Saved sorting number figure to: {fig_path_sorting}")
    plt.show()
