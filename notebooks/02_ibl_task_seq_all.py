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


# Great for CP and MOp:
# pid = '26118c10-35dd-4ab1-9f0f-b9a89a1da070'

# Main one for VISp and EnTm:
pid = "c9664185-d3fd-4e0e-89cf-77c402038938"

# The one that is not working well:
# pid = # '3d3d5a5e-df26-43ee-80b6-2d72d85668a5'
print(f"\nProcessing PID: {pid}")

ssl, spikes, clusters, sl = load_session_data(pid, one, ba)
pupil_features, pupil_times = load_pupil_data(sl)

# Resolve cluster IDs for safe indexing.
cluster_ids, cid_to_idx = build_cluster_id_map(clusters)

# Map acronyms once for calculations and for plots (can be different atlas choices)
cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

# Build event-aligned arrays for each requested event.
events_by_name, contrasts_by_name = ana_utils.build_event_dicts(
    sl, CONFIG_CALC["EVENT_NAMES"], CONFIG_CALC["MIN_TRIALS"]
)

eid = one.pid2eid(pid)[0]

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
    regions_to_plot_comparison = ['ENTm']  # Or specify like: ['VISp', 'ENTm']

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
trial_idx = 125
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
        'PLOT_ONLY_GOOD_UNITS': True,
        'SORT_BY_SPONT': True,
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
    df_rastermap=df_rastermap,
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
    coupling_strength_thr=0.05,
    region_acronyms=regions_to_plot,
)

# %%
CONFIG_PLOT.update(
    {
        'PLOT_ONLY_GOOD_UNITS': True,
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
    t_start=180,
    t_end=182,
    region_acronyms=["VISp", "ENTm"],
    df_res=df_res,
    df_coupling=df_coupling,
    # df_rastermap=df_rastermap,
    sort_mode="spont"
    )
# dealy spont default rastermap

# %% Calculate stPR for spontaneous vs task periods

# Filter by regions if specified
if regions_to_plot_comparison is not None:
    region_mask = df_comparison['region'].isin(regions_to_plot_comparison)
    df_comparison = df_comparison[region_mask]
    print(f"Filtered to {len(df_comparison)} neurons in regions: {regions_to_plot_comparison}")
    unique_regions = regions_to_plot_comparison
else:
    unique_regions = df_comparison['region'].unique()
    print(f"Using all regions: {list(unique_regions)}")

# Filter for valid coupling strength data
valid_strength_mask = (
    df_comparison['coupling_strength_spont'].notna() &
    df_comparison['coupling_strength_task'].notna()
)
df_strength = df_comparison[valid_strength_mask]
print(f"\nNeurons with valid coupling strength: {len(df_strength)}")

# Filter for valid coupling delay data
valid_delay_mask = (
    df_comparison['coupling_delay_ms_spont'].notna() &
    df_comparison['coupling_delay_ms_task'].notna()
)
df_delay = df_comparison[valid_delay_mask]
print(f"Neurons with valid coupling delay: {len(df_delay)}")

# Filter for valid sorting number data
valid_sorting_mask = (
    df_comparison['sorting_number_spont'].notna() &
    df_comparison['sorting_number_task'].notna()
)
df_sorting = df_comparison[valid_sorting_mask]
print(f"Neurons with valid sorting number: {len(df_sorting)}")

# Calculate correlations
if len(df_strength) > 0:
    correlation_strength = np.corrcoef(
        df_strength['coupling_strength_spont'],
        df_strength['coupling_strength_task']
    )[0, 1]
    print(f"\nCorrelation - Coupling strength: {correlation_strength:.3f}")

if len(df_delay) > 0:
    correlation_delay = np.corrcoef(
        df_delay['coupling_delay_ms_spont'],
        df_delay['coupling_delay_ms_task']
    )[0, 1]
    print(f"Correlation - Coupling delay: {correlation_delay:.3f}")

if len(df_sorting) > 0:
    correlation_sorting = np.corrcoef(
        df_sorting['sorting_number_spont'],
        df_sorting['sorting_number_task']
    )[0, 1]
    print(f"Correlation - Sorting number: {correlation_sorting:.3f}")

# Create color map for regions
region_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
region_to_color = dict(zip(unique_regions, region_colors))

# Plot 1: Coupling Strength
if len(df_strength) > 0:
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    for region in unique_regions:
        region_mask = df_strength['region'] == region
        if region_mask.sum() > 0:
            ax1.scatter(
                df_strength.loc[region_mask, 'coupling_strength_task'],
                df_strength.loc[region_mask, 'coupling_strength_spont'],
                c=[region_to_color[region]],
                alpha=0.6,
                s=50,
                edgecolors='k',
                linewidths=0.5,
                label=region
            )

    # Add unity line
    min_val = min(df_strength['coupling_strength_task'].min(),
                 df_strength['coupling_strength_spont'].min())
    max_val = max(df_strength['coupling_strength_task'].max(),
                 df_strength['coupling_strength_spont'].max())
    ax1.plot([min_val, max_val], [min_val, max_val],
            'r--', alpha=0.5, linewidth=2, label='Unity')

    ax1.set_xlabel('Task stPR (coupling strength)', fontsize=12)
    ax1.set_ylabel('Spontaneous stPR (coupling strength)', fontsize=12)
    ax1.set_title(f'Coupling Strength: Spontaneous vs Task\n(r = {correlation_strength:.3f}, n = {len(df_strength)})',
                 fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    ax1.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    fig_path_strength = path_fig / f"{pid}_stPR_strength_spontaneous_vs_task.png"
    plt.savefig(fig_path_strength, dpi=300, bbox_inches='tight')
    print(f"\nSaved coupling strength figure to: {fig_path_strength}")
    plt.show()
else:
    print("\nNo valid coupling strength data for plotting.")

# Plot 2: Coupling Delay
if len(df_delay) > 0:
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    for region in unique_regions:
        region_mask = df_delay['region'] == region
        if region_mask.sum() > 0:
            ax2.scatter(
                df_delay.loc[region_mask, 'coupling_delay_ms_task'],
                df_delay.loc[region_mask, 'coupling_delay_ms_spont'],
                c=[region_to_color[region]],
                alpha=0.6,
                s=50,
                edgecolors='k',
                linewidths=0.5,
                label=region
            )

    # Add unity line
    min_val = min(df_delay['coupling_delay_ms_task'].min(),
                 df_delay['coupling_delay_ms_spont'].min())
    max_val = max(df_delay['coupling_delay_ms_task'].max(),
                 df_delay['coupling_delay_ms_spont'].max())
    ax2.plot([min_val, max_val], [min_val, max_val],
            'r--', alpha=0.5, linewidth=2, label='Unity')

    ax2.set_xlabel('Task Coupling Delay (ms)', fontsize=12)
    ax2.set_ylabel('Spontaneous Coupling Delay (ms)', fontsize=12)
    ax2.set_title(f'Coupling Delay: Spontaneous vs Task\n(r = {correlation_delay:.3f}, n = {len(df_delay)})',
                 fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_ylim([-30, 30])
    ax2.set_xlim([-30, 30])
    plt.tight_layout()

    fig_path_delay = path_fig / f"{pid}_stPR_delay_spontaneous_vs_task.png"
    plt.savefig(fig_path_delay, dpi=300, bbox_inches='tight')
    print(f"Saved coupling delay figure to: {fig_path_delay}")
    plt.show()
else:
    print("No valid coupling delay data for plotting.")

# Plot 3: Sorting Number
if len(df_sorting) > 0:
    fig3, ax3 = plt.subplots(figsize=(8, 8))

    for region in unique_regions:
        region_mask = df_sorting['region'] == region
        if region_mask.sum() > 0:
            ax3.scatter(
                df_sorting.loc[region_mask, 'sorting_number_task'],
                df_sorting.loc[region_mask, 'sorting_number_spont'],
                c=[region_to_color[region]],
                alpha=0.6,
                s=50,
                edgecolors='k',
                linewidths=0.5,
                label=region
            )

    # Add unity line
    min_val = min(df_sorting['sorting_number_task'].min(),
                 df_sorting['sorting_number_spont'].min())
    max_val = max(df_sorting['sorting_number_task'].max(),
                 df_sorting['sorting_number_spont'].max())
    ax3.plot([min_val, max_val], [min_val, max_val],
            'r--', alpha=0.5, linewidth=2, label='Unity')

    ax3.set_xlabel('Task Sorting Number', fontsize=12)
    ax3.set_ylabel('Spontaneous Sorting Number', fontsize=12)
    ax3.set_title(f'Sorting Number: Spontaneous vs Task\n(r = {correlation_sorting:.3f}, n = {len(df_sorting)})',
                 fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    ax3.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    fig_path_sorting = path_fig / f"{pid}_stPR_sorting_spontaneous_vs_task.png"
    plt.savefig(fig_path_sorting, dpi=300, bbox_inches='tight')
    print(f"Saved sorting number figure to: {fig_path_sorting}")
    plt.show()
else:
    print("No valid sorting number data for plotting.")
