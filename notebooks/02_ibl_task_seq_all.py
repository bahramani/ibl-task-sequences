# %% Imports
import sys
import os
from pathlib import Path

# --- ROBUST PATH SETUP ---
# 1. Get the folder containing THIS file (notebooks/)
# This handles both "Run as Script" (__file__) and "Run as Notebook" cases
# try:
#     script_dir = Path(__file__).resolve().parent
# except NameError:
#     script_dir = Path.cwd()
#
# # 2. Go up one level to the project root (SeqProject2026/)
# project_root = script_dir.parent
#
# # 3. Add to sys.path if not already there
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))
#     print(f"Project root added to path: {project_root}")
#
# # Double check that 'utils' is actually found there
# if not (project_root / "utils").exists():
#     print("WARNING: 'utils' folder not found at the calculated path!")
# -------------------------

# Enable autoreload
# try:
#     from IPython import get_ipython
#     if get_ipython() is not None:
#         %load_ext autoreload
#         %autoreload 2
# except ImportError:
#     pass

import numpy as np
from iblatlas.atlas import AllenAtlas

# Now these will work
import utils.io as io_utils
import utils.analysis as ana_utils
import utils.plotting as plot_utils


# %% Parameters ##############################################################################

CONFIG_CALC = {
    # Which atlas to use when assigning brain regions ("Beryl" or "Allen")
    "ATLAS_MAPPING": "Beryl",
    # Run calculations only on good units (label == 1) or on all units
    "CALC_ONLY_GOOD_UNITS": True,
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
    "PLOT_EVENT": "response_times",
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
path_data, path_fig, path_data_processed, ibl_cache = io_utils.setup_paths(base_path)
print(f"Directories ready. Cache: {ibl_cache}")

one = io_utils.init_one(ibl_cache)


ba, br, beryl_acronyms, hier_scores = io_utils.prepare_region_dirs(path_data)


# Great for CP and MOp: 
# pid = '26118c10-35dd-4ab1-9f0f-b9a89a1da070'

# Main one for VISp and EnTm:
pid = "c9664185-d3fd-4e0e-89cf-77c402038938"

# The one that is not working well:
# pid = # '3d3d5a5e-df26-43ee-80b6-2d72d85668a5' 
print(f"\nProcessing PID: {pid}")

ssl, spikes, clusters, sl = io_utils.load_session_data(pid, one, ba)
pupil_features, pupil_times = io_utils.load_pupil_data(sl)

# Resolve cluster IDs for safe indexing.
cluster_ids, cid_to_idx = io_utils.build_cluster_id_map(clusters)

# Map acronyms once for calculations and for plots (can be different atlas choices)
cluster_acronyms_calc = io_utils.map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = io_utils.map_acronyms(clusters, br, CONFIG_PLOT["ATLAS_MAPPING"])

# Build event-aligned arrays for each requested event.
events_by_name, contrasts_by_name = ana_utils.build_event_dicts(
    sl, CONFIG_CALC["EVENT_NAMES"], CONFIG_CALC["MIN_TRIALS"]
)

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

# %% Select Trial and Unit to Plot ###########################################################

trial_idx = 79
single_neuron_id = 559

# %% Plot Single Trial Raster ##############################################################################

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
    })

regions_to_plot = CONFIG_PLOT["PLOT_REGIONS"]

plot_utils.plot_sequence_raster(
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
    region_acronyms=regions_to_plot,
)

plot_utils.plot_population_PSTH_sorted(
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


# %% Cross-Correlogram Function ###########################################################################

def plot_cross_correlogram(
    spikes,
    cluster_id_1,
    cluster_id_2,
    clusters,
    cluster_acronyms,
    cid_to_idx,
    df_res=None,
    bin_size=0.0001,
    window_size=0.007,
    save_flag=False,
    path_fig=None,
    pid=None,
):
    """
    Plot cross-correlogram between two neurons.

    Parameters
    ----------
    spikes : object
        Spikes object with .times and .clusters attributes
    cluster_id_1 : int
        First cluster ID (reference neuron)
    cluster_id_2 : int
        Second cluster ID (target neuron)
    clusters : object
        Clusters object
    cluster_acronyms : array
        Array of brain region acronyms for each cluster
    cid_to_idx : dict
        Mapping from cluster ID to index
    df_res : DataFrame, optional
        Results dataframe containing delay information
    bin_size : float
        Bin size in seconds (default: 0.1 ms)
    window_size : float
        Half-window size in seconds (default: 7 ms)
    save_flag : bool
        Whether to save the figure
    path_fig : Path, optional
        Path to save figure
    pid : str, optional
        Session PID for filename
    """
    import matplotlib.pyplot as plt

    # Get spike times for each cluster
    spike_times_1 = spikes.times[spikes.clusters == cluster_id_1]
    spike_times_2 = spikes.times[spikes.clusters == cluster_id_2]

    # Get region information
    idx_1 = cid_to_idx.get(cluster_id_1)
    idx_2 = cid_to_idx.get(cluster_id_2)

    if idx_1 is None or idx_2 is None:
        print(f"Error: One or both cluster IDs not found.")
        return

    region_1 = cluster_acronyms[idx_1]
    region_2 = cluster_acronyms[idx_2]

    # Get delay information if available
    delay_1, delay_2 = np.nan, np.nan
    if df_res is not None:
        match_1 = df_res[df_res['cluster_id'] == cluster_id_1]
        match_2 = df_res[df_res['cluster_id'] == cluster_id_2]
        if not match_1.empty and 'delay_response_times' in df_res.columns:
            delay_1 = match_1.iloc[0]['delay_response_times']
        if not match_2.empty and 'delay_response_times' in df_res.columns:
            delay_2 = match_2.iloc[0]['delay_response_times']

    # Compute cross-correlogram
    # For each spike in neuron 1, find time differences to all spikes in neuron 2
    ccg_values = []

    for spike_t1 in spike_times_1:
        # Find spikes in neuron 2 within the window
        time_diffs = spike_times_2 - spike_t1
        # Keep only those within window
        valid_diffs = time_diffs[(time_diffs >= -window_size) & (time_diffs <= window_size)]
        ccg_values.extend(valid_diffs)

    ccg_values = np.array(ccg_values)

    # Create bins
    bins = np.arange(-window_size, window_size + bin_size, bin_size)

    # Compute histogram
    counts, bin_edges = np.histogram(ccg_values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert to ms for plotting
    bin_centers_ms = bin_centers * 1000

    # Normalize by number of reference spikes and bin size to get rate
    if len(spike_times_1) > 0:
        counts_normalized = counts / (len(spike_times_1) * bin_size)
    else:
        counts_normalized = counts

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(bin_centers_ms, counts_normalized, width=bin_size*1000,
           color='black', edgecolor='black', alpha=0.7)

    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero lag')

    ax.set_xlabel('Time lag (ms)', fontsize=12)
    ax.set_ylabel('Coincidence rate (Hz)', fontsize=12)

    # Build title with delay information
    delay_str_1 = f"{delay_1*1000:.2f} ms" if not np.isnan(delay_1) else "N/A"
    delay_str_2 = f"{delay_2*1000:.2f} ms" if not np.isnan(delay_2) else "N/A"

    title = f'Cross-Correlogram\n'
    title += f'Cluster {cluster_id_1} ({region_1}, delay: {delay_str_1}) → '
    title += f'Cluster {cluster_id_2} ({region_2}, delay: {delay_str_2})'
    ax.set_title(title, fontsize=14)

    ax.set_xlim(-window_size*1000, window_size*1000)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()

    if save_flag and path_fig is not None and pid is not None:
        filename = f"{pid}_ccg_{cluster_id_1}_{cluster_id_2}.png"
        save_path = path_fig / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-correlogram saved to: {save_path}")

    plt.show()

    # Print statistics
    print(f"\nCross-Correlogram Statistics:")
    print(f"  Cluster 1: {cluster_id_1} ({region_1})")
    print(f"  Cluster 2: {cluster_id_2} ({region_2})")
    print(f"  Total spike pairs: {len(ccg_values)}")
    print(f"  Mean coincidence rate: {np.mean(counts_normalized):.2f} Hz")
    print(f"  Peak coincidence rate: {np.max(counts_normalized):.2f} Hz")
    if len(counts_normalized) > 0:
        peak_idx = np.argmax(counts_normalized)
        print(f"  Peak at: {bin_centers_ms[peak_idx]:.2f} ms")


# %% Plot Cross-Correlogram Example ###########################################################################

cluster_1 = 813
cluster_2 = 694

plot_cross_correlogram(
    spikes,
    cluster_1,
    cluster_2,
    clusters,
    cluster_acronyms_plot,
    cid_to_idx,
    df_res=df_res,
    bin_size=0.0001,  # 0.1 ms bins
    window_size=0.007,  # ±7 ms window
    save_flag=True,
    path_fig=path_fig,
    pid=pid,
)

