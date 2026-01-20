# %% Imports
from pathlib import Path


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


