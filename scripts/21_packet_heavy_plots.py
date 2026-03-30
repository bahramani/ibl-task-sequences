# %% Heavy Plot Config
from pathlib import Path
import sys

import numpy as np
import plotly.io as pio

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))

from utils.packet_dashboard import (
    DEFAULT_PACKET_CONFIG,
    PACKET_CACHE_DIR,
    build_embedding_figure,
    build_packet_scatter,
    build_single_neuron_scatter_matrix,
    choose_default_neuron_id,
    get_packet_scatter_options,
    list_available_pids,
    load_packet_cache,
)

DEFAULT_CLUSTER_METHODS = list(DEFAULT_PACKET_CONFIG["CLUSTER_METHODS"])

PID = "f967a527-257f-404a-871d-b91575dca3b4"
REGION = None
CLUSTER_METHOD = None
PROMPT_FOR_REGION_AND_CLUSTER = True
PLOT_TEMPLATE = "plotly_white"
PLOT_RENDERER = "browser"


def _show_browser(fig):
    fig.update_layout(template=PLOT_TEMPLATE)
    fig.show(renderer=PLOT_RENDERER)


def _choose_option(options, label, default_value=None):
    if not options:
        raise ValueError(f"No options available for {label}.")
    default_value = default_value if default_value in options else options[0]
    default_index = options.index(default_value)
    print(f"Available {label}:")
    for idx, value in enumerate(options):
        print(f"  [{idx}] {value}")
    selection = input(f"Choose {label} by index or name (press Enter for {default_value}): ").strip()
    if selection == "":
        return default_value
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(options):
            return options[idx]
        raise ValueError(f"{label} index {idx} out of range.")
    if selection in options:
        return selection
    raise ValueError(f"{label} `{selection}` not found.")


def _ordered_cluster_methods(region_bundle, configured_methods):
    cluster_results = region_bundle.get("cluster_results", {})
    methods = [method for method in configured_methods if cluster_results.get(method) is not None]
    methods.extend(
        method
        for method, result in cluster_results.items()
        if result is not None and method not in methods
    )
    return methods


available_pids = list_available_pids(PACKET_CACHE_DIR)
if not available_pids:
    raise RuntimeError(f"No packet caches found in `{PACKET_CACHE_DIR}`. Run `19_packet_comp.py` first.")
if PID is None:
    PID = available_pids[0]

packet_cache = load_packet_cache(PID, PACKET_CACHE_DIR)
available_regions = sorted(packet_cache.get("region_results", {}).keys())
if not available_regions:
    raise RuntimeError(f"No packet regions are available in cache for PID {PID}.")

cluster_methods_config = list(packet_cache.get("packet_config", {}).get("CLUSTER_METHODS", DEFAULT_CLUSTER_METHODS))
target_region_from_cache = packet_cache.get("packet_config", {}).get("TARGET_REGION")
pio.templates.default = PLOT_TEMPLATE

print(f"PID: {PID}")
print(f"Available regions: {available_regions}")

default_region = REGION if REGION in available_regions else None
if default_region is None and target_region_from_cache in available_regions:
    default_region = target_region_from_cache
if default_region is None:
    default_region = available_regions[0]

if PROMPT_FOR_REGION_AND_CLUSTER:
    REGION = _choose_option(available_regions, "region", default_value=default_region)
else:
    REGION = default_region
if REGION not in packet_cache.get("region_results", {}):
    raise ValueError(f"Invalid REGION `{REGION}`. Options: {available_regions}")

region_bundle = packet_cache["region_results"][REGION]
cluster_methods_available = _ordered_cluster_methods(region_bundle, cluster_methods_config)
if not cluster_methods_available:
    raise RuntimeError(f"No clustering results are available for region `{REGION}`.")
print(f"Available clustering methods for {REGION}: {cluster_methods_available}")

default_cluster_method = CLUSTER_METHOD if CLUSTER_METHOD in cluster_methods_available else cluster_methods_available[0]
if PROMPT_FOR_REGION_AND_CLUSTER:
    CLUSTER_METHOD = _choose_option(
        cluster_methods_available,
        "clustering method",
        default_value=default_cluster_method,
    )
else:
    CLUSTER_METHOD = default_cluster_method

if CLUSTER_METHOD not in cluster_methods_available:
    raise ValueError(f"Invalid CLUSTER_METHOD `{CLUSTER_METHOD}`. Options: {cluster_methods_available}")

cluster_result = region_bundle.get("cluster_results", {}).get(CLUSTER_METHOD)
if cluster_result is None:
    raise RuntimeError(f"No clustering result available for region `{REGION}` with method `{CLUSTER_METHOD}`.")

print(f"Chosen region: {REGION}")
print(f"Chosen clustering method: {CLUSTER_METHOD}")
print(f"Chosen number of clusters: {cluster_result.get('n_clusters')}")
print(
    "Cluster count source: "
    f"{cluster_result.get('cluster_count_source', 'unknown')}"
    f" (suggested_k={cluster_result.get('suggested_k', 'n/a')})"
)


# %% Packet Embedding
# Options:
# PCA_COLOR_BY: "cluster", "context", or "period"
# PCA_SHOW_ALL: True opens one figure for each of the three colorings above
# Uses the PCA embedding stored with the selected clustering method.
PCA_COLOR_BY = "cluster"
PCA_SHOW_ALL = True

feature_pca = cluster_result.get("embedding")
if feature_pca is None:
    print("PCA embedding is not available in this cache.")
else:
    pca_scores = np.asarray(feature_pca["scores"], dtype=float)
    explained = np.asarray(feature_pca["explained_ratio"], dtype=float)
    pc1 = pca_scores[:, 0] if pca_scores.shape[1] >= 1 else np.zeros(pca_scores.shape[0], dtype=float)
    pc2 = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else np.zeros(pca_scores.shape[0], dtype=float)
    pc1_label = f"PC1 ({100.0 * explained[0]:.1f}% var)" if explained.size >= 1 else "PC1"
    pc2_label = f"PC2 ({100.0 * explained[1]:.1f}% var)" if explained.size >= 2 else "PC2"

    color_modes = ["cluster", "context", "period"] if PCA_SHOW_ALL else [PCA_COLOR_BY]
    for color_by in color_modes:
        fig_pca = build_embedding_figure(
            cluster_result["packet_plot_df"],
            pc1,
            pc2,
            color_by=color_by,
            title=f"Region {REGION} | {CLUSTER_METHOD} | Feature-space PCA by {color_by}",
            x_label=pc1_label,
            y_label=pc2_label,
            template=PLOT_TEMPLATE,
        )
        _show_browser(fig_pca)


# %% Packet Scatter
# Options:
# PACKET_SCATTER_X / PACKET_SCATTER_Y:
#   "pc1", "pc2", "packet_score", "packet_size",
#   "packet_fraction", "peak_rate", "template_dot",
#   "t_com", "relative_rank_order", "temporal_width",
#   "other_unit_mean_activity_at_t_com", "packet_rate_over_recording_rate"
# PACKET_SCATTER_COLOR_BY: "cluster", "context", or "period"
PACKET_SCATTER_X = None
PACKET_SCATTER_Y = None
PROMPT_FOR_PACKET_SCATTER_AXES = True
PACKET_SCATTER_COLOR_BY = "cluster"

packet_scatter_options = get_packet_scatter_options()
default_packet_x = PACKET_SCATTER_X if PACKET_SCATTER_X in packet_scatter_options else packet_scatter_options[0]
default_packet_y = PACKET_SCATTER_Y if PACKET_SCATTER_Y in packet_scatter_options else (
    packet_scatter_options[1] if len(packet_scatter_options) > 1 else packet_scatter_options[0]
)
if PROMPT_FOR_PACKET_SCATTER_AXES:
    PACKET_SCATTER_X = _choose_option(packet_scatter_options, "packet scatter X", default_value=default_packet_x)
    PACKET_SCATTER_Y = _choose_option(packet_scatter_options, "packet scatter Y", default_value=default_packet_y)
else:
    PACKET_SCATTER_X = default_packet_x
    PACKET_SCATTER_Y = default_packet_y

fig_packet_scatter = build_packet_scatter(
    region_bundle,
    CLUSTER_METHOD,
    PACKET_SCATTER_X,
    PACKET_SCATTER_Y,
    color_by=PACKET_SCATTER_COLOR_BY,
    template=PLOT_TEMPLATE,
)
_show_browser(fig_packet_scatter)


# %% Single-Neuron Scatter Matrix
# Options:
# SINGLE_NEURON_CLUSTER_ID:
#   None -> use the neuron with the highest combined variance across the 8 main packet features
#   int  -> a specific neuron cluster_id from region_bundle["neuron_summary_df"]["cluster_id"]
SINGLE_NEURON_CLUSTER_ID = None

if SINGLE_NEURON_CLUSTER_ID is None:
    SINGLE_NEURON_CLUSTER_ID = choose_default_neuron_id(region_bundle)

fig_single_neuron, resolved_neuron_id = build_single_neuron_scatter_matrix(
    region_bundle,
    CLUSTER_METHOD,
    cluster_id=SINGLE_NEURON_CLUSTER_ID,
    template=PLOT_TEMPLATE,
)
print(f"Single-neuron scatter matrix cluster_id: {resolved_neuron_id}")
_show_browser(fig_single_neuron)


# %% Packet Scatter: Cached Extra Features
# Plot packet-level summaries of:
#   x = mean across active neurons of "other_unit_mean_activity_at_t_com"
#   y = mean across neurons of "packet_rate_over_recording_rate"
# Colors:
#   "cluster" and "context"
EXTRA_PACKET_SCATTER_COLOR_MODES = ["cluster", "context"]

feature_df_local = region_bundle["feature_df"].copy()
packet_plot_df_extra = cluster_result["packet_plot_df"].copy()

if feature_df_local.empty:
    print("Extra packet scatter: feature table is empty.")
else:
    packet_idx_arr = np.asarray(feature_df_local["packet_idx"], dtype=int)
    other_unit_activity_arr = np.asarray(feature_df_local["other_unit_mean_activity_at_t_com"], dtype=float)
    packet_rate_ratio_arr = np.asarray(feature_df_local["packet_rate_over_recording_rate"], dtype=float)
    packet_idx_unique = np.unique(packet_idx_arr)
    packet_x_lookup = {}
    packet_y_lookup = {}
    for pkt_idx in packet_idx_unique:
        pkt_mask = packet_idx_arr == int(pkt_idx)

        x_vals_pkt = other_unit_activity_arr[pkt_mask]
        valid_x = np.isfinite(x_vals_pkt)
        packet_x_lookup[int(pkt_idx)] = float(np.mean(x_vals_pkt[valid_x])) if np.any(valid_x) else np.nan

        y_vals_pkt = packet_rate_ratio_arr[pkt_mask]
        valid_y = np.isfinite(y_vals_pkt)
        packet_y_lookup[int(pkt_idx)] = float(np.mean(y_vals_pkt[valid_y])) if np.any(valid_y) else np.nan

    packet_idx_plot = np.asarray(packet_plot_df_extra["packet_idx"], dtype=int)
    packet_x = np.array([packet_x_lookup.get(int(pkt_idx), np.nan) for pkt_idx in packet_idx_plot], dtype=float)
    packet_y = np.array([packet_y_lookup.get(int(pkt_idx), np.nan) for pkt_idx in packet_idx_plot], dtype=float)

    n_valid_points = int(np.sum(np.isfinite(packet_x) & np.isfinite(packet_y)))
    print(f"Extra packet scatter valid packets: {n_valid_points}/{len(packet_idx_plot)}")

    for color_by in EXTRA_PACKET_SCATTER_COLOR_MODES:
        fig_extra_packet_scatter = build_embedding_figure(
            packet_plot_df_extra,
            packet_x,
            packet_y,
            color_by=color_by,
            title=f"Region {REGION} | {CLUSTER_METHOD} | Extra packet scatter by {color_by}",
            x_label="Mean other-unit activity at own COM across active neurons (Hz)",
            y_label="Mean packet rate / recording rate across neurons (fold)",
            template=PLOT_TEMPLATE,
        )
        _show_browser(fig_extra_packet_scatter)
