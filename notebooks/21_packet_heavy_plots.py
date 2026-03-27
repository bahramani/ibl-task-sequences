# %% Heavy Plot Config
from pathlib import Path
import sys

import numpy as np
import plotly.io as pio

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))

from utils.packet_dashboard import (
    PACKET_CACHE_DIR,
    build_embedding_figure,
    build_packet_scatter,
    build_single_neuron_scatter_matrix,
    choose_default_neuron_id,
    compute_umap_embedding,
    get_packet_scatter_options,
    list_available_pids,
    load_packet_cache,
)


PID = None
REGION = None
SHAPE_MODE = None
PROMPT_FOR_REGION_AND_SHAPE = True
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


available_pids = list_available_pids(PACKET_CACHE_DIR)
if not available_pids:
    raise RuntimeError(f"No packet caches found in `{PACKET_CACHE_DIR}`. Run `19_packet_comp.py` first.")
if PID is None:
    PID = available_pids[0]

packet_cache = load_packet_cache(PID, PACKET_CACHE_DIR)
available_regions = sorted(packet_cache.get("region_results", {}).keys())
if not available_regions:
    raise RuntimeError(f"No packet regions are available in cache for PID {PID}.")

shape_modes_available = list(packet_cache.get("packet_config", {}).get("SHAPE_MODES", ["raw", "normalized", "residual"]))
pio.templates.default = PLOT_TEMPLATE

print(f"PID: {PID}")
print(f"Available regions: {available_regions}")
print(f"Available shape modes: {shape_modes_available}")

if PROMPT_FOR_REGION_AND_SHAPE:
    REGION = _choose_option(available_regions, "region", default_value=REGION)
    SHAPE_MODE = _choose_option(shape_modes_available, "shape mode", default_value=SHAPE_MODE)
else:
    if REGION is None:
        REGION = available_regions[0]
    if SHAPE_MODE is None:
        SHAPE_MODE = shape_modes_available[0]

if SHAPE_MODE not in shape_modes_available:
    raise ValueError(f"Invalid SHAPE_MODE `{SHAPE_MODE}`. Options: {shape_modes_available}")
if REGION not in packet_cache.get("region_results", {}):
    raise ValueError(f"Invalid REGION `{REGION}`. Options: {available_regions}")

region_bundle = packet_cache["region_results"][REGION]
shape_result = region_bundle["shape_results"].get(SHAPE_MODE)
if shape_result is None:
    raise RuntimeError(f"No clustering result available for region `{REGION}` in shape mode `{SHAPE_MODE}`.")

print(f"Chosen region: {REGION}")
print(f"Chosen shape mode: {SHAPE_MODE}")
print(f"Chosen number of clusters: {shape_result.get('n_clusters')}")


# %% Packet Embeddings (PCA)
# Options:
# PCA_COLOR_BY: "cluster", "context", or "period"
# PCA_SHOW_ALL: True opens one figure for each of the three colorings above
PCA_COLOR_BY = "cluster"
PCA_SHOW_ALL = True

pca_scores = np.asarray(shape_result["cluster_pca"]["scores"], dtype=float)
explained = np.asarray(shape_result["cluster_pca"]["explained_ratio"], dtype=float)
pc1 = pca_scores[:, 0] if pca_scores.shape[1] >= 1 else np.zeros(pca_scores.shape[0], dtype=float)
pc2 = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else np.zeros(pca_scores.shape[0], dtype=float)
pc1_label = f"PC1 ({100.0 * explained[0]:.1f}% var)" if explained.size >= 1 else "PC1"
pc2_label = f"PC2 ({100.0 * explained[1]:.1f}% var)" if explained.size >= 2 else "PC2"

color_modes = ["cluster", "context", "period"] if PCA_SHOW_ALL else [PCA_COLOR_BY]
for color_by in color_modes:
    fig_pca = build_embedding_figure(
        shape_result["packet_plot_df"],
        pc1,
        pc2,
        color_by=color_by,
        title=f"Region {REGION} | {SHAPE_MODE} | PCA by {color_by}",
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
#   "t_com", "relative_rank_order", "temporal_width"
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
    SHAPE_MODE,
    PACKET_SCATTER_X,
    PACKET_SCATTER_Y,
    color_by=PACKET_SCATTER_COLOR_BY,
    template=PLOT_TEMPLATE,
)
_show_browser(fig_packet_scatter)


# %% Single-Neuron Scatter Matrix
# Options:
# SINGLE_NEURON_CLUSTER_ID:
#   None -> use the neuron with the highest combined variance across the 6 main packet features
#   int  -> a specific neuron cluster_id from region_bundle["neuron_summary_df"]["cluster_id"]
SINGLE_NEURON_CLUSTER_ID = None

if SINGLE_NEURON_CLUSTER_ID is None:
    SINGLE_NEURON_CLUSTER_ID = choose_default_neuron_id(region_bundle)

fig_single_neuron, resolved_neuron_id = build_single_neuron_scatter_matrix(
    region_bundle,
    SHAPE_MODE,
    cluster_id=SINGLE_NEURON_CLUSTER_ID,
    template=PLOT_TEMPLATE,
)
print(f"Single-neuron scatter matrix cluster_id: {resolved_neuron_id}")
_show_browser(fig_single_neuron)


# %% Packet UMAP
# Options:
# UMAP_COLOR_BY: "cluster", "context", or "period"
UMAP_COLOR_BY = "cluster"

umap_scores = compute_umap_embedding(
    np.asarray(shape_result["cluster_input"]["cluster_matrix"], dtype=float),
    pre_pca_components=packet_cache["packet_config"].get("UMAP_PRE_PCA_COMPONENTS", 10),
    n_neighbors=packet_cache["packet_config"].get("UMAP_NEIGHBORS", 20),
    min_dist=packet_cache["packet_config"].get("UMAP_MIN_DIST", 0.2),
    random_state=packet_cache["packet_config"].get("UMAP_RANDOM_STATE", 0),
)
if umap_scores is None:
    print("UMAP is not available in this environment.")
else:
    fig_umap = build_embedding_figure(
        shape_result["packet_plot_df"],
        umap_scores[:, 0],
        umap_scores[:, 1],
        color_by=UMAP_COLOR_BY,
        title=f"Region {REGION} | {SHAPE_MODE} | UMAP by {UMAP_COLOR_BY}",
        x_label="UMAP1",
        y_label="UMAP2",
        template=PLOT_TEMPLATE,
    )
    _show_browser(fig_umap)
