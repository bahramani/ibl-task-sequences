import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from scipy.stats import pearsonr, spearmanr
from types import SimpleNamespace

from .analysis import (
    compute_psth_for_clusters,
    event_label,
    delay_column_name,
    delay_split_column_name,
)

DEFAULT_TEMPLATE = "plotly_white"


def _get_cluster_attr(clusters, key, fallback=None):
    if clusters is None:
        return fallback
    if hasattr(clusters, key):
        return getattr(clusters, key)
    if isinstance(clusters, dict) and key in clusters:
        return clusters.get(key)
    return fallback


def _get_metrics_df(clusters):
    if clusters is None:
        return None
    if hasattr(clusters, "metrics"):
        return clusters.metrics
    if isinstance(clusters, dict) and "metrics" in clusters:
        return clusters.get("metrics")
    return None


def _get_metric_cluster_ids(metrics):
    if metrics is None:
        return None
    if isinstance(metrics, pd.DataFrame):
        if "cluster_id" in metrics.columns:
            return np.asarray(metrics["cluster_id"])
        try:
            return np.asarray(metrics.index)
        except Exception:
            return None
    if isinstance(metrics, dict) and "cluster_id" in metrics:
        return np.asarray(metrics["cluster_id"])
    if hasattr(metrics, "cluster_id"):
        return np.asarray(metrics.cluster_id)
    return None


def _map_labels_to_cluster_ids(labels, cluster_ids, id_all):
    if labels is None:
        return None
    labels = np.asarray(labels)
    cluster_ids = np.asarray(cluster_ids)
    if labels.shape[0] == len(cluster_ids):
        return labels.astype(float)
    if id_all is None:
        return None
    id_all = np.asarray(id_all)
    if id_all.shape[0] != labels.shape[0]:
        return None
    label_lookup = dict(zip(id_all.tolist(), labels.tolist()))
    return np.array([label_lookup.get(cid, np.nan) for cid in cluster_ids], dtype=float)


def _get_session_field(sl, key):
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl.get(key)
    return getattr(sl, key, None)


def _extract_motion_energy_series(motion_energy, camera_key, t_start, t_end, t_offset=0.0):
    if motion_energy is None or camera_key not in motion_energy:
        return np.array([]), np.array([])
    df = motion_energy.get(camera_key)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.array([]), np.array([])
    time_col = "times" if "times" in df.columns else None
    if time_col is None:
        times = df.index.values
    else:
        times = df[time_col].values
    value_col = None
    for candidate in ("whiskerMotionEnergy", "motionEnergy", "energy"):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_cols = [
            col
            for col in df.columns
            if col != time_col and pd.api.types.is_numeric_dtype(df[col])
        ]
        if numeric_cols:
            value_col = numeric_cols[0]
    if value_col is None:
        return np.array([]), np.array([])
    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        return np.array([]), np.array([])
    return times[mask] - t_offset, df[value_col].values[mask]


def _get_trials_array_field(trials, key):
    if trials is None:
        return None
    if hasattr(trials, "keys") and key in trials.keys():
        return np.asarray(trials[key])
    if hasattr(trials, key):
        return np.asarray(getattr(trials, key))
    return None


def _normalize_colorscale(cmap_name):
    if not isinstance(cmap_name, str):
        return cmap_name
    cmap = cmap_name.strip().lower()
    if cmap in ("bwr", "rdbu"):
        return "rdbu_r"
    if cmap == "rdbu_r":
        return "rdbu_r"
    if cmap == "rdgy":
        return "rdgy"
    return cmap_name


def _white_theme():
    template = DEFAULT_TEMPLATE or "plotly_white"
    base_color = "white" if "dark" in str(template).lower() else "black"
    return template, base_color


def _delay_units(config_plot):
    units = str(config_plot.get("DELAY_UNITS", "s")).lower()
    return "ms" if units.startswith("ms") else "s"


def _delay_to_seconds(values, config_plot):
    vals = np.asarray(values, dtype=float)
    if _delay_units(config_plot) == "ms":
        vals = vals / 1000.0
    return vals


def _delay_to_ms(values, config_plot):
    vals = np.asarray(values, dtype=float)
    if _delay_units(config_plot) != "ms":
        vals = vals * 1000.0
    return vals


def _color_to_rgba(color, alpha=0.15):
    if color is None:
        return f"rgba(200,200,200,{alpha})"
    if isinstance(color, str):
        c = color.strip()
        if c.startswith("rgba("):
            parts = c[5:-1].split(",")
            if len(parts) >= 3:
                r, g, b = [p.strip() for p in parts[:3]]
                return f"rgba({r},{g},{b},{alpha})"
        if c.startswith("rgb("):
            parts = c[4:-1].split(",")
            if len(parts) >= 3:
                r, g, b = [p.strip() for p in parts[:3]]
                return f"rgba({r},{g},{b},{alpha})"
        if c.startswith("#") and len(c) == 7:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(200,200,200,{alpha})"


def _region_color_map(regions):
    regions = [str(r) for r in regions]
    colors = {}
    try:
        from iblatlas.regions import BrainRegions

        br = BrainRegions()
        for region in regions:
            try:
                idx = br.acronym2index(region)[1][0][0]
                rgb = br.rgb[idx]
                colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
            except Exception:
                continue
    except Exception:
        pass

    if len(colors) < len(regions):
        fallback = px.colors.qualitative.Dark24
        for i, region in enumerate(regions):
            if region not in colors:
                colors[region] = fallback[i % len(fallback)]
    return colors


def _compute_variability_curve(
    bin_centers, rate, window_s=0.1, step_s=0.025, metric="fano"
):
    if bin_centers is None or rate is None:
        return None, None
    bin_centers = np.asarray(bin_centers, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if bin_centers.size == 0 or rate.size == 0:
        return None, None
    n = min(bin_centers.size, rate.size)
    bin_centers = bin_centers[:n]
    rate = rate[:n]
    if step_s <= 0 or window_s <= 0:
        return None, None
    start = float(bin_centers[0])
    end = float(bin_centers[-1])
    if end - start < window_s:
        return None, None
    centers = []
    fano_vals = []
    win_start = start
    while win_start + window_s <= end + 1e-9:
        win_end = win_start + window_s
        mask = (bin_centers >= win_start) & (bin_centers < win_end)
        if np.sum(mask) >= 2:
            vals = rate[mask]
            mean_val = np.nanmean(vals)
            var_val = np.nanvar(vals)
            std_val = np.sqrt(var_val)
            if metric == "cv":
                fano = std_val / mean_val if np.isfinite(mean_val) and mean_val != 0 else np.nan
            else:
                fano = var_val / mean_val if np.isfinite(mean_val) and mean_val != 0 else np.nan
        else:
            fano = np.nan
        centers.append(win_start + window_s / 2)
        fano_vals.append(fano)
        win_start += step_s
    if len(centers) == 0:
        return None, None
    return np.asarray(centers, dtype=float), np.asarray(fano_vals, dtype=float)


def _moving_mean(values, window_bins):
    values = np.asarray(values, dtype=float)
    if window_bins <= 1 or values.size == 0:
        return values
    kernel = np.ones(int(window_bins), dtype=float) / float(window_bins)
    return np.convolve(values, kernel, mode="same")


def _get_trial_event(trials, event_name, trial_idx):
    if trials is None:
        return np.nan
    if hasattr(trials, "keys"):
        if event_name not in trials.keys():
            return np.nan
        events = np.asarray(trials[event_name])
    else:
        return np.nan
    if trial_idx < 0 or trial_idx >= len(events):
        return np.nan
    return events[trial_idx]


def _get_quality_mask(clusters, cluster_ids, only_good):
    if not only_good:
        return np.ones(len(cluster_ids), dtype=bool)
    label_values = _get_label_values(clusters, cluster_ids)
    if label_values is None:
        return np.ones(len(cluster_ids), dtype=bool)
    return np.asarray(label_values == 1, dtype=bool)

def _get_label_values(clusters, cluster_ids):
    cluster_ids = np.asarray(cluster_ids)
    metrics = _get_metrics_df(clusters)
    labels = None
    if isinstance(metrics, pd.DataFrame) and "label" in metrics.columns:
        labels = np.asarray(metrics["label"])
    elif isinstance(metrics, dict) and "label" in metrics:
        labels = np.asarray(metrics["label"])
    elif hasattr(metrics, "label"):
        labels = np.asarray(metrics.label)
    values = _map_labels_to_cluster_ids(labels, cluster_ids, _get_metric_cluster_ids(metrics))
    if values is not None:
        return values

    labels = None
    if hasattr(clusters, "label"):
        labels = np.asarray(clusters.label)
    if labels is None and isinstance(clusters, dict) and "label" in clusters:
        labels = np.asarray(clusters.get("label"))
    return _map_labels_to_cluster_ids(labels, cluster_ids, _get_cluster_attr(clusters, "cluster_id", None))


def _get_depths(clusters, n_units):
    depths = _get_cluster_attr(clusters, "depths", None)
    if depths is None:
        depths = _get_cluster_attr(clusters, "depth", None)
    if depths is None:
        return np.arange(n_units)
    depths = np.asarray(depths)
    if depths.shape[0] == n_units:
        return depths
    # Fall back to mapping by cluster_id if lengths mismatch.
    cluster_id_all = _get_cluster_attr(clusters, "cluster_id", None)
    if cluster_id_all is None:
        return np.arange(n_units)
    cluster_id_all = np.asarray(cluster_id_all)
    if cluster_id_all.shape[0] != depths.shape[0]:
        return np.arange(n_units)
    depth_lookup = dict(zip(cluster_id_all.tolist(), depths.tolist()))
    return depth_lookup


def _prepare_units_df(cluster_ids, cluster_acronyms, clusters, only_good, label_min=None):
    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    quality_mask = _get_quality_mask(clusters, cluster_ids, only_good)
    if label_min is not None:
        label_values = _get_label_values(clusters, cluster_ids)
        if label_values is not None:
            label_mask = label_values >= float(label_min)
            quality_mask = quality_mask & label_mask
    depths = _get_depths(clusters, len(cluster_ids))
    if isinstance(depths, dict):
        depths = np.array([depths.get(cid, np.nan) for cid in cluster_ids], dtype=float)

    df_units = pd.DataFrame(
        {
            "cluster_id": cluster_ids[quality_mask],
            "acronym": cluster_acronyms[quality_mask],
            "depth": depths[quality_mask],
        }
    )
    df_units["acronym"] = df_units["acronym"].astype(str)
    df_units = df_units[~df_units["acronym"].isin(["root", "void"])]
    return df_units, quality_mask


def _merge_metric(
    df_units,
    metric_key,
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
):
    metric_key = (metric_key or "depth").strip().lower()
    df_units = df_units.copy()
    sort_label = "Depth"

    def _merge_coupling_metric(df_local, df_src, columns, label):
        if df_src is not None:
            for col in columns:
                if col in df_src.columns:
                    df_local = df_local.merge(
                        df_src[["cluster_id", col]].rename(columns={col: "sort_metric"}),
                        on="cluster_id",
                        how="left",
                    )
                    return df_local, label
        df_local["sort_metric"] = df_local["depth"]
        return df_local, "Depth"

    if metric_key in ("depth", "default"):
        df_units["sort_metric"] = df_units["depth"]
        return df_units, sort_label

    if "stim" in metric_key:
        delay_col = delay_column_name("stimOn_times")
        sort_label = "Delay (Stim)"
        if df_res is not None and delay_col in df_res.columns:
            df_units = df_units.merge(
                df_res[["cluster_id", delay_col]].rename(columns={delay_col: "sort_metric"}),
                on="cluster_id",
                how="left",
            )
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "move" in metric_key:
        delay_col = delay_column_name("firstMovement_times")
        sort_label = "Delay (First Move)"
        if df_res is not None and delay_col in df_res.columns:
            df_units = df_units.merge(
                df_res[["cluster_id", delay_col]].rename(columns={delay_col: "sort_metric"}),
                on="cluster_id",
                how="left",
            )
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "feedback" in metric_key:
        delay_col = delay_column_name("feedback_times")
        sort_label = "Delay (Feedback)"
        if df_res is not None and delay_col in df_res.columns:
            df_units = df_units.merge(
                df_res[["cluster_id", delay_col]].rename(columns={delay_col: "sort_metric"}),
                on="cluster_id",
                how="left",
            )
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "response" in metric_key:
        delay_col = delay_column_name("response_times")
        sort_label = "Delay (Response)"
        if df_res is not None and delay_col in df_res.columns:
            df_units = df_units.merge(
                df_res[["cluster_id", delay_col]].rename(columns={delay_col: "sort_metric"}),
                on="cluster_id",
                how="left",
            )
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "firing" in metric_key:
        sort_label = "Firing rate"
        if df_firing_rate is not None:
            for col in ("firing_rate", "firing_rate_h1", "firing_rate_h2"):
                if col in df_firing_rate.columns:
                    df_units = df_units.merge(
                        df_firing_rate[["cluster_id", col]].rename(columns={col: "sort_metric"}),
                        on="cluster_id",
                        how="left",
                    )
                    return df_units, sort_label
        df_units["sort_metric"] = df_units["depth"]
        return df_units, "Depth"

    if "strength" in metric_key:
        if "spont" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling,
                ["coupling_strength", "coupling_strength_h1", "coupling_strength_h2"],
                "stPR Strength (Spont)",
            )
        if "task" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling_task,
                ["coupling_strength", "coupling_strength_odd", "coupling_strength_even"],
                "stPR Strength (Task)",
            )
        if "iti" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling_iti,
                ["coupling_strength", "coupling_strength_odd", "coupling_strength_even"],
                "stPR Strength (ITI)",
            )

    if "max" in metric_key:
        if "spont" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling,
                ["coupling_max", "coupling_max_h1", "coupling_max_h2"],
                "stPR Max (Spont)",
            )
        if "task" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling_task,
                ["coupling_max", "coupling_max_odd", "coupling_max_even"],
                "stPR Max (Task)",
            )
        if "iti" in metric_key:
            return _merge_coupling_metric(
                df_units,
                df_coupling_iti,
                ["coupling_max", "coupling_max_odd", "coupling_max_even"],
                "stPR Max (ITI)",
            )

    if "spont" in metric_key:
        sort_label = "stPR Delay (Spont)"
        if df_coupling is not None:
            if "sorting_number" in df_coupling.columns:
                df_units = df_units.merge(
                    df_coupling[["cluster_id", "sorting_number"]].rename(
                        columns={"sorting_number": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            elif "coupling_delay_ms" in df_coupling.columns:
                df_units = df_units.merge(
                    df_coupling[["cluster_id", "coupling_delay_ms"]].rename(
                        columns={"coupling_delay_ms": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            else:
                df_units["sort_metric"] = df_units["depth"]
                sort_label = "Depth"
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "task" in metric_key:
        sort_label = "stPR Delay (Task)"
        if df_coupling_task is not None:
            if "sorting_number" in df_coupling_task.columns:
                df_units = df_units.merge(
                    df_coupling_task[["cluster_id", "sorting_number"]].rename(
                        columns={"sorting_number": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            elif "coupling_delay_ms" in df_coupling_task.columns:
                df_units = df_units.merge(
                    df_coupling_task[["cluster_id", "coupling_delay_ms"]].rename(
                        columns={"coupling_delay_ms": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            else:
                df_units["sort_metric"] = df_units["depth"]
                sort_label = "Depth"
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    if "iti" in metric_key:
        sort_label = "stPR Delay (ITI)"
        if df_coupling_iti is not None:
            if "sorting_number" in df_coupling_iti.columns:
                df_units = df_units.merge(
                    df_coupling_iti[["cluster_id", "sorting_number"]].rename(
                        columns={"sorting_number": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            elif "coupling_delay_ms" in df_coupling_iti.columns:
                df_units = df_units.merge(
                    df_coupling_iti[["cluster_id", "coupling_delay_ms"]].rename(
                        columns={"coupling_delay_ms": "sort_metric"}
                    ),
                    on="cluster_id",
                    how="left",
                )
            else:
                df_units["sort_metric"] = df_units["depth"]
                sort_label = "Depth"
        else:
            df_units["sort_metric"] = df_units["depth"]
            sort_label = "Depth"
        return df_units, sort_label

    df_units["sort_metric"] = df_units["depth"]
    return df_units, "Depth"


def _sort_within_regions(df_units, sort_label):
    df_depth_sorted = df_units.sort_values(by="depth", ascending=True).reset_index(drop=True)
    region_order = df_depth_sorted["acronym"].dropna().unique().tolist()
    sorted_groups = []
    for region in region_order:
        region_df = df_depth_sorted[df_depth_sorted["acronym"] == region].copy()
        region_df = region_df.sort_values(
            by="sort_metric", ascending=True, na_position="last"
        ).reset_index(drop=True)
        sorted_groups.append(region_df)
    if sorted_groups:
        df_sorted = pd.concat(sorted_groups, ignore_index=True)
    else:
        df_sorted = df_depth_sorted
    return df_sorted, region_order, sort_label


def plot_trial_raster_plotly(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    sl,
    config_plot,
    trial_idx,
    sorting_metric="depth",
    variability_metric=None,
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
    pupil_features=None,
    pupil_times=None,
    region_colors=None,
):
    """Plot a trial-aligned raster using plotly-resampler with intra-region sorting."""
    trials = _get_session_field(sl, "trials")
    wheel = _get_session_field(sl, "wheel")
    pose = _get_session_field(sl, "pose")
    motion_energy = _get_session_field(sl, "motion_energy")
    if trials is None:
        return go.Figure()

    template = config_plot.get("PLOTLY_TEMPLATE", DEFAULT_TEMPLATE)
    base_color = _template_base_color(template)

    t_stim_on = _get_trial_event(trials, "stimOn_times", trial_idx)
    t_first_move = _get_trial_event(trials, "firstMovement_times", trial_idx)
    t_feedback = _get_trial_event(trials, "feedback_times", trial_idx)

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in trials.keys():
        align_event = "stimOn_times"
    t_align = _get_trial_event(trials, align_event, trial_idx)
    if np.isnan(t_align):
        t_align = t_stim_on

    use_event_window = config_plot.get("TRIAL_RASTER_USE_EVENT_WINDOW", False)
    if use_event_window:
        win_start = config_plot.get("PSTH_WINDOW_START", -config_plot["RASTER_WINDOW_PRE"])
        win_end = config_plot.get("PSTH_WINDOW_END", config_plot["RASTER_WINDOW_POST"])
        valid_events = [t for t in [t_stim_on, t_first_move, t_feedback] if np.isfinite(t)]
        if valid_events:
            t_start = min(valid_events) + win_start
            t_end = max(valid_events) + win_end
        else:
            t_start = t_align - config_plot["RASTER_WINDOW_PRE"]
            t_end = t_align + config_plot["RASTER_WINDOW_POST"]
    else:
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

    cont_l = trials["contrastLeft"][trial_idx]
    cont_r = trials["contrastRight"][trial_idx]
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
    response_str = choice_map.get(trials["choice"][trial_idx], "NA")
    fb_val = trials["feedbackType"][trial_idx]
    outcome_str = "Correct" if fb_val == 1 else "Incorrect"

    plot_title = (
        f"Trial {trial_idx} | Contrast: {contrast_val} ({stim_side}) | "
        f"Response: {response_str} | {outcome_str}"
    )

    df_units, _ = _prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        config_plot["PLOT_ONLY_GOOD_UNITS"],
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )
    if df_units.empty:
        return go.Figure()

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        avg_psth_only_good,
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )

    df_units, sort_label = _merge_metric(
        df_units,
        sorting_metric,
        df_res=df_res,
        df_coupling=df_coupling,
        df_coupling_task=df_coupling_task,
        df_coupling_iti=df_coupling_iti,
        df_firing_rate=df_firing_rate,
    )
    df_units, region_order, sort_label = _sort_within_regions(df_units, sort_label)

    mask_window = (spikes.times >= t_start) & (spikes.times <= t_end)
    window_spike_times_all = spikes.times[mask_window]
    window_spike_clusters_all = spikes.clusters[mask_window]

    cluster_index_map = dict(zip(df_units["cluster_id"].values, df_units.index.values))
    cluster_region_map = dict(zip(df_units["cluster_id"].values, df_units["acronym"].values))

    spike_mask = np.isin(window_spike_clusters_all, df_units["cluster_id"].values)
    window_spike_times = window_spike_times_all[spike_mask] - t_offset
    window_spike_clusters = window_spike_clusters_all[spike_mask]
    spike_y = pd.Series(window_spike_clusters).map(cluster_index_map).to_numpy()
    spike_regions = pd.Series(window_spike_clusters).map(cluster_region_map).to_numpy()

    metric = (
        variability_metric
        if variability_metric is not None
        else config_plot.get("PSTH_VARIABILITY_METRIC", "fano")
    )
    metric = str(metric).lower()
    metric_title = "Fano Factor" if metric == "fano" else "CV"

    fig = FigureResampler(
        make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.52, 0.12, 0.10, 0.10, 0.08, 0.08],
            subplot_titles=(
                "Raster",
                "Avg PSTH",
                metric_title,
                "Wheel",
                "Paw Speed",
                "Motion energy (L/R)",
            ),
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=window_spike_times,
            y=spike_y,
            mode="markers",
            marker=dict(color=base_color, size=4, symbol="line-ns-open"),
            customdata=np.column_stack([window_spike_clusters, spike_regions]),
            hovertemplate=(
                "Time: %{x:.3f}s<br>Unit: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
            ),
            name="Spikes",
        ),
        max_n_samples=len(window_spike_times),
        hf_x=window_spike_times,
        hf_y=spike_y,
        row=1,
        col=1,
    )

    if region_colors is None:
        region_colors = _region_color_map(region_order)

    for acronym in region_order:
        group = df_units[df_units["acronym"] == acronym]
        if group.empty:
            continue
        y0 = group.index.min() - 0.5
        y1 = group.index.max() + 0.5
        fill_color = _color_to_rgba(region_colors.get(acronym), alpha=0.18)
        fig.add_shape(
            type="rect",
            x0=t_start - t_offset,
            x1=t_end - t_offset,
            y0=y0,
            y1=y1,
            line=dict(width=0),
            fillcolor=fill_color,
            layer="below",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=t_end - t_offset,
            y=(y0 + y1) / 2,
            xanchor="left",
            yanchor="middle",
            text=acronym,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xshift=10,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=region_colors.get(acronym), size=8),
                name=acronym,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if wheel is not None:
        mask_wheel = (wheel["times"] >= t_start) & (wheel["times"] <= t_end)
        wheel_t = wheel["times"][mask_wheel] - t_offset
        wheel_pos = wheel["position"][mask_wheel]
    else:
        wheel_t = np.array([])
        wheel_pos = np.array([])
    # Average PSTH by region (single-event window)
    bin_size = config_plot.get("POP_BIN_SIZE", 0.005)
    smooth_window_s = 0.05
    smooth_bins = max(1, int(round(smooth_window_s / bin_size))) if bin_size > 0 else 1
    fano_window_s = 0.1
    fano_step_s = 0.025
    if align_to_event:
        psth_start = t_start - t_offset
        psth_end = t_end - t_offset
    else:
        psth_start = t_start
        psth_end = t_end

    if psth_end > psth_start and len(df_units_psth) > 0:
        bins = np.arange(psth_start, psth_end + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        if align_to_event:
            psth_spike_times_all = window_spike_times_all - t_offset
        else:
            psth_spike_times_all = window_spike_times_all
        for acronym in region_order:
            region_ids = df_units_psth.loc[
                df_units_psth["acronym"] == acronym, "cluster_id"
            ].values
            if len(region_ids) == 0:
                continue
            region_mask = np.isin(window_spike_clusters_all, region_ids)
            region_spike_times = psth_spike_times_all[region_mask]
            counts, _ = np.histogram(region_spike_times, bins=bins)
            rate = counts / (len(region_ids) * bin_size)
            rate_smoothed = _moving_mean(rate, smooth_bins)
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=rate_smoothed,
                    mode="lines",
                    line=dict(color=region_colors.get(acronym)),
                    name=f"{acronym} PSTH",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            var_x, var_y = _compute_variability_curve(
                bin_centers,
                rate,
                window_s=fano_window_s,
                step_s=fano_step_s,
                metric=metric,
            )
            if var_x is not None and var_y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=var_x,
                        y=var_y,
                        mode="lines",
                        line=dict(color=region_colors.get(acronym)),
                        name=f"{acronym} {metric_title}",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

    fig.add_trace(
        go.Scatter(x=wheel_t, y=wheel_pos, mode="lines", line=dict(color=base_color)),
        row=4,
        col=1,
    )

    pose_t = None
    paw_speed = None
    if pose is not None and "leftCamera" in pose:
        pose_df = pose["leftCamera"]
        if "times" in pose_df.columns:
            pose_timestamps = pose_df["times"].values
        else:
            pose_timestamps = pose_df.index.values
        mask_pose = (pose_timestamps >= t_start) & (pose_timestamps <= t_end)
        pose_t = pose_timestamps[mask_pose] - t_offset
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

    if paw_speed is not None:
        fig.add_trace(
            go.Scatter(x=pose_t, y=paw_speed, mode="lines", line=dict(color=base_color)),
            row=5,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y5",
            text="Paw data not available",
            showarrow=False,
            row=5,
            col=1,
        )

    left_t, left_me = _extract_motion_energy_series(
        motion_energy, "leftCamera", t_start, t_end, t_offset=t_offset
    )
    right_t, right_me = _extract_motion_energy_series(
        motion_energy, "rightCamera", t_start, t_end, t_offset=t_offset
    )
    if left_t.size > 0 or right_t.size > 0:
        if left_t.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=left_t,
                    y=left_me,
                    mode="lines",
                    line=dict(color="#1f77b4"),
                    name="Motion L",
                    showlegend=True,
                ),
                row=6,
                col=1,
            )
        if right_t.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=right_t,
                    y=right_me,
                    mode="lines",
                    line=dict(color="#ff7f0e"),
                    name="Motion R",
                    showlegend=True,
                ),
                row=6,
                col=1,
            )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y6",
            text="Motion energy not available",
            showarrow=False,
            row=6,
            col=1,
        )

    event_lines = [
        ("Stim On", t_stim_on, "blue"),
        ("First Move", t_first_move, "green"),
        ("Feedback", t_feedback, "red"),
    ]
    for name, time_val, color in event_lines:
        for row in range(1, 7):
            fig.add_vline(x=time_val - t_offset, line=dict(color=color, width=2), row=row, col=1)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    ylabel_text = (
        f"Good Units (n={len(df_units)})"
        if config_plot["PLOT_ONLY_GOOD_UNITS"]
        else f"All Units (n={len(df_units)})"
    )

    fig.update_yaxes(
        title_text=ylabel_text,
        row=1,
        col=1,
        showticklabels=False,
        range=[-0.5, len(df_units) - 0.5],
    )
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=2, col=1)
    fig.update_yaxes(title_text=metric_title, row=3, col=1)
    fig.update_yaxes(title_text="Wheel (rad)", row=4, col=1)
    fig.update_yaxes(title_text="Paw (px/s)", row=5, col=1)
    fig.update_yaxes(title_text="Motion energy", row=6, col=1)
    fig.update_xaxes(title_text=xlabel_text, row=6, col=1)
    fig.update_xaxes(range=[t_start - t_offset, t_end - t_offset])

    fig.update_layout(
        title=f"{plot_title} | Sort: {sort_label}",
        height=1120,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))

    return fig


def _template_base_color(template):
    if template is None:
        return "black"
    return "white" if "dark" in template.lower() else "black"


def plot_single_neuron_plotly(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    cluster_id,
):
    """Plot PSTHs and rasters for a single neuron using plotly."""
    trials = _get_session_field(sl, "trials")
    if trials is None:
        return go.Figure()

    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    if cluster_id not in cluster_ids:
        fallback_id = cluster_ids[0] if len(cluster_ids) > 0 else None
        if fallback_id is None:
            return go.Figure()
        cluster_id = fallback_id

    idx = int(np.where(cluster_ids == cluster_id)[0][0])
    target_acronym = cluster_acronyms[idx]

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in trials.keys():
        align_event = "stimOn_times"

    delay_col = delay_column_name(align_event)
    delay_odd_col = delay_split_column_name(align_event, "odd")
    delay_even_col = delay_split_column_name(align_event, "even")
    unit_delay = np.nan
    if df_res is not None and delay_col in df_res.columns:
        match = df_res[df_res["cluster_id"] == cluster_id]
        if not match.empty:
            unit_delay = float(match.iloc[0][delay_col])
    delay_s = (
        float(_delay_to_seconds(unit_delay, config_plot))
        if np.isfinite(unit_delay)
        else np.nan
    )
    delay_ms = (
        float(_delay_to_ms(unit_delay, config_plot))
        if np.isfinite(unit_delay)
        else np.nan
    )

    def _get_trials_array(key):
        return _get_trials_array_field(trials, key)

    def _get_spike_array(key):
        if spikes is None:
            return np.array([])
        if isinstance(spikes, dict):
            return np.asarray(spikes.get(key, []))
        return np.asarray(getattr(spikes, key, []))

    spike_times = _get_spike_array("times")
    spike_clusters = _get_spike_array("clusters")
    spikes_obj = spikes
    if isinstance(spikes, dict):
        spikes_obj = SimpleNamespace(times=spike_times, clusters=spike_clusters)
    neuron_spikes = spike_times[spike_clusters == cluster_id]

    raster_pre = config_plot["SINGLE_NEURON_RASTER_PRE"]
    raster_post = config_plot["SINGLE_NEURON_RASTER_POST"]
    bin_size = config_plot["SINGLE_NEURON_BIN_SIZE"]
    smooth_sigma = config_plot["SINGLE_NEURON_SMOOTH_SIGMA"]

    template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
    base_color = _template_base_color(template)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Left Stimuli PSTH",
            "Right Stimuli PSTH",
            "Left Stimuli Raster",
            "Right Stimuli Raster",
            "Global PSTH (All Trials)",
        ),
        row_heights=[0.25, 0.45, 0.3],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    contrasts_to_plot = [1.0, 0.25, 0.125, 0.0625, 0.0]
    contrast_colors = {
        1.0: "rgba(0,0,0,1.0)",
        0.5: "rgba(51,51,51,1.0)",
        0.25: "rgba(102,102,102,1.0)",
        0.125: "rgba(153,153,153,1.0)",
        0.0625: "rgba(191,191,191,1.0)",
        0.0: "rgba(217,217,217,1.0)",
    }

    event_series = _get_trials_array(align_event)
    if event_series is None:
        return go.Figure()

    sides = [("Left", "contrastLeft"), ("Right", "contrastRight")]
    global_curves = []
    global_bin_centers = None
    for col_idx, (side_label, contrast_key) in enumerate(sides, start=1):
        contrast_arr = _get_trials_array(contrast_key)
        if contrast_arr is None:
            continue

        current_raster_y = 0
        raster_x = []
        raster_y = []
        raster_colors = []

        for cont in contrasts_to_plot:
            mask = (contrast_arr == cont) & (~np.isnan(event_series))
            events = event_series[mask]
            if len(events) == 0:
                continue

            psth_by_cluster, bin_centers = compute_psth_for_clusters(
                spikes_obj,
                [cluster_id],
                events,
                -raster_pre,
                raster_post,
                bin_size,
                smooth_sigma,
                show_progress=False,
            )
            psth_entry = psth_by_cluster.get(cluster_id)
            if psth_entry and bin_centers is not None:
                firing_rate = psth_entry["fr_smooth"]
            else:
                firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

            color_val = contrast_colors.get(cont, "rgba(0,0,0,1.0)")
            label_text = f"{cont * 100:.0f}%" if cont > 0 else "0%"
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=firing_rate,
                    mode="lines",
                    line=dict(color=color_val, width=2),
                    name=label_text,
                    showlegend=(col_idx == 1),
                    legendgroup=label_text,
                ),
                row=1,
                col=col_idx,
            )
            if bin_centers is not None and len(firing_rate) > 0:
                global_curves.append(np.asarray(firing_rate, dtype=float))
                if global_bin_centers is None:
                    global_bin_centers = np.asarray(bin_centers, dtype=float)

            for event_t in events:
                t_start = event_t - raster_pre
                t_end = event_t + raster_post
                trial_spikes = neuron_spikes[
                    (neuron_spikes >= t_start) & (neuron_spikes <= t_end)
                ]
                aligned_spikes = trial_spikes - event_t
                if len(aligned_spikes) > 0:
                    raster_x.extend(aligned_spikes.tolist())
                    raster_y.extend([current_raster_y] * len(aligned_spikes))
                    raster_colors.extend([color_val] * len(aligned_spikes))
                current_raster_y += 1

            current_raster_y += 2

        fig.add_trace(
            go.Scattergl(
                x=raster_x,
                y=raster_y,
                mode="markers",
                marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
                showlegend=False,
            ),
            row=2,
            col=col_idx,
        )

        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=col_idx)
        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=col_idx)
        fig.update_xaxes(range=[-raster_pre, raster_post], row=1, col=col_idx)
        fig.update_xaxes(range=[-raster_pre, raster_post], row=2, col=col_idx)
        fig.update_yaxes(title_text="Firing Rate (Hz)", row=1, col=col_idx)
        fig.update_yaxes(title_text="Trials", showticklabels=False, row=2, col=col_idx)

    if global_curves and global_bin_centers is not None:
        min_len = min(len(curve) for curve in global_curves)
        curves = [curve[:min_len] for curve in global_curves]
        global_curve = np.nanmean(np.vstack(curves), axis=0)
        x_vals = global_bin_centers[:min_len]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=global_curve,
                mode="lines",
                line=dict(color=base_color, width=2),
                name="Global PSTH",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        if np.isfinite(delay_s):
            fig.add_vline(
                x=delay_s,
                line=dict(color="red", width=2),
                row=3,
                col=1,
            )
            y_text = float(np.nanmax(global_curve)) * 0.9 if len(global_curve) > 0 else 0
            fig.add_annotation(
                x=delay_s,
                y=y_text,
                text=f"{delay_ms:.0f} ms" if np.isfinite(delay_ms) else "NA",
                showarrow=False,
                font=dict(color="red"),
                row=3,
                col=1,
            )

        fig.add_vline(
            x=0,
            line=dict(color="blue", dash="dash"),
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text=f"Time from {event_label(align_event)} (s)", row=3, col=1)
        fig.update_yaxes(title_text="Firing Rate (Hz)", row=3, col=1)
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No valid trials found",
            showarrow=False,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=f"Cluster #{cluster_id} ({target_acronym}) Response Analysis",
        height=900,
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))

    return fig


def plot_single_neuron_conditioned_event_plotly(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    cluster_id,
    event_name,
    condition_type,
    title=None,
):
    """Plot PSTH + raster + global PSTH for a single neuron conditioned on trial groups."""
    trials = _get_session_field(sl, "trials")
    if trials is None:
        return go.Figure()

    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    if cluster_id not in cluster_ids:
        fallback_id = cluster_ids[0] if len(cluster_ids) > 0 else None
        if fallback_id is None:
            return go.Figure()
        cluster_id = fallback_id

    idx = int(np.where(cluster_ids == cluster_id)[0][0])
    target_acronym = cluster_acronyms[idx]

    event_series = _get_trials_array_field(trials, event_name)
    if event_series is None:
        return go.Figure()

    choice_arr = _get_trials_array_field(trials, "choice")
    feedback_arr = _get_trials_array_field(trials, "feedbackType")

    if condition_type == "choice":
        conditions = [
            ("Left", choice_arr == 1, "#1f77b4"),
            ("Right", choice_arr == -1, "#ff7f0e"),
        ]
        sort_label = "Left/Right"
    elif condition_type == "feedback":
        conditions = [
            ("Correct", feedback_arr == 1, "#2ca02c"),
            ("Incorrect", feedback_arr == -1, "#d62728"),
        ]
        sort_label = "Correct/Incorrect"
    else:
        return go.Figure()

    raster_pre = config_plot["SINGLE_NEURON_RASTER_PRE"]
    raster_post = config_plot["SINGLE_NEURON_RASTER_POST"]
    bin_size = config_plot["SINGLE_NEURON_BIN_SIZE"]
    smooth_sigma = config_plot["SINGLE_NEURON_SMOOTH_SIGMA"]

    template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
    base_color = _template_base_color(template)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{event_label(event_name)} PSTH",
            f"{event_label(event_name)} Raster ({sort_label})",
            f"{event_label(event_name)} Global PSTH",
        ),
    )

    spike_times = np.asarray(spikes.times if hasattr(spikes, "times") else spikes.get("times", []))
    spike_clusters = np.asarray(
        spikes.clusters if hasattr(spikes, "clusters") else spikes.get("clusters", [])
    )
    neuron_spikes = spike_times[spike_clusters == cluster_id]

    global_curves = []
    global_bin_centers = None

    raster_x = []
    raster_y = []
    raster_colors = []
    current_raster_y = 0

    for label, cond_mask, color in conditions:
        if cond_mask is None:
            continue
        mask = cond_mask & (~np.isnan(event_series))
        events = event_series[mask]
        if len(events) == 0:
            continue

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            [cluster_id],
            events,
            -raster_pre,
            raster_post,
            bin_size,
            smooth_sigma,
            show_progress=False,
        )
        psth_entry = psth_by_cluster.get(cluster_id)
        if psth_entry and bin_centers is not None:
            firing_rate = psth_entry["fr_smooth"]
        else:
            firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=firing_rate,
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        if bin_centers is not None and len(firing_rate) > 0:
            global_curves.append(np.asarray(firing_rate, dtype=float))
            if global_bin_centers is None:
                global_bin_centers = np.asarray(bin_centers, dtype=float)

        for event_t in events:
            t_start = event_t - raster_pre
            t_end = event_t + raster_post
            trial_spikes = neuron_spikes[
                (neuron_spikes >= t_start) & (neuron_spikes <= t_end)
            ]
            aligned_spikes = trial_spikes - event_t
            if len(aligned_spikes) > 0:
                raster_x.extend(aligned_spikes.tolist())
                raster_y.extend([current_raster_y] * len(aligned_spikes))
                raster_colors.extend([color] * len(aligned_spikes))
            current_raster_y += 1

        current_raster_y += 2

    fig.add_trace(
        go.Scattergl(
            x=raster_x,
            y=raster_y,
            mode="markers",
            marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=1)
    fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=1)
    fig.update_xaxes(range=[-raster_pre, raster_post], row=1, col=1)
    fig.update_xaxes(range=[-raster_pre, raster_post], row=2, col=1)
    fig.update_yaxes(title_text="Firing Rate (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Trials", showticklabels=False, row=2, col=1)

    delay_col = delay_column_name(event_name)
    unit_delay = np.nan
    if df_res is not None and delay_col in df_res.columns:
        match = df_res[df_res["cluster_id"] == cluster_id]
        if not match.empty:
            unit_delay = float(match.iloc[0][delay_col])
    delay_s = (
        float(_delay_to_seconds(unit_delay, config_plot))
        if np.isfinite(unit_delay)
        else np.nan
    )
    delay_ms = (
        float(_delay_to_ms(unit_delay, config_plot))
        if np.isfinite(unit_delay)
        else np.nan
    )

    if global_curves and global_bin_centers is not None:
        min_len = min(len(curve) for curve in global_curves)
        curves = [curve[:min_len] for curve in global_curves]
        global_curve = np.nanmean(np.vstack(curves), axis=0)
        x_vals = global_bin_centers[:min_len]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=global_curve,
                mode="lines",
                line=dict(color=base_color, width=2),
                name="Global PSTH",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        if np.isfinite(delay_s):
            fig.add_vline(
                x=delay_s,
                line=dict(color="red", width=2),
                row=3,
                col=1,
            )
            y_text = float(np.nanmax(global_curve)) * 0.9 if len(global_curve) > 0 else 0
            fig.add_annotation(
                x=delay_s,
                y=y_text,
                text=f"{delay_ms:.0f} ms" if np.isfinite(delay_ms) else "NA",
                showarrow=False,
                font=dict(color="red"),
                row=3,
                col=1,
            )

        fig.add_vline(
            x=0,
            line=dict(color="blue", dash="dash"),
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text=f"Time from {event_label(event_name)} (s)", row=3, col=1)
        fig.update_yaxes(title_text="Firing Rate (Hz)", row=3, col=1)

    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No valid trials found",
            showarrow=False,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title or f"{event_label(event_name)} Response",
        height=700,
        margin=dict(l=70, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_layout(template=template, font=dict(color=base_color))

    return fig


def _coerce_single_neuron_event_groups(event_groups):
    groups = []
    if event_groups is None:
        return groups
    for item in event_groups:
        label = None
        events = None
        color = None
        if isinstance(item, dict):
            label = item.get("label")
            events = item.get("events")
            color = item.get("color")
        elif isinstance(item, (list, tuple)):
            if len(item) > 0:
                label = item[0]
            if len(item) > 1:
                events = item[1]
            if len(item) > 2:
                color = item[2]
        if label is None:
            label = "Group"
        if events is None:
            continue
        arr = np.asarray(events, dtype=float).ravel()
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        groups.append((str(label), np.sort(arr), color))
    return groups


def plot_single_neuron_event_groups_plotly(
    spikes,
    cluster_ids,
    cluster_acronyms,
    config_plot,
    cluster_id,
    event_groups,
    title=None,
    xaxis_event_label="event onset",
    legend_title=None,
):
    """Plot single-neuron PSTH+raster for custom event groups."""
    cluster_ids = np.asarray(cluster_ids) if cluster_ids is not None else np.array([])
    cluster_acronyms = (
        np.asarray(cluster_acronyms).astype(str)
        if cluster_acronyms is not None
        else np.array([])
    )
    if cluster_ids.size == 0:
        return go.Figure()

    if cluster_id not in cluster_ids:
        cluster_id = cluster_ids[0]
    idx = int(np.where(cluster_ids == cluster_id)[0][0])
    if len(cluster_acronyms) == len(cluster_ids):
        target_acronym = cluster_acronyms[idx]
    else:
        target_acronym = "NA"

    if spikes is None:
        return go.Figure()
    if hasattr(spikes, "times"):
        spike_times = np.asarray(spikes.times)
    elif isinstance(spikes, dict):
        spike_times = np.asarray(spikes.get("times", []))
    else:
        spike_times = np.array([])
    if hasattr(spikes, "clusters"):
        spike_clusters = np.asarray(spikes.clusters)
    elif isinstance(spikes, dict):
        spike_clusters = np.asarray(spikes.get("clusters", []))
    else:
        spike_clusters = np.array([])
    n_spikes = min(len(spike_times), len(spike_clusters))
    spike_times = spike_times[:n_spikes]
    spike_clusters = spike_clusters[:n_spikes]
    spikes_obj = SimpleNamespace(times=spike_times, clusters=spike_clusters)
    neuron_spikes = spike_times[spike_clusters == cluster_id]

    raster_pre = float(config_plot.get("SINGLE_NEURON_RASTER_PRE", 0.2))
    raster_post = float(config_plot.get("SINGLE_NEURON_RASTER_POST", 0.35))
    bin_size = float(config_plot.get("SINGLE_NEURON_BIN_SIZE", 0.03))
    smooth_sigma = float(config_plot.get("SINGLE_NEURON_SMOOTH_SIGMA", 0.5))

    template = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
    base_color = _template_base_color(template)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{xaxis_event_label} PSTH",
            f"{xaxis_event_label} Raster",
            f"{xaxis_event_label} Global PSTH",
        ),
    )

    parsed_groups = _coerce_single_neuron_event_groups(event_groups)
    palette = px.colors.qualitative.Dark24
    global_curves = []
    global_bin_centers = None
    raster_x = []
    raster_y = []
    raster_colors = []
    current_raster_y = 0
    plotted_groups = 0

    for group_idx, (label, events, color) in enumerate(parsed_groups):
        color_use = color or palette[group_idx % len(palette)]
        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes_obj,
            [cluster_id],
            events,
            -raster_pre,
            raster_post,
            bin_size,
            smooth_sigma,
            show_progress=False,
        )
        psth_entry = psth_by_cluster.get(cluster_id)
        if psth_entry and bin_centers is not None:
            firing_rate = psth_entry["fr_smooth"]
        else:
            firing_rate = np.zeros(len(bin_centers) if bin_centers is not None else 0)

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=firing_rate,
                mode="lines",
                line=dict(color=color_use, width=2),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        plotted_groups += 1

        if bin_centers is not None and len(firing_rate) > 0:
            global_curves.append(np.asarray(firing_rate, dtype=float))
            if global_bin_centers is None:
                global_bin_centers = np.asarray(bin_centers, dtype=float)

        for event_t in events:
            t_start = event_t - raster_pre
            t_end = event_t + raster_post
            trial_spikes = neuron_spikes[
                (neuron_spikes >= t_start) & (neuron_spikes <= t_end)
            ]
            aligned_spikes = trial_spikes - event_t
            if len(aligned_spikes) > 0:
                raster_x.extend(aligned_spikes.tolist())
                raster_y.extend([current_raster_y] * len(aligned_spikes))
                raster_colors.extend([color_use] * len(aligned_spikes))
            current_raster_y += 1
        current_raster_y += 2

    fig.add_trace(
        go.Scattergl(
            x=raster_x,
            y=raster_y,
            mode="markers",
            marker=dict(color=raster_colors, size=5, symbol="line-ns-open"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=1, col=1)
    fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=2, col=1)
    fig.update_xaxes(range=[-raster_pre, raster_post], row=1, col=1)
    fig.update_xaxes(range=[-raster_pre, raster_post], row=2, col=1)
    fig.update_xaxes(range=[-raster_pre, raster_post], row=3, col=1)
    fig.update_yaxes(title_text="Firing Rate (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Trials", showticklabels=False, row=2, col=1)

    if global_curves and global_bin_centers is not None:
        min_len = min(len(curve) for curve in global_curves)
        curves = [curve[:min_len] for curve in global_curves]
        global_curve = np.nanmean(np.vstack(curves), axis=0)
        x_vals = global_bin_centers[:min_len]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=global_curve,
                mode="lines",
                line=dict(color=base_color, width=2),
                name="Global PSTH",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_vline(
            x=0,
            line=dict(color="blue", dash="dash"),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="Firing Rate (Hz)", row=3, col=1)
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No valid events found",
            showarrow=False,
            row=3,
            col=1,
        )

    if plotted_groups == 0:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No valid events found",
            showarrow=False,
            row=1,
            col=1,
        )

    legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    if legend_title is not None:
        legend_cfg["title"] = dict(text=str(legend_title))
    fig.update_layout(
        title=title or f"Cluster #{cluster_id} ({target_acronym}) Event Response",
        height=700,
        margin=dict(l=70, r=40, t=80, b=60),
        legend=legend_cfg,
    )
    fig.update_xaxes(title_text=f"Time from {xaxis_event_label} (s)", row=3, col=1)
    fig.update_layout(template=template, font=dict(color=base_color))
    return fig


def _format_passive_contrast_label(contrast_value):
    pct = float(contrast_value) * 100.0
    if np.isclose(pct, round(pct)):
        return f"{int(round(pct))}%"
    if np.isclose(pct * 10.0, round(pct * 10.0)):
        return f"{pct:.1f}%"
    return f"{pct:.2f}%"


def plot_single_neuron_passive_visual_plotly(
    spikes,
    cluster_ids,
    cluster_acronyms,
    config_plot,
    cluster_id,
    visual_events_by_contrast,
    title="Passive Visual Response",
):
    """Plot single-neuron passive visual response grouped by contrast."""
    contrast_colors = {
        1.0: "rgba(0,0,0,1.0)",
        0.5: "rgba(51,51,51,1.0)",
        0.25: "rgba(102,102,102,1.0)",
        0.125: "rgba(153,153,153,1.0)",
        0.0625: "rgba(191,191,191,1.0)",
        0.0: "rgba(217,217,217,1.0)",
    }

    event_groups = []
    if isinstance(visual_events_by_contrast, dict):
        for idx, (contrast_key, events) in enumerate(visual_events_by_contrast.items()):
            try:
                contrast_val = float(contrast_key)
            except Exception:
                continue
            color = contrast_colors.get(contrast_val)
            if color is None:
                shade = int(np.clip(40 + idx * 30, 0, 220))
                color = f"rgba({shade},{shade},{shade},1.0)"
            event_groups.append(
                (
                    _format_passive_contrast_label(contrast_val),
                    events,
                    color,
                )
            )

    return plot_single_neuron_event_groups_plotly(
        spikes,
        cluster_ids,
        cluster_acronyms,
        config_plot,
        cluster_id,
        event_groups,
        title=title,
        xaxis_event_label="passive visual onset",
        legend_title="Contrast",
    )


def plot_single_neuron_passive_auditory_plotly(
    spikes,
    cluster_ids,
    cluster_acronyms,
    config_plot,
    cluster_id,
    auditory_event_times,
    title="Passive Auditory Response (Valve vs Tone vs Noise)",
):
    """Plot single-neuron passive auditory response grouped by valve/tone/noise."""
    valve_events = None
    tone_events = None
    noise_events = None
    if isinstance(auditory_event_times, dict):
        valve_events = auditory_event_times.get("valve")
        if valve_events is None:
            valve_events = auditory_event_times.get("passive_valve")
        tone_events = auditory_event_times.get("tone")
        if tone_events is None:
            tone_events = auditory_event_times.get("passive_tone")
        noise_events = auditory_event_times.get("noise")
        if noise_events is None:
            noise_events = auditory_event_times.get("passive_noise")

    event_groups = [
        ("Valve", valve_events, "#17becf"),
        ("Tone", tone_events, "#bcbd22"),
        ("Noise", noise_events, "#8c564b"),
    ]
    return plot_single_neuron_event_groups_plotly(
        spikes,
        cluster_ids,
        cluster_acronyms,
        config_plot,
        cluster_id,
        event_groups,
        title=title,
        xaxis_event_label="passive auditory onset",
        legend_title="Stimulus",
    )


def _build_stpr_lags(config_calc, curve_len=None):
    bin_size_ms = config_calc.get("STPR_BIN_SIZE", 0.001) * 1000
    if bin_size_ms <= 0:
        bin_size_ms = 1.0
    window_ms = config_calc.get("STPR_WINDOW_MS", 80)
    window_bins = int(round(window_ms / bin_size_ms)) if bin_size_ms > 0 else 0
    if curve_len is None:
        return np.arange(-window_bins, window_bins + 1) * bin_size_ms
    expected_len = window_bins * 2 + 1
    if curve_len == expected_len:
        return np.arange(-window_bins, window_bins + 1) * bin_size_ms
    half = (curve_len - 1) / 2.0
    return (np.arange(curve_len) - half) * bin_size_ms


def plot_stpr_curve_halves_plotly(
    df_coupling,
    config_calc,
    cluster_id,
    title=None,
    template=None,
    split_suffixes=None,
    split_labels=None,
):
    """Plot stPR curves for two splits (defaults to first/second half)."""
    fig = go.Figure()
    if template is None:
        template, base_color = _white_theme()
    else:
        base_color = _template_base_color(template)

    if df_coupling is None or len(df_coupling) == 0:
        fig.add_annotation(text="No coupling data available", showarrow=False)
        fig.update_layout(
            title=title or "stPR Curves (First vs Second Half)",
            template=template,
            font=dict(color=base_color, size=13),
            width=900,
            height=650,
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    match = df_coupling[df_coupling["cluster_id"] == cluster_id]
    if match.empty:
        fig.add_annotation(text="Selected neuron not found in coupling data", showarrow=False)
        fig.update_layout(
            title=title or "stPR Curves (First vs Second Half)",
            template=template,
            font=dict(color=base_color, size=13),
            width=900,
            height=650,
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    suffix_a, suffix_b = split_suffixes or ("h1", "h2")
    label_a, label_b = split_labels or ("First Half", "Second Half")

    row = match.iloc[0]
    curve_a = np.asarray(row.get(f"stpr_curve_{suffix_a}", []), dtype=float)
    curve_b = np.asarray(row.get(f"stpr_curve_{suffix_b}", []), dtype=float)
    if curve_a.size == 0 and "stpr_curve" in row:
        curve_a = np.asarray(row.get("stpr_curve", []), dtype=float)
        label_a = "Mean"
        label_b = ""

    if curve_a.size == 0 and curve_b.size == 0:
        fig.add_annotation(text="No stPR curve data available", showarrow=False)
        fig.update_layout(
            title=title or "stPR Curves (First vs Second Half)",
            template=template,
            font=dict(color=base_color, size=13),
            width=900,
            height=650,
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    colors = ["#1f77b4", "#ff7f0e"]
    if curve_a.size > 0:
        lags_a = _build_stpr_lags(config_calc, curve_a.size)
        fig.add_trace(
            go.Scatter(
                x=lags_a,
                y=curve_a,
                mode="lines",
                line=dict(color=colors[0], width=2),
                name=label_a,
            )
        )
        delay_a = row.get(f"coupling_delay_ms_{suffix_a}", np.nan)
        if np.isfinite(delay_a):
            fig.add_vline(x=delay_a, line=dict(color=colors[0], dash="dash"))

    if curve_b.size > 0:
        lags_b = _build_stpr_lags(config_calc, curve_b.size)
        fig.add_trace(
            go.Scatter(
                x=lags_b,
                y=curve_b,
                mode="lines",
                line=dict(color=colors[1], width=2),
                name=label_b or f"{suffix_b}",
            )
        )
        delay_b = row.get(f"coupling_delay_ms_{suffix_b}", np.nan)
        if np.isfinite(delay_b):
            fig.add_vline(x=delay_b, line=dict(color=colors[1], dash="dash"))

    fig.add_vline(x=0, line=dict(color="gray", dash="dot"))

    fig.update_layout(
        title=title or "stPR Curves (First vs Second Half)",
        xaxis_title="Lag (ms)",
        yaxis_title="stPR (z)",
        template=template,
        font=dict(color=base_color, size=13),
        width=900,
        height=650,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def _add_event_vlines(
    fig,
    times,
    label,
    color,
    t_start,
    t_end,
    n_rows=6,
    dash=None,
):
    if fig is None or times is None:
        return
    times_arr = np.asarray(times, dtype=float)
    if times_arr.size == 0:
        return
    if t_start is not None and t_end is not None:
        mask = (times_arr >= t_start) & (times_arr <= t_end) & np.isfinite(times_arr)
    else:
        mask = np.isfinite(times_arr)
    times_arr = times_arr[mask]
    if times_arr.size == 0:
        return
    line = dict(color=color, width=1.5)
    if dash:
        line["dash"] = dash
    for row in range(1, n_rows + 1):
        for t_event in times_arr:
            fig.add_vline(x=float(t_event), line=line, row=row, col=1)
    legend_line = dict(color=color, width=2)
    if dash:
        legend_line["dash"] = dash
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=legend_line,
            name=label,
            showlegend=True,
        ),
        row=1,
        col=1,
    )


def plot_time_window_raster_plotly(
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    sl,
    config_plot,
    t_start,
    t_end,
    region_acronyms=None,
    sorting_metric="depth",
    variability_metric=None,
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
    pupil_features=None,
    pupil_times=None,
    region_colors=None,
    extra_event_times=None,
    extra_event_styles=None,
):
    """Plot a session-time raster for a specified time window."""
    if t_start >= t_end:
        return go.Figure()

    trials = _get_session_field(sl, "trials")
    wheel = _get_session_field(sl, "wheel")
    pose = _get_session_field(sl, "pose")
    motion_energy = _get_session_field(sl, "motion_energy")
    if trials is None:
        return go.Figure()

    template = config_plot.get("PLOTLY_TEMPLATE", DEFAULT_TEMPLATE)
    base_color = _template_base_color(template)

    df_units, _ = _prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        config_plot["PLOT_ONLY_GOOD_UNITS"],
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )
    if df_units.empty:
        return go.Figure()

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        avg_psth_only_good,
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )

    if region_acronyms is not None:
        if isinstance(region_acronyms, str):
            region_acronyms = [region_acronyms]
        region_mask = np.zeros(len(df_units), dtype=bool)
        for region in region_acronyms:
            region_mask |= df_units["acronym"].astype(str).str.startswith(region)
        df_units = df_units.loc[region_mask].copy()
        if not df_units_psth.empty:
            psth_region_mask = np.zeros(len(df_units_psth), dtype=bool)
            for region in region_acronyms:
                psth_region_mask |= df_units_psth["acronym"].astype(str).str.startswith(region)
            df_units_psth = df_units_psth.loc[psth_region_mask].copy()

    if df_units.empty:
        return go.Figure()

    df_units, sort_label = _merge_metric(
        df_units,
        sorting_metric,
        df_res=df_res,
        df_coupling=df_coupling,
        df_coupling_task=df_coupling_task,
        df_coupling_iti=df_coupling_iti,
        df_firing_rate=df_firing_rate,
    )
    df_units, region_order, sort_label = _sort_within_regions(df_units, sort_label)

    mask_window = (spikes.times >= t_start) & (spikes.times <= t_end)
    window_spike_times_all = spikes.times[mask_window]
    window_spike_clusters_all = spikes.clusters[mask_window]

    cluster_index_map = dict(zip(df_units["cluster_id"].values, df_units.index.values))
    cluster_region_map = dict(zip(df_units["cluster_id"].values, df_units["acronym"].values))
    spike_mask = np.isin(window_spike_clusters_all, df_units["cluster_id"].values)
    window_spike_times = window_spike_times_all[spike_mask]
    window_spike_clusters = window_spike_clusters_all[spike_mask]
    spike_y = pd.Series(window_spike_clusters).map(cluster_index_map).to_numpy()
    spike_regions = pd.Series(window_spike_clusters).map(cluster_region_map).to_numpy()

    metric = (
        variability_metric
        if variability_metric is not None
        else config_plot.get("PSTH_VARIABILITY_METRIC", "fano")
    )
    metric = str(metric).lower()
    metric_title = "Fano Factor" if metric == "fano" else "CV"

    fig = FigureResampler(
        make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.52, 0.12, 0.10, 0.10, 0.08, 0.08],
            subplot_titles=(
                "Raster",
                "Avg PSTH",
                metric_title,
                "Wheel",
                "Paw Speed",
                "Motion energy (L/R)",
            ),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=window_spike_times,
            y=spike_y,
            mode="markers",
            marker=dict(color=base_color, size=3, symbol="line-ns-open"),
            customdata=np.column_stack([window_spike_clusters, spike_regions]),
            hovertemplate=(
                "Time: %{x:.3f}s<br>Unit: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
            ),
            name="Spikes",
        ),
        max_n_samples=len(window_spike_times),
        hf_x=window_spike_times,
        hf_y=spike_y,
        row=1,
        col=1,
    )

    if region_colors is None:
        region_colors = _region_color_map(region_order)
    for acronym in region_order:
        group = df_units[df_units["acronym"] == acronym]
        if group.empty:
            continue
        y0 = group.index.min() - 0.5
        y1 = group.index.max() + 0.5
        fill_color = _color_to_rgba(region_colors.get(acronym), alpha=0.18)
        fig.add_shape(
            type="rect",
            x0=t_start,
            x1=t_end,
            y0=y0,
            y1=y1,
            line=dict(width=0),
            fillcolor=fill_color,
            layer="below",
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=region_colors.get(acronym), size=8),
                name=acronym,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if wheel is not None:
        mask_wheel = (wheel["times"] >= t_start) & (wheel["times"] <= t_end)
        wheel_t = wheel["times"][mask_wheel]
        wheel_pos = wheel["position"][mask_wheel]
    else:
        wheel_t = np.array([])
        wheel_pos = np.array([])
    # Average PSTH by region across the selected time window
    bin_size = config_plot.get("POP_BIN_SIZE", 0.005)
    smooth_window_s = 0.05
    smooth_bins = max(1, int(round(smooth_window_s / bin_size))) if bin_size > 0 else 1
    fano_window_s = 0.1
    fano_step_s = 0.025
    if t_end > t_start and len(df_units_psth) > 0:
        bins = np.arange(t_start, t_end + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        for acronym in region_order:
            region_ids = df_units_psth.loc[
                df_units_psth["acronym"] == acronym, "cluster_id"
            ].values
            if len(region_ids) == 0:
                continue
            region_mask = np.isin(window_spike_clusters_all, region_ids)
            region_spike_times = window_spike_times_all[region_mask]
            counts, _ = np.histogram(region_spike_times, bins=bins)
            rate = counts / (len(region_ids) * bin_size)
            rate_smoothed = _moving_mean(rate, smooth_bins)
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=rate_smoothed,
                    mode="lines",
                    line=dict(color=region_colors.get(acronym)),
                    name=f"{acronym} PSTH",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            var_x, var_y = _compute_variability_curve(
                bin_centers,
                rate,
                window_s=fano_window_s,
                step_s=fano_step_s,
                metric=metric,
            )
            if var_x is not None and var_y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=var_x,
                        y=var_y,
                        mode="lines",
                        line=dict(color=region_colors.get(acronym)),
                        name=f"{acronym} {metric_title}",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

    fig.add_trace(
        go.Scatter(x=wheel_t, y=wheel_pos, mode="lines", line=dict(color=base_color)),
        row=4,
        col=1,
    )

    pose_t = None
    paw_speed = None
    if pose is not None and "leftCamera" in pose:
        pose_df = pose["leftCamera"]
        if "times" in pose_df.columns:
            pose_timestamps = pose_df["times"].values
        else:
            pose_timestamps = pose_df.index.values
        mask_pose = (pose_timestamps >= t_start) & (pose_timestamps <= t_end)
        pose_t = pose_timestamps[mask_pose]
        paw_key = "paw_r" if "paw_r_x" in pose_df.columns else "paw_l"
        if f"{paw_key}_x" in pose_df.columns and pose_t.size >= 2:
            x_vals = pose_df[f"{paw_key}_x"].values[mask_pose]
            y_vals = pose_df[f"{paw_key}_y"].values[mask_pose]
            if x_vals.size >= 2 and y_vals.size >= 2:
                dx = np.gradient(x_vals)
                dy = np.gradient(y_vals)
                dt = np.gradient(pose_t)
                dt[dt == 0] = np.nan
                speed_raw = np.sqrt(dx**2 + dy**2) / dt
                paw_speed = (
                    pd.Series(speed_raw)
                    .fillna(0)
                    .rolling(window=5, center=True)
                    .mean()
                    .values
                )

    if paw_speed is not None:
        fig.add_trace(
            go.Scatter(x=pose_t, y=paw_speed, mode="lines", line=dict(color=base_color)),
            row=5,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y5",
            text="Paw data not available",
            showarrow=False,
            row=5,
            col=1,
        )

    left_t, left_me = _extract_motion_energy_series(
        motion_energy, "leftCamera", t_start, t_end, t_offset=0.0
    )
    right_t, right_me = _extract_motion_energy_series(
        motion_energy, "rightCamera", t_start, t_end, t_offset=0.0
    )
    if left_t.size > 0 or right_t.size > 0:
        if left_t.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=left_t,
                    y=left_me,
                    mode="lines",
                    line=dict(color="#1f77b4"),
                    name="Motion L",
                    showlegend=True,
                ),
                row=6,
                col=1,
            )
        if right_t.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=right_t,
                    y=right_me,
                    mode="lines",
                    line=dict(color="#ff7f0e"),
                    name="Motion R",
                    showlegend=True,
                ),
                row=6,
                col=1,
            )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y6",
            text="Motion energy not available",
            showarrow=False,
            row=6,
            col=1,
        )

    event_style_map = {
        "stimOn_times": ("Stim On", "blue", None),
        "firstMovement_times": ("First Move", "green", None),
        "response_times": ("Response", "purple", None),
        "feedback_times": ("Feedback", "red", None),
    }
    for event_name, (label, color, dash) in event_style_map.items():
        if event_name not in trials.keys():
            continue
        event_times = np.asarray(trials[event_name])
        _add_event_vlines(
            fig,
            event_times,
            label,
            color,
            t_start,
            t_end,
            n_rows=6,
            dash=dash,
        )

    if extra_event_times:
        styles = extra_event_styles or {}
        if isinstance(extra_event_times, dict):
            items = list(extra_event_times.items())
        else:
            items = list(extra_event_times)
        for key, times in items:
            style = styles.get(key)
            label = str(key)
            color = "#666666"
            dash = None
            if isinstance(style, dict):
                label = style.get("label", label)
                color = style.get("color", color)
                dash = style.get("dash", dash)
            elif isinstance(style, (list, tuple)):
                if len(style) > 0 and style[0] is not None:
                    label = style[0]
                if len(style) > 1 and style[1] is not None:
                    color = style[1]
                if len(style) > 2 and style[2] is not None:
                    dash = style[2]
            _add_event_vlines(
                fig,
                times,
                label,
                color,
                t_start,
                t_end,
                n_rows=6,
                dash=dash,
            )

    ylabel_text = (
        f"Good Units (n={len(df_units)})"
        if config_plot["PLOT_ONLY_GOOD_UNITS"]
        else f"All Units (n={len(df_units)})"
    )

    fig.update_yaxes(
        title_text=ylabel_text,
        row=1,
        col=1,
        showticklabels=False,
        range=[-0.5, len(df_units) - 0.5],
    )
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=2, col=1)
    fig.update_yaxes(title_text=metric_title, row=3, col=1)
    fig.update_yaxes(title_text="Wheel (rad)", row=4, col=1)
    fig.update_yaxes(title_text="Paw (px/s)", row=5, col=1)
    fig.update_yaxes(title_text="Motion energy", row=6, col=1)
    fig.update_xaxes(title_text="Time in session (s)", row=6, col=1)

    fig.update_layout(
        title=f"Window {t_start:.2f}-{t_end:.2f}s | Sort: {sort_label}",
        height=1120,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    fig.update_xaxes(range=[t_start, t_end], row=1, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=2, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=3, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=4, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=5, col=1)

    return fig


def plot_population_sorted_plotly(
    sl,
    spikes,
    clusters,
    cluster_ids,
    cluster_acronyms,
    df_res,
    config_plot,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
    region_acronyms=None,
    sort_mode="delay",
):
    """Plot population heatmaps sorted by delay or coupling."""
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    template, base_color = _white_theme()
    df_units, _ = _prepare_units_df(
        cluster_ids,
        cluster_acronyms,
        clusters,
        config_plot.get("PLOT_ONLY_GOOD_UNITS", False),
        label_min=config_plot.get("PLOT_LABEL_MIN"),
    )
    if df_units.empty:
        return go.Figure()
    cluster_ids = df_units["cluster_id"].to_numpy()
    cluster_acronyms = df_units["acronym"].to_numpy()
    region_from_config = region_acronyms is not None
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS")
        if region_acronyms is None:
            region_acronyms = sorted(pd.Series(cluster_acronyms).astype(str).unique().tolist())
    elif isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]

    if len(region_acronyms) == 0:
        return go.Figure()

    window_pre = config_plot["POP_WINDOW_PRE"]
    window_post = config_plot["POP_WINDOW_POST"]
    bin_size = config_plot["POP_BIN_SIZE"]
    smooth_sigma = config_plot["POP_SMOOTH_SIGMA"]
    cmap_name = _normalize_colorscale(config_plot["POP_CMAP_NAME"])
    normalize = config_plot["POP_NORMALIZE"]

    trials = _get_session_field(sl, "trials")
    if trials is None:
        return go.Figure()

    align_event = config_plot.get("PLOT_EVENT", "stimOn_times")
    if align_event not in trials.keys():
        align_event = "stimOn_times"
    delay_col = delay_column_name(align_event)
    delay_odd_col = delay_split_column_name(align_event, "odd")
    delay_even_col = delay_split_column_name(align_event, "even")
    event_series = np.asarray(trials[align_event])
    stim_times = event_series[~np.isnan(event_series)]

    n_rows = len(region_acronyms)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{region} units" for region in region_acronyms],
    )
    use_prefix = bool(config_plot.get("PLOT_REGION_PREFIX_MATCH", False)) and region_from_config

    def _select_delay_sort_key(df_local):
        if "delay_odd" in df_local.columns and np.isfinite(df_local["delay_odd"]).any():
            return "delay_odd", "Delay (Odd)"
        if "delay" in df_local.columns and np.isfinite(df_local["delay"]).any():
            return "delay", "Delay"
        if "delay_even" in df_local.columns and np.isfinite(df_local["delay_even"]).any():
            return "delay_even", "Delay (Even)"
        return "delay", "Delay"

    def _sort_by_coupling_value(df_local, df_src, value_cols, label, ascending=True):
        if df_src is None:
            return None
        for col in value_cols:
            if col in df_src.columns:
                merged = df_local.merge(
                    df_src[["cluster_id", col]].rename(columns={col: "sort_metric"}),
                    on="cluster_id",
                    how="left",
                )
                return (
                    merged.sort_values(by="sort_metric", ascending=ascending, na_position="last"),
                    label,
                )
        return None

    for row_idx, region in enumerate(region_acronyms, start=1):
        if use_prefix:
            region_mask = np.char.startswith(cluster_acronyms.astype(str), region)
        else:
            region_mask = cluster_acronyms.astype(str) == str(region)
        df_region = pd.DataFrame({"cluster_id": cluster_ids[region_mask]})
        sort_desc = sort_mode not in ("depth", "default")
        sort_ascending = not sort_desc
        merge_cols = ["cluster_id"]
        rename_map = {}
        if df_res is not None:
            if delay_col in df_res.columns:
                merge_cols.append(delay_col)
                rename_map[delay_col] = "delay"
            if delay_odd_col in df_res.columns:
                merge_cols.append(delay_odd_col)
                rename_map[delay_odd_col] = "delay_odd"
            if delay_even_col in df_res.columns:
                merge_cols.append(delay_even_col)
                rename_map[delay_even_col] = "delay_even"
        if len(merge_cols) > 1:
            df_region = df_region.merge(df_res[merge_cols], on="cluster_id", how="left")
            df_region = df_region.rename(columns=rename_map)
        for col_name in ("delay", "delay_odd", "delay_even"):
            if col_name not in df_region.columns:
                df_region[col_name] = np.nan

        delay_sort_key, delay_sort_label = _select_delay_sort_key(df_region)

        if sort_mode == "spont" and df_coupling is not None:
            if "sorting_number" in df_coupling.columns:
                df_region = df_region.merge(
                    df_coupling[["cluster_id", "sorting_number"]],
                    on="cluster_id",
                    how="left",
                )
                df_sorted = df_region.sort_values(
                    by="sorting_number", ascending=sort_ascending, na_position="last"
                )
                sort_label = "Coupling (Spont)"
            else:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
        elif sort_mode == "task" and df_coupling_task is not None:
            if "sorting_number" in df_coupling_task.columns:
                df_region = df_region.merge(
                    df_coupling_task[["cluster_id", "sorting_number"]],
                    on="cluster_id",
                    how="left",
                )
                df_sorted = df_region.sort_values(
                    by="sorting_number", ascending=sort_ascending, na_position="last"
                )
                sort_label = "Coupling (Task)"
            else:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
        elif sort_mode == "iti" and df_coupling_iti is not None:
            if "sorting_number" in df_coupling_iti.columns:
                df_region = df_region.merge(
                    df_coupling_iti[["cluster_id", "sorting_number"]],
                    on="cluster_id",
                    how="left",
                )
                df_sorted = df_region.sort_values(
                    by="sorting_number", ascending=sort_ascending, na_position="last"
                )
                sort_label = "Coupling (ITI)"
            else:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
        elif sort_mode == "spont_strength":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling,
                ["coupling_strength", "coupling_strength_h1", "coupling_strength_h2"],
                "stPR Strength (Spont)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "spont_max":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling,
                ["coupling_max", "coupling_max_h1", "coupling_max_h2"],
                "stPR Max (Spont)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "task_strength":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling_task,
                ["coupling_strength", "coupling_strength_odd", "coupling_strength_even"],
                "stPR Strength (Task)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "task_max":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling_task,
                ["coupling_max", "coupling_max_odd", "coupling_max_even"],
                "stPR Max (Task)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "iti_strength":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling_iti,
                ["coupling_strength", "coupling_strength_odd", "coupling_strength_even"],
                "stPR Strength (ITI)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "iti_max":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_coupling_iti,
                ["coupling_max", "coupling_max_odd", "coupling_max_even"],
                "stPR Max (ITI)",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "firing_rate":
            sorted_result = _sort_by_coupling_value(
                df_region,
                df_firing_rate,
                ["firing_rate", "firing_rate_h1", "firing_rate_h2"],
                "Firing rate",
                ascending=sort_ascending,
            )
            if sorted_result is None:
                df_sorted = df_region.sort_values(
                    by=delay_sort_key, ascending=sort_ascending, na_position="last"
                )
                sort_label = delay_sort_label
            else:
                df_sorted, sort_label = sorted_result
        elif sort_mode == "depth":
            all_cluster_ids = _get_cluster_attr(clusters, "cluster_id", None)
            if all_cluster_ids is None:
                all_cluster_ids = np.arange(len(_get_depths(clusters, len(cluster_ids))))
            depths = _get_depths(clusters, len(all_cluster_ids))
            df_depth = pd.DataFrame({"cluster_id": np.asarray(all_cluster_ids), "depth": depths})
            df_region = df_region.merge(df_depth, on="cluster_id", how="left")
            df_sorted = df_region.sort_values(by="depth", ascending=True, na_position="last")
            sort_label = "Depth"
        else:
            df_sorted = df_region.sort_values(
                by=delay_sort_key, ascending=sort_ascending, na_position="last"
            )
            sort_label = delay_sort_label

        df_sorted = df_sorted.reset_index(drop=True)
        n_neurons = len(df_sorted)
        if n_neurons == 0 or len(stim_times) == 0:
            continue

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            df_sorted["cluster_id"].values,
            stim_times,
            -window_pre,
            window_post,
            bin_size,
            smooth_sigma,
            show_progress=False,
        )

        n_bins = len(bin_centers) if bin_centers is not None else 0
        psth_matrix = np.zeros((n_neurons, n_bins))
        for i, row in df_sorted.iterrows():
            cid = row["cluster_id"]
            psth_entry = psth_by_cluster.get(cid)
            if not psth_entry:
                continue
            fr_smooth = psth_entry["fr_smooth"]
            if normalize and fr_smooth.size > 0:
                peak = np.max(fr_smooth)
                if peak > 0:
                    fr_smooth = fr_smooth / peak
            psth_matrix[i, :] = fr_smooth

        def _scale_delay_for_plot(values):
            delay_s = _delay_to_seconds(values, config_plot)
            if bin_centers is None:
                return delay_s
            delay_s = np.asarray(delay_s, dtype=float)
            if delay_s.size == 0 or not np.isfinite(delay_s).any():
                return delay_s
            max_abs = np.nanmax(np.abs(bin_centers))
            if np.isfinite(max_abs) and max_abs > 0:
                median_abs = np.nanmedian(np.abs(delay_s))
                if np.isfinite(median_abs) and median_abs > max_abs * 10:
                    return delay_s / 1000.0
            return delay_s

        df_sorted["delay_plot"] = _scale_delay_for_plot(df_sorted["delay"])
        df_sorted["delay_odd_plot"] = _scale_delay_for_plot(df_sorted["delay_odd"])
        df_sorted["delay_even_plot"] = _scale_delay_for_plot(df_sorted["delay_even"])

        show_scale = row_idx == 1
        fig.add_trace(
            go.Heatmap(
                z=psth_matrix,
                x=bin_centers,
                y=np.arange(n_neurons),
                colorscale=cmap_name,
                colorbar=dict(
                    title="Norm FR" if normalize else "FR",
                    len=0.7,
                    y=0.5,
                    yanchor="middle",
                ),
                showscale=show_scale,
            ),
            row=row_idx,
            col=1,
        )

        odd_available = np.isfinite(df_sorted["delay_odd_plot"]).any()
        even_available = np.isfinite(df_sorted["delay_even_plot"]).any()
        if odd_available or even_available:
            if odd_available:
                valid_odd = df_sorted.dropna(subset=["delay_odd_plot"])
                fig.add_trace(
                    go.Scatter(
                        x=valid_odd["delay_odd_plot"],
                        y=valid_odd.index,
                        mode="markers",
                        marker=dict(color="gray", size=5),
                        name="Odd delay",
                        legendgroup="delay_odd",
                        showlegend=row_idx == 1,
                    ),
                    row=row_idx,
                    col=1,
                )
            if even_available:
                valid_even = df_sorted.dropna(subset=["delay_even_plot"])
                fig.add_trace(
                    go.Scatter(
                        x=valid_even["delay_even_plot"],
                        y=valid_even.index,
                        mode="markers",
                        marker=dict(color="black", size=5),
                        name="Even delay",
                        legendgroup="delay_even",
                        showlegend=row_idx == 1,
                    ),
                    row=row_idx,
                    col=1,
                )
        else:
            valid_delays = df_sorted.dropna(subset=["delay_plot"])
            fig.add_trace(
                go.Scatter(
                    x=valid_delays["delay_plot"],
                    y=valid_delays.index,
                    mode="markers",
                    marker=dict(color=base_color, size=5),
                    name="Delay",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=row_idx, col=1)
        fig.update_yaxes(
            title_text=f"Neurons (Sorted by {sort_label})", row=row_idx, col=1, autorange="reversed"
        )

    for ann in fig.layout.annotations or []:
        ann.update(font=dict(size=12))

    fig.update_layout(
        title=f"Population PSTH Heatmaps | Align: {event_label(align_event)}",
        height=max(450, 280 * n_rows + 140),
        width=1000,
        margin=dict(l=70, r=70, t=90, b=70),
    )
    fig.update_layout(template=template, font=dict(color=base_color, size=12))
    fig.update_xaxes(
        title_text=f"Time from {event_label(align_event)} (s)",
        row=n_rows,
        col=1,
    )

    return fig


def plot_population_coupling_heatmap_plotly(
    df_coupling,
    config_plot,
    config_calc,
    region_acronyms=None,
    coupling_strength_thr=np.nan,
    zscore_by_region=False,
    colorbar_mode="single",
    colorbar_side="right",
    zmin=None,
    zmax=None,
    zmid=None,
):
    """Plot spike-triggered population coupling heatmaps sorted by coupling delay."""
    if df_coupling is None or len(df_coupling) == 0:
        return go.Figure()

    df_coupling = df_coupling[~df_coupling["region"].isin(["root", "void"])]

    region_from_config = region_acronyms is not None
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS")
        if region_acronyms is None:
            region_acronyms = sorted(
                pd.Series(df_coupling["region"]).astype(str).unique().tolist()
            )
    elif isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]

    if len(region_acronyms) == 0:
        return go.Figure()

    template, base_color = _white_theme()
    cmap_name = _normalize_colorscale(config_plot["POP_CMAP_NAME"])
    bin_size_ms = config_calc.get("STPR_BIN_SIZE", 0.001) * 1000
    window_ms = config_calc.get("STPR_WINDOW_MS", 80)
    window_bins = int(round(window_ms / bin_size_ms)) if bin_size_ms > 0 else 0
    lags_ms = np.arange(-window_bins, window_bins + 1) * bin_size_ms
    n_bins = len(lags_ms)

    n_rows = len(region_acronyms)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{region} units" for region in region_acronyms],
    )
    use_prefix = bool(config_plot.get("PLOT_REGION_PREFIX_MATCH", False)) and region_from_config
    has_split_curves = (
        "stpr_curve_h1" in df_coupling.columns and "stpr_curve_h2" in df_coupling.columns
    )

    def _normalize_curve(curve_array):
        if curve_array.size == 0:
            return curve_array
        curve_mean = np.nanmean(curve_array)
        curve_std = np.nanstd(curve_array)
        if curve_std > 0:
            return (curve_array - curve_mean) / curve_std
        return curve_array

    def _extract_curve(row):
        if has_split_curves:
            curve_h1 = np.asarray(row.get("stpr_curve_h1", []), dtype=float)
            curve_h2 = np.asarray(row.get("stpr_curve_h2", []), dtype=float)
            if zscore_by_region:
                curve_h1 = _normalize_curve(curve_h1)
                curve_h2 = _normalize_curve(curve_h2)
            if curve_h1.size > 0 and curve_h2.size > 0:
                min_len = min(curve_h1.size, curve_h2.size)
                return (curve_h1[:min_len] + curve_h2[:min_len]) / 2
            if curve_h1.size > 0:
                return curve_h1
            if curve_h2.size > 0:
                return curve_h2
            return np.array([])
        curve = np.asarray(row.get("stpr_curve", []), dtype=float)
        if zscore_by_region:
            curve = _normalize_curve(curve)
        return curve

    for row_idx, region in enumerate(region_acronyms, start=1):
        if use_prefix:
            region_mask = df_coupling["region"].astype(str).str.startswith(region)
        else:
            region_mask = df_coupling["region"].astype(str) == str(region)
        df_region = df_coupling.loc[region_mask].copy()
        if pd.notna(coupling_strength_thr):
            df_region = df_region.loc[df_region["coupling_strength"] > coupling_strength_thr]

        df_sorted = df_region.sort_values(by="sorting_number", ascending=True, na_position="last")
        df_sorted = df_sorted.reset_index(drop=True)

        n_neurons = len(df_sorted)
        if n_neurons == 0:
            continue

        stpr_matrix = np.full((n_neurons, n_bins), np.nan)

        for row_i, row in df_sorted.iterrows():
            curve = _extract_curve(row)
            if curve.size == 0:
                continue

            if curve.size == n_bins:
                stpr_matrix[row_i, :] = curve
            elif curve.size < n_bins:
                start_idx = int((n_bins - curve.size) // 2)
                end_idx = start_idx + curve.size
                stpr_matrix[row_i, start_idx:end_idx] = curve
            else:
                trim_start = int((curve.size - n_bins) // 2)
                stpr_matrix[row_i, :] = curve[trim_start : trim_start + n_bins]

        zmin_val = zmin
        zmax_val = zmax
        zmid_val = zmid

        show_scale = False
        if colorbar_mode == "per_row":
            show_scale = True
        elif colorbar_mode == "single":
            show_scale = row_idx == 1

        colorbar_title = "stPR (z-score)" if zscore_by_region else "stPR"
        colorbar = dict(
            title=colorbar_title,
            len=0.7,
            y=0.5,
            yanchor="middle",
        )
        if colorbar_mode == "per_row":
            yaxis_name = "yaxis" if row_idx == 1 else f"yaxis{row_idx}"
            yaxis = getattr(fig.layout, yaxis_name, None)
            if yaxis is not None and getattr(yaxis, "domain", None):
                domain = yaxis.domain
                colorbar["len"] = float(domain[1] - domain[0])
                colorbar["y"] = float((domain[0] + domain[1]) / 2.0)
            if colorbar_side == "left":
                colorbar["x"] = -0.08
                colorbar["xanchor"] = "right"
            else:
                colorbar["x"] = 1.02
                colorbar["xanchor"] = "left"
        elif colorbar_side == "left":
            colorbar["x"] = -0.08
            colorbar["xanchor"] = "right"
        fig.add_trace(
            go.Heatmap(
                z=stpr_matrix,
                x=lags_ms,
                y=np.arange(n_neurons),
                colorscale=cmap_name,
                zmin=zmin_val,
                zmax=zmax_val,
                zmid=zmid_val,
                colorbar=colorbar,
                showscale=show_scale,
            ),
            row=row_idx,
            col=1,
        )

        split_specs = [
            ("h1", "h2", "First half", "Second half"),
            ("odd", "even", "Odd trials", "Even trials"),
            ("true", "false", "True trials", "False trials"),
        ]
        split_plotted = False
        for split_a, split_b, label_a, label_b in split_specs:
            col_a = f"coupling_delay_ms_{split_a}"
            col_b = f"coupling_delay_ms_{split_b}"
            if col_a in df_sorted.columns and col_b in df_sorted.columns:
                show_legend = row_idx == 1
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[col_a],
                        y=df_sorted.index,
                        mode="markers",
                        marker=dict(
                            color="gray" if base_color == "black" else "lightgray", size=4
                        ),
                        name=label_a,
                        legendgroup=f"delay_{split_a}",
                        showlegend=show_legend,
                    ),
                    row=row_idx,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[col_b],
                        y=df_sorted.index,
                        mode="markers",
                        marker=dict(color=base_color, size=4),
                        name=label_b,
                        legendgroup=f"delay_{split_b}",
                        showlegend=show_legend,
                    ),
                    row=row_idx,
                    col=1,
                )
                split_plotted = True
                break
        if not split_plotted and "coupling_delay_ms" in df_sorted.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["coupling_delay_ms"],
                    y=df_sorted.index,
                    mode="markers",
                    marker=dict(color=base_color, size=4),
                    name="Delay",
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )

        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=row_idx, col=1)
        fig.update_yaxes(title_text="Neurons", row=row_idx, col=1, autorange="reversed")

    for ann in fig.layout.annotations or []:
        ann.update(font=dict(size=12))

    fig.update_layout(
        title="Spike-triggered Population Coupling (stPR)",
        height=max(450, 280 * n_rows + 140),
        width=1000,
        margin=dict(
            l=110 if colorbar_mode == "per_row" and colorbar_side == "left" else 70,
            r=70,
            t=90,
            b=70,
        ),
    )
    fig.update_layout(template=template, font=dict(color=base_color, size=12))
    fig.update_xaxes(
        title_text="Lag (ms)",
        range=[-window_ms, window_ms],
        row=n_rows,
        col=1,
    )

    return fig


def _corr_stat(res):
    return getattr(res, "statistic", res[0])


def _compute_corrs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, int(mask.sum())
    rp = _corr_stat(pearsonr(x[mask], y[mask]))
    rs = _corr_stat(spearmanr(x[mask], y[mask]))
    return float(rp), float(rs), int(mask.sum())


def _scatter_with_unity_plotly(
    df,
    xcol,
    ycol,
    xlabel,
    ylabel,
    title,
    region_order=None,
    region_colors=None,
    template=None,
    highlight_cluster_id=None,
    other_alpha=0.4,
):
    if region_colors is None and "region" in df.columns:
        region_colors = _region_color_map(
            pd.Series(df["region"]).astype(str).unique().tolist()
        )
    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color="region",
        category_orders={"region": region_order} if region_order is not None else None,
        color_discrete_map=region_colors,
    )
    if highlight_cluster_id is not None:
        for trace in fig.data:
            if getattr(trace, "mode", "") and "markers" in trace.mode:
                trace.update(marker=dict(opacity=other_alpha, size=7))
    min_val = np.nanmin([df[xcol].min(), df[ycol].min()])
    max_val = np.nanmax([df[xcol].max(), df[ycol].max()])
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Unity",
        )
    )
    if template is None:
        template, base_color = _white_theme()
    else:
        base_color = _template_base_color(template)
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template=template,
        font=dict(color=base_color, size=13),
        width=900,
        height=650,
        margin=dict(l=70, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if highlight_cluster_id is not None and "cluster_id" in df.columns:
        highlight_rows = df[df["cluster_id"] == highlight_cluster_id]
        if not highlight_rows.empty:
            highlight_row = highlight_rows.iloc[0]
            region_val = str(highlight_row.get("region", "NA"))
            highlight_color = None
            if region_colors and region_val in region_colors:
                highlight_color = region_colors[region_val]
            fig.add_trace(
                go.Scatter(
                    x=[highlight_row[xcol]],
                    y=[highlight_row[ycol]],
                    mode="markers",
                    marker=dict(
                        color=highlight_color or "red",
                        size=12,
                        opacity=1.0,
                        line=dict(color="white", width=1),
                    ),
                    name=f"Selected {highlight_cluster_id}",
                    showlegend=False,
                )
            )
    return fig


def plot_coupling_strength_summary_plotly(
    df_comparison,
    region_order=None,
    region_colors=None,
    template=None,
    highlight_cluster_id=None,
):
    df_strength = df_comparison.dropna(
        subset=["coupling_strength_spont", "coupling_strength_task"]
    )
    if len(df_strength) == 0:
        return go.Figure()
    rp, rs, n = _compute_corrs(
        df_strength["coupling_strength_spont"], df_strength["coupling_strength_task"]
    )
    title = (
        f"Coupling Strength: Spont vs Task (Pearson r={rp:.3f}, Spearman rho={rs:.3f}, n={n})"
    )
    return _scatter_with_unity_plotly(
        df_strength,
        "coupling_strength_task",
        "coupling_strength_spont",
        "Task stPR (coupling strength)",
        "Spont stPR (coupling strength)",
        title,
        region_order,
        region_colors,
        template,
        highlight_cluster_id=highlight_cluster_id,
    )


def plot_coupling_delay_summary_plotly(
    df_comparison,
    region_order=None,
    region_colors=None,
    template=None,
    highlight_cluster_id=None,
):
    df_delay = df_comparison.dropna(
        subset=["coupling_delay_ms_spont", "coupling_delay_ms_task"]
    )
    if len(df_delay) == 0:
        return go.Figure()
    rp, rs, n = _compute_corrs(
        df_delay["coupling_delay_ms_spont"], df_delay["coupling_delay_ms_task"]
    )
    title = f"Coupling Delay: Spont vs Task (Pearson r={rp:.3f}, Spearman rho={rs:.3f}, n={n})"
    return _scatter_with_unity_plotly(
        df_delay,
        "coupling_delay_ms_task",
        "coupling_delay_ms_spont",
        "Task Coupling Delay (ms)",
        "Spont Coupling Delay (ms)",
        title,
        region_order,
        region_colors,
        template,
        highlight_cluster_id=highlight_cluster_id,
    )


def plot_coupling_sorting_summary_plotly(
    df_comparison, region_order=None, region_colors=None, template=None
):
    df_sorting = df_comparison.dropna(
        subset=["sorting_number_spont", "sorting_number_task"]
    )
    if len(df_sorting) == 0:
        return go.Figure()
    rp, rs, n = _compute_corrs(
        df_sorting["sorting_number_spont"], df_sorting["sorting_number_task"]
    )
    title = f"Sorting Number: Spont vs Task (Pearson r={rp:.3f}, Spearman rho={rs:.3f}, n={n})"
    return _scatter_with_unity_plotly(
        df_sorting,
        "sorting_number_task",
        "sorting_number_spont",
        "Task Sorting Number",
        "Spont Sorting Number",
        title,
        region_order,
        region_colors,
        template,
    )
