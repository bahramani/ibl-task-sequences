import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from scipy.stats import pearsonr, spearmanr

from .analysis import compute_psth_for_clusters, event_label, delay_column_name


def _get_cluster_attr(clusters, key, fallback=None):
    if clusters is None:
        return fallback
    if hasattr(clusters, key):
        return getattr(clusters, key)
    if isinstance(clusters, dict) and key in clusters:
        return clusters.get(key)
    return fallback


def _get_session_field(sl, key):
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl.get(key)
    return getattr(sl, key, None)


def _normalize_colorscale(cmap_name):
    if not isinstance(cmap_name, str):
        return cmap_name
    cmap = cmap_name.strip().lower()
    if cmap in ("bwr", "rdbu"):
        return "rdbu"
    if cmap == "rdgy":
        return "rdgy"
    return cmap_name


def _white_theme():
    return "plotly_white", "black"


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
    labels = None
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            labels = np.asarray(clusters.metrics.label)
    if labels is None and hasattr(clusters, "label"):
        labels = np.asarray(clusters.label)
    if labels is None and isinstance(clusters, dict) and "label" in clusters:
        labels = np.asarray(clusters.get("label"))
    if labels is None:
        return np.ones(len(cluster_ids), dtype=bool)
    return labels == 1


def _get_depths(clusters, n_units):
    depths = _get_cluster_attr(clusters, "depths", None)
    if depths is None:
        depths = _get_cluster_attr(clusters, "depth", None)
    if depths is None:
        return np.arange(n_units)
    return np.asarray(depths)


def _prepare_units_df(cluster_ids, cluster_acronyms, clusters, only_good):
    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    quality_mask = _get_quality_mask(clusters, cluster_ids, only_good)
    depths = _get_depths(clusters, len(cluster_ids))

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


def _merge_metric(df_units, metric_key, df_res=None, df_coupling=None, df_coupling_task=None):
    metric_key = (metric_key or "depth").strip().lower()
    df_units = df_units.copy()
    sort_label = "Depth"

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

    if "spont" in metric_key:
        sort_label = "Coupling (Spont)"
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
        sort_label = "Coupling (Task)"
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
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    pupil_features=None,
    pupil_times=None,
    region_colors=None,
):
    """Plot a trial-aligned raster using plotly-resampler with intra-region sorting."""
    trials = _get_session_field(sl, "trials")
    wheel = _get_session_field(sl, "wheel")
    pose = _get_session_field(sl, "pose")
    if trials is None:
        return go.Figure()

    template, base_color = _white_theme()

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
        cluster_ids, cluster_acronyms, clusters, config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    if df_units.empty:
        return go.Figure()

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids, cluster_acronyms, clusters, avg_psth_only_good
    )

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids, cluster_acronyms, clusters, avg_psth_only_good
    )

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids, cluster_acronyms, clusters, avg_psth_only_good
    )

    df_units, sort_label = _merge_metric(
        df_units,
        sorting_metric,
        df_res=df_res,
        df_coupling=df_coupling,
        df_coupling_task=df_coupling_task,
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

    fig = FigureResampler(
        make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.55, 0.12, 0.11, 0.11, 0.11],
            subplot_titles=("Raster", "Avg PSTH", "Wheel", "Paw Speed", "Pupil Diameter"),
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
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=rate,
                    mode="lines",
                    line=dict(color=region_colors.get(acronym)),
                    name=f"{acronym} PSTH",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.add_trace(
        go.Scatter(x=wheel_t, y=wheel_pos, mode="lines", line=dict(color=base_color)),
        row=3,
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
            row=4,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y4",
            text="Paw data not available",
            showarrow=False,
            row=4,
            col=1,
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
            pupil_t = pt[mask_pupil] - t_offset
            pupil_diam = pd_vals[mask_pupil]

    if pupil_diam is not None:
        fig.add_trace(
            go.Scatter(x=pupil_t, y=pupil_diam, mode="lines", line=dict(color=base_color)),
            row=5,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y5",
            text="Pupil data not available",
            showarrow=False,
            row=5,
            col=1,
        )

    event_lines = [
        ("Stim On", t_stim_on, "blue"),
        ("First Move", t_first_move, "green"),
        ("Feedback", t_feedback, "red"),
    ]
    for name, time_val, color in event_lines:
        for row in range(1, 6):
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
    fig.update_yaxes(title_text="Wheel (rad)", row=3, col=1)
    fig.update_yaxes(title_text="Paw (px/s)", row=4, col=1)
    fig.update_yaxes(title_text="Pupil (mm)", row=5, col=1)
    fig.update_xaxes(title_text=xlabel_text, row=5, col=1)
    fig.update_xaxes(range=[t_start - t_offset, t_end - t_offset])

    fig.update_layout(
        title=f"{plot_title} | Sort: {sort_label}",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_layout(template=template, font=dict(color=base_color))

    return fig


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
    df_res=None,
    df_coupling=None,
    df_coupling_task=None,
    pupil_features=None,
    pupil_times=None,
    region_colors=None,
):
    """Plot a session-time raster for a specified time window."""
    if t_start >= t_end:
        return go.Figure()

    trials = _get_session_field(sl, "trials")
    wheel = _get_session_field(sl, "wheel")
    pose = _get_session_field(sl, "pose")
    if trials is None:
        return go.Figure()

    template, base_color = _white_theme()

    df_units, _ = _prepare_units_df(
        cluster_ids, cluster_acronyms, clusters, config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    if df_units.empty:
        return go.Figure()

    avg_psth_only_good = config_plot.get(
        "AVG_PSTH_ONLY_GOOD", config_plot["PLOT_ONLY_GOOD_UNITS"]
    )
    df_units_psth, _ = _prepare_units_df(
        cluster_ids, cluster_acronyms, clusters, avg_psth_only_good
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

    fig = FigureResampler(
        make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.55, 0.12, 0.11, 0.11, 0.11],
            subplot_titles=("Raster", "Avg PSTH", "Wheel", "Paw Speed", "Pupil Diameter"),
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
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=rate,
                    mode="lines",
                    line=dict(color=region_colors.get(acronym)),
                    name=f"{acronym} PSTH",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.add_trace(
        go.Scatter(x=wheel_t, y=wheel_pos, mode="lines", line=dict(color=base_color)),
        row=3,
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
            row=4,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y4",
            text="Paw data not available",
            showarrow=False,
            row=4,
            col=1,
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

    if pupil_diam is not None:
        fig.add_trace(
            go.Scatter(x=pupil_t, y=pupil_diam, mode="lines", line=dict(color=base_color)),
            row=5,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y5",
            text="Pupil data not available",
            showarrow=False,
            row=5,
            col=1,
        )

    event_style_map = {
        "stimOn_times": ("Stim On", "blue"),
        "firstMovement_times": ("First Move", "green"),
        "response_times": ("Response", "purple"),
        "feedback_times": ("Feedback", "red"),
    }
    for event_name, (label, color) in event_style_map.items():
        if event_name not in trials.keys():
            continue
        event_times = np.asarray(trials[event_name])
        valid_times = event_times[(event_times >= t_start) & (event_times <= t_end)]
        if len(valid_times) == 0:
            continue
        for row in range(1, 6):
            for t_event in valid_times:
                fig.add_vline(x=t_event, line=dict(color=color, width=1.5), row=row, col=1)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
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
    fig.update_yaxes(title_text="Wheel (rad)", row=3, col=1)
    fig.update_yaxes(title_text="Paw (px/s)", row=4, col=1)
    fig.update_yaxes(title_text="Pupil (mm)", row=5, col=1, showticklabels=False)
    fig.update_xaxes(title_text="Time in session (s)", row=5, col=1)

    fig.update_layout(
        title=f"Window {t_start:.2f}-{t_end:.2f}s | Sort: {sort_label}",
        height=900,
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
    region_acronyms=None,
    sort_mode="delay",
):
    """Plot population heatmaps sorted by delay or coupling."""
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    template, base_color = _white_theme()
    quality_mask = _get_quality_mask(
        clusters, np.asarray(cluster_ids), config_plot.get("PLOT_ONLY_GOOD_UNITS", False)
    )
    cluster_ids = np.asarray(cluster_ids)[quality_mask]
    cluster_acronyms = cluster_acronyms[quality_mask]
    keep_mask = ~np.isin(cluster_acronyms, ["root", "void"])
    cluster_ids = cluster_ids[keep_mask]
    cluster_acronyms = cluster_acronyms[keep_mask]
    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS", ["VISp"])
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
    event_series = np.asarray(trials[align_event])
    stim_times = event_series[~np.isnan(event_series)]

    fig = make_subplots(
        rows=len(region_acronyms),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{region} units" for region in region_acronyms],
    )

    for row_idx, region in enumerate(region_acronyms, start=1):
        region_mask = np.char.startswith(cluster_acronyms.astype(str), region)
        df_region = pd.DataFrame({"cluster_id": cluster_ids[region_mask]})
        if df_res is not None and delay_col in df_res.columns:
            df_region = df_region.merge(
                df_res[["cluster_id", delay_col]].rename(columns={delay_col: "delay"}),
                on="cluster_id",
                how="left",
            )
        else:
            df_region["delay"] = np.nan

        if sort_mode == "spont" and df_coupling is not None:
            if "sorting_number" in df_coupling.columns:
                df_region = df_region.merge(
                    df_coupling[["cluster_id", "sorting_number"]],
                    on="cluster_id",
                    how="left",
                )
                df_sorted = df_region.sort_values(
                    by="sorting_number", ascending=True, na_position="last"
                )
                sort_label = "Coupling (Spont)"
            else:
                df_sorted = df_region.sort_values(by="delay", ascending=True, na_position="last")
                sort_label = "Delay"
        elif sort_mode == "task" and df_coupling_task is not None:
            if "sorting_number" in df_coupling_task.columns:
                df_region = df_region.merge(
                    df_coupling_task[["cluster_id", "sorting_number"]],
                    on="cluster_id",
                    how="left",
                )
                df_sorted = df_region.sort_values(
                    by="sorting_number", ascending=True, na_position="last"
                )
                sort_label = "Coupling (Task)"
            else:
                df_sorted = df_region.sort_values(by="delay", ascending=True, na_position="last")
                sort_label = "Delay"
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
            df_sorted = df_region.sort_values(by="delay", ascending=True, na_position="last")
            sort_label = "Delay"

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

        fig.add_trace(
            go.Heatmap(
                z=psth_matrix,
                x=bin_centers,
                y=np.arange(n_neurons),
                colorscale=cmap_name,
                colorbar=dict(title="Norm FR" if normalize else "FR", len=0.35),
                showscale=True,
            ),
            row=row_idx,
            col=1,
        )

        valid_delays = df_sorted.dropna(subset=["delay"])
        fig.add_trace(
            go.Scatter(
                x=valid_delays["delay"],
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

    fig.update_layout(
        title=f"Population PSTH Heatmaps | Align: {event_label(align_event)}",
        height=350 * len(region_acronyms),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    fig.update_xaxes(title_text=f"Time from {event_label(align_event)} (s)")

    return fig


def plot_population_coupling_heatmap_plotly(
    df_coupling,
    config_plot,
    config_calc,
    region_acronyms=None,
    coupling_strength_thr=np.nan,
):
    """Plot spike-triggered population coupling heatmaps sorted by coupling delay."""
    if df_coupling is None or len(df_coupling) == 0:
        return go.Figure()

    df_coupling = df_coupling[~df_coupling["region"].isin(["root", "void"])]

    if region_acronyms is None:
        region_acronyms = config_plot.get("PLOT_REGIONS", ["VISp"])
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

    fig = make_subplots(
        rows=len(region_acronyms),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{region} units" for region in region_acronyms],
    )

    for row_idx, region in enumerate(region_acronyms, start=1):
        region_mask = df_coupling["region"].astype(str).str.startswith(region)
        df_region = df_coupling.loc[region_mask].copy()
        if pd.notna(coupling_strength_thr):
            df_region = df_region.loc[df_region["coupling_strength"] > coupling_strength_thr]

        df_sorted = df_region.sort_values(by="sorting_number", ascending=True, na_position="last")
        df_sorted = df_sorted.reset_index(drop=True)

        n_neurons = len(df_sorted)
        n_bins = len(lags_ms)
        if n_neurons == 0:
            continue

        stpr_matrix = np.full((n_neurons, n_bins), np.nan)

        def _normalize_curve(curve_array):
            if curve_array.size == 0:
                return curve_array
            mean = np.nanmean(curve_array)
            std = np.nanstd(curve_array)
            if std > 0:
                return (curve_array - mean) / std
            return curve_array

        for row_i, row in df_sorted.iterrows():
            if "stpr_curve_h1" in df_sorted.columns and "stpr_curve_h2" in df_sorted.columns:
                curve_h1 = np.asarray(row.get("stpr_curve_h1", []), dtype=float)
                curve_h2 = np.asarray(row.get("stpr_curve_h2", []), dtype=float)
                curve_h1 = _normalize_curve(curve_h1)
                curve_h2 = _normalize_curve(curve_h2)
                if curve_h1.size > 0 and curve_h2.size > 0:
                    min_len = min(curve_h1.size, curve_h2.size)
                    curve = (curve_h1[:min_len] + curve_h2[:min_len]) / 2
                elif curve_h1.size > 0:
                    curve = curve_h1
                elif curve_h2.size > 0:
                    curve = curve_h2
                else:
                    continue
            else:
                curve = np.asarray(row.get("stpr_curve", []), dtype=float)
                if curve.size == 0:
                    continue
                curve = _normalize_curve(curve)

            if curve.size == n_bins:
                stpr_matrix[row_i, :] = curve
            elif curve.size < n_bins:
                start_idx = int((n_bins - curve.size) // 2)
                end_idx = start_idx + curve.size
                stpr_matrix[row_i, start_idx:end_idx] = curve
            else:
                trim_start = int((curve.size - n_bins) // 2)
                stpr_matrix[row_i, :] = curve[trim_start : trim_start + n_bins]

        fig.add_trace(
            go.Heatmap(
                z=stpr_matrix,
                x=lags_ms,
                y=np.arange(n_neurons),
                colorscale=cmap_name,
                colorbar=dict(title="stPR z", len=0.35),
                showscale=True,
            ),
            row=row_idx,
            col=1,
        )

        if "coupling_delay_ms_h1" in df_sorted.columns and "coupling_delay_ms_h2" in df_sorted.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["coupling_delay_ms_h1"],
                    y=df_sorted.index,
                    mode="markers",
                    marker=dict(color="gray" if base_color == "black" else "lightgray", size=4),
                    name="Delay H1",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["coupling_delay_ms_h2"],
                    y=df_sorted.index,
                    mode="markers",
                    marker=dict(color=base_color, size=4),
                    name="Delay H2",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
        elif "coupling_delay_ms" in df_sorted.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["coupling_delay_ms"],
                    y=df_sorted.index,
                    mode="markers",
                    marker=dict(color=base_color, size=4),
                    name="Delay",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        fig.add_vline(x=0, line=dict(color="black", dash="dash"), row=row_idx, col=1)
        fig.update_yaxes(title_text="Neurons", row=row_idx, col=1, autorange="reversed")

    fig.update_layout(
        title="Spike-triggered Population Coupling (stPR)",
        height=350 * len(region_acronyms),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    fig.update_layout(template=template, font=dict(color=base_color))
    fig.update_xaxes(title_text="Lag (ms)", range=[-window_ms, window_ms])

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
):
    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color="region",
        category_orders={"region": region_order} if region_order is not None else None,
        color_discrete_map=region_colors,
    )
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
        base_color = "black"
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template=template,
        font=dict(color=base_color),
    )
    return fig


def plot_coupling_strength_summary_plotly(
    df_comparison, region_order=None, region_colors=None, template=None
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
    )


def plot_coupling_delay_summary_plotly(
    df_comparison, region_order=None, region_colors=None, template=None
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
