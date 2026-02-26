# %% Imports

from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))  # if notebook is in /notebooks/

import utils.analysis as ana_utils
import utils.plotting_plotly as plotting_utils
from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    build_cluster_id_map,
    get_cluster_labels_array,
)
from utils.plotting_plotly import (
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
)

# import sys, pathlib
# src = pathlib.Path(r"C:\Users\Experiment\Documents\Amirreza\SeqProject2026\external\latenzy_src\python")
# sys.path.insert(0, str(src))

# try:
#     from latenzy import latenzy
# except Exception:
#     from latenZy import latenzy  # fallback name in some revisions

# print("latenzy import OK")

# %% Helpers
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"


def _in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _set_plotly_renderer(preferred=None):
    if preferred:
        pio.renderers.default = preferred
        return
    if _in_notebook():
        try:
            import nbformat  # noqa: F401

            pio.renderers.default = "notebook_connected"
            return
        except Exception:
            pass
    pio.renderers.default = "browser"


def show_fig(fig, renderer=None):
    if renderer:
        pio.renderers.default = renderer
    try:
        fig.show()
    except ValueError as exc:
        if "nbformat" in str(exc).lower():
            pio.renderers.default = "browser"
            fig.show()
        else:
            raise


def list_cached_pids(cache_dir=CACHE_DIR):
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.pkl")])


def choose_pid(pid_list, default_index=0):
    if not pid_list:
        raise RuntimeError("No PIDs found in data/dashboard_cache.")
    print("Available PIDs:")
    for idx, pid_val in enumerate(pid_list):
        print(f"  [{idx}] {pid_val}")
    prompt = f"Enter PID or index (press Enter for {pid_list[default_index]}): "
    selection = input(prompt).strip()
    if selection == "":
        return pid_list[default_index]
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(pid_list):
            return pid_list[idx]
        raise ValueError(f"Index {idx} out of range.")
    if selection in pid_list:
        return selection
    raise ValueError(f"PID '{selection}' not found.")


def _init_one_with_fallback(ibl_cache, preferred_mode="local", allow_remote=True):
    modes = []
    if preferred_mode:
        modes.append(preferred_mode)
    if "local" not in modes:
        modes.append("local")
    if allow_remote and "remote" not in modes:
        modes.append("remote")
    last_error = None
    for mode in modes:
        try:
            one = init_one(ibl_cache, mode=mode)
            return one, mode
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize ONE. Last error: {last_error}")


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


def _get_wheel_field(wheel, key):
    if wheel is None:
        return None
    try:
        if hasattr(wheel, "keys") and key in wheel.keys():
            return np.asarray(wheel[key], dtype=float).reshape(-1)
    except Exception:
        pass
    if isinstance(wheel, dict) and key in wheel:
        try:
            return np.asarray(wheel.get(key), dtype=float).reshape(-1)
        except Exception:
            return None
    if hasattr(wheel, key):
        try:
            return np.asarray(getattr(wheel, key), dtype=float).reshape(-1)
        except Exception:
            return None
    return None


def _extract_wheel_speed_cm_s(wheel, wheel_radius_cm=3.1):
    """
    Return wheel speed in cm/s.

    Assumes wheel position/velocity from ONE are in radians or radians/s and
    converts using ``wheel_radius_cm``.
    """
    times = _get_wheel_field(wheel, "times")
    if times is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    vel = _get_wheel_field(wheel, "velocity")
    if vel is not None and vel.shape[0] == times.shape[0]:
        mask = np.isfinite(times) & np.isfinite(vel)
        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=float)
        t = np.asarray(times[mask], dtype=float)
        speed_cm_s = np.abs(np.asarray(vel[mask], dtype=float)) * float(wheel_radius_cm)
        order = np.argsort(t)
        return t[order], speed_cm_s[order]

    pos = _get_wheel_field(wheel, "position")
    if pos is None or pos.shape[0] != times.shape[0]:
        return np.array([], dtype=float), np.array([], dtype=float)

    mask = np.isfinite(times) & np.isfinite(pos)
    if int(mask.sum()) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    t = np.asarray(times[mask], dtype=float)
    p = np.asarray(pos[mask], dtype=float)
    order = np.argsort(t)
    t = t[order]
    p = p[order]
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    p = p[uniq_idx]
    t = uniq_t
    if t.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    vel_rad_s = np.gradient(p, t)
    speed_cm_s = np.abs(vel_rad_s) * float(wheel_radius_cm)
    keep = np.isfinite(t) & np.isfinite(speed_cm_s)
    return t[keep], speed_cm_s[keep]


def _classify_whisk_onsets_by_locomotion(
    onsets_s,
    wheel,
    lookahead_s=6.0,
    speed_thr_cm_s=1.0,
    wheel_radius_cm=3.1,
):
    onsets = np.asarray(onsets_s, dtype=float).reshape(-1)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return {
            "all_onsets": np.array([], dtype=float),
            "loco_flags": np.array([], dtype=bool),
            "loco_onsets": np.array([], dtype=float),
            "non_loco_onsets": np.array([], dtype=float),
            "wheel_times": np.array([], dtype=float),
            "wheel_speed_cm_s": np.array([], dtype=float),
        }

    wheel_t, wheel_speed = _extract_wheel_speed_cm_s(wheel, wheel_radius_cm=wheel_radius_cm)
    flags = np.zeros(onsets.size, dtype=bool)
    if wheel_t.size > 0 and wheel_speed.size > 0:
        for idx, t0 in enumerate(onsets):
            i0 = int(np.searchsorted(wheel_t, t0, side="left"))
            i1 = int(np.searchsorted(wheel_t, t0 + float(lookahead_s), side="right"))
            if i1 > i0:
                max_speed = float(np.nanmax(wheel_speed[i0:i1]))
                flags[idx] = np.isfinite(max_speed) and (max_speed > float(speed_thr_cm_s))

    return {
        "all_onsets": onsets,
        "loco_flags": flags,
        "loco_onsets": onsets[flags],
        "non_loco_onsets": onsets[~flags],
        "wheel_times": wheel_t,
        "wheel_speed_cm_s": wheel_speed,
    }


def _build_whisk_trace(df_trace, config):
    columns = [
        "bin_idx",
        "bin_start_s",
        "bin_center_s",
        "bin_end_s",
        "wh_norm",
        "n_views",
    ]
    target_wh_mean = config.get("WH_TARGET_MEAN", None)
    wh_signal_scale = 1.0
    use_binning = bool(config.get("WH_BIN_SIGNAL", True))
    if use_binning:
        out = ana_utils.bin_normalize_whisk_trace(
            df_trace,
            bin_s=config["WH_BIN_S"],
            norm_pctl=config["WH_NORM_PCTL"],
            norm_top_pctl=config.get("WH_NORM_TOP_PCTL", 100.0),
            bin_reduce=config.get("WH_BIN_REDUCE", "mean"),
            normalize_after_bin=config.get("WH_NORMALIZE_AFTER_BIN", False),
        )
        # Rescale whisk signal right at creation so downstream calculations use the new mean level.
        if target_wh_mean is not None and not out.empty:
            wh_vals = out["wh_norm"].to_numpy(dtype=float)
            curr_mean = float(np.nanmean(wh_vals))
            if np.isfinite(curr_mean) and not np.isclose(curr_mean, 0.0):
                wh_signal_scale = float(target_wh_mean) / curr_mean
                out = out.copy()
                out["wh_norm"] = wh_vals * wh_signal_scale
        if isinstance(config, dict):
            # Reuse this same scale for left/right camera traces in plotting.
            config["WH_SIGNAL_SCALE"] = wh_signal_scale
        return out

    if df_trace is None or len(df_trace) == 0:
        return pd.DataFrame(columns=columns)

    df = df_trace.copy()
    if not {"times", "view", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=columns)
    df = df.dropna(subset=["times", "value"]).copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["times"] = df["times"].astype(float)
    df["value"] = df["value"].astype(float)
    df["view"] = df["view"].astype(str)

    norm_pctl = float(config.get("WH_NORM_PCTL", 1.0))
    norm_top_pctl = float(config.get("WH_NORM_TOP_PCTL", 100.0))
    if norm_pctl < 0 or norm_pctl > 100:
        raise ValueError("WH_NORM_PCTL must be in [0, 100].")
    if norm_top_pctl < 0 or norm_top_pctl > 100:
        raise ValueError("WH_NORM_TOP_PCTL must be in [0, 100].")
    if norm_top_pctl <= norm_pctl:
        raise ValueError("WH_NORM_TOP_PCTL must be > WH_NORM_PCTL.")

    def _normalize_values(vals_fin):
        lo = float(np.nanpercentile(vals_fin, norm_pctl))
        hi = float(np.nanpercentile(vals_fin, norm_top_pctl))
        if not np.isfinite(lo):
            lo = float(np.nanmin(vals_fin))
        if not np.isfinite(hi):
            hi = float(np.nanmax(vals_fin))
        if hi <= lo:
            hi = float(np.nanmax(vals_fin))
        if hi <= lo:
            return np.zeros_like(vals_fin, dtype=float)
        return np.clip((vals_fin - lo) / (hi - lo), 0.0, 1.0)

    by_view = {}
    for view, grp_view in df.groupby("view", sort=False):
        vals = grp_view["value"].to_numpy(dtype=float)
        times = grp_view["times"].to_numpy(dtype=float)
        finite = np.isfinite(times) & np.isfinite(vals)
        if not np.any(finite):
            continue
        vals_fin = vals[finite]
        t_fin = times[finite]
        norm = _normalize_values(vals_fin)
        tmp = pd.DataFrame({"times": t_fin, "value_norm": norm})
        tmp = (
            tmp.groupby("times", as_index=False)["value_norm"]
            .mean()
            .sort_values("times")
            .reset_index(drop=True)
        )
        t_view = tmp["times"].to_numpy(dtype=float)
        y_view = tmp["value_norm"].to_numpy(dtype=float)
        if t_view.size == 0:
            continue
        by_view[str(view)] = (t_view, y_view)
    if not by_view:
        return pd.DataFrame(columns=columns)

    t_ref = np.concatenate([vals[0] for vals in by_view.values()])
    t_ref = np.sort(np.unique(t_ref[np.isfinite(t_ref)]))
    if t_ref.size == 0:
        return pd.DataFrame(columns=columns)

    wh_mat = np.full((len(by_view), t_ref.size), np.nan, dtype=float)
    for row_idx, (_view, (t_view, y_view)) in enumerate(by_view.items()):
        if t_view.size == 1:
            exact = np.isclose(t_ref, t_view[0], rtol=0.0, atol=1e-12)
            wh_mat[row_idx, exact] = y_view[0]
            continue
        interp_vals = np.interp(t_ref, t_view, y_view)
        in_range = (t_ref >= t_view[0]) & (t_ref <= t_view[-1])
        interp_vals[~in_range] = np.nan
        wh_mat[row_idx, :] = interp_vals

    n_views = np.sum(np.isfinite(wh_mat), axis=0).astype(int)
    wh_sum = np.nansum(wh_mat, axis=0)
    wh_norm = np.divide(
        wh_sum,
        n_views,
        out=np.full(t_ref.shape, np.nan, dtype=float),
        where=n_views > 0,
    )
    keep = np.isfinite(wh_norm) & (n_views > 0)
    if not np.any(keep):
        return pd.DataFrame(columns=columns)

    t_out = t_ref[keep]
    wh_out = wh_norm[keep]
    # Rescale whisk signal right at creation so downstream calculations use the new mean level.
    if target_wh_mean is not None:
        curr_mean = float(np.nanmean(wh_out))
        if np.isfinite(curr_mean) and not np.isclose(curr_mean, 0.0):
            wh_signal_scale = float(target_wh_mean) / curr_mean
            wh_out = wh_out * wh_signal_scale
    if isinstance(config, dict):
        # Reuse this same scale for left/right camera traces in plotting.
        config["WH_SIGNAL_SCALE"] = wh_signal_scale
    n_views_out = n_views[keep]
    out = pd.DataFrame(
        {
            "bin_idx": np.arange(t_out.size, dtype=int),
            "bin_start_s": t_out,
            "bin_center_s": t_out,
            "bin_end_s": t_out,
            "wh_norm": wh_out,
            "n_views": n_views_out,
        }
    )
    return out[columns]


def _get_cluster_firing_rate(clusters, cluster_ids=None):
    if clusters is None:
        return None
    rate = None
    if hasattr(clusters, "firing_rate"):
        rate = np.asarray(clusters.firing_rate)
    elif isinstance(clusters, dict) and "firing_rate" in clusters:
        rate = np.asarray(clusters.get("firing_rate"))
    elif hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "firing_rate" in clusters.metrics.columns:
            rate = np.asarray(clusters.metrics["firing_rate"])
    if rate is None:
        return None
    if cluster_ids is None:
        return rate
    cluster_ids = np.asarray(cluster_ids)
    if len(rate) == len(cluster_ids):
        return rate
    cluster_id_all = None
    if hasattr(clusters, "cluster_id"):
        cluster_id_all = np.asarray(clusters.cluster_id)
    elif isinstance(clusters, dict) and "cluster_id" in clusters:
        cluster_id_all = np.asarray(clusters.get("cluster_id"))
    if cluster_id_all is None or len(cluster_id_all) != len(rate):
        return None
    rate_map = dict(zip(cluster_id_all, rate))
    return np.asarray([rate_map.get(cid, np.nan) for cid in cluster_ids], dtype=float)


def _select_cluster_ids_by_label(cluster_ids, clusters, label_min=None):
    cluster_ids = np.asarray(cluster_ids)
    if label_min is None:
        return cluster_ids
    labels = get_cluster_labels_array(clusters)
    if labels is None:
        return cluster_ids
    labels = np.asarray(labels)
    if labels.shape[0] != cluster_ids.shape[0]:
        return cluster_ids
    try:
        labels_float = labels.astype(float)
        mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return cluster_ids[mask]


def _build_trial_table(trials, trial_idx):
    contrast_left = np.asarray(trials["contrastLeft"], dtype=float)
    contrast_right = np.asarray(trials["contrastRight"], dtype=float)
    trial_contrasts = np.nanmax(np.vstack([np.abs(contrast_left), np.abs(contrast_right)]), axis=0)
    trial_contrasts = np.where(np.isnan(trial_contrasts), 0, trial_contrasts)
    reaction_time = (
        np.asarray(trials["response_times"], dtype=float)
        - np.asarray(trials["stimOn_times"], dtype=float)
    )
    choice_map = {1: "Left", -1: "Right", 0: "NoGo"}
    subject_response = choice_map.get(int(np.asarray(trials["choice"])[trial_idx]), "NA")
    correct_response = bool(np.asarray(trials["feedbackType"])[trial_idx] == 1)
    return pd.DataFrame(
        {
            "Trial": [int(trial_idx)],
            "Contrast": [float(trial_contrasts[trial_idx])],
            "Reaction Time (s)": [float(reaction_time[trial_idx])],
            "Correct Response": [correct_response],
            "Subject Response": [subject_response],
        }
    )


def _build_task_event_inputs(sl):
    trials = sl.trials
    trial_contrasts = ana_utils.get_trial_contrasts(sl)
    events_by_name = {}
    contrasts_by_name = {}
    trial_idx_by_name = {}
    for event_name in ("stimOn_times", "firstMovement_times", "feedback_times"):
        if event_name not in trials.keys():
            events_by_name[event_name] = np.array([], dtype=float)
            contrasts_by_name[event_name] = np.array([], dtype=float)
            trial_idx_by_name[event_name] = np.array([], dtype=int)
            continue
        times_all = np.asarray(trials[event_name], dtype=float)
        valid = np.isfinite(times_all)
        idx = np.nonzero(valid)[0]
        events_by_name[event_name] = times_all[valid]
        contrasts_by_name[event_name] = trial_contrasts[idx]
        trial_idx_by_name[event_name] = idx.astype(int)
    return events_by_name, contrasts_by_name, trial_idx_by_name


def _pearsonr_with_n(x, y, min_n=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


def _spearmanr_with_n(x, y, min_n=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    x = x[mask]
    y = y[mask]
    if spearmanr is not None:
        res = spearmanr(x, y)
        return float(res.correlation), n
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan, n
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), n


def _build_corr_heatmap_fig(corr_mat, n_mat, names, title, template):
    text_mat = np.empty(corr_mat.shape, dtype=object)
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            val = corr_mat[i, j]
            n_val = int(n_mat[i, j])
            if np.isfinite(val):
                text_mat[i, j] = f"{val:.2f}<br>(n={n_val})"
            else:
                text_mat[i, j] = f"nan<br>(n={n_val})"
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_mat,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=text_mat,
            texttemplate="%{text}",
            hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=1150,
        height=900,
        template=template,
        margin=dict(l=90, r=30, t=90, b=90),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def _build_event_session(event_times, event_name):
    return {"trials": {event_name: np.asarray(event_times, dtype=float)}}


def _extract_event_times_from_session(event_session, event_name):
    if not isinstance(event_session, dict):
        return np.array([], dtype=float)
    trials_obj = event_session.get("trials", {})
    if isinstance(trials_obj, dict):
        arr = np.asarray(trials_obj.get(event_name, np.array([])), dtype=float).reshape(-1)
    else:
        try:
            arr = np.asarray(trials_obj[event_name], dtype=float).reshape(-1)
        except Exception:
            arr = np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.sort(arr)


def _compute_event_locked_whisk_mean(whisk_df, event_times, x_axis):
    x_axis = np.asarray(x_axis, dtype=float).reshape(-1)
    if x_axis.size == 0:
        return np.array([], dtype=float)
    out = np.full(x_axis.shape, np.nan, dtype=float)
    if whisk_df is None or not isinstance(whisk_df, pd.DataFrame):
        return out
    if "bin_center_s" not in whisk_df.columns or "wh_norm" not in whisk_df.columns:
        return out

    t_wh = np.asarray(whisk_df["bin_center_s"], dtype=float).reshape(-1)
    v_wh = np.asarray(whisk_df["wh_norm"], dtype=float).reshape(-1)
    keep = np.isfinite(t_wh) & np.isfinite(v_wh)
    t_wh = t_wh[keep]
    v_wh = v_wh[keep]
    if t_wh.size < 2:
        return out
    order = np.argsort(t_wh)
    t_wh = t_wh[order]
    v_wh = v_wh[order]

    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    event_times = event_times[np.isfinite(event_times)]
    if event_times.size == 0:
        return out

    aligned = []
    for t_ev in event_times:
        x_query = x_axis + float(t_ev)
        vals = np.interp(x_query, t_wh, v_wh, left=np.nan, right=np.nan)
        aligned.append(vals)
    if len(aligned) == 0:
        return out
    return np.nanmean(np.vstack(aligned), axis=0)


def _build_multi_event_population_panel(
    event_specs,
    event_sessions,
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    df_res,
    plot_config,
    sort_mode,
    region_name,
    df_coupling=None,
    df_coupling_task=None,
    df_coupling_iti=None,
    df_firing_rate=None,
    whisk_df=None,
):
    n_events = len(event_specs)
    n_cols = int(plot_config.get("HEATMAP_PANEL_COLS", 4))
    n_cols = max(1, n_cols)
    n_event_rows = int(np.ceil(n_events / n_cols))
    heatmap_row_weight = float(plot_config.get("HEATMAP_MAIN_ROW_WEIGHT", 0.74))
    fr_row_weight = float(plot_config.get("HEATMAP_FR_ROW_WEIGHT", 0.16))
    whisk_row_weight = float(plot_config.get("HEATMAP_WHISK_ROW_WEIGHT", 0.10))
    panel_vertical_spacing = float(plot_config.get("HEATMAP_VERTICAL_SPACING", 0.07))
    title_yshift = float(plot_config.get("HEATMAP_TITLE_YSHIFT", 2))

    no_aux_events = {"stimOn_times", "firstMovement_times", "feedback_times"}
    panel_row_has_aux = []
    for panel_row in range(n_event_rows):
        i0 = panel_row * n_cols
        i1 = min((panel_row + 1) * n_cols, n_events)
        has_aux = any(
            bool(event_specs[i][1]) and (event_specs[i][1] not in no_aux_events)
            for i in range(i0, i1)
        )
        panel_row_has_aux.append(has_aux)

    row_heights = []
    panel_row_to_rows = []
    next_row = 1
    for panel_row in range(n_event_rows):
        has_aux = panel_row_has_aux[panel_row]
        row_heat = next_row
        next_row += 1
        if has_aux:
            row_heights.extend([heatmap_row_weight, fr_row_weight, whisk_row_weight])
            row_fr = next_row
            next_row += 1
            row_wh = next_row
            next_row += 1
        else:
            row_heights.append(heatmap_row_weight)
            row_fr = None
            row_wh = None
        panel_row_to_rows.append((row_heat, row_fr, row_wh))
    n_rows = next_row - 1

    subplot_titles = [""] * (n_rows * n_cols)
    for idx, (label, _event_name) in enumerate(event_specs):
        panel_row = idx // n_cols
        col = idx % n_cols
        row_heat, _row_fr, _row_wh = panel_row_to_rows[panel_row]
        title_idx = (row_heat - 1) * n_cols + col
        if 0 <= title_idx < len(subplot_titles):
            subplot_titles[title_idx] = str(label)
    subplot_specs = [[{"secondary_y": True} for _ in range(n_cols)] for _ in range(n_rows)]
    fig_panel = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=panel_vertical_spacing,
        row_heights=row_heights,
        specs=subplot_specs,
    )
    if fig_panel.layout.annotations:
        for ann in fig_panel.layout.annotations:
            ann.yshift = title_yshift
            ann.xanchor = "center"

    def _xaxis_name(row, col):
        axis_idx = (row - 1) * n_cols + col
        return "x" if axis_idx == 1 else f"x{axis_idx}"

    whisk_brief_summary_key = (
        "wh_brief_times_spont"
        if "wh_brief_times_spont" in event_sessions
        else "wh_brief_times"
    )
    whisk_long_summary_key = (
        "wh_long_times_spont"
        if "wh_long_times_spont" in event_sessions
        else "wh_long_times"
    )
    brief_times_summary = _extract_event_times_from_session(
        event_sessions.get(whisk_brief_summary_key),
        whisk_brief_summary_key,
    )
    long_times_summary = _extract_event_times_from_session(
        event_sessions.get(whisk_long_summary_key),
        whisk_long_summary_key,
    )
    whisk_count_brief = int(brief_times_summary.size)
    whisk_count_long = int(long_times_summary.size)

    region_cluster_ids = np.asarray(
        [
            cid
            for cid, reg in zip(plot_cluster_ids, plot_cluster_acronyms)
            if str(reg) == str(region_name)
        ],
        dtype=np.asarray(plot_cluster_ids).dtype if len(plot_cluster_ids) else int,
    )
    arousal_plus_count = 0
    arousal_minus_count = 0
    arousal_neutral_count = 0
    if (
        isinstance(df_res, pd.DataFrame)
        and not df_res.empty
        and ("cluster_id" in df_res.columns)
        and ("arousal_group" in df_res.columns)
        and (region_cluster_ids.size > 0)
    ):
        df_region_arousal = (
            df_res.loc[
                df_res["cluster_id"].isin(region_cluster_ids),
                ["cluster_id", "arousal_group"],
            ]
            .drop_duplicates(subset=["cluster_id"], keep="first")
            .copy()
        )
        if not df_region_arousal.empty:
            group_vals = df_region_arousal["arousal_group"].astype(str).str.strip().str.lower()
            arousal_plus_count = int((group_vals == "arousal_plus").sum())
            arousal_minus_count = int((group_vals == "arousal_minus").sum())
            arousal_neutral_count = int((group_vals == "neutral").sum())

    fr_axis_meta = {}
    show_heatmap_colorbar = bool(plot_config.get("HEATMAP_SHOW_COLORBAR", True))
    heatmap_colorbar_added = False
    event_window_overrides = plot_config.get("POP_WINDOWS_BY_EVENT", {})

    def _finite_float(value):
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out):
            return None
        return out

    for idx, (_event_label, event_name) in enumerate(event_specs):
        if not event_name:
            continue
        panel_row = idx // n_cols
        col = idx % n_cols + 1
        row_heat, row_fr, row_wh = panel_row_to_rows[panel_row]
        keep_aux = (event_name not in no_aux_events) and (row_fr is not None) and (row_wh is not None)
        event_session = event_sessions.get(event_name)
        if event_session is None:
            continue

        cfg = dict(plot_config)
        cfg["PLOT_EVENT"] = event_name
        pop_window_pre = float(cfg.get("POP_WINDOW_PRE", 0.1))
        pop_window_post = float(cfg.get("POP_WINDOW_POST", 0.2))
        event_window = event_window_overrides.get(event_name)
        if event_window is not None and len(event_window) == 2:
            pop_window_pre = float(event_window[0])
            pop_window_post = float(event_window[1])
        cfg["POP_WINDOW_PRE"] = pop_window_pre
        cfg["POP_WINDOW_POST"] = pop_window_post
        fig_event = plot_population_sorted_plotly(
            event_session,
            spikes,
            clusters,
            plot_cluster_ids,
            plot_cluster_acronyms,
            df_res,
            cfg,
            df_coupling=df_coupling,
            df_coupling_task=df_coupling_task,
            df_coupling_iti=df_coupling_iti,
            df_firing_rate=df_firing_rate,
            region_acronyms=[region_name],
            sort_mode=sort_mode,
        )
        if fig_event is None or len(fig_event.data) == 0:
            continue

        heatmap_trace = None
        for trace in fig_event.data:
            if isinstance(trace, go.Heatmap):
                heatmap_trace = go.Figure(data=[trace]).data[0]
                break
        if heatmap_trace is None:
            continue

        x_vals = np.asarray(heatmap_trace.x, dtype=float).reshape(-1)
        z_vals = np.asarray(heatmap_trace.z, dtype=float)
        if z_vals.ndim == 1:
            z_vals = z_vals.reshape(1, -1)
        if x_vals.size == 0 and z_vals.ndim == 2 and z_vals.shape[1] > 0:
            x_vals = np.linspace(-pop_window_pre, pop_window_post, z_vals.shape[1])
        heatmap_trace.z = z_vals

        pop_zscore = bool(cfg.get("POP_ZSCORE", False))
        pop_zmin = _finite_float(cfg.get("POP_ZMIN", None))
        pop_zmax = _finite_float(cfg.get("POP_ZMAX", None))
        if pop_zscore:
            heatmap_trace.zmid = 0.0
        if pop_zmin is not None and pop_zmax is not None and pop_zmax > pop_zmin:
            heatmap_trace.zmin = pop_zmin
            heatmap_trace.zmax = pop_zmax

        show_scale = show_heatmap_colorbar and (not heatmap_colorbar_added)
        heatmap_trace.showscale = show_scale
        if show_scale:
            heatmap_trace.colorbar = dict(
                title="Baseline z-score" if pop_zscore else (
                    "Norm FR" if bool(cfg.get("POP_NORMALIZE", False)) else "FR"
                ),
                len=0.80,
                y=0.5,
                yanchor="middle",
            )
            heatmap_colorbar_added = True
        fig_panel.add_trace(heatmap_trace, row=row_heat, col=col)

        for trace in fig_event.data:
            if not isinstance(trace, go.Scatter):
                continue
            tr_name = str(getattr(trace, "name", ""))
            if tr_name not in {"Delay", "Odd delay", "Even delay"}:
                continue
            trace_copy = go.Figure(data=[trace]).data[0]
            trace_copy.showlegend = False
            fig_panel.add_trace(trace_copy, row=row_heat, col=col)

        if keep_aux:
            fr_mean_plus = np.full(x_vals.shape, np.nan, dtype=float)
            fr_mean_minus = np.full(x_vals.shape, np.nan, dtype=float)
            fr_mean_all = np.full(x_vals.shape, np.nan, dtype=float)
            if z_vals.ndim == 2 and z_vals.shape[1] == x_vals.size and z_vals.shape[0] > 0:
                fr_mean_all = np.nanmean(z_vals, axis=0)
                row_groups = None
                group_info_available = False
                heatmap_meta = getattr(heatmap_trace, "meta", None)
                if isinstance(heatmap_meta, dict):
                    row_groups = heatmap_meta.get("row_group_values")
                    if row_groups is None:
                        row_cluster_ids = heatmap_meta.get("row_cluster_ids")
                        if (
                            row_cluster_ids is not None
                            and len(row_cluster_ids) == z_vals.shape[0]
                            and isinstance(df_res, pd.DataFrame)
                            and ("cluster_id" in df_res.columns)
                            and ("arousal_group" in df_res.columns)
                        ):
                            df_group_lookup = (
                                df_res[["cluster_id", "arousal_group"]]
                                .drop_duplicates(subset=["cluster_id"], keep="first")
                                .copy()
                            )
                            group_lookup = dict(
                                zip(
                                    df_group_lookup["cluster_id"].tolist(),
                                    df_group_lookup["arousal_group"].tolist(),
                                )
                            )
                            row_groups = [group_lookup.get(cid, "neutral") for cid in row_cluster_ids]
                if row_groups is not None and len(row_groups) == z_vals.shape[0]:
                    group_info_available = True
                    group_vals = pd.Series(row_groups).astype(str).str.strip().str.lower().to_numpy()
                    plus_mask = np.isin(
                        group_vals,
                        ["arousal_plus", "exc", "excitatory", "increase"],
                    )
                    minus_mask = np.isin(
                        group_vals,
                        ["arousal_minus", "inh", "inhibitory", "decrease"],
                    )
                    if np.any(plus_mask):
                        fr_mean_plus = np.nanmean(z_vals[plus_mask, :], axis=0)
                    if np.any(minus_mask):
                        fr_mean_minus = np.nanmean(z_vals[minus_mask, :], axis=0)
                if not group_info_available:
                    fr_mean_plus = fr_mean_all.copy()
                    fr_mean_minus = fr_mean_all.copy()
            plus_trace_name = "Arousal+ mean z-score" if pop_zscore else "Arousal+ mean FR"
            fig_panel.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=fr_mean_plus,
                    mode="lines",
                    line=dict(color="#8b0000", width=2),
                    showlegend=False,
                    name=plus_trace_name,
                    hovertemplate=(
                        "Time: %{x:.3f}s<br>"
                        + f"{plus_trace_name}: "
                        + "%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row_fr,
                col=col,
            )
            minus_trace_name = "Arousal- mean z-score" if pop_zscore else "Arousal- mean FR"
            fig_panel.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=fr_mean_minus,
                    mode="lines",
                    line=dict(color="#00008b", width=2),
                    showlegend=False,
                    name=minus_trace_name,
                    hovertemplate=(
                        "Time: %{x:.3f}s<br>"
                        + f"{minus_trace_name}: "
                        + "%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row_fr,
                col=col,
            )
            fr_finite = np.concatenate(
                [
                    np.asarray(fr_mean_plus, dtype=float).reshape(-1),
                    np.asarray(fr_mean_minus, dtype=float).reshape(-1),
                ]
            )
            fr_finite = fr_finite[np.isfinite(fr_finite)]
            if fr_finite.size > 0:
                fr_axis_meta[event_name] = {
                    "row": row_fr,
                    "col": col,
                    "ymin": float(np.nanmin(fr_finite)),
                    "ymax": float(np.nanmax(fr_finite)),
                }

            ev_times = _extract_event_times_from_session(event_session, event_name)
            whisk_mean = _compute_event_locked_whisk_mean(whisk_df, ev_times, x_vals)
            fig_panel.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=whisk_mean,
                    mode="lines",
                    line=dict(color="#c77c2e", width=2),
                    showlegend=False,
                    name="Mean whisk",
                    hovertemplate="Time: %{x:.3f}s<br>Mean whisk: %{y:.3f}<extra></extra>",
                ),
                row=row_wh,
                col=col,
            )

        target_rows = [row_heat]
        if keep_aux:
            target_rows.extend([row_fr, row_wh])
        for target_row in target_rows:
            fig_panel.add_vline(
                x=0,
                row=target_row,
                col=col,
                line=dict(color="black", dash="dash"),
            )
            fig_panel.update_xaxes(
                range=[-pop_window_pre, pop_window_post],
                row=target_row,
                col=col,
            )

        if keep_aux:
            # Lock zoom/pan among heatmap + FR + whisk for this event panel.
            x_axis_ref = _xaxis_name(row_heat, col)
            fig_panel.update_xaxes(matches=x_axis_ref, row=row_fr, col=col)
            fig_panel.update_xaxes(matches=x_axis_ref, row=row_wh, col=col)

        fig_panel.update_yaxes(autorange="reversed", row=row_heat, col=col)
        fig_panel.update_yaxes(showticklabels=False, row=row_heat, col=col)
        if panel_row == n_event_rows - 1 and keep_aux:
            fig_panel.update_xaxes(title_text="Time from event (s)", row=row_wh, col=col)
        elif panel_row == n_event_rows - 1:
            fig_panel.update_xaxes(title_text="Time from event (s)", row=row_heat, col=col)

        if col == 1:
            fig_panel.update_yaxes(
                title_text="Neurons",
                title_standoff=28,
                row=row_heat,
                col=col,
            )
            if keep_aux:
                fig_panel.update_yaxes(
                    title_text="Mean z-score" if pop_zscore else "Firing rate",
                    row=row_fr,
                    col=col,
                )
                fig_panel.update_yaxes(title_text="Whisk", row=row_wh, col=col)

    summary_slots = [idx for idx, (_label, event_name) in enumerate(event_specs) if not event_name]
    if len(summary_slots) > 0:
        summary_x_whisk = ["Wh brief", "Wh long"]
        summary_y_whisk = [whisk_count_brief, whisk_count_long]
        summary_x_arousal = ["Arousal+", "Arousal-", "Neutral"]
        summary_y_arousal = [arousal_plus_count, arousal_minus_count, arousal_neutral_count]
        whisk_y_max = max(summary_y_whisk + [1])
        neuron_y_max = max(summary_y_arousal + [1])

        for idx in summary_slots:
            panel_row = idx // n_cols
            col = idx % n_cols + 1
            row_heat, _row_fr, _row_wh = panel_row_to_rows[panel_row]

            fig_panel.add_trace(
                go.Bar(
                    x=summary_x_whisk,
                    y=summary_y_whisk,
                    name="Whisk events",
                    legendgroup="summary_whisk",
                    marker=dict(color=["#ff8c00", "#2ca02c"]),
                    text=summary_y_whisk,
                    textposition="outside",
                    showlegend=(idx == summary_slots[0]),
                    hovertemplate="%{x}: %{y}<extra>Whisk events</extra>",
                ),
                row=row_heat,
                col=col,
                secondary_y=False,
            )
            fig_panel.add_trace(
                go.Bar(
                    x=summary_x_arousal,
                    y=summary_y_arousal,
                    name="Neurons (arousal group)",
                    legendgroup="summary_arousal",
                    marker=dict(color=["#8b0000", "#00008b", "#7f7f7f"]),
                    text=summary_y_arousal,
                    textposition="outside",
                    showlegend=(idx == summary_slots[0]),
                    hovertemplate="%{x}: %{y}<extra>Neurons</extra>",
                ),
                row=row_heat,
                col=col,
                secondary_y=True,
            )
            fig_panel.update_xaxes(tickangle=-20, row=row_heat, col=col)
            fig_panel.update_yaxes(
                title_text="Whisk count",
                range=[0, whisk_y_max * 1.15],
                row=row_heat,
                col=col,
                secondary_y=False,
            )
            fig_panel.update_yaxes(
                title_text="Neuron count",
                range=[0, neuron_y_max * 1.15],
                row=row_heat,
                col=col,
                secondary_y=True,
            )

    # Match Wh Long onset/offset FR axis ranges using the combined extrema of both.
    fr_ref = fr_axis_meta.get("wh_long_times_spont")
    if fr_ref is None:
        fr_ref = fr_axis_meta.get("wh_long_times")
    fr_target = fr_axis_meta.get("wh_long_offset_times_spont")
    if fr_target is None:
        fr_target = fr_axis_meta.get("wh_long_offset_times")
    if fr_ref is not None and fr_target is not None:
        y_min = min(float(fr_ref["ymin"]), float(fr_target["ymin"]))
        y_max = max(float(fr_ref["ymax"]), float(fr_target["ymax"]))
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_max <= y_min:
                pad = max(abs(y_min) * 0.05, 1e-6)
                y_min -= pad
                y_max += pad
            y_range = [y_min, y_max]
            fig_panel.update_yaxes(
                range=y_range,
                row=int(fr_ref["row"]),
                col=int(fr_ref["col"]),
            )
            fig_panel.update_yaxes(
                range=y_range,
                row=int(fr_target["row"]),
                col=int(fr_target["col"]),
            )

    fig_panel.update_layout(
        title=f"Response Analysis (Region {region_name})",
        width=max(1200, 360 * n_cols + 120),
        height=max(640, int(340 * float(sum(row_heights))) + 220),
        template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
        barmode="group",
        margin=dict(l=80, r=40, t=90, b=70),
    )
    fig_panel.update_xaxes(showgrid=False)
    fig_panel.update_yaxes(showgrid=False)
    return fig_panel


def _choose_option(options, prompt, default_value):
    options = list(options or [])
    if not options:
        raise ValueError("No options available.")
    if default_value not in options:
        default_value = options[0]
    default_idx = options.index(default_value)
    print(prompt)
    for idx, name in enumerate(options):
        print(f"  [{idx}] {name}")
    try:
        selection = input(
            f"Choose index or exact name (Enter for [{default_idx}] {default_value}): "
        ).strip()
    except Exception:
        return default_value
    if selection == "":
        return default_value
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(options):
            return options[idx]
        print(f"Index {idx} out of range. Using default: {default_value}")
        return default_value
    if selection in options:
        return selection
    lower_map = {str(opt).lower(): opt for opt in options}
    if selection.lower() in lower_map:
        return lower_map[selection.lower()]
    print(f"Unknown selection '{selection}'. Using default: {default_value}")
    return default_value


def _format_corr_value(val):
    return "nan" if not np.isfinite(val) else f"{val:.3f}"


_set_plotly_renderer(PLOTLY_RENDERER)


# %% PID and ONE session loading
PID = "c9664185-d3fd-4e0e-89cf-77c402038938"
# PID = "3eb6e6e0-8a57-49d6-b7c9-f39d5834e682" # "afe87fbb-3a17-461f-b333-e22903f1d70d"  
TARGET_REGION = None # "VISa"
CALC_LABEL_MIN = 0.9
MIN_REGION_NEURONS = 20
ONE_PREFERRED_MODE = "remote"  # "local" or "remote"
ALLOW_REMOTE_FALLBACK = True

LOAD_WHEEL = True
LOAD_POSE = True
LOAD_MOTION_ENERGY = True
LOAD_PUPIL = False

path_data, _path_fig, path_data_processed, ibl_cache = setup_paths(BASE_PATH)
one, one_mode = _init_one_with_fallback(
    ibl_cache,
    preferred_mode=ONE_PREFERRED_MODE,
    allow_remote=ALLOW_REMOTE_FALLBACK,
)
print(f"Using ONE mode: {one_mode}")

if PID is None:
    pid_list = list_cached_pids()
    if not pid_list:
        raise RuntimeError("No cached PID list found in data/dashboard_cache; set PID manually.")
    pid = choose_pid(pid_list, default_index=0)
else:
    pid = PID
print(f"Selected PID: {pid}")

ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)
ssl, spikes, clusters, sl = load_session_data(
    pid,
    one,
    ba=ba,
    load_wheel=LOAD_WHEEL,
    load_pose=LOAD_POSE,
    load_motion_energy=LOAD_MOTION_ENERGY,
    load_pupil=LOAD_PUPIL,
)
eid = getattr(ssl, "eid", None)
if eid is None:
    eid, _ = one.pid2eid(pid)
print(f"EID: {eid}")

if not hasattr(sl, "trials") or sl.trials is None:
    raise RuntimeError(f"Selected PID {pid} has no trials object.")

cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
cluster_acronyms_calc = map_acronyms(clusters, br, "Beryl")
cluster_acronyms_plot = np.asarray(cluster_acronyms_calc).astype(str)

cluster_ids_arr = np.asarray(cluster_ids)
cluster_labels = get_cluster_labels_array(clusters)
good_label_mask = np.ones(cluster_ids_arr.shape[0], dtype=bool)
if cluster_labels is not None:
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.shape[0] == cluster_ids_arr.shape[0]:
        try:
            good_label_mask = cluster_labels.astype(float) >= float(CALC_LABEL_MIN)
        except (TypeError, ValueError):
            good_label_mask = cluster_labels == 1

if TARGET_REGION is None:
    eligible_region_series = pd.Series(cluster_acronyms_plot[good_label_mask]).astype(str)
    eligible_region_series = eligible_region_series[
        ~eligible_region_series.isin(["root", "void", "nan", "None"])
    ]
    eligible_region_counts = eligible_region_series.value_counts()
    auto_plot_regions = sorted(
        eligible_region_counts[eligible_region_counts > int(MIN_REGION_NEURONS)].index.tolist()
    )
    if len(auto_plot_regions) == 0:
        raise RuntimeError(
            f"No regions have >{MIN_REGION_NEURONS} neurons with label >= {CALC_LABEL_MIN} "
            f"for PID {pid} (EID {eid})."
        )
    target_region_mask = np.asarray(
        [str(acr) in auto_plot_regions for acr in cluster_acronyms_plot],
        dtype=bool,
    )
    print(
        "TARGET_REGION=None -> selected regions: "
        + ", ".join(
            [
                f"{reg} (n={int(eligible_region_counts.get(reg, 0))})"
                for reg in auto_plot_regions
            ]
        )
    )
else:
    target_region_mask = np.asarray(
        [str(acr).startswith(str(TARGET_REGION)) for acr in cluster_acronyms_plot],
        dtype=bool,
    )
    auto_plot_regions = [str(TARGET_REGION)]

target_region_cluster_ids = np.asarray(cluster_ids)[target_region_mask]
if target_region_cluster_ids.size == 0:
    if TARGET_REGION is None:
        raise RuntimeError(
            f"No clusters found for auto-selected regions (PID {pid}, EID {eid})."
        )
    raise RuntimeError(
        f"No clusters from region '{TARGET_REGION}' found for PID {pid} (EID {eid})."
    )
target_region_cluster_id_set = set(int(cid) for cid in target_region_cluster_ids.tolist())
target_region_cid_to_idx = {
    int(cid): cid_to_idx[int(cid)]
    for cid in target_region_cluster_ids
    if int(cid) in cid_to_idx
}
target_region_cluster_acronyms = np.asarray(
    [cluster_acronyms_calc[cid_to_idx[int(cid)]] for cid in target_region_cluster_ids],
    dtype=str,
)

spont_intervals = _load_spontaneous_intervals(one, eid)
if spont_intervals is None or len(np.asarray(spont_intervals).reshape(-1)) < 2:
    raise RuntimeError(
        f"Selected PID {pid} (EID {eid}) has no spontaneous interval in *passivePeriods*."
    )
spont_intervals = np.asarray(spont_intervals, dtype=float)
spont_interval_list = [tuple(row) for row in spont_intervals]

df_me_raw = ana_utils.extract_motion_energy_trace(
    getattr(sl, "motion_energy", None),
    max_interp_gap_frames=3,
    ensure_positive_motion=True,
)
if df_me_raw.empty:
    raise RuntimeError(
        f"Selected PID {pid} (EID {eid}) has no usable motion energy trace from left/right camera."
    )
print(
    f"Region units: {len(target_region_cluster_ids)} | "
    f"Motion-energy samples: {len(df_me_raw)} | "
    f"Spont intervals: {len(spont_interval_list)}"
)


# %% CONFIG (analysis + plotting)
CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_LABEL_MIN": CALC_LABEL_MIN,
    "CALC_SPONT": True,
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "feedback_times"],
    # Options: "com", "com_signed", "psth_peak", "psth_peak_signed", "tfs", "latenzy"
    "DELAY_METHOD": "com_signed",
    "WH_DELAY_METHOD": "com_signed",
    # latenzy options (used only when DELAY_METHOD or WH_DELAY_METHOD == "latenzy")
    # None -> call latenzy(spikes, events) without use_dur.
    # Set to scalar or (start, end) only if you explicitly want use_dur.
    "LATENZY_USE_DUR": None,
    # Optional certainty threshold on latenzy's second output; None disables thresholding.
    "LATENZY_MIN_SCORE": None,
    "DELAY_UNITS": "ms",
    "FULL_CONTRAST_VALUES": (1.0, 100.0),
    "DELAY_WINDOWS": {
        "stimOn_times": (0.02, 0.35),
        "firstMovement_times": (-0.1, 0.2),
        "feedback_times": (-0.1, 0.2),
    },
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -1.0,
    "PSTH_WINDOW_END": 1.0,
    "RESPONSIVE_WINDOW_START": 0.02,
    "RESPONSIVE_WINDOW_END": 0.35,
    # If True, responsiveness/sign are computed on baseline z-scored PSTHs.
    "RESPONSIVE_USE_ZSCORE": True,
    # "smooth" (default) uses smoothed PSTH for z-scoring; "raw" uses unsmoothed.
    "RESPONSIVE_ZSCORE_SOURCE": "smooth",
    # COM methods: True -> compute COM on threshold-crossing bins; False -> full response window.
    "COM_USE_THRESHOLD": True,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 10,
    "MIN_TRIALS_SPLIT": 5,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_LOW_PASS_ORDER": 3,
    "STPR_POP_USE_GOOD_UNITS": False,
    "TASK_POST_EVENT_S": 1.0,
    "ITI_SKIP_FIRST_LAST": True,
    "WH_BIN_SIGNAL": False,  # Set False to keep whisk signal unbinned.
    "WH_BIN_S": 0.3,
    "WH_NORM_PCTL": 1.0,
    "WH_NORM_TOP_PCTL": 100.0,
    "WH_BIN_REDUCE": "mean",
    "WH_NORMALIZE_AFTER_BIN": True,
    # Target mean for the final whisk trace; scaling is applied when the signal is created.
    "WH_TARGET_MEAN": 0.06, #0.05465713847833459,
    "WH_START_THR": 0.10,
    "WH_END_THR": 0.04,
    "WH_END_QUIET_WINDOW_S": 0.5,
    "WH_MERGE_GAP_S": 0.3,
    "WH_BRIEF_RANGE_S": (0.25, 2.0),
    "WH_LONG_MIN_S": 2.0,
    "WH_LOCO_LOOKAHEAD_S": 6.0,
    "WH_LOCO_SPEED_THR_CM_S": 1.0,
    "WH_WHEEL_RADIUS_CM": 3.1,
    "WH_DELAY_WINDOW": (0, 0.4), # (-0.2, 0.5),   
    "WH_SPLIT_MODE": "odd_even",
    "AROUSAL_GROUP_MODE": "response_sign",  # "corr" or "response_sign"
    "AROUSAL_SIGN_EVENT": "wh_brief_times_spont",
    "AROUSAL_POS_THR": 0.05,
    "AROUSAL_NEG_THR": -0.05,
    "AROUSAL_MIN_FR_HZ": 0.5,
    "AROUSAL_MIN_BRIEF_EVENTS": 5,
    "AROUSAL_MIN_LONG_EVENTS": 5,
    "AROUSAL_BIN_S": 0.3,
    "AROUSAL_SMOOTH_SIGMA": 5,
    "AROUSAL_MIN_CORR_BINS": 10,
    # If True, arousal corr z-scoring uses pre-event baseline bins [t0-BASELINE_PRE, t0).
    "AROUSAL_USE_EVENT_BASELINE_ZSCORE": True,
    # Optional override; when omitted, BASELINE_PRE is used.
    "AROUSAL_BASELINE_PRE": 0.2,
    "AROUSAL_MIN_BASELINE_BINS": 3,
    "AROUSAL_REQUIRE_SPLIT_HALF": True,
}

CONFIG_PLOT = {
    "ATLAS_MAPPING": "Beryl",
    "PLOT_ONLY_GOOD_UNITS": False,
    "PLOT_EVENT": "stimOn_times",
    "PLOT_REGIONS": list(auto_plot_regions),
    "RASTER_WINDOW_PRE": 1,
    "RASTER_WINDOW_POST": 2,
    "RASTER_ALIGN_TO_EVENT": True,
    "RASTER_ALIGN_TO_STIM_ON": True,
    "SINGLE_NEURON_RASTER_PRE": 0.5,
    "SINGLE_NEURON_RASTER_POST": 1.0,
    "SINGLE_NEURON_BIN_SIZE": 0.05,
    "SINGLE_NEURON_SMOOTH_SIGMA": 1,
    "SEQUENCE_WINDOW_PRE": 0.5,
    "SEQUENCE_WINDOW_POST": 1.0,
    "SEQUENCE_ALIGN_TO_EVENT": True,
    "SEQUENCE_ALIGN_TO_STIM": True,
    "POP_WINDOW_PRE": 0.1,
    "POP_WINDOW_POST": 0.2,
    "POP_BIN_SIZE": 0.005,
    "POP_SMOOTH_SIGMA": 2,
    "POP_CMAP_NAME": "bwr",
    "POP_NORMALIZE": True,
    "HEATMAP_PANEL_COLS": 4,
    "HEATMAP_GROUP_BY_AROUSAL": True,  # Whisk: arousal-/neutral/arousal+; task: response-sign inh/none/exc.
    "SORT_BY_SPONT": True,
}

PLOT_DARK_THEME = False
PLOT_LABEL_MIN = float(CONFIG_CALC.get("CALC_LABEL_MIN", 0.5))
CORR_MIN_N = 2

TRIAL_INDEX = 334
TRIAL_RASTER_SORT = "Whisk Corr |r|"
GENERAL_RASTER_START = 2950
GENERAL_RASTER_END = 3050
GENERAL_RASTER_SORT = TRIAL_RASTER_SORT
HEATMAP_SORT = "Own Event Delay"

RASTER_SORT_MAP = {
    "Default (Depth)": "depth",
    "Delay to Stim On": "delay:stimOn_times",
    "Delay to First Move": "delay:firstMovement_times",
    "Delay to Feedback": "delay:feedback_times",
    "Delay to Wh Brief": "delay:wh_brief_times_spont",
    "Delay to Wh Long": "delay:wh_long_times_spont",
    "Delay to Wh All": "delay:wh_all_times_spont",
    "Task stPR Delay": "task",
    "Task stPR Strength": "task_strength",
    "Task stPR Max": "task_max",
    "ITI stPR Delay": "iti",
    "ITI stPR Strength": "iti_strength",
    "ITI stPR Max": "iti_max",
    "Spont stPR Delay": "spont",
    "Spont stPR Strength": "spont_strength",
    "Spont stPR Max": "spont_max",
    "Firing rate": "firing_rate",
    "Whisk Corr |r|": "whisk_corr_abs",
}

HEATMAP_SORT_MAP = {
    "Own Event Delay": "delay",
    "Default (Depth)": "depth",
    "Delay to Stim On": "delay:stimOn_times",
    "Delay to First Move": "delay:firstMovement_times",
    "Delay to Feedback": "delay:feedback_times",
    "Delay to Wh Brief": "delay:wh_brief_times_spont",
    "Delay to Wh Long": "delay:wh_long_times_spont",
    "Delay to Wh All": "delay:wh_all_times_spont",
    "Task stPR Delay": "task",
    "Task stPR Strength": "task_strength",
    "ITI stPR Delay": "iti",
    "ITI stPR Strength": "iti_strength",
    "Spont stPR Delay": "spont",
    "Spont stPR Strength": "spont_strength",
    "Firing rate": "firing_rate",
}

plot_config = dict(CONFIG_PLOT)
plot_config["PSTH_WINDOW_START"] = CONFIG_CALC.get("PSTH_WINDOW_START", -0.2)
plot_config["PSTH_WINDOW_END"] = CONFIG_CALC.get("PSTH_WINDOW_END", 0.35)
plot_config["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
plot_config["SINGLE_NEURON_SMOOTH_SIGMA"] = 0.5
plot_config["SINGLE_NEURON_BIN_SIZE"] = 0.01
plot_config["PLOT_LABEL_MIN"] = PLOT_LABEL_MIN
plot_config["DELAY_UNITS"] = CONFIG_CALC.get("DELAY_UNITS", "s")
plot_config["PLOTLY_TEMPLATE"] = "plotly_dark" if PLOT_DARK_THEME else "plotly_white"
plotting_utils.DEFAULT_TEMPLATE = plot_config["PLOTLY_TEMPLATE"]
pio.templates.default = plot_config["PLOTLY_TEMPLATE"]


# %% Whisk normalization diagnostics (data stats only; no effect on calculations)
WH_NORM_DEBUG = False
if WH_NORM_DEBUG:
    dbg_df = df_me_raw[["times", "view", "value"]].copy()
    dbg_df = dbg_df.dropna(subset=["times", "value"])
    if dbg_df.empty:
        print("Whisk normalization diagnostics skipped: no finite motion-energy samples.")
    else:
        dbg_df["times"] = dbg_df["times"].astype(float)
        dbg_df["value"] = dbg_df["value"].astype(float)
        dbg_df["view"] = dbg_df["view"].astype(str)

        bin_s_dbg = float(CONFIG_CALC["WH_BIN_S"])
        norm_pctl_dbg = float(CONFIG_CALC["WH_NORM_PCTL"])
        norm_top_pctl_dbg = float(CONFIG_CALC.get("WH_NORM_TOP_PCTL", 100.0))
        bin_reduce_dbg = str(CONFIG_CALC.get("WH_BIN_REDUCE", "mean")).lower()
        t0_dbg = float(dbg_df["times"].min())
        use_bin_signal_dbg = bool(CONFIG_CALC.get("WH_BIN_SIGNAL", True))
        normalize_after_bin_dbg = bool(CONFIG_CALC.get("WH_NORMALIZE_AFTER_BIN", False))
        if use_bin_signal_dbg:
            order_label = (
                "bin -> normalize per camera -> average"
                if normalize_after_bin_dbg
                else "normalize per camera -> bin -> average"
            )
        else:
            order_label = "normalize per camera raw samples -> interpolate/average (no binning)"
        print(f"\nWhisk normalization order: {order_label}")
        top_label = "bins" if use_bin_signal_dbg else "samples"

        view_rows = []
        top_rows = []
        for view_name, grp in dbg_df.groupby("view", sort=False):
            vals = grp["value"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            vals_fin = vals[finite]
            tmp_raw = grp.loc[finite, ["times", "value"]].copy()
            if use_bin_signal_dbg:
                tmp_raw["bin_idx"] = np.floor(
                    (tmp_raw["times"].to_numpy(dtype=float) - t0_dbg) / bin_s_dbg
                ).astype(int)

                if normalize_after_bin_dbg:
                    grp_raw = tmp_raw.groupby("bin_idx", as_index=False)["value"]
                    if bin_reduce_dbg == "max":
                        by_bin_raw = grp_raw.max()
                    elif bin_reduce_dbg == "median":
                        by_bin_raw = grp_raw.median()
                    else:
                        by_bin_raw = grp_raw.mean()
                    vals_for_norm = by_bin_raw["value"].to_numpy(dtype=float)
                    if vals_for_norm.size == 0:
                        continue
                    lo = float(np.nanpercentile(vals_for_norm, norm_pctl_dbg))
                    hi = float(np.nanpercentile(vals_for_norm, norm_top_pctl_dbg))
                    if not np.isfinite(lo):
                        lo = float(np.nanmin(vals_for_norm))
                    if not np.isfinite(hi):
                        hi = float(np.nanmax(vals_for_norm))
                    if hi <= lo:
                        hi = float(np.nanmax(vals_for_norm))
                    if hi <= lo:
                        y_bin = np.zeros_like(vals_for_norm, dtype=float)
                    else:
                        y_bin = np.clip((vals_for_norm - lo) / (hi - lo), 0.0, 1.0)
                    x_bin = (
                        t0_dbg
                        + by_bin_raw["bin_idx"].to_numpy(dtype=float) * bin_s_dbg
                        + 0.5 * bin_s_dbg
                    )
                    norm_stats_vals = y_bin
                    norm_basis = "binned_raw"
                else:
                    lo = float(np.nanpercentile(vals_fin, norm_pctl_dbg))
                    hi = float(np.nanpercentile(vals_fin, norm_top_pctl_dbg))
                    if not np.isfinite(lo):
                        lo = float(np.nanmin(vals_fin))
                    if not np.isfinite(hi):
                        hi = float(np.nanmax(vals_fin))
                    if hi <= lo:
                        hi = float(np.nanmax(vals_fin))

                    if hi <= lo:
                        norm_samples = np.zeros_like(vals_fin, dtype=float)
                    else:
                        norm_samples = np.clip((vals_fin - lo) / (hi - lo), 0.0, 1.0)

                    tmp_norm = tmp_raw.copy()
                    tmp_norm["norm"] = norm_samples
                    grp_norm = tmp_norm.groupby("bin_idx", as_index=False)["norm"]
                    if bin_reduce_dbg == "max":
                        by_bin_norm = grp_norm.max()
                    elif bin_reduce_dbg == "median":
                        by_bin_norm = grp_norm.median()
                    else:
                        by_bin_norm = grp_norm.mean()
                    x_bin = (
                        t0_dbg
                        + by_bin_norm["bin_idx"].to_numpy(dtype=float) * bin_s_dbg
                        + 0.5 * bin_s_dbg
                    )
                    y_bin = by_bin_norm["norm"].to_numpy(dtype=float)
                    norm_stats_vals = norm_samples
                    norm_basis = "raw_samples"
            else:
                lo = float(np.nanpercentile(vals_fin, norm_pctl_dbg))
                hi = float(np.nanpercentile(vals_fin, norm_top_pctl_dbg))
                if not np.isfinite(lo):
                    lo = float(np.nanmin(vals_fin))
                if not np.isfinite(hi):
                    hi = float(np.nanmax(vals_fin))
                if hi <= lo:
                    hi = float(np.nanmax(vals_fin))

                if hi <= lo:
                    norm_samples = np.zeros_like(vals_fin, dtype=float)
                else:
                    norm_samples = np.clip((vals_fin - lo) / (hi - lo), 0.0, 1.0)
                x_bin = tmp_raw["times"].to_numpy(dtype=float)
                y_bin = norm_samples
                norm_stats_vals = norm_samples
                norm_basis = "raw_samples_no_bin"

            view_rows.append(
                {
                    "view": view_name,
                    "norm_basis": norm_basis,
                    "n_samples": int(vals_fin.size),
                    "raw_min": float(np.nanmin(vals_fin)),
                    "raw_p1": float(np.nanpercentile(vals_fin, 1)),
                    "raw_p50": float(np.nanpercentile(vals_fin, 50)),
                    "raw_p95": float(np.nanpercentile(vals_fin, 95)),
                    "raw_p99": float(np.nanpercentile(vals_fin, 99)),
                    "raw_max": float(np.nanmax(vals_fin)),
                    "norm_lo_used": lo,
                    "norm_hi_used": hi,
                    "norm_scale_hi_minus_lo": float(hi - lo),
                    "norm_min": float(np.nanmin(norm_stats_vals)),
                    "norm_p95": float(np.nanpercentile(norm_stats_vals, 95)),
                    "norm_p99": float(np.nanpercentile(norm_stats_vals, 99)),
                    "norm_max": float(np.nanmax(norm_stats_vals)),
                    "frac_norm_ge_0p5": float(np.mean(norm_stats_vals >= 0.5)),
                    "frac_norm_ge_0p8": float(np.mean(norm_stats_vals >= 0.8)),
                    "binned_max": float(np.nanmax(y_bin)) if y_bin.size else np.nan,
                    "binned_p99": float(np.nanpercentile(y_bin, 99)) if y_bin.size else np.nan,
                }
            )

            if y_bin.size > 0:
                top_idx = np.argsort(y_bin)[::-1][:5]
                for idx_top in top_idx:
                    top_rows.append(
                        {
                            "view": view_name,
                            "bin_center_s": float(x_bin[idx_top]),
                            "norm_bin_value": float(y_bin[idx_top]),
                        }
                    )

        print("\n=== Per-view normalization stats (values actually used) ===")
        df_view_norm = pd.DataFrame(view_rows)
        if not df_view_norm.empty:
            display(df_view_norm.sort_values("view").reset_index(drop=True))
        else:
            print("No per-view rows available.")

        df_wh_dbg = _build_whisk_trace(dbg_df, CONFIG_CALC)
        if not df_wh_dbg.empty:
            wh_dbg = df_wh_dbg["wh_norm"].to_numpy(dtype=float)
            wh_times_dbg = df_wh_dbg["bin_center_s"].to_numpy(dtype=float)
            mask_wh_dbg = np.isfinite(wh_dbg) & np.isfinite(wh_times_dbg)
            wh_dbg = wh_dbg[mask_wh_dbg]
            wh_times_dbg = wh_times_dbg[mask_wh_dbg]
            if wh_dbg.size > 0:
                print("\n=== Final mean trace stats (df_wh['wh_norm']) ===")
                print(
                    "wh_norm summary | "
                    f"min={np.nanmin(wh_dbg):.4f}, p50={np.nanpercentile(wh_dbg,50):.4f}, "
                    f"p95={np.nanpercentile(wh_dbg,95):.4f}, p99={np.nanpercentile(wh_dbg,99):.4f}, "
                    f"max={np.nanmax(wh_dbg):.4f}"
                )
                top_idx_wh = np.argsort(wh_dbg)[::-1][:10]
                df_top_mean = pd.DataFrame(
                    {
                        "bin_center_s": wh_times_dbg[top_idx_wh],
                        "wh_norm": wh_dbg[top_idx_wh],
                    }
                )
                df_top_mean = df_top_mean.merge(
                    df_wh_dbg[["bin_center_s", "n_views"]],
                    on="bin_center_s",
                    how="left",
                )
                print(f"\nTop 10 {top_label} of mean whisk trace:")
                display(df_top_mean.sort_values("wh_norm", ascending=False).reset_index(drop=True))
            else:
                print("No finite values in final mean trace.")
        else:
            print("No normalized mean whisk trace produced in diagnostics.")

        print(f"\nTop 5 {top_label} per view (after per-view normalization):")
        df_top_view_bins = pd.DataFrame(top_rows)
        if not df_top_view_bins.empty:
            display(
                df_top_view_bins.sort_values(["view", "norm_bin_value"], ascending=[True, False])
                .reset_index(drop=True)
            )
        else:
            print(f"No top-{top_label} table available.")


# %% Whisk calculations (fast; rerun this cell for threshold/settings tuning)
task_events_by_name, task_contrasts_by_name, task_trial_idx_by_name = _build_task_event_inputs(sl)

task_windows = ana_utils.build_task_window_table(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
trial_end_times = ana_utils.compute_trial_end_times(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
stim_on_times = np.asarray(sl.trials["stimOn_times"], dtype=float)
iti_windows = ana_utils.build_iti_windows(
    trial_end_times,
    stim_on_times,
    skip_first_last=CONFIG_CALC["ITI_SKIP_FIRST_LAST"],
)

df_wh = _build_whisk_trace(df_me_raw, CONFIG_CALC)
if df_wh.empty:
    raise RuntimeError("Whisk signal is empty after normalization.")

wh_vals = df_wh["wh_norm"].to_numpy(dtype=float)
if np.isfinite(wh_vals).any():
    q50, q90, q95, q99, qmax = np.nanpercentile(wh_vals, [50, 90, 95, 99, 100])
    print(
        "wh_norm stats | "
        f"median={q50:.3f}, p90={q90:.3f}, p95={q95:.3f}, p99={q99:.3f}, max={qmax:.3f}"
    )

wh_detect = ana_utils.detect_wh_bouts(
    df_wh["bin_center_s"].to_numpy(dtype=float),
    df_wh["wh_norm"].to_numpy(dtype=float),
    start_thr=CONFIG_CALC["WH_START_THR"],
    end_thr=CONFIG_CALC["WH_END_THR"],
    end_quiet_window_s=CONFIG_CALC.get("WH_END_QUIET_WINDOW_S", 4.0),
    merge_gap_s=CONFIG_CALC["WH_MERGE_GAP_S"],
    brief_range_s=CONFIG_CALC["WH_BRIEF_RANGE_S"],
    long_min_s=CONFIG_CALC["WH_LONG_MIN_S"],
)

# Keep only categorized whisk bouts/events (brief + long) in downstream analyses.
brief_bouts = np.asarray(wh_detect.get("brief_bouts", np.empty((0, 2))), dtype=float).reshape(-1, 2)
long_bouts = np.asarray(wh_detect.get("long_bouts", np.empty((0, 2))), dtype=float).reshape(-1, 2)
categorized_bouts = (
    np.vstack([brief_bouts, long_bouts])
    if (brief_bouts.size + long_bouts.size) > 0
    else np.empty((0, 2), dtype=float)
)
if categorized_bouts.size:
    finite_mask = (
        np.isfinite(categorized_bouts[:, 0])
        & np.isfinite(categorized_bouts[:, 1])
        & (categorized_bouts[:, 1] > categorized_bouts[:, 0])
    )
    categorized_bouts = categorized_bouts[finite_mask]
    if categorized_bouts.shape[0] > 1:
        categorized_bouts = np.unique(categorized_bouts, axis=0)
    categorized_bouts = categorized_bouts[np.argsort(categorized_bouts[:, 0])]
categorized_onsets = categorized_bouts[:, 0].copy() if categorized_bouts.size else np.array([], dtype=float)
categorized_durations = (
    categorized_bouts[:, 1] - categorized_bouts[:, 0]
    if categorized_bouts.size
    else np.array([], dtype=float)
)
wh_detect["all_bouts"] = categorized_bouts
wh_detect["all_onsets"] = categorized_onsets
wh_detect["all_durations"] = categorized_durations

wh_event_base = {
    "wh_brief_times": np.asarray(wh_detect.get("brief_onsets", np.array([])), dtype=float),
    "wh_long_times": np.asarray(wh_detect.get("long_onsets", np.array([])), dtype=float),
    "wh_all_times": np.asarray(wh_detect.get("all_onsets", np.array([])), dtype=float),
}

wh_loco = _classify_whisk_onsets_by_locomotion(
    wh_event_base["wh_all_times"],
    getattr(sl, "wheel", None),
    lookahead_s=CONFIG_CALC.get("WH_LOCO_LOOKAHEAD_S", 6.0),
    speed_thr_cm_s=CONFIG_CALC.get("WH_LOCO_SPEED_THR_CM_S", 1.0),
    wheel_radius_cm=CONFIG_CALC.get("WH_WHEEL_RADIUS_CM", 3.1),
)
wh_event_base["wh_all_times_loco"] = np.asarray(wh_loco["loco_onsets"], dtype=float)
wh_event_base["wh_all_times_non_loco"] = np.asarray(wh_loco["non_loco_onsets"], dtype=float)
if wh_loco["wheel_speed_cm_s"].size == 0:
    print("Whisk-locomotion tagging skipped: wheel speed unavailable.")
else:
    n_all_wh = len(wh_event_base["wh_all_times"])
    n_loco_wh = len(wh_event_base["wh_all_times_loco"])
    n_non_loco_wh = len(wh_event_base["wh_all_times_non_loco"])
    print(
        "Whisk-locomotion split "
        f"(>{CONFIG_CALC['WH_LOCO_SPEED_THR_CM_S']:.2f} cm/s in "
        f"{CONFIG_CALC['WH_LOCO_LOOKAHEAD_S']:.1f} s): "
        f"all={n_all_wh}, loco={n_loco_wh}, non_loco={n_non_loco_wh}"
    )

wh_events_by_period = ana_utils.split_wh_events_by_period(
    wh_event_base,
    spont_intervals=spont_intervals,
    task_windows=task_windows,
    iti_windows=iti_windows,
)

for base_name in ("wh_brief_times", "wh_long_times", "wh_all_times"):
    n_all = len(wh_events_by_period.get(base_name, []))
    n_spont = len(wh_events_by_period.get(f"{base_name}_spont", []))
    n_task = len(wh_events_by_period.get(f"{base_name}_task", []))
    n_iti = len(wh_events_by_period.get(f"{base_name}_iti", []))
    print(
        f"{base_name}: all={n_all}, spont={n_spont}, task={n_task}, iti={n_iti} "
        f"(split_sum={n_spont + n_task + n_iti})"
    )

events_by_name = dict(task_events_by_name)
contrasts_by_name = dict(task_contrasts_by_name)
trial_idx_by_name = dict(task_trial_idx_by_name)
for event_name, times in wh_events_by_period.items():
    arr = np.asarray(times, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = np.sort(arr)
    events_by_name[event_name] = arr
    contrasts_by_name[event_name] = np.ones(len(arr), dtype=float)
    trial_idx_by_name[event_name] = np.arange(len(arr), dtype=int)


print(
    "Whisk variables updated (df_wh, wh_detect, wh_events_by_period, events_by_name). "
    "Delay/stPR outputs were not recomputed."
)


# %% Delay + stPR calculations (slow; rerun after whisk settings are finalized)
_required_wh_vars = [
    "events_by_name",
    "contrasts_by_name",
    "trial_idx_by_name",
    "wh_events_by_period",
    "df_wh",
    "wh_detect",
    "wh_event_base",
    "task_windows",
    "iti_windows",
]
_missing_wh_vars = [name for name in _required_wh_vars if name not in globals()]
if _missing_wh_vars:
    raise RuntimeError(
        "Run the previous whisk-calculation cell first. Missing: "
        + ", ".join(_missing_wh_vars)
    )

wh_delay_names = [
    "wh_brief_times_spont",
    "wh_long_times_spont",
    "wh_all_times_spont",
]

DELAY_EVENT_NAMES = ["stimOn_times", "firstMovement_times", "feedback_times", *wh_delay_names]

delay_windows = dict(CONFIG_CALC.get("DELAY_WINDOWS", {}))
for event_name in wh_delay_names:
    delay_windows[event_name] = tuple(CONFIG_CALC["WH_DELAY_WINDOW"])

task_delay_names = ["stimOn_times", "firstMovement_times", "feedback_times"]
delay_methods_by_event = {
    event_name: str(CONFIG_CALC.get("DELAY_METHOD", "com"))
    for event_name in task_delay_names
}
for event_name in wh_delay_names:
    delay_methods_by_event[event_name] = str(CONFIG_CALC.get("WH_DELAY_METHOD", "tfs"))

delay_config = dict(CONFIG_CALC)
delay_config["DELAY_WINDOWS"] = delay_windows
delay_config["EVENT_NAMES"] = list(DELAY_EVENT_NAMES)
delay_config["DELAY_METHODS_BY_EVENT"] = dict(delay_methods_by_event)
df_res = ana_utils.calculate_event_delays(
    spikes,
    clusters,
    cluster_acronyms_calc,
    events_by_name,
    delay_config,
    target_region_cid_to_idx,
    contrasts_by_name=contrasts_by_name,
    trial_idx_by_name=trial_idx_by_name,
    include_splits=True,
    output_path=path_data_processed / f"{pid}_wh_delay_results.csv",
)

df_arousal = ana_utils.compute_arousal_groups_from_whisk(
    spikes,
    clusters,
    target_region_cluster_acronyms,
    target_region_cid_to_idx,
    target_region_cluster_ids,
    wh_events_by_period.get("wh_brief_times_spont", np.array([])),
    CONFIG_CALC,
    whisk_times=df_wh["bin_center_s"].to_numpy(dtype=float),
    whisk_values=df_wh["wh_norm"].to_numpy(dtype=float),
    spont_intervals=spont_intervals,
)
if df_arousal is not None and not df_arousal.empty:
    arousal_cols = [
        "cluster_id",
        "arousal_corr",
        "arousal_corr_h1",
        "arousal_corr_h2",
        "arousal_corr_abs",
        "arousal_mod",
        "arousal_fr_hz",
        "arousal_group",
        "arousal_n_events",
        "arousal_n_bins",
    ]
    df_res = df_res.merge(df_arousal[arousal_cols], on="cluster_id", how="left")
else:
    df_res["arousal_corr"] = np.nan
    df_res["arousal_corr_h1"] = np.nan
    df_res["arousal_corr_h2"] = np.nan
    df_res["arousal_corr_abs"] = np.nan
    df_res["arousal_mod"] = np.nan
    df_res["arousal_fr_hz"] = np.nan
    df_res["arousal_group"] = "neutral"
    df_res["arousal_n_events"] = 0
    df_res["arousal_n_bins"] = 0
arousal_group_mode = str(CONFIG_CALC.get("AROUSAL_GROUP_MODE", "corr")).strip().lower()
if arousal_group_mode not in {"corr", "response_sign"}:
    print(
        f"Unknown AROUSAL_GROUP_MODE='{arousal_group_mode}'. "
        "Falling back to 'corr'."
    )
    arousal_group_mode = "corr"
if arousal_group_mode == "response_sign":
    arousal_sign_event = str(
        CONFIG_CALC.get("AROUSAL_SIGN_EVENT", "wh_brief_times_spont")
    ).strip()
    arousal_sign_col = ana_utils.response_sign_column_name(arousal_sign_event)
    sign_to_group = {"exc": "arousal_plus", "inh": "arousal_minus", "none": "neutral"}
    if arousal_sign_col in df_res.columns:
        sign_vals = df_res[arousal_sign_col].astype(str).str.lower()
        df_res["arousal_group"] = sign_vals.map(sign_to_group).fillna("neutral")
    else:
        print(
            f"AROUSAL_GROUP_MODE='response_sign' requested but '{arousal_sign_col}' "
            "is missing. Keeping correlation-based arousal groups."
        )
df_res["arousal_group"] = df_res["arousal_group"].fillna("neutral")

wh_sort_delay_cols = [
    ana_utils.delay_column_name("wh_brief_times_spont"),
    ana_utils.delay_column_name("wh_long_times_spont"),
    ana_utils.delay_column_name("wh_all_times_spont"),
]
df_res = ana_utils.add_wh_delay_sorting(df_res, wh_sort_delay_cols, group_col="arousal_group")

calc_label_min = CONFIG_CALC.get("CALC_LABEL_MIN", None)
if calc_label_min is None and CONFIG_CALC.get("CALC_ONLY_GOOD_UNITS", False):
    calc_label_min = 1.0
calc_cluster_ids = _select_cluster_ids_by_label(
    cluster_ids,
    clusters,
    label_min=calc_label_min,
)
calc_cluster_ids = np.asarray(
    [cid for cid in calc_cluster_ids if int(cid) in target_region_cluster_id_set],
    dtype=np.asarray(cluster_ids).dtype,
)

df_coupling = None
if CONFIG_CALC.get("CALC_SPONT", True) and spont_interval_list:
    spikes_spont = ana_utils.slice_spikes_by_intervals(spikes, spont_interval_list)
    df_coupling = ana_utils.compute_population_coupling(
        spikes_spont,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=True,
        intervals=spont_interval_list,
        context_label="Spont",
    )

if not task_windows.empty:
    task_odd_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy()
    task_even_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy()
else:
    task_odd_intervals = np.empty((0, 2))
    task_even_intervals = np.empty((0, 2))

df_task_odd = None
df_task_even = None
if len(task_odd_intervals) > 0:
    spikes_task_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_task_odd = ana_utils.compute_population_coupling(
        spikes_task_odd,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=task_odd_intervals,
        context_label="Task odd",
    )
if len(task_even_intervals) > 0:
    spikes_task_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_task_even = ana_utils.compute_population_coupling(
        spikes_task_even,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=task_even_intervals,
        context_label="Task even",
    )
if df_task_odd is not None and df_task_odd.empty:
    df_task_odd = None
if df_task_even is not None and df_task_even.empty:
    df_task_even = None
df_coupling_task = ana_utils.merge_stpr_splits(
    df_task_odd,
    df_task_even,
    CONFIG_CALC,
    split_a="odd",
    split_b="even",
) if (df_task_odd is not None or df_task_even is not None) else None

if not iti_windows.empty:
    iti_odd_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy()
    iti_even_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy()
else:
    iti_odd_intervals = np.empty((0, 2))
    iti_even_intervals = np.empty((0, 2))

df_iti_odd = None
df_iti_even = None
if len(iti_odd_intervals) > 0:
    spikes_iti_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_iti_odd = ana_utils.compute_population_coupling(
        spikes_iti_odd,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=iti_odd_intervals,
        context_label="ITI odd",
    )
if len(iti_even_intervals) > 0:
    spikes_iti_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_iti_even = ana_utils.compute_population_coupling(
        spikes_iti_even,
        clusters,
        cluster_acronyms_calc,
        CONFIG_CALC,
        cluster_ids=calc_cluster_ids,
        split_halves=False,
        intervals=iti_even_intervals,
        context_label="ITI even",
    )
if df_iti_odd is not None and df_iti_odd.empty:
    df_iti_odd = None
if df_iti_even is not None and df_iti_even.empty:
    df_iti_even = None
df_coupling_iti = ana_utils.merge_stpr_splits(
    df_iti_odd,
    df_iti_even,
    CONFIG_CALC,
    split_a="odd",
    split_b="even",
) if (df_iti_odd is not None or df_iti_even is not None) else None

cluster_firing_rate = _get_cluster_firing_rate(clusters, cluster_ids)
df_firing_rate = None
if cluster_firing_rate is not None and cluster_ids is not None:
    df_firing_rate = pd.DataFrame(
        {
            "cluster_id": np.asarray(cluster_ids),
            "firing_rate_h1": np.asarray(cluster_firing_rate, dtype=float),
            "firing_rate_h2": np.asarray(cluster_firing_rate, dtype=float),
        }
    )

plot_cluster_ids = _select_cluster_ids_by_label(
    cluster_ids,
    clusters,
    label_min=PLOT_LABEL_MIN,
)
plot_cluster_ids = np.asarray(
    [cid for cid in plot_cluster_ids if int(cid) in target_region_cluster_id_set],
    dtype=np.asarray(cluster_ids).dtype,
)
plot_cluster_acronyms = np.asarray(
    [cluster_acronyms_plot[cid_to_idx[int(cid)]] for cid in plot_cluster_ids],
    dtype=str,
)
df_res_plot = df_res[df_res["cluster_id"].isin(plot_cluster_ids)].copy()
df_coupling_plot = (
    df_coupling[df_coupling["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling is not None
    else None
)
df_coupling_task_plot = (
    df_coupling_task[df_coupling_task["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling_task is not None
    else None
)
df_coupling_iti_plot = (
    df_coupling_iti[df_coupling_iti["cluster_id"].isin(plot_cluster_ids)].copy()
    if df_coupling_iti is not None
    else None
)
if df_firing_rate is not None:
    df_firing_rate = df_firing_rate[df_firing_rate["cluster_id"].isin(plot_cluster_ids)].copy()

print(f"Delay table shape: {df_res_plot.shape}")
if df_coupling_plot is not None:
    print(f"Spont stPR shape: {df_coupling_plot.shape}")
if df_coupling_task_plot is not None:
    print(f"Task stPR shape: {df_coupling_task_plot.shape}")
if df_coupling_iti_plot is not None:
    print(f"ITI stPR shape: {df_coupling_iti_plot.shape}")


# %% Whisk (normalized signal + detected bouts)
df_me_vis = df_me_raw[["times", "view", "value"]].copy()
df_me_vis = df_me_vis.dropna(subset=["times", "value"])
df_me_vis["times"] = df_me_vis["times"].astype(float)
df_me_vis["value"] = df_me_vis["value"].astype(float)
df_me_vis["view"] = df_me_vis["view"].astype(str)

view_curves = {}
use_bin_signal_qc = bool(CONFIG_CALC.get("WH_BIN_SIGNAL", True))
# Apply the same whisk scaling to right/left camera curves so all whisk plots are consistent.
wh_signal_scale = float(CONFIG_CALC.get("WH_SIGNAL_SCALE", 1.0))
if not df_me_vis.empty:
    norm_pctl = float(CONFIG_CALC["WH_NORM_PCTL"])
    norm_top_pctl = float(CONFIG_CALC.get("WH_NORM_TOP_PCTL", 100.0))
    if use_bin_signal_qc:
        bin_s = float(CONFIG_CALC["WH_BIN_S"])
        bin_reduce = str(CONFIG_CALC.get("WH_BIN_REDUCE", "mean")).lower()
        normalize_after_bin = bool(CONFIG_CALC.get("WH_NORMALIZE_AFTER_BIN", False))
        t0_global = float(df_me_vis["times"].min())
        for view_name, grp in df_me_vis.groupby("view", sort=False):
            vals = grp["value"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            tmp = grp.loc[finite, ["times", "value"]].copy()
            tmp["bin_idx"] = np.floor(
                (tmp["times"].to_numpy(dtype=float) - t0_global) / bin_s
            ).astype(int)

            if normalize_after_bin:
                grp_raw = tmp.groupby("bin_idx", as_index=False)["value"]
                if bin_reduce == "max":
                    agg_raw = grp_raw.max()
                elif bin_reduce == "median":
                    agg_raw = grp_raw.median()
                else:
                    agg_raw = grp_raw.mean()
                vals_for_norm = agg_raw["value"].to_numpy(dtype=float)
                if vals_for_norm.size == 0:
                    continue
                lo = float(np.nanpercentile(vals_for_norm, norm_pctl))
                hi = float(np.nanpercentile(vals_for_norm, norm_top_pctl))
                if not np.isfinite(lo):
                    lo = float(np.nanmin(vals_for_norm))
                if not np.isfinite(hi):
                    hi = float(np.nanmax(vals_for_norm))
                if hi <= lo:
                    hi = float(np.nanmax(vals_for_norm))
                if hi <= lo:
                    y_vals = np.zeros_like(vals_for_norm, dtype=float)
                else:
                    y_vals = np.clip((vals_for_norm - lo) / (hi - lo), 0.0, 1.0)
                x_vals = (
                    t0_global + agg_raw["bin_idx"].to_numpy(dtype=float) * bin_s + 0.5 * bin_s
                )
            else:
                vals_fin = vals[finite]
                lo = float(np.nanpercentile(vals_fin, norm_pctl))
                hi = float(np.nanpercentile(vals_fin, norm_top_pctl))
                if not np.isfinite(lo):
                    lo = float(np.nanmin(vals_fin))
                if not np.isfinite(hi):
                    hi = float(np.nanmax(vals_fin))
                if hi <= lo:
                    hi = float(np.nanmax(vals_fin))
                if hi <= lo:
                    norm_vals = np.zeros_like(vals_fin, dtype=float)
                else:
                    norm_vals = np.clip((vals_fin - lo) / (hi - lo), 0.0, 1.0)
                tmp["value_norm"] = norm_vals
                grp_norm = tmp.groupby("bin_idx", as_index=False)["value_norm"]
                if bin_reduce == "max":
                    agg_norm = grp_norm.max()
                elif bin_reduce == "median":
                    agg_norm = grp_norm.median()
                else:
                    agg_norm = grp_norm.mean()
                x_vals = (
                    t0_global + agg_norm["bin_idx"].to_numpy(dtype=float) * bin_s + 0.5 * bin_s
                )
                y_vals = agg_norm["value_norm"].to_numpy(dtype=float)

            view_curves[view_name] = (x_vals, np.asarray(y_vals, dtype=float) * wh_signal_scale)
    else:
        for view_name, grp in df_me_vis.groupby("view", sort=False):
            vals = grp["value"].to_numpy(dtype=float)
            times = grp["times"].to_numpy(dtype=float)
            finite = np.isfinite(times) & np.isfinite(vals)
            if not np.any(finite):
                continue
            vals_fin = vals[finite]
            times_fin = times[finite]
            lo = float(np.nanpercentile(vals_fin, norm_pctl))
            hi = float(np.nanpercentile(vals_fin, norm_top_pctl))
            if not np.isfinite(lo):
                lo = float(np.nanmin(vals_fin))
            if not np.isfinite(hi):
                hi = float(np.nanmax(vals_fin))
            if hi <= lo:
                hi = float(np.nanmax(vals_fin))
            if hi <= lo:
                norm_vals = np.zeros_like(vals_fin, dtype=float)
            else:
                norm_vals = np.clip((vals_fin - lo) / (hi - lo), 0.0, 1.0)
            order = np.argsort(times_fin)
            view_curves[view_name] = (times_fin[order], norm_vals[order] * wh_signal_scale)

if use_bin_signal_qc:
    bin_ms = 1000.0 * float(CONFIG_CALC["WH_BIN_S"])
    view_label_suffix = f" ({bin_ms:.0f} ms)"
    mean_label = f"Mean across available cameras ({bin_ms:.0f} ms)"
else:
    view_label_suffix = " (raw, no bin)"
    mean_label = "Mean across available cameras (raw, no bin)"

sig_t_all = df_wh["bin_center_s"].to_numpy(dtype=float)
sig_t_all = sig_t_all[np.isfinite(sig_t_all)]
if sig_t_all.size == 0:
    raise RuntimeError("Whisk QC plotting failed: no finite whisk timestamps.")
x_min_all = float(np.nanmin(sig_t_all))
x_max_all = float(np.nanmax(sig_t_all))
wh_zoom_start = x_min_all if GENERAL_RASTER_START is None else float(GENERAL_RASTER_START)
wh_zoom_end = x_max_all if GENERAL_RASTER_END is None else float(GENERAL_RASTER_END)
wh_zoom_start = float(np.clip(wh_zoom_start, x_min_all, x_max_all))
wh_zoom_end = float(np.clip(wh_zoom_end, x_min_all, x_max_all))
if wh_zoom_end <= wh_zoom_start:
    wh_zoom_start, wh_zoom_end = x_min_all, x_max_all

fig_wh = go.Figure()
for view_name, (x_vals, y_vals) in view_curves.items():
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    mask_plot = (
        np.isfinite(x_vals)
        & np.isfinite(y_vals)
        & (x_vals >= wh_zoom_start)
        & (x_vals <= wh_zoom_end)
    )
    if not np.any(mask_plot):
        continue
    fig_wh.add_trace(
        go.Scatter(
            x=x_vals[mask_plot],
            y=y_vals[mask_plot],
            mode="lines",
            line=dict(width=1.0, dash="dash"),
            opacity=0.7,
            name=f"{view_name} norm{view_label_suffix}",
        )
    )
mean_x = df_wh["bin_center_s"].to_numpy(dtype=float)
mean_y = df_wh["wh_norm"].to_numpy(dtype=float)
mean_mask = (
    np.isfinite(mean_x)
    & np.isfinite(mean_y)
    & (mean_x >= wh_zoom_start)
    & (mean_x <= wh_zoom_end)
)
fig_wh.add_trace(
    go.Scatter(
        x=mean_x[mean_mask],
        y=mean_y[mean_mask],
        mode="lines",
        line=dict(color="#1f77b4", width=2.2),
        name=mean_label,
    )
)
for x0, x1 in np.asarray(wh_detect.get("brief_bouts", np.empty((0, 2))), dtype=float):
    x0 = float(x0)
    x1 = float(x1)
    if x1 < wh_zoom_start or x0 > wh_zoom_end:
        continue
    fig_wh.add_vrect(
        x0=max(x0, wh_zoom_start),
        x1=min(x1, wh_zoom_end),
        fillcolor="rgba(23,190,207,0.20)",
        line_width=0,
        layer="below",
    )
for x0, x1 in np.asarray(wh_detect.get("long_bouts", np.empty((0, 2))), dtype=float):
    x0 = float(x0)
    x1 = float(x1)
    if x1 < wh_zoom_start or x0 > wh_zoom_end:
        continue
    fig_wh.add_vrect(
        x0=max(x0, wh_zoom_start),
        x1=min(x1, wh_zoom_end),
        fillcolor="rgba(214,39,40,0.17)",
        line_width=0,
        layer="below",
    )
for t_on in np.asarray(wh_event_base.get("wh_brief_times", np.array([])), dtype=float):
    t_on = float(t_on)
    if wh_zoom_start <= t_on <= wh_zoom_end:
        fig_wh.add_vline(x=t_on, line=dict(color="#17becf", dash="dot", width=1.1))
for t_on in np.asarray(wh_event_base.get("wh_long_times", np.array([])), dtype=float):
    t_on = float(t_on)
    if wh_zoom_start <= t_on <= wh_zoom_end:
        fig_wh.add_vline(x=t_on, line=dict(color="#d62728", dash="dash", width=1.1))
for t_on in np.asarray(wh_event_base.get("wh_all_times_loco", np.array([])), dtype=float):
    t_on = float(t_on)
    if wh_zoom_start <= t_on <= wh_zoom_end:
        fig_wh.add_vline(x=t_on, line=dict(color="#9467bd", dash="dashdot", width=1.2))

start_thr = float(CONFIG_CALC["WH_START_THR"])
end_thr = float(CONFIG_CALC["WH_END_THR"])
brief_lo = float(CONFIG_CALC["WH_BRIEF_RANGE_S"][0])
brief_hi = float(CONFIG_CALC["WH_BRIEF_RANGE_S"][1])
long_min = float(CONFIG_CALC["WH_LONG_MIN_S"])
fig_wh.add_trace(
    go.Scatter(
        x=[wh_zoom_start, wh_zoom_end],
        y=[start_thr, start_thr],
        mode="lines",
        line=dict(color="#2ca02c", dash="dot", width=1.5),
        name=f"Start threshold ({start_thr:.2f})",
    )
)
fig_wh.add_trace(
    go.Scatter(
        x=[wh_zoom_start, wh_zoom_end],
        y=[end_thr, end_thr],
        mode="lines",
        line=dict(color="#ff7f0e", dash="dash", width=1.5),
        name=f"End threshold ({end_thr:.2f})",
    )
)

# Legend-only traces for bout and onset styles.
fig_wh.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(23,190,207,0.35)"),
        name=f"Brief bouts ({brief_lo:g}-{brief_hi:g} s)",
        hoverinfo="skip",
    )
)
fig_wh.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(214,39,40,0.30)"),
        name=f"Long bouts (>{long_min:g} s)",
        hoverinfo="skip",
    )
)
fig_wh.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#17becf", dash="dot", width=1.4),
        name=f"Brief bout onsets ({brief_lo:g}-{brief_hi:g} s)",
        hoverinfo="skip",
    )
)
fig_wh.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#9467bd", dash="dashdot", width=1.4),
        name=(
            "Wh onsets with locomotion "
            f"(>{CONFIG_CALC['WH_LOCO_SPEED_THR_CM_S']:.1f} cm/s in "
            f"{CONFIG_CALC['WH_LOCO_LOOKAHEAD_S']:.0f} s)"
        ),
        hoverinfo="skip",
    )
)
fig_wh.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="#d62728", dash="dash", width=1.4),
        name=f"Long bout onsets (>{long_min:g} s)",
        hoverinfo="skip",
    )
)
fig_wh.update_layout(
    title="Whisking (normalized trace; mean across cameras + brief/long bouts)",
    xaxis_title="Time in session (s)",
    yaxis_title="Normalized whisk signal",
    template=plot_config["PLOTLY_TEMPLATE"],
    width=1280,
    height=420,
    margin=dict(l=70, r=40, t=70, b=120),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="left",
        x=0.0,
    ),
)
fig_wh.update_xaxes(range=[wh_zoom_start, wh_zoom_end])
fig_wh.update_yaxes(range=[-0.02, 1.05])
show_fig(fig_wh)


# %% Trial raster (wh brief + wh long overlays)
trial_sort_metric = RASTER_SORT_MAP.get(TRIAL_RASTER_SORT, "depth")
n_trials = len(sl.trials["stimOn_times"])
trial_idx = 80 # = int(np.clip(int(TRIAL_INDEX), 0, max(0, n_trials - 1)))
display(_build_trial_table(sl.trials, trial_idx))

wh_event_styles = {
    "wh_brief_times": ("Wh brief", "#17becf", "dot"),
    "wh_long_times": ("Wh long", "#d62728", "dash"),
}
wh_event_overlay = {
    "wh_brief_times": wh_events_by_period.get("wh_brief_times", np.array([])),
    "wh_long_times": wh_events_by_period.get("wh_long_times", np.array([])),
}
wh_event_span_styles = {
    "wh_brief_bouts": {"label": "Wh brief bouts", "color": "#17becf", "alpha": 0.20},
    "wh_long_bouts": {"label": "Wh long bouts", "color": "#d62728", "alpha": 0.17},
}
wh_event_spans = {
    "wh_brief_bouts": np.asarray(wh_detect.get("brief_bouts", np.empty((0, 2))), dtype=float),
    "wh_long_bouts": np.asarray(wh_detect.get("long_bouts", np.empty((0, 2))), dtype=float),
}

fig_trial = plot_trial_raster_plotly(
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    sl,
    plot_config,
    trial_idx,
    sorting_metric=trial_sort_metric,
    variability_metric="fano",
    df_res=df_res_plot,
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    df_coupling_iti=df_coupling_iti_plot,
    df_firing_rate=df_firing_rate,
    motion_mean_df=df_wh,
    extra_event_times=wh_event_overlay,
    extra_event_styles=wh_event_styles,
    extra_event_spans=wh_event_spans,
    extra_event_span_styles=wh_event_span_styles,
)
show_fig(fig_trial)


# %% General raster (wh brief + wh long overlays)
general_sort_metric = RASTER_SORT_MAP.get(GENERAL_RASTER_SORT, "depth")
spike_times = np.asarray(spikes["times"], dtype=float)
if spike_times.size == 0:
    raise RuntimeError("No spikes found for this PID.")
min_t = float(np.nanmin(spike_times))
max_t = float(np.nanmax(spike_times))
t_start = min_t if GENERAL_RASTER_START is None else float(GENERAL_RASTER_START)
t_end = min(min_t + 10.0, max_t) if GENERAL_RASTER_END is None else float(GENERAL_RASTER_END)
t_start = max(min_t, t_start)
t_end = min(max_t, t_end)

wh_event_styles = {
    "wh_brief_times": ("Wh brief", "#17becf", "dot"),
    "wh_long_times": ("Wh long", "#d62728", "dash"),
}
wh_event_overlay = {
    "wh_brief_times": wh_events_by_period.get("wh_brief_times", np.array([])),
    "wh_long_times": wh_events_by_period.get("wh_long_times", np.array([])),
}
wh_event_span_styles = {
    "wh_brief_bouts": {"label": "Wh brief bouts", "color": "#17becf", "alpha": 0.20},
    "wh_long_bouts": {"label": "Wh long bouts", "color": "#d62728", "alpha": 0.17},
}
wh_event_spans = {
    "wh_brief_bouts": np.asarray(wh_detect.get("brief_bouts", np.empty((0, 2))), dtype=float),
    "wh_long_bouts": np.asarray(wh_detect.get("long_bouts", np.empty((0, 2))), dtype=float),
}

fig_general = plot_time_window_raster_plotly(
    spikes,
    clusters,
    plot_cluster_ids,
    plot_cluster_acronyms,
    sl,
    plot_config,
    t_start,
    t_end,
    sorting_metric=general_sort_metric,
    variability_metric="fano",
    df_res=df_res_plot,
    df_coupling=df_coupling_plot,
    df_coupling_task=df_coupling_task_plot,
    df_coupling_iti=df_coupling_iti_plot,
    df_firing_rate=df_firing_rate,
    motion_mean_df=df_wh,
    extra_event_times=wh_event_overlay,
    extra_event_styles=wh_event_styles,
    extra_event_spans=wh_event_spans,
    extra_event_span_styles=wh_event_span_styles,
)
show_fig(fig_general)


# %% Response heatmaps (task + spontaneous whisk events)
heatmap_sort_mode = HEATMAP_SORT_MAP.get(HEATMAP_SORT, "delay:wh_all_times_spont")
heatmap_plot_config = dict(plot_config)
heatmap_plot_config["HEATMAP_PANEL_COLS"] = 4
heatmap_plot_config["POP_WINDOW_PRE"] = 0.1
heatmap_plot_config["POP_WINDOW_POST"] = 0.2
heatmap_plot_config["POP_NORMALIZE"] = False
heatmap_plot_config["POP_ZSCORE"] = True
heatmap_plot_config["POP_ZSCORE_SOURCE"] = str(
    CONFIG_CALC.get("RESPONSIVE_ZSCORE_SOURCE", "smooth")
).strip().lower()
heatmap_plot_config["POP_BASELINE_PRE"] = float(CONFIG_CALC.get("BASELINE_PRE", 0.2))
heatmap_plot_config["POP_ZMIN"] = -10.0
heatmap_plot_config["POP_ZMAX"] = 10.0
heatmap_plot_config["HEATMAP_SHOW_COLORBAR"] = True
heatmap_plot_config["POP_SPLIT_AROUSAL_WHISK"] = bool(
    plot_config.get("HEATMAP_GROUP_BY_AROUSAL", True)
)
heatmap_plot_config["POP_SPLIT_GROUP_ANY_EVENT"] = True
heatmap_plot_config["POP_AROUSAL_GROUP_COL"] = "arousal_group"
heatmap_plot_config["POP_GROUP_COL_BY_EVENT"] = {
    "stimOn_times": ana_utils.response_sign_column_name("stimOn_times"),
    "firstMovement_times": ana_utils.response_sign_column_name("firstMovement_times"),
    "feedback_times": ana_utils.response_sign_column_name("feedback_times"),
}
heatmap_plot_config["POP_WINDOWS_BY_EVENT"] = {
    "stimOn_times": (0.5, 1.0),
    "firstMovement_times": (0.5, 1.0),
    "feedback_times": (0.5, 1.0),
    "wh_long_offset_times_spont": (0.5, 2.0),
    "wh_all_times_spont": (0.5, 2.0),
    "wh_brief_times_spont": (0.5, 2.0),
    "wh_long_times_spont": (0.5, 2.0),
}

long_bouts_for_offsets = np.asarray(
    wh_detect.get("long_bouts", np.empty((0, 2))),
    dtype=float,
).reshape(-1, 2)
if long_bouts_for_offsets.size > 0:
    valid_long_offset = (
        np.isfinite(long_bouts_for_offsets[:, 0])
        & np.isfinite(long_bouts_for_offsets[:, 1])
        & (long_bouts_for_offsets[:, 1] > long_bouts_for_offsets[:, 0])
    )
    long_bouts_for_offsets = long_bouts_for_offsets[valid_long_offset]
    if long_bouts_for_offsets.size > 0:
        spont_arr = np.asarray(spont_intervals, dtype=float).reshape(-1, 2)
        long_onsets = long_bouts_for_offsets[:, 0]
        onset_spont_mask = np.zeros(long_onsets.shape[0], dtype=bool)
        for t0, t1 in spont_arr:
            if not np.isfinite(t0) or not np.isfinite(t1) or (t1 <= t0):
                continue
            onset_spont_mask |= (long_onsets >= float(t0)) & (long_onsets <= float(t1))
        long_bouts_for_offsets = long_bouts_for_offsets[onset_spont_mask]
    if long_bouts_for_offsets.size > 0:
        wh_long_offset_times_spont = np.sort(long_bouts_for_offsets[:, 1])
    else:
        wh_long_offset_times_spont = np.array([], dtype=float)
else:
    wh_long_offset_times_spont = np.array([], dtype=float)

heatmap_event_specs = [
    ("Stim On", "stimOn_times"),
    ("First Move", "firstMovement_times"),
    ("Feedback", "feedback_times"),
    ("Whisk / Arousal Counts (Spont)", None),
    ("Wh All (Spont)", "wh_all_times_spont"),
    ("Wh Brief (Spont)", "wh_brief_times_spont"),
    ("Wh Long (Spont)", "wh_long_times_spont"),
    ("Wh Long Offset (Spont)", "wh_long_offset_times_spont"),
]

event_time_lookup = dict(events_by_name)
event_time_lookup["wh_long_offset_times_spont"] = wh_long_offset_times_spont

event_sessions = {}
for _label, event_name in heatmap_event_specs:
    if not event_name:
        continue
    event_sessions[event_name] = _build_event_session(
        event_time_lookup.get(event_name, np.array([])),
        event_name,
    )

unique_regions = sorted(pd.Series(plot_cluster_acronyms).astype(str).unique().tolist())
plot_regions = plot_config.get("PLOT_REGIONS")
if plot_regions:
    selected_regions = []
    for region_prefix in plot_regions:
        matches = [reg for reg in unique_regions if str(reg).startswith(str(region_prefix))]
        if matches:
            selected_regions.extend(matches)
    selected_regions = list(dict.fromkeys(selected_regions))
else:
    selected_regions = unique_regions

if not selected_regions:
    print("No plotting regions matched current filters.")
else:
    for region_name in selected_regions:
        fig_panel = _build_multi_event_population_panel(
            heatmap_event_specs,
            event_sessions,
            spikes,
            clusters,
            plot_cluster_ids,
            plot_cluster_acronyms,
            df_res_plot,
            heatmap_plot_config,
            sort_mode=heatmap_sort_mode,
            region_name=region_name,
            df_coupling=df_coupling_plot,
            df_coupling_task=df_coupling_task_plot,
            df_coupling_iti=df_coupling_iti_plot,
            df_firing_rate=df_firing_rate,
            whisk_df=df_wh,
        )
        show_fig(fig_panel)


# %% Debug grouped z-PSTHs helper (exact traces + thresholds used for categorization)
def _debug_grouped_z_psths(
    region_name,
    event_name,
    df_res_debug,
    group_prefix="Arousal",
    compare_with_arousal_group=False,
):
    event_name = str(event_name).strip()
    sign_col = ana_utils.response_sign_column_name(event_name)
    group_map = {"exc": "arousal_plus", "inh": "arousal_minus", "none": "neutral"}
    use_zscore_dbg = bool(CONFIG_CALC.get("RESPONSIVE_USE_ZSCORE", False))
    zscore_source_dbg = str(
        CONFIG_CALC.get("RESPONSIVE_ZSCORE_SOURCE", "smooth")
    ).strip().lower()
    z_thr_dbg = float(CONFIG_CALC.get("RESPONSIVE_Z_THR", 2.0))

    if not use_zscore_dbg:
        print(
            "[Debug PSTH] WARNING: RESPONSIVE_USE_ZSCORE is False. "
            "Plots remain z-scored for inspection, but categorization is not."
        )

    df_debug = df_res_debug.copy()
    if sign_col in df_debug.columns:
        sign_vals_dbg = df_debug[sign_col].astype(str).str.strip().str.lower()
        df_debug["group_from_sign"] = sign_vals_dbg.map(group_map).fillna("neutral")
    else:
        print(
            f"[Debug PSTH] Warning: '{sign_col}' missing. "
            "Falling back to arousal_group if available."
        )
        if "arousal_group" in df_debug.columns:
            df_debug["group_from_sign"] = (
                df_debug["arousal_group"].astype(str).str.strip().str.lower().fillna("neutral")
            )
        else:
            df_debug["group_from_sign"] = "neutral"

    if compare_with_arousal_group and "arousal_group" in df_debug.columns:
        arousal_orig = (
            df_debug["arousal_group"].astype(str).str.strip().str.lower().fillna("neutral")
        )
        mismatch_mask = arousal_orig != df_debug["group_from_sign"]
        n_mismatch = int(mismatch_mask.sum())
        print(
            f"[Debug PSTH] mismatch (stored arousal_group vs sign-derived) "
            f"{n_mismatch}/{len(df_debug)}"
        )

    region_cluster_ids_dbg = np.asarray(
        [
            int(cid)
            for cid, reg in zip(plot_cluster_ids, plot_cluster_acronyms)
            if str(reg) == str(region_name)
        ],
        dtype=int,
    )
    if region_cluster_ids_dbg.size == 0:
        region_cluster_ids_dbg = np.asarray(
            [
                int(cid)
                for cid, reg in zip(plot_cluster_ids, plot_cluster_acronyms)
                if str(reg).startswith(str(region_name))
            ],
            dtype=int,
        )

    if region_cluster_ids_dbg.size == 0:
        print(f"[Debug PSTH] No neurons found for region '{region_name}'.")
        return

    events_dbg = np.asarray(events_by_name.get(event_name, np.array([])), dtype=float)
    events_dbg = events_dbg[np.isfinite(events_dbg)]
    if events_dbg.size == 0:
        print(f"[Debug PSTH] No finite events found for '{event_name}'.")
        return

    delay_cfg_ref = delay_config if "delay_config" in globals() else CONFIG_CALC
    win_start_dbg, win_end_dbg = ana_utils.get_event_delay_window(delay_cfg_ref, event_name)
    psth_by_cluster_dbg, bin_centers_dbg = ana_utils.compute_psth_for_clusters(
        spikes,
        region_cluster_ids_dbg,
        events_dbg,
        CONFIG_CALC["PSTH_WINDOW_START"],
        CONFIG_CALC["PSTH_WINDOW_END"],
        CONFIG_CALC["BIN_SIZE"],
        CONFIG_CALC["SMOOTH_SIGMA"],
        show_progress=False,
        desc=f"Debug PSTH ({event_name})",
    )
    if bin_centers_dbg is None or len(bin_centers_dbg) == 0:
        print("[Debug PSTH] Could not compute PSTH bin centers.")
        return

    bin_centers_dbg = np.asarray(bin_centers_dbg, dtype=float)
    idx_baseline_dbg = (
        (bin_centers_dbg >= -float(CONFIG_CALC["BASELINE_PRE"]))
        & (bin_centers_dbg < 0)
    )
    if not np.any(idx_baseline_dbg):
        print("[Debug PSTH] Baseline window is empty.")
        return

    df_region_dbg = df_debug[
        pd.to_numeric(df_debug["cluster_id"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .isin(region_cluster_ids_dbg)
    ].copy()
    group_specs_dbg = [
        ("arousal_plus", f"{group_prefix} +", "#1f77b4"),
        ("neutral", "Neutral", "#7f7f7f"),
        ("arousal_minus", f"{group_prefix} -", "#ff7f0e"),
    ]

    print(
        f"[Debug PSTH] Region={region_name} | Event={event_name} | "
        f"n_events={len(events_dbg)} | use_zscore={use_zscore_dbg} | "
        f"z_source={zscore_source_dbg} | z_thr={z_thr_dbg:.2f} | "
        f"window=({win_start_dbg:.3f}, {win_end_dbg:.3f}) s"
    )

    for group_key_dbg, group_label_dbg, color_dbg in group_specs_dbg:
        cids_dbg = (
            df_region_dbg.loc[
                df_region_dbg["group_from_sign"].astype(str).str.strip().str.lower()
                == group_key_dbg,
                "cluster_id",
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        if len(cids_dbg) == 0:
            print(f"[Debug PSTH] {group_label_dbg}: no neurons in {region_name}.")
            continue

        fig_dbg = go.Figure()
        traces_z = []
        used_cids = []
        for cid_dbg in cids_dbg:
            psth_entry_dbg = psth_by_cluster_dbg.get(int(cid_dbg))
            if psth_entry_dbg is None:
                continue
            fr_raw_dbg = np.asarray(
                psth_entry_dbg.get("fr_raw", np.array([])),
                dtype=float,
            ).reshape(-1)
            fr_smooth_dbg = np.asarray(
                psth_entry_dbg.get("fr_smooth", np.array([])),
                dtype=float,
            ).reshape(-1)
            if fr_raw_dbg.size != bin_centers_dbg.size:
                continue
            if fr_smooth_dbg.size != bin_centers_dbg.size:
                fr_smooth_dbg = fr_raw_dbg.copy()

            if use_zscore_dbg and zscore_source_dbg == "raw":
                z_src_dbg = fr_raw_dbg
            else:
                z_src_dbg = fr_smooth_dbg

            baseline_dbg = np.asarray(z_src_dbg[idx_baseline_dbg], dtype=float)
            baseline_dbg = baseline_dbg[np.isfinite(baseline_dbg)]
            if baseline_dbg.size == 0:
                continue

            base_mean_dbg = float(np.mean(baseline_dbg))
            base_std_dbg = float(np.std(baseline_dbg))
            if (not np.isfinite(base_std_dbg)) or base_std_dbg <= 0:
                continue
            z_trace_dbg = (np.asarray(z_src_dbg, dtype=float) - base_mean_dbg) / base_std_dbg

            traces_z.append(z_trace_dbg)
            used_cids.append(int(cid_dbg))
            fig_dbg.add_trace(
                go.Scatter(
                    x=bin_centers_dbg,
                    y=z_trace_dbg,
                    mode="lines",
                    line=dict(color=color_dbg, width=1),
                    opacity=0.16,
                    showlegend=False,
                    hovertemplate=(
                        f"cid={int(cid_dbg)}<br>"
                        "t=%{x:.3f}s<br>"
                        "z=%{y:.3f}<extra></extra>"
                    ),
                )
            )

        if len(traces_z) == 0:
            print(f"[Debug PSTH] {group_label_dbg}: no valid z-scored PSTH traces.")
            continue

        mat_z_dbg = np.vstack(traces_z)
        mean_z_dbg = np.nanmean(mat_z_dbg, axis=0)
        fig_dbg.add_trace(
            go.Scatter(
                x=bin_centers_dbg,
                y=mean_z_dbg,
                mode="lines",
                line=dict(color=color_dbg, width=3),
                name=f"{group_label_dbg} mean z-PSTH",
            )
        )
        fig_dbg.add_hline(
            y=float(z_thr_dbg),
            line=dict(color="black", width=2, dash="dash"),
            annotation_text=f"+z threshold ({z_thr_dbg:.1f})",
            annotation_position="top right",
        )
        fig_dbg.add_hline(
            y=-float(z_thr_dbg),
            line=dict(color="black", width=2, dash="dash"),
            annotation_text=f"-z threshold ({z_thr_dbg:.1f})",
            annotation_position="bottom right",
        )
        fig_dbg.add_vrect(
            x0=float(win_start_dbg),
            x1=float(win_end_dbg),
            fillcolor="gray",
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        fig_dbg.add_vline(x=0, line=dict(color="black", dash="dash"))
        fig_dbg.update_layout(
            title=(
                f"{group_label_dbg} neurons | Region {region_name} | "
                f"Event {ana_utils.event_label(event_name)}<br>"
                f"(n={len(used_cids)}; z-scored PSTH traces used for categorization)"
            ),
            template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
            width=1000,
            height=480,
            margin=dict(l=70, r=40, t=90, b=70),
        )
        fig_dbg.update_xaxes(title_text=f"Time from {ana_utils.event_label(event_name)} (s)")
        fig_dbg.update_yaxes(title_text="Baseline z-score")
        show_fig(fig_dbg)


# %% Debug arousal grouping PSTHs (ProS, exact z-scored traces + thresholds used for categorization)
_debug_grouped_z_psths(
    region_name="ProS",
    event_name=str(CONFIG_CALC.get("AROUSAL_SIGN_EVENT", "wh_brief_times_spont")).strip(),
    df_res_debug=df_res_plot,
    group_prefix="Arousal",
    compare_with_arousal_group=True,
)


# %% Debug event grouping PSTHs (VISp, Stim On; exact z-scored traces + thresholds used for categorization)
_debug_grouped_z_psths(
    region_name="VISp",
    event_name="stimOn_times",
    df_res_debug=df_res_plot,
    group_prefix="Response",
    compare_with_arousal_group=False,
)


# %% Correlation matrices (Pearson + Spearman)
CORR_VARIABLE_SPECS = [
    {
        "key": "delay_stim",
        "name": "Delay (Stim On)",
        "df": "df_res",
        "v1": "delay_stimOn_times_odd",
        "v2": "delay_stimOn_times_even",
    },
    {
        "key": "delay_move",
        "name": "Delay (First Move)",
        "df": "df_res",
        "v1": "delay_firstMovement_times_odd",
        "v2": "delay_firstMovement_times_even",
    },
    {
        "key": "delay_feedback",
        "name": "Delay (Feedback)",
        "df": "df_res",
        "v1": "delay_feedback_times_odd",
        "v2": "delay_feedback_times_even",
    },
    {
        "key": "delay_wh_brief",
        "name": "Delay (Wh Brief)",
        "df": "df_res",
        "v1": "delay_wh_brief_times_spont_odd",
        "v2": "delay_wh_brief_times_spont_even",
    },
    {
        "key": "delay_wh_long",
        "name": "Delay (Wh Long)",
        "df": "df_res",
        "v1": "delay_wh_long_times_odd",
        "v2": "delay_wh_long_times_even",
    },
    {
        "key": "delay_wh_all",
        "name": "Delay (Wh All)",
        "df": "df_res",
        "v1": "delay_wh_all_times_odd",
        "v2": "delay_wh_all_times_even",
    },
    {
        "key": "stpr_delay_spont",
        "name": "stPR Delay (Spont)",
        "df": "df_coupling",
        "v1": "coupling_delay_ms_h1",
        "v2": "coupling_delay_ms_h2",
    },
    {
        "key": "stpr_delay_task",
        "name": "stPR Delay (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "key": "stpr_delay_iti",
        "name": "stPR Delay (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "key": "stpr_strength_spont",
        "name": "stPR Strength (Spont)",
        "df": "df_coupling",
        "v1": "coupling_strength_h1",
        "v2": "coupling_strength_h2",
    },
    {
        "key": "stpr_strength_task",
        "name": "stPR Strength (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "key": "stpr_strength_iti",
        "name": "stPR Strength (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
]

source_tables = {
    "df_res": df_res_plot,
    "df_coupling": df_coupling_plot,
    "df_coupling_task": df_coupling_task_plot,
    "df_coupling_iti": df_coupling_iti_plot,
}

region_lookup = pd.DataFrame(
    {
        "cluster_id": plot_cluster_ids,
        "region": plot_cluster_acronyms,
    }
)
region_lookup = region_lookup[~region_lookup["region"].isin(["root", "void"])].copy()


def _build_corr_variable_table(df_src, spec, region_df):
    if df_src is None:
        return None
    v1 = spec["v1"]
    v2 = spec["v2"]
    if v1 not in df_src.columns or v2 not in df_src.columns:
        return None
    df_var = df_src[["cluster_id", v1, v2]].copy()
    df_var = df_var.groupby("cluster_id", as_index=False).mean(numeric_only=True)
    if region_df is not None:
        df_var = df_var.merge(region_df[["cluster_id", "region"]], on="cluster_id", how="inner")
    vals_1 = df_var[v1].to_numpy(dtype=float)
    vals_2 = df_var[v2].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(vals_1) & np.isfinite(vals_2)
    mean_vals[valid] = (vals_1[valid] + vals_2[valid]) / 2.0
    df_var = df_var.rename(columns={v1: "half_1", v2: "half_2"})
    df_var["mean"] = mean_vals
    return df_var


if region_lookup.empty:
    print("No neurons available for correlation analysis.")
else:
    var_tables_all = {}
    for spec in CORR_VARIABLE_SPECS:
        df_var_all = _build_corr_variable_table(
            source_tables.get(spec["df"]),
            spec,
            region_lookup,
        )
        if df_var_all is not None and not df_var_all.empty:
            var_tables_all[spec["name"]] = df_var_all

    for region_name in selected_regions:
        region_ids = region_lookup.loc[region_lookup["region"] == region_name, "cluster_id"].to_numpy()
        if len(region_ids) == 0:
            continue

        var_tables_region = {}
        for spec in CORR_VARIABLE_SPECS:
            name = spec["name"]
            df_var_all = var_tables_all.get(name)
            if df_var_all is None:
                continue
            df_var_reg = df_var_all[df_var_all["region"] == region_name].copy()
            if not df_var_reg.empty:
                var_tables_region[name] = df_var_reg

        available_specs = [spec for spec in CORR_VARIABLE_SPECS if spec["name"] in var_tables_region]
        available_names = [spec["name"] for spec in available_specs]
        if len(available_names) < 2:
            print(f"Region {region_name}: not enough variables with finite data.")
            continue

        rel_pearson = {}
        rel_spearman = {}
        rel_n_pearson = {}
        rel_n_spearman = {}
        for spec in available_specs:
            name = spec["name"]
            df_var = var_tables_region[name]
            r_p, n_p = _pearsonr_with_n(df_var["half_1"], df_var["half_2"], min_n=CORR_MIN_N)
            r_s, n_s = _spearmanr_with_n(df_var["half_1"], df_var["half_2"], min_n=CORR_MIN_N)
            rel_pearson[name] = r_p
            rel_spearman[name] = r_s
            rel_n_pearson[name] = n_p
            rel_n_spearman[name] = n_s

        mean_wide = pd.DataFrame({"cluster_id": region_ids})
        for spec in available_specs:
            name = spec["name"]
            mean_wide = mean_wide.merge(
                var_tables_region[name][["cluster_id", "mean"]].rename(columns={"mean": name}),
                on="cluster_id",
                how="left",
            )

        n_vars = len(available_names)
        pearson_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
        n_mat_p = np.zeros((n_vars, n_vars), dtype=int)
        spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
        n_mat_s = np.zeros((n_vars, n_vars), dtype=int)

        for i, name_i in enumerate(available_names):
            for j, name_j in enumerate(available_names):
                if i == j:
                    pearson_mat[i, j] = rel_pearson.get(name_i, np.nan)
                    n_mat_p[i, j] = int(rel_n_pearson.get(name_i, 0))
                    spearman_mat[i, j] = rel_spearman.get(name_i, np.nan)
                    n_mat_s[i, j] = int(rel_n_spearman.get(name_i, 0))
                else:
                    x = mean_wide[name_i].to_numpy(dtype=float)
                    y = mean_wide[name_j].to_numpy(dtype=float)
                    r_p, n_p = _pearsonr_with_n(x, y, min_n=CORR_MIN_N)
                    r_s, n_s = _spearmanr_with_n(x, y, min_n=CORR_MIN_N)
                    pearson_mat[i, j] = r_p
                    n_mat_p[i, j] = int(n_p)
                    spearman_mat[i, j] = r_s
                    n_mat_s[i, j] = int(n_s)

        fig_p = _build_corr_heatmap_fig(
            pearson_mat,
            n_mat_p,
            available_names,
            (
                "Reliability (diag) + Pairwise Pearson (off-diag) | "
                f"Region {region_name} | N units={len(region_ids)} | min_n={CORR_MIN_N}"
            ),
            plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_p)

        fig_s = _build_corr_heatmap_fig(
            spearman_mat,
            n_mat_s,
            available_names,
            (
                "Reliability (diag) + Pairwise Spearman (off-diag) | "
                f"Region {region_name} | N units={len(region_ids)} | min_n={CORR_MIN_N}"
            ),
            plot_config["PLOTLY_TEMPLATE"],
        )
        show_fig(fig_s)


# %% Variable correlation scatter (interactive)
if region_lookup.empty:
    print("No neurons available for variable correlation scatter.")
else:
    scatter_regions = (
        selected_regions
        if "selected_regions" in globals() and len(selected_regions) > 0
        else sorted(region_lookup["region"].unique().tolist())
    )
    region_lookup_scatter = region_lookup[region_lookup["region"].isin(scatter_regions)].copy()
    merged_scatter = region_lookup_scatter[["cluster_id", "region"]].drop_duplicates().copy()

    available_specs_scatter = []
    for spec in CORR_VARIABLE_SPECS:
        df_var = _build_corr_variable_table(
            source_tables.get(spec["df"]),
            spec,
            region_lookup_scatter,
        )
        if df_var is None or df_var.empty:
            continue
        name = spec["name"]
        tmp = df_var[["cluster_id", "mean"]].rename(columns={"mean": name})
        merged_scatter = merged_scatter.merge(tmp, on="cluster_id", how="left")
        if np.isfinite(tmp[name].to_numpy(dtype=float)).any():
            available_specs_scatter.append(spec)

    available_var_names = [spec["name"] for spec in available_specs_scatter if spec["name"] in merged_scatter.columns]
    if len(available_var_names) < 2:
        print("Not enough variables with finite data to draw a scatter plot.")
    else:
        default_x = "Delay (Wh All)"
        default_y = "stPR Delay (Task)"
        if default_x not in available_var_names:
            default_x = available_var_names[0]
        if default_y not in available_var_names:
            default_y = available_var_names[1] if len(available_var_names) > 1 else available_var_names[0]

        var_x_name = _choose_option(
            available_var_names,
            "Variable X options:",
            default_x,
        )
        var_y_name = _choose_option(
            available_var_names,
            "Variable Y options:",
            default_y,
        )
        if var_y_name == var_x_name:
            fallback = [name for name in available_var_names if name != var_x_name]
            if fallback:
                var_y_name = fallback[0]
                print(f"Variable Y matched X; using '{var_y_name}' for Y.")

        plot_df = merged_scatter[["cluster_id", "region", var_x_name, var_y_name]].copy()
        x_vals = plot_df[var_x_name].to_numpy(dtype=float)
        y_vals = plot_df[var_y_name].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        plot_df = plot_df.loc[valid_mask].reset_index(drop=True)

        if plot_df.empty:
            print(f"No overlapping finite values for '{var_x_name}' and '{var_y_name}'.")
        else:
            x_vals = plot_df[var_x_name].to_numpy(dtype=float)
            y_vals = plot_df[var_y_name].to_numpy(dtype=float)
            r_p, n_p = _pearsonr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)
            r_s, n_s = _spearmanr_with_n(x_vals, y_vals, min_n=CORR_MIN_N)
            region_names = sorted(plot_df["region"].astype(str).unique().tolist())
            region_color_map = plotting_utils._region_color_map(region_names)

            fig_corr = go.Figure()
            for reg_name in region_names:
                reg_df = plot_df.loc[plot_df["region"].astype(str) == str(reg_name)].copy()
                if reg_df.empty:
                    continue
                fig_corr.add_trace(
                    go.Scatter(
                        x=reg_df[var_x_name].to_numpy(dtype=float),
                        y=reg_df[var_y_name].to_numpy(dtype=float),
                        mode="markers",
                        name=str(reg_name),
                        customdata=reg_df["cluster_id"].to_numpy(dtype=int),
                        marker=dict(
                            size=7,
                            opacity=0.7,
                            color=region_color_map.get(str(reg_name), "gray"),
                        ),
                        hovertemplate=(
                            "Region: %{fullData.name}<br>"
                            "Cluster ID: %{customdata}<br>"
                            f"{var_x_name}: %{{x:.3f}}<br>"
                            f"{var_y_name}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
            max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
            if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                fig_corr.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Unity line (y=x)",
                        line=dict(color="red", dash="dash", width=2),
                        hovertemplate="Unity line<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                    )
                )

            fit_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_fit = x_vals[fit_mask]
            y_fit = y_vals[fit_mask]
            if x_fit.size >= 2 and np.nanstd(x_fit) > 0:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                x_line = np.array([float(np.nanmin(x_fit)), float(np.nanmax(x_fit))], dtype=float)
                y_line = slope * x_line + intercept
                fig_corr.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"Linear fit (y={slope:.2f}x+{intercept:.2f})",
                        line=dict(color="black", width=2),
                        hovertemplate="Linear fit<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                    )
                )

            fig_corr.update_layout(
                title=f"Variable Correlation | {var_x_name} vs {var_y_name}",
                template=plot_config.get("PLOTLY_TEMPLATE", "plotly_white"),
                width=900,
                height=700,
                margin=dict(l=70, r=40, t=95, b=140),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                ),
            )
            fig_corr.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="left",
                bordercolor="rgba(120,120,120,0.6)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                text=(
                    f"Pearson r={_format_corr_value(r_p)} (n={n_p})<br>"
                    f"Spearman rho={_format_corr_value(r_s)} (n={n_s})"
                ),
            )
            fig_corr.update_xaxes(title_text=var_x_name)
            fig_corr.update_yaxes(title_text=var_y_name)
            show_fig(fig_corr)


# %% Export annotated left+right camera clip for the general raster window
try:
    import cv2
    import ibllib.io.video as vidio
except Exception as exc:
    print(f"Skipping camera clip export (missing dependency): {exc}")
else:
    try:
        right_times_all = np.asarray(
            one.load_dataset(eid, "*rightCamera.times*", collection="alf"),
            dtype=float,
        ).reshape(-1)
        left_times_all = np.asarray(
            one.load_dataset(eid, "*leftCamera.times*", collection="alf"),
            dtype=float,
        ).reshape(-1)

        right_valid_mask = np.isfinite(right_times_all)
        right_valid_frame_idx = np.flatnonzero(right_valid_mask)
        right_valid_times = right_times_all[right_valid_mask]
        if right_valid_times.size == 0:
            raise RuntimeError("No finite right-camera timestamps found.")

        left_valid_mask = np.isfinite(left_times_all)
        left_valid_frame_idx = np.flatnonzero(left_valid_mask)
        left_valid_times = left_times_all[left_valid_mask]
        if left_valid_times.size == 0:
            raise RuntimeError("No finite left-camera timestamps found.")

        overlap_min = max(float(right_valid_times.min()), float(left_valid_times.min()))
        overlap_max = min(float(right_valid_times.max()), float(left_valid_times.max()))
        if overlap_max <= overlap_min:
            raise RuntimeError("No overlapping left/right camera timestamps found.")

        clip_t_start = (
            overlap_min
            if GENERAL_RASTER_START is None
            else float(GENERAL_RASTER_START)
        )
        clip_t_end = (
            min(clip_t_start + 10.0, overlap_max)
            if GENERAL_RASTER_END is None
            else float(GENERAL_RASTER_END)
        )
        clip_t_start = max(overlap_min, clip_t_start)
        clip_t_end = min(overlap_max, clip_t_end)
        if clip_t_end <= clip_t_start:
            raise RuntimeError(
                f"Invalid clip window [{clip_t_start:.3f}, {clip_t_end:.3f}] s."
            )

        start_pos = int(np.searchsorted(right_valid_times, clip_t_start, side="left"))
        end_pos = int(np.searchsorted(right_valid_times, clip_t_end, side="right"))
        frame_numbers = right_valid_frame_idx[start_pos:end_pos]
        frame_times = right_valid_times[start_pos:end_pos]
        if frame_numbers.size == 0:
            raise RuntimeError(
                f"No right-camera frames found in [{clip_t_start:.3f}, {clip_t_end:.3f}] s."
            )

        try:
            right_url = vidio.url_from_eid(eid, label="right", one=one)
        except TypeError:
            right_url = vidio.url_from_eid(eid, one=one)["right"]
        try:
            left_url = vidio.url_from_eid(eid, label="left", one=one)
        except TypeError:
            left_url = vidio.url_from_eid(eid, one=one)["left"]

        try:
            meta_right = vidio.get_video_meta(right_url, one=one)
        except TypeError:
            meta_right = vidio.get_video_meta(right_url)
        fps = float(meta_right.get("fps", np.nan))
        if not np.isfinite(fps) or fps <= 0:
            dt = np.diff(right_valid_times)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            fps = float(1.0 / np.nanmedian(dt)) if dt.size else 30.0

        # Use the same averaged whisk signal shown in the Whisk QC trace.
        sig_t_all = np.asarray(df_wh["bin_center_s"], dtype=float)
        sig_v_all = np.asarray(df_wh["wh_norm"], dtype=float)
        sig_mask = (
            np.isfinite(sig_t_all)
            & np.isfinite(sig_v_all)
            & (sig_t_all >= clip_t_start)
            & (sig_t_all <= clip_t_end)
        )
        signal_t = sig_t_all[sig_mask]
        signal_v = sig_v_all[sig_mask]
        if signal_t.size == 0:
            signal_t = np.array([clip_t_start, clip_t_end], dtype=float)
            signal_v = np.array([0.0, 0.0], dtype=float)
        signal_v = np.nan_to_num(signal_v, nan=0.0, posinf=0.0, neginf=0.0)
        signal_v = np.maximum(signal_v, 0.0)
        signal_peak = float(np.nanmax(signal_v)) if signal_v.size else 0.0
        signal_y_max = 2.0 * signal_peak if signal_peak > 0 else 1.0

        wheel_src = globals().get("wh_loco", {})
        paw_t_all = np.asarray(wheel_src.get("wheel_times", np.array([])), dtype=float).reshape(-1)
        paw_v_all = np.asarray(wheel_src.get("wheel_speed_cm_s", np.array([])), dtype=float).reshape(-1)
        paw_mask = (
            np.isfinite(paw_t_all)
            & np.isfinite(paw_v_all)
            & (paw_t_all >= clip_t_start)
            & (paw_t_all <= clip_t_end)
        )
        paw_t = paw_t_all[paw_mask]
        paw_v = paw_v_all[paw_mask]
        if paw_t.size == 0:
            paw_t = np.array([clip_t_start, clip_t_end], dtype=float)
            paw_v = np.array([0.0, 0.0], dtype=float)
        paw_v = np.nan_to_num(paw_v, nan=0.0, posinf=0.0, neginf=0.0)
        paw_v = np.maximum(paw_v, 0.0)
        paw_peak = float(np.nanmax(paw_v)) if paw_v.size else 0.0
        paw_y_max = 2.0 * paw_peak if paw_peak > 0 else 1.0

        brief_bouts = np.asarray(
            wh_detect.get("brief_bouts", np.empty((0, 2))),
            dtype=float,
        ).reshape(-1, 2)
        long_bouts = np.asarray(
            wh_detect.get("long_bouts", np.empty((0, 2))),
            dtype=float,
        ).reshape(-1, 2)
        brief_onsets = np.asarray(
            wh_event_base.get("wh_brief_times", np.array([])),
            dtype=float,
        ).reshape(-1)
        long_onsets = np.asarray(
            wh_event_base.get("wh_long_times", np.array([])),
            dtype=float,
        ).reshape(-1)

        out_dir = path_data_processed / "video_clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        start_tag = f"{clip_t_start:.3f}".replace(".", "p")
        end_tag = f"{clip_t_end:.3f}".replace(".", "p")
        clip_path = out_dir / f"{pid}_right_left_{start_tag}_{end_tag}_annotated.mp4"

        def _time_to_x(t_val, left, plot_w):
            rel = (float(t_val) - clip_t_start) / max(clip_t_end - clip_t_start, 1e-9)
            rel = float(np.clip(rel, 0.0, 1.0))
            return int(round(left + rel * (plot_w - 1)))

        def _blend_rect(img, x0, x1, y0, y1, color_bgr, alpha=0.25):
            x0 = int(max(0, min(x0, img.shape[1] - 1)))
            x1 = int(max(0, min(x1, img.shape[1] - 1)))
            y0 = int(max(0, min(y0, img.shape[0] - 1)))
            y1 = int(max(0, min(y1, img.shape[0] - 1)))
            if x1 < x0 or y1 < y0:
                return
            roi = img[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
            color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
            mixed = (1.0 - float(alpha)) * roi + float(alpha) * color
            img[y0 : y1 + 1, x0 : x1 + 1] = np.clip(mixed, 0, 255).astype(np.uint8)

        def _draw_dashed_vertical(img, x, y0, y1, color, thickness=1, dash=6, gap=5):
            x = int(x)
            y0 = int(y0)
            y1 = int(y1)
            for y in range(y0, y1 + 1, dash + gap):
                y_end = min(y + dash, y1)
                cv2.line(img, (x, y), (x, y_end), color, thickness, cv2.LINE_AA)

        def _annotate_text_with_outline(
            img,
            text,
            org,
            font_scale=0.9,
            fg=(255, 255, 255),
            bg=(0, 0, 0),
            fg_thickness=2,
            bg_thickness=3,
        ):
            cv2.putText(
                img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                bg,
                bg_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                fg,
                fg_thickness,
                cv2.LINE_AA,
            )

        def _build_panel(width, panel_height):
            panel = np.full((panel_height, width, 3), 246, dtype=np.uint8)
            left = min(56, max(40, int(round(width * 0.08))))
            right = 16
            top = 14
            bottom = 30
            mid_gap = 22
            plot_w = max(24, width - left - right)
            available_h = max(48, panel_height - top - bottom - mid_gap)
            plot_h = max(20, available_h // 2)
            x0 = left
            x1 = left + plot_w - 1
            y0_wh = top
            y1_wh = y0_wh + plot_h - 1
            y0_paw = y1_wh + mid_gap
            y1_paw = y0_paw + plot_h - 1

            for y0_sub in (y0_wh, y0_paw):
                for frac in (0.2, 0.4, 0.6, 0.8):
                    y = int(round(y0_sub + frac * (plot_h - 1)))
                    cv2.line(panel, (x0, y), (x1, y), (234, 234, 234), 1, cv2.LINE_AA)

            n_ticks = max(4, min(8, int(round(width / 170.0))))
            tick_times = np.linspace(clip_t_start, clip_t_end, n_ticks)
            for tick_t in tick_times:
                x_tick = _time_to_x(tick_t, x0, plot_w)
                cv2.line(panel, (x_tick, y0_wh), (x_tick, y1_paw), (230, 230, 230), 1, cv2.LINE_AA)
                tick_lbl = f"{tick_t:.0f}"
                (tw, _th), _base = cv2.getTextSize(
                    tick_lbl,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    1,
                )
                tx = int(np.clip(x_tick - tw // 2, 0, max(0, width - tw - 2)))
                cv2.putText(
                    panel,
                    tick_lbl,
                    (tx, panel_height - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (95, 95, 95),
                    1,
                    cv2.LINE_AA,
                )

            for b0, b1 in brief_bouts:
                if not (np.isfinite(b0) and np.isfinite(b1)):
                    continue
                b0 = max(float(b0), clip_t_start)
                b1 = min(float(b1), clip_t_end)
                if b1 <= b0:
                    continue
                xb0 = _time_to_x(b0, x0, plot_w)
                xb1 = _time_to_x(b1, x0, plot_w)
                _blend_rect(panel, xb0, xb1, y0_wh, y1_wh, color_bgr=(208, 238, 241), alpha=0.30)

            for b0, b1 in long_bouts:
                if not (np.isfinite(b0) and np.isfinite(b1)):
                    continue
                b0 = max(float(b0), clip_t_start)
                b1 = min(float(b1), clip_t_end)
                if b1 <= b0:
                    continue
                xb0 = _time_to_x(b0, x0, plot_w)
                xb1 = _time_to_x(b1, x0, plot_w)
                _blend_rect(panel, xb0, xb1, y0_wh, y1_wh, color_bgr=(220, 205, 236), alpha=0.28)

            for t_on in brief_onsets:
                if not np.isfinite(t_on) or t_on < clip_t_start or t_on > clip_t_end:
                    continue
                x_on = _time_to_x(float(t_on), x0, plot_w)
                _draw_dashed_vertical(
                    panel,
                    x_on,
                    y0_wh,
                    y1_wh,
                    color=(185, 150, 20),
                    thickness=1,
                    dash=4,
                    gap=4,
                )

            for t_on in long_onsets:
                if not np.isfinite(t_on) or t_on < clip_t_start or t_on > clip_t_end:
                    continue
                x_on = _time_to_x(float(t_on), x0, plot_w)
                _draw_dashed_vertical(
                    panel,
                    x_on,
                    y0_wh,
                    y1_wh,
                    color=(100, 80, 175),
                    thickness=1,
                    dash=4,
                    gap=4,
                )

            xs = np.asarray([_time_to_x(t, x0, plot_w) for t in signal_t], dtype=np.int32)
            y_norm = np.clip(signal_v / max(signal_y_max, 1e-9), 0.0, 1.0)
            ys = y0_wh + np.round((1.0 - y_norm) * (plot_h - 1)).astype(np.int32)
            if xs.size >= 2:
                pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
                cv2.polylines(panel, [pts], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            paw_xs = np.asarray([_time_to_x(t, x0, plot_w) for t in paw_t], dtype=np.int32)
            paw_norm = np.clip(paw_v / max(paw_y_max, 1e-9), 0.0, 1.0)
            paw_ys = y0_paw + np.round((1.0 - paw_norm) * (plot_h - 1)).astype(np.int32)
            if paw_xs.size >= 2:
                paw_pts = np.column_stack([paw_xs, paw_ys]).reshape(-1, 1, 2)
                cv2.polylines(panel, [paw_pts], isClosed=False, color=(85, 85, 85), thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(panel, (x0, y0_wh), (x1, y1_wh), (214, 214, 214), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (x0, y0_paw), (x1, y1_paw), (214, 214, 214), 1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"Motion energy / whisk mean (0 to {signal_y_max:.2f})",
                (x0 + 2, y0_wh - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (85, 85, 85),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"Paw speed (0 to {paw_y_max:.2f} cm/s)",
                (x0 + 2, y0_paw - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (85, 85, 85),
                1,
                cv2.LINE_AA,
            )
            return panel, x0, y0_wh, y1_paw, plot_w

        chunk_size = 250
        writer = None
        written = 0
        width = int(meta_right.get("width", 0) or 0)
        height = int(meta_right.get("height", 0) or 0)
        panel_height = 220
        panel_static = None
        plot_left = None
        plot_top = None
        plot_bottom = None
        plot_w = None

        left_pos = np.searchsorted(left_valid_times, frame_times, side="left")
        left_pos = np.clip(left_pos, 0, max(left_valid_times.size - 1, 0))
        left_prev_pos = np.clip(left_pos - 1, 0, max(left_valid_times.size - 1, 0))
        left_curr_dist = np.abs(left_valid_times[left_pos] - frame_times)
        left_prev_dist = np.abs(left_valid_times[left_prev_pos] - frame_times)
        left_match_pos = np.where(left_prev_dist < left_curr_dist, left_prev_pos, left_pos)
        left_frame_numbers = left_valid_frame_idx[left_match_pos]

        try:
            for i0 in range(0, frame_numbers.size, chunk_size):
                chunk_ids = frame_numbers[i0 : i0 + chunk_size]
                chunk_times = frame_times[i0 : i0 + chunk_size]
                left_chunk_ids = left_frame_numbers[i0 : i0 + chunk_size]
                try:
                    right_frames = vidio.get_video_frames_preload(
                        right_url,
                        chunk_ids,
                        quiet=True,
                    )
                except TypeError:
                    right_frames = vidio.get_video_frames_preload(right_url, chunk_ids)
                if right_frames is None or len(right_frames) == 0:
                    continue

                left_chunk_unique = np.unique(left_chunk_ids.astype(np.int64))
                left_frame_map = {}
                if left_chunk_unique.size:
                    try:
                        left_frames_loaded = vidio.get_video_frames_preload(
                            left_url,
                            left_chunk_unique,
                            quiet=True,
                        )
                    except TypeError:
                        left_frames_loaded = vidio.get_video_frames_preload(left_url, left_chunk_unique)
                    if left_frames_loaded is not None:
                        for k, frm in zip(left_chunk_unique, left_frames_loaded):
                            left_frame_map[int(k)] = frm

                n_pairs = min(len(right_frames), len(chunk_times), len(left_chunk_ids))
                for i_pair in range(n_pairs):
                    frame_right = right_frames[i_pair]
                    frame_left = left_frame_map.get(int(left_chunk_ids[i_pair]), None)
                    frame_t = float(chunk_times[i_pair])
                    if frame_right is None:
                        continue

                    if frame_right.dtype != np.uint8:
                        frame_right = np.nan_to_num(frame_right, nan=0.0, posinf=255.0, neginf=0.0)
                        if np.nanmax(frame_right) <= 1.5:
                            frame_right = (255.0 * np.clip(frame_right, 0.0, 1.0)).astype(np.uint8)
                        else:
                            frame_right = np.clip(frame_right, 0, 255).astype(np.uint8)
                    if frame_right.ndim == 2:
                        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)
                    elif frame_right.ndim == 3 and frame_right.shape[2] == 4:
                        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGRA2BGR)

                    if width <= 0 or height <= 0:
                        height, width = frame_right.shape[:2]
                    if frame_right.shape[1] != width or frame_right.shape[0] != height:
                        frame_right = cv2.resize(frame_right, (width, height), interpolation=cv2.INTER_AREA)

                    if frame_left is None:
                        frame_left = np.zeros_like(frame_right)
                    else:
                        if frame_left.dtype != np.uint8:
                            frame_left = np.nan_to_num(frame_left, nan=0.0, posinf=255.0, neginf=0.0)
                            if np.nanmax(frame_left) <= 1.5:
                                frame_left = (255.0 * np.clip(frame_left, 0.0, 1.0)).astype(np.uint8)
                            else:
                                frame_left = np.clip(frame_left, 0, 255).astype(np.uint8)
                        if frame_left.ndim == 2:
                            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
                        elif frame_left.ndim == 3 and frame_left.shape[2] == 4:
                            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGRA2BGR)
                        if frame_left.shape[1] != width or frame_left.shape[0] != height:
                            frame_left = cv2.resize(frame_left, (width, height), interpolation=cv2.INTER_AREA)

                    if writer is None:
                        panel_static, plot_left, plot_top, plot_bottom, plot_w = _build_panel(
                            2 * width,
                            panel_height,
                        )
                        out_h = int(height + panel_height)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(clip_path),
                            fourcc,
                            float(fps),
                            (int(2 * width), out_h),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open VideoWriter for {clip_path}.")

                    timer_text = f"{frame_t:.2f} s"
                    timer_y = max(30, int(height) - 16)
                    _annotate_text_with_outline(
                        frame_right,
                        timer_text,
                        (12, timer_y),
                        font_scale=0.9,
                        fg=(255, 255, 255),
                        bg=(0, 0, 0),
                        fg_thickness=2,
                        bg_thickness=3,
                    )

                    panel_frame = panel_static.copy()
                    cursor_x = _time_to_x(frame_t, plot_left, plot_w)
                    _draw_dashed_vertical(
                        panel_frame,
                        cursor_x,
                        plot_top,
                        plot_bottom,
                        color=(40, 40, 255),
                        thickness=1,
                        dash=7,
                        gap=6,
                    )

                    composite = np.zeros((height + panel_height, 2 * width, 3), dtype=np.uint8)
                    composite[:height, :width, :] = frame_right
                    composite[:height, width : 2 * width, :] = frame_left

                    left_label = "Right"
                    right_label = "Left"
                    (left_tw, _left_th), _ = cv2.getTextSize(
                        left_label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        2,
                    )
                    (right_tw, _right_th), _ = cv2.getTextSize(
                        right_label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        2,
                    )
                    label_y = max(30, int(height) - 16)
                    left_x = max(8, width - left_tw - 12)
                    right_x = max(width + 8, 2 * width - right_tw - 12)
                    _annotate_text_with_outline(
                        composite,
                        left_label,
                        (left_x, label_y),
                        font_scale=0.8,
                        fg=(255, 255, 255),
                        bg=(0, 0, 0),
                        fg_thickness=2,
                        bg_thickness=3,
                    )
                    _annotate_text_with_outline(
                        composite,
                        right_label,
                        (right_x, label_y),
                        font_scale=0.8,
                        fg=(255, 255, 255),
                        bg=(0, 0, 0),
                        fg_thickness=2,
                        bg_thickness=3,
                    )

                    composite[height:, :, :] = panel_frame
                    writer.write(composite)
                    written += 1
        finally:
            if writer is not None:
                writer.release()

        if written == 0:
            raise RuntimeError("No frames were written to the output video.")

        print(
            "Saved annotated dual-camera clip (right|left): "
            f"{clip_path} | frames={written} | fps={fps:.2f} | "
            f"window=[{clip_t_start:.3f}, {clip_t_end:.3f}] s"
        )
    except Exception as exc:
        print(f"Failed to export annotated dual-camera clip: {exc}")
