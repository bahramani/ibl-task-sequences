# %%
from pathlib import Path
import importlib
import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from one.alf import io as alfio
from IPython.display import display

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None


def _find_base_path():
    candidates = []
    if "__file__" in globals():
        file_path = Path(__file__).resolve()
        candidates.extend([file_path.parent, *file_path.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "dashboard_region_cache").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/dashboard_region_cache")


BASE_PATH = _find_base_path()
CACHE_DIR = BASE_PATH / "data" / "dashboard_region_cache"
RAW_CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
RAW_DIR = BASE_PATH / "data" / "raw"
REST_DIR = RAW_DIR / ".rest"
PLOTLY_TEMPLATE = "plotly_white"
pio.templates.default = PLOTLY_TEMPLATE
pio.renderers.default = "browser"

MIN_NEURONS = 30
GOOD_ONLY = False
USE_GOOD_STPR = False
AVG_BY_PID = False
METHOD = "pearson"  # "pearson" or "spearman"
DEFAULT_LABEL_MIN = 0.59
SPIKE_SORTER = "pykilosort"
FIRST_MOVE_WINDOW = (-0.5, 1.0)
FIRST_MOVE_RESPONSE_WINDOW = (-0.1, 0.2)
FIRST_MOVE_PSTH_BIN = 0.02
FIRST_MOVE_PSTH_SMOOTH_SIGMA_BINS = 1.5
WHISK_WINDOW = (-0.5, 2.0)
WHISK_RESPONSE_WINDOW = (0.0, 0.4)
WHISK_PSTH_BIN = 0.02
WHISK_PSTH_SMOOTH_SIGMA_BINS = 1.5
EXAMPLE_MIN_UNITS = 20
EXAMPLE_REGION_METHOD = "pearson"
EXAMPLE_PID_SCAN_LIMIT = 30
EXAMPLE_RANDOM_SEED = 12345

VAR_X = "Coupling Strength (Spont)"

HIGHLIGHT_REGIONS = {
    "VISp",
    "MOs",
    "CP",
    "CA1",
    "SCm",
    "ZI",
    "AUDp",
    "GRN",
    "PO",
    "VPM",
    "VISa",
    "MOp",
    "SSp-ul",
    "SUB",
}

VAR_SPECS = {
    "Coupling Strength (Spont)": {
        "name": "Coupling Strength (Spont)",
        "df": "df_coupling",
        "v1": "coupling_strength_h1",
        "v2": "coupling_strength_h2",
    },
    "Response to Whisking Events": {
        "name": "Response to Whisking Events",
        "df": "df_res",
        "v1": "response_zmean_wh_brief_times_spont_odd",
        "v2": "response_zmean_wh_brief_times_spont_even",
    },
    "Response to First Move": {
        "name": "Response to First Move",
        "df": "df_res",
        "v1": "response_zmean_firstMovement_times_odd",
        "v2": "response_zmean_firstMovement_times_even",
    },
    "Response to Stim On": {
        "name": "Response to Stim On",
        "df": "df_res",
        "v1": "response_zmean_stimOn_times_odd",
        "v2": "response_zmean_stimOn_times_even",
    },
    "Response to Feedback": {
        "name": "Response to Feedback",
        "df": "df_res",
        "v1": "response_zmean_feedback_times_odd",
        "v2": "response_zmean_feedback_times_even",
    },
}

EXAMPLE_EVENT_SPECS = {
    "Response to First Move": {
        "event_source": "trials",
        "event_key": "firstMovement_times",
        "window": FIRST_MOVE_WINDOW,
        "response_window": FIRST_MOVE_RESPONSE_WINDOW,
        "bin_size": FIRST_MOVE_PSTH_BIN,
        "smooth_sigma_bins": FIRST_MOVE_PSTH_SMOOTH_SIGMA_BINS,
        "time_axis_title": "Time from first move (s)",
        "response_value_label": "first-move resp",
        "scatter_y_title": "Response to First Move",
        "title_suffix": "first move",
    },
    "Response to Whisking Events": {
        "event_source": "wh_events_by_period",
        "event_key": "wh_brief_times_spont",
        "window": WHISK_WINDOW,
        "response_window": WHISK_RESPONSE_WINDOW,
        "bin_size": WHISK_PSTH_BIN,
        "smooth_sigma_bins": WHISK_PSTH_SMOOTH_SIGMA_BINS,
        "time_axis_title": "Time from whisking event (s)",
        "response_value_label": "whisk resp",
        "scatter_y_title": "Response to Whisking Events",
        "title_suffix": "whisking",
    },
}


def _to_rgb(hex_value):
    if pd.isna(hex_value):
        return None
    hv = str(hex_value).strip()
    if hv.lower().startswith("0x"):
        hv = hv[2:]
    hv = "".join(ch for ch in hv if ch in "0123456789abcdefABCDEF")
    if not hv:
        return None
    hv = hv.zfill(6)
    if len(hv) > 6:
        hv = hv[-6:]
    try:
        r = int(hv[0:2], 16)
        g = int(hv[2:4], 16)
        b = int(hv[4:6], 16)
        return f"rgb({r},{g},{b})"
    except Exception:
        return None


def _get_allen_lookup():
    try:
        iblatlas_pkg = importlib.import_module("iblatlas")
        csv_path = Path(iblatlas_pkg.__file__).resolve().parent / "allen_structure_tree.csv"
        if not csv_path.exists():
            return None
        df_allen = pd.read_csv(csv_path, dtype={"color_hex_triplet": "string"})
    except Exception:
        return None

    required_cols = {"id", "acronym", "graph_order", "color_hex_triplet"}
    if not required_cols.issubset(set(df_allen.columns)):
        return None

    color_by_acr = {}
    order_by_acr = {}
    for _, row in df_allen.iterrows():
        acr = str(row.get("acronym", "")).strip()
        if not acr:
            continue
        rgb = _to_rgb(row.get("color_hex_triplet", ""))
        if rgb is not None:
            color_by_acr[acr] = rgb
        try:
            order_by_acr[acr] = int(float(row.get("graph_order")))
        except Exception:
            pass
    return {
        "color_by_acr": color_by_acr,
        "order_by_acr": order_by_acr,
    }


def _build_region_colors(regions):
    allen_lookup = _get_allen_lookup()
    if allen_lookup is not None:
        color_by_acr = allen_lookup.get("color_by_acr", {})
        return {str(region): color_by_acr[str(region)] for region in regions if str(region) in color_by_acr}

    if BrainRegions is None:
        return {}
    try:
        br = BrainRegions()
    except Exception:
        return {}

    colors = {}
    for region in regions:
        try:
            idx = br.acronym2index(region)[1][0][0]
            rgb = br.rgb[idx]
            colors[str(region)] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        except Exception:
            continue
    return colors


def _calc_total_reliability(rx, ry):
    if pd.notnull(rx) and pd.notnull(ry) and rx * ry >= 0:
        return float(np.sqrt(rx * ry))
    if pd.notnull(rx) and pd.isnull(ry):
        return float(rx)
    if pd.isnull(rx) and pd.notnull(ry):
        return float(ry)
    return np.nan


def _compute_axis_range(values, pad_frac=0.04):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        delta = 0.05 if vmin == 0 else abs(vmin) * pad_frac
        return [vmin - delta, vmax + delta]
    span = vmax - vmin
    pad = span * pad_frac
    return [vmin - pad, vmax + pad]


def _apply_cartesian_grid(fig):
    if fig is None:
        return
    if PLOTLY_TEMPLATE == "plotly_dark":
        gridcolor = "rgba(148, 163, 184, 0.24)"
    else:
        gridcolor = "rgba(15, 23, 42, 0.12)"
    fig.update_xaxes(showgrid=True, gridcolor=gridcolor)
    fig.update_yaxes(showgrid=True, gridcolor=gridcolor)


def _add_corr_rel_summary_annotation(fig, corr_values, rel_values):
    corr_arr = np.asarray(corr_values, dtype=float)
    rel_arr = np.asarray(rel_values, dtype=float)
    corr_arr = corr_arr[np.isfinite(corr_arr)]
    rel_arr = rel_arr[np.isfinite(rel_arr)]
    if corr_arr.size == 0 or rel_arr.size == 0:
        return

    if PLOTLY_TEMPLATE == "plotly_dark":
        bgcolor = "rgba(15, 23, 42, 0.82)"
        bordercolor = "rgba(226, 232, 240, 0.55)"
        font_color = "#f8fafc"
    else:
        bgcolor = "rgba(255, 255, 255, 0.88)"
        bordercolor = "rgba(15, 23, 42, 0.25)"
        font_color = "#0f172a"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=(
            f"mean corr = {float(np.nanmean(corr_arr)):.3f}<br>"
            f"mean rel = {float(np.nanmean(rel_arr)):.3f}"
        ),
        font=dict(size=12, color=font_color),
        bordercolor=bordercolor,
        borderwidth=1,
        bgcolor=bgcolor,
    )


def _load_dashboard_region_cache():
    corr_path = CACHE_DIR / "summary_correlations.pkl"
    region_path = CACHE_DIR / "summary_region_counts.pkl"
    if not corr_path.exists() or not region_path.exists():
        raise FileNotFoundError("summary_correlations.pkl or summary_region_counts.pkl is missing")

    df_corr = pd.read_pickle(corr_path)
    df_region_counts = pd.read_pickle(region_path)

    if "delay_mode" in df_corr.columns:
        df_corr = df_corr.copy()
        df_corr["delay_mode"] = df_corr["delay_mode"].astype(str).str.lower().fillna("com")
        if (df_corr["delay_mode"] == "com").any():
            df_corr = df_corr[df_corr["delay_mode"] == "com"].copy()

    return df_corr, df_region_counts


def _save_summary_correlations(df_corr_updated):
    corr_pkl = CACHE_DIR / "summary_correlations.pkl"
    corr_parquet = CACHE_DIR / "summary_correlations.parquet"
    df_corr_updated.to_pickle(corr_pkl)
    try:
        df_corr_updated.to_parquet(corr_parquet, index=False)
    except Exception:
        pass


def _has_spont_interval(meta):
    if not meta:
        return False
    interval = meta.get("spont_interval")
    if interval is None:
        return False
    try:
        start, end = interval
        start_val = float(start)
        end_val = float(end)
        return np.isfinite(start_val) and np.isfinite(end_val) and end_val > start_val
    except Exception:
        return False


def _clean_table_with_pid(df, pid):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df_out = df.copy()
    df_out["pid"] = str(pid)
    df_out["cluster_id"] = pd.to_numeric(df_out["cluster_id"], errors="coerce")
    df_out = df_out[np.isfinite(df_out["cluster_id"])].copy()
    if df_out.empty:
        return None
    df_out["cluster_id"] = df_out["cluster_id"].astype(int)
    return df_out


def _build_variable_table(df, spec, region_lookup):
    if df is None or df.empty:
        return None
    if spec["v1"] not in df.columns or spec["v2"] not in df.columns:
        return None
    df_var = df[["pid", "cluster_id", spec["v1"], spec["v2"]]].copy()
    df_var["cluster_id"] = pd.to_numeric(df_var["cluster_id"], errors="coerce")
    df_var = df_var[np.isfinite(df_var["cluster_id"])].copy()
    if df_var.empty:
        return None
    df_var["cluster_id"] = df_var["cluster_id"].astype(int)
    df_var = df_var.groupby(["pid", "cluster_id"], as_index=False).mean(numeric_only=True)
    df_var = df_var.merge(region_lookup, on=["pid", "cluster_id"], how="inner")
    if df_var.empty:
        return None

    v1 = df_var[spec["v1"]].to_numpy(dtype=float)
    v2 = df_var[spec["v2"]].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(v1) & np.isfinite(v2)
    mean_vals[valid] = (v1[valid] + v2[valid]) / 2.0
    df_var["mean"] = mean_vals
    return df_var


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
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan, n
    return float(np.corrcoef(x_rank, y_rank)[0, 1]), n


def _mean_with_count(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan, 0
    return float(np.nanmean(arr[finite])), int(np.sum(finite))


def _rows_for_region_pair(region_lookup, df_var_x, df_var_y, var_y):
    rows = []
    region_name = str(region_lookup["region"].iloc[0])

    if AVG_BY_PID:
        pid_list = region_lookup["pid"].astype(str).unique().tolist()
        x_rel_p_vals, x_rel_s_vals = [], []
        y_rel_p_vals, y_rel_s_vals = [], []
        xy_p_vals, xy_s_vals = [], []
        yx_p_vals, yx_s_vals = [], []

        for pid in pid_list:
            region_pid = region_lookup[region_lookup["pid"].astype(str) == str(pid)][["pid", "cluster_id"]].drop_duplicates()
            if region_pid.empty:
                continue

            df_var_x_pid = None
            if df_var_x is not None:
                df_var_x_pid = df_var_x[df_var_x["pid"].astype(str) == str(pid)].copy()
            df_var_y_pid = None
            if df_var_y is not None:
                df_var_y_pid = df_var_y[df_var_y["pid"].astype(str) == str(pid)].copy()

            if df_var_x_pid is not None and not df_var_x_pid.empty:
                p_val, _ = _pearsonr_with_n(df_var_x_pid[VAR_SPECS[VAR_X]["v1"]], df_var_x_pid[VAR_SPECS[VAR_X]["v2"]])
                s_val, _ = _spearmanr_with_n(df_var_x_pid[VAR_SPECS[VAR_X]["v1"]], df_var_x_pid[VAR_SPECS[VAR_X]["v2"]])
                x_rel_p_vals.append(p_val)
                x_rel_s_vals.append(s_val)

            if df_var_y_pid is not None and not df_var_y_pid.empty:
                p_val, _ = _pearsonr_with_n(df_var_y_pid[VAR_SPECS[var_y]["v1"]], df_var_y_pid[VAR_SPECS[var_y]["v2"]])
                s_val, _ = _spearmanr_with_n(df_var_y_pid[VAR_SPECS[var_y]["v1"]], df_var_y_pid[VAR_SPECS[var_y]["v2"]])
                y_rel_p_vals.append(p_val)
                y_rel_s_vals.append(s_val)

            if (
                df_var_x_pid is not None and not df_var_x_pid.empty
                and df_var_y_pid is not None and not df_var_y_pid.empty
            ):
                mean_wide = region_pid.copy()
                mean_wide = mean_wide.merge(
                    df_var_x_pid[["pid", "cluster_id", "mean"]],
                    on=["pid", "cluster_id"],
                    how="left",
                ).rename(columns={"mean": "mean_x"})
                mean_wide = mean_wide.merge(
                    df_var_y_pid[["pid", "cluster_id", "mean"]],
                    on=["pid", "cluster_id"],
                    how="left",
                ).rename(columns={"mean": "mean_y"})
                p_xy, _ = _pearsonr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
                s_xy, _ = _spearmanr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
                xy_p_vals.append(p_xy)
                xy_s_vals.append(s_xy)
                yx_p_vals.append(p_xy)
                yx_s_vals.append(s_xy)

        x_p, x_p_n = _mean_with_count(x_rel_p_vals)
        x_s, x_s_n = _mean_with_count(x_rel_s_vals)
        y_p, y_p_n = _mean_with_count(y_rel_p_vals)
        y_s, y_s_n = _mean_with_count(y_rel_s_vals)
        xy_p, xy_p_n = _mean_with_count(xy_p_vals)
        xy_s, xy_s_n = _mean_with_count(xy_s_vals)
        yx_p, yx_p_n = _mean_with_count(yx_p_vals)
        yx_s, yx_s_n = _mean_with_count(yx_s_vals)
    else:
        x_p = x_s = x_p_n = x_s_n = None
        y_p = y_s = y_p_n = y_s_n = None
        xy_p = xy_s = xy_p_n = xy_s_n = None
        yx_p = yx_s = yx_p_n = yx_s_n = None

        if df_var_x is not None and not df_var_x.empty:
            x_p, x_p_n = _pearsonr_with_n(df_var_x[VAR_SPECS[VAR_X]["v1"]], df_var_x[VAR_SPECS[VAR_X]["v2"]])
            x_s, x_s_n = _spearmanr_with_n(df_var_x[VAR_SPECS[VAR_X]["v1"]], df_var_x[VAR_SPECS[VAR_X]["v2"]])

        if df_var_y is not None and not df_var_y.empty:
            y_p, y_p_n = _pearsonr_with_n(df_var_y[VAR_SPECS[var_y]["v1"]], df_var_y[VAR_SPECS[var_y]["v2"]])
            y_s, y_s_n = _spearmanr_with_n(df_var_y[VAR_SPECS[var_y]["v1"]], df_var_y[VAR_SPECS[var_y]["v2"]])

        if (
            df_var_x is not None and not df_var_x.empty
            and df_var_y is not None and not df_var_y.empty
        ):
            mean_wide = region_lookup[["pid", "cluster_id"]].drop_duplicates()
            mean_wide = mean_wide.merge(
                df_var_x[["pid", "cluster_id", "mean"]],
                on=["pid", "cluster_id"],
                how="left",
            ).rename(columns={"mean": "mean_x"})
            mean_wide = mean_wide.merge(
                df_var_y[["pid", "cluster_id", "mean"]],
                on=["pid", "cluster_id"],
                how="left",
            ).rename(columns={"mean": "mean_y"})
            xy_p, xy_p_n = _pearsonr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
            xy_s, xy_s_n = _spearmanr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
            yx_p, yx_p_n = xy_p, xy_p_n
            yx_s, yx_s_n = xy_s, xy_s_n

    base_row = {
        "good_only": bool(GOOD_ONLY),
        "use_good_stpr": bool(USE_GOOD_STPR),
        "avg_by_pid": bool(AVG_BY_PID),
        "arousal_group": "all",
        "region": region_name,
    }

    if x_p is not None:
        rows.append({
            **base_row,
            "var1": VAR_X,
            "var2": VAR_X,
            "pearson_r": x_p,
            "pearson_n": x_p_n,
            "spearman_rho": x_s,
            "spearman_n": x_s_n,
        })
    if y_p is not None:
        rows.append({
            **base_row,
            "var1": var_y,
            "var2": var_y,
            "pearson_r": y_p,
            "pearson_n": y_p_n,
            "spearman_rho": y_s,
            "spearman_n": y_s_n,
        })
    if xy_p is not None:
        rows.append({
            **base_row,
            "var1": VAR_X,
            "var2": var_y,
            "pearson_r": xy_p,
            "pearson_n": xy_p_n,
            "spearman_rho": xy_s,
            "spearman_n": xy_s_n,
        })
    if yx_p is not None:
        rows.append({
            **base_row,
            "var1": var_y,
            "var2": VAR_X,
            "pearson_r": yx_p,
            "pearson_n": yx_p_n,
            "spearman_rho": yx_s,
            "spearman_n": yx_s_n,
        })

    return rows


def _compute_missing_rows_for_pair(var_y):
    spec_x = VAR_SPECS[VAR_X]
    spec_y = VAR_SPECS[var_y]
    rows = []
    region_lookup_parts = {}
    region_var_x_parts = {}
    region_var_y_parts = {}

    cache_paths = sorted(RAW_CACHE_DIR.glob("*.pkl"))
    for path in cache_paths:
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            continue
        if not isinstance(cache, dict):
            continue

        pid = str(cache.get("pid", path.stem))
        config_calc = cache.get("config_calc") or {}
        meta = cache.get("meta") or {}
        has_spont = _has_spont_interval(meta)

        df_res = cache.get("df_res")
        if df_res is None or not isinstance(df_res, pd.DataFrame) or df_res.empty:
            continue
        if "cluster_id" not in df_res.columns or "label" not in df_res.columns:
            continue
        if "acronym" in df_res.columns:
            region_col = "acronym"
        elif "region" in df_res.columns:
            region_col = "region"
        else:
            continue

        df_res_copy = df_res.copy()
        df_res_copy["pid"] = pid
        df_res_copy["cluster_id"] = pd.to_numeric(df_res_copy["cluster_id"], errors="coerce")
        df_res_copy = df_res_copy[np.isfinite(df_res_copy["cluster_id"])].copy()
        if df_res_copy.empty:
            continue
        df_res_copy["cluster_id"] = df_res_copy["cluster_id"].astype(int)

        labels = pd.to_numeric(df_res_copy["label"], errors="coerce")
        label_values = labels.to_numpy(dtype=float)
        if GOOD_ONLY:
            mask_keep = np.isfinite(label_values) & np.isclose(label_values, 1.0)
        else:
            try:
                label_min = float(config_calc.get("CALC_LABEL_MIN", DEFAULT_LABEL_MIN))
            except Exception:
                label_min = float(DEFAULT_LABEL_MIN)
            mask_keep = np.isfinite(label_values) & (label_values >= label_min)

        df_units = df_res_copy.loc[mask_keep, ["pid", "cluster_id", region_col]].copy()
        if df_units.empty:
            continue
        df_units["region"] = df_units[region_col].astype(str)
        df_units = df_units[~df_units["region"].isin(["root", "void", "NA", "nan"])]
        if df_units.empty:
            continue

        df_var_x_source = None
        if has_spont:
            coupling_key = "df_coupling_good" if USE_GOOD_STPR else "df_coupling"
            df_var_x_source = _clean_table_with_pid(cache.get(coupling_key), pid)
        df_var_y_source = df_res_copy

        for region_name in sorted(df_units["region"].astype(str).unique().tolist()):
            region_lookup = df_units.loc[
                df_units["region"].astype(str) == str(region_name),
                ["pid", "cluster_id", "region"],
            ].drop_duplicates()
            if region_lookup.empty:
                continue

            df_var_x = _build_variable_table(df_var_x_source, spec_x, region_lookup) if has_spont else None
            df_var_y = _build_variable_table(df_var_y_source, spec_y, region_lookup)

            if ((df_var_x is None or df_var_x.empty) and (df_var_y is None or df_var_y.empty)):
                continue

            region_lookup_parts.setdefault(region_name, []).append(region_lookup)
            if df_var_x is not None and not df_var_x.empty:
                region_var_x_parts.setdefault(region_name, []).append(df_var_x)
            if df_var_y is not None and not df_var_y.empty:
                region_var_y_parts.setdefault(region_name, []).append(df_var_y)

    for region_name in sorted(region_lookup_parts.keys()):
        region_lookup = pd.concat(region_lookup_parts[region_name], ignore_index=True)
        region_lookup = region_lookup.drop_duplicates(subset=["pid", "cluster_id", "region"]).reset_index(drop=True)
        df_var_x = None
        if region_name in region_var_x_parts:
            df_var_x = pd.concat(region_var_x_parts[region_name], ignore_index=True)
            df_var_x = df_var_x.drop_duplicates(subset=["pid", "cluster_id"], keep="last").reset_index(drop=True)
        df_var_y = None
        if region_name in region_var_y_parts:
            df_var_y = pd.concat(region_var_y_parts[region_name], ignore_index=True)
            df_var_y = df_var_y.drop_duplicates(subset=["pid", "cluster_id"], keep="last").reset_index(drop=True)
        rows.extend(_rows_for_region_pair(region_lookup, df_var_x, df_var_y, var_y))

    if not rows:
        return pd.DataFrame()

    df_new = pd.DataFrame(rows)
    dedup_cols = ["good_only", "use_good_stpr", "avg_by_pid", "arousal_group", "region", "var1", "var2"]
    df_new = df_new.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    return df_new


def _refresh_corr_views():
    global df_rc_filtered, df_corr_mode
    df_rc_filtered = df_region_counts.query("good_only == @GOOD_ONLY").copy()
    df_corr_mode = df_corr.query(
        "good_only == @GOOD_ONLY and "
        "use_good_stpr == @USE_GOOD_STPR and "
        "avg_by_pid == @AVG_BY_PID and "
        "arousal_group == 'all'"
    ).copy()


def ensure_pair_rows(var_y, persist=True, force=False):
    global df_corr
    if force:
        needed_masks = []
    else:
        needed_masks = [
            (df_corr_mode["var1"] == VAR_X) & (df_corr_mode["var2"] == VAR_X),
            (df_corr_mode["var1"] == var_y) & (df_corr_mode["var2"] == var_y),
            (df_corr_mode["var1"] == VAR_X) & (df_corr_mode["var2"] == var_y),
        ]
    if needed_masks and all(mask.any() for mask in needed_masks):
        print(f"Using existing summary rows for {VAR_X} vs {var_y}")
        return

    if force:
        print(f"Force recomputing summary rows for {VAR_X} vs {var_y} ...")
    else:
        print(f"Computing only the missing rows needed for {VAR_X} vs {var_y} ...")
    df_new = _compute_missing_rows_for_pair(var_y)
    if df_new.empty:
        raise ValueError(f"Could not compute any rows for {VAR_X} vs {var_y}")
    if "delay_mode" in df_corr.columns and "delay_mode" not in df_new.columns:
        df_new["delay_mode"] = "com"

    df_corr = pd.concat([df_corr, df_new], ignore_index=True)
    dedup_cols = ["good_only", "use_good_stpr", "avg_by_pid", "arousal_group", "region", "var1", "var2"]
    df_corr = df_corr.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    _refresh_corr_views()

    if persist:
        _save_summary_correlations(df_corr)
        print(f"Saved {len(df_new):,} newly computed rows into {CACHE_DIR / 'summary_correlations.pkl'}")


def _load_pid_cache(pid):
    cache_path = RAW_CACHE_DIR / f"{pid}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Raw cache not found for PID {pid}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _extract_insertions_from_json(payload):
    found = []
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            probe_insertions = current.get("probe_insertion", None)
            if isinstance(probe_insertions, list):
                stack.extend(probe_insertions)
            if (
                isinstance(current.get("id"), str)
                and isinstance(current.get("session"), str)
                and isinstance(current.get("session_info"), dict)
            ):
                found.append(current)
                continue
            for key, value in current.items():
                if key in {"datasets", "data_dataset_session_related"}:
                    continue
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return found


def _find_pid_rest_record(pid):
    if not REST_DIR.exists():
        return None
    for rest_path in sorted(REST_DIR.iterdir()):
        if not rest_path.is_file():
            continue
        try:
            text = rest_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pid not in text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        for rec in _extract_insertions_from_json(payload):
            if str(rec.get("id", "")).strip() == str(pid):
                return rec
    return None


def _find_probe_revision_dir(session_path, pname):
    probe_root = session_path / "alf" / str(pname) / SPIKE_SORTER
    if not probe_root.exists():
        raise FileNotFoundError(f"Probe path not found: {probe_root}")

    required = ("spikes.times.npy", "spikes.clusters.npy")
    revision_dirs = []
    for child in probe_root.iterdir():
        if child.is_dir() and child.name.startswith("#") and child.name.endswith("#"):
            if all((child / fname).exists() for fname in required):
                revision_dirs.append(child)

    if revision_dirs:
        return sorted(revision_dirs, key=lambda p: p.name)[-1]
    if all((probe_root / fname).exists() for fname in required):
        return probe_root
    raise FileNotFoundError(f"No valid spike-sorting directory found under {probe_root}")


def _load_spikes_for_pid(pid, cache_payload):
    rest_record = _find_pid_rest_record(pid)
    if rest_record is None:
        raise RuntimeError(f"Could not resolve PID {pid} from data/raw/.rest")

    session_info = rest_record["session_info"]
    lab = str(session_info["lab"])
    subject = str(session_info["subject"])
    date = str(session_info["start_time"])[:10]
    number = int(session_info["number"])
    pname = str(rest_record["name"])

    session_path = RAW_DIR / lab / "Subjects" / subject / date / f"{number:03d}"
    revision_dir = _find_probe_revision_dir(session_path, pname)
    return alfio.load_object(revision_dir, "spikes", attribute=["times", "clusters"])


def _trial_df_from_cache(cache_payload):
    trials = cache_payload.get("trials", None)
    if isinstance(trials, pd.DataFrame):
        df = trials.copy()
    elif trials is None:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(trials)
    if df.empty:
        return df
    if "trial_idx" not in df.columns:
        df["trial_idx"] = np.arange(len(df), dtype=int)
    return df.sort_values("trial_idx").reset_index(drop=True)


def _prepare_pid_region_tables(cache_payload, pid, region_name, response_var_name="Response to First Move"):
    config_calc = cache_payload.get("config_calc") or {}
    df_res = cache_payload.get("df_res")
    if df_res is None or not isinstance(df_res, pd.DataFrame) or df_res.empty:
        return None, None, None, None
    if "cluster_id" not in df_res.columns or "label" not in df_res.columns:
        return None, None, None, None
    if "acronym" in df_res.columns:
        region_col = "acronym"
    elif "region" in df_res.columns:
        region_col = "region"
    else:
        return None, None, None, None

    df_res_copy = df_res.copy()
    df_res_copy["pid"] = str(pid)
    df_res_copy["cluster_id"] = pd.to_numeric(df_res_copy["cluster_id"], errors="coerce")
    df_res_copy = df_res_copy[np.isfinite(df_res_copy["cluster_id"])].copy()
    if df_res_copy.empty:
        return None, None, None, None
    df_res_copy["cluster_id"] = df_res_copy["cluster_id"].astype(int)

    labels = pd.to_numeric(df_res_copy["label"], errors="coerce")
    label_values = labels.to_numpy(dtype=float)
    if GOOD_ONLY:
        mask_keep = np.isfinite(label_values) & np.isclose(label_values, 1.0)
    else:
        try:
            label_min = float(config_calc.get("CALC_LABEL_MIN", DEFAULT_LABEL_MIN))
        except Exception:
            label_min = float(DEFAULT_LABEL_MIN)
        mask_keep = np.isfinite(label_values) & (label_values >= label_min)

    df_units = df_res_copy.loc[mask_keep, ["pid", "cluster_id", region_col]].copy()
    if df_units.empty:
        return None, None, None, None
    df_units["region"] = df_units[region_col].astype(str)
    df_units = df_units[df_units["region"].astype(str) == str(region_name)].copy()
    if df_units.empty:
        return None, None, None, None
    region_lookup = df_units[["pid", "cluster_id", "region"]].drop_duplicates()

    coupling_key = "df_coupling_good" if USE_GOOD_STPR else "df_coupling"
    df_coupling = _clean_table_with_pid(cache_payload.get(coupling_key), pid)
    spec_x = VAR_SPECS[VAR_X]
    spec_y = VAR_SPECS[response_var_name]

    df_var_x = None
    if _has_spont_interval(cache_payload.get("meta") or {}):
        df_var_x = _build_variable_table(df_coupling, spec_x, region_lookup)
    df_var_y = _build_variable_table(df_res_copy, spec_y, region_lookup)
    return df_res_copy, region_lookup, df_var_x, df_var_y


def _compute_pid_pair_stats(region_name, response_var_name="Response to First Move"):
    rows = []
    for path in sorted(RAW_CACHE_DIR.glob("*.pkl")):
        pid = path.stem
        try:
            cache_payload = _load_pid_cache(pid)
        except Exception:
            continue
        df_res_copy, region_lookup, df_var_x, df_var_y = _prepare_pid_region_tables(
            cache_payload,
            pid,
            region_name,
            response_var_name=response_var_name,
        )
        if region_lookup is None or region_lookup.empty:
            continue
        n_units = int(region_lookup["cluster_id"].nunique())
        if n_units <= EXAMPLE_MIN_UNITS:
            continue
        if df_var_x is None or df_var_x.empty or df_var_y is None or df_var_y.empty:
            continue

        mean_wide = region_lookup[["pid", "cluster_id"]].drop_duplicates()
        mean_wide = mean_wide.merge(
            df_var_x[["pid", "cluster_id", "mean"]],
            on=["pid", "cluster_id"],
            how="left",
        ).rename(columns={"mean": "mean_x"})
        mean_wide = mean_wide.merge(
            df_var_y[["pid", "cluster_id", "mean"]],
            on=["pid", "cluster_id"],
            how="left",
        ).rename(columns={"mean": "mean_y"})

        p_corr, _ = _pearsonr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
        s_corr, _ = _spearmanr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
        p_rel_x, _ = _pearsonr_with_n(df_var_x[VAR_SPECS[VAR_X]["v1"]], df_var_x[VAR_SPECS[VAR_X]["v2"]])
        s_rel_x, _ = _spearmanr_with_n(df_var_x[VAR_SPECS[VAR_X]["v1"]], df_var_x[VAR_SPECS[VAR_X]["v2"]])
        p_rel_y, _ = _pearsonr_with_n(df_var_y[VAR_SPECS[response_var_name]["v1"]], df_var_y[VAR_SPECS[response_var_name]["v2"]])
        s_rel_y, _ = _spearmanr_with_n(df_var_y[VAR_SPECS[response_var_name]["v1"]], df_var_y[VAR_SPECS[response_var_name]["v2"]])

        rows.append(
            {
                "pid": str(pid),
                "region": str(region_name),
                "n_units": n_units,
                "pearson_corr": p_corr,
                "spearman_corr": s_corr,
                "pearson_reliability": _calc_total_reliability(p_rel_x, p_rel_y),
                "spearman_reliability": _calc_total_reliability(s_rel_x, s_rel_y),
            }
        )

    return pd.DataFrame(rows)


def _choose_pid_for_region(region_name, desired_sign, response_var_name="Response to First Move", selection_rank=0):
    df_stats = _compute_pid_pair_stats(region_name, response_var_name=response_var_name)
    if df_stats.empty:
        raise ValueError(f"No eligible PID found for region {region_name}")

    corr_col = f"{EXAMPLE_REGION_METHOD}_corr"
    rel_col = f"{EXAMPLE_REGION_METHOD}_reliability"
    df_stats = df_stats[
        np.isfinite(pd.to_numeric(df_stats[corr_col], errors="coerce"))
        & np.isfinite(pd.to_numeric(df_stats[rel_col], errors="coerce"))
        & (pd.to_numeric(df_stats["n_units"], errors="coerce") > EXAMPLE_MIN_UNITS)
    ].copy()
    if df_stats.empty:
        raise ValueError(f"No finite corr/rel PID found for region {region_name}")

    df_stats["base_score"] = np.abs(pd.to_numeric(df_stats[corr_col], errors="coerce")) * np.clip(
        pd.to_numeric(df_stats[rel_col], errors="coerce"), 0.0, None
    )
    if desired_sign == "positive":
        sign_mask = pd.to_numeric(df_stats[corr_col], errors="coerce") > 0
        fallback_ascending = False
    else:
        sign_mask = pd.to_numeric(df_stats[corr_col], errors="coerce") < 0
        fallback_ascending = True

    if sign_mask.any():
        df_pref = df_stats.loc[sign_mask].copy()
        df_pref = df_pref.sort_values(["base_score", "n_units"], ascending=[False, False]).reset_index(drop=True)
    else:
        df_pref = df_stats.sort_values([corr_col, "base_score", "n_units"], ascending=[fallback_ascending, False, False]).reset_index(drop=True)

    df_scan = df_pref.head(EXAMPLE_PID_SCAN_LIMIT).copy()
    evaluated_rows = []
    for _, row in df_scan.iterrows():
        pid = str(row["pid"])
        row_dict = row.to_dict()
        try:
            cache_payload = _load_pid_cache(pid)
            unit_df = _build_unit_df_for_region(
                cache_payload,
                pid,
                region_name,
                response_var_name=response_var_name,
            )
            if unit_df is None:
                continue
            examples = _select_example_neurons(unit_df, desired_sign=desired_sign)
            row_dict.update(_score_example_selection(examples, desired_sign=desired_sign))
        except Exception:
            row_dict.update({"example_valid": False, "example_score": -np.inf})
        evaluated_rows.append(row_dict)

    if evaluated_rows:
        df_eval = pd.DataFrame(evaluated_rows)
        if df_eval["example_valid"].any():
            df_eval = df_eval[df_eval["example_valid"]].copy()
        df_eval = df_eval.sort_values(
            ["example_score", "base_score", "n_units"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        pick_idx = min(max(int(selection_rank), 0), len(df_eval) - 1)
        return df_eval.iloc[pick_idx].to_dict(), df_eval

    pick_idx = min(max(int(selection_rank), 0), len(df_pref) - 1)
    return df_pref.iloc[pick_idx].to_dict(), df_pref


def _safe_z(arr):
    arr = np.asarray(arr, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return out
    mu = float(np.nanmean(arr[finite]))
    sd = float(np.nanstd(arr[finite]))
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[finite] = (arr[finite] - mu) / sd
    return out


def _pick_rows(df, mask, score_col, n_pick, ascending=False, used_ids=None):
    if used_ids is None:
        used_ids = set()
    sub = df.loc[mask].copy()
    if used_ids:
        sub = sub[~sub["cluster_id"].isin(list(used_ids))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(score_col, ascending=ascending)
    return sub.head(n_pick).copy()


def _select_example_neurons(unit_df, desired_sign):
    df = unit_df.copy()
    df = df[
        np.isfinite(pd.to_numeric(df["coupling_mean"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["response_mean"], errors="coerce"))
    ].copy()
    if df.empty:
        raise ValueError("No finite neurons available for example selection")

    df["coupling_z"] = _safe_z(df["coupling_mean"])
    df["response_z"] = _safe_z(df["response_mean"])
    df["coupling_positive"] = pd.to_numeric(df["coupling_mean"], errors="coerce") > 0

    used_ids = set()
    picked_frames = []
    positive_df = df[df["coupling_positive"]].copy()
    if positive_df.empty:
        positive_df = df.copy()
    pool_n = max(4, int(np.ceil(0.2 * len(positive_df))))
    pool_n = min(max(pool_n, 2), len(positive_df))

    hi_coupling_pool = (
        positive_df.sort_values(["coupling_mean", "response_mean"], ascending=[False, True])
        .head(pool_n)
        .copy()
    )
    lo_coupling_pool = (
        positive_df.sort_values(["coupling_mean", "response_mean"], ascending=[True, False])
        .head(pool_n)
        .copy()
    )

    if len(hi_coupling_pool) < 2:
        hi_coupling_pool = positive_df.sort_values("coupling_mean", ascending=False).head(max(2, min(6, len(positive_df)))).copy()
    if len(lo_coupling_pool) < 2:
        lo_coupling_pool = positive_df.sort_values("coupling_mean", ascending=True).head(max(2, min(6, len(positive_df)))).copy()

    if desired_sign == "positive":
        hi_coupling_pool["score_hi"] = hi_coupling_pool["response_z"] + 0.15 * hi_coupling_pool["coupling_z"]
        lo_coupling_pool["score_lo"] = lo_coupling_pool["response_z"] + 0.15 * lo_coupling_pool["coupling_z"]

        picked_hi = _pick_rows(
            hi_coupling_pool,
            pd.Series(True, index=hi_coupling_pool.index),
            "score_hi",
            2,
            ascending=False,
            used_ids=used_ids,
        )
        if len(picked_hi) < 2:
            df["score_hi"] = df["response_z"] + 0.15 * df["coupling_z"]
            fallback = _pick_rows(
                df,
                df["coupling_positive"],
                "score_hi",
                2 - len(picked_hi),
                ascending=False,
                used_ids=used_ids | set(picked_hi["cluster_id"]),
            )
            picked_hi = pd.concat([picked_hi, fallback], ignore_index=True)
        picked_hi["example_group"] = "High coupling / high response"
        picked_hi["example_color"] = "#ef4444"
        picked_frames.append(picked_hi)
        used_ids.update(picked_hi["cluster_id"].tolist())

        picked_lo = _pick_rows(
            lo_coupling_pool,
            pd.Series(True, index=lo_coupling_pool.index),
            "score_lo",
            2,
            ascending=True,
            used_ids=used_ids,
        )
        if len(picked_lo) < 2:
            df["score_lo"] = df["response_z"] + 0.15 * df["coupling_z"]
            fallback = _pick_rows(
                df,
                df["coupling_positive"],
                "score_lo",
                2 - len(picked_lo),
                ascending=True,
                used_ids=used_ids | set(picked_lo["cluster_id"]),
            )
            picked_lo = pd.concat([picked_lo, fallback], ignore_index=True)
        picked_lo["example_group"] = "Low non-zero coupling / low response"
        picked_lo["example_color"] = "#7c3aed"
        picked_frames.append(picked_lo)
    else:
        hi_coupling_pool["score_hl"] = 2.0 * hi_coupling_pool["coupling_z"] - 0.25 * hi_coupling_pool["response_z"]
        lo_coupling_pool["score_lh"] = lo_coupling_pool["response_z"] - 0.15 * lo_coupling_pool["coupling_z"]

        picked_hl = _pick_rows(
            hi_coupling_pool,
            pd.Series(True, index=hi_coupling_pool.index),
            "score_hl",
            2,
            ascending=False,
            used_ids=used_ids,
        )
        if len(picked_hl) < 2:
            df["score_hl"] = 2.0 * df["coupling_z"] - 0.25 * df["response_z"]
            fallback = _pick_rows(
                df,
                df["coupling_positive"],
                "score_hl",
                2 - len(picked_hl),
                ascending=False,
                used_ids=used_ids | set(picked_hl["cluster_id"]),
            )
            picked_hl = pd.concat([picked_hl, fallback], ignore_index=True)
        picked_hl["example_group"] = "High coupling / low response"
        picked_hl["example_color"] = "#0f766e"
        picked_frames.append(picked_hl)
        used_ids.update(picked_hl["cluster_id"].tolist())

        picked_lh = _pick_rows(
            lo_coupling_pool,
            pd.Series(True, index=lo_coupling_pool.index),
            "score_lh",
            2,
            ascending=False,
            used_ids=used_ids,
        )
        if len(picked_lh) < 2:
            df["score_lh"] = df["response_z"] - 0.15 * df["coupling_z"]
            fallback = _pick_rows(
                df,
                df["coupling_positive"],
                "score_lh",
                2 - len(picked_lh),
                ascending=False,
                used_ids=used_ids | set(picked_lh["cluster_id"]),
            )
            picked_lh = pd.concat([picked_lh, fallback], ignore_index=True)
        picked_lh["example_group"] = "Low non-zero coupling / high response"
        picked_lh["example_color"] = "#d97706"
        picked_frames.append(picked_lh)

    picked = pd.concat(picked_frames, ignore_index=True)
    if len(picked) < 4:
        raise ValueError("Could not find four example neurons for the selected PID/region")
    picked = picked.head(4).copy()
    picked["display_label"] = [
        f"{row['example_group']} #{idx + 1}"
        for idx, (_, row) in enumerate(picked.iterrows())
    ]
    return picked


def _score_example_selection(examples, desired_sign):
    if examples is None or len(examples) < 4:
        return {"example_valid": False, "example_score": -np.inf}

    if desired_sign == "positive":
        hi = examples[examples["example_group"] == "High coupling / high response"].copy()
        lo = examples[examples["example_group"] == "Low non-zero coupling / low response"].copy()
        if len(hi) < 2 or len(lo) < 2:
            return {"example_valid": False, "example_score": -np.inf}
        coupling_gap = float(hi["coupling_mean"].mean() - lo["coupling_mean"].mean())
        response_gap = float(hi["response_mean"].mean() - lo["response_mean"].mean())
        high_coupling = float(hi["coupling_mean"].mean())
    else:
        hi = examples[examples["example_group"] == "High coupling / low response"].copy()
        lo = examples[examples["example_group"] == "Low non-zero coupling / high response"].copy()
        if len(hi) < 2 or len(lo) < 2:
            return {"example_valid": False, "example_score": -np.inf}
        coupling_gap = float(hi["coupling_mean"].mean() - lo["coupling_mean"].mean())
        response_gap = float(lo["response_mean"].mean() - hi["response_mean"].mean())
        high_coupling = float(hi["coupling_mean"].mean())

    example_valid = (
        np.isfinite(coupling_gap)
        and np.isfinite(response_gap)
        and np.isfinite(high_coupling)
        and (coupling_gap > 0)
        and (response_gap > 0)
    )
    example_score = coupling_gap + 0.6 * response_gap + 0.3 * max(high_coupling, 0.0)
    return {
        "example_valid": bool(example_valid),
        "example_score": float(example_score),
        "example_coupling_gap": float(coupling_gap),
        "example_response_gap": float(response_gap),
        "example_high_coupling_mean": float(high_coupling),
    }


def _build_unit_df_for_region(cache_payload, pid, region_name, response_var_name="Response to First Move"):
    _, region_lookup, df_var_x, df_var_y = _prepare_pid_region_tables(
        cache_payload,
        pid,
        region_name,
        response_var_name=response_var_name,
    )
    if region_lookup is None or df_var_x is None or df_var_y is None:
        return None

    unit_df = region_lookup[["pid", "cluster_id", "region"]].drop_duplicates()
    unit_df = unit_df.merge(
        df_var_x[["pid", "cluster_id", "mean"]].rename(columns={"mean": "coupling_mean"}),
        on=["pid", "cluster_id"],
        how="inner",
    )
    unit_df = unit_df.merge(
        df_var_y[["pid", "cluster_id", "mean"]].rename(columns={"mean": "response_mean"}),
        on=["pid", "cluster_id"],
        how="inner",
    )
    unit_df = unit_df[
        np.isfinite(pd.to_numeric(unit_df["coupling_mean"], errors="coerce"))
        & np.isfinite(pd.to_numeric(unit_df["response_mean"], errors="coerce"))
    ].copy()
    if unit_df.empty:
        return None
    return unit_df


def _replace_examples_with_random_high_coupling(unit_df, examples, desired_sign, random_seed=EXAMPLE_RANDOM_SEED):
    if desired_sign == "positive":
        target_group = "High coupling / high response"
        replacement_group = "High coupling example"
        replacement_color = "#ef4444"
    else:
        target_group = "High coupling / low response"
        replacement_group = "High coupling example"
        replacement_color = "#0f766e"

    keep_examples = examples[examples["example_group"] != target_group].copy()
    high_examples = examples[examples["example_group"] == target_group].copy()
    n_replace = len(high_examples)
    if n_replace == 0:
        return examples.copy()

    df = unit_df.copy()
    df = df[
        np.isfinite(pd.to_numeric(df["coupling_mean"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["response_mean"], errors="coerce"))
    ].copy()
    if df.empty:
        return examples.copy()

    df["coupling_positive"] = pd.to_numeric(df["coupling_mean"], errors="coerce") > 0
    positive_df = df[df["coupling_positive"]].copy()
    if positive_df.empty:
        positive_df = df.copy()

    pool_n = max(8, int(np.ceil(0.25 * len(positive_df))))
    pool_n = min(max(pool_n, n_replace), len(positive_df))
    candidate_pool = (
        positive_df.sort_values("coupling_mean", ascending=False)
        .head(pool_n)
        .copy()
    )
    exclude_ids = set(examples["cluster_id"].tolist())
    candidate_pool = candidate_pool[~candidate_pool["cluster_id"].isin(exclude_ids)].copy()
    if len(candidate_pool) < n_replace:
        exclude_ids = set(keep_examples["cluster_id"].tolist())
        candidate_pool = positive_df[~positive_df["cluster_id"].isin(exclude_ids)].copy()

    if len(candidate_pool) < n_replace:
        return examples.copy()

    rng = np.random.default_rng(random_seed)
    picked_idx = rng.choice(candidate_pool.index.to_numpy(), size=n_replace, replace=False)
    picked_high = candidate_pool.loc[picked_idx].copy()
    picked_high = picked_high.sort_values("coupling_mean", ascending=False).reset_index(drop=True)
    picked_high["example_group"] = replacement_group
    picked_high["example_color"] = replacement_color

    combined = pd.concat(
        [
            picked_high.reindex(columns=examples.columns),
            keep_examples.reindex(columns=examples.columns),
        ],
        ignore_index=True,
    )
    combined["display_label"] = [
        f"{row['example_group']} #{idx + 1}"
        for idx, (_, row) in enumerate(combined.iterrows())
    ]
    return combined


def _extract_example_event_times(cache_payload, response_var_name):
    spec = EXAMPLE_EVENT_SPECS[response_var_name]
    if spec["event_source"] == "trials":
        trials = _trial_df_from_cache(cache_payload)
        event_key = str(spec["event_key"])
        if event_key not in trials.columns:
            raise ValueError(f"Trials table is missing {event_key}")
        event_times = pd.to_numeric(trials[event_key], errors="coerce").to_numpy(dtype=float)
    else:
        wh_events = cache_payload.get("wh_events_by_period") or {}
        event_times = np.asarray(wh_events.get(spec["event_key"], np.array([])), dtype=float)
    event_times = event_times[np.isfinite(event_times)]
    if event_times.size == 0:
        raise ValueError(f"No finite events found for {response_var_name}")
    return np.sort(event_times)


def _gaussian_smooth(values, sigma_bins):
    arr = np.asarray(values, dtype=float)
    if sigma_bins is None or sigma_bins <= 0:
        return arr.copy()
    radius = max(1, int(np.ceil(4.0 * float(sigma_bins))))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma_bins)) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(arr, kernel, mode="same")


def _compute_aligned_raster_psth(unit_spike_times, event_times, window, bin_size, smooth_sigma_bins):
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]
    if event_times.size == 0:
        raise ValueError("No valid first-move events found")
    spike_times = np.asarray(unit_spike_times, dtype=float)
    spike_times = spike_times[np.isfinite(spike_times)]

    raster_x = []
    raster_y = []
    all_aligned = []
    for trial_idx, t_event in enumerate(event_times, start=1):
        t0 = t_event + float(window[0])
        t1 = t_event + float(window[1])
        idx0 = int(np.searchsorted(spike_times, t0, side="left"))
        idx1 = int(np.searchsorted(spike_times, t1, side="right"))
        rel_spikes = spike_times[idx0:idx1] - t_event
        if rel_spikes.size:
            raster_x.extend(rel_spikes.tolist())
            raster_y.extend([trial_idx] * int(rel_spikes.size))
            all_aligned.append(rel_spikes)

    if all_aligned:
        aligned_concat = np.concatenate(all_aligned)
    else:
        aligned_concat = np.array([], dtype=float)

    bin_edges = np.arange(float(window[0]), float(window[1]) + float(bin_size), float(bin_size))
    if bin_edges[-1] < float(window[1]):
        bin_edges = np.append(bin_edges, float(window[1]))
    counts, edges = np.histogram(aligned_concat, bins=bin_edges)
    fr = counts.astype(float) / (event_times.size * float(bin_size))
    fr_smooth = _gaussian_smooth(fr, smooth_sigma_bins)
    bin_centers = edges[:-1] + np.diff(edges) / 2.0

    return {
        "raster_x": np.asarray(raster_x, dtype=float),
        "raster_y": np.asarray(raster_y, dtype=float),
        "bin_centers": np.asarray(bin_centers, dtype=float),
        "fr": np.asarray(fr_smooth, dtype=float),
        "n_trials": int(event_times.size),
    }


def _build_region_example_figure(
    region_name,
    desired_sign,
    response_var_name="Response to First Move",
    selection_rank=0,
    random_high_examples=False,
):
    example_spec = EXAMPLE_EVENT_SPECS[response_var_name]
    selected_pid, candidate_pids = _choose_pid_for_region(
        region_name,
        desired_sign=desired_sign,
        response_var_name=response_var_name,
        selection_rank=selection_rank,
    )
    pid = str(selected_pid["pid"])
    cache_payload = _load_pid_cache(pid)
    spikes = _load_spikes_for_pid(pid, cache_payload)
    df_res_copy, region_lookup, df_var_x, df_var_y = _prepare_pid_region_tables(
        cache_payload,
        pid,
        region_name,
        response_var_name=response_var_name,
    )
    if region_lookup is None or df_var_x is None or df_var_y is None:
        raise ValueError(f"Could not build region tables for PID {pid} / {region_name}")

    unit_df = region_lookup[["pid", "cluster_id", "region"]].drop_duplicates()
    unit_df = unit_df.merge(
        df_var_x[["pid", "cluster_id", "mean"]].rename(columns={"mean": "coupling_mean"}),
        on=["pid", "cluster_id"],
        how="inner",
    )
    unit_df = unit_df.merge(
        df_var_y[["pid", "cluster_id", "mean"]].rename(columns={"mean": "response_mean"}),
        on=["pid", "cluster_id"],
        how="inner",
    )
    examples = _select_example_neurons(unit_df, desired_sign=desired_sign)
    if random_high_examples:
        examples = _replace_examples_with_random_high_coupling(
            unit_df,
            examples,
            desired_sign=desired_sign,
        )

    event_times = _extract_example_event_times(cache_payload, response_var_name)

    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"])

    example_titles = []
    for _, row in examples.iterrows():
        title_text = (
            f"{row['display_label']}<br>"
            f"cluster_id={int(row['cluster_id'])} | "
            f"coupling={float(row['coupling_mean']):.3f} | "
            f"{example_spec['response_value_label']}={float(row['response_mean']):.3f}"
        )
        example_titles.append(title_text)
    subplot_titles = [
        example_titles[0], example_titles[1],
        "", "",
        example_titles[2], example_titles[3],
        "", "",
    ]
    fig = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.045,
        horizontal_spacing=0.08,
        row_heights=[0.20, 0.11, 0.20, 0.11],
        subplot_titles=subplot_titles,
    )

    title = (
        f"{region_name} | PID {pid} | "
        f"{EXAMPLE_REGION_METHOD.title()} corr={float(selected_pid[f'{EXAMPLE_REGION_METHOD}_corr']):.3f} | "
        f"rel={float(selected_pid[f'{EXAMPLE_REGION_METHOD}_reliability']):.3f} | "
        f"n={int(selected_pid['n_units'])} | {example_spec['title_suffix']}"
    )
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=1000,
        width=1250,
        margin=dict(l=80, r=40, t=100, b=60),
        showlegend=False,
    )

    for idx, (_, row) in enumerate(examples.iterrows()):
        cluster_id = int(row["cluster_id"])
        unit_spikes = spike_times[spike_clusters == cluster_id]
        aligned = _compute_aligned_raster_psth(
            unit_spikes,
            event_times,
            window=example_spec["window"],
            bin_size=example_spec["bin_size"],
            smooth_sigma_bins=example_spec["smooth_sigma_bins"],
        )

        if idx < 2:
            raster_row = 1
            psth_row = 2
            col = idx + 1
        else:
            raster_row = 3
            psth_row = 4
            col = idx - 1
        color = str(row["example_color"])

        fig.add_trace(
            go.Scatter(
                x=aligned["raster_x"],
                y=aligned["raster_y"],
                mode="markers",
                marker=dict(size=3, color="black", opacity=0.75),
                hoverinfo="skip",
            ),
            row=raster_row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=aligned["bin_centers"],
                y=aligned["fr"],
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="t=%{x:.2f}s<br>FR=%{y:.2f} Hz<extra></extra>",
            ),
            row=psth_row,
            col=col,
        )

        fig.update_yaxes(title_text="Trials", row=raster_row, col=col)
        fig.update_yaxes(title_text="Hz", row=psth_row, col=col)
        fig.update_yaxes(autorange="reversed", row=raster_row, col=col)

        fig.add_vline(x=0.0, line=dict(color="black", dash="dash"), row=raster_row, col=col)
        fig.add_vline(x=0.0, line=dict(color="black", dash="dash"), row=psth_row, col=col)
        fig.add_vrect(
            x0=float(example_spec["response_window"][0]),
            x1=float(example_spec["response_window"][1]),
            fillcolor="lightgray",
            opacity=0.18,
            line_width=0,
            row=raster_row,
            col=col,
        )
        fig.add_vrect(
            x0=float(example_spec["response_window"][0]),
            x1=float(example_spec["response_window"][1]),
            fillcolor="lightgray",
            opacity=0.18,
            line_width=0,
            row=psth_row,
            col=col,
        )

    _apply_cartesian_grid(fig)
    fig.update_xaxes(title_text=example_spec["time_axis_title"], row=4, col=1)
    fig.update_xaxes(title_text=example_spec["time_axis_title"], row=4, col=2)
    return fig, examples, selected_pid, candidate_pids, unit_df


def _build_region_pid_scatter(unit_df, examples, region_name, pid_info, response_var_name="Response to First Move"):
    example_spec = EXAMPLE_EVENT_SPECS[response_var_name]
    region_color = _build_region_colors([region_name]).get(region_name, "#2563eb")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=unit_df["coupling_mean"].to_numpy(dtype=float),
            y=unit_df["response_mean"].to_numpy(dtype=float),
            mode="markers",
            marker=dict(size=8, color=region_color, opacity=0.75),
            name=f"{region_name} neurons",
            hovertemplate=(
                "cluster_id=%{customdata[0]}<br>"
                "coupling=%{x:.3f}<br>"
                f"{example_spec['response_value_label']}=%{{y:.3f}}<extra></extra>"
            ),
            customdata=np.stack([unit_df["cluster_id"].astype(int).to_numpy()], axis=-1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=examples["coupling_mean"].to_numpy(dtype=float),
            y=examples["response_mean"].to_numpy(dtype=float),
            mode="markers+text",
            text=[f"{i + 1}" for i in range(len(examples))],
            textposition="top center",
            marker=dict(
                size=13,
                color=examples["example_color"].astype(str).tolist(),
                line=dict(color="black", width=1.5),
            ),
            name="Example neurons",
            customdata=np.stack([examples["cluster_id"].astype(int).to_numpy()], axis=-1),
            hovertemplate=(
                "example cluster_id=%{customdata[0]}<br>"
                "coupling=%{x:.3f}<br>"
                f"{example_spec['response_value_label']}=%{{y:.3f}}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0.0, line=dict(color="gray", dash="dash"))
    fig.add_hline(y=0.0, line=dict(color="gray", dash="dash"))
    _apply_cartesian_grid(fig)
    fig.update_layout(
        title=(
            f"{region_name} neuron scatter | PID {pid_info['pid']} | "
            f"{EXAMPLE_REGION_METHOD.title()} corr={float(pid_info[f'{EXAMPLE_REGION_METHOD}_corr']):.3f} | "
            f"rel={float(pid_info[f'{EXAMPLE_REGION_METHOD}_reliability']):.3f} | {example_spec['title_suffix']}"
        ),
        xaxis_title="Coupling Strength (Spont)",
        yaxis_title=example_spec["scatter_y_title"],
        template=PLOTLY_TEMPLATE,
        height=650,
        width=850,
        margin=dict(l=80, r=40, t=90, b=70),
    )
    return fig


df_corr, df_region_counts = _load_dashboard_region_cache()
df_rc_filtered = df_region_counts.query("good_only == @GOOD_ONLY").copy()
df_corr_mode = df_corr.query(
    "good_only == @GOOD_ONLY and "
    "use_good_stpr == @USE_GOOD_STPR and "
    "avg_by_pid == @AVG_BY_PID and "
    "arousal_group == 'all'"
).copy()

print(f"Loaded {len(df_corr_mode):,} correlation rows from {CACHE_DIR}")


def build_corr_rel_figure(var_y, method=METHOD):
    if df_corr_mode.empty:
        raise ValueError("No matching rows found in summary_correlations for the selected toggles.")

    col_r = "pearson_r" if str(method).lower() == "pearson" else "spearman_rho"

    df_x_diag = df_corr_mode.query("var1 == @VAR_X and var2 == @VAR_X").copy()
    df_y_diag = df_corr_mode.query("var1 == @var_y and var2 == @var_y").copy()
    df_xy = df_corr_mode.query("var1 == @VAR_X and var2 == @var_y").copy()
    if df_xy.empty:
        df_xy = df_corr_mode.query("var1 == @var_y and var2 == @VAR_X").copy()

    if df_x_diag.empty:
        raise ValueError(f"Missing diagonal reliability row for {VAR_X}")
    if df_y_diag.empty:
        raise ValueError(f"Missing diagonal reliability row for {var_y}")
    if df_xy.empty:
        raise ValueError(f"Missing cross-correlation row for {VAR_X} vs {var_y}")

    df_plot = (
        df_xy.merge(df_x_diag[["region", col_r]], on="region", suffixes=("", "_x_diag"))
        .merge(df_y_diag[["region", col_r]], on="region", suffixes=("", "_y_diag"))
        .merge(df_rc_filtered[["region", "n_neurons"]], on="region", how="inner")
    )
    df_plot = df_plot[df_plot["n_neurons"] >= MIN_NEURONS].copy()
    if df_plot.empty:
        raise ValueError(f"No eligible regions with >= {MIN_NEURONS} neurons for {VAR_X} vs {var_y}")

    df_plot["reliability_x"] = df_plot[f"{col_r}_x_diag"]
    df_plot["reliability_y"] = df_plot[f"{col_r}_y_diag"]
    df_plot["corr"] = df_plot[col_r]
    df_plot["reliability"] = df_plot.apply(
        lambda row: _calc_total_reliability(row["reliability_x"], row["reliability_y"]),
        axis=1,
    )
    df_plot = df_plot[
        np.isfinite(pd.to_numeric(df_plot["corr"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df_plot["reliability"], errors="coerce"))
    ].copy()
    if df_plot.empty:
        raise ValueError(f"No finite corr/rel points available for {VAR_X} vs {var_y}")

    region_colors = _build_region_colors(df_plot["region"].astype(str).tolist())
    fig = go.Figure()
    for _, row in df_plot.sort_values("region").iterrows():
        region = str(row["region"])
        color = region_colors.get(region)
        marker = dict(size=8, color=color) if color else dict(size=8)
        if region in HIGHLIGHT_REGIONS:
            marker["line"] = dict(color="black", width=1)

        fig.add_trace(
            go.Scatter(
                x=[float(row["reliability"])],
                y=[float(row["corr"])],
                mode="markers",
                name=region,
                marker=marker,
                showlegend=region in HIGHLIGHT_REGIONS,
                hovertemplate=(
                    f"Region: {region}<br>"
                    f"corr={float(row['corr']):.2f}<br>"
                    f"rel={float(row['reliability']):.2f}<br>"
                    f"n={int(row['n_neurons'])}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{str(method).title()} correlation vs total reliability | "
            f"{VAR_X} vs {var_y} | regions with >= {MIN_NEURONS} neurons"
        ),
        xaxis_title="Total reliability (sqrt(rel1 * rel2))",
        yaxis_title="Correlation between variables",
        template=PLOTLY_TEMPLATE,
        legend=dict(x=1.02, y=1, yanchor="top"),
        margin=dict(l=80, r=200, t=90, b=70),
        height=600,
        width=900,
    )
    _apply_cartesian_grid(fig)

    x_range = _compute_axis_range(df_plot["reliability"].to_numpy(dtype=float))
    y_range = _compute_axis_range(df_plot["corr"].to_numpy(dtype=float))
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    if x_range is not None and y_range is not None:
        min_val = float(min(x_range[0], y_range[0]))
        max_val = float(max(x_range[1], y_range[1]))
        if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
            fig.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(color="red", dash="dash"),
            )

    _add_corr_rel_summary_annotation(
        fig,
        df_plot["corr"].to_numpy(dtype=float),
        df_plot["reliability"].to_numpy(dtype=float),
    )
    return fig, df_plot


# %%
ensure_pair_rows("Response to Whisking Events")
fig_whisk, df_whisk = build_corr_rel_figure("Response to Whisking Events", method=METHOD)
fig_whisk.show()


# %%
ensure_pair_rows("Response to First Move", force=True)
fig_move, df_move = build_corr_rel_figure("Response to First Move", method=METHOD)
fig_move.show()


# %%
ensure_pair_rows("Response to Stim On")
fig_stim, df_stim = build_corr_rel_figure("Response to Stim On", method=METHOD)
fig_stim.show()


# %%
ensure_pair_rows("Response to Feedback")
fig_feedback, df_feedback = build_corr_rel_figure("Response to Feedback", method=METHOD)
fig_feedback.show()


# %%
fig_scm_examples, df_scm_examples, scm_pid_info, df_scm_pid_candidates, df_scm_units = _build_region_example_figure(
    "SCm",
    desired_sign="positive",
)
fig_scm_scatter = _build_region_pid_scatter(df_scm_units, df_scm_examples, "SCm", scm_pid_info)
print("Selected SCm PID:", scm_pid_info)
display(df_scm_examples[["cluster_id", "example_group", "coupling_mean", "response_mean"]])
fig_scm_examples.show()
fig_scm_scatter.show()


# %%
fig_vpl_examples, df_vpl_examples, vpl_pid_info, df_vpl_pid_candidates, df_vpl_units = _build_region_example_figure(
    "VPL",
    desired_sign="negative",
    random_high_examples=True,
)
fig_vpl_scatter = _build_region_pid_scatter(df_vpl_units, df_vpl_examples, "VPL", vpl_pid_info)
print("Selected VPL PID:", vpl_pid_info)
display(df_vpl_examples[["cluster_id", "example_group", "coupling_mean", "response_mean"]])
fig_vpl_examples.show()
fig_vpl_scatter.show()


# %%
fig_visp_whisk_examples, df_visp_whisk_examples, visp_whisk_pid_info, df_visp_whisk_pid_candidates, df_visp_whisk_units = _build_region_example_figure(
    "VISp",
    desired_sign="positive",
    response_var_name="Response to Whisking Events",
    selection_rank=1,
)
fig_visp_whisk_scatter = _build_region_pid_scatter(
    df_visp_whisk_units,
    df_visp_whisk_examples,
    "VISp",
    visp_whisk_pid_info,
    response_var_name="Response to Whisking Events",
)
print("Selected VISp PID:", visp_whisk_pid_info)
display(df_visp_whisk_examples[["cluster_id", "example_group", "coupling_mean", "response_mean"]])
fig_visp_whisk_examples.show()
fig_visp_whisk_scatter.show()


# %%
fig_dg_whisk_examples, df_dg_whisk_examples, dg_whisk_pid_info, df_dg_whisk_pid_candidates, df_dg_whisk_units = _build_region_example_figure(
    "DG",
    desired_sign="negative",
    response_var_name="Response to Whisking Events",
)
fig_dg_whisk_scatter = _build_region_pid_scatter(
    df_dg_whisk_units,
    df_dg_whisk_examples,
    "DG",
    dg_whisk_pid_info,
    response_var_name="Response to Whisking Events",
)
print("Selected DG PID:", dg_whisk_pid_info)
display(df_dg_whisk_examples[["cluster_id", "example_group", "coupling_mean", "response_mean"]])
fig_dg_whisk_examples.show()
fig_dg_whisk_scatter.show()
