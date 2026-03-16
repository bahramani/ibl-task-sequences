import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import pickle
import importlib
import sys

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None


@st.cache_resource(show_spinner=False)
def _get_brain_regions():
    if BrainRegions is None:
        return None
    try:
        return BrainRegions()
    except Exception:
        # iblatlas may require parquet readers internally; keep dashboard functional without it.
        return None


@st.cache_resource(show_spinner=False)
def _get_allen_lookup():
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

    def _category_from_acronyms(acronyms):
        if "Isocortex" in acronyms:
            return "Isocortex"
        if "HPF" in acronyms:
            return "HPF"
        if "OLF" in acronyms:
            return "OLF"
        if "CTXsp" in acronyms:
            return "CTXsp"
        if "STR" in acronyms:
            return "Striatum"
        if "PAL" in acronyms:
            return "Pallidum"
        if "TH" in acronyms:
            return "Thal."
        if "HY" in acronyms:
            return "Hyp."
        if "MB" in acronyms:
            return "Midbrain"
        if "P" in acronyms:
            return "Pons"
        if "MY" in acronyms:
            return "Medulla"
        if "CB" in acronyms:
            return "Cereb."
        return "Other"

    try:
        iblatlas_pkg = importlib.import_module("iblatlas")
        csv_path = Path(iblatlas_pkg.__file__).resolve().parent / "allen_structure_tree.csv"
        if not csv_path.exists():
            return None
        # Keep hex color column as text to preserve leading zeros (e.g., "019399" for AUD*).
        df_allen = pd.read_csv(csv_path, dtype={"color_hex_triplet": "string"})
    except Exception:
        return None

    required_cols = {"id", "acronym", "graph_order", "structure_id_path", "color_hex_triplet"}
    if not required_cols.issubset(set(df_allen.columns)):
        return None

    id_to_acr = {}
    for _, row in df_allen[["id", "acronym"]].dropna().iterrows():
        try:
            id_to_acr[int(row["id"])] = str(row["acronym"])
        except Exception:
            continue

    color_by_acr = {}
    order_by_acr = {}
    category_by_acr = {}

    for _, row in df_allen.iterrows():
        acr = str(row.get("acronym", ""))
        if not acr:
            continue

        rgb = _to_rgb(row.get("color_hex_triplet", ""))
        if rgb is not None:
            color_by_acr[acr] = rgb

        try:
            order_by_acr[acr] = int(float(row.get("graph_order")))
        except Exception:
            pass

        path = str(row.get("structure_id_path", ""))
        ancestor_ids = []
        for tok in path.split("/"):
            tok = tok.strip()
            if tok and tok.isdigit():
                ancestor_ids.append(int(tok))
        ancestor_acronyms = [id_to_acr.get(i, "") for i in ancestor_ids]
        category_by_acr[acr] = _category_from_acronyms(ancestor_acronyms)

    return {
        "color_by_acr": color_by_acr,
        "order_by_acr": order_by_acr,
        "category_by_acr": category_by_acr,
    }


def _build_region_colors(regions):
    allen_lookup = _get_allen_lookup()
    if allen_lookup is not None:
        color_by_acr = allen_lookup.get("color_by_acr", {})
        return {str(region): color_by_acr[str(region)] for region in regions if str(region) in color_by_acr}

    br = _get_brain_regions()
    if br is None:
        return {}
    colors = {}
    for region in regions:
        try:
            idx = br.acronym2index(region)[1][0][0]
            rgb = br.rgb[idx]
            colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        except Exception:
            continue
    return colors


def _show_table(df, width="stretch", max_rows=400):
    try:
        st.dataframe(df, width=width)
        return
    except Exception:
        # Fallback for environments where Streamlit dataframe serialization fails
        # (e.g. broken/missing pyarrow binary after environment updates).
        if "_table_fallback_warned" not in st.session_state:
            st.warning("`st.dataframe` is unavailable in this environment. Showing HTML table fallback.")
            st.session_state["_table_fallback_warned"] = True

    if df is None or df.empty:
        st.info("No rows to display.")
        return

    df_show = df.head(max_rows).copy()
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} rows (of {len(df):,}).")

    html = df_show.to_html(index=False, classes="fallback-table", border=0)
    st.markdown(
        f"""
<style>
.fallback-wrap {{
  overflow: auto;
  max-height: 430px;
  border: 1px solid #d4d9e1;
  border-radius: 10px;
  background: #ffffff;
}}
.fallback-table {{
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: 0.93rem;
  line-height: 1.35;
}}
.fallback-table thead th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: #f4f7fb;
  color: #1f2937;
  text-align: left;
  font-weight: 700;
  padding: 10px 12px;
  border-bottom: 1px solid #d4d9e1;
  white-space: nowrap;
}}
.fallback-table tbody td {{
  padding: 9px 12px;
  border-bottom: 1px solid #e8ebf0;
  color: #1f2937;
  white-space: nowrap;
}}
.fallback-table tbody tr:nth-child(even) {{
  background: #f9fbfe;
}}
.fallback-table tbody tr:hover {{
  background: #eef4ff;
}}
</style>
<div class="fallback-wrap">{html}</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Region Dashboard (Fast)", layout="wide")
BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "dashboard_region_cache"
RAW_CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
DEFAULT_LABEL_MIN = 0.59

if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

CORR_VARIABLES = [
    {"name": "Depth", "df": "df_depth", "v1": "depth_h1", "v2": "depth_h2"},
    {"name": "Firing Rate", "df": "df_firing_rate", "v1": "firing_rate_h1", "v2": "firing_rate_h2"},
    {"name": "Correlation to Whisking", "df": "df_arousal_corr", "v1": "arousal_corr_abs_h1", "v2": "arousal_corr_abs_h2"},
    {"name": "Delay to Stim On", "df": "df_res", "v1": "delay_stimOn_times_odd", "v2": "delay_stimOn_times_even"},
    {"name": "Delay to First Move", "df": "df_res", "v1": "delay_firstMovement_times_odd", "v2": "delay_firstMovement_times_even"},
    {"name": "Delay to Feedback", "df": "df_res", "v1": "delay_feedback_times_odd", "v2": "delay_feedback_times_even"},
    {"name": "Delay to Whisking Events", "df": "df_res", "v1": "delay_wh_brief_times_spont_odd", "v2": "delay_wh_brief_times_spont_even"},
    {"name": "Delay to Passive Visual", "df": "df_res", "v1": "delay_passive_visual_times_odd", "v2": "delay_passive_visual_times_even"},
    {"name": "Delay to Passive Tone", "df": "df_res", "v1": "delay_passive_tone_times_odd", "v2": "delay_passive_tone_times_even"},
    {"name": "Delay to Passive Valve", "df": "df_res", "v1": "delay_passive_valve_times_odd", "v2": "delay_passive_valve_times_even"},
    {"name": "Delay to Passive Noise", "df": "df_res", "v1": "delay_passive_noise_times_odd", "v2": "delay_passive_noise_times_even"},
    {"name": "Coupling Delay (Spont)", "df": "df_coupling", "v1": "coupling_delay_ms_h1", "v2": "coupling_delay_ms_h2"},
    {"name": "Coupling Delay (ITI)", "df": "df_coupling_iti", "v1": "coupling_delay_ms_odd", "v2": "coupling_delay_ms_even"},
    {"name": "Coupling Delay (Task)", "df": "df_coupling_task", "v1": "coupling_delay_ms_odd", "v2": "coupling_delay_ms_even"},
    {"name": "Coupling Strength (Spont)", "df": "df_coupling", "v1": "coupling_strength_h1", "v2": "coupling_strength_h2"},
    {"name": "Coupling Strength (ITI)", "df": "df_coupling_iti", "v1": "coupling_strength_odd", "v2": "coupling_strength_even"},
    {"name": "Coupling Strength (Task)", "df": "df_coupling_task", "v1": "coupling_strength_odd", "v2": "coupling_strength_even"},
    {"name": "Coupling Max (Spont)", "df": "df_coupling", "v1": "coupling_max_h1", "v2": "coupling_max_h2"},
    {"name": "Coupling Max (ITI)", "df": "df_coupling_iti", "v1": "coupling_max_odd", "v2": "coupling_max_even"},
    {"name": "Coupling Max (Task)", "df": "df_coupling_task", "v1": "coupling_max_odd", "v2": "coupling_max_even"},
]


def _build_corr_variables():
    return [dict(spec) for spec in CORR_VARIABLES]


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


def _is_firing_rate_spec(spec):
    return spec.get("df") in {"df_firing_rate", "df_depth", "df_arousal_corr"}


def _is_spont_spec(spec):
    return spec.get("df") == "df_coupling"


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
    except (TypeError, ValueError):
        return False


def _filter_region_lookup_for_spec(region_lookup, spec, has_spont_interval):
    if region_lookup is None or region_lookup.empty:
        return region_lookup
    if not _is_spont_spec(spec):
        return region_lookup
    if has_spont_interval:
        return region_lookup
    return region_lookup.iloc[0:0]


def _build_variable_table(df, spec, region_lookup):
    if df is None or df.empty:
        return None
    if spec["v1"] not in df.columns or spec["v2"] not in df.columns:
        return None
    if "pid" not in df.columns or "cluster_id" not in df.columns:
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


def _extract_cluster_depths(cache, cluster_ids):
    if cluster_ids is None:
        return None
    try:
        cluster_ids_arr = np.asarray(cluster_ids)
    except Exception:
        return None
    n_clusters = int(cluster_ids_arr.size)
    if n_clusters == 0:
        return None

    def _to_depth_array(values):
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == n_clusters:
                return arr
        except Exception:
            pass
        return None

    for key in ("cluster_depths", "cluster_depth", "depths", "cluster_depth_um", "depth_um"):
        arr = _to_depth_array(cache.get(key))
        if arr is not None:
            return arr

    clusters = cache.get("clusters")
    if clusters is not None:
        if isinstance(clusters, dict):
            for key in ("depths", "depth", "axial_um", "depth_um"):
                arr = _to_depth_array(clusters.get(key))
                if arr is not None:
                    return arr
        else:
            for attr in ("depths", "depth", "axial_um", "depth_um"):
                arr = _to_depth_array(getattr(clusters, attr, None))
                if arr is not None:
                    return arr
    return None


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


def _clean_table_with_pid(df, pid):
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if not isinstance(df, pd.DataFrame):
        return None

    df_out = df.copy()
    df_out["pid"] = str(pid)
    df_out["cluster_id"] = pd.to_numeric(df_out["cluster_id"], errors="coerce")
    df_out = df_out[np.isfinite(df_out["cluster_id"])].copy()
    if df_out.empty:
        return None
    df_out["cluster_id"] = df_out["cluster_id"].astype(int)
    return df_out


@st.cache_data(show_spinner="Computing per-PID points from raw cache...")
def _compute_region_pid_points(selected_region, var_x, var_y, good_only, use_good_stpr):
    if not RAW_CACHE_DIR.exists():
        return pd.DataFrame()

    spec_map = {spec["name"]: spec for spec in _build_corr_variables()}
    spec_x = spec_map.get(var_x)
    spec_y = spec_map.get(var_y)
    if spec_x is None or spec_y is None:
        return pd.DataFrame()

    selected_region_norm = None if selected_region in (None, "") else str(selected_region)
    rows = []
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
        config_calc = cache.get("config_calc")
        if not isinstance(config_calc, dict):
            config_calc = {}

        df_res = cache.get("df_res")
        if df_res is None or not isinstance(df_res, pd.DataFrame) or df_res.empty:
            continue
        if "cluster_id" not in df_res.columns:
            continue
        if "acronym" in df_res.columns:
            region_col = "acronym"
        elif "region" in df_res.columns:
            region_col = "region"
        else:
            continue
        if "label" not in df_res.columns:
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
        if good_only:
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

        table_name_x = spec_x.get("df")
        table_name_y = spec_y.get("df")
        needed_tables = set([table_name_x, table_name_y])
        table_data = {"df_res": df_res_copy}

        if "df_coupling" in needed_tables:
            table_data["df_coupling"] = _clean_table_with_pid(
                cache.get("df_coupling_good" if use_good_stpr else "df_coupling"),
                pid,
            )
        if "df_coupling_task" in needed_tables:
            table_data["df_coupling_task"] = _clean_table_with_pid(
                cache.get("df_coupling_task_good" if use_good_stpr else "df_coupling_task"),
                pid,
            )
        if "df_coupling_iti" in needed_tables:
            table_data["df_coupling_iti"] = _clean_table_with_pid(
                cache.get("df_coupling_iti_good" if use_good_stpr else "df_coupling_iti"),
                pid,
            )
        if "df_arousal_corr" in needed_tables:
            if "arousal_corr_abs" in df_res_copy.columns:
                df_ar = df_res_copy[["pid", "cluster_id", "arousal_corr_abs"]].copy()
                df_ar["arousal_corr_abs_h1"] = pd.to_numeric(df_ar["arousal_corr_abs"], errors="coerce")
                df_ar["arousal_corr_abs_h2"] = df_ar["arousal_corr_abs_h1"]
                table_data["df_arousal_corr"] = df_ar[["pid", "cluster_id", "arousal_corr_abs_h1", "arousal_corr_abs_h2"]]
            else:
                table_data["df_arousal_corr"] = None
        if "df_firing_rate" in needed_tables:
            cluster_ids = cache.get("cluster_ids")
            cluster_fr = cache.get("cluster_firing_rate")
            df_rate = None
            if cluster_ids is not None and cluster_fr is not None:
                cluster_ids_clean = pd.to_numeric(np.asarray(cluster_ids), errors="coerce")
                cluster_fr = np.asarray(cluster_fr, dtype=float)
                if len(cluster_ids_clean) == len(cluster_fr):
                    df_rate = pd.DataFrame({
                        "pid": pid,
                        "cluster_id": cluster_ids_clean,
                        "firing_rate_h1": cluster_fr,
                        "firing_rate_h2": cluster_fr,
                    })
                    df_rate = df_rate[np.isfinite(df_rate["cluster_id"])].copy()
                    if not df_rate.empty:
                        df_rate["cluster_id"] = df_rate["cluster_id"].astype(int)
                    else:
                        df_rate = None
            table_data["df_firing_rate"] = df_rate
        if "df_depth" in needed_tables:
            cluster_ids = cache.get("cluster_ids")
            depths = _extract_cluster_depths(cache, cluster_ids)
            df_depth = None
            if cluster_ids is not None and depths is not None:
                cluster_ids_clean = pd.to_numeric(np.asarray(cluster_ids), errors="coerce")
                depths = np.asarray(depths, dtype=float)
                if len(cluster_ids_clean) == len(depths):
                    df_depth = pd.DataFrame({
                        "pid": pid,
                        "cluster_id": cluster_ids_clean,
                        "depth_h1": depths,
                        "depth_h2": depths,
                    })
                    df_depth = df_depth[np.isfinite(df_depth["cluster_id"])].copy()
                    if not df_depth.empty:
                        df_depth["cluster_id"] = df_depth["cluster_id"].astype(int)
                    else:
                        df_depth = None
            table_data["df_depth"] = df_depth

        region_names = sorted(df_units["region"].astype(str).unique().tolist())
        if selected_region_norm is not None:
            region_names = [r for r in region_names if r == selected_region_norm]
        if not region_names:
            continue

        has_spont = _has_spont_interval((cache.get("meta") or {}))
        for region_name in region_names:
            region_lookup = df_units.loc[
                df_units["region"].astype(str) == str(region_name),
                ["pid", "cluster_id", "region"],
            ].drop_duplicates()
            if region_lookup.empty:
                continue

            rl_x = _filter_region_lookup_for_spec(region_lookup, spec_x, has_spont)
            rl_y = _filter_region_lookup_for_spec(region_lookup, spec_y, has_spont)
            if rl_x is None or rl_x.empty or rl_y is None or rl_y.empty:
                continue

            df_var_x = _build_variable_table(table_data.get(table_name_x), spec_x, rl_x)
            df_var_y = _build_variable_table(table_data.get(table_name_y), spec_y, rl_y)
            if df_var_x is None or df_var_x.empty or df_var_y is None or df_var_y.empty:
                continue

            if _is_firing_rate_spec(spec_x):
                p_rel_x, p_rel_x_n = np.nan, 0
                s_rel_x, s_rel_x_n = np.nan, 0
            else:
                p_rel_x, p_rel_x_n = _pearsonr_with_n(df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]])
                s_rel_x, s_rel_x_n = _spearmanr_with_n(df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]])

            if _is_firing_rate_spec(spec_y):
                p_rel_y, p_rel_y_n = np.nan, 0
                s_rel_y, s_rel_y_n = np.nan, 0
            else:
                p_rel_y, p_rel_y_n = _pearsonr_with_n(df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]])
                s_rel_y, s_rel_y_n = _spearmanr_with_n(df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]])

            if str(var_x) == str(var_y):
                p_corr, p_corr_n = p_rel_x, p_rel_x_n
                s_corr, s_corr_n = s_rel_x, s_rel_x_n
            else:
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
                p_corr, p_corr_n = _pearsonr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])
                s_corr, s_corr_n = _spearmanr_with_n(mean_wide["mean_x"], mean_wide["mean_y"])

            p_total_rel = _calc_total_reliability(p_rel_x, p_rel_y)
            s_total_rel = _calc_total_reliability(s_rel_x, s_rel_y)

            rows.append(
                {
                    "region": str(region_name),
                    "pid": pid,
                    "n_units": int(region_lookup["cluster_id"].nunique()),
                    "pearson_corr": p_corr,
                    "pearson_reliability": p_total_rel,
                    "pearson_n": p_corr_n,
                    "spearman_corr": s_corr,
                    "spearman_reliability": s_total_rel,
                    "spearman_n": s_corr_n,
                }
            )

    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    dedup_cols = ["region", "pid"] if "region" in df_out.columns else ["pid"]
    df_out = df_out.drop_duplicates(subset=dedup_cols).sort_values(dedup_cols)
    return df_out.reset_index(drop=True)

@st.cache_data(show_spinner="Loading dashboard tables from saved cache...")
def load_precomputed_data():
    def _read_parquet_any(path):
        errors = []

        try:
            return pd.read_parquet(path)
        except Exception as e:
            errors.append(f"pandas.read_parquet failed: {e}")

        try:
            import polars as pl  # type: ignore
            return pl.read_parquet(path).to_pandas()
        except Exception as e:
            errors.append(f"polars.read_parquet failed: {e}")

        try:
            import duckdb  # type: ignore
            con = duckdb.connect()
            try:
                return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
            finally:
                con.close()
        except Exception as e:
            errors.append(f"duckdb read_parquet failed: {e}")

        raise RuntimeError(" | ".join(errors))

    def _read_label_min_text():
        try:
            with open(CACHE_DIR / "summary_metadata.pkl", "rb") as f:
                meta = pickle.load(f)
                return meta.get("label_min_text", "NA")
        except Exception:
            return "NA"

    def _load_from_pickle_cache():
        corr_pkl = CACHE_DIR / "summary_correlations.pkl"
        region_pkl = CACHE_DIR / "summary_region_counts.pkl"
        pids_pkl = CACHE_DIR / "summary_pids.pkl"
        arousal_pkl = CACHE_DIR / "summary_arousal_fractions.pkl"

        if not (corr_pkl.exists() and region_pkl.exists() and pids_pkl.exists()):
            raise FileNotFoundError("Pickle cache not found for one or more required tables.")

        df_corr = pd.read_pickle(corr_pkl)
        df_region_counts = pd.read_pickle(region_pkl)
        df_pids = pd.read_pickle(pids_pkl)
        df_arousal = pd.read_pickle(arousal_pkl) if arousal_pkl.exists() else pd.DataFrame()
        return df_corr, df_arousal, df_region_counts, df_pids, _read_label_min_text()

    try:
        return _load_from_pickle_cache()
    except Exception as e:
        pkl_err = str(e)

    try:
        df_corr = _read_parquet_any(CACHE_DIR / "summary_correlations.parquet")
        df_arousal = (
            _read_parquet_any(CACHE_DIR / "summary_arousal_fractions.parquet")
            if (CACHE_DIR / "summary_arousal_fractions.parquet").exists()
            else pd.DataFrame()
        )
        df_region_counts = _read_parquet_any(CACHE_DIR / "summary_region_counts.parquet")
        df_pids = _read_parquet_any(CACHE_DIR / "summary_pids.parquet")
        return df_corr, df_arousal, df_region_counts, df_pids, _read_label_min_text()
    except Exception as e:
        return None, None, None, None, f"Pickle load failed: {pkl_err} | Parquet load failed: {e}"


df_corr, df_arousal, df_region_counts, df_pids, label_min_text_or_err = load_precomputed_data()

plotly_dark_mode = st.toggle("Plotly dark mode", value=False)
PLOTLY_TEMPLATE = "plotly_dark" if plotly_dark_mode else "plotly_white"
pio.templates.default = PLOTLY_TEMPLATE

st.title("Region Dashboard (Fast Edition)")

if df_corr is None:
    st.error(
        "Could not load saved dashboard tables from `data/dashboard_region_cache`. "
        "No recomputation was run. "
        "Tried local pickle cache first, then parquet readers (pandas/pyarrow-fastparquet, polars, duckdb). "
        f"Error: {label_min_text_or_err}"
    )
    st.code("pip install pyarrow", language="bash")
    st.stop()

st.caption(f"Label threshold(s) applied during precomputation: {label_min_text_or_err}")

st.subheader("PID Summary")
_show_table(df_pids, width="stretch", max_rows=800)

good_only_toggle = st.toggle(
    "Use only good neurons (label=1) for correlation/reliability",
    value=False,
)
use_good_stpr = st.toggle(
    "Use Coupling computed from good neuron population",
    value=False,
)
avg_by_pid_toggle = st.toggle(
    "Average corr/rel per PID (within region)",
    value=False,
)
if "delay_mode" in df_corr.columns:
    df_corr = df_corr.copy()
    df_corr["delay_mode"] = (
        df_corr["delay_mode"].astype(str).str.lower().fillna("com")
    )
    has_com_rows = (df_corr["delay_mode"] == "com").any()
    if has_com_rows:
        df_corr = df_corr[df_corr["delay_mode"] == "com"].copy()
    else:
        st.warning("COM coupling-delay rows were not found; using all available rows.")

# Region Summary
df_rc_filtered = df_region_counts.query("good_only == @good_only_toggle")
st.subheader("Region Summary")
_show_table(df_rc_filtered.drop(columns=["good_only"], errors='ignore'), width="stretch", max_rows=800)


# ==========================================
# Correlation Matrices by Region
# ==========================================
st.subheader("Correlation Matrices by Region")
if df_rc_filtered.empty:
    st.info("No regions available for this selection.")
else:
    region_labels = [f"{row['region']} (n={row['n_neurons']})" for _, row in df_rc_filtered.sort_values("region").iterrows()]
    label_to_region = dict(zip(region_labels, df_rc_filtered.sort_values("region")["region"]))
    selected_label = st.selectbox("Select region", region_labels)
    selected_region = label_to_region.get(selected_label)

    if selected_region:
        df_mat = df_corr.query(
            "good_only == @good_only_toggle and "
            "use_good_stpr == @use_good_stpr and "
            "avg_by_pid == @avg_by_pid_toggle and "
            "arousal_group == 'all' and "
            "region == @selected_region"
        )
        
        if df_mat.empty:
            st.info("No correlation data available for this region with the current toggles.")
        else:
            names = list(pd.unique(df_mat["var1"]))
            n_vars = len(names)
            
            corr_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
            spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
            text_mat = np.empty((n_vars, n_vars), dtype=object)
            spearman_text = np.empty((n_vars, n_vars), dtype=object)
            
            name_idx = {n: i for i, n in enumerate(names)}
            
            for _, row in df_mat.iterrows():
                i, j = name_idx[row["var1"]], name_idx[row["var2"]]
                r_val = float(row["pearson_r"]) if pd.notnull(row["pearson_r"]) else np.nan
                s_val = float(row["spearman_rho"]) if pd.notnull(row["spearman_rho"]) else np.nan
                n_val = int(row["pearson_n"]) if pd.notnull(row["pearson_n"]) else 0
                sn_val = int(row["spearman_n"]) if pd.notnull(row["spearman_n"]) else 0
                
                corr_mat[i, j] = r_val
                spearman_mat[i, j] = s_val
                
                prefix = "rel=" if i == j else "r="
                text_mat[i, j] = f"{prefix}{r_val:.2f}<br>(n={n_val})" if np.isfinite(r_val) else f"{prefix}nan<br>(n={n_val})"
                
                prefix_s = "rel=" if i == j else "rho="
                spearman_text[i, j] = f"{prefix_s}{s_val:.2f}<br>(n={sn_val})" if np.isfinite(s_val) else f"{prefix_s}nan<br>(n={sn_val})"
                
            n_total_units = df_rc_filtered[df_rc_filtered["region"] == selected_region]["n_neurons"].values[0]
            avg_text = f" | avg across PIDs" if avg_by_pid_toggle else ""
            
            fig_pearson = go.Figure(data=go.Heatmap(
                z=corr_mat, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                text=text_mat, texttemplate="%{text}", hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>"
            ))
            fig_pearson.update_layout(
                title=f"Reliability (diag) + Pairwise Pearson (off-diag) | Region {selected_region} | N total: {n_total_units}{avg_text}",
                height=min(1000, max(500, 40 * n_vars + 200)), margin=dict(l=90, r=30, t=90, b=90), template=PLOTLY_TEMPLATE
            )
            fig_pearson.update_xaxes(tickangle=45)

            fig_spearman = go.Figure(data=go.Heatmap(
                z=spearman_mat, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                text=spearman_text, texttemplate="%{text}", hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>"
            ))
            fig_spearman.update_layout(
                title=f"Reliability (diag) + Pairwise Spearman (off-diag) | Region {selected_region} | N total: {n_total_units}{avg_text}",
                height=min(1000, max(500, 40 * n_vars + 200)), margin=dict(l=90, r=30, t=90, b=90), template=PLOTLY_TEMPLATE
            )
            fig_spearman.update_xaxes(tickangle=45)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Pearson**")
                st.plotly_chart(fig_pearson, width="stretch")
            with col_b:
                st.markdown("**Spearman**")
                st.plotly_chart(fig_spearman, width="stretch")


# ==========================================
# Scatter Plot Controls (Global)
# ==========================================
st.subheader("Scatter Plot Settings")
df_corr_mode = df_corr
if df_corr_mode.empty:
    st.info("No correlation rows available.")
    st.stop()
var_names_raw = pd.unique(df_corr_mode["var1"])
var_names = [v for v in var_names_raw if pd.notnull(v)]
col1, col2, col3 = st.columns(3)
with col1:
    var_x = st.selectbox("Variable 1", var_names, index=0)
with col2:
    default_idx = 1 if len(var_names) > 1 else 0
    var_y = st.selectbox("Variable 2", var_names, index=default_idx)
with col3:
    min_neurons = int(st.number_input("Min neurons per region", min_value=0, value=30, step=10))

def _prepare_scatter_df(df_source, method="pearson"):
    df_x_diag = df_source.query("var1 == @var_x and var2 == @var_x")
    df_y_diag = df_source.query("var1 == @var_y and var2 == @var_y")
    df_xy = df_source.query("var1 == @var_x and var2 == @var_y")

    col_r = "pearson_r" if method == "pearson" else "spearman_rho"

    df_plot = df_xy.merge(df_x_diag[["region", col_r]], on="region", suffixes=("", "_x_diag")) \
                   .merge(df_y_diag[["region", col_r]], on="region", suffixes=("", "_y_diag"))
    
    # Merge with region counts to filter by min_neurons
    df_plot = df_plot.merge(df_rc_filtered[["region", "n_neurons"]], on="region", how="inner")
    df_plot = df_plot[df_plot["n_neurons"] >= min_neurons]
    
    if df_plot.empty:
        return pd.DataFrame()

    df_plot["reliability_x"] = df_plot[f"{col_r}_x_diag"]
    df_plot["reliability_y"] = df_plot[f"{col_r}_y_diag"]
    df_plot["corr"] = df_plot[col_r]

    df_plot["reliability"] = df_plot.apply(
        lambda row: _calc_total_reliability(row["reliability_x"], row["reliability_y"]),
        axis=1,
    )
    df_plot = df_plot[np.isfinite(df_plot["corr"]) & np.isfinite(df_plot["reliability"])]
    return df_plot


def _aggregate_region_points_from_pid(df_pid_all, method="pearson"):
    corr_col = "pearson_corr" if method == "pearson" else "spearman_corr"
    rel_col = "pearson_reliability" if method == "pearson" else "spearman_reliability"

    if df_pid_all is None or df_pid_all.empty:
        return pd.DataFrame()
    if "region" not in df_pid_all.columns:
        return pd.DataFrame()

    df_plot = df_pid_all.copy()
    df_plot = df_plot[np.isfinite(df_plot[corr_col]) & np.isfinite(df_plot[rel_col])]
    if df_plot.empty:
        return pd.DataFrame()

    df_plot = (
        df_plot.groupby("region", as_index=False)
        .agg(
            corr=(corr_col, "mean"),
            reliability=(rel_col, "mean"),
            n_pids=("pid", "nunique"),
        )
    )
    df_plot = df_plot.merge(df_rc_filtered[["region", "n_neurons"]], on="region", how="inner")
    df_plot = df_plot[df_plot["n_neurons"] >= min_neurons]
    return df_plot


def _build_scatter_from_region_points(df_plot, method="pearson", group_title="all", return_meta=False):
    if df_plot is None or df_plot.empty:
        if return_meta:
            return None, pd.DataFrame(), None, None
        return None

    region_colors = _build_region_colors(df_plot["region"])
    highlight_regions = {"VISp", "MOs", "CP", "CA1", "SCm", "ZI", "AUDp", "GRN", "PO", "VPM", "VISa", "MOp"}

    fig = go.Figure()
    for _, row in df_plot.sort_values("region").iterrows():
        region = row["region"]
        color = region_colors.get(region)
        marker = dict(size=8, color=color) if color else dict(size=8)
        if region in highlight_regions:
            marker["line"] = dict(color="black", width=1)

        fig.add_trace(
            go.Scatter(
                x=[row["reliability"]],
                y=[row["corr"]],
                mode="markers",
                name=region,
                marker=marker,
                showlegend=region in highlight_regions,
                hovertemplate=(
                    f"Region: {region}<br>"
                    f"corr={row['corr']:.2f}<br>"
                    f"rel={row['reliability']:.2f}<br>"
                    f"n={int(row['n_neurons'])}<br>"
                    f"n_pids={int(row.get('n_pids', np.nan)) if pd.notnull(row.get('n_pids', np.nan)) else 0}<extra></extra>"
                ),
            )
        )

    avg_text = " (avg across PIDs)" if avg_by_pid_toggle else ""
    title_prefix = f"{group_title} | " if group_title != "all" else ""
    fig.update_layout(
        title=f"{method.title()} correlation vs total reliability | {title_prefix}{var_x} vs {var_y} | regions with >= {min_neurons} neurons{avg_text}",
        xaxis_title="Total reliability (sqrt(rel1 * rel2))",
        yaxis_title="Correlation between variables",
        template=PLOTLY_TEMPLATE,
        legend=dict(x=1.02, y=1, yanchor="top"),
        margin=dict(l=80, r=200, t=90, b=70),
        height=600,
        width=900,
    )
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

    if return_meta:
        return fig, df_plot.copy(), x_range, y_range
    return fig


def _build_scatter(df_source, method="pearson", group_title="all", return_meta=False):
    df_plot = _prepare_scatter_df(df_source, method=method)
    if df_plot.empty:
        if return_meta:
            return None, pd.DataFrame(), None, None
        return None

    region_colors = _build_region_colors(df_plot["region"])
    highlight_regions = {"VISp", "MOs", "CP", "CA1", "SCm", "ZI", "AUDp", "GRN", "PO", "VPM", "VISa", "MOp"}

    fig = go.Figure()
    for _, row in df_plot.sort_values("region").iterrows():
        region = row["region"]
        color = region_colors.get(region)
        marker = dict(size=8, color=color) if color else dict(size=8)
        if region in highlight_regions: marker["line"] = dict(color="black", width=1)
        
        fig.add_trace(go.Scatter(
            x=[row["reliability"]], y=[row["corr"]], mode="markers", name=region, marker=marker,
            showlegend=region in highlight_regions,
            hovertemplate=f"Region: {region}<br>corr={row['corr']:.2f}<br>rel={row['reliability']:.2f}<br>n={row['n_neurons']}<extra></extra>"
        ))

    avg_text = " (avg across PIDs)" if avg_by_pid_toggle else ""
    title_prefix = f"{group_title} | " if group_title != "all" else ""
    fig.update_layout(
        title=f"{method.title()} correlation vs total reliability | {title_prefix}{var_x} vs {var_y} | regions with >= {min_neurons} neurons{avg_text}",
        xaxis_title="Total reliability (sqrt(rel1 * rel2))", yaxis_title="Correlation between variables",
        template=PLOTLY_TEMPLATE, legend=dict(x=1.02, y=1, yanchor="top"), margin=dict(l=80, r=200, t=90, b=70),
        height=600, width=900
    )
    x_range = _compute_axis_range(df_plot["reliability"].to_numpy(dtype=float))
    y_range = _compute_axis_range(df_plot["corr"].to_numpy(dtype=float))
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    if x_range is not None and y_range is not None:
        min_val = float(min(x_range[0], y_range[0]))
        max_val = float(max(x_range[1], y_range[1]))
    else:
        x_vals, y_vals = df_plot["reliability"].to_numpy(dtype=float), df_plot["corr"].to_numpy(dtype=float)
        min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
        max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
    if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))

    if return_meta:
        return fig, df_plot.copy(), x_range, y_range
    return fig


def _build_pid_detail_scatter(
    df_pid_points,
    method,
    selected_region,
    axis_x_range,
    axis_y_range,
    avg_point=None,
    marker_color=None,
):
    corr_col = "pearson_corr" if method == "pearson" else "spearman_corr"
    rel_col = "pearson_reliability" if method == "pearson" else "spearman_reliability"
    n_col = "pearson_n" if method == "pearson" else "spearman_n"

    df_plot = df_pid_points.copy()
    df_plot = df_plot[np.isfinite(df_plot[corr_col]) & np.isfinite(df_plot[rel_col])]
    if df_plot.empty:
        return None

    color = marker_color if marker_color else "#1f77b4"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[rel_col].to_numpy(dtype=float),
            y=df_plot[corr_col].to_numpy(dtype=float),
            mode="markers",
            name="PIDs",
            marker=dict(size=9, color=color, opacity=0.7, line=dict(color="rgba(0,0,0,0.35)", width=0.8)),
            customdata=np.stack(
                [
                    df_plot["pid"].astype(str).to_numpy(),
                    df_plot[n_col].to_numpy(dtype=float),
                    df_plot["n_units"].to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "PID: %{customdata[0]}<br>"
                "corr=%{y:.2f}<br>"
                "rel=%{x:.2f}<br>"
                "n_corr=%{customdata[1]:.0f}<br>"
                "n_units=%{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )

    if avg_point is not None:
        avg_rel = avg_point.get("reliability", np.nan)
        avg_corr = avg_point.get("corr", np.nan)
        if np.isfinite(avg_rel) and np.isfinite(avg_corr):
            fig.add_trace(
                go.Scatter(
                    x=[float(avg_rel)],
                    y=[float(avg_corr)],
                    mode="markers",
                    name="Region average",
                    marker=dict(
                        size=17,
                        color=color,
                        line=dict(color="black", width=2),
                        symbol="circle",
                    ),
                    hovertemplate=(
                        "Region average<br>"
                        f"Region: {selected_region}<br>"
                        "corr=%{y:.2f}<br>"
                        "rel=%{x:.2f}<extra></extra>"
                    ),
                )
            )

    avg_text = " (avg across PIDs)" if avg_by_pid_toggle else ""
    fig.update_layout(
        title=f"{method.title()} per-PID correlation vs total reliability | Region {selected_region}{avg_text}",
        xaxis_title="Total reliability (sqrt(rel1 * rel2))",
        yaxis_title="Correlation between variables",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=90, b=70),
        legend=dict(x=1.02, y=1, yanchor="top"),
        height=550,
    )

    if axis_x_range is not None:
        fig.update_xaxes(range=axis_x_range)
    if axis_y_range is not None:
        fig.update_yaxes(range=axis_y_range)

    if axis_x_range is not None and axis_y_range is not None:
        dmin = float(min(axis_x_range[0], axis_y_range[0]))
        dmax = float(max(axis_x_range[1], axis_y_range[1]))
        if np.isfinite(dmin) and np.isfinite(dmax) and dmin < dmax:
            fig.add_shape(
                type="line",
                x0=dmin,
                y0=dmin,
                x1=dmax,
                y1=dmax,
                line=dict(color="red", dash="dash"),
            )
    return fig


def _compute_pid_average_point(df_pid_points, method):
    corr_col = "pearson_corr" if method == "pearson" else "spearman_corr"
    rel_col = "pearson_reliability" if method == "pearson" else "spearman_reliability"

    df_plot = df_pid_points.copy()
    df_plot = df_plot[np.isfinite(df_plot[corr_col]) & np.isfinite(df_plot[rel_col])]
    if df_plot.empty:
        return None

    return {
        "corr": float(np.nanmean(df_plot[corr_col].to_numpy(dtype=float))),
        "reliability": float(np.nanmean(df_plot[rel_col].to_numpy(dtype=float))),
    }

# ==========================================
# Correlation vs Reliability Scatter
# ==========================================
st.subheader("Correlation vs Reliability by Region")
df_scat_all = df_corr.query(
    "good_only == @good_only_toggle and "
    "use_good_stpr == @use_good_stpr and "
    "avg_by_pid == @avg_by_pid_toggle and "
    "arousal_group == 'all'"
)

df_pid_all_for_scatter = None
if avg_by_pid_toggle:
    df_pid_all_for_scatter = _compute_region_pid_points(
        None,
        var_x,
        var_y,
        bool(good_only_toggle),
        bool(use_good_stpr),
    )
    df_plot_p_agg = _aggregate_region_points_from_pid(df_pid_all_for_scatter, method="pearson")
    df_plot_s_agg = _aggregate_region_points_from_pid(df_pid_all_for_scatter, method="spearman")
    fig_p, df_plot_p, x_range_p, y_range_p = _build_scatter_from_region_points(
        df_plot_p_agg,
        method="pearson",
        return_meta=True,
    )
    fig_s, df_plot_s, x_range_s, y_range_s = _build_scatter_from_region_points(
        df_plot_s_agg,
        method="spearman",
        return_meta=True,
    )
elif df_scat_all.empty:
    fig_p, df_plot_p, x_range_p, y_range_p = None, pd.DataFrame(), None, None
    fig_s, df_plot_s, x_range_s, y_range_s = None, pd.DataFrame(), None, None
else:
    fig_p, df_plot_p, x_range_p, y_range_p = _build_scatter(df_scat_all, "pearson", return_meta=True)
    fig_s, df_plot_s, x_range_s, y_range_s = _build_scatter(df_scat_all, "spearman", return_meta=True)

if (fig_p is None) and (fig_s is None):
    st.info("No data available for scatter plots.")
else:
    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("**Pearson**")
        if fig_p:
            st.plotly_chart(fig_p, width="stretch")
        else:
            st.info("No data for Pearson plot.")
    with col_s:
        st.markdown("**Spearman**")
        if fig_s:
            st.plotly_chart(fig_s, width="stretch")
        else:
            st.info("No data for Spearman plot.")

    if avg_by_pid_toggle:
        st.markdown("**Per-PID Detail by Region**")
        regions_p = df_plot_p["region"].astype(str).tolist() if "region" in df_plot_p.columns else []
        regions_s = df_plot_s["region"].astype(str).tolist() if "region" in df_plot_s.columns else []
        region_options = sorted(
            set(regions_p) | set(regions_s)
        )
        if not region_options:
            st.info("No eligible regions for per-PID detail plots.")
        else:
            selected_pid_region = st.selectbox(
                "Select brain region for per-PID plots",
                region_options,
                key="pid_region_scatter_select",
            )
            df_pid_points = _compute_region_pid_points(
                selected_pid_region,
                var_x,
                var_y,
                bool(good_only_toggle),
                bool(use_good_stpr),
            )
            if df_pid_all_for_scatter is not None and not df_pid_all_for_scatter.empty and "region" in df_pid_all_for_scatter.columns:
                df_pid_points = df_pid_all_for_scatter[
                    df_pid_all_for_scatter["region"].astype(str) == str(selected_pid_region)
                ].copy()
            if df_pid_points.empty:
                st.info("No per-PID data available for the selected region/variables.")
            else:
                region_color = _build_region_colors([selected_pid_region]).get(selected_pid_region)
                avg_point_p = _compute_pid_average_point(df_pid_points, method="pearson")
                avg_point_s = _compute_pid_average_point(df_pid_points, method="spearman")

                fig_pid_p = _build_pid_detail_scatter(
                    df_pid_points,
                    method="pearson",
                    selected_region=selected_pid_region,
                    axis_x_range=x_range_p,
                    axis_y_range=y_range_p,
                    avg_point=avg_point_p,
                    marker_color=region_color,
                )
                fig_pid_s = _build_pid_detail_scatter(
                    df_pid_points,
                    method="spearman",
                    selected_region=selected_pid_region,
                    axis_x_range=x_range_s,
                    axis_y_range=y_range_s,
                    avg_point=avg_point_s,
                    marker_color=region_color,
                )

                col_pp, col_ps = st.columns(2)
                with col_pp:
                    st.markdown("**Pearson (per PID)**")
                    if fig_pid_p:
                        st.plotly_chart(fig_pid_p, width="stretch")
                    else:
                        st.info("No per-PID Pearson points.")
                with col_ps:
                    st.markdown("**Spearman (per PID)**")
                    if fig_pid_s:
                        st.plotly_chart(fig_pid_s, width="stretch")
                    else:
                        st.info("No per-PID Spearman points.")


# ==========================================
# Arousal-Split Analysis
# ==========================================
st.subheader("Arousal-Split Correlation Analysis")
df_split = df_corr.query(
    "good_only == @good_only_toggle and "
    "use_good_stpr == @use_good_stpr and "
    "avg_by_pid == @avg_by_pid_toggle and "
    "arousal_group in ['arousal_plus', 'arousal_minus']"
)

if df_split.empty:
    st.info("Arousal-group split data is unavailable or not precomputed.")
else:
    arousal_groups = [("arousal_plus", "Arousal +"), ("arousal_minus", "Arousal -")]
    for group_key, group_title in arousal_groups:
        st.markdown(f"**{group_title}**")
        df_group = df_split.query("arousal_group == @group_key")
        
        fig_p_split = _build_scatter(df_group, "pearson", group_title)
        fig_s_split = _build_scatter(df_group, "spearman", group_title)
        
        col_p, col_s = st.columns(2)
        with col_p:
            st.markdown("**Pearson**")
            if fig_p_split: st.plotly_chart(fig_p_split, width="stretch")
            else: st.info("No data.")
        with col_s:
            st.markdown("**Spearman**")
            if fig_s_split: st.plotly_chart(fig_s_split, width="stretch")
            else: st.info("No data.")


# ==========================================
# Arousal Fractions
# ==========================================
st.subheader("Arousal + vs Arousal - Fractions by Region")
df_ar = df_arousal.query("good_only == @good_only_toggle and avg_by_pid == @avg_by_pid_toggle")
if df_ar.empty:
    st.info("Arousal fractions data is unavailable.")
else:
    region_colors = _build_region_colors(df_ar["region"])
    fig_arousal = go.Figure()
    for _, row in df_ar.sort_values("region").iterrows():
        region = str(row["region"])
        color = region_colors.get(region)
        marker = dict(size=9, color=color) if color else dict(size=9)
        fig_arousal.add_trace(go.Scatter(
            x=[float(row["frac_minus_pct"])], y=[float(row["frac_plus_pct"])], mode="markers", marker=marker,
            showlegend=False,
            hovertemplate=(f"Region: {region}<br>Arousal-: {float(row['frac_minus_pct']):.1f}%<br>"
                           f"Arousal+: {float(row['frac_plus_pct']):.1f}%<br>"
                           f"n_neurons={int(row['n_neurons'])}<br>n_pids={int(row['n_pids'])}<extra></extra>")
        ))
    avg_text = " (avg across PIDs)" if avg_by_pid_toggle else ""
    fig_arousal.update_layout(
        title=f"Arousal + vs Arousal - Fractions by Region{avg_text}",
        xaxis_title="Fraction of Arousal- neurons (%)", yaxis_title="Fraction of Arousal+ neurons (%)",
        template=PLOTLY_TEMPLATE, margin=dict(l=80, r=40, t=80, b=70), height=620, width=850
    )
    fig_arousal.update_xaxes(title_font=dict(color="blue"), tickfont=dict(color="blue"))
    fig_arousal.update_yaxes(title_font=dict(color="red"), tickfont=dict(color="red"))
    st.plotly_chart(fig_arousal, width="stretch")

# ==========================================
# Multi-Level Region Plot (Reference Figure)
# ==========================================
st.markdown("---")
st.subheader("Region Coupling Reliability and Median Strength")
st.caption(
    "Reliability aggregation: "
    + ("avg across PIDs within region" if avg_by_pid_toggle else "all neurons within region")
)

coupling_source = st.selectbox("Coupling source for regional summary", ["Spont", "ITI", "Task"], index=0, key="ml_source")

@st.cache_data(show_spinner="Extracting medians from raw cache (takes ~1 min on first load)...")
def load_raw_coupling_for_medians():
    CACHE_DIR_RAW = BASE_PATH / "data" / "dashboard_cache"
    if not CACHE_DIR_RAW.exists(): return None
    cache_paths = list(CACHE_DIR_RAW.glob("*.pkl"))
    
    rows = []
    
    for path in cache_paths:
        with open(path, "rb") as f:
            cache = pickle.load(f)
            
        pid = cache.get("pid", path.stem)
        
        df_res = cache.get("df_res")
        if df_res is None or df_res.empty: continue
        if "label" not in df_res.columns: continue
        
        if "region" in df_res.columns: region_col = "region"
        elif "acronym" in df_res.columns: region_col = "acronym"
        else: continue
        
        df_labels = df_res[["cluster_id", "label", region_col]].copy()
        
        for key, name in [("df_coupling", "Spont"), ("df_coupling_iti", "ITI"), ("df_coupling_task", "Task")]:
            df = cache.get(key)
            if df is not None and not df.empty:
                df = df.copy()
                df["pid"] = pid
                df["source"] = name
                if name == "Spont":
                    if "coupling_strength_h1" in df.columns and "coupling_strength_h2" in df.columns:
                        df["strength"] = df[["coupling_strength_h1", "coupling_strength_h2"]].mean(axis=1)
                    else: continue
                else:
                    if "coupling_strength_odd" in df.columns and "coupling_strength_even" in df.columns:
                        df["strength"] = df[["coupling_strength_odd", "coupling_strength_even"]].mean(axis=1)
                    else: continue
                    
                df_merged = df.merge(df_labels, on="cluster_id", how="inner")
                # Use all good neurons (label >= label_min from meta, which user configured usually to 0.59)
                df_merged = df_merged[pd.to_numeric(df_merged["label"], errors="coerce") >= 0.59]
                
                if not df_merged.empty:
                    df_merged["cluster_id"] = pd.to_numeric(df_merged["cluster_id"])
                    df_merged = df_merged[np.isfinite(df_merged["cluster_id"])].copy()
                    df_merged["region"] = df_merged[region_col].astype(str)
                    
                    rows.append(df_merged[["pid", "cluster_id", "region", "source", "strength"]])
                    
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


with st.spinner("Preparing regional multilevel scatter plots (computing median)..."):
    df_raw = load_raw_coupling_for_medians()

if df_raw is not None and not df_raw.empty:
    from plotly.subplots import make_subplots
    
    source_var_delay = f"Coupling Delay ({coupling_source})"
    source_var_strength = f"Coupling Strength ({coupling_source})"
    
    # Keep "all good neurons" and coupling-from-all-neuron-population settings as before,
    # but let avg_by_pid follow the global toggle.
    df_f = df_corr.query(
        "var1 == @source_var_delay and var2 == @source_var_delay and "
        "arousal_group == 'all' and good_only == True and use_good_stpr == False and "
        "avg_by_pid == @avg_by_pid_toggle"
    )
    df_g = df_corr.query(
        "var1 == @source_var_strength and var2 == @source_var_strength and "
        "arousal_group == 'all' and good_only == True and use_good_stpr == False and "
        "avg_by_pid == @avg_by_pid_toggle"
    )
    
    # Strength aggregation follows avg_by_pid toggle:
    # False: median across all neurons in region
    # True: mean of per-PID regional medians
    df_h_raw = df_raw[df_raw["source"] == coupling_source]
    if avg_by_pid_toggle:
        df_h = (
            df_h_raw.groupby(["region", "pid"])["strength"].median().reset_index()
            .groupby("region", as_index=False)
            .agg(
                strength=("strength", "mean"),
                n_pids_strength=("pid", "nunique"),
            )
        )
        strength_subplot_title = "Mean per-PID median coupling strength"
        strength_trace_name = "Mean PID median strength"
        strength_hover_name = "Mean PID median strength"
        strength_yaxis_title = "Mean PID med. str."
    else:
        df_h = df_h_raw.groupby("region", as_index=False)["strength"].median()
        df_h["n_pids_strength"] = (
            df_h_raw.groupby("region")["pid"].nunique().reindex(df_h["region"]).to_numpy()
        )
        strength_subplot_title = "Median coupling strength"
        strength_trace_name = "Median strength"
        strength_hover_name = "Median strength"
        strength_yaxis_title = "Median str."
    
    # Brain Region Category Map for x-axis Ordering
    allen_lookup = _get_allen_lookup()
    if allen_lookup is not None:
        order_by_acr = allen_lookup.get("order_by_acr", {})
        category_by_acr = allen_lookup.get("category_by_acr", {})
        region_map = {}
        region_orders = {}
        for reg in df_h["region"].unique():
            reg_s = str(reg)
            region_map[reg_s] = category_by_acr.get(reg_s, "Unknown")
            region_orders[reg_s] = int(order_by_acr.get(reg_s, 999999))
    else:
        region_map = {reg: "Unknown" for reg in df_h["region"].unique()}
        region_orders = {reg: 999999 for reg in df_h["region"].unique()}
        
    df_f = df_f.rename(columns={"pearson_r": "delay_rel"})
    df_g = df_g.rename(columns={"pearson_r": "strength_rel"})
    
    df_merged = df_h[["region", "strength", "n_pids_strength"]].merge(
        df_f[["region", "delay_rel"]],
        on="region",
        how="left",
    )
    df_merged = df_merged.merge(df_g[["region", "strength_rel"]], on="region", how="left")
    
    # Apply global 'min_neurons' constraint
    df_rc_filtered_local = df_region_counts.query("good_only == True")
    df_merged = df_merged.merge(df_rc_filtered_local[["region", "n_neurons"]], on="region", how="inner")
    df_merged = df_merged[df_merged["n_neurons"] >= min_neurons]
    
    category_order = ["Isocortex", "HPF", "OLF", "CTXsp", "Striatum", "Pallidum", "Thal.", "Hyp.", "Midbrain", "Pons", "Medulla", "Cereb.", "Unknown", "Other"]
    df_merged["category"] = df_merged["region"].map(region_map).fillna("Unknown")
    df_merged["allen_order"] = df_merged["region"].map(region_orders).fillna(999999)
    if (df_merged["allen_order"] != 999999).any():
        df_merged = df_merged[df_merged["allen_order"] != 999999]
    df_merged = df_merged[df_merged["region"] != "root"]
    
    # Sort primarily by the anatomical order to ensure grouping is strict
    category_rank = {cat: i for i, cat in enumerate(category_order)}
    df_merged["category_rank"] = df_merged["category"].map(category_rank).fillna(len(category_rank)).astype(int)
    df_merged = df_merged.sort_values(["category_rank", "allen_order", "region"])
    
    if not df_merged.empty:
        fig_multi = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Coupling delay reliability", "Coupling strength reliability", strength_subplot_title),
        )
        
        regions = df_merged["region"].tolist()
        x_vals = list(range(len(regions)))
        n_neurons_list = df_merged["n_neurons"].tolist()
        region_colors_dict = _build_region_colors(regions)
        marker_colors = [region_colors_dict.get(r, "whitesmoke") for r in regions]
        
        
        fig_multi.add_trace(go.Scatter(x=x_vals, y=df_merged["delay_rel"], mode="markers", 
                                       marker={"color": marker_colors, "size": 8}, name="Delay Rel", showlegend=False,
                                       hovertemplate="Region: %{customdata[0]}<br>Delay Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
                                       customdata=np.stack((df_merged["region"], df_merged["n_neurons"]), axis=-1)), row=1, col=1)
        fig_multi.add_trace(go.Scatter(x=x_vals, y=df_merged["strength_rel"], mode="markers", 
                                       marker={"color": marker_colors, "size": 8}, name="Strength Rel", showlegend=False,
                                       hovertemplate="Region: %{customdata[0]}<br>Strength Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
                                       customdata=np.stack((df_merged["region"], df_merged["n_neurons"]), axis=-1)), row=2, col=1)
        fig_multi.add_trace(go.Scatter(x=x_vals, y=df_merged["strength"], mode="markers", 
                                       marker={"color": marker_colors, "size": 8}, name=strength_trace_name, showlegend=False,
                                       hovertemplate=(
                                           f"Region: %{{customdata[0]}}<br>{strength_hover_name}: %{{y:.2f}}<br>"
                                           "Neurons: %{customdata[1]}<br>PIDs: %{customdata[2]}<extra></extra>"
                                       ),
                                       customdata=np.stack(
                                           (
                                               df_merged["region"],
                                               df_merged["n_neurons"],
                                               df_merged["n_pids_strength"],
                                           ),
                                           axis=-1,
                                       )), row=3, col=1)
        
        fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Add broad region x-axis group labels as annotations
        df_merged_reset = df_merged.reset_index(drop=True)
        cat_positions = df_merged_reset.groupby("category", sort=False).apply(lambda x: x.index.to_numpy().mean())
        
        # Pre-resolve true acronyms for broad groups so they get the generic Allen broad-level color
        cat_to_acronym = {"Isocortex": "Isocortex", "HPF": "HPF", "Striatum": "STR", "Pallidum": "PAL", 
                          "Thal.": "TH", "Hyp.": "HY", "Midbrain": "MB", "Pons": "P", "Medulla": "MY", 
                          "Cereb.": "CB", "OLF": "OLF", "CTXsp": "CTXsp"}
        cat_acronyms_list = [cat_to_acronym.get(c, c) for c in cat_positions.index]
        cat_colors_dict = _build_region_colors(cat_acronyms_list)
        
        for cat, pos in cat_positions.items():
            if cat not in ["Unknown", "Other"]:
                acronym = cat_to_acronym.get(cat, cat)
                font_color = cat_colors_dict.get(acronym, "gray")
                if font_color == "rgb(255,255,255)": font_color = "gray"
                fig_multi.add_annotation(x=pos, y=-0.15, xref="x", yref="paper", text=cat, 
                                         showarrow=False, font={"color": font_color, "size": 14}, xanchor="center")

        # Draw per-region x labels as annotations so each label can match its marker color.
        for xi, reg in enumerate(regions):
            label_color = region_colors_dict.get(reg, "gray")
            if label_color == "rgb(255,255,255)":
                label_color = "gray"
            fig_multi.add_annotation(
                x=xi,
                y=-0.06,
                xref="x",
                yref="paper",
                text=reg,
                showarrow=False,
                textangle=90,
                font={"color": label_color, "size": 11},
                xanchor="right",
                yanchor="middle",
            )

        fig_multi.update_layout(height=750, template=PLOTLY_TEMPLATE, margin=dict(b=180, t=50))
        fig_multi.update_yaxes(range=[-1.1, 1.1], title_text="Delay rel.", row=1, col=1)
        fig_multi.update_yaxes(range=[-1.1, 1.1], title_text="Strength rel.", row=2, col=1)
        fig_multi.update_yaxes(title_text=strength_yaxis_title, row=3, col=1)
        fig_multi.update_xaxes(title_text="", tickmode="array", tickvals=x_vals, showticklabels=False)
        fig_multi.update_layout(xaxis=dict(tickmode='array', tickvals=x_vals, showticklabels=False))
        
        st.plotly_chart(fig_multi, use_container_width=True)

        df_rel = df_merged[["region", "delay_rel", "strength_rel", "n_neurons"]].copy()
        df_rel = df_rel[
            np.isfinite(df_rel["delay_rel"].to_numpy(dtype=float))
            & np.isfinite(df_rel["strength_rel"].to_numpy(dtype=float))
        ].copy()

        if not df_rel.empty:
            rel_colors = [region_colors_dict.get(str(r), "whitesmoke") for r in df_rel["region"]]
            fig_rel = go.Figure()
            fig_rel.add_trace(
                go.Scatter(
                    x=df_rel["delay_rel"],
                    y=df_rel["strength_rel"],
                    mode="markers",
                    marker=dict(color=rel_colors, size=9),
                    showlegend=False,
                    customdata=np.stack(
                        (df_rel["region"].astype(str), df_rel["n_neurons"].astype(int)),
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Region: %{customdata[0]}<br>"
                        "Delay rel: %{x:.2f}<br>"
                        "Strength rel: %{y:.2f}<br>"
                        "Neurons: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
            fig_rel.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_rel.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rel.add_shape(
                type="line",
                x0=-1.1,
                y0=-1.1,
                x1=1.1,
                y1=1.1,
                line=dict(color="gray", dash="dot"),
            )
            fig_rel.update_layout(
                title="Delay Reliability vs Strength Reliability",
                xaxis_title="Delay rel.",
                yaxis_title="Strength rel.",
                template=PLOTLY_TEMPLATE,
                height=520,
                margin=dict(l=70, r=40, t=70, b=60),
            )
            fig_rel.update_xaxes(range=[-1.1, 1.1])
            fig_rel.update_yaxes(range=[-1.1, 1.1])
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.info("No finite delay/strength reliability pairs available for the added scatter plot.")
    else:
        st.info("No matching data found for the multi-level plot.")
else:
    st.info("Raw coupling strength medians could not be extracted.")
