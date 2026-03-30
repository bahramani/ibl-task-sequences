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


def _add_region_metadata(df, region_col="region"):
    if not isinstance(df, pd.DataFrame):
        return df

    df_out = df.copy()
    if df_out.empty or region_col not in df_out.columns:
        return df_out

    regions = df_out[region_col].astype(str)
    allen_lookup = _get_allen_lookup() or {}
    color_by_acr = allen_lookup.get("color_by_acr", {})
    order_by_acr = allen_lookup.get("order_by_acr", {})
    category_by_acr = allen_lookup.get("category_by_acr", {})

    inferred_color = regions.map(color_by_acr)
    if "allen_color" not in df_out.columns:
        df_out["allen_color"] = inferred_color
    else:
        df_out["allen_color"] = df_out["allen_color"].where(df_out["allen_color"].notna(), inferred_color)

    inferred_category = regions.map(category_by_acr).fillna("Unknown")
    if "category" not in df_out.columns:
        df_out["category"] = inferred_category
    else:
        df_out["category"] = df_out["category"].where(df_out["category"].notna(), inferred_category)
    df_out["category"] = df_out["category"].fillna("Unknown")

    inferred_order = pd.to_numeric(regions.map(order_by_acr), errors="coerce")
    if "allen_order" not in df_out.columns:
        df_out["allen_order"] = inferred_order
    else:
        existing_order = pd.to_numeric(df_out["allen_order"], errors="coerce")
        df_out["allen_order"] = existing_order.where(existing_order.notna(), inferred_order)
    df_out["allen_order"] = pd.to_numeric(df_out["allen_order"], errors="coerce").fillna(999999).astype(int)

    category_rank = {cat: idx for idx, cat in enumerate(CANONICAL_CATEGORY_ORDER)}
    inferred_rank = df_out["category"].map(category_rank)
    if "category_rank" not in df_out.columns:
        df_out["category_rank"] = inferred_rank
    else:
        existing_rank = pd.to_numeric(df_out["category_rank"], errors="coerce")
        df_out["category_rank"] = existing_rank.where(existing_rank.notna(), inferred_rank)
    df_out["category_rank"] = pd.to_numeric(df_out["category_rank"], errors="coerce").fillna(len(category_rank)).astype(int)
    return df_out


def _apply_region_scatter_filters(df_plot, region_meta_df, selected_categories=None):
    if not isinstance(df_plot, pd.DataFrame):
        return df_plot

    df_out = df_plot.copy()
    if df_out.empty or "region" not in df_out.columns:
        return df_out

    if isinstance(region_meta_df, pd.DataFrame) and not region_meta_df.empty:
        meta_cols = [
            col for col in ["region", "category", "allen_color", "allen_order", "category_rank"]
            if col in region_meta_df.columns
        ]
        if meta_cols:
            df_out = df_out.merge(
                region_meta_df[meta_cols].drop_duplicates(subset=["region"]),
                on="region",
                how="left",
                suffixes=("", "_meta"),
            )
            for col in meta_cols:
                if col == "region":
                    continue
                meta_col = f"{col}_meta"
                if meta_col not in df_out.columns:
                    continue
                if col in df_out.columns:
                    df_out[col] = df_out[col].where(df_out[col].notna(), df_out[meta_col])
                else:
                    df_out[col] = df_out[meta_col]
                df_out = df_out.drop(columns=[meta_col])

    if selected_categories is not None:
        if "category" not in df_out.columns:
            df_out["category"] = "Unknown"
        df_out = df_out[df_out["category"].isin(selected_categories)].copy()

    return df_out


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
SEQ_REGION_CACHE_DIR = BASE_PATH / "data" / "processed" / "16_ibl_seq_batch_comp"
DEFAULT_LABEL_MIN = 0.59
CANONICAL_CATEGORY_ORDER = [
    "Isocortex",
    "HPF",
    "OLF",
    "CTXsp",
    "Striatum",
    "Pallidum",
    "Thal.",
    "Hyp.",
    "Midbrain",
    "Pons",
    "Medulla",
    "Cereb.",
    "Unknown",
    "Other",
]
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

if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

CORR_VARIABLES = [
    {"name": "Depth", "df": "df_depth", "v1": "depth_h1", "v2": "depth_h2"},
    {"name": "Firing Rate", "df": "df_firing_rate", "v1": "firing_rate_h1", "v2": "firing_rate_h2"},
    {"name": "Correlation to Whisking", "df": "df_arousal_corr", "v1": "arousal_corr_abs_h1", "v2": "arousal_corr_abs_h2"},
    {"name": "Delay to Stim On", "df": "df_res", "v1": "delay_stimOn_times_odd", "v2": "delay_stimOn_times_even"},
    {"name": "Response to Stim On", "df": "df_res", "v1": "response_zmean_stimOn_times_odd", "v2": "response_zmean_stimOn_times_even"},
    {"name": "Delay to First Move", "df": "df_res", "v1": "delay_firstMovement_times_odd", "v2": "delay_firstMovement_times_even"},
    {"name": "Response to First Move", "df": "df_res", "v1": "response_zmean_firstMovement_times_odd", "v2": "response_zmean_firstMovement_times_even"},
    {"name": "Delay to Feedback", "df": "df_res", "v1": "delay_feedback_times_odd", "v2": "delay_feedback_times_even"},
    {"name": "Response to Feedback", "df": "df_res", "v1": "response_zmean_feedback_times_odd", "v2": "response_zmean_feedback_times_even"},
    {"name": "Delay to Feedback (Correct Trials)", "df": "df_res", "v1": "delay_feedback_correct_times_odd", "v2": "delay_feedback_correct_times_even"},
    {"name": "Delay to Feedback (Incorrect Trials)", "df": "df_res", "v1": "delay_feedback_incorrect_times_odd", "v2": "delay_feedback_incorrect_times_even"},
    {"name": "Response to Feedback (Correct Trials)", "df": "df_res", "v1": "response_zmean_feedback_correct_times_odd", "v2": "response_zmean_feedback_correct_times_even"},
    {"name": "Response to Feedback (Incorrect Trials)", "df": "df_res", "v1": "response_zmean_feedback_incorrect_times_odd", "v2": "response_zmean_feedback_incorrect_times_even"},
    {"name": "Delay to Whisking Events", "df": "df_res", "v1": "delay_wh_brief_times_spont_odd", "v2": "delay_wh_brief_times_spont_even"},
    {"name": "Response to Whisking Events", "df": "df_res", "v1": "response_zmean_wh_brief_times_spont_odd", "v2": "response_zmean_wh_brief_times_spont_even"},
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


def _format_metric_value(value, n=None, prefix=""):
    try:
        value_num = float(value)
    except Exception:
        value_num = np.nan
    if not np.isfinite(value_num):
        metric_text = "nan"
    else:
        metric_text = f"{value_num:.3f}"
    if n is None:
        return f"{prefix}{metric_text}"
    try:
        n_val = int(n)
    except Exception:
        n_val = 0
    return f"{prefix}{metric_text} (n={n_val})"


def _select_preferred_metric(pearson_val, spearman_val):
    try:
        pearson_num = float(pearson_val)
    except Exception:
        pearson_num = np.nan
    try:
        spearman_num = float(spearman_val)
    except Exception:
        spearman_num = np.nan
    p_ok = np.isfinite(pearson_num)
    s_ok = np.isfinite(spearman_num)
    if p_ok and s_ok:
        if spearman_num > pearson_num:
            return spearman_num, "Spearman"
        return pearson_num, "Pearson"
    if p_ok:
        return pearson_num, "Pearson"
    if s_ok:
        return spearman_num, "Spearman"
    return np.nan, "NA"


def _safe_polyfit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return np.nan, np.nan
    x_fit = x[mask]
    y_fit = y[mask]
    if np.unique(x_fit).size < 2:
        return np.nan, np.nan
    try:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        return float(slope), float(intercept)
    except Exception:
        return np.nan, np.nan


@st.cache_data(show_spinner="Loading neuron-level region scatter data from raw cache...")
def _compute_region_neuron_scatter_data(selected_region, var_x, var_y, good_only, use_good_stpr):
    if not RAW_CACHE_DIR.exists():
        return {
            "points": pd.DataFrame(),
            "pearson_corr": np.nan,
            "pearson_corr_n": 0,
            "spearman_corr": np.nan,
            "spearman_corr_n": 0,
            "pearson_rel_x": np.nan,
            "pearson_rel_x_n": 0,
            "spearman_rel_x": np.nan,
            "spearman_rel_x_n": 0,
            "pearson_rel_y": np.nan,
            "pearson_rel_y_n": 0,
            "spearman_rel_y": np.nan,
            "spearman_rel_y_n": 0,
        }

    selected_region_norm = None if selected_region in (None, "") else str(selected_region)
    if selected_region_norm is None:
        return {
            "points": pd.DataFrame(),
            "pearson_corr": np.nan,
            "pearson_corr_n": 0,
            "spearman_corr": np.nan,
            "spearman_corr_n": 0,
            "pearson_rel_x": np.nan,
            "pearson_rel_x_n": 0,
            "spearman_rel_x": np.nan,
            "spearman_rel_x_n": 0,
            "pearson_rel_y": np.nan,
            "pearson_rel_y_n": 0,
            "spearman_rel_y": np.nan,
            "spearman_rel_y_n": 0,
        }

    spec_map = {spec["name"]: spec for spec in _build_corr_variables()}
    spec_x = spec_map.get(var_x)
    spec_y = spec_map.get(var_y)
    if spec_x is None or spec_y is None:
        return {
            "points": pd.DataFrame(),
            "pearson_corr": np.nan,
            "pearson_corr_n": 0,
            "spearman_corr": np.nan,
            "spearman_corr_n": 0,
            "pearson_rel_x": np.nan,
            "pearson_rel_x_n": 0,
            "spearman_rel_x": np.nan,
            "spearman_rel_x_n": 0,
            "pearson_rel_y": np.nan,
            "pearson_rel_y_n": 0,
            "spearman_rel_y": np.nan,
            "spearman_rel_y_n": 0,
        }

    point_frames = []
    rel_x_frames = []
    rel_y_frames = []
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
        df_units = df_units[df_units["region"].astype(str) == selected_region_norm].copy()
        if df_units.empty:
            continue

        table_name_x = spec_x.get("df")
        table_name_y = spec_y.get("df")
        needed_tables = {table_name_x, table_name_y}
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
                table_data["df_arousal_corr"] = df_ar[
                    ["pid", "cluster_id", "arousal_corr_abs_h1", "arousal_corr_abs_h2"]
                ]
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
                    df_rate = pd.DataFrame(
                        {
                            "pid": pid,
                            "cluster_id": cluster_ids_clean,
                            "firing_rate_h1": cluster_fr,
                            "firing_rate_h2": cluster_fr,
                        }
                    )
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
                    df_depth = pd.DataFrame(
                        {
                            "pid": pid,
                            "cluster_id": cluster_ids_clean,
                            "depth_h1": depths,
                            "depth_h2": depths,
                        }
                    )
                    df_depth = df_depth[np.isfinite(df_depth["cluster_id"])].copy()
                    if not df_depth.empty:
                        df_depth["cluster_id"] = df_depth["cluster_id"].astype(int)
                    else:
                        df_depth = None
            table_data["df_depth"] = df_depth

        has_spont = _has_spont_interval((cache.get("meta") or {}))
        region_lookup = df_units[["pid", "cluster_id", "region"]].drop_duplicates()
        rl_x = _filter_region_lookup_for_spec(region_lookup, spec_x, has_spont)
        rl_y = _filter_region_lookup_for_spec(region_lookup, spec_y, has_spont)
        if rl_x is None or rl_x.empty or rl_y is None or rl_y.empty:
            continue

        df_var_x = _build_variable_table(table_data.get(table_name_x), spec_x, rl_x)
        df_var_y = _build_variable_table(table_data.get(table_name_y), spec_y, rl_y)
        if df_var_x is None or df_var_x.empty or df_var_y is None or df_var_y.empty:
            continue

        point_df = region_lookup[["pid", "cluster_id", "region"]].drop_duplicates()
        point_df = point_df.merge(
            df_var_x[["pid", "cluster_id", "mean"]].rename(columns={"mean": "mean_x"}),
            on=["pid", "cluster_id"],
            how="inner",
        )
        point_df = point_df.merge(
            df_var_y[["pid", "cluster_id", "mean"]].rename(columns={"mean": "mean_y"}),
            on=["pid", "cluster_id"],
            how="inner",
        )
        if not point_df.empty:
            point_frames.append(point_df)

        if not _is_firing_rate_spec(spec_x):
            rel_x_frames.append(
                df_var_x[["pid", "cluster_id", spec_x["v1"], spec_x["v2"]]].rename(
                    columns={spec_x["v1"]: "v1", spec_x["v2"]: "v2"}
                )
            )
        if not _is_firing_rate_spec(spec_y):
            rel_y_frames.append(
                df_var_y[["pid", "cluster_id", spec_y["v1"], spec_y["v2"]]].rename(
                    columns={spec_y["v1"]: "v1", spec_y["v2"]: "v2"}
                )
            )

    if point_frames:
        df_points = pd.concat(point_frames, ignore_index=True)
        df_points = df_points.drop_duplicates(subset=["pid", "cluster_id"]).copy()
        df_points = df_points[
            np.isfinite(pd.to_numeric(df_points["mean_x"], errors="coerce"))
            & np.isfinite(pd.to_numeric(df_points["mean_y"], errors="coerce"))
        ].copy()
    else:
        df_points = pd.DataFrame(columns=["pid", "cluster_id", "region", "mean_x", "mean_y"])

    pearson_corr, pearson_corr_n = _pearsonr_with_n(
        df_points.get("mean_x", pd.Series(dtype=float)),
        df_points.get("mean_y", pd.Series(dtype=float)),
    )
    spearman_corr, spearman_corr_n = _spearmanr_with_n(
        df_points.get("mean_x", pd.Series(dtype=float)),
        df_points.get("mean_y", pd.Series(dtype=float)),
    )

    if rel_x_frames:
        df_rel_x = pd.concat(rel_x_frames, ignore_index=True)
        df_rel_x = df_rel_x.drop_duplicates(subset=["pid", "cluster_id"]).copy()
        pearson_rel_x, pearson_rel_x_n = _pearsonr_with_n(df_rel_x["v1"], df_rel_x["v2"])
        spearman_rel_x, spearman_rel_x_n = _spearmanr_with_n(df_rel_x["v1"], df_rel_x["v2"])
    else:
        pearson_rel_x, pearson_rel_x_n = np.nan, 0
        spearman_rel_x, spearman_rel_x_n = np.nan, 0

    if rel_y_frames:
        df_rel_y = pd.concat(rel_y_frames, ignore_index=True)
        df_rel_y = df_rel_y.drop_duplicates(subset=["pid", "cluster_id"]).copy()
        pearson_rel_y, pearson_rel_y_n = _pearsonr_with_n(df_rel_y["v1"], df_rel_y["v2"])
        spearman_rel_y, spearman_rel_y_n = _spearmanr_with_n(df_rel_y["v1"], df_rel_y["v2"])
    else:
        pearson_rel_y, pearson_rel_y_n = np.nan, 0
        spearman_rel_y, spearman_rel_y_n = np.nan, 0

    return {
        "points": df_points.reset_index(drop=True),
        "pearson_corr": pearson_corr,
        "pearson_corr_n": int(pearson_corr_n),
        "spearman_corr": spearman_corr,
        "spearman_corr_n": int(spearman_corr_n),
        "pearson_rel_x": pearson_rel_x,
        "pearson_rel_x_n": int(pearson_rel_x_n),
        "spearman_rel_x": spearman_rel_x,
        "spearman_rel_x_n": int(spearman_rel_x_n),
        "pearson_rel_y": pearson_rel_y,
        "pearson_rel_y_n": int(pearson_rel_y_n),
        "spearman_rel_y": spearman_rel_y,
        "spearman_rel_y_n": int(spearman_rel_y_n),
    }


def _build_region_neuron_scatter(
    selected_region,
    var_x,
    var_y,
    good_only,
    use_good_stpr,
):
    scatter_data = _compute_region_neuron_scatter_data(
        selected_region,
        var_x,
        var_y,
        bool(good_only),
        bool(use_good_stpr),
    )
    df_points = scatter_data.get("points", pd.DataFrame()).copy()
    if df_points.empty:
        return None

    region_color = _build_region_colors([selected_region]).get(selected_region, "#1f77b4")
    x_vals = pd.to_numeric(df_points["mean_x"], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df_points["mean_y"], errors="coerce").to_numpy(dtype=float)
    df_points = df_points[np.isfinite(x_vals) & np.isfinite(y_vals)].copy()
    if df_points.empty:
        return None
    x_vals = pd.to_numeric(df_points["mean_x"], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(df_points["mean_y"], errors="coerce").to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=str(selected_region),
            marker=dict(
                size=8,
                color=region_color,
                opacity=0.78,
                line=dict(color="rgba(0,0,0,0.25)", width=0.6),
            ),
            customdata=np.stack(
                (
                    df_points["pid"].astype(str).to_numpy(),
                    df_points["cluster_id"].astype(str).to_numpy(),
                ),
                axis=-1,
            ),
            hovertemplate=(
                "Region: "
                + str(selected_region)
                + "<br>PID: %{customdata[0]}<br>cluster_id: %{customdata[1]}<br>"
                + f"{var_x}: "
                + "%{x:.3f}<br>"
                + f"{var_y}: "
                + "%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    x_range = _compute_axis_range(x_vals)
    y_range = _compute_axis_range(y_vals)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    combined_range = None
    if x_range is not None and y_range is not None:
        combined_range = [float(min(x_range[0], y_range[0])), float(max(x_range[1], y_range[1]))]
    elif np.isfinite(x_vals).any() and np.isfinite(y_vals).any():
        combined_range = _compute_axis_range(np.concatenate([x_vals, y_vals]))

    if combined_range is not None:
        fig.add_shape(
            type="line",
            x0=float(combined_range[0]),
            y0=float(combined_range[0]),
            x1=float(combined_range[1]),
            y1=float(combined_range[1]),
            line=dict(color="red", dash="dash", width=1.5),
        )

    slope, intercept = _safe_polyfit_line(x_vals, y_vals)
    if np.isfinite(slope) and np.isfinite(intercept):
        fit_x0 = float(np.nanmin(x_vals))
        fit_x1 = float(np.nanmax(x_vals))
        fit_y0 = float(slope * fit_x0 + intercept)
        fit_y1 = float(slope * fit_x1 + intercept)
        fig.add_trace(
            go.Scatter(
                x=[fit_x0, fit_x1],
                y=[fit_y0, fit_y1],
                mode="lines",
                line=dict(color="black", dash="dash", width=1.8),
                name="Fit",
                hovertemplate="Fit<extra></extra>",
                showlegend=False,
            )
        )

    rel_x_best, rel_x_label = _select_preferred_metric(
        scatter_data.get("pearson_rel_x", np.nan),
        scatter_data.get("spearman_rel_x", np.nan),
    )
    rel_y_best, rel_y_label = _select_preferred_metric(
        scatter_data.get("pearson_rel_y", np.nan),
        scatter_data.get("spearman_rel_y", np.nan),
    )

    if PLOTLY_TEMPLATE == "plotly_dark":
        bgcolor = "rgba(15, 23, 42, 0.82)"
        bordercolor = "rgba(226, 232, 240, 0.55)"
        font_color = "#f8fafc"
    else:
        bgcolor = "rgba(255, 255, 255, 0.9)"
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
            _format_metric_value(scatter_data.get("pearson_corr", np.nan), scatter_data.get("pearson_corr_n", 0), "Pearson r = ")
            + "<br>"
            + _format_metric_value(scatter_data.get("spearman_corr", np.nan), scatter_data.get("spearman_corr_n", 0), "Spearman rho = ")
            + "<br>"
            + _format_metric_value(rel_x_best, None, f"Rel {var_x} ({rel_x_label}) = ")
            + "<br>"
            + _format_metric_value(rel_y_best, None, f"Rel {var_y} ({rel_y_label}) = ")
            + "<br>"
            + f"Neurons = {int(len(df_points))}<br>"
            + f"PIDs = {int(df_points['pid'].astype(str).nunique())}<br>"
            + (
                f"Fit: y = {slope:.3f}x {intercept:+.3f}"
                if np.isfinite(slope) and np.isfinite(intercept)
                else "Fit: unavailable"
            )
        ),
        font=dict(size=12, color=font_color),
        bordercolor=bordercolor,
        borderwidth=1,
        bgcolor=bgcolor,
    )

    fig.update_layout(
        title=f"Neuron Scatter | Region {selected_region} | {var_x} vs {var_y}",
        xaxis_title=str(var_x),
        yaxis_title=str(var_y),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=90, b=70),
        height=620,
    )
    _apply_cartesian_grid(fig)
    return fig


def _read_saved_table_bundle(cache_dir, stem):
    pkl_path = cache_dir / f"{stem}.pkl"
    parquet_path = cache_dir / f"{stem}.parquet"
    csv_path = cache_dir / f"{stem}.csv"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find cached table for '{stem}'.")


def _seq_diag_mask(df_pairs):
    if df_pairs is None or df_pairs.empty or "is_diagonal" not in df_pairs.columns:
        return np.zeros(0, dtype=bool)
    series = df_pairs["is_diagonal"]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    ).to_numpy(dtype=bool)


def _extract_seq_diag_reliability(df_pairs, var_key, col_name):
    if df_pairs is None or df_pairs.empty:
        return pd.DataFrame(columns=["region", col_name])
    required_cols = {"region", "var_x_key", "var_y_key", "pearson_r"}
    if not required_cols.issubset(set(df_pairs.columns)):
        return pd.DataFrame(columns=["region", col_name])

    diag_mask = _seq_diag_mask(df_pairs)
    if diag_mask.size != len(df_pairs):
        return pd.DataFrame(columns=["region", col_name])

    df_rel = df_pairs.loc[
        diag_mask
        & (df_pairs["var_x_key"].astype(str) == str(var_key))
        & (df_pairs["var_y_key"].astype(str) == str(var_key)),
        ["region", "pearson_r"],
    ].copy()
    if df_rel.empty:
        return pd.DataFrame(columns=["region", col_name])
    df_rel["region"] = df_rel["region"].astype(str)
    return df_rel.rename(columns={"pearson_r": col_name})


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


@st.cache_data(show_spinner=False)
def load_seq_region_pair_data():
    try:
        df_pairs_pooled = _read_saved_table_bundle(SEQ_REGION_CACHE_DIR, "summary_region_pairs_pooled")
        df_pairs_pidmean = _read_saved_table_bundle(SEQ_REGION_CACHE_DIR, "summary_region_pairs_pidmean")
        df_region_counts_seq = _read_saved_table_bundle(SEQ_REGION_CACHE_DIR, "summary_region_counts")
        return df_pairs_pooled, df_pairs_pidmean, df_region_counts_seq, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(e)


df_corr, df_arousal, df_region_counts, df_pids, label_min_text_or_err = load_precomputed_data()
seq_region_pairs_pooled, seq_region_pairs_pidmean, seq_region_counts_seq, seq_region_err = load_seq_region_pair_data()

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
# Whisking vs First-Move Reliability
# ==========================================
st.subheader("Whisking vs First-Move Reliability")
seq_region_pair_table = (
    seq_region_pairs_pidmean if avg_by_pid_toggle else seq_region_pairs_pooled
).copy()
df_spont_whisk_rel = _extract_seq_diag_reliability(
    seq_region_pair_table,
    "spont_whisk_event_delay",
    "spont_whisk_rel",
)
df_first_move_rel_seq = _extract_seq_diag_reliability(
    seq_region_pair_table,
    "first_move_delay",
    "first_move_rel",
)

df_whisk_move = df_spont_whisk_rel.merge(df_first_move_rel_seq, on="region", how="inner")
if not df_whisk_move.empty:
    df_seq_counts_local = seq_region_counts_seq.rename(
        columns={"n_units_total": "n_neurons_seq"}
    )
    df_whisk_move = df_whisk_move.merge(
        df_seq_counts_local[["region", "n_neurons_seq"]],
        on="region",
        how="left",
    )
    df_whisk_move = df_whisk_move.merge(
        df_rc_filtered[["region", "n_neurons"]],
        on="region",
        how="left",
    )
    df_whisk_move["n_neurons_plot"] = pd.to_numeric(
        df_whisk_move.get("n_neurons"), errors="coerce"
    ).fillna(pd.to_numeric(df_whisk_move.get("n_neurons_seq"), errors="coerce"))
    df_whisk_move = df_whisk_move[
        np.isfinite(df_whisk_move["spont_whisk_rel"].to_numpy(dtype=float))
        & np.isfinite(df_whisk_move["first_move_rel"].to_numpy(dtype=float))
        & np.isfinite(df_whisk_move["n_neurons_plot"].to_numpy(dtype=float))
    ].copy()

if df_whisk_move.empty:
    if seq_region_err:
        st.info(
            "Sequence-region reliability tables are unavailable for the whisking/first-move scatter: "
            + str(seq_region_err)
        )
    else:
        st.info("No finite spontaneous-whisk/first-move reliability pairs are available.")
else:
    region_colors_whisk_move = _build_region_colors(df_whisk_move["region"].astype(str).tolist())
    marker_colors_whisk_move = [
        region_colors_whisk_move.get(str(region), "whitesmoke")
        for region in df_whisk_move["region"]
    ]

    fig_whisk_move = go.Figure()
    fig_whisk_move.add_trace(
        go.Scatter(
            x=df_whisk_move["spont_whisk_rel"],
            y=df_whisk_move["first_move_rel"],
            mode="markers",
            marker=dict(color=marker_colors_whisk_move, size=9),
            showlegend=False,
            customdata=np.stack(
                (
                    df_whisk_move["region"].astype(str),
                    df_whisk_move["n_neurons_plot"].astype(int),
                ),
                axis=-1,
            ),
            hovertemplate=(
                "Region: %{customdata[0]}<br>"
                "Spont whisk rel: %{x:.2f}<br>"
                "First-move rel: %{y:.2f}<br>"
                "Neurons: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    combined_range = _compute_axis_range(
        np.concatenate(
            [
                df_whisk_move["spont_whisk_rel"].to_numpy(dtype=float),
                df_whisk_move["first_move_rel"].to_numpy(dtype=float),
            ]
        )
    )
    if combined_range is not None:
        fig_whisk_move.update_xaxes(range=combined_range)
        fig_whisk_move.update_yaxes(range=combined_range)
        line_min, line_max = float(combined_range[0]), float(combined_range[1])
    else:
        line_min = float(
            np.nanmin(
                np.concatenate(
                    [
                        df_whisk_move["spont_whisk_rel"].to_numpy(dtype=float),
                        df_whisk_move["first_move_rel"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        line_max = float(
            np.nanmax(
                np.concatenate(
                    [
                        df_whisk_move["spont_whisk_rel"].to_numpy(dtype=float),
                        df_whisk_move["first_move_rel"].to_numpy(dtype=float),
                    ]
                )
            )
        )

    if np.isfinite(line_min) and np.isfinite(line_max) and line_min < line_max:
        fig_whisk_move.add_shape(
            type="line",
            x0=line_min,
            y0=line_min,
            x1=line_max,
            y1=line_max,
            line=dict(color="red", dash="dash"),
        )

    avg_text = " (avg across PIDs)" if avg_by_pid_toggle else ""
    fig_whisk_move.update_layout(
        title=f"Spont Whisk Delay Reliability vs First-Move Delay Reliability{avg_text}",
        xaxis_title="Spont Whisk Delay reliability",
        yaxis_title="First Move Delay reliability",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=80, b=70),
        height=620,
    )
    _apply_cartesian_grid(fig_whisk_move)
    st.plotly_chart(fig_whisk_move, width="stretch")


df_corr_mode = df_corr.copy()
if df_corr_mode.empty:
    var_names = []
    var_x = None
    var_y = None
else:
    registry_var_names = [spec["name"] for spec in _build_corr_variables()]
    var_names_raw = [str(v) for v in pd.unique(df_corr_mode["var1"]) if pd.notnull(v)]
    available_var_name_set = set(var_names_raw)
    var_names = [name for name in registry_var_names if name in available_var_name_set]
    var_names.extend([name for name in var_names_raw if name not in set(var_names)])
    missing_registry_vars = [name for name in registry_var_names if name not in available_var_name_set]
    if missing_registry_vars:
        st.warning(
            "Some registered variables are missing from the saved region-summary cache and "
            "will not appear until `08_batch_compute_region.py` is rerun: "
            + ", ".join(missing_registry_vars)
        )
    default_var_x = var_names[0] if var_names else None
    default_var_y = var_names[1] if len(var_names) > 1 else default_var_x
    if default_var_x is not None:
        if (
            "scatter_var_x" not in st.session_state
            or st.session_state.get("scatter_var_x") not in var_names
        ):
            st.session_state["scatter_var_x"] = default_var_x
    if default_var_y is not None:
        if (
            "scatter_var_y" not in st.session_state
            or st.session_state.get("scatter_var_y") not in var_names
        ):
            st.session_state["scatter_var_y"] = default_var_y
    if default_var_x is not None:
        if (
            "region_neuron_var_x" not in st.session_state
            or st.session_state.get("region_neuron_var_x") not in var_names
        ):
            st.session_state["region_neuron_var_x"] = st.session_state.get(
                "scatter_var_x",
                default_var_x,
            )
    if default_var_y is not None:
        if (
            "region_neuron_var_y" not in st.session_state
            or st.session_state.get("region_neuron_var_y") not in var_names
        ):
            st.session_state["region_neuron_var_y"] = st.session_state.get(
                "scatter_var_y",
                default_var_y,
            )
    var_x = st.session_state.get("scatter_var_x", default_var_x)
    var_y = st.session_state.get("scatter_var_y", default_var_y)


# ==========================================
# Correlation Matrices by Region
# ==========================================
st.subheader("Correlation Matrices by Region")
selected_region = None
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

        st.markdown("**Neuron Scatter for Selected Region**")
        neuron_var_x = st.session_state.get("region_neuron_var_x")
        neuron_var_y = st.session_state.get("region_neuron_var_y")
        neuron_col1, neuron_col2 = st.columns(2)
        with neuron_col1:
            selected_neuron_var_x = neuron_var_x if neuron_var_x in var_names else var_names[0]
            st.selectbox(
                "Neuron scatter variable 1",
                var_names,
                index=var_names.index(selected_neuron_var_x),
                key="region_neuron_var_x",
            )
        with neuron_col2:
            neuron_default_idx = 1 if len(var_names) > 1 else 0
            selected_neuron_var_y = (
                neuron_var_y if neuron_var_y in var_names else var_names[neuron_default_idx]
            )
            st.selectbox(
                "Neuron scatter variable 2",
                var_names,
                index=var_names.index(selected_neuron_var_y),
                key="region_neuron_var_y",
            )
        neuron_var_x = st.session_state.get("region_neuron_var_x", var_names[0])
        neuron_var_y = st.session_state.get(
            "region_neuron_var_y",
            var_names[1] if len(var_names) > 1 else var_names[0],
        )
        st.caption("These controls are independent from the global Scatter Plot Settings below.")
        if neuron_var_x is None or neuron_var_y is None:
            st.info("Select neuron-scatter variables above to render the neuron-level scatter.")
        else:
            fig_region_neuron = _build_region_neuron_scatter(
                selected_region,
                neuron_var_x,
                neuron_var_y,
                bool(good_only_toggle),
                bool(use_good_stpr),
            )
            if fig_region_neuron is None:
                st.info("No neuron-level values are available for the selected region and variables.")
            else:
                st.plotly_chart(fig_region_neuron, width="stretch")


# ==========================================
# Scatter Plot Controls (Global)
# ==========================================
st.subheader("Scatter Plot Settings")
if df_corr_mode.empty:
    st.info("No correlation rows available.")
    st.stop()
col1, col2, col3 = st.columns(3)
with col1:
    st.selectbox(
        "Variable 1",
        var_names,
        index=var_names.index(st.session_state.get("scatter_var_x", var_names[0])),
        key="scatter_var_x",
    )
with col2:
    default_idx = 1 if len(var_names) > 1 else 0
    selected_var_y = st.session_state.get("scatter_var_y", var_names[default_idx])
    if selected_var_y not in var_names:
        selected_var_y = var_names[default_idx]
    st.selectbox(
        "Variable 2",
        var_names,
        index=var_names.index(selected_var_y),
        key="scatter_var_y",
    )
with col3:
    min_neurons = int(
        st.number_input(
            "Min neurons per region",
            min_value=0,
            value=int(st.session_state.get("scatter_min_neurons", 30)),
            step=10,
            key="scatter_min_neurons",
        )
    )
var_x = st.session_state.get("scatter_var_x", var_names[0])
var_y = st.session_state.get("scatter_var_y", var_names[1] if len(var_names) > 1 else var_names[0])

df_scatter_region_meta = _add_region_metadata(
    df_rc_filtered[["region", "n_neurons"]].drop_duplicates(subset=["region"]).copy()
)
present_category_set = set(df_scatter_region_meta.get("category", pd.Series(dtype="object")).dropna().astype(str))
scatter_categories = [
    cat for cat in CANONICAL_CATEGORY_ORDER
    if (cat in present_category_set) and (cat not in {"Unknown", "Other"})
]
extra_scatter_categories = sorted(
    present_category_set.difference(CANONICAL_CATEGORY_ORDER).difference({"Unknown", "Other"})
)
scatter_categories.extend(extra_scatter_categories)

if scatter_categories:
    st.markdown("**Broad anatomy categories**")
    cat_cols = st.columns(6)
    selected_scatter_categories = []
    for idx, cat in enumerate(scatter_categories):
        checked = cat_cols[idx % 6].checkbox(
            cat,
            value=True,
            key=f"scatter_cat_{cat}",
        )
        if checked:
            selected_scatter_categories.append(cat)
    if not selected_scatter_categories:
        st.warning("Select at least one broad anatomy category to plot region points.")
else:
    selected_scatter_categories = None
    st.caption("Broad anatomy category metadata is unavailable for the current region set.")

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
    df_plot = _apply_region_scatter_filters(
        df_plot,
        df_scatter_region_meta,
        selected_categories=selected_scatter_categories,
    )
    
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
    df_plot = _apply_region_scatter_filters(
        df_plot,
        df_scatter_region_meta,
        selected_categories=selected_scatter_categories,
    )
    return df_plot


def _build_scatter_from_region_points(df_plot, method="pearson", group_title="all", return_meta=False):
    if df_plot is None or df_plot.empty:
        if return_meta:
            return None, pd.DataFrame(), None, None
        return None

    region_colors = _build_region_colors(df_plot["region"])
    fig = go.Figure()
    for _, row in df_plot.sort_values("region").iterrows():
        region = row["region"]
        color = region_colors.get(region)
        marker = dict(size=8, color=color) if color else dict(size=8)
        if region in HIGHLIGHT_REGIONS:
            marker["line"] = dict(color="black", width=1)

        fig.add_trace(
            go.Scatter(
                x=[row["reliability"]],
                y=[row["corr"]],
                mode="markers",
                name=region,
                marker=marker,
                showlegend=region in HIGHLIGHT_REGIONS,
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
    fig = go.Figure()
    for _, row in df_plot.sort_values("region").iterrows():
        region = row["region"]
        color = region_colors.get(region)
        marker = dict(size=8, color=color) if color else dict(size=8)
        if region in HIGHLIGHT_REGIONS: marker["line"] = dict(color="black", width=1)
        
        fig.add_trace(go.Scatter(
            x=[row["reliability"]], y=[row["corr"]], mode="markers", name=region, marker=marker,
            showlegend=region in HIGHLIGHT_REGIONS,
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
    else:
        x_vals, y_vals = df_plot["reliability"].to_numpy(dtype=float), df_plot["corr"].to_numpy(dtype=float)
        min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
        max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
    if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))

    _add_corr_rel_summary_annotation(
        fig,
        df_plot["corr"].to_numpy(dtype=float),
        df_plot["reliability"].to_numpy(dtype=float),
    )

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
    _apply_cartesian_grid(fig)

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
    _add_corr_rel_summary_annotation(
        fig,
        df_plot[corr_col].to_numpy(dtype=float),
        df_plot[rel_col].to_numpy(dtype=float),
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
    _apply_cartesian_grid(fig_arousal)
    st.plotly_chart(fig_arousal, width="stretch")

# ==========================================
# Multi-Level Region Plot (Reference Figure)
# ==========================================
st.markdown("---")
st.subheader("Region Reliability Summary")
st.caption(
    "Reliability aggregation: "
    + ("avg across PIDs within region" if avg_by_pid_toggle else "all neurons within region")
)

coupling_source = st.selectbox("Coupling source for regional summary", ["Spont", "ITI", "Task"], index=0, key="ml_source")
from plotly.subplots import make_subplots

source_var_strength = f"Coupling Strength ({coupling_source})"
source_var_delay = f"Coupling Delay ({coupling_source})"

# Keep "all good neurons" and coupling-from-all-neuron-population settings as before,
# but let avg_by_pid follow the global toggle. Delay/whisk event rows come from the
# cached sequence-region comparison tables used by 17_ibl_seq_regions_dash.py.
df_strength = df_corr.query(
    "var1 == @source_var_strength and var2 == @source_var_strength and "
    "arousal_group == 'all' and good_only == True and use_good_stpr == False and "
    "avg_by_pid == @avg_by_pid_toggle"
)
df_delay = df_corr.query(
    "var1 == @source_var_delay and var2 == @source_var_delay and "
    "arousal_group == 'all' and good_only == True and use_good_stpr == False and "
    "avg_by_pid == @avg_by_pid_toggle"
)
df_seq_region_pairs = (
    seq_region_pairs_pidmean if avg_by_pid_toggle else seq_region_pairs_pooled
).copy()
df_first_move = _extract_seq_diag_reliability(
    df_seq_region_pairs,
    "first_move_delay",
    "first_move_rel",
)
df_task_whisk = _extract_seq_diag_reliability(
    df_seq_region_pairs,
    "task_whisk_event_delay",
    "task_whisk_rel",
)
df_spont_whisk = _extract_seq_diag_reliability(
    df_seq_region_pairs,
    "spont_whisk_event_delay",
    "spont_whisk_rel",
)

df_strength = df_strength.rename(columns={"pearson_r": "strength_rel"})
df_delay = df_delay.rename(columns={"pearson_r": "delay_rel"})

region_series = [
    df_src["region"].astype(str)
    for df_src in (df_strength, df_delay, df_first_move, df_task_whisk, df_spont_whisk)
    if ("region" in df_src.columns) and (not df_src.empty)
]
if region_series:
    df_merged = pd.DataFrame({"region": pd.unique(pd.concat(region_series, ignore_index=True))})
else:
    df_merged = pd.DataFrame(columns=["region"])

if not df_merged.empty:
    df_merged = df_merged.merge(df_strength[["region", "strength_rel"]], on="region", how="left")
    df_merged = df_merged.merge(df_delay[["region", "delay_rel"]], on="region", how="left")
    df_merged = df_merged.merge(df_first_move[["region", "first_move_rel"]], on="region", how="left")
    df_merged = df_merged.merge(df_task_whisk[["region", "task_whisk_rel"]], on="region", how="left")
    df_merged = df_merged.merge(df_spont_whisk[["region", "spont_whisk_rel"]], on="region", how="left")
    category_order = ["Isocortex", "HPF", "OLF", "CTXsp", "Striatum", "Pallidum", "Thal.", "Hyp.", "Midbrain", "Pons", "Medulla", "Cereb.", "Unknown", "Other"]

    # Apply global 'min_neurons' constraint.
    df_rc_filtered_local = df_region_counts.query("good_only == True").rename(
        columns={"n_neurons": "n_neurons_dashboard"}
    )
    df_merged = df_merged.merge(
        df_rc_filtered_local[["region", "n_neurons_dashboard"]],
        on="region",
        how="left",
    )
    if not seq_region_counts_seq.empty:
        df_seq_region_meta = seq_region_counts_seq.rename(
            columns={"n_units_total": "n_neurons_seq"}
        )
        seq_meta_cols = [
            col
            for col in ["region", "n_neurons_seq", "allen_color", "allen_order", "category", "category_rank"]
            if col in df_seq_region_meta.columns
        ]
        df_merged = df_merged.merge(df_seq_region_meta[seq_meta_cols], on="region", how="left")
    df_merged["n_neurons_plot"] = pd.to_numeric(
        df_merged.get("n_neurons_dashboard"), errors="coerce"
    ).fillna(pd.to_numeric(df_merged.get("n_neurons_seq"), errors="coerce"))
    df_merged = df_merged[
        np.isfinite(df_merged["n_neurons_plot"].to_numpy(dtype=float))
        & (df_merged["n_neurons_plot"].to_numpy(dtype=float) >= float(min_neurons))
    ].copy()

    has_seq_order = (
        "allen_order" in df_merged.columns
        and pd.to_numeric(df_merged["allen_order"], errors="coerce").notna().any()
    )
    if has_seq_order:
        df_merged["allen_order"] = pd.to_numeric(df_merged["allen_order"], errors="coerce").fillna(999999)
        if "category" not in df_merged.columns:
            df_merged["category"] = "Unknown"
        df_merged["category"] = df_merged["category"].fillna("Unknown")
        if "category_rank" in df_merged.columns:
            df_merged["category_rank"] = pd.to_numeric(
                df_merged["category_rank"], errors="coerce"
            ).fillna(len(category_order)).astype(int)
        else:
            category_rank = {cat: i for i, cat in enumerate(category_order)}
            df_merged["category_rank"] = df_merged["category"].map(category_rank).fillna(len(category_rank)).astype(int)
    else:
        allen_lookup = _get_allen_lookup()
        if allen_lookup is not None:
            order_by_acr = allen_lookup.get("order_by_acr", {})
            category_by_acr = allen_lookup.get("category_by_acr", {})
            region_map = {}
            region_orders = {}
            for reg in df_merged["region"].astype(str).unique():
                region_map[reg] = category_by_acr.get(reg, "Unknown")
                region_orders[reg] = int(order_by_acr.get(reg, 999999))
        else:
            region_map = {reg: "Unknown" for reg in df_merged["region"].astype(str).unique()}
            region_orders = {reg: 999999 for reg in df_merged["region"].astype(str).unique()}

        df_merged["category"] = df_merged["region"].map(region_map).fillna("Unknown")
        df_merged["allen_order"] = df_merged["region"].map(region_orders).fillna(999999)
        category_rank = {cat: i for i, cat in enumerate(category_order)}
        df_merged["category_rank"] = df_merged["category"].map(category_rank).fillna(len(category_rank)).astype(int)

    if (pd.to_numeric(df_merged["allen_order"], errors="coerce") != 999999).any():
        df_merged = df_merged[pd.to_numeric(df_merged["allen_order"], errors="coerce") != 999999]
    df_merged = df_merged[df_merged["region"] != "root"]
    df_merged = df_merged.sort_values(["category_rank", "allen_order", "region"])

if not df_merged.empty:
    fig_multi = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Coupling strength reliability",
            "Coupling delay reliability",
            "First Move Delay reliability",
            "Task Whisk Delay reliability",
            "Spont Whisk Delay reliability",
        ),
    )

    regions = df_merged["region"].tolist()
    x_vals = list(range(len(regions)))
    region_colors_dict = _build_region_colors(regions)
    marker_colors = [region_colors_dict.get(r, "whitesmoke") for r in regions]
    region_customdata = np.stack((df_merged["region"], df_merged["n_neurons_plot"].astype(int)), axis=-1)

    fig_multi.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_merged["strength_rel"],
            mode="markers",
            marker={"color": marker_colors, "size": 8},
            name="Strength Rel",
            showlegend=False,
            hovertemplate="Region: %{customdata[0]}<br>Strength Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
            customdata=region_customdata,
        ),
        row=1,
        col=1,
    )
    fig_multi.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_merged["delay_rel"],
            mode="markers",
            marker={"color": marker_colors, "size": 8},
            name="Delay Rel",
            showlegend=False,
            hovertemplate="Region: %{customdata[0]}<br>Delay Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
            customdata=region_customdata,
        ),
        row=2,
        col=1,
    )
    fig_multi.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_merged["first_move_rel"],
            mode="markers",
            marker={"color": marker_colors, "size": 8},
            name="First-Move Rel",
            showlegend=False,
            hovertemplate="Region: %{customdata[0]}<br>First-Move Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
            customdata=region_customdata,
        ),
        row=3,
        col=1,
    )
    fig_multi.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_merged["task_whisk_rel"],
            mode="markers",
            marker={"color": marker_colors, "size": 8},
            name="Task Whisk Rel",
            showlegend=False,
            hovertemplate="Region: %{customdata[0]}<br>Task Whisk Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
            customdata=region_customdata,
        ),
        row=4,
        col=1,
    )
    fig_multi.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_merged["spont_whisk_rel"],
            mode="markers",
            marker={"color": marker_colors, "size": 8},
            name="Spont Whisk Rel",
            showlegend=False,
            hovertemplate="Region: %{customdata[0]}<br>Spont Whisk Rel: %{y:.2f}<br>Neurons: %{customdata[1]}<extra></extra>",
            customdata=region_customdata,
        ),
        row=5,
        col=1,
    )

    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", row=5, col=1)

    df_merged_reset = df_merged.reset_index(drop=True)
    cat_positions = df_merged_reset.groupby("category", sort=False).apply(lambda x: x.index.to_numpy().mean())

    cat_to_acronym = {"Isocortex": "Isocortex", "HPF": "HPF", "Striatum": "STR", "Pallidum": "PAL",
                      "Thal.": "TH", "Hyp.": "HY", "Midbrain": "MB", "Pons": "P", "Medulla": "MY",
                      "Cereb.": "CB", "OLF": "OLF", "CTXsp": "CTXsp"}
    cat_acronyms_list = [cat_to_acronym.get(c, c) for c in cat_positions.index]
    cat_colors_dict = _build_region_colors(cat_acronyms_list)

    for cat, pos in cat_positions.items():
        if cat not in ["Unknown", "Other"]:
            acronym = cat_to_acronym.get(cat, cat)
            font_color = cat_colors_dict.get(acronym, "gray")
            if font_color == "rgb(255,255,255)":
                font_color = "gray"
            fig_multi.add_annotation(
                x=pos,
                y=-0.15,
                xref="x",
                yref="paper",
                text=cat,
                showarrow=False,
                font={"color": font_color, "size": 14},
                xanchor="center",
            )

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

    fig_multi.update_layout(height=1080, template=PLOTLY_TEMPLATE, margin=dict(b=180, t=50))
    _apply_cartesian_grid(fig_multi)
    fig_multi.update_yaxes(range=[-0.05, 1.0], title_text="Strength rel.", row=1, col=1)
    fig_multi.update_yaxes(range=[-0.05, 1.0], title_text="Delay rel.", row=2, col=1)
    fig_multi.update_yaxes(range=[-0.05, 1.0], title_text="First-move rel.", row=3, col=1)
    fig_multi.update_yaxes(range=[-0.05, 1.0], title_text="Task whisk rel.", row=4, col=1)
    fig_multi.update_yaxes(range=[-0.05, 1.0], title_text="Spont whisk rel.", row=5, col=1)
    fig_multi.update_xaxes(title_text="", tickmode="array", tickvals=x_vals, showticklabels=False)
    fig_multi.update_layout(xaxis=dict(tickmode="array", tickvals=x_vals, showticklabels=False))

    st.plotly_chart(fig_multi, use_container_width=True)

    df_rel = df_merged[["region", "delay_rel", "strength_rel", "n_neurons_plot"]].copy()
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
                    (df_rel["region"].astype(str), df_rel["n_neurons_plot"].astype(int)),
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
        _apply_cartesian_grid(fig_rel)
        fig_rel.update_xaxes(range=[-1.1, 1.1])
        fig_rel.update_yaxes(range=[-1.1, 1.1])
        st.plotly_chart(fig_rel, use_container_width=True)
    else:
        st.info("No finite delay/strength reliability pairs available for the added scatter plot.")
else:
    if seq_region_err:
        st.info(
            "No matching data found for the regional reliability plot. "
            f"Sequence-region source error: {seq_region_err}"
        )
    else:
        st.info("No matching data found for the regional reliability plot.")
