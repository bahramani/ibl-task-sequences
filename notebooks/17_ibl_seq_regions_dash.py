import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


st.set_page_config(page_title="Sequence Regions Dashboard", layout="wide")

BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "processed" / "16_ibl_seq_batch_comp"
FIXED_REL_CORR_AXIS_RANGE = [-0.2, 1.0]


def _show_table(df, width="stretch", max_rows=600):
    try:
        st.dataframe(df, width=width)
        return
    except Exception:
        pass

    if df is None or df.empty:
        st.info("No rows to display.")
        return

    df_show = df.head(max_rows).copy()
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} rows (of {len(df):,}).")
    st.markdown(df_show.to_html(index=False), unsafe_allow_html=True)


def _read_table_bundle(stem):
    pkl_path = CACHE_DIR / f"{stem}.pkl"
    parquet_path = CACHE_DIR / f"{stem}.parquet"
    csv_path = CACHE_DIR / f"{stem}.csv"
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


@st.cache_data(show_spinner="Loading batch sequence dashboard tables...")
def load_batch_outputs():
    meta_path_json = CACHE_DIR / "summary_metadata.json"
    meta_path_pkl = CACHE_DIR / "summary_metadata.pkl"
    if meta_path_json.exists():
        metadata = json.loads(meta_path_json.read_text(encoding="utf-8"))
    elif meta_path_pkl.exists():
        with open(meta_path_pkl, "rb") as f:
            metadata = pickle.load(f)
    else:
        raise FileNotFoundError("Missing summary_metadata.json / summary_metadata.pkl.")

    outputs = {
        "metadata": metadata,
        "summary_pids": _read_table_bundle("summary_pids"),
        "summary_pid_reason_counts": _read_table_bundle("summary_pid_reason_counts"),
        "summary_region_counts": _read_table_bundle("summary_region_counts"),
        "summary_pid_region_counts": _read_table_bundle("summary_pid_region_counts"),
        "summary_pid_pairs": _read_table_bundle("summary_pid_pairs"),
        "summary_region_pairs_pooled": _read_table_bundle("summary_region_pairs_pooled"),
        "summary_region_pairs_pidmean": _read_table_bundle("summary_region_pairs_pidmean"),
    }
    return outputs


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


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


def _build_var_maps(variable_specs):
    by_name = {spec["name"]: spec for spec in variable_specs}
    by_key = {spec["key"]: spec for spec in variable_specs}
    ordered_names = [spec["name"] for spec in sorted(variable_specs, key=lambda item: item["order"])]
    return by_name, by_key, ordered_names


def _normalize_pair(var_x_name, var_y_name, by_name):
    spec_x = by_name[var_x_name]
    spec_y = by_name[var_y_name]
    if int(spec_x["order"]) <= int(spec_y["order"]):
        return spec_x, spec_y
    return spec_y, spec_x


def _build_matrix_figure(
    df_region_pairs,
    variable_specs,
    method,
    aggregation_mode,
    template,
    region_label,
    support_text,
):
    ordered_specs = sorted(variable_specs, key=lambda item: item["order"])
    names = [spec["name"] for spec in ordered_specs]
    name_to_idx = {spec["name"]: idx for idx, spec in enumerate(ordered_specs)}
    n_vars = len(ordered_specs)
    z = np.full((n_vars, n_vars), np.nan, dtype=float)
    text = np.empty((n_vars, n_vars), dtype=object)
    text[:] = ""

    value_col = "pearson_r" if method == "pearson" else "spearman_rho"
    count_col = "pearson_n" if method == "pearson" else "spearman_n"
    pid_count_col = "pearson_pid_count" if method == "pearson" else "spearman_pid_count"

    for _, row in df_region_pairs.iterrows():
        x_name = str(row["var_x_name"])
        y_name = str(row["var_y_name"])
        i = name_to_idx[x_name]
        j = name_to_idx[y_name]
        val = row.get(value_col, np.nan)
        if aggregation_mode == "pid_first_mean":
            count_val = row.get(pid_count_col, row.get("n_pids", np.nan))
        else:
            count_val = row.get(count_col, row.get("n_shared", np.nan))
        is_diag = bool(row.get("is_diagonal", False))
        prefix = "rel=" if is_diag else ("r=" if method == "pearson" else "rho=")
        text_str = (
            f"{prefix}{float(val):.3f}<br>n={int(count_val)}"
            if pd.notnull(val) and np.isfinite(float(val)) and pd.notnull(count_val)
            else f"{prefix}nan<br>n={int(count_val) if pd.notnull(count_val) else 0}"
        )
        z[i, j] = float(val) if pd.notnull(val) else np.nan
        z[j, i] = z[i, j]
        text[i, j] = text_str
        text[j, i] = text_str

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}",
            hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
        )
    )
    mode_text = "Pooled neurons" if aggregation_mode == "pooled_neurons" else "PID first"
    fig.update_layout(
        title=f"{method.title()} Matrix | {region_label} | {mode_text}<br><sup>{support_text}</sup>",
        template=template,
        margin=dict(l=90, r=40, t=95, b=95),
        height=600,
    )
    fig.update_xaxes(tickangle=40)
    return fig


def _build_region_scatter(df_plot, method, aggregation_mode, template, title):
    if df_plot.empty:
        return None

    corr_col = "pearson_r" if method == "pearson" else "spearman_rho"
    rel_col = "pearson_total_reliability" if method == "pearson" else "spearman_total_reliability"
    count_col = "pearson_pid_count" if aggregation_mode == "pid_first_mean" and method == "pearson" else None
    if aggregation_mode == "pid_first_mean" and method == "spearman":
        count_col = "spearman_pid_count"
    if aggregation_mode == "pooled_neurons":
        count_col = "pearson_n" if method == "pearson" else "spearman_n"

    df_plot = df_plot.copy()
    df_plot["plot_color"] = df_plot["allen_color"].fillna("rgba(180,180,180,0.95)")
    df_plot = df_plot[
        pd.to_numeric(df_plot[corr_col], errors="coerce").notna()
        & pd.to_numeric(df_plot[rel_col], errors="coerce").notna()
    ].copy()
    if df_plot.empty:
        return None

    custom_cols = [
        "region",
        "category",
        "n_units_total",
        "n_pids",
        count_col,
    ]
    for col in custom_cols:
        if col not in df_plot.columns:
            df_plot[col] = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[rel_col].to_numpy(dtype=float),
            y=df_plot[corr_col].to_numpy(dtype=float),
            mode="markers",
            marker=dict(
                size=10,
                color=df_plot["plot_color"].tolist(),
                line=dict(color="rgba(0,0,0,0.45)", width=0.8),
            ),
            showlegend=False,
            customdata=np.stack(
                [
                    df_plot["region"].astype(str).to_numpy(),
                    df_plot["category"].astype(str).to_numpy(),
                    pd.to_numeric(df_plot["n_units_total"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot["n_pids"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot[count_col], errors="coerce").fillna(0).to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Region: %{customdata[0]}<br>"
                "Category: %{customdata[1]}<br>"
                "Total rel=%{x:.3f}<br>"
                "Corr=%{y:.3f}<br>"
                "Units=%{customdata[2]:.0f}<br>"
                "PIDs=%{customdata[3]:.0f}<br>"
                "Count=%{customdata[4]:.0f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.update_yaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.add_shape(
        type="line",
        x0=FIXED_REL_CORR_AXIS_RANGE[0],
        y0=FIXED_REL_CORR_AXIS_RANGE[0],
        x1=FIXED_REL_CORR_AXIS_RANGE[1],
        y1=FIXED_REL_CORR_AXIS_RANGE[1],
        line=dict(color="red", dash="dash"),
    )

    mean_corr = _safe_mean(df_plot[corr_col])
    mean_rel = _safe_mean(df_plot[rel_col])
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bordercolor="rgba(120,120,120,0.55)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.78)" if template == "plotly_white" else "rgba(30,30,30,0.82)",
        text=(
            f"Mean corr = {mean_corr:.3f}<br>"
            f"Mean total rel = {mean_rel:.3f}<br>"
            f"Regions = {len(df_plot):,}"
        ),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Total reliability",
        yaxis_title="Correlation",
        template=template,
        margin=dict(l=80, r=30, t=85, b=70),
        height=580,
    )
    return fig


def _build_pid_detail_scatter(df_pid_points, avg_row, method, region, template, marker_color=None):
    corr_col = "pearson_r" if method == "pearson" else "spearman_rho"
    rel_col = "pearson_total_reliability" if method == "pearson" else "spearman_total_reliability"
    n_col = "pearson_n" if method == "pearson" else "spearman_n"

    df_plot = df_pid_points.copy()
    df_plot = df_plot[
        pd.to_numeric(df_plot[corr_col], errors="coerce").notna()
        & pd.to_numeric(df_plot[rel_col], errors="coerce").notna()
    ].copy()
    if df_plot.empty:
        return None

    color = marker_color if pd.notnull(marker_color) else "#1f77b4"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[rel_col].to_numpy(dtype=float),
            y=df_plot[corr_col].to_numpy(dtype=float),
            mode="markers",
            name="PIDs",
            marker=dict(size=10, color=color, opacity=0.72, line=dict(color="rgba(0,0,0,0.4)", width=0.8)),
            customdata=np.stack(
                [
                    df_plot["pid"].astype(str).to_numpy(),
                    pd.to_numeric(df_plot[n_col], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot["n_units_region"], errors="coerce").fillna(0).to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "PID: %{customdata[0]}<br>"
                "Total rel=%{x:.3f}<br>"
                "Corr=%{y:.3f}<br>"
                "Count=%{customdata[1]:.0f}<br>"
                "Units=%{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )

    if avg_row is not None:
        avg_rel = avg_row.get(rel_col, np.nan)
        avg_corr = avg_row.get(corr_col, np.nan)
        if pd.notnull(avg_rel) and pd.notnull(avg_corr):
            fig.add_trace(
                go.Scatter(
                    x=[float(avg_rel)],
                    y=[float(avg_corr)],
                    mode="markers",
                    name="Region average",
                    marker=dict(size=18, color=color, line=dict(color="black", width=2)),
                    hovertemplate=(
                        f"Region average<br>Region: {region}<br>"
                        "Total rel=%{x:.3f}<br>"
                        "Corr=%{y:.3f}<extra></extra>"
                    ),
                )
            )

    fig.update_xaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.update_yaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.add_shape(
        type="line",
        x0=FIXED_REL_CORR_AXIS_RANGE[0],
        y0=FIXED_REL_CORR_AXIS_RANGE[0],
        x1=FIXED_REL_CORR_AXIS_RANGE[1],
        y1=FIXED_REL_CORR_AXIS_RANGE[1],
        line=dict(color="red", dash="dash"),
    )

    fig.update_layout(
        title=f"{method.title()} Per-PID Detail | Region {region}",
        xaxis_title="Total reliability",
        yaxis_title="Correlation",
        template=template,
        margin=dict(l=80, r=30, t=80, b=70),
        height=550,
    )
    return fig


def _build_pid_reliability_scatter(
    df_rel_points,
    avg_x,
    avg_y,
    method,
    region,
    template,
    marker_color=None,
):
    value_col = "pearson_rel" if method == "pearson" else "spearman_rel"
    n_col = "pearson_n" if method == "pearson" else "spearman_n"

    df_plot = df_rel_points.copy()
    df_plot = df_plot[
        pd.to_numeric(df_plot[f"{value_col}_x"], errors="coerce").notna()
        & pd.to_numeric(df_plot[f"{value_col}_y"], errors="coerce").notna()
    ].copy()
    if df_plot.empty:
        return None

    color = marker_color if pd.notnull(marker_color) else "#1f77b4"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[f"{value_col}_x"].to_numpy(dtype=float),
            y=df_plot[f"{value_col}_y"].to_numpy(dtype=float),
            mode="markers",
            name="PIDs",
            marker=dict(size=10, color=color, opacity=0.72, line=dict(color="rgba(0,0,0,0.4)", width=0.8)),
            customdata=np.stack(
                [
                    df_plot["pid"].astype(str).to_numpy(),
                    pd.to_numeric(df_plot[f"{n_col}_x"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot[f"{n_col}_y"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot["n_units_region_x"], errors="coerce").fillna(0).to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "PID: %{customdata[0]}<br>"
                "Whisk rel=%{x:.3f}<br>"
                "Non-whisk rel=%{y:.3f}<br>"
                "Whisk count=%{customdata[1]:.0f}<br>"
                "Non-whisk count=%{customdata[2]:.0f}<br>"
                "Units=%{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )

    if pd.notnull(avg_x) and pd.notnull(avg_y):
        fig.add_trace(
            go.Scatter(
                x=[float(avg_x)],
                y=[float(avg_y)],
                mode="markers",
                name="Region average",
                marker=dict(size=18, color=color, line=dict(color="black", width=2)),
                hovertemplate=(
                    f"Region average<br>Region: {region}<br>"
                    "Whisk rel=%{x:.3f}<br>"
                    "Non-whisk rel=%{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.update_yaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.add_shape(
        type="line",
        x0=FIXED_REL_CORR_AXIS_RANGE[0],
        y0=FIXED_REL_CORR_AXIS_RANGE[0],
        x1=FIXED_REL_CORR_AXIS_RANGE[1],
        y1=FIXED_REL_CORR_AXIS_RANGE[1],
        line=dict(color="red", dash="dash"),
    )
    fig.update_layout(
        title=f"{method.title()} Reliability | Spont Whisk vs Spont Non-Whisk | Region {region}",
        xaxis_title="Spont Whisk Coupling Delay Reliability",
        yaxis_title="Spont Non-Whisk Coupling Delay Reliability",
        template=template,
        margin=dict(l=80, r=30, t=80, b=70),
        height=550,
    )
    return fig


def _build_region_reliability_comparison_scatter(
    df_plot,
    method,
    aggregation_mode,
    template,
    title,
):
    if df_plot.empty:
        return None

    x_col = "pearson_rel_x" if method == "pearson" else "spearman_rel_x"
    y_col = "pearson_rel_y" if method == "pearson" else "spearman_rel_y"
    x_count_col = (
        "pearson_pid_count_x"
        if aggregation_mode == "pid_first_mean" and method == "pearson"
        else "spearman_pid_count_x"
        if aggregation_mode == "pid_first_mean"
        else "pearson_n_x"
        if method == "pearson"
        else "spearman_n_x"
    )
    y_count_col = (
        "pearson_pid_count_y"
        if aggregation_mode == "pid_first_mean" and method == "pearson"
        else "spearman_pid_count_y"
        if aggregation_mode == "pid_first_mean"
        else "pearson_n_y"
        if method == "pearson"
        else "spearman_n_y"
    )

    df_plot = df_plot.copy()
    df_plot = df_plot[
        pd.to_numeric(df_plot[x_col], errors="coerce").notna()
        & pd.to_numeric(df_plot[y_col], errors="coerce").notna()
    ].copy()
    if df_plot.empty:
        return None

    df_plot["plot_color"] = df_plot["allen_color"].fillna("rgba(180,180,180,0.95)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[x_col].to_numpy(dtype=float),
            y=df_plot[y_col].to_numpy(dtype=float),
            mode="markers",
            marker=dict(
                size=10,
                color=df_plot["plot_color"].tolist(),
                line=dict(color="rgba(0,0,0,0.45)", width=0.8),
            ),
            showlegend=False,
            customdata=np.stack(
                [
                    df_plot["region"].astype(str).to_numpy(),
                    df_plot["category"].astype(str).to_numpy(),
                    pd.to_numeric(df_plot["n_units_total"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot["n_pids"], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot[x_count_col], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(df_plot[y_count_col], errors="coerce").fillna(0).to_numpy(dtype=float),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Region: %{customdata[0]}<br>"
                "Category: %{customdata[1]}<br>"
                "Whisk rel=%{x:.3f}<br>"
                "Non-whisk rel=%{y:.3f}<br>"
                "Units=%{customdata[2]:.0f}<br>"
                "PIDs=%{customdata[3]:.0f}<br>"
                "Whisk count=%{customdata[4]:.0f}<br>"
                "Non-whisk count=%{customdata[5]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.update_yaxes(range=FIXED_REL_CORR_AXIS_RANGE)
    fig.add_shape(
        type="line",
        x0=FIXED_REL_CORR_AXIS_RANGE[0],
        y0=FIXED_REL_CORR_AXIS_RANGE[0],
        x1=FIXED_REL_CORR_AXIS_RANGE[1],
        y1=FIXED_REL_CORR_AXIS_RANGE[1],
        line=dict(color="red", dash="dash"),
    )
    mean_x = _safe_mean(df_plot[x_col])
    mean_y = _safe_mean(df_plot[y_col])
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bordercolor="rgba(120,120,120,0.55)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.78)" if template == "plotly_white" else "rgba(30,30,30,0.82)",
        text=(
            f"Mean whisk rel = {mean_x:.3f}<br>"
            f"Mean non-whisk rel = {mean_y:.3f}<br>"
            f"Regions = {len(df_plot):,}"
        ),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Spont Whisk Coupling Delay Reliability",
        yaxis_title="Spont Non-Whisk Coupling Delay Reliability",
        template=template,
        margin=dict(l=80, r=30, t=85, b=70),
        height=580,
    )
    return fig


try:
    data = load_batch_outputs()
except Exception as exc:
    st.title("Sequence Regions Dashboard")
    st.error(
        "Could not load precomputed batch tables from "
        f"`{CACHE_DIR}`. Run `16_ibl_seq_batch_comp.py` first.\n\nError: {exc}"
    )
    st.stop()

metadata = data["metadata"]
summary_pids = data["summary_pids"].copy()
summary_pid_reason_counts = data["summary_pid_reason_counts"].copy()
summary_region_counts = data["summary_region_counts"].copy()
summary_pid_pairs = data["summary_pid_pairs"].copy()
summary_region_pairs_pooled = data["summary_region_pairs_pooled"].copy()
summary_region_pairs_pidmean = data["summary_region_pairs_pidmean"].copy()

variable_specs = metadata.get("variable_specs", [])
category_order = metadata.get("category_order", [])
if not variable_specs:
    st.error("`summary_metadata` is missing `variable_specs`.")
    st.stop()

var_by_name, var_by_key, ordered_var_names = _build_var_maps(variable_specs)
canonical_categories = [cat for cat in category_order if cat not in ["Unknown", "Other"]]
if not canonical_categories:
    canonical_categories = ["Isocortex", "HPF", "OLF", "CTXsp", "Striatum", "Pallidum", "Thal.", "Hyp.", "Midbrain", "Pons", "Medulla", "Cereb."]

control_col_a, control_col_b = st.columns([1.6, 1.0])
with control_col_a:
    aggregation_label = st.radio(
        "Aggregation mode",
        ["Pooled neurons", "PID first"],
        index=0,
        horizontal=True,
    )
with control_col_b:
    plotly_dark_mode = st.toggle("Plotly dark mode", value=False)

aggregation_mode = "pooled_neurons" if aggregation_label == "Pooled neurons" else "pid_first_mean"
PLOTLY_TEMPLATE = "plotly_dark" if plotly_dark_mode else "plotly_white"
pio.templates.default = PLOTLY_TEMPLATE

st.title("Sequence Regions Dashboard")
st.caption(
    f"Source 15 config: calc_version={metadata.get('source_calc_version', 'NA')} | "
    f"CALC_LABEL_MIN={metadata.get('source_calc_label_min', 'NA')} | "
    f"MIN_REGION_NEURONS={metadata.get('source_min_region_neurons', 'NA')} | "
    f"tag={metadata.get('all_pids_tag', 'NA')}"
)

metric_cols = st.columns(4)
metric_cols[0].metric("Target PIDs", int(metadata.get("n_target_pids", len(summary_pids))))
metric_cols[1].metric("Matching Cached PIDs", int(metadata.get("n_ok_pids", 0)))
metric_cols[2].metric("Not Available", int(metadata.get("n_not_available_pids", 0)))
metric_cols[3].metric("Config Mismatch", int(metadata.get("n_config_mismatch_pids", 0)))

st.subheader("PID Status Summary")
reason_col, detail_col = st.columns([0.9, 1.7])
with reason_col:
    st.markdown("**Reason Counts**")
    _show_table(summary_pid_reason_counts, max_rows=200)
with detail_col:
    st.markdown("**Detailed PID Table**")
    _show_table(summary_pids, max_rows=900)

region_pair_table = (
    summary_region_pairs_pooled if aggregation_mode == "pooled_neurons" else summary_region_pairs_pidmean
).copy()
if region_pair_table.empty or summary_region_counts.empty:
    st.info("No region-level batch data are available.")
    st.stop()

region_counts_sorted = summary_region_counts.sort_values(["category_rank", "allen_order", "region"]).reset_index(drop=True)
region_labels = [
    f"{row['region']} | units={int(row['n_units_total'])} | pids={int(row['n_pids'])}"
    for _, row in region_counts_sorted.iterrows()
]
label_to_region = dict(zip(region_labels, region_counts_sorted["region"].astype(str)))

st.subheader("Region Matrices")
selected_region_label = st.selectbox("Select region", region_labels, index=0)
selected_region = label_to_region[selected_region_label]
selected_region_row = region_counts_sorted.loc[region_counts_sorted["region"].astype(str) == str(selected_region)].iloc[0]
selected_region_pairs = region_pair_table.loc[
    region_pair_table["region"].astype(str) == str(selected_region)
].copy()
support_text = (
    f"Units total={int(selected_region_row['n_units_total'])} | "
    f"PIDs={int(selected_region_row['n_pids'])} | "
    f"Mean units/PID={float(selected_region_row['n_units_mean_per_pid']):.1f}"
)
st.caption(
    f"Selected region: {selected_region} | category={selected_region_row.get('category', 'NA')} | {support_text}"
)

fig_mat_p = _build_matrix_figure(
    selected_region_pairs,
    variable_specs,
    method="pearson",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    region_label=str(selected_region),
    support_text=support_text,
)
fig_mat_s = _build_matrix_figure(
    selected_region_pairs,
    variable_specs,
    method="spearman",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    region_label=str(selected_region),
    support_text=support_text,
)
col_mat_a, col_mat_b = st.columns(2)
with col_mat_a:
    st.plotly_chart(fig_mat_p, width="stretch")
with col_mat_b:
    st.plotly_chart(fig_mat_s, width="stretch")

st.subheader("Correlation vs Total Reliability")
scatter_cols = st.columns(2)
with scatter_cols[0]:
    var_x_name = st.selectbox("Variable X", ordered_var_names, index=0)
with scatter_cols[1]:
    default_idx = 1 if len(ordered_var_names) > 1 else 0
    var_y_name = st.selectbox("Variable Y", ordered_var_names, index=default_idx)

st.markdown("**Broad anatomy categories**")
cat_cols = st.columns(6)
selected_categories = []
for idx, cat in enumerate(canonical_categories):
    checked = cat_cols[idx % 6].checkbox(cat, value=True, key=f"cat_{cat}")
    if checked:
        selected_categories.append(cat)

if not selected_categories:
    st.warning("Select at least one broad anatomy category to plot region points.")
    st.stop()

spec_x, spec_y = _normalize_pair(var_x_name, var_y_name, var_by_name)
df_pair_selected = region_pair_table.loc[
    (region_pair_table["var_x_key"].astype(str) == str(spec_x["key"]))
    & (region_pair_table["var_y_key"].astype(str) == str(spec_y["key"]))
].copy()
df_pair_selected = df_pair_selected.drop(
    columns=["n_units_total", "n_units_mean_per_pid", "n_units_median_per_pid", "n_pids_region"],
    errors="ignore",
)
df_pair_selected = df_pair_selected.merge(
    summary_region_counts[
        [
            "region",
            "n_pids",
            "n_units_total",
            "n_units_mean_per_pid",
            "n_units_median_per_pid",
            "allen_color",
            "category",
            "category_rank",
            "allen_order",
        ]
    ],
    on="region",
    how="left",
    suffixes=("", "_region"),
)
if "category" in df_pair_selected.columns:
    df_pair_selected = df_pair_selected[df_pair_selected["category"].isin(selected_categories)].copy()

scatter_title_p = (
    f"Pearson corr vs total reliability | {var_x_name} vs {var_y_name} | {aggregation_label}"
)
scatter_title_s = (
    f"Spearman corr vs total reliability | {var_x_name} vs {var_y_name} | {aggregation_label}"
)
fig_scatter_p = _build_region_scatter(
    df_pair_selected,
    method="pearson",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    title=scatter_title_p,
)
fig_scatter_s = _build_region_scatter(
    df_pair_selected,
    method="spearman",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    title=scatter_title_s,
)
col_sc_a, col_sc_b = st.columns(2)
with col_sc_a:
    if fig_scatter_p is None:
        st.info("No Pearson points are available for the current filters.")
    else:
        st.plotly_chart(fig_scatter_p, width="stretch")
with col_sc_b:
    if fig_scatter_s is None:
        st.info("No Spearman points are available for the current filters.")
    else:
        st.plotly_chart(fig_scatter_s, width="stretch")

if aggregation_mode == "pid_first_mean":
    st.subheader("PID-First Detail")
    df_pid_selected = summary_pid_pairs.loc[
        (summary_pid_pairs["region"].astype(str) == str(selected_region))
        & (summary_pid_pairs["var_x_key"].astype(str) == str(spec_x["key"]))
        & (summary_pid_pairs["var_y_key"].astype(str) == str(spec_y["key"]))
    ].copy()
    df_avg_selected = summary_region_pairs_pidmean.loc[
        (summary_region_pairs_pidmean["region"].astype(str) == str(selected_region))
        & (summary_region_pairs_pidmean["var_x_key"].astype(str) == str(spec_x["key"]))
        & (summary_region_pairs_pidmean["var_y_key"].astype(str) == str(spec_y["key"]))
    ].copy()
    avg_row = None if df_avg_selected.empty else df_avg_selected.iloc[0].to_dict()
    region_color = selected_region_row.get("allen_color")
    fig_pid_p = _build_pid_detail_scatter(
        df_pid_selected,
        avg_row,
        method="pearson",
        region=str(selected_region),
        template=PLOTLY_TEMPLATE,
        marker_color=region_color,
    )
    fig_pid_s = _build_pid_detail_scatter(
        df_pid_selected,
        avg_row,
        method="spearman",
        region=str(selected_region),
        template=PLOTLY_TEMPLATE,
        marker_color=region_color,
    )
    pid_col_a, pid_col_b = st.columns(2)
    with pid_col_a:
        if fig_pid_p is None:
            st.info("No per-PID Pearson detail is available for this region/pair.")
        else:
            st.plotly_chart(fig_pid_p, width="stretch")
    with pid_col_b:
        if fig_pid_s is None:
            st.info("No per-PID Spearman detail is available for this region/pair.")
        else:
            st.plotly_chart(fig_pid_s, width="stretch")

st.subheader("Spontaneous Coupling Reliability Comparison")
whisk_key = "spont_whisk_coupling"
nonwhisk_key = "spont_nonwhisk_coupling"
df_rel_x = region_pair_table.loc[
    (region_pair_table["is_diagonal"] == True)
    & (region_pair_table["var_x_key"].astype(str) == whisk_key)
    & (region_pair_table["var_y_key"].astype(str) == whisk_key)
].copy()
df_rel_y = region_pair_table.loc[
    (region_pair_table["is_diagonal"] == True)
    & (region_pair_table["var_x_key"].astype(str) == nonwhisk_key)
    & (region_pair_table["var_y_key"].astype(str) == nonwhisk_key)
].copy()

rel_merge_x_cols = [
    "region",
    "pearson_r",
    "spearman_rho",
]
rel_merge_y_cols = [
    "region",
    "pearson_r",
    "spearman_rho",
]
if aggregation_mode == "pid_first_mean":
    rel_merge_x_cols += ["pearson_pid_count", "spearman_pid_count", "pearson_n_mean", "spearman_n_mean"]
    rel_merge_y_cols += ["pearson_pid_count", "spearman_pid_count", "pearson_n_mean", "spearman_n_mean"]
else:
    rel_merge_x_cols += ["pearson_n", "spearman_n"]
    rel_merge_y_cols += ["pearson_n", "spearman_n"]

df_rel_compare = df_rel_x[rel_merge_x_cols].rename(
        columns={
            "pearson_r": "pearson_rel_x",
            "spearman_rho": "spearman_rel_x",
            "pearson_n": "pearson_n_x",
            "spearman_n": "spearman_n_x",
            "pearson_n_mean": "pearson_n_x",
            "spearman_n_mean": "spearman_n_x",
            "pearson_pid_count": "pearson_pid_count_x",
            "spearman_pid_count": "spearman_pid_count_x",
        }
    )
df_rel_compare = df_rel_compare.merge(
    df_rel_y[rel_merge_y_cols].rename(
        columns={
            "pearson_r": "pearson_rel_y",
            "spearman_rho": "spearman_rel_y",
            "pearson_n": "pearson_n_y",
            "spearman_n": "spearman_n_y",
            "pearson_n_mean": "pearson_n_y",
            "spearman_n_mean": "spearman_n_y",
            "pearson_pid_count": "pearson_pid_count_y",
            "spearman_pid_count": "spearman_pid_count_y",
        }
    ),
    on="region",
    how="inner",
)
df_rel_compare = df_rel_compare.merge(
    summary_region_counts[
        [
            "region",
            "n_pids",
            "n_units_total",
            "allen_color",
            "category",
            "category_rank",
            "allen_order",
        ]
    ],
    on="region",
    how="left",
)
if "category" in df_rel_compare.columns:
    df_rel_compare = df_rel_compare[df_rel_compare["category"].isin(selected_categories)].copy()

rel_title_p = (
    f"Pearson reliability | Spont Whisk Coupling vs Spont Non-Whisk Coupling | {aggregation_label}"
)
rel_title_s = (
    f"Spearman reliability | Spont Whisk Coupling vs Spont Non-Whisk Coupling | {aggregation_label}"
)
fig_rel_p = _build_region_reliability_comparison_scatter(
    df_rel_compare,
    method="pearson",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    title=rel_title_p,
)
fig_rel_s = _build_region_reliability_comparison_scatter(
    df_rel_compare,
    method="spearman",
    aggregation_mode=aggregation_mode,
    template=PLOTLY_TEMPLATE,
    title=rel_title_s,
)
rel_col_a, rel_col_b = st.columns(2)
with rel_col_a:
    if fig_rel_p is None:
        st.info("No Pearson reliability-comparison points are available for the current filters.")
    else:
        st.plotly_chart(fig_rel_p, width="stretch")
with rel_col_b:
    if fig_rel_s is None:
        st.info("No Spearman reliability-comparison points are available for the current filters.")
    else:
        st.plotly_chart(fig_rel_s, width="stretch")
