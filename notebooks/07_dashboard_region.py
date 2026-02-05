# %%
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None


st.set_page_config(page_title="Region Dashboard", layout="wide")

BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
DEFAULT_LABEL_MIN = 0.5


CORR_VARIABLES = [
    {
        "name": "Delay (Stim On)",
        "df": "df_res",
        "v1": "delay_stimOn_times_odd",
        "v2": "delay_stimOn_times_even",
    },
    {
        "name": "Delay (First Move)",
        "df": "df_res",
        "v1": "delay_firstMovement_times_odd",
        "v2": "delay_firstMovement_times_even",
    },
    {
        "name": "Delay (Response)",
        "df": "df_res",
        "v1": "delay_response_times_odd",
        "v2": "delay_response_times_even",
    },
    {
        "name": "Delay (Feedback)",
        "df": "df_res",
        "v1": "delay_feedback_times_odd",
        "v2": "delay_feedback_times_even",
    },
    {
        "name": "stPR Delay (Spont)",
        "df": "df_coupling",
        "v1": "coupling_delay_ms_h1",
        "v2": "coupling_delay_ms_h2",
    },
    {
        "name": "stPR Delay (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "name": "stPR Delay (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_delay_ms_odd",
        "v2": "coupling_delay_ms_even",
    },
    {
        "name": "stPR Strength (Spont)",
        "df": "df_coupling",
        "v1": "coupling_strength_h1",
        "v2": "coupling_strength_h2",
    },
    {
        "name": "stPR Strength (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "name": "stPR Strength (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_strength_odd",
        "v2": "coupling_strength_even",
    },
    {
        "name": "stPR Max (Spont)",
        "df": "df_coupling",
        "v1": "coupling_max_h1",
        "v2": "coupling_max_h2",
    },
    {
        "name": "stPR Max (Task)",
        "df": "df_coupling_task",
        "v1": "coupling_max_odd",
        "v2": "coupling_max_even",
    },
    {
        "name": "stPR Max (ITI)",
        "df": "df_coupling_iti",
        "v1": "coupling_max_odd",
        "v2": "coupling_max_even",
    },
]


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


def _build_region_colors(regions):
    if BrainRegions is None:
        return {}
    br = BrainRegions()
    colors = {}
    for region in regions:
        try:
            idx = br.acronym2index(region)[1][0][0]
            rgb = br.rgb[idx]
            colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        except Exception:
            continue
    return colors


def _build_variable_table(df, spec, region_lookup):
    if df is None or df.empty:
        return None
    if spec["v1"] not in df.columns or spec["v2"] not in df.columns:
        return None
    df_var = df[["pid", "cluster_id", spec["v1"], spec["v2"]]].copy()
    df_var = df_var.groupby(["pid", "cluster_id"], as_index=False).mean(
        numeric_only=True
    )
    df_var = df_var.merge(region_lookup, on=["pid", "cluster_id"], how="inner")
    v1 = df_var[spec["v1"]].to_numpy(dtype=float)
    v2 = df_var[spec["v2"]].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(v1) & np.isfinite(v2)
    mean_vals[valid] = (v1[valid] + v2[valid]) / 2.0
    df_var["mean"] = mean_vals
    return df_var


def _build_label_lookup(df_res):
    if df_res is None or df_res.empty:
        return None
    if "label" not in df_res.columns:
        return None
    df_labels = df_res[["pid", "cluster_id", "label"]].copy()
    df_labels["cluster_id"] = pd.to_numeric(df_labels["cluster_id"], errors="coerce")
    df_labels = df_labels[np.isfinite(df_labels["cluster_id"])].copy()
    if df_labels.empty:
        return None
    df_labels["cluster_id"] = df_labels["cluster_id"].astype(int)
    df_labels["label"] = pd.to_numeric(df_labels["label"], errors="coerce")
    df_labels = df_labels[np.isfinite(df_labels["label"])].copy()
    if df_labels.empty:
        return None
    df_labels = df_labels.groupby(["pid", "cluster_id"], as_index=False)["label"].max()
    return df_labels


@st.cache_data(show_spinner=False)
def _load_cache_tables(cache_dir):
    cache_paths = sorted(Path(cache_dir).glob("*.pkl"))

    rows = []
    pid_summary_rows = []
    label_mins = []
    data_tables = {
        "df_res": [],
        "df_coupling": [],
        "df_coupling_task": [],
        "df_coupling_iti": [],
    }

    for path in cache_paths:
        with open(path, "rb") as f:
            cache = pickle.load(f)

        pid = cache.get("pid", path.stem)
        config_calc = cache.get("config_calc") or {}
        label_min_raw = config_calc.get("CALC_LABEL_MIN", DEFAULT_LABEL_MIN)
        try:
            label_min = float(label_min_raw)
        except (TypeError, ValueError):
            label_min = float(DEFAULT_LABEL_MIN)
        label_mins.append(label_min)

        df_res = cache.get("df_res")
        if df_res is None or len(df_res) == 0:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
            continue

        if "label" in df_res.columns:
            labels = pd.to_numeric(df_res["label"], errors="coerce")
            df_units = df_res[labels >= label_min].copy()
        else:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
            continue

        if "acronym" in df_units.columns:
            region_col = "acronym"
        elif "region" in df_units.columns:
            region_col = "region"
        else:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
            continue

        df_units = df_units[["cluster_id", region_col]].copy()
        df_units["cluster_id"] = pd.to_numeric(df_units["cluster_id"], errors="coerce")
        df_units = df_units[np.isfinite(df_units["cluster_id"])].copy()
        if df_units.empty:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
            continue
        df_units["cluster_id"] = df_units["cluster_id"].astype(int)
        df_units["region"] = df_units[region_col].astype(str)
        df_units = df_units[~df_units["region"].isin(["root", "void", "NA", "nan"])]
        df_units["pid"] = pid

        rows.append(df_units[["pid", "cluster_id", "region"]])
        pid_summary_rows.append(
            {"pid": pid, "n_neurons": int(len(df_units)), "label_min": label_min}
        )

        df_res_copy = df_res.copy()
        df_res_copy["pid"] = pid
        df_res_copy["cluster_id"] = pd.to_numeric(
            df_res_copy["cluster_id"], errors="coerce"
        )
        df_res_copy = df_res_copy[np.isfinite(df_res_copy["cluster_id"])].copy()
        if not df_res_copy.empty:
            df_res_copy["cluster_id"] = df_res_copy["cluster_id"].astype(int)
            data_tables["df_res"].append(df_res_copy)

        for key in ("df_coupling", "df_coupling_task", "df_coupling_iti"):
            df_tbl = cache.get(key)
            if df_tbl is None or df_tbl.empty:
                continue
            df_tbl = df_tbl.copy()
            df_tbl["pid"] = pid
            df_tbl["cluster_id"] = pd.to_numeric(df_tbl["cluster_id"], errors="coerce")
            df_tbl = df_tbl[np.isfinite(df_tbl["cluster_id"])].copy()
            if df_tbl.empty:
                continue
            df_tbl["cluster_id"] = df_tbl["cluster_id"].astype(int)
            data_tables[key].append(df_tbl)

    if rows:
        neurons_df = pd.concat(rows, ignore_index=True)
    else:
        neurons_df = pd.DataFrame(columns=["pid", "cluster_id", "region"])

    pid_summary = (
        pd.DataFrame(pid_summary_rows)
        .sort_values("pid", ascending=True)
        .reset_index(drop=True)
    )

    region_counts = (
        neurons_df["region"]
        .value_counts()
        .rename_axis("region")
        .reset_index(name="n_neurons")
        .sort_values("region")
        .reset_index(drop=True)
    )
    region_pid_counts = (
        neurons_df.groupby("region")["pid"].nunique().rename("n_pids").reset_index()
    )
    region_counts = region_counts.merge(region_pid_counts, on="region", how="left")

    label_mins_clean = [val for val in label_mins if val is not None and np.isfinite(val)]
    label_min_values = sorted(set(np.round(label_mins_clean, 6)))
    if len(label_min_values) == 1:
        label_min_text = f"{label_min_values[0]:.2f}"
    elif label_min_values:
        label_min_text = ", ".join([f"{val:.2f}" for val in label_min_values])
    else:
        label_min_text = "NA"

    data_concat = {}
    for key, tables in data_tables.items():
        data_concat[key] = pd.concat(tables, ignore_index=True) if tables else None

    return neurons_df, pid_summary, region_counts, label_min_text, data_concat


plotly_dark_mode = st.toggle("Plotly dark mode", value=False)
PLOTLY_TEMPLATE = "plotly_dark" if plotly_dark_mode else "plotly_white"
pio.templates.default = PLOTLY_TEMPLATE

neurons_df, pid_summary, region_counts, label_min_text, data_concat = _load_cache_tables(
    CACHE_DIR
)

st.title("Region Dashboard")
st.caption(f"Label threshold(s) detected: {label_min_text}")

st.subheader("PID Summary")
st.dataframe(pid_summary, width="stretch")

st.subheader("Region Summary")
st.dataframe(region_counts, width="stretch")

good_only_toggle = st.toggle(
    "Use only good neurons (label=1) for correlation/reliability",
    value=False,
)

label_lookup = _build_label_lookup(data_concat.get("df_res"))
if good_only_toggle and label_lookup is None:
    st.warning("Label data not available; using all neurons.")
    good_only_toggle = False

if good_only_toggle:
    good_mask = np.isclose(label_lookup["label"].to_numpy(dtype=float), 1.0)
    label_good = label_lookup.loc[good_mask, ["pid", "cluster_id"]]
    neurons_df_calc = neurons_df.merge(
        label_good, on=["pid", "cluster_id"], how="inner"
    )
else:
    neurons_df_calc = neurons_df

region_counts_calc = (
    neurons_df_calc["region"]
    .value_counts()
    .rename_axis("region")
    .reset_index(name="n_neurons")
    .sort_values("region")
    .reset_index(drop=True)
)
region_pid_counts_calc = (
    neurons_df_calc.groupby("region")["pid"].nunique().rename("n_pids").reset_index()
)
region_counts_calc = region_counts_calc.merge(
    region_pid_counts_calc, on="region", how="left"
)

label_filter_text = "label=1 only" if good_only_toggle else f"label>= {label_min_text}"

st.subheader("Correlation Matrices by Region")
if region_counts_calc.empty:
    st.info("No regions available.")
else:
    region_counts_sorted = region_counts_calc.sort_values("region").reset_index(drop=True)
    region_labels = [
        f"{row['region']} (n={row['n_neurons']})"
        for _, row in region_counts_sorted.iterrows()
    ]
    label_to_region = dict(zip(region_labels, region_counts_sorted["region"]))
    selected_label = st.selectbox("Select region", region_labels)
    selected_region = label_to_region.get(selected_label)

    if selected_region is None:
        st.info("No region selected.")
    else:
        region_lookup = neurons_df_calc.loc[
            neurons_df_calc["region"] == selected_region, ["pid", "cluster_id", "region"]
        ]
        n_total_units = int(len(region_lookup))

        available_specs = []
        for spec in CORR_VARIABLES:
            df_src = data_concat.get(spec["df"])
            if df_src is None:
                continue
            if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
                continue
            available_specs.append(spec)

        var_tables_all = {}
        for spec in available_specs:
            df_var = _build_variable_table(data_concat.get(spec["df"]), spec, region_lookup)
            if df_var is None or df_var.empty:
                continue
            var_tables_all[spec["name"]] = df_var

        names = [spec["name"] for spec in available_specs if spec["name"] in var_tables_all]
        n_vars = len(names)

        if n_vars == 0:
            st.info("No variables available for correlation matrices.")
        else:
            reliability = {}
            reliability_n = {}
            for spec in available_specs:
                name = spec["name"]
                df_var = var_tables_all.get(name)
                if df_var is None:
                    reliability[name] = np.nan
                    reliability_n[name] = 0
                    continue
                r_val, n_val = _pearsonr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                reliability[name] = r_val
                reliability_n[name] = n_val

            mean_wide = region_lookup[["pid", "cluster_id"]].drop_duplicates()
            for spec in available_specs:
                name = spec["name"]
                df_var = var_tables_all.get(name)
                if df_var is None:
                    mean_wide[name] = np.nan
                    continue
                mean_wide = mean_wide.merge(
                    df_var[["pid", "cluster_id", "mean"]],
                    on=["pid", "cluster_id"],
                    how="left",
                ).rename(columns={"mean": name})

            corr_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
            text_mat = np.empty((n_vars, n_vars), dtype=object)
            for i, name_i in enumerate(names):
                for j, name_j in enumerate(names):
                    if i == j:
                        r_val = reliability.get(name_i, np.nan)
                        n_val = reliability_n.get(name_i, 0)
                        corr_mat[i, j] = r_val
                        text_mat[i, j] = (
                            f"rel={r_val:.2f}<br>(n={n_val})"
                            if np.isfinite(r_val)
                            else f"rel=nan<br>(n={n_val})"
                        )
                    else:
                        r_val, n_val = _pearsonr_with_n(
                            mean_wide[name_i], mean_wide[name_j]
                        )
                        corr_mat[i, j] = r_val
                        text_mat[i, j] = (
                            f"r={r_val:.2f}<br>(n={n_val})"
                            if np.isfinite(r_val)
                            else f"r=nan<br>(n={n_val})"
                        )

            fig_pearson = go.Figure(
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
            fig_pearson.update_layout(
                title=(
                    "Reliability (diag) + Pairwise Pearson (off-diag) | "
                    f"Region {selected_region} | N total ({label_filter_text}): {n_total_units}"
                ),
                height=min(1000, max(500, 40 * n_vars + 200)),
                margin=dict(l=90, r=30, t=90, b=90),
                template=PLOTLY_TEMPLATE,
            )
            fig_pearson.update_xaxes(tickangle=45)

            spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
            spearman_text = np.empty((n_vars, n_vars), dtype=object)
            for i, name_i in enumerate(names):
                for j, name_j in enumerate(names):
                    if i == j:
                        spec = next((s for s in available_specs if s["name"] == name_i), None)
                        if spec is None or name_i not in var_tables_all:
                            r_val, n_val = np.nan, 0
                        else:
                            r_val, n_val = _spearmanr_with_n(
                                var_tables_all[name_i][spec["v1"]],
                                var_tables_all[name_i][spec["v2"]],
                            )
                        spearman_mat[i, j] = r_val
                        spearman_text[i, j] = (
                            f"rel={r_val:.2f}<br>(n={n_val})"
                            if np.isfinite(r_val)
                            else f"rel=nan<br>(n={n_val})"
                        )
                    else:
                        r_val, n_val = _spearmanr_with_n(
                            mean_wide[name_i], mean_wide[name_j]
                        )
                        spearman_mat[i, j] = r_val
                        spearman_text[i, j] = (
                            f"rho={r_val:.2f}<br>(n={n_val})"
                            if np.isfinite(r_val)
                            else f"rho=nan<br>(n={n_val})"
                        )

            fig_spearman = go.Figure(
                data=go.Heatmap(
                    z=spearman_mat,
                    x=names,
                    y=names,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    reversescale=True,
                    text=spearman_text,
                    texttemplate="%{text}",
                    hovertemplate="X=%{x}<br>Y=%{y}<br>%{text}<extra></extra>",
                )
            )
            fig_spearman.update_layout(
                title=(
                    "Reliability (diag) + Pairwise Spearman (off-diag) | "
                    f"Region {selected_region} | N total ({label_filter_text}): {n_total_units}"
                ),
                height=min(1000, max(500, 40 * n_vars + 200)),
                margin=dict(l=90, r=30, t=90, b=90),
                template=PLOTLY_TEMPLATE,
            )
            fig_spearman.update_xaxes(tickangle=45)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Pearson**")
                st.plotly_chart(fig_pearson, width="stretch")
            with col_b:
                st.markdown("**Spearman**")
                st.plotly_chart(fig_spearman, width="stretch")

st.subheader("Correlation vs Reliability by Region")

available_specs = []
for spec in CORR_VARIABLES:
    df_src = data_concat.get(spec["df"])
    if df_src is None:
        continue
    if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
        continue
    available_specs.append(spec)

var_names = [spec["name"] for spec in available_specs]
if not var_names:
    st.info("No variables available for correlation/reliability plot.")
else:
    min_neurons = int(
        st.number_input(
            "Min neurons per region",
            min_value=0,
            value=100,
            step=10,
        )
    )
    var_x = st.selectbox("Variable 1", var_names, index=0)
    default_idx = 1 if len(var_names) > 1 else 0
    var_y = st.selectbox("Variable 2", var_names, index=default_idx)

    spec_by_name = {spec["name"]: spec for spec in available_specs}
    spec_x = spec_by_name[var_x]
    spec_y = spec_by_name[var_y]

    region_counts_all = (
        neurons_df_calc.groupby("region")["cluster_id"]
        .nunique()
        .rename("n_neurons")
        .reset_index()
    )
    eligible_regions = region_counts_all.loc[
        region_counts_all["n_neurons"] >= min_neurons, "region"
    ].tolist()
    region_colors = _build_region_colors(eligible_regions)

    def _build_scatter(method_name):
        records = []
        for region in sorted(eligible_regions):
            region_ids = neurons_df_calc.loc[
                neurons_df_calc["region"] == region, ["pid", "cluster_id", "region"]
            ]
            if region_ids.empty:
                continue

            df_var_x = _build_variable_table(data_concat.get(spec_x["df"]), spec_x, region_ids)
            df_var_y = _build_variable_table(data_concat.get(spec_y["df"]), spec_y, region_ids)
            if df_var_x is None or df_var_y is None:
                continue

            if method_name == "spearman":
                rel_x, n_rel_x = _spearmanr_with_n(df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]])
                rel_y, n_rel_y = _spearmanr_with_n(df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]])
            else:
                rel_x, n_rel_x = _pearsonr_with_n(df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]])
                rel_y, n_rel_y = _pearsonr_with_n(df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]])

            rel_prod = rel_x * rel_y if np.isfinite(rel_x) and np.isfinite(rel_y) else np.nan
            if rel_prod is not None and np.isfinite(rel_prod) and rel_prod >= 0:
                rel_total = float(np.sqrt(rel_prod))
            else:
                rel_total = np.nan

            merged = df_var_x[["pid", "cluster_id", "mean"]].merge(
                df_var_y[["pid", "cluster_id", "mean"]],
                on=["pid", "cluster_id"],
                how="inner",
                suffixes=("_x", "_y"),
            )
            if merged.empty:
                continue

            if method_name == "spearman":
                corr_val, n_corr = _spearmanr_with_n(merged["mean_x"], merged["mean_y"])
            else:
                corr_val, n_corr = _pearsonr_with_n(merged["mean_x"], merged["mean_y"])

            n_neurons = int(
                region_counts_all.loc[
                    region_counts_all["region"] == region, "n_neurons"
                ].values[0]
            )

            records.append(
                {
                    "region": region,
                    "corr": corr_val,
                    "reliability": rel_total,
                    "n_neurons": n_neurons,
                    "n_corr": n_corr,
                    "n_rel_x": n_rel_x,
                    "n_rel_y": n_rel_y,
                }
            )

        df_plot = pd.DataFrame(records)
        df_plot = df_plot[np.isfinite(df_plot["corr"]) & np.isfinite(df_plot["reliability"])]
        if df_plot.empty:
            return None

        fig = go.Figure()
        for _, row in df_plot.sort_values("region").iterrows():
            region = row["region"]
            color = region_colors.get(region)
            fig.add_trace(
                go.Scatter(
                    x=[row["reliability"]],
                    y=[row["corr"]],
                    mode="markers",
                    name=region,
                    marker=dict(size=10, color=color) if color else dict(size=10),
                    hovertemplate=(
                        f"Region: {region}<br>"
                        f"corr={row['corr']:.2f}<br>"
                        f"rel={row['reliability']:.2f}<br>"
                        f"n={row['n_neurons']}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=(
                f"{method_name.title()} correlation vs total reliability | "
                f"{var_x} vs {var_y} | "
                f"regions with >= {min_neurons} neurons ({label_filter_text})"
            ),
            xaxis_title="Total reliability (sqrt(rel1 * rel2))",
            yaxis_title="Correlation between variables",
            template=PLOTLY_TEMPLATE,
            legend=dict(x=1.02, y=1, yanchor="top"),
            margin=dict(l=80, r=200, t=90, b=70),
            height=600,
            width=900,
        )
        return fig

    fig_p = _build_scatter("pearson")
    fig_s = _build_scatter("spearman")

    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("**Pearson**")
        if fig_p is None:
            st.info("No data for Pearson plot.")
        else:
            st.plotly_chart(fig_p, width="stretch")
    with col_s:
        st.markdown("**Spearman**")
        if fig_s is None:
            st.info("No data for Spearman plot.")
        else:
            st.plotly_chart(fig_s, width="stretch")
