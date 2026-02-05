# %% Imports
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None

USE_DARK_PLOTLY = False
PLOTLY_TEMPLATE = "plotly_dark" if USE_DARK_PLOTLY else "plotly_white"

# %% Load cached PIDs and collect neuron/region info
BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
DEFAULT_LABEL_MIN = 0.5

cache_paths = sorted(CACHE_DIR.glob("*.pkl"))
print(f"Cache directory: {CACHE_DIR}")
print(f"Found {len(cache_paths)} cached PIDs")

rows = []
pid_summary_rows = []
label_mins = []
missing_df_res = []
missing_label = []
missing_region = []

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
        missing_df_res.append(pid)
        pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
        continue

    df = df_res.copy()
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        df = df[labels >= label_min]
    else:
        missing_label.append(pid)
        pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
        continue

    if "acronym" in df.columns:
        region_col = "acronym"
    elif "region" in df.columns:
        region_col = "region"
    else:
        missing_region.append(pid)
        pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min})
        continue

    df = df[[region_col]].copy()
    df["region"] = df[region_col].astype(str)
    df = df[~df["region"].isin(["root", "void", "NA", "nan"])]
    df["pid"] = pid

    rows.append(df[["pid", "region"]])
    pid_summary_rows.append({"pid": pid, "n_neurons": int(len(df)), "label_min": label_min})

if rows:
    neurons_df = pd.concat(rows, ignore_index=True)
else:
    neurons_df = pd.DataFrame(columns=["pid", "region"])

pid_summary = (
    pd.DataFrame(pid_summary_rows)
    .sort_values("pid", ascending=True)
    .reset_index(drop=True)
)

label_mins_clean = [val for val in label_mins if val is not None and np.isfinite(val)]
label_min_values = sorted(set(np.round(label_mins_clean, 6)))
if len(label_min_values) == 1:
    label_min_text = f"{label_min_values[0]:.2f}"
elif label_min_values:
    label_min_text = ", ".join([f"{val:.2f}" for val in label_min_values])
else:
    label_min_text = "NA"

print(f"Label threshold(s) found: {label_min_text}")
print(f"Total neurons (label >= threshold): {len(neurons_df)}")
print(f"Unique regions: {neurons_df['region'].nunique()}")
if missing_df_res:
    print(f"Warning: {len(missing_df_res)} PIDs missing df_res (skipped)")
if missing_label:
    print(f"Warning: {len(missing_label)} PIDs missing label column (skipped)")
if missing_region:
    print(f"Warning: {len(missing_region)} PIDs missing region/acronym column (skipped)")

# %% PID summary (optional)
pid_summary

# %% Region table (neurons per region)
region_counts = (
    neurons_df["region"]
    .value_counts()
    .rename_axis("region")
    .reset_index(name="n_neurons")
    .sort_values("n_neurons", ascending=False)
    .reset_index(drop=True)
)
region_pid_counts = (
    neurons_df.groupby("region")["pid"].nunique().rename("n_pids").reset_index()
)
region_counts = region_counts.merge(region_pid_counts, on="region", how="left")
region_counts

# %% PIDs that contain a region (label-filtered neurons)
REGION_QUERY = "VISp"  # set region acronym here

region_pid_table = (
    neurons_df.loc[neurons_df["region"] == REGION_QUERY]
    .groupby("pid")["region"]
    .size()
    .rename("n_neurons")
    .reset_index()
    .sort_values("pid")
    .reset_index(drop=True)
)

print(
    f"Region {REGION_QUERY}: {len(region_pid_table)} PIDs | "
    f"label >= {label_min_text}"
)
region_pid_table

# %% Correlation matrices for a region (all PIDs, Plotly)
REGION_CORR = "MOs"  # set region acronym here

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


def _format_corr_value(val):
    if np.isfinite(val):
        return f"{val:.2f}"
    return "nan"


data_tables = {
    "df_res": [],
    "df_coupling": [],
    "df_coupling_task": [],
    "df_coupling_iti": [],
}
region_lookup_rows = []

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

    df_res = cache.get("df_res")
    if df_res is not None and not df_res.empty and "label" in df_res.columns:
        labels = pd.to_numeric(df_res["label"], errors="coerce")
        df_units = df_res[labels >= label_min].copy()
        if "acronym" in df_units.columns:
            region_col = "acronym"
        elif "region" in df_units.columns:
            region_col = "region"
        else:
            region_col = None

        if region_col is not None:
            df_units = df_units[["cluster_id", region_col]].copy()
            df_units["cluster_id"] = pd.to_numeric(
                df_units["cluster_id"], errors="coerce"
            )
            df_units = df_units[np.isfinite(df_units["cluster_id"])].copy()
            if not df_units.empty:
                df_units["cluster_id"] = df_units["cluster_id"].astype(int)
                df_units["region"] = df_units[region_col].astype(str)
                df_units = df_units[
                    ~df_units["region"].isin(["root", "void", "NA", "nan"])
                ]
                df_units["pid"] = pid
                region_lookup_rows.append(df_units[["pid", "cluster_id", "region"]])

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

region_lookup = (
    pd.concat(region_lookup_rows, ignore_index=True)
    if region_lookup_rows
    else pd.DataFrame(columns=["pid", "cluster_id", "region"])
)

if REGION_CORR is not None:
    region_lookup = region_lookup[region_lookup["region"] == REGION_CORR]

n_total_units = int(len(region_lookup))
if n_total_units == 0:
    print(f"No neurons found for region {REGION_CORR}.")
else:
    available_specs = []
    data_concat = {}
    for key, tables in data_tables.items():
        if tables:
            data_concat[key] = pd.concat(tables, ignore_index=True)
        else:
            data_concat[key] = None

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
        print(f"No correlation variables available for region {REGION_CORR}.")
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
                f"Region {REGION_CORR} | N total (label>= {label_min_text}): {n_total_units}"
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
                f"Region {REGION_CORR} | N total (label>= {label_min_text}): {n_total_units}"
            ),
            height=min(1000, max(500, 40 * n_vars + 200)),
            margin=dict(l=90, r=30, t=90, b=90),
            template=PLOTLY_TEMPLATE,
        )
        fig_spearman.update_xaxes(tickangle=45)

        fig_pearson.show()
        fig_spearman.show()

# %% Correlation vs reliability by region (Plotly)
MIN_NEURONS_FOR_REGION = 50
CORR_METHOD = "pearson"  # "spearman" or "pearson"

if "_pearsonr_with_n" not in globals():
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

if "_spearmanr_with_n" not in globals():
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

if "_build_variable_table" not in globals():
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


def _corr_with_n(method, x, y, min_n=2):
    if method.lower().startswith("s"):
        return _spearmanr_with_n(x, y, min_n=min_n)
    return _pearsonr_with_n(x, y, min_n=min_n)


def _build_region_colors(regions):
    try:
        from iblatlas.regions import BrainRegions as _BrainRegions
    except Exception:
        _BrainRegions = None
    if _BrainRegions is None:
        return {}
    br = _BrainRegions()
    colors = {}
    for region in regions:
        try:
            idx = br.acronym2index(region)[1][0][0]
            rgb = br.rgb[idx]
            colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        except Exception:
            continue
    return colors


data_tables = {
    "df_res": [],
    "df_coupling": [],
    "df_coupling_task": [],
    "df_coupling_iti": [],
}
region_lookup_rows = []

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

    df_res = cache.get("df_res")
    if df_res is not None and not df_res.empty and "label" in df_res.columns:
        labels = pd.to_numeric(df_res["label"], errors="coerce")
        df_units = df_res[labels >= label_min].copy()
        if "acronym" in df_units.columns:
            region_col = "acronym"
        elif "region" in df_units.columns:
            region_col = "region"
        else:
            region_col = None

        if region_col is not None:
            df_units = df_units[["cluster_id", region_col]].copy()
            df_units["cluster_id"] = pd.to_numeric(
                df_units["cluster_id"], errors="coerce"
            )
            df_units = df_units[np.isfinite(df_units["cluster_id"])].copy()
            if not df_units.empty:
                df_units["cluster_id"] = df_units["cluster_id"].astype(int)
                df_units["region"] = df_units[region_col].astype(str)
                df_units = df_units[
                    ~df_units["region"].isin(["root", "void", "NA", "nan"])
                ]
                df_units["pid"] = pid
                region_lookup_rows.append(df_units[["pid", "cluster_id", "region"]])

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

region_lookup = (
    pd.concat(region_lookup_rows, ignore_index=True)
    if region_lookup_rows
    else pd.DataFrame(columns=["pid", "cluster_id", "region"])
)

data_concat = {}
for key, tables in data_tables.items():
    data_concat[key] = pd.concat(tables, ignore_index=True) if tables else None

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
    print("No variables available for correlation/reliability plot.")
else:
    print("Available variables:")
    for idx, name in enumerate(var_names):
        print(f"  [{idx}] {name}")

    def _choose_var(prompt, names):
        choice = input(f"{prompt} (index or name): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(names):
                return names[idx]
        for name in names:
            if choice.lower() == name.lower():
                return name
        return names[0]

    var_x = _choose_var("Select first variable", var_names)
    var_y = _choose_var("Select second variable", var_names)

    spec_by_name = {spec["name"]: spec for spec in available_specs}
    spec_x = spec_by_name[var_x]
    spec_y = spec_by_name[var_y]

    region_counts_all = (
        region_lookup.groupby("region")["cluster_id"]
        .nunique()
        .rename("n_neurons")
        .reset_index()
    )
    eligible_regions = region_counts_all.loc[
        region_counts_all["n_neurons"] >= MIN_NEURONS_FOR_REGION, "region"
    ].tolist()

    region_colors = _build_region_colors(eligible_regions)

    records = []
    for region in sorted(eligible_regions):
        region_ids = region_lookup.loc[
            region_lookup["region"] == region, ["pid", "cluster_id"]
        ]
        if region_ids.empty:
            continue

        df_var_x = _build_variable_table(data_concat.get(spec_x["df"]), spec_x, region_ids)
        df_var_y = _build_variable_table(data_concat.get(spec_y["df"]), spec_y, region_ids)
        if df_var_x is None or df_var_y is None:
            continue

        rel_x, n_rel_x = _corr_with_n(CORR_METHOD, df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]])
        rel_y, n_rel_y = _corr_with_n(CORR_METHOD, df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]])
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
        corr_val, n_corr = _corr_with_n(CORR_METHOD, merged["mean_x"], merged["mean_y"])

        n_neurons = int(region_counts_all.loc[
            region_counts_all["region"] == region, "n_neurons"
        ].values[0])

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
        print("No regions with valid correlation/reliability values.")
    else:
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
                f"{CORR_METHOD.title()} correlation vs total reliability | "
                f"{var_x} vs {var_y} | "
                f"regions with >= {MIN_NEURONS_FOR_REGION} neurons"
            ),
            xaxis_title="Total reliability (sqrt(rel1 * rel2))",
            yaxis_title="Correlation between variables",
            template=PLOTLY_TEMPLATE,
            legend=dict(x=1.02, y=1, yanchor="top"),
            margin=dict(l=80, r=200, t=90, b=70),
            height=600,
            width=1000,
        )
        fig.show()

# %% Histogram of neurons per region
if not region_counts.empty:
    threshold_label = label_min_text if len(label_min_values) == 1 else "varies"
    plt.figure(figsize=(9, 5))
    plt.hist(region_counts["n_neurons"], bins="auto", color="#4C72B0", edgecolor="white")
    plt.xlabel(f"Neurons per region (label >= {threshold_label})")
    plt.ylabel("Number of regions")
    plt.title("Distribution of neuron counts across regions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No region counts available to plot.")
