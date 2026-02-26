# %% Imports
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None

# %% Config
USE_DARK_PLOTLY = False
PLOTLY_TEMPLATE = "plotly_dark" if USE_DARK_PLOTLY else "plotly_white"
PLOTLY_RENDERER = None  # e.g. "browser", "notebook", "png"

BASE_PATH = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
DEFAULT_LABEL_MIN = 0.9

REGION_QUERY = "VISp"  # set to region name to skip prompt
REGION_CORR = "VISp"  # set to region name to skip prompt

GOOD_ONLY_FOR_CORR = True
USE_GOOD_STPR = False
AVG_BY_PID = False

MIN_NEURONS_FOR_REGION = 100
SCATTER_VAR_X = None  # set to variable name to skip prompt
SCATTER_VAR_Y = None
USE_INTERACTIVE_PROMPTS = False


def _set_plotly_renderer():
    if PLOTLY_RENDERER:
        pio.renderers.default = PLOTLY_RENDERER
        return
    try:
        import nbformat  # noqa: F401
    except Exception:
        pio.renderers.default = "browser"


_set_plotly_renderer()

# %% Correlation variable definitions
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
    {
        "name": "Firing rate",
        "df": "df_firing_rate",
        "v1": "firing_rate_h1",
        "v2": "firing_rate_h2",
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


def _is_firing_rate_spec(spec):
    return spec.get("df") == "df_firing_rate"


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
    except (TypeError, ValueError):
        return False
    if start is None or end is None:
        return False
    try:
        start_val = float(start)
        end_val = float(end)
    except (TypeError, ValueError):
        return False
    return np.isfinite(start_val) and np.isfinite(end_val) and end_val > start_val


def _filter_region_lookup_for_spec(region_lookup, spec, spont_pids):
    if region_lookup is None or region_lookup.empty:
        return region_lookup
    if not _is_spont_spec(spec):
        return region_lookup
    if not spont_pids:
        return region_lookup.iloc[0:0]
    if "pid" not in region_lookup.columns:
        return region_lookup
    return region_lookup[region_lookup["pid"].isin(spont_pids)]


def _mean_with_count(values):
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.nan, 0
    return float(np.nanmean(vals[finite])), int(np.sum(finite))

def _load_cache_tables(cache_dir):
    cache_paths = sorted(Path(cache_dir).glob("*.pkl"))

    rows = []
    pid_summary_rows = []
    label_mins = []
    spont_pids = set()
    data_tables = {
        "df_res": [],
        "df_coupling": [],
        "df_coupling_task": [],
        "df_coupling_iti": [],
        "df_coupling_good": [],
        "df_coupling_task_good": [],
        "df_coupling_iti_good": [],
        "df_firing_rate": [],
    }

    for path in cache_paths:
        with open(path, "rb") as f:
            cache = pickle.load(f)

        pid = cache.get("pid", path.stem)
        meta = cache.get("meta") or {}
        has_spont = _has_spont_interval(meta)
        if has_spont:
            spont_pids.add(pid)
        config_calc = cache.get("config_calc") or {}
        label_min_raw = config_calc.get("CALC_LABEL_MIN", DEFAULT_LABEL_MIN)
        try:
            label_min = float(label_min_raw)
        except (TypeError, ValueError):
            label_min = float(DEFAULT_LABEL_MIN)
        label_mins.append(label_min)

        df_res = cache.get("df_res")
        if df_res is None or len(df_res) == 0:
            pid_summary_rows.append(
                {
                    "pid": pid,
                    "n_neurons": 0,
                    "label_min": label_min,
                    "has_spont_interval": has_spont,
                }
            )
            continue

        if "label" in df_res.columns:
            labels = pd.to_numeric(df_res["label"], errors="coerce")
            df_units = df_res[labels >= label_min].copy()
        else:
            pid_summary_rows.append(
                {
                    "pid": pid,
                    "n_neurons": 0,
                    "label_min": label_min,
                    "has_spont_interval": has_spont,
                }
            )
            continue

        if "acronym" in df_units.columns:
            region_col = "acronym"
        elif "region" in df_units.columns:
            region_col = "region"
        else:
            pid_summary_rows.append(
                {
                    "pid": pid,
                    "n_neurons": 0,
                    "label_min": label_min,
                    "has_spont_interval": has_spont,
                }
            )
            continue

        df_units = df_units[["cluster_id", region_col]].copy()
        df_units["cluster_id"] = pd.to_numeric(df_units["cluster_id"], errors="coerce")
        df_units = df_units[np.isfinite(df_units["cluster_id"])].copy()
        if df_units.empty:
            pid_summary_rows.append(
                {
                    "pid": pid,
                    "n_neurons": 0,
                    "label_min": label_min,
                    "has_spont_interval": has_spont,
                }
            )
            continue
        df_units["cluster_id"] = df_units["cluster_id"].astype(int)
        df_units["region"] = df_units[region_col].astype(str)
        df_units = df_units[~df_units["region"].isin(["root", "void", "NA", "nan"])]
        df_units["pid"] = pid

        rows.append(df_units[["pid", "cluster_id", "region"]])
        pid_summary_rows.append(
            {
                "pid": pid,
                "n_neurons": int(len(df_units)),
                "label_min": label_min,
                "has_spont_interval": has_spont,
            }
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

        for key in (
            "df_coupling",
            "df_coupling_task",
            "df_coupling_iti",
            "df_coupling_good",
            "df_coupling_task_good",
            "df_coupling_iti_good",
        ):
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

        cluster_ids = cache.get("cluster_ids")
        cluster_firing_rate = cache.get("cluster_firing_rate")
        if cluster_ids is not None and cluster_firing_rate is not None:
            cluster_ids = pd.to_numeric(np.asarray(cluster_ids), errors="coerce")
            cluster_firing_rate = np.asarray(cluster_firing_rate, dtype=float)
            if len(cluster_ids) == len(cluster_firing_rate):
                df_rate = pd.DataFrame(
                    {
                        "pid": pid,
                        "cluster_id": cluster_ids,
                        "firing_rate_h1": cluster_firing_rate,
                        "firing_rate_h2": cluster_firing_rate,
                    }
                )
                df_rate = df_rate[np.isfinite(df_rate["cluster_id"])].copy()
                if not df_rate.empty:
                    df_rate["cluster_id"] = df_rate["cluster_id"].astype(int)
                    data_tables["df_firing_rate"].append(df_rate)

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

    return (
        cache_paths,
        neurons_df,
        pid_summary,
        region_counts,
        label_min_text,
        label_min_values,
        data_concat,
        spont_pids,
    )

(
    cache_paths,
    neurons_df,
    pid_summary,
    region_counts,
    label_min_text,
    label_min_values,
    data_concat,
    spont_pids,
) = _load_cache_tables(CACHE_DIR)
spont_pids = set(spont_pids) if spont_pids is not None else set()

print(f"Cache directory: {CACHE_DIR}")
print(f"Found {len(cache_paths)} cached PIDs")
print(f"Label threshold(s) found: {label_min_text}")
print(f"Total neurons (label >= threshold): {len(neurons_df)}")
print(f"Unique regions: {neurons_df['region'].nunique()}")

label_lookup = _build_label_lookup(data_concat.get("df_res"))
if GOOD_ONLY_FOR_CORR and label_lookup is None:
    print("Warning: Label data not available; using all neurons.")
    GOOD_ONLY_FOR_CORR = False

if GOOD_ONLY_FOR_CORR:
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

label_filter_text = "label=1 only" if GOOD_ONLY_FOR_CORR else f"label>= {label_min_text}"

pid_summary_calc = pid_summary
if isinstance(pid_summary, pd.DataFrame):
    pid_counts = (
        neurons_df_calc.groupby("pid")["cluster_id"]
        .nunique()
        .rename("n_neurons")
        .reset_index()
    )
    pid_summary_calc = pid_summary.drop(columns=["n_neurons"], errors="ignore").merge(
        pid_counts, on="pid", how="left"
    )
    pid_summary_calc["n_neurons"] = (
        pid_summary_calc["n_neurons"].fillna(0).astype(int)
    )

# %% PID summary (optional)
pid_summary_calc

# %% Region table (neurons per region)
region_counts_calc

# %% PIDs that contain a region (analysis-filtered neurons)
region_pid_table = (
    neurons_df_calc.loc[neurons_df_calc["region"] == REGION_QUERY]
    .groupby("pid")["region"]
    .size()
    .rename("n_neurons")
    .reset_index()
    .sort_values("pid")
    .reset_index(drop=True)
)

print(
    f"Region {REGION_QUERY}: {len(region_pid_table)} PIDs | "
    f"{label_filter_text}"
)
region_pid_table

data_for_corr = dict(data_concat)
if USE_GOOD_STPR:
    missing = []
    if data_concat.get("df_coupling_good") is None:
        missing.append("Spont")
    if data_concat.get("df_coupling_task_good") is None:
        missing.append("Task")
    if data_concat.get("df_coupling_iti_good") is None:
        missing.append("ITI")
    if len(missing) == 3:
        print(
            "Warning: Good-neuron stPR not available in cache; using all neurons for stPR metrics."
        )
        USE_GOOD_STPR = False
    elif missing:
        print(
            "Warning: Good-neuron stPR missing for: "
            + ", ".join(missing)
            + ". Using all neurons for those contexts."
        )

if USE_GOOD_STPR:
    data_for_corr["df_coupling"] = data_concat.get("df_coupling_good")
    data_for_corr["df_coupling_task"] = data_concat.get("df_coupling_task_good")
    data_for_corr["df_coupling_iti"] = data_concat.get("df_coupling_iti_good")

# %% Correlation matrices for a region (Plotly)
if REGION_CORR is None:
    print("REGION_CORR is None; skipping correlation matrices.")
else:
    region_lookup = neurons_df_calc.loc[
        neurons_df_calc["region"] == REGION_CORR, ["pid", "cluster_id", "region"]
    ]
    n_total_units = int(len(region_lookup))
    if n_total_units == 0:
        print(f"No neurons found for region {REGION_CORR}.")
    else:
        available_specs = []
        for spec in CORR_VARIABLES:
            df_src = data_for_corr.get(spec["df"])
            if df_src is None:
                continue
            if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
                continue
            available_specs.append(spec)

        var_tables_all = {}
        for spec in available_specs:
            region_lookup_spec = _filter_region_lookup_for_spec(
                region_lookup, spec, spont_pids
            )
            df_var = _build_variable_table(
                data_for_corr.get(spec["df"]), spec, region_lookup_spec
            )
            if df_var is None or df_var.empty:
                continue
            var_tables_all[spec["name"]] = df_var

        names = [spec["name"] for spec in available_specs if spec["name"] in var_tables_all]
        n_vars = len(names)

        if n_vars == 0:
            print(f"No correlation variables available for region {REGION_CORR}.")
        else:
            n_pids = int(region_lookup["pid"].nunique())
            if AVG_BY_PID:
                pid_list = sorted(region_lookup["pid"].unique().tolist())
                rel_vals = {name: [] for name in names}
                rel_s_vals = {name: [] for name in names}
                corr_vals = {(a, b): [] for a in names for b in names if a != b}
                corr_s_vals = {(a, b): [] for a in names for b in names if a != b}

                for pid in pid_list:
                    region_pid = region_lookup.loc[region_lookup["pid"] == pid]
                    if region_pid.empty:
                        continue

                    var_tables_pid = {}
                    for spec in available_specs:
                        name = spec["name"]
                        if name not in names:
                            continue
                        region_pid_spec = _filter_region_lookup_for_spec(
                            region_pid, spec, spont_pids
                        )
                        df_var = _build_variable_table(
                            data_for_corr.get(spec["df"]), spec, region_pid_spec
                        )
                        if df_var is None or df_var.empty:
                            continue
                        var_tables_pid[name] = df_var

                    if not var_tables_pid:
                        continue

                    for spec in available_specs:
                        name = spec["name"]
                        if name not in names:
                            continue
                        if _is_firing_rate_spec(spec):
                            rel_vals[name].append(np.nan)
                            rel_s_vals[name].append(np.nan)
                            continue
                        df_var = var_tables_pid.get(name)
                        if df_var is None:
                            rel_vals[name].append(np.nan)
                            rel_s_vals[name].append(np.nan)
                            continue
                        r_val, _ = _pearsonr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                        r_s, _ = _spearmanr_with_n(
                            df_var[spec["v1"]], df_var[spec["v2"]]
                        )
                        rel_vals[name].append(r_val)
                        rel_s_vals[name].append(r_s)

                    mean_wide = region_pid[["pid", "cluster_id"]].drop_duplicates()
                    for spec in available_specs:
                        name = spec["name"]
                        df_var = var_tables_pid.get(name)
                        if df_var is None:
                            mean_wide[name] = np.nan
                            continue
                        mean_wide = mean_wide.merge(
                            df_var[["pid", "cluster_id", "mean"]],
                            on=["pid", "cluster_id"],
                            how="left",
                        ).rename(columns={"mean": name})

                    for name_i in names:
                        for name_j in names:
                            if name_i == name_j:
                                continue
                            r_val, _ = _pearsonr_with_n(mean_wide[name_i], mean_wide[name_j])
                            r_s, _ = _spearmanr_with_n(mean_wide[name_i], mean_wide[name_j])
                            corr_vals[(name_i, name_j)].append(r_val)
                            corr_s_vals[(name_i, name_j)].append(r_s)

                corr_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
                text_mat = np.empty((n_vars, n_vars), dtype=object)
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        if i == j:
                            r_val, n_val = _mean_with_count(rel_vals.get(name_i, []))
                            corr_mat[i, j] = r_val
                            text_mat[i, j] = (
                                f"rel={r_val:.2f}<br>(pids={n_val})"
                                if np.isfinite(r_val)
                                else f"rel=nan<br>(pids={n_val})"
                            )
                        else:
                            r_val, n_val = _mean_with_count(
                                corr_vals.get((name_i, name_j), [])
                            )
                            corr_mat[i, j] = r_val
                            text_mat[i, j] = (
                                f"r={r_val:.2f}<br>(pids={n_val})"
                                if np.isfinite(r_val)
                                else f"r=nan<br>(pids={n_val})"
                            )

                spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
                spearman_text = np.empty((n_vars, n_vars), dtype=object)
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        if i == j:
                            r_val, n_val = _mean_with_count(rel_s_vals.get(name_i, []))
                            spearman_mat[i, j] = r_val
                            spearman_text[i, j] = (
                                f"rel={r_val:.2f}<br>(pids={n_val})"
                                if np.isfinite(r_val)
                                else f"rel=nan<br>(pids={n_val})"
                            )
                        else:
                            r_val, n_val = _mean_with_count(
                                corr_s_vals.get((name_i, name_j), [])
                            )
                            spearman_mat[i, j] = r_val
                            spearman_text[i, j] = (
                                f"rho={r_val:.2f}<br>(pids={n_val})"
                                if np.isfinite(r_val)
                                else f"rho=nan<br>(pids={n_val})"
                            )
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
                    if _is_firing_rate_spec(spec):
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

                spearman_mat = np.full((n_vars, n_vars), np.nan, dtype=float)
                spearman_text = np.empty((n_vars, n_vars), dtype=object)
                for i, name_i in enumerate(names):
                    for j, name_j in enumerate(names):
                        if i == j:
                            spec = next(
                                (s for s in available_specs if s["name"] == name_i), None
                            )
                            if spec is None or name_i not in var_tables_all:
                                r_val, n_val = np.nan, 0
                            elif _is_firing_rate_spec(spec):
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
            avg_text = f" | avg across PIDs (n_pids={n_pids})" if AVG_BY_PID else ""
            fig_pearson.update_layout(
                title=(
                    "Reliability (diag) + Pairwise Pearson (off-diag) | "
                    f"Region {REGION_CORR} | N total ({label_filter_text}): {n_total_units}"
                    f"{avg_text}"
                ),
                height=min(1000, max(500, 40 * n_vars + 200)),
                margin=dict(l=90, r=30, t=90, b=90),
                template=PLOTLY_TEMPLATE,
            )
            fig_pearson.update_xaxes(tickangle=45)

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
                    f"Region {REGION_CORR} | N total ({label_filter_text}): {n_total_units}"
                    f"{avg_text}"
                ),
                height=min(1000, max(500, 40 * n_vars + 200)),
                margin=dict(l=90, r=30, t=90, b=90),
                template=PLOTLY_TEMPLATE,
            )
            fig_spearman.update_xaxes(tickangle=45)

            fig_pearson.show()
            fig_spearman.show()

# %% Correlation vs reliability by region (Plotly)
available_specs = []
for spec in CORR_VARIABLES:
    df_src = data_for_corr.get(spec["df"])
    if df_src is None:
        continue
    if spec["v1"] not in df_src.columns or spec["v2"] not in df_src.columns:
        continue
    available_specs.append(spec)

var_names = [spec["name"] for spec in available_specs]
if not var_names:
    print("No variables available for correlation/reliability plot.")
else:
    if USE_INTERACTIVE_PROMPTS:
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
    else:
        def _pick_var(preferred, names, fallback_idx):
            if preferred in names:
                return preferred
            idx = min(fallback_idx, len(names) - 1)
            return names[idx]

        var_x = _pick_var(SCATTER_VAR_X, var_names, 0)
        var_y = _pick_var(SCATTER_VAR_Y, var_names, 1 if len(var_names) > 1 else 0)
        if var_y == var_x and len(var_names) > 1:
            var_y = var_names[1]

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
        region_counts_all["n_neurons"] >= MIN_NEURONS_FOR_REGION, "region"
    ].tolist()

    if not eligible_regions:
        print("No regions meet the minimum neuron threshold for scatter plots.")
    else:
        region_colors = _build_region_colors(eligible_regions)

        def _build_scatter(method_name):
            records = []
            x_is_fr = _is_firing_rate_spec(spec_x)
            y_is_fr = _is_firing_rate_spec(spec_y)
            for region in sorted(eligible_regions):
                region_ids = neurons_df_calc.loc[
                    neurons_df_calc["region"] == region, ["pid", "cluster_id", "region"]
                ]
                if region_ids.empty:
                    continue

                if AVG_BY_PID:
                    pid_list = sorted(region_ids["pid"].unique().tolist())
                    rel_x_vals = []
                    rel_y_vals = []
                    corr_vals = []
                    for pid in pid_list:
                        region_pid = region_ids.loc[region_ids["pid"] == pid]
                        if region_pid.empty:
                            continue
                        region_pid_x = _filter_region_lookup_for_spec(
                            region_pid, spec_x, spont_pids
                        )
                        region_pid_y = _filter_region_lookup_for_spec(
                            region_pid, spec_y, spont_pids
                        )
                        df_var_x = _build_variable_table(
                            data_for_corr.get(spec_x["df"]), spec_x, region_pid_x
                        )
                        df_var_y = _build_variable_table(
                            data_for_corr.get(spec_y["df"]), spec_y, region_pid_y
                        )
                        if df_var_x is None or df_var_y is None:
                            continue

                        if method_name == "spearman":
                            rel_x_pid, _ = _spearmanr_with_n(
                                df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]]
                            )
                            rel_y_pid, _ = _spearmanr_with_n(
                                df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]]
                            )
                        else:
                            rel_x_pid, _ = _pearsonr_with_n(
                                df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]]
                            )
                            rel_y_pid, _ = _pearsonr_with_n(
                                df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]]
                            )

                        if x_is_fr:
                            rel_x_pid = np.nan
                        if y_is_fr:
                            rel_y_pid = np.nan

                        merged = df_var_x[["pid", "cluster_id", "mean"]].merge(
                            df_var_y[["pid", "cluster_id", "mean"]],
                            on=["pid", "cluster_id"],
                            how="inner",
                            suffixes=("_x", "_y"),
                        )
                        if merged.empty:
                            continue
                        if method_name == "spearman":
                            corr_pid, _ = _spearmanr_with_n(
                                merged["mean_x"], merged["mean_y"]
                            )
                        else:
                            corr_pid, _ = _pearsonr_with_n(
                                merged["mean_x"], merged["mean_y"]
                            )

                        rel_x_vals.append(rel_x_pid)
                        rel_y_vals.append(rel_y_pid)
                        corr_vals.append(corr_pid)

                    rel_x, n_rel_x = _mean_with_count(rel_x_vals)
                    rel_y, n_rel_y = _mean_with_count(rel_y_vals)
                    corr_val, n_corr = _mean_with_count(corr_vals)
                else:
                    region_ids_x = _filter_region_lookup_for_spec(
                        region_ids, spec_x, spont_pids
                    )
                    region_ids_y = _filter_region_lookup_for_spec(
                        region_ids, spec_y, spont_pids
                    )
                    df_var_x = _build_variable_table(
                        data_for_corr.get(spec_x["df"]), spec_x, region_ids_x
                    )
                    df_var_y = _build_variable_table(
                        data_for_corr.get(spec_y["df"]), spec_y, region_ids_y
                    )
                    if df_var_x is None or df_var_y is None:
                        continue

                    if method_name == "spearman":
                        rel_x, n_rel_x = _spearmanr_with_n(
                            df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]]
                        )
                        rel_y, n_rel_y = _spearmanr_with_n(
                            df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]]
                        )
                    else:
                        rel_x, n_rel_x = _pearsonr_with_n(
                            df_var_x[spec_x["v1"]], df_var_x[spec_x["v2"]]
                        )
                        rel_y, n_rel_y = _pearsonr_with_n(
                            df_var_y[spec_y["v1"]], df_var_y[spec_y["v2"]]
                        )

                    if x_is_fr:
                        rel_x, n_rel_x = np.nan, 0
                    if y_is_fr:
                        rel_y, n_rel_y = np.nan, 0

                    merged = df_var_x[["pid", "cluster_id", "mean"]].merge(
                        df_var_y[["pid", "cluster_id", "mean"]],
                        on=["pid", "cluster_id"],
                        how="inner",
                        suffixes=("_x", "_y"),
                    )
                    if merged.empty:
                        continue
                    if method_name == "spearman":
                        corr_val, n_corr = _spearmanr_with_n(
                            merged["mean_x"], merged["mean_y"]
                        )
                    else:
                        corr_val, n_corr = _pearsonr_with_n(
                            merged["mean_x"], merged["mean_y"]
                        )

                if x_is_fr and y_is_fr:
                    rel_total = np.nan
                elif x_is_fr:
                    rel_total = rel_y
                elif y_is_fr:
                    rel_total = rel_x
                else:
                    rel_prod = (
                        rel_x * rel_y if np.isfinite(rel_x) and np.isfinite(rel_y) else np.nan
                    )
                    if rel_prod is not None and np.isfinite(rel_prod) and rel_prod >= 0:
                        rel_total = float(np.sqrt(rel_prod))
                    else:
                        rel_total = np.nan

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
            df_plot = df_plot[
                np.isfinite(df_plot["corr"]) & np.isfinite(df_plot["reliability"])
            ]
            if df_plot.empty:
                return None

            highlight_regions = {
                "VISp",
                "MOs",
                "CP",
                "CA1",
                "SCm",
                "ZI",
                "AUDp",
                "GRN",
            }

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
                            f"n={row['n_neurons']}<extra></extra>"
                        ),
                    )
                )

            avg_text = " (avg across PIDs)" if AVG_BY_PID else ""
            fig.update_layout(
                title=(
                    f"{method_name.title()} correlation vs total reliability | "
                    f"{var_x} vs {var_y} | "
                    f"regions with >= {MIN_NEURONS_FOR_REGION} neurons ({label_filter_text})"
                    f"{avg_text}"
                ),
                xaxis_title="Total reliability (sqrt(rel1 * rel2))",
                yaxis_title="Correlation between variables",
                template=PLOTLY_TEMPLATE,
                legend=dict(x=1.02, y=1, yanchor="top"),
                margin=dict(l=80, r=200, t=90, b=70),
                height=600,
                width=900,
            )
            x_vals = df_plot["reliability"].to_numpy(dtype=float)
            y_vals = df_plot["corr"].to_numpy(dtype=float)
            min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
            max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
                fig.add_shape(
                    type="line",
                    x0=min_val,
                    y0=min_val,
                    x1=max_val,
                    y1=max_val,
                    line=dict(color="red", dash="dash"),
                )
            return fig

        fig_p = _build_scatter("pearson")
        fig_s = _build_scatter("spearman")

        if fig_p is None:
            print("No data for Pearson scatter.")
        else:
            fig_p.show()
        if fig_s is None:
            print("No data for Spearman scatter.")
        else:
            fig_s.show()

# %% Histogram of neurons per region
if not region_counts_calc.empty:
    threshold_label = label_min_text if len(label_min_values) == 1 else "varies"
    if GOOD_ONLY_FOR_CORR:
        threshold_label = "label=1 only"
    plt.figure(figsize=(9, 5))
    plt.hist(
        region_counts_calc["n_neurons"],
        bins="auto",
        color="#4C72B0",
        edgecolor="white",
    )
    plt.xlabel(f"Neurons per region ({threshold_label})")
    plt.ylabel("Number of regions")
    plt.title("Distribution of neuron counts across regions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No region counts available to plot.")

# %% stPR delay vs task delays by PID (AUD regions)
REGIONS_STPR = ["AUDp", "AUDpo", "AUDv"]
SPEC_STPR_NAME = "stPR Delay (Task)"
SPEC_STIM_NAME = "Delay (Stim On)"
SPEC_FEEDBACK_NAME = "Delay (Feedback)"


def _find_spec(name):
    return next((spec for spec in CORR_VARIABLES if spec["name"] == name), None)


def _format_corr_pair(pearson_val, spearman_val):
    p_txt = f"{pearson_val:.2f}" if np.isfinite(pearson_val) else "nan"
    s_txt = f"{spearman_val:.2f}" if np.isfinite(spearman_val) else "nan"
    return f"{p_txt} ({s_txt})"


try:
    from IPython.display import display as _display
except Exception:
    _display = None


def _show_df(df, title):
    print(title)
    if _display is not None:
        _display(df)
    else:
        print(df.to_string(index=False))


spec_stpr = _find_spec(SPEC_STPR_NAME)
spec_stim = _find_spec(SPEC_STIM_NAME)
spec_feedback = _find_spec(SPEC_FEEDBACK_NAME)

if spec_stpr is None or spec_stim is None or spec_feedback is None:
    print(
        "Missing spec(s) for stPR/stim/feedback delay. "
        "Check CORR_VARIABLES for required entries."
    )
else:
    for region in REGIONS_STPR:
        region_lookup = neurons_df_calc.loc[
            neurons_df_calc["region"] == region, ["pid", "cluster_id", "region"]
        ]
        pids = sorted(region_lookup["pid"].unique().tolist())
        if not pids:
            df_out = pd.DataFrame(
                columns=["pid", "stpr_vs_stim", "stpr_vs_feedback"]
            )
            _show_df(df_out, f"Region {region} | stPR delay vs task delays")
            continue

        region_lookup_stpr = _filter_region_lookup_for_spec(
            region_lookup, spec_stpr, spont_pids
        )
        region_lookup_stim = _filter_region_lookup_for_spec(
            region_lookup, spec_stim, spont_pids
        )
        region_lookup_fb = _filter_region_lookup_for_spec(
            region_lookup, spec_feedback, spont_pids
        )

        df_stpr = _build_variable_table(
            data_for_corr.get(spec_stpr["df"]), spec_stpr, region_lookup_stpr
        )
        df_stim = _build_variable_table(
            data_for_corr.get(spec_stim["df"]), spec_stim, region_lookup_stim
        )
        df_fb = _build_variable_table(
            data_for_corr.get(spec_feedback["df"]), spec_feedback, region_lookup_fb
        )

        if df_stpr is None or df_stim is None or df_fb is None:
            print(f"Missing data for region {region}; returning NaNs.")
            rows = [
                {"pid": pid, "stpr_vs_stim": "nan (nan)", "stpr_vs_feedback": "nan (nan)"}
                for pid in pids
            ]
            df_out = pd.DataFrame(rows)
            _show_df(df_out, f"Region {region} | stPR delay vs task delays")
            continue

        rows = []
        for pid in pids:
            stpr_pid = df_stpr.loc[df_stpr["pid"] == pid, ["pid", "cluster_id", "mean"]]
            stim_pid = df_stim.loc[df_stim["pid"] == pid, ["pid", "cluster_id", "mean"]]
            fb_pid = df_fb.loc[df_fb["pid"] == pid, ["pid", "cluster_id", "mean"]]

            merged_stim = stpr_pid.merge(
                stim_pid, on=["pid", "cluster_id"], how="inner", suffixes=("_stpr", "_stim")
            )
            if merged_stim.empty:
                r_p_stim, r_s_stim = np.nan, np.nan
            else:
                r_p_stim, _ = _pearsonr_with_n(
                    merged_stim["mean_stpr"], merged_stim["mean_stim"]
                )
                r_s_stim, _ = _spearmanr_with_n(
                    merged_stim["mean_stpr"], merged_stim["mean_stim"]
                )

            merged_fb = stpr_pid.merge(
                fb_pid, on=["pid", "cluster_id"], how="inner", suffixes=("_stpr", "_fb")
            )
            if merged_fb.empty:
                r_p_fb, r_s_fb = np.nan, np.nan
            else:
                r_p_fb, _ = _pearsonr_with_n(
                    merged_fb["mean_stpr"], merged_fb["mean_fb"]
                )
                r_s_fb, _ = _spearmanr_with_n(
                    merged_fb["mean_stpr"], merged_fb["mean_fb"]
                )

            rows.append(
                {
                    "pid": pid,
                    "stpr_vs_stim": _format_corr_pair(r_p_stim, r_s_stim),
                    "stpr_vs_feedback": _format_corr_pair(r_p_fb, r_s_fb),
                }
            )

        df_out = pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)
        _show_df(df_out, f"Region {region} | stPR delay vs task delays")
