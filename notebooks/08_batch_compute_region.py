"""
Batch compute region dashboard metrics to be consumed by 09_dashboard_region_fast.py

Run this script directly to precalculate correlations and arousal statistics into Parquet files.
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import warnings
import sys
from tqdm import tqdm

warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))

CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
OUTPUT_DIR = BASE_PATH / "data" / "dashboard_region_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LABEL_MIN = 0.59
MIN_NEURONS = 30  # As per user request, setting min filter statically

# =========================================================
# 1. Dashboard Constants & Helper Functions
# =========================================================

COUPLING_TABLE_KEYS = (
    "df_coupling",
    "df_coupling_task",
    "df_coupling_iti",
    "df_coupling_good",
    "df_coupling_task_good",
    "df_coupling_iti_good",
)


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
NEW_EVENT_RESPONSE_VARIABLES = {
    "Response to Stim On",
    "Response to First Move",
    "Response to Feedback",
    "Response to Whisking Events",
    "Delay to Feedback (Correct Trials)",
    "Delay to Feedback (Incorrect Trials)",
    "Response to Feedback (Correct Trials)",
    "Response to Feedback (Incorrect Trials)",
}
SPONT_COUPLING_STRENGTH_VAR = "Coupling Strength (Spont)"


def _build_corr_variables():
    return [dict(spec) for spec in CORR_VARIABLES]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build region-summary cache tables consumed by 09_dashboard_region_fast.py."
        )
    )
    parser.add_argument(
        "--append-missing-only",
        action="store_true",
        help=(
            "Reuse existing dashboard-region outputs and only compute correlation rows "
            "for variables missing from the saved summary cache."
        ),
    )
    parser.add_argument(
        "--focus-spont-strength-newvars",
        action="store_true",
        help=(
            "Only compute the rows needed for Coupling Strength (Spont) versus the new "
            "event-response variables, plus the diagonal reliability rows those plots need."
        ),
    )
    return parser.parse_args()

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
    df_var = df_var.groupby(["pid", "cluster_id"], as_index=False).mean(numeric_only=True)
    df_var = df_var.merge(region_lookup, on=["pid", "cluster_id"], how="inner")
    v1 = df_var[spec["v1"]].to_numpy(dtype=float)
    v2 = df_var[spec["v2"]].to_numpy(dtype=float)
    mean_vals = np.full(len(df_var), np.nan, dtype=float)
    valid = np.isfinite(v1) & np.isfinite(v2)
    mean_vals[valid] = (v1[valid] + v2[valid]) / 2.0
    df_var["mean"] = mean_vals
    return df_var

def _build_label_lookup(df_res):
    if df_res is None or df_res.empty: return None
    if "label" not in df_res.columns: return None
    df_labels = df_res[["pid", "cluster_id", "label"]].copy()
    df_labels["cluster_id"] = pd.to_numeric(df_labels["cluster_id"], errors="coerce")
    df_labels = df_labels[np.isfinite(df_labels["cluster_id"])].copy()
    if df_labels.empty: return None
    df_labels["cluster_id"] = df_labels["cluster_id"].astype(int)
    df_labels["label"] = pd.to_numeric(df_labels["label"], errors="coerce")
    df_labels = df_labels[np.isfinite(df_labels["label"])].copy()
    if df_labels.empty: return None
    df_labels = df_labels.groupby(["pid", "cluster_id"], as_index=False)["label"].max()
    return df_labels

def _is_firing_rate_spec(spec):
    return spec.get("df") in {"df_firing_rate", "df_depth", "df_arousal_corr"}

def _is_spont_spec(spec):
    return spec.get("df") == "df_coupling"

def _has_spont_interval(meta):
    if not meta: return False
    interval = meta.get("spont_interval")
    if interval is None: return False
    try:
        start, end = interval
        start_val = float(start)
        end_val = float(end)
        return np.isfinite(start_val) and np.isfinite(end_val) and end_val > start_val
    except (TypeError, ValueError):
        return False

def _filter_region_lookup_for_spec(region_lookup, spec, spont_pids):
    if region_lookup is None or region_lookup.empty: return region_lookup
    if not _is_spont_spec(spec): return region_lookup
    if not spont_pids: return region_lookup.iloc[0:0]
    if "pid" not in region_lookup.columns: return region_lookup
    return region_lookup[region_lookup["pid"].isin(spont_pids)]

def _mean_with_count(values):
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite): return np.nan, 0
    return float(np.nanmean(vals[finite])), int(np.sum(finite))

def _is_nonempty_arraylike(value):
    if value is None: return False
    if isinstance(value, pd.DataFrame): return not value.empty
    if isinstance(value, dict): return len(value) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True

def _has_whisk_signal(cache):
    df_wh = cache.get("df_wh")
    return isinstance(df_wh, pd.DataFrame) and (not df_wh.empty)

def _has_passive_stimuli(cache):
    passive = cache.get("passive_events")
    if not isinstance(passive, dict): return False
    keys = ("passive_visual_top2_right_times", "passive_visual_top2_left_times", "passive_tone_times", "passive_valve_times", "passive_noise_times")
    for key in keys:
        val = passive.get(key)
        if not _is_nonempty_arraylike(val): continue
        try:
            arr = np.asarray(val).reshape(-1)
            if arr.size > 0: return True
        except Exception:
            continue
    return False

def _extract_cluster_depths(cache, cluster_ids):
    if cluster_ids is None: return None
    try:
        cluster_ids_arr = np.asarray(cluster_ids)
    except Exception:
        return None
    n_clusters = int(cluster_ids_arr.size)
    if n_clusters == 0: return None

    def _to_depth_array(values):
        if values is None: return None
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == n_clusters: return arr
        except Exception:
            pass
        return None

    for key in ("cluster_depths", "cluster_depth", "depths", "cluster_depth_um", "depth_um"):
        arr = _to_depth_array(cache.get(key))
        if arr is not None: return arr

    clusters = cache.get("clusters")
    if clusters is not None:
        if isinstance(clusters, dict):
            for key in ("depths", "depth", "axial_um", "depth_um"):
                arr = _to_depth_array(clusters.get(key))
                if arr is not None: return arr
        else:
            for attr in ("depths", "depth", "axial_um", "depth_um"):
                arr = _to_depth_array(getattr(clusters, attr, None))
                if arr is not None: return arr
    return None


def _read_saved_output_table(stem):
    pkl_path = OUTPUT_DIR / f"{stem}.pkl"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    parquet_path = OUTPUT_DIR / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None


def _load_existing_output_bundle():
    df_corr = _read_saved_output_table("summary_correlations")
    df_arousal = _read_saved_output_table("summary_arousal_fractions")
    df_region = _read_saved_output_table("summary_region_counts")
    df_pids = _read_saved_output_table("summary_pids")
    label_min_text = "0.50 (batch minimum)"
    meta_path = OUTPUT_DIR / "summary_metadata.pkl"
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            label_min_text = str(meta.get("label_min_text", label_min_text))
        except Exception:
            pass
    return {
        "df_corr": df_corr,
        "df_arousal": df_arousal,
        "df_region": df_region,
        "df_pids": df_pids,
        "label_min_text": label_min_text,
    }


def _corr_record_key(good_only, use_stpr, avg_by_pid, arousal_group, region, var1, var2):
    return (
        bool(good_only),
        bool(use_stpr),
        bool(avg_by_pid),
        str(arousal_group),
        str(region),
        str(var1),
        str(var2),
    )


def _build_existing_corr_key_set(df_corr):
    if df_corr is None or df_corr.empty:
        return set()
    required_cols = ["good_only", "use_good_stpr", "avg_by_pid", "arousal_group", "region", "var1", "var2"]
    if not set(required_cols).issubset(df_corr.columns):
        return set()
    key_set = set()
    for row in df_corr[required_cols].itertuples(index=False, name=None):
        key_set.add(_corr_record_key(*row))
    return key_set


def _should_compute_pair(name_i, name_j, target_var_names=None, focus_source_var=None):
    name_i = str(name_i)
    name_j = str(name_j)
    target_set = None if target_var_names is None else {str(v) for v in target_var_names}

    if focus_source_var is not None:
        focus_source = str(focus_source_var)
        if name_i == name_j:
            if name_i == focus_source:
                return True
            return target_set is None or name_i in target_set
        return (
            (name_i == focus_source and (target_set is None or name_j in target_set))
            or (name_j == focus_source and (target_set is None or name_i in target_set))
        )

    if target_set is None:
        return True
    if name_i == name_j:
        return name_i in target_set
    return (name_i in target_set) or (name_j in target_set)

def _build_arousal_lookup(df_res, region_lookup):
    if (df_res is None or df_res.empty or "pid" not in df_res.columns or "cluster_id" not in df_res.columns or "arousal_group" not in df_res.columns): return None
    if region_lookup is None or region_lookup.empty: return None

    df_arousal = df_res[["pid", "cluster_id", "arousal_group"]].copy()
    df_arousal["cluster_id"] = pd.to_numeric(df_arousal["cluster_id"], errors="coerce")
    df_arousal = df_arousal[np.isfinite(df_arousal["cluster_id"])].copy()
    if df_arousal.empty: return None
    df_arousal["cluster_id"] = df_arousal["cluster_id"].astype(int)

    plus_labels = {"arousal_plus", "plus", "+", "arousal +", "arousal+"}
    minus_labels = {"arousal_minus", "minus", "-", "arousal -", "arousal-"}
    groups_raw = df_arousal["arousal_group"].astype(str).str.strip().str.lower()
    groups_norm = pd.Series("arousal_neutral", index=df_arousal.index, dtype=object)
    groups_norm[groups_raw.isin(plus_labels)] = "arousal_plus"
    groups_norm[groups_raw.isin(minus_labels)] = "arousal_minus"
    df_arousal["arousal_group"] = groups_norm
    if df_arousal.empty: return None
    df_arousal = df_arousal.groupby(["pid", "cluster_id"])["arousal_group"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).reset_index()

    region_lookup_all = region_lookup[["pid", "cluster_id", "region"]].drop_duplicates()
    df_arousal = df_arousal.merge(region_lookup_all, on=["pid", "cluster_id"], how="inner")
    return df_arousal if not df_arousal.empty else None

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
        "df_depth": [],
        "df_arousal_corr": [],
    }

    print(f"Loading {len(cache_paths)} cache files...")
    for path in tqdm(cache_paths, desc="Load raw cache", unit="pid"):
        with open(path, "rb") as f:
            cache = pickle.load(f)

        pid = cache.get("pid", path.stem)
        meta = cache.get("meta") or {}
        has_spont = _has_spont_interval(meta)
        has_whisk = _has_whisk_signal(cache)
        has_passive = _has_passive_stimuli(cache)
        if has_spont: spont_pids.add(pid)
        config_calc = cache.get("config_calc") or {}
        try:
            label_min = float(config_calc.get("CALC_LABEL_MIN", DEFAULT_LABEL_MIN))
        except (TypeError, ValueError):
            label_min = float(DEFAULT_LABEL_MIN)
        label_mins.append(label_min)

        df_res = cache.get("df_res")
        if df_res is None or len(df_res) == 0:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min, "has_spont_interval": has_spont, "has_whisk_signal": has_whisk, "has_passive_stimuli": has_passive})
            continue

        if "label" in df_res.columns:
            labels = pd.to_numeric(df_res["label"], errors="coerce")
            df_units = df_res[labels >= label_min].copy()
        else:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min, "has_spont_interval": has_spont, "has_whisk_signal": has_whisk, "has_passive_stimuli": has_passive})
            continue

        if "acronym" in df_units.columns:
            region_col = "acronym"
        elif "region" in df_units.columns:
            region_col = "region"
        else:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min, "has_spont_interval": has_spont, "has_whisk_signal": has_whisk, "has_passive_stimuli": has_passive})
            continue

        df_units = df_units[["cluster_id", region_col]].copy()
        df_units["cluster_id"] = pd.to_numeric(df_units["cluster_id"], errors="coerce")
        df_units = df_units[np.isfinite(df_units["cluster_id"])].copy()
        if df_units.empty:
            pid_summary_rows.append({"pid": pid, "n_neurons": 0, "label_min": label_min, "has_spont_interval": has_spont, "has_whisk_signal": has_whisk, "has_passive_stimuli": has_passive})
            continue
        df_units["cluster_id"] = df_units["cluster_id"].astype(int)
        df_units["region"] = df_units[region_col].astype(str)
        df_units = df_units[~df_units["region"].isin(["root", "void", "NA", "nan"])]
        df_units["pid"] = pid

        rows.append(df_units[["pid", "cluster_id", "region"]])
        pid_summary_rows.append({"pid": pid, "n_neurons": int(len(df_units)), "label_min": label_min, "has_spont_interval": has_spont, "has_whisk_signal": has_whisk, "has_passive_stimuli": has_passive})

        df_res_copy = df_res.copy()
        df_res_copy["pid"] = pid
        df_res_copy["cluster_id"] = pd.to_numeric(df_res_copy["cluster_id"], errors="coerce")
        df_res_copy = df_res_copy[np.isfinite(df_res_copy["cluster_id"])].copy()
        if not df_res_copy.empty:
            df_res_copy["cluster_id"] = df_res_copy["cluster_id"].astype(int)
            data_tables["df_res"].append(df_res_copy)
            if "arousal_corr_abs" in df_res_copy.columns:
                df_ar = df_res_copy[["pid", "cluster_id", "arousal_corr_abs"]].copy()
                df_ar["arousal_corr_abs_h1"] = pd.to_numeric(df_ar["arousal_corr_abs"], errors="coerce")
                df_ar["arousal_corr_abs_h2"] = df_ar["arousal_corr_abs_h1"]
                df_ar = df_ar[["pid", "cluster_id", "arousal_corr_abs_h1", "arousal_corr_abs_h2"]]
                data_tables["df_arousal_corr"].append(df_ar)

        for key in COUPLING_TABLE_KEYS:
            df_tbl = cache.get(key)
            if df_tbl is None or df_tbl.empty: continue
            df_tbl = df_tbl.copy()
            df_tbl["pid"] = pid
            df_tbl["cluster_id"] = pd.to_numeric(df_tbl["cluster_id"], errors="coerce")
            df_tbl = df_tbl[np.isfinite(df_tbl["cluster_id"])].copy()
            if df_tbl.empty: continue
            df_tbl["cluster_id"] = df_tbl["cluster_id"].astype(int)
            data_tables[key].append(df_tbl)

        cluster_ids = cache.get("cluster_ids")
        cluster_firing_rate = cache.get("cluster_firing_rate")
        if cluster_ids is not None and cluster_firing_rate is not None:
            cluster_ids_clean = pd.to_numeric(np.asarray(cluster_ids), errors="coerce")
            cluster_firing_rate = np.asarray(cluster_firing_rate, dtype=float)
            if len(cluster_ids_clean) == len(cluster_firing_rate):
                df_rate = pd.DataFrame({"pid": pid, "cluster_id": cluster_ids_clean, "firing_rate_h1": cluster_firing_rate, "firing_rate_h2": cluster_firing_rate})
                df_rate = df_rate[np.isfinite(df_rate["cluster_id"])].copy()
                if not df_rate.empty:
                    df_rate["cluster_id"] = df_rate["cluster_id"].astype(int)
                    data_tables["df_firing_rate"].append(df_rate)

        cluster_depths = _extract_cluster_depths(cache, cluster_ids)
        if cluster_ids is not None and cluster_depths is not None:
            cluster_ids_depth = pd.to_numeric(np.asarray(cluster_ids), errors="coerce")
            cluster_depths = np.asarray(cluster_depths, dtype=float)
            if len(cluster_ids_depth) == len(cluster_depths):
                df_depth = pd.DataFrame({"pid": pid, "cluster_id": cluster_ids_depth, "depth_h1": cluster_depths, "depth_h2": cluster_depths})
                df_depth = df_depth[np.isfinite(df_depth["cluster_id"])].copy()
                if not df_depth.empty:
                    df_depth["cluster_id"] = df_depth["cluster_id"].astype(int)
                    data_tables["df_depth"].append(df_depth)

    if rows:
        neurons_df = pd.concat(rows, ignore_index=True)
    else:
        neurons_df = pd.DataFrame(columns=["pid", "cluster_id", "region"])
    
    pid_summary = pd.DataFrame(pid_summary_rows).sort_values("pid").reset_index(drop=True)
    data_concat = {key: (pd.concat(tables, ignore_index=True) if tables else None) for key, tables in data_tables.items()}
    return neurons_df, pid_summary, data_concat, list(spont_pids)


def _write_region_cache_outputs(df_corr, df_arousal_frac, df_region_summary, pid_summary):
    print("Exporting cache tables to", OUTPUT_DIR, "...")
    pid_summary.to_parquet(OUTPUT_DIR / "summary_pids.parquet", index=False)
    pid_summary.to_pickle(OUTPUT_DIR / "summary_pids.pkl")

    df_corr.to_parquet(OUTPUT_DIR / "summary_correlations.parquet", index=False)
    df_corr.to_pickle(OUTPUT_DIR / "summary_correlations.pkl")

    df_arousal_frac.to_parquet(OUTPUT_DIR / "summary_arousal_fractions.parquet", index=False)
    df_arousal_frac.to_pickle(OUTPUT_DIR / "summary_arousal_fractions.pkl")

    df_region_summary.to_parquet(OUTPUT_DIR / "summary_region_counts.parquet", index=False)
    df_region_summary.to_pickle(OUTPUT_DIR / "summary_region_counts.pkl")

    with open(OUTPUT_DIR / "summary_metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "label_min_text": "0.50 (batch minimum)",
            },
            f,
        )

def precompute_all_dashboards(
    write_outputs=True,
    append_missing_only=False,
    focus_spont_strength_newvars=False,
):
    print("Loading data from cache... This may take a moment.")
    neurons_df_raw, pid_summary, data_concat, spont_pids_list = _load_cache_tables(CACHE_DIR)
    spont_pids = set(spont_pids_list)
    print("Data loading completed.")

    corr_variables = _build_corr_variables()
    label_min_text = "0.50 (batch minimum)"
    existing_bundle = None
    existing_corr = None
    existing_corr_keys = set()
    target_var_names = None
    focus_source_var = None
    reuse_existing_side_tables = False

    if focus_spont_strength_newvars:
        append_missing_only = True
        focus_source_var = SPONT_COUPLING_STRENGTH_VAR
        target_var_names = set(NEW_EVENT_RESPONSE_VARIABLES)

    if append_missing_only:
        existing_bundle = _load_existing_output_bundle()
        existing_corr = existing_bundle.get("df_corr")
        reuse_existing_side_tables = all(
            isinstance(existing_bundle.get(key), pd.DataFrame)
            for key in ("df_arousal", "df_region", "df_pids")
        )
        if isinstance(existing_corr, pd.DataFrame) and not existing_corr.empty:
            existing_corr_keys = _build_existing_corr_key_set(existing_corr)
            label_min_text = existing_bundle.get("label_min_text", label_min_text)
            if focus_spont_strength_newvars:
                target_var_names = set(
                    name for name in NEW_EVENT_RESPONSE_VARIABLES
                    if any(spec["name"] == name for spec in corr_variables)
                )
                print(
                    "Focused mode: computing only rows for "
                    f"{SPONT_COUPLING_STRENGTH_VAR} versus: "
                    + ", ".join(sorted(target_var_names))
                )
            else:
                existing_var_series = pd.concat(
                    [
                        existing_corr.get("var1", pd.Series(dtype=object)),
                        existing_corr.get("var2", pd.Series(dtype=object)),
                    ],
                    ignore_index=True,
                )
                existing_var_names = set(existing_var_series.dropna().astype(str))
                registry_var_names = [spec["name"] for spec in corr_variables]
                missing_var_names = [name for name in registry_var_names if name not in existing_var_names]
                if missing_var_names:
                    target_var_names = set(missing_var_names)
                    print(
                        "Append-missing-only mode: computing rows involving missing variables: "
                        + ", ".join(missing_var_names)
                    )
                else:
                    print("Append-missing-only mode: no missing variables were found in saved outputs.")
                    df_arousal_existing = existing_bundle.get("df_arousal")
                    df_region_existing = existing_bundle.get("df_region")
                    df_pids_existing = existing_bundle.get("df_pids")
                    if existing_corr is None:
                        existing_corr = pd.DataFrame()
                    return (
                        existing_corr,
                        df_arousal_existing if isinstance(df_arousal_existing, pd.DataFrame) else pd.DataFrame(),
                        df_region_existing if isinstance(df_region_existing, pd.DataFrame) else pd.DataFrame(),
                        df_pids_existing if isinstance(df_pids_existing, pd.DataFrame) else pid_summary,
                        label_min_text,
                    )
        else:
            print("Append-missing-only mode requested, but no existing summary_correlations cache was found. Falling back to full recompute.")
            append_missing_only = False
            focus_spont_strength_newvars = False
            target_var_names = None
            focus_source_var = None

    # Label lookup
    label_lookup = _build_label_lookup(data_concat.get("df_res"))

    # Prepare export structures
    corr_records = []
    arousal_frac_records = []
    region_summary_records = []

    # Iteration logic
    good_options = [False, True]
    stpr_options = [False, True]
    avg_options = [False, True]
    
    # Calculate total iterations for the progress bar
    total_iters = len(good_options) * len(stpr_options) * len(avg_options)
    
    print("Starting computations matrix generation...")
    pbar = tqdm(total=total_iters, desc="Toggle Sets", unit="toggle")
    
    for good_only in good_options:
        if good_only and label_lookup is not None:
            good_mask = np.isclose(label_lookup["label"].to_numpy(dtype=float), 1.0)
            label_good = label_lookup.loc[good_mask, ["pid", "cluster_id"]]
            neurons_df_calc = neurons_df_raw.merge(label_good, on=["pid", "cluster_id"], how="inner")
        else:
            neurons_df_calc = neurons_df_raw

        # Save Region Counts
        if not reuse_existing_side_tables:
            region_counts_tmp = (
                neurons_df_calc["region"].value_counts().rename_axis("region").reset_index(name="n_neurons")
            )
            region_pid_counts_tmp = neurons_df_calc.groupby("region")["pid"].nunique().rename("n_pids").reset_index()
            region_counts_tmp = region_counts_tmp.merge(region_pid_counts_tmp, on="region", how="left")
            region_counts_tmp["good_only"] = good_only
            region_summary_records.append(region_counts_tmp)

        region_lookup_all = neurons_df_calc[["pid", "cluster_id", "region"]].drop_duplicates()
        df_arousal_split = _build_arousal_lookup(data_concat.get("df_res"), region_lookup_all)

        for use_stpr in stpr_options:
            data_for_corr = data_concat.copy()
            if use_stpr:
                data_for_corr["df_coupling"] = data_concat.get("df_coupling_good")
                data_for_corr["df_coupling_task"] = data_concat.get("df_coupling_task_good")
                data_for_corr["df_coupling_iti"] = data_concat.get("df_coupling_iti_good")
            
            available_specs = [
                spec for spec in corr_variables
                if data_for_corr.get(spec["df"]) is not None and
                   spec["v1"] in data_for_corr[spec["df"]].columns and
                   spec["v2"] in data_for_corr[spec["df"]].columns
            ]
            if focus_source_var is not None and target_var_names is not None:
                focus_allowed = set(target_var_names) | {focus_source_var}
                available_specs = [spec for spec in available_specs if spec["name"] in focus_allowed]
            
            for avg_by_pid in avg_options:
                toggle_text = f"good={int(good_only)} stpr={int(use_stpr)} avgpid={int(avg_by_pid)}"
                pbar.set_postfix_str(toggle_text)
                # ---------------------------------------------------------
                # AROUSAL FRACTIONS EXPORT (Only depends on good_only and avg_by_pid)
                # ---------------------------------------------------------
                if (not reuse_existing_side_tables) and df_arousal_split is not None and not df_arousal_split.empty and not use_stpr:
                    # we only need to compute this once per (good_only, avg_by_pid)
                    if avg_by_pid:
                        df_pid = df_arousal_split.groupby(["region", "pid"])["arousal_group"].agg(
                            n_total="count", n_plus=lambda s: int((s == "arousal_plus").sum()), n_minus=lambda s: int((s == "arousal_minus").sum())
                        ).reset_index()
                        df_pid = df_pid[df_pid["n_total"] > 0]
                        if not df_pid.empty:
                            df_pid["frac_plus_pct"] = 100.0 * df_pid["n_plus"] / df_pid["n_total"]
                            df_pid["frac_minus_pct"] = 100.0 * df_pid["n_minus"] / df_pid["n_total"]
                            df_plot = df_pid.groupby("region").agg(
                                frac_plus_pct=("frac_plus_pct", "mean"), frac_minus_pct=("frac_minus_pct", "mean"),
                                n_pids=("pid", "nunique"), n_neurons=("n_total", "sum")
                            ).reset_index()
                            df_plot["good_only"] = good_only
                            df_plot["avg_by_pid"] = avg_by_pid
                            arousal_frac_records.append(df_plot)
                    else:
                        df_plot = df_arousal_split.groupby("region")["arousal_group"].agg(
                            n_total="count", n_plus=lambda s: int((s == "arousal_plus").sum()), n_minus=lambda s: int((s == "arousal_minus").sum())
                        ).reset_index()
                        df_plot = df_plot[df_plot["n_total"] > 0]
                        if not df_plot.empty:
                            df_plot["frac_plus_pct"] = 100.0 * df_plot["n_plus"] / df_plot["n_total"]
                            df_plot["frac_minus_pct"] = 100.0 * df_plot["n_minus"] / df_plot["n_total"]
                            df_plot["n_neurons"] = df_plot["n_total"].astype(int)
                            df_plot["n_pids"] = df_arousal_split.groupby("region")["pid"].nunique().reindex(df_plot["region"]).to_numpy()
                            df_plot["good_only"] = good_only
                            df_plot["avg_by_pid"] = avg_by_pid
                            arousal_frac_records.append(df_plot)

                # ---------------------------------------------------------
                # CORRELATIONS EXPORT (Depends on all toggles)
                # ---------------------------------------------------------
                groups_to_process = [("all", neurons_df_calc)]
                if (not focus_spont_strength_newvars) and df_arousal_split is not None and not df_arousal_split.empty:
                    groups_to_process.append(("arousal_plus", df_arousal_split[df_arousal_split["arousal_group"] == "arousal_plus"]))
                    groups_to_process.append(("arousal_minus", df_arousal_split[df_arousal_split["arousal_group"] == "arousal_minus"]))

                group_region_specs = []
                total_regions = 0
                for group_name, df_group in groups_to_process:
                    region_counts_grp = df_group.groupby("region")["cluster_id"].nunique().reset_index(name="n_neurons")
                    eligible_regions = region_counts_grp[region_counts_grp["n_neurons"] >= MIN_NEURONS]["region"].tolist()
                    group_region_specs.append((group_name, df_group, eligible_regions))
                    total_regions += len(eligible_regions)

                region_pbar = tqdm(
                    total=total_regions,
                    desc=f"Regions {toggle_text}",
                    unit="region",
                    leave=False,
                )

                for group_name, df_group, eligible_regions in group_region_specs:
                    for region in eligible_regions:
                        region_lookup = df_group[df_group["region"] == region][["pid", "cluster_id", "region"]].drop_duplicates()
                        if region_lookup.empty:
                            region_pbar.update(1)
                            continue
                        
                        var_tables_all = {}
                        for spec in available_specs:
                            rls = _filter_region_lookup_for_spec(region_lookup, spec, spont_pids)
                            df_var = _build_variable_table(data_for_corr.get(spec["df"]), spec, rls)
                            if df_var is not None and not df_var.empty:
                                var_tables_all[spec["name"]] = df_var
                        
                        names = [s["name"] for s in available_specs if s["name"] in var_tables_all]
                        if not names:
                            region_pbar.update(1)
                            continue
                        if target_var_names is not None and not any(
                            _should_compute_pair(name, name, target_var_names=target_var_names, focus_source_var=focus_source_var)
                            or _should_compute_pair(name, focus_source_var, target_var_names=target_var_names, focus_source_var=focus_source_var)
                            for name in names
                        ):
                            region_pbar.update(1)
                            continue
                        
                        # Process based on avg_by_pid
                        if avg_by_pid:
                            pid_list = region_lookup["pid"].unique()
                            # Dict to hold pid values before averaging
                            diag_names = [
                                name for name in names
                                if _should_compute_pair(
                                    name,
                                    name,
                                    target_var_names=target_var_names,
                                    focus_source_var=focus_source_var,
                                )
                            ]
                            pair_keys = [
                                (a, b) for a in names for b in names
                                if a != b and _should_compute_pair(
                                    a,
                                    b,
                                    target_var_names=target_var_names,
                                    focus_source_var=focus_source_var,
                                )
                            ]
                            rel_vals = {name: [] for name in diag_names}
                            rel_s_vals = {name: [] for name in diag_names}
                            corr_vals = {key: [] for key in pair_keys}
                            corr_s_vals = {key: [] for key in pair_keys}

                            for pid in pid_list:
                                region_pid = region_lookup[region_lookup["pid"] == pid]
                                if region_pid.empty: continue
                                
                                var_tables_pid = {}
                                for spec in available_specs:
                                    if spec["name"] not in names: continue
                                    rls = _filter_region_lookup_for_spec(region_pid, spec, spont_pids)
                                    df_var = _build_variable_table(data_for_corr.get(spec["df"]), spec, rls)
                                    if df_var is not None and not df_var.empty:
                                        var_tables_pid[spec["name"]] = df_var
                                
                                if not var_tables_pid: continue
                                
                                for spec in available_specs:
                                    name = spec["name"]
                                    if name not in rel_vals: continue
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
                                    r_s, _ = _spearmanr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                                    rel_vals[name].append(r_val)
                                    rel_s_vals[name].append(r_s)

                                mean_wide = region_pid[["pid", "cluster_id"]].drop_duplicates()
                                for name in names:
                                    df_var = var_tables_pid.get(name)
                                    if df_var is None:
                                        mean_wide[name] = np.nan
                                        continue
                                    mean_wide = mean_wide.merge(df_var[["pid", "cluster_id", "mean"]], on=["pid", "cluster_id"], how="left").rename(columns={"mean": name})

                                for name_i in names:
                                    for name_j in names:
                                        if name_i == name_j:
                                            continue
                                        if (name_i, name_j) not in corr_vals:
                                            continue
                                        r_val, _ = _pearsonr_with_n(mean_wide[name_i], mean_wide[name_j])
                                        r_s, _ = _spearmanr_with_n(mean_wide[name_i], mean_wide[name_j])
                                        corr_vals[(name_i, name_j)].append(r_val)
                                        corr_s_vals[(name_i, name_j)].append(r_s)

                            for name_i in names:
                                for name_j in names:
                                    if not _should_compute_pair(
                                        name_i,
                                        name_j,
                                        target_var_names=target_var_names,
                                        focus_source_var=focus_source_var,
                                    ):
                                        continue
                                    record_key = _corr_record_key(good_only, use_stpr, avg_by_pid, group_name, region, name_i, name_j)
                                    if record_key in existing_corr_keys:
                                        continue
                                    is_diag = (name_i == name_j)
                                    if is_diag:
                                        r_val, n_val = _mean_with_count(rel_vals[name_i])
                                        s_val, _ = _mean_with_count(rel_s_vals[name_i])
                                        corr_records.append({
                                            "good_only": good_only, "use_good_stpr": use_stpr, "avg_by_pid": avg_by_pid,
                                            "arousal_group": group_name, "region": region,
                                            "var1": name_i, "var2": name_j,
                                            "pearson_r": r_val, "pearson_n": n_val,
                                            "spearman_rho": s_val, "spearman_n": n_val,
                                        })
                                    else:
                                        r_val, n_val = _mean_with_count(corr_vals[(name_i, name_j)])
                                        s_val, _ = _mean_with_count(corr_s_vals[(name_i, name_j)])
                                        corr_records.append({
                                            "good_only": good_only, "use_good_stpr": use_stpr, "avg_by_pid": avg_by_pid,
                                            "arousal_group": group_name, "region": region,
                                            "var1": name_i, "var2": name_j,
                                            "pearson_r": r_val, "pearson_n": n_val,
                                            "spearman_rho": s_val, "spearman_n": n_val,
                                        })
                        else:
                            reliability = {}
                            reliability_n = {}
                            reliability_s = {}
                            reliability_s_n = {}
                            for spec in available_specs:
                                name = spec["name"]
                                if name not in names:
                                    continue
                                if not _should_compute_pair(
                                    name,
                                    name,
                                    target_var_names=target_var_names,
                                    focus_source_var=focus_source_var,
                                ):
                                    continue
                                df_var = var_tables_all.get(name)
                                if df_var is None or _is_firing_rate_spec(spec):
                                    reliability[name] = np.nan; reliability_n[name] = 0
                                    reliability_s[name] = np.nan; reliability_s_n[name] = 0
                                    continue
                                r_val, n_val = _pearsonr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                                s_val, sn_val = _spearmanr_with_n(df_var[spec["v1"]], df_var[spec["v2"]])
                                reliability[name] = r_val; reliability_n[name] = n_val
                                reliability_s[name] = s_val; reliability_s_n[name] = sn_val

                            mean_wide = region_lookup[["pid", "cluster_id"]].drop_duplicates()
                            for name in names:
                                df_var = var_tables_all.get(name)
                                if df_var is None:
                                    mean_wide[name] = np.nan
                                    continue
                                mean_wide = mean_wide.merge(df_var[["pid", "cluster_id", "mean"]], on=["pid", "cluster_id"], how="left").rename(columns={"mean": name})

                            for name_i in names:
                                for name_j in names:
                                    if not _should_compute_pair(
                                        name_i,
                                        name_j,
                                        target_var_names=target_var_names,
                                        focus_source_var=focus_source_var,
                                    ):
                                        continue
                                    record_key = _corr_record_key(good_only, use_stpr, avg_by_pid, group_name, region, name_i, name_j)
                                    if record_key in existing_corr_keys:
                                        continue
                                    is_diag = (name_i == name_j)
                                    if is_diag:
                                        r_val, n_val = reliability[name_i], reliability_n[name_i]
                                        s_val, sn_val = reliability_s[name_i], reliability_s_n[name_i]
                                    else:
                                        r_val, n_val = _pearsonr_with_n(mean_wide[name_i], mean_wide[name_j])
                                        s_val, sn_val = _spearmanr_with_n(mean_wide[name_i], mean_wide[name_j])
                                    
                                    corr_records.append({
                                        "good_only": good_only, "use_good_stpr": use_stpr, "avg_by_pid": avg_by_pid,
                                        "arousal_group": group_name, "region": region,
                                        "var1": name_i, "var2": name_j,
                                        "pearson_r": r_val, "pearson_n": n_val,
                                        "spearman_rho": s_val, "spearman_n": sn_val,
                                    })
                        region_pbar.update(1)
                region_pbar.close()
                pbar.update(1)

    pbar.close()
    print("Concatenating into DataFrames...")
    df_corr_new = pd.DataFrame(corr_records)
    if reuse_existing_side_tables and existing_bundle is not None:
        df_arousal_frac = existing_bundle.get("df_arousal", pd.DataFrame())
        df_region_summary = existing_bundle.get("df_region", pd.DataFrame())
        pid_summary = existing_bundle.get("df_pids", pid_summary)
    else:
        df_arousal_frac = pd.concat(arousal_frac_records, ignore_index=True) if arousal_frac_records else pd.DataFrame()
        df_region_summary = pd.concat(region_summary_records, ignore_index=True) if region_summary_records else pd.DataFrame()

    if append_missing_only and isinstance(existing_corr, pd.DataFrame):
        if df_corr_new.empty:
            df_corr = existing_corr.copy()
            print("No new correlation rows were generated.")
        else:
            df_corr = pd.concat([existing_corr, df_corr_new], ignore_index=True)
            dedup_cols = ["good_only", "use_good_stpr", "avg_by_pid", "arousal_group", "region", "var1", "var2"]
            df_corr = df_corr.drop_duplicates(subset=dedup_cols, keep="last")
            df_corr = df_corr.sort_values(dedup_cols).reset_index(drop=True)
            print(f"Added {len(df_corr_new):,} new correlation rows on top of {len(existing_corr):,} existing rows.")
    else:
        df_corr = df_corr_new

    if write_outputs:
        _write_region_cache_outputs(
            df_corr,
            df_arousal_frac,
            df_region_summary,
            pid_summary,
        )
        print("SUCCESS: Batch export complete.")
    else:
        print("Computed dashboard tables in memory (no parquet export).")

    return df_corr, df_arousal_frac, df_region_summary, pid_summary, label_min_text

if __name__ == "__main__":
    args = parse_args()
    precompute_all_dashboards(
        append_missing_only=bool(args.append_missing_only),
        focus_spont_strength_newvars=bool(args.focus_spont_strength_newvars),
    )
