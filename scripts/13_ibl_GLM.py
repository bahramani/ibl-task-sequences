# %% Imports
import importlib
from pathlib import Path
import pickle
import sys
import warnings
import time

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import nemos as nmo

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))

from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    load_session_data,
    load_pupil_data,
    build_cluster_id_map,
    get_cluster_labels_array,
    map_acronyms,
)
import utils.analysis as ana_utils

warnings.filterwarnings("ignore")


# %% User settings
MIN_LABEL_VALUE = 0.9
PID_LIST = [
    # "39883ded-f5a2-4f4f-a98e-fb138eb8433e",
    # "57c5856a-c7bd-4d0f-87c6-37005b1484aa",
    # "7332e6cf-9847-4aca-b2e3-d864989dd0fb",
    # "3eb6e6e0-8a57-49d6-b7c9-f39d5834e682",
    # "ce24bbe9-ae70-4659-9e9c-564d1a865de8",
    # "2e720cee-05cc-440e-a24b-13794b1ac01d",
    "f967a527-257f-404a-871d-b91575dca3b4",
]
USE_REGULARIZATION = True
REGULARIZER_STRENGTH = 0.1
BIN_SIZE_S = 0.001
# When BIN_SIZE_S == 0.001, post-fit GLM outputs are forced to binary 0/1 per bin.
# We keep the Poisson GLM fit itself unchanged, but convert the predicted Poisson
# mean lambda to a spike probability p = 1 - exp(-lambda) and then apply a fixed
# deterministic threshold (see _postprocess_predicted_count_output).
BINARY_1MS_SPIKE_PROB_THRESHOLD = 0.5
SPONT_HOLDOUT_RANDOM_SEED = 0
SPONT_TEST_BLOCK_COUNT = 1
GLM_REGIONS = None # ["SSp-ul"]  # Set to None to run all regions
SHOW_FIGURES = False


# %% Internal defaults
PLOTLY_RENDERER = None
ATLAS_MAPPING = "Beryl"
TRAIN_TEST_MODULUS = 10
TRAIN_TEST_TEST_INDEX = 9
TASK_EVENT_NAMES = ["stimOn_times", "firstMovement_times", "feedback_times"]
TASK_POST_EVENT_S = 1.0
ITI_SKIP_FIRST_LAST = True
MIN_TRAIN_BINS = 100
MIN_TEST_BINS = 20
MIN_WINDOWS_FOR_SPLIT = 3
OUTPUT_ROOT = BASE_PATH / "results" / "13_ibl_glm"
OUTPUT_DIR = OUTPUT_ROOT
DASHBOARD_CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"

COUPLING_CONFIG = {
    "ATLAS_MAPPING": ATLAS_MAPPING,
    "CALC_LABEL_MIN": MIN_LABEL_VALUE,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_LOW_PASS_ORDER": 3,
    "STPR_POP_USE_GOOD_UNITS": False,
    "TASK_POST_EVENT_S": TASK_POST_EVENT_S,
    "ITI_SKIP_FIRST_LAST": ITI_SKIP_FIRST_LAST,
}

WHISK_CONFIG = {
    "WH_BIN_SIGNAL": False,
    "WH_BIN_S": 0.3,
    "WH_NORM_PCTL": 1.0,
    "WH_NORM_TOP_PCTL": 100.0,
    "WH_BIN_REDUCE": "mean",
    "WH_NORMALIZE_AFTER_BIN": True,
    "WH_TARGET_MEAN": 0.06,
    "WH_START_THR": 0.10,
    "WH_END_THR": 0.04,
    "WH_END_QUIET_WINDOW_S": 0.5,
    "WH_MERGE_GAP_S": 0.3,
    "WH_BRIEF_RANGE_S": (0.25, 2.0),
    "WH_LONG_MIN_S": 2.0,
    "WH_LOCO_LOOKAHEAD_S": 6.0,
    "WH_LOCO_SPEED_THR_CM_S": 1.0,
    "WH_WHEEL_RADIUS_CM": 3.1,
    "WH_DELAY_WINDOW": (0.0, 0.4),
    "WH_SPLIT_MODE": "odd_even",
}


# %% Plot helpers
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
    if not SHOW_FIGURES:
        return
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


def save_fig(fig, file_name):
    path = OUTPUT_DIR / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_dataframe(df, stem):
    csv_path = OUTPUT_DIR / (stem + ".csv")
    parquet_path = OUTPUT_DIR / (stem + ".parquet")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"Warning: could not save parquet for {stem}: {exc}")
    return csv_path


def _color_to_rgb(color):
    if not isinstance(color, str):
        return None
    val = color.strip()
    if val.startswith("rgb(") and val.endswith(")"):
        parts = [part.strip() for part in val[4:-1].split(",")]
        if len(parts) != 3:
            return None
        try:
            return tuple(int(float(part)) for part in parts)
        except Exception:
            return None
    if val.startswith("#") and len(val) == 7:
        try:
            return tuple(int(val[idx : idx + 2], 16) for idx in (1, 3, 5))
        except Exception:
            return None
    return None


def _rgb_to_css(rgb_triplet):
    if rgb_triplet is None or len(rgb_triplet) != 3:
        return None
    r, g, b = rgb_triplet
    return f"rgb({int(r)},{int(g)},{int(b)})"


def _with_alpha(color, alpha):
    rgb = _color_to_rgb(color)
    if rgb is None:
        return color
    r, g, b = rgb
    return f"rgba({r},{g},{b},{float(alpha):.3f})"


def _mix_with_white(color, mix_frac):
    rgb = _color_to_rgb(color)
    if rgb is None:
        return color
    frac = float(np.clip(mix_frac, 0.0, 1.0))
    mixed = tuple(int(round((1.0 - frac) * channel + frac * 255.0)) for channel in rgb)
    return _rgb_to_css(mixed)


def _get_allen_color_lookup():
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
            return _rgb_to_css((int(hv[0:2], 16), int(hv[2:4], 16), int(hv[4:6], 16)))
        except Exception:
            return None

    try:
        iblatlas_pkg = importlib.import_module("iblatlas")
        csv_path = Path(iblatlas_pkg.__file__).resolve().parent / "allen_structure_tree.csv"
        if not csv_path.exists():
            return {}
        df_allen = pd.read_csv(csv_path, dtype={"color_hex_triplet": "string"})
    except Exception:
        return {}

    if "acronym" not in df_allen.columns or "color_hex_triplet" not in df_allen.columns:
        return {}

    colors = {}
    for _, row in df_allen[["acronym", "color_hex_triplet"]].dropna(subset=["acronym"]).iterrows():
        rgb = _to_rgb(row.get("color_hex_triplet", ""))
        if rgb is not None:
            colors[str(row["acronym"])] = rgb
    return colors


def _build_region_colors(acronyms):
    unique_regions = pd.Series(acronyms).astype(str).dropna().unique().tolist()
    allen_lookup = _get_allen_color_lookup()
    colors = {region: allen_lookup.get(region) for region in unique_regions if allen_lookup.get(region)}
    if colors:
        return colors
    fallback_palette = [
        "rgb(31,119,180)",
        "rgb(255,127,14)",
        "rgb(44,160,44)",
        "rgb(214,39,40)",
        "rgb(148,103,189)",
        "rgb(140,86,75)",
    ]
    return {region: fallback_palette[idx % len(fallback_palette)] for idx, region in enumerate(unique_regions)}


# %% General helpers
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


def _is_pid_resolution_error(exc):
    message = str(exc)
    snippets = (
        "Converting probe IDs requires remote connection",
        "Cannot infer session id and probe name from pid",
        "pid2eid",
    )
    return any(snippet in message for snippet in snippets)


def _load_session_data_with_retry(
    pid,
    ibl_cache,
    ba,
    preferred_mode="local",
    allow_remote=True,
    **load_kwargs,
):
    one, one_mode = _init_one_with_fallback(
        ibl_cache,
        preferred_mode=preferred_mode,
        allow_remote=allow_remote,
    )
    try:
        ssl, spikes, clusters, sl = load_session_data(pid, one, ba=ba, **load_kwargs)
        return one, one_mode, ssl, spikes, clusters, sl
    except Exception as exc:
        if one_mode == "remote" or not allow_remote or not _is_pid_resolution_error(exc):
            raise
        print(
            "Local ONE client could not resolve PID metadata. "
            "Retrying session load with remote ONE."
        )
        one_remote = init_one(ibl_cache, mode="remote")
        ssl, spikes, clusters, sl = load_session_data(pid, one_remote, ba=ba, **load_kwargs)
        return one_remote, "remote", ssl, spikes, clusters, sl


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


def _trials_to_dataframe(trials):
    if trials is None:
        return pd.DataFrame()
    if isinstance(trials, pd.DataFrame):
        return trials.copy()
    try:
        return pd.DataFrame(trials)
    except Exception:
        rows = {}
        if hasattr(trials, "keys"):
            for key in trials.keys():
                rows[key] = np.asarray(trials[key])
            return pd.DataFrame(rows)
    return pd.DataFrame()


def _safe_cache_value_equal(a, b, atol=1e-12):
    if a is None and b is None:
        return True
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        try:
            arr_a = np.asarray(a, dtype=float)
            arr_b = np.asarray(b, dtype=float)
            return arr_a.shape == arr_b.shape and np.allclose(arr_a, arr_b, atol=atol, equal_nan=True)
        except Exception:
            return a == b
    if isinstance(a, (float, int, np.floating, np.integer)) or isinstance(b, (float, int, np.floating, np.integer)):
        try:
            return bool(np.isclose(float(a), float(b), atol=atol, equal_nan=True))
        except Exception:
            return a == b
    return a == b


def _load_dashboard_cache(pid, cache_dir):
    cache_path = Path(cache_dir) / f"{pid}.pkl"
    if not cache_path.exists():
        return None, cache_path
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f), cache_path
    except Exception as exc:
        print(f"Warning: could not load dashboard cache {cache_path}: {exc}")
        return None, cache_path


def _cluster_id_set_matches(df, cluster_ids):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "cluster_id" not in df.columns:
        return False
    cache_ids = set(pd.to_numeric(df["cluster_id"], errors="coerce").dropna().astype(int).tolist())
    live_ids = set(np.asarray(cluster_ids, dtype=int).tolist())
    return cache_ids == live_ids


def _collect_cached_coupling_tables(cache, cluster_ids, spont_intervals, coupling_config, min_label_value):
    cached_tables = {}
    cache_notes = []
    if not isinstance(cache, dict):
        return cached_tables, ["dashboard cache missing or unreadable"]

    config_calc = cache.get("config_calc", {})
    meta = cache.get("meta", {})
    config_pairs = {
        "ATLAS_MAPPING": coupling_config.get("ATLAS_MAPPING"),
        "CALC_LABEL_MIN": min_label_value,
        "STPR_BIN_SIZE": coupling_config.get("STPR_BIN_SIZE"),
        "STPR_WINDOW_MS": coupling_config.get("STPR_WINDOW_MS"),
        "STPR_LOW_PASS_HZ": coupling_config.get("STPR_LOW_PASS_HZ"),
        "STPR_LOW_PASS_ORDER": coupling_config.get("STPR_LOW_PASS_ORDER"),
        "STPR_POP_USE_GOOD_UNITS": coupling_config.get("STPR_POP_USE_GOOD_UNITS"),
        "TASK_POST_EVENT_S": coupling_config.get("TASK_POST_EVENT_S"),
        "ITI_SKIP_FIRST_LAST": coupling_config.get("ITI_SKIP_FIRST_LAST"),
    }
    mismatches = []
    for key, expected in config_pairs.items():
        cached_val = config_calc.get(key, None)
        if not _safe_cache_value_equal(cached_val, expected):
            mismatches.append(f"{key}: cache={cached_val} current={expected}")

    spont_expected = None
    spont_arr = _coerce_interval_array(spont_intervals)
    if spont_arr.shape[0] >= 1:
        spont_expected = tuple(np.asarray(spont_arr[0], dtype=float).tolist())
    spont_cached = meta.get("spont_interval", None)
    if spont_expected is not None and not _safe_cache_value_equal(spont_cached, spont_expected, atol=1e-6):
        mismatches.append(f"spont_interval: cache={spont_cached} current={spont_expected}")

    if mismatches:
        return cached_tables, mismatches

    table_map = {
        "spont": "df_coupling",
        "task": "df_coupling_task",
        "iti": "df_coupling_iti",
    }
    for label, cache_key in table_map.items():
        df = cache.get(cache_key, None)
        if isinstance(df, pd.DataFrame) and not df.empty and _cluster_id_set_matches(df, cluster_ids):
            cached_tables[label] = df.copy()
            cache_notes.append(f"{label}=loaded")
        else:
            cache_notes.append(f"{label}=recompute")
    return cached_tables, cache_notes


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
        mask = labels.astype(float) >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return cluster_ids[mask]


def _coerce_interval_array(intervals):
    if intervals is None:
        return np.empty((0, 2), dtype=float)
    if isinstance(intervals, pd.DataFrame):
        if {"start", "end"}.issubset(intervals.columns):
            arr = intervals[["start", "end"]].to_numpy(dtype=float)
        else:
            return np.empty((0, 2), dtype=float)
    else:
        arr = np.asarray(intervals, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, 2)
    if arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 1] > arr[:, 0])
    arr = arr[mask]
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr[np.argsort(arr[:, 0])]


def _intervals_to_mask(bin_centers, intervals):
    arr = _coerce_interval_array(intervals)
    mask = np.zeros(bin_centers.shape[0], dtype=bool)
    if arr.size == 0:
        return mask
    for start, end in arr:
        mask |= (bin_centers >= float(start)) & (bin_centers < float(end))
    return mask


def _make_holdout_blocks(start, end, modulus=10, test_index=9, n_test_blocks=1, rng=None):
    if not (np.isfinite(start) and np.isfinite(end) and end > start):
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.empty((0,), dtype=int)
    modulus = max(int(modulus), 1)
    n_test_blocks = int(np.clip(int(n_test_blocks), 1, modulus))
    edges = np.linspace(float(start), float(end), modulus + 1)
    if rng is None:
        selected = np.array([int(test_index) % modulus], dtype=int)
    else:
        selected = np.sort(np.asarray(rng.choice(modulus, size=n_test_blocks, replace=False), dtype=int))
    selected_set = set(selected.tolist())
    train_blocks = []
    test_blocks = []
    for idx in range(modulus):
        pair = [edges[idx], edges[idx + 1]]
        if idx in selected_set:
            test_blocks.append(pair)
        else:
            train_blocks.append(pair)
    return np.asarray(train_blocks, dtype=float), np.asarray(test_blocks, dtype=float), selected


def _split_windows_by_holdout(windows_df, modulus=10, test_index=9):
    if windows_df is None or windows_df.empty:
        return windows_df.iloc[0:0].copy(), windows_df.iloc[0:0].copy()
    windows_df = windows_df.sort_values("start").reset_index(drop=True).copy()
    if len(windows_df) < MIN_WINDOWS_FOR_SPLIT:
        return windows_df.iloc[0:0].copy(), windows_df.iloc[0:0].copy()
    is_test = (np.arange(len(windows_df)) % modulus) == test_index
    return windows_df.loc[~is_test].reset_index(drop=True), windows_df.loc[is_test].reset_index(drop=True)


def _build_global_bin_grid(session_end_s, bin_size_s):
    end_val = float(session_end_s)
    if not np.isfinite(end_val) or end_val <= 0:
        raise ValueError("session_end_s must be positive and finite.")
    edges = np.arange(0.0, end_val + bin_size_s, bin_size_s, dtype=float)
    if edges[-1] < end_val:
        edges = np.append(edges, end_val + bin_size_s)
    centers = edges[:-1] + 0.5 * float(bin_size_s)
    return edges, centers


def _get_session_end(spikes, trials_df, sl, pupil_features, pupil_times, spont_intervals):
    candidates = []
    if spikes is not None and "times" in spikes:
        spike_times = np.asarray(spikes["times"], dtype=float)
        spike_times = spike_times[np.isfinite(spike_times)]
        if spike_times.size:
            candidates.append(float(np.nanmax(spike_times)))
    if not trials_df.empty:
        for col in ("intervals_1", "intervals_bpod_1", "feedback_times", "response_times", "firstMovement_times", "stimOn_times"):
            if col in trials_df.columns:
                vals = np.asarray(trials_df[col], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    candidates.append(float(np.nanmax(vals)))
    wheel = getattr(sl, "wheel", None)
    if wheel is not None:
        try:
            wheel_times = np.asarray(wheel["times"], dtype=float)
            wheel_times = wheel_times[np.isfinite(wheel_times)]
            if wheel_times.size:
                candidates.append(float(np.nanmax(wheel_times)))
        except Exception:
            pass
    if pupil_times is not None:
        pupil_arr = np.asarray(pupil_times, dtype=float)
        pupil_arr = pupil_arr[np.isfinite(pupil_arr)]
        if pupil_arr.size:
            candidates.append(float(np.nanmax(pupil_arr)))
    motion_energy = getattr(sl, "motion_energy", None)
    if motion_energy is not None:
        df_me = ana_utils.extract_motion_energy_trace(motion_energy)
        if not df_me.empty:
            candidates.append(float(df_me["times"].max()))
    spont_arr = _coerce_interval_array(spont_intervals)
    if spont_arr.size:
        candidates.append(float(np.nanmax(spont_arr[:, 1])))
    if not candidates:
        raise RuntimeError("Could not determine session end.")
    return max(candidates)


def _accumulate_event_series(event_times, amplitudes, bin_edges):
    series = np.zeros(bin_edges.shape[0] - 1, dtype=np.float32)
    times = np.asarray(event_times, dtype=float).reshape(-1)
    amps = np.asarray(amplitudes, dtype=float).reshape(-1)
    if times.size == 0:
        return series
    if amps.size == 1 and times.size > 1:
        amps = np.full(times.shape, float(amps[0]), dtype=float)
    if amps.shape[0] != times.shape[0]:
        raise ValueError("Event times and amplitudes must match.")
    valid = np.isfinite(times) & np.isfinite(amps)
    times = times[valid]
    amps = amps[valid]
    if times.size == 0:
        return series
    idx = np.searchsorted(bin_edges, times, side="right") - 1
    keep = (idx >= 0) & (idx < series.shape[0])
    idx = idx[keep]
    amps = amps[keep]
    if idx.size == 0:
        return series
    np.add.at(series, idx, amps.astype(np.float32))
    return series


def _basis_convolve(series, n_basis, window_s, bin_size_s, predictor_causality):
    window_bins = int(round(float(window_s) / float(bin_size_s)))
    window_bins = max(window_bins, 2)
    basis = nmo.basis.RaisedCosineLinearConv(
        n_basis_funcs=int(n_basis),
        window_size=window_bins,
        conv_kwargs={"predictor_causality": predictor_causality, "shift": False},
    )
    features = np.asarray(basis.compute_features(np.asarray(series, dtype=float)), dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _interpolate_to_bins(source_times, source_values, target_times):
    source_times = np.asarray(source_times, dtype=float).reshape(-1)
    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    target_times = np.asarray(target_times, dtype=float).reshape(-1)
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    if int(valid.sum()) < 2:
        return np.zeros(target_times.shape[0], dtype=np.float32)
    source_times = source_times[valid]
    source_values = source_values[valid]
    order = np.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]
    uniq_t, uniq_idx = np.unique(source_times, return_index=True)
    source_values = source_values[uniq_idx]
    interp = np.interp(target_times, uniq_t, source_values, left=0.0, right=0.0)
    return np.asarray(interp, dtype=np.float32)


def _get_wheel_speed_regressor(wheel, bin_centers):
    wheel_t, wheel_speed = ana_utils.extract_wheel_speed_cm_s(wheel, wheel_radius_cm=3.1)
    return _interpolate_to_bins(wheel_t, wheel_speed, bin_centers)


def _get_whisk_regressor(sl, bin_centers):
    df_motion = ana_utils.extract_motion_energy_trace(
        getattr(sl, "motion_energy", None),
        max_interp_gap_frames=3,
        ensure_positive_motion=True,
    )
    df_whisk = ana_utils.build_whisk_trace(df_motion, dict(WHISK_CONFIG))
    whisk_values = (
        _interpolate_to_bins(df_whisk["bin_center_s"].to_numpy(dtype=float), df_whisk["wh_norm"].to_numpy(dtype=float), bin_centers)
        if not df_whisk.empty
        else np.zeros(bin_centers.shape[0], dtype=np.float32)
    )
    return df_motion, df_whisk, whisk_values


def _get_pupil_regressor(pupil_features, pupil_times, bin_centers):
    if pupil_features is None or pupil_times is None:
        return np.zeros(bin_centers.shape[0], dtype=np.float32)
    if not isinstance(pupil_features, pd.DataFrame):
        return np.zeros(bin_centers.shape[0], dtype=np.float32)
    if "pupilDiameter_smooth" not in pupil_features.columns:
        return np.zeros(bin_centers.shape[0], dtype=np.float32)
    return _interpolate_to_bins(np.asarray(pupil_times, dtype=float), pupil_features["pupilDiameter_smooth"].to_numpy(dtype=float), bin_centers)

def _trial_contrasts(trials_df):
    left = np.abs(np.asarray(trials_df["contrastLeft"], dtype=float))
    right = np.abs(np.asarray(trials_df["contrastRight"], dtype=float))
    out = np.nanmax(np.vstack([left, right]), axis=0)
    return np.where(np.isfinite(out), out, 0.0)


def _build_task_features(trials_df, bin_edges, bin_centers, wheel_series):
    block_entries = []
    if trials_df.empty:
        return block_entries

    valid_stim = np.isfinite(trials_df["stimOn_times"].to_numpy(dtype=float))
    contrast_left = np.nan_to_num(trials_df["contrastLeft"].to_numpy(dtype=float), nan=0.0)
    contrast_right = np.nan_to_num(trials_df["contrastRight"].to_numpy(dtype=float), nan=0.0)

    stim_left_mask = valid_stim & (contrast_left > 0)
    stim_right_mask = valid_stim & (contrast_right > 0)
    stim_left = _accumulate_event_series(
        trials_df.loc[stim_left_mask, "stimOn_times"].to_numpy(dtype=float),
        np.tanh(5.0 * contrast_left[stim_left_mask]) / np.tanh(5.0),
        bin_edges,
    )
    stim_right = _accumulate_event_series(
        trials_df.loc[stim_right_mask, "stimOn_times"].to_numpy(dtype=float),
        np.tanh(5.0 * contrast_right[stim_right_mask]) / np.tanh(5.0),
        bin_edges,
    )
    block_entries.append({"name": "Stim Onset L", "category": "Task", "data": _basis_convolve(stim_left, 5, 0.400, BIN_SIZE_S, "causal")})
    block_entries.append({"name": "Stim Onset R", "category": "Task", "data": _basis_convolve(stim_right, 5, 0.400, BIN_SIZE_S, "causal")})

    valid_move = np.isfinite(trials_df["firstMovement_times"].to_numpy(dtype=float))
    choice_vals = np.asarray(trials_df["choice"], dtype=float)
    move_left = _accumulate_event_series(
        trials_df.loc[valid_move & (choice_vals == 1), "firstMovement_times"].to_numpy(dtype=float),
        np.ones(int(np.sum(valid_move & (choice_vals == 1))), dtype=float),
        bin_edges,
    )
    move_right = _accumulate_event_series(
        trials_df.loc[valid_move & (choice_vals == -1), "firstMovement_times"].to_numpy(dtype=float),
        np.ones(int(np.sum(valid_move & (choice_vals == -1))), dtype=float),
        bin_edges,
    )
    move_init = _accumulate_event_series(
        trials_df.loc[valid_move, "firstMovement_times"].to_numpy(dtype=float),
        np.ones(int(np.sum(valid_move)), dtype=float),
        bin_edges,
    )
    block_entries.append({"name": "First Movement L", "category": "Task", "data": _basis_convolve(move_left, 3, 0.200, BIN_SIZE_S, "anti-causal")})
    block_entries.append({"name": "First Movement R", "category": "Task", "data": _basis_convolve(move_right, 3, 0.200, BIN_SIZE_S, "anti-causal")})

    valid_feedback = np.isfinite(trials_df["feedback_times"].to_numpy(dtype=float))
    feedback_type = np.asarray(trials_df["feedbackType"], dtype=float)
    fb_correct = _accumulate_event_series(
        trials_df.loc[valid_feedback & (feedback_type == 1), "feedback_times"].to_numpy(dtype=float),
        np.ones(int(np.sum(valid_feedback & (feedback_type == 1))), dtype=float),
        bin_edges,
    )
    fb_incorrect = _accumulate_event_series(
        trials_df.loc[valid_feedback & (feedback_type == -1), "feedback_times"].to_numpy(dtype=float),
        np.ones(int(np.sum(valid_feedback & (feedback_type == -1))), dtype=float),
        bin_edges,
    )
    block_entries.append({"name": "Feedback Correct", "category": "Task", "data": _basis_convolve(fb_correct, 5, 0.400, BIN_SIZE_S, "causal")})
    block_entries.append({"name": "Feedback Incorrect", "category": "Task", "data": _basis_convolve(fb_incorrect, 5, 0.400, BIN_SIZE_S, "causal")})
    block_entries.append({"name": "Wheel Speed", "category": "Task", "data": _basis_convolve(wheel_series, 3, 0.300, BIN_SIZE_S, "anti-causal")})

    prob_series = np.zeros(bin_centers.shape[0], dtype=np.float32)
    if "probabilityLeft" in trials_df.columns:
        if {"intervals_0", "intervals_1"}.issubset(trials_df.columns):
            starts = trials_df["intervals_0"].to_numpy(dtype=float)
            ends = trials_df["intervals_1"].to_numpy(dtype=float)
        else:
            starts = trials_df["stimOn_times"].to_numpy(dtype=float)
            ends = ana_utils.compute_trial_end_times(trials_df, TASK_EVENT_NAMES, post_event_s=TASK_POST_EVENT_S)
        prob_vals = trials_df["probabilityLeft"].to_numpy(dtype=float)
        for start, end, prob in zip(starts, ends, prob_vals):
            if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(prob) and end > start):
                continue
            mask = (bin_centers >= float(start)) & (bin_centers < float(end))
            prob_series[mask] = float(prob)
    block_entries.append({"name": "Block Probability", "category": "Task", "data": prob_series[:, None]})
    block_entries.append({"name": "Movement Initiation", "category": "Task", "data": move_init[:, None].astype(np.float32)})
    return block_entries


def _build_uninstructed_blocks(whisk_series, pupil_series):
    return [
        {"name": "Whisking", "category": "Uninstructed", "data": whisk_series[:, None].astype(np.float32)},
        {"name": "Pupil Diameter", "category": "Uninstructed", "data": pupil_series[:, None].astype(np.float32)},
    ]


def _assemble_design_matrix(block_entries, expected_rows):
    arrays = []
    block_specs = []
    feature_names = []
    cursor = 0
    for entry in block_entries:
        data = np.asarray(entry["data"], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[0] != expected_rows:
            raise ValueError(f"Block '{entry['name']}' has wrong row count: {data.shape[0]} != {expected_rows}")
        start = cursor
        stop = cursor + data.shape[1]
        block_specs.append({"name": entry["name"], "category": entry["category"], "start": start, "stop": stop, "width": data.shape[1]})
        for idx in range(data.shape[1]):
            feature_names.append(entry["name"] if data.shape[1] == 1 else f"{entry['name']} [{idx + 1}]")
        arrays.append(data)
        cursor = stop
    X = np.concatenate(arrays, axis=1).astype(np.float32) if arrays else np.empty((expected_rows, 0), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, block_specs, feature_names


def _build_region_count_matrix(spikes, region_cluster_ids, bin_edges):
    region_cluster_ids = np.asarray(region_cluster_ids, dtype=int)
    if region_cluster_ids.size == 0:
        return np.empty((bin_edges.shape[0] - 1, 0), dtype=np.float32), {}
    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=int)
    counts = np.zeros((bin_edges.shape[0] - 1, region_cluster_ids.shape[0]), dtype=np.float32)
    cid_to_col = {}
    for col_idx, cid in enumerate(region_cluster_ids):
        cid_to_col[int(cid)] = col_idx
        unit_times = spike_times[spike_clusters == int(cid)]
        if unit_times.size == 0:
            continue
        hist, _ = np.histogram(unit_times, bins=bin_edges)
        counts[:, col_idx] = hist.astype(np.float32)
    return counts, cid_to_col


def _extract_training_coupling_params(df_coupling_train):
    params = {}
    if df_coupling_train is None or df_coupling_train.empty:
        return params
    for _, row in df_coupling_train.iterrows():
        cid = int(row["cluster_id"])
        params[cid] = {
            "region": row.get("region", "NA"),
            "strength": float(row.get("coupling_strength", np.nan)),
            "delay_ms": float(row.get("coupling_delay_ms", np.nan)),
        }
    return params


def _build_internal_sequence_block(
    neuron_id,
    region_cluster_ids,
    region_counts,
    region_cid_to_col,
    coupling_params,
    bin_centers,
    context_mask,
    delay_ms_override=None,
):
    if int(neuron_id) not in coupling_params:
        return np.zeros(bin_centers.shape[0], dtype=np.float32), False
    params = coupling_params[int(neuron_id)]
    strength = float(params.get("strength", np.nan))
    delay_ms = float(delay_ms_override) if delay_ms_override is not None else float(params.get("delay_ms", np.nan))
    if not np.isfinite(strength) or region_counts.shape[1] <= 1:
        return np.zeros(bin_centers.shape[0], dtype=np.float32), False
    if delay_ms_override is None and not np.isfinite(delay_ms):
        return np.zeros(bin_centers.shape[0], dtype=np.float32), False
    region_cluster_ids = np.asarray(region_cluster_ids, dtype=int)
    region_sum = region_counts.sum(axis=1, dtype=np.float64)
    self_col = region_cid_to_col.get(int(neuron_id), None)
    if self_col is None:
        return np.zeros(bin_centers.shape[0], dtype=np.float32), False
    pop_counts = region_sum - region_counts[:, self_col]
    n_other = max(int(region_cluster_ids.shape[0] - 1), 1)
    pop_rate = np.asarray(pop_counts / (float(n_other) * float(BIN_SIZE_S)), dtype=np.float32)
    pop_rate[~context_mask] = 0.0
    delay_s = float(delay_ms) / 1000.0
    shifted = np.interp(bin_centers - delay_s, bin_centers, pop_rate, left=0.0, right=0.0)
    return np.asarray(strength * shifted, dtype=np.float32), True


class _SafeNemosGLM:
    def __init__(self, model):
        self._model = model

    def _default_init_params(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
        mean_rate = float(np.nanmean(y_arr)) if y_arr.size else 0.0
        if not np.isfinite(mean_rate) or mean_rate <= 0:
            mean_rate = 1e-8
        init_coef = np.zeros(n_features, dtype=np.float32)
        init_intercept = np.array([np.log(mean_rate)], dtype=np.float32)
        return init_coef, init_intercept

    def fit(self, X, y, init_params=None):
        if init_params is None:
            init_params = self._default_init_params(X, y)
        try:
            fitted = self._model.fit(X, y, init_params=init_params)
        except (ValueError, TypeError) as exc:
            # Some NeMoS versions validate default initialization even when
            # init_params are provided. Re-raise so the caller's try/except
            # can record the failure and skip this neuron gracefully.
            raise ValueError(
                f"NeMoS fit failed (init guard): {type(exc).__name__}: {exc}"
            ) from exc
        if fitted is not None:
            self._model = fitted
        return self

    def score(self, X, y, score_type="pseudo-r2-Cohen"):
        return self._model.score(X, y, score_type=score_type)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _build_model():
    if USE_REGULARIZATION:
        model = nmo.glm.GLM(observation_model="Poisson", regularizer="Ridge", regularizer_strength=REGULARIZER_STRENGTH)
    else:
        model = nmo.glm.GLM(observation_model="Poisson")
    return _SafeNemosGLM(model)


def _safe_pearson_corr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _postprocess_predicted_count_output(pred_mean_count):
    pred_mean_count = np.asarray(pred_mean_count, dtype=float).reshape(-1)
    pred_mean_count = np.nan_to_num(pred_mean_count, nan=0.0, posinf=0.0, neginf=0.0)
    pred_mean_count = np.clip(pred_mean_count, 0.0, None)
    if not np.isclose(float(BIN_SIZE_S), 0.001, atol=1e-12):
        return pred_mean_count
    # For 1 ms bins, a neuron cannot emit more than one spike in the bin.
    # We therefore convert the Poisson mean lambda to the probability of at
    # least one spike, p = 1 - exp(-lambda), and apply a deterministic
    # threshold so the saved/output prediction is exactly 0 or 1.
    spike_prob = 1.0 - np.exp(-pred_mean_count)
    return (spike_prob >= float(BINARY_1MS_SPIKE_PROB_THRESHOLD)).astype(float)


def _fit_and_score_subset(X, y, train_mask, test_mask):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the same number of rows.")
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    result = {
        "valid": False,
        "score_cohen": np.nan,
        "score_mcfadden": np.nan,
        "test_rate_count_corr": np.nan,
        "test_pred_mean_count": None,
        "n_train_bins": int(train_idx.size),
        "n_test_bins": int(test_idx.size),
        "n_features_used": 0,
        "train_spike_count": np.nan,
        "test_spike_count": np.nan,
        "status": "not_run",
    }
    if train_idx.size < MIN_TRAIN_BINS or test_idx.size < MIN_TEST_BINS:
        result["status"] = "insufficient_bins"
        return result
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    result["train_spike_count"] = float(np.nansum(y_train))
    result["test_spike_count"] = float(np.nansum(y_test))
    if not (np.all(np.isfinite(y_train)) and np.all(np.isfinite(y_test))):
        result["status"] = "non_finite_target"
        return result
    if result["train_spike_count"] <= 0:
        result["status"] = "no_train_spikes"
        return result
    if result["test_spike_count"] <= 0:
        result["status"] = "no_test_spikes"
        return result
    means = np.nanmean(X_train, axis=0, keepdims=True)
    stds = np.nanstd(X_train, axis=0, keepdims=True)
    keep_cols = np.isfinite(stds).reshape(-1) & (stds.reshape(-1) > 0)
    if int(np.sum(keep_cols)) == 0:
        result["status"] = "no_variable_features"
        return result
    X_train = X_train[:, keep_cols]
    X_test = X_test[:, keep_cols]
    means = means[:, keep_cols]
    stds = stds[:, keep_cols]
    X_train = np.nan_to_num((X_train - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num((X_test - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    model = _build_model()
    # NeMoS initializes the intercept from the mean firing rate. We provide it explicitly
    # so rare/low-rate neurons do not fail during internal initialization.
    mean_rate = float(np.nanmean(y_train))
    mean_rate = max(mean_rate, 1e-8)
    init_coef = np.zeros(X_train.shape[1], dtype=np.float32)
    init_intercept = np.array([np.log(mean_rate)], dtype=np.float32)
    try:
        model.fit(X_train, y_train, init_params=(init_coef, init_intercept))
        score_cohen = float(model.score(X_test, y_test, score_type="pseudo-r2-Cohen"))
        score_mcfadden = float(model.score(X_test, y_test, score_type="pseudo-r2-McFadden"))
    except Exception as exc:
        result["status"] = f"fit_error: {type(exc).__name__}: {exc}"
        return result
    pred_mean_count = None
    test_rate_count_corr = np.nan
    try:
        pred_mean_count = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        pred_mean_count = _postprocess_predicted_count_output(pred_mean_count)
        pred_rate_hz = pred_mean_count / max(float(BIN_SIZE_S), np.finfo(float).eps)
        test_rate_count_corr = _safe_pearson_corr(pred_rate_hz, y_test)
    except Exception:
        pred_mean_count = None
        test_rate_count_corr = np.nan
    result.update(
        {
            "valid": np.isfinite(score_cohen) and np.isfinite(score_mcfadden),
            "score_cohen": score_cohen,
            "score_mcfadden": score_mcfadden,
            "test_rate_count_corr": test_rate_count_corr,
            "test_pred_mean_count": pred_mean_count,
            "n_features_used": int(np.sum(keep_cols)),
            "status": "ok" if np.isfinite(score_cohen) and np.isfinite(score_mcfadden) else "non_finite_score",
        }
    )
    return result


def _safe_fit_and_score(X, y, train_mask, test_mask):
    """Wrapper around _fit_and_score_subset that never raises.

    Catches any exception that escapes the inner try/except (e.g. from a
    stale Jupyter kernel or a JAX deferred-compilation error) and returns
    a safe invalid result dict instead of crashing the neuron loop.
    """
    try:
        return _fit_and_score_subset(X, y, train_mask, test_mask)
    except Exception as exc:
        return {
            "valid": False,
            "score_cohen": np.nan,
            "score_mcfadden": np.nan,
            "test_rate_count_corr": np.nan,
            "test_pred_mean_count": None,
            "n_train_bins": int(np.sum(train_mask)),
            "n_test_bins": int(np.sum(test_mask)),
            "n_features_used": 0,
            "train_spike_count": np.nan,
            "test_spike_count": np.nan,
            "status": f"outer_catch: {type(exc).__name__}: {exc}",
        }


def _evaluate_model_family(X, y, train_mask, test_mask, block_specs):
    out = {
        "full": _safe_fit_and_score(X, y, train_mask, test_mask),
        "block_delta": {},
        "block_score_without": {},
        "category_delta": {},
        "category_score_without": {},
        "category_score_alone": {},
    }
    if not out["full"]["valid"]:
        return out
    n_features = X.shape[1]
    categories = []
    for block in block_specs:
        if block["category"] not in categories:
            categories.append(block["category"])
    for block in block_specs:
        keep = np.ones(n_features, dtype=bool)
        keep[block["start"] : block["stop"]] = False
        reduced = _safe_fit_and_score(X[:, keep], y, train_mask, test_mask)
        out["block_score_without"][block["name"]] = reduced["score_cohen"]
        out["block_delta"][block["name"]] = out["full"]["score_cohen"] - reduced["score_cohen"] if reduced["valid"] else np.nan
    for category in categories:
        keep = np.ones(n_features, dtype=bool)
        alone = np.zeros(n_features, dtype=bool)
        for block in block_specs:
            block_slice = slice(block["start"], block["stop"])
            if block["category"] == category:
                keep[block_slice] = False
                alone[block_slice] = True
        reduced = _safe_fit_and_score(X[:, keep], y, train_mask, test_mask)
        alone_fit = _safe_fit_and_score(X[:, alone], y, train_mask, test_mask)
        out["category_score_without"][category] = reduced["score_cohen"]
        out["category_score_alone"][category] = alone_fit["score_cohen"]
        out["category_delta"][category] = out["full"]["score_cohen"] - reduced["score_cohen"] if reduced["valid"] else np.nan
    return out

def _compute_region_reliability(df_coupling, context_label, suffix_a, suffix_b):
    rows = []
    if df_coupling is None or df_coupling.empty:
        return pd.DataFrame(columns=["context", "region", "n_delay", "r_delay", "n_strength", "r_strength"])
    delay_a = f"coupling_delay_ms_{suffix_a}"
    delay_b = f"coupling_delay_ms_{suffix_b}"
    strength_a = f"coupling_strength_{suffix_a}"
    strength_b = f"coupling_strength_{suffix_b}"
    for region, grp in df_coupling.groupby("region"):
        delay_valid = grp[[delay_a, delay_b]].dropna()
        strength_valid = grp[[strength_a, strength_b]].dropna()
        rows.append(
            {
                "context": context_label,
                "region": region,
                "n_delay": int(len(delay_valid)),
                "r_delay": delay_valid[delay_a].corr(delay_valid[delay_b]) if len(delay_valid) >= 2 else np.nan,
                "n_strength": int(len(strength_valid)),
                "r_strength": strength_valid[strength_a].corr(strength_valid[strength_b]) if len(strength_valid) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


def _print_reliability_table(df_rel, label):
    print(f"\n{label} reliability by region")
    if df_rel.empty:
        print("No reliability rows available.")
        return
    display(df_rel)


def _make_region_subplot_count(n_items, n_cols=3):
    return int(np.ceil(max(n_items, 1) / float(n_cols))), n_cols


def _plot_reliability_scatter(df_coupling, df_rel, context_label, metric):
    if df_coupling is None or df_coupling.empty or df_rel.empty:
        return None
    metric = metric.lower().strip()
    suffix_a, suffix_b = ("h1", "h2") if context_label == "Spont" else ("odd", "even")
    if metric == "delay":
        col_a = f"coupling_delay_ms_{suffix_a}"
        col_b = f"coupling_delay_ms_{suffix_b}"
        rel_col = "r_delay"
    else:
        col_a = f"coupling_strength_{suffix_a}"
        col_b = f"coupling_strength_{suffix_b}"
        rel_col = "r_strength"
    regions = sorted(df_rel["region"].astype(str).unique().tolist())
    n_rows, n_cols = _make_region_subplot_count(len(regions), n_cols=3)
    titles = []
    for region in regions:
        match = df_rel.loc[df_rel["region"] == region, rel_col]
        corr_val = float(match.iloc[0]) if not match.empty else np.nan
        titles.append(f"{region} | r={corr_val:.3f}" if np.isfinite(corr_val) else f"{region} | r=NA")
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
    region_colors = _build_region_colors(regions)
    for idx, region in enumerate(regions):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        grp = df_coupling.loc[df_coupling["region"] == region, ["cluster_id", col_a, col_b]].dropna()
        if grp.empty:
            fig.add_annotation(x=0.5, y=0.5, xref=f"x{idx + 1} domain", yref=f"y{idx + 1} domain", text="No data", showarrow=False)
            continue
        fig.add_trace(
            go.Scatter(
                x=grp[col_a],
                y=grp[col_b],
                mode="markers",
                marker=dict(size=7, opacity=0.75, color=region_colors.get(region, "rgb(31,119,180)")),
                text=[f"cluster_id={cid}" for cid in grp["cluster_id"]],
                name=region,
                showlegend=False,
                hovertemplate="region=%{fullData.name}<br>%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        xy = np.concatenate([grp[col_a].to_numpy(dtype=float), grp[col_b].to_numpy(dtype=float)])
        finite = xy[np.isfinite(xy)]
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            fig.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(100,100,100,0.5)", dash="dash"), showlegend=False),
                row=row,
                col=col,
            )
    fig.update_layout(title=f"{context_label} reliability scatter | {metric.title()}", template="plotly_white", height=max(350, 320 * n_rows), width=1280)
    return fig


def _plot_reliability_summary(df_rel_all):
    if df_rel_all.empty:
        return None
    summary = df_rel_all.groupby("context", as_index=False)[["r_delay", "r_strength"]].mean(numeric_only=True).sort_values("context")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Delay reliability", "Strength reliability"])
    fig.add_trace(go.Bar(x=summary["context"], y=summary["r_delay"], marker_color="#1f77b4", name="Delay"), row=1, col=1)
    fig.add_trace(go.Bar(x=summary["context"], y=summary["r_strength"], marker_color="#ff7f0e", name="Strength"), row=1, col=2)
    fig.update_layout(title="Mean coupling reliability across contexts", template="plotly_white", showlegend=False)
    return fig


def _plot_region_box(df, score_col, title, yaxis_title=None):
    if df.empty or score_col not in df.columns:
        return None
    regions = sorted(df["region"].astype(str).unique().tolist())
    region_colors = _build_region_colors(regions)
    x_map = {region: idx for idx, region in enumerate(regions)}
    jitter_rng = np.random.default_rng(0)
    fig = go.Figure()
    for region in regions:
        grp = df.loc[df["region"] == region]
        color = region_colors.get(region, "rgb(31,119,180)")
        xpos = np.full(len(grp), x_map[region], dtype=float)
        fig.add_trace(
            go.Box(
                x=xpos,
                y=grp[score_col],
                name=region,
                boxpoints=False,
                marker_color=color,
                line=dict(color=color),
                fillcolor=_with_alpha(color, 0.35),
                showlegend=False,
            )
        )
        if len(grp) > 0:
            fig.add_trace(
                go.Scatter(
                    x=xpos + jitter_rng.uniform(-0.16, 0.16, size=len(grp)),
                    y=grp[score_col],
                    mode="markers",
                    marker=dict(color=color, size=7, opacity=0.75),
                    customdata=grp[["cluster_id"]].to_numpy() if "cluster_id" in grp.columns else None,
                    name=region,
                    showlegend=False,
                    hovertemplate=(
                        "region=" + region + "<br>"
                        + "cluster_id=%{customdata[0]}<br>"
                        + f"{score_col}=%{{y:.4f}}<extra></extra>"
                        if "cluster_id" in grp.columns
                        else f"region={region}<br>{score_col}=%{{y:.4f}}<extra></extra>"
                    ),
                )
            )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_map.values()),
        ticktext=regions,
        range=[-0.5, max(len(regions) - 0.5, 0.5)],
        title="Region",
    )
    fig.update_layout(title=title, yaxis_title=yaxis_title or score_col, template="plotly_white")
    return fig


def _plot_stacked_region_means(df, columns, title):
    if df.empty:
        return None
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return None
    summary = df.groupby("region", as_index=False)[columns].mean(numeric_only=True)
    if summary.empty:
        return None
    fig = go.Figure()
    if len(summary) == 1:
        base_color = _build_region_colors(summary["region"].tolist()).get(str(summary["region"].iloc[0]), "rgb(31,119,180)")
        mix_levels = np.linspace(0.05, 0.55, num=len(columns), endpoint=True)
        colors = [_mix_with_white(base_color, mix) for mix in mix_levels]
    else:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for col, color in zip(columns, colors):
        if col in summary.columns:
            fig.add_trace(go.Bar(x=summary["region"], y=summary[col], name=col, marker_color=color))
    fig.update_layout(title=title, barmode="stack", template="plotly_white", xaxis_title="Region", yaxis_title="Mean delta pseudo-R2")
    return fig


def _plot_region_heatmap(df, columns, title, colorbar_title):
    if df.empty or not columns:
        return None
    summary = df.groupby("region", as_index=False)[columns].mean(numeric_only=True)
    if summary.empty:
        return None
    colorscale = "Viridis"
    if len(summary) == 1:
        base_color = _build_region_colors(summary["region"].tolist()).get(str(summary["region"].iloc[0]), "rgb(31,119,180)")
        colorscale = [(0.0, "rgb(255,255,255)"), (1.0, base_color)]
    fig = go.Figure(data=go.Heatmap(z=summary[columns].to_numpy(dtype=float), x=columns, y=summary["region"], colorbar_title=colorbar_title, colorscale=colorscale))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def _plot_internal_vs_coupling(df, title):
    required = {"coupling_strength_train_spont", "delta_Internal Sequence", "region", "cluster_id"}
    if df.empty or not required.issubset(df.columns):
        return None
    fig = go.Figure()
    region_colors = _build_region_colors(df["region"].astype(str).unique().tolist())
    for region in sorted(df["region"].astype(str).unique().tolist()):
        grp = df.loc[df["region"] == region]
        fig.add_trace(
            go.Scatter(
                x=grp["coupling_strength_train_spont"],
                y=grp["delta_Internal Sequence"],
                mode="markers",
                name=region,
                marker=dict(color=region_colors.get(region, "rgb(31,119,180)"), size=8, opacity=0.8),
                text=[f"cluster_id={cid}" for cid in grp["cluster_id"]],
                hovertemplate="region=%{fullData.name}<br>%{text}<br>coupling=%{x:.4f}<br>delta_Internal Sequence=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Spontaneous training coupling strength", yaxis_title="Internal sequence delta pseudo-R2")
    return fig


def _plot_full_vs_without_internal(df, title):
    if df.empty or "score_without_Internal" not in df.columns:
        return None
    fig = go.Figure()
    region_colors = _build_region_colors(df["region"].astype(str).unique().tolist())
    for region in sorted(df["region"].astype(str).unique().tolist()):
        grp = df.loc[df["region"] == region]
        fig.add_trace(
            go.Scatter(
                x=grp["pseudo_r2_cohen"],
                y=grp["score_without_Internal"],
                mode="markers",
                name=region,
                marker=dict(color=region_colors.get(region, "rgb(31,119,180)"), size=8, opacity=0.8),
                text=[f"cluster_id={cid}" for cid in grp["cluster_id"]],
                hovertemplate="region=%{fullData.name}<br>%{text}<br>full=%{x:.4f}<br>without Internal=%{y:.4f}<extra></extra>",
            )
        )
    finite = np.concatenate([df["pseudo_r2_cohen"].to_numpy(dtype=float), df["score_without_Internal"].to_numpy(dtype=float)])
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(100,100,100,0.5)", dash="dash"), showlegend=False))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Full-model held-out pseudo-R2 (Cohen)", yaxis_title="Held-out pseudo-R2 without Internal")
    return fig


def _plot_ranked_region(df, title):
    if df.empty:
        return None
    region_counts = df.groupby("region").size().sort_values(ascending=False)
    if region_counts.empty:
        return None
    region = region_counts.index[0]
    grp = df.loc[df["region"] == region].sort_values("pseudo_r2_cohen", ascending=False).reset_index(drop=True)
    if grp.empty:
        return None
    color = _build_region_colors([region]).get(region, "rgb(31,119,180)")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=np.arange(len(grp)),
            y=grp["pseudo_r2_cohen"],
            marker_color=color,
            text=grp["cluster_id"].astype(str),
            hovertemplate="rank=%{x}<br>cluster_id=%{text}<br>pseudo_r2=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"{title} | region={region}", template="plotly_white", xaxis_title="Neuron rank", yaxis_title="Held-out pseudo-R2 (Cohen)")
    return fig


def _safe_stem(text):
    text = str(text)
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "plot"


def _segment_series_with_gaps(times_s, values, max_gap_s):
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if times_s.size == 0 or values.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if values.shape[0] != times_s.shape[0]:
        if values.shape[0] == 1:
            values = np.full(times_s.shape[0], float(values[0]), dtype=float)
        else:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
    valid = np.isfinite(times_s) & np.isfinite(values)
    times_s = times_s[valid]
    values = values[valid]
    if times_s.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_out = [float(times_s[0])]
    y_out = [float(values[0])]
    for idx in range(1, times_s.size):
        if float(times_s[idx] - times_s[idx - 1]) > float(max_gap_s):
            x_out.append(np.nan)
            y_out.append(np.nan)
        x_out.append(float(times_s[idx]))
        y_out.append(float(values[idx]))
    return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float)


def _counts_to_raster_xy(times_s, counts, y_value):
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)
    if times_s.size == 0 or counts.shape[0] != times_s.shape[0]:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=int)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.clip(np.rint(counts).astype(int), 0, None)
    keep = counts > 0
    if not np.any(keep):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=int)
    x_out = times_s[keep]
    y_out = np.full(x_out.shape[0], float(y_value), dtype=float)
    c_out = counts[keep]
    return x_out, y_out, c_out


def _discretize_predicted_counts(predicted_mean_count, seed_value=None):
    lam = np.asarray(predicted_mean_count, dtype=float).reshape(-1)
    lam = np.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)
    lam = np.clip(lam, 0.0, None)
    if lam.size == 0:
        return np.asarray([], dtype=int)
    seed_int = int(seed_value) if seed_value is not None and np.isfinite(seed_value) else 0
    seed_int = (seed_int * 2654435761 + 1013904223) % (2**32 - 1)
    rng = np.random.default_rng(seed_int)
    return np.asarray(rng.poisson(lam), dtype=int)


def _get_cluster_depth_lookup_local(clusters, cid_to_idx=None):
    depth_arr = None
    try:
        if hasattr(clusters, "keys") and "depths" in clusters.keys():
            depth_arr = np.asarray(clusters["depths"], dtype=float)
        elif hasattr(clusters, "keys") and "depth" in clusters.keys():
            depth_arr = np.asarray(clusters["depth"], dtype=float)
    except Exception:
        depth_arr = None
    if depth_arr is None and isinstance(clusters, pd.DataFrame):
        if "depths" in clusters.columns:
            depth_arr = np.asarray(clusters["depths"], dtype=float)
        elif "depth" in clusters.columns:
            depth_arr = np.asarray(clusters["depth"], dtype=float)
    if depth_arr is None:
        return {}
    if isinstance(cid_to_idx, dict) and cid_to_idx:
        out = {}
        for cid, idx in cid_to_idx.items():
            try:
                idx_int = int(idx)
                if 0 <= idx_int < depth_arr.shape[0]:
                    out[int(cid)] = float(depth_arr[idx_int])
            except Exception:
                continue
        return out
    return {int(idx): float(val) for idx, val in enumerate(np.asarray(depth_arr, dtype=float).tolist())}


def _sort_glm_population_like_dashboard(df_region, cluster_depth_lookup=None, cluster_region_lookup=None):
    if df_region is None or df_region.empty:
        return pd.DataFrame(), []
    df_sorted = df_region.copy()
    df_sorted["cluster_id"] = pd.to_numeric(df_sorted["cluster_id"], errors="coerce")
    df_sorted = df_sorted[np.isfinite(df_sorted["cluster_id"])].copy()
    if df_sorted.empty:
        return pd.DataFrame(), []
    df_sorted["cluster_id"] = df_sorted["cluster_id"].astype(int)
    if "region" not in df_sorted.columns:
        df_sorted["region"] = np.nan
    if isinstance(cluster_region_lookup, dict) and cluster_region_lookup:
        missing_region = df_sorted["region"].isna() | (df_sorted["region"].astype(str).str.len() == 0)
        if missing_region.any():
            df_sorted.loc[missing_region, "region"] = df_sorted.loc[missing_region, "cluster_id"].map(cluster_region_lookup)
    df_sorted["region"] = df_sorted["region"].fillna("Unknown").astype(str)
    if isinstance(cluster_depth_lookup, dict) and cluster_depth_lookup:
        df_sorted["depth"] = df_sorted["cluster_id"].map(cluster_depth_lookup)
    else:
        df_sorted["depth"] = np.nan
    df_depth_sorted = df_sorted.sort_values(["depth", "cluster_id"], ascending=[True, True], na_position="last").reset_index(drop=True)
    region_order = df_depth_sorted["region"].dropna().astype(str).drop_duplicates().tolist()
    sorted_groups = []
    for region_name in region_order:
        region_df = df_depth_sorted.loc[df_depth_sorted["region"] == region_name].copy()
        region_df["coupling_delay_ms_train_spont"] = pd.to_numeric(region_df["coupling_delay_ms_train_spont"], errors="coerce")
        region_df = region_df.sort_values(
            ["coupling_delay_ms_train_spont", "depth", "cluster_id"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        sorted_groups.append(region_df)
    if sorted_groups:
        return pd.concat(sorted_groups, ignore_index=True), region_order
    return df_depth_sorted, region_order


def _merge_spont_test_plot_payload_legacy_aware(
    payload,
    fallback_times_s,
    fallback_whisk_signal,
    fallback_pupil_signal,
    fallback_whisk_available,
    fallback_pupil_available,
):
    merged_payload = {
        "times_s": np.asarray(fallback_times_s, dtype=float).reshape(-1),
        "observed_count_by_cid": {},
        "predicted_mean_count_by_cid": {},
        "whisk_signal": np.asarray(fallback_whisk_signal, dtype=float).reshape(-1),
        "pupil_signal": np.asarray(fallback_pupil_signal, dtype=float).reshape(-1),
        "whisk_available": bool(fallback_whisk_available),
        "pupil_available": bool(fallback_pupil_available),
    }

    def _convert_rate_dict_to_count_dict(data_dict):
        out = {}
        if not isinstance(data_dict, dict):
            return out
        for cid, arr in data_dict.items():
            arr = np.asarray(arr, dtype=float).reshape(-1)
            out[int(cid)] = arr * float(BIN_SIZE_S)
        return out

    def _merge_one_dict(dct):
        if not isinstance(dct, dict):
            return
        if merged_payload["times_s"].size == 0:
            candidate_times = np.asarray(dct.get("times_s", np.asarray([], dtype=float)), dtype=float).reshape(-1)
            if candidate_times.size > 0:
                merged_payload["times_s"] = candidate_times
        if "whisk_signal" in dct:
            candidate = np.asarray(dct.get("whisk_signal", np.asarray([], dtype=float)), dtype=float).reshape(-1)
            if candidate.size > 0:
                merged_payload["whisk_signal"] = candidate
        if "pupil_signal" in dct:
            candidate = np.asarray(dct.get("pupil_signal", np.asarray([], dtype=float)), dtype=float).reshape(-1)
            if candidate.size > 0:
                merged_payload["pupil_signal"] = candidate
        if "whisk_available" in dct:
            merged_payload["whisk_available"] = bool(dct.get("whisk_available"))
        if "pupil_available" in dct:
            merged_payload["pupil_available"] = bool(dct.get("pupil_available"))
        merged_payload["observed_count_by_cid"].update(dct.get("observed_count_by_cid", {}))
        merged_payload["predicted_mean_count_by_cid"].update(dct.get("predicted_mean_count_by_cid", {}))
        if "observed_rate_by_cid" in dct:
            merged_payload["observed_count_by_cid"].update(_convert_rate_dict_to_count_dict(dct.get("observed_rate_by_cid", {})))
        if "predicted_rate_by_cid" in dct:
            merged_payload["predicted_mean_count_by_cid"].update(_convert_rate_dict_to_count_dict(dct.get("predicted_rate_by_cid", {})))

    if isinstance(payload, dict):
        if any(key in payload for key in ("observed_count_by_cid", "predicted_mean_count_by_cid", "observed_rate_by_cid", "predicted_rate_by_cid")):
            _merge_one_dict(payload)
        else:
            for sub_payload in payload.values():
                _merge_one_dict(sub_payload)
    return merged_payload


def _plot_spont_test_observed_vs_predicted(
    title_label,
    df_region,
    times_s,
    observed_count_by_cid,
    predicted_mean_count_by_cid,
    whisk_signal=None,
    pupil_signal=None,
    whisk_available=False,
    pupil_available=False,
    cluster_depth_lookup=None,
    cluster_region_lookup=None,
):
    if df_region is None or df_region.empty:
        return None
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    finite_times = times_s[np.isfinite(times_s)]
    if finite_times.size == 0:
        return None
    whisk_signal = np.asarray(whisk_signal if whisk_signal is not None else np.asarray([], dtype=float), dtype=float).reshape(-1)
    pupil_signal = np.asarray(pupil_signal if pupil_signal is not None else np.asarray([], dtype=float), dtype=float).reshape(-1)
    df_region, region_order = _sort_glm_population_like_dashboard(
        df_region,
        cluster_depth_lookup=cluster_depth_lookup,
        cluster_region_lookup=cluster_region_lookup,
    )
    if df_region.empty:
        return None
    region_colors = _build_region_colors(region_order)

    actual_color = "rgb(0,0,0)"
    predicted_color = "rgb(214,39,40)"
    t_min = float(np.nanmin(finite_times))
    t_max = float(np.nanmax(finite_times))
    gap_threshold_s = max(float(BIN_SIZE_S) * 1.5, 1e-6)
    n_units = len(df_region)
    y_positions = np.arange(n_units, dtype=float)
    zero_row = np.zeros(times_s.shape[0], dtype=float)
    z_obs_rows = []
    z_pred_rows = []
    cluster_id_rows = []

    for _, row in df_region.iterrows():
        cid = int(row["cluster_id"])
        obs_counts = np.asarray(observed_count_by_cid.get(cid, zero_row), dtype=float).reshape(-1)
        if obs_counts.shape[0] != times_s.shape[0]:
            obs_counts = zero_row.copy()
        pred_mean_count = np.asarray(predicted_mean_count_by_cid.get(cid, zero_row), dtype=float).reshape(-1)
        if pred_mean_count.shape[0] != times_s.shape[0]:
            pred_mean_count = zero_row.copy()
        z_obs_rows.append(np.nan_to_num(obs_counts, nan=0.0, posinf=0.0, neginf=0.0))
        z_pred_rows.append(np.nan_to_num(pred_mean_count, nan=0.0, posinf=0.0, neginf=0.0))
        cluster_id_rows.append(cid)

    if not z_obs_rows:
        return None

    z_obs = np.vstack(z_obs_rows)
    z_pred = np.vstack(z_pred_rows)
    cluster_id_grid = np.repeat(np.asarray(cluster_id_rows, dtype=int)[:, None], times_s.shape[0], axis=1)
    finite_counts = np.concatenate(
        [
            z_obs[np.isfinite(z_obs)],
            z_pred[np.isfinite(z_pred)],
        ]
    )
    finite_counts = finite_counts[finite_counts > 0]
    if finite_counts.size > 0:
        cmax = float(np.nanpercentile(finite_counts, 99.0))
        if not np.isfinite(cmax) or cmax <= 0:
            cmax = float(np.nanmax(finite_counts))
    else:
        cmax = 1.0
    cmax = max(cmax, 1.0)

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes="all",
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(
            "GLM predicted",
            "Observed",
            "Whisking",
            "Whisking",
            "Pupil diameter (smooth)",
            "Pupil diameter (smooth)",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=z_pred,
            x=times_s,
            y=y_positions,
            customdata=cluster_id_grid,
            zmin=0.0,
            zmax=cmax,
            colorscale=[(0.0, "rgb(255,255,255)"), (1.0, "rgb(0,0,0)")],
            hovertemplate=(
                "time=%{x:.3f}s<br>"
                "cluster_id=%{customdata}<br>"
                "predicted output/bin=%{z:.3f}<extra></extra>"
            ),
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_obs,
            x=times_s,
            y=y_positions,
            customdata=cluster_id_grid,
            zmin=0.0,
            zmax=cmax,
            colorscale=[
                (0.0, "rgb(255,255,255)"),
                (0.2, "rgb(225,225,225)"),
                (0.5, "rgb(150,150,150)"),
                (1.0, "rgb(0,0,0)"),
            ],
            hovertemplate=(
                "time=%{x:.3f}s<br>"
                "cluster_id=%{customdata}<br>"
                "observed count/bin=%{z:.0f}<extra></extra>"
            ),
            showscale=False,
        ),
        row=1,
        col=2,
    )

    for region_idx, region_name in enumerate(region_order):
        group = df_region.loc[df_region["region"] == region_name]
        if group.empty:
            continue
        y0 = float(group.index.min()) - 0.5
        if region_idx > 0:
            for col in (1, 2):
                fig.add_shape(
                    type="line",
                    x0=t_min,
                    x1=t_max,
                    y0=y0,
                    y1=y0,
                    line=dict(color=_with_alpha(region_colors.get(region_name, "rgb(31,119,180)"), 0.5), width=1),
                    layer="above",
                    row=1,
                    col=col,
                )
        fig.add_annotation(
            x=t_max,
            y=(float(group.index.min()) + float(group.index.max())) / 2.0,
            xanchor="left",
            yanchor="middle",
            text=region_name,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xshift=10,
            row=1,
            col=2,
        )

    whisk_x, whisk_y = _segment_series_with_gaps(times_s, whisk_signal, gap_threshold_s)
    if bool(whisk_available) and whisk_x.size > 0:
        for col in (1, 2):
            fig.add_trace(
                go.Scatter(
                    x=whisk_x,
                    y=whisk_y,
                    mode="lines",
                    line=dict(color="rgb(255,127,14)", width=1.8),
                    name="Whisking",
                    showlegend=False,
                ),
                row=2,
                col=col,
            )
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="y3 domain", text="Whisking signal not available", showarrow=False)
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="y4 domain", text="Whisking signal not available", showarrow=False)

    pupil_x, pupil_y = _segment_series_with_gaps(times_s, pupil_signal, gap_threshold_s)
    if bool(pupil_available) and pupil_x.size > 0:
        for col in (1, 2):
            fig.add_trace(
                go.Scatter(
                    x=pupil_x,
                    y=pupil_y,
                    mode="lines",
                    line=dict(color="rgb(0,0,0)", width=1.6),
                    name="Pupil diameter",
                    showlegend=False,
                ),
                row=3,
                col=col,
            )
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="y5 domain", text="Pupil signal not available", showarrow=False)
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="y6 domain", text="Pupil signal not available", showarrow=False)

    fig.update_layout(
        title=f"Spont test binned raster vs GLM prediction | {title_label}",
        template="plotly_white",
        height=min(max(620, 4 * len(df_region) + 220), 1800),
        width=1650,
        hovermode="closest",
        margin=dict(l=70, r=70, t=80, b=60),
    )
    for col in (1, 2):
        fig.update_yaxes(
            title_text="Neurons sorted by coupling delay" if col == 1 else None,
            showticklabels=False,
            range=[-0.5, n_units - 0.5],
            row=1,
            col=col,
        )
        fig.update_yaxes(title_text="Whisking" if col == 1 else None, row=2, col=col)
        fig.update_yaxes(title_text="Pupil" if col == 1 else None, row=3, col=col)
        fig.update_xaxes(showgrid=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, row=1, col=col)
        fig.update_xaxes(range=[t_min, t_max], row=1, col=col)
        fig.update_xaxes(range=[t_min, t_max], row=2, col=col)
        fig.update_xaxes(range=[t_min, t_max], row=3, col=col)
        fig.update_xaxes(title_text="Time in session (s)", row=3, col=col)
    fig.update_xaxes(matches="x", row=1, col=2)
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=2, col=2)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_xaxes(matches="x", row=3, col=2)
    fig.update_yaxes(matches="y", row=1, col=2)
    return fig


def _plot_internal_variant_box_subplots(df, variant_specs, title, yaxis_title):
    if df.empty:
        return None
    available_specs = [(col, label, mix_frac) for col, label, mix_frac in variant_specs if col in df.columns]
    if not available_specs:
        return None
    regions = sorted(df["region"].astype(str).unique().tolist())
    n_rows, n_cols = _make_region_subplot_count(len(regions), n_cols=3)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=regions)
    region_colors = _build_region_colors(regions)
    for idx, region in enumerate(regions):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        grp = df.loc[df["region"] == region].copy()
        if grp.empty:
            continue
        base_color = region_colors.get(region, "rgb(31,119,180)")
        for spec_idx, (score_col, label, mix_frac) in enumerate(available_specs):
            vals = pd.to_numeric(grp[score_col], errors="coerce")
            valid_mask = np.isfinite(vals.to_numpy(dtype=float))
            if not valid_mask.any():
                continue
            grp_valid = grp.loc[valid_mask]
            color = _mix_with_white(base_color, mix_frac)
            fig.add_trace(
                go.Box(
                    x=[label] * len(grp_valid),
                    y=vals.loc[valid_mask],
                    name=label,
                    legendgroup=label,
                    boxpoints="all",
                    jitter=0.28,
                    pointpos=0.0,
                    marker=dict(color=color, opacity=0.75, size=6),
                    line=dict(color=color),
                    fillcolor=_with_alpha(color, 0.35),
                    text=[f"cluster_id={int(cid)}" for cid in grp_valid["cluster_id"]],
                    customdata=grp_valid[["cluster_id"]].to_numpy(),
                    hovertemplate=(
                        f"region={region}<br>variant={label}<br>cluster_id=%{{customdata[0]}}"
                        f"<br>{score_col}=%{{y:.4f}}<extra></extra>"
                    ),
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Variant", row=row, col=col)
        fig.update_yaxes(title_text=yaxis_title if col == 1 else None, row=row, col=col)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(420, 320 * n_rows),
        width=1400,
        boxmode="group",
    )
    return fig


def _summarize_region_results(df, prefix):
    if df.empty:
        return pd.DataFrame()
    numeric_cols = [col for col in df.columns if col not in {"region", "cluster_id"} and pd.api.types.is_numeric_dtype(df[col])]
    summary = df.groupby("region", as_index=False)[numeric_cols].agg(["mean", "median", "count"])
    summary.columns = ["region" if col[0] == "region" else f"{col[0]}_{col[1]}" for col in summary.columns.to_flat_index()]
    summary.insert(0, "table", prefix)
    return summary

# %% Data loading
def process_pid(pid):
    global OUTPUT_DIR
    PID = str(pid)
    OUTPUT_DIR = OUTPUT_ROOT / PID
    _set_plotly_renderer(PLOTLY_RENDERER)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_data, path_fig, path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    ba, br, beryl_acronyms, hier_scores = prepare_region_dirs(path_data)

    print(f"Loading session for PID: {PID}")
    one, one_mode, ssl, spikes, clusters, sl = _load_session_data_with_retry(
        PID,
        ibl_cache=ibl_cache,
        ba=ba,
        preferred_mode="local",
        allow_remote=True,
        load_trials=True,
        load_wheel=True,
        load_pose=False,
        load_motion_energy=True,
        load_pupil=False,
    )
    pupil_features, pupil_times = load_pupil_data(sl)
    trials_df = _trials_to_dataframe(sl.trials)
    eid = getattr(sl, "eid", None)
    if eid is None:
        eid, _ = one.pid2eid(PID)
    spont_intervals = _load_spontaneous_intervals(one, eid)
    cluster_ids_all, cid_to_idx = build_cluster_id_map(clusters)
    cluster_ids = _select_cluster_ids_by_label(cluster_ids_all, clusters, label_min=MIN_LABEL_VALUE)
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if cluster_ids.size == 0:
        raise RuntimeError(f"No neurons passed MIN_LABEL_VALUE={MIN_LABEL_VALUE}.")
    cluster_acronyms_all = np.asarray(map_acronyms(clusters, br, ATLAS_MAPPING), dtype=str)
    cluster_region_map = {int(cid): str(cluster_acronyms_all[cid_to_idx[int(cid)]]) for cid in cluster_ids}
    region_to_cluster_ids = {}
    for cid in cluster_ids:
        region_to_cluster_ids.setdefault(cluster_region_map[int(cid)], []).append(int(cid))

    session_end_s = _get_session_end(spikes, trials_df, sl, pupil_features, pupil_times, spont_intervals)
    bin_edges, bin_centers = _build_global_bin_grid(session_end_s, BIN_SIZE_S)

    if spont_intervals is None or _coerce_interval_array(spont_intervals).size == 0:
        raise RuntimeError("No spontaneous interval was found for this session.")
    spont_start = float(spont_intervals[0][0])
    spont_end = float(spont_intervals[0][1])
    if spont_end > session_end_s:
        spont_end = session_end_s
    spont_intervals = np.array([[spont_start, spont_end]], dtype=float)

    task_windows = ana_utils.build_task_window_table(trials_df, TASK_EVENT_NAMES, post_event_s=TASK_POST_EVENT_S)
    trial_end_times = ana_utils.compute_trial_end_times(trials_df, TASK_EVENT_NAMES, post_event_s=TASK_POST_EVENT_S)
    stim_on_times = np.asarray(trials_df["stimOn_times"], dtype=float)
    iti_windows = ana_utils.build_iti_windows(trial_end_times, stim_on_times, skip_first_last=ITI_SKIP_FIRST_LAST)

    spont_holdout_rng = np.random.default_rng(SPONT_HOLDOUT_RANDOM_SEED)
    spont_train_intervals, spont_test_intervals, spont_test_block_indices = _make_holdout_blocks(
        spont_start,
        spont_end,
        modulus=TRAIN_TEST_MODULUS,
        test_index=TRAIN_TEST_TEST_INDEX,
        n_test_blocks=SPONT_TEST_BLOCK_COUNT,
        rng=spont_holdout_rng,
    )
    task_train_windows, task_test_windows = _split_windows_by_holdout(task_windows, modulus=TRAIN_TEST_MODULUS, test_index=TRAIN_TEST_TEST_INDEX)

    mask_spont = _intervals_to_mask(bin_centers, spont_intervals)
    mask_spont_train = _intervals_to_mask(bin_centers, spont_train_intervals)
    mask_spont_test = _intervals_to_mask(bin_centers, spont_test_intervals)
    mask_task = _intervals_to_mask(bin_centers, task_windows)
    mask_task_train = _intervals_to_mask(bin_centers, task_train_windows)
    mask_task_test = _intervals_to_mask(bin_centers, task_test_windows)
    mask_iti = _intervals_to_mask(bin_centers, iti_windows)

    assert not np.any(mask_spont_train & mask_spont_test), "Spont train/test overlap."
    assert not np.any(mask_task_train & mask_task_test), "Task train/test overlap."
    assert np.all(mask_task_train <= mask_task), "Task train mask leaked outside task windows."
    assert np.all(mask_task_test <= mask_task), "Task test mask leaked outside task windows."

    print(f"ONE mode: {one_mode}")
    print(f"Selected neurons after label filter: {cluster_ids.size}")
    print(f"Regions after label filter: {len(region_to_cluster_ids)}")
    print(f"Spont interval: {spont_start:.3f}s to {spont_end:.3f}s")
    print(f"Spont holdout blocks: test={spont_test_block_indices.tolist()} of {TRAIN_TEST_MODULUS}")
    print(f"Task windows: {len(task_windows)} total | {len(task_train_windows)} train | {len(task_test_windows)} test")
    print(f"ITI windows: {len(iti_windows)} total")

    dashboard_cache, dashboard_cache_path = _load_dashboard_cache(PID, DASHBOARD_CACHE_DIR)
    cached_coupling_tables, cached_coupling_notes = _collect_cached_coupling_tables(
        dashboard_cache,
        cluster_ids,
        spont_intervals,
        COUPLING_CONFIG,
        MIN_LABEL_VALUE,
    )
    if dashboard_cache is None:
        print(f"Dashboard cache: none at {dashboard_cache_path}")
    else:
        print(f"Dashboard cache: {dashboard_cache_path}")
        print("Coupling cache check:", "; ".join(cached_coupling_notes))


    # %% Shared regressors on the global 10 ms grid
    wheel_speed_series = _get_wheel_speed_regressor(getattr(sl, "wheel", None), bin_centers)
    df_motion_energy, df_whisk, whisk_series = _get_whisk_regressor(sl, bin_centers)
    pupil_series = _get_pupil_regressor(pupil_features, pupil_times, bin_centers)

    task_block_entries = _build_task_features(trials_df, bin_edges, bin_centers, wheel_speed_series)
    uninstructed_block_entries = _build_uninstructed_blocks(whisk_series, pupil_series)

    X_task_shared, task_shared_specs, task_shared_feature_names = _assemble_design_matrix(task_block_entries + uninstructed_block_entries, expected_rows=bin_centers.shape[0])
    X_spont_shared, spont_shared_specs, spont_shared_feature_names = _assemble_design_matrix(uninstructed_block_entries, expected_rows=bin_centers.shape[0])

    assert X_task_shared.shape[0] == bin_centers.shape[0]
    assert X_spont_shared.shape[0] == bin_centers.shape[0]


    # %% Coupling calculations for reliability and spontaneous-training internal-sequence parameters
    print("\nComputing / loading coupling tables...")

    spikes_spont_full = ana_utils.slice_spikes_by_intervals(spikes, spont_intervals)
    spikes_spont_train = ana_utils.slice_spikes_by_intervals(spikes, spont_train_intervals)

    if "spont" in cached_coupling_tables:
        df_coupling_spont = cached_coupling_tables["spont"].copy()
        print("Loaded spontaneous full coupling from dashboard cache.")
    else:
        df_coupling_spont = ana_utils.compute_population_coupling(
            spikes_spont_full,
            clusters,
            cluster_acronyms_all,
            COUPLING_CONFIG,
            cluster_ids=cluster_ids,
            split_halves=True,
            intervals=spont_intervals,
            context_label="Spont",
        )
        print("Recomputed spontaneous full coupling.")

    df_coupling_spont_train = ana_utils.compute_population_coupling(
        spikes_spont_train,
        clusters,
        cluster_acronyms_all,
        COUPLING_CONFIG,
        cluster_ids=cluster_ids,
        split_halves=False,
        intervals=spont_train_intervals,
        context_label="Spont train",
    )

    task_odd_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy(dtype=float) if not task_windows.empty else np.empty((0, 2), dtype=float)
    task_even_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy(dtype=float) if not task_windows.empty else np.empty((0, 2), dtype=float)
    iti_odd_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float) if not iti_windows.empty else np.empty((0, 2), dtype=float)
    iti_even_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy(dtype=float) if not iti_windows.empty else np.empty((0, 2), dtype=float)

    df_task_odd = None
    df_task_even = None
    if "task" in cached_coupling_tables:
        df_coupling_task = cached_coupling_tables["task"].copy()
        print("Loaded task coupling from dashboard cache.")
    else:
        if len(task_odd_intervals) > 0:
            spikes_task_odd = ana_utils.slice_spikes_by_intervals(spikes, task_odd_intervals, exclude_intervals=spont_intervals)
            df_task_odd = ana_utils.compute_population_coupling(spikes_task_odd, clusters, cluster_acronyms_all, COUPLING_CONFIG, cluster_ids=cluster_ids, split_halves=False, intervals=task_odd_intervals, context_label="Task odd")
        if len(task_even_intervals) > 0:
            spikes_task_even = ana_utils.slice_spikes_by_intervals(spikes, task_even_intervals, exclude_intervals=spont_intervals)
            df_task_even = ana_utils.compute_population_coupling(spikes_task_even, clusters, cluster_acronyms_all, COUPLING_CONFIG, cluster_ids=cluster_ids, split_halves=False, intervals=task_even_intervals, context_label="Task even")
        df_coupling_task = ana_utils.merge_stpr_splits(df_task_odd, df_task_even, COUPLING_CONFIG, split_a="odd", split_b="even") if (df_task_odd is not None or df_task_even is not None) else pd.DataFrame()
        print("Recomputed task coupling.")

    df_iti_odd = None
    df_iti_even = None
    if "iti" in cached_coupling_tables:
        df_coupling_iti = cached_coupling_tables["iti"].copy()
        print("Loaded ITI coupling from dashboard cache.")
    else:
        if len(iti_odd_intervals) > 0:
            spikes_iti_odd = ana_utils.slice_spikes_by_intervals(spikes, iti_odd_intervals, exclude_intervals=spont_intervals)
            df_iti_odd = ana_utils.compute_population_coupling(spikes_iti_odd, clusters, cluster_acronyms_all, COUPLING_CONFIG, cluster_ids=cluster_ids, split_halves=False, intervals=iti_odd_intervals, context_label="ITI odd")
        if len(iti_even_intervals) > 0:
            spikes_iti_even = ana_utils.slice_spikes_by_intervals(spikes, iti_even_intervals, exclude_intervals=spont_intervals)
            df_iti_even = ana_utils.compute_population_coupling(spikes_iti_even, clusters, cluster_acronyms_all, COUPLING_CONFIG, cluster_ids=cluster_ids, split_halves=False, intervals=iti_even_intervals, context_label="ITI even")
        df_coupling_iti = ana_utils.merge_stpr_splits(df_iti_odd, df_iti_even, COUPLING_CONFIG, split_a="odd", split_b="even") if (df_iti_odd is not None or df_iti_even is not None) else pd.DataFrame()
        print("Recomputed ITI coupling.")

    df_rel_spont = _compute_region_reliability(df_coupling_spont, "Spont", "h1", "h2")
    df_rel_task = _compute_region_reliability(df_coupling_task, "Task", "odd", "even")
    df_rel_iti = _compute_region_reliability(df_coupling_iti, "ITI", "odd", "even")
    df_reliability_all = pd.concat([df_rel_spont, df_rel_task, df_rel_iti], ignore_index=True)

    _print_reliability_table(df_rel_spont, "Spont")
    _print_reliability_table(df_rel_task, "Task")
    _print_reliability_table(df_rel_iti, "ITI")

    save_dataframe(df_coupling_spont, "coupling_spont_full")
    save_dataframe(df_coupling_spont_train, "coupling_spont_train")
    save_dataframe(df_coupling_task, "coupling_task_full")
    save_dataframe(df_coupling_iti, "coupling_iti_full")
    save_dataframe(df_reliability_all, "coupling_reliability_by_region")
    save_dataframe(task_windows, "task_windows")
    save_dataframe(iti_windows, "iti_windows")

    train_coupling_params = _extract_training_coupling_params(df_coupling_spont_train)


    # %% GLM fitting loop

    tic = time.perf_counter()

    print("\nFitting single-neuron NeMoS GLMs...")
    GLM_LOOP_VERSION = "skip-zero-spike-v2"
    print(f"GLM loop version: {GLM_LOOP_VERSION}")


    def _make_skipped_eval(y_counts, train_mask, test_mask, status):
        y_counts = np.asarray(y_counts, dtype=float).reshape(-1)
        train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
        test_mask = np.asarray(test_mask, dtype=bool).reshape(-1)
        full = {
            "valid": False,
            "score_cohen": np.nan,
            "score_mcfadden": np.nan,
            "test_rate_count_corr": np.nan,
            "test_pred_mean_count": None,
            "n_train_bins": int(np.sum(train_mask)),
            "n_test_bins": int(np.sum(test_mask)),
            "n_features_used": 0,
            "train_spike_count": float(np.nansum(y_counts[train_mask])),
            "test_spike_count": float(np.nansum(y_counts[test_mask])),
            "status": status,
        }
        return {
            "full": full,
            "block_delta": {},
            "block_score_without": {},
            "category_delta": {},
            "category_score_without": {},
            "category_score_alone": {},
        }


    glm_spont_rows = []
    glm_task_rows = []
    spont_test_times_s = np.asarray(bin_centers[mask_spont_test], dtype=float)
    spont_test_plot_data = {
        "times_s": spont_test_times_s.copy(),
        "observed_count_by_cid": {},
        "predicted_mean_count_by_cid": {},
        "whisk_signal": np.asarray(whisk_series[mask_spont_test], dtype=float),
        "pupil_signal": np.asarray(pupil_series[mask_spont_test], dtype=float),
        "whisk_available": isinstance(df_whisk, pd.DataFrame) and not df_whisk.empty,
        "pupil_available": (
            isinstance(pupil_features, pd.DataFrame)
            and pupil_times is not None
            and "pupilDiameter_smooth" in pupil_features.columns
        ),
    }

    _glm_region_items = sorted(region_to_cluster_ids.items())
    if GLM_REGIONS is not None:
        _glm_region_items = [(r, cids) for r, cids in _glm_region_items if r in GLM_REGIONS]
        print(f"Filtering to GLM_REGIONS={GLM_REGIONS} → {len(_glm_region_items)} region(s)")

    for region_idx, (region, region_cluster_ids) in enumerate(_glm_region_items, start=1):
        region_cluster_ids = np.asarray(region_cluster_ids, dtype=int)
        region_counts, region_cid_to_col = _build_region_count_matrix(spikes, region_cluster_ids, bin_edges)
        observed_count_by_cid = {
            int(cid): np.asarray(region_counts[mask_spont_test, region_cid_to_col[int(cid)]], dtype=float)
            for cid in region_cluster_ids
        }
        predicted_mean_count_by_cid = {
            int(cid): np.full(spont_test_times_s.shape, np.nan, dtype=float)
            for cid in region_cluster_ids
        }
        spont_test_plot_data["observed_count_by_cid"].update(observed_count_by_cid)
        spont_test_plot_data["predicted_mean_count_by_cid"].update(predicted_mean_count_by_cid)
        spont_train_region_spikes = np.nansum(region_counts[mask_spont_train], axis=0)
        task_train_region_spikes = np.nansum(region_counts[mask_task_train], axis=0)
        kept_cluster_ids = []
        skipped_cluster_ids = []
        for cid in region_cluster_ids:
            col_idx = region_cid_to_col[int(cid)]
            if float(spont_train_region_spikes[col_idx]) <= 0 and float(task_train_region_spikes[col_idx]) <= 0:
                skipped_cluster_ids.append(int(cid))
            else:
                kept_cluster_ids.append(int(cid))
        print(
            f"Region {region_idx:02d}/{len(region_to_cluster_ids)} | {region} | "
            f"neurons={len(region_cluster_ids)} | kept={len(kept_cluster_ids)} | skipped_zero_train={len(skipped_cluster_ids)}"
        )
        region_progress = (
            tqdm(
                total=len(region_cluster_ids),
                desc=f"{region} neurons",
                unit="neuron",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        for cid in skipped_cluster_ids:
            coupling_match = df_coupling_spont_train.loc[df_coupling_spont_train["cluster_id"] == int(cid)]
            coupling_delay = float(coupling_match["coupling_delay_ms"].iloc[0]) if not coupling_match.empty else np.nan
            coupling_strength = float(coupling_match["coupling_strength"].iloc[0]) if not coupling_match.empty else np.nan
            skipped_spont = _make_skipped_eval(np.zeros(bin_centers.shape[0], dtype=float), mask_spont_train, mask_spont_test, "skip_zero_train_spikes_both_contexts")
            skipped_task = _make_skipped_eval(np.zeros(bin_centers.shape[0], dtype=float), mask_task_train, mask_task_test, "skip_zero_train_spikes_both_contexts")
            glm_spont_rows.append(
                {
                    "pid": PID,
                    "region": region,
                    "cluster_id": int(cid),
                    "pseudo_r2_cohen": skipped_spont["full"]["score_cohen"],
                    "pseudo_r2_mcfadden": skipped_spont["full"]["score_mcfadden"],
                    "test_rate_count_corr": skipped_spont["full"]["test_rate_count_corr"],
                    "pseudo_r2_cohen_internal_zero_delay": np.nan,
                    "pseudo_r2_mcfadden_internal_zero_delay": np.nan,
                    "test_rate_count_corr_internal_zero_delay": np.nan,
                    "fit_status_internal_zero_delay": skipped_spont["full"]["status"],
                    "pseudo_r2_cohen_without_internal": np.nan,
                    "pseudo_r2_mcfadden_without_internal": np.nan,
                    "test_rate_count_corr_without_internal": np.nan,
                    "fit_status_without_internal": skipped_spont["full"]["status"],
                    "delay_gain_pseudo_r2": np.nan,
                    "delay_gain_test_rate_count_corr": np.nan,
                    "internal_gain_zero_delay_pseudo_r2": np.nan,
                    "internal_gain_zero_delay_test_rate_count_corr": np.nan,
                    "n_train_bins": skipped_spont["full"]["n_train_bins"],
                    "n_test_bins": skipped_spont["full"]["n_test_bins"],
                    "train_spike_count": 0.0,
                    "test_spike_count": 0.0,
                    "n_features_used": skipped_spont["full"]["n_features_used"],
                    "fit_status": skipped_spont["full"]["status"],
                    "coupling_delay_ms_train_spont": coupling_delay,
                    "coupling_strength_train_spont": coupling_strength,
                    "has_internal_sequence": False,
                }
            )
            glm_task_rows.append(
                {
                    "pid": PID,
                    "region": region,
                    "cluster_id": int(cid),
                    "pseudo_r2_cohen": skipped_task["full"]["score_cohen"],
                    "pseudo_r2_mcfadden": skipped_task["full"]["score_mcfadden"],
                    "test_rate_count_corr": skipped_task["full"]["test_rate_count_corr"],
                    "pseudo_r2_cohen_internal_zero_delay": np.nan,
                    "pseudo_r2_mcfadden_internal_zero_delay": np.nan,
                    "test_rate_count_corr_internal_zero_delay": np.nan,
                    "fit_status_internal_zero_delay": skipped_task["full"]["status"],
                    "pseudo_r2_cohen_without_internal": np.nan,
                    "pseudo_r2_mcfadden_without_internal": np.nan,
                    "test_rate_count_corr_without_internal": np.nan,
                    "fit_status_without_internal": skipped_task["full"]["status"],
                    "delay_gain_pseudo_r2": np.nan,
                    "delay_gain_test_rate_count_corr": np.nan,
                    "internal_gain_zero_delay_pseudo_r2": np.nan,
                    "internal_gain_zero_delay_test_rate_count_corr": np.nan,
                    "n_train_bins": skipped_task["full"]["n_train_bins"],
                    "n_test_bins": skipped_task["full"]["n_test_bins"],
                    "train_spike_count": 0.0,
                    "test_spike_count": 0.0,
                    "n_features_used": skipped_task["full"]["n_features_used"],
                    "fit_status": skipped_task["full"]["status"],
                    "coupling_delay_ms_train_spont": coupling_delay,
                    "coupling_strength_train_spont": coupling_strength,
                    "has_internal_sequence": False,
                    "score_without_Internal": np.nan,
                }
            )
            if region_progress is not None:
                region_progress.update(1)

        for cid in kept_cluster_ids:
            col_idx = region_cid_to_col[int(cid)]
            y_counts = region_counts[:, col_idx].astype(float)
            internal_spont, internal_spont_ok = _build_internal_sequence_block(cid, region_cluster_ids, region_counts, region_cid_to_col, train_coupling_params, bin_centers, mask_spont)
            internal_task, internal_task_ok = _build_internal_sequence_block(cid, region_cluster_ids, region_counts, region_cid_to_col, train_coupling_params, bin_centers, mask_task)
            internal_spont_d0, _ = _build_internal_sequence_block(
                cid,
                region_cluster_ids,
                region_counts,
                region_cid_to_col,
                train_coupling_params,
                bin_centers,
                mask_spont,
                delay_ms_override=0.0,
            )
            internal_task_d0, _ = _build_internal_sequence_block(
                cid,
                region_cluster_ids,
                region_counts,
                region_cid_to_col,
                train_coupling_params,
                bin_centers,
                mask_task,
                delay_ms_override=0.0,
            )
            X_spont, spont_specs, _ = _assemble_design_matrix(uninstructed_block_entries + [{"name": "Internal Sequence", "category": "Internal", "data": internal_spont[:, None]}], expected_rows=bin_centers.shape[0])
            X_spont_d0, _, _ = _assemble_design_matrix(
                uninstructed_block_entries + [{"name": "Internal Sequence", "category": "Internal", "data": internal_spont_d0[:, None]}],
                expected_rows=bin_centers.shape[0],
            )
            X_task, task_specs, _ = _assemble_design_matrix(task_block_entries + uninstructed_block_entries + [{"name": "Internal Sequence", "category": "Internal", "data": internal_task[:, None]}], expected_rows=bin_centers.shape[0])
            X_task_d0, _, _ = _assemble_design_matrix(
                task_block_entries + uninstructed_block_entries + [{"name": "Internal Sequence", "category": "Internal", "data": internal_task_d0[:, None]}],
                expected_rows=bin_centers.shape[0],
            )
            assert X_spont.shape[0] == y_counts.shape[0]
            assert X_spont_d0.shape[0] == y_counts.shape[0]
            assert X_spont_shared.shape[0] == y_counts.shape[0]
            assert X_task.shape[0] == y_counts.shape[0]
            assert X_task_d0.shape[0] == y_counts.shape[0]
            assert X_task_shared.shape[0] == y_counts.shape[0]
            spont_train_spikes = float(np.nansum(y_counts[mask_spont_train]))
            spont_test_spikes = float(np.nansum(y_counts[mask_spont_test]))
            task_train_spikes = float(np.nansum(y_counts[mask_task_train]))
            task_test_spikes = float(np.nansum(y_counts[mask_task_test]))

            if spont_train_spikes <= 0:
                spont_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_train_spikes")
                spont_d0_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_train_spikes")["full"]
                spont_no_internal_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_train_spikes")["full"]
            elif spont_test_spikes <= 0:
                spont_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_test_spikes")
                spont_d0_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_test_spikes")["full"]
                spont_no_internal_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, "no_test_spikes")["full"]
            else:
                try:
                    spont_eval = _evaluate_model_family(X_spont, y_counts, train_mask=mask_spont_train, test_mask=mask_spont_test, block_specs=spont_specs)
                except Exception as exc:
                    spont_eval = _make_skipped_eval(y_counts, mask_spont_train, mask_spont_test, f"eval_error: {type(exc).__name__}: {exc}")
                spont_d0_eval = _safe_fit_and_score(X_spont_d0, y_counts, mask_spont_train, mask_spont_test)
                spont_no_internal_eval = _safe_fit_and_score(X_spont_shared, y_counts, mask_spont_train, mask_spont_test)

            if task_train_spikes <= 0:
                task_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_train_spikes")
                task_d0_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_train_spikes")["full"]
                task_no_internal_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_train_spikes")["full"]
            elif task_test_spikes <= 0:
                task_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_test_spikes")
                task_d0_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_test_spikes")["full"]
                task_no_internal_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, "no_test_spikes")["full"]
            else:
                try:
                    task_eval = _evaluate_model_family(X_task, y_counts, train_mask=mask_task_train, test_mask=mask_task_test, block_specs=task_specs)
                except Exception as exc:
                    task_eval = _make_skipped_eval(y_counts, mask_task_train, mask_task_test, f"eval_error: {type(exc).__name__}: {exc}")
                task_d0_eval = _safe_fit_and_score(X_task_d0, y_counts, mask_task_train, mask_task_test)
                task_no_internal_eval = _safe_fit_and_score(X_task_shared, y_counts, mask_task_train, mask_task_test)
            coupling_match = df_coupling_spont_train.loc[df_coupling_spont_train["cluster_id"] == int(cid)]
            coupling_delay = float(coupling_match["coupling_delay_ms"].iloc[0]) if not coupling_match.empty else np.nan
            coupling_strength = float(coupling_match["coupling_strength"].iloc[0]) if not coupling_match.empty else np.nan

            spont_row = {
                "pid": PID,
                "region": region,
                "cluster_id": int(cid),
                "pseudo_r2_cohen": spont_eval["full"]["score_cohen"],
                "pseudo_r2_mcfadden": spont_eval["full"]["score_mcfadden"],
                "test_rate_count_corr": spont_eval["full"]["test_rate_count_corr"],
                "pseudo_r2_cohen_internal_zero_delay": spont_d0_eval["score_cohen"],
                "pseudo_r2_mcfadden_internal_zero_delay": spont_d0_eval["score_mcfadden"],
                "test_rate_count_corr_internal_zero_delay": spont_d0_eval["test_rate_count_corr"],
                "fit_status_internal_zero_delay": spont_d0_eval["status"],
                "pseudo_r2_cohen_without_internal": spont_no_internal_eval["score_cohen"],
                "pseudo_r2_mcfadden_without_internal": spont_no_internal_eval["score_mcfadden"],
                "test_rate_count_corr_without_internal": spont_no_internal_eval["test_rate_count_corr"],
                "fit_status_without_internal": spont_no_internal_eval["status"],
                "delay_gain_pseudo_r2": (
                    spont_eval["full"]["score_cohen"] - spont_d0_eval["score_cohen"]
                    if np.isfinite(spont_eval["full"]["score_cohen"]) and np.isfinite(spont_d0_eval["score_cohen"])
                    else np.nan
                ),
                "delay_gain_test_rate_count_corr": (
                    spont_eval["full"]["test_rate_count_corr"] - spont_d0_eval["test_rate_count_corr"]
                    if np.isfinite(spont_eval["full"]["test_rate_count_corr"]) and np.isfinite(spont_d0_eval["test_rate_count_corr"])
                    else np.nan
                ),
                "internal_gain_zero_delay_pseudo_r2": (
                    spont_d0_eval["score_cohen"] - spont_no_internal_eval["score_cohen"]
                    if np.isfinite(spont_d0_eval["score_cohen"]) and np.isfinite(spont_no_internal_eval["score_cohen"])
                    else np.nan
                ),
                "internal_gain_zero_delay_test_rate_count_corr": (
                    spont_d0_eval["test_rate_count_corr"] - spont_no_internal_eval["test_rate_count_corr"]
                    if np.isfinite(spont_d0_eval["test_rate_count_corr"]) and np.isfinite(spont_no_internal_eval["test_rate_count_corr"])
                    else np.nan
                ),
                "n_train_bins": spont_eval["full"]["n_train_bins"],
                "n_test_bins": spont_eval["full"]["n_test_bins"],
                "train_spike_count": spont_eval["full"]["train_spike_count"],
                "test_spike_count": spont_eval["full"]["test_spike_count"],
                "n_features_used": spont_eval["full"]["n_features_used"],
                "fit_status": spont_eval["full"]["status"],
                "coupling_delay_ms_train_spont": coupling_delay,
                "coupling_strength_train_spont": coupling_strength,
                "has_internal_sequence": bool(internal_spont_ok),
            }
            for name, delta in spont_eval["block_delta"].items():
                spont_row[f"delta_{name}"] = delta
            for name, score in spont_eval["block_score_without"].items():
                spont_row[f"score_without_{name}"] = score
            for category, delta in spont_eval["category_delta"].items():
                spont_row[f"cat_delta_{category}"] = delta
            for category, score in spont_eval["category_score_without"].items():
                spont_row[f"cat_score_without_{category}"] = score
            for category, score in spont_eval["category_score_alone"].items():
                spont_row[f"cat_score_alone_{category}"] = score
            glm_spont_rows.append(spont_row)

            task_row = {
                "pid": PID,
                "region": region,
                "cluster_id": int(cid),
                "pseudo_r2_cohen": task_eval["full"]["score_cohen"],
                "pseudo_r2_mcfadden": task_eval["full"]["score_mcfadden"],
                "test_rate_count_corr": task_eval["full"]["test_rate_count_corr"],
                "pseudo_r2_cohen_internal_zero_delay": task_d0_eval["score_cohen"],
                "pseudo_r2_mcfadden_internal_zero_delay": task_d0_eval["score_mcfadden"],
                "test_rate_count_corr_internal_zero_delay": task_d0_eval["test_rate_count_corr"],
                "fit_status_internal_zero_delay": task_d0_eval["status"],
                "pseudo_r2_cohen_without_internal": task_no_internal_eval["score_cohen"],
                "pseudo_r2_mcfadden_without_internal": task_no_internal_eval["score_mcfadden"],
                "test_rate_count_corr_without_internal": task_no_internal_eval["test_rate_count_corr"],
                "fit_status_without_internal": task_no_internal_eval["status"],
                "delay_gain_pseudo_r2": (
                    task_eval["full"]["score_cohen"] - task_d0_eval["score_cohen"]
                    if np.isfinite(task_eval["full"]["score_cohen"]) and np.isfinite(task_d0_eval["score_cohen"])
                    else np.nan
                ),
                "delay_gain_test_rate_count_corr": (
                    task_eval["full"]["test_rate_count_corr"] - task_d0_eval["test_rate_count_corr"]
                    if np.isfinite(task_eval["full"]["test_rate_count_corr"]) and np.isfinite(task_d0_eval["test_rate_count_corr"])
                    else np.nan
                ),
                "internal_gain_zero_delay_pseudo_r2": (
                    task_d0_eval["score_cohen"] - task_no_internal_eval["score_cohen"]
                    if np.isfinite(task_d0_eval["score_cohen"]) and np.isfinite(task_no_internal_eval["score_cohen"])
                    else np.nan
                ),
                "internal_gain_zero_delay_test_rate_count_corr": (
                    task_d0_eval["test_rate_count_corr"] - task_no_internal_eval["test_rate_count_corr"]
                    if np.isfinite(task_d0_eval["test_rate_count_corr"]) and np.isfinite(task_no_internal_eval["test_rate_count_corr"])
                    else np.nan
                ),
                "n_train_bins": task_eval["full"]["n_train_bins"],
                "n_test_bins": task_eval["full"]["n_test_bins"],
                "train_spike_count": task_eval["full"]["train_spike_count"],
                "test_spike_count": task_eval["full"]["test_spike_count"],
                "n_features_used": task_eval["full"]["n_features_used"],
                "fit_status": task_eval["full"]["status"],
                "coupling_delay_ms_train_spont": coupling_delay,
                "coupling_strength_train_spont": coupling_strength,
                "has_internal_sequence": bool(internal_task_ok),
                "score_without_Internal": task_no_internal_eval["score_cohen"],
            }
            for name, delta in task_eval["block_delta"].items():
                task_row[f"delta_{name}"] = delta
            for name, score in task_eval["block_score_without"].items():
                task_row[f"score_without_{name}"] = score
            for category, delta in task_eval["category_delta"].items():
                task_row[f"cat_delta_{category}"] = delta
            for category, score in task_eval["category_score_without"].items():
                task_row[f"cat_score_without_{category}"] = score
            for category, score in task_eval["category_score_alone"].items():
                task_row[f"cat_score_alone_{category}"] = score
            glm_task_rows.append(task_row)
            spont_pred_mean_count = spont_eval["full"].get("test_pred_mean_count", None)
            if spont_pred_mean_count is not None:
                spont_pred_mean_count = np.asarray(spont_pred_mean_count, dtype=float).reshape(-1)
                if spont_pred_mean_count.shape[0] == spont_test_times_s.shape[0]:
                    spont_test_plot_data["predicted_mean_count_by_cid"][int(cid)] = spont_pred_mean_count
            if region_progress is not None:
                region_progress.update(1)

        if region_progress is not None:
            region_progress.close()

    df_glm_spont = pd.DataFrame(glm_spont_rows)
    df_glm_task = pd.DataFrame(glm_task_rows)
    df_glm_spont_region = _summarize_region_results(df_glm_spont, "spont")
    df_glm_task_region = _summarize_region_results(df_glm_task, "task")

    save_dataframe(df_glm_spont, "glm_spont_per_neuron")
    save_dataframe(df_glm_task, "glm_task_per_neuron")
    save_dataframe(df_glm_spont_region, "glm_spont_region_summary")
    save_dataframe(df_glm_task_region, "glm_task_region_summary")

    toc = time.perf_counter()

    # Calculate the difference
    elapsed_time = toc - tic
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # %% Metadata
    metadata_df = pd.DataFrame(
        [
            {
                "pid": PID,
                "one_mode": one_mode,
                "atlas_mapping": ATLAS_MAPPING,
                "min_label_value": MIN_LABEL_VALUE,
                "use_regularization": USE_REGULARIZATION,
                "regularizer_strength": REGULARIZER_STRENGTH,
                "bin_size_s": BIN_SIZE_S,
                "spont_holdout_random_seed": SPONT_HOLDOUT_RANDOM_SEED,
                "spont_test_block_count": SPONT_TEST_BLOCK_COUNT,
                "spont_test_block_indices": ",".join(str(int(idx)) for idx in np.asarray(spont_test_block_indices, dtype=int).tolist()),
                "session_end_s": session_end_s,
                "spont_start_s": spont_start,
                "spont_end_s": spont_end,
                "n_selected_neurons": int(cluster_ids.size),
                "n_regions": int(len(region_to_cluster_ids)),
                "n_task_windows": int(len(task_windows)),
                "n_task_train_windows": int(len(task_train_windows)),
                "n_task_test_windows": int(len(task_test_windows)),
                "n_iti_windows": int(len(iti_windows)),
            }
        ]
    )
    save_dataframe(metadata_df, "metadata")

    print("\nSaved tables:")
    for stem in (
        "coupling_reliability_by_region",
        "glm_spont_per_neuron",
        "glm_task_per_neuron",
        "glm_spont_region_summary",
        "glm_task_region_summary",
        "metadata",
    ):
        print(f"  - {OUTPUT_DIR / (stem + '.csv')}")


    # %% Plotting: reliability
    fig_rel_spont_delay = _plot_reliability_scatter(df_coupling_spont, df_rel_spont, "Spont", "delay")
    fig_rel_spont_strength = _plot_reliability_scatter(df_coupling_spont, df_rel_spont, "Spont", "strength")
    fig_rel_task_delay = _plot_reliability_scatter(df_coupling_task, df_rel_task, "Task", "delay")
    fig_rel_task_strength = _plot_reliability_scatter(df_coupling_task, df_rel_task, "Task", "strength")
    fig_rel_iti_delay = _plot_reliability_scatter(df_coupling_iti, df_rel_iti, "ITI", "delay")
    fig_rel_iti_strength = _plot_reliability_scatter(df_coupling_iti, df_rel_iti, "ITI", "strength")
    fig_rel_summary = _plot_reliability_summary(df_reliability_all)

    for fig, name in (
        (fig_rel_spont_delay, "fig_reliability_spont_delay.html"),
        (fig_rel_spont_strength, "fig_reliability_spont_strength.html"),
        (fig_rel_task_delay, "fig_reliability_task_delay.html"),
        (fig_rel_task_strength, "fig_reliability_task_strength.html"),
        (fig_rel_iti_delay, "fig_reliability_iti_delay.html"),
        (fig_rel_iti_strength, "fig_reliability_iti_strength.html"),
        (fig_rel_summary, "fig_reliability_summary.html"),
    ):
        if fig is not None:
            save_fig(fig, name)
            show_fig(fig)


    # %% Plotting: spontaneous GLM
    internal_variant_specs_pseudo = [
        ("pseudo_r2_cohen_without_internal", "No Internal", 0.60),
        ("pseudo_r2_cohen_internal_zero_delay", "d=0", 0.30),
        ("pseudo_r2_cohen", "Fitted d", 0.00),
    ]
    internal_variant_specs_corr = [
        ("test_rate_count_corr_without_internal", "No Internal", 0.60),
        ("test_rate_count_corr_internal_zero_delay", "d=0", 0.30),
        ("test_rate_count_corr", "Fitted d", 0.00),
    ]

    fig_spont_scores = _plot_region_box(df_glm_spont, "pseudo_r2_cohen", "Spont GLM held-out pseudo-R2 (Cohen)")
    fig_spont_corr = _plot_region_box(
        df_glm_spont,
        "test_rate_count_corr",
        "Spont GLM test predicted-rate vs observed-count correlation",
        yaxis_title="Test corr(predicted rate, observed counts)",
    )
    fig_spont_internal_variants_pseudo = _plot_internal_variant_box_subplots(
        df_glm_spont,
        internal_variant_specs_pseudo,
        "Spont GLM internal variants | held-out pseudo-R2 (Cohen)",
        "Held-out pseudo-R2 (Cohen)",
    )
    fig_spont_internal_variants_corr = _plot_internal_variant_box_subplots(
        df_glm_spont,
        internal_variant_specs_corr,
        "Spont GLM internal variants | test predicted-rate vs observed-count correlation",
        "Test corr(predicted rate, observed counts)",
    )
    fig_spont_stack = _plot_stacked_region_means(df_glm_spont, ["delta_Whisking", "delta_Pupil Diameter", "delta_Internal Sequence"], "Spont GLM mean drop-one contribution by region")
    fig_spont_internal = _plot_internal_vs_coupling(df_glm_spont, "Spont GLM internal contribution vs spontaneous training coupling strength")
    fig_spont_test_heatmaps = []
    payload = _merge_spont_test_plot_payload_legacy_aware(
        spont_test_plot_data if isinstance(spont_test_plot_data, dict) else {},
        fallback_times_s=np.asarray(bin_centers[mask_spont_test], dtype=float),
        fallback_whisk_signal=np.asarray(whisk_series[mask_spont_test], dtype=float),
        fallback_pupil_signal=np.asarray(pupil_series[mask_spont_test], dtype=float),
        fallback_whisk_available=isinstance(df_whisk, pd.DataFrame) and not df_whisk.empty,
        fallback_pupil_available=(
            isinstance(pupil_features, pd.DataFrame)
            and pupil_times is not None
            and "pupilDiameter_smooth" in pupil_features.columns
        ),
    )
    if not payload["observed_count_by_cid"]:
        cluster_ids_plot = np.asarray(pd.to_numeric(df_glm_spont.get("cluster_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique(), dtype=int)
        if cluster_ids_plot.size > 0:
            observed_counts_all, observed_cid_to_col = _build_region_count_matrix(spikes, cluster_ids_plot, bin_edges)
            payload["observed_count_by_cid"] = {
                int(cid): np.asarray(observed_counts_all[mask_spont_test, observed_cid_to_col[int(cid)]], dtype=float)
                for cid in cluster_ids_plot
                if int(cid) in observed_cid_to_col
            }

    cluster_depth_lookup_plot = _get_cluster_depth_lookup_local(clusters, cid_to_idx=cid_to_idx if "cid_to_idx" in globals() else None)
    cluster_region_lookup_plot = cluster_region_map if "cluster_region_map" in globals() else {}

    fig_region = _plot_spont_test_observed_vs_predicted(
        "all regions",
        df_glm_spont.copy(),
        payload.get("times_s", np.asarray([], dtype=float)),
        payload.get("observed_count_by_cid", {}),
        payload.get("predicted_mean_count_by_cid", {}),
        whisk_signal=payload.get("whisk_signal", np.asarray(whisk_series[mask_spont_test], dtype=float)),
        pupil_signal=payload.get("pupil_signal", np.asarray(pupil_series[mask_spont_test], dtype=float)),
        whisk_available=payload.get("whisk_available", isinstance(df_whisk, pd.DataFrame) and not df_whisk.empty),
        pupil_available=payload.get(
            "pupil_available",
            isinstance(pupil_features, pd.DataFrame)
            and pupil_times is not None
            and "pupilDiameter_smooth" in pupil_features.columns,
        ),
        cluster_depth_lookup=cluster_depth_lookup_plot,
        cluster_region_lookup=cluster_region_lookup_plot,
    )
    if fig_region is not None:
        fig_spont_test_heatmaps.append((fig_region, "fig_spont_test_activity_vs_prediction_all_regions.html"))

    for fig, name in (
        (fig_spont_scores, "fig_spont_pseudo_r2.html"),
        (fig_spont_corr, "fig_spont_test_rate_count_corr.html"),
        (fig_spont_internal_variants_pseudo, "fig_spont_internal_variants_pseudo_r2.html"),
        (fig_spont_internal_variants_corr, "fig_spont_internal_variants_test_rate_count_corr.html"),
        (fig_spont_stack, "fig_spont_regressor_stack.html"),
        (fig_spont_internal, "fig_spont_internal_vs_coupling.html"),
    ):
        if fig is not None:
            save_fig(fig, name)
            show_fig(fig)
    for fig, name in fig_spont_test_heatmaps:
        save_fig(fig, name)
        show_fig(fig)


    # %% Plotting: task GLM
    task_category_cols = ["cat_delta_Task", "cat_delta_Uninstructed", "cat_delta_Internal"]
    task_regressor_cols = [
        "delta_Stim Onset L",
        "delta_Stim Onset R",
        "delta_First Movement L",
        "delta_First Movement R",
        "delta_Feedback Correct",
        "delta_Feedback Incorrect",
        "delta_Wheel Speed",
        "delta_Block Probability",
        "delta_Movement Initiation",
        "delta_Whisking",
        "delta_Pupil Diameter",
        "delta_Internal Sequence",
    ]

    fig_task_scores = _plot_region_box(df_glm_task, "pseudo_r2_cohen", "Task GLM held-out pseudo-R2 (Cohen)")
    fig_task_corr = _plot_region_box(
        df_glm_task,
        "test_rate_count_corr",
        "Task GLM test predicted-rate vs observed-count correlation",
        yaxis_title="Test corr(predicted rate, observed counts)",
    )
    fig_task_internal_variants_pseudo = _plot_internal_variant_box_subplots(
        df_glm_task,
        internal_variant_specs_pseudo,
        "Task GLM internal variants | held-out pseudo-R2 (Cohen)",
        "Held-out pseudo-R2 (Cohen)",
    )
    fig_task_internal_variants_corr = _plot_internal_variant_box_subplots(
        df_glm_task,
        internal_variant_specs_corr,
        "Task GLM internal variants | test predicted-rate vs observed-count correlation",
        "Test corr(predicted rate, observed counts)",
    )
    fig_task_category = _plot_region_heatmap(df_glm_task, [col for col in task_category_cols if col in df_glm_task.columns], "Task GLM mean category contribution by region", "Mean delta pseudo-R2")
    fig_task_regressor = _plot_region_heatmap(df_glm_task, [col for col in task_regressor_cols if col in df_glm_task.columns], "Task GLM mean regressor contribution by region", "Mean delta pseudo-R2")
    fig_task_internal_compare = _plot_full_vs_without_internal(df_glm_task, "Task GLM full score vs score without internal sequence")
    fig_task_ranked = _plot_ranked_region(df_glm_task, "Task GLM ranked neurons")

    for fig, name in (
        (fig_task_scores, "fig_task_pseudo_r2.html"),
        (fig_task_corr, "fig_task_test_rate_count_corr.html"),
        (fig_task_internal_variants_pseudo, "fig_task_internal_variants_pseudo_r2.html"),
        (fig_task_internal_variants_corr, "fig_task_internal_variants_test_rate_count_corr.html"),
        (fig_task_category, "fig_task_category_heatmap.html"),
        (fig_task_regressor, "fig_task_regressor_heatmap.html"),
        (fig_task_internal_compare, "fig_task_internal_compare.html"),
        (fig_task_ranked, "fig_task_ranked_region.html"),
    ):
        if fig is not None:
            save_fig(fig, name)
            show_fig(fig)


    # %% Quick validation checks
    assert df_reliability_all["context"].isin(["Spont", "Task", "ITI"]).all()
    assert X_task_shared.shape[0] == X_spont_shared.shape[0] == bin_centers.shape[0]
    if not df_glm_spont.empty:
        assert {"pseudo_r2_cohen", "pseudo_r2_mcfadden", "cluster_id", "region"}.issubset(df_glm_spont.columns)
    if not df_glm_task.empty:
        assert {"pseudo_r2_cohen", "pseudo_r2_mcfadden", "cluster_id", "region"}.issubset(df_glm_task.columns)
    print("\nNotebook construction checks passed.")

# %% Batch run
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
batch_status_rows = []
for pid in PID_LIST:
    print(f"\n{'=' * 80}\nProcessing PID: {pid}\n{'=' * 80}")
    try:
        process_pid(pid)
        batch_status_rows.append({"pid": str(pid), "status": "ok"})
    except Exception as exc:
        batch_status_rows.append({"pid": str(pid), "status": f"error: {type(exc).__name__}: {exc}"})
        print(f"PID {pid} failed: {type(exc).__name__}: {exc}")

batch_status_df = pd.DataFrame(batch_status_rows)
try:
    batch_status_df.to_csv(OUTPUT_ROOT / "batch_run_summary.csv", index=False)
except Exception as exc:
    print(f"Warning: could not save batch summary: {exc}")
display(batch_status_df)
