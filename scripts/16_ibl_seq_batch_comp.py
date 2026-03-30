"""
Compile `15_ibl_seq_comparison` per-PID caches into lightweight batch tables for
`17_ibl_seq_regions_dash.py`.

This script treats the saved outputs of notebook 15 as the source of truth.
It does not recompute any single-PID sequence analysis. Instead, it:

1. Selects the strict, dominant `15` cache signature.
2. Resolves the full PID target set for the tagged dataset.
3. Summarizes which target PIDs have a matching cache, which are missing, and why.
4. Recomputes region-level pooled statistics from cached neuron summaries.
5. Computes PID-first region means from cached per-PID pair summaries.
6. Saves fast dashboard tables under `data/processed/16_ibl_seq_batch_comp/`.
"""

# %%
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib
import io
import json
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, disable=False, leave=True, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.disable = disable
            self.leave = leave

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n=1):
            return None

        def set_description(self, desc=None, refresh=True):
            self.desc = desc

        def set_postfix_str(self, s="", refresh=True):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return _TqdmFallback(iterable=iterable, **kwargs)

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:  # pragma: no cover
    kendalltau = None
    spearmanr = None

BASE_PATH = Path(__file__).resolve().parents[1]
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import utils.analysis as ana_utils
from utils.io import (
    build_cluster_id_map,
    get_cluster_labels_array,
    init_one,
    load_session_data,
    map_acronyms,
    setup_paths,
)

try:
    from iblatlas.atlas import AllenAtlas
except Exception:  # pragma: no cover
    AllenAtlas = None

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None

warnings.filterwarnings("ignore", category=FutureWarning)


# %% Config
BATCH_VERSION = "16_seq_batch_comp_v1.0"
SOURCE_CACHE_ROOT = BASE_PATH / "data" / "processed" / "15_ibl_seq_comparison"
OUTPUT_ROOT = BASE_PATH / "data" / "processed" / "16_ibl_seq_batch_comp"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_PIDS_TAG = "2025_Q3_IBL_et_al_BWM"
ONE_PREFERRED_MODE = "remote"
SHOW_PROGRESS = True

SOURCE_CALC_VERSION = "15_seq_comparison_v1.2"
SOURCE_CALC_LABEL_MIN = 0.9
SOURCE_MIN_REGION_NEURONS = 15
PAIR_MIN_N = 2

CATEGORY_ORDER = [
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

VARIABLE_SPECS = [
    {
        "order": 0,
        "key": "spont_whisk_coupling",
        "name": "Spont Whisk Coupling Delay",
        "mean_col": "spont_whisk_coupling_delay_ms",
        "odd_col": "spont_whisk_coupling_delay_ms_odd",
        "even_col": "spont_whisk_coupling_delay_ms_even",
    },
    {
        "order": 1,
        "key": "spont_nonwhisk_coupling",
        "name": "Spont Non-Whisk Coupling Delay",
        "mean_col": "spont_nonwhisk_coupling_delay_ms",
        "odd_col": "spont_nonwhisk_coupling_delay_ms_odd",
        "even_col": "spont_nonwhisk_coupling_delay_ms_even",
    },
    {
        "order": 2,
        "key": "first_move_delay",
        "name": "First Move Delay",
        "mean_col": "first_move_delay_ms",
        "odd_col": "first_move_delay_ms_odd",
        "even_col": "first_move_delay_ms_even",
    },
    {
        "order": 3,
        "key": "feedback_delay",
        "name": "Feedback Delay",
        "mean_col": "feedback_delay_ms",
        "odd_col": "feedback_delay_ms_odd",
        "even_col": "feedback_delay_ms_even",
    },
    {
        "order": 4,
        "key": "spont_whisk_event_delay",
        "name": "Spont Whisk Event Delay",
        "mean_col": "spont_whisk_event_delay_ms",
        "odd_col": "spont_whisk_event_delay_ms_odd",
        "even_col": "spont_whisk_event_delay_ms_even",
    },
    {
        "order": 5,
        "key": "task_whisk_event_delay",
        "name": "Task Whisk Event Delay",
        "mean_col": "task_whisk_event_delay_ms",
        "odd_col": "task_whisk_event_delay_ms_odd",
        "even_col": "task_whisk_event_delay_ms_even",
    },
]
VARIABLE_BY_KEY = {spec["key"]: spec for spec in VARIABLE_SPECS}
VARIABLE_BY_NAME = {spec["name"]: spec for spec in VARIABLE_SPECS}


# %% Helpers
class _PidSkipError(RuntimeError):
    """Raised when a PID should be treated as unavailable rather than failed."""


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _stable_hash(payload):
    blob = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _run_quietly(func, *args, suppress_stdout=True, **kwargs):
    if not suppress_stdout:
        return func(*args, **kwargs)
    captured = io.StringIO()
    with redirect_stdout(captured):
        result = func(*args, **kwargs)
    for line in captured.getvalue().splitlines():
        lower = line.strip().lower()
        if any(token in lower for token in ("warning", "error", "failed", "traceback")):
            print(line)
    return result


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _median_abs_dev(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.nanmedian(arr))
    return float(np.nanmedian(np.abs(arr - med)))


def _pearsonr_with_n(x, y, min_n=PAIR_MIN_N):
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


def _spearmanr_with_n(x, y, min_n=PAIR_MIN_N):
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


def _order_agreement_score(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return np.nan, 0, np.nan

    if kendalltau is not None:
        try:
            tau = kendalltau(x, y, nan_policy="omit")
            score = float(tau.correlation)
            n_pairs = int(n * (n - 1) // 2)
            preserved_fraction = np.nan if not np.isfinite(score) else float((score + 1.0) / 2.0)
            return score, n_pairs, preserved_fraction
        except Exception:
            pass

    concordant = 0
    comparable = 0
    for idx in range(n - 1):
        dx = x[idx + 1 :] - x[idx]
        dy = y[idx + 1 :] - y[idx]
        valid = (dx != 0) & (dy != 0)
        if not np.any(valid):
            continue
        comparable += int(np.sum(valid))
        concordant += int(np.sum(np.sign(dx[valid]) == np.sign(dy[valid])))
    if comparable == 0:
        return np.nan, 0, np.nan
    preserved_fraction = float(concordant) / float(comparable)
    score = 2.0 * preserved_fraction - 1.0
    return score, comparable, preserved_fraction


def _calc_total_reliability(rx, ry):
    if pd.notnull(rx) and pd.notnull(ry) and float(rx) * float(ry) >= 0:
        return float(np.sqrt(float(rx) * float(ry)))
    if pd.notnull(rx) and pd.isnull(ry):
        return float(rx)
    if pd.isnull(rx) and pd.notnull(ry):
        return float(ry)
    return np.nan


def _normalize_reason_group(reason):
    reason = str(reason or "").strip()
    if not reason:
        return ""
    return re.sub(r"^PID [0-9a-f-]+ ", "", reason)


def _write_table_bundle(df, stem, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    pkl_path = out_dir / f"{stem}.pkl"
    parquet_path = out_dir / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"Could not write parquet for {stem}: {exc}")
    return {"csv": csv_path, "pkl": pkl_path, "parquet": parquet_path}


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


def _cluster_id_set(values):
    return {int(v) for v in np.asarray(values, dtype=int).reshape(-1)}


def _select_cluster_ids_by_label(cluster_ids, clusters, label_min=None, strict_gt=False):
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
        if strict_gt:
            keep_mask = labels_float > float(label_min)
        else:
            keep_mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        keep_mask = labels == 1
    return cluster_ids[keep_mask]


def _compute_region_eligibility(
    cluster_ids,
    cluster_acronyms,
    clusters,
    label_min,
    min_region_neurons,
    strict_gt=False,
):
    cluster_ids = np.asarray(cluster_ids)
    selected_cluster_ids = _select_cluster_ids_by_label(
        cluster_ids,
        clusters,
        label_min=label_min,
        strict_gt=bool(strict_gt),
    )
    selected_set = _cluster_id_set(selected_cluster_ids)
    region_df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "region": np.asarray(cluster_acronyms).astype(str),
        }
    )
    region_df = region_df[region_df["cluster_id"].isin(selected_set)].copy()
    region_df = region_df[~region_df["region"].isin(["root", "void", "nan", "NA"])].copy()
    region_counts = (
        region_df["region"]
        .value_counts()
        .sort_index()
        .rename("n_units")
        .reset_index()
        .rename(columns={"index": "region"})
    )
    if region_counts.empty:
        return {"eligible_regions": [], "eligible_cluster_ids": np.array([], dtype=int)}
    eligible_regions = sorted(
        region_counts.loc[region_counts["n_units"] >= int(min_region_neurons), "region"]
        .astype(str)
        .tolist()
    )
    eligible_cluster_ids = region_df.loc[
        region_df["region"].isin(eligible_regions),
        "cluster_id",
    ].to_numpy(dtype=int)
    return {
        "eligible_regions": eligible_regions,
        "eligible_cluster_ids": np.asarray(eligible_cluster_ids, dtype=int),
    }


def _probe_pid_status(pid, one, ba, br):
    pid = str(pid)
    eid, _ = one.pid2eid(pid)
    spont_intervals = _load_spontaneous_intervals(one, eid)
    if spont_intervals is None or len(np.asarray(spont_intervals).reshape(-1)) < 2:
        raise _PidSkipError(f"PID {pid} has no spontaneous interval in *passivePeriods*.")

    ssl, spikes, clusters, sl = _run_quietly(
        load_session_data,
        pid,
        one,
        ba=ba,
        load_trials=False,
        load_wheel=False,
        load_pose=False,
        load_motion_energy=True,
        load_pupil=False,
        suppress_stdout=True,
    )

    cluster_ids, _cid_to_idx = build_cluster_id_map(clusters)
    cluster_ids = np.asarray(cluster_ids)
    if br is None:
        if hasattr(clusters, "acronym"):
            cluster_acronyms = np.asarray(clusters.acronym).astype(str)
        elif isinstance(clusters, dict) and "acronym" in clusters:
            cluster_acronyms = np.asarray(clusters["acronym"]).astype(str)
        else:
            raise RuntimeError("Could not resolve cluster acronyms while probing PID status.")
    else:
        cluster_acronyms = np.asarray(map_acronyms(clusters, br, "Beryl")).astype(str)

    df_me_raw = ana_utils.extract_motion_energy_trace(
        getattr(sl, "motion_energy", None),
        max_interp_gap_frames=3,
        ensure_positive_motion=True,
    )
    if df_me_raw.empty:
        raise _PidSkipError(f"PID {pid} has no usable motion energy trace.")

    eligibility = _compute_region_eligibility(
        cluster_ids,
        cluster_acronyms,
        clusters,
        label_min=SOURCE_CALC_LABEL_MIN,
        min_region_neurons=SOURCE_MIN_REGION_NEURONS,
        strict_gt=False,
    )
    eligible_cluster_ids = np.asarray(eligibility["eligible_cluster_ids"], dtype=int)
    if eligible_cluster_ids.size == 0:
        raise _PidSkipError(
            f"PID {pid} has no eligible regions after label>={SOURCE_CALC_LABEL_MIN} "
            f"and MIN_REGION_NEURONS={SOURCE_MIN_REGION_NEURONS}."
        )

    return {
        "pid": pid,
        "eid": str(eid),
        "eligible_units": int(eligible_cluster_ids.size),
        "eligible_regions": list(eligibility["eligible_regions"]),
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


def _get_allen_lookup():
    try:
        iblatlas_pkg = importlib.import_module("iblatlas")
        csv_path = Path(iblatlas_pkg.__file__).resolve().parent / "allen_structure_tree.csv"
        if not csv_path.exists():
            return None
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
        acr = str(row.get("acronym", "")).strip()
        if not acr:
            continue
        rgb = _to_rgb(row.get("color_hex_triplet", ""))
        if rgb:
            color_by_acr[acr] = rgb
        try:
            order_by_acr[acr] = int(float(row.get("graph_order")))
        except Exception:
            pass
        path = str(row.get("structure_id_path", ""))
        ancestor_ids = []
        for token in path.split("/"):
            token = token.strip()
            if token and token.isdigit():
                ancestor_ids.append(int(token))
        ancestor_acronyms = [id_to_acr.get(idx, "") for idx in ancestor_ids]
        category_by_acr[acr] = _category_from_acronyms(ancestor_acronyms)

    return {
        "color_by_acr": color_by_acr,
        "order_by_acr": order_by_acr,
        "category_by_acr": category_by_acr,
    }


def _build_region_meta(regions):
    regions = pd.Series(regions).dropna().astype(str).unique().tolist()
    allen_lookup = _get_allen_lookup() or {}
    color_by_acr = allen_lookup.get("color_by_acr", {})
    order_by_acr = allen_lookup.get("order_by_acr", {})
    category_by_acr = allen_lookup.get("category_by_acr", {})
    rows = []
    for region in regions:
        rows.append(
            {
                "region": region,
                "allen_color": color_by_acr.get(region),
                "allen_order": int(order_by_acr.get(region, 999999)),
                "category": category_by_acr.get(region, "Unknown"),
            }
        )
    df_meta = pd.DataFrame(rows)
    if df_meta.empty:
        return df_meta
    category_rank = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    df_meta["category_rank"] = (
        df_meta["category"].map(category_rank).fillna(len(category_rank)).astype(int)
    )
    return df_meta


def _build_pair_specs():
    pair_specs = []
    for idx, spec_x in enumerate(VARIABLE_SPECS):
        pair_specs.append((spec_x, spec_x, True))
        for spec_y in VARIABLE_SPECS[idx + 1 :]:
            pair_specs.append((spec_x, spec_y, False))
    return pair_specs


PAIR_SPECS = _build_pair_specs()


def _manifest_signature(calc_version, config_calc):
    payload = {
        "calc_version": str(calc_version or ""),
        "config_calc": config_calc if isinstance(config_calc, dict) else {},
    }
    return _stable_hash(payload), payload


def _scan_source_manifests(cache_root):
    rows = []
    manifest_paths = sorted(Path(cache_root).glob("*/*/manifest.json"))
    manifest_iter = tqdm(
        manifest_paths,
        total=len(manifest_paths),
        desc="Scanning 15 manifests",
        unit="manifest",
        disable=not SHOW_PROGRESS,
        dynamic_ncols=True,
    )
    try:
        for manifest_path in manifest_iter:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Skipping unreadable manifest {manifest_path}: {exc}")
                continue
            pid = str(manifest.get("pid") or manifest_path.parent.parent.name)
            calc_hash = str(manifest.get("calc_hash") or manifest_path.parent.name)
            cache_dir = manifest_path.parent
            config_calc = manifest.get("config_calc", {})
            calc_version = str(manifest.get("calc_version", ""))
            signature_hash, signature_payload = _manifest_signature(calc_version, config_calc)
            rows.append(
                {
                    "pid": pid,
                    "eid": str(manifest.get("eid", "")),
                    "calc_hash": calc_hash,
                    "cache_dir": str(cache_dir),
                    "manifest_path": str(manifest_path),
                    "manifest_mtime": float(cache_dir.stat().st_mtime),
                    "calc_version": calc_version,
                    "config_calc": config_calc,
                    "calc_label_min": float(config_calc.get("CALC_LABEL_MIN", np.nan)),
                    "signature_hash": signature_hash,
                    "signature_payload": signature_payload,
                }
            )
    finally:
        if hasattr(manifest_iter, "close"):
            manifest_iter.close()
    return pd.DataFrame(rows)


def _select_source_caches(df_manifest):
    if df_manifest.empty:
        raise RuntimeError(f"No `15` cache manifests found under {SOURCE_CACHE_ROOT}.")

    sig_counts = (
        df_manifest.groupby("signature_hash", as_index=False)
        .agg(
            n_manifests=("pid", "size"),
            n_pids=("pid", "nunique"),
            manifest_mtime=("manifest_mtime", "max"),
        )
        .sort_values(
            ["n_pids", "n_manifests", "manifest_mtime", "signature_hash"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    dominant_hash = str(sig_counts.iloc[0]["signature_hash"])
    dominant_row = df_manifest.loc[df_manifest["signature_hash"] == dominant_hash].iloc[0]
    dominant_payload = dict(dominant_row["signature_payload"])
    dominant_calc_version = str(dominant_payload.get("calc_version", ""))
    dominant_config = dict(dominant_payload.get("config_calc", {}))
    if dominant_calc_version != SOURCE_CALC_VERSION:
        raise RuntimeError(
            f"Dominant 15 signature uses calc_version={dominant_calc_version}, "
            f"expected {SOURCE_CALC_VERSION}."
        )
    dominant_label_min = float(dominant_config.get("CALC_LABEL_MIN", np.nan))
    if not np.isfinite(dominant_label_min) or not np.isclose(dominant_label_min, SOURCE_CALC_LABEL_MIN):
        raise RuntimeError(
            f"Dominant 15 signature uses CALC_LABEL_MIN={dominant_label_min}, "
            f"expected {SOURCE_CALC_LABEL_MIN}."
        )

    selected_by_pid = {}
    source_summary_rows = []
    for pid, df_pid in df_manifest.groupby("pid", sort=True):
        df_pid = df_pid.sort_values("manifest_mtime", ascending=False).reset_index(drop=True)
        df_match = df_pid.loc[df_pid["signature_hash"] == dominant_hash].copy()
        selected_row = None
        if not df_match.empty:
            selected_row = df_match.iloc[0].to_dict()
            selected_by_pid[str(pid)] = dict(selected_row)
        source_summary_rows.append(
            {
                "pid": str(pid),
                "matching_cache_count": int((df_pid["signature_hash"] == dominant_hash).sum()),
                "nonmatching_cache_count": int((df_pid["signature_hash"] != dominant_hash).sum()),
                "selected_cache_dir": "" if selected_row is None else str(selected_row["cache_dir"]),
                "selected_calc_hash": "" if selected_row is None else str(selected_row["calc_hash"]),
            }
        )
    df_source_summary = pd.DataFrame(source_summary_rows)
    return {
        "dominant_hash": dominant_hash,
        "dominant_payload": dominant_payload,
        "signature_counts": sig_counts,
        "selected_by_pid": selected_by_pid,
        "source_summary": df_source_summary,
        "ignored_nonmatching_cache_rows": int((df_manifest["signature_hash"] != dominant_hash).sum()),
    }


def _get_all_remote_pids(one, tag=None, page_size=500):
    if tag:
        eids, _ = one.search(tag=tag, details=True, query_type="remote")
        pids = []
        for eid in eids:
            insertions = one.alyx.rest("insertions", "list", session=eid)
            records = insertions.get("results", []) if isinstance(insertions, dict) else insertions
            pids.extend([str(ins["id"]) for ins in list(records or []) if ins.get("id")])
        return list(dict.fromkeys(pids))

    all_pids = []
    offset = 0
    while True:
        batch = one.alyx.rest("insertions", "list", limit=page_size, offset=offset)
        records = batch.get("results", []) if isinstance(batch, dict) else batch
        records = list(records or [])
        if not records:
            break
        all_pids.extend([str(ins["id"]) for ins in records if ins.get("id")])
        if len(records) < page_size:
            break
        offset += len(records)
    return list(dict.fromkeys(all_pids))


def _read_pid_cache(cache_dir):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    df_neurons = pd.read_csv(cache_dir / "summary_neurons.csv")
    df_pairs = pd.read_csv(cache_dir / "summary_pairs.csv")
    return manifest, df_neurons, df_pairs


def _clean_summary_neurons(df_neurons, pid):
    required_cols = ["cluster_id", "region"]
    missing = [col for col in required_cols if col not in df_neurons.columns]
    if missing:
        raise RuntimeError(f"summary_neurons.csv is missing columns: {missing}")

    for spec in VARIABLE_SPECS:
        for col in (spec["mean_col"], spec["odd_col"], spec["even_col"]):
            if col not in df_neurons.columns:
                raise RuntimeError(f"summary_neurons.csv is missing column '{col}'.")

    df = df_neurons.copy()
    df["pid"] = str(pid)
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df[np.isfinite(df["cluster_id"])].copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["region"] = df["region"].astype(str)
    df = df[~df["region"].isin(["root", "void", "nan", "NA"])].copy()
    return df.reset_index(drop=True)


def _build_diag_rows_for_pid(df_region, pid, calc_hash, cache_dir):
    rows = []
    n_units_region = int(df_region["cluster_id"].nunique())
    region = str(df_region["region"].iloc[0])
    for spec in VARIABLE_SPECS:
        pearson_r, pearson_n = _pearsonr_with_n(df_region[spec["odd_col"]], df_region[spec["even_col"]])
        spearman_rho, spearman_n = _spearmanr_with_n(
            df_region[spec["odd_col"]], df_region[spec["even_col"]]
        )
        rows.append(
            {
                "pid": str(pid),
                "calc_hash": str(calc_hash),
                "cache_dir": str(cache_dir),
                "region": region,
                "is_diagonal": True,
                "var_x_key": spec["key"],
                "var_y_key": spec["key"],
                "var_x_name": spec["name"],
                "var_y_name": spec["name"],
                "var_x_order": int(spec["order"]),
                "var_y_order": int(spec["order"]),
                "pair_label": spec["name"],
                "n_units_region": n_units_region,
                "n_shared": int(pearson_n),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "pearson_n": int(pearson_n),
                "spearman_n": int(spearman_n),
                "reliability_x_pearson": pearson_r,
                "reliability_y_pearson": pearson_r,
                "reliability_x_spearman": spearman_rho,
                "reliability_y_spearman": spearman_rho,
                "reliability_x_n": int(pearson_n),
                "reliability_y_n": int(pearson_n),
                "pearson_total_reliability": pearson_r,
                "spearman_total_reliability": spearman_rho,
                "reliability_floor": pearson_r,
                "order_agreement_score": np.nan,
                "order_agreement_n_pairs": 0,
                "order_agreement_preserved_fraction": np.nan,
                "delay_diff_mean_ms": 0.0,
                "delay_diff_median_ms": 0.0,
                "delay_diff_std_ms": 0.0,
                "delay_diff_mad_ms": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_pid_pair_rows(df_neurons, df_pairs, pid, calc_hash, cache_dir):
    region_counts = (
        df_neurons.groupby("region", as_index=False)
        .agg(n_units_region=("cluster_id", "nunique"))
        .sort_values("region")
        .reset_index(drop=True)
    )

    df_pairs_out = df_pairs.copy()
    if not df_pairs_out.empty:
        df_pairs_out["pid"] = str(pid)
        df_pairs_out["calc_hash"] = str(calc_hash)
        df_pairs_out["cache_dir"] = str(cache_dir)
        df_pairs_out["is_diagonal"] = False
        df_pairs_out["var_x_order"] = df_pairs_out["var_x_key"].map(
            lambda key: VARIABLE_BY_KEY[str(key)]["order"]
        )
        df_pairs_out["var_y_order"] = df_pairs_out["var_y_key"].map(
            lambda key: VARIABLE_BY_KEY[str(key)]["order"]
        )
        df_pairs_out["pearson_n"] = pd.to_numeric(df_pairs_out.get("n_shared"), errors="coerce").fillna(0).astype(int)
        df_pairs_out["spearman_n"] = df_pairs_out["pearson_n"]
        df_pairs_out["pearson_total_reliability"] = df_pairs_out.apply(
            lambda row: _calc_total_reliability(
                row.get("reliability_x_pearson"),
                row.get("reliability_y_pearson"),
            ),
            axis=1,
        )
        df_pairs_out["spearman_total_reliability"] = df_pairs_out.apply(
            lambda row: _calc_total_reliability(
                row.get("reliability_x_spearman"),
                row.get("reliability_y_spearman"),
            ),
            axis=1,
        )
        df_pairs_out = df_pairs_out.merge(region_counts, on="region", how="left")

    diag_frames = []
    for region, df_region in df_neurons.groupby("region", sort=True):
        diag_frames.append(_build_diag_rows_for_pid(df_region, pid, calc_hash, cache_dir))
    df_diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()

    if df_pairs_out.empty:
        df_out = df_diag
    elif df_diag.empty:
        df_out = df_pairs_out
    else:
        df_out = pd.concat([df_pairs_out, df_diag], ignore_index=True, sort=False)

    return df_out.reset_index(drop=True), region_counts


def _compute_pair_metrics(df_region, spec_x, spec_y, is_diagonal):
    x_mean = df_region[spec_x["mean_col"]]
    y_mean = df_region[spec_y["mean_col"]]
    rel_x_p, rel_x_n = _pearsonr_with_n(df_region[spec_x["odd_col"]], df_region[spec_x["even_col"]])
    rel_y_p, rel_y_n = _pearsonr_with_n(df_region[spec_y["odd_col"]], df_region[spec_y["even_col"]])
    rel_x_s, _rel_x_s_n = _spearmanr_with_n(df_region[spec_x["odd_col"]], df_region[spec_x["even_col"]])
    rel_y_s, _rel_y_s_n = _spearmanr_with_n(df_region[spec_y["odd_col"]], df_region[spec_y["even_col"]])

    if is_diagonal:
        pearson_r = rel_x_p
        spearman_rho = rel_x_s
        pearson_n = int(rel_x_n)
        spearman_n = int(rel_x_n)
        order_score, order_n_pairs, order_preserved = np.nan, 0, np.nan
        delay_diff_mean_ms = 0.0
        delay_diff_median_ms = 0.0
        delay_diff_std_ms = 0.0
        delay_diff_mad_ms = 0.0
        reliability_floor = rel_x_p
    else:
        pearson_r, pearson_n = _pearsonr_with_n(x_mean, y_mean)
        spearman_rho, spearman_n = _spearmanr_with_n(x_mean, y_mean)
        order_score, order_n_pairs, order_preserved = _order_agreement_score(x_mean, y_mean)
        diff_vals = (
            pd.to_numeric(x_mean, errors="coerce") - pd.to_numeric(y_mean, errors="coerce")
        ).to_numpy(dtype=float)
        diff_vals = diff_vals[np.isfinite(diff_vals)]
        if diff_vals.size == 0:
            delay_diff_mean_ms = np.nan
            delay_diff_median_ms = np.nan
            delay_diff_std_ms = np.nan
            delay_diff_mad_ms = np.nan
        else:
            delay_diff_mean_ms = float(np.nanmean(diff_vals))
            delay_diff_median_ms = float(np.nanmedian(diff_vals))
            delay_diff_std_ms = float(np.nanstd(diff_vals))
            delay_diff_mad_ms = _median_abs_dev(diff_vals)
        reliability_floor = _calc_total_reliability(rel_x_p, rel_y_p)

    return {
        "n_shared": int(pearson_n),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "pearson_n": int(pearson_n),
        "spearman_n": int(spearman_n),
        "reliability_x_pearson": rel_x_p,
        "reliability_y_pearson": rel_y_p,
        "reliability_x_spearman": rel_x_s,
        "reliability_y_spearman": rel_y_s,
        "reliability_x_n": int(rel_x_n),
        "reliability_y_n": int(rel_y_n),
        "pearson_total_reliability": _calc_total_reliability(rel_x_p, rel_y_p),
        "spearman_total_reliability": _calc_total_reliability(rel_x_s, rel_y_s),
        "reliability_floor": reliability_floor,
        "order_agreement_score": order_score,
        "order_agreement_n_pairs": int(order_n_pairs),
        "order_agreement_preserved_fraction": order_preserved,
        "delay_diff_mean_ms": delay_diff_mean_ms,
        "delay_diff_median_ms": delay_diff_median_ms,
        "delay_diff_std_ms": delay_diff_std_ms,
        "delay_diff_mad_ms": delay_diff_mad_ms,
    }


def _build_pooled_region_pair_rows(df_neurons_all):
    rows = []
    grouped = list(df_neurons_all.groupby("region", sort=True))
    region_iter = tqdm(
        grouped,
        total=len(grouped),
        desc="Computing pooled region pairs",
        unit="region",
        disable=not SHOW_PROGRESS,
        dynamic_ncols=True,
    )
    try:
        for region, df_region in region_iter:
            n_units_region = int(df_region["cluster_id"].nunique())
            n_pids_region = int(df_region["pid"].nunique())
            for spec_x, spec_y, is_diagonal in PAIR_SPECS:
                metrics = _compute_pair_metrics(df_region, spec_x, spec_y, is_diagonal=is_diagonal)
                rows.append(
                    {
                        "region": str(region),
                        "is_diagonal": bool(is_diagonal),
                        "var_x_key": spec_x["key"],
                        "var_y_key": spec_y["key"],
                        "var_x_name": spec_x["name"],
                        "var_y_name": spec_y["name"],
                        "var_x_order": int(spec_x["order"]),
                        "var_y_order": int(spec_y["order"]),
                        "pair_label": spec_x["name"]
                        if is_diagonal
                        else f"{spec_x['name']} vs {spec_y['name']}",
                        "n_units_region": n_units_region,
                        "n_pids": n_pids_region,
                        **metrics,
                    }
                )
    finally:
        if hasattr(region_iter, "close"):
            region_iter.close()
    return pd.DataFrame(rows)


def _aggregate_pidmean_pairs(df_pid_pairs):
    rows = []
    if df_pid_pairs.empty:
        return pd.DataFrame(rows)
    group_cols = [
        "region",
        "is_diagonal",
        "var_x_key",
        "var_y_key",
        "var_x_name",
        "var_y_name",
        "var_x_order",
        "var_y_order",
        "pair_label",
    ]
    grouped = list(df_pid_pairs.groupby(group_cols, sort=True))
    pair_iter = tqdm(
        grouped,
        total=len(grouped),
        desc="Computing PID-first region means",
        unit="pair",
        disable=not SHOW_PROGRESS,
        dynamic_ncols=True,
    )
    try:
        for group_key, df_group in pair_iter:
            (
                region,
                is_diagonal,
                var_x_key,
                var_y_key,
                var_x_name,
                var_y_name,
                var_x_order,
                var_y_order,
                pair_label,
            ) = group_key
            rows.append(
                {
                    "region": str(region),
                    "is_diagonal": bool(is_diagonal),
                    "var_x_key": str(var_x_key),
                    "var_y_key": str(var_y_key),
                    "var_x_name": str(var_x_name),
                    "var_y_name": str(var_y_name),
                    "var_x_order": int(var_x_order),
                    "var_y_order": int(var_y_order),
                    "pair_label": str(pair_label),
                    "n_pids": int(df_group["pid"].nunique()),
                    "n_units_region": float(np.nanmean(pd.to_numeric(df_group["n_units_region"], errors="coerce"))),
                    "n_shared_mean": _safe_mean(df_group.get("n_shared")),
                    "pearson_r": _safe_mean(df_group.get("pearson_r")),
                    "spearman_rho": _safe_mean(df_group.get("spearman_rho")),
                    "pearson_n_mean": _safe_mean(df_group.get("pearson_n")),
                    "spearman_n_mean": _safe_mean(df_group.get("spearman_n")),
                    "pearson_pid_count": int(pd.to_numeric(df_group.get("pearson_r"), errors="coerce").notna().sum()),
                    "spearman_pid_count": int(pd.to_numeric(df_group.get("spearman_rho"), errors="coerce").notna().sum()),
                    "reliability_x_pearson": _safe_mean(df_group.get("reliability_x_pearson")),
                    "reliability_y_pearson": _safe_mean(df_group.get("reliability_y_pearson")),
                    "reliability_x_spearman": _safe_mean(df_group.get("reliability_x_spearman")),
                    "reliability_y_spearman": _safe_mean(df_group.get("reliability_y_spearman")),
                    "reliability_x_pid_count": int(
                        pd.to_numeric(df_group.get("reliability_x_pearson"), errors="coerce").notna().sum()
                    ),
                    "reliability_y_pid_count": int(
                        pd.to_numeric(df_group.get("reliability_y_pearson"), errors="coerce").notna().sum()
                    ),
                    "pearson_total_reliability": _safe_mean(df_group.get("pearson_total_reliability")),
                    "spearman_total_reliability": _safe_mean(df_group.get("spearman_total_reliability")),
                    "pearson_total_rel_pid_count": int(
                        pd.to_numeric(df_group.get("pearson_total_reliability"), errors="coerce").notna().sum()
                    ),
                    "spearman_total_rel_pid_count": int(
                        pd.to_numeric(df_group.get("spearman_total_reliability"), errors="coerce").notna().sum()
                    ),
                    "reliability_floor": _safe_mean(df_group.get("reliability_floor")),
                    "order_agreement_score": _safe_mean(df_group.get("order_agreement_score")),
                    "order_agreement_n_pairs_mean": _safe_mean(df_group.get("order_agreement_n_pairs")),
                    "order_agreement_preserved_fraction": _safe_mean(
                        df_group.get("order_agreement_preserved_fraction")
                    ),
                    "delay_diff_mean_ms": _safe_mean(df_group.get("delay_diff_mean_ms")),
                    "delay_diff_median_ms": _safe_mean(df_group.get("delay_diff_median_ms")),
                    "delay_diff_std_ms": _safe_mean(df_group.get("delay_diff_std_ms")),
                    "delay_diff_mad_ms": _safe_mean(df_group.get("delay_diff_mad_ms")),
                }
            )
    finally:
        if hasattr(pair_iter, "close"):
            pair_iter.close()
    return pd.DataFrame(rows)


def _sort_region_tables(df):
    if df is None or df.empty or "category_rank" not in df.columns:
        return df
    return df.sort_values(["category_rank", "allen_order", "region"]).reset_index(drop=True)


def main():
    print("Running 16_ibl_seq_batch_comp.py")
    if not SOURCE_CACHE_ROOT.exists():
        raise RuntimeError(f"Source cache root does not exist: {SOURCE_CACHE_ROOT}")

    _path_data, _path_fig, _path_processed, ibl_cache = setup_paths(BASE_PATH)

    print("Scanning saved 15 caches...")
    df_manifest = _scan_source_manifests(SOURCE_CACHE_ROOT)
    source_selection = _select_source_caches(df_manifest)
    dominant_hash = source_selection["dominant_hash"]
    selected_by_pid = dict(source_selection["selected_by_pid"])
    df_source_summary = source_selection["source_summary"].copy()
    print(
        "Dominant 15 source signature:",
        dominant_hash,
        f"| selected PIDs with matching cache={len(selected_by_pid)}",
        f"| ignored non-matching cache rows={source_selection['ignored_nonmatching_cache_rows']}",
    )

    print(f"Initializing ONE (mode={ONE_PREFERRED_MODE})...")
    one = init_one(ibl_cache, mode=ONE_PREFERRED_MODE)
    ba = AllenAtlas() if AllenAtlas is not None else None
    br = BrainRegions() if BrainRegions is not None else None

    print(f"Resolving target PIDs from Alyx tag '{ALL_PIDS_TAG}'...")
    target_pids = _get_all_remote_pids(one, tag=ALL_PIDS_TAG)
    if not target_pids:
        raise RuntimeError(f"No target PIDs resolved for tag '{ALL_PIDS_TAG}'.")
    print(f"Resolved {len(target_pids)} target PID(s).")

    summary_rows = []
    neurons_frames = []
    pid_pairs_frames = []
    pid_region_count_frames = []

    status_iter = tqdm(
        target_pids,
        total=len(target_pids),
        desc="Processing target PIDs",
        unit="pid",
        disable=not SHOW_PROGRESS,
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for pid in status_iter:
            pid = str(pid)
            df_source_pid = df_source_summary.loc[df_source_summary["pid"] == pid]
            matching_cache_count = int(df_source_pid["matching_cache_count"].iloc[0]) if not df_source_pid.empty else 0
            nonmatching_cache_count = int(df_source_pid["nonmatching_cache_count"].iloc[0]) if not df_source_pid.empty else 0

            if pid in selected_by_pid:
                selected = dict(selected_by_pid[pid])
                cache_dir = Path(selected["cache_dir"])
                try:
                    manifest, df_neurons_raw, df_pairs_raw = _read_pid_cache(cache_dir)
                    df_neurons = _clean_summary_neurons(df_neurons_raw, pid)
                    df_pid_pairs, df_pid_region_counts = _build_pid_pair_rows(
                        df_neurons,
                        df_pairs_raw,
                        pid,
                        selected["calc_hash"],
                        cache_dir,
                    )
                    neurons_frames.append(df_neurons)
                    pid_pairs_frames.append(df_pid_pairs)
                    df_pid_region_counts = df_pid_region_counts.copy()
                    df_pid_region_counts["pid"] = pid
                    df_pid_region_counts["calc_hash"] = selected["calc_hash"]
                    df_pid_region_counts["cache_dir"] = str(cache_dir)
                    pid_region_count_frames.append(df_pid_region_counts)
                    summary_rows.append(
                        {
                            "pid": pid,
                            "eid": str(manifest.get("eid", "")),
                            "status": "ok",
                            "reason": "",
                            "reason_group": "ok",
                            "cache_match": True,
                            "matching_cache_count": matching_cache_count,
                            "nonmatching_cache_count": nonmatching_cache_count,
                            "cache_dir": str(cache_dir),
                            "calc_hash": str(selected["calc_hash"]),
                            "eligible_units": int(df_neurons["cluster_id"].nunique()),
                        }
                    )
                    if hasattr(status_iter, "set_postfix_str"):
                        ok_count = sum(1 for row in summary_rows if row["status"] == "ok")
                        status_iter.set_postfix_str(f"ok={ok_count}")
                except Exception as exc:
                    reason = f"Cache read/transform failed: {exc}"
                    summary_rows.append(
                        {
                            "pid": pid,
                            "eid": str(selected.get("eid", "")),
                            "status": "failed",
                            "reason": reason,
                            "reason_group": _normalize_reason_group(reason),
                            "cache_match": False,
                            "matching_cache_count": matching_cache_count,
                            "nonmatching_cache_count": nonmatching_cache_count,
                            "cache_dir": str(cache_dir),
                            "calc_hash": str(selected["calc_hash"]),
                            "eligible_units": 0,
                        }
                    )
                continue

            if nonmatching_cache_count > 0:
                reason = "Only non-matching 15 cache folders are available for this PID."
                summary_rows.append(
                    {
                        "pid": pid,
                        "eid": "",
                        "status": "config_mismatch",
                        "reason": reason,
                        "reason_group": _normalize_reason_group(reason),
                        "cache_match": False,
                        "matching_cache_count": matching_cache_count,
                        "nonmatching_cache_count": nonmatching_cache_count,
                        "cache_dir": "",
                        "calc_hash": "",
                        "eligible_units": 0,
                    }
                )
                continue

            try:
                probe = _probe_pid_status(pid, one, ba, br)
                reason = "PID has no matching 15 cache even though the source data pass availability checks."
                summary_rows.append(
                    {
                        "pid": pid,
                        "eid": str(probe.get("eid", "")),
                        "status": "failed",
                        "reason": reason,
                        "reason_group": _normalize_reason_group(reason),
                        "cache_match": False,
                        "matching_cache_count": 0,
                        "nonmatching_cache_count": 0,
                        "cache_dir": "",
                        "calc_hash": "",
                        "eligible_units": int(probe.get("eligible_units", 0)),
                    }
                )
            except _PidSkipError as exc:
                reason = str(exc)
                summary_rows.append(
                    {
                        "pid": pid,
                        "eid": "",
                        "status": "skipped",
                        "reason": reason,
                        "reason_group": _normalize_reason_group(reason),
                        "cache_match": False,
                        "matching_cache_count": 0,
                        "nonmatching_cache_count": 0,
                        "cache_dir": "",
                        "calc_hash": "",
                        "eligible_units": 0,
                    }
                )
            except Exception as exc:
                reason = str(exc)
                summary_rows.append(
                    {
                        "pid": pid,
                        "eid": "",
                        "status": "failed",
                        "reason": reason,
                        "reason_group": _normalize_reason_group(reason),
                        "cache_match": False,
                        "matching_cache_count": 0,
                        "nonmatching_cache_count": 0,
                        "cache_dir": "",
                        "calc_hash": "",
                        "eligible_units": 0,
                    }
                )
    finally:
        if hasattr(status_iter, "close"):
            status_iter.close()

    summary_pids = pd.DataFrame(summary_rows).sort_values(["status", "pid"]).reset_index(drop=True)
    status_counts = summary_pids["status"].value_counts().to_dict()
    print(f"PID status summary: {status_counts}")

    if neurons_frames:
        summary_neurons_all = pd.concat(neurons_frames, ignore_index=True, sort=False)
    else:
        summary_neurons_all = pd.DataFrame(columns=["pid", "cluster_id", "region"])

    if pid_pairs_frames:
        summary_pid_pairs = pd.concat(pid_pairs_frames, ignore_index=True, sort=False)
    else:
        summary_pid_pairs = pd.DataFrame()

    if pid_region_count_frames:
        summary_pid_region_counts = pd.concat(pid_region_count_frames, ignore_index=True, sort=False)
    else:
        summary_pid_region_counts = pd.DataFrame(columns=["pid", "region", "n_units_region"])

    region_meta = _build_region_meta(summary_pid_region_counts.get("region", pd.Series(dtype=object)))
    if region_meta.empty and not summary_neurons_all.empty:
        region_meta = _build_region_meta(summary_neurons_all["region"])

    if not summary_pid_region_counts.empty:
        summary_pid_region_counts = summary_pid_region_counts.merge(region_meta, on="region", how="left")
        summary_pid_region_counts = _sort_region_tables(summary_pid_region_counts)

    if not summary_pid_pairs.empty:
        summary_pid_pairs = summary_pid_pairs.merge(region_meta, on="region", how="left")
        summary_pid_pairs = _sort_region_tables(summary_pid_pairs)

    if not summary_pid_region_counts.empty:
        summary_region_counts = (
            summary_pid_region_counts.groupby("region", as_index=False)
            .agg(
                n_pids=("pid", "nunique"),
                n_units_total=("n_units_region", "sum"),
                n_units_mean_per_pid=("n_units_region", "mean"),
                n_units_median_per_pid=("n_units_region", "median"),
                n_units_min_per_pid=("n_units_region", "min"),
                n_units_max_per_pid=("n_units_region", "max"),
            )
            .merge(region_meta, on="region", how="left")
        )
        summary_region_counts = _sort_region_tables(summary_region_counts)
    else:
        summary_region_counts = pd.DataFrame(
            columns=[
                "region",
                "n_pids",
                "n_units_total",
                "n_units_mean_per_pid",
                "n_units_median_per_pid",
                "allen_color",
                "allen_order",
                "category",
                "category_rank",
            ]
        )

    if not summary_neurons_all.empty:
        summary_region_pairs_pooled = _build_pooled_region_pair_rows(summary_neurons_all)
        summary_region_pairs_pooled = summary_region_pairs_pooled.merge(region_meta, on="region", how="left")
        summary_region_pairs_pooled = _sort_region_tables(summary_region_pairs_pooled)
    else:
        summary_region_pairs_pooled = pd.DataFrame()

    if not summary_pid_pairs.empty:
        summary_region_pairs_pidmean = _aggregate_pidmean_pairs(summary_pid_pairs)
        summary_region_pairs_pidmean = summary_region_pairs_pidmean.merge(region_meta, on="region", how="left")
        summary_region_pairs_pidmean = _sort_region_tables(summary_region_pairs_pidmean)
    else:
        summary_region_pairs_pidmean = pd.DataFrame()

    summary_pid_reason_counts = (
        summary_pids.groupby(["status", "reason_group"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "n_pids"})
        .sort_values(["status", "n_pids", "reason_group"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    metadata = {
        "batch_version": BATCH_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_cache_root": str(SOURCE_CACHE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "all_pids_tag": ALL_PIDS_TAG,
        "one_preferred_mode": ONE_PREFERRED_MODE,
        "source_calc_version": SOURCE_CALC_VERSION,
        "source_calc_label_min": SOURCE_CALC_LABEL_MIN,
        "source_min_region_neurons": SOURCE_MIN_REGION_NEURONS,
        "source_signature_hash": dominant_hash,
        "source_signature_payload": source_selection["dominant_payload"],
        "source_signature_counts": source_selection["signature_counts"].to_dict(orient="records"),
        "ignored_nonmatching_cache_rows": int(source_selection["ignored_nonmatching_cache_rows"]),
        "n_target_pids": int(len(target_pids)),
        "n_ok_pids": int((summary_pids["status"] == "ok").sum()),
        "n_config_mismatch_pids": int((summary_pids["status"] == "config_mismatch").sum()),
        "n_not_available_pids": int(summary_pids["status"].isin(["skipped", "failed"]).sum()),
        "status_counts": status_counts,
        "variable_specs": VARIABLE_SPECS,
        "category_order": CATEGORY_ORDER,
    }

    print(f"Saving batch tables to {OUTPUT_ROOT}")
    with open(OUTPUT_ROOT / "summary_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    (OUTPUT_ROOT / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_table_bundle(summary_pids, "summary_pids", OUTPUT_ROOT)
    _write_table_bundle(summary_pid_reason_counts, "summary_pid_reason_counts", OUTPUT_ROOT)
    _write_table_bundle(summary_region_counts, "summary_region_counts", OUTPUT_ROOT)
    _write_table_bundle(summary_pid_region_counts, "summary_pid_region_counts", OUTPUT_ROOT)
    _write_table_bundle(summary_pid_pairs, "summary_pid_pairs", OUTPUT_ROOT)
    _write_table_bundle(summary_region_pairs_pooled, "summary_region_pairs_pooled", OUTPUT_ROOT)
    _write_table_bundle(summary_region_pairs_pidmean, "summary_region_pairs_pidmean", OUTPUT_ROOT)

    print("Finished 16 batch computation.")
    print(f"Saved metadata and tables under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
