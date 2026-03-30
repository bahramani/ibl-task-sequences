from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from one.alf import io as alfio

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _NullTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, n=1):
            return None

        def set_postfix_str(self, s="", refresh=True):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable=iterable, **kwargs)

BASE_PATH = Path(__file__).resolve().parents[1]
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from utils import analysis as ana_utils


RAW_DIR = BASE_PATH / "data" / "raw"
DASHBOARD_CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
PROCESSED_DIR = BASE_PATH / "data" / "processed"
SUMMARY_DIR = PROCESSED_DIR / "24_calc_dashboard_event_response_z"
REST_DIR = RAW_DIR / ".rest"

DEFAULT_CONFIG = {
    "BIN_SIZE": 0.005,
    "PSTH_WINDOW_START": -1.0,
    "PSTH_WINDOW_END": 1.0,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 10,
    "MIN_TRIALS_SPLIT": 5,
    "RESPONSIVE_ZSCORE_SOURCE": "smooth",
}

SPIKE_SORTER = "pykilosort"
AUDITORY_FEEDBACK_ACRONYMS = frozenset({"AUDp", "AUDv", "AUDd"})

BASE_RESPONSE_TARGET_COLUMNS = (
    "response_zmean_stimOn_times",
    "response_zmean_stimOn_times_odd",
    "response_zmean_stimOn_times_even",
    "response_zmean_firstMovement_times",
    "response_zmean_firstMovement_times_odd",
    "response_zmean_firstMovement_times_even",
    "response_zmean_feedback_times",
    "response_zmean_feedback_times_odd",
    "response_zmean_feedback_times_even",
    "response_zmean_wh_brief_times_spont",
    "response_zmean_wh_brief_times_spont_odd",
    "response_zmean_wh_brief_times_spont_even",
)
AUDITORY_RESPONSE_TARGET_COLUMNS = (
    "response_zmean_feedback_correct_times",
    "response_zmean_feedback_correct_times_odd",
    "response_zmean_feedback_correct_times_even",
    "response_zmean_feedback_incorrect_times",
    "response_zmean_feedback_incorrect_times_odd",
    "response_zmean_feedback_incorrect_times_even",
)
AUDITORY_DELAY_TARGET_COLUMNS = (
    "delay_feedback_correct_times",
    "delay_feedback_correct_times_odd",
    "delay_feedback_correct_times_even",
    "delay_feedback_incorrect_times",
    "delay_feedback_incorrect_times_odd",
    "delay_feedback_incorrect_times_even",
)
TARGET_COLUMNS = (
    *BASE_RESPONSE_TARGET_COLUMNS,
    *AUDITORY_RESPONSE_TARGET_COLUMNS,
    *AUDITORY_DELAY_TARGET_COLUMNS,
)


@dataclass(frozen=True)
class EventSpec:
    event_name: str
    response_window: tuple[float, float]
    baseline_window: tuple[float, float]
    source: str

    @property
    def full_col(self) -> str:
        return f"response_zmean_{self.event_name}"

    @property
    def odd_col(self) -> str:
        return f"{self.full_col}_odd"

    @property
    def even_col(self) -> str:
        return f"{self.full_col}_even"


EVENT_SPECS = (
    EventSpec(
        event_name="stimOn_times",
        response_window=(0.02, 0.35),
        baseline_window=(-0.2, 0.0),
        source="task",
    ),
    EventSpec(
        event_name="firstMovement_times",
        response_window=(-0.1, 0.2),
        baseline_window=(-0.3, -0.1),
        source="task",
    ),
    EventSpec(
        event_name="feedback_times",
        response_window=(-0.1, 0.2),
        baseline_window=(-0.3, -0.1),
        source="task",
    ),
    EventSpec(
        event_name="wh_brief_times_spont",
        response_window=(0.0, 0.4),
        baseline_window=(-0.2, 0.0),
        source="whisk",
    ),
)
AUDITORY_FEEDBACK_EVENT_SPECS = (
    EventSpec(
        event_name="feedback_correct_times",
        response_window=(0.0, 0.1),
        baseline_window=(-0.2, 0.0),
        source="task",
    ),
    EventSpec(
        event_name="feedback_incorrect_times",
        response_window=(0.0, 0.1),
        baseline_window=(-0.2, 0.0),
        source="task",
    ),
)
SUMMARY_EVENT_NAMES = tuple(spec.event_name for spec in EVENT_SPECS) + tuple(
    spec.event_name for spec in AUDITORY_FEEDBACK_EVENT_SPECS
)


def _coerce_pid_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-neuron event-response z-mean metrics offline from existing "
            "dashboard caches and local IBL ALF files."
        )
    )
    parser.add_argument(
        "--pids",
        nargs="*",
        default=None,
        help="Optional PID list. Accepts space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the target columns already exist in the dashboard cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PID workers. Default is 1 to limit memory pressure.",
    )
    args = parser.parse_args()
    args.pids = _coerce_pid_list(args.pids)
    args.workers = max(int(args.workers), 1)
    return args


def _extract_insertions_from_json(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
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


def build_pid_lookup(rest_dir: Path, target_pids: set[str]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    if not target_pids:
        return mapping
    if not rest_dir.exists():
        return mapping

    rest_paths = sorted(path for path in rest_dir.iterdir() if path.is_file())
    for rest_path in tqdm(rest_paths, desc="REST scan", unit="file"):
        try:
            text = rest_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if '"session_info"' not in text and '"probe_insertion"' not in text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue

        for record in _extract_insertions_from_json(payload):
            pid = str(record.get("id", "")).strip()
            if pid not in target_pids or pid in mapping:
                continue
            session_info = record.get("session_info", {}) or {}
            start_time = str(session_info.get("start_time", "")).strip()
            date = start_time[:10] if start_time else ""
            number = session_info.get("number", None)
            try:
                number = int(number)
            except Exception:
                number = None

            pname = str(record.get("name", "")).strip()
            if not pname or not date or number is None:
                continue

            lab = str(session_info.get("lab", "")).strip()
            subject = str(session_info.get("subject", "")).strip()
            if not lab or not subject:
                continue

            mapping[pid] = {
                "pid": pid,
                "eid": str(record.get("session", "")).strip(),
                "pname": pname,
                "lab": lab,
                "subject": subject,
                "date": date,
                "number": number,
                "source": "rest",
            }
            if len(mapping) == len(target_pids):
                return mapping
    return mapping


def _date_only(value: Any) -> str:
    text = str(value or "").strip()
    return text[:10] if text else ""


def _load_single_attribute(alf_dir: Path, object_name: str, attribute: str) -> Any:
    bunch = alfio.load_object(alf_dir, object_name, attribute=attribute)
    return bunch[attribute]


def _find_probe_revision_dir(session_path: Path, pname: str) -> Path:
    probe_root = session_path / "alf" / str(pname) / SPIKE_SORTER
    if not probe_root.exists():
        raise FileNotFoundError(f"Probe path not found: {probe_root}")

    required = ("spikes.times.npy", "spikes.clusters.npy")
    revision_dirs = []
    for child in probe_root.iterdir():
        if child.is_dir() and child.name.startswith("#") and child.name.endswith("#"):
            if all((child / filename).exists() for filename in required):
                revision_dirs.append(child)

    if revision_dirs:
        return sorted(revision_dirs, key=lambda p: p.name)[-1]

    if all((probe_root / filename).exists() for filename in required):
        return probe_root

    raise FileNotFoundError(
        f"No spike-sorting revision with required files found under {probe_root}"
    )


def _fallback_probe_lookup_from_disk(
    pid: str,
    cache_payload: dict[str, Any],
    raw_dir: Path,
) -> dict[str, Any] | None:
    meta = cache_payload.get("meta", {}) or {}
    lab = str(meta.get("lab", "")).strip()
    subject = str(meta.get("subject", "")).strip()
    date = _date_only(meta.get("date"))
    if not lab or not subject or not date:
        return None

    date_dir = raw_dir / lab / "Subjects" / subject / date
    if not date_dir.exists():
        return None

    expected_cluster_ids = np.asarray(cache_payload.get("cluster_ids", []))
    expected_count = int(expected_cluster_ids.size)
    if expected_count <= 0:
        return None

    matches: list[dict[str, Any]] = []
    for session_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
        try:
            number = int(session_dir.name)
        except Exception:
            continue
        alf_dir = session_dir / "alf"
        if not alf_dir.exists():
            continue
        for probe_dir in sorted(path for path in alf_dir.iterdir() if path.is_dir()):
            pname = probe_dir.name
            try:
                revision_dir = _find_probe_revision_dir(session_dir, pname)
                channels = _load_single_attribute(revision_dir, "clusters", "channels")
            except Exception:
                continue
            if int(np.asarray(channels).size) != expected_count:
                continue
            matches.append(
                {
                    "pid": pid,
                    "eid": str(meta.get("eid", "")).strip(),
                    "pname": pname,
                    "lab": lab,
                    "subject": subject,
                    "date": date,
                    "number": number,
                    "source": "disk_fallback",
                }
            )
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_probe_lookup(
    pid: str,
    cache_payload: dict[str, Any],
    raw_dir: Path,
    rest_lookup: dict[str, Any] | None,
) -> dict[str, Any]:
    if rest_lookup is not None:
        session_path = (
            raw_dir
            / rest_lookup["lab"]
            / "Subjects"
            / rest_lookup["subject"]
            / rest_lookup["date"]
            / str(int(rest_lookup["number"])).zfill(3)
        )
        if session_path.exists():
            out = dict(rest_lookup)
            out["session_path"] = session_path
            return out

    fallback = _fallback_probe_lookup_from_disk(pid, cache_payload, raw_dir)
    if fallback is None:
        raise RuntimeError("Could not resolve probe metadata from local .rest cache or disk fallback.")

    session_path = (
        raw_dir
        / fallback["lab"]
        / "Subjects"
        / fallback["subject"]
        / fallback["date"]
        / str(int(fallback["number"])).zfill(3)
    )
    fallback["session_path"] = session_path
    return fallback


def _safe_config(cache_payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_calc = cache_payload.get("config_calc", {}) or {}
    config.update(config_calc)
    delay_windows = dict(config.get("DELAY_WINDOWS", {}) or {})
    delay_windows.setdefault("feedback_correct_times", (0.0, 0.1))
    delay_windows.setdefault("feedback_incorrect_times", (0.0, 0.1))
    config["DELAY_WINDOWS"] = delay_windows
    return config


def _trial_df_from_cache(cache_payload: dict[str, Any]) -> pd.DataFrame:
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
    df = df.sort_values("trial_idx").reset_index(drop=True)
    return df


def _build_event_payloads(
    cache_payload: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    trials = _trial_df_from_cache(cache_payload)
    payloads: dict[str, dict[str, np.ndarray]] = {}

    if not trials.empty:
        trial_idx = pd.to_numeric(trials["trial_idx"], errors="coerce").to_numpy()
        trial_idx = np.where(np.isfinite(trial_idx), trial_idx, np.arange(len(trials), dtype=float))
        trial_idx = trial_idx.astype(int)
        contrast = None
        if "contrast" in trials.columns:
            contrast = pd.to_numeric(trials["contrast"], errors="coerce").to_numpy(dtype=float)

        if {"stimOn_times", "contrast"}.issubset(trials.columns):
            stim = pd.to_numeric(trials["stimOn_times"], errors="coerce").to_numpy(dtype=float)
            stim_contrast = np.asarray(contrast, dtype=float)
            mask = np.isfinite(stim) & np.isfinite(stim_contrast) & (stim_contrast > 0)
            payloads["stimOn_times"] = {
                "events": np.asarray(stim[mask], dtype=float),
                "split_index": trial_idx[mask].astype(int),
                "contrasts": np.asarray(stim_contrast[mask], dtype=float),
            }

        for event_name in ("firstMovement_times", "feedback_times"):
            if event_name not in trials.columns:
                continue
            values = pd.to_numeric(trials[event_name], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            payloads[event_name] = {
                "events": np.asarray(values[mask], dtype=float),
                "split_index": trial_idx[mask].astype(int),
                "contrasts": (
                    np.asarray(contrast[mask], dtype=float)
                    if contrast is not None
                    else np.ones(int(np.sum(mask)), dtype=float)
                ),
            }

        if {"feedback_times", "correct_response"}.issubset(trials.columns):
            feedback = pd.to_numeric(trials["feedback_times"], errors="coerce").to_numpy(dtype=float)
            correct_response = trials["correct_response"]
            correct_mask = np.isfinite(feedback) & correct_response.eq(True).to_numpy(dtype=bool)
            incorrect_mask = np.isfinite(feedback) & correct_response.eq(False).to_numpy(dtype=bool)
            for event_name, mask in (
                ("feedback_correct_times", correct_mask),
                ("feedback_incorrect_times", incorrect_mask),
            ):
                payloads[event_name] = {
                    "events": np.asarray(feedback[mask], dtype=float),
                    "split_index": trial_idx[mask].astype(int),
                    "contrasts": (
                        np.asarray(contrast[mask], dtype=float)
                        if contrast is not None
                        else np.ones(int(np.sum(mask)), dtype=float)
                    ),
                }

    wh_events_by_period = cache_payload.get("wh_events_by_period", {}) or {}
    wh_brief = np.asarray(wh_events_by_period.get("wh_brief_times_spont", np.array([])), dtype=float)
    wh_brief = np.sort(wh_brief[np.isfinite(wh_brief)])
    payloads["wh_brief_times_spont"] = {
        "events": wh_brief.astype(float),
        "split_index": np.arange(wh_brief.size, dtype=int),
        "contrasts": np.ones(wh_brief.size, dtype=float),
    }
    return payloads


def _zscore_trace(arr: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    baseline_mask = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    if arr.size == 0 or baseline_mask.size != arr.size:
        return np.full(arr.shape, np.nan, dtype=float)
    ref = arr[baseline_mask]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return (arr - mu) / sd


def _response_zmean_from_psth(
    psth_entry: dict[str, np.ndarray] | None,
    bin_centers: np.ndarray | None,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    zscore_source: str,
) -> float:
    if psth_entry is None or bin_centers is None:
        return np.nan

    if zscore_source == "raw":
        trace = np.asarray(psth_entry.get("fr_raw", np.array([])), dtype=float)
    else:
        smooth = psth_entry.get("fr_smooth", None)
        if smooth is None:
            trace = np.asarray(psth_entry.get("fr_raw", np.array([])), dtype=float)
        else:
            trace = np.asarray(smooth, dtype=float)

    centers = np.asarray(bin_centers, dtype=float)
    if trace.size == 0 or centers.size != trace.size:
        return np.nan

    baseline_mask = (centers >= baseline_window[0]) & (centers < baseline_window[1])
    response_mask = (centers >= response_window[0]) & (centers <= response_window[1])
    if not np.any(response_mask):
        return np.nan

    ztrace = _zscore_trace(trace, baseline_mask=baseline_mask)
    response_vals = ztrace[response_mask]
    response_vals = response_vals[np.isfinite(response_vals)]
    if response_vals.size == 0:
        return np.nan
    return float(np.mean(response_vals))


def _compute_event_metrics(
    spikes: Any,
    cluster_ids: np.ndarray,
    events: np.ndarray,
    split_index: np.ndarray,
    config: dict[str, Any],
    spec: EventSpec,
    zscore_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_units = int(cluster_ids.size)
    full_vals = np.full(n_units, np.nan, dtype=float)
    odd_vals = np.full(n_units, np.nan, dtype=float)
    even_vals = np.full(n_units, np.nan, dtype=float)

    min_trials = int(config.get("MIN_TRIALS", DEFAULT_CONFIG["MIN_TRIALS"]))
    min_trials_split = int(config.get("MIN_TRIALS_SPLIT", DEFAULT_CONFIG["MIN_TRIALS_SPLIT"]))
    if events.size < min_trials or n_units == 0:
        return full_vals, odd_vals, even_vals

    psth_kwargs = {
        "window_start": float(config.get("PSTH_WINDOW_START", DEFAULT_CONFIG["PSTH_WINDOW_START"])),
        "window_end": float(config.get("PSTH_WINDOW_END", DEFAULT_CONFIG["PSTH_WINDOW_END"])),
        "bin_size": float(config.get("BIN_SIZE", DEFAULT_CONFIG["BIN_SIZE"])),
        "smooth_sigma": float(config.get("SMOOTH_SIGMA", DEFAULT_CONFIG["SMOOTH_SIGMA"])),
        "show_progress": False,
        "desc": f"PSTH {spec.event_name}",
    }

    psth_full, bin_centers_full = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events,
        **psth_kwargs,
    )
    for row_idx, cid in enumerate(cluster_ids):
        full_vals[row_idx] = _response_zmean_from_psth(
            psth_full.get(int(cid)),
            bin_centers_full,
            spec.baseline_window,
            spec.response_window,
            zscore_source=zscore_source,
        )

    odd_mask = (split_index % 2) == 1
    even_mask = ~odd_mask
    events_odd = events[odd_mask]
    events_even = events[even_mask]
    if events_odd.size < min_trials_split or events_even.size < min_trials_split:
        return full_vals, odd_vals, even_vals

    psth_odd, bin_centers_odd = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events_odd,
        **psth_kwargs,
    )
    psth_even, bin_centers_even = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events_even,
        **psth_kwargs,
    )
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        odd_vals[row_idx] = _response_zmean_from_psth(
            psth_odd.get(cid_int),
            bin_centers_odd,
            spec.baseline_window,
            spec.response_window,
            zscore_source=zscore_source,
        )
        even_vals[row_idx] = _response_zmean_from_psth(
            psth_even.get(cid_int),
            bin_centers_even,
            spec.baseline_window,
            spec.response_window,
            zscore_source=zscore_source,
        )
    return full_vals, odd_vals, even_vals


def _apply_delay_units(arr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if str(config.get("DELAY_UNITS", "s")).lower().startswith("ms"):
        out *= 1000.0
    return out


def _compute_delay_metrics(
    spikes: Any,
    cluster_ids: np.ndarray,
    spike_times_by_cluster: dict[int, np.ndarray],
    events: np.ndarray,
    split_index: np.ndarray,
    contrasts: np.ndarray,
    config: dict[str, Any],
    event_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_units = int(cluster_ids.size)
    full_vals = np.full(n_units, np.nan, dtype=float)
    odd_vals = np.full(n_units, np.nan, dtype=float)
    even_vals = np.full(n_units, np.nan, dtype=float)

    min_trials = int(config.get("MIN_TRIALS", DEFAULT_CONFIG["MIN_TRIALS"]))
    min_trials_split = int(config.get("MIN_TRIALS_SPLIT", DEFAULT_CONFIG["MIN_TRIALS_SPLIT"]))
    if events.size < min_trials or n_units == 0:
        return full_vals, odd_vals, even_vals

    win_start, win_end = ana_utils.get_event_delay_window(config, event_name)
    event_config = {
        **config,
        "RESPONSIVE_WINDOW_START": float(win_start),
        "RESPONSIVE_WINDOW_END": float(win_end),
    }
    event_method = ana_utils.get_event_delay_method(config, event_name)
    psth_kwargs = {
        "window_start": float(config.get("PSTH_WINDOW_START", DEFAULT_CONFIG["PSTH_WINDOW_START"])),
        "window_end": float(config.get("PSTH_WINDOW_END", DEFAULT_CONFIG["PSTH_WINDOW_END"])),
        "bin_size": float(config.get("BIN_SIZE", DEFAULT_CONFIG["BIN_SIZE"])),
        "smooth_sigma": float(config.get("SMOOTH_SIGMA", DEFAULT_CONFIG["SMOOTH_SIGMA"])),
        "show_progress": False,
        "desc": f"PSTH delay {event_name}",
    }

    psth_full, bin_centers_full = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events,
        **psth_kwargs,
    )
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        psth_entry = psth_full.get(cid_int)
        delay, is_responsive = ana_utils.calculate_delay(
            psth_entry.get("fr_raw") if psth_entry else None,
            psth_entry.get("fr_smooth") if psth_entry else None,
            bin_centers_full,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events,
            trial_contrasts=contrasts,
            return_sign=False,
        )
        if is_responsive:
            full_vals[row_idx] = delay

    odd_mask = (split_index % 2) == 1
    even_mask = ~odd_mask
    events_odd = events[odd_mask]
    events_even = events[even_mask]
    if events_odd.size < min_trials_split or events_even.size < min_trials_split:
        return _apply_delay_units(full_vals, config), odd_vals, even_vals

    contrasts_odd = contrasts[odd_mask]
    contrasts_even = contrasts[even_mask]
    psth_odd, bin_centers_odd = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events_odd,
        **psth_kwargs,
    )
    psth_even, bin_centers_even = ana_utils.compute_psth_for_clusters(
        spikes,
        cluster_ids,
        events_even,
        **psth_kwargs,
    )
    for row_idx, cid in enumerate(cluster_ids):
        cid_int = int(cid)

        odd_entry = psth_odd.get(cid_int)
        delay_odd, resp_odd = ana_utils.calculate_delay(
            odd_entry.get("fr_raw") if odd_entry else None,
            odd_entry.get("fr_smooth") if odd_entry else None,
            bin_centers_odd,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events_odd,
            trial_contrasts=contrasts_odd,
            return_sign=False,
        )
        if resp_odd:
            odd_vals[row_idx] = delay_odd

        even_entry = psth_even.get(cid_int)
        delay_even, resp_even = ana_utils.calculate_delay(
            even_entry.get("fr_raw") if even_entry else None,
            even_entry.get("fr_smooth") if even_entry else None,
            bin_centers_even,
            event_config,
            method=event_method,
            neuron_spikes=spike_times_by_cluster.get(cid_int, np.array([])),
            event_times=events_even,
            trial_contrasts=contrasts_even,
            return_sign=False,
        )
        if resp_even:
            even_vals[row_idx] = delay_even

    return (
        _apply_delay_units(full_vals, config),
        _apply_delay_units(odd_vals, config),
        _apply_delay_units(even_vals, config),
    )


def _columns_exist(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and all(col in df.columns for col in TARGET_COLUMNS)


def _mirror_columns_to_csv(csv_path: Path, df_res: pd.DataFrame) -> tuple[bool, str]:
    if not csv_path.exists():
        return False, "csv_missing"
    df_csv = pd.read_csv(csv_path)
    if "cluster_id" not in df_csv.columns or "cluster_id" not in df_res.columns:
        raise RuntimeError(f"Cannot mirror response columns because cluster_id is missing in {csv_path}")
    merge_cols = ["cluster_id", *TARGET_COLUMNS]
    df_target = df_res[merge_cols].copy()
    df_csv = df_csv.drop(columns=[col for col in TARGET_COLUMNS if col in df_csv.columns])
    df_csv = df_csv.merge(df_target, on="cluster_id", how="left")
    df_csv.to_csv(csv_path, index=False)
    return True, "csv_updated"


def _save_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as stream:
        pickle.dump(payload, stream)
    os.replace(tmp_path, cache_path)


def _load_spikes_times_clusters(revision_dir: Path) -> Any:
    spikes = alfio.load_object(revision_dir, "spikes", attribute=["times", "clusters"])
    if "times" not in spikes or "clusters" not in spikes:
        raise RuntimeError(f"Missing spikes.times or spikes.clusters in {revision_dir}")
    return spikes


def _pid_progress_total() -> int:
    # load cache, resolve/load spikes, build payloads, save/mirror
    base_steps = 4
    return base_steps + len(EVENT_SPECS) + len(AUDITORY_FEEDBACK_EVENT_SPECS)


def _pid_progress_desc(pid: str) -> str:
    return f"PID {str(pid)[:8]}"


def _set_pid_stage(progress: Any, stage: str) -> None:
    if progress is None:
        return
    try:
        progress.set_postfix_str(str(stage), refresh=False)
    except Exception:
        return


def _advance_pid_stage(progress: Any, stage: str) -> None:
    if progress is None:
        return
    _set_pid_stage(progress, stage)
    try:
        progress.update(1)
    except Exception:
        return


def process_pid(
    pid: str,
    cache_path_str: str,
    raw_dir_str: str,
    processed_dir_str: str,
    rest_lookup: dict[str, Any] | None,
    force: bool,
    show_inner_progress: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_path_str)
    raw_dir = Path(raw_dir_str)
    processed_dir = Path(processed_dir_str)
    csv_path = processed_dir / f"{pid}_delay_results_dashboard.csv"

    summary: dict[str, Any] = {
        "pid": pid,
        "status": "failed",
        "reason": "",
        "mapping_source": "",
        "pname": "",
        "session_path": "",
        "spike_sorting_dir": "",
        "n_units": 0,
        "n_auditory_units": 0,
    }
    for event_name in SUMMARY_EVENT_NAMES:
        summary[f"n_events_{event_name}"] = 0
        summary[f"n_odd_{event_name}"] = 0
        summary[f"n_even_{event_name}"] = 0

    pid_progress = (
        tqdm(total=_pid_progress_total(), desc=_pid_progress_desc(pid), unit="step", leave=False)
        if show_inner_progress
        else None
    )
    try:
        with open(cache_path, "rb") as stream:
            cache_payload = pickle.load(stream)
        _advance_pid_stage(pid_progress, "cache")

        df_res = cache_payload.get("df_res", None)
        if not isinstance(df_res, pd.DataFrame):
            raise RuntimeError("Cache does not contain a DataFrame in df_res.")
        df_res = df_res.copy()
        summary["n_units"] = int(len(df_res))

        if not force and _columns_exist(df_res):
            csv_updated, csv_status = _mirror_columns_to_csv(csv_path, df_res)
            _advance_pid_stage(pid_progress, "mirror")
            summary["status"] = "skipped_existing"
            summary["reason"] = csv_status if csv_updated else "columns already present in cache"
            return summary

        if "cluster_id" not in df_res.columns:
            raise RuntimeError("df_res is missing cluster_id.")
        if "acronym" not in df_res.columns:
            raise RuntimeError("df_res is missing acronym.")
        cluster_ids = pd.to_numeric(df_res["cluster_id"], errors="coerce").to_numpy()
        if np.isnan(cluster_ids).any():
            raise RuntimeError("cluster_id contains non-numeric values.")
        cluster_ids = cluster_ids.astype(int)
        auditory_mask = df_res["acronym"].astype(str).isin(AUDITORY_FEEDBACK_ACRONYMS).to_numpy(
            dtype=bool
        )
        summary["n_auditory_units"] = int(np.sum(auditory_mask))

        probe_lookup = resolve_probe_lookup(
            pid=pid,
            cache_payload=cache_payload,
            raw_dir=raw_dir,
            rest_lookup=rest_lookup,
        )
        session_path = Path(probe_lookup["session_path"])
        pname = str(probe_lookup["pname"])
        revision_dir = _find_probe_revision_dir(session_path, pname)
        spikes = _load_spikes_times_clusters(revision_dir)
        _advance_pid_stage(pid_progress, "spikes")
        spike_times_by_cluster = None
        if np.any(auditory_mask):
            spike_times_by_cluster = {
                int(cid): spikes.times[spikes.clusters == int(cid)]
                for cid in cluster_ids[auditory_mask]
            }

        config = _safe_config(cache_payload)
        zscore_source = str(
            config.get("RESPONSIVE_ZSCORE_SOURCE", DEFAULT_CONFIG["RESPONSIVE_ZSCORE_SOURCE"])
        ).strip().lower()
        if zscore_source not in {"raw", "smooth"}:
            zscore_source = "smooth"

        event_payloads = _build_event_payloads(cache_payload)
        _advance_pid_stage(pid_progress, "events")
        for spec in EVENT_SPECS:
            _set_pid_stage(pid_progress, spec.event_name)
            payload = event_payloads.get(
                spec.event_name,
                {
                    "events": np.array([], dtype=float),
                    "split_index": np.array([], dtype=int),
                    "contrasts": np.array([], dtype=float),
                },
            )
            events = np.asarray(payload["events"], dtype=float)
            split_index = np.asarray(payload["split_index"], dtype=int)
            summary[f"n_events_{spec.event_name}"] = int(events.size)
            summary[f"n_odd_{spec.event_name}"] = int(np.sum((split_index % 2) == 1))
            summary[f"n_even_{spec.event_name}"] = int(np.sum((split_index % 2) == 0))

            full_vals, odd_vals, even_vals = _compute_event_metrics(
                spikes=spikes,
                cluster_ids=cluster_ids,
                events=events,
                split_index=split_index,
                config=config,
                spec=spec,
                zscore_source=zscore_source,
            )
            df_res[spec.full_col] = full_vals
            df_res[spec.odd_col] = odd_vals
            df_res[spec.even_col] = even_vals
            _advance_pid_stage(pid_progress, spec.event_name)

        for spec in AUDITORY_FEEDBACK_EVENT_SPECS:
            _set_pid_stage(pid_progress, spec.event_name)
            payload = event_payloads.get(
                spec.event_name,
                {
                    "events": np.array([], dtype=float),
                    "split_index": np.array([], dtype=int),
                    "contrasts": np.array([], dtype=float),
                },
            )
            events = np.asarray(payload["events"], dtype=float)
            split_index = np.asarray(payload["split_index"], dtype=int)
            contrasts = np.asarray(payload.get("contrasts", np.ones(events.size, dtype=float)), dtype=float)
            summary[f"n_events_{spec.event_name}"] = int(events.size)
            summary[f"n_odd_{spec.event_name}"] = int(np.sum((split_index % 2) == 1))
            summary[f"n_even_{spec.event_name}"] = int(np.sum((split_index % 2) == 0))

            df_res[spec.full_col] = np.nan
            df_res[spec.odd_col] = np.nan
            df_res[spec.even_col] = np.nan
            delay_col = ana_utils.delay_column_name(spec.event_name)
            delay_odd_col = ana_utils.delay_split_column_name(spec.event_name, "odd")
            delay_even_col = ana_utils.delay_split_column_name(spec.event_name, "even")
            df_res[delay_col] = np.nan
            df_res[delay_odd_col] = np.nan
            df_res[delay_even_col] = np.nan

            if not np.any(auditory_mask):
                _advance_pid_stage(pid_progress, spec.event_name)
                continue

            auditory_cluster_ids = cluster_ids[auditory_mask]
            full_vals, odd_vals, even_vals = _compute_event_metrics(
                spikes=spikes,
                cluster_ids=auditory_cluster_ids,
                events=events,
                split_index=split_index,
                config=config,
                spec=spec,
                zscore_source=zscore_source,
            )
            df_res.loc[auditory_mask, spec.full_col] = full_vals
            df_res.loc[auditory_mask, spec.odd_col] = odd_vals
            df_res.loc[auditory_mask, spec.even_col] = even_vals

            delay_full, delay_odd, delay_even = _compute_delay_metrics(
                spikes=spikes,
                cluster_ids=auditory_cluster_ids,
                spike_times_by_cluster=spike_times_by_cluster or {},
                events=events,
                split_index=split_index,
                contrasts=contrasts,
                config=config,
                event_name=spec.event_name,
            )
            df_res.loc[auditory_mask, delay_col] = delay_full
            df_res.loc[auditory_mask, delay_odd_col] = delay_odd
            df_res.loc[auditory_mask, delay_even_col] = delay_even
            _advance_pid_stage(pid_progress, spec.event_name)

        cache_payload["df_res"] = df_res
        _save_cache(cache_path, cache_payload)
        csv_updated, csv_status = _mirror_columns_to_csv(csv_path, df_res)
        _advance_pid_stage(pid_progress, "save")

        summary["status"] = "ok"
        summary["reason"] = csv_status if csv_updated else "cache_updated"
        summary["mapping_source"] = str(probe_lookup.get("source", ""))
        summary["pname"] = pname
        summary["session_path"] = str(session_path)
        summary["spike_sorting_dir"] = str(revision_dir)
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        return summary
    finally:
        if pid_progress is not None:
            pid_progress.close()


def _list_dashboard_pids(cache_dir: Path) -> list[str]:
    return sorted(path.stem for path in cache_dir.glob("*.pkl"))


def _save_summary(summary_rows: list[dict[str, Any]]) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(summary_rows).sort_values(["status", "pid"]).reset_index(drop=True)
    summary_path = SUMMARY_DIR / "batch_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    timestamp_path = SUMMARY_DIR / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_summary.to_csv(timestamp_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    available_pids = _list_dashboard_pids(DASHBOARD_CACHE_DIR)
    if not available_pids:
        raise RuntimeError(f"No dashboard cache files found in {DASHBOARD_CACHE_DIR}")

    if args.pids:
        selected = [pid for pid in args.pids if pid in set(available_pids)]
        missing = sorted(set(args.pids) - set(selected))
        if missing:
            print(f"Warning: {len(missing)} requested PIDs not found in dashboard cache. First few: {missing[:10]}")
        pids = selected
    else:
        pids = available_pids

    if not pids:
        raise RuntimeError("No PIDs selected for processing.")

    print(f"Selected {len(pids)} PIDs.")
    print("Scanning local .rest cache for probe metadata...")
    rest_lookup = build_pid_lookup(REST_DIR, set(pids))
    print(f"Resolved {len(rest_lookup)} / {len(pids)} PIDs from .rest metadata.")

    jobs = [
        {
            "pid": pid,
            "cache_path_str": str(DASHBOARD_CACHE_DIR / f"{pid}.pkl"),
            "raw_dir_str": str(RAW_DIR),
            "processed_dir_str": str(PROCESSED_DIR),
            "rest_lookup": rest_lookup.get(pid),
            "force": bool(args.force),
            "show_inner_progress": bool(args.workers == 1),
        }
        for pid in pids
    ]

    summary_rows: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = tqdm(jobs, desc="PIDs", unit="pid")
        for job in iterator:
            summary_rows.append(process_pid(**job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_pid, **job) for job in jobs]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="PIDs",
                unit="pid",
            ):
                summary_rows.append(future.result())

    summary_path = _save_summary(summary_rows)
    df_summary = pd.DataFrame(summary_rows)
    ok_count = int((df_summary["status"] == "ok").sum()) if not df_summary.empty else 0
    skipped_count = int((df_summary["status"] == "skipped_existing").sum()) if not df_summary.empty else 0
    fail_count = int((df_summary["status"] == "failed").sum()) if not df_summary.empty else 0
    print(f"Finished. ok={ok_count}, skipped={skipped_count}, failed={fail_count}")
    print(f"Summary written to {summary_path}")
    if fail_count:
        failed = df_summary.loc[df_summary["status"] == "failed", ["pid", "reason"]]
        print("Failed PIDs:")
        print(failed.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
