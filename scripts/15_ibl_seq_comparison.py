# %%
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import importlib
import io
import json
import os
import pickle
import re
import sys
import uuid

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
from plotly.subplots import make_subplots

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _TqdmFallback:
        def __init__(
            self,
            iterable=None,
            total=None,
            desc=None,
            unit=None,
            leave=True,
            disable=False,
            dynamic_ncols=True,
            **kwargs,
        ):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.leave = leave
            self.disable = disable

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

        def refresh(self):
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
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))

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
    from one.api import ONE as OneAPI
except Exception:  # pragma: no cover
    OneAPI = None

try:
    from iblatlas.regions import BrainRegions
except Exception:  # pragma: no cover
    BrainRegions = None


# %% Helpers
PLOTLY_RENDERER = None
CALC_VERSION = "15_seq_comparison_v1.2"
PAIR_MIN_N = 2


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


def _choose_option(options, prompt, default_value):
    options = list(options or [])
    if not options:
        raise ValueError("No options available.")
    if default_value not in options:
        default_value = options[0]
    default_idx = options.index(default_value)
    print(prompt)
    for idx, name in enumerate(options):
        print(f"  [{idx}] {name}")
    try:
        selection = input(
            f"Choose index or exact name (Enter for [{default_idx}] {default_value}): "
        ).strip()
    except Exception:
        return default_value
    if selection == "":
        return default_value
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(options):
            return options[idx]
        print(f"Index {idx} out of range. Using default: {default_value}")
        return default_value
    if selection in options:
        return selection
    lower_map = {str(opt).lower(): opt for opt in options}
    if selection.lower() in lower_map:
        return lower_map[selection.lower()]
    print(f"Unknown selection '{selection}'. Using default: {default_value}")
    return default_value


def _format_corr_value(val):
    return "nan" if not np.isfinite(val) else f"{val:.3f}"


def _slugify(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    text = text.strip("_")
    return text or "item"


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, uuid.UUID):
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


def _make_progress(total, desc, unit="step", leave=False, enabled=True):
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=not enabled,
        dynamic_ncols=True,
    )


def _progress_iter(iterable, desc, total=None, unit="item", leave=False, enabled=True):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=not enabled,
        dynamic_ncols=True,
    )


def _set_progress_stage(progress_bar, stage_idx, total_stages, label, detail=None):
    if progress_bar is None:
        return
    progress_bar.set_description(f"[{stage_idx}/{total_stages}] {label}")
    if detail is not None:
        progress_bar.set_postfix_str(detail)


def _merge_delay_result_frames(frames):
    merged = None
    for df in list(frames or []):
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if merged is None:
            merged = df.copy()
            continue
        add_cols = ["cluster_id"] + [col for col in df.columns if col != "cluster_id" and col not in merged.columns]
        merged = merged.merge(df[add_cols], on="cluster_id", how="outer")
    return merged if merged is not None else pd.DataFrame()


class _NoOpProgress:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, n=1):
        return None

    def set_description(self, desc=None, refresh=True):
        return None

    def set_postfix_str(self, s="", refresh=True):
        return None

    def refresh(self):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _PatchedModuleTqdm:
    def __init__(self, module, enabled=True, leave=False):
        self.module = module
        self.enabled = enabled
        self.leave = leave
        self.original = None

    def __enter__(self):
        self.original = getattr(self.module, "tqdm", None)
        if self.original is None:
            return self

        def _wrapped_tqdm(iterable=None, **kwargs):
            if not self.enabled:
                return _NoOpProgress(iterable=iterable)
            kwargs.setdefault("leave", self.leave)
            kwargs.setdefault("disable", False)
            kwargs.setdefault("dynamic_ncols", True)
            return tqdm(iterable, **kwargs)

        setattr(self.module, "tqdm", _wrapped_tqdm)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            setattr(self.module, "tqdm", self.original)
        return False


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


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


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
            if mode == "local" and OneAPI is not None:
                one_local = OneAPI(mode="local", cache_dir=ibl_cache)
                if hasattr(one_local, "pid2eid"):
                    return one_local, mode
                raise RuntimeError(
                    "local ONE initialized without Alyx conversion methods; local cache tables are unavailable"
                )
            return init_one(ibl_cache, mode=mode), mode
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize ONE. Last error: {last_error}")


def _fetch_session_metadata(one, eid):
    meta = {"lab": None, "date": None, "subject": None}
    try:
        sess = one.alyx.rest("sessions", "read", id=eid)
        if isinstance(sess, dict):
            meta["lab"] = sess.get("lab") or sess.get("lab_name") or sess.get("location")
            meta["subject"] = sess.get("subject") or sess.get("subject_nickname")
            meta["date"] = sess.get("start_time") or sess.get("date")
    except Exception:
        pass
    return meta


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
    finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 1] > arr[:, 0])
    arr = arr[finite_mask]
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return arr


def _merge_intervals(intervals):
    arr = _coerce_interval_array(intervals)
    if arr.size == 0:
        return arr
    merged = [arr[0].tolist()]
    for start, end in arr[1:]:
        prev_start, prev_end = merged[-1]
        if float(start) <= float(prev_end):
            merged[-1][1] = float(max(prev_end, end))
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float)


def _intersect_interval_sets(intervals_a, intervals_b):
    a = _merge_intervals(intervals_a)
    b = _merge_intervals(intervals_b)
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), dtype=float)
    out = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(float(a[i, 0]), float(b[j, 0]))
        end = min(float(a[i, 1]), float(b[j, 1]))
        if end > start:
            out.append([start, end])
        if float(a[i, 1]) < float(b[j, 1]):
            i += 1
        else:
            j += 1
    return _merge_intervals(out)


def _subtract_intervals(base_intervals, subtract_intervals):
    base = _merge_intervals(base_intervals)
    sub = _merge_intervals(subtract_intervals)
    if base.size == 0:
        return np.empty((0, 2), dtype=float)
    if sub.size == 0:
        return base
    out = []
    sub_idx = 0
    for base_start, base_end in base:
        cur = float(base_start)
        while sub_idx < len(sub) and float(sub[sub_idx, 1]) <= cur:
            sub_idx += 1
        tmp_idx = sub_idx
        while tmp_idx < len(sub) and float(sub[tmp_idx, 0]) < float(base_end):
            sub_start = float(sub[tmp_idx, 0])
            sub_end = float(sub[tmp_idx, 1])
            if sub_start > cur:
                out.append([cur, min(sub_start, float(base_end))])
            cur = max(cur, sub_end)
            if cur >= float(base_end):
                break
            tmp_idx += 1
        if cur < float(base_end):
            out.append([cur, float(base_end)])
    return _merge_intervals(out)


def _pad_intervals(intervals, pre_s=0.0, post_s=0.0):
    arr = _coerce_interval_array(intervals)
    if arr.size == 0:
        return arr
    out = arr.copy()
    out[:, 0] = out[:, 0] - float(pre_s)
    out[:, 1] = out[:, 1] + float(post_s)
    return _merge_intervals(out)


def _split_alternating_intervals(intervals):
    arr = _merge_intervals(intervals)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    odd = arr[::2].copy()
    even = arr[1::2].copy()
    return odd, even


def _intervals_total_duration(intervals):
    arr = _coerce_interval_array(intervals)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr[:, 1] - arr[:, 0]))


def _intervals_to_df(intervals, segment_type, split_label):
    arr = _coerce_interval_array(intervals)
    if arr.size == 0:
        return pd.DataFrame(
            columns=["segment_type", "split", "segment_idx", "start_s", "end_s", "duration_s"]
        )
    return pd.DataFrame(
        {
            "segment_type": str(segment_type),
            "split": str(split_label),
            "segment_idx": np.arange(len(arr), dtype=int),
            "start_s": arr[:, 0],
            "end_s": arr[:, 1],
            "duration_s": arr[:, 1] - arr[:, 0],
        }
    )


def _extract_spont_whisk_intervals(wh_detect, spont_intervals, pad_pre_s=0.05, pad_post_s=0.05):
    all_bouts = np.asarray(wh_detect.get("all_bouts", np.empty((0, 2))), dtype=float)
    all_bouts = _coerce_interval_array(all_bouts)
    if all_bouts.size == 0:
        return np.empty((0, 2), dtype=float)
    padded = _pad_intervals(all_bouts, pre_s=pad_pre_s, post_s=pad_post_s)
    return _intersect_interval_sets(padded, spont_intervals)


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
        return np.nan, 0, 0.0
    concordant = 0
    comparable = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
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


def _median_abs_dev(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.nanmedian(arr))
    return float(np.nanmedian(np.abs(arr - med)))


def _zscore_nan(vals):
    arr = np.asarray(vals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return out
    mu = float(np.nanmean(arr[mask]))
    sd = float(np.nanstd(arr[mask]))
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[mask] = (arr[mask] - mu) / sd
    return out


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
            r = int(hv[0:2], 16)
            g = int(hv[2:4], 16)
            b = int(hv[4:6], 16)
            return f"rgb({r},{g},{b})"
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
        acr = str(row["acronym"])
        rgb = _to_rgb(row.get("color_hex_triplet", ""))
        if rgb is not None:
            colors[acr] = rgb
    return colors


def _build_region_colors(acronyms):
    unique_regions = pd.Series(acronyms).dropna().astype(str).unique().tolist()
    all_lookup = _get_allen_color_lookup()
    if all_lookup:
        colors = {reg: all_lookup.get(reg) for reg in unique_regions}
        colors = {k: v for k, v in colors.items() if v}
        if len(colors) == len(unique_regions):
            return colors
    colors = {}
    if BrainRegions is not None:
        try:
            br = BrainRegions()
            for region in unique_regions:
                try:
                    idx = br.acronym2index(region)[1][0][0]
                    rgb = br.rgb[idx]
                    colors[region] = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
                except Exception:
                    continue
        except Exception:
            pass
    if len(colors) < len(unique_regions):
        fallback = qualitative.Dark24
        for idx, region in enumerate(unique_regions):
            if region not in colors:
                colors[region] = fallback[idx % len(fallback)]
    return colors


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
        # strict_gt=False keeps units with label exactly equal to label_min.
        # strict_gt=True requires units to be strictly above label_min.
        if strict_gt:
            mask = labels_float > float(label_min)
        else:
            mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return cluster_ids[mask]


def _save_fig(fig, out_dir, prefix, save_flag=True):
    if not save_flag or fig is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{prefix}.html"
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"Saved figure: {save_path}")
    return save_path


def _cluster_id_set(values):
    return {int(v) for v in np.asarray(values, dtype=int).reshape(-1)}


def _map_whisk_group(raw_group):
    raw = str(raw_group).strip().lower()
    if raw in {"arousal_plus", "wh_plus", "plus", "positive"}:
        return "wh+"
    if raw in {"arousal_minus", "wh_minus", "minus", "negative"}:
        return "wh-"
    return "neutral"


def _build_task_event_inputs(sl, event_names):
    events_by_name, contrasts_by_name, trial_idx_by_name = ana_utils.build_event_dicts(
        sl,
        list(event_names),
        min_trials=0,
        return_trial_idx=True,
    )
    return events_by_name, contrasts_by_name, trial_idx_by_name


def _resolve_task_whisk_move_control_window(config):
    delay_windows = dict(config.get("DELAY_WINDOWS", {}))
    fm_start, fm_end = delay_windows.get("firstMovement_times", (-0.1, 0.2))
    tw_start, tw_end = delay_windows.get("wh_all_times_task", (0.0, 0.4))
    pre_override = config.get("TASK_WHISK_MOVE_EXCLUDE_PRE_S", None)
    post_override = config.get("TASK_WHISK_MOVE_EXCLUDE_POST_S", None)
    pre_s = (
        float(pre_override)
        if pre_override is not None
        else max(0.0, float(tw_end) - float(fm_start))
    )
    post_s = (
        float(post_override)
        if post_override is not None
        else max(0.0, float(fm_end) - float(tw_start))
    )
    return pre_s, post_s


def _exclude_events_near_reference_times(events, reference_times, pre_s=0.0, post_s=0.0):
    events = np.asarray(events, dtype=float).reshape(-1)
    events = np.sort(events[np.isfinite(events)])
    reference_times = np.asarray(reference_times, dtype=float).reshape(-1)
    reference_times = np.sort(reference_times[np.isfinite(reference_times)])
    if events.size == 0:
        return events, np.zeros(0, dtype=bool)
    if reference_times.size == 0:
        return events.copy(), np.ones(events.shape[0], dtype=bool)
    left = np.searchsorted(reference_times, events - float(pre_s), side="left")
    right = np.searchsorted(reference_times, events + float(post_s), side="right")
    keep_mask = right == left
    return events[keep_mask], keep_mask


def _build_whisk_delay_inputs(wh_events_by_period, first_move_times=None, config=None):
    events_by_name = {}
    contrasts_by_name = {}
    trial_idx_by_name = {}
    control_info = {
        "enabled": False,
        "mode": "disabled",
        "exclude_pre_s": 0.0,
        "exclude_post_s": 0.0,
        "original_task_whisk_n": 0,
        "controlled_task_whisk_n": 0,
        "excluded_task_whisk_n": 0,
        "keep_fraction": np.nan,
    }
    for event_name in ("wh_all_times_spont", "wh_all_times_task"):
        ev = np.asarray(wh_events_by_period.get(event_name, np.array([])), dtype=float).reshape(-1)
        ev = ev[np.isfinite(ev)]
        ev = np.sort(ev)
        events_by_name[event_name] = ev
        contrasts_by_name[event_name] = np.ones(ev.shape[0], dtype=float)
        trial_idx_by_name[event_name] = np.arange(ev.shape[0], dtype=int)
    if bool((config or {}).get("TASK_WHISK_MOVE_CONTROL_ENABLED", True)):
        control_info["enabled"] = True
        control_info["mode"] = "exclude_window_overlap"
        pre_s, post_s = _resolve_task_whisk_move_control_window(config or {})
        control_info["exclude_pre_s"] = float(pre_s)
        control_info["exclude_post_s"] = float(post_s)
        task_events = events_by_name.get("wh_all_times_task", np.array([], dtype=float))
        controlled_events, _keep_mask = _exclude_events_near_reference_times(
            task_events,
            first_move_times,
            pre_s=pre_s,
            post_s=post_s,
        )
        control_info["original_task_whisk_n"] = int(task_events.shape[0])
        control_info["controlled_task_whisk_n"] = int(controlled_events.shape[0])
        control_info["excluded_task_whisk_n"] = int(task_events.shape[0] - controlled_events.shape[0])
        control_info["keep_fraction"] = (
            float(controlled_events.shape[0] / task_events.shape[0])
            if task_events.shape[0] > 0
            else np.nan
        )
        # Task whisk events are filtered in-place so the analysis keeps a single
        # task-whisk variable based only on non-overlapping events.
        events_by_name["wh_all_times_task"] = controlled_events
        contrasts_by_name["wh_all_times_task"] = np.ones(controlled_events.shape[0], dtype=float)
        trial_idx_by_name["wh_all_times_task"] = np.arange(controlled_events.shape[0], dtype=int)
    return events_by_name, contrasts_by_name, trial_idx_by_name, control_info


def _compute_region_eligibility(cluster_ids, cluster_acronyms, clusters, label_min, min_region_neurons, strict_gt=False):
    cluster_ids = np.asarray(cluster_ids)
    selected_cluster_ids = _select_cluster_ids_by_label(
        cluster_ids,
        clusters,
        label_min=label_min,
        strict_gt=strict_gt,
    )
    selected_set = _cluster_id_set(selected_cluster_ids)
    region_df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "region": np.asarray(cluster_acronyms).astype(str),
        }
    )
    region_df = region_df[region_df["cluster_id"].isin(selected_set)].copy()
    region_df = region_df[~region_df["region"].isin(["root", "void"])].copy()
    region_counts = (
        region_df["region"].value_counts().sort_index().rename("n_units").reset_index().rename(columns={"index": "region"})
    )
    if region_counts.empty:
        return {
            "eligible_regions": [],
            "region_counts": region_counts.rename(columns={"region": "region"}),
            "eligible_cluster_ids": np.array([], dtype=int),
        }
    eligible_regions = sorted(
        region_counts.loc[region_counts["n_units"] >= int(min_region_neurons), "region"].astype(str).tolist()
    )
    eligible_cluster_ids = region_df.loc[region_df["region"].isin(eligible_regions), "cluster_id"].to_numpy(dtype=int)
    region_counts["eligible"] = region_counts["region"].astype(str).isin(eligible_regions)
    return {
        "eligible_regions": eligible_regions,
        "region_counts": region_counts,
        "eligible_cluster_ids": np.asarray(eligible_cluster_ids, dtype=int),
    }


def _compute_spont_context_coupling(
    spikes,
    clusters,
    cluster_acronyms,
    config_calc,
    cluster_ids,
    intervals_odd,
    intervals_even,
    context_label,
    show_progress=True,
):
    df_odd = None
    df_even = None
    total_splits = int(len(intervals_odd) > 0) + int(len(intervals_even) > 0)
    total_splits = max(total_splits, 1)
    split_bar = _make_progress(
        total=total_splits,
        desc=f"{context_label}: stPR splits",
        unit="split",
        leave=False,
        enabled=show_progress,
    )
    try:
        split_bar.set_postfix_str(f"n_units={len(np.asarray(cluster_ids).reshape(-1))}")
        if len(intervals_odd) > 0:
            split_bar.set_description(f"{context_label}: odd split")
            spikes_odd = ana_utils.slice_spikes_by_intervals(spikes, intervals_odd)
            with _PatchedModuleTqdm(ana_utils, enabled=False, leave=False):
                df_odd = _run_quietly(
                    ana_utils.compute_population_coupling,
                    spikes_odd,
                    clusters,
                    cluster_acronyms,
                    config_calc,
                    cluster_ids=cluster_ids,
                    split_halves=False,
                    intervals=intervals_odd,
                    context_label=f"{context_label} odd",
            )
            if df_odd is not None and df_odd.empty:
                df_odd = None
            split_bar.update(1)
        if len(intervals_even) > 0:
            split_bar.set_description(f"{context_label}: even split")
            spikes_even = ana_utils.slice_spikes_by_intervals(spikes, intervals_even)
            with _PatchedModuleTqdm(ana_utils, enabled=False, leave=False):
                df_even = _run_quietly(
                    ana_utils.compute_population_coupling,
                    spikes_even,
                    clusters,
                    cluster_acronyms,
                    config_calc,
                    cluster_ids=cluster_ids,
                    split_halves=False,
                    intervals=intervals_even,
                    context_label=f"{context_label} even",
            )
            if df_even is not None and df_even.empty:
                df_even = None
            split_bar.update(1)
        if total_splits == 1 and df_odd is None and df_even is None:
            split_bar.update(1)
    finally:
        split_bar.close()
    if df_odd is None and df_even is None:
        return None
    return ana_utils.merge_stpr_splits(
        df_odd,
        df_even,
        config_calc,
        split_a="odd",
        split_b="even",
    )


def _build_variable_specs():
    return [
        {
            "key": "spont_whisk_coupling",
            "name": "Spont Whisk Coupling Delay",
            "odd_col": "spont_whisk_coupling_delay_ms_odd",
            "even_col": "spont_whisk_coupling_delay_ms_even",
            "mean_col": "spont_whisk_coupling_delay_ms",
            "source_df": "df_coupling_whisk",
            "source_odd": "coupling_delay_ms_odd",
            "source_even": "coupling_delay_ms_even",
        },
        {
            "key": "spont_nonwhisk_coupling",
            "name": "Spont Non-Whisk Coupling Delay",
            "odd_col": "spont_nonwhisk_coupling_delay_ms_odd",
            "even_col": "spont_nonwhisk_coupling_delay_ms_even",
            "mean_col": "spont_nonwhisk_coupling_delay_ms",
            "source_df": "df_coupling_nonwhisk",
            "source_odd": "coupling_delay_ms_odd",
            "source_even": "coupling_delay_ms_even",
        },
        {
            "key": "first_move_delay",
            "name": "First Move Delay",
            "odd_col": "first_move_delay_ms_odd",
            "even_col": "first_move_delay_ms_even",
            "mean_col": "first_move_delay_ms",
            "source_df": "df_delay",
            "source_odd": ana_utils.delay_split_column_name("firstMovement_times", "odd"),
            "source_even": ana_utils.delay_split_column_name("firstMovement_times", "even"),
        },
        {
            "key": "feedback_delay",
            "name": "Feedback Delay",
            "odd_col": "feedback_delay_ms_odd",
            "even_col": "feedback_delay_ms_even",
            "mean_col": "feedback_delay_ms",
            "source_df": "df_delay",
            "source_odd": ana_utils.delay_split_column_name("feedback_times", "odd"),
            "source_even": ana_utils.delay_split_column_name("feedback_times", "even"),
        },
        {
            "key": "spont_whisk_event_delay",
            "name": "Spont Whisk Event Delay",
            "odd_col": "spont_whisk_event_delay_ms_odd",
            "even_col": "spont_whisk_event_delay_ms_even",
            "mean_col": "spont_whisk_event_delay_ms",
            "source_df": "df_delay",
            "source_odd": ana_utils.delay_split_column_name("wh_all_times_spont", "odd"),
            "source_even": ana_utils.delay_split_column_name("wh_all_times_spont", "even"),
        },
        {
            "key": "task_whisk_event_delay",
            "name": "Task Whisk Event Delay",
            "odd_col": "task_whisk_event_delay_ms_odd",
            "even_col": "task_whisk_event_delay_ms_even",
            "mean_col": "task_whisk_event_delay_ms",
            "source_df": "df_delay",
            "source_odd": ana_utils.delay_split_column_name("wh_all_times_task", "odd"),
            "source_even": ana_utils.delay_split_column_name("wh_all_times_task", "even"),
        },
    ]

def _merge_variable_into_summary(summary_df, source_df, spec):
    out = summary_df.copy()
    source_df = source_df if isinstance(source_df, pd.DataFrame) else pd.DataFrame()
    odd_col = spec["source_odd"]
    even_col = spec["source_even"]
    needed = ["cluster_id"]
    if odd_col in source_df.columns:
        needed.append(odd_col)
    if even_col in source_df.columns:
        needed.append(even_col)
    if len(needed) == 1:
        out[spec["odd_col"]] = np.nan
        out[spec["even_col"]] = np.nan
        out[spec["mean_col"]] = np.nan
        return out
    tmp = source_df[needed].copy()
    tmp = tmp.groupby("cluster_id", as_index=False).mean(numeric_only=True)
    tmp = tmp.rename(columns={odd_col: spec["odd_col"], even_col: spec["even_col"]})
    out = out.merge(tmp, on="cluster_id", how="left")
    odd_vals = pd.to_numeric(out.get(spec["odd_col"]), errors="coerce").to_numpy(dtype=float)
    even_vals = pd.to_numeric(out.get(spec["even_col"]), errors="coerce").to_numpy(dtype=float)
    mean_vals = np.full(out.shape[0], np.nan, dtype=float)
    valid = np.isfinite(odd_vals) & np.isfinite(even_vals)
    mean_vals[valid] = (odd_vals[valid] + even_vals[valid]) / 2.0
    out[spec["mean_col"]] = mean_vals
    return out


def _build_summary_neurons(
    eligible_cluster_ids,
    eligible_regions,
    cluster_acronyms,
    cid_to_idx,
    clusters,
    df_delay,
    df_coupling_whisk,
    df_coupling_nonwhisk,
    extra_variable_specs=None,
    show_progress=True,
):
    summary = pd.DataFrame({"cluster_id": np.asarray(eligible_cluster_ids, dtype=int)})
    summary["region"] = [
        str(cluster_acronyms[cid_to_idx[int(cid)]]) for cid in summary["cluster_id"].to_numpy(dtype=int)
    ]
    labels = get_cluster_labels_array(clusters)
    summary["label"] = np.nan
    if labels is not None:
        labels = np.asarray(labels)
        label_vals = []
        label_iter = _progress_iter(
            summary["cluster_id"].to_numpy(dtype=int),
            desc="Summary labels",
            total=int(summary.shape[0]),
            unit="unit",
            leave=False,
            enabled=show_progress,
        )
        for cid in label_iter:
            idx = cid_to_idx.get(int(cid))
            if idx is None or idx >= len(labels):
                label_vals.append(np.nan)
            else:
                try:
                    label_vals.append(float(labels[idx]))
                except Exception:
                    label_vals.append(np.nan)
        summary["label"] = np.asarray(label_vals, dtype=float)

    if isinstance(df_delay, pd.DataFrame) and not df_delay.empty:
        arousal_cols = [col for col in ("cluster_id", "arousal_group", "arousal_corr", "arousal_corr_abs") if col in df_delay.columns]
        if "cluster_id" in arousal_cols:
            df_whisk_group = df_delay[arousal_cols].drop_duplicates("cluster_id").copy()
            summary = summary.merge(df_whisk_group, on="cluster_id", how="left")
    if "arousal_group" not in summary.columns:
        summary["arousal_group"] = "neutral"
    summary["whisk_group"] = summary["arousal_group"].map(_map_whisk_group).fillna("neutral")

    source_tables = {
        "df_delay": df_delay,
        "df_coupling_whisk": df_coupling_whisk,
        "df_coupling_nonwhisk": df_coupling_nonwhisk,
    }
    variable_specs = _build_variable_specs() + list(extra_variable_specs or [])
    merge_iter = _progress_iter(
        variable_specs,
        desc="Summary variables",
        total=len(variable_specs),
        unit="variable",
        leave=False,
        enabled=show_progress,
    )
    for spec in merge_iter:
        summary = _merge_variable_into_summary(summary, source_tables.get(spec["source_df"]), spec)

    summary = summary[summary["region"].isin(list(eligible_regions))].copy()
    summary = summary.sort_values(["region", "cluster_id"]).reset_index(drop=True)
    return summary


def _build_summary_pairs(summary_neurons, variable_specs, show_progress=True):
    rows = []
    if summary_neurons is None or summary_neurons.empty:
        return pd.DataFrame()
    regions = sorted(summary_neurons["region"].astype(str).unique().tolist())
    pair_specs = [(idx_x, idx_y) for idx_x in range(len(variable_specs)) for idx_y in range(idx_x + 1, len(variable_specs))]
    total_pairs = len(regions) * len(pair_specs)
    pair_bar = _make_progress(
        total=total_pairs,
        desc="Pair summaries",
        unit="pair",
        leave=False,
        enabled=show_progress,
    )
    try:
        for region in regions:
            pair_bar.set_postfix_str(region)
            pair_bar.set_description(f"Pair summaries ({region})")
            df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region)].copy()
            if df_region.empty:
                continue
            rel_stats = {}
            for spec in variable_specs:
                r_p, n_p = _pearsonr_with_n(df_region[spec["odd_col"]], df_region[spec["even_col"]])
                r_s, n_s = _spearmanr_with_n(df_region[spec["odd_col"]], df_region[spec["even_col"]])
                rel_stats[spec["key"]] = {
                    "pearson": r_p,
                    "spearman": r_s,
                    "n_pearson": n_p,
                    "n_spearman": n_s,
                }
            for idx_x, idx_y in pair_specs:
                spec_x = variable_specs[idx_x]
                spec_y = variable_specs[idx_y]
                x = df_region[spec_x["mean_col"]].to_numpy(dtype=float)
                y = df_region[spec_y["mean_col"]].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x_pair = x[mask]
                y_pair = y[mask]
                pearson_r, n_shared = _pearsonr_with_n(x_pair, y_pair)
                spearman_rho, _ = _spearmanr_with_n(x_pair, y_pair)
                order_score, n_order_pairs, preserved_fraction = _order_agreement_score(x_pair, y_pair)
                diff = x_pair - y_pair
                rows.append(
                    {
                        "region": region,
                        "var_x_key": spec_x["key"],
                        "var_y_key": spec_y["key"],
                        "var_x_name": spec_x["name"],
                        "var_y_name": spec_y["name"],
                        "pair_label": f"{spec_x['name']} vs {spec_y['name']}",
                        "n_shared": int(n_shared),
                        "pearson_r": pearson_r,
                        "spearman_rho": spearman_rho,
                        "reliability_x_pearson": rel_stats[spec_x["key"]]["pearson"],
                        "reliability_y_pearson": rel_stats[spec_y["key"]]["pearson"],
                        "reliability_x_spearman": rel_stats[spec_x["key"]]["spearman"],
                        "reliability_y_spearman": rel_stats[spec_y["key"]]["spearman"],
                        "reliability_x_n": int(rel_stats[spec_x["key"]]["n_pearson"]),
                        "reliability_y_n": int(rel_stats[spec_y["key"]]["n_pearson"]),
                        "reliability_floor": np.nanmin(
                            [rel_stats[spec_x["key"]]["pearson"], rel_stats[spec_y["key"]]["pearson"]]
                        ),
                        "order_agreement_score": order_score,
                        "order_agreement_n_pairs": int(n_order_pairs),
                        "order_agreement_preserved_fraction": preserved_fraction,
                        "delay_diff_mean_ms": float(np.nanmean(diff)) if diff.size else np.nan,
                        "delay_diff_median_ms": float(np.nanmedian(diff)) if diff.size else np.nan,
                        "delay_diff_sd_ms": float(np.nanstd(diff)) if diff.size else np.nan,
                        "delay_diff_mad_ms": _median_abs_dev(diff),
                    }
                )
                pair_bar.update(1)
    finally:
        pair_bar.close()
    return pd.DataFrame(rows)


def _build_pairwise_matrix(summary_neurons, variable_specs, region_name, metric="pearson"):
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region_name)].copy()
    names = [spec["name"] for spec in variable_specs]
    n_vars = len(variable_specs)
    mat = np.full((n_vars, n_vars), np.nan, dtype=float)
    text = np.empty((n_vars, n_vars), dtype=object)
    for i, spec_i in enumerate(variable_specs):
        if metric == "spearman":
            rel_val, rel_n = _spearmanr_with_n(df_region[spec_i["odd_col"]], df_region[spec_i["even_col"]])
        else:
            rel_val, rel_n = _pearsonr_with_n(df_region[spec_i["odd_col"]], df_region[spec_i["even_col"]])
        mat[i, i] = rel_val
        text[i, i] = f"rel={_format_corr_value(rel_val)}<br>n={rel_n}"
        for j in range(i + 1, n_vars):
            spec_j = variable_specs[j]
            x = df_region[spec_i["mean_col"]].to_numpy(dtype=float)
            y = df_region[spec_j["mean_col"]].to_numpy(dtype=float)
            if metric == "spearman":
                val, n_val = _spearmanr_with_n(x, y)
                prefix = "rho"
            else:
                val, n_val = _pearsonr_with_n(x, y)
                prefix = "r"
            mat[i, j] = val
            mat[j, i] = val
            text[i, j] = f"{prefix}={_format_corr_value(val)}<br>n={n_val}"
            text[j, i] = f"{prefix}={_format_corr_value(val)}<br>n={n_val}"
    return names, mat, text


def _build_order_agreement_matrix(summary_neurons, variable_specs, region_name):
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region_name)].copy()
    names = [spec["name"] for spec in variable_specs]
    n_vars = len(variable_specs)
    mat = np.full((n_vars, n_vars), np.nan, dtype=float)
    text = np.empty((n_vars, n_vars), dtype=object)
    for i, spec_i in enumerate(variable_specs):
        rel_val, rel_n = _pearsonr_with_n(df_region[spec_i["odd_col"]], df_region[spec_i["even_col"]])
        mat[i, i] = rel_val
        text[i, i] = f"rel={_format_corr_value(rel_val)}<br>n={rel_n}"
        for j in range(i + 1, n_vars):
            spec_j = variable_specs[j]
            x = df_region[spec_i["mean_col"]].to_numpy(dtype=float)
            y = df_region[spec_j["mean_col"]].to_numpy(dtype=float)
            score, n_pairs, preserved_fraction = _order_agreement_score(x, y)
            mat[i, j] = score
            mat[j, i] = score
            txt = (
                f"oa={_format_corr_value(score)}<br>pairs={n_pairs}<br>pres={preserved_fraction:.2f}"
                if np.isfinite(score)
                else f"oa=nan<br>pairs={n_pairs}"
            )
            text[i, j] = txt
            text[j, i] = txt
    return names, mat, text


def _add_grouped_scatter_points(fig, df_plot, x_col, y_col, region_color, row, col):
    symbol_map = {"wh+": "circle", "neutral": "diamond", "wh-": "x"}
    for group_name in ("wh+", "neutral", "wh-"):
        grp = df_plot.loc[df_plot["whisk_group"] == group_name].copy()
        if grp.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=grp[x_col].to_numpy(dtype=float),
                y=grp[y_col].to_numpy(dtype=float),
                mode="markers",
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                customdata=grp[["cluster_id", "region", "whisk_group"]].to_numpy(),
                marker=dict(
                    color=region_color,
                    size=7,
                    opacity=0.75,
                    symbol=symbol_map.get(group_name, "circle"),
                    line=dict(width=0.8, color="black"),
                ),
                hovertemplate=(
                    "Cluster %{customdata[0]}<br>"
                    "Region %{customdata[1]}<br>"
                    "Whisk group %{customdata[2]}<br>"
                    f"{x_col}: %{{x:.3f}}<br>"
                    f"{y_col}: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def _add_unity_line(fig, x_vals, y_vals, row=1, col=1):
    if len(x_vals) == 0 or len(y_vals) == 0:
        return
    min_val = float(np.nanmin([np.nanmin(x_vals), np.nanmin(y_vals)]))
    max_val = float(np.nanmax([np.nanmax(x_vals), np.nanmax(y_vals)]))
    if not (np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val):
        return
    trace = go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        showlegend=False,
        hovertemplate="Unity line<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    )
    try:
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)
    except Exception:
        fig.add_trace(trace)


_set_plotly_renderer(PLOTLY_RENDERER)


def _configure_local_one_params_dir(base_path):
    params_root = Path(base_path) / "data" / "_appdata"
    params_root.mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(params_root)
    return params_root


class _PidSkipError(RuntimeError):
    pass


def _coerce_pid_list(pid_values):
    if pid_values is None:
        return []
    if isinstance(pid_values, str):
        pid_values = [pid_values]
    elif not isinstance(pid_values, (list, tuple, set, np.ndarray, pd.Series)):
        pid_values = [pid_values]
    out = []
    for value in pid_values:
        if value is None:
            continue
        pid_str = str(value).strip()
        if pid_str:
            out.append(pid_str)
    return list(dict.fromkeys(out))


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


def _resolve_pid_targets(one, pid, pid_list=None, run_all_pids=False, all_pids_tag=None):
    if run_all_pids:
        targets = _get_all_remote_pids(one, tag=all_pids_tag)
    else:
        targets = _coerce_pid_list(pid_list)
        if not targets and pid is not None:
            targets = _coerce_pid_list([pid])
    return list(dict.fromkeys(targets))


def _find_cache_dir_for_pid(processed_root, pid, calc_hash=None):
    pid_root = Path(processed_root) / str(pid)
    if not pid_root.exists():
        return None
    if calc_hash is not None:
        target_dir = pid_root / str(calc_hash)
        if (target_dir / "manifest.json").exists() and (target_dir / "calc_cache.pkl").exists():
            return target_dir
        return None
    candidates = []
    for child in pid_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "manifest.json").exists() and (child / "calc_cache.pkl").exists():
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_cached_plot_context(processed_root, fig_root, pid, calc_hash=None):
    cache_dir = _find_cache_dir_for_pid(processed_root, pid, calc_hash=calc_hash)
    if cache_dir is None:
        raise FileNotFoundError(f"No cached calculations found for PID {pid}.")
    manifest_path = cache_dir / "manifest.json"
    cache_path = cache_dir / "calc_cache.pkl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with open(cache_path, "rb") as f:
        cache_payload = pickle.load(f)
    fig_dir = Path(fig_root) / str(pid)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {
        "pid": str(pid),
        "calc_hash": str(manifest.get("calc_hash", cache_dir.name)),
        "cache_dir": cache_dir,
        "fig_dir": fig_dir,
        "manifest": manifest,
        "cache_payload": cache_payload,
        "cache_hit": True,
    }


def _prepare_pid_context(
    pid,
    one,
    ba,
    br,
    config_calc,
    min_region_neurons,
    suppress_load_stdout=False,
):
    pid = str(pid)
    eid, _ = one.pid2eid(pid)
    spont_intervals = _load_spontaneous_intervals(one, eid)
    if spont_intervals is None or len(np.asarray(spont_intervals).reshape(-1)) < 2:
        raise _PidSkipError(f"PID {pid} has no spontaneous interval in *passivePeriods*.")
    spont_intervals = _coerce_interval_array(spont_intervals)

    ssl, spikes, clusters, sl = _run_quietly(
        load_session_data,
        pid,
        one,
        ba=ba,
        load_trials=True,
        load_wheel=True,
        load_pose=False,
        load_motion_energy=True,
        load_pupil=False,
        suppress_stdout=suppress_load_stdout,
    )

    cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
    cluster_ids = np.asarray(cluster_ids)
    if br is None:
        if hasattr(clusters, "acronym"):
            cluster_acronyms_calc = np.asarray(clusters.acronym).astype(str)
        else:
            raise RuntimeError("BrainRegions is unavailable and clusters.acronym could not be resolved.")
    else:
        cluster_acronyms_calc = np.asarray(map_acronyms(clusters, br, "Beryl")).astype(str)

    df_me_raw = ana_utils.extract_motion_energy_trace(
        getattr(sl, "motion_energy", None),
        max_interp_gap_frames=3,
        ensure_positive_motion=True,
    )
    if df_me_raw.empty:
        raise _PidSkipError(f"PID {pid} has no usable motion energy trace.")

    task_windows = ana_utils.build_task_window_table(
        sl.trials,
        ["firstMovement_times", "feedback_times"],
        post_event_s=1.0,
    )
    trial_end_times = ana_utils.compute_trial_end_times(
        sl.trials,
        ["firstMovement_times", "feedback_times"],
        post_event_s=1.0,
    )
    stim_on_times = np.asarray(sl.trials["stimOn_times"], dtype=float)
    iti_windows = ana_utils.build_iti_windows(trial_end_times, stim_on_times, skip_first_last=True)

    eligibility = _compute_region_eligibility(
        cluster_ids,
        cluster_acronyms_calc,
        clusters,
        label_min=config_calc.get("CALC_LABEL_MIN"),
        min_region_neurons=min_region_neurons,
        strict_gt=bool(config_calc.get("CALC_LABEL_STRICT_GT", False)),
    )
    eligible_regions = list(eligibility["eligible_regions"])
    eligible_cluster_ids = np.asarray(eligibility["eligible_cluster_ids"], dtype=int)
    eligible_cid_to_idx = {
        int(cid): cid_to_idx[int(cid)]
        for cid in eligible_cluster_ids
        if int(cid) in cid_to_idx
    }
    if eligible_cluster_ids.size == 0:
        raise _PidSkipError(
            f"PID {pid} has no eligible regions after label>={config_calc.get('CALC_LABEL_MIN')} "
            f"and MIN_REGION_NEURONS={min_region_neurons}."
        )

    return {
        "pid": pid,
        "eid": eid,
        "ssl": ssl,
        "spikes": spikes,
        "clusters": clusters,
        "sl": sl,
        "cluster_ids": cluster_ids,
        "cid_to_idx": cid_to_idx,
        "cluster_acronyms_calc": cluster_acronyms_calc,
        "spont_intervals": spont_intervals,
        "df_me_raw": df_me_raw,
        "task_windows": task_windows,
        "iti_windows": iti_windows,
        "eligibility": eligibility,
        "eligible_regions": eligible_regions,
        "eligible_cluster_ids": eligible_cluster_ids,
        "eligible_cid_to_idx": eligible_cid_to_idx,
    }


def _run_single_pid_calculations(
    pid,
    one,
    one_mode,
    ba,
    br,
    config_calc,
    config_plot,
    seq_processed_root,
    seq_fig_root,
    min_region_neurons,
    calc_version,
    show_calc_progress=True,
    return_payload=False,
    suppress_load_stdout=False,
):
    pid_context = _prepare_pid_context(
        pid,
        one,
        ba,
        br,
        config_calc=config_calc,
        min_region_neurons=min_region_neurons,
        suppress_load_stdout=suppress_load_stdout,
    )
    pid = str(pid_context["pid"])
    eid = pid_context["eid"]
    spikes = pid_context["spikes"]
    clusters = pid_context["clusters"]
    sl = pid_context["sl"]
    cluster_acronyms_calc = pid_context["cluster_acronyms_calc"]
    cid_to_idx = pid_context["cid_to_idx"]
    spont_intervals = pid_context["spont_intervals"]
    df_me_raw = pid_context["df_me_raw"]
    task_windows = pid_context["task_windows"]
    iti_windows = pid_context["iti_windows"]
    eligibility = pid_context["eligibility"]
    eligible_regions = pid_context["eligible_regions"]
    eligible_cluster_ids = pid_context["eligible_cluster_ids"]
    eligible_cid_to_idx = pid_context["eligible_cid_to_idx"]

    calc_hash_payload = {
        "calc_version": calc_version,
        "pid": pid,
        "min_region_neurons": int(min_region_neurons),
        "config_calc": config_calc,
        "eligible_regions": eligible_regions,
    }
    calc_hash = _stable_hash(calc_hash_payload)
    cache_dir = Path(seq_processed_root) / pid / calc_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(seq_fig_root) / pid
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    cache_pkl_path = cache_dir / "calc_cache.pkl"

    cache_payload = None
    cache_hit = False
    if manifest_path.exists() and cache_pkl_path.exists():
        try:
            manifest_existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest_existing.get("calc_hash") == calc_hash
                and manifest_existing.get("calc_version") == calc_version
            ):
                if return_payload:
                    with open(cache_pkl_path, "rb") as f:
                        cache_payload = pickle.load(f)
                cache_hit = True
                print(f"[{pid}] Loaded cached calculations from {cache_pkl_path}")
        except Exception as exc:
            print(f"[{pid}] Cache load failed. Recomputing. ({exc})")

    if not cache_hit:
        print(f"[{pid}] Running calculations...")
        calc_stage_total = 10
        calc_stage_bar = _make_progress(
            total=calc_stage_total,
            desc=f"{pid}: seq comparison calculations",
            unit="stage",
            leave=True,
            enabled=show_calc_progress,
        )
        try:
            calc_stage_bar.set_postfix_str(f"pid={pid}")

            _set_progress_stage(calc_stage_bar, 1, calc_stage_total, "Building whisk trace")
            df_wh = ana_utils.build_whisk_trace(df_me_raw, dict(config_calc))
            if df_wh is None or df_wh.empty:
                raise RuntimeError("Whisk signal is empty after normalization.")
            calc_stage_bar.set_postfix_str(f"n_bins={len(df_wh)}")
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 2, calc_stage_total, "Detecting whisk events and spontaneous segments")
            whisk_bundle = ana_utils.build_whisk_events(
                df_wh,
                dict(config_calc),
                spont_intervals=spont_intervals,
                task_windows=task_windows,
                iti_windows=iti_windows,
                wheel=getattr(sl, "wheel", None),
            )
            wh_detect = whisk_bundle.get("wh_detect", {})
            wh_events_by_period = whisk_bundle.get("wh_events_by_period", {})

            spont_whisk_intervals = _extract_spont_whisk_intervals(
                wh_detect,
                spont_intervals=spont_intervals,
                pad_pre_s=config_calc["SPONT_WHISK_PAD_PRE_S"],
                pad_post_s=config_calc["SPONT_WHISK_PAD_POST_S"],
            )
            spont_nonwhisk_intervals = _subtract_intervals(spont_intervals, spont_whisk_intervals)
            spont_whisk_odd, spont_whisk_even = _split_alternating_intervals(spont_whisk_intervals)
            spont_nonwhisk_odd, spont_nonwhisk_even = _split_alternating_intervals(spont_nonwhisk_intervals)

            df_segments = pd.concat(
                [
                    _intervals_to_df(spont_whisk_intervals, "whisk", "all"),
                    _intervals_to_df(spont_whisk_odd, "whisk", "odd"),
                    _intervals_to_df(spont_whisk_even, "whisk", "even"),
                    _intervals_to_df(spont_nonwhisk_intervals, "nonwhisk", "all"),
                    _intervals_to_df(spont_nonwhisk_odd, "nonwhisk", "odd"),
                    _intervals_to_df(spont_nonwhisk_even, "nonwhisk", "even"),
                ],
                ignore_index=True,
            )
            calc_stage_bar.set_postfix_str(
                f"whisk seg={len(spont_whisk_intervals)} | nonwhisk seg={len(spont_nonwhisk_intervals)}"
            )
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 3, calc_stage_total, "Preparing event dictionaries")
            task_events_by_name, task_contrasts_by_name, task_trial_idx_by_name = _build_task_event_inputs(
                sl,
                ["firstMovement_times", "feedback_times"],
            )
            whisk_events_by_name, whisk_contrasts_by_name, whisk_trial_idx_by_name, task_whisk_control_info = _build_whisk_delay_inputs(
                wh_events_by_period,
                first_move_times=task_events_by_name.get("firstMovement_times", np.array([])),
                config=config_calc,
            )
            events_by_name = {**task_events_by_name, **whisk_events_by_name}
            contrasts_by_name = {**task_contrasts_by_name, **whisk_contrasts_by_name}
            trial_idx_by_name = {**task_trial_idx_by_name, **whisk_trial_idx_by_name}
            calc_stage_bar.set_postfix_str(
                f"{len(events_by_name)} event streams | task whisk kept={task_whisk_control_info['controlled_task_whisk_n']}"
            )
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 4, calc_stage_total, "Calculating event delays")
            delay_config = dict(config_calc)
            delay_event_names = [
                "firstMovement_times",
                "feedback_times",
                "wh_all_times_spont",
                "wh_all_times_task",
            ]
            delay_frames = []
            delay_subbar = _make_progress(
                total=len(delay_event_names),
                desc="Event delays",
                unit="event",
                leave=False,
                enabled=show_calc_progress,
            )
            try:
                for event_name in delay_event_names:
                    delay_subbar.set_description(f"Event delays: {event_name}")
                    delay_subbar.set_postfix_str(
                        f"n_events={len(np.asarray(events_by_name.get(event_name, np.array([]))).reshape(-1))}"
                    )
                    event_delay_config = dict(delay_config)
                    event_delay_config["EVENT_NAMES"] = [event_name]
                    with _PatchedModuleTqdm(ana_utils, enabled=False, leave=False):
                        df_event = _run_quietly(
                            ana_utils.calculate_event_delays,
                            spikes,
                            clusters,
                            cluster_acronyms_calc,
                            events_by_name,
                            event_delay_config,
                            eligible_cid_to_idx,
                            contrasts_by_name=contrasts_by_name,
                            trial_idx_by_name=trial_idx_by_name,
                            include_splits=True,
                            include_splits_events=[event_name],
                            output_path=None,
                        )
                    delay_frames.append(df_event)
                    delay_subbar.update(1)
            finally:
                delay_subbar.close()
            df_delay = _merge_delay_result_frames(delay_frames)
            calc_stage_bar.set_postfix_str(f"n_units={0 if df_delay is None else len(df_delay)}")
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 5, calc_stage_total, "Assigning whisk groups")
            selected_ids = (
                df_delay["cluster_id"].to_numpy(dtype=int)
                if isinstance(df_delay, pd.DataFrame) and "cluster_id" in df_delay.columns and not df_delay.empty
                else np.asarray(eligible_cluster_ids, dtype=int)
            )
            selected_ids = np.asarray(
                [cid for cid in selected_ids if int(cid) in eligible_cid_to_idx],
                dtype=int,
            )
            selected_acronyms = np.asarray(
                [cluster_acronyms_calc[eligible_cid_to_idx[int(cid)]] for cid in selected_ids],
                dtype=str,
            )
            selected_cid_to_idx = {int(cid): eligible_cid_to_idx[int(cid)] for cid in selected_ids}

            df_arousal = ana_utils.compute_arousal_groups_from_whisk(
                spikes,
                clusters,
                selected_acronyms,
                selected_cid_to_idx,
                selected_ids,
                wh_events_by_period.get("wh_brief_times_spont", np.array([])),
                dict(config_calc),
                whisk_times=df_wh["bin_center_s"].to_numpy(dtype=float),
                whisk_values=df_wh["wh_norm"].to_numpy(dtype=float),
                spont_intervals=spont_intervals,
            )

            if df_arousal is not None and not df_arousal.empty and not df_delay.empty:
                arousal_cols = [
                    "cluster_id",
                    "arousal_corr",
                    "arousal_corr_h1",
                    "arousal_corr_h2",
                    "arousal_corr_abs",
                    "arousal_mod",
                    "arousal_fr_hz",
                    "arousal_group",
                    "arousal_n_events",
                    "arousal_n_bins",
                ]
                df_delay = df_delay.merge(df_arousal[arousal_cols], on="cluster_id", how="left")
            else:
                for col in (
                    "arousal_corr",
                    "arousal_corr_h1",
                    "arousal_corr_h2",
                    "arousal_corr_abs",
                    "arousal_mod",
                    "arousal_fr_hz",
                ):
                    df_delay[col] = np.nan
                df_delay["arousal_group"] = "neutral"
                df_delay["arousal_n_events"] = 0
                df_delay["arousal_n_bins"] = 0

            arousal_group_mode = str(config_calc.get("AROUSAL_GROUP_MODE", "corr")).strip().lower()
            if arousal_group_mode == "response_sign":
                arousal_sign_event = str(config_calc.get("AROUSAL_SIGN_EVENT", "wh_brief_times_spont")).strip()
                arousal_sign_col = ana_utils.response_sign_column_name(arousal_sign_event)
                sign_to_group = {"exc": "arousal_plus", "inh": "arousal_minus", "none": "neutral"}
                if arousal_sign_col in df_delay.columns:
                    sign_vals = df_delay[arousal_sign_col].astype(str).str.lower()
                    df_delay["arousal_group"] = sign_vals.map(sign_to_group).fillna("neutral")
            df_delay["arousal_group"] = df_delay["arousal_group"].fillna("neutral")
            calc_stage_bar.set_postfix_str(f"n_units={len(selected_ids)}")
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 6, calc_stage_total, "Computing spontaneous whisk stPR")
            df_coupling_whisk = _compute_spont_context_coupling(
                spikes,
                clusters,
                cluster_acronyms_calc,
                dict(config_calc),
                cluster_ids=eligible_cluster_ids,
                intervals_odd=spont_whisk_odd,
                intervals_even=spont_whisk_even,
                context_label="Spont whisk",
                show_progress=show_calc_progress,
            )
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 7, calc_stage_total, "Computing spontaneous non-whisk stPR")
            df_coupling_nonwhisk = _compute_spont_context_coupling(
                spikes,
                clusters,
                cluster_acronyms_calc,
                dict(config_calc),
                cluster_ids=eligible_cluster_ids,
                intervals_odd=spont_nonwhisk_odd,
                intervals_even=spont_nonwhisk_even,
                context_label="Spont non-whisk",
                show_progress=show_calc_progress,
            )
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 8, calc_stage_total, "Building neuron summary table")
            variable_specs = _build_variable_specs()
            summary_neurons = _build_summary_neurons(
                eligible_cluster_ids,
                eligible_regions,
                cluster_acronyms_calc,
                cid_to_idx,
                clusters,
                df_delay,
                df_coupling_whisk,
                df_coupling_nonwhisk,
                show_progress=show_calc_progress,
            )
            calc_stage_bar.set_postfix_str(f"n_units={len(summary_neurons)}")
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 9, calc_stage_total, "Building region-pair summary table")
            summary_pairs = _build_summary_pairs(
                summary_neurons,
                variable_specs,
                show_progress=show_calc_progress,
            )
            calc_stage_bar.set_postfix_str(f"n_pairs={len(summary_pairs)}")
            calc_stage_bar.update(1)

            _set_progress_stage(calc_stage_bar, 10, calc_stage_total, "Saving lightweight cache")
            spont_coverage = {
                "spont_total_s": _intervals_total_duration(spont_intervals),
                "whisk_total_s": _intervals_total_duration(spont_whisk_intervals),
                "nonwhisk_total_s": _intervals_total_duration(spont_nonwhisk_intervals),
            }

            region_colors = _build_region_colors(summary_neurons["region"].astype(str).tolist())
            region_group_counts = (
                summary_neurons.groupby(["region", "whisk_group"]).size().rename("n_units").reset_index()
            )
            region_group_counts["region_color"] = region_group_counts["region"].map(region_colors)

            meta = _fetch_session_metadata(one, eid)
            meta.update(
                {
                    "pid": pid,
                    "eid": eid,
                    "one_mode": one_mode,
                    "n_trials": int(len(sl.trials)),
                    "eligible_regions": list(eligible_regions),
                    "eligible_units": int(len(eligible_cluster_ids)),
                }
            )

            cache_payload = {
                "meta": meta,
                "config_calc": dict(config_calc),
                "config_plot": dict(config_plot),
                "calc_version": calc_version,
                "calc_hash": calc_hash,
                "spont_intervals": spont_intervals,
                "spont_whisk_intervals": spont_whisk_intervals,
                "spont_nonwhisk_intervals": spont_nonwhisk_intervals,
                "spont_whisk_odd": spont_whisk_odd,
                "spont_whisk_even": spont_whisk_even,
                "spont_nonwhisk_odd": spont_nonwhisk_odd,
                "spont_nonwhisk_even": spont_nonwhisk_even,
                "spont_coverage": spont_coverage,
                "summary_neurons": summary_neurons,
                "summary_pairs": summary_pairs,
                "df_segments": df_segments,
                "df_delay": df_delay,
                "df_coupling_whisk": df_coupling_whisk,
                "df_coupling_nonwhisk": df_coupling_nonwhisk,
                "region_counts": eligibility["region_counts"],
                "region_group_counts": region_group_counts,
                "region_colors": region_colors,
                "variable_specs": variable_specs,
                "task_whisk_control_info": task_whisk_control_info,
            }

            summary_neurons.to_csv(cache_dir / "summary_neurons.csv", index=False)
            summary_pairs.to_csv(cache_dir / "summary_pairs.csv", index=False)
            df_segments.to_csv(cache_dir / "spont_segments.csv", index=False)
            with open(cache_pkl_path, "wb") as f:
                pickle.dump(cache_payload, f)
            manifest = {
                "pid": pid,
                "eid": eid,
                "calc_hash": calc_hash,
                "calc_version": calc_version,
                "config_calc": config_calc,
                "config_plot": config_plot,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
            calc_stage_bar.set_postfix_str("saved")
            calc_stage_bar.update(1)
        finally:
            calc_stage_bar.close()
        print(f"[{pid}] Saved cache to {cache_dir}")

    if return_payload and cache_payload is None:
        with open(cache_pkl_path, "rb") as f:
            cache_payload = pickle.load(f)

    return {
        "pid": pid,
        "eid": eid,
        "calc_hash": calc_hash,
        "cache_dir": cache_dir,
        "fig_dir": fig_dir,
        "cache_payload": cache_payload,
        "cache_hit": cache_hit,
        "eligible_regions": eligible_regions,
        "eligible_units": int(len(eligible_cluster_ids)),
    }


# %% PID and session loading params
PID = "f967a527-257f-404a-871d-b91575dca3b4"
PID_LIST = []
# When True, ignore PID and PID_LIST and process all insertions available to ONE.
# Set ALL_PIDS_TAG if you want to restrict this remote query to a specific Alyx tag.
RUN_ALL_PIDS = True
ALL_PIDS_TAG = "2025_Q3_IBL_et_al_BWM"
# Set False to skip recalculation and only load a saved cache for plotting.
RUN_CALCULATIONS = True
# Optional cache-only plotting mode. Use this to generate figures later from saved calculations
# without rerunning the full pipeline. If None, single-PID calculation mode plots automatically.
PLOT_FROM_CACHE_PID = None
PLOT_FROM_CACHE_HASH = None
CALC_LABEL_MIN = 0.9
# False: include neurons with label exactly equal to CALC_LABEL_MIN.
# True: keep only neurons with label strictly greater than CALC_LABEL_MIN.
CALC_LABEL_STRICT_GT = False
MIN_REGION_NEURONS = 15
ONE_PREFERRED_MODE = "remote"
ONE_ALLOW_REMOTE_FALLBACK = True
PLOTLY_DARK_THEME = False
SAVE_FIGURES = True
# Shows an overall calculation-stage bar plus nested bars for the heavier loops.
SHOW_CALC_PROGRESS = True


# %% Paths and plotting defaults
LOCAL_ONE_PARAMS_DIR = _configure_local_one_params_dir(BASE_PATH)
path_data, path_fig, path_data_processed, ibl_cache = setup_paths(BASE_PATH)
SEQ_PROCESSED_ROOT = path_data_processed / "15_ibl_seq_comparison"
SEQ_FIG_ROOT = path_fig / "15_ibl_seq_comparison"
SEQ_PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
SEQ_FIG_ROOT.mkdir(parents=True, exist_ok=True)

PLOTLY_TEMPLATE = "plotly_dark" if PLOTLY_DARK_THEME else "plotly_white"
pio.templates.default = PLOTLY_TEMPLATE


# %% ONE init and PID target resolution
try:
    one, one_mode = _init_one_with_fallback(
        ibl_cache,
        preferred_mode=ONE_PREFERRED_MODE,
        allow_remote=ONE_ALLOW_REMOTE_FALLBACK,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not initialize ONE. This script needs either a populated local ONE cache in "
        f"{ibl_cache} with PID lookup tables available, or remote Alyx access."
    ) from exc
print(f"ONE mode in use: {one_mode}")

ba = AllenAtlas() if AllenAtlas is not None else None
br = BrainRegions() if BrainRegions is not None else None

PID_TARGETS = _resolve_pid_targets(
    one,
    PID,
    pid_list=PID_LIST,
    run_all_pids=RUN_ALL_PIDS,
    all_pids_tag=ALL_PIDS_TAG,
)
if not RUN_CALCULATIONS and PLOT_FROM_CACHE_PID is None:
    raise RuntimeError("Nothing to do. Set RUN_CALCULATIONS=True or provide PLOT_FROM_CACHE_PID.")
if RUN_CALCULATIONS and not PID_TARGETS:
    raise RuntimeError("No target PIDs resolved. Set PID, PID_LIST, or RUN_ALL_PIDS.")

TARGET_MODE = (
    "all"
    if RUN_ALL_PIDS
    else ("list" if len(_coerce_pid_list(PID_LIST)) > 0 else "single")
)
PLOT_FROM_CACHE_PID = str(PLOT_FROM_CACHE_PID).strip() if PLOT_FROM_CACHE_PID is not None else None
if RUN_CALCULATIONS and len(PID_TARGETS) > 1 and PLOT_FROM_CACHE_PID is not None:
    print("Batch calculation mode disables plotting. Ignoring PLOT_FROM_CACHE_PID for this run.")
    PLOT_FROM_CACHE_PID = None
AUTO_PLOT_AFTER_CALC = RUN_CALCULATIONS and len(PID_TARGETS) == 1 and PLOT_FROM_CACHE_PID is None
SHOW_INNER_CALC_PROGRESS = bool(SHOW_CALC_PROGRESS and len(PID_TARGETS) == 1)

print(f"Target mode: {TARGET_MODE}")
if RUN_CALCULATIONS:
    print(f"Resolved {len(PID_TARGETS)} PID(s) for calculation.")
    if len(PID_TARGETS) <= 10:
        print("Targets:", PID_TARGETS)
    else:
        print("First 10 targets:", PID_TARGETS[:10])
if RUN_ALL_PIDS:
    print(
        "RUN_ALL_PIDS=True. "
        + (
            f"Restricting remote query to tag '{ALL_PIDS_TAG}'."
            if ALL_PIDS_TAG
            else "Querying all insertions available to the current ONE endpoint."
        )
    )
if PLOT_FROM_CACHE_PID is not None:
    print(f"Plot-from-cache PID: {PLOT_FROM_CACHE_PID}")


# %% Calculation config
CONFIG_CALC = {
    "CALC_LABEL_MIN": float(CALC_LABEL_MIN),
    # Cached copy of the threshold rule above:
    # False -> label >= CALC_LABEL_MIN
    # True  -> label >  CALC_LABEL_MIN
    "CALC_LABEL_STRICT_GT": bool(CALC_LABEL_STRICT_GT),
    "DELAY_METHOD": "com_signed",
    "DELAY_UNITS": "ms",
    "BIN_SIZE": 0.005,
    "BASELINE_PRE": 0.2,
    "PSTH_WINDOW_START": -1.0,
    "PSTH_WINDOW_END": 1.0,
    "RESPONSIVE_WINDOW_START": -0.1,
    "RESPONSIVE_WINDOW_END": 0.2,
    "RESPONSIVE_USE_ZSCORE": True,
    "RESPONSIVE_ZSCORE_SOURCE": "smooth",
    "COM_USE_THRESHOLD": True,
    "SMOOTH_SIGMA": 1,
    "MIN_TRIALS": 10,
    "MIN_TRIALS_SPLIT": 5,
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_POP_USE_GOOD_UNITS": False,
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
    "SPONT_WHISK_PAD_PRE_S": 0.05,
    "SPONT_WHISK_PAD_POST_S": 0.05,
    # Task whisk events are filtered before delay calculation so they exclude
    # events whose response window overlaps the first-move response window.
    # With the default windows below this becomes:
    # first move within [task_whisk - 0.3 s, task_whisk + 0.2 s].
    "TASK_WHISK_MOVE_CONTROL_ENABLED": True,
    "TASK_WHISK_MOVE_EXCLUDE_PRE_S": None,
    "TASK_WHISK_MOVE_EXCLUDE_POST_S": None,
    # "corr" uses spontaneous whisk/firing correlations in full + split halves.
    # "response_sign" uses the sign of the whisk-event response column below.
    "AROUSAL_GROUP_MODE": "response_sign",
    "AROUSAL_SIGN_EVENT": "wh_brief_times_spont",
    "AROUSAL_POS_THR": 0.05,
    "AROUSAL_NEG_THR": -0.05,
    "AROUSAL_BIN_S": 0.3,
    "AROUSAL_SMOOTH_SIGMA": 5,
    "AROUSAL_MIN_CORR_BINS": 10,
    "AROUSAL_USE_EVENT_BASELINE_ZSCORE": True,
    "AROUSAL_BASELINE_PRE": 0.2,
    "AROUSAL_MIN_BASELINE_BINS": 3,
    "AROUSAL_REQUIRE_SPLIT_HALF": True,
    "DELAY_WINDOWS": {
        "firstMovement_times": (-0.1, 0.2),
        "feedback_times": (0.0, 0.4),
        "wh_all_times_spont": (0.0, 0.2),
        "wh_all_times_task": (0.0, 0.2),
    },
}

CONFIG_PLOT = {
    "PLOTLY_TEMPLATE": PLOTLY_TEMPLATE,
    "SAVE_FIGURES": bool(SAVE_FIGURES),
    "FIG_DIR_ROOT": str(SEQ_FIG_ROOT),
}


# %% Cache-aware calculations
calculation_results = []
successful_results = []
pid_run_summary = pd.DataFrame(
    columns=["pid", "status", "reason", "cache_hit", "calc_hash", "eligible_units", "cache_dir"]
)
if RUN_CALCULATIONS:
    summary_rows = []
    pid_iter = _progress_iter(
        PID_TARGETS,
        desc="Processing PIDs",
        total=len(PID_TARGETS),
        unit="pid",
        leave=True,
        enabled=SHOW_CALC_PROGRESS and len(PID_TARGETS) > 1,
    )
    try:
        for pid_value in pid_iter:
            pid_str = str(pid_value)
            try:
                result = _run_single_pid_calculations(
                    pid_str,
                    one,
                    one_mode,
                    ba,
                    br,
                    CONFIG_CALC,
                    CONFIG_PLOT,
                    SEQ_PROCESSED_ROOT,
                    SEQ_FIG_ROOT,
                    MIN_REGION_NEURONS,
                    CALC_VERSION,
                    show_calc_progress=SHOW_INNER_CALC_PROGRESS,
                    return_payload=AUTO_PLOT_AFTER_CALC,
                    suppress_load_stdout=len(PID_TARGETS) > 1,
                )
                calculation_results.append(result)
                successful_results.append(result)
                summary_rows.append(
                    {
                        "pid": pid_str,
                        "status": "ok",
                        "reason": "",
                        "cache_hit": bool(result.get("cache_hit", False)),
                        "calc_hash": result.get("calc_hash"),
                        "eligible_units": int(result.get("eligible_units", 0)),
                        "cache_dir": str(result.get("cache_dir", "")),
                    }
                )
                if hasattr(pid_iter, "set_postfix_str"):
                    pid_iter.set_postfix_str(f"ok={len(successful_results)}")
            except _PidSkipError as exc:
                if len(PID_TARGETS) == 1:
                    raise
                summary_rows.append(
                    {
                        "pid": pid_str,
                        "status": "skipped",
                        "reason": str(exc),
                        "cache_hit": False,
                        "calc_hash": "",
                        "eligible_units": 0,
                        "cache_dir": "",
                    }
                )
                if hasattr(pid_iter, "set_postfix_str"):
                    pid_iter.set_postfix_str(f"ok={len(successful_results)} | skipped")
            except Exception as exc:
                if len(PID_TARGETS) == 1:
                    raise
                summary_rows.append(
                    {
                        "pid": pid_str,
                        "status": "failed",
                        "reason": str(exc),
                        "cache_hit": False,
                        "calc_hash": "",
                        "eligible_units": 0,
                        "cache_dir": "",
                    }
                )
                if hasattr(pid_iter, "set_postfix_str"):
                    pid_iter.set_postfix_str(f"ok={len(successful_results)} | failed")
    finally:
        if hasattr(pid_iter, "close"):
            pid_iter.close()
    pid_run_summary = pd.DataFrame(summary_rows)
    if not pid_run_summary.empty:
        status_counts = pid_run_summary["status"].value_counts().to_dict()
        print(f"PID run summary: {status_counts}")
        display(pid_run_summary.head(20))
else:
    print("RUN_CALCULATIONS=False. Skipping calculation pass.")


# %% Load cached outputs into working variables
plot_context = None
if PLOT_FROM_CACHE_PID is not None:
    plot_context = _load_cached_plot_context(
        SEQ_PROCESSED_ROOT,
        SEQ_FIG_ROOT,
        PLOT_FROM_CACHE_PID,
        calc_hash=PLOT_FROM_CACHE_HASH,
    )
elif AUTO_PLOT_AFTER_CALC and len(successful_results) == 1:
    plot_context = successful_results[0]
    if plot_context.get("cache_payload") is None:
        plot_context = _load_cached_plot_context(
            SEQ_PROCESSED_ROOT,
            SEQ_FIG_ROOT,
            plot_context["pid"],
            calc_hash=plot_context["calc_hash"],
        )

PLOTTING_ACTIVE = plot_context is not None
PLOT_PID_ACTIVE = None
PLOT_CALC_HASH_ACTIVE = None
PLOT_FIG_DIR = None
PLOT_SAVE_FIGURES = False
cache_payload = None
summary_neurons = pd.DataFrame()
summary_pairs = pd.DataFrame()
df_segments = pd.DataFrame()
df_delay = pd.DataFrame()
df_coupling_whisk = None
df_coupling_nonwhisk = None
region_counts = pd.DataFrame()
region_group_counts = pd.DataFrame()
region_colors = {}
variable_specs = []
task_whisk_control_info = {}
spont_coverage = {}
regions_order = []

if PLOTTING_ACTIVE:
    cache_payload = plot_context["cache_payload"]
    PLOT_PID_ACTIVE = str(plot_context["pid"])
    PLOT_CALC_HASH_ACTIVE = str(plot_context.get("calc_hash", cache_payload.get("calc_hash", "")))
    PLOT_FIG_DIR = Path(plot_context["fig_dir"])
    PLOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_SAVE_FIGURES = bool(SAVE_FIGURES)

    summary_neurons = cache_payload["summary_neurons"].copy()
    summary_pairs = cache_payload["summary_pairs"].copy()
    df_segments = cache_payload["df_segments"].copy()
    df_delay = cache_payload["df_delay"].copy()
    df_coupling_whisk = cache_payload["df_coupling_whisk"]
    df_coupling_nonwhisk = cache_payload["df_coupling_nonwhisk"]
    region_counts = cache_payload["region_counts"].copy()
    region_group_counts = cache_payload["region_group_counts"].copy()
    region_colors = dict(cache_payload["region_colors"])
    variable_specs = list(cache_payload["variable_specs"])
    task_whisk_control_info = dict(cache_payload.get("task_whisk_control_info", {}))
    spont_coverage = dict(cache_payload["spont_coverage"])

    eligible_regions_active = list(cache_payload.get("meta", {}).get("eligible_regions", []))
    if not eligible_regions_active:
        eligible_regions_active = sorted(summary_neurons["region"].astype(str).dropna().unique().tolist())
    regions_order = [r for r in eligible_regions_active if r in region_group_counts["region"].astype(str).tolist()]

    print(f"Plot PID: {PLOT_PID_ACTIVE}")
    print(f"Plot cache hash: {PLOT_CALC_HASH_ACTIVE}")
    print(f"Summary neurons: {summary_neurons.shape}")
    print(f"Summary pairs: {summary_pairs.shape}")
    display(summary_neurons.head())
else:
    print("Plotting disabled. Use a single PID target or set PLOT_FROM_CACHE_PID to load a saved cache.")


# %% Plot 1: spontaneous whisk vs non-whisk coverage
if PLOTTING_ACTIVE:
    coverage_df = pd.DataFrame(
        {
            "state": ["Whisking", "Non-whisking"],
            "seconds": [
                float(spont_coverage.get("whisk_total_s", 0.0)),
                float(spont_coverage.get("nonwhisk_total_s", 0.0)),
            ],
            "color": ["rgb(255,127,14)", "rgb(120,120,120)"],
        }
    )
    fig_spont_coverage = go.Figure(
        data=[
            go.Bar(
                x=coverage_df["state"],
                y=coverage_df["seconds"],
                marker=dict(color=coverage_df["color"]),
                text=np.round(coverage_df["seconds"], 2),
                textposition="outside",
            )
        ]
    )
    fig_spont_coverage.update_layout(
        title=f"Spontaneous Coverage | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        xaxis_title="State",
        yaxis_title="Seconds",
        height=520,
    )
    show_fig(fig_spont_coverage)
    _save_fig(
        fig_spont_coverage,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_01_spont_coverage",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 2: whisk group counts per region
if PLOTTING_ACTIVE:
    count_lookup = {}
    for group_name in ("wh+", "neutral", "wh-"):
        group_df = region_group_counts.loc[region_group_counts["whisk_group"] == group_name].copy()
        count_lookup[group_name] = {
            str(row["region"]): int(row["n_units"]) for _, row in group_df.iterrows()
        }

    fig_group_counts = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("wh+", "neutral", "wh-"),
        shared_yaxes=True,
    )
    for col_idx, group_name in enumerate(("wh+", "neutral", "wh-"), start=1):
        x_vals = regions_order
        y_vals = [count_lookup[group_name].get(region, 0) for region in x_vals]
        colors = [region_colors.get(region, "gray") for region in x_vals]
        fig_group_counts.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                marker=dict(color=colors),
                text=y_vals,
                textposition="outside",
                showlegend=False,
                hovertemplate="Region %{x}<br>Units %{y}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
    fig_group_counts.update_layout(
        title=f"Whisk Group Counts by Region | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        height=560,
        width=max(900, 180 * len(regions_order)),
    )
    fig_group_counts.update_xaxes(tickangle=45)
    fig_group_counts.update_yaxes(title_text="Units", row=1, col=1)
    show_fig(fig_group_counts)
    _save_fig(
        fig_group_counts,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_02_whisk_group_counts",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 3: reliability scatter grids by region
for region_name in regions_order:
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region_name)].copy()
    if df_region.empty:
        continue
    fig_rel = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[spec["name"] for spec in variable_specs],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    region_color = region_colors.get(region_name, "gray")
    for idx_spec, spec in enumerate(variable_specs):
        row = idx_spec // 3 + 1
        col = idx_spec % 3 + 1
        x_vals = df_region[spec["odd_col"]].to_numpy(dtype=float)
        y_vals = df_region[spec["even_col"]].to_numpy(dtype=float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        plot_df = df_region.loc[mask, ["cluster_id", "region", "whisk_group", spec["odd_col"], spec["even_col"]]].copy()
        plot_df = plot_df.rename(columns={spec["odd_col"]: "x", spec["even_col"]: "y"})
        if not plot_df.empty:
            _add_grouped_scatter_points(fig_rel, plot_df, "x", "y", region_color, row=row, col=col)
            _add_unity_line(fig_rel, plot_df["x"].to_numpy(dtype=float), plot_df["y"].to_numpy(dtype=float), row=row, col=col)
        r_p, n_p = _pearsonr_with_n(x_vals, y_vals)
        r_s, _ = _spearmanr_with_n(x_vals, y_vals)
        fig_rel.add_annotation(
            x=0.98,
            y=0.98,
            xref=f"x{idx_spec + 1} domain" if idx_spec > 0 else "x domain",
            yref=f"y{idx_spec + 1} domain" if idx_spec > 0 else "y domain",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(120,120,120,0.5)",
            borderwidth=1,
            text=f"r={_format_corr_value(r_p)}<br>rho={_format_corr_value(r_s)}<br>n={n_p}",
        )
        fig_rel.update_xaxes(title_text="Odd", row=row, col=col)
        fig_rel.update_yaxes(title_text="Even", row=row, col=col)
    fig_rel.update_layout(
        title=f"Reliability Scatter Grid | Region {region_name} | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        height=820,
        width=1180,
    )
    show_fig(fig_rel)
    _save_fig(
        fig_rel,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_03_reliability_{region_name}",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 4: Pearson and Spearman matrices by region
for region_name in regions_order:
    names_p, pearson_mat, pearson_text = _build_pairwise_matrix(summary_neurons, variable_specs, region_name, metric="pearson")
    names_s, spearman_mat, spearman_text = _build_pairwise_matrix(summary_neurons, variable_specs, region_name, metric="spearman")
    fig_corr = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pearson", "Spearman"),
        horizontal_spacing=0.10,
    )
    fig_corr.add_trace(
        go.Heatmap(
            z=pearson_mat,
            x=names_p,
            y=names_p,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=pearson_text,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_corr.add_trace(
        go.Heatmap(
            z=spearman_mat,
            x=names_s,
            y=names_s,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=spearman_text,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_corr.update_layout(
        title=f"Correlation Matrices | Region {region_name} | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        height=720,
        width=1440,
        margin=dict(l=70, r=50, t=90, b=140),
    )
    fig_corr.update_xaxes(tickangle=45)
    fig_corr.update_yaxes(autorange="reversed")
    show_fig(fig_corr)
    _save_fig(
        fig_corr,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_04_corr_matrices_{region_name}",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 5: interactive all-region variable scatter
available_var_names = [spec["name"] for spec in variable_specs]
if len(available_var_names) >= 2:
    default_x = "First Move Delay" if "First Move Delay" in available_var_names else available_var_names[0]
    default_y = "Task Whisk Event Delay" if "Task Whisk Event Delay" in available_var_names else available_var_names[1]
    var_x_name = _choose_option(available_var_names, "Variable X options:", default_x)
    var_y_name = _choose_option(available_var_names, "Variable Y options:", default_y)
    if var_y_name == var_x_name:
        fallback = [name for name in available_var_names if name != var_x_name]
        if fallback:
            var_y_name = fallback[0]
            print(f"Variable Y matched X; using '{var_y_name}' for Y.")
    spec_x = next(spec for spec in variable_specs if spec["name"] == var_x_name)
    spec_y = next(spec for spec in variable_specs if spec["name"] == var_y_name)
    plot_df = summary_neurons[["cluster_id", "region", "whisk_group", spec_x["mean_col"], spec_y["mean_col"]]].copy()
    plot_df = plot_df.rename(columns={spec_x["mean_col"]: "x", spec_y["mean_col"]: "y"})
    mask = np.isfinite(plot_df["x"].to_numpy(dtype=float)) & np.isfinite(plot_df["y"].to_numpy(dtype=float))
    plot_df = plot_df.loc[mask].reset_index(drop=True)
    if plot_df.empty:
        print("No overlapping finite values for the selected variables.")
    else:
        pearson_r, n_p = _pearsonr_with_n(plot_df["x"], plot_df["y"])
        spearman_rho, n_s = _spearmanr_with_n(plot_df["x"], plot_df["y"])
        rel_x, n_rel_x = _pearsonr_with_n(summary_neurons[spec_x["odd_col"]], summary_neurons[spec_x["even_col"]])
        rel_y, n_rel_y = _pearsonr_with_n(summary_neurons[spec_y["odd_col"]], summary_neurons[spec_y["even_col"]])
        symbol_map = {"wh+": "circle", "neutral": "diamond", "wh-": "x"}
        fig_scatter = go.Figure()
        for region_name in regions_order:
            for whisk_group in ("wh+", "neutral", "wh-"):
                grp = plot_df.loc[
                    (plot_df["region"].astype(str) == str(region_name))
                    & (plot_df["whisk_group"].astype(str) == str(whisk_group))
                ].copy()
                if grp.empty:
                    continue
                fig_scatter.add_trace(
                    go.Scatter(
                        x=grp["x"].to_numpy(dtype=float),
                        y=grp["y"].to_numpy(dtype=float),
                        mode="markers",
                        name=f"{region_name} | {whisk_group}",
                        customdata=grp[["cluster_id", "region", "whisk_group"]].to_numpy(),
                        marker=dict(
                            color=region_colors.get(region_name, "gray"),
                            size=7,
                            opacity=0.75,
                            symbol=symbol_map.get(whisk_group, "circle"),
                            line=dict(width=0.8, color="black"),
                        ),
                        hovertemplate=(
                            "Cluster %{customdata[0]}<br>"
                            "Region %{customdata[1]}<br>"
                            "Whisk group %{customdata[2]}<br>"
                            f"{var_x_name}: %{{x:.3f}}<br>"
                            f"{var_y_name}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )
        _add_unity_line(fig_scatter, plot_df["x"].to_numpy(dtype=float), plot_df["y"].to_numpy(dtype=float))
        fig_scatter.update_layout(
            title=f"All-Region Variable Scatter | {var_x_name} vs {var_y_name} | PID {PLOT_PID_ACTIVE}",
            template=PLOTLY_TEMPLATE,
            width=980,
            height=760,
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
            margin=dict(l=70, r=40, t=95, b=150),
        )
        fig_scatter.update_xaxes(title_text=var_x_name)
        fig_scatter.update_yaxes(title_text=var_y_name)
        fig_scatter.add_annotation(
            x=0.99,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            align="left",
            bordercolor="rgba(120,120,120,0.6)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            text=(
                f"Pearson r={_format_corr_value(pearson_r)} (n={n_p})<br>"
                f"Spearman rho={_format_corr_value(spearman_rho)} (n={n_s})<br>"
                f"Reliability X={_format_corr_value(rel_x)} (n={n_rel_x})<br>"
                f"Reliability Y={_format_corr_value(rel_y)} (n={n_rel_y})"
            ),
        )
        show_fig(fig_scatter)
        _save_fig(
            fig_scatter,
            PLOT_FIG_DIR,
            f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_05_scatter_{_slugify(var_x_name)}__{_slugify(var_y_name)}",
            save_flag=PLOT_SAVE_FIGURES,
        )


# %% Plot 6: interactive cross-trigger rank-order plot
if regions_order:
    region_default = regions_order[0]
    region_choice = _choose_option(regions_order, "Region options for rank-order plot:", region_default)
    anchor_name = _choose_option(
        available_var_names,
        "Anchor variable options:",
        "First Move Delay" if "First Move Delay" in available_var_names else available_var_names[0],
    )
    anchor_spec = next(spec for spec in variable_specs if spec["name"] == anchor_name)
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region_choice)].copy()
    df_region = df_region.loc[np.isfinite(df_region[anchor_spec["mean_col"]].to_numpy(dtype=float))].copy()
    if df_region.empty:
        print(f"No finite anchor values for region {region_choice}.")
    else:
        df_region = df_region.sort_values(anchor_spec["mean_col"]).reset_index(drop=True)
        rank = np.arange(df_region.shape[0], dtype=int)
        fig_rank = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Raw Delays (ms)", "Z-scored Shape"),
            horizontal_spacing=0.10,
        )
        dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
        region_color = region_colors.get(region_choice, "gray")
        for idx_spec, spec in enumerate(variable_specs):
            y_raw = df_region[spec["mean_col"]].to_numpy(dtype=float)
            y_z = _zscore_nan(y_raw)
            style = dash_styles[idx_spec % len(dash_styles)]
            base_kwargs = dict(
                mode="lines+markers",
                name=spec["name"],
                line=dict(color=region_color, dash=style, width=2),
                marker=dict(size=5, color=region_color),
                hovertemplate="Rank %{x}<br>Value %{y:.3f}<extra></extra>",
            )
            fig_rank.add_trace(go.Scatter(x=rank, y=y_raw, **base_kwargs), row=1, col=1)
            fig_rank.add_trace(go.Scatter(x=rank, y=y_z, **base_kwargs, showlegend=False), row=1, col=2)
        fig_rank.update_layout(
            title=f"Cross-Trigger Rank Order | Region {region_choice} | Anchor {anchor_name} | PID {PLOT_PID_ACTIVE}",
            template=PLOTLY_TEMPLATE,
            width=1420,
            height=620,
        )
        fig_rank.update_xaxes(title_text="Neuron rank", row=1, col=1)
        fig_rank.update_xaxes(title_text="Neuron rank", row=1, col=2)
        fig_rank.update_yaxes(title_text="Delay (ms)", row=1, col=1)
        fig_rank.update_yaxes(title_text="Z-score", row=1, col=2)
        show_fig(fig_rank)
        _save_fig(
            fig_rank,
            PLOT_FIG_DIR,
            f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_06_rank_order_{region_choice}__{_slugify(anchor_name)}",
            save_flag=PLOT_SAVE_FIGURES,
        )


# %% Plot 7: interactive delay-difference distribution
if regions_order:
    diff_region = _choose_option(regions_order, "Region options for delay-difference plot:", regions_order[0])
    diff_x_name = _choose_option(available_var_names, "Difference variable A options:", available_var_names[0])
    diff_y_name = _choose_option(
        available_var_names,
        "Difference variable B options:",
        available_var_names[1] if len(available_var_names) > 1 else available_var_names[0],
    )
    if diff_y_name == diff_x_name:
        fallback = [name for name in available_var_names if name != diff_x_name]
        if fallback:
            diff_y_name = fallback[0]
    diff_x_spec = next(spec for spec in variable_specs if spec["name"] == diff_x_name)
    diff_y_spec = next(spec for spec in variable_specs if spec["name"] == diff_y_name)
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(diff_region)].copy()
    vals_x = df_region[diff_x_spec["mean_col"]].to_numpy(dtype=float)
    vals_y = df_region[diff_y_spec["mean_col"]].to_numpy(dtype=float)
    valid = np.isfinite(vals_x) & np.isfinite(vals_y)
    diff_vals = vals_x[valid] - vals_y[valid]
    if diff_vals.size == 0:
        print("No overlapping finite values for the selected difference plot.")
    else:
        rel_x, _ = _pearsonr_with_n(df_region[diff_x_spec["odd_col"]], df_region[diff_x_spec["even_col"]])
        rel_y, _ = _pearsonr_with_n(df_region[diff_y_spec["odd_col"]], df_region[diff_y_spec["even_col"]])
        fig_diff = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        fig_diff.add_trace(
            go.Histogram(
                x=diff_vals,
                nbinsx=max(12, min(60, int(np.sqrt(diff_vals.size) * 3))),
                marker=dict(color=region_colors.get(diff_region, "gray")),
                showlegend=False,
                hovertemplate="Diff %{x:.3f}<br>Count %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig_diff.add_trace(
            go.Box(
                x=diff_vals,
                marker=dict(color=region_colors.get(diff_region, "gray")),
                boxpoints="all",
                jitter=0.25,
                pointpos=0.0,
                showlegend=False,
                hovertemplate="Diff %{x:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        mean_diff = float(np.nanmean(diff_vals))
        median_diff = float(np.nanmedian(diff_vals))
        fig_diff.add_vline(x=0.0, line_dash="dash", line_color="red")
        fig_diff.add_vline(x=mean_diff, line_dash="dot", line_color="black")
        fig_diff.update_layout(
            title=f"Delay Difference | {diff_x_name} - {diff_y_name} | Region {diff_region} | PID {PLOT_PID_ACTIVE}",
            template=PLOTLY_TEMPLATE,
            width=980,
            height=700,
        )
        fig_diff.update_xaxes(title_text="A - B (ms)", row=2, col=1)
        fig_diff.update_yaxes(title_text="Count", row=1, col=1)
        fig_diff.add_annotation(
            x=0.99,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(120,120,120,0.6)",
            borderwidth=1,
            text=(
                f"Mean={mean_diff:.3f}<br>"
                f"Median={median_diff:.3f}<br>"
                f"SD={float(np.nanstd(diff_vals)):.3f}<br>"
                f"MAD={_median_abs_dev(diff_vals):.3f}<br>"
                f"Rel A={_format_corr_value(rel_x)}<br>"
                f"Rel B={_format_corr_value(rel_y)}"
            ),
        )
        show_fig(fig_diff)
        _save_fig(
            fig_diff,
            PLOT_FIG_DIR,
            f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_07_diff_{diff_region}__{_slugify(diff_x_name)}__{_slugify(diff_y_name)}",
            save_flag=PLOT_SAVE_FIGURES,
        )


# %% Plot 8: order-agreement heatmaps by region
for region_name in regions_order:
    names_oa, oa_mat, oa_text = _build_order_agreement_matrix(summary_neurons, variable_specs, region_name)
    fig_oa = go.Figure(
        data=go.Heatmap(
            z=oa_mat,
            x=names_oa,
            y=names_oa,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=oa_text,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig_oa.update_layout(
        title=f"Order-Agreement Matrix | Region {region_name} | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        height=760,
        width=820,
        margin=dict(l=80, r=40, t=90, b=150),
    )
    fig_oa.update_xaxes(tickangle=45)
    fig_oa.update_yaxes(autorange="reversed")
    show_fig(fig_oa)
    _save_fig(
        fig_oa,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_08_order_agreement_{region_name}",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 9: reliability vs similarity summary
if not summary_pairs.empty:
    fig_rel_vs = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Reliability Floor vs Pearson", "Reliability Floor vs Order Agreement"),
        horizontal_spacing=0.10,
    )
    for region_name in regions_order:
        grp = summary_pairs.loc[summary_pairs["region"].astype(str) == str(region_name)].copy()
        if grp.empty:
            continue
        color = region_colors.get(region_name, "gray")
        customdata = grp[["pair_label", "n_shared"]].to_numpy()
        fig_rel_vs.add_trace(
            go.Scatter(
                x=grp["reliability_floor"].to_numpy(dtype=float),
                y=grp["pearson_r"].to_numpy(dtype=float),
                mode="markers",
                name=region_name,
                legendgroup=region_name,
                marker=dict(color=color, size=8, opacity=0.75, line=dict(width=0.5, color="black")),
                customdata=customdata,
                hovertemplate="Region: " + region_name + "<br>%{customdata[0]}<br>n=%{customdata[1]}<br>Rel floor=%{x:.3f}<br>Pearson=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig_rel_vs.add_trace(
            go.Scatter(
                x=grp["reliability_floor"].to_numpy(dtype=float),
                y=grp["order_agreement_score"].to_numpy(dtype=float),
                mode="markers",
                name=region_name,
                legendgroup=region_name,
                showlegend=False,
                marker=dict(color=color, size=8, opacity=0.75, line=dict(width=0.5, color="black")),
                customdata=customdata,
                hovertemplate="Region: " + region_name + "<br>%{customdata[0]}<br>n=%{customdata[1]}<br>Rel floor=%{x:.3f}<br>Order agreement=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig_rel_vs.update_layout(
        title=f"Reliability vs Similarity Summary | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        width=1320,
        height=620,
    )
    fig_rel_vs.update_xaxes(title_text="Reliability floor", row=1, col=1)
    fig_rel_vs.update_yaxes(title_text="Pearson r", row=1, col=1)
    fig_rel_vs.update_xaxes(title_text="Reliability floor", row=1, col=2)
    fig_rel_vs.update_yaxes(title_text="Order agreement", row=1, col=2)
    show_fig(fig_rel_vs)
    _save_fig(
        fig_rel_vs,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_09_reliability_vs_similarity",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 10: region-by-pair summary heatmap
if not summary_pairs.empty:
    pair_order = (
        summary_pairs.groupby("pair_label", as_index=False)["order_agreement_score"]
        .mean(numeric_only=True)
        .sort_values("order_agreement_score", ascending=False)["pair_label"]
        .tolist()
    )
    heat_rows = []
    text_rows = []
    for region_name in regions_order:
        grp = summary_pairs.loc[summary_pairs["region"].astype(str) == str(region_name)].copy()
        row_vals = []
        row_text = []
        for pair_label in pair_order:
            row = grp.loc[grp["pair_label"].astype(str) == str(pair_label)]
            if row.empty:
                row_vals.append(np.nan)
                row_text.append("oa=nan<br>r=nan<br>n=0")
            else:
                row0 = row.iloc[0]
                row_vals.append(float(row0["order_agreement_score"]) if pd.notna(row0["order_agreement_score"]) else np.nan)
                row_text.append(
                    f"oa={_format_corr_value(row0['order_agreement_score'])}<br>"
                    f"r={_format_corr_value(row0['pearson_r'])}<br>"
                    f"n={int(row0['n_shared'])}"
                )
        heat_rows.append(row_vals)
        text_rows.append(row_text)
    fig_region_pair = go.Figure(
        data=go.Heatmap(
            z=np.asarray(heat_rows, dtype=float),
            x=pair_order,
            y=regions_order,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.asarray(text_rows, dtype=object),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
        )
    )
    fig_region_pair.update_layout(
        title=f"Region by Variable Pair Summary | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        height=max(560, 70 * len(regions_order) + 180),
        width=max(1200, 120 * len(pair_order) + 220),
        margin=dict(l=90, r=50, t=90, b=170),
    )
    fig_region_pair.update_xaxes(tickangle=45)
    fig_region_pair.update_yaxes(autorange="reversed")
    show_fig(fig_region_pair)
    _save_fig(
        fig_region_pair,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_10_region_by_pair_summary",
        save_flag=PLOT_SAVE_FIGURES,
    )


# %% Plot 11: coupling-vs-response grids by region
coupling_specs = [spec for spec in variable_specs if "Coupling" in spec["name"]]
response_specs = [spec for spec in variable_specs if "Coupling" not in spec["name"]]
for region_name in regions_order:
    df_region = summary_neurons.loc[summary_neurons["region"].astype(str) == str(region_name)].copy()
    if df_region.empty:
        continue
    fig_grid = make_subplots(
        rows=len(coupling_specs),
        cols=len(response_specs),
        subplot_titles=[
            f"{spec_x['name']} vs {spec_y['name']}"
            for spec_x in coupling_specs
            for spec_y in response_specs
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )
    region_color = region_colors.get(region_name, "gray")
    for row_idx, spec_x in enumerate(coupling_specs, start=1):
        for col_idx, spec_y in enumerate(response_specs, start=1):
            x = df_region[spec_x["mean_col"]].to_numpy(dtype=float)
            y = df_region[spec_y["mean_col"]].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            plot_df = df_region.loc[mask, ["cluster_id", "region", "whisk_group", spec_x["mean_col"], spec_y["mean_col"]]].copy()
            plot_df = plot_df.rename(columns={spec_x["mean_col"]: "x", spec_y["mean_col"]: "y"})
            if not plot_df.empty:
                _add_grouped_scatter_points(fig_grid, plot_df, "x", "y", region_color, row=row_idx, col=col_idx)
            pearson_r, n_p = _pearsonr_with_n(x, y)
            spearman_rho, _ = _spearmanr_with_n(x, y)
            rel_x, _ = _pearsonr_with_n(df_region[spec_x["odd_col"]], df_region[spec_x["even_col"]])
            rel_y, _ = _pearsonr_with_n(df_region[spec_y["odd_col"]], df_region[spec_y["even_col"]])
            rel_floor = np.nanmin([rel_x, rel_y])
            axis_num = (row_idx - 1) * len(response_specs) + col_idx
            xref = "x domain" if axis_num == 1 else f"x{axis_num} domain"
            yref = "y domain" if axis_num == 1 else f"y{axis_num} domain"
            fig_grid.add_annotation(
                x=0.98,
                y=0.98,
                xref=xref,
                yref=yref,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="rgba(120,120,120,0.5)",
                borderwidth=1,
                text=(
                    f"r={_format_corr_value(pearson_r)}<br>"
                    f"rho={_format_corr_value(spearman_rho)}<br>"
                    f"n={n_p}<br>"
                    f"rel floor={_format_corr_value(rel_floor)}"
                ),
            )
            fig_grid.update_xaxes(title_text=spec_x["name"], row=row_idx, col=col_idx)
            fig_grid.update_yaxes(title_text=spec_y["name"], row=row_idx, col=col_idx)
    fig_grid.update_layout(
        title=f"Coupling vs Response Delay Grid | Region {region_name} | PID {PLOT_PID_ACTIVE}",
        template=PLOTLY_TEMPLATE,
        width=1850,
        height=820,
        margin=dict(l=60, r=40, t=90, b=80),
    )
    show_fig(fig_grid)
    _save_fig(
        fig_grid,
        PLOT_FIG_DIR,
        f"{PLOT_PID_ACTIVE}_{PLOT_CALC_HASH_ACTIVE}_11_coupling_vs_response_{region_name}",
        save_flag=PLOT_SAVE_FIGURES,
    )
