import numpy as np
import utils.io as io_utils
import pandas as pd
from scipy.ndimage import gaussian_filter1d
try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None
try:
    from tqdm.auto import tqdm
except ImportError:  # Graceful fallback if tqdm is unavailable.
    def tqdm(iterable, **kwargs):
        return iterable

from rastermap import Rastermap, utils

_LATENZY_FN = None
_LATENZY_IMPORT_ATTEMPTED = False
_LATENZY_IMPORT_ERROR = None
_LATENZY_WARNED_UNAVAILABLE = False
_LATENZY_WARNED_RUNTIME = False


def _get_latenzy_callable():
    """Lazy-load optional latenzy dependency."""
    global _LATENZY_FN
    global _LATENZY_IMPORT_ATTEMPTED
    global _LATENZY_IMPORT_ERROR
    if _LATENZY_IMPORT_ATTEMPTED:
        return _LATENZY_FN
    _LATENZY_IMPORT_ATTEMPTED = True
    try:
        from latenzy import latenzy as _latenzy_fn
        _LATENZY_FN = _latenzy_fn
        return _LATENZY_FN
    except Exception as exc:
        _LATENZY_IMPORT_ERROR = exc
        try:
            import latenzy as _latenzy_mod
            candidate = getattr(_latenzy_mod, "latenzy", None)
            if callable(candidate):
                _LATENZY_FN = candidate
                return _LATENZY_FN
            _LATENZY_IMPORT_ERROR = RuntimeError(
                "imported module 'latenzy' has no callable 'latenzy' attribute"
            )
        except Exception:
            pass
    return None


def _warn_latenzy_unavailable_once():
    global _LATENZY_WARNED_UNAVAILABLE
    if _LATENZY_WARNED_UNAVAILABLE:
        return
    _LATENZY_WARNED_UNAVAILABLE = True
    msg = (
        "Warning: DELAY_METHOD='latenzy' requested but package 'latenzy' could not be "
        "imported."
    )
    if _LATENZY_IMPORT_ERROR is not None:
        msg = f"{msg} ({_LATENZY_IMPORT_ERROR})"
    print(msg)


def _warn_latenzy_runtime_once(exc):
    global _LATENZY_WARNED_RUNTIME
    if _LATENZY_WARNED_RUNTIME:
        return
    _LATENZY_WARNED_RUNTIME = True
    print(f"Warning: latenzy failed at runtime; returning NaN delays. ({exc})")


def _first_finite_scalar(value):
    if value is None:
        return np.nan
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.nan
    if arr.size == 0:
        return np.nan
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        return float(finite[0])
    return np.nan


def _parse_latenzy_output(output):
    latency = np.nan
    score = np.nan
    if isinstance(output, (tuple, list)):
        if len(output) >= 1:
            latency = _first_finite_scalar(output[0])
        if len(output) >= 2:
            score = _first_finite_scalar(output[1])
    else:
        latency = _first_finite_scalar(output)
    return latency, score


def get_trial_contrasts(sl):
    """Return per-trial contrasts (abs max of left/right), NaNs -> 0."""
    contrast_left = np.abs(sl.trials.contrastLeft)
    contrast_right = np.abs(sl.trials.contrastRight)
    trial_contrasts = np.nanmax(np.vstack([contrast_left, contrast_right]), axis=0)
    return np.where(np.isnan(trial_contrasts), 0, trial_contrasts)


def build_event_dicts(sl, event_names, min_trials, return_trial_idx=False):
    """Build per-event arrays of valid event times, aligned contrasts, and trial indices."""
    events_by_name = {}
    contrasts_by_name = {}
    trial_idx_by_name = {}
    trial_contrasts = get_trial_contrasts(sl)
    for event_name in event_names:
        if event_name not in sl.trials.keys():
            print(f"Warning: Event '{event_name}' not found in trials.")
            events_by_name[event_name] = np.array([])
            contrasts_by_name[event_name] = np.array([])
            trial_idx_by_name[event_name] = np.array([], dtype=int)
            continue
        events_all = np.asarray(sl.trials[event_name])
        valid_mask = ~np.isnan(events_all)
        events = events_all[valid_mask]
        contrasts = trial_contrasts[valid_mask]
        trial_idx = np.nonzero(valid_mask)[0]
        if len(events) < min_trials:
            print(
                f"Warning: {event_name} has only {len(events)} trials "
                f"(min {min_trials})."
            )
        events_by_name[event_name] = events
        contrasts_by_name[event_name] = contrasts
        trial_idx_by_name[event_name] = trial_idx
    if return_trial_idx:
        return events_by_name, contrasts_by_name, trial_idx_by_name
    return events_by_name, contrasts_by_name


def get_event_time(sl, event_name, trial_idx):
    """Return a single-trial event time (NaN if missing)."""
    if event_name not in sl.trials.keys():
        return np.nan
    events = np.asarray(sl.trials[event_name])
    if trial_idx < 0 or trial_idx >= len(events):
        return np.nan
    return events[trial_idx]


def event_label(event_name):
    """Human-readable labels for event names."""
    label_map = {
        "stimOn_times": "Stim On",
        "firstMovement_times": "First Move",
        "response_times": "Response",
        "feedback_times": "Feedback",
        "feedback_correct_times": "Feedback (Correct)",
        "feedback_incorrect_times": "Feedback (Incorrect)",
        "passive_tone_times": "Passive Tone",
        "passive_noise_times": "Passive Noise",
        "passive_valve_times": "Passive Valve",
        "passive_all_times": "Passive (All)",
        "wh_brief_times": "Whisking Brief",
        "wh_long_times": "Whisking Long",
        "wh_all_times": "Whisking All",
        "wh_brief_times_spont": "Whisking Brief (Spont)",
        "wh_brief_times_task": "Whisking Brief (Task)",
        "wh_brief_times_iti": "Whisking Brief (ITI)",
        "wh_long_times_spont": "Whisking Long (Spont)",
        "wh_long_times_task": "Whisking Long (Task)",
        "wh_long_times_iti": "Whisking Long (ITI)",
        "wh_all_times_spont": "Whisking All (Spont)",
        "wh_all_times_task": "Whisking All (Task)",
        "wh_all_times_iti": "Whisking All (ITI)",
    }
    return label_map.get(event_name, event_name)


def delay_column_name(event_name):
    return f"delay_{event_name}"


def responsive_column_name(event_name):
    return f"responsive_{event_name}"


def response_sign_column_name(event_name):
    return f"response_sign_{event_name}"


def delay_split_column_name(event_name, split_label):
    return f"{delay_column_name(event_name)}_{split_label}"


def responsive_split_column_name(event_name, split_label):
    return f"{responsive_column_name(event_name)}_{split_label}"


def response_sign_split_column_name(event_name, split_label):
    return f"{response_sign_column_name(event_name)}_{split_label}"


def reliability_column_names(event_name):
    return f"delay_h1_{event_name}", f"delay_h2_{event_name}"


def get_event_delay_window(config, event_name):
    """Return (start, end) window for delay/response detection for this event."""
    windows = config.get("DELAY_WINDOWS", {})
    if isinstance(windows, dict) and event_name in windows:
        start, end = windows[event_name]
        return float(start), float(end)
    return float(config.get("RESPONSIVE_WINDOW_START", 0.0)), float(
        config.get("RESPONSIVE_WINDOW_END", 0.0)
    )


def get_event_delay_method(config, event_name):
    """Return delay method for an event (falls back to global DELAY_METHOD)."""
    methods = config.get("DELAY_METHODS_BY_EVENT", {})
    if isinstance(methods, dict):
        method = methods.get(event_name, None)
        if method is not None:
            return str(method)
    return str(config.get("DELAY_METHOD", "com"))


def compute_psth_for_clusters(
    spikes,
    cluster_ids,
    event_times,
    window_start,
    window_end,
    bin_size,
    smooth_sigma,
    show_progress=False,
    desc="PSTH",
):
    """Compute raw/smoothed PSTHs for multiple clusters and event times."""
    if event_times is None or len(event_times) == 0 or len(cluster_ids) == 0:
        return {}, None

    bins = np.arange(window_start, window_end + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    s_min = event_times.min() + window_start
    s_max = event_times.max() + window_end
    mask = (spikes.times >= s_min) & (spikes.times <= s_max)
    times_window = spikes.times[mask]
    clusters_window = spikes.clusters[mask]

    psth_by_cluster = {}
    iterator = tqdm(cluster_ids, desc=desc, unit="cluster") if show_progress else cluster_ids
    for cid in iterator:
        neuron_spikes = times_window[clusters_window == cid]
        if len(neuron_spikes) == 0:
            continue

        relative_spikes = []
        for t_ev in event_times:
            t0 = t_ev + window_start
            t1 = t_ev + window_end
            in_trial = neuron_spikes[(neuron_spikes >= t0) & (neuron_spikes <= t1)]
            if len(in_trial) > 0:
                relative_spikes.append(in_trial - t_ev)

        if len(relative_spikes) == 0:
            fr_raw = np.zeros(len(bin_centers))
        else:
            all_rel_spikes = np.concatenate(relative_spikes)
            counts, _ = np.histogram(all_rel_spikes, bins=bins)
            fr_raw = counts / len(event_times) / bin_size

        if smooth_sigma and smooth_sigma > 0:
            fr_smooth = gaussian_filter1d(fr_raw, sigma=smooth_sigma)
        else:
            fr_smooth = fr_raw.copy()

        psth_by_cluster[cid] = {"fr_raw": fr_raw, "fr_smooth": fr_smooth}

    return psth_by_cluster, bin_centers


def calculate_delay(
    fr_raw,
    fr_smooth,
    bin_centers,
    config,
    method=None,
    neuron_spikes=None,
    event_times=None,
    trial_contrasts=None,
    return_sign=False,
):
    """
    Compute delay within a responsive window and responsiveness.

    Supported methods:
    - "com": PSTH center of mass within the responsive window.
    - "com_signed": signed center of mass; supports both excitation and inhibition.
    - "psth_peak": peak time of the PSTH within the responsive window.
    - "psth_peak_signed": signed peak/trough time in the responsive window.
    - "tfs": time to first spike after event onset (100% contrast trials only).
    - "latenzy": non-parametric latency estimate from spike/event times.

    Optional config knobs for PSTH-based methods:
    - RESPONSIVE_USE_ZSCORE (bool): if True, classify responsiveness/sign on a
      baseline z-scored PSTH trace.
    - RESPONSIVE_ZSCORE_SOURCE ("smooth" or "raw"): PSTH trace used for
      z-scoring when RESPONSIVE_USE_ZSCORE is True.
    - RESPONSIVE_Z_THR (float): z-threshold magnitude (default: 2.0).
    - COM_USE_THRESHOLD (bool): COM methods only. If True (default), COM is
      computed on threshold-crossing bins; if False, COM uses all bins in the
      responsive window.
    """
    def _result(delay_val, is_responsive, sign_val="none"):
        if return_sign:
            return delay_val, bool(is_responsive), str(sign_val)
        return delay_val, bool(is_responsive)

    # Default to the configured delay method if none is provided explicitly.
    method = str(method or config.get("DELAY_METHOD", "com")).strip().lower()

    if method == "latenzy":
        # latenzy operates directly on spike/event times.
        if neuron_spikes is None or event_times is None:
            return _result(np.nan, False, "none")

        spikes_arr = np.asarray(neuron_spikes, dtype=float).reshape(-1)
        events_arr = np.asarray(event_times, dtype=float).reshape(-1)
        spikes_arr = spikes_arr[np.isfinite(spikes_arr)]
        events_arr = events_arr[np.isfinite(events_arr)]
        if spikes_arr.size == 0 or events_arr.size == 0:
            return _result(np.nan, False, "none")
        spikes_arr = np.sort(spikes_arr)
        events_arr = np.sort(events_arr)

        latenzy_fn = _get_latenzy_callable()
        if latenzy_fn is None:
            _warn_latenzy_unavailable_once()
            return _result(np.nan, False, "none")

        # LATENZY_USE_DUR is optional.
        # If None, call latenzy with (spikes, events) only.
        use_dur = config.get("LATENZY_USE_DUR", None)
        if use_dur is not None:
            try:
                use_dur_arr = np.asarray(use_dur, dtype=float).reshape(-1)
                if use_dur_arr.size == 1:
                    use_dur = float(use_dur_arr[0])
                elif use_dur_arr.size >= 2:
                    use_dur = use_dur_arr[:2]
                    # latenzy expects start <= 0 when end > 0.
                    if (
                        np.isfinite(use_dur[0])
                        and np.isfinite(use_dur[1])
                        and (use_dur[1] > 0)
                        and (use_dur[0] > 0)
                    ):
                        use_dur = np.asarray([0.0, float(use_dur[1])], dtype=float)
                else:
                    use_dur = None
            except Exception:
                use_dur = None

        try:
            if use_dur is None:
                latenzy_out = latenzy_fn(spikes_arr, events_arr)
            else:
                latenzy_out = latenzy_fn(spikes_arr, events_arr, use_dur)
        except TypeError:
            # Some versions only accept (spikes, events).
            try:
                latenzy_out = latenzy_fn(spikes_arr, events_arr)
            except Exception as exc:
                _warn_latenzy_runtime_once(exc)
                return _result(np.nan, False, "none")
        except Exception as exc:
            _warn_latenzy_runtime_once(exc)
            return _result(np.nan, False, "none")

        delay, score = _parse_latenzy_output(latenzy_out)
        if not np.isfinite(delay):
            return _result(np.nan, False, "none")

        min_score = config.get("LATENZY_MIN_SCORE", None)
        if min_score is not None:
            try:
                min_score = float(min_score)
            except (TypeError, ValueError):
                min_score = None
        if min_score is not None:
            if not np.isfinite(score):
                return _result(np.nan, False, "none")
            if score < min_score:
                return _result(np.nan, False, "none")
        return _result(float(delay), True, "exc")

    if method == "tfs":
        # TFS relies on single-trial spike times, not the PSTH.
        if neuron_spikes is None or event_times is None or trial_contrasts is None:
            return _result(np.nan, False, "none")
        if len(neuron_spikes) == 0 or len(event_times) == 0:
            return _result(np.nan, False, "none")

        # Identify 100% contrast trials. Accept 1.0 or 100.0 by default.
        full_contrast_values = config.get("FULL_CONTRAST_VALUES", (1.0, 100.0))
        full_mask = np.zeros(len(trial_contrasts), dtype=bool)
        for val in full_contrast_values:
            full_mask |= np.isclose(trial_contrasts, val)
        if not np.any(full_mask):
            return _result(np.nan, False, "none")

        # For each full-contrast trial, take the first spike in the responsive window.
        first_spike_offsets = []
        window_start = config["RESPONSIVE_WINDOW_START"]
        window_end = config["RESPONSIVE_WINDOW_END"]
        for t_ev in event_times[full_mask]:
            t0 = t_ev + window_start
            t1 = t_ev + window_end
            idx0 = np.searchsorted(neuron_spikes, t0, side="left")
            idx1 = np.searchsorted(neuron_spikes, t1, side="right")
            if idx0 < idx1:
                first_spike_offsets.append(neuron_spikes[idx0] - t_ev)

        if len(first_spike_offsets) == 0:
            return _result(np.nan, False, "none")
        return _result(float(np.mean(first_spike_offsets)), True, "exc")

    # From here on we rely on the PSTH representation.
    if fr_raw is None or bin_centers is None:
        return _result(np.nan, False, "none")

    fr_raw = np.asarray(fr_raw, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    if fr_raw.size != bin_centers.size or fr_raw.size == 0:
        return _result(np.nan, False, "none")
    if fr_smooth is not None:
        fr_smooth = np.asarray(fr_smooth, dtype=float)
        if fr_smooth.size != fr_raw.size:
            fr_smooth = None

    resp_signal = fr_smooth if fr_smooth is not None else fr_raw
    use_zscore = bool(config.get("RESPONSIVE_USE_ZSCORE", False))
    zscore_source = str(config.get("RESPONSIVE_ZSCORE_SOURCE", "smooth")).strip().lower()
    if use_zscore and zscore_source == "raw":
        z_source = fr_raw
    else:
        z_source = resp_signal

    # Compute baseline statistics for thresholding.
    idx_baseline = (bin_centers >= -config["BASELINE_PRE"]) & (bin_centers < 0)
    baseline_trace = z_source if use_zscore else fr_raw
    baseline_fr = np.asarray(baseline_trace[idx_baseline], dtype=float)
    baseline_fr = baseline_fr[np.isfinite(baseline_fr)]
    if len(baseline_fr) == 0:
        return _result(np.nan, False, "none")

    baseline_mean = float(np.mean(baseline_fr))
    baseline_std = float(np.std(baseline_fr))
    z_thr = config.get("RESPONSIVE_Z_THR", 2.0)
    try:
        z_thr = float(z_thr)
    except (TypeError, ValueError):
        z_thr = 2.0
    if not np.isfinite(z_thr) or z_thr <= 0:
        z_thr = 2.0

    if use_zscore:
        if (not np.isfinite(baseline_std)) or baseline_std <= 0:
            return _result(np.nan, False, "none")
        eval_signal = (z_source - baseline_mean) / baseline_std
        threshold_hi = z_thr
        threshold_lo = -z_thr
    else:
        eval_signal = fr_raw
        threshold_hi = baseline_mean + z_thr * baseline_std
        threshold_lo = baseline_mean - z_thr * baseline_std

    idx_responsive = (bin_centers >= config["RESPONSIVE_WINDOW_START"]) & (
        bin_centers <= config["RESPONSIVE_WINDOW_END"]
    )
    com_use_threshold = bool(config.get("COM_USE_THRESHOLD", True))

    if method in {"com_signed", "signed_com"}:
        exc_mask_thr = idx_responsive & (eval_signal > threshold_hi)
        inh_mask_thr = idx_responsive & (eval_signal < threshold_lo)
        if not np.any(exc_mask_thr) and not np.any(inh_mask_thr):
            return _result(np.nan, False, "none")

        if use_zscore:
            exc_weights_thr = np.clip(eval_signal[exc_mask_thr] - threshold_hi, 0.0, None)
            inh_weights_thr = np.clip(threshold_lo - eval_signal[inh_mask_thr], 0.0, None)
        else:
            exc_weights_thr = np.clip(resp_signal[exc_mask_thr] - baseline_mean, 0.0, None)
            inh_weights_thr = np.clip(baseline_mean - resp_signal[inh_mask_thr], 0.0, None)
        exc_strength = float(np.nansum(exc_weights_thr)) if exc_weights_thr.size else 0.0
        inh_strength = float(np.nansum(inh_weights_thr)) if inh_weights_thr.size else 0.0
        if exc_strength <= 0 and inh_strength <= 0:
            return _result(np.nan, False, "none")

        if inh_strength > exc_strength:
            chosen_sign = "inh"
        else:
            chosen_sign = "exc"

        if com_use_threshold:
            if chosen_sign == "inh":
                chosen_time = np.asarray(bin_centers[inh_mask_thr], dtype=float)
                chosen_weights = np.asarray(inh_weights_thr, dtype=float)
            else:
                chosen_time = np.asarray(bin_centers[exc_mask_thr], dtype=float)
                chosen_weights = np.asarray(exc_weights_thr, dtype=float)
        else:
            com_mask = np.asarray(idx_responsive, dtype=bool)
            chosen_time = np.asarray(bin_centers[com_mask], dtype=float)
            if use_zscore:
                metric = np.asarray(eval_signal[com_mask], dtype=float)
                if chosen_sign == "inh":
                    chosen_weights = np.clip(-metric, 0.0, None)
                else:
                    chosen_weights = np.clip(metric, 0.0, None)
            else:
                metric = np.asarray(resp_signal[com_mask], dtype=float)
                if chosen_sign == "inh":
                    chosen_weights = np.clip(baseline_mean - metric, 0.0, None)
                else:
                    chosen_weights = np.clip(metric - baseline_mean, 0.0, None)

        finite = np.isfinite(chosen_time) & np.isfinite(chosen_weights)
        if not np.any(finite):
            return _result(np.nan, False, "none")
        chosen_time = chosen_time[finite]
        chosen_weights = chosen_weights[finite]

        weight_sum = float(np.nansum(chosen_weights))
        if weight_sum <= 0:
            return _result(np.nan, False, "none")
        delay = float(np.nansum(chosen_weights * chosen_time) / weight_sum)
        return _result(delay, True, chosen_sign)

    if method in {"psth_peak_signed", "signed_psth_peak", "psth_signed_peak"}:
        exc_mask = idx_responsive & (eval_signal > threshold_hi)
        inh_mask = idx_responsive & (eval_signal < threshold_lo)
        if not np.any(exc_mask) and not np.any(inh_mask):
            return _result(np.nan, False, "none")

        exc_strength = 0.0
        inh_strength = 0.0
        if np.any(exc_mask):
            exc_vals = np.asarray(eval_signal[exc_mask], dtype=float)
            exc_peak = float(np.nanmax(exc_vals))
            if np.isfinite(exc_peak):
                exc_strength = max(exc_peak - threshold_hi, 0.0)
        if np.any(inh_mask):
            inh_vals = np.asarray(eval_signal[inh_mask], dtype=float)
            inh_trough = float(np.nanmin(inh_vals))
            if np.isfinite(inh_trough):
                inh_strength = max(threshold_lo - inh_trough, 0.0)
        if exc_strength <= 0 and inh_strength <= 0:
            return _result(np.nan, False, "none")

        if inh_strength > exc_strength:
            chosen_time = np.asarray(bin_centers[inh_mask], dtype=float)
            chosen_signal = np.asarray(resp_signal[inh_mask], dtype=float)
            finite = np.isfinite(chosen_time) & np.isfinite(chosen_signal)
            if not np.any(finite):
                return _result(np.nan, False, "none")
            chosen_time = chosen_time[finite]
            chosen_signal = chosen_signal[finite]
            trough_idx = int(np.argmin(chosen_signal))
            return _result(float(chosen_time[trough_idx]), True, "inh")

        chosen_time = np.asarray(bin_centers[exc_mask], dtype=float)
        chosen_signal = np.asarray(resp_signal[exc_mask], dtype=float)
        finite = np.isfinite(chosen_time) & np.isfinite(chosen_signal)
        if not np.any(finite):
            return _result(np.nan, False, "none")
        chosen_time = chosen_time[finite]
        chosen_signal = chosen_signal[finite]
        peak_idx = int(np.argmax(chosen_signal))
        return _result(float(chosen_time[peak_idx]), True, "exc")

    responsive_mask = idx_responsive & (eval_signal > threshold_hi)
    if not np.any(responsive_mask):
        return _result(np.nan, False, "none")

    if method == "psth_peak":
        # Peak time is the bin center at maximum firing rate within the window.
        resp_fr = fr_smooth[responsive_mask] if fr_smooth is not None else fr_raw[responsive_mask]
        resp_time = bin_centers[responsive_mask]
        peak_idx = int(np.argmax(resp_fr))
        return _result(float(resp_time[peak_idx]), True, "exc")

    com_mask = responsive_mask if com_use_threshold else idx_responsive
    resp_fr = fr_smooth[com_mask] if fr_smooth is not None else fr_raw[com_mask]
    resp_time = bin_centers[com_mask]

    # Default: center of mass (previous behavior).
    sum_fr = np.sum(resp_fr)
    if sum_fr <= 0:
        return _result(np.nan, False, "none")
    delay = np.sum(resp_fr * resp_time) / sum_fr
    return _result(delay, True, "exc")


def compute_trial_end_times(trials, event_names, post_event_s=1.0):
    """Return per-trial end times based on the max available event + post_event_s."""
    if trials is None:
        return np.array([])
    n_trials = len(trials)
    end_times = np.full(n_trials, np.nan, dtype=float)
    arrays = {}
    for event_name in event_names:
        if hasattr(trials, "keys") and event_name in trials.keys():
            arrays[event_name] = np.asarray(trials[event_name])
    for idx in range(n_trials):
        ev_times = [arr[idx] for arr in arrays.values() if np.isfinite(arr[idx])]
        if len(ev_times) == 0:
            continue
        end_times[idx] = np.nanmax(ev_times) + float(post_event_s)
    return end_times


def build_task_window_table(trials, event_names, post_event_s=1.0):
    """Build a DataFrame of task windows with trial metadata."""
    if trials is None:
        return pd.DataFrame(columns=["trial_idx", "start", "end", "correct", "odd"])
    stim_on = np.asarray(trials["stimOn_times"])
    n_trials = len(stim_on)
    end_times = compute_trial_end_times(trials, event_names, post_event_s=post_event_s)
    correct = None
    if hasattr(trials, "keys") and "feedbackType" in trials.keys():
        correct = np.asarray(trials["feedbackType"]) == 1
    rows = []
    for idx in range(n_trials):
        t_start = stim_on[idx]
        t_end = end_times[idx]
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
            continue
        rows.append(
            {
                "trial_idx": idx,
                "start": float(t_start),
                "end": float(t_end),
                "correct": bool(correct[idx]) if correct is not None else False,
                "odd": bool(idx % 2 == 1),
            }
        )
    return pd.DataFrame(rows)


def build_iti_windows(trial_end_times, stim_on_times, skip_first_last=True):
    """Return ITI windows (trial_end -> next stim_on), labeled by preceding trial index."""
    if trial_end_times is None or stim_on_times is None:
        return pd.DataFrame(columns=["trial_idx", "start", "end", "odd"])
    n_trials = len(stim_on_times)
    rows = []
    for idx in range(n_trials - 1):
        if skip_first_last and (idx == 0 or idx == n_trials - 2):
            # Skip ITIs adjacent to the first and last trials.
            continue
        t_end = trial_end_times[idx]
        t_next = stim_on_times[idx + 1]
        if not np.isfinite(t_end) or not np.isfinite(t_next) or t_next <= t_end:
            continue
        rows.append(
            {
                "trial_idx": idx,
                "start": float(t_end),
                "end": float(t_next),
                "odd": bool(idx % 2 == 1),
            }
        )
    return pd.DataFrame(rows)


def _mask_times_by_intervals(times, intervals):
    if times is None or len(times) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    if intervals is None or len(intervals) == 0:
        return mask
    times = np.asarray(times)
    for start, end in intervals:
        idx0 = np.searchsorted(times, start, side="left")
        idx1 = np.searchsorted(times, end, side="right")
        if idx1 > idx0:
            mask[idx0:idx1] = True
    return mask


def slice_spikes_by_intervals(spikes, intervals, exclude_intervals=None):
    """Return spikes dict sliced to the provided intervals (optionally excluding others)."""
    if spikes is None or "times" not in spikes or "clusters" not in spikes:
        return {"times": np.array([]), "clusters": np.array([])}
    times = np.asarray(spikes["times"])
    base_mask = _mask_times_by_intervals(times, intervals)
    if exclude_intervals is not None and len(exclude_intervals) > 0:
        excl_mask = _mask_times_by_intervals(times, exclude_intervals)
        base_mask &= ~excl_mask
    return {key: np.asarray(val)[base_mask] for key, val in spikes.items()}


def _mean_stpr_curve(curve_a, curve_b):
    curve_a = np.asarray(curve_a, dtype=float) if curve_a is not None else np.array([])
    curve_b = np.asarray(curve_b, dtype=float) if curve_b is not None else np.array([])
    if curve_a.size == 0 and curve_b.size == 0:
        return np.array([])
    if curve_a.size == 0:
        return curve_b.copy()
    if curve_b.size == 0:
        return curve_a.copy()
    if curve_a.size != curve_b.size:
        min_len = min(curve_a.size, curve_b.size)
        curve_a = curve_a[:min_len]
        curve_b = curve_b[:min_len]
    return (curve_a + curve_b) / 2.0


def _stpr_metrics_from_curve(curve, lags_ms):
    """Return (delay_ms, strength_at_zero, peak) for a stPR curve."""
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0 or not np.isfinite(curve).any():
        return np.nan, np.nan, np.nan
    zero_idx = int(curve.size // 2)
    strength = float(curve[zero_idx])  # value at lag 0
    peak = float(np.nanmax(curve))
    curve_min = np.nanmin(curve)
    curve_shifted = curve - curve_min
    stpr_sum = np.nansum(curve_shifted)
    if stpr_sum > 0:
        delay_val = float(
            np.nansum(lags_ms[: curve.size] * curve_shifted) / stpr_sum
        )
    else:
        delay_val = np.nan
    return delay_val, strength, peak


def _lowpass_filter(signal, fs_hz, cutoff_hz, order=3):
    if butter is None or filtfilt is None:
        return signal
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal
    nyq = 0.5 * fs_hz
    if cutoff_hz >= nyq:
        return signal
    try:
        b, a = butter(order, cutoff_hz / nyq, btype="low")
        if signal.size <= 3 * max(len(a), len(b)):
            return signal
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _interpolate_short_nan_runs(values, max_gap):
    """Linearly interpolate NaN runs up to ``max_gap`` samples."""
    arr = np.asarray(values, dtype=float).copy()
    if arr.size == 0:
        return arr
    if max_gap is None or int(max_gap) <= 0:
        return arr
    max_gap = int(max_gap)
    nan_mask = ~np.isfinite(arr)
    if not np.any(nan_mask):
        return arr
    n = arr.size
    idx = 0
    while idx < n:
        if not nan_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and nan_mask[idx]:
            idx += 1
        end = idx
        gap_len = end - start
        left = start - 1
        right = end
        if (
            gap_len <= max_gap
            and left >= 0
            and right < n
            and np.isfinite(arr[left])
            and np.isfinite(arr[right])
        ):
            arr[start:end] = np.interp(
                np.arange(start, end, dtype=float),
                [float(left), float(right)],
                [float(arr[left]), float(arr[right])],
            )
    return arr


def _extract_me_view_df(
    motion_energy,
    view,
    value_keys,
    max_interp_gap_frames=3,
    ensure_positive_motion=True,
):
    if motion_energy is None or view not in motion_energy:
        return None
    df = motion_energy.get(view)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "times" in df.columns:
        times = np.asarray(df["times"], dtype=float)
    else:
        times = np.asarray(df.index, dtype=float)
    value_col = None
    for key in value_keys:
        if key in df.columns:
            value_col = key
            break
    if value_col is None:
        numeric_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        numeric_cols = [col for col in numeric_cols if col != "times"]
        if numeric_cols:
            value_col = numeric_cols[0]
    if value_col is None:
        return None
    out = pd.DataFrame({"times": times, "value": np.asarray(df[value_col], dtype=float)})
    out = out[np.isfinite(out["times"].to_numpy(dtype=float))].copy()
    if out.empty:
        return None
    out = out.sort_values("times").reset_index(drop=True)

    vals = out["value"].to_numpy(dtype=float)
    vals = _interpolate_short_nan_runs(vals, max_interp_gap_frames)
    if ensure_positive_motion:
        vals_fin = vals[np.isfinite(vals)]
        if vals_fin.size:
            lo = float(np.nanpercentile(vals_fin, 1.0))
            hi = float(np.nanpercentile(vals_fin, 99.0))
            if abs(lo) > abs(hi):
                vals = -vals
    out["value"] = vals
    out = out[np.isfinite(out["value"].to_numpy(dtype=float))].copy()
    if out.empty:
        return None
    out["view"] = str(view)
    out = out[["times", "view", "value"]].reset_index(drop=True)
    return out


def extract_motion_energy_trace(
    motion_energy,
    views=("leftCamera", "rightCamera"),
    value_keys=("whiskerMotionEnergy", "motionEnergy", "energy"),
    max_interp_gap_frames=3,
    ensure_positive_motion=True,
):
    """
    Extract a long-form motion-energy table from one or more camera views.

    Short tracking gaps are interpolated per camera view up to
    ``max_interp_gap_frames`` samples. Longer gaps are kept missing.

    Returns
    -------
    pd.DataFrame
        Columns: ``times`` (s), ``view``, ``value``.
    """
    if motion_energy is None:
        return pd.DataFrame(columns=["times", "view", "value"])
    rows = []
    for view in views:
        df_view = _extract_me_view_df(
            motion_energy,
            view,
            value_keys,
            max_interp_gap_frames=max_interp_gap_frames,
            ensure_positive_motion=ensure_positive_motion,
        )
        if df_view is not None and not df_view.empty:
            rows.append(df_view)
    if not rows:
        return pd.DataFrame(columns=["times", "view", "value"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["times", "view"]).reset_index(drop=True)
    return out


def bin_normalize_whisk_trace(
    df_trace,
    bin_s=0.3,
    norm_pctl=1.0,
    norm_top_pctl=100.0,
    bin_reduce="mean",
    normalize_after_bin=False,
):
    """
    Normalize whisking motion energy per camera, then bin into fixed windows.

    Two normalization orders are supported:
    - ``normalize_after_bin=False`` (default): normalize per camera on raw
      samples first, then bin normalized samples.
    - ``normalize_after_bin=True``: bin raw samples per camera first, then
      normalize binned values per camera.

    In both cases normalization is per camera between percentile
    ``norm_pctl`` and percentile ``norm_top_pctl`` (or camera max when
    ``norm_top_pctl=100``), and final whisking is the mean across available
    cameras in each bin.
    """
    columns = [
        "bin_idx",
        "bin_start_s",
        "bin_center_s",
        "bin_end_s",
        "wh_norm",
        "n_views",
    ]
    if df_trace is None or len(df_trace) == 0:
        return pd.DataFrame(columns=columns)
    if bin_s is None or float(bin_s) <= 0:
        raise ValueError("bin_s must be > 0.")
    norm_pctl = float(norm_pctl)
    norm_top_pctl = float(norm_top_pctl)
    if norm_pctl < 0 or norm_pctl > 100:
        raise ValueError("norm_pctl must be in [0, 100].")
    if norm_top_pctl < 0 or norm_top_pctl > 100:
        raise ValueError("norm_top_pctl must be in [0, 100].")
    if norm_top_pctl <= norm_pctl:
        raise ValueError("norm_top_pctl must be > norm_pctl.")
    bin_reduce = str(bin_reduce).lower()
    if bin_reduce not in {"mean", "median", "max"}:
        raise ValueError("bin_reduce must be one of: mean, median, max.")

    df = df_trace.copy()
    if not {"times", "view", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=columns)
    df = df.dropna(subset=["times", "value"]).copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    df["times"] = df["times"].astype(float)
    df["value"] = df["value"].astype(float)
    df["view"] = df["view"].astype(str)

    def _normalize_values(vals_fin):
        lo = float(np.nanpercentile(vals_fin, norm_pctl))
        hi = float(np.nanpercentile(vals_fin, norm_top_pctl))
        if not np.isfinite(lo):
            lo = float(np.nanmin(vals_fin))
        if not np.isfinite(hi):
            hi = float(np.nanmax(vals_fin))
        if hi <= lo:
            hi = float(np.nanmax(vals_fin))
        if hi <= lo:
            norm = np.zeros_like(vals_fin, dtype=float)
        else:
            norm = (vals_fin - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)
        return norm

    t0 = float(df["times"].min())

    if normalize_after_bin:
        by_view_rows = []
        for view, grp_view in df.groupby("view", sort=False):
            tmp = grp_view[["times", "value"]].copy()
            vals = tmp["value"].to_numpy(dtype=float)
            times = tmp["times"].to_numpy(dtype=float)
            finite = np.isfinite(times) & np.isfinite(vals)
            if not np.any(finite):
                continue
            tmp = tmp.loc[finite].copy()
            tmp["bin_idx"] = np.floor((tmp["times"] - t0) / float(bin_s)).astype(int)
            grp_raw = tmp.groupby("bin_idx", as_index=False)["value"]
            if bin_reduce == "max":
                by_bin_raw = grp_raw.max()
            elif bin_reduce == "median":
                by_bin_raw = grp_raw.median()
            else:
                by_bin_raw = grp_raw.mean()
            vals_bin = by_bin_raw["value"].to_numpy(dtype=float)
            if vals_bin.size == 0:
                continue
            by_bin_raw["view"] = view
            by_bin_raw["value_norm"] = _normalize_values(vals_bin)
            by_view_rows.append(by_bin_raw[["bin_idx", "view", "value_norm"]])
        if not by_view_rows:
            return pd.DataFrame(columns=columns)
        by_view_bin = pd.concat(by_view_rows, ignore_index=True)
    else:
        norm_rows = []
        for view, grp_view in df.groupby("view", sort=False):
            vals = grp_view["value"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            vals_fin = vals[finite]
            norm = _normalize_values(vals_fin)
            tmp = grp_view.loc[finite, ["times"]].copy()
            tmp["view"] = view
            tmp["value_norm"] = norm
            norm_rows.append(tmp)
        if not norm_rows:
            return pd.DataFrame(columns=columns)

        norm_df = pd.concat(norm_rows, ignore_index=True)
        norm_df = norm_df.sort_values(["times", "view"]).reset_index(drop=True)
        norm_df["bin_idx"] = np.floor((norm_df["times"] - t0) / float(bin_s)).astype(int)
        grp_norm = norm_df.groupby(["bin_idx", "view"], sort=True)["value_norm"]
        if bin_reduce == "max":
            by_view_bin = grp_norm.max().reset_index()
        elif bin_reduce == "median":
            by_view_bin = grp_norm.median().reset_index()
        else:
            by_view_bin = grp_norm.mean().reset_index()

    wide = by_view_bin.pivot(index="bin_idx", columns="view", values="value_norm")
    wide = wide.sort_index()
    out = pd.DataFrame(index=wide.index.copy())
    out["wh_norm"] = wide.mean(axis=1, skipna=True).astype(float)
    out["n_views"] = wide.notna().sum(axis=1).astype(int)
    out["bin_idx"] = out.index.astype(int)
    out["bin_start_s"] = t0 + out["bin_idx"] * float(bin_s)
    out["bin_center_s"] = out["bin_start_s"] + 0.5 * float(bin_s)
    out["bin_end_s"] = out["bin_start_s"] + float(bin_s)
    out = out.reset_index(drop=True)
    out = out[columns]
    return out


def detect_wh_bouts(
    times,
    wh,
    start_thr=0.15,
    end_thr=0.05,
    merge_gap_s=0.0,
    end_quiet_window_s=4.0,
    end_quiet_thr=None,
    brief_range_s=(1.3, 3.5),
    long_min_s=3.5,
):
    """
    Detect whisking bouts from a continuous whisking trace.

    Bouts start when signal exceeds ``start_thr``. A bout can end when signal
    drops below ``end_thr`` and the mean whisking in the next
    ``end_quiet_window_s`` seconds stays below ``end_quiet_thr`` (defaults to
    ``end_thr``). Optional post-hoc merge is available via ``merge_gap_s``.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    wh = np.asarray(wh, dtype=float).reshape(-1)
    mask = np.isfinite(times) & np.isfinite(wh)
    times = times[mask]
    wh = wh[mask]
    if times.size == 0:
        empty = np.empty((0, 2), dtype=float)
        return {
            "all_bouts": empty,
            "brief_bouts": empty,
            "long_bouts": empty,
            "all_onsets": np.array([], dtype=float),
            "brief_onsets": np.array([], dtype=float),
            "long_onsets": np.array([], dtype=float),
            "all_durations": np.array([], dtype=float),
            "brief_durations": np.array([], dtype=float),
            "long_durations": np.array([], dtype=float),
        }

    order = np.argsort(times)
    times = times[order]
    wh = wh[order]
    start_thr = float(start_thr)
    end_thr = float(end_thr)
    merge_gap_s = float(merge_gap_s)
    end_quiet_window_s = float(end_quiet_window_s)
    if end_quiet_thr is None:
        end_quiet_thr = end_thr
    end_quiet_thr = float(end_quiet_thr)

    bouts = []
    start_idx = None
    for idx in range(times.size):
        val = wh[idx]
        if start_idx is None:
            if val >= start_thr:
                start_idx = idx
        else:
            if val <= end_thr:
                if end_quiet_window_s > 0:
                    t_now = float(times[idx])
                    future_mask = (times > t_now) & (
                        times <= (t_now + end_quiet_window_s)
                    )
                    if np.any(future_mask):
                        future_mean = float(np.nanmean(wh[future_mask]))
                    else:
                        future_mean = float(val)
                    quiet_ok = np.isfinite(future_mean) and (future_mean <= end_quiet_thr)
                else:
                    quiet_ok = True
                if quiet_ok:
                    bouts.append([float(times[start_idx]), float(times[idx])])
                    start_idx = None
    if start_idx is not None:
        bouts.append([float(times[start_idx]), float(times[-1])])

    if not bouts:
        return {
            "all_bouts": np.empty((0, 2), dtype=float),
            "brief_bouts": np.empty((0, 2), dtype=float),
            "long_bouts": np.empty((0, 2), dtype=float),
            "all_onsets": np.array([], dtype=float),
            "brief_onsets": np.array([], dtype=float),
            "long_onsets": np.array([], dtype=float),
            "all_durations": np.array([], dtype=float),
            "brief_durations": np.array([], dtype=float),
            "long_durations": np.array([], dtype=float),
        }

    bouts = np.asarray(bouts, dtype=float)
    bouts = bouts[np.argsort(bouts[:, 0])]
    if merge_gap_s > 0 and bouts.shape[0] > 1:
        merged = [bouts[0].tolist()]
        for start, end in bouts[1:]:
            prev_start, prev_end = merged[-1]
            if float(start) - float(prev_end) < merge_gap_s:
                merged[-1][1] = float(max(prev_end, end))
            else:
                merged.append([float(start), float(end)])
        merged = np.asarray(merged, dtype=float)
    else:
        merged = bouts.copy()
    durations = merged[:, 1] - merged[:, 0]

    brief_lo, brief_hi = brief_range_s
    brief_mask = (durations >= float(brief_lo)) & (durations <= float(brief_hi))
    long_mask = durations > float(long_min_s)
    brief_bouts = merged[brief_mask]
    long_bouts = merged[long_mask]
    brief_durations = durations[brief_mask]
    long_durations = durations[long_mask]

    return {
        "all_bouts": merged,
        "brief_bouts": brief_bouts,
        "long_bouts": long_bouts,
        "all_onsets": merged[:, 0].copy(),
        "brief_onsets": brief_bouts[:, 0].copy() if len(brief_bouts) else np.array([], dtype=float),
        "long_onsets": long_bouts[:, 0].copy() if len(long_bouts) else np.array([], dtype=float),
        "all_durations": durations,
        "brief_durations": brief_durations,
        "long_durations": long_durations,
    }


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


def _mask_times_in_intervals(times, intervals):
    times = np.asarray(times, dtype=float).reshape(-1)
    mask = np.zeros(times.shape[0], dtype=bool)
    if times.size == 0:
        return mask
    intervals_arr = _coerce_interval_array(intervals)
    if intervals_arr.size == 0:
        return mask
    for start, end in intervals_arr:
        mask |= (times >= float(start)) & (times <= float(end))
    return mask


def split_wh_events_by_period(wh_event_times, spont_intervals, task_windows, iti_windows):
    """
    Split whisk event times by period using event onset timestamps.

    Returns a dictionary containing both global and period-specific event arrays:
    ``<name>``, ``<name>_spont``, ``<name>_task``, ``<name>_iti``.
    """
    wh_event_times = wh_event_times or {}
    split_events = {}
    spont_arr = _coerce_interval_array(spont_intervals)
    task_arr = _coerce_interval_array(task_windows)
    iti_arr = _coerce_interval_array(iti_windows)

    for event_name, times in wh_event_times.items():
        arr = np.asarray(times, dtype=float).reshape(-1)
        arr = np.sort(arr[np.isfinite(arr)])
        split_events[event_name] = arr
        if arr.size == 0:
            split_events[f"{event_name}_spont"] = np.array([], dtype=float)
            split_events[f"{event_name}_task"] = np.array([], dtype=float)
            split_events[f"{event_name}_iti"] = np.array([], dtype=float)
            continue
        split_events[f"{event_name}_spont"] = arr[_mask_times_in_intervals(arr, spont_arr)]
        split_events[f"{event_name}_task"] = arr[_mask_times_in_intervals(arr, task_arr)]
        split_events[f"{event_name}_iti"] = arr[_mask_times_in_intervals(arr, iti_arr)]
    return split_events


def _cluster_firing_rate_lookup(clusters):
    if clusters is None:
        return {}
    rates = None
    if hasattr(clusters, "firing_rate"):
        rates = np.asarray(clusters.firing_rate, dtype=float)
    elif isinstance(clusters, dict) and "firing_rate" in clusters:
        rates = np.asarray(clusters.get("firing_rate"), dtype=float)
    elif hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "firing_rate" in clusters.metrics.columns:
            rates = np.asarray(clusters.metrics["firing_rate"], dtype=float)
    if rates is None:
        return {}
    cluster_id_all = None
    if hasattr(clusters, "cluster_id"):
        cluster_id_all = np.asarray(clusters.cluster_id)
    elif isinstance(clusters, dict) and "cluster_id" in clusters:
        cluster_id_all = np.asarray(clusters.get("cluster_id"))
    if cluster_id_all is None or len(cluster_id_all) != len(rates):
        return {}
    return dict(zip(cluster_id_all.tolist(), rates.tolist()))


def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _fill_nan_linear(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0 or np.isfinite(arr).all():
        return arr
    idx = np.arange(arr.size, dtype=float)
    finite = np.isfinite(arr)
    if int(finite.sum()) == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    if int(finite.sum()) == 1:
        out = np.full(arr.shape, np.nan, dtype=float)
        out[:] = float(arr[finite][0])
        return out
    return np.interp(idx, idx[finite], arr[finite])


def _zscore_trace(arr, baseline_mask=None):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.isfinite(arr).any():
        return np.full(arr.shape, np.nan, dtype=float)

    ref = arr
    if baseline_mask is not None:
        baseline_mask = np.asarray(baseline_mask, dtype=bool).reshape(-1)
        if baseline_mask.size != arr.size:
            return np.full(arr.shape, np.nan, dtype=float)
        ref = arr[baseline_mask]
        if ref.size == 0:
            return np.full(arr.shape, np.nan, dtype=float)

    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(arr.shape, np.nan, dtype=float)

    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return (arr - mu) / sd


def compute_arousal_groups_from_whisk(
    spikes,
    clusters,
    cluster_acronyms,
    cid_to_idx,
    cluster_ids,
    wh_brief_times,
    config,
    whisk_times=None,
    whisk_values=None,
    spont_intervals=None,
):
    """
    Assign arousal groups from whisking-neuron correlations in spontaneous periods.

    This mirrors the paper protocol:
    - bin whisking and firing traces in fixed windows (default: 300 ms),
    - smooth traces with a Gaussian kernel (default sigma=5 bins),
    - z-score traces (full trace by default, or pre-event baseline bins if
      ``AROUSAL_USE_EVENT_BASELINE_ZSCORE`` is enabled),
    - compute Pearson correlations in full recording and split halves.
    Neurons are ``arousal_plus`` if correlation exceeds threshold in full + both
    halves, ``arousal_minus`` if below negative threshold in full + both halves.
    """
    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    if cluster_ids.size == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "acronym",
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
        )

    spike_times = (
        np.asarray(spikes.times, dtype=float)
        if hasattr(spikes, "times")
        else np.asarray(spikes.get("times", []), dtype=float)
    )
    spike_clusters = (
        np.asarray(spikes.clusters)
        if hasattr(spikes, "clusters")
        else np.asarray(spikes.get("clusters", []))
    )

    wh_brief_times = np.asarray(wh_brief_times, dtype=float).reshape(-1)
    wh_brief_times = np.sort(wh_brief_times[np.isfinite(wh_brief_times)])
    legacy_n_events = int(wh_brief_times.size)
    corr_pos_thr = float(config.get("AROUSAL_POS_THR", 0.05))
    corr_neg_thr = float(config.get("AROUSAL_NEG_THR", -0.05))
    corr_pos_thr = float(max(corr_pos_thr, 0.0))
    corr_neg_thr = float(min(corr_neg_thr, 0.0))
    bin_s = float(config.get("AROUSAL_BIN_S", 0.3))
    smooth_sigma = float(config.get("AROUSAL_SMOOTH_SIGMA", 5.0))
    min_bins = int(config.get("AROUSAL_MIN_CORR_BINS", 10))
    require_split_half = bool(config.get("AROUSAL_REQUIRE_SPLIT_HALF", True))
    use_event_baseline_zscore = bool(
        config.get("AROUSAL_USE_EVENT_BASELINE_ZSCORE", False)
    )
    try:
        baseline_pre = float(
            config.get("AROUSAL_BASELINE_PRE", config.get("BASELINE_PRE", 0.2))
        )
    except Exception:
        baseline_pre = 0.2
    if not np.isfinite(baseline_pre) or baseline_pre <= 0:
        baseline_pre = 0.2
    min_baseline_bins = int(config.get("AROUSAL_MIN_BASELINE_BINS", 3))
    if min_baseline_bins < 1:
        min_baseline_bins = 1
    if bin_s <= 0:
        bin_s = 0.3

    firing_rate_map = _cluster_firing_rate_lookup(clusters)
    duration_s = np.nan
    if spike_times.size > 1:
        duration_s = float(np.nanmax(spike_times) - np.nanmin(spike_times))
    cluster_count_map = pd.Series(spike_clusters).value_counts().to_dict()

    output = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(int),
            "acronym": cluster_acronyms,
        }
    )
    output["arousal_corr"] = np.nan
    output["arousal_corr_h1"] = np.nan
    output["arousal_corr_h2"] = np.nan
    output["arousal_corr_abs"] = np.nan
    output["arousal_mod"] = np.nan
    output["arousal_fr_hz"] = np.nan
    output["arousal_group"] = "neutral"
    output["arousal_n_events"] = legacy_n_events
    output["arousal_n_bins"] = 0

    fr_vals = []
    for cid in output["cluster_id"].to_numpy(dtype=int):
        fr_val = firing_rate_map.get(cid, np.nan)
        if not np.isfinite(fr_val) and np.isfinite(duration_s) and duration_s > 0:
            fr_val = float(cluster_count_map.get(cid, 0)) / duration_s
        fr_vals.append(fr_val)
    output["arousal_fr_hz"] = np.asarray(fr_vals, dtype=float)

    if whisk_times is None or whisk_values is None:
        return output

    whisk_times = np.asarray(whisk_times, dtype=float).reshape(-1)
    whisk_values = np.asarray(whisk_values, dtype=float).reshape(-1)
    n_wh = int(min(whisk_times.size, whisk_values.size))
    if n_wh <= 0:
        return output
    whisk_times = whisk_times[:n_wh]
    whisk_values = whisk_values[:n_wh]
    wh_mask = np.isfinite(whisk_times) & np.isfinite(whisk_values)
    whisk_times = whisk_times[wh_mask]
    whisk_values = whisk_values[wh_mask]
    if whisk_times.size == 0:
        return output

    spont_arr = _coerce_interval_array(spont_intervals)
    if spont_arr.size == 0:
        return output
    spont_arr = spont_arr[np.isfinite(spont_arr).all(axis=1)]
    spont_arr = spont_arr[spont_arr[:, 1] > spont_arr[:, 0]]
    if spont_arr.size == 0:
        return output
    spont_arr = spont_arr[np.argsort(spont_arr[:, 0])]

    t_min = float(np.nanmin(spont_arr[:, 0]))
    t_max = float(np.nanmax(spont_arr[:, 1]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min) < bin_s:
        return output

    bin_edges = np.arange(t_min, t_max + bin_s, bin_s, dtype=float)
    if bin_edges.size < 2:
        return output
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    spont_bin_mask = _mask_times_in_intervals(bin_centers, spont_arr)
    if int(np.sum(spont_bin_mask)) < max(3, min_bins):
        return output

    # Bin whisking signal and keep only spontaneous bins.
    wh_sum, _ = np.histogram(whisk_times, bins=bin_edges, weights=whisk_values)
    wh_cnt, _ = np.histogram(whisk_times, bins=bin_edges)
    whisk_binned = np.full(bin_centers.shape[0], np.nan, dtype=float)
    valid_wh = wh_cnt > 0
    whisk_binned[valid_wh] = wh_sum[valid_wh] / wh_cnt[valid_wh]
    whisk_trace = whisk_binned[spont_bin_mask]
    if int(np.isfinite(whisk_trace).sum()) < max(3, min_bins):
        return output
    whisk_trace = _fill_nan_linear(whisk_trace)
    if smooth_sigma > 0 and whisk_trace.size > 1:
        whisk_trace = gaussian_filter1d(whisk_trace, sigma=smooth_sigma)

    centers_spont = bin_centers[spont_bin_mask]
    baseline_mask_spont = None
    if use_event_baseline_zscore:
        baseline_mask_spont = np.zeros(centers_spont.shape[0], dtype=bool)
        for t_ev in wh_brief_times:
            if not np.isfinite(t_ev):
                continue
            baseline_mask_spont |= (
                (centers_spont >= float(t_ev) - baseline_pre)
                & (centers_spont < float(t_ev))
            )
        if int(np.sum(baseline_mask_spont)) < min_baseline_bins:
            return output

    t_half = float(np.nanmin(centers_spont) + np.nanmax(centers_spont)) / 2.0
    h1_mask = centers_spont <= t_half
    h2_mask = centers_spont > t_half
    min_half_bins = max(3, min_bins // 2)

    in_span = (spike_times >= bin_edges[0]) & (spike_times <= bin_edges[-1])
    spike_times_span = spike_times[in_span]
    spike_clusters_span = spike_clusters[in_span]

    corr_vals = np.full(len(output), np.nan, dtype=float)
    corr_h1_vals = np.full(len(output), np.nan, dtype=float)
    corr_h2_vals = np.full(len(output), np.nan, dtype=float)
    for i, cid in enumerate(output["cluster_id"].to_numpy(dtype=int)):
        neuron_spikes = spike_times_span[spike_clusters_span == cid]
        if neuron_spikes.size == 0:
            continue
        spike_counts, _ = np.histogram(neuron_spikes, bins=bin_edges)
        fr_trace = spike_counts.astype(float) / bin_s
        fr_trace = fr_trace[spont_bin_mask]
        if smooth_sigma > 0 and fr_trace.size > 1:
            fr_trace = gaussian_filter1d(fr_trace, sigma=smooth_sigma)

        fr_z = _zscore_trace(fr_trace, baseline_mask=baseline_mask_spont)
        wh_z = _zscore_trace(whisk_trace, baseline_mask=baseline_mask_spont)
        corr_vals[i] = _safe_corrcoef(fr_z, wh_z)

        if int(np.sum(h1_mask)) >= min_half_bins:
            baseline_h1 = (
                baseline_mask_spont[h1_mask] if baseline_mask_spont is not None else None
            )
            corr_h1_vals[i] = _safe_corrcoef(
                _zscore_trace(fr_trace[h1_mask], baseline_mask=baseline_h1),
                _zscore_trace(whisk_trace[h1_mask], baseline_mask=baseline_h1),
            )
        if int(np.sum(h2_mask)) >= min_half_bins:
            baseline_h2 = (
                baseline_mask_spont[h2_mask] if baseline_mask_spont is not None else None
            )
            corr_h2_vals[i] = _safe_corrcoef(
                _zscore_trace(fr_trace[h2_mask], baseline_mask=baseline_h2),
                _zscore_trace(whisk_trace[h2_mask], baseline_mask=baseline_h2),
            )

    output["arousal_corr"] = corr_vals
    output["arousal_corr_h1"] = corr_h1_vals
    output["arousal_corr_h2"] = corr_h2_vals
    output["arousal_corr_abs"] = np.abs(corr_vals)
    output["arousal_mod"] = corr_vals
    n_spont_bins = int(np.isfinite(whisk_trace).sum())
    output["arousal_n_events"] = n_spont_bins
    output["arousal_n_bins"] = n_spont_bins

    group_vals = []
    for corr_val, corr_h1, corr_h2 in zip(
        output["arousal_corr"].to_numpy(dtype=float),
        output["arousal_corr_h1"].to_numpy(dtype=float),
        output["arousal_corr_h2"].to_numpy(dtype=float),
    ):
        if not np.isfinite(corr_val):
            group_vals.append("neutral")
            continue
        if require_split_half:
            pos_ok = (
                np.isfinite(corr_h1)
                and np.isfinite(corr_h2)
                and corr_val >= corr_pos_thr
                and corr_h1 >= corr_pos_thr
                and corr_h2 >= corr_pos_thr
            )
            neg_ok = (
                np.isfinite(corr_h1)
                and np.isfinite(corr_h2)
                and corr_val <= corr_neg_thr
                and corr_h1 <= corr_neg_thr
                and corr_h2 <= corr_neg_thr
            )
        else:
            pos_ok = corr_val >= corr_pos_thr
            neg_ok = corr_val <= corr_neg_thr
        if pos_ok:
            group_vals.append("arousal_plus")
        elif neg_ok:
            group_vals.append("arousal_minus")
        else:
            group_vals.append("neutral")
    output["arousal_group"] = group_vals
    return output


def add_wh_delay_sorting(df, delay_cols, group_col="arousal_group"):
    """
    Add whisk-delay sorting indices within and across arousal +/- groups.

    For each delay column ``delay_wh_*_times`` this function adds:
    ``wh_*_sort_group`` and ``wh_*_sort_all``.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if group_col not in out.columns:
        out[group_col] = "neutral"
    arousal_mask = out[group_col].isin(["arousal_plus", "arousal_minus"])

    for delay_col in list(delay_cols or []):
        if delay_col not in out.columns:
            continue
        key = str(delay_col)
        if key.startswith("delay_"):
            key = key[len("delay_") :]
        if key.endswith("_times"):
            key = key[: -len("_times")]

        col_group = f"{key}_sort_group"
        col_all = f"{key}_sort_all"
        out[col_group] = np.nan
        out[col_all] = np.nan

        delay_vals = out[delay_col].to_numpy(dtype=float)
        finite_delay = np.isfinite(delay_vals)

        for group_name in ("arousal_plus", "arousal_minus"):
            mask = (out[group_col].to_numpy(dtype=object) == group_name) & finite_delay
            idx = out.index[mask]
            if len(idx) == 0:
                continue
            sorted_idx = out.loc[idx, delay_col].sort_values().index
            out.loc[sorted_idx, col_group] = np.arange(len(sorted_idx), dtype=float)

        mask_all = arousal_mask.to_numpy(dtype=bool) & finite_delay
        idx_all = out.index[mask_all]
        if len(idx_all) > 0:
            sorted_idx_all = out.loc[idx_all, delay_col].sort_values().index
            out.loc[sorted_idx_all, col_all] = np.arange(len(sorted_idx_all), dtype=float)

        out[col_group] = out[col_group].astype("Int64")
        out[col_all] = out[col_all].astype("Int64")

    return out


def calculate_delays(
    spikes,
    clusters,
    cluster_acronyms,
    events_by_name,
    contrasts_by_name,
    config,
    path_data_processed,
    pid,
    cid_to_idx,
    trial_idx_by_name=None,
):
    """Compute event delays for all clusters across multiple events and trial splits."""
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for delay calculation.")
        return pd.DataFrame()

    min_trials = config.get("MIN_TRIALS", 0)
    min_trials_split = config.get(
        "MIN_TRIALS_SPLIT", max(5, int(np.ceil(min_trials / 2)))
    )

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]
    print(f"Found {len(cluster_ids)} clusters.")

    results = []
    selected_cluster_ids = []
    label_min = config.get("CALC_LABEL_MIN", None)
    if label_min is None and config.get("CALC_ONLY_GOOD_UNITS", False):
        label_min = 1.0

    def _label_ok(label_val):
        if label_min is None:
            return True
        if label_val is None:
            return False
        try:
            return float(label_val) >= float(label_min)
        except (TypeError, ValueError):
            return False

    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = io_utils.get_cluster_label(clusters, idx)
        if not _label_ok(label):
            continue
        results.append(
            {
                "cluster_id": cid,
                "acronym": cluster_acronyms[idx],
                "label": label,
            }
        )
        selected_cluster_ids.append(cid)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No clusters met the selection criteria.")
        return df_res

    spike_times_by_cluster = {
        cid: spikes.times[spikes.clusters == cid] for cid in selected_cluster_ids
    }

    for event_name in event_names:
        events = events_by_name.get(event_name, np.array([]))
        contrasts = contrasts_by_name.get(event_name, np.array([]))
        trial_idx = (
            trial_idx_by_name.get(event_name, np.array([], dtype=int))
            if trial_idx_by_name is not None
            else np.arange(len(events))
        )

        delay_col = delay_column_name(event_name)
        delay_odd_col = delay_split_column_name(event_name, "odd")
        delay_even_col = delay_split_column_name(event_name, "even")
        resp_col = responsive_column_name(event_name)
        sign_col = response_sign_column_name(event_name)
        sign_odd_col = response_sign_split_column_name(event_name, "odd")
        sign_even_col = response_sign_split_column_name(event_name, "even")

        if events is None or len(events) < min_trials:
            df_res[delay_col] = np.nan
            df_res[delay_odd_col] = np.nan
            df_res[delay_even_col] = np.nan
            df_res[resp_col] = False
            df_res[sign_col] = "none"
            df_res[sign_odd_col] = "none"
            df_res[sign_even_col] = "none"
            continue

        odd_mask = (trial_idx % 2) == 1
        even_mask = ~odd_mask
        events_odd = events[odd_mask]
        events_even = events[even_mask]
        contrasts_odd = contrasts[odd_mask]
        contrasts_even = contrasts[even_mask]

        if len(events_odd) < min_trials_split or len(events_even) < min_trials_split:
            df_res[delay_col] = np.nan
            df_res[delay_odd_col] = np.nan
            df_res[delay_even_col] = np.nan
            df_res[resp_col] = False
            df_res[sign_col] = "none"
            df_res[sign_odd_col] = "none"
            df_res[sign_even_col] = "none"
            continue

        win_start, win_end = get_event_delay_window(config, event_name)
        event_config = {
            **config,
            "RESPONSIVE_WINDOW_START": win_start,
            "RESPONSIVE_WINDOW_END": win_end,
        }
        event_method = get_event_delay_method(config, event_name)

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH ({event_name})",
        )
        psth_by_cluster_odd, bin_centers_odd = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_odd,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH Odd ({event_name})",
        )
        psth_by_cluster_even, bin_centers_even = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_even,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH Even ({event_name})",
        )

        delays = []
        delays_odd = []
        delays_even = []
        responsive_flags = []
        response_signs = []
        response_signs_odd = []
        response_signs_even = []
        for cid in tqdm(selected_cluster_ids, desc=f"Delays ({event_name})", unit="cluster"):
            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry = psth_by_cluster.get(cid)
            fr_raw = psth_entry["fr_raw"] if psth_entry else None
            fr_smooth = psth_entry["fr_smooth"] if psth_entry else None

            delay, is_responsive, response_sign = calculate_delay(
                fr_raw,
                fr_smooth,
                bin_centers,
                event_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events,
                trial_contrasts=contrasts,
                return_sign=True,
            )

            psth_entry_odd = psth_by_cluster_odd.get(cid)
            fr_raw_odd = psth_entry_odd["fr_raw"] if psth_entry_odd else None
            fr_smooth_odd = psth_entry_odd["fr_smooth"] if psth_entry_odd else None
            delay_odd, resp_odd, response_sign_odd = calculate_delay(
                fr_raw_odd,
                fr_smooth_odd,
                bin_centers_odd,
                event_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events_odd,
                trial_contrasts=contrasts_odd,
                return_sign=True,
            )

            psth_entry_even = psth_by_cluster_even.get(cid)
            fr_raw_even = psth_entry_even["fr_raw"] if psth_entry_even else None
            fr_smooth_even = psth_entry_even["fr_smooth"] if psth_entry_even else None
            delay_even, resp_even, response_sign_even = calculate_delay(
                fr_raw_even,
                fr_smooth_even,
                bin_centers_even,
                event_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events_even,
                trial_contrasts=contrasts_even,
                return_sign=True,
            )

            if not (resp_odd and resp_even) or not (
                np.isfinite(delay_odd) and np.isfinite(delay_even)
            ):
                # If either split is non-responsive, mark both halves as NaN.
                delay_odd = np.nan
                delay_even = np.nan
                delay = np.nan
                is_responsive = False
                response_sign = "none"
                response_sign_odd = "none"
                response_sign_even = "none"

            delays.append(delay)
            delays_odd.append(delay_odd)
            delays_even.append(delay_even)
            responsive_flags.append(is_responsive)
            response_signs.append(response_sign if is_responsive else "none")
            response_signs_odd.append(response_sign_odd if np.isfinite(delay_odd) else "none")
            response_signs_even.append(response_sign_even if np.isfinite(delay_even) else "none")

        df_res[delay_col] = delays
        df_res[delay_odd_col] = delays_odd
        df_res[delay_even_col] = delays_even
        df_res[resp_col] = responsive_flags
        df_res[sign_col] = response_signs
        df_res[sign_odd_col] = response_signs_odd
        df_res[sign_even_col] = response_signs_even

    output_path = path_data_processed / f"{pid}_delay_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Computed delays for {len(df_res)} neurons. Saved to {output_path}.")
    return df_res


def calculate_event_delays(
    spikes,
    clusters,
    cluster_acronyms,
    events_by_name,
    config,
    cid_to_idx,
    contrasts_by_name=None,
    trial_idx_by_name=None,
    include_splits=False,
    output_path=None,
):
    """
    Compute per-event delays without enforcing split-half responsiveness.

    This helper is intended for custom event dictionaries (for example feedback
    correct/incorrect or passive replay events) where odd/even reliability
    gating should not zero-out the overall delay estimate.
    """
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for delay calculation.")
        return pd.DataFrame()

    min_trials = int(config.get("MIN_TRIALS", 0))
    min_trials_split = int(
        config.get("MIN_TRIALS_SPLIT", max(5, int(np.ceil(max(min_trials, 1) / 2))))
    )
    contrasts_by_name = contrasts_by_name or {}

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]
    print(f"Found {len(cluster_ids)} clusters.")

    label_min = config.get("CALC_LABEL_MIN", None)
    if label_min is None and config.get("CALC_ONLY_GOOD_UNITS", False):
        label_min = 1.0

    def _label_ok(label_val):
        if label_min is None:
            return True
        if label_val is None:
            return False
        try:
            return float(label_val) >= float(label_min)
        except (TypeError, ValueError):
            return False

    results = []
    selected_cluster_ids = []
    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = io_utils.get_cluster_label(clusters, idx)
        if not _label_ok(label):
            continue
        results.append(
            {
                "cluster_id": cid,
                "acronym": cluster_acronyms[idx],
                "label": label,
            }
        )
        selected_cluster_ids.append(cid)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No clusters met the selection criteria.")
        return df_res

    spike_times_by_cluster = {
        cid: spikes.times[spikes.clusters == cid] for cid in selected_cluster_ids
    }

    def _prepare_event_arrays(event_name):
        events = events_by_name.get(event_name, np.array([]))
        if events is None:
            events = np.array([])
        events = np.asarray(events, dtype=float).reshape(-1)

        contrasts = contrasts_by_name.get(event_name, None)
        if contrasts is None:
            contrasts = np.ones(len(events), dtype=float)
        else:
            contrasts = np.asarray(contrasts, dtype=float).reshape(-1)
            if contrasts.shape[0] != events.shape[0]:
                if contrasts.size == 1:
                    contrasts = np.full(len(events), float(contrasts.ravel()[0]), dtype=float)
                else:
                    contrasts = np.ones(len(events), dtype=float)

        if trial_idx_by_name is not None:
            trial_idx = trial_idx_by_name.get(event_name, None)
        else:
            trial_idx = None
        if trial_idx is None:
            trial_idx = np.arange(len(events), dtype=int)
        else:
            trial_idx = np.asarray(trial_idx).reshape(-1)
            if trial_idx.shape[0] != events.shape[0]:
                trial_idx = np.arange(len(events), dtype=int)

        finite_mask = np.isfinite(events)
        if not np.all(finite_mask):
            events = events[finite_mask]
            contrasts = contrasts[finite_mask]
            trial_idx = trial_idx[finite_mask]

        return events, contrasts, trial_idx.astype(int)

    delay_columns = []
    for event_name in event_names:
        events, contrasts, trial_idx = _prepare_event_arrays(event_name)

        delay_col = delay_column_name(event_name)
        resp_col = responsive_column_name(event_name)
        sign_col = response_sign_column_name(event_name)
        delay_columns.append(delay_col)
        if include_splits:
            delay_odd_col = delay_split_column_name(event_name, "odd")
            delay_even_col = delay_split_column_name(event_name, "even")
            sign_odd_col = response_sign_split_column_name(event_name, "odd")
            sign_even_col = response_sign_split_column_name(event_name, "even")
            delay_columns.extend([delay_odd_col, delay_even_col])

        if events.size < min_trials:
            df_res[delay_col] = np.nan
            df_res[resp_col] = False
            df_res[sign_col] = "none"
            if include_splits:
                df_res[delay_odd_col] = np.nan
                df_res[delay_even_col] = np.nan
                df_res[sign_odd_col] = "none"
                df_res[sign_even_col] = "none"
            continue

        win_start, win_end = get_event_delay_window(config, event_name)
        event_config = {
            **config,
            "RESPONSIVE_WINDOW_START": win_start,
            "RESPONSIVE_WINDOW_END": win_end,
        }
        event_method = get_event_delay_method(config, event_name)

        psth_by_cluster, bin_centers = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH ({event_name})",
        )

        split_ready = False
        psth_by_cluster_odd = {}
        psth_by_cluster_even = {}
        bin_centers_odd = None
        bin_centers_even = None
        events_odd = np.array([])
        events_even = np.array([])
        contrasts_odd = np.array([])
        contrasts_even = np.array([])
        if include_splits:
            odd_mask = (trial_idx % 2) == 1
            even_mask = ~odd_mask
            events_odd = events[odd_mask]
            events_even = events[even_mask]
            contrasts_odd = contrasts[odd_mask]
            contrasts_even = contrasts[even_mask]
            if len(events_odd) >= min_trials_split and len(events_even) >= min_trials_split:
                split_ready = True
                psth_by_cluster_odd, bin_centers_odd = compute_psth_for_clusters(
                    spikes,
                    selected_cluster_ids,
                    events_odd,
                    config["PSTH_WINDOW_START"],
                    config["PSTH_WINDOW_END"],
                    config["BIN_SIZE"],
                    config["SMOOTH_SIGMA"],
                    show_progress=True,
                    desc=f"PSTH Odd ({event_name})",
                )
                psth_by_cluster_even, bin_centers_even = compute_psth_for_clusters(
                    spikes,
                    selected_cluster_ids,
                    events_even,
                    config["PSTH_WINDOW_START"],
                    config["PSTH_WINDOW_END"],
                    config["BIN_SIZE"],
                    config["SMOOTH_SIGMA"],
                    show_progress=True,
                    desc=f"PSTH Even ({event_name})",
                )

        delays = []
        responsive_flags = []
        response_signs = []
        delays_odd = []
        delays_even = []
        response_signs_odd = []
        response_signs_even = []
        for cid in tqdm(selected_cluster_ids, desc=f"Delays ({event_name})", unit="cluster"):
            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry = psth_by_cluster.get(cid)
            fr_raw = psth_entry["fr_raw"] if psth_entry else None
            fr_smooth = psth_entry["fr_smooth"] if psth_entry else None
            delay, is_responsive, response_sign = calculate_delay(
                fr_raw,
                fr_smooth,
                bin_centers,
                event_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events,
                trial_contrasts=contrasts,
                return_sign=True,
            )
            if not is_responsive:
                delay = np.nan
                response_sign = "none"
            delays.append(delay)
            responsive_flags.append(bool(is_responsive))
            response_signs.append(response_sign)

            if include_splits:
                delay_odd = np.nan
                delay_even = np.nan
                response_sign_odd = "none"
                response_sign_even = "none"
                if split_ready:
                    odd_entry = psth_by_cluster_odd.get(cid)
                    fr_raw_odd = odd_entry["fr_raw"] if odd_entry else None
                    fr_smooth_odd = odd_entry["fr_smooth"] if odd_entry else None
                    delay_odd, resp_odd, response_sign_odd = calculate_delay(
                        fr_raw_odd,
                        fr_smooth_odd,
                        bin_centers_odd,
                        event_config,
                        method=event_method,
                        neuron_spikes=neuron_spikes,
                        event_times=events_odd,
                        trial_contrasts=contrasts_odd,
                        return_sign=True,
                    )
                    if not resp_odd:
                        delay_odd = np.nan
                        response_sign_odd = "none"

                    even_entry = psth_by_cluster_even.get(cid)
                    fr_raw_even = even_entry["fr_raw"] if even_entry else None
                    fr_smooth_even = even_entry["fr_smooth"] if even_entry else None
                    delay_even, resp_even, response_sign_even = calculate_delay(
                        fr_raw_even,
                        fr_smooth_even,
                        bin_centers_even,
                        event_config,
                        method=event_method,
                        neuron_spikes=neuron_spikes,
                        event_times=events_even,
                        trial_contrasts=contrasts_even,
                        return_sign=True,
                    )
                    if not resp_even:
                        delay_even = np.nan
                        response_sign_even = "none"
                delays_odd.append(delay_odd)
                delays_even.append(delay_even)
                response_signs_odd.append(response_sign_odd if np.isfinite(delay_odd) else "none")
                response_signs_even.append(response_sign_even if np.isfinite(delay_even) else "none")

        df_res[delay_col] = delays
        df_res[resp_col] = responsive_flags
        df_res[sign_col] = response_signs
        if include_splits:
            df_res[delay_odd_col] = delays_odd
            df_res[delay_even_col] = delays_even
            df_res[sign_odd_col] = response_signs_odd
            df_res[sign_even_col] = response_signs_even

    if str(config.get("DELAY_UNITS", "s")).lower().startswith("ms"):
        for col in delay_columns:
            if col in df_res.columns:
                df_res[col] = df_res[col].astype(float) * 1000.0

    if output_path is not None:
        df_res.to_csv(output_path, index=False)
        print(f"Computed delays for {len(df_res)} neurons. Saved to {output_path}.")
    else:
        print(f"Computed delays for {len(df_res)} neurons.")
    return df_res


def calculate_delay_reliability(
    spikes,
    clusters,
    cluster_acronyms,
    events_by_name,
    contrasts_by_name,
    config,
    path_data_processed,
    pid,
    cid_to_idx,
    df_res=None,
):
    """Compute split-half delay reliability across multiple events."""
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for reliability calculation.")
        return pd.DataFrame()

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]

    results = []
    selected_cluster_ids = []
    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = io_utils.get_cluster_label(clusters, idx)
        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
            continue
        results.append({"cluster_id": cid, "acronym": cluster_acronyms[idx]})
        selected_cluster_ids.append(cid)

    df_reliability = pd.DataFrame(results)
    if df_reliability.empty:
        print("No clusters met the selection criteria.")
        return df_reliability

    spike_times_by_cluster = {
        cid: spikes.times[spikes.clusters == cid] for cid in selected_cluster_ids
    }

    for event_name in event_names:
        events = events_by_name.get(event_name, np.array([]))
        contrasts = contrasts_by_name.get(event_name, np.array([]))
        col_h1, col_h2 = reliability_column_names(event_name)
        df_reliability[col_h1] = np.nan
        df_reliability[col_h2] = np.nan

        if events is None or len(events) < config["MIN_TRIALS"]:
            continue

        mid_idx = len(events) // 2
        events_h1 = events[:mid_idx]
        events_h2 = events[mid_idx:]
        contrasts_h1 = contrasts[:mid_idx]
        contrasts_h2 = contrasts[mid_idx:]

        psth_h1, bin_centers_h1 = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_h1,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH H1 ({event_name})",
        )
        psth_h2, bin_centers_h2 = compute_psth_for_clusters(
            spikes,
            selected_cluster_ids,
            events_h2,
            config["PSTH_WINDOW_START"],
            config["PSTH_WINDOW_END"],
            config["BIN_SIZE"],
            config["SMOOTH_SIGMA"],
            show_progress=True,
            desc=f"PSTH H2 ({event_name})",
        )

        if df_res is not None and responsive_column_name(event_name) in df_res.columns:
            resp_lookup = dict(
                zip(df_res["cluster_id"].values, df_res[responsive_column_name(event_name)].values)
            )
            responsive_mask = np.array(
                [resp_lookup.get(cid, False) for cid in selected_cluster_ids], dtype=bool
            )
        else:
            responsive_mask = np.ones(len(selected_cluster_ids), dtype=bool)

        delays_h1 = []
        delays_h2 = []
        rel_config = {
            **config,
            "RESPONSIVE_WINDOW_START": config["RELIABILITY_WINDOW_START"],
            "RESPONSIVE_WINDOW_END": config["RELIABILITY_WINDOW_END"],
        }
        event_method = get_event_delay_method(config, event_name)

        for cid, is_resp in tqdm(
            list(zip(selected_cluster_ids, responsive_mask)),
            desc=f"Reliability ({event_name})",
            unit="cluster",
        ):
            if not is_resp:
                delays_h1.append(np.nan)
                delays_h2.append(np.nan)
                continue

            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry_h1 = psth_h1.get(cid)
            psth_entry_h2 = psth_h2.get(cid)
            fr_raw_h1 = psth_entry_h1["fr_raw"] if psth_entry_h1 else None
            fr_smooth_h1 = psth_entry_h1["fr_smooth"] if psth_entry_h1 else None
            fr_raw_h2 = psth_entry_h2["fr_raw"] if psth_entry_h2 else None
            fr_smooth_h2 = psth_entry_h2["fr_smooth"] if psth_entry_h2 else None

            delay_h1, _ = calculate_delay(
                fr_raw_h1,
                fr_smooth_h1,
                bin_centers_h1,
                rel_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events_h1,
                trial_contrasts=contrasts_h1,
            )
            delay_h2, _ = calculate_delay(
                fr_raw_h2,
                fr_smooth_h2,
                bin_centers_h2,
                rel_config,
                method=event_method,
                neuron_spikes=neuron_spikes,
                event_times=events_h2,
                trial_contrasts=contrasts_h2,
            )

            delays_h1.append(delay_h1)
            delays_h2.append(delay_h2)

        df_reliability[col_h1] = delays_h1
        df_reliability[col_h2] = delays_h2

    output_path = path_data_processed / f"{pid}_delay_reliability.csv"
    df_reliability.to_csv(output_path, index=False)
    print(
        f"Found {len(df_reliability)} responsive neurons (both halves). Saved to {output_path}."
    )
    return df_reliability

def compute_population_coupling(
    spikes,
    clusters,
    cluster_acronyms,
    config,
    cluster_ids=None,
    split_halves=False,
    by_region=True,
    intervals=None,
    context_label=None,
):
    """
    Compute spike-triggered population coupling metrics for each neuron.

    When by_region is True, the population activity is computed within each region,
    excluding the neuron under consideration. Coupling strength is the stPR value
    at lag 0, and coupling max is the peak of the stPR curve.

    The stPR curve uses leave-one-out population activity as a mean-normalized
    population rate.

    Delay sign convention in this function:
    - negative coupling_delay_ms => neuron tends to lead the population
      (neuron spikes before the population modulation);
    - positive coupling_delay_ms => neuron tends to follow the population.
    This convention is enforced by mirroring each spike-triggered segment
    before averaging.

    If intervals is provided, coupling is computed only within those windows, and
    spike-triggered segments that cross interval boundaries are excluded.
    """
    bin_size = config.get("STPR_BIN_SIZE", 0.001)
    window_ms = config.get("STPR_WINDOW_MS", 80)
    lowpass_hz = config.get("STPR_LOW_PASS_HZ", 20)
    lowpass_order = 3
    use_good_population = config.get("STPR_POP_USE_GOOD_UNITS", False)

    def _base_columns(include_halves):
        base = [
            "cluster_id",
            "region",
            "coupling_delay_ms",
            "coupling_strength",
            "coupling_max",
            "stpr_curve",
            "stpr_curve_raw",
            "sorting_number",
        ]
        if include_halves:
            base = [
                "cluster_id",
                "region",
                "coupling_delay_ms",
                "coupling_strength",
                "coupling_max",
                "stpr_curve",
                "stpr_curve_raw",
                "coupling_delay_ms_h1",
                "coupling_strength_h1",
                "coupling_max_h1",
                "stpr_curve_h1",
                "stpr_curve_raw_h1",
                "coupling_delay_ms_h2",
                "coupling_strength_h2",
                "coupling_max_h2",
                "stpr_curve_h2",
                "stpr_curve_raw_h2",
                "sorting_number",
            ]
        return base

    if spikes is None or len(spikes.get("times", [])) == 0:
        return pd.DataFrame(columns=_base_columns(split_halves))

    spike_times = np.asarray(spikes["times"])
    spike_clusters = np.asarray(spikes["clusters"])

    def _get_array(obj, key):
        if obj is None:
            return None
        if hasattr(obj, key):
            return np.asarray(getattr(obj, key))
        if isinstance(obj, dict) and key in obj:
            return np.asarray(obj[key])
        return None

    cluster_id_all = _get_array(clusters, "cluster_id")
    if cluster_id_all is None:
        cluster_id_all = np.arange(len(cluster_acronyms))

    if cluster_ids is None:
        cluster_ids = np.asarray(cluster_id_all)
    else:
        cluster_ids = np.asarray(cluster_ids)

    if cluster_ids is None or len(cluster_ids) == 0:
        return pd.DataFrame(columns=_base_columns(split_halves))

    labels = None
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            labels = np.asarray(clusters.metrics.label)
    if labels is None:
        labels = _get_array(clusters, "label")

    if cluster_ids is not None and len(cluster_ids) > 0:
        population_cluster_ids = np.asarray(cluster_ids)
    elif use_good_population and labels is not None:
        population_cluster_ids = np.asarray(cluster_id_all)[labels == 1]
    else:
        population_cluster_ids = np.asarray(cluster_id_all)

    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    region_lookup = dict(zip(cluster_id_all, cluster_acronyms))

    if by_region:
        region_groups = {}
        for cid in cluster_ids:
            region = region_lookup.get(cid, "NA")
            region_groups.setdefault(region, []).append(cid)
        population_groups = {}
        for cid in population_cluster_ids:
            region = region_lookup.get(cid, "NA")
            population_groups.setdefault(region, []).append(cid)
    else:
        region_groups = {"all": list(cluster_ids)}
        population_groups = {"all": list(population_cluster_ids)}

    bin_size_ms = bin_size * 1000
    window_bins = int(round(window_ms / bin_size_ms)) if bin_size_ms > 0 else 0
    lags_ms = np.arange(-window_bins, window_bins + 1) * bin_size_ms
    fs_hz = 1.0 / bin_size if bin_size > 0 else 0.0

    def _compute_for_spikes(spike_times_local, spike_clusters_local, desc_suffix=""):
        if spike_times_local is None or len(spike_times_local) == 0:
            return pd.DataFrame(
                [
                    {
                        "cluster_id": cid,
                        "region": region_lookup.get(cid, "NA"),
                        "coupling_delay_ms": np.nan,
                        "coupling_strength": np.nan,
                        "coupling_max": np.nan,
                        "stpr_curve": [],
                        "stpr_curve_raw": [],
                    }
                    for cid in cluster_ids
                ]
            )

        start_time = spike_times_local.min()
        end_time = spike_times_local.max()
        bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
        bins_count = len(bin_edges) - 1
        if bins_count <= 0:
            return pd.DataFrame(
                [
                    {
                        "cluster_id": cid,
                        "region": region_lookup.get(cid, "NA"),
                        "coupling_delay_ms": np.nan,
                        "coupling_strength": np.nan,
                        "coupling_max": np.nan,
                        "stpr_curve": [],
                        "stpr_curve_raw": [],
                    }
                    for cid in cluster_ids
                ]
            )

        use_cluster_set = set(cluster_ids).union(set(population_cluster_ids))
        unique_clusters = [
            cid for cid in np.unique(spike_clusters_local) if cid in use_cluster_set
        ]
        cluster_to_counts = {}
        spike_times_by_cluster = {}
        for cid in unique_clusters:
            cluster_spikes = spike_times_local[spike_clusters_local == cid]
            spike_times_by_cluster[cid] = cluster_spikes
            cluster_to_counts[cid], _ = np.histogram(cluster_spikes, bins=bin_edges)

        if intervals is not None:
            intervals_arr = np.asarray(intervals, dtype=float)
            if intervals_arr.ndim == 1 and intervals_arr.size == 2:
                intervals_arr = intervals_arr.reshape(1, 2)
            bin_centers = bin_edges[:-1] + bin_size / 2.0
            valid_bins = np.zeros(bins_count, dtype=bool)
            for start, end in intervals_arr:
                if not np.isfinite(start) or not np.isfinite(end):
                    continue
                if end <= start:
                    continue
                start = max(start, bin_edges[0])
                end = min(end, bin_edges[-1])
                if end <= start:
                    continue
                valid_bins |= (bin_centers >= start) & (bin_centers < end)
            if not np.any(valid_bins):
                return pd.DataFrame(
                    [
                        {
                            "cluster_id": cid,
                            "region": region_lookup.get(cid, "NA"),
                            "coupling_delay_ms": np.nan,
                            "coupling_strength": np.nan,
                            "coupling_max": np.nan,
                            "stpr_curve": [],
                            "stpr_curve_raw": [],
                        }
                        for cid in cluster_ids
                    ]
                )
        else:
            valid_bins = np.ones(bins_count, dtype=bool)

        invalid_prefix = np.concatenate(
            ([0], np.cumsum((~valid_bins).astype(int)))
        )
        valid_bins_count = int(np.sum(valid_bins))

        results = []
        for region, target_cids in region_groups.items():
            population_cids = population_groups.get(region, [])
            if len(target_cids) == 0:
                continue
            if len(population_cids) == 0:
                for cid in target_cids:
                    results.append(
                        {
                            "cluster_id": cid,
                            "region": region_lookup.get(cid, "NA"),
                            "coupling_delay_ms": np.nan,
                            "coupling_strength": np.nan,
                            "coupling_max": np.nan,
                            "stpr_curve": [],
                            "stpr_curve_raw": [],
                        }
                    )
                continue

            population_counts = np.zeros(bins_count, dtype=float)
            mean_rate_by_cid = {}
            for cid in population_cids:
                counts = cluster_to_counts.get(cid, None)
                if counts is None:
                    continue
                population_counts += counts
                if valid_bins_count > 0:
                    mean_rate_by_cid[cid] = float(
                        np.sum(counts[valid_bins]) / (valid_bins_count * bin_size)
                    ) if bin_size > 0 else 0.0
                else:
                    mean_rate_by_cid[cid] = 0.0
            population_cluster_set = set(population_cids)
            sum_mu_all = float(np.sum(list(mean_rate_by_cid.values()))) if mean_rate_by_cid else 0.0

            label = context_label or "coupling"
            desc = f"stPR {label}{desc_suffix} ({region})"
            for cid in tqdm(target_cids, desc=desc, unit="cluster"):
                neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
                if len(neuron_spikes) == 0:
                    results.append(
                        {
                            "cluster_id": cid,
                            "region": region_lookup.get(cid, "NA"),
                            "coupling_delay_ms": np.nan,
                            "coupling_strength": np.nan,
                            "coupling_max": np.nan,
                            "stpr_curve": [],
                            "stpr_curve_raw": [],
                        }
                    )
                    continue

                neuron_counts = cluster_to_counts.get(
                    cid, np.zeros_like(population_counts)
                )
                if cid in population_cluster_set:
                    population_counts_excl = population_counts - neuron_counts
                else:
                    population_counts_excl = population_counts

                population_rate = (
                    population_counts_excl / bin_size if bin_size > 0 else population_counts_excl
                )
                mu_i = mean_rate_by_cid.get(cid, 0.0) if cid in population_cluster_set else 0.0
                sum_mu_excl = sum_mu_all - mu_i
                if sum_mu_excl <= 0:
                    results.append(
                        {
                            "cluster_id": cid,
                            "region": region_lookup.get(cid, "NA"),
                            "coupling_delay_ms": np.nan,
                            "coupling_strength": np.nan,
                            "coupling_max": np.nan,
                            "stpr_curve": [],
                            "stpr_curve_raw": [],
                        }
                    )
                    continue
                pop_trace = (population_rate - sum_mu_excl) / sum_mu_excl

                segments = []
                for spike_time in neuron_spikes:
                    bin_idx = np.searchsorted(bin_edges, spike_time, side="right") - 1
                    start_idx = bin_idx - window_bins
                    end_idx = bin_idx + window_bins + 1
                    if start_idx < 0 or end_idx > bins_count:
                        continue
                    if invalid_prefix[end_idx] != invalid_prefix[start_idx]:
                        continue
                    # Mirror the lag axis so negative delays represent leaders
                    # (neuron activity precedes population activity).
                    segments.append(pop_trace[start_idx:end_idx][::-1])

                if len(segments) == 0:
                    results.append(
                        {
                            "cluster_id": cid,
                            "region": region_lookup.get(cid, "NA"),
                            "coupling_delay_ms": np.nan,
                            "coupling_strength": np.nan,
                            "coupling_max": np.nan,
                            "stpr_curve": [],
                            "stpr_curve_raw": [],
                        }
                    )
                    continue

                stpr_raw = np.mean(np.vstack(segments), axis=0)
                stpr = _lowpass_filter(stpr_raw, fs_hz, lowpass_hz, order=lowpass_order)

                delay_ms, strength, peak = _stpr_metrics_from_curve(stpr, lags_ms)
                if not np.isfinite(delay_ms) or abs(delay_ms) > window_ms:
                    delay_ms = np.nan

                results.append(
                    {
                        "cluster_id": cid,
                        "region": region_lookup.get(cid, "NA"),
                        "coupling_delay_ms": delay_ms,
                        "coupling_strength": strength,
                        "coupling_max": peak,
                        "stpr_curve": stpr.tolist(),
                        "stpr_curve_raw": stpr_raw.tolist(),
                    }
                )

        return pd.DataFrame(results)

    if not split_halves:
        df = _compute_for_spikes(spike_times, spike_clusters)
        valid_mask = df["coupling_delay_ms"].notna()
        sorted_indices = df.loc[valid_mask, "coupling_delay_ms"].sort_values().index
        df["sorting_number"] = np.nan
        df.loc[sorted_indices, "sorting_number"] = np.arange(len(sorted_indices))
        df["sorting_number"] = df["sorting_number"].astype("Int64")
        return df

    start_time = spike_times.min()
    end_time = spike_times.max()
    mid_time = (start_time + end_time) / 2
    mask_h1 = spike_times <= mid_time
    mask_h2 = spike_times > mid_time

    df_h1 = _compute_for_spikes(spike_times[mask_h1], spike_clusters[mask_h1], " (H1)")
    df_h2 = _compute_for_spikes(spike_times[mask_h2], spike_clusters[mask_h2], " (H2)")

    df = df_h1.merge(df_h2, on=["cluster_id", "region"], how="outer", suffixes=("_h1", "_h2"))

    mean_curves = []
    mean_raw_curves = []
    delay_means = []
    strength_means = []
    peak_means = []
    for curve_h1, curve_h2, curve_raw_h1, curve_raw_h2 in zip(
        df["stpr_curve_h1"], df["stpr_curve_h2"], df["stpr_curve_raw_h1"], df["stpr_curve_raw_h2"]
    ):
        mean_curve = _mean_stpr_curve(curve_h1, curve_h2)
        mean_raw_curve = _mean_stpr_curve(curve_raw_h1, curve_raw_h2)
        delay_ms, strength, peak = _stpr_metrics_from_curve(mean_curve, lags_ms)
        if not np.isfinite(delay_ms) or abs(delay_ms) > window_ms:
            delay_ms = np.nan
        mean_curves.append(mean_curve.tolist())
        mean_raw_curves.append(mean_raw_curve.tolist())
        delay_means.append(delay_ms)
        strength_means.append(strength)
        peak_means.append(peak)

    df["stpr_curve"] = mean_curves
    df["stpr_curve_raw"] = mean_raw_curves
    df["coupling_delay_ms"] = delay_means
    df["coupling_strength"] = strength_means
    df["coupling_max"] = peak_means

    valid_mask = df["coupling_delay_ms_h1"].notna()
    sorted_indices = df.loc[valid_mask, "coupling_delay_ms_h1"].sort_values().index
    df["sorting_number"] = np.nan
    # Sorting is intentionally based on the first half to preserve downstream ordering.
    df.loc[sorted_indices, "sorting_number"] = np.arange(len(sorted_indices))
    df["sorting_number"] = df["sorting_number"].astype("Int64")

    return df


def merge_stpr_splits(df_a, df_b, config, split_a="a", split_b="b", sort_on_split_a=True):
    """Merge two stPR result tables and compute mean-curve metrics."""
    if df_a is None and df_b is None:
        return pd.DataFrame()
    if df_a is None:
        df = df_b.copy()
        if f"stpr_curve_{split_b}" in df.columns:
            df["stpr_curve"] = df[f"stpr_curve_{split_b}"]
        elif "stpr_curve" not in df.columns:
            df["stpr_curve"] = [[] for _ in range(len(df))]
        if f"stpr_curve_raw_{split_b}" in df.columns:
            df["stpr_curve_raw"] = df[f"stpr_curve_raw_{split_b}"]
        elif "stpr_curve_raw" not in df.columns:
            df["stpr_curve_raw"] = df.get("stpr_curve", [[] for _ in range(len(df))])
        return df
    if df_b is None:
        df = df_a.copy()
        if f"stpr_curve_{split_a}" in df.columns:
            df["stpr_curve"] = df[f"stpr_curve_{split_a}"]
        elif "stpr_curve" not in df.columns:
            df["stpr_curve"] = [[] for _ in range(len(df))]
        if f"stpr_curve_raw_{split_a}" in df.columns:
            df["stpr_curve_raw"] = df[f"stpr_curve_raw_{split_a}"]
        elif "stpr_curve_raw" not in df.columns:
            df["stpr_curve_raw"] = df.get("stpr_curve", [[] for _ in range(len(df))])
        return df

    df = df_a.merge(
        df_b,
        on=["cluster_id", "region"],
        how="outer",
        suffixes=(f"_{split_a}", f"_{split_b}"),
    )

    bin_size_ms = config.get("STPR_BIN_SIZE", 0.001) * 1000
    if bin_size_ms <= 0:
        bin_size_ms = 1.0
    window_ms = config.get("STPR_WINDOW_MS", 80)
    window_bins = int(round(window_ms / bin_size_ms)) if bin_size_ms > 0 else 0
    lags_ms = np.arange(-window_bins, window_bins + 1) * bin_size_ms

    mean_curves = []
    mean_raw_curves = []
    delay_means = []
    strength_means = []
    peak_means = []
    for _, row in df.iterrows():
        curve_a = row.get(f"stpr_curve_{split_a}", [])
        curve_b = row.get(f"stpr_curve_{split_b}", [])
        curve_a_raw = row.get(f"stpr_curve_raw_{split_a}", curve_a)
        curve_b_raw = row.get(f"stpr_curve_raw_{split_b}", curve_b)
        mean_curve = _mean_stpr_curve(curve_a, curve_b)
        mean_raw_curve = _mean_stpr_curve(curve_a_raw, curve_b_raw)
        delay_ms, strength, peak = _stpr_metrics_from_curve(mean_curve, lags_ms)
        if not np.isfinite(delay_ms) or abs(delay_ms) > window_ms:
            delay_ms = np.nan
        mean_curves.append(mean_curve.tolist())
        mean_raw_curves.append(mean_raw_curve.tolist())
        delay_means.append(delay_ms)
        strength_means.append(strength)
        peak_means.append(peak)

    df["stpr_curve"] = mean_curves
    df["stpr_curve_raw"] = mean_raw_curves
    df["coupling_delay_ms"] = delay_means
    df["coupling_strength"] = strength_means
    df["coupling_max"] = peak_means

    if sort_on_split_a and f"coupling_delay_ms_{split_a}" in df.columns:
        valid_mask = df[f"coupling_delay_ms_{split_a}"].notna()
        sorted_indices = df.loc[valid_mask, f"coupling_delay_ms_{split_a}"].sort_values().index
        df["sorting_number"] = np.nan
        df.loc[sorted_indices, "sorting_number"] = np.arange(len(sorted_indices))
        df["sorting_number"] = df["sorting_number"].astype("Int64")

    return df

def compute_rastermap_sorting(
    spikes,
    cluster_ids,
    cluster_acronyms,
    bin_size=0.01,
    rastermap_params=None,
    separate_by_region=True,
    region_acronyms=None,
):
    """
    Compute Rastermap sorting indices for neurons.

    Parameters
    ----------
    separate_by_region : bool, default True
        If True, compute Rastermap sorting separately for each region. If False,
        all neurons are sorted together.
    region_acronyms : list[str] or None
        Optional list of region prefixes to include when separate_by_region is True.
    """
    from rastermap import Rastermap

    spike_times = np.asarray(spikes["times"])
    spike_clusters = np.asarray(spikes["clusters"])
    cluster_ids = np.asarray(cluster_ids)
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)

    if len(cluster_ids) == 0:
        return pd.DataFrame(columns=["cluster_id", "region", "rastermap_sort"])

    start_time = spike_times.min()
    end_time = spike_times.max()
    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)

    if rastermap_params is None:
        rastermap_params = {
            "n_clusters": 100,
            "n_PCs": 64,
            "locality": 0.5,
            "time_lag_window": 15,
            "grid_upsample": 0,
        }

    if separate_by_region:
        if region_acronyms is None:
            region_list = sorted(np.unique(cluster_acronyms))
        elif isinstance(region_acronyms, str):
            region_list = [region_acronyms]
        else:
            region_list = list(region_acronyms)

        region_masks = []
        for region in region_list:
            region_masks.append((region, np.char.startswith(cluster_acronyms, region)))
    else:
        region_masks = [("all", np.ones(len(cluster_ids), dtype=bool))]

    results = []
    for region, mask in region_masks:
        region_cluster_ids = cluster_ids[mask]
        if len(region_cluster_ids) == 0:
            continue

        spike_raster = np.zeros((len(region_cluster_ids), len(bin_edges) - 1))
        for idx, cid in enumerate(region_cluster_ids):
            cluster_spikes = spike_times[spike_clusters == cid]
            spike_raster[idx], _ = np.histogram(cluster_spikes, bins=bin_edges)

        n_pcs = rastermap_params.get("n_PCs", 64)
        if n_pcs is not None:
            n_pcs = int(min(n_pcs, max(1, len(region_cluster_ids))))

        rastermap_kwargs = {**rastermap_params, "n_PCs": n_pcs}
        model = Rastermap(**rastermap_kwargs).fit(spike_raster)
        sorted_indices = np.asarray(model.isort)

        if len(sorted_indices) != len(region_cluster_ids):
            sorted_indices = np.arange(len(region_cluster_ids))

        for sort_rank, neuron_idx in enumerate(sorted_indices):
            results.append(
                {
                    "cluster_id": int(region_cluster_ids[neuron_idx]),
                    "region": region,
                    "rastermap_sort": sort_rank,
                }
            )

    df = pd.DataFrame(results)
    df["rastermap_sort"] = df["rastermap_sort"].astype("Int64")
    return df
