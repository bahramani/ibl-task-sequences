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
    }
    return label_map.get(event_name, event_name)


def delay_column_name(event_name):
    return f"delay_{event_name}"


def responsive_column_name(event_name):
    return f"responsive_{event_name}"


def delay_split_column_name(event_name, split_label):
    return f"{delay_column_name(event_name)}_{split_label}"


def responsive_split_column_name(event_name, split_label):
    return f"{responsive_column_name(event_name)}_{split_label}"


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
):
    """
    Compute delay within a responsive window and responsiveness.

    Supported methods:
    - "center_of_mass": PSTH center of mass within the responsive window.
    - "psth_peak": peak time of the PSTH within the responsive window.
    - "tfs": time to first spike after event onset (100% contrast trials only).
    """
    # Default to the configured delay method if none is provided explicitly.
    method = method or config.get("DELAY_METHOD", "center_of_mass")

    if method == "tfs":
        # TFS relies on single-trial spike times, not the PSTH.
        if neuron_spikes is None or event_times is None or trial_contrasts is None:
            return np.nan, False
        if len(neuron_spikes) == 0 or len(event_times) == 0:
            return np.nan, False

        # Identify 100% contrast trials. Accept 1.0 or 100.0 by default.
        full_contrast_values = config.get("FULL_CONTRAST_VALUES", (1.0, 100.0))
        full_mask = np.zeros(len(trial_contrasts), dtype=bool)
        for val in full_contrast_values:
            full_mask |= np.isclose(trial_contrasts, val)
        if not np.any(full_mask):
            return np.nan, False

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
            return np.nan, False
        return float(np.mean(first_spike_offsets)), True

    # From here on we rely on the PSTH representation.
    if fr_raw is None or bin_centers is None:
        return np.nan, False

    # Compute a baseline threshold from the unsmoothed PSTH to avoid smoothing bias.
    idx_baseline = (bin_centers >= -config["BASELINE_PRE"]) & (bin_centers < 0)
    baseline_fr = fr_raw[idx_baseline]
    if len(baseline_fr) == 0:
        return np.nan, False

    threshold = np.mean(baseline_fr) + 2 * np.std(baseline_fr)
    idx_responsive = (bin_centers >= config["RESPONSIVE_WINDOW_START"]) & (
        bin_centers <= config["RESPONSIVE_WINDOW_END"]
    )

    responsive_mask = idx_responsive & (fr_raw > threshold)
    if not np.any(responsive_mask):
        return np.nan, False

    resp_fr = fr_smooth[responsive_mask] if fr_smooth is not None else fr_raw[responsive_mask]
    resp_time = bin_centers[responsive_mask]
    if method == "psth_peak":
        # Peak time is the bin center at maximum firing rate within the window.
        peak_idx = int(np.argmax(resp_fr))
        return float(resp_time[peak_idx]), True

    # Default: center of mass (previous behavior).
    sum_fr = np.sum(resp_fr)
    if sum_fr <= 0:
        return np.nan, False
    delay = np.sum(resp_fr * resp_time) / sum_fr
    return delay, True


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
    stpr_sum = np.nansum(curve)
    if stpr_sum != 0:
        delay_val = float(np.nansum(lags_ms[: curve.size] * curve) / stpr_sum)
    else:
        delay_val = np.nan
    return delay_val, strength, peak


def _lowpass_filter(signal, fs_hz, cutoff_hz, order=2):
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

        if events is None or len(events) < min_trials:
            df_res[delay_col] = np.nan
            df_res[delay_odd_col] = np.nan
            df_res[delay_even_col] = np.nan
            df_res[resp_col] = False
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
            continue

        win_start, win_end = get_event_delay_window(config, event_name)
        event_config = {
            **config,
            "RESPONSIVE_WINDOW_START": win_start,
            "RESPONSIVE_WINDOW_END": win_end,
        }

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
        for cid in tqdm(selected_cluster_ids, desc=f"Delays ({event_name})", unit="cluster"):
            neuron_spikes = spike_times_by_cluster.get(cid, np.array([]))
            psth_entry = psth_by_cluster.get(cid)
            fr_raw = psth_entry["fr_raw"] if psth_entry else None
            fr_smooth = psth_entry["fr_smooth"] if psth_entry else None

            delay, is_responsive = calculate_delay(
                fr_raw,
                fr_smooth,
                bin_centers,
                event_config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events,
                trial_contrasts=contrasts,
            )

            psth_entry_odd = psth_by_cluster_odd.get(cid)
            fr_raw_odd = psth_entry_odd["fr_raw"] if psth_entry_odd else None
            fr_smooth_odd = psth_entry_odd["fr_smooth"] if psth_entry_odd else None
            delay_odd, resp_odd = calculate_delay(
                fr_raw_odd,
                fr_smooth_odd,
                bin_centers_odd,
                event_config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events_odd,
                trial_contrasts=contrasts_odd,
            )

            psth_entry_even = psth_by_cluster_even.get(cid)
            fr_raw_even = psth_entry_even["fr_raw"] if psth_entry_even else None
            fr_smooth_even = psth_entry_even["fr_smooth"] if psth_entry_even else None
            delay_even, resp_even = calculate_delay(
                fr_raw_even,
                fr_smooth_even,
                bin_centers_even,
                event_config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events_even,
                trial_contrasts=contrasts_even,
            )

            if not (resp_odd and resp_even) or not (
                np.isfinite(delay_odd) and np.isfinite(delay_even)
            ):
                # If either split is non-responsive, mark both halves as NaN.
                delay_odd = np.nan
                delay_even = np.nan
                delay = np.nan
                is_responsive = False

            delays.append(delay)
            delays_odd.append(delay_odd)
            delays_even.append(delay_even)
            responsive_flags.append(is_responsive)

        df_res[delay_col] = delays
        df_res[delay_odd_col] = delays_odd
        df_res[delay_even_col] = delays_even
        df_res[resp_col] = responsive_flags

    output_path = path_data_processed / f"{pid}_delay_results.csv"
    df_res.to_csv(output_path, index=False)
    print(f"Computed delays for {len(df_res)} neurons. Saved to {output_path}.")
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
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events_h1,
                trial_contrasts=contrasts_h1,
            )
            delay_h2, _ = calculate_delay(
                fr_raw_h2,
                fr_smooth_h2,
                bin_centers_h2,
                rel_config,
                method=config.get("DELAY_METHOD"),
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

    If intervals is provided, coupling is computed only within those windows, and
    spike-triggered segments that cross interval boundaries are excluded.
    """
    bin_size = config.get("STPR_BIN_SIZE", 0.001)
    window_ms = config.get("STPR_WINDOW_MS", 80)
    lowpass_hz = config.get("STPR_LOW_PASS_HZ", 20)
    lowpass_order = config.get("STPR_LOW_PASS_ORDER", 2)
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

                population_rate = population_counts_excl / bin_size if bin_size > 0 else population_counts_excl
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
                normalized_pop = (population_rate - sum_mu_excl) / sum_mu_excl

                segments = []
                for spike_time in neuron_spikes:
                    bin_idx = np.searchsorted(bin_edges, spike_time, side="right") - 1
                    start_idx = bin_idx - window_bins
                    end_idx = bin_idx + window_bins + 1
                    if start_idx < 0 or end_idx > bins_count:
                        continue
                    if invalid_prefix[end_idx] != invalid_prefix[start_idx]:
                        continue
                    segments.append(normalized_pop[start_idx:end_idx])

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
