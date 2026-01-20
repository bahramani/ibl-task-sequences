import numpy as np
import utils.io as io_utils
import pandas as pd
from scipy.ndimage import gaussian_filter1d
try:
    from tqdm.auto import tqdm
except ImportError:  # Graceful fallback if tqdm is unavailable.
    def tqdm(iterable, **kwargs):
        return iterable

def get_trial_contrasts(sl):
    """Return per-trial contrasts (abs max of left/right), NaNs -> 0."""
    contrast_left = np.abs(sl.trials.contrastLeft)
    contrast_right = np.abs(sl.trials.contrastRight)
    trial_contrasts = np.nanmax(np.vstack([contrast_left, contrast_right]), axis=0)
    return np.where(np.isnan(trial_contrasts), 0, trial_contrasts)


def build_event_dicts(sl, event_names, min_trials):
    """Build per-event arrays of valid event times and aligned contrasts."""
    events_by_name = {}
    contrasts_by_name = {}
    trial_contrasts = get_trial_contrasts(sl)
    for event_name in event_names:
        if event_name not in sl.trials.keys():
            print(f"Warning: Event '{event_name}' not found in trials.")
            events_by_name[event_name] = np.array([])
            contrasts_by_name[event_name] = np.array([])
            continue
        events_all = np.asarray(sl.trials[event_name])
        valid_mask = ~np.isnan(events_all)
        events = events_all[valid_mask]
        contrasts = trial_contrasts[valid_mask]
        if len(events) < min_trials:
            print(
                f"Warning: {event_name} has only {len(events)} trials "
                f"(min {min_trials})."
            )
        events_by_name[event_name] = events
        contrasts_by_name[event_name] = contrasts
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


def reliability_column_names(event_name):
    return f"delay_h1_{event_name}", f"delay_h2_{event_name}"


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
):
    """Compute delays for all clusters across multiple events."""
    event_names = config.get("EVENT_NAMES", list(events_by_name.keys()))
    if len(event_names) == 0:
        print("No events provided for delay calculation.")
        return pd.DataFrame()

    cluster_ids = np.unique(spikes.clusters)
    cluster_ids = [cid for cid in cluster_ids if cid in cid_to_idx]
    print(f"Found {len(cluster_ids)} clusters.")

    results = []
    selected_cluster_ids = []
    for cid in cluster_ids:
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue
        label = io_utils.get_cluster_label(clusters, idx)
        if config["CALC_ONLY_GOOD_UNITS"] and label != 1:
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
        delay_col = delay_column_name(event_name)
        resp_col = responsive_column_name(event_name)

        if events is None or len(events) < config["MIN_TRIALS"]:
            df_res[delay_col] = np.nan
            df_res[resp_col] = False
            continue

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

        delays = []
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
                config,
                method=config.get("DELAY_METHOD"),
                neuron_spikes=neuron_spikes,
                event_times=events,
                trial_contrasts=contrasts,
            )

            delays.append(delay)
            responsive_flags.append(is_responsive)

        df_res[delay_col] = delays
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
):
    """
    Compute spike-triggered population coupling metrics for each neuron.

    The population activity is defined as the summed firing rate of all neurons
    (optionally restricted to good units) excluding the neuron under consideration.
    Coupling strength is the peak of the z-scored spike-triggered population rate,
    and coupling delay is the lag (ms) at which this peak occurs.
    """
    bin_size = config.get("STPR_BIN_SIZE", 0.001)
    window_ms = config.get("STPR_WINDOW_MS", 80)
    smooth_sigma_ms = config.get("STPR_SMOOTH_SIGMA_MS", 5)
    use_good_population = config.get("STPR_POP_USE_GOOD_UNITS", False)

    if spikes is None or len(spikes.get("times", [])) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "region",
                "coupling_delay_ms",
                "coupling_strength",
                "stpr_curve",
                "sorting_number",
            ]
        )

    spike_times = np.asarray(spikes["times"])
    spike_clusters = np.asarray(spikes["clusters"])

    if cluster_ids is None:
        cluster_ids = np.asarray(clusters["cluster_id"])
    else:
        cluster_ids = np.asarray(cluster_ids)

    if use_good_population and "label" in clusters:
        population_cluster_ids = clusters["cluster_id"][clusters["label"] == 1]
    else:
        population_cluster_ids = clusters["cluster_id"]

    population_cluster_ids = np.asarray(population_cluster_ids)
    population_cluster_set = set(population_cluster_ids.tolist())
    region_lookup = dict(zip(clusters["cluster_id"], cluster_acronyms))

    start_time = spike_times.min()
    end_time = spike_times.max()
    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bins_count = len(bin_centers)

    cluster_to_counts = {}
    for cid in np.unique(spike_clusters):
        cluster_spikes = spike_times[spike_clusters == cid]
        cluster_to_counts[cid], _ = np.histogram(cluster_spikes, bins=bin_edges)

    population_counts = np.zeros(bins_count, dtype=float)
    for cid in population_cluster_ids:
        population_counts += cluster_to_counts.get(cid, 0)

    bin_size_ms = bin_size * 1000
    window_bins = int(round(window_ms / bin_size_ms))
    lags_ms = np.arange(-window_bins, window_bins + 1) * bin_size_ms

    if smooth_sigma_ms > 0:
        smooth_sigma_bins = smooth_sigma_ms / bin_size_ms
    else:
        smooth_sigma_bins = 0

    results = []
    for cid in tqdm(cluster_ids, desc="stPR coupling", unit="cluster"):
        neuron_spikes = spike_times[spike_clusters == cid]
        if len(neuron_spikes) == 0:
            results.append(
                {
                    "cluster_id": cid,
                    "region": region_lookup.get(cid, "NA"),
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
            )
            continue

        neuron_counts = cluster_to_counts.get(cid, np.zeros_like(population_counts))
        if cid in population_cluster_set:
            population_counts_excl = population_counts - neuron_counts
        else:
            population_counts_excl = population_counts

        population_rate = population_counts_excl / bin_size
        pop_mean = np.mean(population_rate)
        pop_std = np.std(population_rate)
        if pop_std == 0:
            pop_std = 1.0

        segments = []
        for spike_time in neuron_spikes:
            bin_idx = np.searchsorted(bin_edges, spike_time, side="right") - 1
            start_idx = bin_idx - window_bins
            end_idx = bin_idx + window_bins + 1
            if start_idx < 0 or end_idx > bins_count:
                continue
            segments.append(population_rate[start_idx:end_idx])

        if len(segments) == 0:
            results.append(
                {
                    "cluster_id": cid,
                    "region": region_lookup.get(cid, "NA"),
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
            )
            continue

        stpr = np.mean(np.vstack(segments), axis=0)
        if smooth_sigma_bins > 0:
            stpr = gaussian_filter1d(stpr, sigma=smooth_sigma_bins)

        stpr_z = (stpr - pop_mean) / pop_std
        peak_idx = int(np.argmax(stpr_z))
        coupling_strength = float(stpr_z[peak_idx])
        stpr_sum = np.sum(stpr_z)
        if stpr_sum != 0:
            coupling_delay_ms = float(np.sum(lags_ms * stpr_z) / stpr_sum)
        else:
            coupling_delay_ms = float(lags_ms[peak_idx])

        results.append(
            {
                "cluster_id": cid,
                "region": region_lookup.get(cid, "NA"),
                "coupling_delay_ms": coupling_delay_ms,
                "coupling_strength": coupling_strength,
                "stpr_curve": stpr_z.tolist(),
            }
        )

    df = pd.DataFrame(results)
    valid_mask = df["coupling_delay_ms"].notna()
    sorted_indices = df.loc[valid_mask, "coupling_delay_ms"].sort_values().index
    sorting_numbers = pd.Series(np.arange(len(sorted_indices)), index=sorted_indices)
    df["sorting_number"] = np.nan
    df.loc[sorted_indices, "sorting_number"] = sorting_numbers
    df["sorting_number"] = df["sorting_number"].astype("Int64")

    return df