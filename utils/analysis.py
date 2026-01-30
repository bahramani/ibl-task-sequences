import numpy as np
import utils.io as io_utils
import pandas as pd
from scipy.ndimage import gaussian_filter1d
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
    split_halves=False,
):
    """
    Compute spike-triggered population coupling metrics for each neuron.

    The population activity is defined as the summed firing rate of all neurons
    (optionally restricted to good units) excluding the neuron under consideration.
    Coupling strength is the peak of the z-scored spike-triggered population rate,
    and coupling delay is the lag (ms) at which this peak occurs.

    If split_halves is True, the coupling is computed separately for the first and
    second half of the provided spikes (by time). The returned DataFrame includes
    *_h1 and *_h2 columns, while the base coupling_* columns are derived from the
    mean curve across halves. Sorting is intentionally based on the first half
    to preserve downstream ordering expectations.
    """
    bin_size = config.get("STPR_BIN_SIZE", 0.001)
    window_ms = config.get("STPR_WINDOW_MS", 80)
    smooth_sigma_ms = config.get("STPR_SMOOTH_SIGMA_MS", 5)
    use_good_population = config.get("STPR_POP_USE_GOOD_UNITS", False)

    if spikes is None or len(spikes.get("times", [])) == 0:
        base_columns = [
            "cluster_id",
            "region",
            "coupling_delay_ms",
            "coupling_strength",
            "stpr_curve",
            "sorting_number",
        ]
        if split_halves:
            base_columns = [
                "cluster_id",
                "region",
                "coupling_delay_ms",
                "coupling_strength",
                "stpr_curve",
                "coupling_delay_ms_h1",
                "coupling_strength_h1",
                "stpr_curve_h1",
                "coupling_delay_ms_h2",
                "coupling_strength_h2",
                "stpr_curve_h2",
                "sorting_number",
            ]
        return pd.DataFrame(columns=base_columns)

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

    def _compute_coupling_for_spikes(spike_times_local, spike_clusters_local, desc_suffix=""):
        results_local = {}
        if spike_times_local is None or len(spike_times_local) == 0:
            for cid_local in cluster_ids:
                results_local[cid_local] = {
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
            return results_local

        start_time_local = spike_times_local.min()
        end_time_local = spike_times_local.max()
        bin_edges_local = np.arange(start_time_local, end_time_local + bin_size, bin_size)
        bin_centers_local = (bin_edges_local[:-1] + bin_edges_local[1:]) / 2
        bins_count_local = len(bin_centers_local)
        if bins_count_local == 0:
            for cid_local in cluster_ids:
                results_local[cid_local] = {
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
            return results_local

        cluster_to_counts_local = {}
        for cid_local in np.unique(spike_clusters_local):
            cluster_spikes = spike_times_local[spike_clusters_local == cid_local]
            cluster_to_counts_local[cid_local], _ = np.histogram(
                cluster_spikes, bins=bin_edges_local
            )

        population_counts_local = np.zeros(bins_count_local, dtype=float)
        for cid_local in population_cluster_ids:
            population_counts_local += cluster_to_counts_local.get(cid_local, 0)

        desc = "stPR coupling" + desc_suffix
        for cid_local in tqdm(cluster_ids, desc=desc, unit="cluster"):
            neuron_spikes = spike_times_local[spike_clusters_local == cid_local]
            if len(neuron_spikes) == 0:
                results_local[cid_local] = {
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
                continue

            neuron_counts = cluster_to_counts_local.get(
                cid_local, np.zeros_like(population_counts_local)
            )
            if cid_local in population_cluster_set:
                population_counts_excl = population_counts_local - neuron_counts
            else:
                population_counts_excl = population_counts_local

            population_rate = population_counts_excl / bin_size
            pop_mean = np.mean(population_rate)
            pop_std = np.std(population_rate)
            if pop_std == 0:
                pop_std = 1.0

            segments = []
            for spike_time in neuron_spikes:
                bin_idx = np.searchsorted(bin_edges_local, spike_time, side="right") - 1
                start_idx = bin_idx - window_bins
                end_idx = bin_idx + window_bins + 1
                if start_idx < 0 or end_idx > bins_count_local:
                    continue
                segments.append(population_rate[start_idx:end_idx])

            if len(segments) == 0:
                results_local[cid_local] = {
                    "coupling_delay_ms": np.nan,
                    "coupling_strength": np.nan,
                    "stpr_curve": [],
                }
                continue

            stpr = np.mean(np.vstack(segments), axis=0)
            if smooth_sigma_bins > 0:
                stpr = gaussian_filter1d(stpr, sigma=smooth_sigma_bins)

            stpr_z = (stpr - pop_mean) / pop_std
            peak_idx = int(np.argmax(stpr_z))
            coupling_strength = float(stpr_z[peak_idx])
            stpr_pos = np.clip(stpr_z, 0, None)
            stpr_sum = np.sum(stpr_pos)
            if stpr_sum > 0:
                coupling_delay_ms = float(np.sum(lags_ms * stpr_pos) / stpr_sum)
            else:
                coupling_delay_ms = np.nan
            if not np.isfinite(coupling_delay_ms) or abs(coupling_delay_ms) > window_ms:
                coupling_delay_ms = np.nan

            results_local[cid_local] = {
                "coupling_delay_ms": coupling_delay_ms,
                "coupling_strength": coupling_strength,
                "stpr_curve": stpr_z.tolist(),
            }

        return results_local

    if not split_halves:
        results = []
        results_map = _compute_coupling_for_spikes(spike_times, spike_clusters)
        for cid in cluster_ids:
            res = results_map.get(
                cid,
                {"coupling_delay_ms": np.nan, "coupling_strength": np.nan, "stpr_curve": []},
            )
            results.append(
                {
                    "cluster_id": cid,
                    "region": region_lookup.get(cid, "NA"),
                    "coupling_delay_ms": res["coupling_delay_ms"],
                    "coupling_strength": res["coupling_strength"],
                    "stpr_curve": res["stpr_curve"],
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

    # Split into halves by time and compute separately.
    start_time = spike_times.min()
    end_time = spike_times.max()
    mid_time = (start_time + end_time) / 2
    mask_h1 = spike_times <= mid_time
    mask_h2 = spike_times > mid_time

    results_h1 = _compute_coupling_for_spikes(
        spike_times[mask_h1], spike_clusters[mask_h1], desc_suffix=" (H1)"
    )
    results_h2 = _compute_coupling_for_spikes(
        spike_times[mask_h2], spike_clusters[mask_h2], desc_suffix=" (H2)"
    )

    df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "region": [region_lookup.get(cid, "NA") for cid in cluster_ids],
        }
    )
    df["coupling_delay_ms_h1"] = [
        results_h1.get(cid, {}).get("coupling_delay_ms", np.nan) for cid in cluster_ids
    ]
    df["coupling_strength_h1"] = [
        results_h1.get(cid, {}).get("coupling_strength", np.nan) for cid in cluster_ids
    ]
    df["stpr_curve_h1"] = [
        results_h1.get(cid, {}).get("stpr_curve", []) for cid in cluster_ids
    ]
    df["coupling_delay_ms_h2"] = [
        results_h2.get(cid, {}).get("coupling_delay_ms", np.nan) for cid in cluster_ids
    ]
    df["coupling_strength_h2"] = [
        results_h2.get(cid, {}).get("coupling_strength", np.nan) for cid in cluster_ids
    ]
    df["stpr_curve_h2"] = [
        results_h2.get(cid, {}).get("stpr_curve", []) for cid in cluster_ids
    ]

    mean_curves = []
    delay_means = []
    strength_means = []

    for curve_h1, curve_h2 in zip(df["stpr_curve_h1"], df["stpr_curve_h2"]):
        curve_h1 = curve_h1 if curve_h1 is not None else []
        curve_h2 = curve_h2 if curve_h2 is not None else []
        if len(curve_h1) == 0 and len(curve_h2) == 0:
            mean_curve = np.array([])
        elif len(curve_h1) == 0:
            mean_curve = np.asarray(curve_h2, dtype=float)
        elif len(curve_h2) == 0:
            mean_curve = np.asarray(curve_h1, dtype=float)
        else:
            if len(curve_h1) != len(curve_h2):
                min_len = min(len(curve_h1), len(curve_h2))
                curve_h1 = curve_h1[:min_len]
                curve_h2 = curve_h2[:min_len]
            mean_curve = (np.asarray(curve_h1, dtype=float) + np.asarray(curve_h2, dtype=float)) / 2

        mean_curves.append(mean_curve.tolist())
        if len(mean_curve) == 0 or not np.isfinite(mean_curve).any():
            delay_means.append(np.nan)
            strength_means.append(np.nan)
        else:
            peak_idx = int(np.nanargmax(mean_curve))
            strength_means.append(float(mean_curve[peak_idx]))
            lags = lags_ms[: len(mean_curve)]
            stpr_pos = np.clip(mean_curve, 0, None)
            stpr_sum = np.nansum(stpr_pos)
            if stpr_sum > 0:
                delay_val = float(np.nansum(lags * stpr_pos) / stpr_sum)
            else:
                delay_val = np.nan
            if not np.isfinite(delay_val) or abs(delay_val) > window_ms:
                delay_val = np.nan
            delay_means.append(delay_val)

    df["stpr_curve"] = mean_curves
    df["coupling_delay_ms"] = delay_means
    df["coupling_strength"] = strength_means

    valid_mask = df["coupling_delay_ms_h1"].notna()
    sorted_indices = df.loc[valid_mask, "coupling_delay_ms_h1"].sort_values().index
    sorting_numbers = pd.Series(np.arange(len(sorted_indices)), index=sorted_indices)
    df["sorting_number"] = np.nan
    # Sorting is intentionally based on the first half to preserve downstream ordering.
    df.loc[sorted_indices, "sorting_number"] = sorting_numbers
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
