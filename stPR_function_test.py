from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.analysis import compute_population_coupling
from utils.plotting_plotly import plot_time_window_raster_plotly


def _simulate_spike_population(
    n_neurons: int = 100,
    n_coupled: int = 80,
    duration_s: float = 180.0,
    sequence_period_s: float = 0.35,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    cluster_ids = np.arange(n_neurons, dtype=int)
    cluster_acronyms = np.array(["SIM"] * n_neurons)

    leader_ids = np.arange(0, 20, dtype=int)
    follower_ids = np.arange(20, 40, dtype=int)
    near_zero_ids = np.arange(40, 60, dtype=int)
    mixed_coupled_ids = np.arange(60, n_coupled, dtype=int)
    uncoupled_ids = np.arange(n_coupled, n_neurons, dtype=int)

    group = np.array(["uncoupled"] * n_neurons, dtype=object)
    group[leader_ids] = "leader"
    group[follower_ids] = "follower"
    group[near_zero_ids] = "near_zero"
    group[mixed_coupled_ids] = "mixed_coupled"

    offsets_s = np.zeros(n_neurons, dtype=float)
    offsets_s[leader_ids] = rng.uniform(-0.060, -0.020, size=leader_ids.size)
    offsets_s[follower_ids] = rng.uniform(0.020, 0.060, size=follower_ids.size)
    offsets_s[near_zero_ids] = rng.uniform(-0.008, 0.008, size=near_zero_ids.size)
    offsets_s[mixed_coupled_ids] = rng.uniform(-0.030, 0.030, size=mixed_coupled_ids.size)

    coupling_prob = np.zeros(n_neurons, dtype=float)
    coupling_prob[leader_ids] = rng.uniform(0.60, 0.95, size=leader_ids.size)
    coupling_prob[follower_ids] = rng.uniform(0.50, 0.90, size=follower_ids.size)
    coupling_prob[near_zero_ids] = rng.uniform(0.35, 0.75, size=near_zero_ids.size)
    coupling_prob[mixed_coupled_ids] = rng.uniform(0.20, 0.70, size=mixed_coupled_ids.size)

    noise_rate_hz = np.zeros(n_neurons, dtype=float)
    noise_rate_hz[:n_coupled] = rng.uniform(0.2, 1.2, size=n_coupled)
    noise_rate_hz[n_coupled:] = rng.uniform(1.0, 4.0, size=n_neurons - n_coupled)

    seq_event_times = np.arange(0.5, duration_s - 0.5, sequence_period_s)
    seq_event_times += rng.normal(0.0, 0.008, size=seq_event_times.size)
    seq_event_times = seq_event_times[(seq_event_times > 0.1) & (seq_event_times < (duration_s - 0.1))]

    all_times = []
    all_clusters = []
    for cid in cluster_ids:
        coupled_spikes = np.array([], dtype=float)
        if cid < n_coupled:
            keep_mask = rng.random(seq_event_times.size) < coupling_prob[cid]
            base_times = seq_event_times[keep_mask]
            jitter = rng.normal(0.0, 0.004, size=base_times.size)
            coupled_spikes = base_times + offsets_s[cid] + jitter

            burst_mask = rng.random(coupled_spikes.size) < (0.10 * coupling_prob[cid])
            burst_spikes = coupled_spikes[burst_mask] + rng.normal(0.004, 0.0015, size=burst_mask.sum())
            coupled_spikes = np.concatenate([coupled_spikes, burst_spikes])
            coupled_spikes = coupled_spikes[(coupled_spikes >= 0.0) & (coupled_spikes <= duration_s)]

        n_noise = int(rng.poisson(noise_rate_hz[cid] * duration_s))
        noise_spikes = rng.uniform(0.0, duration_s, size=n_noise) if n_noise > 0 else np.array([])

        neuron_spikes = np.sort(np.concatenate([coupled_spikes, noise_spikes]))
        all_times.append(neuron_spikes)
        all_clusters.append(np.full(neuron_spikes.shape, cid, dtype=int))

    spike_times = np.concatenate(all_times)
    spike_clusters = np.concatenate(all_clusters)
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]

    spikes_dict = {"times": spike_times, "clusters": spike_clusters}
    spikes_obj = SimpleNamespace(times=spike_times, clusters=spike_clusters)

    clusters = SimpleNamespace(
        cluster_id=cluster_ids,
        depths=cluster_ids.astype(float),
        metrics=pd.DataFrame({"cluster_id": cluster_ids, "label": np.ones(n_neurons, dtype=int)}),
    )

    event_base = np.arange(1.0, duration_s - 1.0, 2.0)
    trials = {
        "stimOn_times": event_base,
        "firstMovement_times": event_base + 0.2,
        "response_times": event_base + 0.45,
        "feedback_times": event_base + 0.75,
    }
    sl = {"trials": trials, "wheel": None, "pose": None, "motion_energy": None}

    df_truth = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "true_group": group,
            "true_offset_ms": offsets_s * 1000.0,
            "true_coupling_prob": coupling_prob,
            "noise_rate_hz": noise_rate_hz,
        }
    )
    return spikes_dict, spikes_obj, clusters, cluster_ids, cluster_acronyms, sl, df_truth


def _pick_example_ids(df: pd.DataFrame, group_name: str, n_pick: int = 5):
    mask = df["true_group"] == group_name
    subset = df.loc[mask].copy()
    subset = subset[subset["stpr_curve"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    subset = subset.sort_values("coupling_strength", ascending=False)
    return subset["cluster_id"].head(n_pick).tolist()


def _plot_stpr_group_curves(df: pd.DataFrame, lags_ms: np.ndarray, out_png: Path):
    # User requested 4 groups (leader, follower, around-zero, uncoupled).
    groups = [
        ("leader", "Leaders (expected + delay)"),
        ("follower", "Followers (expected - delay)"),
        ("near_zero", "Near-zero (expected ~0 delay)"),
        ("uncoupled", "Uncoupled (expected weak/flat)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (group_name, title) in zip(axes, groups):
        picked_ids = _pick_example_ids(df, group_name, n_pick=5)
        for cid in picked_ids:
            row = df.loc[df["cluster_id"] == cid].iloc[0]
            curve = np.asarray(row["stpr_curve"], dtype=float)
            if curve.size == 0:
                continue
            x = lags_ms[: curve.size]
            delay_val = row["coupling_delay_ms"]
            ax.plot(x, curve, lw=1.6, alpha=0.9, label=f"id {cid}, d={delay_val:.1f}ms")
        ax.axvline(0.0, color="k", ls="--", lw=1.0)
        ax.axhline(0.0, color="gray", ls="-", lw=0.8, alpha=0.5)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[2:]:
        ax.set_xlabel("Lag (ms)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("stPR")

    fig.suptitle("stPR Curves for Simulated Neuron Classes", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    out_dir = Path("stpr_function_test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        spikes_dict,
        spikes_obj,
        clusters,
        cluster_ids,
        cluster_acronyms,
        sl,
        df_truth,
    ) = _simulate_spike_population()

    config_calc = {
        "STPR_BIN_SIZE": 0.001,
        "STPR_WINDOW_MS": 80,
        "STPR_LOW_PASS_HZ": 20,
        "STPR_LOW_PASS_ORDER": 3,
        "STPR_POP_USE_GOOD_UNITS": False,
    }
    config_plot = {
        "PLOT_ONLY_GOOD_UNITS": False,
        "AVG_PSTH_ONLY_GOOD": False,
        "PLOT_LABEL_MIN": 0.0,
        "POP_BIN_SIZE": 0.005,
        "PLOTLY_TEMPLATE": "plotly_white",
    }

    df_coupling = compute_population_coupling(
        spikes_dict,
        clusters,
        cluster_acronyms,
        config_calc,
        cluster_ids=cluster_ids,
        split_halves=False,
        by_region=True,
        context_label="Sim",
    )
    df_eval = df_truth.merge(df_coupling, on="cluster_id", how="left")
    df_eval.to_csv(out_dir / "simulated_stpr_results.csv", index=False)

    leaders = df_eval[df_eval["true_group"] == "leader"]["coupling_delay_ms"].dropna()
    followers = df_eval[df_eval["true_group"] == "follower"]["coupling_delay_ms"].dropna()
    near_zero = df_eval[df_eval["true_group"] == "near_zero"]["coupling_delay_ms"].dropna()
    uncoupled = df_eval[df_eval["true_group"] == "uncoupled"]["coupling_delay_ms"].dropna()

    leader_pos_frac = float((leaders > 0).mean()) if leaders.size else np.nan
    follower_neg_frac = float((followers < 0).mean()) if followers.size else np.nan
    coupled_mask = df_eval["true_group"].isin(["leader", "follower", "near_zero", "mixed_coupled"])
    rho = np.nan
    if coupled_mask.any():
        x = df_eval.loc[coupled_mask, "true_offset_ms"].to_numpy(dtype=float)
        y = df_eval.loc[coupled_mask, "coupling_delay_ms"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2:
            rho = float(np.corrcoef(x[valid], y[valid])[0, 1])

    print("Simulation summary")
    print(f"  spikes total: {len(spikes_dict['times'])}")
    print(f"  units: {len(cluster_ids)} (coupled={int((cluster_ids < 80).sum())}, uncoupled=20)")
    print("Delay sign check (expected: leaders positive, followers negative)")
    print(f"  leaders with positive delay:  {leader_pos_frac:.2%}")
    print(f"  followers with negative delay: {follower_neg_frac:.2%}")
    print("Group median delays (ms)")
    print(f"  leaders:   {leaders.median():.2f}")
    print(f"  followers: {followers.median():.2f}")
    print(f"  near_zero: {near_zero.median():.2f}")
    print(f"  uncoupled: {uncoupled.median():.2f}")
    print(f"  corr(true_offset_ms, estimated_delay_ms) for coupled units: {rho:.3f}")

    window_bins = int(round(config_calc["STPR_WINDOW_MS"] / (config_calc["STPR_BIN_SIZE"] * 1000.0)))
    lags_ms = np.arange(-window_bins, window_bins + 1) * (config_calc["STPR_BIN_SIZE"] * 1000.0)
    _plot_stpr_group_curves(df_eval, lags_ms, out_dir / "stpr_group_curves.png")

    t_start, t_end = 20.0, 30.0
    fig_original = plot_time_window_raster_plotly(
        spikes_obj,
        clusters,
        cluster_ids,
        cluster_acronyms,
        sl,
        config_plot,
        t_start,
        t_end,
        sorting_metric="depth",
        variability_metric="fano",
        df_coupling=df_coupling,
    )
    fig_sorted = plot_time_window_raster_plotly(
        spikes_obj,
        clusters,
        cluster_ids,
        cluster_acronyms,
        sl,
        config_plot,
        t_start,
        t_end,
        sorting_metric="spont",
        variability_metric="fano",
        df_coupling=df_coupling,
    )

    fig_original.write_html(out_dir / "general_raster_original_depth.html")
    fig_sorted.write_html(out_dir / "general_raster_sorted_by_stpr_delay.html")

    print("\nSaved outputs")
    print(f"  {out_dir / 'simulated_stpr_results.csv'}")
    print(f"  {out_dir / 'stpr_group_curves.png'}")
    print(f"  {out_dir / 'general_raster_original_depth.html'}")
    print(f"  {out_dir / 'general_raster_sorted_by_stpr_delay.html'}")


if __name__ == "__main__":
    main()
