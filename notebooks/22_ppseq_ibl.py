# ruff: noqa: E402
# %% Imports
from pathlib import Path
import importlib
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd


BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))

from utils.io import init_one, load_session_data, setup_paths


# %% Config
CONFIG_PPSEQ = {
    "PID": "f967a527-257f-404a-871d-b91575dca3b4",
    "REGION": "SSp-ul",
    "LABEL_MIN": 0.6,
    "FIT_SCOPE": "session",
    "MIN_REGION_NEURONS": 10,
    "BIN_SIZE_S": 0.01,
    "TEMPLATE_DURATION_BINS": 25,
    "K_CANDIDATES": [2, 3, 4, 5, 6],
    "FORCE_K": 4,
    "ALLOW_FORCE_K_FALLBACK": True,
    "NUM_ITER": 60,
    "NUM_RESTARTS": 3,
    "INITIALIZATION": "default",
    "INITIAL_SEQUENCE_FRAC": 0.15,
    "INITIAL_TEMPLATE_CONCENTRATION": 0.6,
    "ALPHA_A0": 0.5,
    "BETA_A0": 0.0,
    "ALPHA_B0": 0.0,
    "BETA_B0": 0.0,
    "ALPHA_T0": 0.0,
    "BETA_T0": 0.0,
    "DEVICE": "auto",
    "SAVE_CACHE": True,
    "CACHE_DIR": BASE_PATH / "data" / "ppseq_cache",
    "PLOT_WINDOW_START_S": 4000.0,
    "PLOT_WINDOW_END_S": 4010.0,
    "ASSIGNMENT_MIN_SEQUENCE_RESP": 0.6,
    "ASSIGNMENT_MIN_AMPLITUDE_ZSCORE": 1.0,
    "PLOT_SEQUENCE_COLORS": {0: "gray", 1: "red", 2: "blue", 3: "green"},
    "PLOTLY_RENDERER": None,
    "RANDOM_SEED": 0,
}


# %% Bootstrap
def _run_pip_install(packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        cmd.extend(list(extra_args))
    cmd.extend(list(packages))
    print("Running install command:")
    print(" ".join(str(part) for part in cmd))
    subprocess.check_call(cmd)


def _bootstrap_ppseq_runtime():
    try:
        import plotly  # noqa: F401
    except Exception:
        print("Plotly is missing. Installing `plotly`.")
        _run_pip_install(["plotly"])

    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(
            "PyTorch import failed. Reinstalling CPU-safe PyTorch wheels in the "
            "current interpreter."
        )
        print(f"Original error: {type(exc).__name__}: {exc}")
        _run_pip_install(
            ["torch", "torchvision", "torchaudio"],
            extra_args=[
                "--upgrade",
                "--force-reinstall",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
        )
        raise RuntimeError(
            "PyTorch was reinstalled with CPU-safe wheels. Restart the kernel "
            "or interpreter, then rerun this notebook."
        ) from exc

    for package_name in ("fastprogress", "jaxtyping"):
        try:
            importlib.import_module(package_name)
        except Exception:
            print(f"Installing missing PP-Seq dependency: {package_name}")
            _run_pip_install([package_name])

    try:
        from ppseq.model import PPSeq  # noqa: F401
    except Exception:
        print("Installing `ppseq-pytorch` from GitHub.")
        _run_pip_install(["git+https://github.com/lindermanlab/ppseq-pytorch.git@main"])
        importlib.invalidate_caches()
        try:
            from ppseq.model import PPSeq  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Could not import `ppseq.model.PPSeq` after installation. "
                "Restart the kernel and rerun, or inspect the environment."
            ) from exc


_bootstrap_ppseq_runtime()

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
from fastprogress import progress_bar
from ppseq.model import PPSeq

import utils.plotting_plotly as plotting_utils
from utils.plotting_plotly import build_whisk_raster_overlay_inputs


# %% Helpers
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
    fig.show()


def _load_base_cache(pid):
    cache_path = BASE_PATH / "data" / "dashboard_cache" / f"{pid}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Base cache not found: {cache_path}. Run 03_calc_dashboard.py first."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None


def _load_raw_session(pid):
    path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    last_exc = None
    for mode in ("local", "remote"):
        try:
            one = init_one(ibl_cache, mode=mode)
            ssl, spikes, clusters, sl = load_session_data(
                pid,
                one,
                load_wheel=False,
                load_pose=False,
                load_motion_energy=False,
                load_pupil=False,
            )
            return {
                "one": one,
                "ssl": ssl,
                "spikes": spikes,
                "clusters": clusters,
                "session_loader": sl,
                "mode": mode,
                "path_data": path_data,
            }
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Raw session load failed for PID {pid}: {last_exc}")


def _prepare_units_table(base_cache, clusters, label_min):
    cluster_ids = np.asarray(base_cache.get("cluster_ids", []), dtype=int)
    cluster_acronyms = np.asarray(base_cache.get("cluster_acronyms_plot", []), dtype=str)
    if cluster_ids.size == 0 or cluster_acronyms.size == 0:
        raise RuntimeError("Base cache does not contain `cluster_ids`/`cluster_acronyms_plot`.")

    df_units = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "acronym": cluster_acronyms.astype(str),
        }
    )
    label_values = plotting_utils._get_label_values(clusters, cluster_ids)
    if label_values is not None:
        df_units["label"] = pd.to_numeric(label_values, errors="coerce")
    else:
        df_units["label"] = np.nan
        df_res = base_cache.get("df_res", pd.DataFrame())
        if isinstance(df_res, pd.DataFrame) and not df_res.empty and "cluster_id" in df_res.columns:
            merge_cols = ["cluster_id"]
            if "label" in df_res.columns:
                merge_cols.append("label")
            df_units = df_units.merge(
                df_res[merge_cols].drop_duplicates("cluster_id"),
                on="cluster_id",
                how="left",
                suffixes=("", "_df_res"),
            )
            if "label_df_res" in df_units.columns:
                df_units["label"] = df_units["label"].where(
                    df_units["label"].notna(),
                    pd.to_numeric(df_units["label_df_res"], errors="coerce"),
                )
                df_units = df_units.drop(columns=["label_df_res"])

    df_units["acronym"] = df_units["acronym"].astype(str)
    label_vals = pd.to_numeric(df_units["label"], errors="coerce")
    keep_mask = ~df_units["acronym"].isin(["root", "void"]) & (label_vals >= float(label_min))
    return df_units.loc[keep_mask].reset_index(drop=True)


def _resolve_target_regions(df_units, region_config):
    if df_units.empty:
        raise RuntimeError("No eligible units remain after region and label filtering.")

    all_regions = sorted(df_units["acronym"].astype(str).unique().tolist())
    if region_config is None:
        return all_regions

    region_name = str(region_config).strip()
    if region_name not in all_regions:
        raise ValueError(
            f"REGION `{region_name}` not found among eligible exact regions: {all_regions}"
        )
    return [region_name]


def _resolve_spont_interval(base_cache, one):
    meta = base_cache.get("meta", {}) if isinstance(base_cache, dict) else {}
    spont_interval = meta.get("spont_interval")
    if spont_interval is not None:
        arr = np.asarray(spont_interval, dtype=float).reshape(-1)
        if arr.size == 2 and np.isfinite(arr).all() and arr[1] > arr[0]:
            return float(arr[0]), float(arr[1])

    eid = meta.get("eid")
    if not eid:
        raise RuntimeError("No EID found in the base cache metadata.")

    fallback = _load_spontaneous_intervals(one, eid)
    if fallback is None or fallback.size == 0:
        raise RuntimeError("Could not resolve a spontaneous interval for this PID.")
    return float(fallback[0, 0]), float(fallback[0, 1])


def _resolve_fit_interval(base_cache, one, spikes, fit_scope):
    scope = str(fit_scope).strip().lower()
    if scope == "spont":
        return _resolve_spont_interval(base_cache, one), "spont"
    if scope in {"session", "all", "full"}:
        meta = base_cache.get("meta", {}) if isinstance(base_cache, dict) else {}
        recording_length_s = meta.get("recording_length_s")
        try:
            recording_length_s = float(recording_length_s)
        except Exception:
            recording_length_s = np.nan
        if not np.isfinite(recording_length_s) or recording_length_s <= 0:
            recording_length_s = float(np.nanmax(np.asarray(spikes.times, dtype=float)))
        return (0.0, float(recording_length_s)), "session"
    raise ValueError(f"Unsupported FIT_SCOPE `{fit_scope}`. Use 'spont' or 'session'.")


def _resolve_device(device_config):
    device_text = str(device_config).strip().lower()
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def _select_plot_window(config_ppseq, fit_interval):
    fit_start, fit_end = map(float, fit_interval)
    plot_start = max(float(config_ppseq["PLOT_WINDOW_START_S"]), fit_start)
    plot_end = min(float(config_ppseq["PLOT_WINDOW_END_S"]), fit_end)
    if plot_end <= plot_start:
        plot_start = fit_start
        plot_end = min(fit_end, fit_start + 20.0)
    return float(plot_start), float(plot_end)


def _prepare_region_cluster_ids(df_units, region_name):
    region_df = df_units.loc[df_units["acronym"].astype(str) == str(region_name)].copy()
    cluster_ids = region_df["cluster_id"].to_numpy(dtype=int)
    return region_df.reset_index(drop=True), cluster_ids


def _bin_region_spikes(spikes, region_cluster_ids, interval, bin_size_s):
    interval_start, interval_end = map(float, interval)
    if interval_end <= interval_start:
        raise ValueError("Invalid fit interval.")

    region_cluster_ids = np.asarray(region_cluster_ids, dtype=int)
    num_units = int(region_cluster_ids.size)
    if num_units == 0:
        raise ValueError("Cannot bin spikes for an empty region.")

    spike_times_all = np.asarray(spikes.times, dtype=float)
    spike_clusters_all = np.asarray(spikes.clusters, dtype=int)

    time_mask = (spike_times_all >= interval_start) & (spike_times_all < interval_end)
    spike_times = spike_times_all[time_mask]
    spike_clusters = spike_clusters_all[time_mask]

    max_region_cluster = int(region_cluster_ids.max()) if region_cluster_ids.size > 0 else 0
    max_spike_cluster = int(spike_clusters.max()) if spike_clusters.size > 0 else 0
    max_cluster_id = int(max(max_region_cluster, max_spike_cluster))
    cluster_lookup = np.full(max_cluster_id + 1, -1, dtype=int)
    cluster_lookup[region_cluster_ids] = np.arange(num_units, dtype=int)

    row_idx = cluster_lookup[spike_clusters]
    region_mask = row_idx >= 0
    spike_times = spike_times[region_mask]
    spike_clusters = spike_clusters[region_mask]
    row_idx = row_idx[region_mask]

    num_bins = max(int(np.ceil((interval_end - interval_start) / float(bin_size_s))), 1)
    bin_edges = interval_start + np.arange(num_bins + 1, dtype=float) * float(bin_size_s)
    if bin_edges[-1] < interval_end:
        bin_edges = np.r_[bin_edges, interval_end]
    else:
        bin_edges[-1] = interval_end
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts = np.zeros((num_units, len(bin_edges) - 1), dtype=np.float32)
    if spike_times.size > 0:
        bin_idx = np.floor((spike_times - interval_start) / float(bin_size_s)).astype(int)
        bin_idx = np.clip(bin_idx, 0, counts.shape[1] - 1)
        np.add.at(counts, (row_idx, bin_idx), 1.0)

    return {
        "counts": counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "fit_spike_times": spike_times,
        "fit_spike_clusters": spike_clusters,
    }


def _extract_model_state(model, template_duration_bins):
    return {
        "base_rates": model.base_rates.detach().cpu().numpy().astype(np.float32),
        "template_scales": model.template_scales.detach().cpu().numpy().astype(np.float32),
        "template_offsets": model.template_offsets.detach().cpu().numpy().astype(np.float32),
        "template_widths": model.template_widths.detach().cpu().numpy().astype(np.float32),
        "template_duration_bins": int(template_duration_bins),
    }


def _initialize_ppseq_model(model, data_t, config_ppseq):
    initialization = str(config_ppseq["INITIALIZATION"]).strip().lower()
    sequence_frac = float(config_ppseq.get("INITIAL_SEQUENCE_FRAC", 0.2))
    concentration = float(config_ppseq.get("INITIAL_TEMPLATE_CONCENTRATION", 0.6))

    if initialization == "default":
        return model.initialize_default(
            data_t,
            sequence_frac=sequence_frac,
            concentration=concentration,
        )
    if initialization == "random":
        return model.initialize_random(
            data_t,
            sequence_frac=sequence_frac,
            concentration=concentration,
        )
    if initialization == "none":
        return model.initialize_none(data_t)
    raise ValueError(
        f"Unsupported INITIALIZATION `{config_ppseq['INITIALIZATION']}`. "
        "Use 'default', 'random', or 'none'."
    )


def _fit_ppseq_model(model, data_t, config_ppseq):
    amplitudes = _initialize_ppseq_model(model, data_t, config_ppseq)
    num_iter = int(config_ppseq["NUM_ITER"])
    fit_templates = bool(config_ppseq.get("FIT_TEMPLATES", True))
    fit_base_rates = bool(config_ppseq.get("FIT_BASE_RATES", True))

    lps = []
    for _ in progress_bar(range(num_iter)):
        amplitudes = model._update_amplitudes(data_t, amplitudes)
        if fit_base_rates:
            model._update_base_rates(data_t, amplitudes)
        if fit_templates:
            model._update_templates(data_t, amplitudes)
        lps.append(model.log_likelihood(data_t, amplitudes))

    lps = torch.stack(lps) if lps else torch.tensor([], device=model.device)
    return lps, amplitudes


def _fit_region_ppseq(counts, config_ppseq):
    if counts.ndim != 2:
        raise ValueError("PP-Seq counts must be a 2D neuron x time matrix.")

    device = _resolve_device(config_ppseq["DEVICE"])
    data_t = torch.as_tensor(counts, dtype=torch.float32, device=device)
    seed = int(config_ppseq["RANDOM_SEED"])
    force_k = config_ppseq.get("FORCE_K")
    if force_k is not None:
        candidate_ks = [int(force_k)]
    else:
        candidate_ks = [
            int(k)
            for k in config_ppseq["K_CANDIDATES"]
            if int(k) >= 1 and int(k) <= counts.shape[0]
        ]
    candidate_ks = [k for k in candidate_ks if int(k) >= 1 and int(k) <= counts.shape[0]]
    if not candidate_ks:
        raise RuntimeError("No valid K candidates remain for this region.")

    fit_records = {}
    best_record = None
    num_restarts = max(1, int(config_ppseq.get("NUM_RESTARTS", 1)))

    for k in candidate_ks:
        restart_records = []
        best_record_k = None
        for restart_idx in range(num_restarts):
            run_seed = seed + 1000 * int(k) + restart_idx
            torch.manual_seed(run_seed)
            model = PPSeq(
                num_templates=int(k),
                num_neurons=int(counts.shape[0]),
                template_duration=int(config_ppseq["TEMPLATE_DURATION_BINS"]),
                alpha_a0=float(config_ppseq["ALPHA_A0"]),
                beta_a0=float(config_ppseq["BETA_A0"]),
                alpha_b0=float(config_ppseq["ALPHA_B0"]),
                beta_b0=float(config_ppseq["BETA_B0"]),
                alpha_t0=float(config_ppseq["ALPHA_T0"]),
                beta_t0=float(config_ppseq["BETA_T0"]),
                device=device,
            )

            try:
                lps, amplitudes = _fit_ppseq_model(model, data_t, config_ppseq)
                lp_vals = lps.detach().cpu().numpy().astype(np.float64).tolist()
                final_ll = float(lp_vals[-1]) if lp_vals else np.nan
                record = {
                    "k": int(k),
                    "restart": int(restart_idx),
                    "seed": int(run_seed),
                    "final_log_likelihood": final_ll,
                    "log_likelihoods": lp_vals,
                    "model_state": _extract_model_state(
                        model,
                        config_ppseq["TEMPLATE_DURATION_BINS"],
                    ),
                    "amplitudes": amplitudes.detach().cpu().numpy().astype(np.float32),
                }
                restart_records.append(
                    {
                        "restart": int(restart_idx),
                        "seed": int(run_seed),
                        "final_log_likelihood": final_ll,
                        "log_likelihoods": lp_vals,
                    }
                )
                if best_record_k is None or final_ll > best_record_k["final_log_likelihood"]:
                    best_record_k = record
                if best_record is None or final_ll > best_record["final_log_likelihood"]:
                    best_record = record
            except Exception as exc:
                restart_records.append(
                    {
                        "restart": int(restart_idx),
                        "seed": int(run_seed),
                        "final_log_likelihood": np.nan,
                        "log_likelihoods": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

        fit_records[int(k)] = {
            "best_final_log_likelihood": (
                float(best_record_k["final_log_likelihood"]) if best_record_k is not None else np.nan
            ),
            "best_restart": (
                int(best_record_k["restart"]) if best_record_k is not None else None
            ),
            "restarts": restart_records,
        }

    if best_record is None:
        if force_k is not None and bool(config_ppseq.get("ALLOW_FORCE_K_FALLBACK", True)):
            fallback_config = dict(config_ppseq)
            fallback_config["FORCE_K"] = None
            fallback_result = _fit_region_ppseq(counts, fallback_config)
            fallback_result["forced_k_failed"] = True
            fallback_result["forced_k_attempt"] = {
                "forced_k": int(force_k),
                "fit_records": fit_records,
            }
            return fallback_result
        raise RuntimeError("PP-Seq fitting did not produce any successful candidate models.")

    best_record["log_likelihoods_by_k"] = fit_records
    return best_record


def _templates_from_state(model_state):
    scales = np.asarray(model_state["template_scales"], dtype=np.float32)
    offsets = np.asarray(model_state["template_offsets"], dtype=np.float32)
    widths = np.asarray(model_state["template_widths"], dtype=np.float32)
    widths = np.clip(widths, 1e-4, None)

    duration = int(model_state["template_duration_bins"])
    ds = np.arange(duration, dtype=np.float32)[None, None, :]
    gaussian = np.exp(-0.5 * ((ds - offsets[:, :, None]) / widths[:, :, None]) ** 2)
    gaussian /= (widths[:, :, None] * np.sqrt(2.0 * np.pi))
    gaussian_sum = np.clip(gaussian.sum(axis=2, keepdims=True), 1e-8, None)
    return gaussian / gaussian_sum * scales[:, :, None]


def _reconstruct_component_intensities(model_state, amplitudes):
    templates = _templates_from_state(model_state)
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    base_rates = np.asarray(model_state["base_rates"], dtype=np.float32)

    num_templates, num_units, _duration = templates.shape
    num_timebins = amplitudes.shape[1]
    template_contrib = np.zeros((num_templates, num_units, num_timebins), dtype=np.float32)

    for template_idx in range(num_templates):
        amp = amplitudes[template_idx]
        for unit_idx in range(num_units):
            template = templates[template_idx, unit_idx]
            template_contrib[template_idx, unit_idx] = np.convolve(
                amp,
                template,
                mode="full",
            )[:num_timebins]

    background = np.repeat(base_rates[:, None], num_timebins, axis=1)
    component_intensities = np.concatenate([background[None, ...], template_contrib], axis=0)
    dominant_component = np.argmax(component_intensities, axis=0).astype(np.int16)
    return component_intensities, dominant_component


def _compute_component_responsibilities(component_intensities):
    component_intensities = np.asarray(component_intensities, dtype=np.float32)
    total_intensity = np.clip(component_intensities.sum(axis=0, keepdims=True), 1e-8, None)
    return component_intensities / total_intensity


def _robust_zscore(values):
    values = np.asarray(values, dtype=np.float32)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if np.isfinite(mad) and mad > 0:
        return 0.6745 * (values - median) / mad
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - mean) / std


def _detect_peaks_1d(values, min_height, min_gap_bins):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.array([], dtype=int)

    candidates = []
    for idx in range(values.size):
        val = values[idx]
        if not np.isfinite(val) or val < float(min_height):
            continue
        left = values[idx - 1] if idx > 0 else -np.inf
        right = values[idx + 1] if idx < values.size - 1 else -np.inf
        if val >= left and val >= right:
            candidates.append(idx)

    if not candidates:
        return np.array([], dtype=int)

    selected = []
    for idx in sorted(candidates, key=lambda i: float(values[i]), reverse=True):
        if all(abs(idx - chosen) >= int(min_gap_bins) for chosen in selected):
            selected.append(int(idx))

    selected.sort()
    return np.asarray(selected, dtype=int)


def _compute_amplitude_zscores(amplitudes):
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    amp_zscores = np.zeros_like(amplitudes, dtype=np.float32)
    for template_idx in range(amplitudes.shape[0]):
        amp_zscores[template_idx] = _robust_zscore(amplitudes[template_idx])
    return amp_zscores


def _build_sequence_support_mask(amp_zscores, min_amp_z, template_duration_bins):
    amp_zscores = np.asarray(amp_zscores, dtype=np.float32)
    duration = max(1, int(template_duration_bins))
    support_mask = np.zeros_like(amp_zscores, dtype=bool)
    for template_idx in range(amp_zscores.shape[0]):
        active_bins = np.flatnonzero(amp_zscores[template_idx] >= float(min_amp_z))
        if active_bins.size == 0:
            continue
        for active_bin in active_bins:
            end_bin = min(active_bin + duration, amp_zscores.shape[1])
            support_mask[template_idx, active_bin:end_bin] = True
    return support_mask


def _summarize_window_sequence_support(region_result, config_ppseq, plot_window):
    fit_start, fit_end = map(float, region_result["fit_interval"])
    plot_start, plot_end = map(float, plot_window)
    plot_start = max(plot_start, fit_start)
    plot_end = min(plot_end, fit_end)
    if plot_end <= plot_start:
        return {}

    amplitudes = np.asarray(region_result["amplitudes"], dtype=np.float32)
    amp_zscores = _compute_amplitude_zscores(amplitudes)
    min_amp_z = float(config_ppseq.get("ASSIGNMENT_MIN_AMPLITUDE_ZSCORE", 1.0))
    duration = int(region_result["model_state"]["template_duration_bins"])
    support_mask = _build_sequence_support_mask(amp_zscores, min_amp_z, duration)
    min_gap_bins = max(1, duration // 2)

    bin_size_s = float(region_result["bin_size_s"])
    lo = max(0, int(np.floor((plot_start - fit_start) / bin_size_s)))
    hi = min(support_mask.shape[1], int(np.ceil((plot_end - fit_start) / bin_size_s)))
    if hi <= lo:
        return {}

    summary = {}
    for template_idx in range(support_mask.shape[0]):
        peaks = _detect_peaks_1d(
            amp_zscores[template_idx],
            min_height=min_amp_z,
            min_gap_bins=min_gap_bins,
        ).astype(np.int32)
        peaks_before = peaks[peaks < lo]
        peaks_after = peaks[peaks >= hi]
        summary[int(template_idx + 1)] = {
            "support_frac": float(support_mask[template_idx, lo:hi].mean()),
            "last_peak_before_s": (
                float(fit_start + peaks_before[-1] * bin_size_s) if peaks_before.size else None
            ),
            "next_peak_after_s": (
                float(fit_start + peaks_after[0] * bin_size_s) if peaks_after.size else None
            ),
        }
    return summary


def _sort_units_ppseq(model_state, region_cluster_ids):
    scales = np.asarray(model_state["template_scales"], dtype=np.float32)
    offsets = np.asarray(model_state["template_offsets"], dtype=np.float32)
    region_cluster_ids = np.asarray(region_cluster_ids, dtype=int)
    num_templates, num_units = scales.shape

    dominant_template = np.argmax(scales, axis=0)
    order = []
    group_bounds = []
    cursor = 0
    seen = np.zeros(num_units, dtype=bool)

    for template_idx in range(num_templates):
        members = np.flatnonzero(dominant_template == template_idx)
        if members.size == 0:
            continue
        members = members[np.argsort(offsets[template_idx, members], kind="stable")]
        order.extend(members.tolist())
        seen[members] = True
        group_bounds.append(
            {
                "template_idx": int(template_idx + 1),
                "start_rank": int(cursor),
                "end_rank": int(cursor + members.size - 1),
                "n_units": int(members.size),
            }
        )
        cursor += members.size

    remaining = np.flatnonzero(~seen)
    if remaining.size > 0:
        order.extend(remaining.tolist())

    order = np.asarray(order, dtype=int)
    ordered_cluster_ids = region_cluster_ids[order]
    return order, ordered_cluster_ids, group_bounds, dominant_template.astype(np.int16) + 1


def _component_color_map(config_ppseq, best_k):
    configured = dict(config_ppseq.get("PLOT_SEQUENCE_COLORS", {}))
    fallback = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "magenta",
    ]
    color_map = {0: str(configured.get(0, "gray"))}
    for component_idx in range(1, int(best_k) + 1):
        color_map[component_idx] = str(
            configured.get(component_idx, fallback[(component_idx - 1) % len(fallback)])
        )
    return color_map


def _build_spike_lookup(cluster_ids, values, fill_value):
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if cluster_ids.size == 0:
        return np.array([], dtype=type(fill_value))
    max_cluster_id = int(cluster_ids.max())
    if np.issubdtype(np.asarray(values).dtype, np.integer):
        lookup = np.full(max_cluster_id + 1, int(fill_value), dtype=int)
    else:
        lookup = np.full(max_cluster_id + 1, fill_value, dtype=np.float32)
    lookup[cluster_ids] = values
    return lookup


def _build_region_raster_figure(region_name, region_result, raw_bundle, base_cache, config_ppseq):
    if region_result.get("status") != "fit_ok":
        raise ValueError(f"Region `{region_name}` does not contain a successful PP-Seq fit.")

    spikes = raw_bundle["spikes"]
    session_loader = raw_bundle["session_loader"]
    fit_start, fit_end = map(float, region_result["fit_interval"])
    plot_start, plot_end = map(float, region_result["plot_window"])
    plot_start = max(plot_start, fit_start)
    plot_end = min(plot_end, fit_end)
    if plot_end <= plot_start:
        raise ValueError("Plot window does not overlap the PP-Seq fit interval.")

    fit_cluster_ids = np.asarray(region_result["cluster_ids"], dtype=int)
    ordered_cluster_ids = np.asarray(region_result["unit_order_ppseq"], dtype=int)
    best_k = int(region_result["best_k"])

    model_state = region_result["model_state"]
    amplitudes = np.asarray(region_result["amplitudes"], dtype=np.float32)
    component_intensities, dominant_component = _reconstruct_component_intensities(
        model_state,
        amplitudes,
    )
    responsibilities = _compute_component_responsibilities(component_intensities)
    amp_zscores = _compute_amplitude_zscores(amplitudes)
    min_amp_z = float(config_ppseq.get("ASSIGNMENT_MIN_AMPLITUDE_ZSCORE", 1.0))
    template_support = _build_sequence_support_mask(
        amp_zscores,
        min_amp_z,
        model_state["template_duration_bins"],
    )
    min_gap_bins = max(1, int(model_state["template_duration_bins"]) // 2)
    peak_bins_by_template = {}
    for template_idx in range(amp_zscores.shape[0]):
        peak_bins_by_template[int(template_idx + 1)] = _detect_peaks_1d(
            amp_zscores[template_idx],
            min_height=min_amp_z,
            min_gap_bins=min_gap_bins,
        ).astype(np.int32)
    row_lookup = _build_spike_lookup(
        fit_cluster_ids,
        np.arange(fit_cluster_ids.size, dtype=int),
        fill_value=-1,
    )
    display_lookup = _build_spike_lookup(
        ordered_cluster_ids,
        np.arange(ordered_cluster_ids.size, dtype=int),
        fill_value=-1,
    )

    spike_times = np.asarray(spikes.times, dtype=float)
    spike_clusters = np.asarray(spikes.clusters, dtype=int)
    in_window = (spike_times >= plot_start) & (spike_times < plot_end)
    spike_times = spike_times[in_window]
    spike_clusters = spike_clusters[in_window]

    valid_cluster_mask = spike_clusters <= (len(row_lookup) - 1)
    spike_times = spike_times[valid_cluster_mask]
    spike_clusters = spike_clusters[valid_cluster_mask]
    fit_rows = row_lookup[spike_clusters]
    display_rows = display_lookup[spike_clusters]
    valid_fit = (fit_rows >= 0) & (display_rows >= 0)
    spike_times = spike_times[valid_fit]
    spike_clusters = spike_clusters[valid_fit]
    fit_rows = fit_rows[valid_fit]
    display_rows = display_rows[valid_fit]

    bin_size_s = float(region_result["bin_size_s"])
    num_bins = dominant_component.shape[1]
    bin_idx = np.floor((spike_times - fit_start) / bin_size_s).astype(int)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    raw_component_labels = dominant_component[fit_rows, bin_idx].astype(int)
    winning_responsibility = responsibilities[raw_component_labels, fit_rows, bin_idx].astype(np.float32)
    winning_amp_z = np.full_like(winning_responsibility, np.nan, dtype=np.float32)
    winning_support = np.zeros_like(raw_component_labels, dtype=bool)
    seq_mask_all = raw_component_labels > 0
    if np.any(seq_mask_all):
        winning_amp_z[seq_mask_all] = amp_zscores[
            raw_component_labels[seq_mask_all] - 1,
            bin_idx[seq_mask_all],
        ]
        winning_support[seq_mask_all] = template_support[
            raw_component_labels[seq_mask_all] - 1,
            bin_idx[seq_mask_all],
        ]
    component_labels = np.zeros_like(raw_component_labels)
    min_sequence_resp = float(config_ppseq.get("ASSIGNMENT_MIN_SEQUENCE_RESP", 0.7))
    for component_idx in range(1, best_k + 1):
        raw_mask = raw_component_labels == component_idx
        if not np.any(raw_mask):
            continue
        strong_mask = winning_responsibility >= min_sequence_resp
        support_mask = winning_support
        keep_mask = raw_mask & strong_mask & support_mask
        component_labels[keep_mask] = component_idx

    color_map = _component_color_map(config_ppseq, best_k)
    config_plot = dict(base_cache.get("config_plot", {}))
    config_plot["PLOTLY_TEMPLATE"] = config_plot.get("PLOTLY_TEMPLATE", "plotly_white")
    region_color_map = plotting_utils._region_color_map([region_name])
    region_fill = plotting_utils._color_to_rgba(
        region_color_map.get(region_name, "#1f77b4"),
        alpha=0.18,
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.62, 0.20, 0.18],
        subplot_titles=("", "Population PSTH", "Whisking"),
    )

    present_components = set(np.unique(component_labels).tolist()) if component_labels.size > 0 else set()
    if best_k >= 0:
        present_components.add(0)
    fig.add_shape(
        type="rect",
        x0=plot_start,
        x1=plot_end,
        y0=-0.5,
        y1=len(ordered_cluster_ids) - 0.5,
        line=dict(width=0),
        fillcolor=region_fill,
        layer="below",
        row=1,
        col=1,
    )
    for component_idx in sorted(present_components):
        comp_mask = component_labels == component_idx
        comp_name = "Background" if component_idx == 0 else f"Sequence {component_idx}"
        fig.add_trace(
            go.Scattergl(
                x=spike_times[comp_mask] if np.any(comp_mask) else [None],
                y=display_rows[comp_mask] if np.any(comp_mask) else [None],
                mode="markers",
                marker=dict(
                    color=color_map.get(component_idx, "gray"),
                    size=3,
                    symbol="line-ns-open",
                ),
                customdata=(
                    np.column_stack(
                        [
                            spike_clusters[comp_mask],
                            component_labels[comp_mask],
                            raw_component_labels[comp_mask],
                            winning_responsibility[comp_mask],
                            winning_amp_z[comp_mask],
                            winning_support[comp_mask].astype(int),
                        ]
                    )
                    if np.any(comp_mask)
                    else None
                ),
                hovertemplate=(
                    "Time: %{x:.3f}s<br>"
                    "Cluster: %{customdata[0]}<br>"
                    "Shown assignment: %{customdata[1]}<br>"
                    "Raw winning component: %{customdata[2]}<br>"
                    "Winning responsibility: %{customdata[3]:.3f}<br>"
                    "Winning amplitude z: %{customdata[4]:.3f}<br>"
                    "Inside event support: %{customdata[5]}<extra></extra>"
                ),
                name=comp_name,
            )
            ,
            row=1,
            col=1,
        )

    group_bounds = list(region_result.get("template_group_bounds", []))
    for group in group_bounds[1:]:
        y_line = float(group["start_rank"]) - 0.5
        fig.add_hline(y=y_line, line=dict(color="black", width=1), row=1, col=1)

    for group in group_bounds:
        y_mid = 0.5 * (float(group["start_rank"]) + float(group["end_rank"]))
        fig.add_annotation(
            x=plot_end,
            y=y_mid,
            xanchor="left",
            yanchor="middle",
            text=f"Seq {group['template_idx']}",
            showarrow=False,
            font=dict(size=10, color=color_map.get(int(group["template_idx"]), "gray")),
            xshift=8,
            row=1,
            col=1,
        )

    for template_idx, peak_bins in peak_bins_by_template.items():
        if peak_bins.size == 0:
            continue
        peak_times = fit_start + peak_bins.astype(np.float32) * bin_size_s
        for peak_time in peak_times:
            if plot_start <= float(peak_time) <= plot_end:
                fig.add_vline(
                    x=float(peak_time),
                    line=dict(
                        color=color_map.get(int(template_idx), "gray"),
                        width=1,
                        dash="dot",
                    ),
                    row=1,
                    col=1,
                )

    fig.add_annotation(
        x=plot_end,
        y=0.5 * max(len(ordered_cluster_ids) - 1, 0),
        xanchor="left",
        yanchor="middle",
        text=region_name,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xshift=10,
        row=1,
        col=1,
    )

    pop_bin_size = float(config_plot.get("POP_BIN_SIZE", 0.005))
    smooth_window_s = 0.05
    smooth_bins = max(1, int(round(smooth_window_s / pop_bin_size))) if pop_bin_size > 0 else 1
    psth_bins = np.arange(plot_start, plot_end + pop_bin_size, pop_bin_size)
    if psth_bins.size < 2:
        psth_bins = np.array([plot_start, plot_end], dtype=float)
    psth_centers = 0.5 * (psth_bins[:-1] + psth_bins[1:])
    psth_counts, _ = np.histogram(spike_times, bins=psth_bins)
    psth_rate = psth_counts / (max(len(ordered_cluster_ids), 1) * pop_bin_size)
    psth_rate_smoothed = plotting_utils._moving_mean(psth_rate, smooth_bins)
    fig.add_trace(
        go.Scatter(
            x=psth_centers,
            y=psth_rate_smoothed,
            mode="lines",
            line=dict(color=region_color_map.get(region_name, "#1f77b4"), width=2),
            name=f"{region_name} PSTH",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    whisk_inputs = build_whisk_raster_overlay_inputs(
        df_wh=base_cache.get("df_wh"),
        wh_detect=base_cache.get("wh_detect"),
        wh_event_base=base_cache.get("wh_event_base"),
        wh_events_by_period=base_cache.get("wh_events_by_period"),
    )
    motion_mean_df = whisk_inputs.get("motion_mean_df")
    mean_t, mean_wh = plotting_utils._extract_precomputed_motion_mean_series(
        motion_mean_df,
        plot_start,
        plot_end,
        t_offset=0.0,
    )
    if mean_t.size > 0:
        fig.add_trace(
            go.Scatter(
                x=mean_t,
                y=mean_wh,
                mode="lines",
                line=dict(color="#ff7f0e"),
                name="Mean whisk",
                showlegend=True,
            ),
            row=3,
            col=1,
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="y3",
            text="Whisking trace not available",
            showarrow=False,
            row=3,
            col=1,
        )

    trials = plotting_utils._get_session_field(session_loader, "trials")
    if trials is not None:
        event_style_map = {
            "stimOn_times": ("Stim On", "blue", None),
            "firstMovement_times": ("First Move", "green", None),
            "response_times": ("Response", "purple", None),
            "feedback_times": ("Feedback", "red", None),
        }
        for event_name, (label, color, dash) in event_style_map.items():
            if hasattr(trials, "keys") and event_name in trials.keys():
                plotting_utils._add_event_vlines(
                    fig,
                    np.asarray(trials[event_name], dtype=float),
                    label,
                    color,
                    plot_start,
                    plot_end,
                    n_rows=3,
                    dash=dash,
                )

    extra_event_times = whisk_inputs.get("extra_event_times")
    extra_event_styles = whisk_inputs.get("extra_event_styles")
    if extra_event_times:
        for key, times in dict(extra_event_times).items():
            style = (extra_event_styles or {}).get(key, (str(key), "#666666", None))
            label = style[0] if len(style) > 0 else str(key)
            color = style[1] if len(style) > 1 else "#666666"
            dash = style[2] if len(style) > 2 else None
            plotting_utils._add_event_vlines(
                fig,
                times,
                label,
                color,
                plot_start,
                plot_end,
                n_rows=3,
                dash=dash,
            )

    extra_event_spans = whisk_inputs.get("extra_event_spans")
    extra_event_span_styles = whisk_inputs.get("extra_event_span_styles")
    if extra_event_spans:
        for key, spans in dict(extra_event_spans).items():
            style = (extra_event_span_styles or {}).get(key, {})
            plotting_utils._add_event_spans(
                fig,
                spans,
                style.get("label", str(key)),
                style.get("color", "#666666"),
                plot_start,
                plot_end,
                row=3,
                col=1,
                alpha=float(style.get("alpha", 0.18)),
            )

    fig.update_layout(
        template=config_plot["PLOTLY_TEMPLATE"],
        title=(
            f"PP-Seq Raster | PID {CONFIG_PPSEQ['PID']} | Region {region_name} | "
            f"K={best_k}"
        ),
        legend_title="Assignment",
        height=max(720, 320 + 8 * len(ordered_cluster_ids)),
    )
    fig.update_xaxes(range=[plot_start, plot_end], row=3, col=1, title_text="time (s)")
    fig.update_yaxes(
        title_text=f"Units (n={len(ordered_cluster_ids)})",
        row=1,
        col=1,
        showticklabels=False,
        range=[-0.5, len(ordered_cluster_ids) - 0.5],
    )
    fig.update_yaxes(title_text="Avg PSTH (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Normalized whisk signal", row=3, col=1)
    return fig


def _fit_single_region(region_name, region_df, raw_bundle, fit_interval, config_ppseq):
    cluster_ids = region_df["cluster_id"].to_numpy(dtype=int)
    if cluster_ids.size < int(config_ppseq["MIN_REGION_NEURONS"]):
        return {
            "status": "skipped_small_region",
            "n_units": int(cluster_ids.size),
            "cluster_ids": cluster_ids,
            "best_k": None,
            "fit_interval": tuple(map(float, fit_interval)),
        }

    binned = _bin_region_spikes(
        raw_bundle["spikes"],
        cluster_ids,
        fit_interval,
        config_ppseq["BIN_SIZE_S"],
    )
    counts = binned["counts"]
    if counts.size == 0 or float(counts.sum()) <= 0:
        return {
            "status": "skipped_no_spikes",
            "n_units": int(cluster_ids.size),
            "cluster_ids": cluster_ids,
            "best_k": None,
            "fit_interval": tuple(map(float, fit_interval)),
        }

    best_fit = _fit_region_ppseq(counts, config_ppseq)
    order_idx, ordered_cluster_ids, group_bounds, template_membership = _sort_units_ppseq(
        best_fit["model_state"],
        cluster_ids,
    )
    plot_window = _select_plot_window(config_ppseq, fit_interval)

    return {
        "status": "fit_ok",
        "n_units": int(cluster_ids.size),
        "cluster_ids": cluster_ids.astype(np.int32),
        "unit_order_ppseq": ordered_cluster_ids.astype(np.int32),
        "best_k": int(best_fit["k"]),
        "best_restart": int(best_fit.get("restart", 0)),
        "bin_size_s": float(config_ppseq["BIN_SIZE_S"]),
        "fit_interval": tuple(map(float, fit_interval)),
        "time_bin_edges": binned["bin_edges"].astype(np.float32),
        "time_bin_centers": binned["bin_centers"].astype(np.float32),
        "model_state": best_fit["model_state"],
        "amplitudes": best_fit["amplitudes"],
        "log_likelihoods_by_k": best_fit["log_likelihoods_by_k"],
        "plot_window": tuple(map(float, plot_window)),
        "template_group_bounds": group_bounds,
        "template_membership": template_membership.astype(np.int16),
        "fit_index_order": order_idx.astype(np.int32),
        "n_fit_spikes": int(binned["fit_spike_times"].size),
        "final_log_likelihood": float(best_fit["final_log_likelihood"]),
    }


def _save_ppseq_cache(pid, payload, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pid}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path


# %% Load data
_set_plotly_renderer(CONFIG_PPSEQ.get("PLOTLY_RENDERER"))

pid = str(CONFIG_PPSEQ["PID"]).strip()

print(f"Loading base cache for PID: {pid}")
base_cache = _load_base_cache(pid)

print(f"Loading raw session for PID: {pid}")
raw_bundle = _load_raw_session(pid)
spikes = raw_bundle["spikes"]
clusters = raw_bundle["clusters"]
session_loader = raw_bundle["session_loader"]
one = raw_bundle["one"]

fit_interval, fit_scope_resolved = _resolve_fit_interval(
    base_cache,
    one,
    spikes,
    CONFIG_PPSEQ["FIT_SCOPE"],
)
print(
    f"Resolved {fit_scope_resolved} fit interval: "
    f"{fit_interval[0]:.4f} to {fit_interval[1]:.4f} s"
)

df_units = _prepare_units_table(base_cache, clusters, CONFIG_PPSEQ["LABEL_MIN"])
print(
    f"Units after LABEL_MIN>={CONFIG_PPSEQ['LABEL_MIN']}: "
    f"{len(df_units)}"
)
target_regions = _resolve_target_regions(df_units, CONFIG_PPSEQ["REGION"])
print(f"Target region list: {target_regions}")


# %% Fit PP-Seq per region
region_results = {}
summary_rows = []

for region_name in target_regions:
    print("")
    print(f"Fitting region: {region_name}")
    region_df, region_cluster_ids = _prepare_region_cluster_ids(df_units, region_name)
    print(f"Eligible units in {region_name}: {len(region_cluster_ids)}")

    try:
        region_result = _fit_single_region(
            region_name,
            region_df,
            raw_bundle,
            fit_interval,
            CONFIG_PPSEQ,
        )
    except Exception as exc:
        region_result = {
            "status": "fit_failed",
            "n_units": int(len(region_cluster_ids)),
            "cluster_ids": region_cluster_ids.astype(np.int32),
            "best_k": None,
            "fit_interval": tuple(map(float, fit_interval)),
            "error": f"{type(exc).__name__}: {exc}",
        }

    region_results[region_name] = region_result
    if region_result.get("status") == "fit_ok":
        ll_by_k = region_result.get("log_likelihoods_by_k", {})
        print("PP-Seq fit summary by K:")
        for k, info in sorted(ll_by_k.items()):
            best_ll = info.get("best_final_log_likelihood")
            best_restart = info.get("best_restart")
            print(f"  K={k}: best_ll={best_ll}, best_restart={best_restart}")
        print(
            f"Chosen K={region_result.get('best_k')} "
            f"(restart={region_result.get('best_restart')})"
        )
    summary_rows.append(
        {
            "region": region_name,
            "status": region_result.get("status"),
            "n_units": int(region_result.get("n_units", 0) or 0),
            "best_k": region_result.get("best_k"),
            "n_fit_spikes": region_result.get("n_fit_spikes"),
            "final_log_likelihood": region_result.get("final_log_likelihood"),
            "plot_window_start_s": (
                region_result.get("plot_window", (np.nan, np.nan))[0]
                if region_result.get("plot_window") is not None
                else np.nan
            ),
            "plot_window_end_s": (
                region_result.get("plot_window", (np.nan, np.nan))[1]
                if region_result.get("plot_window") is not None
                else np.nan
            ),
            "error": region_result.get("error"),
        }
    )

region_summary_df = pd.DataFrame(summary_rows)
print("")
print("Region summary:")
print(region_summary_df.to_string(index=False))


# %% Save cache
ppseq_cache = {
    "meta": dict(base_cache.get("meta", {})),
    "config_ppseq": dict(CONFIG_PPSEQ),
    "region_summary_df": region_summary_df.copy(),
    "region_results": region_results,
}

cache_path = None
if bool(CONFIG_PPSEQ.get("SAVE_CACHE", True)):
    cache_path = _save_ppseq_cache(pid, ppseq_cache, CONFIG_PPSEQ["CACHE_DIR"])
    print("")
    print(f"Saved PP-Seq cache to: {cache_path}")


# %% Plot results
# PLOT_RESULTS_T_START = 3910  
# PLOT_RESULTS_T_END = 3950  

PLOT_RESULTS_T_START = 4005.5  
PLOT_RESULTS_T_END = 4007  

PLOT_RESULTS_ASSIGNMENT_MIN_SEQUENCE_RESP = float(
    CONFIG_PPSEQ["ASSIGNMENT_MIN_SEQUENCE_RESP"]
)
PLOT_RESULTS_ASSIGNMENT_MIN_AMPLITUDE_ZSCORE = float(
    CONFIG_PPSEQ["ASSIGNMENT_MIN_AMPLITUDE_ZSCORE"]
)

region_figures = {}
for region_name in target_regions:
    region_result = region_results.get(region_name, {})
    if region_result.get("status") != "fit_ok":
        print(f"Skipping raster plot for {region_name}: status={region_result.get('status')}")
        continue
    region_result_plot = dict(region_result)
    region_result_plot["plot_window"] = (
        float(PLOT_RESULTS_T_START),
        float(PLOT_RESULTS_T_END),
    )
    config_ppseq_plot = dict(CONFIG_PPSEQ)
    config_ppseq_plot["ASSIGNMENT_MIN_SEQUENCE_RESP"] = float(
        PLOT_RESULTS_ASSIGNMENT_MIN_SEQUENCE_RESP
    )
    config_ppseq_plot["ASSIGNMENT_MIN_AMPLITUDE_ZSCORE"] = float(
        PLOT_RESULTS_ASSIGNMENT_MIN_AMPLITUDE_ZSCORE
    )
    support_summary = _summarize_window_sequence_support(
        region_result_plot,
        config_ppseq_plot,
        region_result_plot["plot_window"],
    )
    if support_summary and all(
        float(item.get("support_frac", 0.0)) <= 0.0 for item in support_summary.values()
    ):
        print(
            f"Requested plot window {region_result_plot['plot_window'][0]:.3f}-"
            f"{region_result_plot['plot_window'][1]:.3f}s contains no detected PP-Seq "
            f"support in {region_name}."
        )
        for template_idx, item in sorted(support_summary.items()):
            print(
                f"  Template {template_idx}: last peak before="
                f"{item['last_peak_before_s']}, next peak after={item['next_peak_after_s']}"
            )
    fig = _build_region_raster_figure(
        region_name,
        region_result_plot,
        raw_bundle,
        base_cache,
        config_ppseq_plot,
    )
    region_figures[region_name] = fig
    show_fig(fig, renderer=CONFIG_PPSEQ.get("PLOTLY_RENDERER"))


# %% Final objects
print("")
print("Finished PP-Seq run.")
print(f"PID: {pid}")
print(f"Successful regions: {sorted(region_figures.keys())}")
if cache_path is not None:
    print(f"PP-Seq cache path: {cache_path}")
