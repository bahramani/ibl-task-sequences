# %% Imports
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))

HELPER_SOURCE_PATH = BASE_PATH / "scripts" / "22_ppseq_ibl.py"


# %% Config
CONFIG_PPSEQ_BATCH = {
    "PID_LIST": [
        "c9664185-d3fd-4e0e-89cf-77c402038938",
        "f967a527-257f-404a-871d-b91575dca3b4",
        "0dfc64c5-80bc-43b6-a7c4-2fb6e638e9f3",
    ],
    "LABEL_MIN": 0.6,
    "FIT_SCOPE": "session",
    "MIN_REGION_NEURONS": 10,
    "BIN_SIZE_S": 0.01,
    "TEMPLATE_DURATION_BINS": 25,
    "K_CANDIDATES": [2, 3, 4, 5, 6],
    "FORCE_K": None,
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
    "PLOT_WINDOW_START_S": 0.0,
    "PLOT_WINDOW_END_S": 20.0,
    "SAVE_PID_CACHE": True,
    "SAVE_BATCH_CACHE": True,
    "SAVE_SUMMARY_CSV": True,
    "CACHE_DIR": BASE_PATH / "data" / "ppseq_batch_cache",
    "RANDOM_SEED": 0,
}


# %% Load PP-Seq helpers from 22
def _load_22_helper_namespace():
    if not HELPER_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing helper source: {HELPER_SOURCE_PATH}")

    source = HELPER_SOURCE_PATH.read_text(encoding="utf-8")
    marker = "\n# %% Load data\n"
    if marker not in source:
        raise RuntimeError(
            f"Could not find the load-data marker in {HELPER_SOURCE_PATH}. "
            "Refactor 22_ppseq_ibl.py or update this loader."
        )

    prefix = source.split(marker, 1)[0]
    namespace = {"__file__": str(HELPER_SOURCE_PATH)}
    exec(compile(prefix, str(HELPER_SOURCE_PATH), "exec"), namespace)
    return namespace


_HELPERS = _load_22_helper_namespace()
_load_base_cache = _HELPERS["_load_base_cache"]
_load_raw_session = _HELPERS["_load_raw_session"]
_prepare_units_table = _HELPERS["_prepare_units_table"]
_resolve_fit_interval = _HELPERS["_resolve_fit_interval"]
_prepare_region_cluster_ids = _HELPERS["_prepare_region_cluster_ids"]
_fit_single_region = _HELPERS["_fit_single_region"]


# %% Batch helpers
def _resolve_target_regions_batch(df_units, min_region_neurons):
    if df_units.empty:
        raise RuntimeError("No eligible units remain after label filtering.")

    region_counts = (
        df_units.groupby("acronym", dropna=False)["cluster_id"]
        .nunique()
        .rename("n_units")
        .reset_index()
        .sort_values(["n_units", "acronym"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    target_regions = (
        region_counts.loc[
            region_counts["n_units"] >= int(min_region_neurons),
            "acronym",
        ]
        .astype(str)
        .tolist()
    )
    return target_regions, region_counts


def _save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def _run_pid_batch(pid, config_ppseq_batch):
    print("")
    print("=" * 80)
    print(f"Running PP-Seq batch fit for PID: {pid}")

    base_cache = _load_base_cache(pid)
    raw_bundle = _load_raw_session(pid)
    spikes = raw_bundle["spikes"]
    clusters = raw_bundle["clusters"]
    one = raw_bundle["one"]

    fit_interval, fit_scope_resolved = _resolve_fit_interval(
        base_cache,
        one,
        spikes,
        config_ppseq_batch["FIT_SCOPE"],
    )
    print(
        f"Resolved {fit_scope_resolved} fit interval: "
        f"{fit_interval[0]:.4f} to {fit_interval[1]:.4f} s"
    )

    df_units = _prepare_units_table(base_cache, clusters, config_ppseq_batch["LABEL_MIN"])
    print(
        f"Units after LABEL_MIN>={config_ppseq_batch['LABEL_MIN']}: "
        f"{len(df_units)}"
    )

    target_regions, region_counts_df = _resolve_target_regions_batch(
        df_units,
        config_ppseq_batch["MIN_REGION_NEURONS"],
    )
    print(f"Target region list: {target_regions}")

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
                config_ppseq_batch,
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
                "pid": pid,
                "region": region_name,
                "status": region_result.get("status"),
                "n_units": int(region_result.get("n_units", 0) or 0),
                "best_k": region_result.get("best_k"),
                "best_restart": region_result.get("best_restart"),
                "n_fit_spikes": region_result.get("n_fit_spikes"),
                "final_log_likelihood": region_result.get("final_log_likelihood"),
                "fit_interval_start_s": (
                    region_result.get("fit_interval", (np.nan, np.nan))[0]
                    if region_result.get("fit_interval") is not None
                    else np.nan
                ),
                "fit_interval_end_s": (
                    region_result.get("fit_interval", (np.nan, np.nan))[1]
                    if region_result.get("fit_interval") is not None
                    else np.nan
                ),
                "forced_k_failed": bool(region_result.get("forced_k_failed", False)),
                "error": region_result.get("error"),
            }
        )

    region_summary_df = pd.DataFrame(summary_rows)
    print("")
    print("Region summary:")
    print(region_summary_df.to_string(index=False))

    payload = {
        "meta": dict(base_cache.get("meta", {})),
        "config_ppseq_batch": dict(config_ppseq_batch),
        "helper_source_path": str(HELPER_SOURCE_PATH),
        "fit_scope_resolved": fit_scope_resolved,
        "fit_interval": tuple(map(float, fit_interval)),
        "eligible_units_df": df_units.copy(),
        "region_counts_df": region_counts_df.copy(),
        "region_summary_df": region_summary_df.copy(),
        "region_results": region_results,
    }
    return payload


# %% Run batch PP-Seq
cache_dir = Path(CONFIG_PPSEQ_BATCH["CACHE_DIR"])
cache_dir.mkdir(parents=True, exist_ok=True)

pid_payloads = {}
pid_cache_paths = {}
pid_summary_rows = []
combined_region_summaries = []

for pid in CONFIG_PPSEQ_BATCH["PID_LIST"]:
    try:
        pid_payload = _run_pid_batch(pid, CONFIG_PPSEQ_BATCH)
        pid_payloads[pid] = pid_payload
        region_summary_df = pid_payload["region_summary_df"].copy()
        combined_region_summaries.append(region_summary_df)

        fit_ok_mask = region_summary_df["status"].astype(str) == "fit_ok"
        fit_failed_mask = region_summary_df["status"].astype(str) == "fit_failed"
        pid_summary_rows.append(
            {
                "pid": pid,
                "status": "ok",
                "n_regions_total": int(len(region_summary_df)),
                "n_regions_fit_ok": int(fit_ok_mask.sum()),
                "n_regions_failed": int(fit_failed_mask.sum()),
                "n_regions_skipped": int((~fit_ok_mask & ~fit_failed_mask).sum()),
            }
        )

        if bool(CONFIG_PPSEQ_BATCH.get("SAVE_PID_CACHE", True)):
            pid_cache_path = cache_dir / f"{pid}.pkl"
            _save_pickle(pid_cache_path, pid_payload)
            pid_cache_paths[pid] = pid_cache_path
            print("")
            print(f"Saved PID PP-Seq cache to: {pid_cache_path}")
    except Exception as exc:
        pid_summary_rows.append(
            {
                "pid": pid,
                "status": "pid_failed",
                "n_regions_total": 0,
                "n_regions_fit_ok": 0,
                "n_regions_failed": 0,
                "n_regions_skipped": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print("")
        print(f"PID failed: {pid}")
        print(f"{type(exc).__name__}: {exc}")

pid_summary_df = pd.DataFrame(pid_summary_rows)
combined_region_summary_df = (
    pd.concat(combined_region_summaries, ignore_index=True)
    if combined_region_summaries
    else pd.DataFrame()
)


# %% Save batch summary
batch_summary_payload = {
    "config_ppseq_batch": dict(CONFIG_PPSEQ_BATCH),
    "helper_source_path": str(HELPER_SOURCE_PATH),
    "pid_summary_df": pid_summary_df.copy(),
    "combined_region_summary_df": combined_region_summary_df.copy(),
    "pid_cache_paths": {pid: str(path) for pid, path in pid_cache_paths.items()},
}

batch_summary_path = None
batch_summary_csv_path = None
if bool(CONFIG_PPSEQ_BATCH.get("SAVE_BATCH_CACHE", True)):
    batch_summary_path = _save_pickle(cache_dir / "ppseq_batch_summary.pkl", batch_summary_payload)
    print("")
    print(f"Saved batch summary cache to: {batch_summary_path}")

if bool(CONFIG_PPSEQ_BATCH.get("SAVE_SUMMARY_CSV", True)):
    batch_summary_csv_path = cache_dir / "ppseq_batch_region_summary.csv"
    combined_region_summary_df.to_csv(batch_summary_csv_path, index=False)
    print(f"Saved batch region summary CSV to: {batch_summary_csv_path}")


# %% Final objects
print("")
print("Batch PP-Seq run finished.")
print("PID summary:")
print(pid_summary_df.to_string(index=False))
