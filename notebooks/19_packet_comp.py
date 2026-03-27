# %%
from pathlib import Path
import sys

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))

from utils.io import setup_paths, init_one
from utils.packet_dashboard import (
    BASE_CACHE_DIR,
    PACKET_CACHE_DIR,
    DEFAULT_PACKET_CONFIG,
    compute_packet_dashboard_cache,
    list_available_pids,
    load_base_cache,
    load_packet_cache,
)


# %% Compute Config
# Follow the same selection style as 03_calc_dashboard.py.
COMPUTE_ALL = False  # If True, ignore PIDS/SUBJECT/REGIONS and process all insertions for TAG.
PIDS = [
    "c9664185-d3fd-4e0e-89cf-77c402038938",
    # "49c2ea3d-2b50-4e8a-b124-9e190960784e",
]
SUBJECT = None  # "CSH_ZAD_029"
REGIONS = None  # ["VISp", "MOp", "PO", "CA1", "AUDp", "CP", "VPM"]
TAG = None  # "2025_Q3_IBL_et_al_BWM"

FORCE_RECOMPUTE = True
PACKET_CONFIG = dict(DEFAULT_PACKET_CONFIG)


def _tag_search_kwargs(tag):
    # IBL docs: tags are attached to datasets, and insertion queries should use
    # django='datasets__tags__name,<tag>' when restricting by a data-release tag.
    if not tag:
        return {}
    return {"django": f"datasets__tags__name,{tag}"}


# %%
def _get_pids_for_subject(one, subject, tag):
    return list(
        dict.fromkeys(
            one.search_insertions(
                subject=subject,
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    )


def _get_pids_for_regions(one, regions, tag):
    if not regions:
        return []
    all_pids = []
    for region in regions:
        all_pids.extend(
            one.search_insertions(
                atlas_acronym=region,
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    return list(dict.fromkeys(all_pids))


def _get_all_tag_pids(one, tag):
    return list(
        dict.fromkeys(
            one.search_insertions(
                query_type="remote",
                **_tag_search_kwargs(tag),
            )
        )
    )


def _local_subject_filter(pid_list, subject):
    subject = str(subject).strip()
    keep = []
    for pid in pid_list:
        try:
            base_cache = load_base_cache(pid, BASE_CACHE_DIR)
        except Exception:
            continue
        if str(base_cache.get("meta", {}).get("subject", "")).strip() == subject:
            keep.append(pid)
    return keep


def _local_region_filter(pid_list, regions):
    if not regions:
        return []
    region_list = [str(region).strip() for region in regions]
    keep = []
    for pid in pid_list:
        try:
            base_cache = load_base_cache(pid, BASE_CACHE_DIR)
        except Exception:
            continue
        acronyms = base_cache.get("cluster_acronyms_plot")
        if acronyms is None:
            continue
        acronyms = [str(acronym) for acronym in acronyms]
        if any(any(acronym.startswith(region) for acronym in acronyms) for region in region_list):
            keep.append(pid)
    return keep


def _resolve_pids_to_process():
    pid_list = list_available_pids(BASE_CACHE_DIR)
    if not pid_list:
        raise RuntimeError(f"No base dashboard caches found in {BASE_CACHE_DIR}.")

    if COMPUTE_ALL:
        remote_mode = "all"
    elif PIDS:
        return list(dict.fromkeys([str(pid).strip() for pid in PIDS]))
    elif SUBJECT is not None:
        remote_mode = "subject"
    else:
        remote_mode = "regions"

    path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    try:
        one = init_one(ibl_cache)
        if remote_mode == "all":
            pids = _get_all_tag_pids(one, TAG)
        elif remote_mode == "subject":
            pids = _get_pids_for_subject(one, SUBJECT, TAG)
        else:
            pids = _get_pids_for_regions(one, REGIONS, TAG)
        pids = [pid for pid in pids if pid]
        if pids:
            return list(dict.fromkeys(pids))
    except Exception as exc:
        print(f"Remote PID selection failed, falling back to local base-cache filtering. ({type(exc).__name__}: {exc})")

    if remote_mode == "all":
        return pid_list
    if remote_mode == "subject":
        return _local_subject_filter(pid_list, SUBJECT)
    return _local_region_filter(pid_list, REGIONS)


# %%
pids_to_run = _resolve_pids_to_process()
if not pids_to_run:
    raise RuntimeError("No PIDs found. Set COMPUTE_ALL, PIDS, SUBJECT, or REGIONS.")

summary_rows = []
failed = []
print(f"Running packet cache computation for {len(pids_to_run)} PID(s).")

for run_idx, pid in enumerate(tqdm(pids_to_run, desc="Processing PIDs"), start=1):
    print("")
    print(f"[{run_idx}/{len(pids_to_run)}] PID: {pid}")
    cache_path = PACKET_CACHE_DIR / f"{pid}.pkl"
    try:
        if cache_path.exists() and not FORCE_RECOMPUTE:
            packet_cache = load_packet_cache(pid, PACKET_CACHE_DIR)
            status = "loaded"
            print(f"Loaded existing packet cache: {cache_path}")
        else:
            packet_cache = compute_packet_dashboard_cache(
                pid,
                packet_config=PACKET_CONFIG,
                packet_cache_dir=PACKET_CACHE_DIR,
            )
            status = "computed"
            print(f"Saved packet cache: {cache_path}")

        region_summary_df = packet_cache.get("region_summary_df", pd.DataFrame())
        print(f"Packet dashboard version: {packet_cache.get('packet_dashboard_version')}")
        if region_summary_df.empty:
            print("No packet regions were computed.")
            n_regions = 0
            n_packets = 0
        else:
            print(region_summary_df.to_string(index=False))
            n_regions = int(len(region_summary_df))
            n_packets = int(
                pd.to_numeric(
                    region_summary_df.get("n_packets", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).sum()
            )
        summary_rows.append(
            {
                "pid": pid,
                "status": status,
                "n_regions": n_regions,
                "n_packets_total": n_packets,
                "cache_path": str(cache_path),
            }
        )
    except Exception as exc:
        failed.append({"pid": pid, "error": f"{type(exc).__name__}: {exc}"})
        print(f"Failed: {type(exc).__name__}: {exc}")

print("")
print("Run summary:")
if summary_rows:
    print(pd.DataFrame(summary_rows).to_string(index=False))
else:
    print("No successful computations.")

if failed:
    print("")
    print("Failures:")
    print(pd.DataFrame(failed).to_string(index=False))
