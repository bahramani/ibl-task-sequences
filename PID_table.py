from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions


BASE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_PATH))

from utils.io import (
    get_cluster_labels_array,
    init_one,
    load_session_data,
    map_acronyms,
    setup_paths,
)
from utils.packet_dashboard import BASE_CACHE_DIR, load_base_cache


TAG = "2024_Q2_IBL_et_al_RepeatedSite" # "2024_Q2_IBL_et_al_RepeatedSite" # "RepeatedSite"
TOP_N = 10
ATLAS_MAPPING = "Beryl"
ALL_PIDS_OUTPUT = BASE_PATH / "results" / f"{TAG}_all_pids.csv"
SUMMARY_OUTPUT = BASE_PATH / "results" / f"{TAG}_pid_good_neuron_summary.csv"


def _get_all_tag_pids(one, tag):
    django_query = f"datasets__tags__name,{tag}"
    insertions = one.alyx.rest("insertions", "list", django=django_query)
    return list(dict.fromkeys([ins["id"] for ins in insertions if ins.get("id")]))


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


def _good_label_mask(labels):
    if labels is None:
        return None

    series = pd.Series(np.asarray(labels).reshape(-1))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return pd.Series(
            np.isclose(numeric.to_numpy(dtype=float), 1.0),
            index=series.index,
        )

    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "good", "true"})


def _clean_regions(regions):
    if regions is None:
        return []
    series = pd.Series(np.asarray(regions).reshape(-1)).dropna().astype(str).str.strip()
    exclude = {"", "nan", "none", "void", "root"}
    series = series.loc[~series.str.lower().isin(exclude)]
    return sorted(series.drop_duplicates().tolist())


def _region_text(regions):
    return ", ".join(regions)


def _summary_from_cache(pid):
    cache = load_base_cache(pid, BASE_CACHE_DIR)
    meta = cache.get("meta", {})
    df_res = cache.get("df_res", pd.DataFrame())

    all_regions = _clean_regions(cache.get("cluster_acronyms_plot"))
    if not all_regions and isinstance(df_res, pd.DataFrame) and "acronym" in df_res.columns:
        all_regions = _clean_regions(df_res["acronym"])

    if not isinstance(df_res, pd.DataFrame) or df_res.empty or "label" not in df_res.columns:
        raise ValueError("Cache does not contain per-cluster labels.")

    good_mask = _good_label_mask(df_res["label"])
    if good_mask is None or len(good_mask) != len(df_res):
        raise ValueError("Cache label array is missing or malformed.")

    good_regions = []
    if "acronym" in df_res.columns:
        good_regions = _clean_regions(df_res.loc[good_mask.to_numpy(dtype=bool), "acronym"])

    return {
        "pid": pid,
        "eid": meta.get("eid"),
        "subject": meta.get("subject"),
        "lab": meta.get("lab"),
        "date": meta.get("date"),
        "good_neurons": int(good_mask.sum()),
        "n_regions": int(len(all_regions)),
        "regions": _region_text(all_regions),
        "n_good_regions": int(len(good_regions)),
        "good_regions": _region_text(good_regions),
        "source": "cache",
    }


def _summary_from_raw(pid, one, ba, br):
    ssl, _spikes, clusters, _sl = load_session_data(
        pid,
        one,
        ba=ba,
        load_trials=False,
        load_wheel=False,
        load_pose=False,
        load_motion_energy=False,
        load_pupil=False,
    )

    labels = get_cluster_labels_array(clusters)
    good_mask = _good_label_mask(labels)
    if good_mask is None:
        raise ValueError("Spike-sorting labels are unavailable for this PID.")

    acronyms = np.asarray(map_acronyms(clusters, br, ATLAS_MAPPING)).astype(str)
    if len(good_mask) != len(acronyms):
        raise ValueError("Label count does not match acronym count.")

    meta = _fetch_session_metadata(one, ssl.eid)
    all_regions = _clean_regions(acronyms)
    good_regions = _clean_regions(acronyms[good_mask.to_numpy(dtype=bool)])

    return {
        "pid": pid,
        "eid": ssl.eid,
        "subject": meta.get("subject"),
        "lab": meta.get("lab"),
        "date": meta.get("date"),
        "good_neurons": int(good_mask.sum()),
        "n_regions": int(len(all_regions)),
        "regions": _region_text(all_regions),
        "n_good_regions": int(len(good_regions)),
        "good_regions": _region_text(good_regions),
        "source": "raw",
    }


def build_pid_summary(tag=TAG):
    _path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
    one = init_one(ibl_cache)
    pids = _get_all_tag_pids(one, tag)
    if not pids:
        raise RuntimeError(f"No PIDs found for tag '{tag}'.")

    ba = AllenAtlas()
    br = BrainRegions()

    rows = []
    failures = []

    for pid in tqdm(pids, desc=f"Summarizing tag={tag}"):
        try:
            try:
                row = _summary_from_cache(pid)
            except Exception:
                row = _summary_from_raw(pid, one, ba, br)
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "pid": pid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["good_neurons", "pid"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    failed_df = pd.DataFrame(failures)
    return pids, summary_df, failed_df


def main():
    pids, summary_df, failed_df = build_pid_summary(TAG)

    ALL_PIDS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pid": pids}).to_csv(ALL_PIDS_OUTPUT, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("")
    print(f"All PIDs with tag '{TAG}' ({len(pids)} total):")
    print(pd.Series(pids, name="pid").to_string(index=False))

    print("")
    print(f"Top {min(TOP_N, len(summary_df))} PIDs by number of good neurons:")
    if summary_df.empty:
        print("No PID summaries could be built.")
    else:
        display_cols = ["pid", "subject", "good_neurons", "n_regions", "regions"]
        print(summary_df.loc[:, display_cols].head(TOP_N).to_string(index=False))

    print("")
    print(f"Saved PID list to: {ALL_PIDS_OUTPUT}")
    print(f"Saved full summary to: {SUMMARY_OUTPUT}")

    if not failed_df.empty:
        print("")
        print("Failed PIDs:")
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()
