"""
IBL Data Loading Functions
--------------------------
Functions for loading neural, behavioral, and anatomical data from the IBL database.
"""

import os
import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions


def load_session_data(one, pid, config, atlas=None):
    """
    Load neural and behavioral data for a given probe insertion.

    Inputs:
        one: ONE instance
        pid: str, probe insertion ID
        config: dict with keys:
            - CALC_ONLY_GOOD_UNITS: bool
            - CALC_SPONT: bool
            - LOAD_PUPIL: bool
            - LOAD_WHEEL: bool
            - LOAD_POSE: bool
        atlas: AllenAtlas instance or None (will create if needed)

    Outputs:
        dict with keys:
            - eid: session ID
            - pname: probe name
            - ssl: SpikeSortingLoader instance
            - sl: SessionLoader instance
            - trials, spikes, clusters, channels: neural/behavioral data
            - wheel, pose: movement data (if enabled)
            - spont_intervals, spikes_spont: spontaneous period (if enabled)
            - clusters_good: good clusters DataFrame
            - pupil_features, pupil_times: pupil data (if enabled)
            - cluster_ids, cid_to_idx: cluster ID mapping
    """
    print(f"Loading data for probe insertion: {pid}")

    # Create atlas if not provided
    if atlas is None:
        atlas = AllenAtlas()

    # Load spike sorting with atlas for coordinate mapping
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    print(f"Session ID: {ssl.eid}, Probe: {ssl.pname}")

    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    print(f"Spikes: {len(spikes.times)}, Clusters: {len(clusters.cluster_id)}, Channels: {len(channels.rawInd)}")
    if hasattr(clusters, 'acronym'):
        print(f"Regions found: {set(clusters.acronym)}")

    # Build cluster ID mapping for safe indexing
    cluster_ids, cid_to_idx = _build_cluster_id_map(clusters)

    # Load session data via SessionLoader
    sl = SessionLoader(eid=ssl.eid, one=one)
    sl.load_trials()
    print(f"Trials loaded: {len(sl.trials.intervals)} trials")

    # Initialize output
    result = {
        'eid': ssl.eid,
        'pname': ssl.pname,
        'ssl': ssl,
        'sl': sl,
        'trials': sl.trials,
        'spikes': spikes,
        'clusters': clusters,
        'channels': channels,
        'cluster_ids': cluster_ids,
        'cid_to_idx': cid_to_idx,
        'wheel': None,
        'pose': None,
        'spont_intervals': None,
        'spikes_spont': None,
        'clusters_good': None,
        'pupil_features': None,
        'pupil_times': None,
    }

    # Load wheel data
    if config.get("LOAD_WHEEL", False):
        sl.load_wheel()
        result['wheel'] = sl.wheel
        print(f"Wheel loaded: {list(sl.wheel.keys())}")

    # Load pose data
    if config.get("LOAD_POSE", False):
        sl.load_pose(views=["left", "right"])
        result['pose'] = sl.pose
        print(f"Pose loaded: {list(sl.pose.keys())}")

    # Filter for good units
    calc_only_good = config.get("CALC_ONLY_GOOD_UNITS", True)
    if calc_only_good:
        good_cluster_ids = clusters.cluster_id[clusters.label == 1]
        n_good = len(good_cluster_ids)
        print(f"Good units: {n_good}/{len(clusters.cluster_id)}")
    else:
        good_cluster_ids = clusters.cluster_id
        print("Using all units")

    # Load spontaneous activity
    if config.get("CALC_SPONT", False):
        print("\nLoading spontaneous activity period...")
        try:
            passive_times = one.load_dataset(ssl.eid, '*passivePeriods*', collection='alf')
            spont_intervals = np.array([[passive_times['spontaneousActivity'][0],
                                         passive_times['spontaneousActivity'][1]]])
            duration = spont_intervals[0][1] - spont_intervals[0][0]
            print(f"Spontaneous: {spont_intervals[0][0]:.2f}s to {spont_intervals[0][1]:.2f}s ({duration:.2f}s)")

            # Filter spikes to spontaneous period
            valid_mask = np.zeros(len(spikes.times), dtype=bool)
            for start, end in spont_intervals:
                valid_mask |= (spikes.times >= start) & (spikes.times <= end)

            spikes_spont = {k: getattr(spikes, k)[valid_mask] for k in spikes.keys()}
            print(f"Spontaneous spikes: {len(spikes_spont['times'])}/{len(spikes.times)}")

            # Filter for good units
            if calc_only_good:
                good_spk_mask = np.isin(spikes_spont['clusters'], good_cluster_ids)
                spikes_spont = {k: v[good_spk_mask] for k, v in spikes_spont.items()}

                cluster_mask = np.isin(clusters.cluster_id, good_cluster_ids)
                clusters_good = pd.DataFrame({k: getattr(clusters, k)[cluster_mask] for k in clusters.keys()})
                print(f"Good units in spont: {len(np.unique(spikes_spont['clusters']))} units, {len(spikes_spont['times'])} spikes")
            else:
                clusters_good = pd.DataFrame({k: getattr(clusters, k) for k in clusters.keys()})

            result['spont_intervals'] = spont_intervals
            result['spikes_spont'] = spikes_spont
            result['clusters_good'] = clusters_good

        except Exception as e:
            print(f"Could not load spontaneous period: {e}")

    # Load pupil data
    if config.get("LOAD_PUPIL", False):
        print("\nLoading pupil data...")
        try:
            left_camera = one.load_object(ssl.eid, "leftCamera", collection="alf")
            if hasattr(left_camera, "features") and hasattr(left_camera, "times"):
                result['pupil_features'] = left_camera.features
                result['pupil_times'] = left_camera.times
                print("Pupil data loaded successfully")
            else:
                print("Camera object missing 'features' or 'times' attribute")
        except Exception as e:
            print(f"Could not load pupil data: {e}")

    return result


def setup_brain_regions(clusters, config, clusters_good=None, path_data=None, hierarchy_file=None):
    """
    Initialize brain atlas and map cluster regions. Optionally create output directories.

    Inputs:
        clusters: cluster metadata from spike sorting
        config: dict with key ATLAS_MAPPING ('Beryl' or 'Allen')
        clusters_good: DataFrame or None, filtered good clusters
        path_data: str or None, if provided creates per-region output folders
        hierarchy_file: str or None, path to hierarchy CSV for loading hierarchy scores

    Outputs:
        dict with keys:
            - atlas: AllenAtlas instance
            - brain_regions: BrainRegions instance
            - beryl_acronyms: array of Beryl region names
            - cluster_acronyms: mapped region acronyms for all clusters
            - unique_regions: unique regions in this probe
            - clusters_good: updated DataFrame with 'region' column (if provided)
            - hier_scores: hierarchy scores per Beryl region (if hierarchy_file provided)
    """
    atlas_mapping = config.get("ATLAS_MAPPING", "Beryl")
    print(f"\nInitializing brain atlas with {atlas_mapping} mapping...")

    ba = AllenAtlas()
    br = BrainRegions()

    # Get Beryl acronyms (excluding void/root)
    beryl_indices = np.unique(br.mappings["Beryl"])
    beryl_acronyms = br.acronym[beryl_indices]
    beryl_acronyms = beryl_acronyms[(beryl_acronyms != "void") & (beryl_acronyms != "root")]
    print(f"Beryl regions: {len(beryl_acronyms)} unique")

    # Create per-region output directories if path provided
    if path_data is not None:
        for region in beryl_acronyms:
            os.makedirs(f"{path_data}/session_plots_integrated/{region}/", exist_ok=True)
        print(f"Created output directories in {path_data}")

    # Load hierarchy scores if file provided
    hier_scores = None
    if hierarchy_file is not None and os.path.exists(hierarchy_file):
        hier_df = pd.read_csv(hierarchy_file)
        area_to_hier = dict(zip(hier_df["areas"], hier_df["CC+TC+CT iterated"]))
        hier_scores = np.array([area_to_hier.get(r, np.nan) for r in beryl_acronyms])
        print(f"Loaded hierarchy scores for {np.sum(~np.isnan(hier_scores))} regions")

    # Map cluster acronyms
    acronym_data = clusters.acronym if hasattr(clusters, 'acronym') else clusters['acronym']
    if atlas_mapping == "Beryl":
        cluster_acronyms = br.acronym2acronym(acronym_data, mapping="Beryl")
    else:
        cluster_acronyms = np.asarray(acronym_data)

    unique_regions = np.unique(cluster_acronyms)
    print(f"Probe regions: {len(unique_regions)} - {unique_regions}")

    result = {
        'atlas': ba,
        'brain_regions': br,
        'beryl_acronyms': beryl_acronyms,
        'cluster_acronyms': cluster_acronyms,
        'unique_regions': unique_regions,
        'clusters_good': clusters_good,
        'hier_scores': hier_scores,
    }

    # Add region column to clusters_good if provided
    if clusters_good is not None and isinstance(clusters_good, pd.DataFrame):
        cid_data = clusters.cluster_id if hasattr(clusters, 'cluster_id') else clusters['cluster_id']
        good_mask = np.isin(cid_data, clusters_good["cluster_id"].to_numpy())
        clusters_good["region"] = cluster_acronyms[good_mask]
        result['clusters_good'] = clusters_good

    return result


# ---- Helper functions ----

def _build_cluster_id_map(clusters):
    """
    Build cluster ID array and mapping dict for safe indexing.

    Inputs:
        clusters: cluster metadata object

    Outputs:
        cluster_ids: array of cluster IDs
        cid_to_idx: dict mapping cluster_id -> array index
    """
    if hasattr(clusters, 'cluster_id'):
        cluster_ids = np.asarray(clusters.cluster_id)
    elif 'cluster_id' in clusters:
        cluster_ids = np.asarray(clusters['cluster_id'])
    else:
        cluster_ids = np.arange(len(clusters.acronym))

    cid_to_idx = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
    return cluster_ids, cid_to_idx


def get_cluster_label(clusters, idx):
    """
    Fetch unit quality label using a safe cluster index.

    Inputs:
        clusters: cluster metadata object
        idx: int, index into clusters

    Output:
        label: int, quality label (1=good, default if not found)
    """
    if hasattr(clusters, 'metrics') and 'label' in clusters.metrics.columns:
        return clusters.metrics.label[idx]
    if hasattr(clusters, 'label'):
        return clusters.label[idx]
    return 1