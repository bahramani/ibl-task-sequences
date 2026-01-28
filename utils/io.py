import os
from pathlib import Path
import pandas as pd
import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

def setup_paths(base_path):
    """Create the main project folders and return their paths."""
    path_data = base_path / "data"
    path_fig = base_path / "results" / "figures"
    path_data_processed = path_data / "processed"
    ibl_cache = path_data / "raw"
    for p in [ibl_cache, path_fig, path_data_processed]:
        p.mkdir(exist_ok=True, parents=True)
    return path_data, path_fig, path_data_processed, ibl_cache


def init_one(ibl_cache):
    """Initialize the ONE API with a custom cache directory."""
    try:
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            silent=True,
            mode='remote',
            cache_dir=ibl_cache,
        )
        print("ONE API initialized.")
        return one
    except Exception as exc:
        raise RuntimeError(f"Error initializing ONE: {exc}")


def prepare_region_dirs(path_data):
    """Create per-region output folders for Beryl acronyms."""
    ba = AllenAtlas()
    br = BrainRegions()
    beryl_indices = np.unique(br.mappings["Beryl"])
    beryl_acronyms = br.acronym[beryl_indices]
    beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["void"]))
    beryl_acronyms = np.delete(beryl_acronyms, np.where(beryl_acronyms == ["root"]))
    for region in beryl_acronyms:
        os.makedirs(f"{path_data}/session_plots_integrated/{region}/", exist_ok=True)
    
    hier_file = os.path.join(path_data, "hierarchy_summary_CreConf_all_regions.csv")
    hier_df = pd.read_csv(hier_file)
    area_to_hier_score = dict(zip(hier_df["areas"], hier_df["CC+TC+CT iterated"]))
    hier_scores = np.array([area_to_hier_score.get(region, np.nan) for region in beryl_acronyms])

    return ba, br, beryl_acronyms, hier_scores

def map_acronyms(clusters, br, mapping):
    """Map cluster acronyms to the requested atlas (Beryl or Allen)."""
    if mapping == "Beryl":
        return br.acronym2acronym(clusters.acronym, mapping="Beryl")
    return clusters.acronym


def load_session_data(pid, one, ba):
    """Load spikes, clusters, and session data for a given probe insertion."""
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    print(f"Session ID (EID): {ssl.eid}")
    print(f"Probe Name: {ssl.pname}")

    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    print(f"Spikes loaded: {spikes.times.shape[0]} spikes")
    if "acronym" in clusters:
        print(f"Cluster regions found: {set(clusters.acronym)}")

    sl = SessionLoader(eid=ssl.eid, one=one)
    sl.load_trials()
    print(f"Trials loaded. Found keys: {list(sl.trials.keys())}")
    sl.load_wheel()
    print(f"Wheel data loaded. Found keys: {list(sl.wheel.keys())}")
    sl.load_pose(views=["left", "right"])
    print(f"Pose data loaded. Found keys: {list(sl.pose.keys())}")

    return ssl, spikes, clusters, sl


def load_pupil_data(sl):
    """Load pupil features and times from the left camera, if available."""
    try:
        left_camera = sl.one.load_object(sl.eid, "leftCamera", collection="alf")
        if hasattr(left_camera, "features") and hasattr(left_camera, "times"):
            print("Camera data loaded successfully.")
            return left_camera.features, left_camera.times
        print("Camera object loaded but 'features' or 'times' attribute missing.")
    except Exception as exc:
        print(f"Could not load pupil data: {exc}")
    return None, None

def build_cluster_id_map(clusters):
    """Return (cluster_ids, cid_to_idx) for safe cluster-id indexing."""
    if hasattr(clusters, "cluster_id"):
        cluster_ids = np.asarray(clusters.cluster_id)
    elif "cluster_id" in clusters:
        cluster_ids = np.asarray(clusters["cluster_id"])
    else:
        cluster_ids = np.arange(len(clusters.acronym))
    cid_to_idx = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
    return cluster_ids, cid_to_idx


def get_cluster_label(clusters, idx):
    """Fetch a unit quality label using a safe cluster index."""
    if hasattr(clusters, "metrics") and "label" in clusters.metrics.columns:
        return clusters.metrics.label[idx]
    if hasattr(clusters, "label"):
        return clusters.label[idx]
    return 1


def get_session_pids(eid, one):
    """Get all probe insertion IDs (PIDs) for a given session (EID)."""
    insertions = one.alyx.rest('insertions', 'list', session=eid)
    pids = [ins['id'] for ins in insertions]
    return pids


def print_region_info(clusters, cluster_acronyms, label=""):
    """Print region names and neuron counts.

    Parameters
    ----------
    clusters : dict or DataFrame
        Cluster data containing cluster_id
    cluster_acronyms : array-like
        Brain region acronyms for each cluster
    label : str
        Label to print before the region info (e.g., "original PID" or "other simultaneous PID")
    """
    unique_regions, counts = np.unique(cluster_acronyms, return_counts=True)
    # Sort by count descending
    sort_idx = np.argsort(-counts)
    unique_regions = unique_regions[sort_idx]
    counts = counts[sort_idx]

    region_str = ", ".join([f"{region}: {count}" for region, count in zip(unique_regions, counts)])
    print(f"Regions in {label}: {region_str}")
    return dict(zip(unique_regions, counts))


def load_and_combine_multi_probe_data(pid, one, ba, br, atlas_mapping="Beryl"):
    """Load data from all probes in the same session and combine spikes.

    Parameters
    ----------
    pid : str
        The primary probe insertion ID
    one : ONE
        ONE API instance
    ba : AllenAtlas
        Brain atlas instance
    br : BrainRegions
        Brain regions instance
    atlas_mapping : str
        Atlas mapping to use ("Beryl" or "Allen")

    Returns
    -------
    ssl : SpikeSortingLoader
        Spike sorting loader for the primary PID
    spikes_combined : dict
        Combined spikes from all probes
    clusters_combined : DataFrame
        Combined cluster info from all probes
    sl : SessionLoader
        Session loader
    other_pids : list
        List of other PIDs in the same session
    """
    # Load primary probe data
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    eid = ssl.eid
    print(f"Session ID (EID): {eid}")
    print(f"Primary Probe Name: {ssl.pname}")

    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    print(f"Primary probe spikes loaded: {spikes.times.shape[0]} spikes")

    # Get cluster acronyms for primary probe
    cluster_acronyms_primary = map_acronyms(clusters, br, atlas_mapping)
    print_region_info(clusters, cluster_acronyms_primary, label="original PID")

    # Convert spikes to dict if needed for combining
    if hasattr(spikes, 'times'):
        spikes_dict = {
            'times': np.array(spikes.times),
            'clusters': np.array(spikes.clusters),
            'amps': np.array(spikes.amps) if hasattr(spikes, 'amps') else None,
            'depths': np.array(spikes.depths) if hasattr(spikes, 'depths') else None,
        }
    else:
        spikes_dict = dict(spikes)

    # Convert clusters to DataFrame
    if not isinstance(clusters, pd.DataFrame):
        clusters_df = pd.DataFrame(clusters)
    else:
        clusters_df = clusters.copy()
    clusters_df['source_pid'] = pid

    # Get all PIDs for this session
    all_pids = get_session_pids(eid, one)
    other_pids = [p for p in all_pids if p != pid]

    if len(other_pids) == 0:
        print("No other probes found in this session.")
        sl = SessionLoader(eid=eid, one=one)
        sl.load_trials()
        sl.load_wheel()
        sl.load_pose(views=["left", "right"])
        return ssl, spikes_dict, clusters_df, sl, other_pids

    print(f"\nFound {len(other_pids)} other probe(s) in this session")

    # Load and combine data from other probes
    all_spikes_times = [spikes_dict['times']]
    all_spikes_clusters = [spikes_dict['clusters']]
    all_spikes_amps = [spikes_dict['amps']] if spikes_dict['amps'] is not None else []
    all_spikes_depths = [spikes_dict['depths']] if spikes_dict['depths'] is not None else []
    all_clusters = [clusters_df]

    # Track max cluster ID to offset new clusters
    max_cluster_id = int(np.max(clusters_df['cluster_id'])) + 1

    for other_pid in other_pids:
        print(f"\nLoading other probe: {other_pid}")
        try:
            ssl_other = SpikeSortingLoader(pid=other_pid, one=one, atlas=ba)
            print(f"  Probe Name: {ssl_other.pname}")

            spikes_other, clusters_other, channels_other = ssl_other.load_spike_sorting()
            clusters_other = ssl_other.merge_clusters(spikes_other, clusters_other, channels_other)
            print(f"  Spikes loaded: {spikes_other.times.shape[0]} spikes")

            # Get cluster acronyms for this probe
            cluster_acronyms_other = map_acronyms(clusters_other, br, atlas_mapping)
            print_region_info(clusters_other, cluster_acronyms_other, label="other simultaneous PID")

            # Convert clusters to DataFrame and offset cluster IDs
            if not isinstance(clusters_other, pd.DataFrame):
                clusters_other_df = pd.DataFrame(clusters_other)
            else:
                clusters_other_df = clusters_other.copy()

            # Offset cluster IDs to avoid collisions
            original_cluster_ids = np.array(clusters_other_df['cluster_id'])
            offset_cluster_ids = original_cluster_ids + max_cluster_id
            clusters_other_df['cluster_id'] = offset_cluster_ids
            clusters_other_df['source_pid'] = other_pid

            # Update spike cluster assignments
            spikes_clusters_offset = np.array(spikes_other.clusters) + max_cluster_id

            # Update max cluster ID for next probe
            max_cluster_id = int(np.max(offset_cluster_ids)) + 1

            # Collect spike data
            all_spikes_times.append(np.array(spikes_other.times))
            all_spikes_clusters.append(spikes_clusters_offset)
            if hasattr(spikes_other, 'amps') and spikes_dict['amps'] is not None:
                all_spikes_amps.append(np.array(spikes_other.amps))
            if hasattr(spikes_other, 'depths') and spikes_dict['depths'] is not None:
                all_spikes_depths.append(np.array(spikes_other.depths))

            all_clusters.append(clusters_other_df)

        except Exception as e:
            print(f"  Error loading probe {other_pid}: {e}")
            continue

    # Combine all spikes
    spikes_combined = {
        'times': np.concatenate(all_spikes_times),
        'clusters': np.concatenate(all_spikes_clusters),
    }
    if len(all_spikes_amps) == len(all_spikes_times):
        spikes_combined['amps'] = np.concatenate(all_spikes_amps)
    if len(all_spikes_depths) == len(all_spikes_times):
        spikes_combined['depths'] = np.concatenate(all_spikes_depths)

    # Sort by time
    sort_idx = np.argsort(spikes_combined['times'])
    for key in spikes_combined:
        if spikes_combined[key] is not None:
            spikes_combined[key] = spikes_combined[key][sort_idx]

    # Combine all clusters
    clusters_combined = pd.concat(all_clusters, ignore_index=True)

    print(f"\nCombined data:")
    print(f"  Total spikes: {len(spikes_combined['times'])}")
    print(f"  Total clusters: {len(clusters_combined)}")

    # Load session data
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    print(f"Trials loaded. Found keys: {list(sl.trials.keys())}")
    sl.load_wheel()
    print(f"Wheel data loaded. Found keys: {list(sl.wheel.keys())}")
    sl.load_pose(views=["left", "right"])
    print(f"Pose data loaded. Found keys: {list(sl.pose.keys())}")

    return ssl, spikes_combined, clusters_combined, sl, other_pids
