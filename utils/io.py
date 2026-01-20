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
