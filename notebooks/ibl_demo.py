from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader

def demo_ibl_one():
    # Setup ONE
    # Note: password is strictly for internal example, normally use credentials file
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)
    
    # Example PID from your notebook
    pid = "29418b18-4db6-436f-a85e-b0f1d38d2e62"
    eid, pname = one.pid2eid(pid)
    print(f"EID: {eid}, Probe: {pname}")

    # 1. Load Trials using SessionLoader
    # SessionLoader is for session-wide data like behavior (trials), wheel, pose, etc.
    print("Loading trials...")
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    # Access trials using sl.trials
    if not sl.trials.empty:
        print("Trials keys:", sl.trials.keys())
    else:
        print("No trials found.")
    
    # 2. Load Spike Sorting using SpikeSortingLoader
    # SpikeSortingLoader is specifically for ephys data (spikes, clusters, channels)
    print("Loading spike sorting...")
    ssl = SpikeSortingLoader(pid=pid, one=one)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    print(f"Loaded {len(spikes['times'])} spikes")
    
    # Common confusion: ssl.trials does not exist. Use sl.trials.
    # Also, to check the collection for spikes:
    print("Spike sorting collection:", ssl.collection)

if __name__ == "__main__":
    demo_ibl_one()
