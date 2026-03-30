# %% Imports
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

BASE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(Path.cwd().parent))

import utils.analysis as ana_utils
import utils.plotting_plotly as plotting_utils
from utils.io import (
    setup_paths,
    init_one,
    prepare_region_dirs,
    map_acronyms,
    load_session_data,
    build_cluster_id_map,
    get_cluster_labels_array,
)


# %% Helpers
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"
PLOTLY_RENDERER = None  # "browser", "notebook_connected", "png", "svg"


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
    try:
        fig.show()
    except ValueError as exc:
        if "nbformat" in str(exc).lower():
            pio.renderers.default = "browser"
            fig.show()
        else:
            raise



def list_cached_pids(cache_dir=CACHE_DIR):
    if not cache_dir.exists():
        return []
    return sorted([p.stem for p in cache_dir.glob("*.pkl")])



def choose_pid(pid_list, default_index=0):
    if not pid_list:
        raise RuntimeError("No PIDs found in data/dashboard_cache.")
    print("Available PIDs:")
    for idx, pid_val in enumerate(pid_list):
        print(f"  [{idx}] {pid_val}")
    prompt = f"Enter PID or index (press Enter for {pid_list[default_index]}): "
    selection = input(prompt).strip()
    if selection == "":
        return pid_list[default_index]
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(pid_list):
            return pid_list[idx]
        raise ValueError(f"Index {idx} out of range.")
    if selection in pid_list:
        return selection
    raise ValueError(f"PID '{selection}' not found.")



def choose_region(region_list, default_index=0):
    if not region_list:
        raise RuntimeError("No regions found for the selected PID.")
    print("Available regions:")
    for idx, region in enumerate(region_list):
        print(f"  [{idx}] {region}")
    prompt = f"Enter region or index (press Enter for {region_list[default_index]}): "
    selection = input(prompt).strip()
    if selection == "":
        return region_list[default_index]
    if selection.isdigit():
        idx = int(selection)
        if 0 <= idx < len(region_list):
            return region_list[idx]
        raise ValueError(f"Index {idx} out of range.")
    if selection in region_list:
        return selection
    raise ValueError(f"Region '{selection}' not found.")



def _init_one_with_fallback(ibl_cache, preferred_mode="local", allow_remote=True):
    modes = []
    if preferred_mode:
        modes.append(preferred_mode)
    if "local" not in modes:
        modes.append("local")
    if allow_remote and "remote" not in modes:
        modes.append("remote")
    last_error = None
    for mode in modes:
        try:
            one = init_one(ibl_cache, mode=mode)
            return one, mode
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize ONE. Last error: {last_error}")



def _load_spontaneous_intervals(one, eid):
    try:
        passive_times = one.load_dataset(eid, "*passivePeriods*", collection="alf")
        spont = passive_times.get("spontaneousActivity", None)
        if spont is not None:
            return np.array([[spont[0], spont[1]]], dtype=float)
    except Exception:
        return None
    return None



def _select_cluster_ids_by_label(cluster_ids, clusters, label_min=None):
    cluster_ids = np.asarray(cluster_ids)
    if label_min is None:
        return cluster_ids
    labels = get_cluster_labels_array(clusters)
    if labels is None:
        return cluster_ids
    labels = np.asarray(labels)
    if labels.shape[0] != cluster_ids.shape[0]:
        return cluster_ids
    try:
        labels_float = labels.astype(float)
        mask = labels_float >= float(label_min)
    except (TypeError, ValueError):
        mask = labels == 1
    return cluster_ids[mask]



def _list_available_regions(cluster_acronyms):
    regions = []
    for region in np.asarray(cluster_acronyms).astype(str):
        region = str(region).strip()
        if region == "" or region.lower() in {"nan", "root", "void"}:
            continue
        regions.append(region)
    return sorted(pd.Series(regions).drop_duplicates().tolist())



def _region_cluster_ids(cluster_ids, cluster_acronyms, region_name):
    region_name = str(region_name)
    mask = np.asarray(
        [str(acr).startswith(region_name) for acr in np.asarray(cluster_acronyms).astype(str)],
        dtype=bool,
    )
    return np.asarray(cluster_ids)[mask]



def _empty_to_none(df):
    if df is None:
        return None
    if isinstance(df, pd.DataFrame) and df.empty:
        return None
    return df



def _rename_method_columns(df, suffix):
    if df is None:
        return None
    out = df.copy()
    rename_map = {
        col: f"{col}_{suffix}"
        for col in out.columns
        if col not in {"cluster_id", "region"}
    }
    return out.rename(columns=rename_map)



def _merge_method_tables(df_current, df_okun):
    df_current = _rename_method_columns(_empty_to_none(df_current), "current")
    df_okun = _rename_method_columns(_empty_to_none(df_okun), "okun")
    if df_current is None and df_okun is None:
        return None
    if df_current is None:
        return df_okun
    if df_okun is None:
        return df_current
    return df_current.merge(df_okun, on=["cluster_id", "region"], how="outer")



def _curve_array(values):
    if values is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr



def _row_has_any_curve(row):
    for col in row.index:
        if str(col).startswith("stpr_curve") and _curve_array(row[col]).size > 0:
            return True
    return False



def _normalize_cluster_id_list(values):
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return []
    out = []
    for val in arr.tolist():
        try:
            out.append(int(val))
        except (TypeError, ValueError):
            continue
    return out



def _select_plot_cluster_ids(comparison_tables, plot_n, seed=0, manual_ids=None):
    plot_n = int(max(1, plot_n))
    available_ids = set()
    for df in comparison_tables.values():
        if df is None or df.empty:
            continue
        valid_rows = df.apply(_row_has_any_curve, axis=1)
        available_ids.update(df.loc[valid_rows, "cluster_id"].astype(int).tolist())
    available_ids = sorted(available_ids)
    if not available_ids:
        raise RuntimeError("No neurons have valid stPR curves for the selected region.")

    manual_ids = _normalize_cluster_id_list(manual_ids)
    if manual_ids is not None:
        chosen = [cid for cid in manual_ids if cid in available_ids]
        missing = [cid for cid in manual_ids if cid not in available_ids]
        if missing:
            raise ValueError(
                "PLOT_CLUSTER_IDS contains IDs with no valid comparison data: "
                f"{missing}"
            )
        return chosen[:plot_n]

    rng = np.random.default_rng(seed)
    sample_size = min(plot_n, len(available_ids))
    chosen = rng.choice(np.asarray(available_ids, dtype=int), size=sample_size, replace=False)
    return chosen.astype(int).tolist()



def _subplot_titles(cluster_ids, cluster_acronym_lookup, n_rows, n_cols):
    titles = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if row_idx == 0 and col_idx < len(cluster_ids):
                cid = int(cluster_ids[col_idx])
                region = cluster_acronym_lookup.get(cid, "NA")
                titles.append(f"Cluster {cid} | {region}")
            elif row_idx == 0:
                titles.append("No neuron")
            else:
                titles.append("")
    return titles



def _annotate_subplot(fig, row, col, text):
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="x domain",
        yref="y domain",
        showarrow=False,
        font=dict(size=12, color="#666666"),
        row=row,
        col=col,
    )



def plot_stpr_method_comparison(
    comparison_tables,
    plot_cluster_ids,
    cluster_acronym_lookup,
    current_config,
    okun_config,
    region_name,
    pid,
    template,
    fig_width,
    fig_height,
):
    context_specs = [
        {
            "label": "Spont",
            "table": comparison_tables.get("Spont"),
            "split_suffixes": ("h1", "h2"),
            "split_labels": ("H1", "H2"),
        },
        {
            "label": "ITI",
            "table": comparison_tables.get("ITI"),
            "split_suffixes": ("odd", "even"),
            "split_labels": ("Odd", "Even"),
        },
        {
            "label": "Task",
            "table": comparison_tables.get("Task"),
            "split_suffixes": ("odd", "even"),
            "split_labels": ("Odd", "Even"),
        },
    ]
    n_rows = len(context_specs)
    n_cols = 4
    subplot_titles = _subplot_titles(plot_cluster_ids, cluster_acronym_lookup, n_rows, n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"secondary_y": True} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    colors = {
        "current": "#1f77b4",
        "okun": "#d95f02",
    }
    dashes = {
        "a": "solid",
        "b": "dash",
    }
    legend_seen = set()

    for row_idx, context in enumerate(context_specs, start=1):
        context_label = context["label"]
        split_a, split_b = context["split_suffixes"]
        split_label_a, split_label_b = context["split_labels"]
        df_context = context["table"]

        for col_idx in range(1, n_cols + 1):
            fig.add_vline(
                x=0.0,
                line=dict(color="gray", dash="dot", width=1),
                row=row_idx,
                col=col_idx,
            )
            if col_idx > len(plot_cluster_ids):
                _annotate_subplot(fig, row_idx, col_idx, "No neuron selected")
                continue

            cluster_id = int(plot_cluster_ids[col_idx - 1])
            if df_context is None or df_context.empty:
                _annotate_subplot(fig, row_idx, col_idx, f"{context_label} unavailable")
                continue

            match = df_context[df_context["cluster_id"] == cluster_id]
            if match.empty:
                _annotate_subplot(fig, row_idx, col_idx, "No data")
                continue

            row = match.iloc[0]
            trace_specs = [
                ("current", split_a, split_label_a, False, "a"),
                ("current", split_b, split_label_b, False, "b"),
                ("okun", split_a, split_label_a, True, "a"),
                ("okun", split_b, split_label_b, True, "b"),
            ]
            plotted_any = False
            for method_name, split_suffix, split_label, secondary_y, dash_key in trace_specs:
                curve_col = f"stpr_curve_{split_suffix}_{method_name}"
                curve = _curve_array(row.get(curve_col, []))
                if curve.size == 0:
                    continue
                config_local = current_config if method_name == "current" else okun_config
                lags_ms = ana_utils.build_stpr_lags(config_local, curve_len=curve.size)
                trace_name = f"{context_label} {method_name.title()} {split_label}"
                showlegend = trace_name not in legend_seen
                legend_seen.add(trace_name)
                fig.add_trace(
                    go.Scatter(
                        x=lags_ms,
                        y=curve,
                        mode="lines",
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=showlegend,
                        line=dict(
                            color=colors[method_name],
                            dash=dashes[dash_key],
                            width=2,
                        ),
                        hovertemplate=(
                            f"Cluster {cluster_id}<br>"
                            f"{context_label} | {method_name.title()} {split_label}<br>"
                            "Lag: %{x:.1f} ms<br>"
                            "Value: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                    secondary_y=secondary_y,
                )
                plotted_any = True

            if not plotted_any:
                _annotate_subplot(fig, row_idx, col_idx, "No valid curves")

    for row_idx, context in enumerate(context_specs, start=1):
        fig.update_yaxes(
            title_text=f"{context['label']} Current stPR",
            row=row_idx,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Okun stPR (spk/s above baseline)",
            row=row_idx,
            col=n_cols,
            secondary_y=True,
        )
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Lag (ms)", row=n_rows, col=col_idx)

    fig.add_annotation(
        text=(
            "Left y-axis: current repo stPR | Right y-axis: Okun-style baseline-subtracted "
            "population rate"
        ),
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13),
    )

    fig.update_layout(
        template=template,
        width=int(fig_width),
        height=int(fig_height),
        title=(
            f"stPR Method Comparison | PID {pid} | Region {region_name} | "
            f"n={len(plot_cluster_ids)} shown"
        ),
        margin=dict(l=70, r=70, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    return fig


# %% Config
PID = "c9664185-d3fd-4e0e-89cf-77c402038938"
TARGET_REGION = None
ONE_PREFERRED_MODE = "remote"
ALLOW_REMOTE_FALLBACK = True

CONFIG_CALC = {
    "ATLAS_MAPPING": "Beryl",
    "CALC_LABEL_MIN": 0.5,
    "CALC_SPONT": True,
    "EVENT_NAMES": ["stimOn_times", "firstMovement_times", "response_times", "feedback_times"],
    "STPR_BIN_SIZE": 0.001,
    "STPR_WINDOW_MS": 80,
    "STPR_LOW_PASS_HZ": 20,
    "STPR_LOW_PASS_ORDER": 3,
    "STPR_POP_USE_GOOD_UNITS": False,
    "TASK_POST_EVENT_S": 1.0,
    "ITI_SKIP_FIRST_LAST": True,
    "OKUN_BIN_SIZE_S": 0.001,
    "OKUN_GAUSS_HALF_WIDTH_MS": 12,
}

PLOT_N_NEURONS = 4
PLOT_RANDOM_SEED = 0
PLOT_CLUSTER_IDS = None  # Example: [123, 456, 789, 1011]
PLOTLY_TEMPLATE = "plotly_white"
FIG_WIDTH = 1850
FIG_HEIGHT = 1100


# %% Session loading
_set_plotly_renderer(PLOTLY_RENDERER)
pio.templates.default = PLOTLY_TEMPLATE
plotting_utils.DEFAULT_TEMPLATE = PLOTLY_TEMPLATE

path_data, _path_fig, _path_data_processed, ibl_cache = setup_paths(BASE_PATH)
one, one_mode = _init_one_with_fallback(
    ibl_cache,
    preferred_mode=ONE_PREFERRED_MODE,
    allow_remote=ALLOW_REMOTE_FALLBACK,
)
print(f"Using ONE mode: {one_mode}")

if PID is None:
    pid_list = list_cached_pids()
    if not pid_list:
        raise RuntimeError("Set PID manually. No cached PID list found in data/dashboard_cache.")
    pid = choose_pid(pid_list, default_index=0)
else:
    pid = PID
print(f"Selected PID: {pid}")

ba, br, _beryl_acronyms, _hier_scores = prepare_region_dirs(path_data)
ssl, spikes, clusters, sl = load_session_data(
    pid,
    one,
    ba=ba,
    load_wheel=False,
    load_pose=False,
    load_motion_energy=False,
    load_pupil=False,
)
eid = getattr(ssl, "eid", None)
if eid is None:
    eid, _ = one.pid2eid(pid)
print(f"EID: {eid}")

cluster_ids, cid_to_idx = build_cluster_id_map(clusters)
cluster_acronyms_calc = map_acronyms(clusters, br, CONFIG_CALC["ATLAS_MAPPING"])
cluster_acronyms_plot = np.asarray(cluster_acronyms_calc).astype(str)
cluster_acronym_lookup = {
    int(cid): str(cluster_acronyms_plot[cid_to_idx[int(cid)]])
    for cid in cluster_ids
    if int(cid) in cid_to_idx
}
available_regions = _list_available_regions(cluster_acronyms_plot)
if TARGET_REGION is None:
    target_region = choose_region(available_regions, default_index=0)
else:
    target_region = str(TARGET_REGION)
print(f"Selected region: {target_region}")

region_cluster_ids = _region_cluster_ids(cluster_ids, cluster_acronyms_plot, target_region)
if region_cluster_ids.size == 0:
    raise RuntimeError(f"No clusters from region '{target_region}' were found for PID {pid}.")
matched_regions = sorted(
    {
        str(cluster_acronym_lookup.get(int(cid), "NA"))
        for cid in region_cluster_ids
    }
)
print("Matched acronyms:", matched_regions)

calc_label_min = CONFIG_CALC.get("CALC_LABEL_MIN", None)
if calc_label_min is None and CONFIG_CALC.get("CALC_ONLY_GOOD_UNITS", False):
    calc_label_min = 1.0
calc_cluster_ids = _select_cluster_ids_by_label(cluster_ids, clusters, label_min=calc_label_min)
calc_region_cluster_id_set = set(int(cid) for cid in region_cluster_ids.tolist())
calc_cluster_ids = np.asarray(
    [cid for cid in calc_cluster_ids if int(cid) in calc_region_cluster_id_set],
    dtype=np.asarray(cluster_ids).dtype,
)
if calc_cluster_ids.size == 0:
    raise RuntimeError(
        f"No neurons passed CALC_LABEL_MIN={calc_label_min} in region '{target_region}'."
    )
print(
    f"Region clusters after label filter: {calc_cluster_ids.size} / {region_cluster_ids.size} "
    f"(label_min={calc_label_min})"
)


# %% stPR calculations
CURRENT_CONFIG = dict(CONFIG_CALC)
OKUN_CONFIG = dict(CONFIG_CALC)
OKUN_LAG_CONFIG = dict(CONFIG_CALC)
OKUN_LAG_CONFIG["STPR_BIN_SIZE"] = CONFIG_CALC.get("OKUN_BIN_SIZE_S", CONFIG_CALC.get("STPR_BIN_SIZE", 0.001))


def _compute_both_methods(spikes_local, intervals_local, split_halves, context_label):
    df_current = ana_utils.compute_population_coupling(
        spikes_local,
        clusters,
        cluster_acronyms_calc,
        CURRENT_CONFIG,
        cluster_ids=calc_cluster_ids,
        split_halves=split_halves,
        intervals=intervals_local,
        context_label=context_label,
    )
    df_okun = ana_utils.compute_population_coupling_okun(
        spikes_local,
        clusters,
        cluster_acronyms_calc,
        OKUN_CONFIG,
        cluster_ids=calc_cluster_ids,
        split_halves=split_halves,
        intervals=intervals_local,
        context_label=context_label,
    )
    return _empty_to_none(df_current), _empty_to_none(df_okun)


spont_intervals = _load_spontaneous_intervals(one, eid) if CONFIG_CALC.get("CALC_SPONT", True) else None
spont_interval_list = []
if spont_intervals is not None:
    spont_intervals = np.asarray(spont_intervals, dtype=float)
    spont_interval_list = [tuple(row) for row in spont_intervals]

df_current_spont = None
df_okun_spont = None
if CONFIG_CALC.get("CALC_SPONT", True) and spont_interval_list:
    spikes_spont = ana_utils.slice_spikes_by_intervals(spikes, spont_interval_list)
    df_current_spont, df_okun_spont = _compute_both_methods(
        spikes_spont,
        spont_interval_list,
        split_halves=True,
        context_label="Spont",
    )


task_windows = ana_utils.build_task_window_table(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
if not task_windows.empty:
    task_odd_intervals = task_windows.loc[task_windows["odd"], ["start", "end"]].to_numpy()
    task_even_intervals = task_windows.loc[~task_windows["odd"], ["start", "end"]].to_numpy()
else:
    task_odd_intervals = np.empty((0, 2))
    task_even_intervals = np.empty((0, 2))

df_current_task_odd = None
df_current_task_even = None
df_okun_task_odd = None
df_okun_task_even = None
if len(task_odd_intervals) > 0:
    spikes_task_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_current_task_odd, df_okun_task_odd = _compute_both_methods(
        spikes_task_odd,
        task_odd_intervals,
        split_halves=False,
        context_label="Task odd",
    )
if len(task_even_intervals) > 0:
    spikes_task_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        task_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_current_task_even, df_okun_task_even = _compute_both_methods(
        spikes_task_even,
        task_even_intervals,
        split_halves=False,
        context_label="Task even",
    )

df_current_task = None
df_okun_task = None
if df_current_task_odd is not None or df_current_task_even is not None:
    df_current_task = ana_utils.merge_stpr_splits(
        df_current_task_odd,
        df_current_task_even,
        CURRENT_CONFIG,
        split_a="odd",
        split_b="even",
    )
if df_okun_task_odd is not None or df_okun_task_even is not None:
    df_okun_task = ana_utils.merge_stpr_splits(
        df_okun_task_odd,
        df_okun_task_even,
        OKUN_LAG_CONFIG,
        split_a="odd",
        split_b="even",
    )
df_current_task = _empty_to_none(df_current_task)
df_okun_task = _empty_to_none(df_okun_task)


trial_end_times = ana_utils.compute_trial_end_times(
    sl.trials,
    CONFIG_CALC["EVENT_NAMES"],
    post_event_s=CONFIG_CALC["TASK_POST_EVENT_S"],
)
stim_on_times = np.asarray(sl.trials["stimOn_times"], dtype=float)
iti_windows = ana_utils.build_iti_windows(
    trial_end_times,
    stim_on_times,
    skip_first_last=CONFIG_CALC["ITI_SKIP_FIRST_LAST"],
)
if not iti_windows.empty:
    iti_odd_intervals = iti_windows.loc[iti_windows["odd"], ["start", "end"]].to_numpy()
    iti_even_intervals = iti_windows.loc[~iti_windows["odd"], ["start", "end"]].to_numpy()
else:
    iti_odd_intervals = np.empty((0, 2))
    iti_even_intervals = np.empty((0, 2))

df_current_iti_odd = None
df_current_iti_even = None
df_okun_iti_odd = None
df_okun_iti_even = None
if len(iti_odd_intervals) > 0:
    spikes_iti_odd = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_odd_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_current_iti_odd, df_okun_iti_odd = _compute_both_methods(
        spikes_iti_odd,
        iti_odd_intervals,
        split_halves=False,
        context_label="ITI odd",
    )
if len(iti_even_intervals) > 0:
    spikes_iti_even = ana_utils.slice_spikes_by_intervals(
        spikes,
        iti_even_intervals,
        exclude_intervals=spont_interval_list,
    )
    df_current_iti_even, df_okun_iti_even = _compute_both_methods(
        spikes_iti_even,
        iti_even_intervals,
        split_halves=False,
        context_label="ITI even",
    )

df_current_iti = None
df_okun_iti = None
if df_current_iti_odd is not None or df_current_iti_even is not None:
    df_current_iti = ana_utils.merge_stpr_splits(
        df_current_iti_odd,
        df_current_iti_even,
        CURRENT_CONFIG,
        split_a="odd",
        split_b="even",
    )
if df_okun_iti_odd is not None or df_okun_iti_even is not None:
    df_okun_iti = ana_utils.merge_stpr_splits(
        df_okun_iti_odd,
        df_okun_iti_even,
        OKUN_LAG_CONFIG,
        split_a="odd",
        split_b="even",
    )
df_current_iti = _empty_to_none(df_current_iti)
df_okun_iti = _empty_to_none(df_okun_iti)


# %% Comparison tables
comparison_tables = {
    "Spont": _merge_method_tables(df_current_spont, df_okun_spont),
    "ITI": _merge_method_tables(df_current_iti, df_okun_iti),
    "Task": _merge_method_tables(df_current_task, df_okun_task),
}
for context_label, df_context in comparison_tables.items():
    shape = None if df_context is None else df_context.shape
    print(f"{context_label} comparison shape: {shape}")

plot_cluster_ids = _select_plot_cluster_ids(
    comparison_tables,
    plot_n=PLOT_N_NEURONS,
    seed=PLOT_RANDOM_SEED,
    manual_ids=PLOT_CLUSTER_IDS,
)
print("Plot cluster IDs:", plot_cluster_ids)


# %% Plot
fig_comparison = plot_stpr_method_comparison(
    comparison_tables=comparison_tables,
    plot_cluster_ids=plot_cluster_ids,
    cluster_acronym_lookup=cluster_acronym_lookup,
    current_config=CURRENT_CONFIG,
    okun_config=OKUN_LAG_CONFIG,
    region_name=target_region,
    pid=pid,
    template=PLOTLY_TEMPLATE,
    fig_width=FIG_WIDTH,
    fig_height=FIG_HEIGHT,
)
show_fig(fig_comparison)
