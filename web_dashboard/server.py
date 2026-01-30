# %%
import json
import pickle
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import plotly.io as pio

BASE_PATH = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CACHE_DIR = BASE_PATH / "data" / "dashboard_cache"

sys.path.insert(0, str(BASE_PATH))

from utils.plotting_plotly import (  # noqa: E402
    plot_trial_raster_plotly,
    plot_time_window_raster_plotly,
    plot_population_sorted_plotly,
    plot_population_coupling_heatmap_plotly,
    plot_coupling_strength_summary_plotly,
    plot_coupling_delay_summary_plotly,
    plot_single_neuron_plotly,
    plot_single_neuron_conditioned_event_plotly,
    plot_stpr_curve_halves_plotly,
)


DATA_CACHE = {}
FIG_CACHE = {}


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _list_pids():
    if not CACHE_DIR.exists():
        return []
    return sorted([p.stem for p in CACHE_DIR.glob("*.pkl")])


def _load_pid(pid):
    if pid in DATA_CACHE:
        return DATA_CACHE[pid]
    path = CACHE_DIR / f"{pid}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    DATA_CACHE[pid] = data
    return data


def _get_label_array(clusters):
    if hasattr(clusters, "metrics") and hasattr(clusters.metrics, "columns"):
        if "label" in clusters.metrics.columns:
            return np.asarray(clusters.metrics.label)
    if hasattr(clusters, "label"):
        return np.asarray(clusters.label)
    if isinstance(clusters, dict) and "label" in clusters:
        return np.asarray(clusters.get("label"))
    return None


def _format_seconds(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    return f"{val:.2f}s"


def _spont_interval_text(interval):
    if interval is None:
        return "NA"
    start, end = interval
    if start is None or end is None:
        return "NA"
    return f"{start:.2f}-{end:.2f}s"


def _build_region_table(cluster_acronyms, labels):
    cluster_acronyms = np.asarray(cluster_acronyms).astype(str)
    all_counts = pd.Series(cluster_acronyms).value_counts().sort_index()
    if labels is not None:
        good_counts = pd.Series(cluster_acronyms[labels == 1]).value_counts().sort_index()
    else:
        good_counts = pd.Series(dtype=int)
    region_table = (
        pd.DataFrame({"region": all_counts.index, "all": all_counts.values})
        .merge(
            pd.DataFrame({"region": good_counts.index, "good": good_counts.values}),
            on="region",
            how="left",
        )
        .fillna(0)
    )
    region_table["all"] = region_table["all"].astype(int)
    region_table["good"] = region_table["good"].astype(int)
    return region_table


def _fig_to_dict(fig):
    return json.loads(pio.to_json(fig, pretty=False))


def _parse_bool(val, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")


def _get_plot_config(data, plot_only_good):
    config_plot = dict(data.get("config_plot", {}))
    config_calc = data.get("config_calc", {})
    config_plot["PLOT_ONLY_GOOD_UNITS"] = plot_only_good
    config_plot["PSTH_WINDOW_START"] = config_calc.get("PSTH_WINDOW_START", -0.2)
    config_plot["PSTH_WINDOW_END"] = config_calc.get("PSTH_WINDOW_END", 0.35)
    config_plot["TRIAL_RASTER_USE_EVENT_WINDOW"] = True
    config_plot["PLOTLY_TEMPLATE"] = "plotly_white"
    return config_plot, config_calc


def _filter_by_good(data, plot_only_good):
    df_coupling = data.get("df_coupling")
    df_coupling_task = data.get("df_coupling_task")
    df_comparison = data.get("df_comparison")
    if not plot_only_good:
        return df_coupling, df_coupling_task, df_comparison

    cluster_ids = data.get("cluster_ids")
    labels = _get_label_array(data.get("clusters"))
    if labels is None:
        return df_coupling, df_coupling_task, df_comparison
    good_cluster_ids = np.asarray(cluster_ids)[labels == 1]

    if df_coupling is not None:
        df_coupling = df_coupling[df_coupling["cluster_id"].isin(good_cluster_ids)]
    if df_coupling_task is not None:
        df_coupling_task = df_coupling_task[
            df_coupling_task["cluster_id"].isin(good_cluster_ids)
        ]
    if df_comparison is not None:
        df_comparison = df_comparison[
            df_comparison["cluster_id"].isin(good_cluster_ids)
        ]
    return df_coupling, df_coupling_task, df_comparison


def _get_cached_fig(key, builder):
    if key in FIG_CACHE:
        return FIG_CACHE[key]
    FIG_CACHE[key] = builder()
    return FIG_CACHE[key]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self._handle_api(parsed)
        return super().do_GET()

    def _handle_api(self, parsed):
        params = parse_qs(parsed.query)
        pid = params.get("pid", [None])[0]

        if parsed.path == "/api/pids":
            return self._send_json({"pids": _list_pids()})

        if parsed.path == "/api/session":
            if pid is None:
                return self._send_json({"error": "pid required"}, status=400)
            data = _load_pid(pid)
            clusters = data.get("clusters")
            labels = _get_label_array(clusters)
            region_table = _build_region_table(data.get("cluster_acronyms_plot"), labels)
            spikes = data.get("spikes")
            min_time = float(np.nanmin(spikes["times"])) if spikes is not None else 0.0
            max_time = float(np.nanmax(spikes["times"])) if spikes is not None else 0.0
            meta = data.get("meta", {})
            info = {
                "Lab": meta.get("lab"),
                "Num trials": meta.get("num_trials"),
                "PID": meta.get("pid"),
                "EID": meta.get("eid"),
                "PIDs Numbers in this session": meta.get("num_other_pids"),
                "Date": meta.get("date"),
                "Recording length": _format_seconds(meta.get("recording_length_s")),
                "Spont length": _format_seconds(meta.get("spont_length_s")),
                "Subject": meta.get("subject"),
                "Spont interval": _spont_interval_text(meta.get("spont_interval")),
            }
            return self._send_json(
                {
                    "info": info,
                    "region_table": region_table.to_dict(orient="records"),
                    "cluster_ids": data.get("cluster_ids"),
                    "cluster_acronyms": data.get("cluster_acronyms_plot"),
                    "labels": labels.tolist() if labels is not None else None,
                    "trial_idx": data.get("trials")["trial_idx"].tolist()
                    if data.get("trials") is not None
                    else [],
                    "min_time": min_time,
                    "max_time": max_time,
                }
            )

        if parsed.path.startswith("/api/fig/"):
            if pid is None:
                return self._send_json({"error": "pid required"}, status=400)
            data = _load_pid(pid)
            plot_only_good = _parse_bool(params.get("plot_only_good", [None])[0], False)
            variability = params.get("variability", [None])[0] or "fano"
            sort = params.get("sort", [None])[0] or "depth"
            trial_idx = params.get("trial_idx", [None])[0]
            cluster_id = params.get("cluster_id", [None])[0]
            mode = params.get("mode", [None])[0]

            plot_config, config_calc = _get_plot_config(data, plot_only_good)
            df_coupling, df_coupling_task, df_comparison = _filter_by_good(data, plot_only_good)
            session = data.get("session")
            spikes = data.get("spikes")
            clusters = data.get("clusters")
            cluster_ids = data.get("cluster_ids")
            cluster_acronyms = data.get("cluster_acronyms_plot")

            if parsed.path == "/api/fig/general_raster":
                t_start = float(params.get("t_start", [0])[0])
                t_end = float(params.get("t_end", [0])[0])
                key = ("general", pid, t_start, t_end, sort, plot_only_good, variability)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_time_window_raster_plotly(
                        spikes,
                        clusters,
                        cluster_ids,
                        cluster_acronyms,
                        session,
                        plot_config,
                        t_start,
                        t_end,
                        sorting_metric=sort,
                        variability_metric=variability,
                        df_res=data.get("df_res"),
                        df_coupling=df_coupling,
                        df_coupling_task=df_coupling_task,
                        pupil_features=data.get("pupil_features"),
                        pupil_times=data.get("pupil_times"),
                        region_colors=None,
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/trial_raster":
                if trial_idx is None:
                    return self._send_json({"error": "trial_idx required"}, status=400)
                trial_idx = int(trial_idx)
                key = ("trial", pid, trial_idx, sort, plot_only_good, variability)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_trial_raster_plotly(
                        spikes,
                        clusters,
                        cluster_ids,
                        cluster_acronyms,
                        session,
                        plot_config,
                        trial_idx,
                        sorting_metric=sort,
                        variability_metric=variability,
                        df_res=data.get("df_res"),
                        df_coupling=df_coupling,
                        df_coupling_task=df_coupling_task,
                        pupil_features=data.get("pupil_features"),
                        pupil_times=data.get("pupil_times"),
                        region_colors=None,
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/population":
                key = ("population", pid, sort, plot_only_good)
                figs = _get_cached_fig(
                    key,
                    lambda: [
                        plot_population_sorted_plotly(
                            session,
                            spikes,
                            clusters,
                            cluster_ids,
                            cluster_acronyms,
                            data.get("df_res"),
                            {**plot_config, "PLOT_EVENT": event_name},
                            df_coupling=df_coupling,
                            df_coupling_task=df_coupling_task,
                            region_acronyms=plot_config.get("PLOT_REGIONS"),
                            sort_mode=sort,
                        )
                        for event_name in ["stimOn_times", "firstMovement_times", "feedback_times"]
                    ],
                )
                return self._send_json({"figs": [_fig_to_dict(fig) for fig in figs]})

            if parsed.path == "/api/fig/coupling":
                key = ("coupling", pid, plot_only_good)
                fig_spont, fig_task = _get_cached_fig(
                    key,
                    lambda: (
                        plot_population_coupling_heatmap_plotly(
                            df_coupling,
                            plot_config,
                            config_calc,
                            region_acronyms=plot_config.get("PLOT_REGIONS"),
                        ),
                        plot_population_coupling_heatmap_plotly(
                            df_coupling_task,
                            plot_config,
                            config_calc,
                            region_acronyms=plot_config.get("PLOT_REGIONS"),
                        ),
                    ),
                )
                return self._send_json(
                    {"spont": _fig_to_dict(fig_spont), "task": _fig_to_dict(fig_task)}
                )

            if parsed.path == "/api/fig/stpr_strength":
                if cluster_id is None:
                    return self._send_json({"error": "cluster_id required"}, status=400)
                cluster_id = int(cluster_id)
                key = ("strength", pid, cluster_id, plot_only_good)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_coupling_strength_summary_plotly(
                        df_comparison,
                        region_order=df_comparison["region"].unique().tolist()
                        if df_comparison is not None
                        else None,
                        region_colors=None,
                        template=plot_config["PLOTLY_TEMPLATE"],
                        highlight_cluster_id=cluster_id,
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/stpr_delay":
                if cluster_id is None:
                    return self._send_json({"error": "cluster_id required"}, status=400)
                cluster_id = int(cluster_id)
                key = ("delay", pid, cluster_id, plot_only_good)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_coupling_delay_summary_plotly(
                        df_comparison,
                        region_order=df_comparison["region"].unique().tolist()
                        if df_comparison is not None
                        else None,
                        region_colors=None,
                        template=plot_config["PLOTLY_TEMPLATE"],
                        highlight_cluster_id=cluster_id,
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/single_stim":
                if cluster_id is None:
                    return self._send_json({"error": "cluster_id required"}, status=400)
                cluster_id = int(cluster_id)
                key = ("single_stim", pid, cluster_id)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_single_neuron_plotly(
                        session,
                        spikes,
                        clusters,
                        cluster_ids,
                        cluster_acronyms,
                        data.get("df_res"),
                        plot_config,
                        cluster_id,
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/single_move":
                if cluster_id is None:
                    return self._send_json({"error": "cluster_id required"}, status=400)
                cluster_id = int(cluster_id)
                key = ("single_move", pid, cluster_id)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_single_neuron_conditioned_event_plotly(
                        session,
                        spikes,
                        clusters,
                        cluster_ids,
                        cluster_acronyms,
                        data.get("df_res"),
                        plot_config,
                        cluster_id,
                        event_name="firstMovement_times",
                        condition_type="choice",
                        title="First Movement Response",
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/single_feedback":
                if cluster_id is None:
                    return self._send_json({"error": "cluster_id required"}, status=400)
                cluster_id = int(cluster_id)
                key = ("single_feedback", pid, cluster_id)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_single_neuron_conditioned_event_plotly(
                        session,
                        spikes,
                        clusters,
                        cluster_ids,
                        cluster_acronyms,
                        data.get("df_res"),
                        plot_config,
                        cluster_id,
                        event_name="feedback_times",
                        condition_type="feedback",
                        title="Feedback Response",
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

            if parsed.path == "/api/fig/stpr_curve":
                if cluster_id is None or mode is None:
                    return self._send_json({"error": "cluster_id and mode required"}, status=400)
                cluster_id = int(cluster_id)
                df_coupling_sel = df_coupling_task if mode == "task" else df_coupling
                key = ("stpr_curve", pid, cluster_id, mode, plot_only_good)
                fig = _get_cached_fig(
                    key,
                    lambda: plot_stpr_curve_halves_plotly(
                        df_coupling_sel,
                        config_calc,
                        cluster_id,
                        title=f"{mode.title()} stPR Curve (First vs Second Half)",
                        template=plot_config["PLOTLY_TEMPLATE"],
                    ),
                )
                return self._send_json(_fig_to_dict(fig))

        return self._send_json({"error": "not found"}, status=404)


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
