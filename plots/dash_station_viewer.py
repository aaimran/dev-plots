#!/usr/bin/env python3
"""Dash app to browse WaveQLab3D station time series.

Looks for datasets named like:
  <dataset>_<q>_<r>_<s>_block<id>.dat
Station name is:
  <q>_<r>_<s>_block<id>

Each file is assumed to have 4 whitespace-delimited columns:
  t  vx  vy  vz

Run:
  python dash_station_viewer.py --data-dir waveqlab3d/simulation/plots
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict


class TimeSeries(TypedDict):
    t: List[float]
    vx: List[float]
    vy: List[float]
    vz: List[float]



import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

try:
    import dash_bootstrap_components as dbc
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'dash-bootstrap-components'. "
        "Install requirements from requirements-dash.txt"
    ) from exc


FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_"  # dataset prefix
    r"(?P<q>[^_]+)_"  # q
    r"(?P<r>[^_]+)_"  # r
    r"(?P<s>[^_]+)_"  # s
    r"(?P<block>block[^.]+)"  # block...
    r"\.dat$"
)


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    dataset: str
    station: str


def iter_dataset_files(data_dir: Path) -> List[Path]:
    patterns = ["*.dat"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(p).resolve() for p in glob.glob(str(data_dir / pat)))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def parse_dataset_info(path: Path) -> Optional[DatasetInfo]:
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    station = f"{m.group('q')}_{m.group('r')}_{m.group('s')}_{m.group('block')}"
    return DatasetInfo(path=path, dataset=m.group("dataset"), station=station)


def load_timeseries(path: Path) -> TimeSeries:
    t: List[float] = []
    vx: List[float] = []
    vy: List[float] = []
    vz: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                tt, vxx, vyy, vzz = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            t.append(tt)
            vx.append(vxx)
            vy.append(vyy)
            vz.append(vzz)

    return {"t": t, "vx": vx, "vy": vy, "vz": vz}


def make_figure(station: str, selected: List[DatasetInfo], plots_selected: List[str]) -> go.Figure:
    plot_meta = {
        "vx": ("t vs vx", "vx"),
        "vy": ("t vs vy", "vy"),
        "vz": ("t vs vz", "vz"),
    }
    plots_selected = [p for p in (plots_selected or []) if p in plot_meta]

    if not plots_selected:
        fig = go.Figure()
        fig.update_layout(
            title=station or "",
            height=800,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.add_annotation(
            text="No plot selected",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    subplot_titles = tuple(plot_meta[p][0] for p in plots_selected)
    fig = make_subplots(
        rows=len(plots_selected),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    for info in selected:
        df = load_timeseries(info.path)
        label = info.dataset
        for row_idx, p in enumerate(plots_selected, start=1):
            showlegend = True
            fig.add_trace(
                go.Scatter(
                    x=df["t"],
                    y=df[p],
                    mode="lines",
                    name=label,
                    legendgroup=f"{p}:{label}",
                    showlegend=showlegend,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        title=station or "",
        height=800,
        margin=dict(l=40, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    fig.update_xaxes(title_text="t", row=len(plots_selected), col=1)
    for row_idx, p in enumerate(plots_selected, start=1):
        fig.update_yaxes(title_text=plot_meta[p][1], row=row_idx, col=1)
    return fig


def build_index(data_dir: Path) -> Tuple[List[DatasetInfo], List[str]]:
    infos: List[DatasetInfo] = []
    for path in iter_dataset_files(data_dir):
        info = parse_dataset_info(path)
        if info is not None:
            infos.append(info)

    stations = sorted({i.station for i in infos})
    return infos, stations


def group_by_station(infos: Iterable[DatasetInfo]) -> Dict[str, List[DatasetInfo]]:
    out: Dict[str, List[DatasetInfo]] = {}
    for i in infos:
        out.setdefault(i.station, []).append(i)
    for st in out:
        out[st].sort(key=lambda x: x.dataset)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing station .dat files",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        candidates = [
            Path.cwd() / "waveqlab3d/simulation/plots",
            script_dir / "waveqlab3d/simulation/plots",
            script_dir,
        ]
        data_dir = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

    if not data_dir.exists():
        raise SystemExit(
            "Data directory not found. Try: "
            "python dash_station_viewer.py --data-dir /path/to/plots\n"
            f"Tried: {data_dir}"
        )

    all_infos, stations = build_index(data_dir)
    by_station = group_by_station(all_infos)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    initial_station = stations[0] if stations else ""
    initial_selected_paths = [str(i.path) for i in by_station.get(initial_station, [])]
    initial_plots = ["vx", "vy", "vz"]

    app.layout = dbc.Container(
        [
            html.H3("Station time series viewer"),
            dbc.Badge(f"Data dir: {data_dir}", color="secondary", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Station"),
                            dcc.Dropdown(
                                id="station-dropdown",
                                options=[{"label": s, "value": s} for s in stations],
                                value=initial_station,
                                multi=False,
                                clearable=False,
                                placeholder="Choose station",
                            ),
                            html.Hr(),
                            html.Label("Datasets"),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Select all", id="dataset-select-all", size="sm", color="secondary"),
                                    dbc.Button("Clear", id="dataset-clear", size="sm", color="secondary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dcc.Checklist(
                                id="dataset-checklist",
                                options=[],
                                value=initial_selected_paths,
                                labelStyle={"display": "block"},
                            ),
                            html.Hr(),
                            html.Label("Select plot"),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Select all", id="plot-select-all", size="sm", color="secondary"),
                                    dbc.Button("Clear", id="plot-clear", size="sm", color="secondary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dcc.Checklist(
                                id="plot-checklist",
                                options=[
                                    {"label": "t vs vx", "value": "vx"},
                                    {"label": "t vs vy", "value": "vy"},
                                    {"label": "t vs vz", "value": "vz"},
                                ],
                                value=initial_plots,
                                labelStyle={"display": "block"},
                            ),
                        ],
                        style={"flex": "0 0 10%", "maxWidth": "10%"},
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="main-plot",
                                figure=make_figure(initial_station, [], initial_plots),
                            )
                        ],
                        style={"flex": "0 0 90%", "maxWidth": "90%"},
                    ),
                ],
                align="start",
            ),
        ],
        fluid=True,
        className="p-3",
    )

    @app.callback(
        [Output("dataset-checklist", "options"), Output("dataset-checklist", "value")],
        Input("station-dropdown", "value"),
    )
    def update_dataset_checklist(selected_station: str):
        infos = by_station.get(selected_station or "", [])
        options = [
            {
                "label": i.dataset,
                "value": str(i.path),
            }
            for i in infos
        ]
        values = [str(i.path) for i in infos]
        return options, values

    @app.callback(
        Output("dataset-checklist", "value"),
        [Input("dataset-select-all", "n_clicks"), Input("dataset-clear", "n_clicks")],
        State("dataset-checklist", "options"),
        prevent_initial_call=True,
    )
    def dataset_select_clear(_select_all: int, _clear: int, options: List[dict]):
        triggered = callback_context.triggered
        if not triggered:
            raise PreventUpdate
        trig_id = triggered[0]["prop_id"].split(".")[0]
        if trig_id == "dataset-clear":
            return []
        # select all
        return [o["value"] for o in (options or []) if "value" in o]

    @app.callback(
        Output("main-plot", "figure"),
        [Input("station-dropdown", "value"), Input("dataset-checklist", "value"), Input("plot-checklist", "value")],
    )
    def update_plot(station: str, selected_paths: List[str], plots_selected: List[str]):
        selected_infos: List[DatasetInfo] = []
        for p in selected_paths or []:
            info = parse_dataset_info(Path(p))
            if info is not None:
                selected_infos.append(info)
        return make_figure(station or "", selected_infos, plots_selected or [])

    @app.callback(
        Output("plot-checklist", "value"),
        [Input("plot-select-all", "n_clicks"), Input("plot-clear", "n_clicks")],
        State("plot-checklist", "options"),
        prevent_initial_call=True,
    )
    def plot_select_clear(_select_all: int, _clear: int, options: List[dict]):
        triggered = callback_context.triggered
        if not triggered:
            raise PreventUpdate
        trig_id = triggered[0]["prop_id"].split(".")[0]
        if trig_id == "plot-clear":
            return []
        return [o["value"] for o in (options or []) if "value" in o]

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
