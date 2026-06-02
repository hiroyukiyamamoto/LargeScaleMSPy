from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


UMAP_FILE = Path(
    r"C:\Users\hyama\Documents\LargeScaleMSPy\release2\demo_output\full_msdial_umap_idf_noautoscale.npz"
)
MSP_FILE = Path(
    r"C:\Users\hyama\Documents\R\MSinfoR\MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
)
MZ_MAX = 2000


def parse_msp_metadata(msp_file):
    line_info = []
    names = []
    classes = []
    start_line = None
    current_name = ""
    current_class = "Unknown"

    with open(msp_file, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text.startswith("NAME:"):
                if start_line is not None:
                    line_info.append((start_line, i - 1))
                    names.append(current_name)
                    classes.append(current_class)
                start_line = i
                current_name = text.split(":", 1)[1].strip()
                current_class = "Unknown"
            elif text.startswith("COMPOUNDCLASS:"):
                current_class = text.split(":", 1)[1].strip() or "Unknown"

    if start_line is not None:
        line_info.append((start_line, i))
        names.append(current_name)
        classes.append(current_class)

    return line_info, names, classes


def spectrum_key_to_index(spectrum_key):
    return int(str(spectrum_key).split("_", 1)[1]) - 1


def load_umap_with_classes(umap_file, msp_file):
    data = np.load(umap_file)
    umap = data["umap_results"]
    valid_keys = data["valid_keys"]

    line_info, names, classes = parse_msp_metadata(msp_file)
    indices = np.array([spectrum_key_to_index(key) for key in valid_keys], dtype=int)

    class_values = [
        classes[idx] if 0 <= idx < len(classes) else "Unknown"
        for idx in indices
    ]
    name_values = [
        names[idx] if 0 <= idx < len(names) else ""
        for idx in indices
    ]

    df = pd.DataFrame(
        {
            "UMAP_1": umap[:, 0],
            "UMAP_2": umap[:, 1],
            "Spectrum_Key": valid_keys,
            "Spectrum_Index": indices,
            "Name": name_values,
            "Lipid_Class": class_values,
        }
    )
    return df, line_info


def read_partial_msp(file_path, start_line, end_line):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[start_line : end_line + 1]


def load_specific_spectrum_from_msp(msp_file, start_line, end_line):
    mz = []
    intensity = []
    for line in read_partial_msp(msp_file, start_line, end_line):
        text = line.strip()
        if not text or ":" in text:
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        try:
            mz.append(float(parts[0]))
            intensity.append(float(parts[1]))
        except ValueError:
            continue
    return mz, intensity


def make_umap_figure(df, selected_classes=None):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df["UMAP_1"],
            y=df["UMAP_2"],
            mode="markers",
            name="All spectra",
            marker={"size": 2, "opacity": 0.12, "color": "lightgray"},
            customdata=np.stack(
                [
                    df["Spectrum_Key"],
                    df["Spectrum_Index"],
                    df["Name"],
                    df["Lipid_Class"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "key=%{customdata[0]}<br>"
                "name=%{customdata[2]}<br>"
                "class=%{customdata[3]}<extra></extra>"
            ),
        )
    )

    if selected_classes:
        highlight_df = df[df["Lipid_Class"].isin(selected_classes)]
    else:
        highlight_df = df.iloc[0:0]

    for lipid_class, group in highlight_df.groupby("Lipid_Class", sort=True):
        fig.add_trace(
            go.Scattergl(
                x=group["UMAP_1"],
                y=group["UMAP_2"],
                mode="markers",
                name=lipid_class,
                marker={"size": 4, "opacity": 0.9},
                customdata=np.stack(
                    [
                        group["Spectrum_Key"],
                        group["Spectrum_Index"],
                        group["Name"],
                        group["Lipid_Class"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "key=%{customdata[0]}<br>"
                    "name=%{customdata[2]}<br>"
                    "class=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="UMAP lipid class highlight",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Lipid class",
        margin={"l": 40, "r": 10, "t": 50, "b": 40},
    )
    return fig


def create_app(df, line_info, msp_file):
    app = Dash(__name__)
    lipid_classes = sorted(df["Lipid_Class"].dropna().unique())

    app.layout = html.Div(
        style={"display": "flex", "height": "100vh"},
        children=[
            html.Div(
                style={"flex": "1.25", "padding": "10px"},
                children=[
                    html.H2("UMAP by Lipid Class"),
                    dcc.Dropdown(
                        id="class-filter",
                        options=[{"label": x, "value": x} for x in lipid_classes],
                        value=[],
                        multi=True,
                        placeholder="Select lipid classes to highlight",
                    ),
                    dcc.Graph(
                        id="umap-plot",
                        style={"height": "88vh"},
                        config={"scrollZoom": True},
                    ),
                ],
            ),
            html.Div(
                style={"flex": "1", "padding": "10px", "overflowY": "auto"},
                children=[
                    html.H2("MS/MS Spectra"),
                    html.Div(id="spectra-container"),
                ],
            ),
        ],
    )

    @app.callback(
        Output("umap-plot", "figure"),
        Input("class-filter", "value"),
    )
    def update_umap(selected_classes):
        return make_umap_figure(df, selected_classes)

    @app.callback(
        Output("spectra-container", "children"),
        Input("umap-plot", "clickData"),
        State("spectra-container", "children"),
    )
    def update_spectrum(click_data, existing_spectra):
        if existing_spectra is None:
            existing_spectra = []
        if not click_data:
            return existing_spectra

        point = click_data["points"][0]
        spectrum_key, spectrum_index, name, lipid_class = point["customdata"]
        spectrum_index = int(spectrum_index)

        if spectrum_index < 0 or spectrum_index >= len(line_info):
            return existing_spectra

        start_line, end_line = line_info[spectrum_index]
        mz, intensity = load_specific_spectrum_from_msp(msp_file, start_line, end_line)
        if not mz:
            return existing_spectra

        fig = go.Figure()
        for mz_value, intensity_value in zip(mz, intensity):
            fig.add_trace(
                go.Scatter(
                    x=[mz_value, mz_value],
                    y=[0, intensity_value],
                    mode="lines",
                    line={"color": "royalblue"},
                    showlegend=False,
                )
            )
        fig.update_layout(
            title=f"{spectrum_key} | {lipid_class} | {name}",
            xaxis_title="m/z",
            yaxis_title="Intensity",
            xaxis={"range": [0, MZ_MAX]},
            height=320,
            margin={"l": 40, "r": 10, "t": 60, "b": 40},
        )

        existing_spectra.append(dcc.Graph(figure=fig))
        return existing_spectra[-3:]

    return app


df, line_info = load_umap_with_classes(UMAP_FILE, MSP_FILE)
print(df["Lipid_Class"].value_counts().head(20))
print("Dash app: http://127.0.0.1:8050/")
app = create_app(df, line_info, str(MSP_FILE))
app.run(debug=False)
