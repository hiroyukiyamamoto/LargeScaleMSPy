import os
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# MGFファイルの読み込み関数
def parse_mgf_to_spectra(mgf_file_path):
    """
    MGFファイルを解析して各スペクトルの情報を取得
    """
    spectra = []
    with open(mgf_file_path, 'r') as f:
        precursor_mz = None
        mz_values = []
        intensity_values = []
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                precursor_mz = None
                mz_values = []
                intensity_values = []
            elif line.startswith("PEPMASS"):
                precursor_mz = float(line.split("=")[1])
            elif line.startswith("END IONS"):
                if precursor_mz is not None and mz_values:
                    spectra.append((precursor_mz, np.array(mz_values), np.array(intensity_values)))
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        continue
    return spectra

# 特定のスペクトルデータを取得する関数
def load_specific_spectrum_from_mgf(spectra, index):
    """
    MGFファイルのスペクトルデータをインデックスで取得
    """
    if index < 0 or index >= len(spectra):
        raise IndexError(f"Spectrum index {index} is out of range.")

    precursor_mz, mz_values, intensity_values = spectra[index]
    return mz_values, intensity_values

# Dashアプリケーションの作成
def create_umap_app(umap_df, mgf_file_path, mz_max=2000):
    """
    Dashアプリケーションを作成
    """
    spectra = parse_mgf_to_spectra(mgf_file_path)

    app = Dash(__name__)
    app.layout = html.Div(style={'display': 'flex'}, children=[
        html.Div(
            style={'flex': '1', 'padding': '10px'},
            children=[
                html.H1("UMAP Plot with MS/MS Viewer"),
                dcc.Graph(
                    id="umap-plot",
                    style={"height": "90vh"},
                    config={"scrollZoom": True}
                )
            ]
        ),
        html.Div(
            style={'flex': '1', 'padding': '10px'},
            children=[
                html.H2("MS/MS Spectra"),
                html.Div(id="spectra-container", style={"height": "90vh", "overflowY": "scroll"})
            ]
        )
    ])

    @app.callback(
        Output("umap-plot", "figure"),
        Output("spectra-container", "children"),
        Input("umap-plot", "clickData"),
        State("spectra-container", "children"),
        State("umap-plot", "relayoutData")
    )
    def update_spectra(click_data, existing_spectra, relayout_data):
        # 初期UMAPプロット
        fig = px.scatter(
            umap_df,
            x="UMAP Dimension 1",
            y="UMAP Dimension 2",
            custom_data=["Index"],
            title="UMAP Plot",
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=5))

        if existing_spectra is None:
            existing_spectra = []

        if relayout_data:
            for key in ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]:
                if key in relayout_data:
                    fig.update_layout(
                        xaxis_range=[
                            relayout_data.get("xaxis.range[0]", None),
                            relayout_data.get("xaxis.range[1]", None)
                        ],
                        yaxis_range=[
                            relayout_data.get("yaxis.range[0]", None),
                            relayout_data.get("yaxis.range[1]", None)
                        ]
                    )

        if click_data:
            try:
                spectrum_key = click_data["points"][0]["customdata"]
                if isinstance(spectrum_key, list):
                    spectrum_key = spectrum_key[0]

                spectrum_index = int(spectrum_key.split("_")[1]) - 1

                if spectrum_index < 0 or spectrum_index >= len(spectra):
                    print(f"Spectrum index {spectrum_index} is out of range.")
                    return fig, existing_spectra

                mz, intensity = load_specific_spectrum_from_mgf(spectra, spectrum_index)
                print(f"Loaded spectrum: mz({len(mz)}), intensity({len(intensity)})")

                if len(mz) > 0 and len(intensity) > 0:
                    spectrum_figure = go.Figure()
                    for mz_val, intensity_val in zip(mz, intensity):
                        spectrum_figure.add_trace(go.Scatter(
                            x=[mz_val, mz_val], y=[0, intensity_val],
                            mode="lines",
                            line=dict(color="blue"),
                            showlegend=False
                        ))
                    spectrum_figure.update_layout(
                        title=f"MS/MS Spectrum for {spectrum_key}",
                        xaxis_title="m/z",
                        yaxis_title="Intensity",
                        xaxis=dict(range=[0, mz_max])
                    )

                    new_figure = dcc.Graph(figure=spectrum_figure)
                    existing_spectra.append(new_figure)
                    existing_spectra = existing_spectra[-3:]
            except Exception as e:
                print(f"Error during spectrum loading: {e}")

        return fig, existing_spectra

    return app

# メイン処理
if __name__ == "__main__":
    umap_results_path = "C:/Users/hyama/data/umap_results.npz"
    mgf_file_path = "C:/Users/hyama/data/combined_spectra.mgf"

    umap_results = np.load(umap_results_path)
    umap_df = pd.DataFrame(
        umap_results["umap_results"],
        columns=["UMAP Dimension 1", "UMAP Dimension 2"]
    )
    umap_df["Index"] = umap_results["valid_keys"]

    app = create_umap_app(umap_df, mgf_file_path, mz_max=2000)
    app.run_server(debug=True)
