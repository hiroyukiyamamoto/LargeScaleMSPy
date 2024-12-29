import os
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# 特定のスペクトルデータを取得する関数
def get_spectrum_from_mgf(mgf_file_path, index):
    """
    MGFファイルから指定されたインデックスのスペクトルを動的に読み込む
    """
    current_index = -1
    precursor_mz = None
    mz_values = []
    intensity_values = []

    with open(mgf_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                current_index += 1
                precursor_mz = None
                mz_values = []
                intensity_values = []
            elif line.startswith("PEPMASS"):
                precursor_mz = float(line.split("=")[1])
            elif line.startswith("END IONS"):
                if current_index == index and precursor_mz is not None:
                    return np.array(mz_values), np.array(intensity_values)
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        continue

    raise IndexError(f"Spectrum index {index} is out of range.")

# Dashアプリケーションの作成
def create_umap_mgf_app(output_file, mgf_file_path, mz_max=2000):
    """
    Dashアプリケーションを作成
    """
    # UMAP結果を読み込んでデータフレームを作成
    umap_results = np.load(output_file)
    umap_df = pd.DataFrame(
        umap_results["umap_results"],
        columns=["UMAP Dimension 1", "UMAP Dimension 2"]
    )
    umap_df["Index"] = umap_results["valid_keys"]

    # Dashアプリケーションの設定
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

                # 必要なスペクトルを動的に取得
                mz, intensity = get_spectrum_from_mgf(mgf_file_path, spectrum_index)
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


## メイン処理
#if __name__ == "__main__":
#    umap_results_path = "C:/Users/hyama/data/umap_results.npz"
#    mgf_file_path = "C:/Users/hyama/data/combined_spectra.mgf"
#    mz_max = 2000
#
#    # create_umap_app にパスを渡してアプリを作成
#    app = create_umap_mgf_app(umap_results_path, mgf_file_path, mz_max)
#    app.run_server(debug=True)
