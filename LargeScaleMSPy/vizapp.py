import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np


def load_umap_results(umap_file_path):
    """
    UMAP結果を読み込む
    """
    umap_results = np.load(umap_file_path)
    umap_data = umap_results["umap_results"]
    valid_keys = umap_results["valid_keys"]
    umap_df = pd.DataFrame(umap_data, columns=["UMAP Dimension 1", "UMAP Dimension 2"])
    umap_df["Index"] = valid_keys
    return umap_df


def parse_msp_to_line_info(msp_file_path):
    """
    MSPファイルを解析して各スペクトルの行範囲を特定
    """
    line_info = []
    start_line = None

    with open(msp_file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("NAME:"):
                if start_line is not None:
                    line_info.append((start_line, i - 1))
                start_line = i
        if start_line is not None:
            line_info.append((start_line, i))

    return line_info


def read_partial_msp(file_path, start_line, end_line):
    """
    MSPファイルの特定の行範囲を部分的に読み込む。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[start_line:end_line + 1]
    return lines


def load_specific_spectrum_from_msp(msp_file_path, start_line, end_line):
    """
    MSPファイルから特定のMS/MSスペクトルをロード
    """
    lines = read_partial_msp(msp_file_path, start_line, end_line)
    mz = []
    intensity = []

    for line in lines:
        line = line.strip()
        if line and not line.startswith("NAME") and not line.startswith("PRECURSOR") and not line.startswith("Num Peaks"):
            try:
                mz_val, intensity_val = map(float, line.split())
                mz.append(mz_val)
                intensity.append(intensity_val)
            except ValueError:
                continue
    return mz, intensity


def create_umap_app(umap_df, msp_file_path, line_info, mz_max=2000):
    """
    Dashアプリケーションを作成
    """
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
                # クリックされたポイントのカスタムデータを取得
                customdata = click_data["points"][0]["customdata"]
                if isinstance(customdata, list):
                    spectrum_key = customdata[0]
                else:
                    spectrum_key = customdata

                # デバッグ用: Spectrum Key
                print("Spectrum Key:", spectrum_key)

                # インデックスの取得
                spectrum_index = int(spectrum_key.split("_")[1]) - 1

                if spectrum_index >= len(line_info) or spectrum_index < 0:
                    print(f"Spectrum index {spectrum_index} is out of range.")
                    return fig, existing_spectra

                start_line, end_line = line_info[spectrum_index]
                mz, intensity = load_specific_spectrum_from_msp(msp_file_path, start_line, end_line)

                if mz and intensity:
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
    umap_results_path = "C:/Users/hyama/Documents/LargeScaleMSPy/umap_results.npz"
    msp_file_path = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"

    line_info = parse_msp_to_line_info(msp_file_path)
    umap_df = load_umap_results(umap_results_path)

    app = create_umap_app(umap_df, msp_file_path, line_info, mz_max=2000)
    app.run_server(debug=True)
