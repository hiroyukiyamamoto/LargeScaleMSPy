import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from pyopenms import MSExperiment, FileHandler


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


def load_mgf_spectra(mgf_file_path):
    """
    MGFファイルからMS/MSスペクトルを読み込む
    """
    exp = MSExperiment()
    handler = FileHandler()
    handler.loadExperiment(mgf_file_path, exp)
    spectra = []
    for spec in exp:
        mz = spec.get_peaks()[0]
        intensity = spec.get_peaks()[1]
        spectra.append({"mz": mz, "intensity": intensity})
    return spectra


def create_umap_app(umap_df, spectra):
    """
    Dashアプリケーションを作成
    """
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("UMAP Plot with MS/MS Viewer"),
        dcc.Graph(
            id="umap-plot",
            style={"height": "70vh"},
            config={"scrollZoom": True}
        ),
        html.Div(id="spectra-container", children=[], style={"marginTop": "20px"})
    ])

    @app.callback(
        Output("umap-plot", "figure"),
        Output("spectra-container", "children"),
        Input("umap-plot", "clickData"),
        State("spectra-container", "children"),
    )
    def update_spectra(click_data, existing_spectra):
        # 初期UMAPプロット
        fig = px.scatter(
            umap_df, 
            x="UMAP Dimension 1", 
            y="UMAP Dimension 2",
            hover_data=["Index"],
            title="UMAP Plot",
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=5))

        # スペクトル表示
        if click_data:
            index = click_data["points"][0]["customdata"]
            mz = spectra[index]["mz"]
            intensity = spectra[index]["intensity"]

            new_figure = dcc.Graph(
                figure=go.Figure(
                    data=go.Scatter(x=mz, y=intensity, mode="lines"),
                    layout=go.Layout(
                        title=f"MS/MS Spectrum for Index {index}",
                        xaxis_title="m/z",
                        yaxis_title="Intensity"
                    )
                )
            )
            existing_spectra.append(new_figure)
            existing_spectra = existing_spectra[-3:]  # 履歴を最大3件に制限

        return fig, existing_spectra

    return app


# メイン処理
if __name__ == "__main__":
    umap_results_path = "umap_results.npz"  # UMAP結果の保存ファイル
    mgf_file_path = "your_data.mgf"         # MS/MSスペクトルのMGFファイル

    # データ読み込み
    umap_df = load_umap_results(umap_results_path)
    spectra = load_mgf_spectra(mgf_file_path)

    # アプリケーションの作成と実行
    app = create_umap_app(umap_df, spectra)
    app.run_server(debug=True)
