import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from pyteomics import mgf


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


def load_specific_spectrum(mgf_file_path, spectrum_index):
    """
    MGFファイルから特定のスペクトルをロード
    """
    with mgf.read(mgf_file_path) as reader:
        for i, spectrum in enumerate(reader):
            if i == spectrum_index:
                mz = spectrum['m/z array']
                intensity = spectrum['intensity array']
                return mz, intensity
    return None, None


def create_umap_app(umap_df, mgf_file_path):
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
            mz, intensity = load_specific_spectrum(mgf_file_path, index)

            if mz is not None and intensity is not None:
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

    # アプリケーションの作成と実行
    app = create_umap_app(umap_df, mgf_file_path)
    app.run_server(debug=True)
