import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse
import scipy.sparse as sp
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

def load_specific_spectrum_from_mgf(spectra, index):
    """
    MGFファイルのスペクトルデータをインデックスで取得
    """
    if index < 0 or index >= len(spectra):
        raise IndexError(f"Spectrum index {index} is out of range.")

    # `spectra` の構造が (precursor_mz, mz_values, intensity_values) である場合
    precursor_mz, mz_values, intensity_values = spectra[index]
    return mz_values, intensity_values  # 必要なデータ形式のみを返す

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
        State("umap-plot", "relayoutData")  # 現在のレイアウトデータ（拡大状態）
    )
    def update_spectra(click_data, existing_spectra, relayout_data):
        # 初期UMAPプロット
        fig = px.scatter(
            umap_df,
            x="UMAP Dimension 1",
            y="UMAP Dimension 2",
            custom_data=["Index"],  # クリックデータ用に `custom_data` を設定
            title="UMAP Plot",
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=5))

        # 初期化: `existing_spectra` が None の場合は空リストにする
        if existing_spectra is None:
            existing_spectra = []

        # UMAPの拡大状態を保持
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

        # スペクトル表示
        if click_data:
            try:
                # デバッグ用: `click_data` の内容をログ出力
                print("ClickData:", click_data)

                # クリックされたポイントのインデックスを取得
                spectrum_key = click_data["points"][0]["customdata"]
                if isinstance(spectrum_key, list):
                    spectrum_key = spectrum_key[0]

                spectrum_index = int(spectrum_key.split("_")[1]) - 1  # インデックスは0ベースに調整

                # 範囲外のインデックスを防止
                if spectrum_index < 0 or spectrum_index >= len(spectra):
                    print(f"Spectrum index {spectrum_index} is out of range.")
                    return fig, existing_spectra

                # スペクトルのロード
                mz, intensity = load_specific_spectrum_from_mgf(spectra, spectrum_index)
                print(f"Loaded spectrum: mz({len(mz)}), intensity({len(intensity)})")

                # スペクトルを描画（縦線プロット）
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
                        xaxis=dict(range=[0, mz_max])  # 横軸の範囲を設定
                    )

                    new_figure = dcc.Graph(figure=spectrum_figure)
                    existing_spectra.append(new_figure)
                    existing_spectra = existing_spectra[-3:]  # 履歴を最大3件に制限
            except Exception as e:
                print(f"Error during spectrum loading: {e}")

        return fig, existing_spectra

    return app

# メイン処理
if __name__ == "__main__":
    # ファイルパスの設定
    umap_results_path = "C:/Users/hyama/data/umap_results.npz"
    mgf_file_path = "C:/Users/hyama/data/combined_spectra.mgf"

    # UMAP結果を読み込み
    umap_df = load_umap_results(umap_results_path)
    print("UMAP DataFrame loaded:", umap_df.head())  # デバッグ用出力

    # アプリケーションの作成と実行
    app = create_umap_app(umap_df, mgf_file_path, mz_max=2000)
    app.run_server(debug=True)
