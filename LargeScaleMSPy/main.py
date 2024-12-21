from .app import create_umap_app, load_umap_results, parse_msp_to_line_info

def run_app():
    # ファイルパスの設定
    umap_results_path = "C:/Users/hyama/umap_results.npz"  # 必要に応じて調整
    msp_file_path = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"

    # 行範囲情報を生成
    line_info = parse_msp_to_line_info(msp_file_path)

    # UMAP結果を読み込み
    umap_df = load_umap_results(umap_results_path)

    # アプリケーションの作成と実行
    app = create_umap_app(umap_df, msp_file_path, line_info, mz_max=2000)
    app.run_server(debug=True)
