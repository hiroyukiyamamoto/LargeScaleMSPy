import os
from LargeScaleMSPy import *

# デモ解析用の設定
msp_file = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
mz_max = 2000       

# HDF5ファイルに保存
hdf5_file = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"
msp2hdf5(msp_file, hdf5_file, bin_size=0.01, mz_range=(0, 2000))

# 各ステップを実行し、データを次に渡す
output_file = "C:/Users/hyama/Documents/LargeScaleMSPy/umap_results.npz"

processed_data = preprocess_data_pca(hdf5_file)
filtered_data = filter_data(processed_data)
pca_scores, valid_spectrum_keys, components, explained_variance_ratio = perform_pca(filtered_data, n_components=10, chunk_size=500)
perform_umap_pca(pca_scores, valid_spectrum_keys, output_file)

# 行範囲情報を生成
line_info = parse_msp_to_line_info(msp_file)

# UMAP結果を読み込み
umap_df = load_umap_results(output_file)

# アプリケーションの作成と実行
print("Dashアプリを起動します。ブラウザで開いて結果を確認してください。")
print("http://127.0.0.1:8050/")
app = create_umap_app(umap_df, msp_file, line_info, mz_max=2000)
app.run_server(debug=False)
