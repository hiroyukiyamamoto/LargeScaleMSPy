import os
from LargeScaleMSPy import *

# デモ解析用の設定
msp_file = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
mz_max = 2000       

# HDF5ファイルに保存
hdf5_file = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"
msp2hdf5(msp_file, hdf5_file, bin_size=0.01, mz_range=(0, 2000))

# 各ステップを実行し、データを次に渡す
output_file = "C:/Users/hyama/data/umap_results.npz"

processed_data = preprocess_data_in_memory(hdf5_file, intensity_threshold=0, normalization_threshold=0.01)
adjacency_matrix, valid_indices = process_similarity_matrix_in_memory(processed_data, min_peaks=3)
pca_scores, components = perform_pca_from_adjacency(adjacency_matrix, valid_indices, chunk_size=10000)
perform_umap(pca_scores, valid_indices, output_file=output_file)

line_info = parse_msp_to_line_info(msp_file)
umap_df = load_umap_results(output_file)

# アプリケーションの作成と実行
print("Dashアプリを起動します。ブラウザで開いて結果を確認してください。")
print("http://127.0.0.1:8050/")
app = create_umap_app(umap_df, msp_file, line_info, mz_max=2000)
app.run_server(debug=False)
