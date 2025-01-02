import os
from LargeScaleMSPy import *

# デモ解析用の設定
massbank_folder = "C:/Users/hyama/Documents/LargeScaleMSPy/MassBank-Human-main/HSA001"  # MassBank形式のテキストファイルが格納されたフォルダ
mgf_file_path = "C:/Users/hyama/data/combined_spectra.mgf"                              # 出力MGFファイルのパス
mz_max = 2000       

# 1. MassBank形式をMGF形式に変換
print("MassBank形式のテキストファイルをMGF形式に変換します...")
massbank_to_mgf(massbank_folder, mgf_file_path)

# HDF5ファイルに保存
hdf5_file = "C:/Users/hyama/data/MassBank-Human.hdf5"
mgf_to_hdf5(mgf_file_path, hdf5_file, bin_size=0.01, mz_range=(0, 2000))

# 各種処理
processed_file = preprocess_data_in_memory(hdf5_file)
similarity_hdf5_file = "C:/Users/hyama/similarity_matrix.hdf5"
compute_similarity(processed_file, similarity_hdf5_file, min_peaks=3) # 類似度行列の計算

# 類似度行列のフィルタリング
filtered_hdf5 = "C:/Users/hyama/filtered_similarity_matrix.hdf5"
filter_zero_rows_and_columns(similarity_hdf5_file, filtered_hdf5)

# Incremental PCAの実行
pca_scores, components, labels, valid_indices = perform_incremental_pca(filtered_hdf5, n_components=10, chunk_size=200)
print(f"Type of pca_scores: {type(pca_scores)}, shape: {pca_scores.shape if pca_scores is not None else 'N/A'}")

# Perform UMAP
print("Performing UMAP...")
umap_results_path = "C:/Users/hyama/umap_results.npz"
perform_umap(pca_scores,labels, umap_results_path)

# create_umap_app にパスを渡してアプリを作成
app = create_umap_mgf_app(umap_results_path, mgf_file_path, mz_max)
app.run_server(debug=False)
