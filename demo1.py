import os
from LargeScaleMSPy import *

# デモ解析用の設定
massbank_folder = "C:/Users/hyama/Documents/LargeScaleMSPy/MassBank-Human-main/HSA001"  # MassBank形式のテキストファイルが格納されたフォルダ
mgf_file_path = "C:/Users/hyama/data/combined_spectra3.mgf"                              # 出力MGFファイルのパス
mz_max = 2000       

# 1. MassBank形式をMGF形式に変換
print("MassBank形式のテキストファイルをMGF形式に変換します...")
massbank_to_mgf(massbank_folder, mgf_file_path)

# HDF5ファイルに保存
hdf5_file = "C:/Users/hyama/data/MassBank-Human.hdf5"
mgf_to_hdf5(mgf_file_path, hdf5_file, bin_size=0.01, mz_range=(0, 2000))

# 各ステップを実行し、データを次に渡す
output_file = "C:/Users/hyama/data/umap_results.npz"

processed_data = preprocess_data_pca(hdf5_file)
filtered_data = filter_data(processed_data)
pca_scores, valid_spectrum_keys, components, explained_variance_ratio = perform_pca(filtered_data)
perform_umap_pca(pca_scores, valid_spectrum_keys, output_file)

# 4. Dashアプリを起動
print("Dashアプリを起動します。ブラウザで開いて結果を確認してください。")
print("http://127.0.0.1:8050/")
app = create_umap_mgf_app(output_file, mgf_file_path, mz_max=mz_max)
app.run_server(debug=False)
