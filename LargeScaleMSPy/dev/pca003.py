#%%
import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import time
from umap.umap_ import UMAP
import matplotlib.pyplot as plt

# ファイルパスの設定
input_file_path = "C:/R/spectrum_data.h5"
processed_file_path = "C:/R/processed_spectrum_data.h5"
chunk_size = 5000  # 1回に処理するスペクトル数
n_components = 10  # 主成分数

import h5sparse
import scipy.sparse as sp

# %% データ前処理と保存
print("Starting preprocessing...")
with h5py.File(input_file_path, "r") as f, h5sparse.File(processed_file_path, "w") as f_out:
    for key in f.keys():
        if key.startswith("spectrum_"):
            intensity = f[f"{key}/intensity"][:]

            # データの前処理
            max_intensity = intensity.max()
            if max_intensity > 0:
                intensity = intensity / max_intensity
            intensity[intensity <= 0.01] = 0
            intensity[intensity > 0] = 1

            # スパース行列に変換
            sparse_intensity = sp.csr_matrix(intensity)

            # 処理済みデータをスパース形式で保存
            f_out.create_dataset(f"{key}/intensity", data=sparse_intensity)
            print(f"Processed and saved {key}")
print("Preprocessing completed.")

# ----------------- ここまでは改めて実行不要

#%%

## 値が全て0の変数と、0のスペクトルを除外

import h5sparse
import scipy.sparse as sp
import numpy as np

# ファイルパス
processed_file_path = "C:/R/processed_spectrum_data.h5"
filtered_file_path = "C:/R/filtered_spectrum_data.h5"

print("Starting filtering...")

with h5sparse.File(processed_file_path, "r") as f_in:
    all_keys = [key for key in f_in.keys() if key.startswith("spectrum_")]  # スペクトルキーを取得
    total_spectra = len(all_keys)

    # 全スペクトルデータを結合
    all_intensities = []
    for key in all_keys:
        sparse_intensity = f_in[f"{key}/intensity"][:]  # スペクトルデータを取得
        all_intensities.append(sparse_intensity)

    # スパース行列を結合
    combined_matrix = sp.vstack(all_intensities)

    # 全て0の列（変数）を特定して除外
    non_zero_columns = combined_matrix.getnnz(axis=0) > 0
    filtered_matrix = combined_matrix[:, non_zero_columns]

    print(f"Original shape: {combined_matrix.shape}, Filtered shape: {filtered_matrix.shape}")

# フィルタリング後のデータを保存
with h5sparse.File(filtered_file_path, "w") as f_out:
    for i, key in enumerate(all_keys):
        row_data = filtered_matrix.getrow(i)  # 各行を取得
        if row_data.nnz > 0:  # 全て0でない行のみ保存
            f_out.create_dataset(f"{key}/intensity", data=row_data)
            print(f"Processed and saved {key}")

        # 進捗表示
        progress = (i + 1) / total_spectra * 100
        print(f"Progress: {progress:.2f}%")

print("Filtering completed.")

#%% 
# サイズ確認

# スパース行列を結合
combined_matrix = sp.vstack(all_intensities)

# 元の行列のサイズを確認
print(f"Original matrix shape: {combined_matrix.shape}")
print(f"Number of non-zero elements in the original matrix: {combined_matrix.nnz}")

# 全て0の列（変数）を特定して除外
non_zero_columns = combined_matrix.getnnz(axis=0) > 0
filtered_matrix = combined_matrix[:, non_zero_columns]

# フィルタリング後のサイズを確認
print(f"Filtered matrix shape: {filtered_matrix.shape}")
print(f"Number of non-zero elements in the filtered matrix: {filtered_matrix.nnz}")

#%%
import h5sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
import time

# パス設定
processed_file_path = "C:/R/filtered_spectrum_data.h5"
chunk_size = 250
n_components = 10

# PCA の設定
ipca = IncrementalPCA(n_components=n_components)
### グラフ隣接行列の場合は、TruncatedSVDを使う
scaler = StandardScaler()  # 完全なautoscaling用

pca_scores = []
valid_spectrum_keys = []

print("Starting Incremental PCA with full autoscaling...")
start_time = time.time()

with h5sparse.File(processed_file_path, "r") as f:
    all_keys = [key for key in f.keys() if key.startswith("spectrum_")]
    total_spectra = len(all_keys)

    # チャンク処理
    for i, chunk_start in enumerate(range(0, total_spectra, chunk_size)):
        chunk_keys = all_keys[chunk_start:chunk_start + chunk_size]
        chunk_data = [f[f"{key}/intensity"][:] for key in chunk_keys]

        # スパース行列を結合
        chunk_array_sparse = sp.vstack(chunk_data)

        # スパース行列を密な形式に変換
        chunk_array = chunk_array_sparse.toarray()

        # Autoscaling（完全な標準化: 平均0、分散1）
        chunk_scaled = scaler.partial_fit(chunk_array).transform(chunk_array)

        # Incremental PCA の適用
        ipca.partial_fit(chunk_scaled)
        pca_chunk_scores = ipca.transform(chunk_scaled)
        pca_scores.append(pca_chunk_scores)

        valid_spectrum_keys.extend(chunk_keys)  # 有効なスペクトルキーを記録

        # 進捗状況を表示
        elapsed_time = time.time() - start_time
        progress = (chunk_start + len(chunk_keys)) / total_spectra * 100
        print(f"Chunk {i + 1}: Processed {chunk_start + len(chunk_keys)} / {total_spectra} spectra "
              f"({progress:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds")

# PCAスコアを結合
pca_scores = np.vstack(pca_scores)
print("Incremental PCA completed.")

# 結果の保存
output_file = "pca_results.npz"
np.savez(output_file,
         components=ipca.components_,
         explained_variance_ratio=ipca.explained_variance_ratio_,
         scores=pca_scores,
         valid_keys=valid_spectrum_keys)
print(f"PCA results saved to {output_file}")

# %% UMAP 処理
print("Starting UMAP...")
umap = UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)
umap_results = umap.fit_transform(pca_scores)

# UMAP結果の保存
umap_output_file = "umap_results.npz"
np.savez(umap_output_file, umap_results=umap_results, valid_keys=valid_spectrum_keys)
print(f"UMAP results saved to {umap_output_file}")

# %% UMAP結果の可視化
print("Plotting UMAP results...")
plt.figure(figsize=(10, 8))
plt.scatter(umap_results[:, 0], umap_results[:, 1], s=1, alpha=0.8)
plt.title("UMAP Projection", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.grid(True)
plt.show()

#%%
import pandas as pd
# UMAP結果をCSVで出力
umap_df = pd.DataFrame(umap_results, columns=["UMAP_1", "UMAP_2"])
umap_df["Spectrum_Key"] = valid_spectrum_keys  # スペクトルキーを追加
umap_csv_file = "umap_results.csv"
umap_df.to_csv(umap_csv_file, index=False)
print(f"UMAP results saved to {umap_csv_file}")
# %%
