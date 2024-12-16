import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import time

# HDF5ファイルのパス
file_path = "C:/R/spectrum_data.h5"

# Incremental PCA の設定
n_components = 10  # 主成分数
ipca = IncrementalPCA(n_components=n_components)
scaler = StandardScaler()

# チャンクサイズの設定
chunk_size = 100  # 1回に処理するスペクトル数

# HDF5ファイルを開いてスペクトルを逐次処理
start_time = time.time()  # 開始時刻を記録
with h5py.File(file_path, "r") as f:
    all_keys = [key for key in f.keys() if key.startswith("spectrum_")]  # スペクトルキーを取得
    total_spectra = len(all_keys)  # 全スペクトル数を記録

    # チャンク処理
    for i, chunk_start in enumerate(range(0, total_spectra, chunk_size)):
        chunk_keys = all_keys[chunk_start:chunk_start + chunk_size]  # チャンクに含まれるキー
        chunk_data = []

        # チャンク内のスペクトルを取得
        for key in chunk_keys:
            intensity = f[f"{key}/intensity"][:]  # intensity データを取得
            chunk_data.append(intensity)

        # チャンクを行列に変換
        chunk_array = np.array(chunk_data)

        # データのスケーリング
        chunk_scaled = scaler.partial_fit(chunk_array).transform(chunk_array)

        # Incremental PCA の適用
        ipca.partial_fit(chunk_scaled)

        # 進捗状況を表示
        elapsed_time = time.time() - start_time
        progress = (chunk_start + len(chunk_keys)) / total_spectra * 100
        print(f"Chunk {i + 1}: Processed {chunk_start + len(chunk_keys)} / {total_spectra} spectra "
              f"({progress:.2f}%) - Elapsed time: {elapsed_time:.2f} seconds")

# PCA結果の出力
print("Principal Components Shape:", ipca.components_.shape)
print("Explained Variance Ratio:", ipca.explained_variance_ratio_)
