import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import time

# データ前処理関数
def preprocess_data_in_memory(input_file_path, intensity_threshold=0, normalization_threshold=0.01):
    """
    データ前処理関数。

    Parameters:
        input_file_path (str): 入力HDF5ファイルのパス。
        intensity_threshold (float): 強度フィルタリングの閾値。
        normalization_threshold (float): 正規化後の最小値フィルタリングの閾値。

    Returns:
        processed_data (numpy.ndarray): 前処理されたデータ。
    """
    print("Starting preprocessing...")
    processed_data = []
    with h5sparse.File(input_file_path, "r") as f:
        for key in f.keys():
            if key.startswith("spectrum_"):
                sparse_matrix = f[key][:]
                dense_matrix = sparse_matrix.toarray()

                # 強度フィルタリング
                dense_matrix[dense_matrix < intensity_threshold] = 0

                # 正規化と最小値フィルタリング
                for i in range(dense_matrix.shape[0]):
                    row = dense_matrix[i, :]
                    max_val = row.max()
                    if max_val > 0:
                        row = row / max_val
                    row[row < normalization_threshold] = 0
                    dense_matrix[i, :] = row

                dense_matrix[dense_matrix > 0] = 1  # 非ゼロ要素を1に置換

                processed_data.append(dense_matrix)

                print(f"Processed {key}")
    print("Preprocessing completed.")
    return np.vstack(processed_data)

# 類似度行列計算関数
def process_similarity_matrix_in_memory(processed_data, min_peaks=3):
    """
    類似度行列を計算する関数。

    Parameters:
        processed_data (numpy.ndarray): 前処理されたデータ。
        min_peaks (int): 共通ピーク数の最小値。

    Returns:
        adjacency_matrix (numpy.ndarray): フィルタリングされた隣接行列。
        valid_indices (numpy.ndarray): 有効なノードのインデックス。
    """
    print("Calculating similarity matrix...")
    X_sparse = csr_matrix(processed_data)
    Z_sparse = X_sparse @ X_sparse.T  # 共通ピーク数の類似度行列
    Z = Z_sparse.toarray()

    # グラフ隣接行列を作成
    np.fill_diagonal(Z, 0)
    adjacency_matrix = (Z >= min_peaks).astype(int)

    # 有効なノードを抽出
    N = np.sum(adjacency_matrix > 0, axis=1)
    valid_indices = np.where(N > 0)[0]  # 接続があるノード

    if len(valid_indices) == 0:
        raise ValueError("No valid nodes found. Check the similarity matrix and filtering criteria.")

    adjacency_matrix = adjacency_matrix[valid_indices][:, valid_indices]
    print(f"Filtered adjacency matrix shape: {adjacency_matrix.shape}")

    return adjacency_matrix, valid_indices

# PCA実行関数（隣接行列を利用）
def perform_pca_from_adjacency(adjacency_matrix, valid_indices, chunk_size=500, n_components=10):
    """
    隣接行列からPCA（SVD）を実行する関数。

    Parameters:
        adjacency_matrix (numpy.ndarray): 隣接行列。
        valid_indices (numpy.ndarray): 有効なノードのインデックス。
        chunk_size (int): チャンクサイズ。
        n_components (int): PCAの次元数。

    Returns:
        pca_scores (numpy.ndarray): PCAスコア。
        components (numpy.ndarray): 主成分ベクトル。
    """
    ipca = IncrementalPCA(n_components=n_components)
    pca_scores = []

    for start in range(0, adjacency_matrix.shape[0], chunk_size):
        end = start + chunk_size
        chunk = adjacency_matrix[start:end]
        print(f"Processing chunk: start={start}, end={end}, shape={chunk.shape}")

        if chunk.size > 0:
            scores = ipca.fit_transform(chunk)
            pca_scores.append(scores)

    # チャンクを結合
    if len(pca_scores) > 0:
        pca_scores = np.vstack(pca_scores)
    else:
        raise ValueError("No data available for PCA after chunk processing.")

    # 主成分ベクトルとスコアを保存
    print("PCA processing completed.")
    return pca_scores, ipca.components_

# UMAP実行関数
def perform_umap(pca_scores, valid_indices, output_file):
    """
    UMAP実行関数。

    Parameters:
        pca_scores (numpy.ndarray): PCAスコア。
        valid_indices (numpy.ndarray): 有効なスペクトルインデックス。
        output_file (str): UMAP結果保存ファイル。
    """
    print("Starting UMAP...")
    umap = UMAP(n_neighbors=10, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    # スペクトルキーの生成
    spectrum_keys = [f"spectrum_{i + 1:04d}" for i in valid_indices]

    # UMAP結果を保存
    np.savez(output_file, umap_results=umap_results, valid_keys=spectrum_keys)
    print(f"UMAP results saved to {output_file}")

    # プロットの生成
    #plt.figure(figsize=(10, 8))
    #plt.scatter(umap_results[:, 0], umap_results[:, 1], s=1, alpha=1)
    #plt.title("UMAP Projection", fontsize=16)
    #plt.xlabel("UMAP Dimension 1", fontsize=12)
    #plt.ylabel("UMAP Dimension 2", fontsize=12)
    #plt.grid(True)
    #plt.show()

# メイン関数
def main():
    input_file_path = "C:/Users/hyama/data/MassBank-Human.hdf5"
    output_file = "C:/Users/hyama/data/umap_results.npz"

    # データ前処理を実行
    processed_data = preprocess_data_in_memory(input_file_path, intensity_threshold=0, normalization_threshold=0.01)

    # 類似度行列を計算
    adjacency_matrix, valid_indices = process_similarity_matrix_in_memory(processed_data, min_peaks=3)

    # 隣接行列のPCAを実行
    pca_scores, components = perform_pca_from_adjacency(adjacency_matrix, valid_indices, chunk_size=10000)

    # UMAPを実行
    perform_umap(pca_scores, valid_indices, output_file=output_file)

if __name__ == "__main__":
    main()
