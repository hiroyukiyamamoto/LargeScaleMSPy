import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import pandas as pd

# データ前処理関数
def preprocess_data(input_file_path, processed_file_path, intensity_threshold=0, normalization_threshold=0.01):
    """
    データ前処理関数。

    Parameters:
        input_file_path (str): 入力HDF5ファイルのパス。
        processed_file_path (str): 出力HDF5ファイルのパス。
        intensity_threshold (float): 強度フィルタリングの閾値。
        normalization_threshold (float): 正規化後の最小値フィルタリングの閾値。
    """
    print("Starting preprocessing...")
    with h5sparse.File(input_file_path, "r") as f, h5sparse.File(processed_file_path, "w") as f_out:
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

                processed_sparse_matrix = sp.csr_matrix(dense_matrix)
                f_out[key] = processed_sparse_matrix

                print(f"Processed and saved {key}")
    print("Preprocessing completed.")

# 類似度行列計算関数
def process_similarity_matrix(input_file_path, min_peaks=3):
    """
    類似度行列を計算する関数。

    Parameters:
        input_file_path (str): スペクトルデータを含む入力HDF5ファイルのパス。
        min_peaks (int): 共通ピーク数の最小値。

    Returns:
        Z (numpy.ndarray): フィルタリングされた類似度行列。
    """
    print(f"Loading input file: {input_file_path}")
    with h5sparse.File(input_file_path, "r") as f:
        all_keys = [key for key in f.keys() if key.startswith("spectrum_")]
        spectra = [f[key][:].toarray() for key in all_keys]

    X = np.vstack(spectra)

    # 類似度行列の計算
    print("Calculating similarity matrix...")
    X_sparse = csr_matrix(X)
    Z_sparse = X_sparse @ X_sparse.T  # 共通ピーク数の類似度行列
    Z = Z_sparse.toarray()

    # フィルタリング
    np.fill_diagonal(Z, 0)  # 対角成分を0に
    Z[Z < min_peaks] = 0  # 共通ピーク数が min_peaks 未満の場合は削除
    Z[Z > 0] = 1  # 接続の有無をバイナリに変更

    return Z

# PCA実行関数（隣接行列を利用）
def perform_pca_from_adjacency(adjacency_matrix, output_file, n_components=10):
    """
    隣接行列からPCA（SVD）を実行する関数。

    Parameters:
        adjacency_matrix (numpy.ndarray): 隣接行列。
        output_file (str): PCA結果を保存するファイルパス。
        n_components (int): PCAの次元数。

    Returns:
        pca_scores (numpy.ndarray): PCAスコア。
        components (numpy.ndarray): 主成分ベクトル。
    """
    print("Starting PCA from adjacency matrix...")
    ipca = IncrementalPCA(n_components=n_components)

    # Incremental PCAに隣接行列を直接入力
    pca_scores = ipca.fit_transform(adjacency_matrix)

    # 主成分ベクトルとスコアを保存
    np.savez(output_file,
             components=ipca.components_,
             explained_variance_ratio=ipca.explained_variance_ratio_,
             scores=pca_scores)
    print(f"PCA results saved to {output_file}")
    return pca_scores, ipca.components_

# UMAP実行関数
def perform_umap(pca_scores, umap_output_file, umap_csv_file):
    """
    UMAPによる次元削減を実行。

    Parameters:
        pca_scores (numpy.ndarray): PCAスコア。
        umap_output_file (str): UMAP結果を保存するファイルパス。
        umap_csv_file (str): UMAP結果をCSV形式で保存するファイルパス。
    """
    print("Starting UMAP...")
    umap = UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    np.savez(umap_output_file, umap_results=umap_results)
    print(f"UMAP results saved to {umap_output_file}")

    umap_df = pd.DataFrame(umap_results, columns=["UMAP_1", "UMAP_2"])
    umap_df.to_csv(umap_csv_file, index=False)
    print(f"UMAP results saved to {umap_csv_file}")

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], s=0.001, alpha=0.1)
    plt.title("UMAP Projection", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.grid(True)
    plt.show()

# メイン関数
def main():
    input_file_path = "C:/Users/hyama/data/MassBank-Human.hdf5"
    processed_file_path = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test2.h5"
    pca_output_file = "pca_results2.npz"
    umap_output_file = "umap_results.npz"
    umap_csv_file = "umap_results.csv"

    preprocess_data(input_file_path, processed_file_path, intensity_threshold=1000, normalization_threshold=0.01)

    # 類似度行列を計算
    similarity_matrix = process_similarity_matrix(processed_file_path, min_peaks=3)

    # 隣接行列のPCAを実行
    pca_scores, components = perform_pca_from_adjacency(similarity_matrix, pca_output_file, n_components=10)

    # UMAPを実行
    perform_umap(pca_scores, umap_output_file=umap_output_file, umap_csv_file=umap_csv_file)

if __name__ == "__main__":
    main()
