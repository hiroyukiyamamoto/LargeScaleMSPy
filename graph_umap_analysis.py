import os
from scipy.sparse import csr_matrix, vstack
import h5sparse
from sklearn.decomposition import IncrementalPCA
from umap.umap_ import UMAP
import numpy as np
import pandas as pd

def preprocess_data_in_memory(input_file_path, intensity_threshold=0, normalization_threshold=0.01):
    print("Starting preprocessing...")
    processed_data = []
    spectrum_labels = []  # ラベルを保存するリスト

    with h5sparse.File(input_file_path, "r") as f:
        for key in f.keys():
            if key.startswith("spectrum_"):
                sparse_matrix = f[key][:]

                # Apply intensity threshold
                sparse_matrix.data[sparse_matrix.data < intensity_threshold] = 0

                # Normalize rows
                row_sums = sparse_matrix.max(axis=1).A.flatten()
                row_sums[row_sums == 0] = 1
                normalized_sparse_matrix = sparse_matrix.multiply(1.0 / row_sums[:, None])

                # Apply normalization threshold
                normalized_sparse_matrix.data[normalized_sparse_matrix.data < normalization_threshold] = 0

                # Binarize the matrix
                binarized_sparse_matrix = normalized_sparse_matrix.sign()
                processed_data.append(binarized_sparse_matrix)
                spectrum_labels.append(key)  # ラベルを保存

                print(f"Processed {key}")

    print("Preprocessing completed.")

    # Combine all processed data into a single sparse matrix
    combined_data = vstack(processed_data)

    # Save to HDF5
    name, ext = os.path.splitext(input_file_path)
    output_hdf5_path = f"{name}_out{ext}"
    with h5sparse.File(output_hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("processed_data", data=combined_data)
        hdf5_file.create_dataset("spectrum_labels", data=np.array(spectrum_labels, dtype="S"))  # ラベルを保存

    print(f"Processed data saved to HDF5: {output_hdf5_path}")
    return output_hdf5_path

from scipy.sparse import csr_matrix
import pandas as pd
import h5py

def compute_similarity(output_hdf5_path, similarity_hdf5_file, min_peaks):
    """
    Compute the full similarity matrix for processed data, save it as a sparse HDF5 dataset,
    and export it to a labeled CSV file.

    Args:
        output_hdf5_path (str): Path to the HDF5 file with processed data.
        similarity_hdf5_file (str): Path to save the computed similarity matrix (HDF5 format).
        similarity_csv_file (str): Path to save the computed similarity matrix as a labeled CSV.
        min_peaks (int): Minimum peaks to consider similarity significant.
    """
    # Load processed data and spectrum labels
    with h5py.File(output_hdf5_path, "r") as f:
        group = f["processed_data"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        processed_data = csr_matrix((data, indices, indptr))

        # Load spectrum labels
        spectrum_labels = [label.decode("utf-8") for label in f["spectrum_labels"][:]]

    # Ensure spectrum_labels matches the size of processed_data
    if len(spectrum_labels) != processed_data.shape[0]:
        raise ValueError(
            f"Mismatch between spectrum labels ({len(spectrum_labels)}) "
            f"and processed data rows ({processed_data.shape[0]})."
        )

    print("Calculating similarity matrix...")

    # Save the similarity matrix to a new sparse HDF5 file
    num_rows = processed_data.shape[0]
    with h5sparse.File(similarity_hdf5_file, "w") as f:

        # スパース行列を保存するデータセットを初期化
        dset = f.create_dataset("similarity_matrix", shape=(num_rows, num_rows), dtype="float32")

        # ラベルも一緒に保存
        f.create_dataset("spectrum_labels", data=np.array(spectrum_labels, dtype="S"))

        # 各行について計算
        row_data = []
        for i in range(num_rows):
            # 行ごとに計算
            similarity_vector = processed_data[i, :] @ processed_data.T

            # 対角成分を0に設定
            similarity_vector = similarity_vector.tolil()
            similarity_vector[0, i] = 0
            similarity_vector = similarity_vector.tocsr()

            # 閾値を適用してフィルタリング
            similarity_vector = similarity_vector.multiply(similarity_vector >= min_peaks)

            # スパース行列として保存するためにデータを一時保存
            row_data.append(similarity_vector)

            # 進行状況を出力
            print(f"Processed and prepared row {i + 1}/{num_rows}")

    # 全行をまとめてスパース行列として保存
    full_sparse_matrix = vstack(row_data)
    # Save the sparse matrix and labels using h5sparse
    with h5sparse.File(similarity_hdf5_file, "w") as f:
        f.create_dataset("similarity_matrix", data=full_sparse_matrix)
        f.create_dataset("spectrum_labels", data=np.array(spectrum_labels, dtype="S"))

    print(f"Similarity matrix saved to sparse HDF5: {similarity_hdf5_file}")

def filter_zero_rows_and_columns(similarity_hdf5_file, filtered_hdf5_file):
    """
    Remove all-zero rows and corresponding columns from a symmetric similarity matrix
    and save the filtered matrix and labels.

    Args:
        similarity_hdf5_file (str): Path to the input HDF5 file with the similarity matrix.
        filtered_hdf5_file (str): Path to save the filtered similarity matrix (HDF5 format).
        filtered_csv_file (str): Path to save the filtered similarity matrix as a CSV file.
    """
    # Load the similarity matrix and labels
    with h5sparse.File(similarity_hdf5_file, "r") as f:
        similarity_matrix = f["similarity_matrix"][:]
        spectrum_labels = [label.decode("utf-8") for label in f["spectrum_labels"][:]]

    print("Filtering all-zero rows and columns...")

    # Identify non-zero rows and columns
    non_zero_rows = np.array(similarity_matrix.sum(axis=1)).flatten() > 0
    non_zero_indices = np.where(non_zero_rows)[0]

    # Filter the similarity matrix
    filtered_matrix = similarity_matrix[non_zero_indices][:, non_zero_indices]

    # Filter the labels
    filtered_labels = [spectrum_labels[i] for i in non_zero_indices]

    # Save the filtered matrix to a new sparse HDF5 file
    with h5sparse.File(filtered_hdf5_file, "w") as f:
        f.create_dataset("filtered_similarity_matrix", data=filtered_matrix)
        f.create_dataset("spectrum_labels", data=np.array(filtered_labels, dtype="S"))

    print(f"Filtered similarity matrix saved to HDF5: {filtered_hdf5_file}")

from scipy.sparse import csr_matrix
from sklearn.decomposition import IncrementalPCA
import numpy as np

def perform_incremental_pca(hdf5_file, n_components, chunk_size=10000):
    """
    Perform Incremental PCA on the filtered similarity matrix.

    Args:
        hdf5_file (str): Path to the HDF5 file with the filtered similarity matrix.
        n_components (int): Number of principal components to compute.

    Returns:
        Tuple: PCA scores, PCA components, spectrum labels, and valid indices.
    """
    ipca = IncrementalPCA(n_components=n_components)
    pca_scores = []
    all_labels = []

    # Load the similarity matrix and labels
    with h5sparse.File(hdf5_file, "r") as f:
        # 修正: スパース行列を正しく読み込む
        similarity_matrix = f["filtered_similarity_matrix"][:]
        spectrum_labels = [label.decode("utf-8") for label in f["spectrum_labels"][:]]

    print(f"Loaded similarity matrix of shape {similarity_matrix.shape}")

    # Perform incremental PCA
    for i in range(0, similarity_matrix.shape[0], chunk_size):  # チャンクサイズ100（例）
        chunk = similarity_matrix[i:i+chunk_size].toarray()  # チャンクごとに密行列に変換
        ipca.partial_fit(chunk)
        print(f"Processed chunk {i + 1} to {min(i + chunk_size, similarity_matrix.shape[0])} for fitting.")

    print("Incremental PCA fitting completed.")

    # Transform the data
    for i in range(0, similarity_matrix.shape[0], chunk_size):
        chunk = similarity_matrix[i:i+chunk_size].toarray()
        scores = ipca.transform(chunk)
        pca_scores.append(scores)

    print("PCA transformation completed.")

    # Combine results
    pca_scores = np.vstack(pca_scores)
    valid_indices = list(range(len(spectrum_labels)))

    return pca_scores, ipca.components_, spectrum_labels, valid_indices

def perform_umap(pca_scores, labels, umap_output_file):
    """
    Perform UMAP dimensionality reduction and save the results with valid keys.

    Args:
        pca_scores (numpy.ndarray): PCA-reduced data (shape: [n_samples, n_components]).
        labels (list): List of spectrum labels corresponding to PCA scores.
        umap_output_file (str): Output file to save UMAP results.
    """
    print("Starting UMAP...")

    # 確認
    print(f"PCA scores shape: {pca_scores.shape}, Number of labels: {len(labels)}")
    
    # PCAスコアとラベル数の一致を確認
    if len(pca_scores) != len(labels):
        raise ValueError(
            f"Mismatch between PCA scores ({len(pca_scores)}) and labels ({len(labels)})."
        )

    # UMAP の適用
    umap = UMAP(n_neighbors=10, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    # 保存するキーをラベルとして利用
    valid_keys = labels

    # 結果を保存
    np.savez(umap_output_file, umap_results=umap_results, valid_keys=valid_keys)
    print(f"UMAP results saved to {umap_output_file}")

    # 結果の一部を表示
    print(f"First few UMAP results:\n{umap_results[:5]}")
    print(f"First few valid keys: {valid_keys[:5]}")

def main():
    #input_file_path = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"
    #processed_file = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos_out.hdf5"    

    input_file_path = "C:/Users/hyama/MassBank-Human.hdf5"
    processed_file = preprocess_data_in_memory(input_file_path)
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
    umap_output_file = "C:/Users/hyama/umap_results.npz"
    perform_umap(pca_scores,labels, umap_output_file)

if __name__ == "__main__":
    main()