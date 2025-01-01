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

def compute_similarity(output_hdf5_path, similarity_hdf5_file, similarity_csv_file, min_peaks):
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

    # Compute similarity matrix for the entire dataset
    similarity_matrix = processed_data @ processed_data.T
    similarity_matrix.setdiag(0)  # Remove self-similarity
    similarity_matrix = similarity_matrix.multiply(similarity_matrix >= min_peaks)

    # Save the similarity matrix to a new sparse HDF5 file
    with h5py.File(similarity_hdf5_file, "w") as f:
        f.create_dataset("data", data=similarity_matrix.data)
        f.create_dataset("indices", data=similarity_matrix.indices)
        f.create_dataset("indptr", data=similarity_matrix.indptr)
        f.create_dataset("spectrum_labels", data=np.array(spectrum_labels, dtype="S"))  # Save labels

    print(f"Similarity matrix saved to HDF5: {similarity_hdf5_file}")

    # Convert the similarity matrix to dense format for CSV export
    print("Converting similarity matrix to dense format for CSV export...")
    dense_similarity_matrix = similarity_matrix.toarray()

    # Save the dense similarity matrix with labels to a CSV file
    df = pd.DataFrame(dense_similarity_matrix, index=spectrum_labels, columns=spectrum_labels)
    df.to_csv(similarity_csv_file)

    print(f"Similarity matrix saved to labeled CSV: {similarity_csv_file}")

def filter_zero_rows_and_columns(similarity_hdf5_file, filtered_hdf5_file, filtered_csv_file):
    """
    Remove all-zero rows and corresponding columns from a symmetric similarity matrix
    and save the filtered matrix and labels.

    Args:
        similarity_hdf5_file (str): Path to the input HDF5 file with the similarity matrix.
        filtered_hdf5_file (str): Path to save the filtered similarity matrix (HDF5 format).
        filtered_csv_file (str): Path to save the filtered similarity matrix as a CSV file.
    """
    # Load the similarity matrix and labels
    with h5py.File(similarity_hdf5_file, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        spectrum_labels = [label.decode("utf-8") for label in f["spectrum_labels"][:]]

    # Reconstruct the sparse similarity matrix
    num_rows = len(indptr) - 1
    similarity_matrix = csr_matrix((data, indices, indptr), shape=(num_rows,num_rows))

    print("Filtering all-zero rows and columns...")

    # Identify non-zero rows
    non_zero_rows = np.array(similarity_matrix.sum(axis=1)).flatten() > 0
    non_zero_indices = np.where(non_zero_rows)[0]

    # Filter the similarity matrix
    filtered_matrix = similarity_matrix[non_zero_indices][:, non_zero_indices]

    # Filter the labels
    filtered_labels = [spectrum_labels[i] for i in non_zero_indices]

    # Save the filtered matrix to a new HDF5 file
    with h5py.File(filtered_hdf5_file, "w") as f:
        f.create_dataset("data", data=filtered_matrix.data)
        f.create_dataset("indices", data=filtered_matrix.indices)
        f.create_dataset("indptr", data=filtered_matrix.indptr)
        f.create_dataset("spectrum_labels", data=np.array(filtered_labels, dtype="S"))

    print(f"Filtered similarity matrix saved to HDF5: {filtered_hdf5_file}")

    # Convert the filtered matrix to dense format for CSV export
    dense_filtered_matrix = filtered_matrix.toarray()

    # Save the filtered matrix with labels to a CSV file
    df = pd.DataFrame(dense_filtered_matrix, index=filtered_labels, columns=filtered_labels)
    df.to_csv(filtered_csv_file)

    print(f"Filtered similarity matrix saved to CSV: {filtered_csv_file}")

def perform_incremental_pca(hdf5_file, n_components):
    """
    Perform Incremental PCA on the data stored in the HDF5 file.

    Args:
        hdf5_file (str): Path to the HDF5 file containing filtered data.
        n_components (int): Number of principal components to retain.

    Returns:
        tuple: (PCA scores, PCA components, spectrum labels, valid indices)
    """
    ipca = IncrementalPCA(n_components=n_components)
    pca_scores = []

    with h5py.File(hdf5_file, "r") as f:
        # ラベルを取得
        spectrum_labels = [label.decode("utf-8") for label in f["spectrum_labels"][:]]
        
        # 元データが有効なラベルに対応している場合、そのインデックスを取得
        valid_indices = list(range(len(spectrum_labels)))  # 初期値として全て有効と仮定
        if "valid_indices" in f:
            valid_indices = f["valid_indices"][:].tolist()  # HDF5内の情報を利用
        
        # スパース行列を読み込む
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        num_rows = len(indptr) - 1
        similarity_matrix = csr_matrix((data, indices, indptr), shape=(num_rows, num_rows))

        # 密行列に変換
        dense_matrix = similarity_matrix.toarray()

        # Incremental PCA
        ipca.partial_fit(dense_matrix)
        pca_scores = ipca.transform(dense_matrix)

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
    input_file_path = "C:/Users/hyama/MassBank-Human.hdf5"
    similarity_hdf5_file = "C:/Users/hyama/similarity_matrix.hdf5"
    umap_output_file = "C:/Users/hyama/data/umap_results.npz"

    similarity_csv_file = "C:/Users/hyama/similarity_matrix.csv"

    processed_file = preprocess_data_in_memory(input_file_path)
    valid_indices = compute_similarity(processed_file, similarity_hdf5_file, similarity_csv_file, min_peaks=3)
    
    filtered_hdf5 = "C:/Users/hyama/filtered_similarity_matrix.hdf5"
    filtered_csv = "C:/Users/hyama/filtered_similarity_matrix.csv"
    
    filter_zero_rows_and_columns(similarity_hdf5_file, filtered_hdf5, filtered_csv)

    pca_scores, components, labels, valid_indices = perform_incremental_pca(filtered_hdf5, n_components=10)
    print(f"Type of pca_scores: {type(pca_scores)}, shape: {pca_scores.shape if pca_scores is not None else 'N/A'}")

    # Perform UMAP
    print("Performing UMAP...")
    perform_umap(pca_scores,labels, umap_output_file)

if __name__ == "__main__":
    main()