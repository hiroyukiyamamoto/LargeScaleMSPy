import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import scipy.sparse as sp
import h5sparse

# データ前処理関数
def preprocess_data_pca(input_file_path):
    print("Starting preprocessing...")
    processed_data = {}
    with h5sparse.File(input_file_path, "r") as f:
        for key in f.keys():
            if key.startswith("spectrum_"):
                sparse_matrix = f[key][:]
                dense_matrix = sparse_matrix.toarray()

                max_intensity = dense_matrix.max()
                if max_intensity > 0:
                    dense_matrix = dense_matrix / max_intensity
                dense_matrix[dense_matrix <= 0.01] = 0
                dense_matrix[dense_matrix > 0] = 1

                processed_sparse_matrix = sp.csr_matrix(dense_matrix)
                processed_data[key] = processed_sparse_matrix
                print(f"Processed {key}")
    print("Preprocessing completed.")
    return processed_data

# フィルタリング関数
def filter_data(processed_data):
    print("Starting filtering...")
    all_keys = list(processed_data.keys())
    total_spectra = len(all_keys)

    combined_matrix = sp.vstack([processed_data[key] for key in all_keys])

    non_zero_columns = combined_matrix.getnnz(axis=0) > 0
    filtered_matrix = combined_matrix[:, non_zero_columns]

    print(f"Original shape: {combined_matrix.shape}, Filtered shape: {filtered_matrix.shape}")

    filtered_data = {key: filtered_matrix.getrow(i) for i, key in enumerate(all_keys) if filtered_matrix.getrow(i).nnz > 0}
    print("Filtering completed.")
    return filtered_data

# PCA実行関数
def perform_pca(filtered_data, n_components=10, chunk_size=250):
    print("Starting Incremental PCA with full autoscaling...")
    ipca = IncrementalPCA(n_components=n_components)
    scaler = StandardScaler()
    pca_scores = []
    valid_spectrum_keys = list(filtered_data.keys())

    all_keys = valid_spectrum_keys
    total_spectra = len(all_keys)

    for i, chunk_start in enumerate(range(0, total_spectra, chunk_size)):
        chunk_keys = all_keys[chunk_start:chunk_start + chunk_size]
        chunk_data = [filtered_data[key].toarray() for key in chunk_keys]

        chunk_array = np.vstack(chunk_data)
        chunk_scaled = scaler.partial_fit(chunk_array).transform(chunk_array)
        ipca.partial_fit(chunk_scaled)
        pca_chunk_scores = ipca.transform(chunk_scaled)
        pca_scores.append(pca_chunk_scores)

        print(f"Processed chunk {i + 1}/{len(range(0, total_spectra, chunk_size))}")

    pca_scores = np.vstack(pca_scores)
    print("PCA completed.")
    return pca_scores, valid_spectrum_keys, ipca.components_, ipca.explained_variance_ratio_

# UMAP実行関数
def perform_umap_pca(pca_scores, valid_spectrum_keys, output_file):
    print("Starting UMAP...")
    umap = UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    # Save UMAP results to .npz file
    np.savez(output_file, umap_results=umap_results, valid_keys=valid_spectrum_keys)
    print(f"UMAP results saved to {output_file}")

# メイン関数
def main():
    input_file_path = "C:/Users/hyama/data/MassBank-Human.hdf5"
    output_file = "C:/Users/hyama/data/umap_results.npz"

    # 各ステップを実行し、データを次に渡す
    processed_data = preprocess_data(input_file_path)
    filtered_data = filter_data(processed_data)
    pca_scores, valid_spectrum_keys, components, explained_variance_ratio = perform_pca(filtered_data)
    perform_umap(pca_scores, valid_spectrum_keys, output_file)

if __name__ == "__main__":
    main()
