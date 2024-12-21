import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import h5sparse
import scipy.sparse as sp
import time

# データ前処理関数
def preprocess_data(input_file_path, processed_file_path):
    print("Starting preprocessing...")
    with h5sparse.File(input_file_path, "r") as f, h5sparse.File(processed_file_path, "w") as f_out:
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
                f_out[key] = processed_sparse_matrix

                print(f"Processed and saved {key}")
    print("Preprocessing completed.")

# フィルタリング関数
def filter_data(processed_file_path, filtered_file_path):
    print("Starting filtering...")
    with h5sparse.File(processed_file_path, "r") as f_in:
        all_keys = [key for key in f_in.keys() if key.startswith("spectrum_")]
        total_spectra = len(all_keys)

        all_intensities = [f_in[key][:] for key in all_keys]
        combined_matrix = sp.vstack(all_intensities)

        non_zero_columns = combined_matrix.getnnz(axis=0) > 0
        filtered_matrix = combined_matrix[:, non_zero_columns]

        print(f"Original shape: {combined_matrix.shape}, Filtered shape: {filtered_matrix.shape}")

    with h5sparse.File(filtered_file_path, "w") as f_out:
        for i, key in enumerate(all_keys):
            row_data = filtered_matrix.getrow(i)
            if row_data.nnz > 0:
                f_out[key] = row_data
                print(f"Processed and saved {key}")

            progress = (i + 1) / total_spectra * 100
            print(f"Progress: {progress:.2f}%")
    print("Filtering completed.")

# PCA実行関数
def perform_pca(filtered_file_path, output_file, chunk_size=250, n_components=10):
    print("Starting Incremental PCA with full autoscaling...")
    ipca = IncrementalPCA(n_components=n_components)
    scaler = StandardScaler()
    pca_scores = []
    valid_spectrum_keys = []

    with h5sparse.File(filtered_file_path, "r") as f:
        all_keys = [key for key in f.keys() if key.startswith("spectrum_")]
        total_spectra = len(all_keys)

        for i, chunk_start in enumerate(range(0, total_spectra, chunk_size)):
            chunk_keys = all_keys[chunk_start:chunk_start + chunk_size]
            chunk_data = [f[key][:] for key in chunk_keys]

            chunk_array_sparse = sp.vstack(chunk_data)
            chunk_array = chunk_array_sparse.toarray()

            chunk_scaled = scaler.partial_fit(chunk_array).transform(chunk_array)
            ipca.partial_fit(chunk_scaled)
            pca_chunk_scores = ipca.transform(chunk_scaled)
            pca_scores.append(pca_chunk_scores)

            valid_spectrum_keys.extend(chunk_keys)
            print(f"Processed chunk {i + 1}/{len(range(0, total_spectra, chunk_size))}")

    pca_scores = np.vstack(pca_scores)
    np.savez(output_file,
             components=ipca.components_,
             explained_variance_ratio=ipca.explained_variance_ratio_,
             scores=pca_scores,
             valid_keys=valid_spectrum_keys)
    print(f"PCA results saved to {output_file}")
    return pca_scores, valid_spectrum_keys

# UMAP実行関数
def perform_umap(pca_scores, valid_spectrum_keys, umap_output_file, umap_csv_file):
    print("Starting UMAP...")
    umap = UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    np.savez(umap_output_file, umap_results=umap_results, valid_keys=valid_spectrum_keys)
    print(f"UMAP results saved to {umap_output_file}")

    umap_df = pd.DataFrame(umap_results, columns=["UMAP_1", "UMAP_2"])
    umap_df["Spectrum_Key"] = valid_spectrum_keys
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
    input_file_path = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"
    processed_file_path = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test2.h5"
    filtered_file_path = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test3.h5"
    pca_output_file = "pca_results2.npz"
    umap_output_file = "umap_results.npz"
    umap_csv_file = "umap_results.csv"

    preprocess_data(input_file_path, processed_file_path)
    filter_data(processed_file_path, filtered_file_path)
    pca_scores, valid_spectrum_keys = perform_pca(filtered_file_path, pca_output_file)
    perform_umap(pca_scores, valid_spectrum_keys, umap_output_file, umap_csv_file)

if __name__ == "__main__":
    main()
