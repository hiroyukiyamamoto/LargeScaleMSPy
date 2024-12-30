import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import h5sparse
import scipy.sparse as sp
import time

def preprocess_data_pca(input_file_path):
    print("Starting preprocessing...")

    # Define the output file path
    processed_file_path = input_file_path.replace(".hdf5", "_processed.hdf5")

    # Remove existing file to avoid conflicts
    if os.path.exists(processed_file_path):
        os.remove(processed_file_path)

    try:
        with h5sparse.File(input_file_path, "r") as f:
            with h5sparse.File(processed_file_path, "w") as f_out:
                for key in f.keys():
                    if key.startswith("spectrum_"):
                        # Read the sparse matrix
                        sparse_matrix = f[key][:]
                        
                        # Convert to dense matrix
                        dense_matrix = sparse_matrix.toarray()

                        # Normalize by the maximum intensity
                        max_intensity = dense_matrix.max()
                        if max_intensity > 0:
                            dense_matrix = dense_matrix / max_intensity

                        # Binarize the data
                        dense_matrix[dense_matrix <= 0.01] = 0
                        dense_matrix[dense_matrix > 0] = 1

                        # Convert back to sparse matrix
                        processed_sparse_matrix = sp.csr_matrix(dense_matrix)

                        # Save the processed matrix to the output file
                        f_out.create_dataset(key, data=processed_sparse_matrix)

                        print(f"Processed and saved {key}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

    print("Preprocessing completed.")
    return processed_file_path

def filter_data_pca(processed_file_path):
    print("Starting filtering...")

    # Define the output file path
    filtered_file_path = processed_file_path.replace(".hdf5", "_filtered.hdf5")

    # Remove existing file to avoid conflicts
    if os.path.exists(filtered_file_path):
        os.remove(filtered_file_path)

    try:
        with h5sparse.File(processed_file_path, "r") as f_in:
            all_keys = [key for key in f_in.keys() if key.startswith("spectrum_")]
            total_spectra = len(all_keys)

            # Combine all spectra into a single sparse matrix
            all_intensities = [f_in[key][:] for key in all_keys]
            combined_matrix = sp.vstack(all_intensities)

            # Filter out columns with no non-zero values
            non_zero_columns = combined_matrix.getnnz(axis=0) > 0
            filtered_matrix = combined_matrix[:, non_zero_columns]

            print(f"Original shape: {combined_matrix.shape}, Filtered shape: {filtered_matrix.shape}")

        with h5sparse.File(filtered_file_path, "w") as f_out:
            for i, key in enumerate(all_keys):
                row_data = filtered_matrix.getrow(i)
                if row_data.nnz > 0:
                    f_out.create_dataset(key, data=row_data)
                    print(f"Processed and saved {key}")

                progress = (i + 1) / total_spectra * 100
                print(f"Progress: {progress:.2f}%")
    except Exception as e:
        print(f"Error during filtering: {e}")
        raise

    print("Filtering completed.")
    return filtered_file_path

def perform_pca(filtered_file_path, chunk_size=250, n_components=10, output_file="pca_results.npz"):
    print("Starting Incremental PCA with full autoscaling...")
    ipca = IncrementalPCA(n_components=n_components)
    scaler = StandardScaler()
    pca_scores = []
    valid_spectrum_keys = []

    # Remove existing file to avoid conflicts
    if os.path.exists(output_file):
        os.remove(output_file)

    try:
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
    except Exception as e:
        print(f"Error during PCA: {e}")
        raise

    return pca_scores, valid_spectrum_keys

def perform_pca2umap(pca_scores, valid_spectrum_keys, umap_output_file):
    print("Starting UMAP...")

    # Remove existing file to avoid conflicts
    if os.path.exists(umap_output_file):
        os.remove(umap_output_file)

    umap = UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)

    try:
        umap_results = umap.fit_transform(pca_scores)
        np.savez(umap_output_file, umap_results=umap_results, valid_keys=valid_spectrum_keys)
        print(f"UMAP results saved to {umap_output_file}")
    except Exception as e:
        print(f"Error during UMAP: {e}")
        raise

    return umap_results

def main():
    input_file_path = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"
    umap_output_file = "umap_results.npz"

    try:
        processed_file_path = preprocess_data_pca(input_file_path)
        filtered_file_path = filter_data_pca(processed_file_path)
        pca_scores, valid_spectrum_keys = perform_pca(filtered_file_path)
        umap_results = perform_pca2umap(pca_scores, valid_spectrum_keys, umap_output_file)

        plt.figure(figsize=(10, 8))
        plt.scatter(umap_results[:, 0], umap_results[:, 1], s=1, alpha=1)
        plt.title("UMAP Projection", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
