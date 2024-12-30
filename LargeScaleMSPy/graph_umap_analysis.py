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

    with h5sparse.File(input_file_path, "r") as f:
        for key in f.keys():
            if key.startswith("spectrum_"):
                sparse_matrix = f[key][:]
                sparse_matrix.data[sparse_matrix.data < intensity_threshold] = 0

                # Normalize rows
                row_sums = sparse_matrix.max(axis=1).A.flatten()
                row_sums[row_sums == 0] = 1
                normalized_sparse_matrix = sparse_matrix.multiply(1.0 / row_sums[:, None])

                # Apply threshold
                normalized_sparse_matrix.data[normalized_sparse_matrix.data < normalization_threshold] = 0
                binarized_sparse_matrix = normalized_sparse_matrix.sign()
                processed_data.append(binarized_sparse_matrix)

                print(f"Processed {key}")

    print("Preprocessing completed.")
    return vstack(processed_data)

def compute_similarity_chunk(processed_data, output_hdf5_file, chunk_size, min_peaks):
    num_samples = processed_data.shape[0]
    valid_indices = []

    with h5sparse.File(output_hdf5_file, "w") as f:
        print("Calculating similarity matrix in chunks...")

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = processed_data[start:end]
            similarity_chunk = chunk @ processed_data.T
            similarity_chunk.setdiag(0)
            similarity_chunk = similarity_chunk.multiply(similarity_chunk >= min_peaks)
            f.create_dataset(f"chunk_{start}_{end}", data=similarity_chunk)
            valid_indices.extend(range(start, end))

            print(f"Processed chunk {start}-{end}")

    print("Similarity matrix calculation completed.")
    return valid_indices

def perform_incremental_pca(hdf5_file, n_components):
    ipca = IncrementalPCA(n_components=n_components)
    pca_scores = []

    with h5sparse.File(hdf5_file, "r") as f:
        chunk_keys = [key for key in f.keys()]
        print(f"Total chunks: {len(chunk_keys)}")

        for i, key in enumerate(chunk_keys):
            print(f"[Fit] Processing chunk {i + 1}/{len(chunk_keys)} - {key}")
            similarity_chunk = f[key][:]
            ipca.partial_fit(similarity_chunk.toarray())

        for i, key in enumerate(chunk_keys):
            print(f"[Transform] Processing chunk {i + 1}/{len(chunk_keys)} - {key}")
            similarity_chunk = f[key][:]
            scores = ipca.transform(similarity_chunk.toarray())
            pca_scores.append(scores)

    pca_scores = np.vstack(pca_scores)
    return pca_scores, ipca.components_

def perform_umap(pca_scores, valid_indices, umap_output_file):
    print("Starting UMAP...")

    if len(pca_scores) != len(valid_indices):
        raise ValueError(f"Mismatch between PCA scores ({len(pca_scores)}) and valid indices ({len(valid_indices)}).")

    umap = UMAP(n_neighbors=10, min_dist=0.01, n_components=2, random_state=42)
    umap_results = umap.fit_transform(pca_scores)

    # Generate spectrum keys
    spectrum_keys = [f"spectrum_{i + 1:04d}" for i in valid_indices]

    # Save results
    np.savez(umap_output_file, umap_results=umap_results, valid_keys=spectrum_keys)
    print(f"UMAP results saved to {umap_output_file}")

    return umap_results

def main():
    input_file_path = "C:/Users/hyama/data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"
    similarity_hdf5_file = "C:/Users/hyama/data/similarity_matrix.hdf5"
    umap_output_file = "C:/Users/hyama/data/umap_results.npz"

    processed_data = preprocess_data_in_memory(input_file_path)
    valid_indices = compute_similarity_chunk(processed_data, similarity_hdf5_file, chunk_size=1000, min_peaks=3)
    pca_scores, components = perform_incremental_pca(similarity_hdf5_file, n_components=10)
    umap_results = perform_umap(pca_scores, valid_indices, umap_output_file)

if __name__ == "__main__":
    main()
