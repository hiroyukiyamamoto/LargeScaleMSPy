import os
from typing import Iterable, List, Tuple

import h5sparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP


def _iter_spectrum_keys(h5f) -> List[str]:
    return sorted(key for key in h5f.keys() if key.startswith("spectrum_"))


def _copy_non_spectrum_datasets(src_h5, dst_h5):
    for key in src_h5.keys():
        if key.startswith("spectrum_"):
            continue
        obj = src_h5[key]
        try:
            data = obj[:]
        except Exception:
            data = obj[()]
        dst_h5.create_dataset(key, data=data)
        if hasattr(obj, "attrs"):
            for attr_key in obj.attrs.keys():
                dst_h5[key].attrs[attr_key] = obj.attrs[attr_key]


def _load_dense_row(h5f, key: str) -> np.ndarray:
    row = h5f[key][:]
    if sp.issparse(row):
        dense = row.toarray()
    else:
        dense = np.asarray(row)
    return np.asarray(dense).reshape(1, -1)


def _preprocess_dense_row(
    dense_row: np.ndarray,
    intensity_threshold: float = 0.0,
    normalization_threshold: float = 0.01,
    binarize: bool = True,
) -> np.ndarray:
    x = np.asarray(dense_row, dtype=float).copy()

    if intensity_threshold > 0:
        x[x < intensity_threshold] = 0

    max_intensity = x.max()
    if max_intensity > 0:
        x = x / max_intensity

    if normalization_threshold > 0:
        x[x < normalization_threshold] = 0

    if binarize:
        x[x > 0] = 1

    return x


def _compute_idf(
    input_file_path: str,
    intensity_threshold: float = 0.0,
    normalization_threshold: float = 0.01,
    binarize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    with h5sparse.File(input_file_path, "r") as f:
        spectrum_keys = _iter_spectrum_keys(f)
        if not spectrum_keys:
            raise ValueError("No spectrum_* datasets were found in the input file.")

        df = None
        n_spectra = 0
        for key in spectrum_keys:
            x = _preprocess_dense_row(
                _load_dense_row(f, key),
                intensity_threshold=intensity_threshold,
                normalization_threshold=normalization_threshold,
                binarize=binarize,
            )
            binary = (x > 0).astype(int)
            if df is None:
                df = binary.reshape(-1)
            else:
                df += binary.reshape(-1)
            n_spectra += 1

    idf = np.log((n_spectra + 1) / (df + 1)) + 1
    return df, idf, n_spectra


def preprocess_data(
    input_file_path,
    intensity_threshold=0.0,
    normalization_threshold=0.01,
    binarize=True,
    weighting_method="none",
    output_file_path=None,
    save_idf=False,
):
    """
    Preprocess spectra stored in an HDF5 file and optionally apply IDF-only weighting.

    Parameters
    ----------
    input_file_path : str
        Input HDF5 path.
    intensity_threshold : float, optional
        Absolute intensity threshold applied before normalization.
    normalization_threshold : float, optional
        Relative threshold applied after base-peak normalization.
    binarize : bool, optional
        If True, positive intensities are converted to 1.
    weighting_method : {"none", "idf_only"}, optional
        Weighting applied after preprocessing.
    output_file_path : str, optional
        Output HDF5 path. If omitted, a suffix is added to the input file path.
    save_idf : bool, optional
        If True, store document frequency and IDF vectors in the output file.
    """
    if weighting_method not in ("none", "idf_only"):
        raise ValueError("weighting_method must be one of: 'none', 'idf_only'")

    if output_file_path is None:
        suffix = "_processed_idf.hdf5" if weighting_method == "idf_only" else "_processed.hdf5"
        output_file_path = input_file_path.replace(".hdf5", suffix)

    print("Starting preprocessing...")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    df = None
    idf = None
    n_spectra = None
    if weighting_method == "idf_only":
        print("Computing document frequencies for IDF-only weighting...")
        df, idf, n_spectra = _compute_idf(
            input_file_path,
            intensity_threshold=intensity_threshold,
            normalization_threshold=normalization_threshold,
            binarize=binarize,
        )

    try:
        with h5sparse.File(input_file_path, "r") as f_in:
            with h5sparse.File(output_file_path, "w") as f_out:
                _copy_non_spectrum_datasets(f_in, f_out)
                spectrum_keys = _iter_spectrum_keys(f_in)
                total_spectra = len(spectrum_keys)

                for i, key in enumerate(spectrum_keys, start=1):
                    dense_row = _preprocess_dense_row(
                        _load_dense_row(f_in, key),
                        intensity_threshold=intensity_threshold,
                        normalization_threshold=normalization_threshold,
                        binarize=binarize,
                    )

                    if weighting_method == "idf_only":
                        dense_row = dense_row * idf.reshape(1, -1)

                    processed_sparse_matrix = sp.csr_matrix(dense_row)
                    f_out.create_dataset(key, data=processed_sparse_matrix)

                    if hasattr(f_in[key], "attrs"):
                        for attr_key in f_in[key].attrs.keys():
                            f_out[key].attrs[attr_key] = f_in[key].attrs[attr_key]

                    if i % 100 == 0 or i == total_spectra:
                        print(f"Processed {i}/{total_spectra} spectra")

                f_out.attrs["intensity_threshold"] = float(intensity_threshold)
                f_out.attrs["normalization_threshold"] = float(normalization_threshold)
                f_out.attrs["binarize"] = bool(binarize)
                f_out.attrs["weighting_method"] = weighting_method

                if save_idf and idf is not None:
                    f_out.create_dataset("document_frequency", data=df.astype(int))
                    f_out.create_dataset("idf", data=idf.astype(float))
                    f_out.attrs["n_spectra"] = int(n_spectra)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

    print("Preprocessing completed.")
    return output_file_path


def preprocess_data_pca(input_file_path):
    """
    Backward-compatible wrapper around preprocess_data().
    """
    return preprocess_data(
        input_file_path=input_file_path,
        intensity_threshold=0.0,
        normalization_threshold=0.01,
        binarize=True,
        weighting_method="none",
        output_file_path=input_file_path.replace(".hdf5", "_processed.hdf5"),
    )


def filter_data_pca(processed_file_path):
    print("Starting filtering...")

    filtered_file_path = processed_file_path.replace(".hdf5", "_filtered.hdf5")

    if os.path.exists(filtered_file_path):
        os.remove(filtered_file_path)

    try:
        with h5sparse.File(processed_file_path, "r") as f_in:
            all_keys = _iter_spectrum_keys(f_in)
            total_spectra = len(all_keys)
            if not all_keys:
                raise ValueError("No spectrum_* datasets were found in the processed file.")

            all_intensities = [f_in[key][:] for key in all_keys]
            combined_matrix = sp.vstack(all_intensities).tocsr()

            non_zero_columns = combined_matrix.getnnz(axis=0) > 0
            filtered_matrix = combined_matrix[:, non_zero_columns]

            print(f"Original shape: {combined_matrix.shape}, Filtered shape: {filtered_matrix.shape}")

            with h5sparse.File(filtered_file_path, "w") as f_out:
                for key in f_in.keys():
                    if key.startswith("spectrum_"):
                        continue
                    obj = f_in[key]
                    try:
                        data = obj[:]
                    except Exception:
                        data = obj[()]
                    if key == "mz":
                        data = np.asarray(data)
                        if data.shape[0] == non_zero_columns.shape[0]:
                            data = data[non_zero_columns]
                        elif data.shape[0] == non_zero_columns.shape[0] + 1:
                            # Some converters store bin edges; convert them to bin centers.
                            bin_centers = (data[:-1] + data[1:]) / 2
                            data = bin_centers[non_zero_columns]
                    f_out.create_dataset(key, data=data)
                    if hasattr(obj, "attrs"):
                        for attr_key in obj.attrs.keys():
                            f_out[key].attrs[attr_key] = obj.attrs[attr_key]

                for attr_key in f_in.attrs.keys():
                    f_out.attrs[attr_key] = f_in.attrs[attr_key]
                f_out.attrs["filter_zero_columns"] = True

                for i, key in enumerate(all_keys):
                    row_data = filtered_matrix.getrow(i)
                    f_out.create_dataset(key, data=row_data)
                    if hasattr(f_in[key], "attrs"):
                        for attr_key in f_in[key].attrs.keys():
                            f_out[key].attrs[attr_key] = f_in[key].attrs[attr_key]

                    progress = (i + 1) / total_spectra * 100
                    if (i + 1) % 100 == 0 or i + 1 == total_spectra:
                        print(f"Progress: {progress:.2f}%")
    except Exception as e:
        print(f"Error during filtering: {e}")
        raise

    print("Filtering completed.")
    return filtered_file_path


def _collect_dense_chunks(filtered_file_path, chunk_size):
    with h5sparse.File(filtered_file_path, "r") as f:
        all_keys = _iter_spectrum_keys(f)
        total_spectra = len(all_keys)
        for chunk_start in range(0, total_spectra, chunk_size):
            chunk_keys = all_keys[chunk_start:chunk_start + chunk_size]
            chunk_data = [f[key][:] for key in chunk_keys]
            chunk_array_sparse = sp.vstack(chunk_data).tocsr()
            chunk_array = chunk_array_sparse.toarray()
            yield chunk_keys, chunk_array


def perform_pca(
    filtered_file_path,
    chunk_size=250,
    n_components=10,
    output_file="pca_results.npz",
    scaling=False,
    centering=False,
):
    print("Starting Incremental PCA...")
    ipca = IncrementalPCA(n_components=n_components)
    scaler = None
    if scaling or centering:
        scaler = StandardScaler(with_mean=centering, with_std=scaling)

    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        with h5sparse.File(filtered_file_path, "r") as f:
            all_keys = _iter_spectrum_keys(f)
            total_spectra = len(all_keys)
            if not all_keys:
                raise ValueError("No spectrum_* datasets were found in the filtered file.")

        total_chunks = len(range(0, total_spectra, chunk_size))

        if scaler is not None:
            print("Fitting scaler...")
            for i, (_, chunk_array) in enumerate(_collect_dense_chunks(filtered_file_path, chunk_size), start=1):
                scaler.partial_fit(chunk_array)
                print(f"Scaler chunk {i}/{total_chunks}")

        print("Fitting Incremental PCA...")
        for i, (_, chunk_array) in enumerate(_collect_dense_chunks(filtered_file_path, chunk_size), start=1):
            if scaler is not None:
                chunk_array = scaler.transform(chunk_array)
            ipca.partial_fit(chunk_array)
            print(f"PCA fit chunk {i}/{total_chunks}")

        print("Collecting PCA scores...")
        pca_scores = []
        valid_spectrum_keys = []
        for i, (chunk_keys, chunk_array) in enumerate(_collect_dense_chunks(filtered_file_path, chunk_size), start=1):
            if scaler is not None:
                chunk_array = scaler.transform(chunk_array)
            pca_chunk_scores = ipca.transform(chunk_array)
            pca_scores.append(pca_chunk_scores)
            valid_spectrum_keys.extend(chunk_keys)
            print(f"PCA transform chunk {i}/{total_chunks}")

        pca_scores = np.vstack(pca_scores)
        np.savez(
            output_file,
            components=ipca.components_,
            explained_variance_ratio=ipca.explained_variance_ratio_,
            scores=pca_scores,
            valid_keys=np.array(valid_spectrum_keys),
            scaling=np.array([bool(scaling)]),
            centering=np.array([bool(centering)]),
        )
        print(f"PCA results saved to {output_file}")
    except Exception as e:
        print(f"Error during PCA: {e}")
        raise

    return pca_scores, valid_spectrum_keys


def perform_umap(
    pca_scores,
    valid_spectrum_keys,
    umap_output_file,
    n_neighbors=100,
    min_dist=0.01,
    n_components=2,
    random_state=42,
):
    print("Starting UMAP...")

    if os.path.exists(umap_output_file):
        os.remove(umap_output_file)

    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )

    try:
        umap_results = umap.fit_transform(pca_scores)
        np.savez(
            umap_output_file,
            umap_results=umap_results,
            valid_keys=np.array(valid_spectrum_keys),
            n_neighbors=np.array([n_neighbors]),
            min_dist=np.array([min_dist]),
            n_components=np.array([n_components]),
            random_state=np.array([random_state]),
        )
        print(f"UMAP results saved to {umap_output_file}")
    except Exception as e:
        print(f"Error during UMAP: {e}")
        raise

    return umap_results


def perform_pca2umap(pca_scores, valid_spectrum_keys, umap_output_file):
    """
    Backward-compatible wrapper around perform_umap().
    """
    return perform_umap(
        pca_scores=pca_scores,
        valid_spectrum_keys=valid_spectrum_keys,
        umap_output_file=umap_output_file,
        n_neighbors=100,
        min_dist=0.01,
        n_components=2,
        random_state=42,
    )


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
