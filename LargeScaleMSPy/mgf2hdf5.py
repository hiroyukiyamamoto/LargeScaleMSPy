import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse
import scipy.sparse as sp

def parse_mgf_file(mgf_file):
    """
    MGFファイルを読み込み、各スペクトルの情報を抽出する。

    Parameters:
        mgf_file (str): MGFファイルのパス。

    Returns:
        list: 各スペクトル情報のリスト [(precursor_mz, mz_values, intensity_values), ...]
    """
    spectra = []
    with open(mgf_file, "r") as f:
        precursor_mz = None
        mz_values = []
        intensity_values = []
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                precursor_mz = None
                mz_values = []
                intensity_values = []
            elif line.startswith("PEPMASS"):
                precursor_mz = float(line.split("=")[1])
            elif line.startswith("END IONS"):
                if precursor_mz is not None and mz_values:
                    spectra.append((precursor_mz, np.array(mz_values), np.array(intensity_values)))
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        pass
    return spectra

def mgf_to_hdf5(input_mgf_file, output_file, bin_size=0.1, mz_range=(0, 2000)):
    """
    MGFファイルを読み込み、ビニングしてHDF5に保存する。

    Parameters:
        input_mgf_file (str): 入力MGFファイルのパス。
        output_file (str): 出力HDF5ファイルのパス。
        bin_size (float): ビニング幅（デフォルトは0.1）。
        mz_range (tuple): ビニング範囲 (min_mz, max_mz)。
    """
    min_mz, max_mz = mz_range
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)
    all_precursor_mzs = []

    with h5sparse.File(output_file, "w") as h5f:
        spectra = parse_mgf_file(input_mgf_file)
        for idx, (precursor_mz, mz_values, intensity_values) in enumerate(spectra):
            binned_intensity, _ = np.histogram(mz_values, bins=bins, weights=intensity_values)

            # スパース形式に変換
            sparse_matrix = csr_matrix(binned_intensity)

            # HDF5に書き込み
            key = f"spectrum_{idx + 1:04d}"  # 一意のキー (例: spectrum_0001)
            h5f.create_dataset(key, data=sparse_matrix)
            all_precursor_mzs.append(precursor_mz)

        # 前駆体m/zリストを保存
        h5f.create_dataset("precursor_mzs", data=np.array(all_precursor_mzs, dtype=float))

    print(f"Sparse HDF5 file saved to {output_file}")

# 実行例
input_mgf_file = "C:/Users/hyama/data/combined_spectra.mgf"
output_file = "C:/Users/hyama/data/MassBank-Human.hdf5"
mgf_to_hdf5(input_mgf_file, output_file, bin_size=0.01, mz_range=(0, 2000))
