import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse
import scipy.sparse as sp

def parse_massbank_record(record_text):
    lines = record_text.strip().split("\n")
    precursor_mz = None
    mz_values = []
    intensity_values = []
    
    in_peak_section = False
    for line in lines:
        if line.startswith("MS$FOCUSED_ION: PRECURSOR_M/Z"):
            precursor_mz = float(line.split()[-1])
        elif line.startswith("PK$PEAK:"):
            in_peak_section = True
        elif in_peak_section:
            if line.startswith("//"):
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    mz_values.append(float(parts[0]))
                    intensity_values.append(float(parts[1]))
                except ValueError:
                    pass
    return precursor_mz, np.array(mz_values), np.array(intensity_values)

def massbankhuman2hdf5(input_folder, output_file, bin_size=0.1, mz_range=(0, 2000)):
    min_mz, max_mz = mz_range
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)
    all_precursor_mzs = []

    with h5sparse.File(output_file, "w") as h5f:
        for idx, filename in enumerate(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, "r") as f:
                    record_text = f.read()
                
                precursor_mz, mz_values, intensity_values = parse_massbank_record(record_text)
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
input_folder = "C:/Users/hyama/Documents/LargeScaleMSPy/MassBank-Human-main/HSA001"
output_file = "C:/Users/hyama/data/MassBank-Human.hdf5"
massbankhuman2hdf5(input_folder, output_file, bin_size=0.01, mz_range=(0, 2000))
