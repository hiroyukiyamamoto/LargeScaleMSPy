import os
import numpy as np
from scipy.sparse import csr_matrix
import h5sparse

def parse_massbank_record(record_text):
    """
    MassBank形式のレコードを解析し、ピークデータと前駆体m/zを抽出する。

    Parameters:
        record_text (str): MassBank形式のテキストレコード。

    Returns:
        tuple: (前駆体m/z, m/z配列, 強度配列)
    """
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
                mz_values.append(float(parts[0]))
                intensity_values.append(float(parts[1]))

    return precursor_mz, np.array(mz_values), np.array(intensity_values)


def massbankhuman2hdf5(input_folder, output_file, bin_size=0.1, mz_range=(0, 2000)):
    """
    フォルダ内のMassBank形式のファイルをパースし、ビニングしてHDF5に保存する。

    Parameters:
        input_folder (str): MassBank形式のファイルが格納されたフォルダのパス。
        output_file (str): 出力HDF5ファイルのパス。
        bin_size (float): ビニング幅（デフォルトは0.1）。
        mz_range (tuple): ビニング範囲 (min_mz, max_mz)。
    """
    # ビニングの設定
    min_mz, max_mz = mz_range
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)

    # データ格納リスト
    all_binned_spectra = []
    all_precursor_mzs = []

    # フォルダ内のファイルを処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # MassBank形式のファイルを指定
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, "r") as f:
                record_text = f.read()
            
            # レコードをパース
            precursor_mz, mz_values, intensity_values = parse_massbank_record(record_text)

            # ビニング処理
            binned_intensity, _ = np.histogram(mz_values, bins=bins, weights=intensity_values)

            # スパース形式に変換して格納
            all_binned_spectra.append(csr_matrix(binned_intensity))
            all_precursor_mzs.append(precursor_mz)

    # スパース行列を1つの大きな行列にまとめる
    if all_binned_spectra:
        sparse_matrix = csr_matrix(np.vstack([spectrum.toarray() for spectrum in all_binned_spectra]))
    else:
        raise ValueError("No valid MassBank files found in the folder.")

    # スパースHDF5ファイルに保存
    with h5sparse.File(output_file, "w") as h5f:
        # スパース行列を保存
        h5f.create_dataset("binned_spectra", data=sparse_matrix)
        # 前駆体m/zを保存
        h5f.create_dataset("precursor_mzs", data=np.array(all_precursor_mzs, dtype=float))

    print(f"Sparse HDF5 file saved to {output_file}")

# example
input_folder = "C:/Users/hyama/Documents/LargeScaleMSPy/MassBank-Human-main/HSA001"
output_file = "MassBank-Human.hdf5"
massbankhuman2hdf5(input_folder, output_file, bin_size=0.01, mz_range=(0, 2000))
