import numpy as np
import h5sparse
from scipy.sparse import csr_matrix

def parse_msp(file_path):
    """
    MSPファイルを読み込み、スペクトル情報を辞書形式で返す関数。

    Parameters:
        file_path (str): MSPファイルのパス。

    Returns:
        list: スペクトル情報を格納した辞書のリスト。
    """
    spectra = []
    with open(file_path, 'r') as f:
        spectrum = {}
        mz_values = []
        intensity_values = []
        read_peaks = False  # ピークデータを読み取るフラグ

        for line in f:
            line = line.strip()
            if line.startswith("NAME:"):
                if spectrum:  # 既存のスペクトルを保存
                    spectrum["m/z array"] = mz_values
                    spectrum["intensity array"] = intensity_values
                    spectra.append(spectrum)
                    mz_values = []
                    intensity_values = []
                # 新しいスペクトルの開始
                spectrum = {"Name": line.split(":", 1)[1].strip()}
                read_peaks = False

            elif line.startswith("PRECURSORMZ:"):
                spectrum["PrecursorMZ"] = float(line.split(":", 1)[1].strip())

            elif line.startswith("Num Peaks:"):
                spectrum["Num Peaks"] = int(line.split(":", 1)[1].strip())
                read_peaks = True  # Num Peaksの後からデータ行が始まる

            elif read_peaks and line:
                # ピークデータ (m/z と intensity) を読み取る
                try:
                    mz, intensity = map(float, line.split())
                    mz_values.append(mz)
                    intensity_values.append(intensity)
                except ValueError:
                    read_peaks = False  # 数値でない行が来たらピーク読み取り終了

        # 最後のスペクトルを追加
        if spectrum:
            spectrum["m/z array"] = mz_values
            spectrum["intensity array"] = intensity_values
            spectra.append(spectrum)

    return spectra

def msp2hdf5(input_file, output_file, bin_size=0.1, mz_range=(0, 2000)):
    """
    MSPファイルを読み込み、ビニング処理を行い、スパース形式でHDF5に保存する関数。

    Parameters:
        input_file (str): 入力MSPファイルのパス。
        output_file (str): 出力HDF5ファイルのパス。
        bin_size (float): ビニング幅（デフォルトは0.1）。
        mz_range (tuple): ビニング範囲 (min_mz, max_mz)。
    """
    # MSPファイルの読み込み
    spectra = parse_msp(input_file)

    # ビニングの設定
    min_mz, max_mz = mz_range
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)

    # スペクトルデータを格納するリスト
    binned_spectra = []
    precursor_mzs = []

    for spectrum in spectra:
        mz_values = spectrum["m/z array"]
        intensity_values = spectrum["intensity array"]
        precursor_mz = spectrum.get("PrecursorMZ", None)

        # ビニング処理
        binned_intensity, _ = np.histogram(mz_values, bins=bins, weights=intensity_values)

        # スパース形式に変換して格納
        binned_spectra.append(csr_matrix(binned_intensity))
        precursor_mzs.append(precursor_mz)

    # スパース行列を1つの大きな行列にまとめる
    sparse_matrix = csr_matrix(np.vstack([spectrum.toarray() for spectrum in binned_spectra]))

    # スパースHDF5ファイルに保存
    with h5sparse.File(output_file, 'w') as h5f:
        # スパース行列を保存
        h5f.create_dataset('binned_spectra', data=sparse_matrix)
        # 前駆体m/zを保存
        h5f.create_dataset('precursor_mzs', data=np.array(precursor_mzs, dtype=float))

    print(f"Sparse HDF5 file saved to {output_file}")

### example
#input_file="C:/Users/hyama/Documents/LargeScaleMSPy/data/MSMS-Pos-MetaboBASE.msp"
#output_file="C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"
#spec = msp2hdf5(input_file, output_file, bin_size=0.01, mz_range=(40, 1700))