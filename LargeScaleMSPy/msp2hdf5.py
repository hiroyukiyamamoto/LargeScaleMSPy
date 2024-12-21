import numpy as np
import h5sparse
from scipy.sparse import csr_matrix

def parse_msp(file_path):
    """
    MSPファイルを解析し、スペクトル情報と行番号範囲を記録。

    Parameters:
        file_path (str): MSPファイルのパス。

    Returns:
        list: 各スペクトル情報（辞書）と行番号範囲のリスト。
    """
    spectra = []
    with open(file_path, 'r') as f:
        spectrum = {}
        mz_values = []
        intensity_values = []
        read_peaks = False
        start_line = None
        end_line = None

        for line_num, line in enumerate(f):
            line = line.strip()
            if line.startswith("NAME:"):
                if spectrum:
                    spectrum["m/z array"] = mz_values
                    spectrum["intensity array"] = intensity_values
                    spectrum["start_line"] = start_line
                    spectrum["end_line"] = end_line
                    spectra.append(spectrum)
                    mz_values = []
                    intensity_values = []
                spectrum = {"Name": line.split(":", 1)[1].strip()}
                read_peaks = False
                start_line = line_num

            elif line.startswith("PRECURSORMZ:"):
                spectrum["PrecursorMZ"] = float(line.split(":", 1)[1].strip())

            elif line.startswith("Num Peaks:"):
                spectrum["Num Peaks"] = int(line.split(":", 1)[1].strip())
                read_peaks = True

            elif read_peaks and line:
                try:
                    mz, intensity = map(float, line.split())
                    mz_values.append(mz)
                    intensity_values.append(intensity)
                except ValueError:
                    read_peaks = False

            end_line = line_num

        if spectrum:
            spectrum["m/z array"] = mz_values
            spectrum["intensity array"] = intensity_values
            spectrum["start_line"] = start_line
            spectrum["end_line"] = end_line
            spectra.append(spectrum)

    return spectra

def msp2hdf5(input_file, output_file, bin_size=0.01, mz_range=(40, 1700)):
    """
    行番号情報を含む逐次処理でHDF5にスパース形式で保存。

    Parameters:
        input_file (str): 入力MSPファイルのパス。
        output_file (str): 出力HDF5ファイルのパス。
        bin_size (float): ビニング幅。
        mz_range (tuple): ビニング範囲。
    """

    spectra = parse_msp(input_file)

    min_mz, max_mz = mz_range
    mz_bins = np.arange(min_mz, max_mz + bin_size, bin_size)

    with h5sparse.File(output_file, 'w') as h5f:
        h5f.create_dataset("mz", data=mz_bins)
        line_info = []

        for i, spectrum in enumerate(spectra):
            print(f"Processing spectrum {i + 1}/{len(spectra)}")

            # スペクトル情報を取得
            mz_values = np.array(spectrum["m/z array"])
            intensity_values = np.array(spectrum["intensity array"])

            # ビニング処理
            binned_intensity, _ = np.histogram(mz_values, bins=mz_bins, weights=intensity_values)

            # スパース行列に変換
            sparse_matrix = csr_matrix(binned_intensity)

            # グループを作成してスパース形式で保存
            group_name = f"spectrum_{i + 1}"
            h5f[group_name] = sparse_matrix

            # 行番号情報を保存
            start_line = spectrum.get("start_line", -1)
            end_line = spectrum.get("end_line", -1)
            h5f[group_name].attrs["start_line"] = start_line
            h5f[group_name].attrs["end_line"] = end_line

            line_info.append((start_line, end_line))

            # メモリ解放
            del mz_values, intensity_values, binned_intensity, sparse_matrix

        # 行番号情報を全体として保存
        h5f.create_dataset("line_info", data=np.array(line_info, dtype=int))

    print(f"HDF5 with sparse data and line info saved to {output_file}")

def load_sparse_spectrum(hdf5_file, spectrum_index):
    """
    HDF5ファイルから指定したスペクトルのスパース行列と属性情報を取得。

    Parameters:
        hdf5_file (str): HDF5ファイルのパス。
        spectrum_index (int): スペクトルのインデックス（1始まり）。

    Returns:
        csr_matrix: 復元されたスパース行列。
        dict: スペクトルに関連する属性情報。
    """
    with h5sparse.File(hdf5_file, 'r') as h5f:
        group_name = f"spectrum_{spectrum_index}"
        group = h5f[group_name]  # HDF5 グループにアクセス
        sparse_matrix = group[:]  # スパース行列を取得
        attributes = {key: group.attrs[key] for key in group.attrs.keys()}  # 属性を辞書形式で取得

    return sparse_matrix, attributes

def get_line_info(hdf5_file, spectrum_index):
    """
    指定したスペクトルの start_line と end_line を取得する。

    Parameters:
        hdf5_file (str): HDF5ファイルのパス。
        spectrum_index (int): スペクトルのインデックス（1始まり）。

    Returns:
        tuple: (start_line, end_line) の値。
    """
    with h5sparse.File(hdf5_file, 'r') as h5f:
        group_name = f"spectrum_{spectrum_index}"
        group = h5f[group_name]  # HDF5 グループにアクセス
        start_line = group.attrs["start_line"]  # start_line の属性を取得
        end_line = group.attrs["end_line"]  # end_line の属性を取得
    return start_line, end_line

def read_partial_msp(file_path, start_line, end_line):
    """
    MSPファイルの特定の行範囲を部分的に読み込む。

    Parameters:
        file_path (str): MSPファイルのパス。
        start_line (int): 読み込み開始行。
        end_line (int): 読み込み終了行。

    Returns:
        list: 指定範囲の行データ。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()[start_line:end_line + 1]
    return lines

### example
#input_file="C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
#output_file="C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"
#spec = msp2hdf5(input_file, output_file, bin_size=0.01, mz_range=(40, 1700))

#spectrum_index = 1
#sparse_spectrum, attrs = load_sparse_spectrum(output_file, spectrum_index)
#start_line, end_line = get_line_info(output_file, spectrum_index)

#msp_data = read_partial_msp(input_file, start_line, end_line)
