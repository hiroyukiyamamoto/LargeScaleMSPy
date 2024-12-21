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

        for line in f:
            line = line.strip()
            if line.startswith("Name:"):
                if spectrum:  # 既存のスペクトルを保存
                    spectrum["m/z array"] = mz_values
                    spectrum["intensity array"] = intensity_values
                    spectra.append(spectrum)
                    mz_values = []
                    intensity_values = []

                # 新しいスペクトルの開始
                spectrum = {"Name": line.split(":", 1)[1].strip()}

            elif line.startswith("PrecursorMZ:"):
                spectrum["PrecursorMZ"] = float(line.split(":", 1)[1].strip())

            elif line.startswith("Num Peaks:"):
                spectrum["Num Peaks"] = int(line.split(":", 1)[1].strip())

            elif line and not line.startswith("Name") and not line.startswith("PrecursorMZ") and not line.startswith("Num Peaks"):
                # m/z と intensity のペアを解析
                mz, intensity = map(float, line.split())
                mz_values.append(mz)
                intensity_values.append(intensity)

        # 最後のスペクトルを追加
        if spectrum:
            spectrum["m/z array"] = mz_values
            spectrum["intensity array"] = intensity_values
            spectra.append(spectrum)

    return spectra