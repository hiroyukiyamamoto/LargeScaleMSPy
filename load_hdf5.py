import numpy as np
import h5sparse
from scipy.sparse import csr_matrix

def load_hdf5(file_path):
    """
    HDF5ファイルを読み込み、すべてのスペクトルのスパース行列を復元する関数。

    Parameters:
        file_path (str): HDF5ファイルのパス。

    Returns:
        list: スペクトルのスパース行列のリスト。
        list: 前駆体m/zのリスト。
    """
    spectra = []
    precursor_mzs = []

    with h5sparse.File(file_path, 'r') as h5f:
        # すべてのグループを確認
        for key in h5f.keys():
            if key.startswith('spectrum_'):  # スペクトルグループを特定
                group = h5f[key]
                sparse_matrix = group[:]
                precursor_mz = group.attrs.get("PrecursorMZ", None)  # 前駆体m/zを属性から取得
                spectra.append(sparse_matrix)
                precursor_mzs.append(precursor_mz)

    return spectra, precursor_mzs


### テスト
# HDF5ファイルの読み込み
output_hdf5_file = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"
spectra, precursor_mzs = load_hdf5(output_hdf5_file)

# 結果の確認
print(f"Number of spectra: {len(spectra)}")
if spectra:
    print(f"Number of bins in first spectrum: {spectra[0].shape[1]}")
print(f"First precursor m/z: {precursor_mzs[0]}")

# 最初のスペクトルのスパース行列を表示
if spectra:
    print(spectra[0])
