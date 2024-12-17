import numpy as np
import h5sparse
from scipy.sparse import csr_matrix

def load_hdf5(file_path):
    """
    スパースHDF5ファイルを読み込み、スパース行列と前駆体m/zを復元する関数。

    Parameters:
        file_path (str): HDF5ファイルのパス。

    Returns:
        csr_matrix: スペクトルのスパース行列。
        np.ndarray: 前駆体m/zの配列。
    """
    with h5sparse.File(file_path, 'r') as h5f:
        # スパース行列を読み込み
        sparse_matrix = h5f['binned_spectra'][:]
        # 前駆体m/zを読み込み
        precursor_mzs = h5f['precursor_mzs'][:]
    
    return sparse_matrix, precursor_mzs


### test
# HDF5ファイルの読み込み
#output_hdf5_file="C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"
#sparse_matrix, precursor_mzs = load_hdf5(output_hdf5_file)

# 結果の確認
#print(f"Number of spectra: {sparse_matrix.shape[0]}")
#print(f"Number of bins: {sparse_matrix.shape[1]}")
#print(f"First precursor m/z: {precursor_mzs[0]}")
