import numpy as np
import h5sparse
from scipy.sparse import csr_matrix
from pyopenms import MSExperiment, MGFFile

def mgf2hdf5(input_file, output_file, bin_size=0.1, mz_range=(0, 2000)):
    """
    MGFファイルを読み込み、ビニング処理を行い、スパース形式でHDF5に保存する関数。

    Parameters:
        input_file (str): 入力MGFファイルのパス。
        output_file (str): 出力HDF5ファイルのパス。
        bin_size (float): ビニング幅（デフォルトは0.1）。
        mz_range (tuple): ビニング範囲 (min_mz, max_mz)。
    """
    # ビニングの設定
    min_mz, max_mz = mz_range
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)

    # MSExperimentオブジェクトを作成してMGFファイルを読み込む
    exp = MSExperiment()
    MGFFile().load(input_file, exp)

    # スペクトルデータを格納するリスト
    binned_spectra = []
    precursor_mzs = []

    for spectrum in exp.getSpectra():
        # m/zと強度を取得
        mz_values, intensity_values = spectrum.get_peaks()
        # 前駆体m/zを取得
        precursor_mz = spectrum.getPrecursors()[0].getMZ() if spectrum.getPrecursors() else None

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
