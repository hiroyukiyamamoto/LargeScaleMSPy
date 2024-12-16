# LargeScaleMSPy

質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPython、Rスクリプト

## スクリプト実行の流れ
- **msinfo009.R**: MSPファイルからスペクトルデータを読み込み、前処理してSpectrum2オブジェクトのリストを作成し、RDS形式で保存します。
- **msinfo010.R**: Spectrum2オブジェクトのリストを読み込み、m/z範囲でビニングした強度データをHDF5形式で保存します。
- **pca003.py、pca004.py**: HDF5形式のスペクトルデータを用いて、PCAおよびUMAPによる次元削減と可視化を実行します。

## 開発中
- **mgf2hdf5.py**: pyopenmsでMGFファイルからスペクトルデータを読み込み、前処理してhdf5形式で保存します。
- **mgf2hdf5_pyteomics.py**: pyteomicsでMGFファイルからスペクトルデータを読み込み、前処理してhdf5形式で保存します。
- **msp2hdf5.py**: MSPファイルからスペクトルデータを読み込み、前処理してhdf5形式で保存します。
- **load_hdf5.py**: hdf5形式のファイルを読み込みます。
- **parse_msp.py**: mspファイルをパースするためのスクリプト。
- **create_umap_app.py**: UMAPのデータポイントをクリックすると、MS/MSスペクトルが表示されます
