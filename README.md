# LargeScaleMSPy

このリポジトリには、質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPythonスクリプトが含まれています。

### スクリプト実行の流れ
1. **msinfo009.R**: MSPファイルからスペクトルデータを読み込み、前処理してSpectrum2オブジェクトのリストを作成し、RDS形式で保存します。
2. **msinfo010.R**: Spectrum2オブジェクトのリストを読み込み、m/z範囲でビン分割した強度データをHDF5形式で保存します。
3. **pca003.py、pca004.py**: HDF5形式のスペクトルデータを用いて、PCAおよびUMAPによる次元削減と可視化を実行します。

### msinfo009.R
MSPファイルからスペクトルデータを読み込み、前処理してSpectrum2オブジェクトのリストを作成し、RDS形式で保存するRスクリプト

### msinfo010.R
Spectrum2オブジェクトのリストを読み込み、m/z範囲でビン分割した強度データをHDF5形式で保存するRスクリプト

### pca003.py、pca004.py
質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPythonスクリプト
