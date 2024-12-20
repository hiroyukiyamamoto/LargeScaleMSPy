# LargeScaleMSPy

質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPython、Rスクリプト

## スクリプト
- **msp2hdf5.py**: MSPファイルからスペクトルデータを読み込み、m/z範囲でビニングした強度データをHDF5形式で保存します。また、MSPファイルの特定のスペクトル情報だけ読み込むことが出来ます。
- **load_hdf5.py**: HDF5形式のファイルを読み込みます。
- **pca_umap_analysis.py**: HDF5形式のスペクトルデータを用いて、PCAおよびUMAPによる次元削減と可視化を実行します。
 
## 開発予定
- **mgf2hdf5.py**: pyopenmsでMGFファイルからスペクトルデータを読み込み、前処理してhdf5形式で保存します。
- **mgf2hdf5_pyteomics.py**: pyteomicsでMGFファイルからスペクトルデータを読み込み、前処理してhdf5形式で保存します。
- **create_umap_app.py**: UMAPのデータポイントをクリックすると、MS/MSスペクトルが表示されます
