# LargeScaleMSPy

質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPythonライブラリ

## インストール手順
ダウンロードしたディレクトリに移動し、インストールを実行した後、ライブラリを読み込む  
pip install -e .  

## サンプル実行
- **demo1.py**: MGFファイルから変換したデータの主成分分析とUMAPを実行し、可視化アプリを実行する。
- **demo2.py**: MGFファイルから変換したグラフ隣接行列の固有ベクトルとUMAPを実行し、可視化アプリを実行する。

## スクリプト
- **extract_mgf.py**: MGFファイルを解析し、スペクトルデータを抽出してテキスト形式で保存します。
- **load_hdf5.py**: HDF5形式のファイルを読み込み、データセットの内容を取得します。
- **mgf2hdf5.py**: MGFファイルからスペクトルデータを読み込み、m/z範囲でビニングした強度データをHDF5形式で保存します。
- **msp2hdf5.py**: MSPファイルからスペクトルデータを読み込み、m/z範囲でビニングした強度データをHDF5形式で保存します。また、MSPファイルの特定のスペクトル情報を読み取ることもできます。
- **pca_umap_analysis.py**: HDF5形式のスペクトルデータを用いて、PCAによる次元削減を実行し、その結果をUMAPでさらに可視化します。
- **vizapp.py**: UMAPのデータポイントをクリックすると、関連するMS/MSスペクトルを表示するインタラクティブな可視化アプリを提供します。
- **vizapp_for_massbank.py**: MassBankデータを解析し、UMAPプロットとMS/MSスペクトルビューアを統合したインタラクティブな可視化アプリを提供します。
- **graph_umap_analysis.py**: グラフ隣接行列の特異値分解に対するUMAPによる次元削減と可視化を実行します。
