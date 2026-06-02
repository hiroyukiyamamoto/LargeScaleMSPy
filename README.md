# LargeScaleMSPy

LargeScaleMSPy は、大規模な MS/MS スペクトルデータを HDF5 形式で扱い、前処理、IDF-only 重み付け、PCA、UMAP、可視化を行う Python パッケージです。

この版では、全スペクトル間のグラフ隣接行列・類似度行列を作る方法は採用せず、次の流れを標準候補にしています。

1. ビニング済み MS/MS スペクトルを HDF5 で読む
2. base peak 正規化
3. relative intensity threshold によるピーク除去
4. 0/1 化
5. IDF-only 重み付け
6. autoscaling なしの Incremental PCA
7. UMAP
8. Dash によるインタラクティブ可視化

## インストール

### ソースコードからインストール

リポジトリを取得したあと、このディレクトリで以下を実行します。

```bash
pip install -e .
```

### 配布ファイルからインストール

wheel が README と同じ階層にある場合は、以下のようにインストールできます。

```bash
pip install largescalemspy-1.1.0-py3-none-any.whl
```

sdist からインストールする場合は以下です。

```bash
pip install largescalemspy-1.1.0.tar.gz
```

配布ファイルを作り直す場合は、以下を実行します。

```bash
python -m pip install build
python -m build
```

## 主な関数

- `msp2hdf5()`: MSP ファイルをビニング済み sparse HDF5 に変換
- `mgf_to_hdf5()`: MGF ファイルをビニング済み sparse HDF5 に変換
- `preprocess_data()`: 正規化、しきい値処理、0/1 化、IDF-only 重み付け
- `filter_data_pca()`: 全スペクトルで 0 の bin を削除
- `perform_pca()`: Incremental PCA を実行
- `perform_umap()`: PCA スコアから UMAP を実行
- `create_umap_app()`: UMAP 点をクリックして MS/MS スペクトルを見る Dash アプリを作成

## 推奨設定

現時点の推奨は、`0/1 + IDF-only + no autoscaling` です。

```python
processed_path = preprocess_data(
    input_file_path="input.hdf5",
    intensity_threshold=0.0,
    normalization_threshold=0.01,
    binarize=True,
    weighting_method="idf_only",
    output_file_path="output_idf_processed.hdf5",
    save_idf=True,
)

filtered_path = filter_data_pca(processed_path)

pca_scores, valid_spectrum_keys = perform_pca(
    filtered_file_path=filtered_path,
    n_components=10,
    chunk_size=250,
    output_file="pca_idf_noautoscale.npz",
    scaling=False,
    centering=False,
)
```

## 使用例: MS-DIAL / LipidBlast 系 MSP データ

既存の HDF5 と MSP ファイルを使う例です。パスは自分の環境に合わせて変更してください。

- HDF5: `data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5`
- MSP: `data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp`

```python
import os

from LargeScaleMSPy import *


msp_file = "data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
hdf5_file = "data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.hdf5"

processed_file = "results/full_msdial_idf_processed.hdf5"
pca_file = "results/full_msdial_pca_idf_noautoscale.npz"
umap_file = "results/full_msdial_umap_idf_noautoscale.npz"
mz_max = 2000

os.makedirs(os.path.dirname(processed_file), exist_ok=True)

processed_path = preprocess_data(
    input_file_path=hdf5_file,
    intensity_threshold=0.0,
    normalization_threshold=0.01,
    binarize=True,
    weighting_method="idf_only",
    output_file_path=processed_file,
    save_idf=True,
)

filtered_path = filter_data_pca(processed_path)

pca_scores, valid_spectrum_keys = perform_pca(
    filtered_file_path=filtered_path,
    n_components=10,
    chunk_size=250,
    output_file=pca_file,
    scaling=False,
    centering=False,
)

umap_results = perform_umap(
    pca_scores=pca_scores,
    valid_spectrum_keys=valid_spectrum_keys,
    umap_output_file=umap_file,
    n_neighbors=100,
    min_dist=0.01,
    n_components=2,
    random_state=42,
)

line_info = parse_msp_to_line_info(msp_file)
umap_df = load_umap_results(umap_file)

print("Dash app: http://127.0.0.1:8050/")
app = create_umap_app(umap_df, msp_file, line_info, mz_max=mz_max)
app.run(debug=False)
```

## 可視化だけを実行する例

UMAP 結果がすでにある場合は、計算をやり直さずに Dash アプリだけを起動できます。

```python
from LargeScaleMSPy import *


msp_file = "data/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
umap_file = "results/full_msdial_umap_idf_noautoscale.npz"
mz_max = 2000

line_info = parse_msp_to_line_info(msp_file)
umap_df = load_umap_results(umap_file)

print("Dash app: http://127.0.0.1:8050/")
app = create_umap_app(umap_df, msp_file, line_info, mz_max=mz_max)
app.run(debug=False)
```

## 出力ファイル

上の例では、以下のようなファイルが作成されます。

- `full_msdial_idf_processed.hdf5`
- `full_msdial_idf_processed_filtered.hdf5`
- `full_msdial_pca_idf_noautoscale.npz`
- `full_msdial_umap_idf_noautoscale.npz`

これらは大きな解析結果ファイルなので、通常は GitHub にはアップロードしません。
