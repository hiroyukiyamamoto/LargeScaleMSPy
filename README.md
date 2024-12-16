# LargeScaleMSPy
質量分析スペクトルデータを前処理し、フィルタリング、PCAおよびUMAPを用いた次元削減、結果の可視化を行うPython、Rスクリプト

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



# MS2DecR

`MS2DecR` is an R package designed for advanced processing and deconvolution of MS/MS spectral data. By integrating Independent Component Analysis (ICA) and Alternating Least Squares (ALS), it provides tools for resolving complex MS/MS spectra and identifying compounds through spectral matching.

## Features

- **MS2 Data Preprocessing**: Filtering of MS2 data by isolation window, retention time, and collision energy.
- **Spectral Deconvolution**: Resolves overlapping spectra into component signals using ICA and ALS.
- **Spectral Matching**: Matches deconvoluted spectra against spectral libraries for compound identification.
- **Customizable Parameters**: Offers flexibility for fine-tuning processing and deconvolution.

## Installation

You can install `MS2DecR` from source using the `devtools` package:

```r
# Install from source
devtools::install_local("path_to_package_directory")
