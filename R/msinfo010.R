library(MsBackendMsp)
library(MSnbase)

spectrum_list <- readRDS(, file="C:/R/spectrum_list.rds")

# min(unlist(mz(sp))) : 43.01784
# max(unlist(mz(sp))) : 1688.33
library(rhdf5)

# HDF5ファイル作成
h5file <- "C:/R/spectrum_data.h5"
if (file.exists(h5file)) file.remove(h5file)
h5createFile(h5file)

# 共通の m/z 範囲を保存
mzrange <- seq(from = 40, to = 1700, by = 0.01)
h5write(mzrange, h5file, "mz")

# スペクトルデータの逐次処理
for (i in seq_along(spectrum_list)) {
  print(paste("Processing spectrum", i))
  spectrum2_obj <- spectrum_list[[i]]
  binned_spectrum <- MSnbase::bin(spectrum2_obj, breaks = mzrange)
  intensity_values <- binned_spectrum@intensity
  
  # グループ作成とデータ保存
  group_name <- paste0("spectrum_", i)
  h5createGroup(h5file, group_name)
  h5write(intensity_values, h5file, paste0(group_name, "/intensity"))
  
  # メモリ解放
  rm(spectrum2_obj, binned_spectrum, intensity_values)
}

# ファイル構造の確認
h5ls(h5file)
