library(MsBackendMsp)
library(MSnbase)

file_msp <- "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
sp <- Spectra(file_msp, source = MsBackendMsp())

# 最初のスペクトルの mz と intensity を取得
#mz_values <- mz(sp[1])         # m/z 値
#intensity_values <- intensity(sp[1])  # 強度

# compound_classes <- spectraData(sp)$COMPOUNDCLASS

### ADGGAの化合物クラス
x <- sp

# データを一括取得
all_mz <- mz(x)              # 全スペクトルの m/z
all_intensity <- intensity(x) # 全スペクトルの強度
all_precursorMz <- spectraData(x)$precursorMz # 全スペクトルの前駆体m/z

# Spectrum2 オブジェクトのリストを作成
spectrum_list <- vector("list", length(all_mz))  # 空のリストを準備

# for文でオブジェクトを作成
for (i in seq_along(all_mz)) {
  # Spectrum2 オブジェクトを作成してリストに格納
  spectrum_list[[i]] <- new("Spectrum2",
                            mz = all_mz[[i]],             # 外で取得した m/z を利用
                            intensity = all_intensity[[i]],  # 外で取得した intensity を利用
                            precursorMz = all_precursorMz[i]) # 外で取得した precursorMz を利用
  
  # 進捗表示（任意）
  if (i %% 100 == 0 || i == length(all_mz)) {
    message("Processed ", i, "/", length(all_mz), " spectra")
  }
}

saveRDS(spectrum_list, file="C:/R/spectrum_list.rds")

