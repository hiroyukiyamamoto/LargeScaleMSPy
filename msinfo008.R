library(MsBackendMsp)

file_msp <- "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
sp <- Spectra(file_msp, source = MsBackendMsp())

# 最初のスペクトルの mz と intensity を取得
mz_values <- mz(sp[1])         # m/z 値
intensity_values <- intensity(sp[1])  # 強度

compound_classes <- spectraData(sp)$COMPOUNDCLASS

### ADGGAの化合物クラス
index <- which(compound_classes=="ADGGA")
x <- sp[index]

# Spectrum2オブジェクトに変換
spectrum_list <- vector("list", length(x))  # 空のリストを作成
for (i in seq_along(index)) {
  # 各スペクトルのデータを取得
  mz_values <- mz(x[i])[[1]]
  intensity_values <- intensity(x[i])[[1]]
  precursor_mz <- spectraData(x[i])$precursorMz
  retention_time <- spectraData(x[i])$rtime
  
  # Spectrum2 オブジェクトを作成
  spectrum_list[[i]] <- new("Spectrum2",
                            mz = mz_values,
                            intensity = intensity_values,
                            precursorMz = precursor_mz)
}

mzrange <- seq(from = 40, to = 1700, by = 0.01) 
# min(unlist(mz(sp))) : 43.01784
# max(unlist(mz(sp))) : 1688.33

X <- matrix(NA,length(x),length(mzrange)-1)
for(i in 1:length(spectrum_list)){
  print(i)
  spectrum2_obj <- spectrum_list[[i]]
  binned_spectrum <- bin(spectrum2_obj, breaks = mzrange)
  
  X[i,] <-  binned_spectrum@intensity
}
X0 <- X

index_d <- which(apply(X,2,sd)!=0)
X <- X0[,index_d]

mzrange_select <- mzrange[index_d]

### コサイン類似度
X1 <- matrix(NA,nrow(X),ncol(X))
for(i in 1:nrow(X)){
  print(i)
  X1[i,] <- X[i,]/sqrt(sum(X[i,]^2))
}
# 
Z <- X1%*%t(X1) # コサイン類似度
Y <- Z

Y[Z>=0.7] <- 1
Y[Z<0.7] <- 0

eig <- eigen(Y) # 近似計算に変えた方がよいか

v <- eig$vectors[,1:15] # 15がよさそうだが、大きめにしておいた方がよいか？
