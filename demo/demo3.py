import os

from LargeScaleMSPy import *


# IDF-only + no autoscaling demo for MS-DIAL / LipidBlast-style MSP data.

msp_file = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
hdf5_file = "C:/Users/hyama/Documents/LargeScaleMSPy/data/test.h5"

processed_file = "C:/Users/hyama/Documents/LargeScaleMSPy/release2/demo_output/full_msdial_idf_processed.hdf5"
pca_file = "C:/Users/hyama/Documents/LargeScaleMSPy/release2/demo_output/full_msdial_pca_idf_noautoscale.npz"
umap_file = "C:/Users/hyama/Documents/LargeScaleMSPy/release2/demo_output/full_msdial_umap_idf_noautoscale.npz"
mz_max = 2000

os.makedirs(os.path.dirname(processed_file), exist_ok=True)

# If the HDF5 file has not been created yet, uncomment this line.
# msp2hdf5(msp_file, hdf5_file, bin_size=0.01, mz_range=(0, 2000))

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
