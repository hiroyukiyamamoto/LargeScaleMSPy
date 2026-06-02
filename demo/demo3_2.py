from LargeScaleMSPy import *


# Viewer-only demo for results created by demo3.py.

msp_file = "C:/Users/hyama/Documents/R/MSinfoR/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp"
umap_file = "C:/Users/hyama/Documents/LargeScaleMSPy/release2/demo_output/full_msdial_umap_idf_noautoscale.npz"
mz_max = 2000

line_info = parse_msp_to_line_info(msp_file)
umap_df = load_umap_results(umap_file)

print("Dash app: http://127.0.0.1:8050/")
app = create_umap_app(umap_df, msp_file, line_info, mz_max=mz_max)
app.run(debug=False)
