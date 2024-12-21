from .create_umap_app import create_dash_app
from .load_hdf5 import load_spectrum_from_hdf5
from .pca_umap_analysis import perform_pca
from .msp2hdf5 import convert_msp_to_hdf5

def run_app():
    # 必要なデータのロードや前処理
    app = create_dash_app(umap_data, spectrum_loader)
    app.run_server(debug=True)
