import os

from .vizapp import create_umap_app, load_umap_results, parse_msp_to_line_info


def run_app():
    """
    Launch the default UMAP viewer app.

    The viewer expects the following environment variables:
      - LARGESCALEMSPY_UMAP_FILE
      - LARGESCALEMSPY_MSP_FILE
      - LARGESCALEMSPY_MZ_MAX (optional, default: 2000)
      - LARGESCALEMSPY_DEBUG (optional, default: false)
    """
    umap_file = os.environ.get("LARGESCALEMSPY_UMAP_FILE")
    msp_file = os.environ.get("LARGESCALEMSPY_MSP_FILE")
    mz_max = float(os.environ.get("LARGESCALEMSPY_MZ_MAX", "2000"))
    debug = os.environ.get("LARGESCALEMSPY_DEBUG", "false").lower() in ("1", "true", "yes")

    if not umap_file or not msp_file:
        raise ValueError(
            "run_app() requires environment variables "
            "LARGESCALEMSPY_UMAP_FILE and LARGESCALEMSPY_MSP_FILE."
        )

    line_info = parse_msp_to_line_info(msp_file)
    umap_df = load_umap_results(umap_file)
    app = create_umap_app(umap_df, msp_file, line_info, mz_max=mz_max)
    app.run_server(debug=debug)
