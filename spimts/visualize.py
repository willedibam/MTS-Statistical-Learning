# pyspi/visualize.py
"""
Visualization-only pipeline:
- Reads artifacts from results/<profile>/<run_dir>/<model>/
- No Calculator() calls; safe to iterate on plots
"""
import os, argparse, numpy as np
from spimts.helpers.utils import ensure_dir, save_fig, list_model_runs, load_numpy_or_none
from spimts.helpers.plotting import (
    plot_mts_heatmap, plot_mpi_heatmap, plot_spi_space,
    plot_spi_fingerprint, plot_spi_dendrogram
)
from spimts.helpers.selection import select_core_spis

def _load_artifacts(model_dir: str):
    arrays = os.path.join(model_dir, "arrays")
    times_path = os.path.join(arrays, "timeseries.npy")
    X = load_numpy_or_none(times_path)
    # gather all MPIs and offdiags from filenames
    matrices, offdiag, spi_names = {}, {}, []
    for fname in os.listdir(arrays):
        if fname.startswith("mpi_") and fname.endswith(".npy"):
            spi = fname[len("mpi_"):-4]
            matrices[spi] = np.load(os.path.join(arrays, fname))
            spi_names.append(spi)
        elif fname.startswith("offdiag_") and fname.endswith(".npy"):
            spi = fname[len("offdiag_"):-4]
            offdiag[spi] = np.load(os.path.join(arrays, fname))
    spi_names = sorted(list(set(spi_names)))
    return X, matrices, offdiag, spi_names

def main():
    p = argparse.ArgumentParser(description="Visualization-only: read artifacts and plot.")
    p.add_argument("--profile", choices=["dev","paper"], default="dev")
    p.add_argument("--root", default="./results", help="Results root (contains dev/ and paper/).")
    p.add_argument("--models", default="", help="Comma-separated filter for model folder names (substr match).")
    p.add_argument("--spi-limit", type=int, default=12, help="Max # of SPI heatmaps per model (visual convenience).")
    args = p.parse_args()

    base = os.path.join(args.root, args.profile)
    if not os.path.isdir(base):
        print(f"[ERR] No such results folder: {base}")
        return

    model_filters = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    for model_dir, model_name in list_model_runs(base):
        if model_filters and not any(f in model_name.lower() for f in model_filters):
            continue
        print(f"[VIZ] {model_name}")

        plots_dir = ensure_dir(os.path.join(model_dir, "plots"))
        mpis_dir = ensure_dir(os.path.join(plots_dir, "mpis"))
        spi_space_dir = ensure_dir(os.path.join(plots_dir, "spi_space"))
        fingerprint_dir = ensure_dir(os.path.join(plots_dir, "fingerprint"))
        mts_dir = ensure_dir(os.path.join(plots_dir, "mts"))

        X, matrices, offdiag, spi_names = _load_artifacts(model_dir)
        if X is not None:
            ax = plot_mts_heatmap(X)
            save_fig(os.path.join(mts_dir, "mts_heatmap.png"), fig=ax.figure)

        # Save up to N MPI heatmaps for quick inspection
        for i, spi in enumerate(spi_names):
            if spi not in matrices: 
                continue
            if i < args.spi_limit:
                ax = plot_mpi_heatmap(matrices[spi], spi, cbar=True)
                save_fig(os.path.join(mpis_dir, f"mpi_{spi}.png"), fig=ax.figure)

        # SPI-space: use ALL SPIs present, preferred targets first
        CORE_SPIS = ["SpearmanR", "Covariance", "MutualInfo", "TransferEntropy",
                     "DynamicTimeWarping", "PairwiseDistance"]
        preferred = select_core_spis(matrices.keys(), CORE_SPIS)
        rest = [s for s in matrices.keys() if s not in preferred]
        order = preferred + rest
        if len(order) >= 2:
            plot_spi_space(matrices, order, method="spearman")
            save_fig(os.path.join(spi_space_dir, "spi_space_spearman.png"))

        # Fingerprints + dendrogram
        if len(offdiag) >= 2:
            # barcode over ALL SPIs available
            fp_spis = list(offdiag.keys())
            if len(fp_spis) >= 2:
                plot_spi_fingerprint(offdiag, fp_spis, method="spearman")
                save_fig(os.path.join(fingerprint_dir, "fingerprint_spearman.png"))
                # dendrogram over all
                plot_spi_dendrogram(offdiag, method="spearman", link="average")
                save_fig(os.path.join(fingerprint_dir, "dendrogram_spearman.png"))

if __name__ == "__main__":
    main()
