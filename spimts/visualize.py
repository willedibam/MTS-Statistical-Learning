# pyspi/visualize.py
"""
Visualization-only pipeline:
- Reads artifacts from results/<profile>/<run_dir>/<model>/
- No Calculator() calls; safe to iterate on plots
"""
import os, argparse, numpy as np
from spimts.helpers.utils import ensure_dir, save_fig, load_numpy_or_none
from spimts.helpers.plotting import (
    plot_mts_heatmap, plot_mpi_heatmap, plot_spi_space, plot_spi_space_individual,
    plot_spi_fingerprint, plot_spi_dendrogram
)
from spimts.helpers.selection import select_core_spis

def _parse_run_dir_name(name: str):
    """Split '<run_id>_<model_safe>' into (run_id, model_safe)."""
    if "_" not in name:
        return None, None
    i = name.find("_")
    return name[:i], name[i+1:]

def _collect_runs(base: str, model_filters=None, run_id: str | None = None, all_runs: bool = False):
    """
    Scan results/<profile>/ and return a list of (run_id, model_safe, run_dir, mtime),
    filtered by model substrings (case-insensitive) and optional run_id (exact or prefix).
    If all_runs=False, keep only the latest per model.
    """
    import glob, os
    if not os.path.isdir(base):
        return []

    # model filters normalized (list of lowercase substrings)
    model_filters = [m.strip().lower() for m in (model_filters or []) if m.strip()]

    cand_dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    rows = []
    for d in cand_dirs:
        rid, model_safe = _parse_run_dir_name(os.path.basename(d))
        if rid is None:
            continue
        mtime = os.path.getmtime(d)
        # filter by models (substring match on safe name)
        if model_filters and not any(f in model_safe.lower() for f in model_filters):
            continue
        # filter by run_id (exact or prefix)
        if run_id and not (rid == run_id or rid.startswith(run_id)):
            continue
        rows.append((rid, model_safe, d, mtime))

    if not rows:
        return []

    if not all_runs:
        # keep the latest per model
        latest = {}
        for rid, msafe, d, mtime in rows:
            if (msafe not in latest) or (mtime > latest[msafe][3]):
                latest[msafe] = (rid, msafe, d, mtime)
        rows = list(latest.values())

    rows.sort(key=lambda r: r[3], reverse=True)  # newest first
    return rows


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
    p.add_argument("--profile", choices=["dev","dev+","paper"], default="dev")
    p.add_argument("--root", default="./results", help="Results root (contains dev/ and paper/).")
    p.add_argument("--models", default="", help="Comma-separated filter for model folder names (substr match).")
    p.add_argument("--spi-limit", type=int, default=12, help="Max # of SPI heatmaps per model (visual convenience).")
    p.add_argument("--run-id", default=None, help="Visualize a specific past run folder under results/<profile> matching <run_id>_<Model>.")
    p.add_argument("--all-runs", action="store_true", help="Process ALL matching runs (by profile/models) instead of only the latest per model.")

    args = p.parse_args()

    base = os.path.join(args.root, args.profile)
    if not os.path.isdir(base):
        print(f"[ERR] No such results folder: {base}")
        return

    model_filters = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    base = os.path.join(args.root, args.profile)
    if not os.path.isdir(base):
        print(f"[ERR] No such results folder: {base}")
        return

    model_filters = [m.strip() for m in args.models.split(",") if m.strip()]

    # NEW: discover runs with optional --run-id / --all-runs filters
    runs = _collect_runs(
        base=base,
        model_filters=model_filters,
        run_id=args.run_id,
        all_runs=args.all_runs
    )

    if not runs:
        print("[INFO] No matching runs found. Check --profile / --models / --run-id.")
        return

    for rid, model_safe, model_dir, _mtime in runs:
        print(f"[VIZ] {model_safe}  (run_id={rid})")

        plots_dir = ensure_dir(os.path.join(model_dir, "plots"))
        mpis_dir = ensure_dir(os.path.join(plots_dir, "mpis"))
        spi_space_dir = ensure_dir(os.path.join(plots_dir, "spi_space"))
        spi_space_individual_dir = ensure_dir(os.path.join(plots_dir, "spi_space_individual"))
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

        # SPI-space: ALL SPIs present, preferred first
        CORE_SPIS = ["SpearmanR", "Covariance", "MutualInfo", "TransferEntropy",
                    "DynamicTimeWarping", "PairwiseDistance"]
        preferred = select_core_spis(matrices.keys(), CORE_SPIS)
        rest = [s for s in matrices.keys() if s not in preferred]
        order = preferred + rest
        if len(order) >= 2:
            plot_spi_space(matrices, order, method="spearman")
            save_fig(os.path.join(spi_space_dir, "spi_space_spearman.png"))
            
            # Individual scatter plots for each SPI pair
            plot_spi_space_individual(matrices, order, spi_space_individual_dir, method="spearman")

        # Fingerprints + dendrogram
        if len(offdiag) >= 2:
            fp_spis = list(offdiag.keys())
            if len(fp_spis) >= 2:
                plot_spi_fingerprint(offdiag, fp_spis, method="spearman")
                save_fig(os.path.join(fingerprint_dir, "fingerprint_spearman.png"))
                plot_spi_dendrogram(offdiag, method="spearman", link="average")
                save_fig(os.path.join(fingerprint_dir, "dendrogram_spearman.png"))

if __name__ == "__main__":
    main()
