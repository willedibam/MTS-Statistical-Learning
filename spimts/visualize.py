# pyspi/visualize.py
"""
Visualization-only pipeline:
- Reads artifacts from results/<profile>/<run_dir>/<model>/
- No Calculator() calls; safe to iterate on plots
"""
import os, argparse, glob, numpy as np
from spimts.helpers.utils import ensure_dir, save_fig, load_numpy_or_none
from spimts.helpers.plotting import (
    plot_mts_heatmap, plot_mpi_heatmap, plot_spi_space, plot_spi_space_individual,
    plot_spi_fingerprint, plot_spi_dendrogram
)
from spimts.helpers.selection import select_core_spis
from spimts.helpers.logging_config import setup_logging

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


def _load_artifacts(model_dir: str, spi_filter: list[str] | None = None):
    """Load artifacts from model directory, optionally filtering to specific SPIs.
    
    Args:
        model_dir: Path to model results directory
        spi_filter: List of exact SPI names to load (None = load all)
    
    Returns:
        (X, matrices, offdiag, spi_names) where matrices/offdiag only contain filtered SPIs
    """
    arrays = os.path.join(model_dir, "arrays")
    times_path = os.path.join(arrays, "timeseries.npy")
    X = load_numpy_or_none(times_path)
    
    # gather all MPIs and offdiags from filenames
    matrices, offdiag, spi_names = {}, {}, []
    for fname in os.listdir(arrays):
        if fname.startswith("mpi_") and fname.endswith(".npy"):
            spi = fname[len("mpi_"):-4]
            # Apply SPI filter if provided
            if spi_filter is None or spi in spi_filter:
                matrices[spi] = np.load(os.path.join(arrays, fname))
                spi_names.append(spi)
        elif fname.startswith("offdiag_") and fname.endswith(".npy"):
            spi = fname[len("offdiag_"):-4]
            # Apply SPI filter if provided
            if spi_filter is None or spi in spi_filter:
                offdiag[spi] = np.load(os.path.join(arrays, fname))
    
    spi_names = sorted(list(set(spi_names)))
    return X, matrices, offdiag, spi_names

def _validate_spi_subset(requested: list[str], available: list[str]) -> list[str]:
    """Validate SPI subset names against available SPIs, with helpful error messages.
    
    Args:
        requested: List of SPI names user requested
        available: List of actually available SPI names from arrays/
    
    Returns:
        List of validated SPI names (subset of available)
    
    Raises:
        SystemExit if any requested SPI is not found (with suggestions)
    """
    missing = [spi for spi in requested if spi not in available]
    
    if missing:
        print(f"\n[ERR] The following SPIs were not found in cached data:")
        for spi in missing:
            # Suggest close matches using simple substring matching
            suggestions = [a for a in available if spi.lower() in a.lower() or a.lower() in spi.lower()]
            if not suggestions:
                # Fallback: Levenshtein-like suggestion (first 3 chars match)
                prefix = spi[:min(3, len(spi))].lower()
                suggestions = [a for a in available if a.lower().startswith(prefix)]
            
            if suggestions:
                print(f"  ✗ '{spi}'  (did you mean: {', '.join(suggestions[:3])}?)")
            else:
                print(f"  ✗ '{spi}'  (no similar matches found)")
        
        print(f"\nAvailable SPIs in this run:")
        for spi in sorted(available):
            print(f"  - {spi}")
        print(f"\n[INFO] Use exact SPI names from arrays/mpi_*.npy files (without 'mpi_' prefix)")
        raise SystemExit(1)
    
    return requested


def _select_mpi_spis(all_spis: list[str], mode: str) -> list[str]:
    """Select which SPIs to generate MPI heatmaps for.
    
    Args:
        all_spis: All available SPI names
        mode: One of 'none', 'core', 'all', or a number string
    
    Returns:
        List of SPI names to plot
    """
    if mode == 'none' or mode == '0':
        return []
    
    if mode == 'all':
        return all_spis
    
    if mode == 'core':
        # Use substring matching for "core" SPIs
        core_patterns = ["spearmanr", "cov_", "mi_", "te_", "dtw", "pairwisedistance"]
        selected = []
        for spi in all_spis:
            spi_lower = spi.lower()
            if any(pattern in spi_lower for pattern in core_patterns):
                selected.append(spi)
        return selected[:6] if len(selected) > 6 else selected  # Limit to 6
    
    # Try parsing as integer
    try:
        limit = int(mode)
        return all_spis[:limit] if limit > 0 else []
    except ValueError:
        print(f"[WARN] Invalid --include-mpi-heatmaps value '{mode}', defaulting to 'core'")
        return _select_mpi_spis(all_spis, 'core')


def main():
    p = argparse.ArgumentParser(description="Visualization-only: read artifacts and plot.")
    p.add_argument("--profile", choices=["dev","dev+","dev++","dev+++","paper"], default="dev")
    p.add_argument("--root", default="./results", help="Results root (contains dev/ and paper/).")
    p.add_argument("--models", default="", help="Comma-separated filter for model folder names (substr match).")
    p.add_argument("--include-mpi-heatmaps", default="core", 
                   help="Which MPI heatmaps to save: 'all', 'core', 'none'/'0', or an integer limit. Default: 'core'")
    p.add_argument("--spi-subset", action="append", dest="spi_subsets",
                   help="Comma-separated exact SPI names for subset visualization. Can be used multiple times for multiple subsets. Example: --spi-subset 'mi_kraskov_NN-4,SpearmanR,cov_EmpiricalCovariance'")
    p.add_argument("--run-id", default=None, help="Visualize a specific past run folder under results/<profile> matching <run_id>_<Model>.")
    p.add_argument("--all-runs", action="store_true", help="Process ALL matching runs (by profile/models) instead of only the latest per model.")
    p.add_argument("--log-file", default=None, 
                   help="Log to file: 'auto' (auto-named in ./logs/), explicit path, or omit for terminal-only output.")

    args = p.parse_args()
    
    # Setup logging
    logger = setup_logging('visualize', args.profile, args.log_file)

    base = os.path.join(args.root, args.profile)
    if not os.path.isdir(base):
        logger.error(f"No such results folder: {base}")
        return

    model_filters = [m.strip() for m in args.models.split(",") if m.strip()]
    
    # Parse SPI subsets if provided
    spi_subsets = []
    if args.spi_subsets:
        for subset_str in args.spi_subsets:
            subset = [s.strip() for s in subset_str.split(",") if s.strip()]
            if subset:
                spi_subsets.append(subset)

    # NEW: discover runs with optional --run-id / --all-runs filters
    runs = _collect_runs(
        base=base,
        model_filters=model_filters,
        run_id=args.run_id,
        all_runs=args.all_runs
    )

    if not runs:
        logger.warning("No matching runs found. Check --profile / --models / --run-id.")
        return
    
    logger.info(f"Found {len(runs)} model run(s) to visualize")
    logger.info(f"Profile: {args.profile} | MPI heatmaps: {args.include_mpi_heatmaps}")
    if spi_subsets:
        logger.info(f"SPI subsets: {len(spi_subsets)} subset(s) requested")
    logger.info("")  # Blank line for readability

    for idx, (rid, model_safe, model_dir, _mtime) in enumerate(runs, 1):
        logger.info(f"[{idx}/{len(runs)}] {model_safe} (run_id={rid})")

        plots_dir = ensure_dir(os.path.join(model_dir, "plots"))
        mpis_dir = ensure_dir(os.path.join(plots_dir, "mpis"))
        spi_space_dir = ensure_dir(os.path.join(plots_dir, "spi_space"))
        spi_space_individual_dir = ensure_dir(os.path.join(plots_dir, "spi_space_individual"))
        fingerprint_dir = ensure_dir(os.path.join(plots_dir, "fingerprint"))
        mts_dir = ensure_dir(os.path.join(plots_dir, "mts"))

        # Load ALL available SPIs first (no filter)
        X, all_matrices, all_offdiag, all_spi_names = _load_artifacts(model_dir, spi_filter=None)
        logger.info(f"  Loaded {len(all_spi_names)} SPIs from cached artifacts")
        
        # MTS heatmap (always plot)
        if X is not None:
            ax = plot_mts_heatmap(X)
            save_fig(os.path.join(mts_dir, "mts_heatmap.png"), fig=ax.figure)
            logger.info(f"  Saved MTS heatmap (T={X.shape[0]}, M={X.shape[1]})")

        # MPI heatmaps: controlled by --include-mpi-heatmaps
        mpi_spis_to_plot = _select_mpi_spis(all_spi_names, args.include_mpi_heatmaps)
        if mpi_spis_to_plot:
            for spi in mpi_spis_to_plot:
                if spi in all_matrices:
                    ax = plot_mpi_heatmap(all_matrices[spi], spi, cbar=True)
                    save_fig(os.path.join(mpis_dir, f"mpi_{spi}.png"), fig=ax.figure)
            logger.info(f"  Saved {len(mpi_spis_to_plot)} MPI heatmaps")
        else:
            logger.info(f"  Skipped MPI heatmaps (mode: {args.include_mpi_heatmaps})")

        # === FULL SPI-SPACE (all SPIs) ===
        CORE_SPIS = ["SpearmanR", "Covariance", "KendallTau", "MutualInfo", "TimeLaggedMutualInfo","TransferEntropy",
                    "DynamicTimeWarping", "PairwiseDistance", "CrossCorrelation"]
        preferred = select_core_spis(all_matrices.keys(), CORE_SPIS)
        rest = [s for s in all_matrices.keys() if s not in preferred]
        order = preferred + rest
        
        if len(order) >= 2:
            # Generate plots for all three correlation methods
            logger.info(f"  Generating SPI-space plots (grid + individual)...")
            for method in ["spearman", "kendall", "pearson"]:
                plot_spi_space(all_matrices, order, method=method)
                save_fig(os.path.join(spi_space_dir, f"spi_space_{method}.png"))
                
                # Individual scatter plots for each SPI pair
                method_dir = ensure_dir(os.path.join(spi_space_individual_dir, method))
                plot_spi_space_individual(all_matrices, order, method_dir, method=method)
            logger.info(f"  Saved 3 grid plots + {len(order)*(len(order)-1)//2} individual plots per method")

        # Fingerprints + dendrogram for all three methods (full SPI set)
        if len(all_offdiag) >= 2:
            fp_spis = list(all_offdiag.keys())
            if len(fp_spis) >= 2:
                logger.info(f"  Generating fingerprints and dendrograms...")
                for method in ["spearman", "kendall", "pearson"]:
                    plot_spi_fingerprint(all_offdiag, fp_spis, method=method)
                    save_fig(os.path.join(fingerprint_dir, f"fingerprint_{method}.png"))
                    plot_spi_dendrogram(all_offdiag, method=method, link="average")
                    save_fig(os.path.join(fingerprint_dir, f"dendrogram_{method}.png"))
                logger.info(f"  Saved 3 fingerprints + 3 dendrograms")
        
        # === SPI SUBSETS (if requested) ===
        if spi_subsets:
            subsets_dir = ensure_dir(os.path.join(plots_dir, "subsets"))
            logger.info(f"  Generating {len(spi_subsets)} SPI subset visualization(s)...")
            
            for subset in spi_subsets:
                # Validate subset SPIs exist
                validated_subset = _validate_spi_subset(subset, all_spi_names)
                
                # Create folder name: alphabetically sorted, joined by '+'
                subset_name = "+".join(sorted(validated_subset))
                subset_dir = ensure_dir(os.path.join(subsets_dir, subset_name))
                subset_spi_space_dir = ensure_dir(os.path.join(subset_dir, "spi_space"))
                subset_spi_space_individual_dir = ensure_dir(os.path.join(subset_dir, "spi_space_individual"))
                subset_fingerprint_dir = ensure_dir(os.path.join(subset_dir, "fingerprint"))
                
                # Filter matrices and offdiag to subset
                subset_matrices = {k: v for k, v in all_matrices.items() if k in validated_subset}
                subset_offdiag = {k: v for k, v in all_offdiag.items() if k in validated_subset}
                
                if len(validated_subset) >= 2:
                    # Generate subset SPI-space plots
                    for method in ["spearman", "kendall", "pearson"]:
                        plot_spi_space(subset_matrices, validated_subset, method=method)
                        save_fig(os.path.join(subset_spi_space_dir, f"spi_space_{method}.png"))
                        
                        # Individual scatter plots
                        method_dir = ensure_dir(os.path.join(subset_spi_space_individual_dir, method))
                        plot_spi_space_individual(subset_matrices, validated_subset, method_dir, method=method)
                    
                    # Fingerprints + dendrograms for subset
                    for method in ["spearman", "kendall", "pearson"]:
                        plot_spi_fingerprint(subset_offdiag, validated_subset, method=method)
                        save_fig(os.path.join(subset_fingerprint_dir, f"fingerprint_{method}.png"))
                        plot_spi_dendrogram(subset_offdiag, method=method, link="average")
                        save_fig(os.path.join(subset_fingerprint_dir, f"dendrogram_{method}.png"))
        
        logger.info(f"  Completed visualization for {model_safe}")
        logger.info("")  # Blank line between models
    
    logger.info("All visualizations complete!")


if __name__ == "__main__":
    main()
