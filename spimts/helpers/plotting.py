# pyspi/helpers/plotting.py
from typing import Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.cluster.hierarchy import linkage, dendrogram
from .utils import save_fig

# ---------- MTS heatmap ----------
def plot_mts_heatmap(data: np.ndarray, vmin: float = -2, vmax: float = 2, ax=None):
    """data shape assumed (T, M); plot with time on X, channels on Y."""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        created = True
    ax.pcolormesh(data.T, shading="flat", vmin=vmin, vmax=vmax,
                  cmap=sns.color_palette("icefire", as_cmap=True))
    ax.grid(False)
    ax.set_xlabel(None); ax.set_ylabel(None); ax.set_xticks([]); ax.set_yticks([])
    if created:
        plt.tight_layout()
    return ax

# ---------- MPI heatmap ----------
def plot_mpi_heatmap(matrix: np.ndarray, spi: str, cbar: bool = False, ax=None):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        created = True
    
    # Determine appropriate bounds based on SPI type and actual data
    spi_lower = spi.lower()
    
    # Correlation-like measures: bounded [-1, 1]
    correlation_spis = ["spearmanr", "pearsonr", "kendalltau", "crosscorr", "partialcorr"]
    # Information-theoretic measures: non-negative, unbounded
    info_spis = ["mi_", "tlmi_", "te_", "cmi_", "mutualinfo"]
    # Distance measures: non-negative, unbounded
    distance_spis = ["dtw", "dynamictimewarping", "pairwisedistance", "euclidean", "manhattan"]
    # Covariance: unbounded, can be negative
    covariance_spis = ["cov", "covariance"]
    
    # Classify SPI and set appropriate bounds
    if any(s in spi_lower for s in correlation_spis):
        # Correlation-like: [-1, 1] with center at 0
        vmin, vmax, center = -1.0, 1.0, 0.0
    elif any(spi_lower.startswith(s) for s in info_spis):
        # Information-theoretic: [0, data_max]
        vmin, vmax, center = 0.0, None, None
    elif any(s in spi_lower for s in distance_spis):
        # Distance: [0, data_max]
        vmin, vmax, center = 0.0, None, None
    elif any(s in spi_lower for s in covariance_spis):
        # Covariance: data-driven symmetric around 0
        vmin, vmax, center = None, None, 0.0
    else:
        # Unknown SPI: use data-driven bounds
        vmin, vmax, center = None, None, None
    
    # For data-driven bounds, compute from actual values
    if vmin is None or vmax is None:
        finite_vals = matrix[np.isfinite(matrix)]
        if len(finite_vals) > 0:
            data_min, data_max = np.min(finite_vals), np.max(finite_vals)
            if vmin is None:
                vmin = data_min
            if vmax is None:
                vmax = data_max
            # For symmetric ranges around center, ensure symmetry
            if center is not None and vmin is not None and vmax is not None:
                max_abs = max(abs(vmin - center), abs(vmax - center))
                vmin, vmax = center - max_abs, center + max_abs
    
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, center=center, cmap="icefire",
                annot=False, square=True, xticklabels=False, yticklabels=False,
                cbar=cbar, cbar_kws={"shrink": .8}, linewidths=.4, ax=ax)
    ax.set_title(f"{spi}")
    if created:
        plt.tight_layout()
    return ax

# ---------- SPI-space (pairwise off-diag scatter) ----------
def _offdiag(vec_or_mat: np.ndarray) -> np.ndarray:
    A = np.asarray(vec_or_mat)
    if A.ndim == 1:
        return A
    iu = np.triu_indices_from(A, k=1)
    return A[iu]

def _score_pair(x, y, method: str):
    if method == "pearson":
        return pearsonr(x, y)[0]
    if method == "spearman":
        return spearmanr(x, y)[0]
    if method == "kendall":
        return kendalltau(x, y)[0]
    raise ValueError("method must be one of: pearson|spearman|kendall")

def plot_spi_space(matrices: Dict[str, np.ndarray], spi_names: List[str], method: str = "spearman"):
    # vectorise off-diagonals
    off = {k: _offdiag(matrices[k]) for k in spi_names if k in matrices}
    names = list(off.keys())
    n = len(names)
    if n < 2:
        print("[WARN] Need ≥2 SPIs for SPI-space; skipping.")
        return
    n_rows, n_cols = n-1, n-1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows), dpi=180)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            if j < n-1-i:
                a, b = names[i], names[i+j+1]
                x, y = off[a], off[b]
                ok = (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                      and np.std(x) > 0 and np.std(y) > 0)
                r = _score_pair(x, y, method) if ok else np.nan
                # Gray dots, larger size, medium transparency
                ax.scatter(x, y, alpha=0.7, s=12, marker='.', color='gray')
                
                # Color-coded regression line based on correlation strength
                if ok:
                    try:
                        # RdYlBu_r: Red(+1) -> Yellow(0) -> Blue(-1), avoids white at zero
                        cmap = plt.cm.RdYlBu_r
                        norm_r = (r + 1) / 2  # Map [-1, 1] to [0, 1]
                        line_color = cmap(norm_r)
                        
                        z = np.polyfit(x, y, 1)
                        ax.plot(x, np.poly1d(z)(x), color=line_color, linestyle='--', 
                               alpha=0.8, linewidth=1.5)
                    except Exception:
                        pass
                
                ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{method}: {r:.2f}" if np.isfinite(r) else f"{method}: nan")
                # Let matplotlib auto-scale axes (no forced aspect ratio)
            else:
                ax.set_visible(False)
    plt.tight_layout()

def plot_spi_space_individual(matrices: Dict[str, np.ndarray], spi_names: List[str], 
                               output_dir: str, method: str = "spearman"):
    """Plot individual scatter plots for all pairwise SPI comparisons.
    
    Creates n(n-1)/2 individual figures, one per SPI pair, saved to output_dir.
    This is useful for large datasets where the full SPI-space grid is too large to view.
    
    Args:
        matrices: Dict of SPI name -> MPI matrix (2D array)
        spi_names: List of SPI names to include
        output_dir: Directory path where individual plots will be saved
        method: Correlation method ('spearman', 'pearson', or 'kendall')
    
    Returns:
        int: Number of plots created
    """
    # vectorise off-diagonals
    off = {k: _offdiag(matrices[k]) for k in spi_names if k in matrices}
    names = list(off.keys())
    n = len(names)
    
    if n < 2:
        print("[WARN] Need ≥2 SPIs for individual SPI-space plots; skipping.")
        return 0
    
    # Map method to Greek symbol
    corr_symbols = {"spearman": r"$\rho$", "pearson": r"$r$", "kendall": r"$\tau$"}
    symbol = corr_symbols.get(method, method)
    
    plot_count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = names[i], names[j]
            x, y = off[a], off[b]
            
            # Check if data is valid
            ok = (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                  and np.std(x) > 0 and np.std(y) > 0)
            r = _score_pair(x, y, method) if ok else np.nan
            
            # Create individual SQUARE figure
            fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
            # Gray dots, larger size, medium transparency
            ax.scatter(x, y, alpha=0.7, s=15, marker='o', color='gray')
            
            # Add regression line in soft teal #4ECDC4/ purple #B19CD9/coral #FA8072 (single pastel color)
            if ok:
                try:
                    z = np.polyfit(x, y, 1)
                    ax.plot(x, np.poly1d(z)(x), color='#4ECDC4', linestyle='--', 
                           alpha=0.85, linewidth=2.5)
                except Exception:
                    pass
            
            ax.set_xlabel(a, fontsize=12)
            ax.set_ylabel(b, fontsize=12)
            # Title shows only correlation value with appropriate symbol
            ax.set_title(f"{symbol} = {r:.3f}" if np.isfinite(r) else f"{symbol} = nan", fontsize=13)
            ax.set_aspect('equal', adjustable='datalim')  # Square aspect, flexible limits
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            safe_a = a.replace("/", "-").replace("\\", "-").replace(" ", "_")
            safe_b = b.replace("/", "-").replace("\\", "-").replace(" ", "_")
            filename = f"{safe_a}_vs_{safe_b}.png"
            filepath = os.path.join(output_dir, filename)
            
            try:
                save_fig(filepath, fig=fig)
                plot_count += 1
            except Exception as e:
                print(f"[WARN] Failed to save {filename}: {e}")
            finally:
                plt.close(fig)
    
    print(f"[INFO] Created {plot_count} individual SPI-space plots in {output_dir}")
    return plot_count

# ---------- Fingerprint barcode ----------
def fingerprint_matrix(offdiag: Dict[str, np.ndarray], method: str = "spearman") -> pd.DataFrame:
    """Compute pairwise correlation matrix between SPIs."""
    spis = list(offdiag.keys())
    K = len(spis)
    Mx = np.eye(K)
    
    for i in range(K):
        for j in range(i+1, K):
            x, y = offdiag[spis[i]], offdiag[spis[j]]
            
            # Check for constant or invalid arrays
            if np.std(x) == 0 or np.std(y) == 0:
                val = 0.0  # Correlation undefined for constant arrays
            elif not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                val = 0.0  # Handle NaN/Inf
            else:
                try:
                    if method == "pearson":
                        val = pearsonr(x, y)[0]
                    elif method == "spearman":
                        val = spearmanr(x, y)[0]
                    elif method == "kendall":
                        val = kendalltau(x, y)[0]
                    else:
                        raise ValueError("Unknown method")
                    
                    # Sanitize result
                    if not np.isfinite(val):
                        val = 0.0
                except Exception:
                    val = 0.0
            
            Mx[i, j] = Mx[j, i] = val
    
    return pd.DataFrame(Mx, index=spis, columns=spis)


def plot_spi_fingerprint(offdiag_map: Dict[str, np.ndarray], spi_names: List[str], method: str = "spearman"):
    """Plot fingerprint barcode showing pairwise SPI correlations.
    
    Args:
        offdiag_map: Dict of SPI name -> already-vectorized off-diagonal values (1D arrays)
        spi_names: List of SPI names to include (order respected)
        method: Correlation method ('spearman', 'pearson', or 'kendall')
    """
    # restrict to provided names (order respected)
    off = {k: offdiag_map[k] for k in spi_names if k in offdiag_map}
    names = list(off.keys())
    if len(names) < 2:
        print("[WARN] Need ≥2 SPIs for fingerprint; skipping.")
        return
    vals = []
    labels = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            # offdiag_map already contains vectorized data, no need to call _offdiag()
            xa, xb = off[a], off[b]
            ok = (np.all(np.isfinite(xa)) and np.all(np.isfinite(xb))
                  and np.std(xa) > 0 and np.std(xb) > 0)
            if ok:
                r = _score_pair(xa, xb, method)
            else:
                r = np.nan
            vals.append(r); labels.append(f"{a} | {b}")
    arr = np.array(vals)[None, :]
    if method == "spearman" or method == "pearson" or method == "kendall":
        vmin, vmax, center, cmap = -1, 1, 0, "icefire" #prev: 'coolwarm'
    else:
        vmin, vmax, center, cmap = None, None, None, "viridis"
    plt.figure(figsize=(max(6, 0.25*arr.shape[1]), 1.6), dpi=180)
    sns.heatmap(arr, vmin=vmin, vmax=vmax, center=center, cmap=cmap,
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={"shrink": .6})
    plt.title(f"SPI fingerprint ({method})")
    plt.tight_layout()

# ---------- Dendrogram over SPIs ----------
def plot_spi_dendrogram(offdiag: dict[str, np.ndarray], method: str = "spearman", 
                        link: str = "average", ax=None):
    """Plot hierarchical clustering dendrogram of SPIs."""
    if len(offdiag) < 2:
        print("[WARN] Need at least 2 SPIs for dendrogram, skipping.")
        return None
    
    F = fingerprint_matrix(offdiag, method=method)
    
    # Convert to distance and sanitize
    D = 1.0 - F.values
    D = np.clip(D, 0, 2)  # Clip to [0, 2] range
    D = np.nan_to_num(D, nan=1.0, posinf=2.0, neginf=0.0)  # Replace invalid values
    np.fill_diagonal(D, 0.0)
    
    # Check if distance matrix is valid
    iu = np.triu_indices_from(D, k=1)
    dvec = D[iu]
    if not np.all(np.isfinite(dvec)):
        print("[WARN] Distance matrix contains non-finite values after sanitization, skipping dendrogram.")
        return None
    
    if np.all(dvec == 0):
        print("[WARN] All distances are zero, skipping dendrogram.")
        return None
    
    Z = linkage(dvec, method=link)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    dendrogram(Z, labels=F.index.to_list(), leaf_rotation=90, ax=ax)
    ax.set_title(f"SPI Dendrogram ({method}, {link})")
    return ax

