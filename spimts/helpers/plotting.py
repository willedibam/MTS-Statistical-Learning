# pyspi/helpers/plotting.py
from typing import Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import gaussian_kde
from .utils import save_fig

# ========== COLOR SCHEME CONFIGURATION ==========
# Uncomment ONE color scheme to use for SPI-space plots

# Option 1: MONOCHROMATIC (grays + teal accent)
# COLOR_SCHEME = "monochrome"
# SCATTER_COLOR = "#4A5568"      # Charcoal gray
# SCATTER_ALPHA = 0.6
# LINE_COLOR = "#0F766E"          # Deep teal
# LINE_ALPHA = 0.85
# MARGINAL_COLOR = "#4A5568"      # Match scatter
# MARGINAL_ALPHA = 0.5
# KDE_COLOR = "#2D3748"           # Darker gray

# Option 2: DUOTONE (navy + amber)
COLOR_SCHEME = "duotone"
SCATTER_COLOR = "#1E3A8A"       # Navy blue
SCATTER_ALPHA = 0.6
LINE_COLOR = "#F59E0B"          # Amber
LINE_ALPHA = 0.85
MARGINAL_COLOR = "#1E3A8A"      # Match scatter
MARGINAL_ALPHA = 0.5
KDE_COLOR = "#1E40AF"           # Darker navy

# # # Option 3: ROYAL BLUE custom (with matching red/purple)
# COLOR_SCHEME = "royal"
# SCATTER_COLOR = "#090088"       # Royal blue
# SCATTER_ALPHA = 0.6
# LINE_COLOR = "#880000"          # Deep red (matched to blue)
# LINE_ALPHA = 0.85
# MARGINAL_COLOR = "#090088"      # Match scatter
# MARGINAL_ALPHA = 0.5
# KDE_COLOR = "#5D3FD3"           # Purple
# # Alternative: Use LINE_COLOR = "#5D3FD3" for purple trendline

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
    
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, center=center, cmap="gray", #icefire
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

def plot_spi_space(matrices: Dict[str, np.ndarray], spi_names: List[str], method: str = "spearman", marginals: bool = False):
    """Plot SPI-space grid with all pairwise comparisons.
    
    Args:
        matrices: Dict of SPI name -> MPI matrix (2D array)
        spi_names: List of SPI names to include
        method: Correlation method ('spearman', 'pearson', or 'kendall')
        marginals: If True, add marginal distributions on grid edges. Default: False.
                  WARNING: Marginals on large grids (>5 SPIs) may be hard to read.
    
    Note: Issue with non-square subplots comes from using (n-1, n-1) subplots
    but plotting different aspect ratios. Fixing by ensuring square aspect.
    """
    # vectorise off-diagonals
    off = {k: _offdiag(matrices[k]) for k in spi_names if k in matrices}
    names = list(off.keys())
    n = len(names)
    if n < 2:
        print("[WARN] Need ≥2 SPIs for SPI-space; skipping.")
        return
    
    # Map method to Greek symbol
    corr_symbols = {"spearman": r"$\rho$", "pearson": r"$r$", "kendall": r"$\tau$"}
    symbol = corr_symbols.get(method, method)
    
    n_rows, n_cols = n-1, n-1
    
    if marginals:
        # Use GridSpec to add space for marginals
        from matplotlib.gridspec import GridSpec
        
        # Create figure with extra space for marginals
        fig = plt.figure(figsize=(4*n_cols + 2, 4*n_rows + 2), dpi=180)
        
        # Create grid: main plot area + top/right marginals
        gs = GridSpec(n_rows + 1, n_cols + 1, figure=fig,
                     width_ratios=[4]*n_cols + [1],
                     height_ratios=[1] + [4]*n_rows,
                     hspace=0.05, wspace=0.05)
        
        # Create main subplot axes
        axes = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                axes[i, j] = fig.add_subplot(gs[i+1, j])
        
        # Create marginal axes
        marginal_top = [fig.add_subplot(gs[0, j]) for j in range(n_cols)]
        marginal_right = [fig.add_subplot(gs[i+1, n_cols]) for i in range(n_rows)]
        
    else:
        # Use square figsize per subplot to avoid rectangular distortion
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=180)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        marginal_top = None
        marginal_right = None
    
    # Track data for marginals (if enabled)
    col_data = {j: [] for j in range(n_cols)}  # x-axis data per column
    row_data = {i: [] for i in range(n_rows)}  # y-axis data per row
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            if j < n-1-i:
                a, b = names[i], names[i+j+1]
                x, y = off[a], off[b]
                ok = (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                      and np.std(x) > 0 and np.std(y) > 0)
                r = _score_pair(x, y, method) if ok else np.nan
                
                # Track data for marginals
                if marginals and ok:
                    col_data[j].extend(x)
                    row_data[i].extend(y)
                
                # Scatter with configured colors
                ax.scatter(x, y, alpha=SCATTER_ALPHA, s=12, marker='.', color=SCATTER_COLOR)
                
                # Polyfit line (NOT regression, just best-fit trend)
                if ok:
                    try:
                        z = np.polyfit(x, y, 1)
                        ax.plot(x, np.poly1d(z)(x), color=LINE_COLOR, linestyle='--', 
                               alpha=LINE_ALPHA, linewidth=1.5)
                    except Exception:
                        pass
                
                # Title shows correlation coefficient with Greek symbol
                ax.set_xlabel(a, fontsize=9)
                ax.set_ylabel(b, fontsize=9)
                ax.set_title(f"{symbol} = {r:.2f}" if np.isfinite(r) else f"{symbol} = nan", fontsize=10)
                
                if not marginals:
                    ax.set_aspect('equal', adjustable='datalim')  # Force square aspect (only if no marginals)
                ax.grid(True, alpha=0.2)
            else:
                ax.set_visible(False)
    
    # Add marginals if requested
    if marginals and marginal_top is not None and marginal_right is not None:
        # Top marginals (x-axis distributions per column)
        for j in range(n_cols):
            if col_data[j]:
                data = np.array(col_data[j])
                marginal_top[j].hist(data, bins=30, color=MARGINAL_COLOR, alpha=MARGINAL_ALPHA, edgecolor='none')
                marginal_top[j].set_xticks([])
                marginal_top[j].set_yticks([])
                marginal_top[j].spines['top'].set_visible(False)
                marginal_top[j].spines['right'].set_visible(False)
                marginal_top[j].spines['left'].set_visible(False)
            else:
                marginal_top[j].set_visible(False)
        
        # Right marginals (y-axis distributions per row)
        for i in range(n_rows):
            if row_data[i]:
                data = np.array(row_data[i])
                marginal_right[i].hist(data, bins=30, color=MARGINAL_COLOR, alpha=MARGINAL_ALPHA, 
                                      edgecolor='none', orientation='horizontal')
                marginal_right[i].set_xticks([])
                marginal_right[i].set_yticks([])
                marginal_right[i].spines['top'].set_visible(False)
                marginal_right[i].spines['right'].set_visible(False)
                marginal_right[i].spines['bottom'].set_visible(False)
            else:
                marginal_right[i].set_visible(False)
    
    plt.tight_layout()

def plot_spi_space_individual(matrices: Dict[str, np.ndarray], spi_names: List[str], 
                               output_dir: str, method: str = "spearman"):
    """Plot individual scatter plots for all pairwise SPI comparisons WITH MARGINALS.
    
    Creates n(n-1)/2 individual figures with marginal distributions, one per SPI pair.
    
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
            
            # Create joint plot with marginals using seaborn
            g = sns.JointGrid(x=x, y=y, height=7, ratio=5, space=0.15)
            
            # Main scatter plot
            g.ax_joint.scatter(x, y, alpha=SCATTER_ALPHA, s=20, marker='o', color=SCATTER_COLOR)
            
            # Add polyfit line (NOT regression - just trend)
            if ok:
                try:
                    z = np.polyfit(x, y, 1)
                    x_sorted = np.sort(x)
                    g.ax_joint.plot(x_sorted, np.poly1d(z)(x_sorted), color=LINE_COLOR, 
                                   linestyle='--', alpha=LINE_ALPHA, linewidth=2.5)
                except Exception:
                    pass
            
            # Marginal histograms with KDE overlay
            g.ax_marg_x.hist(x, bins=30, color=MARGINAL_COLOR, alpha=MARGINAL_ALPHA, edgecolor='none')
            g.ax_marg_y.hist(y, bins=30, color=MARGINAL_COLOR, alpha=MARGINAL_ALPHA, 
                            edgecolor='none', orientation='horizontal')
            
            # Add KDE overlays on marginals (smooth trend line)
            if ok:
                try:
                    # X marginal KDE
                    kde_x = gaussian_kde(x)
                    x_range = np.linspace(x.min(), x.max(), 200)
                    kde_x_vals = kde_x(x_range)
                    # Scale KDE to match histogram height
                    hist_x, _ = np.histogram(x, bins=30)
                    scale_x = hist_x.max() / kde_x_vals.max() if kde_x_vals.max() > 0 else 1
                    ax2_x = g.ax_marg_x.twinx()
                    ax2_x.plot(x_range, kde_x_vals * scale_x, color=KDE_COLOR, 
                              linewidth=2.5, alpha=0.9)
                    ax2_x.set_ylim(0, hist_x.max() * 1.1)
                    ax2_x.axis('off')
                    
                    # Y marginal KDE
                    kde_y = gaussian_kde(y)
                    y_range = np.linspace(y.min(), y.max(), 200)
                    kde_y_vals = kde_y(y_range)
                    hist_y, _ = np.histogram(y, bins=30)
                    scale_y = hist_y.max() / kde_y_vals.max() if kde_y_vals.max() > 0 else 1
                    ax2_y = g.ax_marg_y.twiny()
                    ax2_y.plot(kde_y_vals * scale_y, y_range, color=KDE_COLOR, 
                              linewidth=2.5, alpha=0.9)
                    ax2_y.set_xlim(0, hist_y.max() * 1.1)
                    ax2_y.axis('off')
                except Exception as e:
                    # KDE might fail for some data distributions
                    pass
            
            # Labels and title
            g.ax_joint.set_xlabel(a, fontsize=12)
            g.ax_joint.set_ylabel(b, fontsize=12)
            g.fig.suptitle(f"{symbol} = {r:.3f}" if np.isfinite(r) else f"{symbol} = nan", 
                          fontsize=14, y=0.98)
            g.ax_joint.grid(True, alpha=0.3)
            
            # Save figure
            safe_a = a.replace("/", "-").replace("\\", "-").replace(" ", "_")
            safe_b = b.replace("/", "-").replace("\\", "-").replace(" ", "_")
            filename = f"{safe_a}_vs_{safe_b}.png"
            filepath = os.path.join(output_dir, filename)
            
            try:
                save_fig(filepath, fig=g.fig)
                plot_count += 1
            except Exception as e:
                print(f"[WARN] Failed to save {filename}: {e}")
            finally:
                plt.close(g.fig)
    
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

