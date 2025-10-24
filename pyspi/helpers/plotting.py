# pyspi/helpers/plotting.py
from typing import Dict, List
import numpy as np
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
    # basic bounds: MI >= 0, others often [-1,1]
    lower_zero = spi.lower().startswith("mi_") or spi.lower().startswith("tlmi_")
    vmin = 0.0 if lower_zero else -1.0
    vmax = None if lower_zero else 1.0
    center = None if lower_zero else 0.0
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
                ax.scatter(x, y, alpha=0.5, s=5, marker='o')
                try:
                    z = np.polyfit(x, y, 1); ax.plot(x, np.poly1d(z)(x), "r--", alpha=0.7)
                except Exception:
                    pass
                ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{method}: {r:.2f}" if np.isfinite(r) else f"{method}: nan")
            else:
                ax.set_visible(False)
    plt.tight_layout()

# ---------- Fingerprint barcode ----------
def fingerprint_matrix(offdiag_map: Dict[str, np.ndarray], method: str = "spearman") -> np.ndarray:
    keys = list(offdiag_map.keys())
    K = len(keys)
    F = np.zeros((K, K), float)
    for i in range(K):
        xi = _offdiag(offdiag_map[keys[i]])
        for j in range(K):
            yj = _offdiag(offdiag_map[keys[j]])
            ok = (np.all(np.isfinite(xi)) and np.all(np.isfinite(yj))
                  and np.std(xi) > 0 and np.std(yj) > 0)
            if not ok:
                F[i, j] = np.nan
            else:
                F[i, j] = _score_pair(xi, yj, method)
    return F, keys

def plot_spi_fingerprint(offdiag_map: Dict[str, np.ndarray], spi_names: List[str], method: str = "spearman"):
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
            xa, xb = _offdiag(off[a]), _offdiag(off[b])
            ok = (np.all(np.isfinite(xa)) and np.all(np.isfinite(xb))
                  and np.std(xa) > 0 and np.std(xb) > 0)
            if ok:
                r = _score_pair(xa, xb, method)
            else:
                r = np.nan
            vals.append(r); labels.append(f"{a} | {b}")
    arr = np.array(vals)[None, :]
    if method == "spearman" or method == "pearson" or method == "kendall":
        vmin, vmax, center, cmap = -1, 1, 0, "coolwarm"
    else:
        vmin, vmax, center, cmap = None, None, None, "viridis"
    plt.figure(figsize=(max(6, 0.25*arr.shape[1]), 1.6), dpi=180)
    sns.heatmap(arr, vmin=vmin, vmax=vmax, center=center, cmap=cmap,
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={"shrink": .6})
    plt.title(f"SPI fingerprint ({method})")
    plt.tight_layout()

# ---------- Dendrogram over SPIs ----------
def plot_spi_dendrogram(offdiag_map: Dict[str, np.ndarray], method: str = "spearman", link: str = "average"):
    F, keys = fingerprint_matrix(offdiag_map, method=method)
    # convert similarity to distance; guard NaNs
    sim = np.nan_to_num(F, nan=0.0)
    dist = 1 - sim
    # condensed form: take upper triangle
    iu = np.triu_indices_from(dist, k=1)
    dvec = dist[iu]
    Z = linkage(dvec, method=link)
    plt.title(f"SPI dendrogram ({method}, link={link})")
    dendrogram(Z, labels=keys, leaf_rotation=90)
    plt.tight_layout()
