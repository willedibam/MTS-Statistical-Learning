# pyspi/compute.py
from __future__ import annotations
import os
os.environ["TSLEARN_BACKEND"] = "numpy"  # <-- exact
import argparse, json, time, hashlib, psutil
from dataclasses import dataclass
from typing import Dict, Callable, List

import numpy as np
import pandas as pd
import yaml

from pyspi.calculator import Calculator

# your generators module (copy your current generators.py here)
from spimts.generators import *

# --------------------- profiles (dev vs paper) -------------------------
PROFILES = {
    "dev": {
        # Fast testing: M=4 (6 edges), T=500 (~2-3 min per model)
        "VAR(1)": dict(M=4, T=500),
        "OU-network": dict(M=4, T=500),
        "Kuramoto": dict(M=4, T=500),
        "Stuart-Landau": dict(M=4, T=500),
        "Lorenz-96": dict(M=4, T=500),
        "Rössler-coupled": dict(M=4, T=500),
        "CML-logistic": dict(M=4, T=500),
        "OU-heavyTail": dict(M=4, T=500),
        "GBM-returns": dict(M=4, T=500),
        "TimeWarp-clones": dict(M=4, T=500),
        "Cauchy-OU": dict(M=4, T=500),
        "Unidirectional-Cascade": dict(M=4, T=500),
        "Quadratic-Coupling": dict(M=4, T=500),
        "Exponential-Transform": dict(M=4, T=500),
        "Phase-Lagged-Oscillators": dict(M=4, T=500),
        # Noise models (null hypotheses) - enabled via --include-noise flag
        "Gaussian-Noise": dict(M=4, T=500),
        "Cauchy-Noise": dict(M=4, T=500),
        "t-Noise": dict(M=4, T=500),
        "Exponential-Noise": dict(M=4, T=500),
    },
    "dev+": {
        # Medium testing: M=10 (45 edges), T=1000 (~10-15 min per model)
        "VAR(1)": dict(M=10, T=1000),
        "OU-network": dict(M=10, T=1000),
        "Kuramoto": dict(M=10, T=1000),
        "Stuart-Landau": dict(M=10, T=1000),
        "Lorenz-96": dict(M=10, T=1000),
        "Rössler-coupled": dict(M=10, T=1000),
        "OU-heavyTail": dict(M=10, T=1000),
        "GBM-returns": dict(M=10, T=1000),
        "TimeWarp-clones": dict(M=10, T=1000),
        "CML-logistic": dict(M=20, T=500),
        "Cauchy-OU": dict(M=10, T=1000),
        "Unidirectional-Cascade": dict(M=10, T=1000),
        "Quadratic-Coupling": dict(M=10, T=1000),
        "Exponential-Transform": dict(M=10, T=1000),
        "Phase-Lagged-Oscillators": dict(M=10, T=1500),
        # Noise models (null hypotheses) - enabled via --include-noise flag
        "Gaussian-Noise": dict(M=10, T=1000),
        "Cauchy-Noise": dict(M=10, T=1000),
        "t-Noise": dict(M=10, T=1000),
        "Exponential-Noise": dict(M=10, T=1000),
    },
    "dev++": {
        # Validation: M=15 (105 edges), T=1000-2000 (~20-30 min per model)
        "VAR(1)": dict(M=15, T=1000),
        "OU-network": dict(M=15, T=1000),
        "Kuramoto": dict(M=15, T=1000),
        "Stuart-Landau": dict(M=15, T=1000),
        "Lorenz-96": dict(M=15, T=1000),
        "Rössler-coupled": dict(M=15, T=1000),
        "OU-heavyTail": dict(M=15, T=1000),
        "GBM-returns": dict(M=15, T=1000),
        "TimeWarp-clones": dict(M=15, T=1000),
        "CML-logistic": dict(M=25, T=500),
        "Cauchy-OU": dict(M=15, T=1000),
        "Unidirectional-Cascade": dict(M=15, T=1000),
        "Quadratic-Coupling": dict(M=15, T=1000),
        "Exponential-Transform": dict(M=15, T=1000),
        "Phase-Lagged-Oscillators": dict(M=15, T=2000),
        # Noise models (null hypotheses) - enabled via --include-noise flag
        "Gaussian-Noise": dict(M=15, T=1000),
        "Cauchy-Noise": dict(M=15, T=1000),
        "t-Noise": dict(M=15, T=1000),
        "Exponential-Noise": dict(M=15, T=1000),
    },
    "dev+++": {
        # Overnight run: M=25-35 (300-595 edges), T=1500-2500 (~30-60 min per model)
        # Prioritize M for SPI-SPI correlation power, keep T sufficient
        # Total runtime: ~8-12 hours for 15 models
        "VAR(1)": dict(M=30, T=1500),
        "OU-network": dict(M=30, T=1500),
        "Kuramoto": dict(M=30, T=2500, transients=2000),  # Phase locking needs longer
        "Stuart-Landau": dict(M=30, T=2500, transients=1500),  # Limit cycle convergence
        "Lorenz-96": dict(M=30, T=2000, transients=1000),  # Chaotic settling
        "Rössler-coupled": dict(M=30, T=2000, transients=2000),  # Longer for coupled attractor
        "OU-heavyTail": dict(M=30, T=1500),
        "GBM-returns": dict(M=30, T=2000),
        "TimeWarp-clones": dict(M=30, T=1500),
        "CML-logistic": dict(M=35, T=1000, transients=500),  # Map transients fast, cheap per step
        "Cauchy-OU": dict(M=30, T=1500),
        "Unidirectional-Cascade": dict(M=30, T=1500),
        "Quadratic-Coupling": dict(M=30, T=1500),
        "Exponential-Transform": dict(M=30, T=1500),
        "Phase-Lagged-Oscillators": dict(M=30, T=2500, transients=1000),
        # Noise models (null hypotheses) - enabled via --include-noise flag
        "Gaussian-Noise": dict(M=30, T=1500),
        "Cauchy-Noise": dict(M=30, T=1500),
        "t-Noise": dict(M=30, T=1500),
        "Exponential-Noise": dict(M=30, T=1500),
    },
    "paper": {
        # HPC/Cluster final: M=50 (1225 edges, optimal power), T=2000-4000 (sufficient)
        # Designed for >1000 MTS (synthetic + real-world) on cluster
        "VAR(1)": dict(M=50, T=2000),
        "OU-network": dict(M=50, T=2000),
        "Kuramoto": dict(M=50, T=4000, transients=2000),
        "Stuart-Landau": dict(M=50, T=4000, transients=1500),
        "Lorenz-96": dict(M=50, T=3000, transients=1000),
        "Rössler-coupled": dict(M=40, T=4000, transients=2000),  # M=40 to keep runtime reasonable
        "OU-heavyTail": dict(M=50, T=2000),
        "GBM-returns": dict(M=50, T=2500),
        "TimeWarp-clones": dict(M=50, T=2000),
        "CML-logistic": dict(M=50, T=2000, transients=500),  # Maps scale well
        "Cauchy-OU": dict(M=50, T=2000),
        "Unidirectional-Cascade": dict(M=50, T=2000),
        "Quadratic-Coupling": dict(M=50, T=2000),
        "Exponential-Transform": dict(M=50, T=2000),
        "Phase-Lagged-Oscillators": dict(M=50, T=3000),
        # Noise models (null hypotheses) - enabled via --include-noise flag
        "Gaussian-Noise": dict(M=50, T=2000),
        "Cauchy-Noise": dict(M=50, T=2000),
        "t-Noise": dict(M=50, T=2000),
        "Exponential-Noise": dict(M=50, T=2000),
    }
}

# --------------------- config + labels ---------------------------------
def _get_config_path(config_hint: str | None = None) -> str:
    if config_hint and os.path.exists(config_hint):
        return config_hint
    here = os.path.dirname(__file__)
    cand = os.path.abspath(os.path.join(here, "configs", "pilot0_config.yaml"))
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError("Could not locate pilot0_config.yaml")

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _load_labels(configfile: str) -> dict[str, list[str]]:
    cfg = _load_yaml(configfile)
    labels = {}
    for _, group in cfg.items():
        for spi, entry in group.items():
            labels[spi] = [s.lower() for s in (entry.get("labels", []) or [])]
    return labels

def _uses_java(configfile: str) -> bool:
    cfg = _load_yaml(configfile)
    for _, group in cfg.items():
        for _, entry in group.items():
            deps = [d.lower() for d in (entry.get("dependencies", []) or [])]
            if "java" in deps:
                return True
    return False

def _is_directed(spi: str, labels_map: dict[str, list[str]]) -> bool:
    return "directed" in labels_map.get(spi, [])

# --------------------- reconstruction helpers --------------------------
# Use the same robust reconstruction you validated.
import re
def reconstruct_mpi(calc, spi_name: str, M: int, symmetrize: bool = True) -> np.ndarray:
    cols = [c for c in calc.table.columns
            if (isinstance(c, tuple) and c[0] == spi_name) or (c == spi_name)]
    if cols:
        def _proc_key(col):
            if isinstance(col, tuple) and isinstance(col[1], str):
                m = re.match(r"proc-(\d+)", col[1])
                if m: return int(m.group(1))
            return 0
        cols_sorted = sorted(cols, key=_proc_key)
        vecs = [np.asarray(calc.table[c]).ravel() for c in cols_sorted]
        if len(vecs) >= M and all(v.size == M for v in vecs[:M]):
            A = np.column_stack(vecs[:M])
            np.fill_diagonal(A, 0.0)
            return 0.5*(A + A.T) if symmetrize else A
    vec = np.asarray(calc.table[spi_name]).astype(float).ravel()
    E_dir = M*(M-1); E_und = M*(M-1)//2
    if vec.size == E_dir:
        A = np.zeros((M, M), float); idx = 0
        for i in range(M):
            for j in range(M):
                if i == j: continue
                A[i, j] = vec[idx]; idx += 1
        np.fill_diagonal(A, 0.0)
        return 0.5*(A + A.T) if symmetrize else A
    if vec.size == E_und:
        A = np.zeros((M, M), float)
        iu = np.triu_indices(M, k=1); A[iu] = vec
        if symmetrize:
            A[(iu[1], iu[0])] = vec
        else:
            np.fill_diagonal(A, 0.0)
        return A
    mat = np.array(calc.table[spi_name])
    if mat.ndim == 2 and mat.shape == (M, M):
        return mat
    raise ValueError(f"Cannot reconstruct MPI for '{spi_name}'")

def _offdiag(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]

# ---- consolidated off-diagonal table (wide) ----
def _make_edge_index(M: int):
    """Return (i_idx, j_idx, mask) for all ordered pairs i!=j."""
    mask = ~np.eye(M, dtype=bool)
    i_idx, j_idx = np.where(mask)
    return i_idx, j_idx, mask

def _save_offdiag_table(matrices: Dict[str, np.ndarray], out_csv: str, out_parquet: str | None = None):
    """
    Build a single wide table: columns = ['i','j'] + sorted(SPIs), rows = all ordered edges (i!=j).
    Values are A_ij from each SPI's adjacency matrix.
    """
    if not matrices:
        return
    # all SPIs must be square (M,M)
    any_mat = next(iter(matrices.values()))
    if any_mat.ndim != 2 or any_mat.shape[0] != any_mat.shape[1]:
        raise ValueError("Matrices must be square (M,M) to build consolidated table.")
    M = any_mat.shape[0]
    i_idx, j_idx, mask = _make_edge_index(M)

    spis = sorted(matrices.keys())
    data = np.empty((i_idx.size, len(spis)), dtype=float)
    for k, spi in enumerate(spis):
        A = matrices[spi]
        if A.shape != (M, M):
            raise ValueError(f"Matrix shape mismatch for {spi}: {A.shape} vs {(M,M)}")
        data[:, k] = A[mask]

    df = pd.DataFrame(data, columns=spis)
    df.insert(0, "j", j_idx.astype(int))
    df.insert(0, "i", i_idx.astype(int))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    if out_parquet:
        try:
            df.to_parquet(out_parquet, index=False)  # requires pyarrow or fastparquet
        except Exception as e:
            print(f"[WARN] offdiag_table parquet save failed ({e}); CSV written.")

# ----------------------- pyspi runner -----------------------------------
@dataclass
class ExperimentResult:
    name: str
    data: np.ndarray  # (T, M)
    calc: Calculator
    spi_names: List[str]
    matrices: Dict[str, np.ndarray]
    offdiag: Dict[str, np.ndarray]

def run_pyspi_on(data: np.ndarray, configfile: str, subset: str = "pilot0", normalise: bool = True) -> Calculator:
    # Calculator expects (M, T)
    calc = Calculator(dataset=data.T, subset=subset, configfile=configfile, normalise=normalise)
    calc.compute()
    return calc

def extract_spis(calc: Calculator) -> List[str]:
    cols = calc.table.columns
    try:
        return list(pd.unique(cols.get_level_values(0)))
    except Exception:
        return list(pd.unique([c[0] if isinstance(c, tuple) else c for c in cols]))

# ----------------------- generator wiring -------------------------------
def build_generators(profile: dict, only: list[str] | None = None, include_noise: bool = False) -> Dict[str, Callable[[], np.ndarray]]:
    """Build generator functions from profile.
    
    Args:
        profile: Dictionary of model_name -> parameters
        only: Optional list of model names to filter to
        include_noise: If True, include noise models (Gaussian-Noise, etc.). Default False.
    
    Returns:
        Dictionary of model_name -> generator function
    """
    table = {
        "VAR(1)": lambda p=profile["VAR(1)"]: gen_var(**p),
        "OU-network": lambda p=profile["OU-network"]: gen_ou_network(**p),
        "Kuramoto": lambda p=profile["Kuramoto"]: gen_kuramoto(**p),
        "Stuart-Landau": lambda p=profile["Stuart-Landau"]: gen_stuart_landau(**p),
        "Lorenz-96": lambda p=profile["Lorenz-96"]: gen_lorenz96(**p),
        "Rössler-coupled": lambda p=profile["Rössler-coupled"]: gen_rossler_coupled(**p),
        "CML-logistic": lambda p=profile["CML-logistic"]: gen_cml_logistic(**p),
        "OU-heavyTail": lambda p=profile["OU-heavyTail"]: gen_ou_heavytail(**p),
        "GBM-returns": lambda p=profile["GBM-returns"]: gen_gbm_returns(**p),
        "TimeWarp-clones": lambda p=profile["TimeWarp-clones"]: gen_timewarp_clones(**p),
        "Cauchy-OU": lambda p=profile["Cauchy-OU"]: gen_cauchy_ou(**p),
        "Unidirectional-Cascade": lambda p=profile["Unidirectional-Cascade"]: gen_unidirectional_cascade(**p),
        "Quadratic-Coupling": lambda p=profile["Quadratic-Coupling"]: gen_quadratic_coupling(**p),
        "Exponential-Transform": lambda p=profile["Exponential-Transform"]: gen_exponential_transform(**p),
        "Phase-Lagged-Oscillators": lambda p=profile["Phase-Lagged-Oscillators"]: gen_phase_lagged_oscillators(**p),
    }
    
    # Add noise models if requested
    if include_noise:
        noise_models = {
            "Gaussian-Noise": lambda p=profile.get("Gaussian-Noise", {}): gen_gaussian_noise(**p),
            "Cauchy-Noise": lambda p=profile.get("Cauchy-Noise", {}): gen_cauchy_noise(**p),
            "t-Noise": lambda p=profile.get("t-Noise", {}): gen_t_noise(**p),
            "Exponential-Noise": lambda p=profile.get("Exponential-Noise", {}): gen_exponential_noise(**p),
        }
        table.update(noise_models)
    
    return {k: v for k, v in table.items() if (not only or k in only)}

# ----------------------- caching helpers --------------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]

def _hash_config(configfile: str) -> str:
    with open(configfile, "rb") as f:
        return _hash_bytes(f.read())

def _make_run_id(model: str, M: int, T: int, subset: str, confighash: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    core = f"{model}_{M}x{T}_{subset}_{confighash}"
    return f"{ts}_{hashlib.sha1(core.encode()).hexdigest()[:8]}"

def _save_calc_table(calc: Calculator, path_csv: str, path_parquet: str | None = None):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    # Always save CSV for portability
    calc.table.to_csv(path_csv, index=True)
    # Try parquet (optional)
    if path_parquet:
        try:
            calc.table.to_parquet(path_parquet, index=True)  # requires pyarrow or fastparquet
        except Exception as e:
            print(f"[WARN] parquet save failed ({e}); CSV saved instead.")

# ----------------------- performance tracking ---------------------------
def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hrs}h {mins}m"

def _get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    return {
        "process_mb": mem_info.rss / 1024 / 1024,
        "available_mb": virtual_mem.available / 1024 / 1024,
        "percent_used": virtual_mem.percent
    }

def _check_memory_warning(required_mb: float) -> str | None:
    """Check if memory might be insufficient and return warning if needed."""
    mem = _get_memory_usage()
    if mem["available_mb"] < required_mb * 1.5:  # 1.5x safety margin
        return f"[WARN] Low memory: {mem['available_mb']:.0f}MB available, ~{required_mb:.0f}MB needed"
    return None

def _estimate_memory_required(M: int, T: int, n_spis: int = 50) -> float:
    """Rough estimate of memory needed in MB."""
    # Timeseries: T*M*8 bytes, Matrices: n_spis*M*M*8 bytes, overhead ~2x
    timeseries_mb = (T * M * 8) / 1024 / 1024
    matrices_mb = (n_spis * M * M * 8) / 1024 / 1024
    calc_overhead_mb = timeseries_mb * 3  # pyspi internal overhead
    return (timeseries_mb + matrices_mb + calc_overhead_mb) * 2  # 2x safety

def _save_performance_metrics(perf_data: List[dict], outdir: str):
    """Save performance metrics to CSV in performance/ folder."""
    perf_dir = os.path.join(outdir, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    
    df = pd.DataFrame(perf_data)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(perf_dir, f"perf_metrics_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Create a simple summary plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Runtime vs M*T
        ax = axes[0, 0]
        df['M*T'] = df['M'] * df['T']
        ax.scatter(df['M*T'], df['runtime_seconds'], alpha=0.6, s=100)
        for i, row in df.iterrows():
            ax.text(row['M*T'], row['runtime_seconds'], row['model'], fontsize=8, alpha=0.7)
        ax.set_xlabel('M × T (data size)')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime vs Data Size')
        ax.grid(True, alpha=0.3)
        
        # Memory usage
        ax = axes[0, 1]
        df['edges'] = df['M'] * (df['M'] - 1)
        ax.scatter(df['edges'], df['peak_memory_mb'], alpha=0.6, s=100, c='orange')
        for i, row in df.iterrows():
            ax.text(row['edges'], row['peak_memory_mb'], row['model'], fontsize=8, alpha=0.7)
        ax.set_xlabel('Number of Edges (M×(M-1))')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage vs Network Size')
        ax.grid(True, alpha=0.3)
        
        # SPIs computed
        ax = axes[1, 0]
        ax.bar(range(len(df)), df['n_spis'], alpha=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Number of SPIs')
        ax.set_title('SPIs Computed per Model')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Runtime distribution
        ax = axes[1, 1]
        sorted_df = df.sort_values('runtime_seconds')
        ax.barh(range(len(sorted_df)), sorted_df['runtime_seconds'], alpha=0.6)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['model'])
        ax.set_xlabel('Runtime (seconds)')
        ax.set_title('Runtime by Model')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = os.path.join(perf_dir, f"perf_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PERF] Metrics saved to {perf_dir}/")
    except Exception as e:
        print(f"[WARN] Performance plot generation failed: {e}")

def _model_already_computed(base_dir: str, model_name: str, M: int, T: int) -> tuple[bool, str | None]:
    """Check if this model (regardless of run_id) has already been computed.
    
    Returns:
        (exists, folder_path) - True and path if found, False and None otherwise
    """
    if not os.path.isdir(base_dir):
        return False, None
    
    # Look for any folder ending with this model name
    model_safe = model_name.replace(' ', '_')
    for folder in os.listdir(base_dir):
        if not folder.endswith(f"_{model_safe}"):
            continue
        
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Check meta.json for M and T match
        meta_file = os.path.join(folder_path, "meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get('M') == M and meta.get('T') == T:
                    # Verify it has required files
                    arrays_dir = os.path.join(folder_path, "arrays")
                    timeseries = os.path.join(arrays_dir, "timeseries.npy")
                    if os.path.exists(timeseries) and os.path.isdir(arrays_dir):
                        mpi_files = [f for f in os.listdir(arrays_dir) if f.startswith("mpi_")]
                        if len(mpi_files) > 0:
                            return True, folder_path
            except Exception:
                continue
    
    return False, None

# ---------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute artifacts with pyspi (no plotting).")
    ap.add_argument("--mode", choices=["dev", "dev+", "dev++", "dev+++", "paper"], default="dev", 
                    help="Profile of M,T per model.")
    ap.add_argument("--config", default=None, help="Path to pilot0_config.yaml (auto-detected if not provided).")
    ap.add_argument("--subset", default="pilot0", help="pyspi subset name.")
    ap.add_argument("--outdir", default="./results", help="Output directory root.")
    ap.add_argument("--cache", default="./cache", help="Cache directory for calc.table parquet/CSV.")
    ap.add_argument("--models", type=str, default="", help="Comma-separated model names to run (leave empty for all).")
    ap.add_argument("--normalise", type=int, default=1, help="1/0: z-score normalise input to Calculator.")
    ap.add_argument("--skip-existing", action="store_true", 
                    help="Skip models that already have computed results.")
    ap.add_argument("--dry-run", action="store_true", 
                    help="Show what would be computed without actually running.")
    ap.add_argument("--exclude-noise", action="store_true",
                    help="Exclude noise models (Gaussian-Noise, Cauchy-Noise, t-Noise, Exponential-Noise) from computation. Default: noise included.")
    args = ap.parse_args()

    profile = PROFILES[args.mode]
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    generators = build_generators(profile, only=only, include_noise=not args.exclude_noise)

    configfile = _get_config_path(args.config)
    labels_map = _load_labels(configfile)
    confighash = _hash_config(configfile)

    if _uses_java(configfile):
        print("[INFO] Java-dependent SPIs detected; running serially for JVM safety.")

    base = os.path.join(args.outdir, args.mode)
    os.makedirs(base, exist_ok=True)
    
    # Dry-run mode: preview what will be computed
    if args.dry_run:
        models_to_compute = []
        models_skipped = []
        
        for model, gen in generators.items():
            params = profile[model]
            M, T = params['M'], params['T']
            
            # Check if should skip
            if args.skip_existing:
                exists, existing_path = _model_already_computed(base, model, M, T)
                if exists:
                    models_skipped.append((model, M, T, os.path.basename(existing_path)))
                    continue
            
            models_to_compute.append((model, M, T))
        
        print(f"\n{'='*70}")
        print(f"DRY RUN - Would compute {len(models_to_compute)} model(s) in '{args.mode}' mode")
        if models_skipped:
            print(f"          ({len(models_skipped)} model(s) will be skipped)")
        print(f"{'='*70}\n")
        print(f"Config: {os.path.basename(configfile)} (hash: {confighash})")
        print(f"Output: {base}\n")
        
        if models_to_compute:
            print("Models to compute:")
            total_edges = 0
            total_datapoints = 0
            for model, M, T in models_to_compute:
                edges = M * (M - 1)
                datapoints = M * T
                est_mem = _estimate_memory_required(M, T)
                
                total_edges += edges
                total_datapoints += datapoints
                
                print(f"  - {model:20s} M={M:2d}, T={T:5d}  =>  {edges:4d} edges, {datapoints:7d} datapoints, ~{est_mem:.0f}MB")
            
            print(f"\n{'='*70}")
            print(f"Total: {total_edges:,} edges, {total_datapoints:,} datapoints")
            print(f"{'='*70}\n")
        
        if models_skipped:
            print("Models to skip (already computed):")
            for model, M, T, folder in models_skipped:
                print(f"  [SKIP] {model:20s} M={M:2d}, T={T:5d}  (found: {folder})")
            print()
        
        return

    # Track performance metrics
    perf_data = []
    total_models = len(generators)
    total_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"COMPUTE MODE: {args.mode} | Models: {total_models} | Subset: {args.subset}")
    print(f"{'='*70}\n")

    for idx, (model, gen) in enumerate(generators.items(), 1):
        model_start = time.time()
        params = profile[model]
        T, M = params['T'], params['M']
        
        # Check if should skip (before generating new run_id)
        if args.skip_existing:
            exists, existing_path = _model_already_computed(base, model, M, T)
            if exists:
                print(f"[{idx}/{total_models}] {model:20s} [SKIP] (exists: {os.path.basename(existing_path)})")
                continue
        
        # Generate run_id for new computation
        run_id = _make_run_id(model.replace(" ", "_"), M, T, args.subset, confighash)
        model_dir = os.path.join(base, f"{run_id}_{model.replace(' ', '_')}")
        
        # Memory check
        est_mem = _estimate_memory_required(M, T)
        mem_warn = _check_memory_warning(est_mem)
        
        # Progress header
        print(f"[{idx}/{total_models}] {model:20s} M={M:2d} T={T:5d} ", end="", flush=True)
        if mem_warn:
            print(f"\n        {mem_warn}")
        
        mem_before = _get_memory_usage()
        
        try:
            # Generate data
            data = gen()  # (T, M)
            
            # Setup directories
            arrays_dir = os.path.join(model_dir, "arrays")
            csv_dir = os.path.join(model_dir, "csv")
            os.makedirs(arrays_dir, exist_ok=True)
            os.makedirs(csv_dir, exist_ok=True)

            # 1) Compute SPIs
            calc = run_pyspi_on(data, configfile=configfile, subset=args.subset, normalise=bool(args.normalise))

            # 2) Save calc.table (cache + per-run)
            cache_csv = os.path.join(args.cache, f"{run_id}_{model}_calc_table.csv")
            cache_parq = os.path.join(args.cache, f"{run_id}_{model}_calc_table.parquet")
            _save_calc_table(calc, cache_csv, cache_parq)
            _save_calc_table(calc, os.path.join(csv_dir, "calc_table.csv"), os.path.join(csv_dir, "calc_table.parquet"))

            # 3) Derive + save matrices/off-diagonals
            spi_names = extract_spis(calc)
            matrices, offdiag = {}, {}
            for s in spi_names:
                try:
                    sym = not _is_directed(s, labels_map)
                    mat = reconstruct_mpi(calc, s, M=M, symmetrize=sym)
                    matrices[s] = mat
                    offdiag[s] = _offdiag(mat)
                except Exception as e:
                    print(f"\n        [WARN] reconstruct failed for {s}: {e}")

            # save arrays (timeseries + all MPIs/offdiags)
            np.save(os.path.join(arrays_dir, "timeseries.npy"), data)
            for spi, mat in matrices.items():
                np.save(os.path.join(arrays_dir, f"mpi_{spi}.npy"), mat)
            for spi, vec in offdiag.items():
                np.save(os.path.join(arrays_dir, f"offdiag_{spi}.npy"), vec)

            # consolidated off-diagonal table (wide): one row per (i,j), columns per SPI
            offdiag_csv = os.path.join(csv_dir, "offdiag_table.csv")
            offdiag_parq = os.path.join(csv_dir, "offdiag_table.parquet")
            _save_offdiag_table(matrices, offdiag_csv, offdiag_parq)

            # 4) Write simple meta manifest
            meta = {
                "model": model,
                "run_id": run_id,
                "M": M, "T": T,
                "subset": args.subset,
                "configfile": os.path.abspath(configfile),
                "confighash": confighash,
                "normalise": bool(args.normalise),
                "n_spis": len(spi_names),
            }
            with open(os.path.join(model_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Track performance
            model_elapsed = time.time() - model_start
            mem_after = _get_memory_usage()
            peak_mem = max(mem_before['process_mb'], mem_after['process_mb'])
            
            perf_data.append({
                'model': model,
                'M': M,
                'T': T,
                'n_spis': len(spi_names),
                'runtime_seconds': model_elapsed,
                'peak_memory_mb': peak_mem,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Calculate ETA
            elapsed_total = time.time() - total_start_time
            avg_time_per_model = elapsed_total / idx
            remaining_models = total_models - idx
            eta_seconds = avg_time_per_model * remaining_models
            
            # Success output with timing
            print(f"[OK] {len(spi_names):2d} SPIs | {_format_time(model_elapsed)} | ETA: {_format_time(eta_seconds)}")
            
        except Exception as e:
            print(f"[ERR] FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Save performance metrics
    if perf_data:
        _save_performance_metrics(perf_data, args.outdir)
        
        total_elapsed = time.time() - total_start_time
        print(f"\n{'='*70}")
        print(f"COMPLETED: {len(perf_data)}/{total_models} models in {_format_time(total_elapsed)}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
