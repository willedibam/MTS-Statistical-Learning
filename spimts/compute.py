# pyspi/compute.py
from __future__ import annotations
import os, argparse, json, time, hashlib
from dataclasses import dataclass
from typing import Dict, Callable, List

import numpy as np
import pandas as pd
import yaml

from pyspi.calculator import Calculator

# your generators module (copy your current generators.py here)
from spimts.generators import (
    gen_var, gen_ou_network, gen_kuramoto, gen_stuart_landau,
    gen_lorenz96, gen_rossler_coupled, gen_cml_logistic,
    gen_ou_heavytail, gen_gbm_returns, gen_timewarp_clones
)

# --------------------- profiles (dev vs paper) -------------------------
PROFILES = {
    "dev": {
        "VAR(1)": dict(M=4, T=1000),
        "OU-network": dict(M=4, T=1000),
        "Kuramoto": dict(M=4, T=2000),
        "Stuart-Landau": dict(M=4, T=2000),
        "Lorenz-96": dict(M=4, T=3000),
        "Rössler-coupled": dict(M=4, T=4000),
        "CML-logistic": dict(M=4, T=1500),
        "OU-heavyTail": dict(M=4, T=1000),
        "GBM-returns": dict(M=4, T=1500),
        "TimeWarp-clones": dict(M=4, T=1500),
    },
    "paper": {
        "VAR(1)": dict(M=20, T=4000),
        "OU-network": dict(M=20, T=4000),
        "Kuramoto": dict(M=20, T=10000),
        "Stuart-Landau": dict(M=20, T=10000),
        "Lorenz-96": dict(M=20, T=20000),
        "Rössler-coupled": dict(M=12, T=20000),
        "CML-logistic": dict(M=20, T=8000),
        "OU-heavyTail": dict(M=20, T=4000),
        "GBM-returns": dict(M=20, T=5000),
        "TimeWarp-clones": dict(M=20, T=4000),
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
def build_generators(profile: dict, only: list[str] | None = None) -> Dict[str, Callable[[], np.ndarray]]:
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
    }
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

# ---------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute artifacts with pyspi (no plotting).")
    ap.add_argument("--mode", choices=["dev", "paper"], default="dev", help="Profile of M,T per model.")
    ap.add_argument("--config", default=None, help="Path to pilot0_config.yaml (auto-detected if not provided).")
    ap.add_argument("--subset", default="pilot0", help="pyspi subset name.")
    ap.add_argument("--outdir", default="./results", help="Output directory root.")
    ap.add_argument("--cache", default="./cache", help="Cache directory for calc.table parquet/CSV.")
    ap.add_argument("--models", type=str, default="", help="Comma-separated model names to run (leave empty for all).")
    ap.add_argument("--normalise", type=int, default=1, help="1/0: z-score normalise input to Calculator.")
    args = ap.parse_args()

    profile = PROFILES[args.mode]
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    generators = build_generators(profile, only=only)

    configfile = _get_config_path(args.config)
    labels_map = _load_labels(configfile)
    confighash = _hash_config(configfile)

    if _uses_java(configfile):
        print("[INFO] Java-dependent SPIs detected; running serially for JVM safety.")

    base = os.path.join(args.outdir, args.mode)
    os.makedirs(base, exist_ok=True)

    for model, gen in generators.items():
        print(f"[RUN] {model}...", end=" ", flush=True)
        data = gen()  # (T, M)
        T, M = data.shape

        run_id = _make_run_id(model.replace(" ", "_"), M, T, args.subset, confighash)
        model_dir = os.path.join(base, f"{run_id}_{model.replace(' ', '_')}")
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
                print(f"\n[WARN] {model}: reconstruct failed for {s}: {e}")

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

        print(f"✓ data=({T},{M}) SPIs={len(spi_names)} -> {model_dir}")

if __name__ == "__main__":
    main()
