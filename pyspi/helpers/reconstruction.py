# pyspi/helpers/reconstruction.py
import re
import numpy as np

def _get_M(calc) -> int:
    ds = calc.dataset
    if hasattr(ds, "n_processes"):
        val = ds.n_processes
        return int(val() if callable(val) else val)
    # fallback: DataFrame rows == number of processes/nodes
    return int(calc.table.shape[0])

def reconstruct_mpi(calc, spi_name: str, M: int, symmetrize: bool = True) -> np.ndarray:
    # Try per-proc column layout first
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

    # Vectorized fallback
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

    # If pyspi already gave a square
    mat = np.array(calc.table[spi_name])
    if mat.ndim == 2 and mat.shape == (M, M):
        return mat
    raise ValueError(f"Cannot reconstruct MPI for '{spi_name}'")
