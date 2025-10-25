# MTS-Statistical-Learning (pyspi study)

Compute once with `pyspi.compute`, visualize many times with `pyspi.visualize`.

## Quick Start

```bash
# 1. Compute SPIs for models (one-time, can take hours for paper mode)
python -m spimts compute --mode dev --models "VAR(1),Kuramoto"

# 2. Visualize results (fast, iterate as needed)
python -m spimts visualize --profile dev --models "VAR(1),Kuramoto"
```

---

## Table of Contents
- [Compute Pipeline](#compute-pipeline)
- [Visualize Pipeline](#visualize-pipeline)
- [Profiles & Performance](#profiles--performance)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Compute Pipeline

**Computes Statistical Pairwise Interaction (SPI) matrices for dynamical systems.**

### Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--mode` | `dev`, `dev+`, `dev++`, `paper` | `dev` | Profile determining M (channels) and T (timesteps) |
| `--config` | path | auto-detect | Path to `pilot0_config.yaml` |
| `--subset` | string | `pilot0` | pyspi subset name |
| `--outdir` | path | `./results` | Output directory root |
| `--cache` | path | `./cache` | Cache directory for calc tables |
| `--models` | comma-separated | all | Filter specific models (e.g., `"VAR(1),Kuramoto"`) |
| `--normalise` | `0` or `1` | `1` | Z-score normalize input to Calculator |
| `--skip-existing` | flag | off | Skip models with existing results (resume capability) |
| `--dry-run` | flag | off | Preview what would be computed without running |

### Examples

```bash
# Preview what paper mode would compute
python -m spimts compute --mode paper --dry-run

# Run dev mode for specific models
python -m spimts compute --mode dev --models "VAR(1),Kuramoto"

# Resume interrupted paper run (skip already computed)
python -m spimts compute --mode paper --skip-existing

# Full paper mode (WARNING: ~20-25 hours runtime)
python -m spimts compute --mode paper
```

### Output Structure

```
results/<mode>/<run_id>_<Model>/
├── meta.json                    # Run metadata
├── arrays/
│   ├── timeseries.npy          # (T, M) time series
│   ├── mpi_<SPI>.npy           # (M, M) adjacency matrices
│   └── offdiag_<SPI>.npy       # Upper-triangle vectors
└── csv/
    ├── calc_table.csv          # Raw pyspi output
    ├── calc_table.parquet
    ├── offdiag_table.csv       # Consolidated edge table
    └── offdiag_table.parquet

results/performance/                # Performance metrics & plots
└── perf_metrics_<timestamp>.csv
└── perf_plots_<timestamp>.png
```

---

## Visualize Pipeline

**Reads pre-computed artifacts and generates plots (NO re-computation).**

### Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--profile` | `dev`, `dev+`, `dev++`, `paper` | `dev` | Which results directory to read |
| `--models` | comma-separated | all | Filter models (substring match) |
| `--run-id` | string | latest | Specific run ID (exact or prefix match) |
| `--all-runs` | flag | off | Process all matching runs (not just latest) |
| `--root` | path | `./results` | Base results directory |
| `--spi-limit` | int | `12` | Max MPI heatmaps per model |

### Examples

```bash
# Latest run per model in dev
python -m spimts visualize

# Specific models, latest runs
python -m spimts visualize --profile dev++ --models "VAR(1),Kuramoto" #this gets the **latest** VAR(1),Kuramoto models in the results/dev++/... directory

# Specific past run by ID - YYYYMMDD-HHMMSS or <hash>
python -m spimts visualize --run-id 20251024-233652
python -m spimts visualize --run-id 20251024-233652

# All runs for Kuramoto and VAR(1) in dev+ profile
python -m spimts visualize --profile dev+ --models "Kuramoto,VAR(1)" --all-runs

# Multiple models, latest of each in results/<profile>/...
python -m spimts visualize --profile dev+ --models "Kuramoto,VAR(1)"

# High-quality plots with many SPIs
python -m spimts visualize --spi-limit 100
```

### Generated Plots

```
results/dev++/20251024-170017_8a25574d_VAR(1)/
└── plots/
    ├── mpis/
    │   ├── mpi_SpearmanR.png          ← overwritten
    │   ├── mpi_Covariance.png         ← overwritten
    │   └── ...
    ├── spi_space/
    │   ├── spi_space_spearman.png     ← overwritten
    │   ├── spi_space_kendall.png      ← NEW (first time)
    │   └── spi_space_pearson.png      ← NEW (first time)
    ├── spi_space_individual/
    │   ├── spearman/
    │   │   ├── SpearmanR_vs_Covariance.png  ← overwritten
    │   │   └── ...
    │   ├── kendall/                    ← NEW subfolder
    │   │   └── ...
    │   └── pearson/                    ← NEW subfolder
    │       └── ...
    ├── fingerprint/
    │   ├── fingerprint_spearman.png   ← overwritten
    │   ├── fingerprint_kendall.png    ← NEW
    │   ├── fingerprint_pearson.png    ← NEW
    │   ├── dendrogram_spearman.png    ← overwritten
    │   ├── dendrogram_kendall.png     ← NEW
    │   └── dendrogram_pearson.png     ← NEW
    └── mts/
        └── mts_heatmap.png            ← overwritten
```

---

## Profiles & Performance

### Profile Specifications

| Profile | Description | Typical Use |
|---------|-------------|-------------|
| **dev** | M=4, T≈1000-2000 | Quick iteration, debugging |
| **dev+** | M=10, T≈2000-5000 | Medium-scale testing |
| **dev++** | M=15, T≈3000-12000 | Pre-paper validation |
| **paper** | M=12-30, T≈4000-20000 | Publication-quality results |

### Detailed Profile Parameters

| Model | dev (M,T) | dev+ (M,T) | dev++ (M,T) | paper (M,T) |
|-------|-----------|------------|-------------|-------------|
| VAR(1) | 4, 1000 | 10, 2000 | 15, 3000 | 20, 4000 |
| OU-network | 4, 1000 | 10, 2000 | 15, 3000 | 20, 4000 |
| Kuramoto | 4, 2000 | 10, 5000 | 15, 7500 | 20, 10000 |
| Stuart-Landau | 4, 2000 | 10, 5000 | 15, 7500 | 20, 10000 |
| Lorenz-96 | 4, 3000 | 10, 5000 | 15, 12000 | 20, 20000 |
| Rössler-coupled | 4, 4000 | 10, 4000 | 10, 12000 | 12, 20000 |
| CML-logistic | 4, 1500 | 20, 1000 | 25, 5000 | 30, 8000 |
| OU-heavyTail | 4, 1000 | 10, 2000 | 15, 3000 | 20, 4000 |
| GBM-returns | 4, 1500 | 10, 3000 | 15, 4000 | 20, 5000 |
| TimeWarp-clones | 4, 1500 | 10, 3000 | 15, 3000 | 20, 4000 |

### Runtime Estimates

**Based on dev mode ≈ 60s per model:**

| Profile | Total Models | Est. Runtime | Memory Peak |
|---------|--------------|--------------|-------------|
| dev | 10 | ~10 min | <1 GB |
| dev+ | 10 | ~1-2 hrs | ~2 GB |
| dev++ | 10 | ~5-8 hrs | ~4 GB |
| paper | 10 | **~20-25 hrs** | ~8 GB |

**Per-model scaling:**
- **M² scaling:** Pairwise SPIs scale quadratically with channels
- **T scaling:** Time-dependent SPIs scale linearly with timesteps
- **Java SPIs:** ~2-3x slower due to JVM overhead

### Memory Requirements

| Profile | Minimum RAM | Recommended RAM |
|---------|-------------|-----------------|
| dev | 2 GB | 4 GB |
| dev+ | 4 GB | 8 GB |
| dev++ | 8 GB | 16 GB |
| paper | 16 GB | 32 GB |

**Memory warnings:** The compute pipeline automatically checks available memory and warns when insufficient.

---

## Usage Examples

### Development Workflow

```bash
# 1. Test with small dataset
python -m spimts compute --mode dev --models "VAR(1)" --dry-run

# 2. Run if estimates look good
python -m spimts compute --mode dev --models "VAR(1)"

# 3. Quick visualization
python -m spimts visualize --models "VAR(1)"

# 4. Iterate on plots without recomputing
python -m spimts visualize --models "VAR(1)" --spi-limit 50
```

### Production Pipeline

```bash
# 1. Preview full paper run
python -m spimts compute --mode paper --dry-run

# 2. Start long computation (consider screen/tmux on remote server)
nohup python -m spimts compute --mode paper > compute.log 2>&1 &

# 3. Monitor progress
tail -f compute.log

# 4. If interrupted, resume with skip-existing
python -m spimts compute --mode paper --skip-existing

# 5. Visualize all results
python -m spimts visualize --profile paper --all-runs
```

### Selective Model Runs

```bash
# Run only stochastic models
python -m spimts compute --mode dev+ --models "VAR(1),OU-network,OU-heavyTail,GBM-returns"

# Run only oscillator models
python -m spimts compute --mode dev+ --models "Kuramoto,Stuart-Landau"

# Run only chaotic systems
python -m spimts compute --mode dev+ --models "Lorenz-96,Rössler-coupled,CML-logistic"
```

### Comparing Profiles

```bash
# Compute same models at different scales
python -m spimts compute --mode dev --models "VAR(1),Kuramoto"
python -m spimts compute --mode dev+ --models "VAR(1),Kuramoto"
python -m spimts compute --mode dev++ --models "VAR(1),Kuramoto"

# Visualize to compare
python -m spimts visualize --profile dev --models "VAR(1),Kuramoto"
python -m spimts visualize --profile dev+ --models "VAR(1),Kuramoto"
python -m spimts visualize --profile dev++ --models "VAR(1),Kuramoto"
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```bash
# Check memory before running
python -m spimts compute --mode paper --dry-run  # See estimates

# If insufficient, use smaller profile
python -m spimts compute --mode dev++ --models "Lorenz-96"

# Or run models individually
python -m spimts compute --mode paper --models "VAR(1)" --skip-existing
python -m spimts compute --mode paper --models "Kuramoto" --skip-existing
```

**2. Java-based SPIs Fail**

```bash
# Ensure Java is installed and in PATH
java -version

# Check start_java_for_pyspi.py is in root directory
python start_java_for_pyspi.py
```

**3. Interrupted Long Runs**

```bash
# Resume without recomputing finished models
python -m spimts compute --mode paper --skip-existing
```

**4. Visualization Shows No Results**

```bash
# Check if computation completed
ls results/<profile>/

# Verify run_id format
python -m spimts visualize --profile <mode> --models "<Model>" --all-runs
```

**5. Performance Plots Missing**

Performance plots are automatically generated in `results/performance/` after compute runs complete. If missing:
- Check terminal output for matplotlib warnings
- Ensure matplotlib backend is configured
- Check write permissions on results directory

### Performance Optimization

**For faster computation:**
- Use `--models` to run subsets in parallel on different machines
- Use `dev++` profile to validate before full paper run
- Monitor with `--dry-run` first

**For lower memory:**
- Run models individually with `--models`
- Use smaller profiles (`dev` or `dev+`)
- Close other applications during paper mode

**For resume capability:**
- Always use `--skip-existing` when re-running
- Check `results/<mode>/` for existing folders
- Each run has unique timestamp-based ID

---

## Data Flow

```
┌─────────────┐
│ Generator   │ → (T, M) timeseries
└─────────────┘
       ↓
┌─────────────┐
│ Calculator  │ → (M, T) transposed → calc.table
└─────────────┘
       ↓
┌─────────────┐
│reconstruct_ │ → (M×M) MPI matrices
│    mpi()    │
└─────────────┘
       ↓
┌─────────────┐
│  _offdiag() │ → Upper-triangle vectors + consolidated tables
└─────────────┘
       ↓
┌─────────────┐
│  Save to    │ → arrays/*.npy + csv/*.csv + meta.json
│    disk     │
└─────────────┘
```

---

## Advanced Features

### Custom Configurations

```bash
# Use custom pyspi config
python -m spimts compute --mode dev --config /path/to/custom_config.yaml

# Change output location
python -m spimts compute --mode dev --outdir /data/experiments/run01

# Disable normalization
python -m spimts compute --mode dev --normalise 0
```

### Performance Analysis

After each run, check `results/performance/` for:
- `perf_metrics_<timestamp>.csv` - Raw timing/memory data
- `perf_plots_<timestamp>.png` - 4-panel visualization:
  - Runtime vs data size (M×T)
  - Memory vs network size (edges)
  - SPIs computed per model
  - Runtime distribution

### Batch Processing

```bash
# Process all models sequentially
for model in "VAR(1)" "Kuramoto" "Stuart-Landau"; do
    python -m spimts compute --mode paper --models "$model"
    python -m spimts visualize --profile paper --models "$model"
done
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{mts_spi_study,
  title={MTS Statistical Pairwise Interaction Study},
  author={Your Name},
  year={2025},
  url={https://github.com/willedibam/MTS-Statistical-Learning}
}
```

## Random, iterative list:
- Generator → (T,M) timeseries → Calculator(M,T transposed) → 
calc.table → reconstruct_mpi() → matrices (M×M) → 
offdiag vectors + consolidated tables → save to disk

## Visualize previously computed runs

Visualize **never recomputes**. It reads artifacts from `results/<profile>/<run_id>_<Model>/...`.

### Flags

* `--profile dev|paper|dev+` – which results directory to read (default: `dev`).
* `--models "VAR(1),Kuramoto"` – optional filter (comma-separated).
* `--run-id <id>` – visualize a **specific past run** (exact or prefix match).
* `--all-runs` – process **all** matching runs instead of only the latest per model.
* `--root` – base results dir (default: `./results`).
* `--spi-limit N` – cap how many MPI PNGs per model.

### Examples

```bash
# latest per model under results/dev
python -m spimts visualize

# only these models (latest runs)
python -m spimts visualize --models "VAR(1),Kuramoto"

# specific past run id (e.g., run_X_2025-10-24T12-03-11) under dev
python -m spimts visualize --run-id run_X

# dev+ profile (separate directory), all runs for Kuramoto
python -m spimts visualize --profile dev+ --models Kuramoto --all-runs
```

### Dev iteration workflow

1. Compute a small run:

   ```bash
   python -m spimts compute --mode dev --models "VAR(1),Kuramoto"
   ```
2. Find its `run_id` (folder prefix in `results/dev`).
3. Iterate visuals quickly:

   ```bash
   python -m spimts visualize --profile dev --models "VAR(1),Kuramoto" --run-id <that_run_id>
   ```

---

If you want, I can also add a `--run-dir` flag to point directly to a single folder (bypassing discovery). But with `--run-id` + `--profile` you’re usually covered.
