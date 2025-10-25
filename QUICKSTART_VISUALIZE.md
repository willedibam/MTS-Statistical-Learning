# Quick Start: Visualization Commands

## Summary of Changes
✅ **Fixed aspect ratio bug** in `plot_spi_space()` - now auto-scales axes properly  
✅ **Updated styling**: Gray dots (α=0.7), RdYlBu_r colormap for correlation-colored regression lines  
✅ **Removed `--spi-limit`**, added `--include-mpi-heatmaps {all|core|none|<int>}`  
✅ **Added `--spi-subset`** for targeted SPI combination analysis  
✅ **Error validation** with helpful suggestions for typos  

---

## Available SPIs in Your Data (19 total)
From your cached dev++ runs:
```
cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25
cov_EmpiricalCovariance
dcorr
dtw
dtw_constraint-itakura
dtw_constraint-sakoe-chiba
gc_gaussian_k-1_kt-1_l-1_lt-1
icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25
kendalltau
mi_gaussian
mi_kraskov_NN-4
pdist_cosine
pdist_euclidean
pec_orth
psi_multitaper_mean_fs-1_fmin-0_fmax-0-25
spearmanr
te_kraskov_NN-4_k-1_kt-1_l-1_lt-1
tlmi_kraskov_NN-4
xcorr_max_sig-True
xcorr_mean_sig-True
```
---
## By dward
- `compute.py` will compute whatever SPIs in `configfile=<set>_config.yaml`.
- `visualize.py` will visualize and plot data corresponding to some
    - `--profile`: `[dev, dev+, dev++, paper]`
    - `--models`: "Model1,Model2,...,ModelN"
    - `--all-runs`: all models
    - `--include-mpi-heatmaps`: `[all,core,<int>,0]


---

## Command 1: Full Visualization (Recommended First Run)
**Visualize all dev++ models with all 19 SPIs, core MPI heatmaps**

```powershell
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps core 2>&1 | Tee-Object .\logs\visualize_dev++_full_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

**What this does:**
- Processes all 16 model runs in `results/dev++/`
- Generates spi_space plots (grid + individual) for all 19 SPIs
- Creates 3 versions: Spearman, Kendall, Pearson
- Saves ~6 "core" MPI heatmaps per model (SpearmanR, MI, TE, DTW, Covariance, etc.)
- **Expected runtime:** ~20-30 minutes
- **Output:** ~500-700 total plots

---

## Command 2: Targeted SPI Subsets
**Test specific SPI combinations aligned with CONTEXT.md research questions**

### Example Case Study 1: {MI, ρ, Covariance}
*"Do information-theoretic and correlation-based measures capture similar or different dynamics?"*

```powershell
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps none --spi-subset "mi_kraskov_NN-4,spearmanr,cov_EmpiricalCovariance" 2>&1 | Tee-Object .\logs\visualize_dev++_MI-rho-cov_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

**Output location:** `results/dev++/<run>/plots/subsets/cov_EmpiricalCovariance+mi_kraskov_NN-4+spearmanr/`

---

### Example Case Study 2: {TE, MI}
*"How do directed (TE) vs undirected (MI) information measures compare?"*

```powershell
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps none --spi-subset "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1,mi_kraskov_NN-4" 2>&1 | Tee-Object .\logs\visualize_dev++_TE-MI_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

---

### Example Case Study 3: Multiple Subsets at Once
*Run 3 different subsets in one command*

```powershell
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps none `
    --spi-subset "mi_kraskov_NN-4,spearmanr,cov_EmpiricalCovariance" `
    --spi-subset "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1,mi_kraskov_NN-4" `
    --spi-subset "spearmanr,cov_EmpiricalCovariance" `
    2>&1 | Tee-Object .\logs\visualize_dev++_3subsets_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

---

## Command 3: Single Model Testing
**Test on one model first before running full batch**

```powershell
# Test full visualization on Kuramoto only
python -m spimts visualize --profile dev++ --models "Kuramoto" --include-mpi-heatmaps core 2>&1 | Tee-Object .\logs\visualize_Kuramoto_test_$(Get-Date -Format 'yyyyMMdd-HHmmss').log

# Test subset on Kuramoto only
python -m spimts visualize --profile dev++ --models "Kuramoto" --include-mpi-heatmaps none --spi-subset "mi_kraskov_NN-4,spearmanr,cov_EmpiricalCovariance" 2>&1 | Tee-Object .\logs\visualize_Kuramoto_subset_test_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

---

## Command 4: Specific Models Only
**Visualize only selected models**

```powershell
# Just Kuramoto and Lorenz-96
python -m spimts visualize --profile dev++ --models "Kuramoto,Lorenz-96" --all-runs --include-mpi-heatmaps core 2>&1 | Tee-Object .\logs\visualize_selected_models_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

---

## Understanding --include-mpi-heatmaps

| Value | Behavior | Use Case |
|-------|----------|----------|
| `none` or `0` | No MPI heatmaps | Fastest; focus on SPI-SPI plots |
| `core` | ~6 core SPIs (substring match) | **Default**; quick visual reference |
| `all` | All 19 SPIs | Complete MPI visualization (slower) |
| `10` | First 10 SPIs alphabetically | Custom limit |

**"Core" SPIs** (substring matching):
- `spearmanr` → SpearmanR correlation
- `cov_` → Covariance methods
- `mi_` → Mutual Information variants
- `te_` → Transfer Entropy variants  
- `dtw` → Dynamic Time Warping variants
- `pairwisedistance` or `pdist_` → Distance measures

---

## Error Handling Example

If you make a typo:
```powershell
python -m spimts visualize --profile dev++ --models "Kuramoto" --spi-subset "mi_krasov,spearmanr"
```

You'll get:
```
[ERR] The following SPIs were not found in cached data:
  ✗ 'mi_krasov'  (did you mean: mi_kraskov_NN-4, mi_gaussian?)

Available SPIs in this run:
  - cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25
  - cov_EmpiricalCovariance
  ...

[INFO] Use exact SPI names from arrays/mpi_*.npy files (without 'mpi_' prefix)
```

---

## Output Structure

After running Command 1 (full visualization):
```
results/dev++/20251025-081150_25e48d9a_Kuramoto/
├── arrays/           # Unchanged (cached SPI matrices)
├── csv/              # Unchanged  
├── meta.json         # Unchanged
└── plots/
    ├── mts/
    │   └── mts_heatmap.png
    ├── mpis/         # ~6 core MPI heatmaps
    │   ├── mpi_spearmanr.png
    │   ├── mpi_mi_kraskov_NN-4.png
    │   └── ...
    ├── spi_space/    # Grid plots (all 19 SPIs)
    │   ├── spi_space_spearman.png
    │   ├── spi_space_kendall.png
    │   └── spi_space_pearson.png
    ├── spi_space_individual/  # Individual scatter plots
    │   ├── spearman/  # ~171 plots (19 choose 2)
    │   ├── kendall/
    │   └── pearson/
    └── fingerprint/  # Fingerprint barcodes + dendrograms
        ├── fingerprint_spearman.png
        ├── dendrogram_spearman.png
        └── ...
```

After running Command 2 (with subsets):
```
results/dev++/20251025-081150_25e48d9a_Kuramoto/
└── plots/
    └── subsets/      # NEW: SPI subset visualizations
        ├── cov_EmpiricalCovariance+mi_kraskov_NN-4+spearmanr/
        │   ├── spi_space/
        │   │   ├── spi_space_spearman.png
        │   │   ├── spi_space_kendall.png
        │   │   └── spi_space_pearson.png
        │   ├── spi_space_individual/
        │   │   ├── spearman/  # 3 plots (3 choose 2)
        │   │   ├── kendall/
        │   │   └── pearson/
        │   └── fingerprint/
        │       ├── fingerprint_spearman.png
        │       └── ...
        └── mi_kraskov_NN-4+te_kraskov_NN-4_k-1_kt-1_l-1_lt-1/
            └── ...
```

---

## Recommended Workflow

### Step 1: Test on Single Model
```powershell
python -m spimts visualize --profile dev++ --models "Kuramoto" --include-mpi-heatmaps core
```
- Check output in `results/dev++/<run_id>_Kuramoto/plots/`
- Verify new gray dots, RdYlBu_r colormap, auto-scaled axes

### Step 2: Run Full Visualization
```powershell
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps core 2>&1 | Tee-Object .\logs\visualize_dev++_full_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```
- Go make coffee ☕ (~20-30 min)
- Review plots across all 10 models

### Step 3: Targeted Subset Analysis
```powershell
# Based on CONTEXT.md research questions
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps none `
    --spi-subset "mi_kraskov_NN-4,spearmanr,cov_EmpiricalCovariance" `
    --spi-subset "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1,mi_kraskov_NN-4" `
    2>&1 | Tee-Object .\logs\visualize_dev++_subsets_$(Get-Date -Format 'yyyyMMdd-HHmmss').log
```

---

## Style Reference

### plot_spi_space (Grid)
- **Dots:** Gray, α=0.7, size=12
- **Regression line:** Color-coded by correlation strength
  - **Colormap:** RdYlBu_r (Red[+1] → Yellow[0] → Blue[-1])
  - Avoids white/clear at zero correlation
- **Aspect ratio:** Auto (matplotlib default based on data range)

### plot_spi_space_individual (Single plots)
- **Dots:** Gray, α=0.7, size=15
- **Regression line:** Soft teal (#4ECDC4), fixed color
- **Aspect ratio:** Equal with adjustable datalim (square plots)

---

## Next Steps

1. **Run Command 1** to generate full visualization
2. **Review plots** in `results/dev++/*/plots/spi_space/` and `spi_space_individual/`
3. **Identify interesting SPI combinations** from fingerprint barcodes/dendrograms
4. **Run targeted subsets** with Command 2 for deeper analysis
5. **Compare across models** to see which dynamics show similar SPI-SPI patterns
