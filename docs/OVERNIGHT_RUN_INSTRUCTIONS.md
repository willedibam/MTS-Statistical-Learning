# Overnight Run Setup - Final Instructions

## Overview

This document provides step-by-step instructions to execute the overnight run with the new `dev+++` profile. All preparation work is complete; follow these steps to run.

---

## What's Been Done

✅ **Generator implementation**: 5 new generators added (Cauchy-OU, Unidirectional-Cascade, Quadratic-Coupling, Exponential-Transform, Phase-Lagged-Oscillators)  
✅ **Profile creation**: `dev+++` profile added with M=25-35, T=1500-2500 (optimal M priority, sufficient T)  
✅ **Paper profile optimization**: Updated to M=50, T=2000-4000 (for future HPC/cluster runs)  
✅ **SPI triplet expectations**: Detailed document with 5 triplets and predictions (SPI_TRIPLET_EXPECTATIONS.md)  
✅ **Overnight script**: PowerShell script with safety guards, error handling, logging (overnight_run.ps1)  
✅ **Power settings doc**: Complete instructions for preventing Windows sleep (POWER_SETTINGS.md)  
✅ **Testing**: All 5 generators tested and working  

---

## Current Generator Count

**Total: 15 generators** (organized by type)

**Baseline/Linear (3)**:
1. VAR(1)
2. OU-network  
3. Unidirectional-Cascade *(NEW)*

**Oscillatory (2)**:
4. Kuramoto
5. Stuart-Landau

**Chaotic (3)**:
6. Lorenz-96
7. Rössler-coupled
8. CML-logistic

**Nonlinear/Heavy-Tail (4)**:
9. OU-heavyTail
10. GBM-returns
11. Cauchy-OU *(NEW)*
12. Quadratic-Coupling *(NEW)*

**Transform/Time (3)**:
13. TimeWarp-clones
14. Exponential-Transform *(NEW)*
15. Phase-Lagged-Oscillators *(NEW)*

---

## Profile Comparison

| Profile | M (range) | T (range) | Edges (range) | Runtime/model | Total runtime | Purpose |
|---------|-----------|-----------|---------------|---------------|---------------|---------|
| **dev** | 4 | 500 | 6 | 2-3 min | ~30-45 min | Fast testing |
| **dev+** | 10 | 1000-1500 | 45 | 10-15 min | 2.5-4 hours | Medium testing |
| **dev++** | 15 | 1000-2000 | 105 | 20-30 min | 5-7.5 hours | Validation (ARCHIVED) |
| **dev+++** | 25-35 | 1500-2500 | 300-595 | 30-60 min | **8-12 hours** | **Overnight (NEW)** |
| **paper** | 40-50 | 2000-4000 | 780-1225 | 60-120 min | 15-30 hours | HPC/cluster final |

**Key insight**: M² dominates computational cost AND statistical power (SPI-SPI correlations scale with edges = M(M-1)). dev+++ prioritizes M=30 (435 edges, excellent power) with T=1500-2500 (sufficient for temporal samples).

---

## Step-by-Step Instructions

### 1. Configure Power Settings (CRITICAL)

**Option A: PowerShell (Recommended)**

Open PowerShell **as Administrator** and run:

```powershell
# Disable sleep on AC power
powercfg /change standby-timeout-ac 0

# Verify (should output 0x00000000)
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC Power Setting Index:"
```

**Option B: Windows Settings GUI**

See detailed instructions in `POWER_SETTINGS.md`.

**Verification**:
```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC Power Setting Index:"
```
Should output: `Current AC Power Setting Index: 0x00000000`

---

### 2. Pre-Flight Checks

```powershell
# Verify Python and pyspi
python --version
python -c "import pyspi; print(pyspi.__version__)"

# Check disk space (need ~10 GB free)
Get-PSDrive C | Select-Object Free

# Check existing results (optional - to see current state)
python -m spimts.helpers.inspect summary --profile dev++
```

---

### 3. Run Overnight Script

```powershell
# Navigate to project directory
cd C:\Users\wille\OneDrive\Desktop\2025USYD\USYD\mts-spi-study

# Run overnight script
.\overnight_run.ps1
```

**What the script does**:
1. Pre-run checks (Python, pyspi, power settings, disk space)
2. Runs all 15 generators sequentially with dev+++ profile
3. Logs all output to timestamped log file (`overnight_run_YYYYMMDD-HHMMSS.log`)
4. Continues on error (if one model fails, others still run)
5. Provides final summary (success/failure count, total runtime)

**Expected output** (first few lines):
```
================================================
OVERNIGHT RUN STARTED: 2025-01-XX 22:00:00
Profile: dev+++
Expected runtime: 8-12 hours
Log file: overnight_run_20250125-220000.log
================================================

Starting pre-run checks...
✓ Python version: Python 3.11.x
✓ pyspi version: 0.x.x
✓ Power settings OK: AC standby disabled (never sleep)
✓ Free space on C: 45.67 GB
Pre-run checks complete.

================================================
Running generator: VAR(1)
Start time: 2025-01-XX 22:01:23
Expected runtime: 30-60 min
================================================
...
```

---

### 4. Monitor First Model (Optional but Recommended)

Stay near computer for first ~30-60 minutes to ensure:
- No errors in first model (VAR(1))
- Power settings working (computer doesn't sleep)
- Resource usage reasonable (RAM, CPU)

Once first model completes successfully:
```
✓ SUCCESS: VAR(1) completed in 45.3 minutes
```

You can safely leave the computer overnight.

---

### 5. Check Results Next Morning

```powershell
# View overnight log
Get-Content overnight_run_YYYYMMDD-HHMMSS.log -Tail 50

# Inspect results
python -m spimts.helpers.inspect summary --profile dev+++

# Check for any failures
python -m spimts.helpers.inspect duplicates --profile dev+++
```

**Expected results**:
- 15 runs in `results/dev+++/`
- Each run has subdirectory: `YYYYMMDD-HHMMSS_<hash>_<generator>/`
- Each subdirectory contains: `meta.json`, `arrays/*.npy`, `csv/*.csv`
- Log file shows `✓ Successful: 15 / 15`

---

## Understanding the Output

### Results Directory Structure
```
results/dev+++/
  20250125-220143_abc123de_VAR(1)/
    meta.json              # Metadata: M, T, transients, runtime, etc.
    arrays/
      *.npy                # Raw SPI matrices (M×M) for each SPI
    csv/
      *.csv                # Processed tables (SPI-SPI correlations, etc.)
    plots/                 # (if visualization enabled)
      *.png
```

### Key Files to Check

**meta.json**: Contains configuration and timing info
```json
{
  "generator": "VAR(1)",
  "profile": "dev+++",
  "M": 30,
  "T": 1500,
  "transients": 0,
  "runtime_seconds": 2743.5,
  "subset": "pilot0",
  "timestamp": "20250125-220143"
}
```

**arrays/*.npy**: NumPy arrays (M×M SPI matrices)
```python
import numpy as np
cov_matrix = np.load('results/dev+++/.../arrays/mpi_cov_EmpiricalCovariance.npy')
print(cov_matrix.shape)  # (30, 30)
```

**csv/*.csv**: Processed correlation tables
- SPI-SPI correlation matrices (for triplets)
- Summary statistics

---

## SPI Triplets (Hypothesis Testing)

The overnight run uses the standard `pilot0` subset (~20 SPIs). For focused hypothesis testing with 5 specific triplets, you'll need to:

1. **Modify config** (optional, future enhancement):
   ```yaml
   # configs/triplets.yaml
   triplets:
     - name: "monotonic_vs_info_vs_linear"
       spis: ["SpearmanR", "mi_kraskov_NN-4", "cov_EmpiricalCovariance"]
     # ... (see SPI_TRIPLET_EXPECTATIONS.md for all 5)
   ```

2. **Run with triplets** (requires code modification to support `--spi-triplets` flag):
   ```powershell
   python -m spimts --mode dev+++ --subset pilot0 --spi-triplets configs/triplets.yaml
   ```

**For this overnight run**: Standard `pilot0` subset is fine. You can extract triplet correlations from the full SPI matrices during post-processing.

---

## Troubleshooting

### Computer sleeps mid-run
- **Symptom**: Script stops, no new output in log
- **Solution**: Check power settings again (`powercfg /query`), re-run from failed model
- **Recovery**: `python -m spimts --mode dev+++ --models "Kuramoto,Stuart-Landau,..." --skip-existing`

### Out of memory
- **Symptom**: Python crashes with `MemoryError`
- **Solution**: Close other applications, reduce M (edit PROFILES in compute.py temporarily)
- **Prevention**: dev+++ designed for ~4-8 GB RAM (reasonable for most systems)

### One model fails
- **Symptom**: Log shows `✗ FAILED: <model> failed`
- **Solution**: Script continues with other models (by design)
- **Investigation**: Check error in log, re-run failed model individually:
  ```powershell
  python -m spimts --mode dev+++ --models "<failed_model>"
  ```

### Script stops immediately
- **Symptom**: Pre-run checks fail
- **Solution**: Install missing dependencies (`pip install pyspi-mts`), verify Python environment

---

## Post-Run Analysis (Next Steps)

After overnight run completes:

1. **Validation**: `python -m spimts.helpers.inspect summary --profile dev+++`
2. **Comparison**: Compare dev+++ vs dev++ (if archived runs exist)
3. **Visualization**: Generate plots (see `spimts/visualize.py`)
4. **Extract triplets**: Post-process arrays to compute SPI-SPI correlations for 5 triplets
5. **Hypothesis testing**: Compare results against expectations in `SPI_TRIPLET_EXPECTATIONS.md`

---

## Restoring Power Settings

After run completes, restore normal sleep timeout:

```powershell
# Re-enable sleep after 30 minutes
powercfg /change standby-timeout-ac 30

# Verify
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
```

---

## Summary Checklist

**Before starting**:
- [ ] Power settings configured (`powercfg /change standby-timeout-ac 0`)
- [ ] Power settings verified (`powercfg /query` → `0x00000000`)
- [ ] AC power connected (laptops)
- [ ] Python and pyspi installed
- [ ] ~10 GB free disk space
- [ ] Overnight script ready (`overnight_run.ps1`)

**During run**:
- [ ] Monitor first model (~30-60 min)
- [ ] Verify no sleep (check after 1-2 hours if staying nearby)

**After run**:
- [ ] Check log file for summary
- [ ] Verify 15/15 models successful
- [ ] Inspect results (`python -m spimts.helpers.inspect summary --profile dev+++`)
- [ ] Restore power settings (`powercfg /change standby-timeout-ac 30`)

---

## Quick Start (TL;DR)

```powershell
# 1. Configure power (as Administrator)
powercfg /change standby-timeout-ac 0

# 2. Verify
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC"

# 3. Run
cd C:\Users\wille\OneDrive\Desktop\2025USYD\USYD\mts-spi-study
.\overnight_run.ps1

# 4. Next morning - check results
Get-Content overnight_run_*.log -Tail 50
python -m spimts.helpers.inspect summary --profile dev+++

# 5. Restore power
powercfg /change standby-timeout-ac 30
```

Expected completion: 8-12 hours (15 models × 30-60 min each)

---

**Good luck with the overnight run!** 🚀

If issues arise, check:
1. Log file (`overnight_run_YYYYMMDD-HHMMSS.log`)
2. Power settings (`POWER_SETTINGS.md`)
3. Error messages in console
4. Results directory (`results/dev+++/`)

All tools have been tested and are ready to use.

---

**Document version**: 1.0  
**Last updated**: January 2025  
**Profile**: dev+++ (M=25-35, T=1500-2500)  
**Generators**: 15 total (3 baseline, 2 oscillatory, 3 chaotic, 4 nonlinear, 3 transform/time)
