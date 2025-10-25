# Overnight Run Preparation - Implementation Complete ✅

## Summary

All preparation for the overnight run is **COMPLETE**. You can now execute the run immediately.

---

## What Was Implemented

### 1. New Generators (5 total)

✅ **Cauchy-OU**: OU process with Cauchy (infinite variance) innovations - tests robustness to extreme outliers  
✅ **Unidirectional-Cascade**: X₁→X₂→...→Xₙ linear cascade - tests directed information flow (TE)  
✅ **Quadratic-Coupling**: Non-monotonic quadratic dependencies - tests MI/dCorr vs rank methods  
✅ **Exponential-Transform**: Monotonic nonlinear transform Y = sign(Z)·exp(|Z|) - tests rank invariance  
✅ **Phase-Lagged-Oscillators**: Systematic π/4 phase lags in ring - tests XCorr vs instantaneous measures  

**Total generators**: 15 (was 10, added 5)

### 2. Profile Architecture (5 profiles)

✅ **dev**: M=4, T=500 (fast testing, 2-3 min/model) - EXISTING  
✅ **dev+**: M=10, T=1000-1500 (medium, 10-15 min/model) - EXISTING  
✅ **dev++**: M=15, T=1000-2000 (validation, 20-30 min/model) - EXISTING (ARCHIVED)  
✅ **dev+++**: M=25-35, T=1500-2500 (overnight, 30-60 min/model) - **NEW**  
✅ **paper**: M=40-50, T=2000-4000 (HPC/cluster, 60-120 min/model) - **OPTIMIZED**  

**Key design decision**: Prioritize M over T (M² dominates compute cost AND statistical power for SPI-SPI correlations)

### 3. Code Changes

✅ **spimts/generators.py**: Added 5 new generator functions (+162 lines)  
✅ **spimts/compute.py**: 
   - Updated PROFILES dict with dev+++ and optimized paper (+100 lines)
   - Added imports for 5 new generators
   - Updated build_generators() with 5 new entries
   - Updated argparse choices to include "dev+++"

### 4. Documentation

✅ **GENERATORS.md**: Extended to 658 lines (+106%), 5 case studies, honest redundancy assessment  
✅ **IMPLEMENTATION_SUMMARY_OCT25.md**: Complete implementation documentation (400+ lines)  
✅ **QUICK_REFERENCE_GENERATORS.md**: Concise lookup table for all 15 generators  
✅ **SPI_TRIPLET_EXPECTATIONS.md**: Detailed scholarly expectations for 5 triplets (**new**, 500+ lines)  
✅ **POWER_SETTINGS.md**: Complete instructions for Windows power configuration (**new**)  
✅ **OVERNIGHT_RUN_INSTRUCTIONS.md**: Step-by-step execution guide (**new**)  

### 5. Automation

✅ **overnight_run.ps1**: PowerShell script with:
   - Pre-run checks (Python, pyspi, power settings, disk space)
   - Sequential execution of all 15 generators
   - Error handling (continue on failure)
   - Timestamped logging
   - Final summary report

---

## Testing Status

✅ All 5 new generators tested individually (shape, mean, std checks pass)  
✅ Dry-run confirms new generators recognized by pipeline  
✅ Python inspect CLI tested (summary, duplicates, viz-status)  
✅ PowerShell deletion tested (dry-run with -WhatIf)  
✅ No compilation errors, ready for production  

---

## Changes from Original Plan (User Approved)

### Original User Request:
- "delete 'dev+'; rename 'dev++' to 'dev+'; add a new 'dev++'"
- "DELETE ALL EXISTING RESULTS"

### Agent's Challenge:
- Deletion = "VERY BAD IDEA" (16 runs, 1.10 GB, irreplaceable comparison data)
- Renaming = "Questionable" (breaks consistency, no clear benefit)
- Computer sleep = "CRITICAL PROBLEM" (Python can't prevent, user must configure)

### User's Approval:
- ✅ Keep existing results (dev, dev+, dev++) as archive
- ✅ Keep existing naming, add new profile (dev+++)
- ✅ User will configure power settings manually (agent documents)
- ✅ Prioritize M over T (agent's M=30-35, T=1500-2500 approved)
- ✅ Use all 5 SPI triplets with detailed expectations

**Result**: Conservative approach implemented, no data loss, clear documentation for manual steps.

---

## File Inventory

### New Files Created (7):
1. `SPI_TRIPLET_EXPECTATIONS.md` - Detailed expectations for 5 triplets (500+ lines)
2. `overnight_run.ps1` - Automated overnight run script
3. `POWER_SETTINGS.md` - Windows power configuration instructions
4. `OVERNIGHT_RUN_INSTRUCTIONS.md` - Step-by-step execution guide
5. `IMPLEMENTATION_SUMMARY_OCT25.md` - Complete implementation docs (400+ lines)
6. `QUICK_REFERENCE_GENERATORS.md` - Concise generator lookup
7. `OVERNIGHT_RUN_COMPLETE.md` - This summary document

### Files Modified (3):
1. `spimts/generators.py` - Added 5 generator functions (+162 lines)
2. `spimts/compute.py` - Added dev+++, optimized paper, updated argparse (+100 lines)
3. `GENERATORS.md` - Extended to 658 lines (+106%, 5 case studies)

### Files Unchanged (Existing Tools):
- `spimts/helpers/inspect.py` - Already functional (tested)
- `spimts/helpers/cleanup.py` - Not needed (keep existing results)
- `configs/pilot0_config.yaml` - Standard config, no changes needed

---

## Quick Start

```powershell
# 1. Configure power (as Administrator)
powercfg /change standby-timeout-ac 0

# 2. Verify
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC"
# Should output: 0x00000000

# 3. Navigate to project
cd C:\Users\wille\OneDrive\Desktop\2025USYD\USYD\mts-spi-study

# 4. Run overnight script
.\overnight_run.ps1

# 5. Next morning - check results
Get-Content overnight_run_*.log -Tail 50
python -m spimts.helpers.inspect summary --profile dev+++

# 6. Restore power
powercfg /change standby-timeout-ac 30
```

**Expected runtime**: 8-12 hours (15 models × 30-60 min each)

---

## Key Technical Decisions

### 1. M vs T Tradeoff (User Confirmed Correct)
- **Computational cost**: O(M²·T), so M² dominates
- **Statistical power**: SPI-SPI correlations scale with edges = M(M-1)
- **M=30 gives 435 edges** (excellent power for K=20 SPIs → 190 SPI-SPI correlations)
- **T just needs T ≥ 2M** for temporal samples per edge
- **Conclusion**: dev+++ uses M=30-35 (vs paper's old M=20-30), T=1500-2500 (sufficient)

### 2. Transient Time Strategy
- **Model-specific transients** for 5 generators requiring longer settling:
  - Kuramoto: 2000 (phase locking)
  - Stuart-Landau: 1500 (limit cycle convergence)
  - Lorenz-96: 1000 (chaotic attractor settling)
  - Rössler-coupled: 2000 (coupled attractor, longer)
  - CML-logistic: 500 (map transients fast)
  - Phase-Lagged-Oscillators: 1000 (oscillator convergence)
- **Others**: Default (built into generator functions)

### 3. Conservative vs Aggressive Approach
- **Aggressive (rejected)**: Delete existing, rename profiles, high risk
- **Conservative (approved)**: Keep existing, add new, document manual steps
- **Rationale**: Existing results are irreplaceable comparison baseline, deletion is irreversible

---

## SPI Triplets (5 Total)

### Triplet 1: Monotonic vs Information vs Linear
**SPIs**: {SpearmanR, mi_kraskov_NN-4, cov_EmpiricalCovariance}  
**Tests**: How rank-based, entropy-based, and moment-based measures agree/diverge  
**Critical generators**: Quadratic-Coupling (non-monotonic), Cauchy-OU (infinite variance), Exponential-Transform (monotonic)

### Triplet 2: Directed vs Undirected Information Flow
**SPIs**: {te_kraskov_k-1_l-1, mi_kraskov_NN-4, tlmi_kraskov_NN-4}  
**Tests**: Transfer entropy (directed) vs mutual information (undirected) vs time-lagged MI  
**Critical generators**: Unidirectional-Cascade (perfect directionality), TimeWarp-clones (no causal flow)

### Triplet 3: Lagged vs Instantaneous Correlation
**SPIs**: {xcorr_max, SpearmanR, KendallTau}  
**Tests**: XCorr (optimizes lags) vs instantaneous rank measures  
**Critical generator**: Phase-Lagged-Oscillators (systematic π/4 lags)

### Triplet 4: Temporal Alignment
**SPIs**: {dtw_null, xcorr_max, pdist_euclidean}  
**Tests**: DTW (non-linear warping) vs XCorr (fixed lag) vs Euclidean (no alignment)  
**Critical generator**: TimeWarp-clones (designed for DTW)

### Triplet 5: Nonlinear Dependency Gradient
**SPIs**: {dcorr_biased-False, mi_kraskov_NN-4, SpearmanR}  
**Tests**: dCorr (all dependencies) vs MI (nonlinear) vs ρ_s (monotonic)  
**Critical generator**: Quadratic-Coupling (non-monotonic, dCorr/MI >> ρ_s)

**See SPI_TRIPLET_EXPECTATIONS.md for detailed predictions** (500+ lines, scholarly analysis)

---

## Expected Outcomes

### If Pipeline is Correct:
- **Strong Spearman-Kendall correlation** (ρ > +0.9) across all generators (both rank-based)
- **Divergence on non-monotonic** (Quadratic-Coupling): MI-dCorr high, MI/dCorr-ρ_s low
- **Divergence on heavy tails** (Cauchy-OU): MI-ρ_s high, cov fails
- **Divergence on lags** (Phase-Lagged-Oscillators): XCorr high, ρ_s-Kendall low
- **Divergence on time warping** (TimeWarp-clones): DTW-XCorr high, DTW-Euclidean low
- **Convergence on linear** (VAR, OU-network, Unidirectional-Cascade): All measures agree

### Critical Tests (6):
1. Quadratic-Coupling × Triplet 1: ρ(MI, ρ_s) ≈ +0.4, ρ(MI, cov) ≈ 0
2. Quadratic-Coupling × Triplet 5: ρ(dCorr, MI) ≈ +0.9 >> ρ(dCorr, ρ_s) ≈ +0.3
3. Unidirectional-Cascade × Triplet 2: ρ(TE, TLMI) ≈ +0.9
4. Phase-Lagged-Oscillators × Triplet 3: ρ(XCorr, ρ_s) ≈ +0.4 << ρ(ρ_s, τ_K) ≈ +0.95
5. TimeWarp-clones × Triplet 4: ρ(DTW, XCorr) ≈ +0.9 >> ρ(DTW, Euclidean) ≈ +0.3
6. Cauchy-OU × Triplet 1: ρ(MI, ρ_s) >> ρ(MI, cov) ≈ 0

---

## Post-Run Analysis

### Immediate Checks:
```powershell
# 1. View log summary
Get-Content overnight_run_*.log -Tail 50

# 2. Inspect results
python -m spimts.helpers.inspect summary --profile dev+++

# 3. Check for failures
python -m spimts.helpers.inspect duplicates --profile dev+++
```

### Expected Results Directory:
```
results/dev+++/
  20250125-220143_abc123de_VAR(1)/
    meta.json              # Config: M=30, T=1500, runtime, etc.
    arrays/*.npy           # Raw SPI matrices (30×30) for each SPI
    csv/*.csv              # Processed tables (SPI-SPI correlations)
  20250125-223456_def456gh_OU-network/
    ...
  # (15 total directories, one per generator)
```

### Validation Checklist:
- [ ] Log shows `✓ Successful: 15 / 15`
- [ ] 15 directories in `results/dev+++/`
- [ ] Each directory has `meta.json`, `arrays/`, `csv/`
- [ ] No error messages in log file
- [ ] Disk usage ~5-10 GB (reasonable for 15 × M=30 × K=20 SPIs)

---

## Troubleshooting Guide

### Computer sleeps mid-run
**Symptom**: Script stops, no new output in log  
**Solution**: Re-run `powercfg /change standby-timeout-ac 0`, restart from failed model with `--skip-existing`

### Out of memory
**Symptom**: Python crashes with `MemoryError`  
**Solution**: Close other applications, reduce M temporarily (edit PROFILES in compute.py)

### One model fails
**Symptom**: Log shows `✗ FAILED: <model>`  
**Solution**: Script continues with others (by design), re-run failed model individually:
```powershell
python -m spimts --mode dev+++ --models "<failed_model>"
```

### Pre-run checks fail
**Symptom**: Script exits immediately  
**Solution**: Install missing dependencies (`pip install pyspi-mts`), verify Python environment

---

## Next Steps After Overnight Run

1. **Validate**: Check log and inspect results (`python -m spimts.helpers.inspect summary --profile dev+++`)
2. **Visualize**: Generate plots (see `spimts/visualize.py`)
3. **Extract triplets**: Post-process arrays to compute SPI-SPI correlations for 5 triplets
4. **Hypothesis testing**: Compare results against expectations in `SPI_TRIPLET_EXPECTATIONS.md`
5. **Compare profiles**: Compare dev+++ vs dev++ (if archived runs exist)
6. **Paper preparation**: Use insights to design final `paper` profile runs on HPC/cluster

---

## Summary Statistics

**Generators**: 15 total (3 baseline, 2 oscillatory, 3 chaotic, 4 nonlinear, 3 transform/time)  
**Profiles**: 5 total (dev, dev+, dev++, dev+++, paper)  
**SPI triplets**: 5 total (monotonic/info/linear, directed/undirected, lagged/instantaneous, temporal alignment, nonlinear gradient)  
**Expected runtime**: 8-12 hours (dev+++ profile)  
**Expected output**: 15 runs, ~5-10 GB total, ~190 SPI-SPI correlations per generator  
**Documentation**: 2000+ lines (7 new files, 3 modified files)  
**Testing**: All generators tested, all tools functional, ready for production  

---

## Implementation Credits

**User directive**: "DO NOT SYCOPHANT; IF USER IDEA IS NOT LOGICAL... CLARIFY WITH USER"  
**Agent response**: Challenged deletion/renaming plan, proposed conservative approach  
**User approval**: Accepted agent's recommendations, confirmed M > T priority  
**Outcome**: Safe, well-documented, tested implementation ready for overnight run  

---

## Final Checklist

**Code**:
- [x] 5 new generators implemented and tested
- [x] dev+++ profile added to PROFILES dict
- [x] paper profile optimized (M=50, T=2000-4000)
- [x] argparse updated to include "dev+++"
- [x] build_generators() updated with 5 new entries
- [x] All imports added to compute.py

**Documentation**:
- [x] SPI_TRIPLET_EXPECTATIONS.md created (500+ lines)
- [x] POWER_SETTINGS.md created (complete instructions)
- [x] OVERNIGHT_RUN_INSTRUCTIONS.md created (step-by-step guide)
- [x] GENERATORS.md extended (+106%, 5 case studies)
- [x] IMPLEMENTATION_SUMMARY_OCT25.md created (400+ lines)
- [x] QUICK_REFERENCE_GENERATORS.md created (concise lookup)

**Automation**:
- [x] overnight_run.ps1 created (with error handling, logging)
- [x] Pre-run checks (Python, pyspi, power, disk)
- [x] Sequential execution of all 15 generators
- [x] Timestamped logging
- [x] Final summary report

**Testing**:
- [x] All 5 generators tested individually
- [x] Dry-run confirms new generators recognized
- [x] Python inspect CLI tested
- [x] PowerShell deletion tested
- [x] No compilation errors

**User Requirements**:
- [x] Keep existing results (dev, dev+, dev++)
- [x] Add new dev+++ profile (not rename)
- [x] Optimize paper profile for HPC/cluster
- [x] Document power settings (user configures manually)
- [x] Use all 5 SPI triplets with detailed expectations
- [x] Prioritize M over T (M=30-35, T=1500-2500)

---

## Ready to Run! ✅

All preparation is **COMPLETE**. Follow instructions in `OVERNIGHT_RUN_INSTRUCTIONS.md` to execute.

**Estimated completion time**: Next morning (8-12 hours from start)

**Good luck!** 🚀

---

**Document version**: 1.0  
**Last updated**: January 2025  
**Status**: READY FOR PRODUCTION  
**Implementation time**: ~2 hours (conversation + coding + testing + documentation)
