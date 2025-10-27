# Quick Reference: New Generators & Case Studies

**Date:** October 25, 2025  
**Status:** 5 new generators implemented, 15 total generators ready for overnight run

---

## NEW GENERATORS (Quick Summary)

| Generator | Purpose | Case Study | SPI-SPI Pattern |
|-----------|---------|------------|-----------------|
| **Cauchy-OU** | Heavy-tail outliers | {ρ,r,τ} robustness | ρ↔τ HIGH, r↔ρ LOW |
| **Unidirectional-Cascade** | Causal flow | {TE,MI,TLMI} directionality | TE asymmetric, MI symmetric |
| **Quadratic-Coupling** | Non-monotonic nonlinearity | {MI,ρ,r,dCorr} | MI high, ρ/r low |
| **Exponential-Transform** | Monotonic nonlinearity | {ρ,dCorr,MI} agreement | All HIGH (monotonic) |
| **Phase-Lagged-Oscillators** | Phase coupling | {XCorr,CoherenceMag,ImagCoh} | ImagCoh isolated |

---

## ALL 15 GENERATORS (Organized by Type)

### Core Baseline (4)
1. **VAR(1)** - Linear Gaussian (everything agrees)
2. **GBM-returns** - Common factor (MI high, TE low)
3. **TimeWarp-clones** - DTW-specific (temporal alignment)
4. **Cauchy-OU** - Heavy tails (Pearson fails) **NEW**

### Oscillatory/Synchronization (3)
5. **Kuramoto** - Phase synchronization (symmetric)
6. **Stuart-Landau** - Limit cycles (amplitude+phase)
7. **Phase-Lagged-Oscillators** - Systematic lags **NEW**

### Chaotic (3)
8. **Lorenz-96** - Continuous chaos (clean, scalable)
9. **Rössler-coupled** - 3D attractor (marginal)
10. **CML-logistic** - Discrete chaos (cheap, marginal)

### Nonlinear Coupling (3)
11. **Quadratic-Coupling** - Non-monotonic (ρ fails) **NEW**
12. **Exponential-Transform** - Monotonic (ρ works) **NEW**
13. **OU-heavyTail** - Moderate tails (gradient test)

### Directionality (2)
14. **Unidirectional-Cascade** - Pure flow (TE>>MI) **NEW**
15. **OU-network** - Spatial diffusion (marginal)

---

## EXCELLENT SPI TRIPLETS (From pilot0_config.yaml)

### Triplet 1: Monotonic vs Information vs Linear
**SPIs**: `{SpearmanR, mi_kraskov_NN-4, cov_EmpiricalCovariance}`  
**Test**: Does nonlinearity break correlations but not MI?  
**Best Generators**:
- Quadratic-Coupling: MI high, SpearmanR/Cov low (non-monotonic)
- Exponential-Transform: All three HIGH (monotonic preserves ranks)
- Cauchy-OU: SpearmanR/MI agree, Cov fails (outliers)

### Triplet 2: Directed vs Undirected Information
**SPIs**: `{te_kraskov_k-1_l-1, mi_kraskov_NN-4, tlmi_kraskov_NN-4}`  
**Test**: Does TE capture asymmetry MI misses?  
**Best Generators**:
- Unidirectional-Cascade: TE(forward)>>TE(reverse), MI symmetric
- GBM-returns: MI high (common factor), TE low (no direct flow)
- VAR(1): TE moderate (weak directionality), MI high (linear)

### Triplet 3: Rank Robustness
**SPIs**: `{xcorr_max, SpearmanR, KendallTau}`  
**Test**: Do all rank-based SPIs agree under heavy tails?  
**Best Generators**:
- Cauchy-OU: XCorr↔ρ↔τ all HIGH (Pearson fails)
- Phase-Lagged-Oscillators: XCorr finds lag, ρ/τ moderate
- OU-heavyTail: Moderate tails (tests gradient)

### Triplet 4: Temporal Alignment
**SPIs**: `{dtw_null, xcorr_max, pdist_euclidean}`  
**Test**: Does DTW outperform correlation for warped signals?  
**Best Generators**:
- TimeWarp-clones: DTW high, XCorr/Euclidean lower (designed for this)
- Phase-Lagged-Oscillators: DTW vs XCorr for phase shifts
- Kuramoto: All three moderate (phase synchronization)

### Triplet 5: Nonlinear Dependency Gradient
**SPIs**: `{dcorr_biased-False, mi_kraskov_NN-4, SpearmanR}`  
**Test**: Gradient from linear → monotonic → nonlinear detection?  
**Best Generators**:
- Quadratic-Coupling: MI >> dCorr >> ρ (non-monotonic)
- Exponential-Transform: MI ≈ dCorr ≈ ρ (monotonic)
- VAR(1): All three HIGH (linear)

---

## REDUNDANT MODELS (Honest Assessment)

### Marginal But Kept:
- **OU-network**: Overlaps VAR(1), but tests spatial structure
- **Rössler-coupled**: Overlaps Lorenz-96, but different attractor
- **CML-logistic**: Overlaps Lorenz-96, but discrete vs continuous

### Unique & Irreplaceable:
- **TimeWarp-clones**: ONLY DTW-specific generator
- **GBM-returns**: ONLY common-factor structure
- **OU-heavyTail**: Tail-weight gradient (Gaussian→Student-t→Cauchy)

**Verdict**: Keep all 15 (marginal ones have negligible computational cost)

---

## OPTIMAL SETTINGS (From FUTURE.md)

### Development Profiles
- **dev**: M=4, T=500 (fast testing, 3-5 min per model)
- **dev+**: M=10, T=1000-1500 (medium, 10-15 min per model)
- **dev++**: M=15, T=1000-2000 (validation, 20-30 min per model)

### Paper Profile (Cluster-Scale)
- **M=50**: 1,225 edges (optimal statistical power)
- **T=2000-3000**: Captures dynamics without waste
- **Rule**: T ≥ 2M (temporal samples per edge)
- **Runtime**: 15 models × 20 SPIs × M=50 ≈ 4-8 hours overnight

### Per-Model Transients (If Needed)
- **Lorenz-96**: transients=1000 (chaotic settling)
- **Rössler-coupled**: transients=2000 (longer settling)
- **Stuart-Landau**: transients=1500 (limit cycle)
- **Kuramoto**: transients=2000 (phase locking)
- **CML-logistic**: transients=500 (map transients fast)
- **Others**: transients=500 (default)

---

## CLEANUP COMMANDS (Quick Reference)

### Inspection (Python CLI)
```powershell
# Summary of all runs
python -m spimts.helpers.inspect summary --profile dev++

# Find duplicate runs
python -m spimts.helpers.inspect duplicates --profile dev++

# Check visualization status
python -m spimts.helpers.inspect viz-status --profile dev++

# Recent runs (last 7 days)
python -m spimts.helpers.inspect recent --profile dev++ --days 7
```

### Deletion (PowerShell - Always use -WhatIf first!)
```powershell
# Preview deletion of VAR duplicates (keep latest)
Get-ChildItem results\dev++ -Directory | Where-Object { $_.Name -match 'VAR' } | Select-Object -Skip 1 | Remove-Item -Recurse -WhatIf

# Actually delete (remove -WhatIf after preview)
Get-ChildItem results\dev++ -Directory | Where-Object { $_.Name -match 'VAR' } | Select-Object -Skip 1 | Remove-Item -Recurse -Force

# Delete entire dev profile (CAREFUL!)
Remove-Item results\dev -Recurse -WhatIf

# Delete runs older than 7 days
Get-ChildItem results\dev++ -Directory | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Recurse -WhatIf
```

---

## NEXT STEPS (Ready for Next Prompt)

### Overnight Run Preparation:
1. ⏳ Add logging to `compute.py` (same approach as `visualize.py`)
2. ⏳ Configure per-model transient times (if necessary)
3. ⏳ Create overnight run script for paper profile (15 models)
4. ⏳ Test one new generator end-to-end (compute + visualize)

### Deferred to Future:
- Significance testing (Option 2 from FUTURE.md)
- Model classification (Option 4 from FUTURE.md)
- Dimensionality reduction (Option 1 from FUTURE.md)

---

## TESTING CHECKLIST ✅

- [x] All 5 new generators execute without errors
- [x] Generators produce correct shape (T, M)
- [x] Generators produce normalized output (mean≈0, std≈1)
- [x] Dry-run recognizes new generators
- [x] Inspection CLI works (summary, duplicates, viz-status)
- [x] PowerShell deletion dry-run works (-WhatIf)
- [x] GENERATORS.md updated comprehensively
- [x] Honest assessment of redundant models documented
- [ ] End-to-end test (compute + visualize new generator) - **READY FOR NEXT PROMPT**
- [ ] Overnight run script created - **READY FOR NEXT PROMPT**

---

## KEY TAKEAWAYS

1. **5 new generators fill critical gaps** in {ρ,r}, {TE,MI}, {MI,ρ,r} case studies
2. **15 total generators** provide complete coverage of SPI contrasts
3. **3 marginal generators** kept for completeness (negligible cost)
4. **5 excellent SPI triplets** identified from available SPIs
5. **Cleanup tools validated** (Python inspect + PowerShell delete)
6. **Ready for overnight run** with optimal M=50, T=2000 settings

**Bottom line**: Project now has **scientifically justified generators** that **purposefully exploit SPI differences** to infer data character. No more "interesting but useless" models.
