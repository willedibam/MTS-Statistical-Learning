# Implementation Summary: October 26, 2025

## Changes Made

### 1. Plotting Enhancements (`spimts/helpers/plotting.py`)

#### Color Scheme System
Added configurable color schemes at top of file with **three options** (comment/uncomment to switch):

**Option 1: MONOCHROMATIC (Default - Active)**
- Scatter: Charcoal gray `#4A5568`, alpha=0.6
- Trendline: Deep teal `#0F766E`, alpha=0.85
- Marginals: Match scatter, alpha=0.5
- KDE: Darker gray `#2D3748`

**Option 2: DUOTONE**
- Scatter: Navy blue `#1E3A8A`
- Trendline: Amber `#F59E0B`
- Marginals/KDE: Navy shades

**Option 3: ROYAL BLUE Custom**
- Scatter: Royal blue `#090088`
- Trendline: Deep red `#880000` (matched intensity) OR Purple `#5D3FD3`
- KDE: Purple `#5D3FD3`

#### plot_spi_space() - Grid Plots
**Fixed**: Rectangular subplot issue
- Changed figsize from `(2.5*n_cols, 2.5*n_rows)` to `(4*n_cols, 4*n_rows)` (square subplots)
- Added `ax.set_aspect('equal', adjustable='datalim')` to force square aspect ratio
- **Removed**: Color-coded trendline (was varying with correlation strength - confusing)
- **Added**: Consistent trendline color from scheme
- **Fixed titles**: Now show Greek symbols (ρ, r, τ) with correlation value

#### plot_spi_space_individual() - Individual Plots with Marginals
**Major upgrade**: Now uses `seaborn.JointGrid` for professional joint plots
- **Marginal distributions**: Histograms (30 bins) matching scatter color
- **KDE overlays**: Smooth density curves on marginals (darker shade)
- **Scatter plot**: Uses configured color scheme, alpha < 1.0
- **Trendline**: Polyfit line (NOT regression) - just shows linear trend
- **Title**: Greek symbols (ρ, r, τ) with 3-decimal correlation coefficient
- **No confidence bands**: As requested (not regression analysis)

**Technical implementation**:
- Used `gaussian_kde` from scipy for smooth marginal density
- Scaled KDE to match histogram heights (dual y-axis trick)
- Proper error handling if KDE fails (some distributions may not work)

---

### 2. Noise Generators (`spimts/generators.py`)

Added **4 new generators** for null hypothesis testing and clustering validation:

#### gen_gaussian_noise(M, T)
- **Purpose**: Gold standard null hypothesis - no dependencies
- **Expected**: All SPI-SPI correlations ≈ 0
- **Use case**: Validate that classifier can distinguish structure from noise

#### gen_cauchy_noise(M, T)
- **Purpose**: Test robustness to infinite variance (heavy tails)
- **Expected**: Covariance/Pearson fail, Spearman/MI robust
- **Use case**: Show that moment-based SPIs break, rank-based survive

#### gen_t_noise(M, T, df=3.0)
- **Purpose**: Tunable tail heaviness (df=3: heavy, df=10: near-Gaussian)
- **Expected**: Intermediate between Gaussian and Cauchy
- **Use case**: Continuous spectrum of tail behavior

#### gen_exponential_noise(M, T, rate=1.0)
- **Purpose**: Asymmetry/skewness sensitivity (positive support)
- **Expected**: MI_gaussian vs MI_kraskov diverge
- **Use case**: Test Gaussian assumption violations

**Integration**: Added to `compute.py` build_generators() with .get() fallback for optional profile entries.

---

### 3. Documentation

#### ML_STRATEGY.md (New - 15 pages)
Comprehensive machine learning roadmap covering:

**Sample Size Analysis**:
- Current N=15 → p/N = 12.7 (severe overfitting risk)
- Fast config N=15 → p/N = 81.7 (hopeless without PCA)
- Target N=500 → p/N < 5 (viable)

**What's Achievable Now**:
- Option A: PCA + simple classifier (proof-of-concept)
- Option B: L1-regularized models (feature selection)
- Option C: Qualitative only (acceptable)

**Phase 2 (N=500)**:
- Expanded dataset: 500 synthetic + 500 real-world
- Cluster requirements: 50 nodes × 15 hours for fast config
- Full ML pipeline: classification, clustering, feature importance

**SPI Subset Expansion**:
- Pilot0 (20 SPIs): ~20 min/model, ~190 features
- Fast (~50 SPIs): ~90 min/model, ~1,225 features
- Full (~250 SPIs): ~1 day/model, ~31k features (NOT recommended)

**Key Insight**: Pilot0 likely captures 70-80% of information, fast captures 90-95%, full adds <5% for 50x compute cost.

**Dimensionality Reduction**:
- PCA: Fast, interpretable, linear
- UMAP: Nonlinear, better for visualization
- Autoencoders: Requires N > 1,000 (defer to Phase 3)

**Timeline**:
- Phase 1 (now): Qualitative analysis, no ML
- Phase 1.5 (optional): Fast config + proof-of-concept ML
- Phase 2 (months 2-6): N=500 dataset, full ML
- Phase 3 (months 6-12+): Deep learning, real-world applications

#### MULTILAG_STATISTICS.md (New - 12 pages)
Theoretical justification for multi-lag statistics:

**Existing SPIs That Handle Lags**:
- TLMI: I(X_t ; Y_{t+lag}) - single lag
- TE: I(Y_t ; X_{t-k} | Y_{t-l}) - multi-lag with embeddings
- xcorr_max/mean: Linear correlation over all lags
- Granger Causality: Multi-lag autoregressive

**Proposed: Joint Multi-Lag Mutual Information (JMLMI)**:
```
JMLMI(X, Y; τ) = I(X_t ; Y_t, Y_{t+1}, ..., Y_{t+τ})
```
- **Difference from TLMI**: Joint over trajectory, not single lag
- **Difference from TE**: Predictive (present→future), not causal (past→present)
- **Difference from xcorr**: Nonlinear joint structure, not pairwise linear

**Scientific Justification**:
- Chaotic systems (Lorenz-96): Long-range multi-step predictability
- Financial (GBM): GARCH effects (volatility clustering)
- Phase systems: Predict trajectory shape, not just single point

**Validation Strategy**:
1. Literature review (check if exists already)
2. Pilot test on 3 generators (Lorenz, VAR, noise)
3. If pilot succeeds → full integration
4. If not → defer (TE/TLMI sufficient)

**Most Suitable Generators**:
- Lorenz-96, Rössler (chaotic, long memory)
- GBM-returns (volatility clustering)
- Quadratic-Coupling (non-monotonic nonlinear)

**Least Suitable**:
- Gaussian-Noise (IID, no memory)
- VAR(1) (Markov, lag-1 only)
- TimeWarp-clones (trivial predictability)

**Does xcorr_* already do this?** 
- **No**: xcorr is pairwise linear, JMLMI is joint nonlinear
- **Example**: Phase-lagged oscillators (xcorr finds best lag, JMLMI captures trajectory shape)

**Recommendation**: **DEFER until after baseline work.** Pilot test on 3 generators if time permits. Full integration only if pilot shows JMLMI adds value beyond TE/TLMI.

---

## Color Scheme Rationale

### Why Three Options?

**Monochromatic** (default):
- **Pro**: Maximum clarity, formal, academic
- **Pro**: Colorblind-friendly (no hue dependencies)
- **Con**: Less visually distinctive

**Duotone**:
- **Pro**: Modern, professional, high contrast
- **Pro**: Clear visual separation (scatter vs trendline)
- **Con**: May be too "designed" for academic papers

**Royal Blue Custom**:
- **Pro**: Distinctive, memorable, bold
- **Pro**: Purple accent adds sophistication
- **Con**: User's suggestion - needs validation with actual plots

**Easy switching**: Just comment/uncomment block at top of `plotting.py`. No function arguments needed (global constants).

---

## Technical Details

### Why Marginals on Individual Plots Only?

**Grid plots (plot_spi_space)**:
- Already crowded (n×n subplots)
- Marginals would make each subplot tiny
- Focus: Overview of all SPI pairs

**Individual plots (plot_spi_space_individual)**:
- Large figures (7×7 inches)
- Space for marginals without clutter
- Focus: Deep dive into specific pair

### Why JointGrid Instead of Manual Marginals?

**Seaborn JointGrid advantages**:
- Automatic alignment of marginals with main plot
- Handles aspect ratio correctly
- Professional styling out-of-box
- Can add KDE easily with twinx/twiny

**Manual implementation would require**:
- GridSpec layout (complex)
- Manual axis alignment (error-prone)
- Custom spacing calculations

### Polyfit Line vs Regression

**User clarified**: Keep polyfit, no confidence bands, NOT regression.

**Implementation**:
- `np.polyfit(x, y, 1)` - least squares fit (same as regression for linear)
- But: **Title emphasizes correlation coefficient**, not slope/intercept
- **Interpretation**: Visual aid for correlation strength, not prediction

**Why this works**:
- For correlation analysis, trendline shows "general trend"
- Coefficient (ρ, r, τ) is the actual statistic
- No statistical inference (no p-values, no CI) → just descriptive

---

## Next Steps

### Immediate (User's Current Task)
1. Wait for baseline visualization to complete (~15-20 min remaining)
2. Run case studies: `.\run_case_studies.ps1`
3. Inspect plots with new color scheme/marginals
4. **Feedback**: Does royal blue scheme look good, or stick with monochrome/duotone?

### Short-Term (This Week)
1. Validate noise generators:
   ```powershell
   python -c "from spimts.generators import gen_cauchy_noise; import numpy as np; X = gen_cauchy_noise(M=10, T=2000); print(f'Mean: {np.mean(X)}, Variance: {np.var(X)}')"
   ```
   Expected: Variance >> 1 (infinite variance, but finite sample variance is large)

2. Test plotting changes on one model:
   ```powershell
   python -m spimts visualize --profile dev+++ --root results --models "Exponential" --include-mpi-heatmaps none
   ```
   Check: Marginals visible, square subplots, Greek symbols in titles

### Medium-Term (Next Month)
1. Run fast config on 15 models (~1 day compute)
2. Compare pilot0 vs fast SPI-space fingerprints
3. Pilot ML test (PCA + Random Forest)
4. Decide on Phase 1.5 vs wait for Phase 2

### Long-Term (Months 2-6)
1. Expand dataset to N=500
2. Get cluster access
3. Full ML pipeline
4. Publication

---

## Files Modified

1. `spimts/helpers/plotting.py` - Color schemes, marginals, square subplots
2. `spimts/generators.py` - 4 noise models added
3. `spimts/compute.py` - Import noise generators, wire to build_generators()
4. `docs/ML_STRATEGY.md` - NEW (comprehensive ML roadmap)
5. `docs/MULTILAG_STATISTICS.md` - NEW (multi-lag theory + justification)

**Total additions**: ~800 lines of code + documentation

---

## Critical Reminders

1. **Do NOT run full config (~250 SPIs)** without cluster + scientific justification
2. **Do NOT do ML on N=15 without PCA** (guaranteed overfitting)
3. **Do defer multi-lag statistic** until baseline analysis completes
4. **Do validate color scheme** with actual plots before committing to paper figures
5. **Do prioritize** baseline viz → case studies → qualitative analysis over ML experiments

