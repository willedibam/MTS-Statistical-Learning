# SPI Triplet Expectations for Overnight Run

## Overview

This document provides detailed, scholarly expectations for five carefully selected SPI triplets. Each triplet tests a specific hypothesis about how different statistics for pairwise interaction (SPI) families respond to violation of their core assumptions. The expectations are interpretable, falsifiable, and grounded in the statistical theory underlying each measure.

**Purpose**: These triplets will be computed across all 15 synthetic MTS generators using `--spi-subsets pilot0 --spi-triplets <file>` to generate focused correlation matrices for hypothesis testing.

---

## Triplet 1: Monotonic vs Information-Theoretic vs Linear Dependencies

**SPIs**: `{SpearmanR, mi_kraskov_NN-4, cov_EmpiricalCovariance}`

**Hypothesis**: Different dependency structures (monotonic nonlinear, general nonlinear, strictly linear) should create distinct SPI-SPI correlation patterns, revealing where rank-based, information-theoretic, and moment-based measures agree or diverge.

### Scientific Rationale

- **Spearman's ρ (SpearmanR)**: Rank correlation; detects monotonic relationships but invariant to monotonic transformations
- **Mutual Information (mi_kraskov_NN-4)**: Entropy-based; detects arbitrary nonlinear dependencies including non-monotonic
- **Covariance (cov_EmpiricalCovariance)**: Linear moment-based; only detects linear associations, blind to nonlinear

**Key theoretical distinctions**:
1. MI captures all dependencies (linear + nonlinear); ρ_s captures monotonic; cov captures linear only
2. For Gaussian data: MI ∝ -log(1-ρ²), ρ_s ≈ ρ (Pearson), cov ∝ ρ
3. For non-monotonic nonlinear: MI high, ρ_s low, cov ≈ 0
4. For monotonic nonlinear: MI high, ρ_s high, cov variable (depends on transformation)

### Expected SPI-SPI Correlations by Generator

| Generator | ρ(SpearmanR, MI) | ρ(SpearmanR, Cov) | ρ(MI, Cov) | Interpretation |
|-----------|------------------|-------------------|------------|----------------|
| **VAR(1)** | **High (+0.7 to +0.9)** | **High (+0.8 to +0.95)** | **High (+0.7 to +0.9)** | Linear Gaussian: all three measures agree strongly. MI ≈ -log(1-ρ²) maps monotonically to ρ, cov ∝ ρ. |
| **OU-network** | **High (+0.7 to +0.9)** | **High (+0.8 to +0.95)** | **High (+0.7 to +0.9)** | Linear coupling: same as VAR(1), perfect agreement expected. |
| **Kuramoto** | **Moderate (+0.4 to +0.7)** | **Low-Mod (+0.2 to +0.5)** | **Low (+0.1 to +0.4)** | Nonlinear oscillatory coupling: MI detects phase coherence, ρ_s detects some monotonic trends in phases, cov struggles with nonlinearity. Expect MI-ρ_s > ρ_s-cov. |
| **Stuart-Landau** | **Moderate (+0.4 to +0.7)** | **Low-Mod (+0.2 to +0.5)** | **Low (+0.1 to +0.4)** | Similar to Kuramoto: limit cycle dynamics create nonlinear dependencies. MI > ρ_s > cov in sensitivity. |
| **Lorenz-96** | **Moderate (+0.5 to +0.7)** | **Low (+0.1 to +0.3)** | **Low (0.0 to +0.3)** | Chaotic nonlinear: MI detects information flow, ρ_s partial, cov very weak. Expect divergence: ρ(MI, ρ_s) > ρ(MI, cov) ≈ ρ(ρ_s, cov). |
| **Rössler-coupled** | **Moderate (+0.5 to +0.7)** | **Low (+0.1 to +0.3)** | **Low (0.0 to +0.3)** | Chaotic coupled: same pattern as Lorenz-96. |
| **OU-heavyTail** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Linear + heavy tails: ρ_s robust to outliers, MI robust, cov degraded by outliers. Expect ρ(MI, ρ_s) > ρ(MI, cov) ≈ ρ(ρ_s, cov). |
| **GBM-returns** | **Low-Mod (+0.3 to +0.6)** | **Very Low (0.0 to +0.2)** | **Low (+0.1 to +0.3)** | Common factor + nonlinearity: MI detects shared information, ρ_s some monotonic trends, cov weak. |
| **TimeWarp-clones** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Near-perfect clones: all measures should agree strongly (all detect perfect dependency). |
| **CML-logistic** | **Moderate (+0.4 to +0.7)** | **Low (+0.1 to +0.3)** | **Low (0.0 to +0.3)** | Chaotic map: similar to Lorenz-96, MI > ρ_s > cov. |
| **Cauchy-OU** | **High (+0.7 to +0.9)** | **Very Low (0.0 to +0.2)** | **Low (+0.1 to +0.3)** | Linear + infinite variance: ρ_s robust (rank-based), MI robust, cov FAILS (moment doesn't exist). **Critical test**: expect ρ(MI, ρ_s) >> ρ(MI, cov) ≈ ρ(ρ_s, cov). |
| **Unidirectional-Cascade** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Linear cascade: all measures agree, strong correlations across all pairs. |
| **Quadratic-Coupling** | **Low-Mod (+0.3 to +0.6)** | **Very Low (-0.1 to +0.2)** | **Very Low (-0.1 to +0.2)** | **Non-monotonic quadratic**: MI high (detects X² dependencies), ρ_s low (non-monotonic breaks rank correlation), cov very low (quadratic ⟂ linear). **Critical test**: expect ρ(MI, ρ_s) > 0 but weak, ρ(MI, cov) ≈ 0, ρ(ρ_s, cov) ≈ 0. |
| **Exponential-Transform** | **Very High (+0.85 to +0.95)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | Monotonic nonlinear: ρ_s perfect (invariant to monotonic transform), MI high (detects dependency), cov degraded (exponential ≠ linear). **Critical test**: ρ(MI, ρ_s) ≈ 1 >> ρ(MI, cov) ≈ ρ(ρ_s, cov). |
| **Phase-Lagged-Oscillators** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | Linear oscillators + phase lag: MI and ρ_s detect lagged dependency, cov weaker (lagged correlation < instantaneous). |

### Key Predictions

1. **Linear generators** (VAR, OU-network, Unidirectional-Cascade, TimeWarp-clones): All three measures strongly correlated (ρ > +0.7)
2. **Monotonic nonlinear** (Exponential-Transform): MI-ρ_s near-perfect (ρ ≈ +0.9), but MI-cov and ρ_s-cov degraded (ρ ≈ +0.5)
3. **Non-monotonic nonlinear** (Quadratic-Coupling): MI-ρ_s weak (ρ ≈ +0.4), MI-cov and ρ_s-cov very weak (ρ ≈ 0)
4. **Heavy tails** (OU-heavyTail, Cauchy-OU): Cov degraded/fails, MI and ρ_s robust → ρ(MI, ρ_s) >> ρ(MI, cov)
5. **Chaotic/nonlinear** (Kuramoto, Stuart-Landau, Lorenz-96, Rössler, CML): MI > ρ_s > cov in sensitivity → divergent correlations

---

## Triplet 2: Directed vs Undirected Information Flow

**SPIs**: `{te_kraskov_k-1_l-1, mi_kraskov_NN-4, tlmi_kraskov_NN-4}`

**Hypothesis**: Transfer entropy (TE) measures directed information flow (X→Y), while mutual information (MI) measures undirected dependency, and time-lagged MI (TLMI) measures lagged dependency. Generators with asymmetric coupling should show TE ≠ TE^T (directional asymmetry), while MI and TLMI remain symmetric.

### Scientific Rationale

- **Transfer Entropy (TE)**: TE(X→Y) = I(Y_t ; X_{t-k} | Y_{t-l}) measures information flow from X's past to Y's present, conditioning on Y's own past
- **Mutual Information (MI)**: MI(X,Y) = H(X) + H(Y) - H(X,Y) measures total shared information (symmetric)
- **Time-Lagged MI (TLMI)**: TLMI(X,Y,τ) = MI(X_t, Y_{t+τ}) measures lagged dependency (still symmetric in undirected sense)

**Key distinctions**:
1. TE is **asymmetric**: TE(X→Y) ≠ TE(Y→X) for directed coupling
2. MI is **symmetric**: MI(X,Y) = MI(Y,X) always
3. TLMI is **lagged symmetric**: TLMI(X,Y,τ) = TLMI(Y,X,-τ) but can detect directionality via lag optimization
4. For unidirectional X→Y: TE(X→Y) > TE(Y→X), MI(X,Y) high, TLMI(X,Y,+τ) > TLMI(X,Y,-τ)

### Expected SPI-SPI Correlations by Generator

| Generator | ρ(TE, MI) | ρ(TE, TLMI) | ρ(MI, TLMI) | Directionality Pattern |
|-----------|-----------|-------------|-------------|------------------------|
| **VAR(1)** | **Moderate (+0.4 to +0.7)** | **High (+0.6 to +0.8)** | **High (+0.7 to +0.9)** | VAR(1) has bidirectional coupling: TE and TLMI both detect lagged dependencies, MI detects total. Expect ρ(MI, TLMI) > ρ(TE, MI) because MI captures instantaneous + lagged, TLMI captures lagged only. |
| **OU-network** | **Moderate (+0.4 to +0.7)** | **High (+0.6 to +0.8)** | **High (+0.7 to +0.9)** | Network structure: TE detects directed edges, MI total connectivity, TLMI lagged. Similar to VAR(1). |
| **Kuramoto** | **Low-Mod (+0.3 to +0.6)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.5 to +0.7)** | Symmetric coupling: TE(X→Y) ≈ TE(Y→X) for connected pairs, but nonlinearity weakens correlations. MI detects phase coherence, TLMI lagged coherence. |
| **Stuart-Landau** | **Low-Mod (+0.3 to +0.6)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.5 to +0.7)** | Symmetric coupling + limit cycle: similar to Kuramoto. |
| **Lorenz-96** | **Moderate (+0.5 to +0.7)** | **High (+0.6 to +0.8)** | **Moderate (+0.5 to +0.7)** | Chaotic unidirectional coupling (X_i → X_{i+1}): TE should detect directionality strongly, TLMI detects lags, MI total. Expect ρ(TE, TLMI) > ρ(TE, MI) because TE and TLMI both exploit temporal structure. |
| **Rössler-coupled** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Bidirectional coupling: TE, MI, TLMI all detect dependencies, moderate agreement. |
| **OU-heavyTail** | **Moderate (+0.4 to +0.7)** | **High (+0.6 to +0.8)** | **High (+0.7 to +0.9)** | Linear coupling + heavy tails: MI and TLMI robust, TE less robust to outliers → expect ρ(MI, TLMI) > ρ(TE, TLMI) ≈ ρ(TE, MI). |
| **GBM-returns** | **Low (+0.2 to +0.5)** | **Low-Mod (+0.3 to +0.6)** | **Moderate (+0.5 to +0.7)** | Common factor + nonlinearity: MI detects shared info, TLMI lagged, TE weak (no true directed coupling). Expect ρ(MI, TLMI) > ρ(TE, MI) ≈ ρ(TE, TLMI). |
| **TimeWarp-clones** | **Low (+0.2 to +0.5)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Perfect clones with time warping: MI perfect, TLMI perfect after lag alignment, TE low (no causal flow, just copying). **Critical test**: ρ(MI, TLMI) ≈ 1 >> ρ(TE, MI) ≈ ρ(TE, TLMI). |
| **CML-logistic** | **Moderate (+0.5 to +0.7)** | **High (+0.6 to +0.8)** | **Moderate (+0.5 to +0.7)** | Chaotic map coupling: TE detects map directionality, TLMI lagged, MI total. |
| **Cauchy-OU** | **Moderate (+0.4 to +0.7)** | **High (+0.6 to +0.8)** | **High (+0.7 to +0.9)** | Linear + infinite variance: MI and TLMI robust (entropy-based), TE less robust. Similar to OU-heavyTail. |
| **Unidirectional-Cascade** | **High (+0.7 to +0.9)** | **Very High (+0.8 to +0.95)** | **High (+0.7 to +0.9)** | **Perfect unidirectional coupling (X₁→X₂→...→Xₙ)**: TE(Xᵢ→Xᵢ₊₁) >> TE(Xᵢ₊₁→Xᵢ), TLMI detects lagged cascade, MI total. **Critical test**: All three highly correlated, but TE shows asymmetry matrix (upper triangular pattern). Expect ρ(TE, TLMI) ≈ +0.9. |
| **Quadratic-Coupling** | **Low (+0.2 to +0.5)** | **Low-Mod (+0.3 to +0.6)** | **Moderate (+0.5 to +0.7)** | Non-monotonic + symmetric coupling: TE, TLMI, MI all detect dependencies, but nonlinearity weakens TE. |
| **Exponential-Transform** | **Moderate (+0.5 to +0.7)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Monotonic transform of linear: MI and TLMI high (invariant to monotonic transform via rank-based estimation), TE moderate. |
| **Phase-Lagged-Oscillators** | **Low-Mod (+0.3 to +0.6)** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Unidirectional phase lag cascade**: TLMI perfect (detects systematic π/4 lags), MI moderate (instantaneous correlation weak), TE moderate (phase lag ≠ causal transfer). **Critical test**: ρ(TE, TLMI) strong but ρ(TE, MI) weaker because TE struggles with pure phase shifts. Expect ρ(TE, TLMI) > ρ(MI, TLMI) > ρ(TE, MI). |

### Key Predictions

1. **Unidirectional generators** (Unidirectional-Cascade, Lorenz-96): TE-TLMI very high (both detect directionality), ρ(TE, TLMI) > +0.7
2. **Symmetric generators** (Kuramoto, Stuart-Landau, Quadratic-Coupling): TE(X→Y) ≈ TE(Y→X), moderate correlations across all three
3. **TimeWarp-clones**: MI-TLMI near-perfect (ρ ≈ 1), but TE-MI very weak (ρ ≈ +0.3) because cloning ≠ causal transfer
4. **Phase-Lagged-Oscillators**: TLMI strongest (detects systematic lags), TE moderate, MI weaker → ρ(TE, TLMI) > ρ(MI, TLMI) > ρ(TE, MI)
5. **Heavy-tail generators** (Cauchy-OU, OU-heavyTail): MI and TLMI robust (entropy-based), TE less robust → ρ(MI, TLMI) > ρ(TE, TLMI)

---

## Triplet 3: Lagged vs Instantaneous Correlation Robustness

**SPIs**: `{xcorr_max, SpearmanR, KendallTau}`

**Hypothesis**: Maximum cross-correlation (XCorr) exploits temporal structure by optimizing over lags, while Spearman and Kendall measure instantaneous rank-based associations. For generators with systematic time lags, XCorr should outperform instantaneous measures, creating divergence in SPI-SPI correlations.

### Scientific Rationale

- **Max Cross-Correlation (xcorr_max)**: xcorr_max = max_τ |corr(X_t, Y_{t+τ})| over lag τ ∈ [-L, +L]
- **Spearman's ρ**: Rank correlation on instantaneous observations (X_t, Y_t)
- **Kendall's τ**: Concordance-based rank correlation (X_t, Y_t)

**Key distinctions**:
1. XCorr **optimizes over lags**: detects lagged linear dependencies that instantaneous measures miss
2. Spearman and Kendall are **instantaneous**: only see (X_t, Y_t) pairs, blind to temporal structure
3. Spearman ≈ Kendall for most cases: both rank-based, robust to monotonic transforms, highly correlated
4. For lagged linear: xcorr_max >> ρ_s, τ_K (XCorr finds optimal lag, instantaneous miss it)

### Expected SPI-SPI Correlations by Generator

| Generator | ρ(XCorr, SpearmanR) | ρ(XCorr, KendallTau) | ρ(SpearmanR, KendallTau) | Interpretation |
|-----------|---------------------|----------------------|--------------------------|----------------|
| **VAR(1)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | VAR has instantaneous + lagged correlations: XCorr captures both, Spearman and Kendall capture instantaneous. Expect strong agreement across all three, with ρ(ρ_s, τ_K) ≈ +0.95 (rank measures agree perfectly). |
| **OU-network** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | Network coupling: similar to VAR(1). |
| **Kuramoto** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | **Very High (+0.9 to +0.98)** | Phase coherence: instantaneous coherence strong (Spearman, Kendall agree), XCorr detects lagged coherence too. |
| **Stuart-Landau** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | **Very High (+0.9 to +0.98)** | Limit cycle: similar to Kuramoto. |
| **Lorenz-96** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | **Very High (+0.9 to +0.98)** | Chaotic coupling: XCorr finds lagged dependencies, instantaneous weaker but Spearman-Kendall still agree strongly. |
| **Rössler-coupled** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | **Very High (+0.9 to +0.98)** | Chaotic coupled: similar to Lorenz-96. |
| **OU-heavyTail** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | Linear + heavy tails: all three robust (rank-based), strong agreement. |
| **GBM-returns** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | **Very High (+0.9 to +0.98)** | Common factor: instantaneous correlation present (Spearman, Kendall detect), XCorr weaker (no strong lags). |
| **TimeWarp-clones** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | Time-warped clones: XCorr perfect (finds alignment), Spearman and Kendall also strong (clones similar even without alignment). |
| **CML-logistic** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | **Very High (+0.9 to +0.98)** | Chaotic map: similar to Lorenz-96. |
| **Cauchy-OU** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | Linear + infinite variance: all rank-based, robust, strong agreement. |
| **Unidirectional-Cascade** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | **Very High (+0.9 to +0.98)** | Cascade (X₁→X₂→...): XCorr detects lagged cascade strongly, instantaneous weaker (X₁ ⊥ X₃ at t). **Expect**: ρ(XCorr, ρ_s) < ρ(ρ_s, τ_K) because XCorr exploits lags that instantaneous miss. |
| **Quadratic-Coupling** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | **Very High (+0.9 to +0.98)** | Non-monotonic quadratic: all measures struggle (XCorr linear, ranks fail on non-monotonic), but Spearman-Kendall still agree. |
| **Exponential-Transform** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **Very High (+0.9 to +0.98)** | Monotonic transform: all three robust (XCorr detects linear, ranks invariant), strong agreement. |
| **Phase-Lagged-Oscillators** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | **Very High (+0.9 to +0.98)** | **Systematic π/4 phase lags**: XCorr perfect (finds π/4 lag), Spearman and Kendall weak (instantaneous decorrelated by phase shift). **Critical test**: ρ(XCorr, ρ_s) << ρ(ρ_s, τ_K) because XCorr exploits lags, instantaneous measures fail. Expect ρ(XCorr, ρ_s) ≈ +0.4, ρ(ρ_s, τ_K) ≈ +0.95. |

### Key Predictions

1. **All generators**: ρ(SpearmanR, KendallTau) very high (ρ > +0.9) because both are rank-based, instantaneous, nearly identical assumptions
2. **Lagged generators** (Phase-Lagged-Oscillators, Unidirectional-Cascade): ρ(XCorr, ρ_s) << ρ(ρ_s, τ_K) because XCorr exploits temporal structure
3. **Phase-Lagged-Oscillators**: **Critical test** - ρ(XCorr, ρ_s) ≈ +0.4 (weak) vs ρ(ρ_s, τ_K) ≈ +0.95 (near-perfect)
4. **Instantaneous generators** (GBM-returns): ρ(XCorr, ρ_s) ≈ ρ(XCorr, τ_K) because no temporal structure to exploit
5. **Linear generators** (VAR, OU-network, Cauchy-OU, Exponential-Transform): High agreement across all three (ρ > +0.7)

---

## Triplet 4: Temporal Alignment (DTW vs Cross-Correlation vs Euclidean Distance)

**SPIs**: `{dtw_null, xcorr_max, pdist_euclidean}`

**Hypothesis**: Dynamic time warping (DTW) handles temporal misalignment, cross-correlation handles fixed-lag alignment, and Euclidean distance assumes no alignment. Generators with time warping or variable lags should show DTW advantage, creating distinct SPI-SPI correlation patterns.

### Scientific Rationale

- **DTW (dtw_null)**: Finds optimal non-linear time alignment via dynamic programming; invariant to monotonic time warping
- **XCorr (xcorr_max)**: Finds optimal fixed-lag linear shift; assumes uniform time warping (constant lag)
- **Euclidean distance (pdist_euclidean)**: d = √Σ(X_t - Y_t)²; no alignment, assumes perfect synchronization

**Key distinctions**:
1. DTW **handles non-linear time warping**: optimal for variable-speed processes
2. XCorr **handles fixed lags**: optimal for constant time shifts
3. Euclidean **no alignment**: optimal only for perfectly synchronized signals
4. For time-warped data: DTW >> XCorr > Euclidean
5. For fixed-lag data: XCorr ≈ DTW >> Euclidean
6. For synchronized data: all three agree

### Expected SPI-SPI Correlations by Generator

| Generator | ρ(DTW, XCorr) | ρ(DTW, Euclidean) | ρ(XCorr, Euclidean) | Interpretation |
|-----------|---------------|-------------------|---------------------|----------------|
| **VAR(1)** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | Lagged linear: XCorr optimal (finds lag), DTW also good, Euclidean poor (ignores lag). Expect ρ(DTW, XCorr) > ρ(DTW, Euclidean) ≈ ρ(XCorr, Euclidean). |
| **OU-network** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | Network coupling: similar to VAR(1). |
| **Kuramoto** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Phase coherence: DTW and XCorr both detect phase alignment, Euclidean weaker. |
| **Stuart-Landau** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Limit cycle: similar to Kuramoto. |
| **Lorenz-96** | **Moderate (+0.5 to +0.7)** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | Chaotic coupling: DTW handles chaotic misalignment better, XCorr moderate, Euclidean poor. |
| **Rössler-coupled** | **Moderate (+0.5 to +0.7)** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | Chaotic coupled: similar to Lorenz-96. |
| **OU-heavyTail** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Linear + heavy tails: DTW and XCorr handle lags, Euclidean degraded by outliers. |
| **GBM-returns** | **Low-Mod (+0.3 to +0.6)** | **Low (+0.2 to +0.5)** | **Moderate (+0.4 to +0.7)** | Common factor + nonlinearity: no strong temporal structure, weak alignment benefit. Expect ρ(XCorr, Euclidean) > ρ(DTW, XCorr) because no warping to exploit. |
| **TimeWarp-clones** | **Very High (+0.85 to +0.95)** | **Low (+0.2 to +0.5)** | **Very Low (0.0 to +0.3)** | **Time-warped clones**: DTW perfect (designed for this), XCorr good (fixed lag approximation), Euclidean FAILS (misaligned). **Critical test**: ρ(DTW, XCorr) ≈ +0.9 >> ρ(DTW, Euclidean) ≈ +0.3 >> ρ(XCorr, Euclidean) ≈ +0.1. This generator specifically tests DTW advantage. |
| **CML-logistic** | **Moderate (+0.5 to +0.7)** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | Chaotic map: similar to Lorenz-96. |
| **Cauchy-OU** | **High (+0.7 to +0.9)** | **Low (+0.2 to +0.5)** | **Low (+0.2 to +0.5)** | Linear + infinite variance: DTW and XCorr handle lags, Euclidean FAILS (infinite variance → large distances). **Expect**: ρ(DTW, XCorr) >> ρ(DTW, Euclidean) ≈ ρ(XCorr, Euclidean). |
| **Unidirectional-Cascade** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.4 to +0.7)** | Cascade with lags: DTW and XCorr both detect lagged cascade, Euclidean weaker. |
| **Quadratic-Coupling** | **Low-Mod (+0.3 to +0.6)** | **Low (+0.2 to +0.5)** | **Moderate (+0.4 to +0.7)** | Non-monotonic quadratic: all measures struggle, no strong temporal structure. |
| **Exponential-Transform** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Monotonic transform of linear: DTW and XCorr handle lags, Euclidean weaker. |
| **Phase-Lagged-Oscillators** | **Very High (+0.85 to +0.95)** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | **Systematic π/4 phase lags**: Both DTW and XCorr perfect (detect systematic lag), Euclidean poor (misaligned). **Critical test**: ρ(DTW, XCorr) ≈ +0.9 >> ρ(DTW, Euclidean) ≈ ρ(XCorr, Euclidean) ≈ +0.4. |

### Key Predictions

1. **TimeWarp-clones**: **Critical test** - ρ(DTW, XCorr) ≈ +0.9 >> ρ(DTW, Euclidean) ≈ +0.3 because DTW designed for time warping
2. **Phase-Lagged-Oscillators**: ρ(DTW, XCorr) ≈ +0.9 (both detect systematic lag) >> ρ(DTW, Euclidean) ≈ +0.4
3. **Cauchy-OU**: ρ(DTW, XCorr) >> ρ(DTW, Euclidean) ≈ ρ(XCorr, Euclidean) because Euclidean fails on infinite variance
4. **Linear generators** (VAR, OU-network): ρ(DTW, XCorr) high (both handle lags), but ρ(DTW/XCorr, Euclidean) moderate (lags hurt Euclidean)
5. **GBM-returns**: Weak correlations across all pairs (no strong temporal structure to exploit)

---

## Triplet 5: Nonlinear Dependency Detection Gradient

**SPIs**: `{dcorr_biased-False, mi_kraskov_NN-4, SpearmanR}`

**Hypothesis**: Distance correlation (dCorr) detects all dependencies (linear + nonlinear), mutual information (MI) detects arbitrary nonlinear, and Spearman (ρ_s) detects monotonic. This creates a sensitivity gradient: dCorr ≥ MI ≥ ρ_s in generality, which should manifest in SPI-SPI correlations across generators with varying nonlinearity.

### Scientific Rationale

- **Distance Correlation (dCorr)**: dCorr(X,Y) ∈ [0,1], equals 0 iff X ⊥ Y (detects all dependencies including non-monotonic)
- **Mutual Information (MI)**: MI(X,Y) ≥ 0, equals 0 iff X ⊥ Y (detects all dependencies via entropy)
- **Spearman's ρ**: ρ_s ∈ [-1,+1], detects monotonic associations (rank-based)

**Key distinctions**:
1. dCorr **most general**: detects linear, monotonic nonlinear, non-monotonic nonlinear
2. MI **general nonlinear**: detects all dependencies but less interpretable than dCorr (unbounded)
3. ρ_s **monotonic only**: fails on non-monotonic (e.g., U-shaped, quadratic)
4. For linear: all three agree strongly
5. For monotonic nonlinear: dCorr ≈ MI >> ρ_s (ρ_s perfect, but dCorr and MI higher magnitude)
6. For non-monotonic: dCorr ≈ MI >> ρ_s (ρ_s fails completely)

### Expected SPI-SPI Correlations by Generator

| Generator | ρ(dCorr, MI) | ρ(dCorr, SpearmanR) | ρ(MI, SpearmanR) | Interpretation |
|-----------|--------------|---------------------|------------------|----------------|
| **VAR(1)** | **High (+0.7 to +0.9)** | **High (+0.8 to +0.95)** | **High (+0.7 to +0.9)** | Linear Gaussian: all three agree strongly (dCorr² ≈ ρ², MI ≈ -log(1-ρ²), ρ_s ≈ ρ). |
| **OU-network** | **High (+0.7 to +0.9)** | **High (+0.8 to +0.95)** | **High (+0.7 to +0.9)** | Linear coupling: same as VAR(1). |
| **Kuramoto** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Nonlinear oscillatory: dCorr and MI both detect phase coherence strongly, ρ_s weaker (non-monotonic phase wrapping). Expect ρ(dCorr, MI) > ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s). |
| **Stuart-Landau** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Limit cycle: similar to Kuramoto. |
| **Lorenz-96** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.5 to +0.7)** | Chaotic nonlinear: dCorr and MI both sensitive, ρ_s weaker. Expect ρ(dCorr, MI) > ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s). |
| **Rössler-coupled** | **High (+0.7 to +0.9)** | **Moderate (+0.4 to +0.7)** | **Moderate (+0.5 to +0.7)** | Chaotic coupled: similar to Lorenz-96. |
| **OU-heavyTail** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Linear + heavy tails: all three robust (dCorr and MI use ranks/distances, ρ_s rank-based), strong agreement. |
| **GBM-returns** | **High (+0.7 to +0.9)** | **Low-Mod (+0.3 to +0.6)** | **Low-Mod (+0.3 to +0.6)** | Common factor + nonlinearity: dCorr and MI detect shared dependencies strongly, ρ_s weaker. Expect ρ(dCorr, MI) >> ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s). |
| **TimeWarp-clones** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Perfect clones: all three detect perfect dependency, strong agreement. |
| **CML-logistic** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Chaotic map: similar to Lorenz-96. |
| **Cauchy-OU** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Linear + infinite variance: all three robust (dCorr uses distances, MI entropy, ρ_s ranks), strong agreement. |
| **Unidirectional-Cascade** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | **High (+0.7 to +0.9)** | Linear cascade: all three agree, strong correlations. |
| **Quadratic-Coupling** | **Very High (+0.85 to +0.95)** | **Low (+0.2 to +0.5)** | **Low (+0.2 to +0.5)** | **Non-monotonic quadratic**: dCorr and MI both detect X² dependencies strongly (dCorr perfect for quadratic, MI high), ρ_s FAILS (non-monotonic). **Critical test**: ρ(dCorr, MI) ≈ +0.9 >> ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s) ≈ +0.3. This generator tests the dCorr/MI advantage over rank methods for non-monotonic. |
| **Exponential-Transform** | **Very High (+0.85 to +0.95)** | **Very High (+0.85 to +0.95)** | **Very High (+0.85 to +0.95)** | Monotonic nonlinear: all three perfect (dCorr detects dependency, MI detects info, ρ_s invariant to monotonic). **Expect**: All correlations ≈ +0.9 (perfect agreement). |
| **Phase-Lagged-Oscillators** | **High (+0.7 to +0.9)** | **Moderate (+0.5 to +0.7)** | **Moderate (+0.5 to +0.7)** | Linear oscillators + phase lag: dCorr and MI handle lags better, ρ_s weaker. |

### Key Predictions

1. **Quadratic-Coupling**: **Critical test** - ρ(dCorr, MI) ≈ +0.9 >> ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s) ≈ +0.3 (non-monotonic)
2. **Exponential-Transform**: All three ≈ +0.9 (monotonic nonlinear, perfect agreement)
3. **Linear generators** (VAR, OU-network, Unidirectional-Cascade, Cauchy-OU): All three highly correlated (ρ > +0.7)
4. **Nonlinear generators** (Kuramoto, Stuart-Landau, Lorenz-96, Rössler, CML, GBM-returns): ρ(dCorr, MI) > ρ(dCorr, ρ_s) ≈ ρ(MI, ρ_s)
5. **TimeWarp-clones**: All three ≈ +0.7 to +0.9 (perfect dependency, all detect)

---

## Summary of Critical Tests

### Most Diagnostic Generator-Triplet Pairs

1. **Quadratic-Coupling × Triplet 1**: ρ(MI, ρ_s) ≈ +0.4, ρ(MI, cov) ≈ 0 (non-monotonic breaks rank and linear)
2. **Quadratic-Coupling × Triplet 5**: ρ(dCorr, MI) ≈ +0.9 >> ρ(dCorr, ρ_s) ≈ +0.3 (dCorr/MI vs rank)
3. **Unidirectional-Cascade × Triplet 2**: ρ(TE, TLMI) ≈ +0.9 (directed flow detection)
4. **Phase-Lagged-Oscillators × Triplet 3**: ρ(XCorr, ρ_s) ≈ +0.4 << ρ(ρ_s, τ_K) ≈ +0.95 (lag vs instantaneous)
5. **TimeWarp-clones × Triplet 4**: ρ(DTW, XCorr) ≈ +0.9 >> ρ(DTW, Euclidean) ≈ +0.3 (time warping)
6. **Cauchy-OU × Triplet 1**: ρ(MI, ρ_s) >> ρ(MI, cov) ≈ 0 (infinite variance breaks moments)
7. **Exponential-Transform × Triplet 1**: ρ(MI, ρ_s) ≈ +0.9 >> ρ(MI, cov) ≈ +0.5 (monotonic vs linear)

### Expected Outcomes for Validation

If the pipeline is correct, we expect:
- **Strong Spearman-Kendall correlation** (ρ > +0.9) across all generators (both rank-based, instantaneous)
- **Divergence on non-monotonic** (Quadratic-Coupling): MI-dCorr high, MI/dCorr-ρ_s low
- **Divergence on heavy tails** (Cauchy-OU): MI-ρ_s high, MI/ρ_s-cov very low
- **Divergence on lags** (Phase-Lagged-Oscillators): XCorr-ρ_s low, ρ_s-Kendall high
- **Divergence on time warping** (TimeWarp-clones): DTW-XCorr high, DTW-Euclidean low
- **Convergence on linear** (VAR, OU-network, Unidirectional-Cascade): All measures agree (ρ > +0.7)

---

## Implementation Notes

### YAML Configuration for `--spi-triplets`

Create `configs/triplets.yaml`:

```yaml
triplets:
  - name: "monotonic_vs_info_vs_linear"
    spis:
      - "SpearmanR"
      - "mi_kraskov_NN-4"
      - "cov_EmpiricalCovariance"
  
  - name: "directed_vs_undirected"
    spis:
      - "te_kraskov_k-1_l-1"
      - "mi_kraskov_NN-4"
      - "tlmi_kraskov_NN-4"
  
  - name: "lagged_vs_instantaneous"
    spis:
      - "xcorr_max"
      - "SpearmanR"
      - "KendallTau"
  
  - name: "temporal_alignment"
    spis:
      - "dtw_null"
      - "xcorr_max"
      - "pdist_euclidean"
  
  - name: "nonlinear_gradient"
    spis:
      - "dcorr_biased-False"
      - "mi_kraskov_NN-4"
      - "SpearmanR"
```

### Running the Pipeline

```powershell
# Overnight run with dev+++ profile and all 5 triplets
python -m spimts --mode dev+++ --subset pilot0 --spi-triplets configs/triplets.yaml --outdir results/dev+++
```

### Expected Output Structure

Each generator will produce:
- `arrays/*.npy`: Raw SPI matrices (M×M) for each SPI in the triplet
- `csv/*.csv`: SPI-SPI correlation matrix (5×5 for each triplet)
- `plots/*.png`: Heatmaps of SPI-SPI correlations

### Analysis Plan

1. **Extract SPI-SPI correlations** from `csv/` for each generator-triplet pair
2. **Compare against expectations** in tables above
3. **Identify anomalies**: Cases where ρ(SPI_i, SPI_j) diverges from expectation by >0.3
4. **Validate critical tests**: Quadratic-Coupling, Cauchy-OU, Phase-Lagged-Oscillators, TimeWarp-clones, Unidirectional-Cascade
5. **Statistical robustness**: M=30 (435 edges) provides good power for ρ estimation (SE ≈ 0.05)

---

## References

- Reshef et al. (2011): "Detecting Novel Associations in Large Data Sets" (MIC, maximal information coefficient)
- Székely et al. (2007): "Measuring and Testing Dependence by Correlation of Distances" (distance correlation)
- Schreiber (2000): "Measuring Information Transfer" (transfer entropy)
- Kraskov et al. (2004): "Estimating Mutual Information" (k-NN MI estimation)
- Salvador & Chan (2007): "Toward Accurate Dynamic Time Warping in Linear Time and Space" (DTW)

---

**Document prepared**: January 2025  
**Version**: 1.0  
**Purpose**: Overnight run (dev+++ profile) with 5 SPI triplets across 15 generators  
**Expected runtime**: 8-12 hours (M=25-35, T=1500-2500 per model)  
**Contact**: Research team
