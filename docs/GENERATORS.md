# Generator Analysis & Design

**Last Updated:** October 25, 2025 (Extended with Priority 1-2 generators)  
**Purpose:** Document how each MTS generator exploits specific SPI assumption differences to infer data character

---

## Core Intuition (from CONTEXT.md)

> "For certain spatiotemporal dependency structures in MTS, some SPIs yield similar values; for others, these same SPIs yield very different values."

**Key insight:** By examining which SPIs agree/disagree in the SPI-SPI correlation space, we can infer the **character** of the dynamical system (linearity, monotonicity, directionality, lag structure, etc.).

---

## Three Core Case Studies + 2 Extended Case Studies

### 1. {Spearman ρ, Pearson r}: Monotonicity vs Linearity

**What they measure:**
- **Pearson r**: Linear association (assumes Gaussian, sensitive to outliers)
- **Spearman ρ**: Monotonic association (rank-based, robust to outliers)

**When they agree (high ρ-r correlation in SPI-space):**
- Data is linear AND Gaussian → both capture same information
- Example: VAR(1) with Gaussian noise

**When they disagree (low ρ-r correlation in SPI-space):**
- **Case A**: Monotonic but nonlinear → ρ high, r low (e.g., exponential coupling)
- **Case B**: Heavy-tailed outliers → ρ robust, r distorted (e.g., **Cauchy noise - NEW**)
- **Case C**: Non-monotonic relationship → both low but for different reasons

**Character inference:** Discordance reveals **non-Gaussianity** or **nonlinearity**

---

### 2. {MI, Spearman ρ, Pearson r}: Information vs Correlation

**What they measure:**
- **MI (Mutual Information)**: General statistical dependence (linear + nonlinear, no lags)
- **Pearson r / Spearman ρ**: Only linear/monotonic dependence (instantaneous)

**When they agree:**
- Purely linear/monotonic dependencies with no time lags
- Example: OU network (diffusive coupling, Gaussian)

**When they disagree:**
- **Case A**: Nonlinear coupling → MI high, ρ/r low (e.g., **quadratic interactions - NEW**)
- **Case B**: Time-lagged dependencies → MI captures across lags, correlations miss
- **Case C**: Complex multivariate dependence → MI integrates, correlations pairwise only

**Character inference:** Discordance reveals **nonlinear coupling** or **temporal structure**

---

### 3. {TE, MI}: Directed vs Undirected Information

**What they measure:**
- **TE (Transfer Entropy)**: Directed information flow (X→Y predictive power)
- **MI (Mutual Information)**: Undirected mutual dependence (X↔Y association)

**When they agree:**
- Symmetric bidirectional coupling → TE(X→Y) ≈ TE(Y→X) ≈ MI(X,Y)
- Example: Kuramoto oscillators (symmetric phase coupling)

**When they disagree:**
- **Case A**: Unidirectional cascade → TE(X→Y) >> TE(Y→X), MI moderate (**NEW generator**)
- **Case B**: Delayed feedback → TE asymmetric, MI symmetric
- **Case C**: Common drive (Z→X, Z→Y) → MI high, TE low (no direct X→Y flow)

**Character inference:** Discordance reveals **causal structure** and **directionality**

---

### 4. {SpearmanR, KendallTau, CrossCorr-max}: Rank-Based Robustness (EXTENDED CASE)

**What they measure:**
- **Spearman ρ**: Pearson on ranks (robust to monotonic transforms)
- **Kendall τ**: Concordance-based (even more robust, different sensitivity)
- **CrossCorr-max**: Max lagged correlation (captures phase shifts)

**When they agree (high inter-correlations):**
- Monotonic relationships with consistent lag structure
- All three are rank-based/robust → should converge under heavy tails
- Example: **Cauchy-OU** (all three should agree, contrasting with Pearson)

**When they disagree:**
- Complex lag structure → CrossCorr-max high, Spearman/Kendall lower
- Non-monotonic dependencies → all three low but Kendall more conservative

**Character inference:** Tests **outlier robustness** and **lag sensitivity**

---

### 5. {CrossCorr, CoherenceMag, ImaginaryCoherence}: Time vs Frequency Domain (EXTENDED CASE)

**What they measure:**
- **CrossCorr-max**: Lagged correlation (time-domain phase detection)
- **CoherenceMag**: Frequency-domain amplitude coupling
- **ImaginaryCoherence**: Frequency-domain phase coupling (sensitive to quadrature relationships)

**When they agree:**
- Pure amplitude coupling with no phase shifts → CrossCorr & CoherenceMag high, ImagCoh low
- Example: In-phase oscillators

**When they disagree:**
- **Systematic phase lags** → CrossCorr high (finds lag), ImagCoh high (detects phase shift), CoherenceMag moderate
- Example: **Phase-Lagged Oscillators (NEW)** - each channel π/4 ahead of previous

**Character inference:** Distinguishes **phase coupling** from **amplitude coupling**

---

## Current Generators: Coverage Analysis

### ✅ Generators That Purposefully Exploit SPI Differences

**1. VAR(1) - Linear Baseline**
- **Purpose**: Establish baseline where r, ρ, MI all agree (linear, Gaussian, no lags)
- **SPI-SPI expectations**:
  - r ↔ ρ: High correlation (both capture linearity)
  - MI ↔ r: High correlation (MI reduces to correlation for Gaussian linear)
  - TE ↔ MI: Moderate correlation (VAR has directionality)
- **Character**: Pure linear Gaussian autoregression

**2. Kuramoto Oscillators - Symmetric Nonlinear**
- **Purpose**: Nonlinear (sin coupling) but symmetric → tests MI vs ρ/r, symmetric TE
- **SPI-SPI expectations**:
  - r ↔ ρ: Moderate (nonlinear sin reduces r, ρ more robust)
  - MI ↔ ρ: High (both capture phase synchronization)
  - TE(X→Y) ↔ TE(Y→X): High if symmetric coupling (can be broken with `directed=True`)
- **Character**: Nonlinear symmetric coupling with synchronization

**3. Stuart-Landau - Complex Oscillations**
- **Purpose**: Limit-cycle dynamics with amplitude+phase coupling
- **SPI-SPI expectations**:
  - r ↔ ρ: Low (oscillations break linearity)
  - MI ↔ DTW: High (both capture temporal patterns)
- **Character**: Nonlinear oscillatory coupling

**4. Lorenz-96 - Chaotic Dynamics**
- **Purpose**: Deterministic chaos → strong nonlinearity, no stochastic noise
- **SPI-SPI expectations**:
  - r ↔ MI: Low (chaos is highly nonlinear)
  - TE ↔ GC: Moderate (both capture causal flow, but GC assumes linearity)
- **Character**: Deterministic nonlinear chaos

**5. GBM Returns - Common Factor Model**
- **Purpose**: Common factor (market) + idiosyncratic noise → tests TE vs MI
- **SPI-SPI expectations**:
  - MI ↔ ρ: High (linear factor model)
  - TE: Low (no direct causal flow, only common drive)
- **Character**: Common latent factor (no direct coupling)

**6. OU Heavy-Tail - Fat Tails**
- **Purpose**: Student-t noise instead of Gaussian → tests r vs ρ robustness
- **SPI-SPI expectations**:
  - r ↔ ρ: Moderate-Low (outliers distort r, ρ robust)
  - MI ↔ r: Low (MI handles non-Gaussianity better)
- **Character**: Heavy-tailed marginals

---

### ❌ Generators with Unclear SPI Exploitation

**7. OU Network**
- **Current purpose**: Diffusive coupling on ring
- **Issue**: Similar to VAR but continuous-time → not clearly distinct in SPI-space
- **Verdict**: KEEP (useful as Gaussian baseline), but not purpose-designed for SPI contrast

**8. Rössler Coupled**
- **Current purpose**: Coupled chaotic oscillators
- **Issue**: Overlaps with Lorenz-96 (both chaotic)
- **Verdict**: KEEP (3D vs 1D chaos has different signatures), useful

**9. CML Logistic**
- **Current purpose**: Coupled map lattice with logistic maps
- **Issue**: Chaotic like Lorenz-96, but discrete-time
- **Verdict**: KEEP (tests discrete vs continuous chaos), marginal utility

**10. TimeWarp Clones**
- **Current purpose**: Time-warped copies of base signal → tests DTW
- **Issue**: Designed for DTW specifically, but doesn't exploit SPI contrasts broadly
- **Verdict**: QUESTIONABLE - useful for DTW, but not for core case studies

---

## ⚠️ MISSING Generators (Critical Gaps)

### MUST ADD (Priority 1): {ρ, r} Case Study

**1. Cauchy Noise MTS** (YOUR EXAMPLE from prompt)
- **Purpose**: Exploit r's sensitivity to outliers vs ρ's robustness
- **Implementation**: Ornstein-Uhlenbeck with Cauchy innovations
- **SPI-SPI expectations**:
  - **r ↔ ρ: LOW** (Cauchy outliers destroy r, ρ unaffected by rank)
  - MI ↔ ρ: High (both handle heavy tails)
  - MI ↔ r: Low (r unreliable)
- **Character inference**: Heavy-tailed noise → observe r↔ρ discordance

**2. Monotonic Nonlinear Coupling**
- **Purpose**: Monotonic but not linear (e.g., exponential/log transforms)
- **Implementation**: VAR-like but with X_t = A @ exp(X_{t-1}) + noise
- **SPI-SPI expectations**:
  - **ρ high, r low** (monotonic but curved relationship)
  - MI ↔ ρ: High (both capture monotonicity)
  - MI ↔ r: Moderate (MI handles nonlinearity)
- **Character inference**: Nonlinear monotonic → observe ρ>r with both significant

### MUST ADD (Priority 1): {MI, ρ, r} Case Study

**3. Quadratic Coupling (Non-Monotonic)**
- **Purpose**: Nonlinear, non-monotonic interaction (e.g., X_i · X_j² coupling)
- **Implementation**: dX_i/dt = -X_i + Σ_j w_ij · X_j² + noise
- **SPI-SPI expectations**:
  - **r, ρ low** (non-monotonic breaks both)
  - **MI high** (captures nonlinear dependence)
  - **MI ↔ r: LOW**, MI ↔ ρ: LOW
- **Character inference**: Nonlinear non-monotonic → observe MI alone detects coupling

**4. Lagged Linear Coupling**
- **Purpose**: Pure time-lag dependencies (X_i(t) ← X_j(t-τ))
- **Implementation**: VAR-like but X_t = A @ X_{t-τ} with τ>1
- **SPI-SPI expectations**:
  - **r, ρ low** (instantaneous correlations miss lag)
  - **MI moderate-high** (captures across lags if computed with history)
  - **XCorr high** (explicitly handles lags)
- **Character inference**: Temporal lag → observe MI/XCorr vs ρ/r discordance

### MUST ADD (Priority 1): {TE, MI} Case Study

**5. Unidirectional Cascade**
- **Purpose**: Clear directionality (X₁→X₂→X₃→...→Xₙ, no reverse flow)
- **Implementation**: Linear cascade: X_i(t) = a·X_{i-1}(t-1) + noise
- **SPI-SPI expectations**:
  - **TE(i→i+1) >> TE(i+1→i)** (strong asymmetry)
  - **MI(i, i+1) moderate** (undirected dependence)
  - **TE ↔ MI: LOW** (TE is directional, MI is not)
- **Character inference**: Causal flow → observe TE asymmetry vs MI symmetry

**6. Feedback Loop (Bidirectional)**
- **Purpose**: X→Y→X delayed feedback
- **Implementation**: X(t) = f(Y(t-τ)), Y(t) = g(X(t-τ))
- **SPI-SPI expectations**:
  - **TE(X→Y) ≈ TE(Y→X)** (symmetric feedback)
  - **MI high** (strong mutual dependence)
  - **TE ↔ MI: MODERATE** (both detect coupling, but TE shows bidirectionality)
- **Character inference**: Feedback → observe symmetric TE with high MI

---

## Recommended Additions (Priority Order)

### Priority 1 (MUST HAVE for case studies):
1. **Cauchy Noise OU** → {ρ,r} outlier sensitivity test
2. **Unidirectional Cascade** → {TE,MI} directionality test
3. **Quadratic Coupling** → {MI,ρ,r} nonlinearity test

### Priority 2 (Nice to have):
4. **Monotonic Nonlinear** → {ρ,r} linearity vs monotonicity
5. **Lagged Linear** → {MI,ρ} temporal structure

### Priority 3 (Optional):
6. **Feedback Loop** → {TE,MI} bidirectional causality

---

## Generator Design Principles

**Each generator should:**
1. ✅ **Have a clear SPI contrast hypothesis** ("We expect X and Y to disagree because...")
2. ✅ **Target one case study** ({ρ,r}, {MI,ρ,r}, or {TE,MI})
3. ✅ **Be interpretable** (simple dynamics, clear coupling structure)
4. ✅ **Be controllable** (parameters to tune effect strength)

**Avoid:**
1. ❌ Generators that are "interesting" but don't target SPI contrasts
2. ❌ Redundant generators (too many chaotic systems)
3. ❌ Overly complex generators (hard to interpret SPI-space results)

---

## SPI Availability (from config.yaml)

**Available SPIs for case studies:**

**Correlation family:**
- `spearmanr` (Spearman ρ)
- `pearsonr` (Pearson r)  
- `kendalltau` (Kendall τ)
- `cov_EmpiricalCovariance`

**Information theory:**
- `mi_kraskov_NN-4` (Mutual Information)
- `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1` (Transfer Entropy)
- `tlmi_kraskov_NN-4` (Time-Lagged MI)
- `gc_gaussian_k-1_kt-1_l-1_lt-1` (Granger Causality)

**Distance/Lag:**
- `dtw` (Dynamic Time Warping)
- `xcorr_max_sig-True` (Cross-correlation)
- `pdist_euclidean`, `pdist_cosine`

**Other SPI Triplets to Consider:**
- **{Kendall τ, Spearman ρ, Pearson r}**: All three rank/linear correlations
- **{GC, TE, MI}**: Three causality/information measures (linear vs nonlinear)
- **{DTW, XCorr, ρ}**: Temporal alignment vs instantaneous correlation
- **{MI, TLMI, XCorr}**: Information with/without lag vs correlation with lag

---

## NEW GENERATORS IMPLEMENTED (October 25, 2025)

### Priority 1: Critical Gaps (IMPLEMENTED)

**10. Cauchy-OU** ✅ ADDED
- **File**: `generators.py` line ~190
- **Purpose**: Test {ρ,r} with extreme outliers (Cauchy has infinite variance)
- **Case Study**: {SpearmanR, KendallTau, CrossCorr-max} vs Pearson r
- **Dynamics**: Ornstein-Uhlenbeck with Cauchy innovations (Student-t df=1)
- **SPI-SPI expectations**:
  - **ρ ↔ τ ↔ XCorr: HIGH** (all rank-based/robust → agree under heavy tails)
  - **r ↔ ρ: LOW** (Pearson destroyed by outliers, Spearman robust)
  - **r ↔ τ: LOW** (Pearson unreliable, Kendall robust)
- **Character inference**: Heavy-tail robustness → rank-based SPIs converge, Pearson diverges
- **Implementation notes**: Uses `np.sqrt(dt)` scaling for Cauchy, `zscore(..., nan_policy='omit')` to handle potential infinities

**11. Unidirectional-Cascade** ✅ ADDED
- **File**: `generators.py` line ~215
- **Purpose**: Test {TE,MI,TLMI} directionality detection
- **Case Study**: {TransferEntropy, TimeLaggedMutualInfo, DirectedInfo}
- **Dynamics**: X₁→X₂→X₃...→Xₙ linear cascade, no reverse flow
- **SPI-SPI expectations**:
  - **TE(i→i+1) >> TE(i+1→i)** (strong asymmetry in forward vs reverse direction)
  - **MI symmetric** (undirected dependence → same for (i,i+1) and (i+1,i))
  - **TE ↔ MI: LOW** (TE captures direction, MI doesn't)
  - **TLMI ↔ TE: MEDIUM** (TLMI partially captures lag structure but not full directionality)
- **Character inference**: Causal flow → TE shows asymmetry, MI misses direction
- **Implementation notes**: First channel autonomous OU, each subsequent channel driven by previous only

**12. Quadratic-Coupling** ✅ ADDED
- **File**: `generators.py` line ~240
- **Purpose**: Test {MI,ρ,r} nonlinearity detection (non-monotonic)
- **Case Study**: {MutualInfo, DistanceCorrelation, HilbertSchmidtIndependenceCriterion}
- **Dynamics**: dX_i/dt = -α·X_i + Σ_j w·X_j² + noise (parabolic coupling)
- **SPI-SPI expectations**:
  - **MI high** (captures Y = X² dependence)
  - **ρ low, r low** (parabolic is non-monotonic → rank-based fails, linear definitely fails)
  - **dCorr ↔ MI: MEDIUM** (dCorr captures some nonlinearity but not as complete as MI)
  - **HSIC ↔ MI: MEDIUM-HIGH** (both kernel-based, should partially agree)
  - **MI ↔ ρ: LOW**, MI ↔ r: LOW (gradient: MI >> dCorr/HSIC >> ρ ≈ r)
- **Character inference**: Nonlinear non-monotonic → only MI (and kernel methods) detect coupling
- **Implementation notes**: Ring topology with k=2 neighbors, quadratic mean-field coupling

### Priority 2: Extended Case Studies (IMPLEMENTED)

**13. Exponential-Transform** ✅ ADDED
- **File**: `generators.py` line ~270
- **Purpose**: Test {ρ,dCorr,MI} monotonic nonlinearity
- **Case Study**: {SpearmanR, DistanceCorrelation, MutualInfo}
- **Dynamics**: Latent VAR(1) with observation Y = sign(Z)·exp(|Z|)
- **SPI-SPI expectations**:
  - **ρ ↔ dCorr: HIGH** (both detect monotonic nonlinearity)
  - **ρ ↔ MI: HIGH** (monotonic → rank-preserving → all three agree)
  - **dCorr ↔ MI: HIGH** (both capture nonlinear dependencies)
  - **r ↔ ρ: MEDIUM** (exponential curve reduces Pearson, Spearman robust)
- **Character inference**: Monotonic nonlinear → ρ, dCorr, MI all converge (contrasts with quadratic where ρ fails)
- **Implementation notes**: Monotonic transform preserves ranks perfectly

**14. Phase-Lagged-Oscillators** ✅ ADDED
- **File**: `generators.py` line ~295
- **Purpose**: Test {CrossCorr, CoherenceMag, ImaginaryCoherence} phase vs amplitude coupling
- **Case Study**: Time-domain vs frequency-domain phase detection
- **Dynamics**: Ring of oscillators, each channel π/4 phase ahead of previous
- **SPI-SPI expectations**:
  - **CrossCorr-max ↔ CoherenceMag: MEDIUM-HIGH** (both detect coupling across lags/frequencies)
  - **ImagCoh ↔ CoherenceMag: MEDIUM** (ImagCoh specifically sensitive to phase quadrature)
  - **ImagCoh isolated if pure phase shift** (high when phase≠0 or π, low when in-phase)
  - **CrossCorr finds optimal lag** (should peak at lag corresponding to π/4 phase shift)
- **Character inference**: Phase coupling → ImagCoh high, distinguishes from amplitude-only coupling
- **Implementation notes**: Systematic phase lag π/4 × channel_index, weak coupling maintains structure

---

## Redundant/Marginal Generators (Honest Assessment)

### ⚠️ MARGINAL UTILITY (keep for now, but acknowledge overlap):

**OU-network**
- **Overlap with**: VAR(1) - both are linear Gaussian diffusive processes
- **Difference**: Ring Laplacian topology vs full connectivity
- **Justification**: Spatial structure (neighbors) vs global coupling
- **Verdict**: **MARGINAL** - VAR(1) already establishes linear baseline, OU-network adds only topology variation
- **Keep?**: Yes, but mainly for spatial structure demonstration, not SPI contrast

**Rössler-coupled**
- **Overlap with**: Lorenz-96 - both are chaotic coupled systems
- **Difference**: 3D strange attractor (Rössler) vs 1D cyclic chaos (Lorenz)
- **Justification**: Different chaotic regimes (continuous vs map-like)
- **Verdict**: **MARGINAL** - Lorenz-96 is cleaner, more scalable, equally chaotic
- **Keep?**: Yes if studying attractor geometry, otherwise Lorenz-96 sufficient for chaos

**CML-logistic**
- **Overlap with**: Lorenz-96 - both generate chaotic spatiotemporal patterns
- **Difference**: Discrete map (logistic) vs continuous flow (Lorenz)
- **Justification**: Map dynamics can have different correlation structures than flows
- **Verdict**: **MARGINAL** - Unless studying discrete vs continuous chaos specifically
- **Keep?**: Yes, cheap to compute (no integration), discrete chaos useful contrast

### ✅ UNIQUE GENERATORS (irreplaceable):

**TimeWarp-clones**
- **Unique feature**: DTW-specific generator (clones with temporal distortions)
- **Purpose**: Tests DTW vs other distance measures
- **SPI contrast**: DTW ↔ Euclidean distance, DTW ↔ correlation
- **Verdict**: **UNIQUE** - Only generator explicitly designed for DTW case study
- **Keep?**: YES - irreplaceable for temporal alignment SPI tests

**GBM-returns**
- **Unique feature**: Common factor structure (factor model with ρ correlation parameter)
- **Purpose**: Tests common-drive vs pairwise coupling (Z→X, Z→Y but no X↔Y)
- **SPI contrast**: MI high (due to common Z), TE low (no direct X→Y)
- **Verdict**: **UNIQUE** - Only generator with explicit common factor
- **Keep?**: YES - critical for testing "confounding variable" scenarios

**OU-heavyTail**
- **Unique feature**: Student-t noise (finite variance, controllable tail weight via df)
- **Purpose**: Moderate heavy tails (df=2.5), less extreme than Cauchy
- **SPI contrast**: Tests ρ vs r under "reasonable" outliers
- **Verdict**: **USEFUL** - Complements Cauchy-OU (df=1), provides tail-weight gradient
- **Keep?**: YES - allows studying continuum from Gaussian (df=∞) → Cauchy (df=1)

---

## Final Generator Set (15 total)

### Core Baseline (4):
1. **VAR(1)** - Linear Gaussian baseline
2. **GBM-returns** - Common factor baseline
3. **TimeWarp-clones** - DTW alignment baseline
4. **Cauchy-OU** - Heavy-tail robustness baseline

### Oscillatory/Synchronization (3):
5. **Kuramoto** - Phase synchronization
6. **Stuart-Landau** - Amplitude+phase limit cycles
7. **Phase-Lagged-Oscillators** - Systematic phase lags (NEW)

### Chaotic (3):
8. **Lorenz-96** - Continuous chaotic flow
9. **Rössler-coupled** - 3D strange attractor
10. **CML-logistic** - Discrete chaotic map

### Nonlinear Coupling (3):
11. **Quadratic-Coupling** - Non-monotonic (NEW)
12. **Exponential-Transform** - Monotonic nonlinear (NEW)
13. **OU-heavyTail** - Moderate tail weight

### Directionality (2):
14. **Unidirectional-Cascade** - Pure causal flow (NEW)
15. **OU-network** - Spatial diffusion

**Removed**: None (but OU-network, Rössler, CML flagged as marginal)

---

## SPI Triplet Recommendations (Based on Available SPIs)

### Excellent Triplets to Visualize:

**1. {SpearmanR, mi_kraskov_NN-4, cov_EmpiricalCovariance}**
- Tests: Monotonic vs information vs linear
- Generators: Quadratic-Coupling (MI high, others low), Exponential-Transform (all high), Cauchy-OU (Spearman high, Cov low)

**2. {te_kraskov_k-1_l-1, mi_kraskov_NN-4, tlmi_kraskov_NN-4}**
- Tests: Directed vs undirected information flow
- Generators: Unidirectional-Cascade (TE asymmetric, MI symmetric), GBM-returns (MI high, TE low)

**3. {xcorr_max, SpearmanR, KendallTau}**
- Tests: Lagged vs instantaneous, rank-based robustness
- Generators: Phase-Lagged-Oscillators (XCorr finds lag), Cauchy-OU (all three robust)

**4. {dtw_null, xcorr_max, pdist_euclidean}**
- Tests: Temporal alignment vs distance vs correlation
- Generators: TimeWarp-clones (DTW high, others lower), Phase-Lagged-Oscillators (DTW vs XCorr)

**5. {dcorr_biased-False, hsic_biased-False, mi_kraskov_NN-4}**
- Tests: Distance correlation vs HSIC vs MI (all nonlinear-sensitive)
- Generators: Quadratic-Coupling (gradient: MI >> dCorr ≈ HSIC), Exponential-Transform (all agree)

---

## Overnight Run Considerations (Task 3)

**Models needing extended transients:**
- **Lorenz-96**: T_transients=1000 (chaotic attractor settling)
- **Rössler**: T_transients=2000 (longer settling for coupled systems)
- **Stuart-Landau**: T_transients=1500 (limit cycle convergence)
- **Kuramoto**: T_transients=2000 (phase locking)
- **CML**: T_transients=500 (map transients fast)

**Models fine with short transients:**
- **VAR, OU**: T_transients=50-100 (linear stability fast)
- **GBM**: T_transients=0 (no transients needed)
- **TimeWarp**: T_transients=0 (constructed signal)

**Optimal M,T for paper (from FUTURE.md):**
- **M=50** (sweet spot for statistical power)
- **T=2000** (sufficient for most dynamics)
- **Exceptions**: Lorenz-96, Kuramoto may need T=3000-5000 for full attractor coverage

---

## Next Steps

1. **Implement Priority 1 generators** (Cauchy, Cascade, Quadratic)
2. **Update compute.py** to handle per-model transient times
3. **Run dev++ with new generators** (M=15-25, T=2000-5000)
4. **Analyze SPI-space** to validate hypotheses:
   - Does Cauchy show ρ↔r discordance?
   - Does Cascade show TE asymmetry?
   - Does Quadratic show MI vs ρ/r separation?
5. **If validated, scale to paper** (M=50, T=3000)

---

## Summary

**Current state**: 10 generators, but only ~6 purposefully designed for SPI contrasts

**Critical gaps**: 
- No Cauchy noise ({ρ,r} test)
- No unidirectional cascade ({TE,MI} test)
- No quadratic coupling ({MI,ρ,r} test)

**Recommendation**: Add 3-5 targeted generators (Priority 1 list above)

**Do NOT add**: Generic "interesting" models without clear SPI hypothesis
