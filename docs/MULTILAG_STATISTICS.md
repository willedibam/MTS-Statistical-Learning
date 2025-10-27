# Multi-Lag Statistics: Theoretical Justification and Implementation

**Created:** October 26, 2025  
**Status:** Research exploration - DEFERRED until after baseline analysis  
**Scope:** Investigating whether custom multi-lag statistics add value beyond existing SPIs

---

## Motivation

**User question**: "How would you implement a statistic which looks at the subset {r_t, r_{t+1}, r_{t+τ}}?"

**Scientific context**: Many existing SPIs (TLMI, TE, xcorr_max/mean) already capture lagged dependencies. The question is whether a **joint multi-lag statistic** (considering {X_t, X_{t+1}, ..., X_{t+τ}} simultaneously) provides complementary information.

---

## Existing SPIs That Handle Lags

### 1. Time-Lagged Mutual Information (TLMI)
```
TLMI(X, Y; lag) = I(X_t ; Y_{t+lag})
```
- **What it captures**: Delayed mutual information between X and Y
- **Limitation**: Fixed single lag, pairwise between channels
- **In pilot0**: `tlmi_kraskov_NN-4`

### 2. Transfer Entropy (TE)
```
TE(X→Y) = I(Y_t ; X_{t-k} | Y_{t-l})
```
- **What it captures**: Information flow from X's past to Y's present, conditioning on Y's own history
- **Multi-lag**: Uses embeddings k and l (multiple past time steps)
- **In pilot0**: `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1`

### 3. Cross-Correlation (XCorr)
```
xcorr_max: max_τ corr(X_t, Y_{t+τ})
xcorr_mean: mean_τ corr(X_t, Y_{t+τ})
```
- **What it captures**: Optimal lag for linear correlation
- **Multi-lag**: Considers all lags τ ∈ [0, T_max], reports max or mean
- **In pilot0**: `xcorr_max_sig-True`, `xcorr_mean_sig-True`

### 4. Granger Causality (GC)
```
GC(X→Y) = log(var(Y_t | Y_{t-1:t-p}) / var(Y_t | Y_{t-1:t-p}, X_{t-1:t-p}))
```
- **What it captures**: Whether X's past improves prediction of Y beyond Y's own past
- **Multi-lag**: Uses autoregressive order p (multiple lags)
- **In pilot0**: `gc_gaussian_k-1_kt-1_l-1_lt-1`

---

## What Would a Custom Multi-Lag Statistic Add?

### Proposed Statistic: Joint Multi-Lag Mutual Information (JMLMI)

**Definition**:
```
JMLMI(X, Y; τ) = I(X_t ; Y_t, Y_{t+1}, ..., Y_{t+τ})
```

**Interpretation**: Mutual information between X at time t and the **joint distribution** of Y at times [t, t+1, ..., t+τ].

**Difference from TLMI**:
- TLMI: I(X_t ; Y_{t+lag}) - pairwise at single lag
- JMLMI: I(X_t ; Y_t:t+τ) - joint over multiple lags

**What it captures**: Whether X_t predicts the **trajectory** of Y over the next τ steps, not just a single future point.

### Alternative: Multi-Step Predictability (MSP)

**Definition**:
```
MSP(X; τ) = I(X_t ; X_{t+1}, X_{t+2}, ..., X_{t+τ})
```

**Interpretation**: How much information X_t contains about its own future τ-step trajectory.

**Difference from autocorrelation**:
- Autocorrelation: Linear, pairwise correlations at each lag
- MSP: Nonlinear, joint dependence over entire trajectory

**What it captures**: "Memory depth" or "predictive horizon" of the process.

### Alternative: Lagged Covariance Determinant (LCD)

**Definition**:
```
LCD(X, Y; τ) = det(Cov([X_t, Y_t, Y_{t+1}, ..., Y_{t+τ}]))
```

**Interpretation**: Volume of the joint covariance ellipsoid (measures linear dependencies).

**What it captures**: Whether adding more lags increases predictability (determinant shrinks if redundant).

---

## Scientific Justification: When Would This Be Useful?

### Hypothesis 1: Multi-Step Memory in Chaotic Systems

**Claim**: Chaotic systems (Lorenz-96, Rössler) have long-range dependencies where X_t predicts X_{t+1:t+τ} better than X_{t+1} alone.

**Test**:
- Compute MSP(X; τ) for τ ∈ {1, 2, 5, 10}
- For Lorenz-96: Expect MSP(τ=1) < MSP(τ=5) (multi-step captures more structure)
- For IID noise: Expect MSP(τ=1) ≈ MSP(τ=5) ≈ 0 (no memory)

**Competing hypothesis**: Transfer Entropy (TE) already captures this via embeddings k, l.

**Critical question**: **Does MSP add information beyond TE?**
- TE: I(Y_t ; X_{t-k} | Y_{t-l}) - conditional on Y's own past
- MSP: I(X_t ; X_{t+1:t+τ}) - joint future trajectory
- **Difference**: TE is **causal** (past→present), MSP is **predictive** (present→future)

**Verdict**: **Potentially useful** if you care about multi-step-ahead forecasting. But TE already captures similar information for causal inference.

---

### Hypothesis 2: Financial Volatility Clustering (GARCH Effects)

**Claim**: Financial returns have multi-lag volatility dependencies where σ²_t predicts σ²_{t+1:t+τ}.

**Test**:
- Compute JMLMI(|r_t|, {|r_{t+1}|, ..., |r_{t+τ}|}) (absolute returns as proxy for volatility)
- For GBM-returns: Expect JMLMI(τ=5) > JMLMI(τ=1) (volatility clusters)
- For Gaussian noise: Expect JMLMI(τ) ≈ 0 for all τ (no clustering)

**Competing hypothesis**: CrossCorrelation (xcorr_max) on squared returns already captures this.

**Critical question**: **Does JMLMI on volatility add value beyond xcorr on r²?**
- xcorr: Linear correlation of r²_t and r²_{t+lag}
- JMLMI: Nonlinear joint information in {r²_t, r²_{t+1}, ..., r²_{t+τ}}

**Verdict**: **Possibly useful** for GARCH-like dynamics. But you'd need to:
1. Transform returns → squared returns or absolute returns
2. Compute JMLMI on transformed series
3. Compare to existing xcorr, MI on transformed series

**Simpler alternative**: Use existing MI on |r_t| and |r_{t+lag}| (already captures nonlinear volatility dependence).

---

### Hypothesis 3: Phase-Space Reconstruction (Takens Embedding)

**Claim**: For deterministic dynamical systems, the embedding {X_t, X_{t+1}, ..., X_{t+τ}} reconstructs the attractor.

**Test**:
- Compute correlation dimension or Lyapunov exponents from embedding
- This is NOT a pairwise SPI - it's a global property of the system

**Competing hypothesis**: Existing SPIs (MI, TE, dCorr) already implicitly use embeddings.

**Critical question**: **Do we need attractor reconstruction, or just pairwise statistics?**
- If goal is classification/clustering: Pairwise SPIs are sufficient (SPI-space fingerprint captures dynamics)
- If goal is dynamical systems analysis: Attractor reconstruction is better (but different research question)

**Verdict**: **Out of scope** for SPI-space analysis. Attractor reconstruction is a different paradigm.

---

## Implementation Sketch: JMLMI

### Python Pseudocode

```python
from sklearn.feature_selection import mutual_info_regression
import numpy as np

def jmlmi(x, y, tau=3, estimator='kraskov'):
    """
    Joint Multi-Lag Mutual Information.
    
    Args:
        x: (T,) array - source variable at time t
        y: (T,) array - target variable at all times
        tau: int - number of future lags to include
        estimator: 'kraskov' or 'gaussian'
    
    Returns:
        float - I(x_t ; y_t, y_{t+1}, ..., y_{t+tau})
    """
    T = len(x)
    
    # Create lagged features: [y_t, y_{t+1}, ..., y_{t+tau}]
    y_lagged = np.column_stack([y[lag:T-tau+lag] for lag in range(tau+1)])
    x_aligned = x[:T-tau]  # Align x to match y_lagged length
    
    if estimator == 'kraskov':
        # Use sklearn's MI estimator (k-NN based, equivalent to Kraskov)
        mi = mutual_info_regression(y_lagged, x_aligned, n_neighbors=4)[0]
    elif estimator == 'gaussian':
        # Gaussian MI: I(X;Y) = -0.5 * log(det(Cov(X,Y)) / (det(Cov(X)) * det(Cov(Y))))
        joint = np.column_stack([x_aligned[:, None], y_lagged])
        cov_joint = np.cov(joint.T)
        cov_x = np.var(x_aligned)
        cov_y = np.cov(y_lagged.T)
        
        det_joint = np.linalg.det(cov_joint)
        det_x = cov_x
        det_y = np.linalg.det(cov_y)
        
        mi = -0.5 * np.log(det_joint / (det_x * det_y))
    else:
        raise ValueError("estimator must be 'kraskov' or 'gaussian'")
    
    return mi


# Example usage on synthetic data
from spimts.generators import gen_lorenz96, gen_gaussian_noise

# Chaotic system (expect high JMLMI)
X_lorenz = gen_lorenz96(M=10, T=3000)
jmlmi_lorenz = np.mean([jmlmi(X_lorenz[:, i], X_lorenz[:, j], tau=5) 
                         for i in range(10) for j in range(10) if i != j])

# IID noise (expect low JMLMI)
X_noise = gen_gaussian_noise(M=10, T=3000)
jmlmi_noise = np.mean([jmlmi(X_noise[:, i], X_noise[:, j], tau=5) 
                        for i in range(10) for j in range(10) if i != j])

print(f"Lorenz-96 JMLMI(tau=5): {jmlmi_lorenz:.4f}")  # Expect > 0.1
print(f"Gaussian noise JMLMI(tau=5): {jmlmi_noise:.4f}")  # Expect ≈ 0
```

### Computational Complexity

**Per pair (i, j)**:
- TLMI: O(T log T) - single-lag MI estimation
- JMLMI: O(T × τ × k) - k-NN search in (τ+1)-dimensional space
- **Scaling**: For τ=5, JMLMI is ~5x slower than TLMI

**For M channels**:
- M(M-1) pairs → total time ~ M² × T × τ
- For M=30, T=2000, τ=5: ~10-20 seconds per model (similar to existing SPIs)

**Verdict**: **Computationally feasible** to add as custom SPI.

---

## Recommended Approach: Validation BEFORE Implementation

### Step 1: Literature Review (Do First)

**Search for**:
- "Multi-lag mutual information time series"
- "Joint mutual information embedding"
- "Predictive information time series"

**Check if**:
- This statistic already exists with a standard name
- Prior work shows it adds value beyond TE/TLMI
- Established estimation methods (better than naive sklearn)

**Expected outcome**: Either:
1. **Exists already**: Use existing implementation (don't reinvent wheel)
2. **Novel idea**: Proceed cautiously (needs theoretical justification)

---

### Step 2: Pilot Test on 3 Generators (Do Second)

**Hypothesis**: JMLMI distinguishes chaotic from stochastic systems better than TLMI.

**Test**:
1. Compute TLMI and JMLMI on:
   - Lorenz-96 (chaotic, long memory)
   - Gaussian noise (IID, no memory)
   - VAR(1) (Markov, lag-1 memory only)

2. Expected patterns:
   | Generator | TLMI(lag=1) | TLMI(lag=5) | JMLMI(tau=5) | Interpretation |
   |-----------|-------------|-------------|--------------|----------------|
   | Lorenz-96 | High | Medium | **Very High** | Multi-lag captures chaos |
   | VAR(1) | High | Low | Medium | Only lag-1 matters |
   | Gaussian | Low | Low | Low | No memory at any lag |

3. **Critical test**: Does JMLMI(tau=5) for Lorenz >> VAR(1)?
   - If YES: JMLMI adds value (proceed to Step 3)
   - If NO: JMLMI is redundant with TLMI (stop here)

---

### Step 3: Add to SPI Framework (Do Third, If Step 2 Succeeds)

**Implementation**:
1. Add `jmlmi()` function to `generators.py` or new `custom_spis.py`
2. Modify `pyspi` config to include custom SPI (if possible)
3. Run on all 15 generators (dev+++ profile)
4. Compute SPI-space: How does JMLMI correlate with existing SPIs?

**Expected SPI-SPI correlations**:
- ρ(JMLMI, TLMI): **Medium-High** (0.5-0.8) - related but not identical
- ρ(JMLMI, TE): **Medium** (0.4-0.7) - both capture multi-step dependencies
- ρ(JMLMI, xcorr_max): **Low-Medium** (0.2-0.5) - nonlinear vs linear

**Scientific value**: If JMLMI shows **low correlation** with existing SPIs (ρ < 0.5), it captures complementary information → worth keeping.

---

## Which Generators Are Most/Least Suitable for JMLMI?

### Most Suitable (High JMLMI Expected)

**1. Lorenz-96 (Chaotic)**
- **Why**: Sensitive dependence on initial conditions → X_t contains information about long future trajectory
- **Expected**: JMLMI(tau=5) >> TLMI(lag=1) (multi-step predictability)

**2. Rössler-coupled (Chaotic)**
- **Why**: Coupled chaotic oscillators → cross-channel trajectories are predictable
- **Expected**: High JMLMI between channels (phase-locking)

**3. GBM-returns (Financial Volatility)**
- **Why**: GARCH-like effects → |r_t| predicts {|r_{t+1}|, ..., |r_{t+5}|} jointly
- **Expected**: JMLMI on absolute returns > MI on single-lag returns

**4. Quadratic-Coupling (Non-monotonic Nonlinear)**
- **Why**: X² dependencies create multi-step correlations
- **Expected**: JMLMI captures quadratic trajectory better than linear xcorr

---

### Least Suitable (Low JMLMI Expected)

**1. Gaussian-Noise (IID)**
- **Why**: No memory → X_t independent of X_{t+τ} for all τ > 0
- **Expected**: JMLMI ≈ 0 for all tau (null hypothesis)

**2. VAR(1) (Markov)**
- **Why**: Only lag-1 dependence → X_t → X_{t+1}, but X_t ⊥ X_{t+2} | X_{t+1}
- **Expected**: JMLMI(tau=1) high, but JMLMI(tau=5) ≈ TLMI(lag=1) (no multi-step gain)

**3. TimeWarp-clones (Near-identical channels)**
- **Why**: X_i ≈ X_j for all i, j → JMLMI is trivial (perfect predictability)
- **Expected**: JMLMI ≈ 1 (not discriminative)

**4. OU-network (Linear, short-memory)**
- **Why**: Ornstein-Uhlenbeck has exponential decay → little long-range memory
- **Expected**: JMLMI(tau=5) ≈ TLMI(lag=1) × exp(-5α) (exponential decay, no multi-step benefit)

---

## Does xcorr_* Already Capture Multi-Lag Structure?

### What xcorr_max and xcorr_mean Do

**xcorr_max**:
```python
def xcorr_max(x, y, max_lag=100):
    correlations = [np.corrcoef(x[:-lag], y[lag:])[0,1] for lag in range(1, max_lag)]
    return np.max(np.abs(correlations))
```
- **Captures**: Best single lag for linear correlation
- **Limitation**: Pairwise correlations, doesn't consider joint structure

**xcorr_mean**:
- **Captures**: Average correlation across all lags
- **Limitation**: Same as xcorr_max

**Difference from JMLMI**:
- xcorr: Linear correlations at individual lags (sum of pairwise)
- JMLMI: Nonlinear joint information over all lags simultaneously

**Analogy**:
- xcorr is like "best match in a sequence of pairs"
- JMLMI is like "information in the entire sequence as a whole"

### Example Where JMLMI ≠ xcorr

**Scenario**: Phase-lagged oscillators with τ = π/2 (quadrature)

```
x_t = sin(ωt)
y_t = sin(ωt + π/2) = cos(ωt)
```

- **xcorr(x, y) at lag=0**: 0 (orthogonal)
- **xcorr_max(x, y)**: 1 (at lag=π/2ω)
- **JMLMI(x; y, y_{t+1}, ..., y_{t+τ})**: **High** (x predicts entire phase trajectory of y)

**Why difference?**: xcorr finds best single lag, but JMLMI captures that x predicts the **shape** of y's future trajectory (sinusoidal), not just a single future point.

**Verdict**: **JMLMI is complementary** to xcorr (captures joint structure, not just pairwise).

---

## Theoretical Framework: Information-Theoretic Memory Measures

### Predictive Information (Bialek et al., 2001)

**Definition**:
```
PI(X; τ) = I(X_{-∞:t} ; X_{t+1:t+τ})
```
**Interpretation**: Information about future τ steps contained in entire past.

**Relation to MSP**:
- MSP: I(X_t ; X_{t+1:t+τ}) - finite past (single time step)
- PI: I(X_{-∞:t} ; X_{t+1:t+τ}) - infinite past

**Practical**: MSP is a finite-memory approximation to PI.

### Excess Entropy (Crutchfield & Feldman, 2003)

**Definition**:
```
E(X) = lim_{L→∞} I(X_{t:t+L} ; X_{t-L:t})
```
**Interpretation**: Total predictable information (mutual information between infinite past and infinite future).

**Relation to JMLMI**: JMLMI(tau) is a finite-L approximation to excess entropy.

### Transfer Entropy Generalization

**Standard TE**:
```
TE(X→Y) = I(Y_t ; X_{t-k} | Y_{t-l})
```

**Multi-lag TE** (hypothetical):
```
TE_multi(X→Y; τ) = I(Y_t:t+τ ; X_{t-k} | Y_{t-l})
```
**Interpretation**: Information flow from X's past to Y's **future trajectory** (not just Y_t).

**Novel?**: This may not exist in standard literature - could be a research contribution if it proves useful.

---

## Conclusion: Should You Implement JMLMI?

### Arguments FOR Implementation

1. **Theoretical completeness**: Existing SPIs (TLMI, TE, xcorr) are pairwise; JMLMI is joint
2. **Potential novelty**: Multi-lag joint information may not be well-studied in MTS literature
3. **Complementary information**: Likely low correlation with existing SPIs (new axis in SPI-space)
4. **Chaotic systems**: May better distinguish chaotic (long memory) from Markov (short memory) processes

### Arguments AGAINST Implementation

1. **Unproven value**: No evidence (yet) that JMLMI improves classification/clustering
2. **Existing alternatives**: TE with higher embeddings (k, l > 1) may already capture multi-lag structure
3. **Complexity**: Requires validation on pilot generators before committing to full implementation
4. **Diminishing returns**: Adding one more SPI unlikely to change SPI-space structure significantly

---

### **RECOMMENDED PATH FORWARD**

**Phase 1**: **DEFER** until baseline analysis completes
- Finish visualization, case studies, qualitative analysis first
- Understand SPI-space structure with existing 20 SPIs

**Phase 2**: **PILOT TEST** on 3 generators (if time permits)
- Implement JMLMI(tau=5) on Lorenz-96, VAR(1), Gaussian noise
- Check if JMLMI(Lorenz) >> JMLMI(VAR) >> JMLMI(noise)
- If YES: Proceed to Phase 3
- If NO: Abandon (TE/TLMI already sufficient)

**Phase 3**: **FULL INTEGRATION** (only if pilot succeeds)
- Add JMLMI to custom SPI set
- Run on all 15 generators
- Analyze SPI-space: ρ(JMLMI, other SPIs) < 0.5?
- If low correlation: Keep (adds value)
- If high correlation: Discard (redundant)

**Phase 4**: **PUBLICATION** (if JMLMI proves useful)
- Write up multi-lag information theory
- Show that JMLMI improves model classification
- Contribute to time series analysis literature

---

### Final Verdict

**xcorr_* does NOT fully capture multi-lag structure** (only pairwise linear correlations).

**JMLMI is theoretically distinct** (joint nonlinear information over trajectory).

**But**: Unclear if practical value justifies implementation effort.

**Recommendation**: **Start with pilot test on 3 generators AFTER baseline work completes.** If pilot shows promise, proceed. If not, existing SPIs are likely sufficient.

