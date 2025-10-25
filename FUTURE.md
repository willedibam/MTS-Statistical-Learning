# Forward-Looking Research Avenues

**Last Updated:** October 25, 2025  
**Status:** Implementation complete; these are potential next steps for research analysis

---

## Current Capabilities (Implemented)

Your implementation enables:
1. **Systematic SPI selection**: Data-driven identification of which SPIs capture similar vs different information
2. **Cross-model comparison**: Seeing if SPI relationships differ across dynamical systems (Kuramoto vs Lorenz vs VAR)
3. **Targeted case studies**: Testing specific hypotheses with `--spi-subset`

**The critical unanswered question from CONTEXT.md:**
> "For certain spatiotemporal dependency structures in MTS, some SPIs yield similar values; for others, these same SPIs yield very different values."

---

## Testable Hypotheses (with current tools)

**What you can now test that you couldn't before:**

**Q1: Do information-theoretic SPIs cluster together universally?**
- **Test**: Compare fingerprints/dendrograms across all 16 models
- **Look for**: Do MI, TE, TLMI always cluster together, or does clustering depend on dynamical regime (e.g., linear vs nonlinear)?
- **Scientific value**: Determines if information measures are redundant or capture complementary aspects

**Q2: Are correlation-based SPIs redundant?**
- **Test**: Examine SPI-space scatter plots for Spearman vs Pearson vs Kendall
- **Look for**: High ρ (>0.9) across all models = redundant; variable ρ across models = complementary
- **Scientific value**: Justifies using multiple correlation methods or selecting one

**Q3: Do directed SPIs diverge from undirected SPIs in asymmetric systems?**
- **Test**: Use `--spi-subset "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1,mi_kraskov_NN-4"` on symmetric (Kuramoto) vs asymmetric (VAR) models
- **Look for**: Low correlation in symmetric systems (redundant), high correlation in directed systems (complementary)
- **Scientific value**: Validates that directionality matters for specific dynamics

---

## Potential Extensions (Only if Scientifically Justified)

### **Option 1: Dimensionality Reduction in SPI-Space** (HIGH IMPACT)

**What**: Apply PCA/t-SNE/UMAP to the SPI-SPI correlation matrices across all 16 models

**Why**: Identify latent dimensions that explain variance in SPI relationships across diverse dynamics

**Scientific value**: Could reveal that all dynamical systems vary along just 2-3 fundamental axes of "SPI concordance/discordance"

**Implementation**: 
- Post-processing script that loads all `fingerprint` correlation matrices 
- Flatten each K×K fingerprint matrix to a vector (upper triangle)
- Stack 16 models × 3 methods = 48 observations
- Apply sklearn PCA/UMAP and visualize in 2D/3D
- Color points by model type to see if classes separate

**Effort**: Medium (requires aggregating fingerprints across models, then standard sklearn/umap)

**Data requirements**: Current 16 models sufficient for proof-of-concept

---

### **Option 2: Statistical Significance Testing of SPI-SPI Correlations** (MEDIUM IMPACT)

**What**: Bootstrap confidence intervals or permutation tests for SPI-SPI correlations

**Why**: Current correlations (ρ, τ, r) have no uncertainty quantification - are they statistically significant?

**Scientific value**: Distinguish "true" SPI redundancy from spurious correlations due to small M

**Implementation**: 
- Add permutation tests in `plot_spi_space()` or `fingerprint_matrix()`
- For each SPI pair, shuffle one SPI's edge values 1000 times
- Compute null distribution of correlations
- Report p-values alongside observed correlations
- Mark significant correlations (p<0.01) with asterisks in plots

**Effort**: Medium (requires resampling logic, but computation is cheap once MPIs are cached)

**Limitation**: Requires careful handling of spatial correlation structure in MPIs (edges are not independent)

---

### **Option 3: SPI "Redundancy Scoring" for Automatic Subset Selection** (LOW-MEDIUM IMPACT)

**What**: Given a target number N, automatically select N SPIs that maximize information diversity (minimize redundancy)

**Why**: Currently, "core" SPIs are manually chosen via substring matching - data-driven selection would be more principled

**Scientific value**: Could identify minimal SPI sets that capture 90%+ of information across all models

**Implementation**: 
- Greedy algorithm: Start with SPI that has highest avg dissimilarity to all others
- Iteratively add SPI that is most dissimilar to already-selected set
- Or: Hierarchical clustering on fingerprint matrix, pick one representative per cluster
- Output: Recommended "minimal informative subset" of SPIs

**Effort**: Medium (algorithmic complexity, but uses existing correlation matrices)

**Risk**: Algorithm choices (greedy vs clustering, distance metrics) are somewhat arbitrary

---

### **Option 4: Model Classification Using SPI Fingerprints** (HIGH IMPACT IF IT WORKS)

**What**: Train a classifier (e.g., Random Forest, SVM) to predict model type from SPI fingerprint alone

**Why**: Tests the core hypothesis: "SPI-space signatures capture differences between general classes of dynamics"

**Scientific value**: 
- **If it works well (>80% accuracy)**: Validates that your approach truly captures dynamical equivalence classes. **This is a Nature-level result.**
- **If it fails**: Reveals that SPI relationships alone are NOT sufficient to distinguish dynamics (also valuable - means you need additional features)

**Implementation**: 
- Post-processing script that loads all fingerprints
- Create feature vectors: flatten upper triangle of K×K correlation matrix → ~190 features (for K=19)
- Labels: 16 model types (Kuramoto, Lorenz-96, VAR, etc.)
- Train/test split: Leave-one-out cross-validation (small N)
- Try Random Forest, SVM, XGBoost
- Report classification accuracy, confusion matrix, feature importances

**Effort**: Low-Medium (standard ML pipeline, all tools in sklearn)

**Critical test**: If classifier achieves >random accuracy, your SPI-space approach is validated

**Extension**: Use feature importances to identify which SPI-SPI correlations are most discriminative

---

### **Option 5: Nothing** (PERFECTLY ACCEPTABLE)

**What**: Stop here and analyze results by hand

**Why**: You've built a tool to systematically explore SPI relationships. The *scientific discovery* comes from **looking at the plots and understanding patterns**, not from adding more automation.

**Scientific value**: 
- Manual inspection of dendrograms, fingerprints, and SPI-space scatter plots across models will reveal whether certain SPIs consistently cluster
- E.g., "Do MI and Spearman always correlate? Does TE always diverge from MI in directed systems?"
- This is **actual research**, not feature engineering

**Implementation**: None needed

**Effort**: Zero coding, significant intellectual effort analyzing plots

**Recommended**: Do this first before any extensions

---

## Honest Assessment

**Option 5 (Nothing) or Option 4 (Model classification) are the only defensible next steps.**

- **Options 1-3 are incremental refinements** - nice to have, but don't fundamentally advance the research question.
- **Option 4 is high-risk, high-reward** - if SPI fingerprints can classify models, that's a Nature-level result. If they can't, you learn that SPI relationships alone aren't sufficient (also valuable).
- **Option 5 (current state) is already sufficient** to answer your core questions by manually examining the output plots.

**You should NOT add features just to add features.** The current implementation enables the analysis you set out to do. Whether you need extensions depends on what you find in the results.

---

## Cluster-Scale Considerations (M and T sizing)

### **If you had access to a cluster and many more MTS datasets:**

#### **What is the ideal M (number of channels)?**

**Short answer: M ≈ 50-100 is optimal for most statistical learning tasks with SPIs.**

**Reasoning:**
1. **SPI-space dimensionality**: With K SPIs, you have K(K-1)/2 SPI-SPI correlations (features)
   - For K=20 SPIs: ~190 features
   - For K=50 SPIs: ~1,225 features
   
2. **Computational feasibility**: 
   - M=10: Only 45 edges (too sparse for robust correlation estimates)
   - M=50: 1,225 edges (good statistical power for correlations)
   - M=100: 4,950 edges (excellent statistical power, but 4× computational cost)
   - M=500: 124,750 edges (overkill; diminishing returns, computational bottleneck)

3. **Statistical power for SPI correlations**:
   - Each SPI-SPI correlation is computed across M(M-1)/2 edge pairs
   - For M=50: ~1,200 data points per correlation (excellent)
   - For M=10: ~45 data points per correlation (marginal)
   
4. **Curse of dimensionality for clustering/ML**:
   - With M too large, some SPIs may become rank-deficient or numerically unstable
   - M=50-100 balances statistical power with computational tractability

**Recommendation**: 
- **M=50** is the sweet spot for cluster-scale work
- **M=100** if computational resources allow (better statistical power, minimal added value beyond this)
- **M<30** insufficient for robust SPI-SPI correlation estimates
- **M>150** diminishing returns + numerical stability issues for some SPIs

#### **What about T (number of time steps)?**

**Short answer: T ≈ 1,000-2,000 is sufficient for most dynamical systems.**

**Reasoning:**
1. **Stationarity assumption**: Most SPIs assume stationarity (statistical properties don't change over time)
   - For chaotic systems (Lorenz, Rössler): ~500-1,000 time steps captures full attractor
   - For oscillatory systems (Kuramoto, Stuart-Landau): ~10-20 cycles sufficient (depends on frequency)
   - For stochastic systems (VAR, OU): T should be large enough to estimate autocovariance (typically T>500)

2. **Information-theoretic SPIs (MI, TE)**:
   - Require many samples to estimate probability distributions
   - T=1,000 usually sufficient for k-NN estimators (mi_kraskov, te_kraskov)
   - T<500 may lead to underestimation of MI/TE

3. **Diminishing returns beyond T=2,000**:
   - For most systems, dynamics settle into stationary regime within 1,000-2,000 steps
   - Exception: Systems with very slow transients or multiple timescales may need T>5,000

4. **Computational cost**:
   - Most SPIs scale as O(T) or O(T log T)
   - DTW, GC, TE can be O(T²) or worse for large T
   - T=2,000 is good balance of information vs computation

**Exception cases:**
- **Non-stationary systems**: If dynamics evolve over time (e.g., bifurcations, learning), you may need T>10,000 or sliding window analysis
- **Ultra-high-frequency data** (e.g., neural spikes, financial tick data): T can be millions, but you'd typically downsample or use windowing

**Recommendation**:
- **T=1,000** minimum for reliable SPI estimates
- **T=2,000** optimal for most dynamical systems (captures full dynamics without computational waste)
- **T>5,000** only if system has very slow transients or you're explicitly studying non-stationarity
- **T<500** risky (especially for information-theoretic SPIs)

### **Counterintuitive Result: M >> T is often problematic**

If M=500 and T=1,000, you have:
- 124,750 edges (M(M-1)/2)
- Only 1,000 time points to estimate each edge's SPI value
- For correlation-based SPIs (Pearson, Spearman): This is fine
- For information-theoretic SPIs: Requires very large T to avoid bias

**General rule**: 
- **Aim for T ≥ 2M** to ensure sufficient temporal samples per edge
- For M=50: T=1,000 is comfortable
- For M=500: T should be ≥10,000 (but this is impractical for most SPIs)

### **Practical Cluster-Scale Recommendation**

For large-scale SPI analysis across many datasets:
- **Sweet spot: M=50, T=2,000**
- **Aggressive: M=100, T=2,000**
- **Conservative: M=30, T=1,000**
- **Avoid: M>150 (computational bottleneck + numerical issues) or T<500 (unreliable SPI estimates)**

**Your current dev++ profile (M=10-25, T varies) is appropriate for proof-of-concept but underpowered for definitive conclusions.** If you scale up to cluster, target M=50 for all models.

---

## Summary

**Current state**: Implementation is complete and sufficient for answering core research questions.

**Immediate next step**: Manually analyze the 16 dev++ visualizations to see if SPI relationships vary across dynamical systems (Option 5).

**High-impact extension**: Model classification from SPI fingerprints (Option 4) - only worth doing if manual analysis shows promise.

**Cluster-scale guideline**: M=50, T=2,000 is optimal for robust statistical learning with SPIs.

**Do NOT**: Add features without clear scientific motivation (avoid Options 1-3 unless manual analysis reveals specific needs).
