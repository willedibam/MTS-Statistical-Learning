# Machine Learning Strategy for SPI-Space Analysis

**Created:** October 26, 2025  
**Status:** Planning document for future work (Phase 2+)  
**Scope:** NOT immediate priority - baseline visualization and case studies come first

---

## Executive Summary

This document outlines the machine learning approach for analyzing SPI-space fingerprints to:
1. **Classify** MTS generators based on SPI-SPI correlation patterns
2. **Cluster** dynamics into equivalence classes (linear, chaotic, heavy-tail, etc.)
3. **Identify** which SPI relationships are most discriminative

**Critical constraint**: Current dataset (15 dynamical models) is insufficient for deep learning or complex ML. Full-scale ML requires **N > 500 models** (mix of synthetic + real-world data).

---

## Current Dataset Limitations

### Sample Size Analysis

| Dataset | N (models) | p (features) | p/N Ratio | ML Feasibility |
|---------|-----------|--------------|-----------|----------------|
| **Current (pilot0 subset)** | 15 | ~190 (20 SPIs → 190 pairs) | 12.7 | ❌ Severe overfitting risk |
| **With "fast" config (~50 SPIs)** | 15 | ~1,225 (50 choose 2) | 81.7 | ❌ Hopeless without dimensionality reduction |
| **Full config (~250 SPIs)** | 15 | ~31,125 (250 choose 2) | 2,075 | ❌ Completely infeasible |
| **Target dataset (Phase 2)** | >500 | ~190 or ~1,225 | <2.5 or <5 | ✅ Viable with regularization |

**Statistical rule of thumb**:
- **Regression**: Need N ≥ 10p for reliable coefficient estimates
- **Classification**: Need N ≥ 5p to avoid overfitting (with cross-validation)
- **Deep learning**: Need N ≥ 100p for neural networks

**Current status**: With N=15, p=190 → **p/N = 12.7** (inverted ratio). This is a textbook case of the **curse of dimensionality**.

### What's Achievable NOW (N=15)

**Option A: Dimensionality Reduction First** (RECOMMENDED)
1. Apply PCA to fingerprint matrices (flatten 20×20 → 190D vector)
2. Keep first k principal components explaining 90% variance (likely k < 10)
3. Try simple classifiers (Random Forest, SVM) on reduced space
4. **Expected outcome**: Proof-of-concept that SPI-space contains discriminative signal

**Option B: Regularized Models with Feature Selection**
1. Use L1-regularized logistic regression (Lasso) or Elastic Net
2. Forces sparse solutions (selects only informative SPI pairs)
3. Leave-one-out cross-validation (LOOCV) for evaluation
4. **Expected outcome**: Identify 10-20 most important SPI-SPI correlations

**Option C: Qualitative Analysis Only** (ACCEPTABLE)
1. Manual inspection of dendrograms, fingerprints, SPI-space grids
2. Identify clusters by eye (linear vs nonlinear, chaotic vs stochastic)
3. Write scientific narrative based on patterns observed
4. **Expected outcome**: Publishable results without ML (theory-driven)

---

## Phase 2: Expanded Dataset (N > 500)

### Data Collection Plan

**Synthetic Models** (~500 total):
- Current 15 generators × 10 parameter variations each = 150 models
- Add real-world inspired generators (financial volatility, neural spike trains, climate indices) = +100 models
- Perturb existing generators (different coupling strengths, noise levels, M, T) = +250 models

**Real-World MTS** (~500 total):
- Neuroscience: EEG, fMRI, MEG datasets (preprocessed, parcellated)
- Finance: Stock returns, commodity prices, FX rates (multi-asset portfolios)
- Climate: Temperature, precipitation, sea level across stations
- Physiology: ECG, respiration, blood pressure (multi-organ monitoring)
- Industrial: Sensor networks, manufacturing process data

**Data format**: Each model/dataset → one fingerprint matrix (K×K correlation of SPIs)

### Cluster Access Requirements

**Computational bottleneck**: SPI computation (not ML)
- Current: ~20 min/model for 20 SPIs (pilot0)
- Fast config (~50 SPIs): ~1-2 hours/model
- Full config (~250 SPIs): ~days/model (depends on M, T, specific SPIs)

**Cluster specs needed** (for N=500 models):
- **Pilot0 subset (20 SPIs)**: 500 models × 20 min = 10,000 min ≈ **7 days** on single CPU
  - With 50-node cluster: **3-4 hours** (embarrassingly parallel)
- **Fast config (50 SPIs)**: 500 models × 90 min = 45,000 min ≈ **31 days** on single CPU
  - With 50-node cluster: **15 hours**
- **Full config (250 SPIs)**: **Infeasible** on cluster without significant optimization

**Recommendation**: Start with pilot0 or fast config. Full config only if scientifically justified (marginal value of extra SPIs is likely low).

### ML Pipeline (Phase 2)

**Step 1: Feature Engineering** (from fingerprint matrices)
```
Input: N fingerprints (each K×K correlation matrix)
Output: N × p feature vectors

Options:
A. Flatten upper triangle: K(K-1)/2 features (e.g., 190 for K=20)
B. PCA on flattened: Reduce to k components (k=10-50)
C. Graph embeddings: Treat fingerprint as adjacency matrix → graph2vec
D. Summary statistics: Mean, std, skewness of correlation distributions per fingerprint
```

**Step 2: Classification** (predict model type from fingerprint)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Example pipeline
X = load_fingerprints(flatten=True, pca_components=20)  # (500, 20)
y = load_labels()  # (500,) - model names or categories

# Try multiple classifiers
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
    'SVM_RBF': SVC(kernel='rbf', C=1.0, gamma='scale'),
    'LogisticL1': LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Expected accuracy**:
- **Random baseline**: 1/N_classes (e.g., 1/15 = 6.7% for 15 model types)
- **Weak signal**: 20-40% (better than random, but not strong)
- **Moderate signal**: 50-70% (publishable, shows SPI-space captures dynamics)
- **Strong signal**: >80% (Nature-level, validates entire approach)

**Step 3: Clustering** (unsupervised discovery of equivalence classes)
```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Cluster fingerprints
n_clusters = 5  # Hypothesized: linear, chaotic, stochastic, heavy-tail, oscillatory
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluate clustering quality
silhouette = silhouette_score(X, labels)
print(f"Silhouette score: {silhouette:.3f}")  # Higher is better (max 1.0)

# Visualize clusters in 2D (t-SNE or UMAP)
from umap import UMAP
embedding = UMAP(n_components=2).fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10')
```

**Expected outcome**:
- **If clusters separate cleanly**: SPI-space captures fundamental dynamical classes
- **If clusters overlap**: Need more SPIs, or dynamics are not separable by SPI relationships alone

**Step 4: Feature Importance** (which SPI-SPI correlations matter?)
```python
# Train Random Forest and extract feature importances
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Get importances (one per SPI pair)
importances = rf.feature_importances_
spi_pairs = get_spi_pair_names()  # e.g., ["spearmanr_vs_mi_kraskov", ...]

# Sort and report top 20
top_idx = np.argsort(importances)[-20:]
for idx in top_idx:
    print(f"{spi_pairs[idx]}: {importances[idx]:.4f}")
```

**Scientific value**: Identifies which SPI relationships are most discriminative (e.g., "Spearman-MI correlation distinguishes linear from nonlinear systems").

---

## SPI Subset Expansion: Fast Config

### What is "Fast" Config?

Based on `fast_config.yaml`, the "fast" subset includes ~50-60 SPIs:
- **Basic statistics**: Covariance, Precision, SpearmanR, KendallTau, CrossCorrelation (10-15 SPIs)
- **Distance**: PairwiseDistance, DistanceCorrelation, HSIC, Barycenter, Gromov-Wasserstein (5-10 SPIs)
- **Causal**: AdditiveNoiseModel, ConditionalDistributionSimilarity, RECI, IGCI (4-5 SPIs)
- **Information theory**: JointEntropy, ConditionalEntropy, CrossmapEntropy, StochasticInteraction, MutualInfo, TimeLaggedMutualInfo, TransferEntropy (7-10 SPIs)
- **Spectral**: All coherence/phase measures from pilot0, plus DirectedTransferFunction, PartialDirectedCoherence, SpectralGrangerCausality (15-20 SPIs)
- **Misc**: LinearModel, Cointegration, PowerEnvelopeCorrelation (3 SPIs)

**Total**: ~50 SPIs → ~1,225 SPI pairs (50 choose 2)

### Computational Cost Estimate

**Your machine (without cluster)**:
- Pilot0 (20 SPIs): ~20 min/model (current benchmark)
- Fast (~50 SPIs): **~1-2 hours/model** (2.5x more SPIs, but some overlap in computation)
- Full (~250 SPIs): **~days/model** (not recommended without cluster)

**For 15 models**:
- Pilot0: 15 × 20 min = 5 hours total ✅ (already done)
- Fast: 15 × 90 min = 22.5 hours ≈ **1 day** ✅ (feasible overnight run)
- Full: 15 × 1 day = **15 days** ❌ (not practical)

**Recommendation**: Run fast config on current 15 models as **Phase 1.5** (after baseline visualization completes). This gives you ~1,225 features for ML experimentation while waiting for expanded dataset.

### Expected Marginal Value of Fast Config

**Hypothesis**: Adding 30 extra SPIs (fast - pilot0) will improve classification accuracy IF those SPIs capture complementary information.

**Test**:
1. Run pilot0 subset → get classification accuracy A₀
2. Run fast subset → get classification accuracy A_fast
3. If A_fast >> A₀: Extra SPIs are informative (worth the compute time)
4. If A_fast ≈ A₀: Extra SPIs are redundant (pilot0 is sufficient)

**Expected**: Moderate improvement (+5-15% accuracy), but diminishing returns (going to full ~250 SPIs likely adds <5% more).

---

## Dimensionality Reduction Strategies

### Why Dimensionality Reduction?

**Problem**: Even with N=500 models, p=1,225 features (fast config) gives p/N = 2.45 → still risky.

**Solution**: Reduce p to ~50-100 dimensions before classification/clustering.

### Option 1: PCA (Principal Component Analysis)

**Pros**:
- Fast, interpretable (linear combinations of SPI pairs)
- Can plot "loadings" to see which SPI pairs contribute to each PC
- Preserves variance (choose k to explain 90-95% variance)

**Cons**:
- Linear method (may miss nonlinear structure in fingerprints)
- Components are linear mixtures (hard to interpret scientifically)

**Implementation**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep PCs explaining 95% variance
X_reduced = pca.fit_transform(X_flattened)
print(f"Reduced from {X_flattened.shape[1]} to {X_reduced.shape[1]} dimensions")

# Inspect first PC loadings
pc1_loadings = pca.components_[0]
top_pairs = np.argsort(np.abs(pc1_loadings))[-10:]
print("Top 10 SPI pairs for PC1:", [spi_pairs[i] for i in top_pairs])
```

### Option 2: UMAP (Uniform Manifold Approximation and Projection)

**Pros**:
- Nonlinear dimensionality reduction (captures complex manifolds)
- Better for visualization (2D/3D embeddings preserve local + global structure)
- Often outperforms PCA for classification in low dimensions

**Cons**:
- Slower than PCA (still fast for N=500)
- Harder to interpret (no "loadings" like PCA)
- Stochastic (different runs give slightly different embeddings)

**Implementation**:
```python
from umap import UMAP

umap = UMAP(n_components=50, n_neighbors=15, min_dist=0.1)
X_reduced = umap.fit_transform(X_flattened)

# Visualize in 2D
umap_2d = UMAP(n_components=2).fit_transform(X_flattened)
plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, cmap='tab20')
plt.title("UMAP embedding of SPI fingerprints")
```

### Option 3: Autoencoders (Deep Learning)

**Pros**:
- Learn nonlinear embeddings tailored to reconstruction task
- Can incorporate domain knowledge (e.g., symmetry of correlation matrices)

**Cons**:
- Requires N >> 500 (need 5,000+ samples for stable training)
- Overkill for current problem

**Verdict**: Defer to Phase 3 (if dataset reaches N > 1,000).

---

## Timeline and Priorities

### **Phase 1** (Current - Next 2 Weeks)
- ✅ Complete baseline visualization (all 15 models, all plots)
- ✅ Run case studies (5 SPI triplets, targeted models)
- ✅ Analyze results qualitatively (dendrograms, fingerprints, SPI-space grids)
- ✅ Write up theoretical predictions vs observed patterns

**Deliverable**: Scientific narrative of SPI-space structure across dynamics (no ML yet).

### **Phase 1.5** (Optional - Next Month)
- Run fast config (~50 SPIs) on 15 models (~1 day compute)
- Proof-of-concept ML: PCA + Random Forest on pilot0 vs fast
- Compare classification accuracy to assess marginal value of extra SPIs
- **Decision point**: If fast config improves accuracy, plan cluster run for full dataset

**Deliverable**: Preliminary ML results showing feasibility (or lack thereof) of classification.

### **Phase 2** (Months 2-6)
- Collect/generate expanded dataset (N=500 synthetic + real-world)
- Cluster access for parallel SPI computation (fast config)
- Run full ML pipeline: classification, clustering, feature importance
- Identify most discriminative SPI-SPI correlations

**Deliverable**: Full ML analysis + publication-ready results.

### **Phase 3** (Months 6-12+)
- Scale to full config (~250 SPIs) on subset of models (if justified)
- Explore deep learning (autoencoders, graph neural networks on fingerprints)
- Real-world applications (classify unknown MTS datasets)

**Deliverable**: Applied ML tool for MTS analysis.

---

## Key Insights for ML Success

### 1. Sample Size is King
- **N=15**: Can only do proof-of-concept (PCA + simple models)
- **N=500**: Can do reliable classification/clustering
- **N=5,000+**: Can do deep learning

### 2. Feature Engineering Matters More Than Model Choice
- **Bad features + complex model**: Overfits, poor generalization
- **Good features + simple model**: Often works better
- Current fingerprint (flattened correlation matrix) is a good starting point, but consider:
  - Graph-based features (clustering coefficient, modularity of fingerprint as graph)
  - Summary statistics (mean, variance, skewness of correlation distribution)
  - Hierarchical features (dendrogram cut heights, cluster sizes)

### 3. Interpretability > Accuracy (For Science)
- A Random Forest with 85% accuracy but uninterpretable features is less useful than:
- A logistic regression with 75% accuracy that identifies "Spearman-MI correlation predicts chaos"
- **Always prioritize feature importance analysis** over black-box accuracy maximization

### 4. Validation Strategy
- With N=500: Use 5-fold cross-validation (standard)
- With N=15: Use leave-one-out CV (LOOCV) - necessary due to small N
- **Never** train on all data and report training accuracy (that's data leakage)

---

## Appendix: Full Config (~250 SPIs) Feasibility

### Why Full Config is Problematic

**Computational cost**:
- Estimate: ~1 day/model on single CPU (based on complexity of spectral/causal SPIs)
- For N=500: 500 days on single CPU = **1.4 years**
- Even with 50-node cluster: ~10 days of continuous compute

**Statistical cost**:
- 250 SPIs → 31,125 SPI pairs
- With N=500: p/N = 62 (severe overfitting without aggressive regularization)
- Would need dimensionality reduction to ~50 dimensions just to make ML viable

**Scientific cost**:
- Diminishing returns: Pilot0 (20 SPIs) likely captures 70-80% of information
- Fast config (50 SPIs) likely captures 90-95%
- Full config (250 SPIs) might add only 5-10% more information
- **Not worth 50x computation time for 5% gain**

### When to Consider Full Config

**Only if**:
1. You have evidence that fast config is insufficient (classification accuracy plateaus)
2. You have cluster access and can afford 10+ days of compute
3. You're targeting a top-tier journal (Nature, Science) and need to show "exhaustive analysis"

**Otherwise**: Stick with pilot0 or fast config. Diminishing returns principle applies.

---

## Summary Recommendations

**Immediate (Phase 1)**:
1. Finish baseline visualization (in progress)
2. Analyze case studies qualitatively
3. Write scientific narrative without ML

**Short-term (Phase 1.5, optional)**:
1. Run fast config on 15 models (~1 day)
2. Try PCA + Random Forest as proof-of-concept
3. Assess marginal value of extra SPIs

**Medium-term (Phase 2, when ready for serious ML)**:
1. Expand dataset to N=500 (synthetic + real-world)
2. Get cluster access for parallel SPI computation
3. Run full ML pipeline (classification, clustering, feature importance)
4. Publish results

**Long-term (Phase 3, aspirational)**:
1. Scale to N=5,000+ for deep learning
2. Develop applied ML tool for MTS classification
3. Apply to real-world datasets (neuroscience, finance, climate)

**Never do**:
1. Full config (~250 SPIs) without cluster + scientific justification
2. ML on N=15 without dimensionality reduction (guaranteed overfitting)
3. Report training accuracy without cross-validation (data leakage)

