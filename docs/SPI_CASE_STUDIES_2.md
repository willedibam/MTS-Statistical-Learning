# Case Studies in SPI Divergence and Concordance

## A Pedagogical Exploration of Statistical Assumptions via Pairwise Dependency Structures

---

## Preface: What a Statistic Assumes

The statistics we use to quantify relationships between time series are not interchangeable gadgets; each one encodes a specific view of what “dependence” is. Pearson’s correlation assumes linear covariation with finite second moments; Spearman’s rank correlation assumes a monotone order-preserving mapping; mutual information (MI) assumes only measurability but pays an estimation price. Comparing **one** statistic to another is therefore not redundant—**the way statistics agree or disagree** is often more diagnostic than any single value in isolation.

The central idea here is simple and useful: **the relationships between statistics themselves reveal the character of dependencies**. Where two measures concur, the assumptions they share are likely satisfied; where they diverge, a specific assumption is being violated. The following five case studies make that logic explicit.

> **Practical note (for later simulations):** Wherever MI or transfer entropy (TE) appear, inference depends critically on estimator choice, embedding, and finite-sample effects. We will be explicit about these when we move to code; here, we focus on qualitative, assumption-level signatures.

---

## Case Study I — Linear, Monotone, and General Dependence

### *Pearson’s (r), Spearman’s (\rho), and Mutual Information (I)*

**Hierarchy.** Linear ⇒ monotone ⇒ statistically dependent; converses need not hold.

**Gaussian exemplar.** For bivariate normal pairs, ($I(X;Y) = -\tfrac12\log(1-r^2))$ is a strictly increasing function of ($|r|$). Under elliptical/normal settings, Spearman’s (\rho) is a monotone transform of (r). Consequently:

* Ranking by (I) aligns with ranking by (|r|) (or (r^2)), not necessarily by signed (r) when both signs are present.
* (\rho) tracks (r) closely when assumptions are met.

**Signature.** In Gaussian/elliptical regimes with light tails, ((r,\rho,I)) co-vary strongly (after handling the sign issue for (I)). Discrepancies are primarily sampling noise.

**Heavy-tailed counterexample (e.g., Cauchy margins).**
Pearson’s (r) is moment-based and can be unstable or misleading with infinite-variance margins. Rank-based (\rho) remains well-defined and captures monotone coupling. MI can (in principle) capture dependence without finite moments, but **estimation is fragile** unless you work in copula space (rank-transform margins) or use robust estimators.

**Signature.** Under heavy-tailed monotone coupling, expect (\rho) and (copula) (I) to agree; (r) can decorrelate from both.

**Non-monotone dependence (e.g., quadratic).**
If (Y=\beta X^2+\epsilon) with symmetric (X), then (r\approx 0) and (\rho) is weak—yet (I) is substantial (uncertainty is reduced despite nonlinearity and non-monotonicity).

**Signature.** (I) informative while (r,\rho) are not—classic “dependence beyond correlation.”

---

## Case Study II — Directed vs Symmetric Dependence

### *Transfer Entropy (TE), Mutual Information, and Cross-Correlation*

**Definitions.**

* MI is symmetric: (I(X;Y)=I(Y;X)).
* Cross-correlation (XCorr) at optimal lag reports maximal linear alignment but is symmetric if you take the maximum over positive **and** negative lags.
* TE is directional and time-aware:
  [
  \mathrm{TE}(X!\to!Y) ;=; I!\big(Y_t;, X_{t-}^{(k)} ,\big|, Y_{t-}^{(l)}\big),
  ]
  with vector pasts (X_{t-}^{(k)}=(X_{t-1},\dots,X_{t-k})), (Y_{t-}^{(l)}) similarly. Embedding matters.

**Symmetric coupling (e.g., globally coupled oscillators).**
(\mathrm{TE}(X!\to!Y)\approx \mathrm{TE}(Y!\to!X)); MI high; XCorr peaks at small lags.
**Signature.** Broad agreement across MI, XCorr, and (approximately symmetric) TE—direction adds little.

**Unidirectional cascades (e.g., Lorenz–96 nearest-neighbour).**
Forward TE is large, backward TE small; MI remains symmetric; XCorr often peaks at a lag whose **sign** accords with direction.
**Signature.** TE asymmetry highlights directionality that MI cannot; lag sign in XCorr supports (but does not prove) direction.

**Time-warped clones.**
Two distinct scenarios:

1. **Common-driver clones:** (X) and (Y) are independent realizations of a common generator or share a latent driver; MI and XCorr (after lag) can be high, but (Y)’s past already predicts (Y), so TE from (X) to (Y) can be low.
2. **Deterministic warp:** (Y(t)=X(\phi(t))). With suitable embeddings, TE can be high because (X)’s past contains unique predictive information for (Y).
   **Signature.** Disentangle these by model and embedding; do not assume “high MI, low TE” without checking which case you are in.

---

## Case Study III — Temporal Misalignment

### *Dynamic Time Warping (DTW), Euclidean Distance, and Cross-Correlation*

**Synchronized dynamics.** Euclidean distance small, XCorr high at lag 0, DTW distance small.
**Signature.** Measures largely redundant.

**Fixed-lag offsets.** Euclidean distance inflates; XCorr finds the lag; DTW aligns equally well.
**Signature.** XCorr and DTW agree; Euclidean dissents.

**Nonlinear time warps.** Single fixed lag cannot align signals whose local speed varies; DTW can, at the cost of potential over-alignment.
**Signature.** DTW small while Euclidean large; XCorr intermediate. Regularization (windowing/penalties) is essential to avoid aligning noise.

---

## Case Study IV — Temporal Optimization vs Instantaneous Ranks

### *XCorr at Optimal Lag vs Spearman (\rho) and Kendall (\tau)*

**Instantaneous coupling.** With no meaningful lags, (R_{\max}) (XCorr at optimal lag) occurs at zero-lag and tracks (\rho); (\rho) and (\tau) are typically strongly concordant in continuous, tie-free settings.

**Phase-lagged oscillators.** Instantaneous (\rho,\tau) can be modest while (R_{\max}) reaches 1.0 after shifting by the phase lag.
**Signature.** (R_{\max}) diverges upward from instantaneous ranks; (\rho) and (\tau) remain concordant with each other.

**Mixed regimes (e.g., chaotic coupling).** Moderate improvements from lag optimization; partial agreement between (R_{\max}) and ranks.
**Note.** Spearman vs Kendall agreement is high in many continuous settings but not “universal”—ties/discreteness and some copulas break this near-identity.

---

## Case Study V — Temporal Stability of Dependencies

### *Sliding-Window Correlation Triples ({r_t, r_{t+\Delta}, r_{t+2\Delta}})*

**Stationary systems.** Triples lie near the diagonal (r_t\approx r_{t+\Delta}\approx r_{t+2\Delta}).

**Periodic modulation.** Triples trace a loop as phase advances; pairwise correlations are moderate but highly structured (high predictability conditional on phase).

**Regime shifts/bifurcations.** Triples move from one cluster to another with a transitional cloud; ordering (r_{t_1} < r_{t_2} < r_{t_3}) may hold while pairwise correlations between times are low.

**Stochastic drift.** Scattered triples with limited predictability.

> **Caveat.** The geometry of these embeddings reflects correlation dynamics, not the dynamical system’s attractor dimension. Periodic data often produce loop-like structure; chaotic data can look higher-dimensional, but there is no general equality with Lyapunov spectra.

---

## Appendix — The Temporal Correlation Manifold

Extending to (K) time points embeds correlation trajectories in (\mathbb{R}^K). Periodic systems often produce loop-like manifolds; quasi-periodic systems can look toroidal; drifting systems smear. Topological summaries (e.g., persistent homology) can be informative, but their interpretation must be tied to windowing choices and uncertainty quantification.

---

## Synthesis: Reading Assumptions from Disagreements

Across these cases, a small set of **qualitative signatures** reoccur:

* **Light-tailed linear/elliptical:** (r,\rho) align; (I) aligns with (|r|).
* **Heavy-tailed monotone:** (\rho) and copula-based (I) align; (r) can fail.
* **Non-monotone nonlinear:** (I) informative; (r,\rho) near zero.
* **Directed cascades:** TE asymmetry large; MI symmetric; XCorr lag sign supports direction.
* **Temporal misalignment:** DTW ≪ Euclidean; XCorr helps for fixed lags only.
* **Lag vs ranks:** (R_{\max}) can exceed instantaneous (\rho,\tau) in phase-lagged systems.
* **Nonstationarity:** correlation triples reveal loops (periodic), transitions (regime shifts), or drift.

The point is not to crown a single statistic, but to **learn from their pattern of concordance and divergence**. That pattern is a compact, interpretable fingerprint of which assumptions hold—and which do not.

---

## Implementation Notes (for when we simulate)

* **Preprocessing:** either z-score each channel or work in copula space (rank-transform margins) when comparing to MI. Handle sign explicitly when pairing (I) with (r) (use (|r|) or stratify by sign).
* **MI:** prefer copula MI (Gaussian-copula entropy or kNN on ranks) for tail-robustness; report estimator settings and uncertainty.
* **TE:** specify embeddings ((k,l)), lag grid, estimator (e.g., kNN/IDTxl), and surrogate testing; TE is meaningless without this.
* **XCorr:** report both the maximal value **and** the argmax lag (with sign). Consider prewhitening where appropriate.
* **DTW:** constrain warping (Sakoe–Chiba band), penalize path complexity, and report path statistics to detect over-alignment.
* **Windows:** justify window length ((w)), step, and taper; provide bootstrap CIs over pairs/windows.
* **Cross-system comparison:** use Fisher-z for correlation summaries; or compare full joint clouds (e.g., Earth Mover’s Distance) rather than only a single inter-statistic correlation.

---
