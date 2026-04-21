# Autoencoder Latent Space Logging Guide

## Motivation

Tracking the numerical rank of the latent representation matrix `Z = Encoder(X)` alone discards most of the interesting information about *how* the autoencoder learns to use its latent dimensions. The quantities below give a much richer picture of rank-by-rank learning dynamics.

All quantities are computed over a **fixed reference batch** `X_ref` at regular intervals during training.

---

## 1. Full Singular Value Spectrum

**What:** The complete vector of singular values `σ₁ ≥ σ₂ ≥ … ≥ σ_d` of the latent activation matrix `Z`.

**Why:** The spectrum is the single most informative quantity to log. It lets you:

- See *how* rank emerges: does dimension k turn on suddenly (phase transition) or gradually (smooth growth)?
- Distinguish a sharp rank-k matrix (k large singular values, rest near zero) from a soft one (smooth decay with no clear cutoff).
- Watch the **gap structure** — large jumps between consecutive singular values reveal the natural dimensionality the network is choosing.
- Retroactively apply any rank threshold or effective-rank measure without re-running the experiment.

**How:**

```python
spectra = torch.linalg.svdvals(Z)  # shape: (min(batch, d_latent),)
```

**Visualization:** A heatmap with training step on the x-axis, singular value index on the y-axis, and color representing magnitude. This produces a "waterfall" view of the spectrum evolving over time.

---

## 2. Scale of the Latent Matrix

**What:** Global norms of `Z` that summarize its overall magnitude.

**Why:** Two latent matrices can both be rank-5, but one might concentrate most of its energy in a single direction while the other spreads it evenly. Tracking scale provides essential context for interpreting rank changes — for instance, a rank increase caused by singular values growing from zero is very different from one caused by a threshold artifact while the spectrum stays fixed.

**How:**

| Metric | Code | Interpretation |
|---|---|---|
| Frobenius norm | `torch.linalg.norm(Z, 'fro')` | Total energy in the representation |
| Spectral norm | `torch.linalg.svdvals(Z)[0]` | Energy in the dominant direction |
| Nuclear norm | `torch.linalg.norm(Z, 'nuc')` | Sum of singular values; penalizes low-rank |

---

## 3. Effective Rank

**What:** A smooth, threshold-free scalar summary of dimensionality derived from the entropy of the normalized singular value distribution (Roy & Bhattacharya, 2007).

**Why:** Hard rank (count of singular values above ε) is sensitive to the choice of ε and jumps discretely. Effective rank varies continuously from 1 (all energy in one direction) to d (perfectly uniform spectrum), making it much easier to track trends and correlate with loss dynamics.

**How:**

```python
s = torch.linalg.svdvals(Z)
p = s / s.sum()
effective_rank = torch.exp(-(p * p.log()).sum())
```

---

## 4. PCA-Latent Alignment

**What:** The correlation structure between the principal components of the *input data* `X` and the latent activations `Z`.

**Why:** The SVD of `Z` tells you what the encoder *is doing*; the PCA alignment tells you *what input structure it has learned to capture*. Tracking this over training reveals the order in which the encoder locks onto input modes — for example, whether it captures the top variance direction first and then progressively adds finer ones, or whether it grabs several simultaneously.

**How:**

```python
# Once, at initialization:
U, S_x, Vt = torch.linalg.svd(X_ref - X_ref.mean(0), full_matrices=False)
pca_scores = U * S_x  # shape: (batch, d_input)

# At each logging step:
Z = encoder(X_ref)
cross_corr = torch.corrcoef(torch.cat([pca_scores.T, Z.T], dim=0))
alignment = cross_corr[:pca_scores.shape[1], pca_scores.shape[1]:]
# shape: (n_input_pcs, d_latent)
```

**Visualization:** Animate or tile the alignment matrix as a heatmap at successive training steps. Strong off-diagonal entries reveal which input PCs map to which latent dimensions, and when those mappings form.

---

## Summary Table

| Quantity | Shape per step | Key question it answers |
|---|---|---|
| Singular value spectrum | `(d,)` | How is dimensionality structured? |
| Frobenius / spectral / nuclear norm | scalar | Is the representation growing or shrinking? |
| Effective rank | scalar | How many dimensions are meaningfully used? |
| PCA-latent alignment matrix | `(n_pcs, d_latent)` | Which input modes has the encoder captured? |

---

## Practical Notes

- **Use a fixed reference batch.** Recomputing over random batches introduces sampling noise that obscures the training signal.
- **Log every N steps, not every step.** SVD is O(min(b,d)·max(b,d)²); for large batches or latent dimensions, logging every 50–500 steps is usually sufficient.
- **Store raw spectra.** Derived quantities (rank, effective rank, condition number) can always be recomputed; the raw spectrum cannot be recovered from summaries.
- **Center Z before SVD if appropriate.** Whether to subtract the mean of `Z` across the batch depends on whether you care about the variance structure (centered) or the full signal including the mean (uncentered). For PCA-style analysis, center it.
