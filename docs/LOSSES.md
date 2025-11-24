# Loss Functions

All loss functions used in the experiments.

---

## Triplet Loss

**What it does**: Forces the model to learn embeddings where similar faces are close and different faces are far.

**How it works**: 
- Takes 3 images: Anchor (A), Positive (P - same person), Negative (N - different person)
- Minimizes: `max(0, dist(A,P) - dist(A,N) + margin)`
- Goal: Make `dist(A,P) < dist(A,N)` by at least `margin`

**Why it won**: Creates a structured embedding space. Can't cheat by learning shortcuts - must learn real face features.

**Hyperparameters**:
- `margin`: 1.0 (how far apart positive and negative should be)

---

## Binary Cross-Entropy (BCE)

**What it does**: Treats verification as binary classification - "same person" or "different person".

**How it works**:
- Takes 2 images, outputs similarity score 0-1
- Loss: `-[y*log(p) + (1-y)*log(1-p)]`
- Where y=1 for same person, y=0 for different

**Why it failed**: Weak signal. Model learns shortcuts (background, lighting) instead of faces. Only asks "same or different?" without forcing structure.

**Result**: Stuck at ~59% validation accuracy.

---

## Focal Loss

**What it does**: Modified BCE that focuses on hard examples.

**How it works**:
- Same as BCE but down-weights easy examples
- Loss: `-α(1-p)^γ * log(p)` for positive class
- Focuses training on misclassified examples

**Why it failed**: Still binary classification. Focusing on hard examples doesn't fix the fundamental problem - weak signal.

**Hyperparameters**:
- `alpha`: 0.25 (weight for positive class)
- `gamma`: 2.0 (focusing parameter)

**Result**: 61.52% validation accuracy (slightly better than BCE, still bad).

---

## Contrastive Loss

**What it does**: Pulls similar pairs together, pushes different pairs apart.

**How it works**:
- Takes 2 images and a label (same/different)
- For same: minimize distance
- For different: maximize distance up to margin
- Loss: `y*dist² + (1-y)*max(0, margin-dist)²`

**Why it failed**: Better than BCE but still pairwise. Massive overfitting (91.60% train, 61.82% val). Needs way more data.

**Hyperparameters**:
- `margin`: 2.0

**Result**: 61.82% validation, but 91.60% training = overfitting.

---

## Cosine Embedding Loss

**What it does**: Uses cosine similarity instead of euclidean distance.

**How it works**:
- Measures angle between embedding vectors
- Loss: `1 - cos(A,B)` for similar, `max(0, cos(A,B) - margin)` for different
- Cosine similarity: `A·B / (||A|| ||B||)`

**Why it failed**: Still pairwise. Same fundamental issues as contrastive loss.

**Hyperparameters**:
- `margin`: 0.5

**Result**: 59.39% validation accuracy.

---

## Summary

| Loss | Type | Val Acc | Problem |
|------|------|---------|---------|
| **Triplet** | Metric Learning | **96.28%** | ✅ Works |
| Contrastive | Pairwise | 61.82% | Overfits |
| Focal | Pairwise | 61.52% | Weak signal |
| BCE | Pairwise | 59.39% | Weak signal |
| Cosine | Pairwise | 59.39% | Weak signal |

**Triplet loss wins** because it forces geometric structure. Pairwise losses just ask "same or different?" which is too weak for small datasets.

