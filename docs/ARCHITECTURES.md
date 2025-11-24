# Model Architectures

All architectures tested in the experiments.

---

## Backbone Network (Winner)

**What it is**: Pretrained MobileNetV3-Small from ImageNet with custom embedding head.

**Architecture**:
```
Input (3x105x105)
    ↓
MobileNetV3-Small Features (pretrained)
    ↓
AdaptiveAvgPool2d
    ↓
Flatten → 576 features
    ↓
Dropout(0.0)
    ↓
Linear(576 → 16)
    ↓
BatchNorm1d(16)
    ↓
L2 Normalize
    ↓
16D Embedding
```

**Why it won**:
- Pretrained on ImageNet (already knows features)
- Only needs to learn face embeddings
- Small embedding (16D) prevents overfitting
- ~2.5M parameters

**Results**:
- Val: 96.28%
- Train: 95.58%

**Code**: `models/backbone_network.py`

---

## Custom CNN

**What it is**: Lightweight custom CNN built from scratch.

**Architecture**:
```
Input (3x105x105)
    ↓
Conv(3→32, 5x5) + ReLU + MaxPool
    ↓
Conv(32→64, 3x3) + ReLU + MaxPool
    ↓
Conv(64→128, 3x3) + ReLU + MaxPool
    ↓
Flatten
    ↓
Linear(→512) + ReLU + Dropout(0.3)
    ↓
Linear(512→16)
    ↓
L2 Normalize
    ↓
16D Embedding
```

**Why it's worse**:
- Learns from scratch (no pretrained weights)
- Not enough data to learn good features
- ~1M parameters

**Results**:
- Val: 93.62%
- Train: 91.45%

**Code**: `models/face_verification_net.py`

---

## Siamese Network

**What it is**: Classic Siamese architecture from Koch et al. (2015).

**Architecture**:
```
Input (3x105x105)
    ↓
Conv(3→64, 10x10) + BN + ReLU + MaxPool
    ↓
Conv(64→128, 7x7) + BN + ReLU + MaxPool
    ↓
Conv(128→128, 4x4) + BN + ReLU + MaxPool
    ↓
Conv(128→256, 4x4) + BN + ReLU
    ↓
Flatten → 9216 features
    ↓
Linear(9216→4096) + BN
    ↓
L1 Distance between pairs
    ↓
Linear(4096→1) + Sigmoid
    ↓
Similarity Score (0-1)
```

**Why it failed**:
- Learns from scratch
- Huge FC layer (4096) overfits on small data
- Only works with pairwise losses (BCE/Focal)
- ~38M parameters (way too big)

**Results**:
- Val: 59.39% (with BCE)
- Train: 75.94%

**Code**: `models/siamese_network.py`

---

## SiameseV2

**What it is**: Lightweight version of Siamese Network.

**Architecture**:
```
Input (3x105x105)
    ↓
Conv(3→32, 10x10) + BN + ReLU + MaxPool
    ↓
Conv(32→64, 7x7) + BN + ReLU + MaxPool
    ↓
Conv(64→64, 4x4) + BN + ReLU + MaxPool
    ↓
Conv(64→128, 4x4) + BN + ReLU
    ↓
Flatten → 4608 features
    ↓
Linear(4608→512) + BN + Dropout(0.3)
    ↓
L1 Distance between pairs
    ↓
Linear(512→1) + Sigmoid
    ↓
Similarity Score (0-1)
```

**Why it's better than Siamese but still bad**:
- Smaller (2M params vs 38M)
- Less overfitting
- But still learns from scratch
- Still only works with pairwise losses

**Results**:
- Val: 61.52% (with Focal)
- Train: 74.39%

**Code**: `models/siamese_v2.py`

---

## Summary

| Architecture | Type | Params | Val Acc | Pretrained |
|--------------|------|--------|---------|------------|
| **Backbone** | Transfer Learning | 2.5M | **96.28%** | ✅ Yes |
| Custom | From Scratch | 1M | 93.62% | ❌ No |
| SiameseV2 | From Scratch | 2M | 61.52% | ❌ No |
| Siamese | From Scratch | 38M | 59.39% | ❌ No |

**Key insight**: Pretrained backbone crushes everything. Small data + scratch training = failure.

---

## For Triplet Loss

Only **Backbone** and **Custom** work with triplet loss. They have `forward_once()` method that outputs embeddings.

Siamese models output similarity scores, not embeddings. They're designed for pairwise losses only.

