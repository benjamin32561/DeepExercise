# Configuration Guide

All training configs and where to find them.

---

## Training Scripts

Two main training scripts:

### `train_triplet.py`
For triplet loss experiments.

**Config location**: Lines 164-214

**Key settings**:
```python
{
    "architecture": "backbone",  # or "custom"
    "backbone_name": "mobilenet_v3_small",
    "embedding_dim": 16,
    "dropout": 0.0,
    
    "triplet_margin": 1.0,
    
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "weight_decay": 0.0005,
    
    "lr_factor": 0.75,
    "lr_patience": 20,
    "num_epochs": 400,
    "early_stopping_patience": 200,
    
    "use_augmentation": True,
}
```

### `train.py`
For pairwise loss experiments (BCE, Focal, Contrastive, Cosine).

**Config location**: Lines 259-316

**Key settings**:
```python
{
    "architecture": "siamese",  # or "siamese_v2", "custom", "backbone"
    "loss": "bce",  # or "focal", "contrastive", "cosine"
    
    "use_batchnorm": True,
    "fc_dropout": 0.0,
    "embedding_dim": 16,
    "dropout": 0.1,
    
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "adam",
    "weight_decay": 0.0005,
    
    "lr_factor": 0.5,
    "lr_patience": 10,
    "num_epochs": 200,
    "early_stopping_patience": 100,
}
```

---

## Output Structure

All experiments save to `0outputs/experiments/` with this structure:

```
0outputs/experiments/
├── backbone_triplet_adam_lr0.001_bs32/
│   ├── config.json          # Full config used
│   ├── results.json         # Training history + metrics
│   ├── best_model.pth       # Best model checkpoint
│   └── training_curves.png  # Loss/accuracy plots
│
└── custom_triplet_adam_lr0.001_bs32/
    └── ...
```

**Naming convention**: `{architecture}_{loss}_{optimizer}_lr{lr}_bs{batch_size}`

---

## Saved Configs

Each experiment saves its config to `config.json` in the output directory.

**Example** (`0outputs/experiments/backbone_triplet_adam_lr0.001_bs32/config.json`):
```json
{
  "architecture": "backbone",
  "backbone_name": "mobilenet_v3_small",
  "embedding_dim": 16,
  "dropout": 0.0,
  "pretrained": true,
  "triplet_margin": 1.0,
  "learning_rate": 0.001,
  "batch_size": 32,
  "optimizer": "adam",
  "weight_decay": 0.0005,
  "lr_factor": 0.75,
  "lr_patience": 20,
  "num_epochs": 400,
  "early_stopping_patience": 200,
  "use_augmentation": true,
  "loss": "triplet"
}
```

---

## Augmentation Pipeline

Defined in both training scripts (same config):

```python
augmentations = [
    K.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), p=0.5),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomPerspective(distortion_scale=0.2, p=0.3),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),
    K.RandomGaussianNoise(mean=0.0, std=0.12, p=0.3),
    K.RandomGrayscale(p=0.3),
]
```

Applied on GPU during training.

---

## Dataset Paths

Hardcoded in training scripts:

```python
"train_dataset": "0data/datasets/train_dataset.json",
"val_dataset": "0data/datasets/val_dataset.json",
```

---

## How to Run

**Triplet Loss**:
```bash
python train_triplet.py
```

**Pairwise Loss**:
```bash
python train.py
```

Edit the config dict in the script before running.

---

## Reproducibility

All experiments use `set_seed(42)` for reproducibility:
- Random seed
- NumPy seed
- PyTorch seed (CPU + GPU)
- cuDNN deterministic mode

**Old experiments** (in `0outputs/leakage_db/`) didn't use fixed seeds - can't reproduce them.

