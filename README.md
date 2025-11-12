# Facial Recognition Using One-Shot Learning

Siamese Neural Network implementation for facial recognition based on [Koch et al. (2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

## Dataset

**LFW-a (Labeled Faces in the Wild)**
- Train: 788 people, 4,441 images, 1,100 pairs
- Test: 353 people, 1,939 images, 500 pairs
- No person overlap between train/test

## Model Architecture

**Siamese CNN with 38.9M parameters:**
- Conv Block 1: 64 filters, 10×10, ReLU, MaxPool
- Conv Block 2: 128 filters, 7×7, ReLU, MaxPool
- Conv Block 3: 128 filters, 4×4, ReLU, MaxPool
- Conv Block 4: 256 filters, 4×4, ReLU
- FC Layer: 9,216 → 4,096, Sigmoid
- L1 Distance + FC → Similarity score

**Key Features:**
- Batch normalization for stability
- Data augmentation (affine, flip, color jitter)
- Adam optimizer (LR=0.0001)
- Binary Cross Entropy loss
- Early stopping + LR scheduling

## Quick Start

### 1. Install Dependencies

```bash
conda activate de_env
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow tqdm
```

### 2. Train with Hyperparameter Search

```bash
python train.py
```

**What it does:**
- Runs 10 experiments with random hyperparameter search
- **Parallel training**: 2 models train simultaneously on GPU (2x faster!)
- Tests different: learning rates, batch sizes, optimizers, weight decay, etc.
- Automatically selects best model based on validation accuracy
- **~2-3 hours on GPU** for all experiments (with parallel training)

**Outputs:**
- `experiments_TIMESTAMP/` - All experiment results
  - `exp_001/`, `exp_002/`, ... - Individual experiments
  - `summary.json` - All results ranked by performance
  - `comparison.png` - Hyperparameter analysis plots
- `checkpoints/best_model.pth` - Best model (auto-selected)

### 3. Evaluate

```bash
python evaluate.py
```

**Outputs in `results/`:**
- ROC curve + AUC
- Confusion matrix
- Score distributions
- Example predictions

### 4. Visualize Dataset

```bash
python visualize_dataset.py
```

## Project Structure

```
DeepExercise/
├── 0data/                  # Dataset
│   ├── lfw2/              # Images (gitignored)
│   ├── train.txt          # Train pairs
│   └── test.txt           # Test pairs
├── models/
│   └── siamese_network.py # Model architecture
├── utils/
│   ├── dataset_loader.py  # Memory-efficient loader
│   └── dataset.py         # PyTorch datasets
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── visualize_dataset.py   # Dataset analysis
```

## Hyperparameter Search Space

The training script automatically searches over:

| Parameter | Search Range | Type |
|-----------|--------------|------|
| Learning Rate | **1e-4 to 5e-3** | Log-uniform |
| Batch Size | [64, 128, 256] | Choice |
| Weight Decay | 1e-5 to 1e-3 | Log-uniform |
| Optimizer | [Adam, SGD] | Choice |
| Pairs/Epoch | **[2K, 4K, 6K]** | Choice |
| Batch Norm | [True, False] | Choice |

**Fixed Parameters:**
- Loss: Binary Cross Entropy
- Distance: L1 (Manhattan)
- Augmentation: Affine + Color Jitter + Flip
- Early Stopping: 10 epochs patience
- LR Scheduling: ReduceLROnPlateau

## Expected Performance

- **Accuracy:** 85-95%
- **AUC:** 0.90-0.95
- **Training Time:** 2-4 hours (GPU)
- **Convergence:** 20-30 epochs

## Usage Examples

### Adjust Parallel Training

Edit in `train.py`:
```python
n_experiments = 10  # Number of experiments
n_parallel = 2      # Run 2 at once (increase if you have more GPU memory)
                    # 3-4 parallel for 24GB+ GPU
```

### Modify Search Space

Edit `search_space` dict in `train.py`:
```python
search_space = {
    'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
    'batch_size': {'type': 'choice', 'values': [64, 128, 256]},
    # Add or modify parameters
}
```

### Get Embeddings

```python
from models.siamese_network import SiameseNetwork
import torch

model = SiameseNetwork(input_channels=3, use_batchnorm=True)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

embedding = model.get_embedding(image)  # Returns 4096-dim vector
```

## References

- Koch et al. (2015) - [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- Huang et al. (2007) - Labeled Faces in the Wild Database
