# Facial Recognition Using One-Shot Learning

This project implements a Siamese Neural Network for facial recognition using one-shot learning, based on the paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf).

## Project Goal

Implement a one-shot classification solution that can successfully determine whether two facial images of previously unseen persons belong to the same person, using the Labeled Faces in the Wild (LFW-a) dataset.

## Project Structure

```
DeepExercise/
├── 0data/                       # Dataset directory
│   ├── lfw2/                    # LFW-a images organized by person
│   ├── train.txt                # Training pairs
│   └── test.txt                 # Testing pairs
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── dataset_loader.py        # Memory-efficient dataset loader
├── figures/                     # Generated visualizations
│   ├── dataset_analysis.png
│   └── detailed_distribution.png
├── visualize_dataset.py         # Dataset analysis and visualization
├── analyze_dataset.py           # Dataset statistics script
├── DATASET_SUMMARY.md           # Detailed dataset information
└── README.md                    # This file
```

## Setup

### Requirements

```bash
# Create conda environment (or clone existing one)
conda create --name de_env python=3.10
conda activate de_env

# Install dependencies
pip install numpy matplotlib seaborn pillow torch torchvision
```

### Download Dataset

1. Download LFW-a dataset from [Google Drive](https://drive.google.com/file/d/1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp/view)
2. Extract to `0data/` directory
3. Download train/test splits:
   - [train.txt](https://drive.google.com/file/d/1Ie-8ihDHfS_FmxAq4EMtMZ-PpMdezQB8/view)
   - [test.txt](https://drive.google.com/file/d/11r_8bbGap1skrEzrQtQOu8ZztG6UVisX/view)

## Usage

### Visualize Dataset

```bash
python visualize_dataset.py
```

This will generate comprehensive visualizations in the `figures/` directory showing:
- Dataset size comparisons
- Distribution of images per person
- Cumulative distributions
- Statistics summary

### Dataset Loader

The `LFWDatasetLoader` class provides memory-efficient access to the dataset:

```python
from utils.dataset_loader import LFWDatasetLoader

# Load train dataset
train_loader = LFWDatasetLoader(
    data_dir="/path/to/0data/lfw2",
    split_file="/path/to/0data/train.txt"
)

# Get statistics
stats = train_loader.get_statistics()

# Generate random pairs for training
pairs = train_loader.generate_pairs(num_pairs=100, balanced=True)

# Access specific pair from split file
img1_path, img2_path = train_loader.get_pair_by_index(0)
```

## Implementation Plan

### Phase 1: Data Preparation ✅
- [x] Download and organize LFW-a dataset
- [x] Parse train/test split files
- [x] Create memory-efficient dataset loader
- [x] Analyze dataset statistics
- [x] Generate visualizations

### Phase 2: Model Architecture (Next)
- [ ] Implement Siamese network architecture
- [ ] Design convolutional layers based on paper
- [ ] Implement distance metric (L1/L2)
- [ ] Create loss function (contrastive/triplet loss)

### Phase 3: Training Pipeline
- [ ] Implement data augmentation
- [ ] Create pair generation strategy
- [ ] Set up training loop
- [ ] Implement validation
- [ ] Add checkpointing and logging

### Phase 4: Evaluation
- [ ] Implement evaluation metrics (accuracy, precision, recall)
- [ ] Generate ROC curves
- [ ] Analyze misclassifications
- [ ] Compare with baseline

### Phase 5: Experimentation
- [ ] Hyperparameter tuning
- [ ] Architecture variations
- [ ] Different distance metrics
- [ ] Data augmentation strategies

### Phase 6: Reporting
- [ ] Document experimental setup
- [ ] Record convergence times and final metrics
- [ ] Generate training/validation curves
- [ ] Analyze successful and failed predictions
- [ ] Write final report

## Key Features

1. **Memory Efficient**: Dataset loader stores only file paths, not images
2. **No Data Leakage**: Strict train/test split with no person overlap
3. **Balanced Pairs**: Support for generating balanced same/different person pairs
4. **Comprehensive Analysis**: Detailed statistics and visualizations
5. **Reproducible**: Fixed random seeds and documented parameters

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition. ICML Deep Learning Workshop.
- Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.

## License

This is an educational project for learning purposes.

