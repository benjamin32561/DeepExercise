# Siamese Neural Network Architecture

## Overview

This document provides a complete description of the Siamese Neural Network architecture implemented for facial recognition using one-shot learning, based on Koch et al. (2015) ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

## Architecture Details

### Network Structure

The Siamese Network consists of two identical subnetworks (twins) that share the same weights and parameters. Each subnetwork processes one input image and produces a feature embedding. The similarity between two images is then computed by comparing their embeddings.

```
Input Image 1 (105×105×3)                    Input Image 2 (105×105×3)
         ↓                                            ↓
    [Conv Block 1]                              [Conv Block 1]
         ↓                                            ↓
    [Conv Block 2]                              [Conv Block 2]
         ↓                                            ↓
    [Conv Block 3]                              [Conv Block 3]
         ↓                                            ↓
    [Conv Block 4]                              [Conv Block 4]
         ↓                                            ↓
    [FC Layer]                                  [FC Layer]
         ↓                                            ↓
    Embedding (4096)                            Embedding (4096)
         ↓                                            ↓
         └──────────────────┬──────────────────────────┘
                            ↓
                      L1 Distance
                            ↓
                    [FC + Sigmoid]
                            ↓
                  Similarity Score (0-1)
```

### Layer-by-Layer Specification

#### Convolutional Block 1
- **Input**: (batch, 3, 105, 105)
- **Conv2d**: 64 filters, 10×10 kernel, stride=1
- **BatchNorm2d**: 64 channels
- **Activation**: ReLU
- **MaxPool2d**: 2×2 kernel, stride=2
- **Output**: (batch, 64, 48, 48)

#### Convolutional Block 2
- **Input**: (batch, 64, 48, 48)
- **Conv2d**: 128 filters, 7×7 kernel, stride=1
- **BatchNorm2d**: 128 channels
- **Activation**: ReLU
- **MaxPool2d**: 2×2 kernel, stride=2
- **Output**: (batch, 128, 21, 21)

#### Convolutional Block 3
- **Input**: (batch, 128, 21, 21)
- **Conv2d**: 128 filters, 4×4 kernel, stride=1
- **BatchNorm2d**: 128 channels
- **Activation**: ReLU
- **MaxPool2d**: 2×2 kernel, stride=2
- **Output**: (batch, 128, 9, 9)

#### Convolutional Block 4
- **Input**: (batch, 128, 9, 9)
- **Conv2d**: 256 filters, 4×4 kernel, stride=1
- **BatchNorm2d**: 256 channels
- **Activation**: ReLU
- **Output**: (batch, 256, 6, 6)

#### Fully Connected Layer
- **Input**: (batch, 9216) [flattened from 256×6×6]
- **Linear**: 9216 → 4096
- **BatchNorm1d**: 4096 units
- **Activation**: Sigmoid
- **Output**: (batch, 4096) - Feature embedding

#### Distance and Classification
- **L1 Distance**: |embedding1 - embedding2|
- **Linear**: 4096 → 1
- **Activation**: Sigmoid
- **Output**: (batch, 1) - Similarity score [0, 1]

### Total Parameters

**38,973,889 trainable parameters**

Breakdown:
- Conv1: 19,264 parameters
- Conv2: 401,536 parameters
- Conv3: 131,200 parameters
- Conv4: 131,328 parameters
- FC1: 37,752,832 parameters
- FC_out: 4,097 parameters
- BatchNorm layers: ~533,632 parameters

## Hyperparameters and Training Configuration

### Model Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Input Size | 105×105×3 | Standard size for face images, RGB color |
| Batch Normalization | Enabled | Stabilizes training, allows higher learning rates |
| Weight Initialization | He/Kaiming | Optimal for ReLU activations, prevents vanishing gradients |
| Embedding Dimension | 4096 | Large enough to capture complex facial features |

### Training Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rates, faster convergence than SGD |
| **Learning Rate** | 0.0001 | Small LR for deep network, prevents overshooting |
| **Weight Decay** | 0.0001 | L2 regularization to prevent overfitting |
| **Batch Size** | 128 | Balance between memory and gradient stability |
| **Loss Function** | Binary Cross Entropy | Natural choice for binary classification |
| **LR Scheduler** | ReduceLROnPlateau | Reduces LR when validation plateaus |
| **LR Reduction Factor** | 0.5 | Halves learning rate when triggered |
| **LR Patience** | 5 epochs | Wait 5 epochs before reducing LR |
| **Early Stopping** | 10 epochs | Stop if no improvement for 10 epochs |
| **Max Epochs** | 50 | Sufficient for convergence |

### Data Configuration

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Pairs per Epoch** | 10,000 | Large enough for diverse training |
| **Validation Pairs** | 2,000 | Sufficient for reliable validation |
| **Train/Val Split** | Same people, different pairs | Prevents data leakage |
| **Test Set** | Separate people | True one-shot evaluation |

## Data Augmentation Strategy

### Training Augmentation

Applied to training data only (not validation/test):

1. **Random Affine Transformations**
   - Rotation: ±10 degrees
   - Translation: ±10% in x and y
   - Scale: 90% to 110%
   - **Reasoning**: Simulates natural pose variations and camera angles

2. **Random Horizontal Flip**
   - Probability: 50%
   - **Reasoning**: Faces can appear mirrored, increases data diversity

3. **Color Jitter**
   - Brightness: ±20%
   - Contrast: ±20%
   - Saturation: ±20%
   - Hue: ±10%
   - **Reasoning**: Handles different lighting conditions and camera settings

4. **Normalization**
   - Mean: [0.485, 0.456, 0.406] (ImageNet statistics)
   - Std: [0.229, 0.224, 0.225]
   - **Reasoning**: Standardizes input distribution, stabilizes training

### Test/Validation Augmentation

- Only resize and normalization
- No random augmentations
- **Reasoning**: Ensures reproducible evaluation

## Regularization Techniques

### 1. Batch Normalization
- Applied after each convolutional and FC layer
- **Benefits**:
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Acts as regularization
  - Improves gradient flow

### 2. L2 Weight Decay
- Weight decay: 0.0001
- **Benefits**:
  - Prevents weights from growing too large
  - Encourages simpler models
  - Reduces overfitting

### 3. Data Augmentation
- See above section
- **Benefits**:
  - Increases effective dataset size
  - Improves generalization
  - Reduces overfitting

### 4. Early Stopping
- Patience: 10 epochs
- **Benefits**:
  - Prevents overfitting
  - Saves training time
  - Automatically finds optimal stopping point

### 5. Learning Rate Scheduling
- ReduceLROnPlateau with patience=5
- **Benefits**:
  - Fine-tunes model when approaching optimum
  - Helps escape local minima
  - Improves final performance

## Loss Function

### Binary Cross Entropy (BCE) Loss

```
L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

Where:
- y = true label (0 for same person, 1 for different)
- ŷ = predicted similarity score

**Reasoning**:
- Natural choice for binary classification
- Probabilistic interpretation
- Well-behaved gradients
- Widely used and tested

### Alternative: Contrastive Loss

Also implemented but not used by default:

```
L = (1-y) * 0.5 * D² + y * 0.5 * max(margin - D, 0)²
```

Where:
- D = Euclidean distance between embeddings
- margin = minimum distance for dissimilar pairs

**Trade-offs**:
- BCE: Simpler, works directly with similarity scores
- Contrastive: More principled for metric learning, requires tuning margin

## Distance Metric

### L1 Distance (Manhattan Distance)

```
distance = |embedding1 - embedding2|
```

**Reasoning**:
- Used in original paper
- Less sensitive to outliers than L2
- Computationally efficient
- Works well with high-dimensional embeddings

### Alternative: L2 Distance (Euclidean)

```
distance = ||embedding1 - embedding2||₂
```

**Trade-offs**:
- L1: More robust, used in paper
- L2: More common in metric learning, sensitive to magnitude

## Design Choices and Rationale

### 1. Progressive Feature Extraction

**Choice**: Filter depth increases (64 → 128 → 128 → 256) while spatial dimensions decrease

**Rationale**:
- Early layers capture low-level features (edges, textures)
- Later layers capture high-level features (face parts, identity)
- Reduces computational cost while increasing representational power

### 2. Large Initial Kernels

**Choice**: 10×10 kernel in first layer, then 7×7, then 4×4

**Rationale**:
- Large receptive field captures global face structure
- Smaller kernels in later layers capture fine details
- Follows the principle of hierarchical feature learning

### 3. Sigmoid Activation in Embedding Layer

**Choice**: Sigmoid activation after FC layer (not ReLU)

**Rationale**:
- Bounds embedding values to [0, 1]
- Prevents unbounded growth
- Matches the paper's architecture
- Works well with L1 distance

### 4. Shared Weights (Siamese Architecture)

**Choice**: Both towers share identical weights

**Rationale**:
- Ensures symmetric similarity function
- Reduces parameters by half
- Forces network to learn general features
- Essential for one-shot learning

### 5. Batch Size of 128

**Choice**: 128 samples per batch

**Rationale**:
- Large enough for stable gradient estimates
- Small enough to fit in GPU memory
- Balances training speed and convergence
- Standard choice for deep learning

### 6. Adam Optimizer

**Choice**: Adam instead of SGD

**Rationale**:
- Adaptive learning rates per parameter
- Faster convergence than vanilla SGD
- Less sensitive to learning rate choice
- Handles sparse gradients well
- Industry standard for deep learning

### 7. Small Learning Rate (0.0001)

**Choice**: LR = 0.0001

**Rationale**:
- Deep network requires careful optimization
- Prevents overshooting and instability
- Works well with Adam optimizer
- Can be increased with batch normalization

## Evaluation Metrics

### Primary Metrics

1. **Accuracy**: Percentage of correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under ROC curve

### Why These Metrics?

- **Accuracy**: Overall performance measure
- **Precision/Recall**: Handle class imbalance
- **F1-Score**: Single metric combining precision/recall
- **AUC**: Threshold-independent performance measure

## Expected Performance

Based on the paper and similar implementations:

- **Target Accuracy**: 85-95% on LFW test set
- **Training Time**: 2-4 hours on modern GPU
- **Convergence**: 20-30 epochs typically
- **One-Shot Accuracy**: 70-90% depending on task difficulty

## References

1. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition. ICML Deep Learning Workshop.

2. Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.

3. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.

4. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.

