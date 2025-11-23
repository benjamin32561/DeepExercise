import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from models.face_verification_net import FaceVerificationNet
from models.backbone_network import BackboneNetwork
from utils.dataset import SiameseDataset


# ============================================================================
# HARDCODED PATHS - EDIT THESE
# ============================================================================
EXPERIMENTS_DIR = '0outputs/experiments'
TRAIN_DATASET_JSON = '0data/datasets/train_dataset.json'
VAL_DATASET_JSON = '0data/datasets/val_dataset.json'
TEST_DATASET_JSON = '0data/datasets/test_dataset.json'
BATCH_SIZE = 128
NUM_WORKERS = 8
# ============================================================================


def load_model(checkpoint_path, device):
    """
    Load triplet-trained model from checkpoint.
    Returns None, None if architecture is not supported.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    architecture = config.get('architecture')
    
    # Only support architectures with forward_once method
    if architecture not in ['custom', 'backbone']:
        print(f"  ⚠️  Skipping unsupported architecture: {architecture}")
        print(f"      (Triplet evaluation requires 'custom' or 'backbone' models)")
        return None, None
    
    # Create model
    if architecture == 'custom':
        model = FaceVerificationNet(
            embedding_dim=config.get('embedding_dim', 128),
            dropout=config.get('dropout', 0.4)
        )
    elif architecture == 'backbone':
        backbone_name = config.get('backbone_name', 'mobilenet_v3_small')
        model = BackboneNetwork(
            backbone=backbone_name,
            embedding_dim=config.get('embedding_dim', 128),
            pretrained=config.get('pretrained', True),
            dropout=config.get('dropout', 0.4)
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def compute_distances(model, dataloader, device):
    """
    Compute all pairwise distances and labels.
    
    Returns:
        distances: numpy array of distances
        labels: numpy array of labels (1=same person, 0=different)
    """
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Computing distances", leave=False):
            img1, img2 = img1.to(device), img2.to(device)
            
            # Get embeddings
            emb1 = model.forward_once(img1)
            emb2 = model.forward_once(img2)
            
            # Normalize and compute distance
            emb1_norm = F.normalize(emb1, p=2, dim=1)
            emb2_norm = F.normalize(emb2, p=2, dim=1)
            
            distances = F.pairwise_distance(emb1_norm, emb2_norm)
            
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_distances), np.array(all_labels)


def find_optimal_threshold(distances, labels):
    """
    Find optimal threshold using ROC curve.
    
    For distance metrics:
    - Low distance = same person (positive)
    - High distance = different person (negative)
    
    Returns:
        optimal_threshold: Best threshold
        best_accuracy: Accuracy at optimal threshold
        fpr: False positive rate array
        tpr: True positive rate array
        roc_auc: Area under ROC curve
    """
    # For distance: we need to invert for ROC (since lower distance = positive)
    # So we use negative distances as scores
    scores = -distances
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold that maximizes accuracy
    accuracies = []
    for thresh in thresholds:
        predictions = (scores >= thresh).astype(int)
        accuracy = (predictions == labels).mean()
        accuracies.append(accuracy)
    
    best_idx = np.argmax(accuracies)
    optimal_threshold_score = thresholds[best_idx]
    optimal_threshold_distance = -optimal_threshold_score  # Convert back to distance
    best_accuracy = accuracies[best_idx]
    
    return optimal_threshold_distance, best_accuracy, fpr, tpr, roc_auc


def evaluate_with_threshold(distances, labels, threshold):
    """Evaluate accuracy with a given threshold."""
    # Predict: same person if distance < threshold
    predictions = (distances < threshold).astype(int)
    
    # Overall accuracy
    accuracy = 100.0 * (predictions == labels).mean()
    
    # Per-class accuracy
    positive_mask = (labels == 1)
    negative_mask = (labels == 0)
    
    pos_correct = ((predictions == labels) & positive_mask).sum()
    pos_total = positive_mask.sum()
    pos_acc = 100.0 * pos_correct / pos_total if pos_total > 0 else 0
    
    neg_correct = ((predictions == labels) & negative_mask).sum()
    neg_total = negative_mask.sum()
    neg_acc = 100.0 * neg_correct / neg_total if neg_total > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'positive_accuracy': float(pos_acc),
        'negative_accuracy': float(neg_acc),
        'threshold': float(threshold),
        'total_samples': int(len(labels)),
        'positive_samples': int(pos_total),
        'negative_samples': int(neg_total),
        'correct': int((predictions == labels).sum()),
        'positive_correct': int(pos_correct),
        'negative_correct': int(neg_correct)
    }


def create_roc_visualization(experiment_dir, val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc):
    """Create ROC curve visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curves
    ax.plot(val_fpr, val_tpr, 'b-', linewidth=2, label=f'Validation (AUC = {val_auc:.4f})')
    ax.plot(test_fpr, test_tpr, 'r-', linewidth=2, label=f'Test (AUC = {test_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Triplet Model', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_distance_distribution(experiment_dir, train_dist, train_labels, val_dist, val_labels, 
                                 test_dist, test_labels, optimal_threshold):
    """Create distance distribution visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    splits = ['Train', 'Val', 'Test']
    distances_list = [train_dist, val_dist, test_dist]
    labels_list = [train_labels, val_labels, test_labels]
    
    for ax, split, distances, labels in zip(axes, splits, distances_list, labels_list):
        # Separate distances by label
        same_person_dist = distances[labels == 1]
        diff_person_dist = distances[labels == 0]
        
        # Plot histograms
        ax.hist(same_person_dist, bins=50, alpha=0.6, color='green', label='Same Person', density=True)
        ax.hist(diff_person_dist, bins=50, alpha=0.6, color='red', label='Different Person', density=True)
        ax.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2, 
                  label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        ax.set_xlabel('Distance', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{split} Set', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'distance_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_results_visualization(experiment_dir, train_results, val_results, test_results):
    """Create bar chart of results."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    splits = ['Train', 'Val', 'Test']
    results_list = [train_results, val_results, test_results]
    
    overall_accs = [r['accuracy'] for r in results_list]
    positive_accs = [r['positive_accuracy'] for r in results_list]
    negative_accs = [r['negative_accuracy'] for r in results_list]
    
    x = np.arange(len(splits))
    width = 0.25
    
    bars1 = ax.bar(x - width, overall_accs, width, label='Overall', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, positive_accs, width, label='Positive (Same Person)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, negative_accs, width, label='Negative (Different Person)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset Split', fontsize=14, fontweight='bold')
    ax.set_title(f'Triplet Model Performance (Optimal Threshold = {test_results["threshold"]:.3f})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(experiment_dir / 'triplet_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_summary(experiment_dir, train_results, val_results, test_results, config):
    """Save text summary of results."""
    summary_path = experiment_dir / 'triplet_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRIPLET MODEL EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Experiment: {experiment_dir.name}\n")
        f.write(f"Architecture: {config.get('architecture')}\n")
        f.write(f"Backbone: {config.get('backbone_name', 'N/A')}\n")
        f.write(f"Embedding Dim: {config.get('embedding_dim')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS WITH OPTIMAL THRESHOLD\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Optimal Threshold: {test_results['threshold']:.4f}\n\n")
        
        f.write(f"{'Split':<15} {'Overall':<15} {'Same Person':<15} {'Different Person':<15}\n")
        f.write("-"*80 + "\n")
        
        for split_name, results in [('Train', train_results), ('Val', val_results), ('Test', test_results)]:
            f.write(f"{split_name:<15} "
                   f"{results['accuracy']:>7.2f}%      "
                   f"{results['positive_accuracy']:>7.2f}%        "
                   f"{results['negative_accuracy']:>7.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for split_name, results in [('Train', train_results), ('Val', val_results), ('Test', test_results)]:
            f.write(f"{split_name} Set:\n")
            f.write(f"  Total Samples: {results['total_samples']}\n")
            f.write(f"  Correct: {results['correct']} / {results['total_samples']}\n")
            f.write(f"  Same Person: {results['positive_correct']} / {results['positive_samples']} "
                   f"({results['positive_accuracy']:.2f}%)\n")
            f.write(f"  Different Person: {results['negative_correct']} / {results['negative_samples']} "
                   f"({results['negative_accuracy']:.2f}%)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved: {summary_path}")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main testing function."""
    # Set random seed for reproducibility
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiments_dir = Path(EXPERIMENTS_DIR)
    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        model_path = experiment_dir / 'best_model.pth'
    
        # Skip if no model file
        if not model_path.exists():
            continue
            
        print("="*80)
        print("TRIPLET MODEL EVALUATION")
        print("="*80)
        print(f"Experiment: {experiment_dir.name}")
        print(f"Device: {device}")
        print("="*80)
        print()
        
        # Load model
        print("Loading model...")
        model, config = load_model(model_path, device)
            
        # Skip if model couldn't be loaded
        if model is None:
            print()
            continue
            
        print(f"  Architecture: {config.get('architecture')}")
        print(f"  Backbone: {config.get('backbone_name', 'N/A')}")
        print()
        
        # Create dataloaders with ImageNet normalization (used everywhere in codebase)
        print("Creating dataloaders...")
        
        # Use ImageNet normalization (consistent across all datasets)
        test_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = SiameseDataset(TRAIN_DATASET_JSON)
        train_dataset.transform = test_transform  # Override default transform
        
        val_dataset = SiameseDataset(VAL_DATASET_JSON)
        val_dataset.transform = test_transform
        
        test_dataset = SiameseDataset(TEST_DATASET_JSON)
        test_dataset.transform = test_transform
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        print()
        
        # Compute distances for all splits
        print("Computing distances...")
        train_distances, train_labels = compute_distances(model, train_loader, device)
        val_distances, val_labels = compute_distances(model, val_loader, device)
        test_distances, test_labels = compute_distances(model, test_loader, device)
        print()
        
        # Find optimal threshold on validation set
        print("Finding optimal threshold on validation set...")
        optimal_threshold, best_val_acc, val_fpr, val_tpr, val_auc = find_optimal_threshold(val_distances, val_labels)
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Validation accuracy: {best_val_acc*100:.2f}%")
        print(f"  Validation AUC: {val_auc:.4f}")
        print()
        
        # Compute test AUC
        _, _, test_fpr, test_tpr, test_auc = find_optimal_threshold(test_distances, test_labels)
        print(f"  Test AUC: {test_auc:.4f}")
        print()
        
        # Evaluate all splits with optimal threshold
        print("Evaluating with optimal threshold...")
        train_results = evaluate_with_threshold(train_distances, train_labels, optimal_threshold)
        val_results = evaluate_with_threshold(val_distances, val_labels, optimal_threshold)
        test_results = evaluate_with_threshold(test_distances, test_labels, optimal_threshold)
        print()
        
        # Print results
        print("="*80)
        print("RESULTS")
        print("="*80)
        print(f"{'Split':<15} {'Overall':<15} {'Same Person':<15} {'Different Person':<15}")
        print("-"*80)
        for split_name, results in [('Train', train_results), ('Val', val_results), ('Test', test_results)]:
            print(f"{split_name:<15} "
                f"{results['accuracy']:>7.2f}%      "
                f"{results['positive_accuracy']:>7.2f}%        "
                f"{results['negative_accuracy']:>7.2f}%")
        print("="*80)
        print()
        
        # Save results
        print("Saving results...")
        
        # Save JSON results
        for split_name, results in [('train', train_results), ('val', val_results), ('test', test_results)]:
            with open(experiment_dir / f'{split_name}_triplet_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        # Create visualizations
        create_roc_visualization(experiment_dir, val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc)
        print(f"✓ Saved: {experiment_dir / 'roc_curve.png'}")
        
        create_distance_distribution(experiment_dir, train_distances, train_labels,
                                    val_distances, val_labels, test_distances, test_labels,
                                    optimal_threshold)
        print(f"✓ Saved: {experiment_dir / 'distance_distributions.png'}")
        
        create_results_visualization(experiment_dir, train_results, val_results, test_results)
        print(f"✓ Saved: {experiment_dir / 'triplet_evaluation.png'}")
        
        save_summary(experiment_dir, train_results, val_results, test_results, config)
        
        print()
        print("="*80)
        print("EVALUATION COMPLETE")
        print("="*80)


if __name__ == '__main__':
    main()

