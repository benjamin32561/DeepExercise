import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

from models.siamese_network import SiameseNetwork
from models.siamese_v2 import SiameseNetV2
from models.face_verification_net import FaceVerificationNet
from models.backbone_network import BackboneNetwork, BackboneNetworkWithClassifier
from utils.losses import ContrastiveLoss
from utils.dataloader import create_dataloaders


# ============================================================================
# HARDCODED PATHS - EDIT THESE
# ============================================================================
EXPERIMENTS_DIR = '0outputs/experiments'
TRAIN_DATASET_JSON = '0data/datasets/train_dataset.json'
VAL_DATASET_JSON = '0data/datasets/val_dataset.json'
TEST_DATASET_JSON = '0data/datasets/test_dataset.json'
BATCH_SIZE = 128
NUM_WORKERS = 8  # Reduced to avoid worker termination issues
# ============================================================================


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        config = checkpoint.get('config', {})
        model_state_dict = checkpoint['model_state_dict']
    else:
        config = {}
        model_state_dict = checkpoint
    
    # Get architecture and loss type
    architecture = config.get('architecture', 'siamese')
    loss_type = config.get('loss', 'bce')
    
    # Create model based on config
    if architecture in ['siamese', 'original']:
        model = SiameseNetwork(use_batchnorm=config.get('use_batchnorm', True))
    elif architecture == 'siamese_v2':
        model = SiameseNetV2(
            use_batchnorm=config.get('use_batchnorm', True),
            fc_dropout=config.get('fc_dropout', 0.3)
        )
    elif architecture == 'custom':
        model = FaceVerificationNet(
            embedding_dim=config.get('embedding_dim', 128),
            dropout=config.get('dropout', 0.4)
        )
    elif architecture == 'backbone':
        backbone_name = config.get('backbone_name', 'mobilenet_v3_small')
        if loss_type in ['contrastive', 'cosine', 'triplet']:
            model = BackboneNetwork(
                backbone=backbone_name,
                embedding_dim=config.get('embedding_dim', 128),
                pretrained=config.get('pretrained', True),
                dropout=config.get('dropout', 0.4)
            )
        else:
            model = BackboneNetworkWithClassifier(
                backbone=backbone_name,
                embedding_dim=config.get('embedding_dim', 128),
                pretrained=config.get('pretrained', True),
                dropout=config.get('dropout', 0.4)
            )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model, architecture, loss_type


def evaluate_dataset(model, dataloader, device, loss_type):
    """Evaluate model on a dataset."""
    embedding_losses = ['contrastive', 'cosine', 'triplet']
    use_embeddings = loss_type in embedding_losses
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            if use_embeddings:
                output1, output2 = model(img1, img2)
                distances = F.pairwise_distance(output1, output2)
                predictions = (distances < 1.0).float()
            else:
                outputs = model(img1, img2)
                predictions = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = 100.0 * (all_predictions == all_labels).mean()
    
    # Calculate per-class accuracy
    positive_mask = (all_labels == 1)
    negative_mask = (all_labels == 0)
    pos_acc = 100.0 * (all_predictions[positive_mask] == all_labels[positive_mask]).mean()
    neg_acc = 100.0 * (all_predictions[negative_mask] == all_labels[negative_mask]).mean()
    
    return {
        'accuracy': float(accuracy),
        'positive_accuracy': float(pos_acc),
        'negative_accuracy': float(neg_acc),
        'total_samples': int(len(all_labels)),
        'positive_samples': int(positive_mask.sum()),
        'negative_samples': int(negative_mask.sum())
    }


def create_visualization(experiment_dir, train_results, val_results, test_results):
    """Create grouped bar chart showing overall, positive, and negative accuracy for each split."""
    experiment_dir = Path(experiment_dir)
    
    # Data for the bar chart
    splits = ['Train', 'Val', 'Test']
    results_list = [train_results, val_results, test_results]
    
    overall_accs = [r['accuracy'] for r in results_list]
    positive_accs = [r['positive_accuracy'] for r in results_list]
    negative_accs = [r['negative_accuracy'] for r in results_list]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(splits))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, overall_accs, width, label='Overall', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, positive_accs, width, label='Positive (Same Person)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, negative_accs, width, label='Negative (Different Person)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset Split', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance: Overall, Positive, and Negative Accuracy', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_file = experiment_dir / 'final_evaluation.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def save_summary(experiment_dir, train_results, val_results, test_results):
    """Save text summary of results."""
    experiment_dir = Path(experiment_dir)
    summary_file = experiment_dir / 'final_summary.txt'
    
    # Load training history for best metrics
    results_file = experiment_dir / 'results.json'
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    history = data.get('history', data)
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL MODEL EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {experiment_dir.name}\n\n")
        
        # Performance table
        f.write("PERFORMANCE ON ALL SPLITS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Split':<15} {'Accuracy':<15} {'Pos Acc':<15} {'Neg Acc':<15}\n")
        f.write("-" * 80 + "\n")
        
        for name, res in [('Train', train_results), ('Val', val_results), ('Test', test_results)]:
            f.write(f"{name:<15} {res['accuracy']:<15.2f}% {res['positive_accuracy']:<15.2f}% {res['negative_accuracy']:<15.2f}%\n")
        
        f.write("\n")
        
        # Generalization analysis
        train_test_gap = train_results['accuracy'] - test_results['accuracy']
        f.write("GENERALIZATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train-Test gap: {train_test_gap:+.2f}%\n")
        
        if train_test_gap > 10:
            f.write("⚠️  OVERFITTING DETECTED\n")
        elif train_test_gap < -5:
            f.write("⚠️  UNUSUAL: Test > Train\n")
        else:
            f.write("✓ GOOD GENERALIZATION\n")
        
        f.write("\n")
        
        # Training info
        if 'train_acc' in history:
            best_val_acc = max(history['val_acc'])
            best_val_epoch = history['val_acc'].index(best_val_acc) + 1
            f.write("TRAINING INFO:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_val_epoch})\n")
            f.write(f"Total Epochs: {len(history['train_acc'])}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved: {summary_file}")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Set random seed for reproducibility
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiments_dir = Path(EXPERIMENTS_DIR)
    
    # Iterate through all experiments (skip triplet models - use test_triplet.py for those)
    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        model_path = experiment_dir / 'best_model.pth'
        
        print("=" * 80)
        print("FINAL MODEL EVALUATION")
        print("=" * 80)
        print(f"Experiment: {experiment_dir.name}")
        print(f"Device: {device}")
        print("=" * 80)
        
        # Verify dataset paths
        for name, path in [("Train", TRAIN_DATASET_JSON), ("Val", VAL_DATASET_JSON), ("Test", TEST_DATASET_JSON)]:
            if not Path(path).exists():
                print(f"❌ {name} dataset not found: {path}")
                print(f"   Skipping {experiment_dir.name}")
                continue
        
        # Load model
        print("\nLoading model...")
        model, architecture, loss_type = load_model(model_path, device)
        print(f"  Architecture: {architecture}")
        print(f"  Loss: {loss_type}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        results = {}
        
        for split_name, dataset_json in [
            ("train", TRAIN_DATASET_JSON),
            ("val", VAL_DATASET_JSON),
            ("test", TEST_DATASET_JSON)
        ]:
            # Create temporary dataloader for this split
            loaders = create_dataloaders(
                train_dataset_json=dataset_json,
                val_dataset_json=dataset_json,  # Dummy, won't be used
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS
            )
            dataloader = loaders['train']
            
            print(f"\nEvaluating {split_name.upper()} set...")
            results[split_name] = evaluate_dataset(model, dataloader, device, loss_type)
            print(f"  Accuracy: {results[split_name]['accuracy']:.2f}%")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Split':<15} {'Accuracy':<15} {'Positive':<15} {'Negative':<15}")
        print("-" * 80)
        for split in ['train', 'val', 'test']:
            r = results[split]
            print(f"{split.capitalize():<15} {r['accuracy']:<15.2f}% {r['positive_accuracy']:<15.2f}% {r['negative_accuracy']:<15.2f}%")
        print("=" * 80)
        
        # Save results
        print("\nSaving results...")
        create_visualization(experiment_dir, results['train'], results['val'], results['test'])
        save_summary(experiment_dir, results['train'], results['val'], results['test'])
        
        # Save individual JSON files
        for split in ['train', 'val', 'test']:
            json_file = experiment_dir / f'{split}_results.json'
            with open(json_file, 'w') as f:
                json.dump(results[split], f, indent=2)
            print(f"✓ Saved: {json_file}")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)