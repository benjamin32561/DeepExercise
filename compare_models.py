import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def load_results(results_path):
    """Load training results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def compare_architectures():
    """Compare architecture details."""
    print("=" * 80)
    print("Architecture Comparison")
    print("=" * 80)
    
    # Load models for parameter count
    from models.siamese_network import SiameseNetwork
    from models.face_verification_net import FaceVerificationNetLight
    from models.backbone_network import BackboneNetwork
    
    # Three architectures
    siamese_model = SiameseNetwork(use_batchnorm=True)
    siamese_params = sum(p.numel() for p in siamese_model.parameters())
    
    custom_model = FaceVerificationNetLight()
    custom_params = sum(p.numel() for p in custom_model.parameters())
    
    backbone_model = BackboneNetwork(backbone='resnet18', pretrained=False)
    backbone_params = sum(p.numel() for p in backbone_model.parameters())
    
    print("\nModel Parameters:")
    print(f"  Siamese (Original):              {siamese_params:>12,}  (BCE only)")
    print(f"  Custom (Lightweight):            {custom_params:>12,}  ({(1-custom_params/siamese_params)*100:>5.1f}% reduction, Contrastive only)")
    print(f"  Backbone (ResNet18):             {backbone_params:>12,}  ({(1-backbone_params/siamese_params)*100:>5.1f}% reduction, Both losses)")
    
    print("\nArchitecture Features:")
    print("  Siamese:")
    print("    - Simple CNN with 4 conv blocks")
    print("    - MaxPooling for downsampling")
    print("    - Binary classification output")
    print("    - Works with: BCE Loss")
    
    print("\n  Custom:")
    print("    - Residual connections (ResNet-style)")
    print("    - Squeeze-and-Excitation attention")
    print("    - Depthwise separable convolutions")
    print("    - L2-normalized embeddings")
    print("    - Works with: Contrastive Loss")
    
    print("\n  Backbone:")
    print("    - Pretrained ResNet18 from ImageNet")
    print("    - Transfer learning ready")
    print("    - Flexible output (embeddings or scores)")
    print("    - Works with: Both BCE and Contrastive Loss")
    print("=" * 80)


def compare_results(old_results_path='0outputs/training/results.json',
                   new_results_path='0outputs/training_modern/results.json'):
    """Compare training results."""
    print("\n" + "=" * 80)
    print("Training Results Comparison")
    print("=" * 80)
    
    # Check if files exist
    old_path = Path(old_results_path)
    new_path = Path(new_results_path)
    
    if not old_path.exists():
        print(f"\n⚠️  Original results not found: {old_results_path}")
        print("   Run 'python train.py' first to train the original model.")
    
    if not new_path.exists():
        print(f"\n⚠️  Modern results not found: {new_results_path}")
        print("   Run 'python train_modern.py' first to train the modern model.")
        return
    
    if old_path.exists():
        old_results = load_results(old_results_path)
        print("\nOriginal Model:")
        print(f"  Best Val Accuracy:  {old_results['best_val_acc']:.2f}%")
        print(f"  Best Val Loss:      {old_results['best_val_loss']:.4f}")
        print(f"  Training Epochs:    {old_results['total_epochs']}")
        print(f"  Training Time:      {old_results['total_time_minutes']:.1f} min")
        
        # Calculate overfitting
        history = old_results['history']
        best_idx = old_results['best_epoch'] - 1
        train_acc = history['train_acc'][best_idx]
        val_acc = history['val_acc'][best_idx]
        gap = train_acc - val_acc
        print(f"  Train/Val Gap:      {gap:.2f}%")
    
    if new_path.exists():
        new_results = load_results(new_results_path)
        print("\nModern Model:")
        print(f"  Best Val Accuracy:  {new_results['best_val_acc']:.2f}%")
        print(f"  Best Val Loss:      {new_results['best_val_loss']:.4f}")
        print(f"  Training Epochs:    {new_results['total_epochs']}")
        print(f"  Training Time:      {new_results['total_time_minutes']:.1f} min")
        
        # Calculate overfitting
        history = new_results['history']
        best_idx = new_results['best_epoch'] - 1
        train_acc = history['train_acc'][best_idx]
        val_acc = history['val_acc'][best_idx]
        gap = train_acc - val_acc
        print(f"  Train/Val Gap:      {gap:.2f}%")
        
        # Distance metrics
        if 'val_similar_dist' in history:
            val_sim = history['val_similar_dist'][best_idx]
            val_dis = history['val_dissimilar_dist'][best_idx]
            print(f"\n  Distance Metrics:")
            print(f"    Similar pairs:    {val_sim:.4f}")
            print(f"    Dissimilar pairs: {val_dis:.4f}")
            print(f"    Separation:       {val_dis - val_sim:.4f}")
    
    if old_path.exists() and new_path.exists():
        print("\nImprovement:")
        acc_improvement = new_results['best_val_acc'] - old_results['best_val_acc']
        if acc_improvement > 0:
            print(f"  Accuracy: +{acc_improvement:.2f}% ✓")
        else:
            print(f"  Accuracy: {acc_improvement:.2f}%")
        
        old_gap = old_results['history']['train_acc'][old_results['best_epoch']-1] - old_results['best_val_acc']
        new_gap = new_results['history']['train_acc'][new_results['best_epoch']-1] - new_results['best_val_acc']
        gap_improvement = old_gap - new_gap
        if gap_improvement > 0:
            print(f"  Overfitting: {gap_improvement:.2f}% reduction ✓")
        else:
            print(f"  Overfitting: {gap_improvement:.2f}% (increased)")
    
    print("=" * 80)


def plot_comparison(old_results_path='0outputs/training/results.json',
                   new_results_path='0outputs/training_modern/results.json',
                   output_path='0outputs/figures/model_comparison.png'):
    """Plot side-by-side comparison."""
    old_path = Path(old_results_path)
    new_path = Path(new_results_path)
    
    if not old_path.exists() or not new_path.exists():
        print("\n⚠️  Both models must be trained for comparison plot.")
        return
    
    old_results = load_results(old_results_path)
    new_results = load_results(new_results_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation accuracy comparison
    ax = axes[0]
    old_history = old_results['history']
    new_history = new_results['history']
    
    epochs_old = range(1, len(old_history['val_acc']) + 1)
    epochs_new = range(1, len(new_history['val_acc']) + 1)
    
    ax.plot(epochs_old, old_history['val_acc'], 'b-', label='Original', linewidth=2)
    ax.plot(epochs_new, new_history['val_acc'], 'r-', label='Modern', linewidth=2)
    ax.axhline(y=old_results['best_val_acc'], color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=new_results['best_val_acc'], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overfitting comparison (train-val gap)
    ax = axes[1]
    old_gap = [t - v for t, v in zip(old_history['train_acc'], old_history['val_acc'])]
    new_gap = [t - v for t, v in zip(new_history['train_acc'], new_history['val_acc'])]
    
    ax.plot(epochs_old, old_gap, 'b-', label='Original', linewidth=2)
    ax.plot(epochs_new, new_gap, 'r-', label='Modern', linewidth=2)
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Perfect (no gap)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train-Val Gap (%)', fontsize=12)
    ax.set_title('Overfitting Comparison (Lower = Better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison plot: {output_path}")


if __name__ == '__main__':
    print("\n")
    compare_architectures()
    compare_results()
    
    plot_comparison()
    
    print("\n")

