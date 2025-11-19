import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from tqdm import tqdm
import kornia.augmentation as K

from models.face_verification_net import FaceVerificationNetLight
from models.backbone_network import BackboneNetwork
from utils.losses import TripletLoss
from utils.triplet_dataset import create_triplet_dataloaders


def plot_training_curves(history, best_epoch, best_val_acc, output_dir):
    """Plot and save training curves."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Triplet Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Triplet Accuracy (Best: {best_val_acc:.2f}%)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, train_loader, optimizer, criterion, device, augmentation=None):
    """Train for one epoch with triplet loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="  Training", leave=False)
    
    for anchor, positive, negative, _ in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Apply augmentation on GPU
        if augmentation is not None:
            anchor = augmentation(anchor)
            positive = augmentation(positive)
            negative = augmentation(negative)
        
        optimizer.zero_grad()
        
        # Get embeddings
        # Models should have forward_once method that returns embedding for single image
        anchor_emb = model.forward_once(anchor)
        positive_emb = model.forward_once(positive)
        negative_emb = model.forward_once(negative)
        
        # Compute triplet loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy: is distance(anchor, positive) < distance(anchor, negative)?
        with torch.no_grad():
            anchor_emb_norm = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb_norm = F.normalize(positive_emb, p=2, dim=1)
            negative_emb_norm = F.normalize(negative_emb, p=2, dim=1)
            
            pos_dist = F.pairwise_distance(anchor_emb_norm, positive_emb_norm)
            neg_dist = F.pairwise_distance(anchor_emb_norm, negative_emb_norm)
            
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0 * correct / total:.1f}%'})
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate using triplet loss."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="  Validating", leave=False)
        
        for anchor, positive, negative, _ in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Get embeddings
            anchor_emb = model.forward_once(anchor)
            positive_emb = model.forward_once(positive)
            negative_emb = model.forward_once(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            
            # Calculate accuracy
            anchor_emb_norm = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb_norm = F.normalize(positive_emb, p=2, dim=1)
            negative_emb_norm = F.normalize(negative_emb, p=2, dim=1)
            
            pos_dist = F.pairwise_distance(anchor_emb_norm, positive_emb_norm)
            neg_dist = F.pairwise_distance(anchor_emb_norm, negative_emb_norm)
            
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0 * correct / total:.1f}%'})
    
    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    """Main training function for triplet loss."""
    
    config = {
        # Paths
        'train_dataset': '0data/datasets/train_dataset.json',
        'val_dataset': '0data/datasets/val_dataset.json',
        'output_base_dir': '0outputs/experiments',
        
        # Model Architecture
        # Options: 'custom', 'backbone' (siamese doesn't work with triplet - no forward_once method)
        'architecture': 'backbone',
        
        # Model-specific parameters
        'embedding_dim': 128,
        'dropout': 0.5,
        'pretrained': True,  # For backbone
        'backbone_name': 'mobilenet_v3_small',
        
        # Triplet Loss parameters
        'triplet_margin': 1.0,
        
        # Training
        'learning_rate': 0.001,
        'batch_size': 32,  # Smaller for triplets (3x images per sample)
        'optimizer': 'adam',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        
        # Scheduling
        'lr_factor': 0.5,
        'lr_patience': 10,
        'num_epochs': 200,
        'early_stopping_patience': 100,
        
        # Augmentation
        'use_augmentation': True,
        'augmentations': [
            K.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.3),
        ],
        
        # Data loading
        'num_workers': 8,
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment name
    experiment_name = (
        f"{config['architecture']}_"
        f"triplet_"
        f"{config['optimizer']}_"
        f"lr{config['learning_rate']}_"
        f"bs{config['batch_size']}"
    )
    output_dir = Path(config['output_base_dir']) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"TRIPLET LOSS TRAINING: {experiment_name}")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {output_dir}")
    print()
    
    # Save config
    config_to_save = {k: v for k, v in config.items() if k != 'augmentations'}
    config_to_save['augmentations'] = str(config['augmentations']) if config['use_augmentation'] else None
    config_to_save['loss'] = 'triplet'  # Mark as triplet
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Create augmentation
    augmentation = None
    if config['use_augmentation']:
        augmentation = nn.Sequential(*config['augmentations']).to(device)
        print("Augmentation pipeline:")
        for i, aug in enumerate(config['augmentations'], 1):
            print(f"  {i}. {aug.__class__.__name__}")
        print()
    
    # Create triplet dataloaders
    print("Creating triplet dataloaders...")
    loaders = create_triplet_dataloaders(
        train_json=config['train_dataset'],
        val_json=config['val_dataset'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    if config['architecture'] == 'custom':
        model = FaceVerificationNetLight(
            embedding_dim=config['embedding_dim'],
            dropout=config['dropout']
        ).to(device)
    elif config['architecture'] == 'backbone':
        model = BackboneNetwork(
            backbone=config['backbone_name'],
            embedding_dim=config['embedding_dim'],
            pretrained=config['pretrained'],
            dropout=config['dropout']
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")
    
    print(f"Architecture: {config['architecture']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Setup triplet loss
    criterion = TripletLoss(margin=config['triplet_margin'])
    print(f"Loss: Triplet (margin={config['triplet_margin']})")
    print()
    
    # Optimizer
    if config['optimizer'].lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=config['learning_rate'], 
                        weight_decay=config['weight_decay'])
    else:
        optimizer = SGD(model.parameters(), lr=config['learning_rate'], 
                       momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_factor'], 
        patience=config['lr_patience'], verbose=True
    )
    
    # Training tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Training loop
    print("=" * 80)
    print("Starting Triplet Training")
    print("=" * 80)
    print(f"Max epochs: {config['num_epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print()
    
    start_time = time.time()
    
    epoch_pbar = tqdm(range(1, config['num_epochs'] + 1), desc="Epochs", ncols=120)
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, augmentation)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'TrL': f'{train_loss:.3f}',
            'TrA': f'{train_acc:.1f}%',
            'VaL': f'{val_loss:.3f}',
            'VaA': f'{val_acc:.1f}%',
            'Best': f'{best_val_acc:.1f}%',
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config_to_save,
            }, output_dir / 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        # Save training curves and results after every epoch
        plot_training_curves(history, best_epoch, best_val_acc, output_dir)
        
        intermediate_results = {
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_epochs': len(history['train_loss']),
            'current_epoch': epoch,
            'final_train_acc': history['train_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'history': history,
            'config': config_to_save
        }
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(intermediate_results, f, indent=2)
        
        # Early stopping
        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"\n\nEarly stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n\n{'=' * 80}")
    print("Triplet Training Complete!")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Average time per epoch: {total_time / len(history['train_loss']):.1f}s")
    print("=" * 80)
    
    print(f"\nSaved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

