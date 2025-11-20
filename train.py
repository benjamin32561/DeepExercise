import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import kornia.augmentation as K

from models.siamese_network import SiameseNetwork
from models.siamese_v2 import SiameseNetV2
from models.face_verification_net import FaceVerificationNet
from models.backbone_network import BackboneNetwork, BackboneNetworkWithClassifier
from utils.losses import ContrastiveLoss, FocalLoss, CosineEmbeddingLoss
from utils.dataloader import create_dataloaders


def train_epoch(model, train_loader, optimizer, criterion, device, augmentation=None, loss_type='bce'):
    """Train for one epoch with universal loss support."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Determine if loss needs embeddings or scores
    embedding_losses = ['contrastive', 'cosine']
    score_losses = ['bce', 'focal']
    
    pbar = tqdm(train_loader, desc="  Training", leave=False, dynamic_ncols=True, position=1, file=None)
    
    for img1, img2, labels in pbar:
        # Move to device
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        labels_orig = labels.float().to(device, non_blocking=True)
        
        # Apply augmentation on GPU
        if augmentation is not None:
            img1 = augmentation(img1)
            img2 = augmentation(img2)
            # Clamp to valid range [0, 1] after augmentation
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)
        
        # Forward
        optimizer.zero_grad()
        
        # Get model outputs based on what loss needs
        if loss_type in embedding_losses:
            # Loss needs embeddings
            if hasattr(model, 'get_embeddings'):
                embedding1, embedding2 = model.get_embeddings(img1, img2)
            else:
                # Model outputs embeddings by default (custom/backbone)
                outputs = model(img1, img2)
                if isinstance(outputs, tuple):
                    embedding1, embedding2 = outputs
                else:
                    # Shouldn't happen but handle gracefully
                    raise ValueError(f"Model doesn't output embeddings for {loss_type} loss")
            
            # Compute loss
            loss = criterion(embedding1, embedding2, labels_orig)
            
            # Calculate accuracy based on distance
            distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
            predictions = (distances < 1.0).float()
            correct += (predictions == labels_orig).sum().item()
            
        else:  # score_losses
            # Loss needs similarity scores
            outputs = model(img1, img2)
            if isinstance(outputs, tuple):
                # Model outputs embeddings, but we need scores
                # This shouldn't happen with proper config, but handle it
                embedding1, embedding2 = outputs
                distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
                outputs = torch.sigmoid(-distances + 1.0).unsqueeze(1)
            
            labels_bce = labels_orig.unsqueeze(1) if labels_orig.dim() == 1 else labels_orig
            loss = criterion(outputs, labels_bce)
            
            # Calculate accuracy based on threshold
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels_bce).sum().item()
        
        # Backward
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * img1.size(0)
        total += labels_orig.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })
    
    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, loss_type='bce'):
    """Validate the model with universal loss support."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Determine if loss needs embeddings or scores
    embedding_losses = ['contrastive', 'cosine']
    score_losses = ['bce', 'focal']
    
    pbar = tqdm(val_loader, desc="  Validation", leave=False, dynamic_ncols=True, position=1, file=None)
    
    with torch.no_grad():
        for img1, img2, labels in pbar:
            # Move to device
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels_orig = labels.float().to(device, non_blocking=True)
            
            # Get model outputs based on what loss needs
            if loss_type in embedding_losses:
                # Loss needs embeddings
                if hasattr(model, 'get_embeddings'):
                    embedding1, embedding2 = model.get_embeddings(img1, img2)
                else:
                    outputs = model(img1, img2)
                    if isinstance(outputs, tuple):
                        embedding1, embedding2 = outputs
                    else:
                        raise ValueError(f"Model doesn't output embeddings for {loss_type} loss")
                
                # Compute loss
                loss = criterion(embedding1, embedding2, labels_orig)
                
                # Calculate accuracy based on distance
                distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
                predictions = (distances < 1.0).float()
                correct += (predictions == labels_orig).sum().item()
                
            else:  # score_losses
                # Loss needs similarity scores
                outputs = model(img1, img2)
                if isinstance(outputs, tuple):
                    # Model outputs embeddings, convert to scores
                    embedding1, embedding2 = outputs
                    distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
                    outputs = torch.sigmoid(-distances + 1.0).unsqueeze(1)
                
                labels_bce = labels_orig.unsqueeze(1) if labels_orig.dim() == 1 else labels_orig
                loss = criterion(outputs, labels_bce)
                
                # Calculate accuracy based on threshold
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels_bce).sum().item()
            
            # Metrics
            total_loss += loss.item() * img1.size(0)
            total += labels_orig.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.1f}%'
            })
    
    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def plot_training_curves(history, best_epoch, best_val_acc, output_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Validation Loss', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    ax.axhline(best_val_acc, color='g', linestyle=':', alpha=0.5, label=f'Best Acc ({best_val_acc:.2f}%)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontweight='bold', fontsize=13)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # Epoch time
    ax = axes[1, 1]
    ax.plot(epochs, history['epoch_times'], 'orange', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Time per Epoch', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Training Metrics - Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Configuration
    config = {
        # Paths
        'train_dataset': '0data/datasets/train_dataset.json',
        'val_dataset': '0data/datasets/val_dataset.json',
        'output_base_dir': '0outputs/experiments',  # Experiments will be organized here
        
        # Model Architecture
        # Options: 'siamese', 'siamese_v2', 'custom', 'backbone'
        'architecture': 'siamese', 

        # Loss Function
        # Options: 'bce', 'focal' for siamese/siamese_v2. 'contrastive', 'cosine' for other.
        'loss': 'bce',
        
        # Model-specific parameters
        'use_batchnorm': False,          # For siamese/siamese_v2
        'fc_dropout': 0.3,              # For siamese_v2 (light dropout on FC layer only)
        'embedding_dim': 64,            # For custom/backbone
        'dropout': 0.5,                 # For custom/backbone
        'pretrained': True,             # For backbone models
        'backbone_name': 'mobilenet_v3_small',  # Smaller backbone (~2.5M params vs ResNet18's ~11M)
        
        # Loss-specific parameters
        'contrastive_margin': 2.0,      # For contrastive loss
        'focal_alpha': 0.25,            # For focal loss
        'focal_gamma': 2.0,             # For focal loss
        'cosine_margin': 0.5,           # For cosine embedding loss
        
        # Training
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'sgd',         # Adam is better for custom model
        'momentum': 0.9,             # For SGD (if used)
        'weight_decay': 1e-4,
        
        # Scheduling
        'lr_factor': 0.75,               
        'lr_patience': 15,              
        'num_epochs': 200,
        'early_stopping_patience': 50,
        
        'use_augmentation': True,
        'augmentations': [
            K.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.93, 1.07), p=0.75),
            K.RandomHorizontalFlip(p=0.5),

            # K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=1.0),
            K.RandomGaussianNoise(mean=0.0, std=0.1, p=0.5),
        ],

        # Data loading
        'num_workers': 16,
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment-specific directory
    # Format: architecture_loss_optimizer_lr_bs
    experiment_name = (
        f"{config['architecture']}_"
        f"{config['loss']}_"
        f"{config['optimizer']}_"
        f"lr{config['learning_rate']}_"
        f"bs{config['batch_size']}"
    )
    output_dir = Path(config['output_base_dir']) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    
    # Enable cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("=" * 80)
    print("Siamese Network Training")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    config_to_save = {k: v for k, v in config.items() if k != 'augmentations'}
    config_to_save['augmentations'] = str(config['augmentations']) if config['use_augmentation'] else None
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Create augmentation pipeline and move to GPU
    augmentation = None
    if config['use_augmentation']:
        augmentation = nn.Sequential(*config['augmentations']).to(device)
        print(f"Augmentation pipeline ({len(config['augmentations'])} transforms):")
        for i, aug in enumerate(config['augmentations'], 1):
            print(f"  {i}. {aug.__class__.__name__}")
        print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        train_dataset_json=config['train_dataset'],
        val_dataset_json=config['val_dataset'],
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
    loss_type = config['loss']
    
    # Determine if we need embedding-based model
    embedding_losses = ['contrastive', 'cosine']
    need_embeddings = loss_type in embedding_losses
    
    if config['architecture'] == 'siamese':
        # Original Siamese Network - works with all losses!
        model = SiameseNetwork(use_batchnorm=config['use_batchnorm']).to(device)
    
    elif config['architecture'] == 'siamese_v2':
        # Simplified Siamese V2: Original architecture + light FC dropout
        model = SiameseNetV2(
            use_batchnorm=config.get('use_batchnorm', True),
            fc_dropout=config.get('fc_dropout', 0.3)  # Light dropout for regularization
        ).to(device)
    
    elif config['architecture'] == 'custom':
        # Custom lightweight model - outputs embeddings
        model = FaceVerificationNet(
            embedding_dim=config['embedding_dim'],
            dropout=config['dropout']
        ).to(device)
    
    elif config['architecture'] == 'backbone':
        # Pretrained backbone - works with both score and embedding losses
        # Using MobileNetV3-Small for smaller model size (better for small datasets)
        backbone_name = config.get('backbone_name', 'mobilenet_v3_small')
        
        if need_embeddings or loss_type in ['bce', 'focal']:
            # For embedding losses OR if user wants embeddings with BCE
            model = BackboneNetwork(
                backbone=backbone_name,
                embedding_dim=config['embedding_dim'],
                pretrained=config['pretrained'],
                dropout=config['dropout']
            ).to(device)
        else:
            # Classifier version for score-based losses
            model = BackboneNetworkWithClassifier(
                backbone=backbone_name,
                embedding_dim=config['embedding_dim'],
                pretrained=config['pretrained'],
                dropout=config['dropout']
            ).to(device)
    
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}. Choose from: 'siamese', 'siamese_v2', 'custom', 'backbone'")
    
    print(f"Architecture: {config['architecture']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Setup loss
    loss_type = config['loss']
    print(f"Creating loss: {loss_type}")
    
    if loss_type == 'bce':
        criterion = nn.BCELoss()
        print(f"  Loss: Binary Cross-Entropy")
    
    elif loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=config['contrastive_margin'])
        print(f"  Loss: Contrastive (margin={config['contrastive_margin']})")
    
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        print(f"  Loss: Focal (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    
    elif loss_type == 'cosine':
        criterion = CosineEmbeddingLoss(margin=config['cosine_margin'])
        print(f"  Loss: Cosine Embedding (margin={config['cosine_margin']})")
    
    else:
        raise ValueError(f"Unknown loss: {loss_type}. Choose from: 'bce', 'contrastive', 'focal', 'cosine'")
    
    print()
    
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
    print("Starting Training")
    print("=" * 80)
    print(f"Max epochs: {config['num_epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print()
    
    start_time = time.time()
    
    epoch_pbar = tqdm(range(1, config['num_epochs'] + 1), desc="Epochs", dynamic_ncols=True, position=0, file=None)
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # Train and validate  
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, augmentation, loss_type)
        val_loss, val_acc = validate(model, val_loader, criterion, device, loss_type)
        
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
        
        # Save training curves and results after every epoch (in case of manual stop)
        plot_training_curves(history, best_epoch, best_val_acc, output_dir)
        
        # Save intermediate results
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
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Average time per epoch: {total_time / len(history['train_loss']):.1f}s")
    print("=" * 80)
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(history['train_loss']),
        'total_time': total_time,
        'final_train_acc': history['train_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'history': history,
        'config': config_to_save
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results: {output_dir / 'results.json'}")
    
    # Plot training curves
    plot_training_curves(history, best_epoch, best_val_acc, output_dir)
    
    print("\n" + "=" * 80)
    print("All outputs saved!")
    print("=" * 80)


if __name__ == '__main__':
    main()
