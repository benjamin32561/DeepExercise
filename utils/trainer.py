"""
Trainer class for Siamese Neural Network.
Handles single training run with given hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from tqdm import tqdm

from models.siamese_network import count_parameters


class Trainer:
    """
    Trainer class for Siamese Network.
    Handles a single training run with specified hyperparameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        experiment_dir: Path
    ):
        """
        Initialize trainer.
        
        Args:
            model: Siamese network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on (cuda/cpu)
            config: Configuration dictionary with hyperparameters
            experiment_dir: Directory to save experiment results
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 3),
            verbose=False
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.best_epoch = 0
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for img1, img2, labels in self.train_loader:
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            outputs = self.model(img1, img2)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> tuple:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, labels in self.val_loader:
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int, verbose: bool = True) -> dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with final results
        """
        if verbose:
            print(f"\nTraining with config: {self.config['name']}")
            print(f"Model parameters: {count_parameters(self.model):,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress (every 3 epochs or at milestones)
            if verbose and (epoch % 3 == 0 or epoch == 1 or epoch == num_epochs):
                improvement = "✓" if self.epochs_without_improvement == 0 else f"({self.epochs_without_improvement})"
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                      f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                      f"LR: {current_lr:.6f} {improvement}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Track best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"Training complete! Time: {total_time/60:.1f}min | "
                  f"Best Val Acc: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        
        # Save final results
        results = {
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch,
            'total_time': total_time,
            'final_train_acc': self.history['train_acc'][-1],
            'final_train_loss': self.history['train_loss'][-1],
            'history': self.history
        }
        
        self.save_results(results)
        self.plot_training_curves()
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.experiment_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
    
    def save_results(self, results: dict):
        """Save training results to JSON."""
        # Remove non-serializable items
        results_copy = results.copy()
        results_copy['history']['epoch_times'] = [float(t) for t in results_copy['history']['epoch_times']]
        
        results_path = self.experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=4)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.5, label='Best')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[1].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.5, label='Best')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

