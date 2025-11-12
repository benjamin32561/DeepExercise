"""
Evaluation script for Siamese Neural Network.

Performs comprehensive evaluation including:
- Accuracy, Precision, Recall, F1-score
- ROC curve and AUC
- Confusion matrix
- Example predictions (correct and incorrect)
- One-shot learning evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import random

from models.siamese_network import SiameseNetwork
from utils.dataset import SiameseTestDataset
from PIL import Image
import torchvision.transforms as transforms


class Evaluator:
    """Evaluator for Siamese Network."""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Siamese network
            test_loader: Test data loader
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        
        # Results storage
        self.all_labels = []
        self.all_predictions = []
        self.all_scores = []
        self.all_pairs = []
    
    def evaluate(self):
        """Run evaluation on test set."""
        print("=" * 80)
        print("Evaluating Model")
        print("=" * 80)
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(self.test_loader, desc='Evaluating'):
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                scores = outputs.cpu().numpy().flatten()
                predictions = (outputs > 0.5).float().cpu().numpy().flatten()
                
                # Store results
                self.all_labels.extend(labels.numpy())
                self.all_predictions.extend(predictions)
                self.all_scores.extend(scores)
                self.all_pairs.extend([(img1[i], img2[i], labels[i].item()) 
                                       for i in range(len(labels))])
        
        # Convert to numpy arrays
        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        self.all_scores = np.array(self.all_scores)
        
        # Calculate metrics
        self.calculate_metrics()
        
        return self.get_results()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics."""
        # Basic metrics
        correct = (self.all_predictions == self.all_labels).sum()
        total = len(self.all_labels)
        accuracy = 100 * correct / total
        
        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC and AUC
        fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'auc': roc_auc,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        # Print results
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print(f"AUC:       {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
        print("=" * 80)
    
    def get_results(self):
        """Get evaluation results."""
        return self.results
    
    def plot_roc_curve(self, save_path: str):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.results['fpr'], self.results['tpr'], 
                color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {self.results["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, save_path: str):
        """Plot confusion matrix."""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Same', 'Different'],
                   yticklabels=['Same', 'Different'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_score_distribution(self, save_path: str):
        """Plot distribution of similarity scores."""
        same_scores = self.all_scores[self.all_labels == 0]
        diff_scores = self.all_scores[self.all_labels == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(same_scores, bins=50, alpha=0.5, label='Same Person', color='green')
        plt.hist(diff_scores, bins=50, alpha=0.5, label='Different Person', color='red')
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved score distribution to {save_path}")
        plt.close()
    
    def visualize_predictions(self, save_dir: str, num_examples: int = 10):
        """
        Visualize example predictions (correct and incorrect).
        
        Args:
            save_dir: Directory to save visualizations
            num_examples: Number of examples to show
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Find correct and incorrect predictions
        correct_indices = np.where(self.all_predictions == self.all_labels)[0]
        incorrect_indices = np.where(self.all_predictions != self.all_labels)[0]
        
        # Sample examples
        if len(correct_indices) > 0:
            correct_samples = random.sample(list(correct_indices), 
                                          min(num_examples, len(correct_indices)))
            self._plot_examples(correct_samples, save_dir / 'correct_predictions.png', 
                              'Correct Predictions')
        
        if len(incorrect_indices) > 0:
            incorrect_samples = random.sample(list(incorrect_indices), 
                                            min(num_examples, len(incorrect_indices)))
            self._plot_examples(incorrect_samples, save_dir / 'incorrect_predictions.png', 
                              'Incorrect Predictions')
    
    def _plot_examples(self, indices, save_path, title):
        """Plot example image pairs."""
        n = len(indices)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
        
        if n == 1:
            axes = axes.reshape(1, -1)
        
        # Denormalization transform
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for i, idx in enumerate(indices):
            img1, img2, true_label = self.all_pairs[idx]
            pred_label = self.all_predictions[idx]
            score = self.all_scores[idx]
            
            # Denormalize images
            img1 = img1.cpu().numpy().transpose(1, 2, 0)
            img1 = std * img1 + mean
            img1 = np.clip(img1, 0, 1)
            
            img2 = img2.cpu().numpy().transpose(1, 2, 0)
            img2 = std * img2 + mean
            img2 = np.clip(img2, 0, 1)
            
            # Plot images
            axes[i, 0].imshow(img1)
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Image 1')
            
            axes[i, 1].imshow(img2)
            axes[i, 1].axis('off')
            axes[i, 1].set_title('Image 2')
            
            # Plot info
            axes[i, 2].axis('off')
            true_text = 'Same' if true_label == 0 else 'Different'
            pred_text = 'Same' if pred_label == 0 else 'Different'
            color = 'green' if true_label == pred_label else 'red'
            
            info_text = f"True: {true_text}\nPred: {pred_text}\nScore: {score:.4f}"
            axes[i, 2].text(0.5, 0.5, info_text, ha='center', va='center',
                          fontsize=12, color=color, weight='bold',
                          transform=axes[i, 2].transAxes)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved example predictions to {save_path}")
        plt.close()


def main():
    """Main evaluation function."""
    # Configuration
    config = {
        'data_dir': '/home/ben/Desktop/Ben/DeepExercise/0data/lfw2',
        'test_split': '/home/ben/Desktop/Ben/DeepExercise/0data/test.txt',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'results_dir': 'results',
        'batch_size': 128,
        'num_workers': 4,
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = SiameseTestDataset(
        data_dir=config['data_dir'],
        split_file=config['test_split']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load model
    print("Loading model...")
    model = SiameseNetwork(input_channels=3, use_batchnorm=True)
    
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, device)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Create results directory
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_roc_curve(results_dir / 'roc_curve.png')
    evaluator.plot_confusion_matrix(results_dir / 'confusion_matrix.png')
    evaluator.plot_score_distribution(results_dir / 'score_distribution.png')
    evaluator.visualize_predictions(results_dir / 'examples', num_examples=10)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

