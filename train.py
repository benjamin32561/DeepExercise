"""
Training script with hyperparameter search.
Runs multiple experiments with different hyperparameters using random search.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from models.siamese_network import SiameseNetwork
from utils.dataset import SiameseLFWDataset
from utils.trainer import Trainer


class HyperparameterSearch:
    """
    Hyperparameter search using random search strategy.
    """
    
    def __init__(self, base_config: dict, search_space: dict, n_experiments: int = 10):
        """
        Initialize hyperparameter search.
        
        Args:
            base_config: Base configuration with fixed parameters
            search_space: Dictionary defining search space for each hyperparameter
            n_experiments: Number of random experiments to run
        """
        self.base_config = base_config
        self.search_space = search_space
        self.n_experiments = n_experiments
        self.results = []
        
        # Create experiments directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiments_dir = Path(f"experiments_{timestamp}")
        self.experiments_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Experiments will be saved to: {self.experiments_dir}")
    
    def sample_config(self, experiment_id: int) -> dict:
        """
        Sample a random configuration from search space.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Configuration dictionary
        """
        config = self.base_config.copy()
        config['experiment_id'] = experiment_id
        config['name'] = f"exp_{experiment_id:03d}"
        
        # Sample from search space
        for param, space in self.search_space.items():
            if space['type'] == 'choice':
                config[param] = random.choice(space['values'])
            elif space['type'] == 'uniform':
                config[param] = random.uniform(space['low'], space['high'])
            elif space['type'] == 'loguniform':
                config[param] = 10 ** random.uniform(np.log10(space['low']), np.log10(space['high']))
            elif space['type'] == 'int':
                config[param] = random.randint(space['low'], space['high'])
        
        return config
    
    def run_experiment(self, config: dict, device: torch.device) -> dict:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Experiment configuration
            device: Device to train on
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*80)
        print(f"Experiment {config['experiment_id']}/{self.n_experiments}")
        print("="*80)
        print(f"Config: LR={config['learning_rate']:.6f}, BS={config['batch_size']}, "
              f"WD={config['weight_decay']:.6f}, Opt={config['optimizer']}")
        
        # Create datasets
        train_dataset = SiameseLFWDataset(
            data_dir=config['data_dir'],
            split_file=config['train_split'],
            should_augment=True,
            pairs_per_epoch=config['pairs_per_epoch']
        )
        
        val_dataset = SiameseLFWDataset(
            data_dir=config['data_dir'],
            split_file=config['train_split'],
            should_augment=False,
            pairs_per_epoch=config['val_pairs']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Create model
        model = SiameseNetwork(
            input_channels=3,
            use_batchnorm=config['use_batchnorm']
        )
        
        # Create experiment directory
        experiment_dir = self.experiments_dir / config['name']
        
        # Create trainer and train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            experiment_dir=experiment_dir
        )
        
        results = trainer.train(num_epochs=config['num_epochs'], verbose=True)
        
        return results
    
    def run_all_experiments(self, parallel: bool = False, n_parallel: int = 2):
        """
        Run all experiments with random hyperparameter sampling.
        
        Args:
            parallel: Whether to run experiments in parallel
            n_parallel: Number of experiments to run in parallel
        """
        print("\n" + "="*80)
        print(f"Starting Hyperparameter Search: {self.n_experiments} experiments")
        if parallel:
            print(f"Running {n_parallel} experiments in parallel")
        print("="*80)
        
        if parallel and torch.cuda.is_available():
            # Run experiments in parallel using multiple GPU streams
            self._run_parallel_experiments(n_parallel)
        else:
            # Run experiments sequentially
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i in range(1, self.n_experiments + 1):
                # Sample configuration
                config = self.sample_config(i)
                
                # Run experiment
                try:
                    results = self.run_experiment(config, device)
                    self.results.append(results)
                except Exception as e:
                    print(f"Experiment {i} failed with error: {e}")
                    continue
        
        # Analyze and save results
        self.analyze_results()
    
    def _run_parallel_experiments(self, n_parallel: int):
        """Run experiments in parallel on GPU."""
        import threading
        from queue import Queue
        
        # Create experiment queue
        experiment_queue = Queue()
        for i in range(1, self.n_experiments + 1):
            config = self.sample_config(i)
            experiment_queue.put(config)
        
        # Results lock
        results_lock = threading.Lock()
        
        def worker(worker_id: int):
            """Worker thread for running experiments."""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            while not experiment_queue.empty():
                try:
                    config = experiment_queue.get(timeout=1)
                    print(f"\n[Worker {worker_id}] Starting {config['name']}")
                    
                    # Run experiment
                    results = self.run_experiment(config, device)
                    
                    # Store results thread-safely
                    with results_lock:
                        self.results.append(results)
                    
                    print(f"[Worker {worker_id}] Completed {config['name']} - Val Acc: {results['best_val_acc']:.2f}%")
                    
                except Exception as e:
                    print(f"[Worker {worker_id}] Experiment failed: {e}")
                    continue
        
        # Create and start worker threads
        threads = []
        for i in range(n_parallel):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
    
    def analyze_results(self):
        """Analyze all experiment results and find best model."""
        if not self.results:
            print("No successful experiments to analyze!")
            return
        
        print("\n" + "="*80)
        print("Experiment Results Summary")
        print("="*80)
        
        # Sort by validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['best_val_acc'], reverse=True)
        
        # Print top 5 experiments
        print("\nTop 5 Experiments by Validation Accuracy:")
        print("-" * 80)
        for i, result in enumerate(sorted_results[:5], 1):
            config = result['config']
            print(f"{i}. {config['name']} | Val Acc: {result['best_val_acc']:.2f}% | "
                  f"LR: {config['learning_rate']:.6f} | BS: {config['batch_size']} | "
                  f"Opt: {config['optimizer']}")
        
        # Best model
        best_result = sorted_results[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Experiment: {best_result['config']['name']}")
        print(f"Validation Accuracy: {best_result['best_val_acc']:.2f}%")
        print(f"Validation Loss: {best_result['best_val_loss']:.4f}")
        print(f"Best Epoch: {best_result['best_epoch']}")
        print(f"Total Time: {best_result['total_time']/60:.1f} minutes")
        print(f"\nHyperparameters:")
        for key in ['learning_rate', 'batch_size', 'weight_decay', 'optimizer', 
                    'pairs_per_epoch', 'use_batchnorm']:
            if key in best_result['config']:
                print(f"  {key}: {best_result['config'][key]}")
        
        # Save summary
        self.save_summary(sorted_results)
        
        # Create visualizations
        self.plot_comparison(sorted_results)
        
        # Copy best model to main checkpoints directory
        best_model_path = self.experiments_dir / best_result['config']['name'] / 'best_model.pth'
        if best_model_path.exists():
            import shutil
            checkpoints_dir = Path('checkpoints')
            checkpoints_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(best_model_path, checkpoints_dir / 'best_model.pth')
            print(f"\nBest model copied to: checkpoints/best_model.pth")
    
    def save_summary(self, sorted_results):
        """Save experiment summary to JSON."""
        summary = {
            'n_experiments': self.n_experiments,
            'search_space': self.search_space,
            'results': [
                {
                    'name': r['config']['name'],
                    'val_acc': r['best_val_acc'],
                    'val_loss': r['best_val_loss'],
                    'best_epoch': r['best_epoch'],
                    'total_time': r['total_time'],
                    'config': {k: v for k, v in r['config'].items() 
                              if k in ['learning_rate', 'batch_size', 'weight_decay', 
                                      'optimizer', 'pairs_per_epoch', 'use_batchnorm']}
                }
                for r in sorted_results
            ]
        }
        
        summary_path = self.experiments_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary saved to: {summary_path}")
    
    def plot_comparison(self, sorted_results):
        """Create comparison plots for all experiments."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Hyperparameter Search Results', fontsize=16, fontweight='bold')
        
        # Extract data
        names = [r['config']['name'] for r in sorted_results]
        val_accs = [r['best_val_acc'] for r in sorted_results]
        val_losses = [r['best_val_loss'] for r in sorted_results]
        lrs = [r['config']['learning_rate'] for r in sorted_results]
        batch_sizes = [r['config']['batch_size'] for r in sorted_results]
        weight_decays = [r['config']['weight_decay'] for r in sorted_results]
        
        # 1. Validation accuracy comparison
        axes[0, 0].barh(range(len(names)), val_accs, color='steelblue')
        axes[0, 0].set_yticks(range(len(names)))
        axes[0, 0].set_yticklabels(names, fontsize=8)
        axes[0, 0].set_xlabel('Validation Accuracy (%)')
        axes[0, 0].set_title('Validation Accuracy by Experiment')
        axes[0, 0].grid(axis='x', alpha=0.3)
        axes[0, 0].invert_yaxis()
        
        # 2. Learning rate vs accuracy
        axes[0, 1].scatter(lrs, val_accs, s=100, alpha=0.6, c=val_accs, cmap='viridis')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Validation Accuracy (%)')
        axes[0, 1].set_title('Learning Rate vs Accuracy')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Batch size vs accuracy
        axes[0, 2].scatter(batch_sizes, val_accs, s=100, alpha=0.6, c=val_accs, cmap='viridis')
        axes[0, 2].set_xlabel('Batch Size')
        axes[0, 2].set_ylabel('Validation Accuracy (%)')
        axes[0, 2].set_title('Batch Size vs Accuracy')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Weight decay vs accuracy
        axes[1, 0].scatter(weight_decays, val_accs, s=100, alpha=0.6, c=val_accs, cmap='viridis')
        axes[1, 0].set_xlabel('Weight Decay')
        axes[1, 0].set_ylabel('Validation Accuracy (%)')
        axes[1, 0].set_title('Weight Decay vs Accuracy')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training curves for top 5
        for i, result in enumerate(sorted_results[:5]):
            epochs = range(1, len(result['history']['val_acc']) + 1)
            axes[1, 1].plot(epochs, result['history']['val_acc'], 
                          label=f"{result['config']['name']} ({result['best_val_acc']:.1f}%)",
                          linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy (%)')
        axes[1, 1].set_title('Top 5 Training Curves')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Hyperparameter importance (correlation with accuracy)
        correlations = {
            'Learning Rate': np.corrcoef(np.log10(lrs), val_accs)[0, 1],
            'Batch Size': np.corrcoef(batch_sizes, val_accs)[0, 1],
            'Weight Decay': np.corrcoef(np.log10(weight_decays), val_accs)[0, 1],
        }
        
        params = list(correlations.keys())
        corr_values = list(correlations.values())
        colors = ['green' if v > 0 else 'red' for v in corr_values]
        
        axes[1, 2].barh(params, corr_values, color=colors, alpha=0.6)
        axes[1, 2].set_xlabel('Correlation with Val Accuracy')
        axes[1, 2].set_title('Hyperparameter Importance')
        axes[1, 2].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1, 2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiments_dir / 'comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")
        plt.close()


def main():
    """Main training function with hyperparameter search."""
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Base configuration (fixed parameters)
    base_config = {
        # Data
        'data_dir': '/home/ben/Desktop/Ben/DeepExercise/0data/lfw2',
        'train_split': '/home/ben/Desktop/Ben/DeepExercise/0data/train.txt',
        
        # Training
        'num_epochs': 50,  # Reduced from 50
        'early_stopping_patience': 5,  # Reduced from 10
        'lr_patience': 5,  # Reduced from 5
        'lr_factor': 0.5,
        
        # Data
        'val_pairs': 1000,  # Reduced from 2000 for faster validation
        
        # System
        'num_workers': 4,
    }
    
    # Search space for hyperparameters
    # Paper uses LR=0.00006 with momentum SGD
    search_space = {
        'learning_rate': {
            'type': 'loguniform',
            'low': 1e-5,  # Paper uses 6e-5
            'high': 5e-4  # Lower range based on paper
        },
        'batch_size': {
            'type': 'choice',
            'values': [64, 128, 256]
        },
        'weight_decay': {
            'type': 'loguniform',
            'low': 1e-5,
            'high': 1e-3
        },
        'optimizer': {
            'type': 'choice',
            'values': ['adam', 'sgd']
        },
        'pairs_per_epoch': {
            'type': 'choice',
            'values': [2000, 4000, 6000]  # Reduced for faster training
        },
        'use_batchnorm': {
            'type': 'choice',
            'values': [False, True]  # Paper doesn't use batchnorm, so False first
        }
    }
    
    # Number of experiments to run
    n_experiments = 10  # Reduced for faster search
    
    print("\n" + "="*80)
    print("SIAMESE NETWORK HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Number of experiments: {n_experiments}")
    print(f"Search space:")
    for param, space in search_space.items():
        print(f"  {param}: {space}")
    print("="*80)
    
    # Create and run hyperparameter search
    search = HyperparameterSearch(
        base_config=base_config,
        search_space=search_space,
        n_experiments=n_experiments
    )
    
    # Enable parallel training (2-3 models at once on GPU)
    # Adjust n_parallel based on your GPU memory
    use_parallel = torch.cuda.is_available()
    n_parallel = 4  # Run 4 experiments in parallel (adjust based on GPU memory)
    
    if use_parallel:
        print(f"\n💡 Parallel training enabled: {n_parallel} experiments at once")
        print(f"   This will use more GPU memory but finish ~{n_parallel}x faster!")
    
    search.run_all_experiments(parallel=use_parallel, n_parallel=n_parallel)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"Results saved to: {search.experiments_dir}")
    print("\nNext steps:")
    print("1. Review the comparison plots and summary")
    print("2. Run: python evaluate.py  (to test the best model)")
    print("="*80)


if __name__ == "__main__":
    main()
