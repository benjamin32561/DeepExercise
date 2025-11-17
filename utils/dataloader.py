from torch.utils.data import DataLoader
try:
    from utils.dataset import SiameseDataset
except ImportError:
    from dataset import SiameseDataset


def create_dataloaders(
    train_dataset_json: str,
    val_dataset_json: str,
    test_dataset_json: str = None,
    batch_size: int = 32,
    num_workers: int = 8
):
    """
    Create train/val/test dataloaders.
    
    Args:
        train_dataset_json: Path to train dataset JSON
        val_dataset_json: Path to val dataset JSON  
        test_dataset_json: Optional path to test dataset JSON
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    # Create datasets
    train_dataset = SiameseDataset(train_dataset_json)
    val_dataset = SiameseDataset(val_dataset_json)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Optional test loader
    if test_dataset_json is not None:
        test_dataset = SiameseDataset(test_dataset_json)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )
        loaders['test'] = test_loader
    
    return loaders


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Dataloader Creation")
    print("=" * 80)
    
    # Test dataloader creation
    loaders = create_dataloaders(
        train_dataset_json='/home/ben/Desktop/Ben/DeepExercise/0data/datasets/train_dataset.json',
        val_dataset_json='/home/ben/Desktop/Ben/DeepExercise/0data/datasets/val_dataset.json',
        test_dataset_json='/home/ben/Desktop/Ben/DeepExercise/0data/datasets/test_dataset.json',
        batch_size=32,
        num_workers=4
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    
    # Test loading a batch
    print(f"\nLoading a batch from train:")
    img1, img2, labels = next(iter(loaders['train']))
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    print(f"  Labels shape:  {labels.shape}")
    print(f"  Batch balance: {labels.sum().item()}/{len(labels)} positive")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

