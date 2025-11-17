import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


class SiameseDataset(Dataset):
    """Simple Siamese dataset that loads pairs from JSON file."""
    
    def __init__(self, dataset_json: str):
        """
        Args:
            dataset_json: Path to dataset JSON file (from create_dataset.py)
        """
        self.dataset_json = Path(dataset_json)
        
        # Load dataset
        with open(self.dataset_json, 'r') as f:
            data = json.load(f)
        
        self.name = data['name']
        self.pairs = data['pairs']
        
        # Basic preprocessing (resize + normalize)
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {self.name} dataset: {len(self.pairs)} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            img1: First image tensor [3, 105, 105]
            img2: Second image tensor [3, 105, 105]
            label: 1 if same person, 0 if different
        """
        pair = self.pairs[idx]
        
        # Load images
        img1 = Image.open(pair['image1']).convert('RGB')
        img2 = Image.open(pair['image2']).convert('RGB')
        
        # Apply transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = pair['label']
        
        return img1, img2, label


if __name__ == '__main__':
    # Test the dataset
    print("=" * 80)
    print("Testing SiameseDataset")
    print("=" * 80)
    
    # Test loading
    train_dataset = SiameseDataset('/home/ben/Desktop/Ben/DeepExercise/0data/datasets/train_dataset.json')
    val_dataset = SiameseDataset('/home/ben/Desktop/Ben/DeepExercise/0data/datasets/val_dataset.json')
    test_dataset = SiameseDataset('/home/ben/Desktop/Ben/DeepExercise/0data/datasets/test_dataset.json')
    
    # Test getting items
    print(f"\nTesting data loading:")
    img1, img2, label = train_dataset[0]
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    print(f"  Label: {label}")
    
    # Test balance
    print(f"\nChecking balance (first 100 samples):")
    labels = [train_dataset[i][2] for i in range(100)]
    pos_count = sum(labels)
    print(f"  Positive: {pos_count}/100")
    print(f"  Negative: {100 - pos_count}/100")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
