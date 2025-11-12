"""
PyTorch Dataset classes for Siamese Network training.
Handles pair generation, data augmentation, and efficient loading.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional


class SiameseLFWDataset(Dataset):
    """
    Dataset for training Siamese Networks on LFW-a dataset.
    Generates pairs of images (same person or different people) on-the-fly.
    
    Reasoning for design choices:
    1. On-the-fly pair generation: Prevents memory issues and allows infinite variations
    2. Balanced pairs: Equal number of same/different pairs for stable training
    3. Data augmentation: Improves generalization to unseen variations
    4. Lazy loading: Only loads images when needed, not all at once
    """
    
    def __init__(
        self,
        data_dir: str,
        split_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        should_augment: bool = True,
        pairs_per_epoch: int = 10000
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to lfw2 directory
            split_file: Path to train.txt or test.txt (None for full dataset)
            transform: Torchvision transforms to apply
            should_augment: Whether to apply data augmentation
            pairs_per_epoch: Number of pairs to generate per epoch
        """
        self.data_dir = Path(data_dir)
        self.split_file = split_file
        self.should_augment = should_augment
        self.pairs_per_epoch = pairs_per_epoch
        
        # Load dataset metadata
        self.person_to_images = {}
        self.people_list = []
        self._load_dataset()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform(should_augment)
        else:
            self.transform = transform
    
    def _load_dataset(self):
        """Load file paths and organize by person."""
        # If split file provided, get people from it
        if self.split_file:
            with open(self.split_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip first line (number of pairs)
            people_in_split = set()
            for line in lines:
                if line.strip():
                    person_name = line.split('\t')[0]
                    people_in_split.add(person_name)
        else:
            people_in_split = None
        
        # Load image paths for each person
        for person_dir in sorted(self.data_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            # Skip if not in split
            if people_in_split is not None and person_name not in people_in_split:
                continue
            
            # Get all images for this person
            images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
            
            # Only include people with at least 2 images
            if len(images) >= 2:
                self.person_to_images[person_name] = images
                self.people_list.append(person_name)
        
        print(f"Loaded {len(self.people_list)} people with {sum(len(imgs) for imgs in self.person_to_images.values())} images")
    
    def _get_default_transform(self, should_augment: bool):
        """
        Get default image transforms.
        
        Augmentation strategy (when should_augment=True):
        1. Random affine: rotation, translation, scale (as in paper)
        2. Random horizontal flip: faces can be mirrored
        3. Color jitter: lighting variations
        4. Normalization: standard ImageNet normalization
        
        Reasoning:
        - Affine transforms simulate natural pose variations
        - Color jitter handles different lighting conditions
        - Normalization stabilizes training
        """
        transform_list = [
            transforms.Resize((105, 105)),  # Resize to standard size
        ]
        
        if should_augment:
            transform_list.extend([
                transforms.RandomAffine(
                    degrees=10,  # Rotation up to ±10 degrees
                    translate=(0.1, 0.1),  # Translation up to 10%
                    scale=(0.9, 1.1),  # Scale 90% to 110%
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        """Return the number of pairs per epoch."""
        return self.pairs_per_epoch
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a pair of images and their label.
        
        Args:
            idx: Index (not used, pairs are generated randomly)
            
        Returns:
            Tuple of (img1, img2, label) where label is 1 for same, 0 for different
        """
        # Randomly decide if this should be a same or different pair
        should_get_same_class = random.random() > 0.5
        
        if should_get_same_class:
            # Same person pair - label = 1 (similar)
            person = random.choice(self.people_list)
            images = self.person_to_images[person]
            img1_path, img2_path = random.sample(images, 2)
            label = 1  # Same person = high similarity
        else:
            # Different people pair - label = 0 (dissimilar)
            person1, person2 = random.sample(self.people_list, 2)
            img1_path = random.choice(self.person_to_images[person1])
            img2_path = random.choice(self.person_to_images[person2])
            label = 0  # Different person = low similarity
        
        # Load and transform images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, label


class SiameseTestDataset(Dataset):
    """
    Dataset for testing/evaluation using predefined pairs from test.txt.
    
    This dataset uses the exact pairs specified in the test file,
    ensuring reproducible evaluation results.
    """
    
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize test dataset.
        
        Args:
            data_dir: Path to lfw2 directory
            split_file: Path to test.txt file with predefined pairs
            transform: Torchvision transforms (no augmentation for test)
        """
        self.data_dir = Path(data_dir)
        self.split_file = split_file
        
        # Load dataset
        self.person_to_images = {}
        self.pairs = []
        self._load_dataset()
        
        # Set up transforms (no augmentation for test)
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _load_dataset(self):
        """Load image paths and parse pairs from split file."""
        # Parse split file
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
        
        num_pairs = int(lines[0].strip())
        
        # Get all people in this split
        people_in_split = set()
        for line in lines[1:]:
            if line.strip():
                person_name = line.split('\t')[0]
                people_in_split.add(person_name)
        
        # Load image paths
        for person_dir in sorted(self.data_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            if person_name not in people_in_split:
                continue
            
            images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
            if len(images) > 0:
                self.person_to_images[person_name] = images
        
        # Parse pairs
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) == 3:
                person_name = parts[0]
                img1_num = int(parts[1]) - 1  # Convert to 0-indexed
                img2_num = int(parts[2]) - 1
                
                if person_name in self.person_to_images:
                    images = self.person_to_images[person_name]
                    if img1_num < len(images) and img2_num < len(images):
                        self.pairs.append((images[img1_num], images[img2_num], 1))  # Label 1 = same person (similar)
        
        print(f"Loaded {len(self.pairs)} test pairs from {len(self.person_to_images)} people")
    
    def _get_default_transform(self):
        """Get transforms for test (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        """Return number of test pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a specific test pair.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Tuple of (img1, img2, label)
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load and transform images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, label


if __name__ == "__main__":
    # Test the dataset
    print("Testing Siamese Dataset")
    print("=" * 60)
    
    data_dir = "/home/ben/Desktop/Ben/DeepExercise/0data/lfw2"
    train_split = "/home/ben/Desktop/Ben/DeepExercise/0data/train.txt"
    
    # Test training dataset
    train_dataset = SiameseLFWDataset(
        data_dir=data_dir,
        split_file=train_split,
        should_augment=True,
        pairs_per_epoch=100
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    
    # Get a sample
    img1, img2, label = train_dataset[0]
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Label: {label} ({'same' if label == 0 else 'different'})")
    
    # Test test dataset
    test_split = "/home/ben/Desktop/Ben/DeepExercise/0data/test.txt"
    test_dataset = SiameseTestDataset(
        data_dir=data_dir,
        split_file=test_split
    )
    
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    print("\n" + "=" * 60)
    print("Dataset test completed successfully!")

