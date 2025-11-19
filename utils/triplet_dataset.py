"""
Triplet Dataset for Triplet Loss Training

Creates (anchor, positive, negative) triplets from image pairs.
"""
import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class TripletDataset(Dataset):
    """
    Dataset that generates triplets for triplet loss.
    
    For each anchor:
    - Positive: Same person (different image)
    - Negative: Different person
    """
    
    def __init__(self, dataset_json_path, transform=None):
        """
        Args:
            dataset_json_path: Path to JSON file with pairs
            transform: Optional transform to apply to images
        """
        with open(dataset_json_path, 'r') as f:
            data = json.load(f)
            # Handle nested structure with "pairs" key
            if isinstance(data, dict) and 'pairs' in data:
                self.pairs = data['pairs']
            else:
                self.pairs = data
        
        self.transform = transform
        
        # Organize data by person ID for efficient triplet mining
        self.person_to_images = defaultdict(list)
        self.all_persons = set()
        
        for pair in self.pairs:
            # Extract person IDs from paths (assumes LFW format: person_name/image.jpg)
            img1_path = Path(pair['image1'])
            img2_path = Path(pair['image2'])
            
            person1 = img1_path.parent.name
            person2 = img2_path.parent.name
            
            self.person_to_images[person1].append(pair['image1'])
            self.person_to_images[person2].append(pair['image2'])
            
            self.all_persons.add(person1)
            self.all_persons.add(person2)
        
        # Remove duplicates and ensure each person has at least 2 images
        for person in list(self.person_to_images.keys()):
            self.person_to_images[person] = list(set(self.person_to_images[person]))
            if len(self.person_to_images[person]) < 2:
                del self.person_to_images[person]
                self.all_persons.discard(person)
        
        self.persons_list = list(self.person_to_images.keys())
        
        print(f"Loaded triplet dataset:")
        print(f"  Total persons: {len(self.persons_list)}")
        print(f"  Total unique images: {sum(len(imgs) for imgs in self.person_to_images.values())}")
    
    def __len__(self):
        # Each epoch will sample one triplet per person
        return len(self.persons_list)
    
    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative).
        
        Returns:
            anchor: Image tensor
            positive: Image tensor (same person as anchor)
            negative: Image tensor (different person)
            person_id: Person ID (for debugging)
        """
        # Select anchor person
        anchor_person = self.persons_list[idx]
        anchor_images = self.person_to_images[anchor_person]
        
        # Sample anchor and positive (both from same person)
        if len(anchor_images) >= 2:
            anchor_path, positive_path = random.sample(anchor_images, 2)
        else:
            # Fallback: use same image (should rarely happen)
            anchor_path = positive_path = anchor_images[0]
        
        # Sample negative (from different person)
        negative_person = random.choice([p for p in self.persons_list if p != anchor_person])
        negative_path = random.choice(self.person_to_images[negative_person])
        
        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img, anchor_person


def create_triplet_dataloaders(train_json, val_json, batch_size=32, num_workers=8):
    """
    Create dataloaders for triplet training.
    
    Args:
        train_json: Path to training dataset JSON
        val_json: Path to validation dataset JSON
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        dict with 'train' and 'val' dataloaders
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    # Basic transforms (augmentation will be done on GPU with Kornia)
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TripletDataset(train_json, transform=transform)
    val_dataset = TripletDataset(val_json, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return {'train': train_loader, 'val': val_loader}


if __name__ == '__main__':
    # Test the dataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
    ])
    
    dataset = TripletDataset('0data/datasets/train_dataset.json', transform=transform)
    print(f"\nDataset size: {len(dataset)}")
    
    # Sample a triplet
    anchor, positive, negative, person = dataset[0]
    print(f"\nSample triplet:")
    print(f"  Anchor shape: {anchor.shape}")
    print(f"  Positive shape: {positive.shape}")
    print(f"  Negative shape: {negative.shape}")
    print(f"  Person: {person}")
    print(f"\n✓ Triplet dataset working correctly!")

