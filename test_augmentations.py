import torch
import torch.nn as nn
import kornia.augmentation as K
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np


def load_sample_images(data_dir: str, num_images: int = 6):
    """Load sample images from dataset."""
    data_path = Path(data_dir)
    images = []
    
    # Get first few people's first image
    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue
        
        img_files = sorted(list(person_dir.glob("*.jpg")))
        if img_files:
            images.append(img_files[0])
        
        if len(images) >= num_images:
            break
    
    return images


def load_and_preprocess(img_path):
    """Load image and convert to tensor."""
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    return transform(img)


def denormalize(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def visualize_augmentations(
    data_dir: str,
    augmentation_list: list,
    output_dir: str,
    num_images: int = 6,
    num_augmentations: int = 4
):
    """
    Visualize augmentations on sample images.
    
    Args:
        data_dir: Path to image directory
        augmentation_list: List of Kornia augmentation objects
        output_dir: Directory to save visualization
        num_images: Number of sample images to use
        num_augmentations: Number of augmented versions per image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Augmentation Visualization")
    print("=" * 80)
    
    # Load sample images
    print(f"Loading {num_images} sample images...")
    image_paths = load_sample_images(data_dir, num_images)
    print(f"Loaded {len(image_paths)} images")
    
    # Create augmentation pipeline
    aug_pipeline = nn.Sequential(*augmentation_list)
    
    print(f"\nAugmentation pipeline:")
    for i, aug in enumerate(augmentation_list, 1):
        print(f"  {i}. {aug}")
    
    # Create visualization
    fig, axes = plt.subplots(num_images, num_augmentations + 1, 
                             figsize=(3 * (num_augmentations + 1), 3 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nGenerating {num_augmentations} augmented versions per image...")
    
    for row, img_path in enumerate(image_paths):
        # Load and preprocess image
        img_tensor = load_and_preprocess(img_path).unsqueeze(0)
        
        # Show original
        original_display = denormalize(img_tensor[0]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(original_display)
        axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Show augmented versions
        for col in range(num_augmentations):
            augmented = aug_pipeline(img_tensor)
            augmented_display = denormalize(augmented[0]).clamp(0, 1).permute(1, 2, 0).numpy()
            
            axes[row, col + 1].imshow(augmented_display)
            axes[row, col + 1].set_title(f'Augmented {col + 1}', fontsize=11, fontweight='bold')
            axes[row, col + 1].axis('off')
        
        # Add person name on the left
        person_name = img_path.parent.name
        axes[row, 0].text(-0.1, 0.5, person_name, 
                         transform=axes[row, 0].transAxes,
                         fontsize=10, fontweight='bold',
                         rotation=90, va='center', ha='right')
    
    plt.suptitle('Augmentation Examples: Original vs Augmented', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'augmentation_examples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_file}")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


def visualize_individual_augmentations(
    data_dir: str,
    augmentation_list: list,
    output_dir: str,
    num_images: int = 3
):
    """
    Visualize each augmentation separately to show its individual effect.
    
    Args:
        data_dir: Path to image directory
        augmentation_list: List of Kornia augmentation objects
        output_dir: Directory to save visualization
        num_images: Number of sample images to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Individual Augmentation Effects Visualization")
    print("=" * 80)
    
    # Load sample images
    print(f"Loading {num_images} sample images...")
    image_paths = load_sample_images(data_dir, num_images)
    print(f"Loaded {len(image_paths)} images")
    
    # Create figure: rows = augmentations, cols = images
    num_augs = len(augmentation_list)
    fig, axes = plt.subplots(num_augs + 1, num_images, 
                             figsize=(4 * num_images, 3 * (num_augs + 1)))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"\nShowing effect of each augmentation separately...")
    
    # Load all images
    img_tensors = []
    for img_path in image_paths:
        img_tensor = load_and_preprocess(img_path).unsqueeze(0)
        img_tensors.append(img_tensor)
    
    # Row 0: Original images
    for col, (img_tensor, img_path) in enumerate(zip(img_tensors, image_paths)):
        original_display = denormalize(img_tensor[0]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, col].imshow(original_display)
        axes[0, col].set_title(f'Original\n{img_path.parent.name}', 
                              fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
    
    # Label for original row
    axes[0, 0].text(-0.15, 0.5, 'ORIGINAL', 
                   transform=axes[0, 0].transAxes,
                   fontsize=12, fontweight='bold',
                   rotation=90, va='center', ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Each subsequent row: one augmentation applied
    for row, aug in enumerate(augmentation_list, start=1):
        # Get augmentation name and params
        aug_name = aug.__class__.__name__
        
        # Create a single-item pipeline with just this augmentation
        single_aug_pipeline = nn.Sequential(aug)
        
        # Apply this augmentation to each image
        for col, img_tensor in enumerate(img_tensors):
            augmented = single_aug_pipeline(img_tensor)
            augmented_display = denormalize(augmented[0]).clamp(0, 1).permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(augmented_display)
            if col == 0:
                axes[row, col].set_title(f'{aug_name}', 
                                        fontsize=9, fontweight='bold')
            axes[row, col].axis('off')
        
        # Label for this augmentation row
        axes[row, 0].text(-0.15, 0.5, aug_name.replace('Random', ''), 
                         transform=axes[row, 0].transAxes,
                         fontsize=10, fontweight='bold',
                         rotation=90, va='center', ha='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('Individual Augmentation Effects (Each Row = One Augmentation)', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'individual_augmentation_effects.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved individual effects visualization: {output_file}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Individual Effects Visualization Complete!")
    print("=" * 80)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Set random seed for reproducibility
    set_seed(42)
    # Define augmentations to test
    augmentations = [
        K.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomPerspective(distortion_scale=0.2, p=1.0),
        
        # Blur & Noise
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
        K.RandomGaussianNoise(mean=0.0, std=0.12, p=1.0),
        
        # Intensity & Color
        K.RandomGrayscale(p=1.0),
    ]
    
    # First visualization: Combined augmentation pipeline
    print("Creating combined augmentation visualization...")
    visualize_augmentations(
        data_dir='/home/ben/Desktop/Ben/DeepExercise/0data/lfw2',
        augmentation_list=augmentations,
        output_dir='/home/ben/Desktop/Ben/DeepExercise/0outputs/figures',
        num_images=10,
        num_augmentations=4
    )
    
    # Second visualization: Individual augmentation effects
    print("\nCreating individual augmentation effects visualization...")
    visualize_individual_augmentations(
        data_dir='/home/ben/Desktop/Ben/DeepExercise/0data/lfw2',
        augmentation_list=augmentations,
        output_dir='/home/ben/Desktop/Ben/DeepExercise/0outputs/figures',
        num_images=5
    )

