import json
import random
from pathlib import Path
from typing import List, Tuple, Dict


def load_positive_pairs(lfw_file: str, data_dir: str) -> List[Tuple[str, str, int]]:
    """
    Load positive pairs from LFW format file.
    
    Args:
        lfw_file: Path to train.txt or test.txt
        data_dir: Path to lfw2 directory
        
    Returns:
        List of (image1_path, image2_path, label=1) tuples
    """
    data_path = Path(data_dir)
    pairs = []
    
    with open(lfw_file, 'r') as f:
        lines = f.readlines()
    
    # First line is number of pairs
    num_pairs = int(lines[0].strip())
    
    for line in lines[1:]:
        if not line.strip():
            continue
        
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        
        person_name = parts[0]
        img1_num = int(parts[1])
        img2_num = int(parts[2])
        
        # Construct image paths
        person_dir = data_path / person_name
        img1_path = person_dir / f"{person_name}_{img1_num:04d}.jpg"
        img2_path = person_dir / f"{person_name}_{img2_num:04d}.jpg"
        
        # Verify images exist
        if img1_path.exists() and img2_path.exists():
            pairs.append((str(img1_path), str(img2_path), 1))
    
    return pairs


def get_all_people_images(data_dir: str) -> Dict[str, List[str]]:
    """
    Get all people and their images from data directory.
    
    Args:
        data_dir: Path to lfw2 directory
        
    Returns:
        Dictionary mapping person_name -> list of image paths
    """
    data_path = Path(data_dir)
    person_to_images = {}
    
    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
        
        if len(images) > 0:
            person_to_images[person_name] = [str(img) for img in images]
    
    return person_to_images


def generate_negative_pairs(
    num_pairs: int,
    person_to_images: Dict[str, List[str]],
    random_seed: int
) -> List[Tuple[str, str, int]]:
    """
    Generate negative pairs (different people).
    
    Args:
        num_pairs: Number of negative pairs to generate
        person_to_images: Dictionary of person -> images
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (image1_path, image2_path, label=0) tuples
    """
    random.seed(random_seed)
    people_list = list(person_to_images.keys())
    negative_pairs = []
    
    for _ in range(num_pairs):
        # Sample two different people
        person1, person2 = random.sample(people_list, 2)
        
        # Sample one image from each
        img1 = random.choice(person_to_images[person1])
        img2 = random.choice(person_to_images[person2])
        
        negative_pairs.append((img1, img2, 0))
    
    return negative_pairs


def save_dataset(
    name: str,
    pairs: List[Tuple[str, str, int]],
    output_dir: str
):
    """
    Save dataset to JSON file.
    
    Args:
        name: Dataset name (train/val/test)
        pairs: List of (img1, img2, label) tuples
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count statistics
    num_positive = sum(1 for _, _, label in pairs if label == 1)
    num_negative = len(pairs) - num_positive
    
    dataset = {
        'name': name,
        'num_pairs': len(pairs),
        'num_positive': num_positive,
        'num_negative': num_negative,
        'balance': num_positive / len(pairs) if pairs else 0.0,
        'pairs': [
            {
                'image1': img1,
                'image2': img2,
                'label': label
            }
            for img1, img2, label in pairs
        ]
    }
    
    output_file = output_path / f'{name}_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"{name:5s}: {len(pairs):5d} pairs ({num_positive:5d} pos, {num_negative:5d} neg) -> {output_file.name}")


def create_datasets(
    train_file: str,
    test_file: str,
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test datasets with balanced positive/negative pairs.
    
    Args:
        train_file: Path to train.txt
        test_file: Path to test.txt
        data_dir: Path to lfw2 directory
        output_dir: Output directory for dataset files
        val_ratio: Fraction of train pairs to use for validation
        random_seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("Creating Balanced Datasets (Train/Val/Test)")
    print("=" * 80)
    print(f"Train file:  {train_file}")
    print(f"Test file:   {test_file}")
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Val ratio:   {val_ratio:.0%} (from train.txt)")
    print(f"Random seed: {random_seed}")
    print()
    
    # Load all people and images (needed for negative pairs)
    print("Loading people and images...")
    person_to_images = get_all_people_images(data_dir)
    print(f"  Found {len(person_to_images)} people with {sum(len(imgs) for imgs in person_to_images.values())} images")
    print()
    
    # Load positive pairs from train.txt
    print("Loading positive pairs from train.txt...")
    train_val_positive = load_positive_pairs(train_file, data_dir)
    print(f"  Found {len(train_val_positive)} positive pairs")
    
    # Split train.txt positive pairs into train/val
    random.seed(random_seed)
    random.shuffle(train_val_positive)
    
    n_val = int(len(train_val_positive) * val_ratio)
    val_positive = train_val_positive[:n_val]
    train_positive = train_val_positive[n_val:]
    
    print(f"  Split into {len(train_positive)} train, {len(val_positive)} val")
    print()
    
    # Load positive pairs from test.txt
    print("Loading positive pairs from test.txt...")
    test_positive = load_positive_pairs(test_file, data_dir)
    print(f"  Found {len(test_positive)} positive pairs")
    print()
    
    # Generate negative pairs for each split
    print("Generating negative pairs...")
    train_negative = generate_negative_pairs(len(train_positive), person_to_images, random_seed)
    val_negative = generate_negative_pairs(len(val_positive), person_to_images, random_seed + 1)
    test_negative = generate_negative_pairs(len(test_positive), person_to_images, random_seed + 2)
    print(f"  Train: {len(train_negative)} negative pairs")
    print(f"  Val:   {len(val_negative)} negative pairs")
    print(f"  Test:  {len(test_negative)} negative pairs")
    print()
    
    # Combine positive and negative pairs
    train_pairs = train_positive + train_negative
    val_pairs = val_positive + val_negative
    test_pairs = test_positive + test_negative
    
    # Shuffle each dataset
    random.seed(random_seed)
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)
    
    # Save datasets
    print("Saving datasets...")
    save_dataset('train', train_pairs, output_dir)
    save_dataset('val', val_pairs, output_dir)
    save_dataset('test', test_pairs, output_dir)
    
    print()
    print("=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    create_datasets(
        train_file='/home/ben/Desktop/Ben/DeepExercise/0data/train.txt',
        test_file='/home/ben/Desktop/Ben/DeepExercise/0data/test.txt',
        data_dir='/home/ben/Desktop/Ben/DeepExercise/0data/lfw2',
        output_dir='/home/ben/Desktop/Ben/DeepExercise/0data/datasets',
        val_ratio=0.15,
        random_seed=42
    )

