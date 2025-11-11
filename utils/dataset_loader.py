"""
Dataset loader for LFW-a dataset.
Loads file paths and metadata without loading images into memory.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import defaultdict, Counter


class LFWDatasetLoader:
    """
    Lazy loader for LFW-a dataset that stores file paths and metadata
    without loading images into memory.
    """
    
    def __init__(self, data_dir: str, split_file: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Path to the lfw2 directory containing person folders
            split_file: Optional path to train.txt or test.txt file containing pairs
        """
        self.data_dir = Path(data_dir)
        self.split_file = split_file
        
        # Data structures to store metadata
        self.person_to_images: Dict[str, List[Path]] = defaultdict(list)
        self.image_paths: List[Path] = []
        self.person_names: List[str] = []
        self.pairs: List[Tuple[str, int, int]] = []  # (person_name, img1_num, img2_num)
        self.people_in_split: Set[str] = set()
        
        # Load the dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load file paths and build metadata structures."""
        # If split file is provided, parse it to get the people in this split
        if self.split_file:
            self._parse_split_file()
        
        # Iterate through person directories
        for person_dir in sorted(self.data_dir.iterdir()):
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            
            # Skip if not in allowed people (when using split file)
            if self.split_file and person_name not in self.people_in_split:
                continue
            
            # Get all image files for this person
            image_files = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
            
            if len(image_files) > 0:
                self.person_to_images[person_name] = image_files
                self.image_paths.extend(image_files)
                self.person_names.extend([person_name] * len(image_files))
        
        print(f"Loaded {len(self.person_to_images)} people with {len(self.image_paths)} total images")
        if self.split_file:
            print(f"  Pairs in split file: {len(self.pairs)}")
    
    def _parse_split_file(self):
        """Parse the train.txt or test.txt file to extract pairs and people."""
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
        
        # First line is the number of pairs
        num_pairs = int(lines[0].strip())
        
        # Parse pairs
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) == 3:
                person_name = parts[0]
                img1_num = int(parts[1])
                img2_num = int(parts[2])
                self.pairs.append((person_name, img1_num, img2_num))
                self.people_in_split.add(person_name)
        
        print(f"  Parsed {len(self.pairs)} pairs from split file")
        print(f"  Unique people in split: {len(self.people_in_split)}")
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing various statistics
        """
        images_per_person = [len(images) for images in self.person_to_images.values()]
        
        if not images_per_person:
            return {
                'total_people': 0,
                'total_images': 0,
                'min_images_per_person': 0,
                'max_images_per_person': 0,
                'mean_images_per_person': 0,
                'median_images_per_person': 0,
                'std_images_per_person': 0,
                'images_per_person_distribution': Counter(),
                'num_pairs': len(self.pairs),
            }
        
        stats = {
            'total_people': len(self.person_to_images),
            'total_images': len(self.image_paths),
            'min_images_per_person': min(images_per_person),
            'max_images_per_person': max(images_per_person),
            'mean_images_per_person': np.mean(images_per_person),
            'median_images_per_person': np.median(images_per_person),
            'std_images_per_person': np.std(images_per_person),
            'images_per_person_distribution': Counter(images_per_person),
            'num_pairs': len(self.pairs),
        }
        
        return stats
    
    def get_people_with_min_images(self, min_images: int) -> List[str]:
        """
        Get list of people with at least min_images images.
        
        Args:
            min_images: Minimum number of images required
            
        Returns:
            List of person names
        """
        return [person for person, images in self.person_to_images.items() 
                if len(images) >= min_images]
    
    def get_image_paths_for_person(self, person_name: str) -> List[Path]:
        """
        Get all image paths for a specific person.
        
        Args:
            person_name: Name of the person
            
        Returns:
            List of image paths
        """
        return self.person_to_images.get(person_name, [])
    
    def get_all_people(self) -> List[str]:
        """Get list of all person names in the dataset."""
        return list(self.person_to_images.keys())
    
    def get_pair_by_index(self, index: int) -> Tuple[Path, Path]:
        """
        Get a specific pair from the split file by index.
        
        Args:
            index: Index of the pair in the pairs list
            
        Returns:
            Tuple of (image1_path, image2_path)
        """
        if not self.pairs:
            raise ValueError("No pairs loaded. Make sure to provide a split file.")
        
        person_name, img1_num, img2_num = self.pairs[index]
        images = self.person_to_images[person_name]
        
        # Image numbers in the file are 1-indexed
        img1_path = images[img1_num - 1]
        img2_path = images[img2_num - 1]
        
        return img1_path, img2_path
    
    def get_random_pair(self, same_person: bool = True) -> Tuple[Path, Path, int]:
        """
        Get a random pair of images.
        
        Args:
            same_person: If True, return pair from same person, else different people
            
        Returns:
            Tuple of (image1_path, image2_path, label) where label is 1 for same, 0 for different
        """
        if same_person:
            # Find people with at least 2 images
            valid_people = self.get_people_with_min_images(2)
            if not valid_people:
                raise ValueError("No people with 2+ images for same-person pairs")
            
            person = np.random.choice(valid_people)
            images = self.person_to_images[person]
            img1, img2 = np.random.choice(images, size=2, replace=False)
            return img1, img2, 1
        else:
            # Different people
            if len(self.person_to_images) < 2:
                raise ValueError("Need at least 2 people for different-person pairs")
            
            person1, person2 = np.random.choice(list(self.person_to_images.keys()), size=2, replace=False)
            img1 = np.random.choice(self.person_to_images[person1])
            img2 = np.random.choice(self.person_to_images[person2])
            return img1, img2, 0
    
    def generate_pairs(self, num_pairs: int, balanced: bool = True) -> List[Tuple[Path, Path, int]]:
        """
        Generate pairs of images for training/testing.
        
        Args:
            num_pairs: Number of pairs to generate
            balanced: If True, generate equal number of same/different pairs
            
        Returns:
            List of (image1_path, image2_path, label) tuples
        """
        pairs = []
        
        if balanced:
            num_same = num_pairs // 2
            num_diff = num_pairs - num_same
        else:
            num_same = num_pairs // 2
            num_diff = num_pairs // 2
        
        # Generate same-person pairs
        for _ in range(num_same):
            pairs.append(self.get_random_pair(same_person=True))
        
        # Generate different-person pairs
        for _ in range(num_diff):
            pairs.append(self.get_random_pair(same_person=False))
        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        return pairs
    
    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)
    
    def __repr__(self):
        return f"LFWDatasetLoader(people={len(self.person_to_images)}, images={len(self.image_paths)}, pairs={len(self.pairs)})"
