#!/usr/bin/env python3
"""
Analyze the LFW-a dataset and create train/test splits.
"""
import os
from pathlib import Path
from collections import Counter
import random

# Set random seed for reproducibility
random.seed(42)

# Dataset path
dataset_path = Path("0data/lfw2")

# Get all person directories
person_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

print(f"Total number of people: {len(person_dirs)}")

# Count images per person
person_image_counts = {}
total_images = 0

for person_dir in person_dirs:
    images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    person_image_counts[person_dir.name] = len(images)
    total_images += len(images)

print(f"Total number of images: {total_images}")

# Statistics
image_counts = list(person_image_counts.values())
print(f"\nImage count statistics:")
print(f"  Min images per person: {min(image_counts)}")
print(f"  Max images per person: {max(image_counts)}")
print(f"  Average images per person: {sum(image_counts) / len(image_counts):.2f}")
print(f"  Median images per person: {sorted(image_counts)[len(image_counts)//2]}")

# Distribution of people by number of images
count_distribution = Counter(image_counts)
print(f"\nDistribution of people by number of images:")
for count in sorted(count_distribution.keys())[:10]:
    print(f"  {count} image(s): {count_distribution[count]} people")
if len(count_distribution) > 10:
    print(f"  ... (showing first 10)")

# People with multiple images (suitable for training)
people_with_multiple = [name for name, count in person_image_counts.items() if count >= 2]
print(f"\nPeople with 2+ images: {len(people_with_multiple)}")

# Create train/test split (80/20 split, ensuring no overlap)
random.shuffle(people_with_multiple)
split_idx = int(0.8 * len(people_with_multiple))
train_people = set(people_with_multiple[:split_idx])
test_people = set(people_with_multiple[split_idx:])

print(f"\nTrain set: {len(train_people)} people")
print(f"Test set: {len(test_people)} people")

# Count images in each split
train_images = sum(person_image_counts[name] for name in train_people)
test_images = sum(person_image_counts[name] for name in test_people)

print(f"\nTrain images: {train_images}")
print(f"Test images: {test_images}")

# Save the splits
output_dir = Path("/home/ben/Desktop/Ben/DeepExercise/0data")
with open(output_dir / "train_people.txt", "w") as f:
    for name in sorted(train_people):
        f.write(f"{name}\n")

with open(output_dir / "test_people.txt", "w") as f:
    for name in sorted(test_people):
        f.write(f"{name}\n")

print(f"\nSaved train_people.txt and test_people.txt to {output_dir}")
