"""
Visualize dataset statistics and generate analysis graphs.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from utils.dataset_loader import LFWDatasetLoader

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
data_dir = "/home/ben/Desktop/Ben/DeepExercise/0data/lfw2"
train_split = "/home/ben/Desktop/Ben/DeepExercise/0data/train.txt"
test_split = "/home/ben/Desktop/Ben/DeepExercise/0data/test.txt"
output_dir = Path("/home/ben/Desktop/Ben/DeepExercise/figures")
output_dir.mkdir(exist_ok=True)

# Load datasets
print("Loading train dataset...")
train_loader = LFWDatasetLoader(data_dir, train_split)
train_stats = train_loader.get_statistics()

print("\nLoading test dataset...")
test_loader = LFWDatasetLoader(data_dir, test_split)
test_stats = test_loader.get_statistics()

print("\nLoading full dataset...")
full_loader = LFWDatasetLoader(data_dir, None)
full_stats = full_loader.get_statistics()

# Print statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

print("\nFull Dataset:")
print(f"  Total people: {full_stats['total_people']}")
print(f"  Total images: {full_stats['total_images']}")
print(f"  Images per person - Min: {full_stats['min_images_per_person']}, Max: {full_stats['max_images_per_person']}")
print(f"  Images per person - Mean: {full_stats['mean_images_per_person']:.2f}, Median: {full_stats['median_images_per_person']:.1f}")
print(f"  Images per person - Std: {full_stats['std_images_per_person']:.2f}")

print("\nTrain Dataset:")
print(f"  Total people: {train_stats['total_people']}")
print(f"  Total images: {train_stats['total_images']}")
print(f"  Number of pairs: {train_stats['num_pairs']}")
print(f"  Images per person - Min: {train_stats['min_images_per_person']}, Max: {train_stats['max_images_per_person']}")
print(f"  Images per person - Mean: {train_stats['mean_images_per_person']:.2f}, Median: {train_stats['median_images_per_person']:.1f}")
print(f"  Images per person - Std: {train_stats['std_images_per_person']:.2f}")

print("\nTest Dataset:")
print(f"  Total people: {test_stats['total_people']}")
print(f"  Total images: {test_stats['total_images']}")
print(f"  Number of pairs: {test_stats['num_pairs']}")
print(f"  Images per person - Min: {test_stats['min_images_per_person']}, Max: {test_stats['max_images_per_person']}")
print(f"  Images per person - Mean: {test_stats['mean_images_per_person']:.2f}, Median: {test_stats['median_images_per_person']:.1f}")
print(f"  Images per person - Std: {test_stats['std_images_per_person']:.2f}")

print("\n" + "="*60)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('LFW-a Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Train/Test/Full comparison - Number of people and images
ax = axes[0, 0]
datasets = ['Full', 'Train', 'Test']
people_counts = [full_stats['total_people'], train_stats['total_people'], test_stats['total_people']]
image_counts = [full_stats['total_images'], train_stats['total_images'], test_stats['total_images']]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, people_counts, width, label='People', color='steelblue')
bars2 = ax.bar(x + width/2, image_counts, width, label='Images', color='coral')

ax.set_ylabel('Count')
ax.set_title('Dataset Size Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# 2. Distribution of images per person (Train)
ax = axes[0, 1]
train_dist = train_stats['images_per_person_distribution']
# Show only up to 20 images per person for clarity
max_show = 20
counts = [train_dist.get(i, 0) for i in range(1, max_show + 1)]
ax.bar(range(1, max_show + 1), counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Images per Person')
ax.set_ylabel('Number of People')
ax.set_title('Train Set: Images per Person Distribution')
ax.grid(axis='y', alpha=0.3)

# 3. Distribution of images per person (Test)
ax = axes[0, 2]
test_dist = test_stats['images_per_person_distribution']
counts = [test_dist.get(i, 0) for i in range(1, max_show + 1)]
ax.bar(range(1, max_show + 1), counts, color='coral', alpha=0.7)
ax.set_xlabel('Images per Person')
ax.set_ylabel('Number of People')
ax.set_title('Test Set: Images per Person Distribution')
ax.grid(axis='y', alpha=0.3)

# 4. Cumulative distribution
ax = axes[1, 0]
for name, stats, color in [('Train', train_stats, 'steelblue'), 
                            ('Test', test_stats, 'coral'),
                            ('Full', full_stats, 'green')]:
    dist = stats['images_per_person_distribution']
    sorted_keys = sorted(dist.keys())
    cumulative = []
    total = sum(dist.values())
    cum_sum = 0
    for key in sorted_keys:
        cum_sum += dist[key]
        cumulative.append(cum_sum / total * 100)
    ax.plot(sorted_keys, cumulative, marker='o', label=name, color=color, alpha=0.7)

ax.set_xlabel('Images per Person')
ax.set_ylabel('Cumulative Percentage of People (%)')
ax.set_title('Cumulative Distribution')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)

# 5. Statistics comparison table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Full', 'Train', 'Test'],
    ['People', f"{full_stats['total_people']}", f"{train_stats['total_people']}", f"{test_stats['total_people']}"],
    ['Images', f"{full_stats['total_images']}", f"{train_stats['total_images']}", f"{test_stats['total_images']}"],
    ['Min imgs/person', f"{full_stats['min_images_per_person']}", f"{train_stats['min_images_per_person']}", f"{test_stats['min_images_per_person']}"],
    ['Max imgs/person', f"{full_stats['max_images_per_person']}", f"{train_stats['max_images_per_person']}", f"{test_stats['max_images_per_person']}"],
    ['Mean imgs/person', f"{full_stats['mean_images_per_person']:.2f}", f"{train_stats['mean_images_per_person']:.2f}", f"{test_stats['mean_images_per_person']:.2f}"],
    ['Median imgs/person', f"{full_stats['median_images_per_person']:.1f}", f"{train_stats['median_images_per_person']:.1f}", f"{test_stats['median_images_per_person']:.1f}"],
    ['Std imgs/person', f"{full_stats['std_images_per_person']:.2f}", f"{train_stats['std_images_per_person']:.2f}", f"{test_stats['std_images_per_person']:.2f}"],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax.set_title('Statistics Summary', fontweight='bold', pad=20)

# 6. People with multiple images
ax = axes[1, 2]
thresholds = [2, 3, 4, 5, 10, 15, 20]
train_counts = [len(train_loader.get_people_with_min_images(t)) for t in thresholds]
test_counts = [len(test_loader.get_people_with_min_images(t)) for t in thresholds]

x = np.arange(len(thresholds))
width = 0.35

ax.bar(x - width/2, train_counts, width, label='Train', color='steelblue', alpha=0.7)
ax.bar(x + width/2, test_counts, width, label='Test', color='coral', alpha=0.7)

ax.set_xlabel('Minimum Images per Person')
ax.set_ylabel('Number of People')
ax.set_title('People with N+ Images')
ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to {output_dir / 'dataset_analysis.png'}")

# Create additional detailed distribution plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Detailed Images per Person Distribution', fontsize=14, fontweight='bold')

for idx, (name, stats, color) in enumerate([('Full Dataset', full_stats, 'green'),
                                              ('Train Set', train_stats, 'steelblue'), 
                                              ('Test Set', test_stats, 'coral')]):
    ax = axes[idx]
    dist = stats['images_per_person_distribution']
    
    if len(dist) > 0:
        # Show full distribution
        max_images = max(dist.keys())
        x_vals = list(range(1, min(max_images + 1, 51)))  # Cap at 50 for visibility
        y_vals = [dist.get(i, 0) for i in x_vals]
        
        ax.bar(x_vals, y_vals, color=color, alpha=0.7)
        ax.set_xlabel('Images per Person')
        ax.set_ylabel('Number of People')
        ax.set_title(f'{name}\n(Total: {stats["total_people"]} people, {stats["total_images"]} images)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        if stats['mean_images_per_person'] > 0:
            ax.axvline(stats['mean_images_per_person'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {stats["mean_images_per_person"]:.2f}')
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'{name}\n(No data)')

plt.tight_layout()
plt.savefig(output_dir / 'detailed_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved detailed distribution to {output_dir / 'detailed_distribution.png'}")

print("\nVisualization complete!")

