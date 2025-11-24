import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


def load_dataset(dataset_file: str):
    """Load dataset from JSON file."""
    with open(dataset_file, 'r') as f:
        return json.load(f)


def analyze_dataset(dataset):
    """Extract statistics from dataset."""
    pairs = dataset['pairs']
    
    # Extract unique people and images
    people = set()
    images = set()
    person_pair_count = Counter()
    
    for pair in pairs:
        img1, img2 = pair['image1'], pair['image2']
        images.add(img1)
        images.add(img2)
        
        person1 = Path(img1).parent.name
        person2 = Path(img2).parent.name
        people.add(person1)
        people.add(person2)
        person_pair_count[person1] += 1
        person_pair_count[person2] += 1
    
    freq_values = list(person_pair_count.values())
    
    return {
        'name': dataset['name'],
        'num_pairs': dataset['num_pairs'],
        'num_positive': dataset['num_positive'],
        'num_negative': dataset['num_negative'],
        'balance': dataset['balance'],
        'num_unique_people': len(people),
        'num_unique_images': len(images),
        'avg_pairs_per_person': np.mean(freq_values) if freq_values else 0,
        'std_pairs_per_person': np.std(freq_values) if freq_values else 0,
        'min_pairs_per_person': min(freq_values) if freq_values else 0,
        'max_pairs_per_person': max(freq_values) if freq_values else 0,
    }


def print_stats(stats):
    """Print statistics for a dataset."""
    print(f"\n{stats['name'].upper()} Dataset:")
    print(f"  Total Pairs:       {stats['num_pairs']:>6,}")
    print(f"  Positive (same):   {stats['num_positive']:>6,} ({stats['num_positive']/stats['num_pairs']*100:.1f}%)")
    print(f"  Negative (diff):   {stats['num_negative']:>6,} ({stats['num_negative']/stats['num_pairs']*100:.1f}%)")
    print(f"  Unique People:     {stats['num_unique_people']:>6,}")
    print(f"  Unique Images:     {stats['num_unique_images']:>6,}")
    print(f"  Pairs per person:  {stats['avg_pairs_per_person']:>6.2f} ± {stats['std_pairs_per_person']:.2f}")


def plot_bar(ax, datasets, values, colors, ylabel, title, value_format='{:,}'):
    """Helper to create bar chart."""
    bars = ax.bar(datasets, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = value_format.format(val) if isinstance(val, (int, float)) else value_format.format(int(val))
        ax.text(bar.get_x() + bar.get_width()/2., height, label,
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    return bars


def plot_grouped_bar(ax, datasets, group1, group2, colors, ylabel, title, labels):
    """Helper to create grouped bar chart."""
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, group1, width, label=labels[0], 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, group2, width, label=labels[1], 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    return bars1, bars2




def create_summary_table(ax, all_stats, total_pairs, total_positive, total_negative):
    """Helper to create summary statistics table."""
    ax.axis('off')

    table_data = [
            ['Metric', 'Train', 'Val', 'Test', 'Combined'],
            ['Total Pairs'] + [f"{s['num_pairs']:,}" for s in all_stats] + [f"{total_pairs:,}"],
            ['Positive'] + [f"{s['num_positive']:,}" for s in all_stats] + [f"{total_positive:,}"],
            ['Negative'] + [f"{s['num_negative']:,}" for s in all_stats] + [f"{total_negative:,}"],
            ['Unique People'] + [f"{s['num_unique_people']:,}" for s in all_stats] + ['-'],
            ['Unique Images'] + [f"{s['num_unique_images']:,}" for s in all_stats] + ['-'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

    # Style first column
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')

    ax.set_title('Dataset Statistics Summary', fontsize=13, fontweight='bold', pad=20)


def visualize_datasets(datasets_dir: str, output_dir: str):
    """Create comprehensive visualization of all datasets."""
    datasets_path = Path(datasets_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Dataset Visualization")
    print("=" * 80)
    
    # Load and analyze all datasets
    datasets_names = ['train', 'val', 'test']
    all_data = [load_dataset(datasets_path / f'{name}_dataset.json') for name in datasets_names]
    all_stats = [analyze_dataset(data) for data in all_data]
    
    # Print statistics
    for stats in all_stats:
        print_stats(stats)
    
    # Combined statistics
    total_pairs = sum(s['num_pairs'] for s in all_stats)
    total_positive = sum(s['num_positive'] for s in all_stats)
    total_negative = sum(s['num_negative'] for s in all_stats)
    
    print(f"\nCOMBINED (Train+Val+Test):")
    print(f"  Total Pairs:       {total_pairs:>6,}")
    print(f"  Positive (same):   {total_positive:>6,} ({total_positive/total_pairs*100:.1f}%)")
    print(f"  Negative (diff):   {total_negative:>6,} ({total_negative/total_pairs*100:.1f}%)")
    
    # Create visualization (3x3 grid)
    fig = plt.figure(figsize=(18, 12))
    datasets = ['Train', 'Val', 'Test']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Row 1: Basic metrics
    plot_bar(plt.subplot(3, 3, 1), datasets, 
             [s['num_pairs'] for s in all_stats], colors,
             'Number of Pairs', 'Total Pairs per Dataset')
    
    plot_grouped_bar(plt.subplot(3, 3, 2), datasets,
                     [s['num_positive'] for s in all_stats],
                     [s['num_negative'] for s in all_stats],
                     ['#27ae60', '#c0392b'],
                     'Number of Pairs', 'Positive vs Negative Pairs',
                     ['Positive (same)', 'Negative (diff)'])
    
    ax3 = plt.subplot(3, 3, 3)
    plot_bar(ax3, datasets, [s['balance'] * 100 for s in all_stats], colors,
             'Positive Ratio (%)', 'Dataset Balance', '{:.1f}%')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Perfect Balance')
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=10)
    
    # Row 2: Detailed metrics
    plot_bar(plt.subplot(3, 3, 4), datasets,
             [s['num_unique_people'] for s in all_stats], colors,
             'Number of People', 'Unique People per Dataset')
    
    plot_bar(plt.subplot(3, 3, 5), datasets,
             [s['num_unique_images'] for s in all_stats], colors,
             'Number of Images', 'Unique Images per Dataset')
    
    plot_bar(plt.subplot(3, 3, 6), datasets,
             [s['avg_pairs_per_person'] for s in all_stats], colors,
             'Avg Pairs per Person', 'Average Pairs per Person', '{:.2f}')
    
    # Row 3: Summary table and combined view
    create_summary_table(plt.subplot(3, 3, 7), all_stats, 
                        total_pairs, total_positive, total_negative)
    
    # All splits comparison
    ax8 = plt.subplot(3, 3, 8)
    all_labels = ['Train\nPos', 'Train\nNeg', 'Val\nPos', 'Val\nNeg', 'Test\nPos', 'Test\nNeg']
    all_values = [all_stats[0]['num_positive'], all_stats[0]['num_negative'],
                  all_stats[1]['num_positive'], all_stats[1]['num_negative'],
                  all_stats[2]['num_positive'], all_stats[2]['num_negative']]
    all_colors = ['#27ae60', '#c0392b'] * 3
    ax8.bar(all_labels, all_values, color=all_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax8.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
    ax8.set_title('All Splits: Positive vs Negative', fontsize=13, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Combined statistics
    ax9 = plt.subplot(3, 3, 9)
    combined_labels = ['Positive\n(same)', 'Negative\n(diff)']
    combined_values = [total_positive, total_negative]
    combined_colors = ['#27ae60', '#c0392b']
    bars = ax9.bar(combined_labels, combined_values, color=combined_colors, 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax9.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
    ax9.set_title('Combined Dataset (Train+Val+Test)', fontsize=13, fontweight='bold')
    ax9.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, combined_values):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({val/total_pairs*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('Dataset Statistics: Train / Val / Test', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'dataset_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_file}")
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    visualize_datasets(
        datasets_dir='/home/ben/Desktop/Ben/DeepExercise/0data/datasets',
        output_dir='/home/ben/Desktop/Ben/DeepExercise/0outputs/figures'
    )
