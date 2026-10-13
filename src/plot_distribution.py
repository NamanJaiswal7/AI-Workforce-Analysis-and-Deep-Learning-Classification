import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

# File paths
TRAIN_FILE = "data/processed/train.csv"
ISCO_FILE = "data/raw/isco08_en.csv"
OUTPUT_DIR = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fig_4_3_1_distribution.png")

def generate_plot():
    print(f"Loading data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)
    
    # Extract labels. Since it's a multi-label dataset, labels are comma-separated strings
    all_labels = []
    for labels_str in df["labels"].dropna():
        # Split by comma and strip
        labels = [l.strip() for l in str(labels_str).split(",") if l.strip()]
        all_labels.extend(labels)
        
    # Count frequencies
    label_counts = Counter(all_labels)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['count'])
    plot_df.index.name = 'isco_code'
    plot_df = plot_df.reset_index()
    
    # Sort by count descending
    plot_df = plot_df.sort_values('count', ascending=False)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    bars = plt.bar(plot_df['isco_code'], plot_df['count'], color='#4C72B0', edgecolor='black')
    
    # Add title and labels
    plt.title('Distribution of ISCO-08 2-digit codes across the training dataset', fontsize=16, pad=15)
    plt.xlabel('ISCO-08 2-digit group code', fontsize=12)
    plt.ylabel('Number of Study Programs', fontsize=12)
    
    # Customize x-axis
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight top 3 (which are likely 25, 21, 24 based on our dataset script)
    for i in range(min(3, len(bars))):
        bars[i].set_color('#C44E52')
        bars[i].set_edgecolor('black')
        
    # Add a legend for the highlighted bars
    import matplotlib.patches as mpatches
    top_patch = mpatches.Patch(color='#C44E52', label='Top 3 Dominant Groups (ICT/Business/Science)')
    other_patch = mpatches.Patch(color='#4C72B0', label='Other Groups')
    plt.legend(handles=[top_patch, other_patch])
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_plot()
