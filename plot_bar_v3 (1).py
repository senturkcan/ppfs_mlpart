import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['kNN', 'AB', 'SVM', 'NB', 'RF', 'DT']
datasets = ['Beans', 'Divorce', 'Parkinsons', 'Rice', 'WDBC']

data = {
    'Beans': {
        'MI-Corr': [-6.6, -12.0, -5.7, -12.3, -6.7, -5.8],
        'MI':      [-1.2, -3.8, -1.5, -1.2, -1.8, -0.4],
        'Gini':    [-1.2, -3.8, -1.5, -1.2, -1.8, -0.4]
    },
    'Divorce': {
        'MI-Corr': [0.0, 0.0, 0.0, 0.6, 0.0, 1.8],
        'MI':      [0.0, 0.6, 0.0, 1.2, 0.0, -0.6],
        'Gini':    [0.0, 0.6, 0.0, 1.2, 0.0, -0.6]
    },
    'Parkinsons': {
        'MI-Corr': [-2.5, -2.4, -0.3, -3.5, -0.7, -3.6],
        'MI':      [-2.2, -7.7, -2.2, 4.5, -4.2, -3.0],
        'Gini':    [-7.9, -9.8, 2.6, 0.0, -7.8, -9.7]
    },
    'Rice': {
        'MI-Corr': [-0.5, -1.4, -0.9, -5.9, 1.7, 3.4],
        'MI':      [0.3, 0.0, -0.6, 0.3, 1.4, 3.0],
        'Gini':    [0.3, 0.0, -0.6, 0.3, 1.4, 3.0]
    },
    'WDBC': {
        'MI-Corr': [-4.3, -3.6, -4.5, -6.0, -4.3, -4.0],
        'MI':      [-2.3, -2.0, -2.7, 0.5, -1.9, 1.4],
        'Gini':    [-2.3, -2.0, -2.7, 0.5, -1.9, 1.4]
    }
}

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.0

# Plotting
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(7, 10), dpi=300)
bar_width = 0.21
x = np.arange(len(models))

handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    y_micorr = data[dataset]['MI-Corr']
    y_mi = data[dataset]['MI']
    y_gini = data[dataset]['Gini']
    
    # Create bars
    rects1 = ax.bar(x - bar_width, y_micorr, width=bar_width, label='MI-Corr', color='#1f77b4', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, y_mi, width=bar_width, label='MI', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + bar_width, y_gini, width=bar_width, label='Gini', color='#2ca02c', edgecolor='black', linewidth=0.5)
    
    # Collect handles for legend
    if i == 0:
        handles = [rects1, rects2, rects3]
        labels = ['MI-Corr', 'MI', 'Gini']

    # Check for "No Change" (all three are 0) and annotate
    for j in range(len(models)):
        val1 = y_micorr[j]
        val2 = y_mi[j]
        val3 = y_gini[j]
        
        if val1 == 0 and val2 == 0 and val3 == 0:
            ax.text(x[j], 0, "No Change", ha='center', va='center', 
                   rotation=90, fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", 
                            ec="gray", alpha=0.9, linewidth=0.5))

    # Formatting
    ax.set_title(f'{dataset} Dataset', fontsize=11, fontweight='semibold', 
                pad=8, loc='left')
    ax.set_ylabel('Balanced Accuracy Change (%)', fontsize=8)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.axhline(0, color='black', linewidth=1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    
    # Dynamic y-limits with minimum range check
    all_values = y_micorr + y_mi + y_gini
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Ensure there is enough vertical space if values are close to 0
    if max_val - min_val < 2: 
        max_val = 1.0
        min_val = -1.0
        
    margin = (max_val - min_val) * 0.15 
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add single global legend at the top
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
          ncol=3, fontsize=10, frameon=True, fancybox=False, 
          edgecolor='black', framealpha=1)
# Adjust layout: leave space at the top for title and legend
plt.subplots_adjust(top=0.91, hspace=0.4, bottom=0.05, left=0.1, right=0.95)

plt.savefig('compact_bar_graphs_single_legend.png', dpi=300, bbox_inches='tight')
plt.savefig('compact_bar_graphs_single_legend.pdf', bbox_inches='tight')
