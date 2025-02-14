import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def create_pvalue_heatmap(pvalues):
    """
    Create a color-coded heatmap of p-values between conditions
    
    Parameters:
    -----------
    pvalues : dict
        Dictionary of p-values between conditions
    """
    # Condition names
    conditions = ['GN', 'GS', 'UN', 'US']
    
    # Create a matrix to store p-values
    pvalue_matrix = np.zeros((len(conditions), len(conditions)))
    
    # Populate the matrix
    for (cond1, cond2), p_value in pvalues.items():
        i = conditions.index(cond1)
        j = conditions.index(cond2)
        pvalue_matrix[i, j] = p_value
        pvalue_matrix[j, i] = p_value  # symmetric
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Custom color map with extreme sensitivity
    # Use a diverging colormap that emphasizes low p-values
    def custom_colormap():
        # Create a custom colormap that makes significant differences VERY clear
        colors = [
            '#FFFFFF',   # White for high p-values
            '#FFE5E5',   # Very light red
            '#FF9999',   # Light red
            '#FF4444',   # Bright red
            '#FF0000',   # Pure red
            '#AA0000',   # Dark red
            '#660000'    # Very dark red
        ]
        return mcolors.LinearSegmentedColormap.from_list('custom_pvalue', colors, N=100)
    
    # Create custom color normalization to emphasize low p-values
    def custom_norm():
        # This will compress the color scale to make low p-values stand out
        return mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=0.05)
    
    # Create heatmap
    ax = sns.heatmap(pvalue_matrix, 
                     annot=True, 
                     cmap=custom_colormap(), 
                     norm=custom_norm(),
                     cbar_kws={'label': 'p-value'},
                     xticklabels=conditions, 
                     yticklabels=conditions,
                     vmin=0, 
                     vmax=0.05,
                     fmt='.4f',
                     linewidths=0.5,
                     linecolor='lightgray')
    
    plt.title('P-values Between Linguistic Conditions', fontsize=16)
    plt.xlabel('Conditions', fontsize=12)
    plt.ylabel('Conditions', fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('condition_pvalue_heatmap.png', dpi=300)
    plt.close()

def main():
    # P-values from the statistical analysis
    pvalues = {
        ('GN', 'GS'): 0.0000,
        ('GN', 'UN'): 0.0007,
        ('GN', 'US'): 0.0044,
        ('GS', 'UN'): 0.0022,
        ('GS', 'US'): 0.0004,
        ('UN', 'US'): 0.6645
    }
    
    # Condition full names for reference
    condition_names = {
        'GN': 'Grammatical Nonsensical',
        'GS': 'Grammatical Sensical',
        'UN': 'Ungrammatical Nonsensical',
        'US': 'Ungrammatical Sensical'
    }
    
    # Create the heatmap
    create_pvalue_heatmap(pvalues)
    
    # Print detailed comparison for reference
    print("P-value Comparisons:")
    for (cond1, cond2), p_value in pvalues.items():
        print(f"{condition_names[cond1]} vs {condition_names[cond2]}:")
        print(f"  p-value: {p_value:.4f}")
        
        # Significance interpretation
        if p_value < 0.01:
            significance = "** (Highly Significant)"
        elif p_value < 0.05:
            significance = "* (Significant)"
        else:
            significance = "ns (Not Significant)"
        
        print(f"  Significance: {significance}\n")

if __name__ == '__main__':
    main()