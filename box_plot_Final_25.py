import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def load_and_process_itpc_data(csv_path):
    """
    Load ITPC results and prepare for visualization
    
    Parameters:
    -----------
    csv_path : str
        Path to the ITPC results CSV file
    
    Returns:
    --------
    pd.DataFrame
        Processed ITPC data
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert ITPC to numeric values
    def parse_itpc(x):
        try:
            # If it's a string, evaluate and take mean
            if isinstance(x, str):
                parsed = ast.literal_eval(x)
                return np.mean(parsed)
            # If it's already a number, return it
            return x
        except:
            print(f"Warning: Could not parse ITPC value: {x}")
            return np.nan
    
    # Create a copy and process ITPC values
    processed_df = df.copy()
    processed_df['ITPC'] = processed_df['ITPC'].apply(parse_itpc)
    
    return processed_df

def create_itpc_boxplot(csv_path, output_path='itpc_boxplot.png'):
    """
    Create a boxplot of ITPC values for different conditions
    
    Parameters:
    -----------
    csv_path : str
        Path to the ITPC results CSV file
    output_path : str, optional
        Path to save the output visualization
    """
    # Load and process data
    df = load_and_process_itpc_data(csv_path)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Color palette
    color_palette = {
        'UN': '#FF4500',  # Orange Red
        'US': '#9400D3',  # Violet
        'GN': '#1E90FF',  # Dodger Blue
        'GS': '#32CD32'   # Lime Green
    }
    
    # Create condition full names
    condition_names = {
        'UN': 'Ungrammatical\nNonsensical',
        'US': 'Ungrammatical\nSensical',
        'GN': 'Grammatical\nNonsensical',
        'GS': 'Grammatical\nSensical'
    }
    
    # Create boxplot
    sns.boxplot(
        x='Condition', 
        y='ITPC', 
        data=df, 
        palette=[color_palette[cond] for cond in df['Condition'].unique()]
    )
    
    # Add swarmplot for individual data points
    sns.swarmplot(
        x='Condition', 
        y='ITPC', 
        data=df, 
        color='black', 
        alpha=0.5,
        size=3
    )
    
    # Customize the plot
    plt.title('Inter-Trial Phase Coherence (ITPC) by Condition', fontsize=16)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('ITPC', fontsize=12)
    
    # Replace condition labels with full names
    current_labels = plt.gca().get_xticklabels()
    plt.gca().set_xticklabels([condition_names[label.get_text()] for label in current_labels])
    
    # Add statistical annotation
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ITPC boxplot saved to {output_path}")
    
    # Compute and print summary statistics
    print("\nITPC Summary Statistics:")
    summary = df.groupby('Condition')['ITPC'].agg(['mean', 'std', 'min', 'max'])
    print(summary)

def main():
    # Path to the ITPC results CSV
    csv_path = 'data/csv_files/itpc_results.csv'
    
    # Create visualization
    create_itpc_boxplot(csv_path)

if __name__ == '__main__':
    main()