import pandas as pd
import numpy as np
from scipy import stats
import itertools
import ast

def load_itpc_data(csv_path):
    """
    Load ITPC results and organize by condition
    
    Parameters:
    -----------
    csv_path : str
        Path to the ITPC results CSV file
    
    Returns:
    --------
    dict
        Organized ITPC data by condition
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Organize data by condition
    conditions = df['Condition'].unique()
    processed_data = {}
    
    for condition in conditions:
        # Filter data for the current condition
        condition_df = df[df['Condition'] == condition].copy()
        
        # Convert ITPC to numeric values
        condition_df['ITPC'] = condition_df['ITPC'].apply(
            lambda x: np.mean(ast.literal_eval(x)) if isinstance(x, str) else x
        )
        
        # Group by electrode and compute mean ITPC
        electrode_data = condition_df.groupby('Electrode')['ITPC'].mean().values
        
        processed_data[condition] = electrode_data
    
    return processed_data

def perform_wilcoxon_tests(itpc_data):
    """
    Perform Wilcoxon signed-rank tests between all condition pairs
    
    Parameters:
    -----------
    itpc_data : dict
        ITPC data organized by condition
    
    Returns:
    --------
    dict
        Statistical test results
    """
    # Get all condition combinations
    conditions = list(itpc_data.keys())
    condition_pairs = list(itertools.combinations(conditions, 2))
    
    # Store results
    test_results = {}
    
    # Perform Wilcoxon signed-rank test for each pair
    for condition1, condition2 in condition_pairs:
        # Ensure equal lengths by taking the minimum length
        min_len = min(len(itpc_data[condition1]), len(itpc_data[condition2]))
        data1 = itpc_data[condition1][:min_len]
        data2 = itpc_data[condition2][:min_len]
        
        # Perform two-sided Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(data1, data2)
        
        # Determine significance level
        if p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'
        
        test_results[(condition1, condition2)] = {
            'statistic': statistic,
            'p_value': p_value,
            'significance': significance,
            'mean1': np.mean(data1),
            'std1': np.std(data1),
            'mean2': np.mean(data2),
            'std2': np.std(data2)
        }
    
    return test_results

def print_statistical_summary(test_results):
    """
    Print formatted statistical summary
    
    Parameters:
    -----------
    test_results : dict
        Results from Wilcoxon signed-rank tests
    """
    print("\nStatistical Significance (Wilcoxon Signed-Rank Test):")
    print("-" * 50)
    
    condition_full_names = {
        'GN': 'Grammatical Nonsensical',
        'GS': 'Grammatical Sensical',
        'UN': 'Ungrammatical Nonsensical',
        'US': 'Ungrammatical Sensical'
    }
    
    for (condition1, condition2), result in test_results.items():
        print(f"{condition_full_names[condition1]} vs {condition_full_names[condition2]}:")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.4f}")
        print(f"  Significance: {result['significance']}")
        
        # Descriptive statistics
        print(f"\n  Descriptive Statistics for {condition_full_names[condition1]}:")
        print(f"    Mean ITPC: {result['mean1']:.4f}")
        print(f"    Standard Deviation: {result['std1']:.4f}")
        
        print(f"\n  Descriptive Statistics for {condition_full_names[condition2]}:")
        print(f"    Mean ITPC: {result['mean2']:.4f}")
        print(f"    Standard Deviation: {result['std2']:.4f}")
        
        print()  # Extra line for readability

def main():
    # Path to the ITPC results CSV
    csv_path = 'data/csv_files/itpc_results.csv'
    
    # Load ITPC data
    itpc_data = load_itpc_data(csv_path)
    
    # Perform statistical tests
    test_results = perform_wilcoxon_tests(itpc_data)
    
    # Print summary
    print_statistical_summary(test_results)

if __name__ == '__main__':
    main()