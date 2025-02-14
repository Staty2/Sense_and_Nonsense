import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def parse_itpc_list(itpc_str):
    """
    Parse ITPC string representation to list of floats
    
    Parameters:
    -----------
    itpc_str : str
        String representation of ITPC values
    
    Returns:
    --------
    list
        Parsed ITPC values
    """
    try:
        # If it's already a list, return it
        if isinstance(itpc_str, list):
            return itpc_str
        
        # Try to evaluate the string as a list
        parsed = ast.literal_eval(itpc_str)
        return parsed
    except:
        # If parsing fails, return an empty list
        print(f"Warning: Could not parse ITPC value: {itpc_str}")
        return []

def load_and_process_itpc_data(csv_path):
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
        
        # Parse ITPC values
        condition_df['parsed_itpc'] = condition_df['ITPC'].apply(parse_itpc_list)
        
        # Group by electrode and compute mean ITPC
        electrode_data = condition_df.groupby('Electrode')['parsed_itpc'].apply(
            lambda x: np.mean(x.tolist(), axis=0)
        ).tolist()
        
        processed_data[condition] = np.array(electrode_data)
    
    return processed_data

def visualize_itpc(csv_path, output_path='itpc_visualization.png'):
    """
    Create detailed visualization of ITPC results
    
    Parameters:
    -----------
    csv_path : str
        Path to the ITPC results CSV file
    output_path : str, optional
        Path to save the output visualization
    """
    # Set up the plot with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Inter-Trial Phase Coherence (ITPC) Across Frequencies', fontsize=20)
    
    # Color and label details for each condition
    condition_details = {
        'GN': {
            'color': '#1E90FF',  # Dodger Blue
            'label': 'Grammatical Nonsensical',
            'ax': axs[0, 0]
        },
        'GS': {
            'color': '#32CD32',  # Lime Green
            'label': 'Grammatical Sensical',
            'ax': axs[0, 1]
        },
        'UN': {
            'color': '#FF4500',  # Orange Red
            'label': 'Ungrammatical Nonsensical',
            'ax': axs[1, 0]
        },
        'US': {
            'color': '#9400D3',  # Violet
            'label': 'Ungrammatical Sensical',
            'ax': axs[1, 1]
        }
    }
    
    # Load and process ITPC data
    itpc_data = load_and_process_itpc_data(csv_path)
    
    # Plot for each condition
    for condition, details in condition_details.items():
        # Get condition data
        condition_traces = itpc_data[condition]
        
        # Plot individual traces
        for trace in condition_traces:
            details['ax'].plot(
                range(len(trace)), 
                trace, 
                color=details['color'], 
                alpha=0.1  # Very transparent for individual traces
            )
        
        # Compute and plot mean trace
        mean_trace = np.mean(condition_traces, axis=0)
        details['ax'].plot(
            range(len(mean_trace)), 
            mean_trace, 
            color='black', 
            linewidth=3, 
            label='Mean Trace'
        )
        
        # Compute and display statistical summary
        std_trace = np.std(condition_traces, axis=0)
        details['ax'].fill_between(
            range(len(mean_trace)), 
            mean_trace - std_trace, 
            mean_trace + std_trace, 
            color=details['color'], 
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        # Formatting for each subplot
        details['ax'].set_title(details['label'], fontsize=16)
        details['ax'].set_xlabel('Frequency Index', fontsize=12)
        details['ax'].set_ylabel('ITPC', fontsize=12)
        details['ax'].grid(True, linestyle='--', alpha=0.7)
        details['ax'].legend()
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ITPC visualization saved to {output_path}")
    
    # Additional statistical summary
    print("\nITPC Statistical Summary:")
    for condition, traces in itpc_data.items():
        print(f"\n{condition_details[condition]['label']}:")
        mean_trace = np.mean(traces, axis=0)
        std_trace = np.std(traces, axis=0)
        print(f"  Overall Mean ITPC: {np.mean(mean_trace):.4f}")
        print(f"  Overall ITPC Std Dev: {np.mean(std_trace):.4f}")
        print(f"  Min Mean ITPC: {np.min(mean_trace):.4f}")
        print(f"  Max Mean ITPC: {np.max(mean_trace):.4f}")

def main():
    # Path to the ITPC results CSV
    csv_path = 'data/csv_files/itpc_results.csv'
    
    # Create visualization
    visualize_itpc(csv_path)

if __name__ == '__main__':
    main()