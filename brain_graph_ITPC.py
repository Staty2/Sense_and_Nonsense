import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import ast
from scipy import stats
from datetime import datetime

# Standard 10-20 electrode locations in MNI coordinates
ELECTRODE_LOCATIONS = {
    'Fp1': [-42, 72, -4],
    'Fp2': [42, 72, -4],
    'F7': [-54, 18, -10],
    'F3': [-42, 18, 46],
    'Fz': [0, 20, 46],
    'F4': [42, 18, 46],
    'F8': [54, 18, -10],
    'FC5': [-54, -10, 36],
    'FC1': [-21, -10, 46],
    'FC2': [21, -10, 46],
    'FC6': [54, -10, 36],
    'T7': [-60, -20, 0],
    'C3': [-42, -20, 46],
    'Cz': [0, -20, 46],
    'C4': [42, -20, 46],
    'T8': [60, -20, 0],
    'TP9': [-60, -40, 0],
    'CP5': [-54, -40, 36],
    'CP1': [-21, -40, 46],
    'CP2': [21, -40, 46],
    'CP6': [54, -40, 36],
    'TP10': [60, -40, 0],
    'P7': [-54, -60, 0],
    'P3': [-42, -60, 46],
    'Pz': [0, -60, 46],
    'P4': [42, -60, 46],
    'P8': [54, -60, 0],
    'PO9': [-42, -78, 36],
    'O1': [-42, -82, 0],
    'Oz': [0, -82, 0],
    'O2': [42, -82, 0],
    'PO10': [42, -78, 36]
}

def load_and_process_itpc_data(csv_path, phrase_freq_index):
    """
    Load ITPC results and extract values at phrase frequency
    
    Parameters:
    -----------
    csv_path : str
        Path to the ITPC results CSV file
    phrase_freq_index : int
        Index of the phrase frequency
    
    Returns:
    --------
    dict
        ITPC values for each condition at phrase frequency
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Process ITPC values
    def extract_phrase_freq_itpc(x):
        try:
            # If it's a string, evaluate and extract phrase frequency ITPC
            if isinstance(x, str):
                parsed = ast.literal_eval(x)
                return parsed[phrase_freq_index]
            # If it's already a number, return it
            return x
        except:
            print(f"Warning: Could not parse ITPC value: {x}")
            return np.nan
    
    # Create a copy and process ITPC values
    processed_df = df.copy()
    processed_df['ITPC'] = processed_df['ITPC'].apply(extract_phrase_freq_itpc)
    
    # Organize data by condition
    conditions = processed_df['Condition'].unique()
    condition_data = {}
    
    for condition in conditions:
        condition_subset = processed_df[processed_df['Condition'] == condition]
        
        # Create dictionary mapping electrode names to ITPC values
        condition_itpc = {}
        for _, row in condition_subset.iterrows():
            # Assume Electrode column contains electrode names
            condition_itpc[row['Electrode']] = row['ITPC']
        
        condition_data[condition] = condition_itpc
    
    return condition_data

def create_brain_topography(condition1, condition2, phrase_freq, frequencies):
    """
    Create brain topography for ITPC differences
    
    Parameters:
    -----------
    condition1 : str
        First condition to compare
    condition2 : str
        Second condition to compare
    phrase_freq : float
        Phrase frequency
    frequencies : list
        List of all frequencies
    
    Returns:
    --------
    tuple
        Prepared data for visualization
    """
    # Find phrase frequency index
    phrase_freq_index = np.argmin(np.abs(np.array(frequencies) - phrase_freq))
    
    # Load ITPC data
    csv_path = 'data/csv_files/itpc_results.csv'
    condition_data = load_and_process_itpc_data(csv_path, phrase_freq_index)
    
    # Prepare data for brain visualization
    # Compute ITPC difference
    itpc_diff = {}
    significant_electrodes = []
    
    for electrode, coords in ELECTRODE_LOCATIONS.items():
        val1 = condition_data[condition1].get(electrode, np.nan)
        val2 = condition_data[condition2].get(electrode, np.nan)
        
        if not (np.isnan(val1) or np.isnan(val2)):
            # Compute difference
            diff = val1 - val2
            itpc_diff[electrode] = diff
            
            # Perform statistical test
            _, p_value = stats.ttest_ind([val1], [val2])
            if p_value < 0.05:
                significant_electrodes.append(electrode)
    
    return itpc_diff, significant_electrodes

def visualize_brain_itpc(condition1, condition2, frequencies):
    """
    Visualize ITPC differences on a brain surface
    
    Parameters:
    -----------
    condition1 : str
        First condition to compare
    condition2 : str
        Second condition to compare
    frequencies : list
        List of frequencies
    """
    # Phrase frequency
    phrase_freq = 3.125
    
    # Create topography data
    itpc_diff, significant_electrodes = create_brain_topography(
        condition1, condition2, phrase_freq, frequencies
    )
    
    # Prepare data for MNE visualization
    # Get coordinates and values for all electrodes
    ch_names = list(ELECTRODE_LOCATIONS.keys())
    ch_coords = np.array([ELECTRODE_LOCATIONS[ch] for ch in ch_names])
    
    # Create data array for visualization
    data = np.array([itpc_diff.get(ch, 0) for ch in ch_names])
    
    # Create a custom colormap highlighting significant electrodes
    import matplotlib.colors as mcolors
    
    # Generate figure
    plt.figure(figsize=(15, 10))
    
    # Create subplot for brain surface
    brain = mne.viz.create_3d_figure(size=(800, 800), bgcolor='white')
    
    # Plot brain surface
    brain_mesh = mne.viz.plot_brain_surface(
        subjects_dir=None,  # Use default
        subject='fsaverage',  # Standard brain template
        hemi='both',  # Both hemispheres
        surf='inflated',  # Inflated brain surface
        figure=brain
    )
    
    # Plot electrode locations with color intensity
    for i, (ch, coord) in enumerate(zip(ch_names, ch_coords)):
        color = 'red' if ch in significant_electrodes else 'gray'
        size = 10 if ch in significant_electrodes else 5
        brain.add_point(coord, color=color, scale_factor=size)
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'data/csv_files/brain_itpc_{condition1}_vs_{condition2}_{timestamp}.png'
    brain.save_image(output_path)
    
    print(f"Brain ITPC topography saved to {output_path}")
    plt.close()

def main():
    # Frequencies (from the original script)
    frequencies = [
        0.260416666666667, 0.325520833333333, 0.390625, 0.455729166666667, 
        0.520833333333333, 0.5859375, 0.651041666666667, 0.716145833333333, 
        0.78125, 0.846354166666667, 0.911458333333333, 0.9765625, 
        1.04166666666667, 1.10677083333333, 1.171875, 1.23697916666667, 
        1.30208333333333, 1.3671875, 1.43229166666667, 1.49739583333333, 
        1.5625, 1.62760416666667, 1.69270833333333, 1.7578125, 
        1.82291666666667, 1.88802083333333, 1.953125, 2.01822916666667, 
        2.08333333333333, 2.1484375, 2.21354166666667, 2.27864583333333, 
        2.34375, 2.40885416666667, 2.47395833333333, 2.5390625, 
        2.60416666666667, 2.66927083333333, 2.734375, 2.79947916666667, 
        2.86458333333333, 2.9296875, 2.99479166666667, 3.05989583333333, 
        3.125, 3.19010416666667, 3.25520833333333, 3.3203125, 
        3.38541666666667, 3.45052083333333, 3.515625, 3.58072916666667, 
        3.64583333333333, 3.7109375, 3.77604166666667, 3.84114583333333, 
        3.90625, 3.97135416666667
    ]
    
    # Condition pairs to visualize
    condition_pairs = [
        ('GN', 'GS'), ('GN', 'UN'), ('GN', 'US'),
        ('GS', 'UN'), ('GS', 'US'), ('UN', 'US')
    ]
    
    # Create brain topography for each condition pair
    for cond1, cond2 in condition_pairs:
        visualize_brain_itpc(cond1, cond2, frequencies)

if __name__ == '__main__':
    main()