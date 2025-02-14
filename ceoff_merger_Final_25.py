import pandas as pd

# Load the dataset with explicit dtype handling
#ft_coeff_df = pd.read_csv("data/csv_files/processed_data.csv")
#print(ft_coeff_df.info())  # Check column types
#print(ft_coeff_df.head())  # Preview first few rows


#S5_10_07_2018_ft_coeff

#ft_coeff_dfind = pd.read_csv("data/csv_files/S5_10_07_2018_ft_coeff.csv")
#print(ft_coeff_dfind.info())  # Check column types
#print(ft_coeff_dfind.head())  # Preview first few rows

import os
import pandas as pd
import re

def compile_fourier_transform_coefficients(base_directory='data/csv_files'):
    # List to store DataFrames for each participant
    participant_dfs = []
    
    # Regex pattern to match FT coefficient files
    ft_pattern = re.compile(r'^S(\d+)_\d+_\d+_\d+_ft_coeff\.csv$')
    
    # Iterate through files in the directory
    for filename in os.listdir(base_directory):
        # Check if the file matches the FT coefficient file pattern
        match = ft_pattern.match(filename)
        if match:
            # Extract participant ID
            participant_id = match.group(1)
            
            # Read the CSV file
            file_path = os.path.join(base_directory, filename)
            
            try:
                # Try reading the CSV with different options
                df = pd.read_csv(file_path, header=None)  # No header
                
                # If DataFrame is empty, print a warning
                if df.empty:
                    print(f"Warning: {filename} is empty!")
                    continue
                
                # If only one column, try transposing
                if df.shape[1] == 1:
                    df = df.T
                
                # Add participant ID column
                df['participant_id'] = participant_id
                
                # Append to list of DataFrames
                participant_dfs.append(df)
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                # Print file contents for debugging
                try:
                    with open(file_path, 'r') as f:
                        print(f"File contents of {filename}:")
                        print(f.read())
                except Exception as read_error:
                    print(f"Could not read file contents: {read_error}")
    
    # Combine all participant DataFrames
    if participant_dfs:
        combined_df = pd.concat(participant_dfs, ignore_index=True)
        
        # Reorder columns to put participant_id first, then frequency coefficients
        # Assuming the frequency coefficients are the numeric columns
        numeric_cols = [col for col in combined_df.columns if col not in ['participant_id']]
        columns_order = ['participant_id'] + sorted(numeric_cols, key=lambda x: int(x) if str(x).isdigit() else x)
        
        combined_df = combined_df[columns_order]
        
        return combined_df
    else:
        print("No matching files found or all files were empty!")
        return None

# Usage
result_df = compile_fourier_transform_coefficients()

# Save to CSV if desired
if result_df is not None:
    result_df.to_csv('combined_fourier_transform_coefficients.csv', index=False)
    print(result_df)
else:
    print("No DataFrame was created.")