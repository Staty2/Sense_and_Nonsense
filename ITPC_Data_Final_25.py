import numpy as np
import pandas as pd
import warnings
import math
import cmath

class EEGPhaseAnalysis:
    def __init__(self, data_path, num_frequencies=58, num_electrodes=32):
        """
        Initialize EEG Phase Analysis
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing all participant data
        num_frequencies : int, optional
            Number of frequencies to analyze (default: 58)
        num_electrodes : int, optional
            Number of electrodes (default: 32)
        """
        self.data_path = data_path
        self.num_frequencies = num_frequencies
        self.num_electrodes = num_electrodes
        
        # Define trial conditions and their numeric ranges
        self.trial_conditions = {
            'GN': (1, 30),    # Grammatical Nonsensical
            'GS': (31, 60),   # Grammatical Sensical
            'UN': (61, 90),   # Ungrammatical Nonsensical
            'US': (91, 120)   # Ungrammatical Sensical
        }
        
        # Predefined frequencies
        self.frequencies = np.array([
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
        ])
    
    def load_data(self):
        """
        Load entire dataset
        
        Returns:
        --------
        pd.DataFrame
            Loaded participant data
        """
        # Load data and print diagnostic information
        data = pd.read_csv(self.data_path)
        
        # Diagnostic information
        print("Data Diagnostic Information:")
        print(f"Total rows: {len(data)}")
        print(f"Columns: {list(data.columns)}")
        
        # Print first few rows of complex value columns
        complex_columns = [col for col in data.columns if col.startswith('complex_val_')]
        print("\nSample Complex Values:")
        print(data[complex_columns].head())
        
        # Check data types of complex columns
        print("\nComplex Column Data Types:")
        print(data[complex_columns].dtypes)
        
        return data
    
    def parse_complex_value(self, val):
        """
        Flexibly parse complex values and compute phase
        
        Parameters:
        -----------
        val : str or list
            Complex value to parse
        
        Returns:
        --------
        tuple
            (complex number, phase in radians)
        """
        try:
            # If it's already a complex number, return it with computed phase
            if isinstance(val, complex):
                return val, cmath.phase(val)
            
            # Handle list input (split real and imaginary)
            if isinstance(val, list):
                # Try parsing list with two elements (real and imaginary)
                if len(val) == 2:
                    # Remove 'i' from imaginary part
                    real = float(val[0])
                    imag = float(val[1].rstrip('i'))
                    complex_num = complex(real, imag)
                    return complex_num, math.atan2(imag, real)
                
                # If list has one element, try parsing as string
                elif len(val) == 1:
                    val = val[0]
            
            # Handle string input
            if isinstance(val, str):
                # Remove any surrounding quotes or brackets
                val = val.strip("'\"[]")
                
                # Check if it's in standard complex string format
                if 'i' in val or 'j' in val:
                    # Replace 'i' or 'j' with 'j' for Python complex parsing
                    val = val.replace('i', 'j')
                    complex_num = complex(val)
                    return complex_num, cmath.phase(complex_num)
                
                # If no 'i', assume it's a real number
                real = float(val)
                return complex(real, 0), 0
            
            # Direct conversion for numeric types
            complex_num = complex(val)
            return complex_num, cmath.phase(complex_num)
        
        except Exception as e:
            print(f"Error parsing complex value {val}: {e}")
            return 0+0j, 0
    
    def preprocess_data(self, data):
        """
        Preprocess EEG data by condition and electrode
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input EEG data
        
        Returns:
        --------
        dict
            Preprocessed data organized by condition and electrode
        """
        preprocessed_data = {}
        
        # Process each trial condition
        for condition, (start, end) in self.trial_conditions.items():
            condition_data = {}
            
            # Filter data for this condition
            condition_subset = data[
                (data['stimuli'] >= start) & 
                (data['stimuli'] <= end)
            ]
            
            # Group by electrode
            for electrode in range(1, self.num_electrodes + 1):
                electrode_data = condition_subset[
                    condition_subset['electrode_number'] == electrode
                ]
                
                # Extract complex values
                complex_columns = [f'complex_val_{i}' for i in range(1, 58)]
                
                # Convert to complex numbers and phases
                complex_values = []
                phase_values = []
                for _, row in electrode_data.iterrows():
                    # Parse each complex value column
                    row_complex_with_phase = [self.parse_complex_value(row[col]) for col in complex_columns]
                    
                    # Separate complex numbers and phases
                    row_complex = [val[0] for val in row_complex_with_phase]
                    row_phases = [val[1] for val in row_complex_with_phase]
                    
                    complex_values.append(row_complex)
                    phase_values.append(row_phases)
                
                condition_data[electrode] = {
                    'complex': np.array(complex_values),
                    'phases': np.array(phase_values)
                }
            
            preprocessed_data[condition] = condition_data
        
        return preprocessed_data
    
    def compute_itpc(self, phases):
        """
        Compute Inter-Trial Phase Coherence (ITPC)
        
        Parameters:
        -----------
        phases : np.ndarray
            Phase angles
        
        Returns:
        --------
        np.ndarray
            ITPC values
        """
        # Compute mean resultant vector
        mean_phase_vector = np.mean(np.exp(1j * phases), axis=0)
        
        # Compute ITPC (absolute value of mean phase vector)
        itpc = np.abs(mean_phase_vector)
        
        return itpc
    
    def analyze_itpc(self):
        """
        Perform comprehensive ITPC analysis
        
        Returns:
        --------
        dict
            ITPC results for each condition and electrode
        """
        # Load full dataset
        full_data = self.load_data()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(full_data)
        
        # Compute ITPC for each condition and electrode
        itpc_results = {}
        for condition, electrode_data in preprocessed_data.items():
            condition_itpc = {}
            
            for electrode, data_dict in electrode_data.items():
                # Compute ITPC on phases
                itpc = self.compute_itpc(data_dict['phases'])
                condition_itpc[electrode] = {
                    'itpc': itpc,
                    'complex_data': data_dict['complex']
                }
            
            itpc_results[condition] = condition_itpc
        
        return itpc_results
    
    def export_itpc_results(self, itpc_results, output_path='itpc_results.csv'):
        """
        Export ITPC results to a CSV file
        
        Parameters:
        -----------
        itpc_results : dict
            ITPC results from analysis
        output_path : str, optional
            Path to save the CSV file
        
        Returns:
        --------
        str
            Path to the exported CSV file
        """
        # Prepare data for export
        export_data = []
        for condition, electrode_data in itpc_results.items():
            for electrode, data_dict in electrode_data.items():
                export_data.append({
                    'Condition': condition,
                    'Electrode': electrode,
                    'ITPC': data_dict['itpc'].tolist() if isinstance(data_dict['itpc'], np.ndarray) else data_dict['itpc']
                })
        
        # Convert to DataFrame and export
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_path, index=False)
        
        print(f"ITPC results exported to {output_path}")
        return output_path
    
    def summary_statistics(self, itpc_results):
        """
        Compute summary statistics of ITPC results
        
        Parameters:
        -----------
        itpc_results : dict
            ITPC results from analysis
        
        Returns:
        --------
        dict
            Summary statistics for each condition
        """
        summary = {}
        for condition, electrode_data in itpc_results.items():
            # Convert electrode ITPC values to a numpy array
            itpc_values = np.array([
                data_dict['itpc'].mean() if isinstance(data_dict['itpc'], np.ndarray) else data_dict['itpc'] 
                for data_dict in electrode_data.values()
            ])
            
            summary[condition] = {
                'mean': np.mean(itpc_values),
                'std': np.std(itpc_values),
                'min': np.min(itpc_values),
                'max': np.max(itpc_values),
                'median': np.median(itpc_values)
            }
        
        return summary

def main():
    try:
        # Path to the CSV file
        csv_path = 'data/csv_files/processed_data.csv'
        
        # Initialize analysis
        analyzer = EEGPhaseAnalysis(csv_path)
        
        # Perform ITPC analysis
        itpc_results = analyzer.analyze_itpc()
        
        # Export results to CSV
        output_path = analyzer.export_itpc_results(itpc_results)
        
        # Compute and print summary statistics
        summary = analyzer.summary_statistics(itpc_results)
        print("\nITPC Summary Statistics:")
        for condition, stats in summary.items():
            print(f"\n{condition} Condition:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name.capitalize()}: {stat_value}")
        
        return itpc_results, summary
    
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        warnings.warn(f"Analysis failed: {e}")
        return None, None

if __name__ == '__main__':
    main()