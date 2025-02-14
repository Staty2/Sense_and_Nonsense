# First time setup: uncomment these lines to install required packages
using Printf
using HypothesisTests
using KernelDensity
using Distributions
using CSV
using DataFrames

# Include your existing files
include("General_Jullia_25.jl")
include("Mean_Res_Julia_25.jl")
include("Confidence_Julia_25.jl")

# Constants
const PARTICIPANT_N = 14  # number of participants in your data
const FREQUENCY_N = 57   # Changed from 58 to 57 to match your data
const ELECTRODE_N = 32
const TRIALS_PER_CONDITION = 30

"""
Convert frequency to column index
"""
function freq_to_column_index(freq::Float64)
    # Convert frequency to nearest index
    word_rate = 3.125  # Hz
    index = round(Int, (freq * FREQUENCY_N) / word_rate)
    return max(1, min(index, FREQUENCY_N))  # Ensure index is within bounds
end

"""
Main analysis function to extract phase information
"""
function extractPhaseInfo(filename::String; 
                         electrode::Int64=17,
                         category::String="GN",
                         use_phrase::Bool=false)
    
    # Initialize array for ITPC values
    allITPC = zeros(Complex{Float64}, PARTICIPANT_N)
    
    try
        # Read the main CSV file
        df = CSV.read(filename, DataFrame)
        
        # Get grammar indices from getGrammarPeaks()
        grammar = getGrammarPeaks()
        
        # Convert frequency to column index
        target_freq = use_phrase ? grammar[2] : grammar[4]
        target_index = freq_to_column_index(target_freq)
        complex_col = Symbol("complex_val_$target_index")
        
        println("Using frequency: ", target_freq, " Hz (column index: ", target_index, ")")
        
        # Get unique participant IDs
        participant_ids = unique(df.participant_id)
        
        # Process each participant
        for (participantI, participant_id) in enumerate(participant_ids)
            # Filter data for this participant
            participant_data = filter(row -> row.participant_id == participant_id, df)
            
            # Get condition data based on category
            ranges = Dict(
                "GN" => 1:30,
                "GS" => 31:60,
                "UN" => 61:90,
                "US" => 91:120
            )
            
            category_trials = ranges[category]
            condition_data = filter(row -> row.stimuli in category_trials, participant_data)
            
            # Filter for specific electrode
            electrode_data = filter(row -> row.electrode_number == electrode, condition_data)
            
            if !isempty(electrode_data)
                # Get complex values for the specified frequency
                if complex_col in names(electrode_data)
                    complex_values = [parse_complex(string(val)) for val in electrode_data[:, complex_col]]
                    
                    # Calculate phase measures
                    phases = phase([length(complex_values), reshape(complex_values, :, 1, 1)])
                    allITPC[participantI] = phaseMeasures(phases[2])[1,1]
                else
                    error("Column $complex_col not found. Available frequencies are 1 to $FREQUENCY_N")
                end
            end
        end
        
    catch e
        println("Error processing file: ", e)
        return nothing
    end
    
    return allITPC
end

"""
Print phase information and basic statistics
"""
function analyzeAndPrintPhases(phases::Vector{Complex{Float64}})
    println("\nPhase Information by Participant:")
    println("Real Part | Imaginary Part")
    println("-" ^ 30)
    
    for r in phases
        println(@sprintf("%.6f %.6f", real(r), imag(r)))
    end
    
    # Calculate basic statistics
    mean_phase = mean(phases)
    phase_coherence = abs(mean_phase)
    mean_angle = angle(mean_phase)
    
    println("\nSummary Statistics:")
    println("Mean Phase Coherence: ", phase_coherence)
    println("Mean Angle (rad): ", mean_angle)
    println("Mean Angle (deg): ", rad2deg(mean_angle))
end

# Use the full path to your CSV file
filename = "c:\\Users\\lotti\\OneDrive\\Documents\\Importantdocs\\EMATYear3\\TechnicalProject\\NeuralProcessingOfNonsense\\data\\csv_files\\processed_data.csv"

# Run the analysis function
function runAnalysis(filename::String; electrode::Int64=17, category::String="GN", use_phrase::Bool=false)
    println("Running analysis for:")
    println("Electrode: ", electrode)
    println("Category: ", category)
    println("Analysis type: ", use_phrase ? "Phrase" : "Syllable")
    
    phases = extractPhaseInfo(filename, 
                            electrode=electrode, 
                            category=category, 
                            use_phrase=use_phrase)
    
    if phases !== nothing
        analyzeAndPrintPhases(phases)
    else
        println("Analysis failed - check error messages above")
    end
end

# Run the analysis
runAnalysis(filename, electrode=17, category="GN", use_phrase=false)