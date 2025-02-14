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
        
        # Debug: Print available columns
        println("Available columns in the DataFrame:")
        println(names(df))
        
        # Get grammar indices from getGrammarPeaks()
        grammar = getGrammarPeaks()
        
        # Convert frequency to column index
        target_freq = use_phrase ? grammar[2] : grammar[4]
        target_index = freq_to_column_index(target_freq)
        complex_col = Symbol("complex_val_$target_index")
        
        println("Using frequency: ", target_freq, " Hz (column index: ", target_index, ")")
        
        # Check if the column exists
        if !(complex_col in names(df))
            println("Warning: Column $complex_col not found. Skipping analysis for this frequency.")
            return nothing
        end
        
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
                complex_values = [parse_complex(string(val)) for val in electrode_data[:, complex_col]]
                
                # Calculate phase measures
                phases = phase([length(complex_values), reshape(complex_values, :, 1, 1)])
                allITPC[participantI] = phaseMeasures(phases[2])[1,1]
            end
        end
        
    catch e
        println("Error processing file: ", e)
        return nothing
    end
    
    return allITPC
end

"""
Analyze phases and return mean ITPC and significance
"""
function analyzePhases(phases::Vector{Complex{Float64}})
    # Calculate null hypothesis bounds
    n_permutations = 1000
    null_dist = zeros(n_permutations)
    for i in 1:n_permutations
        random_phases = rand(Uniform(-π, π), TRIALS_PER_CONDITION)
        complex_values = exp.(im .* random_phases)
        null_dist[i] = abs(mean(complex_values))
    end
    
    # Calculate bounds
    lower_bound = quantile(null_dist, 0.025)
    upper_bound = quantile(null_dist, 0.975)
    
    # Calculate ITPC values
    itpc_values = abs.(phases)
    
    # Check significance
    mean_itpc = mean(itpc_values)
    is_significant = mean_itpc > upper_bound
    
    return mean_itpc, is_significant
end

# Run the analysis function
function runAnalysis(filename::String; electrode::Int64=17)
    # Initialize DataFrame to store results
    results_df = DataFrame(Category=String[], UsePhrase=Bool[], MeanITPC=Float64[], Lower_Bound=Float64[], Upper_Bound=Float64[], Is_Significant=Bool[])
    
    # Loop through each category and analysis type
    for category in ["GN", "GS", "UN", "US"]
        for use_phrase in [false, true]
            println("Running analysis for:")
            println("Category: ", category)
            println("Analysis type: ", use_phrase ? "Phrase" : "Syllable")
            
            phases = extractPhaseInfo(filename, 
                                    electrode=electrode, 
                                    category=category, 
                                    use_phrase=use_phrase)
            
            if phases !== nothing
                mean_itpc, is_significant = analyzePhases(phases)
                push!(results_df, (category, use_phrase, mean_itpc, lower_bound, upper_bound, is_significant))
            else
                println("Analysis skipped for this condition.")
            end
        end
    end
    
    # Save results to CSV
    CSV.write("analysis_results.csv", results_df)
    println("Analysis results saved to analysis_results.csv")
end

# Run the analysis
runAnalysis("data/csv_files/processed_data.csv", electrode=17)