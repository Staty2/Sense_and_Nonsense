# Concise Phase Significance Analysis

using Printf
using HypothesisTests
using KernelDensity
using Distributions
using CSV
using DataFrames
using Random

# Include existing files
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
Compute significance across all conditions and analysis types
"""
function computeSignificanceSummary(filename::String; 
                                    output_filename::String="phase_significance_summary.csv", 
                                    electrodes::Vector{Int64}=[17])
    # Prepare results DataFrame
    results = DataFrame(
        Analysis_Type = String[],
        Condition = String[],
        Electrode = Int[],
        Mean_ITPC = Float64[],
        Lower_Bound = Float64[],
        Upper_Bound = Float64[],
        Is_Significant = Bool[]
    )
    
    # Read the main CSV file
    df = CSV.read(filename, DataFrame)
    
    # Define analysis combinations
    conditions = ["GN", "GS", "UN", "US"]
    analysis_types = [false, true]  # syllable and phrase
    
    # Process each combination
    for use_phrase in analysis_types
        for condition in conditions
            for electrode in electrodes
                # Get grammar indices from getGrammarPeaks()
                grammar = getGrammarPeaks()
                
                # Convert frequency to column index
                target_freq = use_phrase ? grammar[2] : grammar[4]
                target_index = freq_to_column_index(target_freq)
                complex_col = Symbol("complex_val_$target_index")
                
                # Get unique participant IDs
                participant_ids = unique(df.participant_id)
                
                # Initialize array for ITPC values
                allITPC = zeros(Complex{Float64}, PARTICIPANT_N)
                
                # Get condition data based on category
                ranges = Dict(
                    "GN" => 1:30,
                    "GS" => 31:60,
                    "UN" => 61:90,
                    "US" => 91:120
                )
                
                category_trials = ranges[condition]
                
                # Process each participant
                for (participantI, participant_id) in enumerate(participant_ids)
                    # Filter data for this participant
                    participant_data = filter(row -> row.participant_id == participant_id, df)
                    
                    # Filter condition data
                    condition_data = filter(row -> row.stimuli in category_trials, participant_data)
                    
                    # Filter for specific electrode
                    electrode_data = filter(row -> row.electrode_number == electrode, condition_data)
                    
                    if !isempty(electrode_data)
                        # Get complex values for the specified frequency
                        if complex_col in names(electrode_data)
                            # Parse complex values
                            complex_values = [parse_complex(string(val)) for val in electrode_data[:, complex_col]]
                            
                            # Calculate phase measures
                            phases = phase([length(complex_values), reshape(complex_values, :, 1, 1)])
                            allITPC[participantI] = phaseMeasures(phases[2])[1,1]
                        end
                    end
                end
                
                # Calculate null hypothesis bounds
                n_permutations = 1000
                Random.seed!(12345)  # For reproducibility
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
                itpc_values = abs.(allITPC)
                mean_itpc = mean(itpc_values)
                
                # Determine significance
                is_significant = mean_itpc > upper_bound
                
                # Add to results
                push!(results, [
                    use_phrase ? "Phrase" : "Syllable",
                    condition,
                    electrode,
                    round(mean_itpc, digits=4),
                    round(lower_bound, digits=7),
                    round(upper_bound, digits=6),
                    is_significant
                ])
                
                # Print null hypothesis bounds
                println("$(use_phrase ? "Phrase" : "Syllable") $condition bounds: $lower_bound - $upper_bound")
                
                # Print significant conditions
                if is_significant
                    println("$(use_phrase ? "Phrase" : "Syllable") $condition condition significant: true")
                end
            end
        end
    end
    
    # Save results to CSV
    CSV.write(output_filename, results)
    
    # Print results to console
    println("\nSignificance Summary:")
    show(results, allrows=true)
    
    return results
end

"""
Ensure parse_complex function is defined
"""
function parse_complex(val::String)
    # Remove brackets and split
    cleaned = replace(val, r"[\[\]]" => "")
    parts = split(cleaned, ",")
    if length(parts) >= 2
        return parse(ComplexF64, "$(parts[1]) + $(parts[2])im")
    else
        return parse(ComplexF64, val)
    end
end

"""
Run comprehensive significance analysis
"""
function runComprehensiveAnalysis(filename::String; 
                                  output_filename::String="phase_significance_summary.csv", 
                                  electrodes::Vector{Int64}=[17])
    println("Running comprehensive significance analysis:")
    println("Input file: ", filename)
    println("Output file: ", output_filename)
    println("Electrodes: ", electrodes)
    
    # Compute and save significance summary
    results = computeSignificanceSummary(filename, 
                                         output_filename=output_filename, 
                                         electrodes=electrodes)
    
    return results
end

# Example usage
results = runComprehensiveAnalysis("data/csv_files/processed_data.csv")