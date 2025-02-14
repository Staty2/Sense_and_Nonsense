# Statistical analysis functions for EEG phase data
using Statistics

"""
Calculate mean resultant length across specified dimension
Default is dimension 1 (trials)
"""
function meanResultant(a)
    meanResultant(a, 1)
end

function meanResultant(a, dim::Int64)
    dropdims(abs.(mean(a, dims=dim)), dims=dim)
end

"""
Calculate circular measures (angle and magnitude) for phase data
Returns tuple of (angles, magnitudes)
"""
function circularMeasures(a)
    r = dropdims(sum(a, dims=1) ./ size(a)[1], dims=1)
    (angle.(r), abs.(r))
end

"""
Calculate phase measures
Returns complex values representing mean phase vectors
"""
function phaseMeasures(a)
    dropdims(sum(a, dims=1) ./ size(a)[1], dims=1)
end

"""
Calculate circular variance
Returns -2*log of mean resultant length
"""
function circularVariance(a)
    -2 .* log.(meanResultant(a))
end

"""
Calculate bias-corrected circular statistics
Based on Kutil (2012) paper
"""
function biasCorrect(a)
    (dropdims(abs.(sum(a, dims=1).^2/size(a)[1]), dims=1) .- ones(Float64, size(a)[2:3])) ./ (size(a)[1] - 1)
end

"""
Calculate power of the signal
Default is across dimension 1 (trials)
"""
function getPower(a)
    getPower(a, 1)
end

function getPower(a, dim::Int64)
    dropdims(mean((abs.(a)).^2, dims=dim), dims=dim)
end

# Example usage with your EEG data:
"""
Process all conditions and calculate statistics
"""
function analyzeEEGData(filename::String)
    # Load the data
    data = load(filename)
    phases = phase(data)
    decoding = collect(1:120)
    
    # Get conditions
    all_conditions = getAllConditions(filename, decoding)
    
    # Initialize results dictionary
    results = Dict()
    
    # Calculate statistics for each condition
    for (condition, condition_data) in all_conditions
        phase_data = condition_data[2]  # Get the phase data from the condition
        
        results[condition] = Dict(
            "mean_resultant" => meanResultant(phase_data),
            "circular_variance" => circularVariance(phase_data),
            "power" => getPower(phase_data),
            "circular_measures" => circularMeasures(phase_data)
        )
    end
    
    return results
end

# Usage:
results = analyzeEEGData("data/csv_files/processed_data.csv")
# Access results like:
gn_power = results["GN"]["power"]
gs_variance = results["GS"]["circular_variance"]