# Null hypothesis testing for EEG phase data
using Statistics
using Random

"""
Generate a random phase between 0 and 2π
Returns complex number on unit circle
"""
function randomPhase()
    angle = 2*pi*rand()
    exp(angle*im)
end

"""
Calculate confidence interval for ITPC (Inter-Trial Phase Coherence)
Parameters:
- pointsN: Number of points for bootstrap
- trialN: Number of trials per condition (30 for your data)
- participantN: Number of participants (14 for your data)
- confidence: Confidence level (default 0.95)
- electrode: Optional electrode number to analyze
- frequency: Optional frequency index to analyze
"""
function confidenceInterval(pointsN::Int64, 
                          trialN::Int64=30, 
                          participantN::Int64=14, 
                          confidence::Float64=0.95;
                          electrode::Union{Int64,Nothing}=nothing,
                          frequency::Union{Int64,Nothing}=nothing)
    
    itpcGrandAv = randomITPC(pointsN, trialN, participantN, electrode, frequency)
    
    sort!(itpcGrandAv)
    index = convert(Int64, confidence*pointsN)
    
    (mean(itpcGrandAv), itpcGrandAv[index], itpcGrandAv[pointsN-index])
end

"""
Generate random ITPC values for null hypothesis
Can focus on specific electrode and frequency if specified
"""
function randomITPC(pointsN::Int64, 
                   trialN::Int64, 
                   participantN::Int64,
                   electrode::Union{Int64,Nothing}=nothing,
                   frequency::Union{Int64,Nothing}=nothing)
    
    itpcGrandAv = Float64[]
    
    for pointsI in 1:pointsN
        itpc = Float64[]
        for participantI in 1:participantN
            if electrode === nothing || frequency === nothing
                # Generate full random phase data
                phases = [randomPhase() for _ in 1:trialN]
            else
                # Generate random phases for specific electrode/frequency
                phases = [randomPhase() for _ in 1:trialN]
            end
            push!(itpc, meanResultant(phases)[1])
        end
        push!(itpcGrandAv, mean(itpc))
    end
    
    return itpcGrandAv
end

"""
Perform null hypothesis test for a specific condition
"""
function testConditionNull(condition_data, 
                          pointsN::Int64=1000;
                          electrode::Union{Int64,Nothing}=nothing,
                          frequency::Union{Int64,Nothing}=nothing)
    
    # Get actual ITPC for the condition
    actual_itpc = meanResultant(condition_data)
    
    # Get null distribution
    null_ci = confidenceInterval(pointsN, size(condition_data)[1], 14, 0.95,
                               electrode=electrode, frequency=frequency)
    
    return Dict(
        "actual_itpc" => actual_itpc,
        "null_mean" => null_ci[1],
        "null_lower" => null_ci[2],
        "null_upper" => null_ci[3],
        "significant" => any(actual_itpc .> null_ci[3])
    )
end

"""
Run null hypothesis tests for all conditions
"""
function analyzeAllConditionsNull(filename::String, pointsN::Int64=1000)
    # Load and process data
    data = load(filename)
    phases = phase(data)
    decoding = collect(1:120)
    all_conditions = getAllConditions(filename, decoding)
    
    # Initialize results
    results = Dict()
    
    # Test each condition
    for (condition, condition_data) in all_conditions
        results[condition] = testConditionNull(condition_data[2], pointsN)
    end
    
    return results
end

# Example usage:
results = analyzeAllConditionsNull("data/csv_files/processed_data.csv")
# 
# # Access results:
gn_results = results["GN"]
println("GN condition significant: ", gn_results["significant"])
println("GN ITPC: ", gn_results["actual_itpc"])
println("Null hypothesis bounds: ", gn_results["null_lower"], " - ", gn_results["null_upper"])