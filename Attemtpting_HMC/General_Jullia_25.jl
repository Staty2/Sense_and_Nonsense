# First time setup: uncomment these lines to install required packages
# import Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")

using CSV
using DataFrames

# General helper functions including load functions and condition function
function getStimuli()
    ["GN", "GS", "UN", "US"]
end

function getStimuliP1to4()
    ["GN", "GS", "UN"]
end

function getGrammarPeaks()
    f = 1/0.32
    [0.25*f, 0.5*f, 0.75*f, f]
end

"""
Parse a string representation of a complex number.
Handles formats like "1.23+4.56im" or "1.23-4.56im"
"""
function parse_complex(s::String)
    # Remove any whitespace
    s = replace(s, " " => "")
    
    # If the string is empty or "missing", return 0+0im
    if isempty(s) || s == "missing"
        return Complex(0.0, 0.0)
    end
    
    try
        # Try to directly parse as a complex number
        return parse(Complex{Float64}, s)
    catch
        try
            # If that fails, try to handle the string manually
            # Find the position of +/- before 'im'
            im_pos = findfirst("im", s)
            if im_pos === nothing
                # No imaginary part
                return Complex(parse(Float64, s), 0.0)
            end
            
            # Find the last + or - before 'im'
            split_pos = findlast(r"[+-]", s[1:im_pos[1]-1])
            
            if split_pos === nothing
                # Only imaginary part
                imag_str = s[1:im_pos[1]-1]
                return Complex(0.0, parse(Float64, imag_str))
            else
                # Both real and imaginary parts
                real_str = s[1:split_pos-1]
                imag_str = s[split_pos:im_pos[1]-1]
                
                real_part = isempty(real_str) ? 0.0 : parse(Float64, real_str)
                imag_part = imag_str == "+" ? 1.0 : (imag_str == "-" ? -1.0 : parse(Float64, imag_str))
                
                return Complex(real_part, imag_part)
            end
        catch e
            @warn "Failed to parse complex number: $s"
            return Complex(0.0, 0.0)
        end
    end
end

"""
Loads and processes CSV file into the required format.
Returns [nTrials, a] where:
- nTrials is the number of trials
- a is a 3D array of Complex{Float64} with dimensions [trial, electrode, frequency]
"""
function load(filename::String)
    # Read CSV file
    df = CSV.read(filename, DataFrame)
    
    # Constants
    nElectrodes = 32
    nFreq = 58
    
    # Get unique trials
    trials = unique(df.stimuli)
    nTrials = length(trials)
    
    # Initialize 3D array for complex values
    a = zeros(Complex{Float64}, (nTrials, nElectrodes, nFreq))
    
    # Process each trial
    for (trial_idx, trial_num) in enumerate(trials)
        trial_data = filter(row -> row.stimuli == trial_num, df)
        
        for electrode in 1:nElectrodes
            electrode_data = filter(row -> row.electrode_number == electrode, trial_data)
            if !isempty(electrode_data)
                # Get complex values for this electrode
                for freq in 1:nFreq
                    complex_col = Symbol("complex_val_$freq")
                    if hasproperty(electrode_data, complex_col)
                        complex_str = string(electrode_data[1, complex_col])
                        a[trial_idx, electrode, freq] = parse_complex(complex_str)
                    end
                end
            end
        end
    end
    
    return [nTrials, a]
end

function phase(bigA)
    phases = bigA[2] ./ (abs.(bigA[2]))
    [bigA[1], phases]
end

function phase(filename::String)
    bigA = load(filename)
    phase(bigA)
end

"""
Splits the input into a dictionary of conditions based on stimulus categories:
GN (1-30), GS (31-60), UN (61-90), US (91-120)
"""
function condition(phases::Array{Complex{Float64},3}, decoding::Vector{Int64}, stimuli::Array{String}, offset::Int64)
    trialsPerCondition = 30::Int64  # Changed from 25 to 30 to match the stimulus ranges
    stride = 30::Int64
    
    condPhases = Dict()
    
    # Map stimulus categories to their ranges
    ranges = Dict(
        "GN" => 1:30,
        "GS" => 31:60,
        "UN" => 61:90,
        "US" => 91:120
    )
    
    for stimulus in stimuli
        if haskey(ranges, stimulus)
            range_trials = ranges[stimulus]
            relevant_trials = findall(x -> x in range_trials, decoding)
            if !isempty(relevant_trials)
                condPhases[stimulus] = [length(relevant_trials), phases[relevant_trials,:,:]]
            end
        end
    end
    
    return condPhases
end

function getGrammaticalConditions(filename::String, decoding::Vector{Int64})
    bigA = load(filename)
    condition(bigA[2], decoding, ["GN", "GS"], 0)
end

function getUngrammaticalConditions(filename::String, decoding::Vector{Int64})
    bigA = load(filename)
    condition(bigA[2], decoding, ["UN", "US"], 0)
end

function getAllConditions(filename::String, decoding::Vector{Int64})
    bigA = load(filename)
    condition(bigA[2], decoding, getStimuli(), 0)
end

# Example usage:
df = load("data/csv_files/processed_data.csv")
phases = phase(df)
decoding = collect(1:120)  # or your actual decoding vector
results_grammatical = getGrammaticalConditions("data/csv_files/processed_data.csv", decoding)
results_ungrammatical = getUngrammaticalConditions("data/csv_files/processed_data.csv", decoding)
results_all = getAllConditions("data/csv_files/processed_data.csv", decoding)

