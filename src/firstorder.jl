# First Order Statistical Features for Radiomics.jl
#
# First-order features are computed directly from voxel intensities within the ROI,
# without considering spatial relationships between voxels. These features describe
# the distribution of voxel gray levels.
#
# Total: 19 features (18 IBSI-compliant + 1 PyRadiomics extension)
#
# References:
# - PyRadiomics: radiomics/firstorder.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - PyRadiomics docs: https://pyradiomics.readthedocs.io/en/latest/features.html

using Statistics
using StatsBase: countmap

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding log(0) - matches np.spacing(1)
const EPSILON = eps(Float64)  # ≈ 2.2e-16

#==============================================================================#
# Feature 1: Energy (IBSI: N8CA)
#==============================================================================#

"""
    energy(voxels::AbstractVector{<:Real}; shift::Real=0) -> Float64

Compute the energy (sum of squared values) of voxel intensities.

# Mathematical Formula
```
Energy = Σᵢ₌₁ᴺᵖ (X(i) + c)²
```
where:
- X(i) = voxel intensity at index i
- Nₚ = number of voxels in ROI
- c = shift parameter (voxelArrayShift for handling negative values)

# Arguments
- `voxels`: Vector of voxel intensity values within the ROI
- `shift::Real=0`: Shift applied to values before squaring (PyRadiomics voxelArrayShift)

# Returns
- `Float64`: Energy value

# Notes
- Volume-confounded: larger ROIs produce larger Energy values
- PyRadiomics uses `np.nansum((targetVoxelArray + voxelArrayShift)**2)`
- IBSI ID: N8CA

# Example
```julia
voxels = [1.0, 2.0, 3.0, 4.0, 5.0]
e = energy(voxels)  # 55.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getEnergyFeatureValue
"""
function energy(voxels::AbstractVector{<:Real}; shift::Real=0)
    return sum(v -> (v + shift)^2, voxels)
end

#==============================================================================#
# Feature 2: Total Energy (NOT IN IBSI - PyRadiomics extension)
#==============================================================================#

"""
    total_energy(voxels::AbstractVector{<:Real}, voxel_volume::Real; shift::Real=0) -> Float64

Compute total energy accounting for voxel volume.

# Mathematical Formula
```
TotalEnergy = Vvoxel × Σᵢ₌₁ᴺᵖ (X(i) + c)²
```
where Vvoxel = voxel volume in mm³

# Arguments
- `voxels`: Vector of voxel intensity values within the ROI
- `voxel_volume`: Volume of a single voxel in mm³
- `shift::Real=0`: Shift applied to values before squaring

# Returns
- `Float64`: Total energy value

# Notes
- NOT in IBSI standard - PyRadiomics extension
- Volume-confounded
- Requires voxel spacing information

# Example
```julia
voxels = [1.0, 2.0, 3.0]
voxel_vol = 2.0 * 2.0 * 3.0  # 12 mm³
te = total_energy(voxels, voxel_vol)  # 14.0 * 12.0 = 168.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getTotalEnergyFeatureValue
"""
function total_energy(voxels::AbstractVector{<:Real}, voxel_volume::Real; shift::Real=0)
    return voxel_volume * energy(voxels; shift=shift)
end

#==============================================================================#
# Feature 3: Entropy (IBSI: TLU2)
#==============================================================================#

"""
    entropy(voxels::AbstractVector{<:Real}) -> Float64

Compute the entropy of voxel intensity distribution.

# Mathematical Formula
```
Entropy = -Σᵢ₌₁ᴺᵍ p(i) × log₂(p(i) + ε)
```
where:
- Nᵧ = number of distinct gray levels
- p(i) = probability of gray level i (normalized histogram)
- ε = machine epsilon (≈ 2.2×10⁻¹⁶) to prevent log(0)

# Arguments
- `voxels`: Vector of voxel intensity values (should be DISCRETIZED for texture analysis)

# Returns
- `Float64`: Entropy value in bits

# Notes
- Higher entropy indicates more heterogeneous distribution
- For texture analysis, input should be discretized voxel values
- ε prevents log(0) errors
- IBSI ID: TLU2

# Example
```julia
# Uniform distribution (high entropy)
voxels = [1, 2, 3, 4, 5, 6, 7, 8]
e = entropy(voxels)  # ≈ 3.0 (log₂(8))

# Constant distribution (zero entropy)
voxels = [5, 5, 5, 5, 5]
e = entropy(voxels)  # ≈ 0.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getEntropyFeatureValue
"""
function entropy(voxels::AbstractVector{<:Real})
    # Count occurrences of each unique value
    counts = countmap(voxels)
    n = length(voxels)

    # Compute probabilities and entropy
    H = 0.0
    for count in values(counts)
        p = count / n
        H -= p * log2(p + EPSILON)
    end

    return H
end

#==============================================================================#
# Feature 4: Minimum (IBSI: 1GSF)
#==============================================================================#

"""
    fo_minimum(voxels::AbstractVector{<:Real}) -> Float64

Compute the minimum voxel intensity.

# Mathematical Formula
```
Minimum = min(X)
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Minimum intensity value

# Notes
- NaN values are ignored (matches np.nanmin behavior)
- IBSI ID: 1GSF

# Example
```julia
voxels = [10.0, 25.0, 5.0, 30.0]
m = fo_minimum(voxels)  # 5.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getMinimumFeatureValue
"""
function fo_minimum(voxels::AbstractVector{<:Real})
    # Filter NaN values like np.nanmin
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(minimum(valid))
end

#==============================================================================#
# Feature 5: 10th Percentile (IBSI: QG58)
#==============================================================================#

"""
    percentile_10(voxels::AbstractVector{<:Real}) -> Float64

Compute the 10th percentile of voxel intensities.

# Mathematical Formula
```
P₁₀ = 10th percentile of X
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: 10th percentile value

# Notes
- Uses linear interpolation (matches numpy default)
- NaN values are ignored
- IBSI ID: QG58

# Example
```julia
voxels = collect(1.0:100.0)
p10 = percentile_10(voxels)  # ≈ 10.0
```

# References
- PyRadiomics: radiomics/firstorder.py:get10PercentileFeatureValue
"""
function percentile_10(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(quantile(valid, 0.10))
end

#==============================================================================#
# Feature 6: 90th Percentile (IBSI: 8DWT)
#==============================================================================#

"""
    percentile_90(voxels::AbstractVector{<:Real}) -> Float64

Compute the 90th percentile of voxel intensities.

# Mathematical Formula
```
P₉₀ = 90th percentile of X
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: 90th percentile value

# Notes
- Uses linear interpolation (matches numpy default)
- NaN values are ignored
- IBSI ID: 8DWT

# Example
```julia
voxels = collect(1.0:100.0)
p90 = percentile_90(voxels)  # ≈ 90.0
```

# References
- PyRadiomics: radiomics/firstorder.py:get90PercentileFeatureValue
"""
function percentile_90(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(quantile(valid, 0.90))
end

#==============================================================================#
# Feature 7: Maximum (IBSI: 84IY)
#==============================================================================#

"""
    fo_maximum(voxels::AbstractVector{<:Real}) -> Float64

Compute the maximum voxel intensity.

# Mathematical Formula
```
Maximum = max(X)
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Maximum intensity value

# Notes
- NaN values are ignored (matches np.nanmax behavior)
- IBSI ID: 84IY

# Example
```julia
voxels = [10.0, 25.0, 5.0, 30.0]
m = fo_maximum(voxels)  # 30.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getMaximumFeatureValue
"""
function fo_maximum(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(maximum(valid))
end

#==============================================================================#
# Feature 8: Mean (IBSI: Q4LE)
#==============================================================================#

"""
    fo_mean(voxels::AbstractVector{<:Real}) -> Float64

Compute the mean voxel intensity.

# Mathematical Formula
```
Mean = (1/Nₚ) × Σᵢ₌₁ᴺᵖ X(i)
```
where Nₚ = number of voxels

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Mean intensity value

# Notes
- NaN values are ignored (matches np.nanmean behavior)
- IBSI ID: Q4LE

# Example
```julia
voxels = [10.0, 20.0, 30.0, 40.0]
m = fo_mean(voxels)  # 25.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getMeanFeatureValue
"""
function fo_mean(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(mean(valid))
end

#==============================================================================#
# Feature 9: Median (IBSI: Y12H)
#==============================================================================#

"""
    fo_median(voxels::AbstractVector{<:Real}) -> Float64

Compute the median voxel intensity.

# Mathematical Formula
```
Median = middle value of sorted X
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Median intensity value

# Notes
- NaN values are ignored (matches np.nanmedian behavior)
- IBSI ID: Y12H

# Example
```julia
voxels = [10.0, 20.0, 30.0, 40.0, 50.0]
m = fo_median(voxels)  # 30.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getMedianFeatureValue
"""
function fo_median(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(median(valid))
end

#==============================================================================#
# Feature 10: Interquartile Range (IBSI: SALO)
#==============================================================================#

"""
    interquartile_range(voxels::AbstractVector{<:Real}) -> Float64

Compute the interquartile range of voxel intensities.

# Mathematical Formula
```
IQR = P₇₅ - P₂₅
```
where P₇₅ and P₂₅ are the 75th and 25th percentiles.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Interquartile range

# Notes
- Measures spread of the middle 50% of data
- More robust to outliers than range
- NaN values are ignored
- IBSI ID: SALO

# Example
```julia
voxels = collect(1.0:100.0)
iqr = interquartile_range(voxels)  # 50.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getInterquartileRangeFeatureValue
"""
function interquartile_range(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(quantile(valid, 0.75) - quantile(valid, 0.25))
end

#==============================================================================#
# Feature 11: Range (IBSI: 2OJQ)
#==============================================================================#

"""
    fo_range(voxels::AbstractVector{<:Real}) -> Float64

Compute the range of voxel intensities.

# Mathematical Formula
```
Range = max(X) - min(X)
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Range of intensity values

# Notes
- NaN values are ignored
- IBSI ID: 2OJQ

# Example
```julia
voxels = [10.0, 25.0, 5.0, 30.0]
r = fo_range(voxels)  # 25.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getRangeFeatureValue
"""
function fo_range(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    return Float64(maximum(valid) - minimum(valid))
end

#==============================================================================#
# Feature 12: Mean Absolute Deviation (IBSI: 4FUA)
#==============================================================================#

"""
    mean_absolute_deviation(voxels::AbstractVector{<:Real}) -> Float64

Compute the mean absolute deviation of voxel intensities.

# Mathematical Formula
```
MAD = (1/Nₚ) × Σᵢ₌₁ᴺᵖ |X(i) - X̄|
```
where X̄ is the mean intensity.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Mean absolute deviation

# Notes
- Robust measure of variability
- NaN values are ignored
- IBSI ID: 4FUA

# Example
```julia
voxels = [10.0, 20.0, 30.0, 40.0]
mad = mean_absolute_deviation(voxels)  # 10.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getMeanAbsoluteDeviationFeatureValue
"""
function mean_absolute_deviation(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    μ = mean(valid)
    return Float64(mean(abs.(valid .- μ)))
end

#==============================================================================#
# Feature 13: Robust Mean Absolute Deviation (IBSI: 1128)
#==============================================================================#

"""
    robust_mean_absolute_deviation(voxels::AbstractVector{<:Real}) -> Float64

Compute the robust mean absolute deviation of voxel intensities.

# Mathematical Formula
```
rMAD = (1/N₁₀₋₉₀) × Σᵢ |X₁₀₋₉₀(i) - X̄₁₀₋₉₀|
```
where only voxels between the 10th and 90th percentile are considered.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Robust mean absolute deviation

# Notes
- More robust to outliers than standard MAD
- Only includes voxels within 10th-90th percentile range
- NaN values are ignored
- IBSI ID: 1128

# Example
```julia
voxels = collect(1.0:100.0)
rmad = robust_mean_absolute_deviation(voxels)
```

# References
- PyRadiomics: radiomics/firstorder.py:getRobustMeanAbsoluteDeviationFeatureValue
"""
function robust_mean_absolute_deviation(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN

    # Get 10th and 90th percentile bounds
    p10 = quantile(valid, 0.10)
    p90 = quantile(valid, 0.90)

    # Filter to robust range (inclusive)
    robust_voxels = filter(v -> p10 <= v <= p90, valid)

    isempty(robust_voxels) && return NaN

    # Compute MAD on robust subset
    μ = mean(robust_voxels)
    return Float64(mean(abs.(robust_voxels .- μ)))
end

#==============================================================================#
# Feature 14: Root Mean Squared (IBSI: 5ZWQ)
#==============================================================================#

"""
    root_mean_squared(voxels::AbstractVector{<:Real}; shift::Real=0) -> Float64

Compute the root mean squared of voxel intensities.

# Mathematical Formula
```
RMS = √[(1/Nₚ) × Σᵢ₌₁ᴺᵖ (X(i) + c)²]
```
where c is the shift parameter.

# Arguments
- `voxels`: Vector of voxel intensity values
- `shift::Real=0`: Shift applied to values before squaring (PyRadiomics voxelArrayShift)

# Returns
- `Float64`: Root mean squared value

# Notes
- Also known as quadratic mean
- Volume-confounded when shift ≠ 0
- NaN values are ignored
- IBSI ID: 5ZWQ

# Example
```julia
voxels = [1.0, 2.0, 3.0, 4.0, 5.0]
rms = root_mean_squared(voxels)  # √(55/5) ≈ 3.317
```

# References
- PyRadiomics: radiomics/firstorder.py:getRootMeanSquaredFeatureValue
"""
function root_mean_squared(voxels::AbstractVector{<:Real}; shift::Real=0)
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    n = length(valid)
    return Float64(sqrt(sum(v -> (v + shift)^2, valid) / n))
end

#==============================================================================#
# Feature 15: Standard Deviation (NOT IN IBSI - DEPRECATED)
#==============================================================================#

"""
    standard_deviation(voxels::AbstractVector{<:Real}) -> Float64

Compute the standard deviation of voxel intensities.

# Mathematical Formula
```
σ = √[(1/Nₚ) × Σᵢ₌₁ᴺᵖ (X(i) - X̄)²]
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Standard deviation (population formula, corrected=false)

# Notes
- DEPRECATED in PyRadiomics (correlated with variance)
- Uses population formula (N divisor, not N-1) to match PyRadiomics
- NOT in IBSI standard

# Example
```julia
voxels = [10.0, 20.0, 30.0, 40.0]
s = standard_deviation(voxels)  # √125 ≈ 11.18
```

# References
- PyRadiomics: radiomics/firstorder.py:getStandardDeviationFeatureValue
"""
function standard_deviation(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    # Use population std (corrected=false) to match PyRadiomics np.nanstd
    return Float64(std(valid; corrected=false))
end

#==============================================================================#
# Feature 16: Skewness (IBSI: KE2A)
#==============================================================================#

"""
    skewness(voxels::AbstractVector{<:Real}) -> Float64

Compute the skewness of voxel intensity distribution.

# Mathematical Formula
```
Skewness = μ₃ / σ³ = [(1/Nₚ) × Σᵢ(X(i) - X̄)³] / σ³
```
where μ₃ is the third central moment and σ is the population standard deviation.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Skewness value (0 for flat regions)

# Notes
- Positive skewness: right tail is longer
- Negative skewness: left tail is longer
- Returns 0 for constant intensity (avoids division by zero)
- NaN values are ignored
- IBSI ID: KE2A

# Example
```julia
voxels = [1.0, 2.0, 3.0, 4.0, 10.0]  # Right-skewed
s = skewness(voxels)  # Positive value
```

# References
- PyRadiomics: radiomics/firstorder.py:getSkewnessFeatureValue
"""
function skewness(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN

    n = length(valid)
    μ = mean(valid)

    # Compute second and third central moments
    m2 = sum((v - μ)^2 for v in valid) / n
    m3 = sum((v - μ)^3 for v in valid) / n

    # Avoid division by zero for constant intensity
    if m2 < eps(Float64)
        return 0.0
    end

    return Float64(m3 / (m2^1.5))
end

#==============================================================================#
# Feature 17: Kurtosis (IBSI: IPH6)
#==============================================================================#

"""
    kurtosis(voxels::AbstractVector{<:Real}) -> Float64

Compute the kurtosis of voxel intensity distribution.

# Mathematical Formula
```
Kurtosis = μ₄ / σ⁴ = [(1/Nₚ) × Σᵢ(X(i) - X̄)⁴] / σ⁴
```
where μ₄ is the fourth central moment and σ is the population standard deviation.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Kurtosis value (0 for flat regions)

# Notes
- **IMPORTANT**: PyRadiomics returns non-excess kurtosis (μ₄/σ⁴)
- IBSI expects excess kurtosis (μ₄/σ⁴ - 3)
- This implementation matches PyRadiomics (NOT excess kurtosis)
- Normal distribution has kurtosis ≈ 3 (or excess kurtosis ≈ 0)
- Returns 0 for constant intensity
- NaN values are ignored
- IBSI ID: IPH6 (with different convention)

# Example
```julia
# Normal-like distribution
voxels = randn(1000)
k = kurtosis(voxels)  # ≈ 3.0 (non-excess)
```

# References
- PyRadiomics: radiomics/firstorder.py:getKurtosisFeatureValue
"""
function kurtosis(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN

    n = length(valid)
    μ = mean(valid)

    # Compute second and fourth central moments
    m2 = sum((v - μ)^2 for v in valid) / n
    m4 = sum((v - μ)^4 for v in valid) / n

    # Avoid division by zero for constant intensity
    if m2 < eps(Float64)
        return 0.0
    end

    # Return non-excess kurtosis (matches PyRadiomics)
    return Float64(m4 / (m2^2))
end

#==============================================================================#
# Feature 18: Variance (IBSI: ECT3)
#==============================================================================#

"""
    fo_variance(voxels::AbstractVector{<:Real}) -> Float64

Compute the variance of voxel intensities.

# Mathematical Formula
```
Variance = (1/Nₚ) × Σᵢ₌₁ᴺᵖ (X(i) - X̄)²
```

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Variance (population formula)

# Notes
- Uses population formula (N divisor) to match PyRadiomics
- NaN values are ignored
- IBSI ID: ECT3

# Example
```julia
voxels = [10.0, 20.0, 30.0, 40.0]
v = fo_variance(voxels)  # 125.0
```

# References
- PyRadiomics: radiomics/firstorder.py:getVarianceFeatureValue
"""
function fo_variance(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN
    # Use population variance (corrected=false) to match PyRadiomics
    return Float64(var(valid; corrected=false))
end

#==============================================================================#
# Feature 19: Uniformity (IBSI: BJ5W)
#==============================================================================#

"""
    uniformity(voxels::AbstractVector{<:Real}) -> Float64

Compute the uniformity of voxel intensity distribution.

# Mathematical Formula
```
Uniformity = Σᵢ₌₁ᴺᵍ p(i)²
```
where p(i) is the probability of gray level i.

# Arguments
- `voxels`: Vector of voxel intensity values

# Returns
- `Float64`: Uniformity value (0 to 1)

# Notes
- Higher values indicate more homogeneous distribution
- Maximum value of 1 when all voxels have same intensity
- Inverse relationship with entropy
- NaN values are ignored
- IBSI ID: BJ5W

# Example
```julia
# Uniform values (high uniformity)
voxels = [5.0, 5.0, 5.0, 5.0]
u = uniformity(voxels)  # 1.0

# Diverse values (lower uniformity)
voxels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
u = uniformity(voxels)  # 0.125
```

# References
- PyRadiomics: radiomics/firstorder.py:getUniformityFeatureValue
"""
function uniformity(voxels::AbstractVector{<:Real})
    valid = filter(!isnan, voxels)
    isempty(valid) && return NaN

    # Count occurrences of each unique value
    counts = countmap(valid)
    n = length(valid)

    # Compute uniformity as sum of squared probabilities
    U = 0.0
    for count in values(counts)
        p = count / n
        U += p^2
    end

    return U
end

#==============================================================================#
# High-Level Extraction Functions
#==============================================================================#

"""
    extract_firstorder(image, mask; label::Int=1, shift::Real=0,
                       voxel_volume::Real=1.0) -> Dict{String, Float64}

Extract all first-order features from an image within a masked region.

# Arguments
- `image`: The image data (Array or RadiomicsImage)
- `mask`: The segmentation mask (Array, RadiomicsMask, BitArray)
- `label::Int=1`: Label value in mask defining the ROI
- `shift::Real=0`: Shift for Energy/RMS calculations (PyRadiomics voxelArrayShift)
- `voxel_volume::Real=1.0`: Volume of a single voxel in mm³ (for TotalEnergy)

# Returns
- `Dict{String, Float64}`: Dictionary of feature names to values

# Feature Names
The returned dictionary uses PyRadiomics-compatible names:
- "Energy", "TotalEnergy", "Entropy", "Minimum", "10Percentile",
- "90Percentile", "Maximum", "Mean", "Median", "InterquartileRange",
- "Range", "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation",
- "RootMeanSquared", "StandardDeviation", "Skewness", "Kurtosis",
- "Variance", "Uniformity"

# Example
```julia
image = rand(64, 64, 64) * 100
mask = rand(Bool, 64, 64, 64)
features = extract_firstorder(image, mask)
println(features["Energy"])
println(features["Entropy"])
```

# Notes
- Extracts all 19 first-order features in a single pass
- For texture analysis, discretize voxels before calling (affects Entropy, Uniformity)
- StandardDeviation is deprecated but included for PyRadiomics compatibility

# See also
- Individual feature functions: `energy`, `entropy`, etc.
"""
function extract_firstorder(image::AbstractArray{T, N}, mask::AbstractArray;
                            label::Int=1,
                            shift::Real=0,
                            voxel_volume::Real=1.0) where {T<:Real, N}
    # Extract voxels from ROI
    voxels = get_voxels(image, mask; label=label)

    # Compute all features
    return Dict{String, Float64}(
        "Energy" => energy(voxels; shift=shift),
        "TotalEnergy" => total_energy(voxels, voxel_volume; shift=shift),
        "Entropy" => entropy(voxels),
        "Minimum" => fo_minimum(voxels),
        "10Percentile" => percentile_10(voxels),
        "90Percentile" => percentile_90(voxels),
        "Maximum" => fo_maximum(voxels),
        "Mean" => fo_mean(voxels),
        "Median" => fo_median(voxels),
        "InterquartileRange" => interquartile_range(voxels),
        "Range" => fo_range(voxels),
        "MeanAbsoluteDeviation" => mean_absolute_deviation(voxels),
        "RobustMeanAbsoluteDeviation" => robust_mean_absolute_deviation(voxels),
        "RootMeanSquared" => root_mean_squared(voxels; shift=shift),
        "StandardDeviation" => standard_deviation(voxels),
        "Skewness" => skewness(voxels),
        "Kurtosis" => kurtosis(voxels),
        "Variance" => fo_variance(voxels),
        "Uniformity" => uniformity(voxels)
    )
end

# Convenience method for RadiomicsImage
function extract_firstorder(image::RadiomicsImage, mask;
                            label::Int=1, shift::Real=0)
    voxel_vol = voxel_volume(image)
    return extract_firstorder(image.data, mask;
                             label=label, shift=shift, voxel_volume=voxel_vol)
end

"""
    extract_firstorder_to_featureset!(fs::FeatureSet, image, mask;
                                      label::Int=1, shift::Real=0,
                                      voxel_volume::Real=1.0,
                                      image_type::String="original")

Extract first-order features and add them to an existing FeatureSet.

# Arguments
- `fs`: FeatureSet to add features to
- `image`: The image data
- `mask`: The segmentation mask
- `label::Int=1`: Label value in mask
- `shift::Real=0`: Shift for Energy/RMS calculations
- `voxel_volume::Real=1.0`: Volume of a single voxel in mm³
- `image_type::String="original"`: Image type label for feature keys

# Returns
- Modified FeatureSet (also modifies in-place)

# Example
```julia
fs = FeatureSet()
extract_firstorder_to_featureset!(fs, image, mask)
println(fs["firstorder_Energy"])
```
"""
function extract_firstorder_to_featureset!(fs::FeatureSet, image::AbstractArray, mask::AbstractArray;
                                           label::Int=1,
                                           shift::Real=0,
                                           voxel_volume::Real=1.0,
                                           image_type::String="original")
    features = extract_firstorder(image, mask; label=label, shift=shift, voxel_volume=voxel_volume)

    for (name, value) in features
        push!(fs, FeatureResult(name, value, "firstorder", image_type))
    end

    return fs
end

#==============================================================================#
# Feature List and Names
#==============================================================================#

"""
    firstorder_feature_names() -> Vector{String}

Return a list of all first-order feature names.

# Returns
- `Vector{String}`: Names of all 19 first-order features

# Example
```julia
names = firstorder_feature_names()
println(length(names))  # 19
```
"""
function firstorder_feature_names()
    return [
        "Energy",
        "TotalEnergy",
        "Entropy",
        "Minimum",
        "10Percentile",
        "90Percentile",
        "Maximum",
        "Mean",
        "Median",
        "InterquartileRange",
        "Range",
        "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation",
        "RootMeanSquared",
        "StandardDeviation",
        "Skewness",
        "Kurtosis",
        "Variance",
        "Uniformity"
    ]
end

"""
    firstorder_ibsi_features() -> Vector{String}

Return a list of IBSI-compliant first-order features.

# Returns
- `Vector{String}`: Names of 18 IBSI-compliant features (excludes TotalEnergy and StandardDeviation)

# Notes
- TotalEnergy is a PyRadiomics extension, not in IBSI
- StandardDeviation is deprecated (use Variance)
"""
function firstorder_ibsi_features()
    return [
        "Energy",
        "Entropy",
        "Minimum",
        "10Percentile",
        "90Percentile",
        "Maximum",
        "Mean",
        "Median",
        "InterquartileRange",
        "Range",
        "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation",
        "RootMeanSquared",
        "Skewness",
        "Kurtosis",
        "Variance",
        "Uniformity"
    ]
end
