# Neighbouring Gray Tone Difference Matrix (NGTDM) for Radiomics.jl
#
# This module implements NGTDM computation and features.
# NGTDM quantifies the difference between a voxel's gray level and the average
# gray level of its neighbors within a specified distance.
#
# Unlike GLCM/GLRLM/GLSZM which are 2D matrices, NGTDM produces two 1D vectors:
# - s_i: Sum of absolute differences from neighborhood averages for each gray level
# - n_i: Count of voxels at each gray level (with valid neighborhoods)
#
# Total: 5 features
#
# References:
# - PyRadiomics: radiomics/ngtdm.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - Amadasun M, King R; "Textural features corresponding to textural properties";
#   IEEE Transactions on Systems, Man and Cybernetics 19:1264-1274 (1989).
#   DOI: 10.1109/21.44046

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding division by zero - matches np.spacing(1)
const NGTDM_EPSILON = eps(Float64)  # ≈ 2.2e-16

# Large value returned for Coarseness when denominator is zero (homogeneous image)
const NGTDM_COARSENESS_MAX = 1e6

#==============================================================================#
# NGTDM Result Type
#==============================================================================#

"""
    NGTDMResult

Container for NGTDM computation results.

# Fields
- `s_i::Vector{Float64}`: Sum of absolute differences for each gray level
- `n_i::Vector{Int}`: Voxel count for each gray level (with valid neighborhoods)
- `p_i::Vector{Float64}`: Probability for each gray level (n_i / N_v_p)
- `gray_levels::Vector{Int}`: Gray level values (indices with n_i > 0)
- `N_v_p::Int`: Total number of valid voxels (sum of n_i)
- `N_g_p::Int`: Number of gray levels with non-zero counts
- `Ng::Int`: Total number of gray levels in discretized image

# Notes
The NGTDM is stored as vectors indexed by gray level. Empty gray levels
(where n_i = 0) are tracked in gray_levels for iteration but the full
s_i and n_i vectors cover all Ng gray levels.

All masked voxels are included in n_i counts. Voxels with no valid neighbors
contribute to n_i but have diff=0 (so they don't contribute to s_i).
This matches PyRadiomics behavior.
"""
struct NGTDMResult
    s_i::Vector{Float64}      # Sum of |i - Ā_i| for all voxels at gray level i
    n_i::Vector{Int}          # Count of voxels at gray level i
    p_i::Vector{Float64}      # Probability n_i / N_v_p
    gray_levels::Vector{Int}  # Non-empty gray level values
    N_v_p::Int               # Total valid voxels
    N_g_p::Int               # Number of non-empty gray levels
    Ng::Int                  # Total gray levels
end

"""
    NGTDMResult2D

Container for 2D NGTDM computation results.

# Fields
Same as NGTDMResult.
"""
struct NGTDMResult2D
    s_i::Vector{Float64}
    n_i::Vector{Int}
    p_i::Vector{Float64}
    gray_levels::Vector{Int}
    N_v_p::Int
    N_g_p::Int
    Ng::Int
end

#==============================================================================#
# Neighborhood Computation Helpers
#==============================================================================#

"""
    _get_chebyshev_offsets_3d(distance::Int) -> Vector{CartesianIndex{3}}

Get all neighbor offsets within Chebyshev distance in 3D.

Chebyshev distance (L∞ norm) gives a cube-shaped neighborhood.
For distance=1, returns 26 neighbors (3×3×3 cube minus center).

# Arguments
- `distance::Int`: Maximum Chebyshev distance (typically 1)

# Returns
- Vector of CartesianIndex offsets for all neighbors
"""
function _get_chebyshev_offsets_3d(distance::Int)
    offsets = CartesianIndex{3}[]
    for dz in -distance:distance
        for dy in -distance:distance
            for dx in -distance:distance
                if !(dz == 0 && dy == 0 && dx == 0)
                    push!(offsets, CartesianIndex(dx, dy, dz))
                end
            end
        end
    end
    return offsets
end

"""
    _get_chebyshev_offsets_2d(distance::Int) -> Vector{CartesianIndex{2}}

Get all neighbor offsets within Chebyshev distance in 2D.

For distance=1, returns 8 neighbors (3×3 square minus center).

# Arguments
- `distance::Int`: Maximum Chebyshev distance (typically 1)

# Returns
- Vector of CartesianIndex offsets for all neighbors
"""
function _get_chebyshev_offsets_2d(distance::Int)
    offsets = CartesianIndex{2}[]
    for dy in -distance:distance
        for dx in -distance:distance
            if !(dy == 0 && dx == 0)
                push!(offsets, CartesianIndex(dx, dy))
            end
        end
    end
    return offsets
end

#==============================================================================#
# NGTDM Matrix Computation - 3D
#==============================================================================#

"""
    compute_ngtdm(image::AbstractArray{<:Integer, 3}, mask::AbstractArray{Bool, 3};
                  Ng::Union{Int, Nothing}=nothing,
                  distance::Int=1) -> NGTDMResult

Compute Neighbouring Gray Tone Difference Matrix for a 3D image.

The NGTDM computes for each gray level i:
- n_i: Number of voxels with gray level i (that have valid neighborhoods)
- s_i: Sum of |i - Ā_i| for all voxels at gray level i

where Ā_i is the average gray level of valid neighbors.

# Arguments
- `image`: 3D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 3D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)
- `distance::Int=1`: Chebyshev distance for neighborhood (default: 1)

# Returns
- `NGTDMResult`: Struct containing:
  - `s_i`: Sum of differences vector
  - `n_i`: Voxel count vector
  - `p_i`: Probability vector (n_i / N_v_p)
  - `gray_levels`: Non-empty gray level indices
  - `N_v_p`: Total valid voxels
  - `N_g_p`: Number of non-empty gray levels
  - `Ng`: Total gray levels

# Notes
- Input image should be discretized (integer gray levels starting from 1)
- All masked voxels are included in n_i counts
- Voxels with no neighbors in mask contribute to n_i but have diff=0 for s_i
- This matches PyRadiomics behavior

# Example
```julia
# Discretize image first
discretized = discretize_image(image, mask, binwidth=25.0)
result = compute_ngtdm(discretized.discretized, mask)

# Access vectors
s = result.s_i
n = result.n_i
```

# References
- PyRadiomics: radiomics/ngtdm.py
- IBSI: Section 3.6.6 (Neighbouring grey tone difference based features)
"""
function compute_ngtdm(image::AbstractArray{<:Integer, 3},
                       mask::AbstractArray{Bool, 3};
                       Ng::Union{Int, Nothing}=nothing,
                       distance::Int=1)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive, got $distance"))

    # Get masked voxels
    masked_values = image[mask]
    if isempty(masked_values)
        throw(ArgumentError("Mask is empty, no voxels to process"))
    end

    # Determine number of gray levels
    if isnothing(Ng)
        Ng = maximum(masked_values)
    end
    Ng > 0 || throw(ArgumentError("Ng must be positive, got $Ng"))

    # Initialize vectors
    s_i = zeros(Float64, Ng)
    n_i = zeros(Int, Ng)

    # Get neighbor offsets
    offsets = _get_chebyshev_offsets_3d(distance)

    sz = size(image)

    # Iterate over all voxels in the mask
    @inbounds for idx in CartesianIndices(image)
        # Skip if not in mask
        mask[idx] || continue

        gray_level = image[idx]
        if gray_level < 1 || gray_level > Ng
            continue  # Invalid gray level
        end

        # Compute neighborhood average
        neighbor_sum = 0.0
        valid_neighbor_count = 0

        for offset in offsets
            neighbor_idx = idx + offset

            # Check bounds
            if checkbounds(Bool, image, neighbor_idx)
                # Check if neighbor is in mask
                if mask[neighbor_idx]
                    neighbor_sum += image[neighbor_idx]
                    valid_neighbor_count += 1
                end
            end
        end

        # Include all voxels in count (matching PyRadiomics behavior)
        # For voxels with no valid neighbors, diff=0 (s_i unchanged)
        n_i[gray_level] += 1
        if valid_neighbor_count > 0
            avg_neighbor = neighbor_sum / valid_neighbor_count
            diff = abs(Float64(gray_level) - avg_neighbor)
            s_i[gray_level] += diff
        end
        # Note: Voxels with no neighbors contribute to n_i but not s_i
        # This matches PyRadiomics behavior where isolated voxels have diff=0
    end

    # Compute derived values
    N_v_p = sum(n_i)
    gray_levels = findall(x -> x > 0, n_i)
    N_g_p = length(gray_levels)

    # Compute probabilities
    p_i = if N_v_p > 0
        n_i ./ N_v_p
    else
        zeros(Float64, Ng)
    end

    return NGTDMResult(s_i, n_i, p_i, gray_levels, N_v_p, N_g_p, Ng)
end

#==============================================================================#
# NGTDM Matrix Computation - 2D
#==============================================================================#

"""
    compute_ngtdm_2d(image::AbstractArray{<:Integer, 2}, mask::AbstractArray{Bool, 2};
                     Ng::Union{Int, Nothing}=nothing,
                     distance::Int=1) -> NGTDMResult2D

Compute Neighbouring Gray Tone Difference Matrix for a 2D image.

# Arguments
- `image`: 2D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 2D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)
- `distance::Int=1`: Chebyshev distance for neighborhood (default: 1)

# Returns
- `NGTDMResult2D`: Struct containing NGTDM vectors

# Example
```julia
image_2d = discretize(image_slice, edges)
mask_2d = mask[:, :, 32]
result = compute_ngtdm_2d(image_2d, mask_2d)
```
"""
function compute_ngtdm_2d(image::AbstractArray{<:Integer, 2},
                          mask::AbstractArray{Bool, 2};
                          Ng::Union{Int, Nothing}=nothing,
                          distance::Int=1)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive"))

    # Get masked voxels
    masked_values = image[mask]
    if isempty(masked_values)
        throw(ArgumentError("Mask is empty"))
    end

    # Determine number of gray levels
    if isnothing(Ng)
        Ng = maximum(masked_values)
    end

    # Initialize vectors
    s_i = zeros(Float64, Ng)
    n_i = zeros(Int, Ng)

    # Get neighbor offsets
    offsets = _get_chebyshev_offsets_2d(distance)

    # Iterate over all voxels in the mask
    @inbounds for idx in CartesianIndices(image)
        mask[idx] || continue

        gray_level = image[idx]
        if gray_level < 1 || gray_level > Ng
            continue
        end

        # Compute neighborhood average
        neighbor_sum = 0.0
        valid_neighbor_count = 0

        for offset in offsets
            neighbor_idx = idx + offset

            if checkbounds(Bool, image, neighbor_idx) && mask[neighbor_idx]
                neighbor_sum += image[neighbor_idx]
                valid_neighbor_count += 1
            end
        end

        # Include all voxels in count (matching PyRadiomics behavior)
        # For voxels with no valid neighbors, diff=0 (s_i unchanged)
        n_i[gray_level] += 1
        if valid_neighbor_count > 0
            avg_neighbor = neighbor_sum / valid_neighbor_count
            diff = abs(Float64(gray_level) - avg_neighbor)
            s_i[gray_level] += diff
        end
        # Note: Voxels with no neighbors contribute to n_i but not s_i
        # This matches PyRadiomics behavior where isolated voxels have diff=0
    end

    # Compute derived values
    N_v_p = sum(n_i)
    gray_levels = findall(x -> x > 0, n_i)
    N_g_p = length(gray_levels)

    p_i = N_v_p > 0 ? n_i ./ N_v_p : zeros(Float64, Ng)

    return NGTDMResult2D(s_i, n_i, p_i, gray_levels, N_v_p, N_g_p, Ng)
end

#==============================================================================#
# High-Level NGTDM Computation Interface
#==============================================================================#

"""
    compute_ngtdm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                  binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                  distance::Int=1) -> NGTDMResult

Compute NGTDM from a non-discretized image (convenience wrapper).

This function discretizes the image before computing the NGTDM.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `distance::Int=1`: Chebyshev distance for neighborhood

# Returns
- For 3D images: `NGTDMResult`
- For 2D images: `NGTDMResult2D`

# Example
```julia
result = compute_ngtdm(image, mask, binwidth=25.0)
```
"""
function compute_ngtdm(image::AbstractArray{<:Real, 3},
                       mask::AbstractArray{Bool, 3};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing,
                       distance::Int=1)

    # Discretize the image
    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Compute NGTDM on discretized image
    return compute_ngtdm(disc_result.discretized, mask; distance=distance)
end

function compute_ngtdm(image::AbstractArray{<:Real, 2},
                       mask::AbstractArray{Bool, 2};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing,
                       distance::Int=1)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    return compute_ngtdm_2d(disc_result.discretized, mask; distance=distance)
end

#==============================================================================#
# NGTDM with Settings
#==============================================================================#

"""
    compute_ngtdm(image, mask, settings::Settings)

Compute NGTDM using parameters from a Settings object.

# Example
```julia
settings = Settings(binwidth=32.0, ngtdm_distance=2)
result = compute_ngtdm(image, mask, settings)
```
"""
function compute_ngtdm(image::AbstractArray{<:Real},
                       mask::AbstractArray{Bool},
                       settings::Settings)

    bincount = settings.discretization_mode == FixedBinCount ? settings.bincount : nothing

    return compute_ngtdm(image, mask;
                         binwidth=settings.binwidth,
                         bincount=bincount,
                         distance=settings.ngtdm_distance)
end

#==============================================================================#
# NGTDM Utility Functions
#==============================================================================#

"""
    ngtdm_num_gray_levels(result::Union{NGTDMResult, NGTDMResult2D}) -> Int

Get the total number of gray levels in the NGTDM.
"""
ngtdm_num_gray_levels(result::NGTDMResult) = result.Ng
ngtdm_num_gray_levels(result::NGTDMResult2D) = result.Ng

"""
    ngtdm_num_valid_gray_levels(result::Union{NGTDMResult, NGTDMResult2D}) -> Int

Get the number of non-empty gray levels (N_g,p).
"""
ngtdm_num_valid_gray_levels(result::NGTDMResult) = result.N_g_p
ngtdm_num_valid_gray_levels(result::NGTDMResult2D) = result.N_g_p

"""
    ngtdm_num_valid_voxels(result::Union{NGTDMResult, NGTDMResult2D}) -> Int

Get the total number of valid voxels (N_v,p).
"""
ngtdm_num_valid_voxels(result::NGTDMResult) = result.N_v_p
ngtdm_num_valid_voxels(result::NGTDMResult2D) = result.N_v_p

"""
    ngtdm_sum_s(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Get the sum of all s_i values (Σ s_i).
"""
ngtdm_sum_s(result::NGTDMResult) = sum(result.s_i)
ngtdm_sum_s(result::NGTDMResult2D) = sum(result.s_i)

#==============================================================================#
# NGTDM Features (5 Total)
#==============================================================================#

"""
    ngtdm_coarseness(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Compute the Coarseness feature from the NGTDM.

# Mathematical Formula
``\\text{Coarseness} = \\frac{1}{\\sum_i p_i \\times s_i}``

# Description
Measures the spatial rate of change in intensity. Higher values indicate locally
uniform textures with low spatial change rates. A coarse texture has larger
homogeneous regions.

# Edge Case
If Σ(p_i × s_i) ≈ 0 (completely homogeneous image), returns 10^6 to avoid
division by zero.

# Returns
- `Float64`: Coarseness value

# References
- PyRadiomics: radiomics/ngtdm.py:getCoarsenessFeatureValue()
- IBSI: Section 3.6.6.1
- Amadasun & King (1989), DOI: 10.1109/21.44046
"""
function ngtdm_coarseness(result::Union{NGTDMResult, NGTDMResult2D})
    # Sum of p_i * s_i over non-empty gray levels
    sum_coarse = 0.0
    @inbounds for i in result.gray_levels
        sum_coarse += result.p_i[i] * result.s_i[i]
    end

    # Handle homogeneous image case
    if sum_coarse < NGTDM_EPSILON
        return NGTDM_COARSENESS_MAX
    end

    return 1.0 / sum_coarse
end

"""
    ngtdm_contrast(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Compute the Contrast feature from the NGTDM.

# Mathematical Formula
``\\text{Contrast} = \\left[\\frac{1}{N_{g,p}(N_{g,p}-1)} \\sum_i \\sum_j p_i p_j (i-j)^2\\right] \\times \\left[\\frac{1}{N_{v,p}} \\sum_i s_i\\right]``

# Description
Measures both the dynamic range of gray levels and the spatial intensity change.
Higher values indicate large changes between voxels and their neighborhoods,
as well as large gray level differences.

# Edge Case
If N_g,p = 1 (only one gray level), returns 0.

# Returns
- `Float64`: Contrast value

# References
- PyRadiomics: radiomics/ngtdm.py:getContrastFeatureValue()
- IBSI: Section 3.6.6.2
- Amadasun & King (1989), DOI: 10.1109/21.44046
"""
function ngtdm_contrast(result::Union{NGTDMResult, NGTDMResult2D})
    N_g_p = result.N_g_p
    N_v_p = result.N_v_p

    # Edge case: only one gray level
    if N_g_p <= 1
        return 0.0
    end

    # Term 1: Gray level variance (weighted by probabilities)
    # Σᵢ Σⱼ p_i × p_j × (i - j)²
    term1 = 0.0
    @inbounds for i in result.gray_levels
        for j in result.gray_levels
            term1 += result.p_i[i] * result.p_i[j] * (i - j)^2
        end
    end
    term1 /= (N_g_p * (N_g_p - 1))

    # Term 2: Sum of differences normalized by voxel count
    term2 = sum(result.s_i) / N_v_p

    return term1 * term2
end

"""
    ngtdm_busyness(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Compute the Busyness feature from the NGTDM.

# Mathematical Formula
``\\text{Busyness} = \\frac{\\sum_i p_i s_i}{\\sum_i \\sum_j |i \\cdot p_i - j \\cdot p_j|}``

# Description
Measures the change from a pixel to its neighbor. A high busyness value
indicates a "busy" image with rapid changes of intensity between pixels
and their neighborhoods.

# Edge Case
If N_g,p = 1, returns 0 (denominator would be zero).

# Returns
- `Float64`: Busyness value

# References
- PyRadiomics: radiomics/ngtdm.py:getBusynessFeatureValue()
- IBSI: Section 3.6.6.3
- Amadasun & King (1989), DOI: 10.1109/21.44046
"""
function ngtdm_busyness(result::Union{NGTDMResult, NGTDMResult2D})
    N_g_p = result.N_g_p

    # Edge case: only one gray level
    if N_g_p <= 1
        return 0.0
    end

    # Numerator: Σ p_i × s_i
    numerator = 0.0
    @inbounds for i in result.gray_levels
        numerator += result.p_i[i] * result.s_i[i]
    end

    # Denominator: Σᵢ Σⱼ |i × p_i - j × p_j|
    denominator = 0.0
    @inbounds for i in result.gray_levels
        for j in result.gray_levels
            denominator += abs(i * result.p_i[i] - j * result.p_i[j])
        end
    end

    # Avoid division by zero
    if denominator < NGTDM_EPSILON
        return 0.0
    end

    return numerator / denominator
end

"""
    ngtdm_complexity(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Compute the Complexity feature from the NGTDM.

# Mathematical Formula
``\\text{Complexity} = \\frac{1}{N_{v,p}} \\sum_i \\sum_j \\frac{|i-j| \\times (p_i s_i + p_j s_j)}{p_i + p_j}``

# Description
An image is considered complex when there are many primitive components
(i.e., the image is non-uniform with many rapid changes in gray level intensity).
The feature measures both the number of primitives and the average difference
from the mean value.

# Edge Case
When p_i + p_j = 0, that term is skipped.

# Returns
- `Float64`: Complexity value

# References
- PyRadiomics: radiomics/ngtdm.py:getComplexityFeatureValue()
- IBSI: Section 3.6.6.4
- Amadasun & King (1989), DOI: 10.1109/21.44046
"""
function ngtdm_complexity(result::Union{NGTDMResult, NGTDMResult2D})
    N_v_p = result.N_v_p

    if N_v_p == 0
        return 0.0
    end

    total = 0.0
    @inbounds for i in result.gray_levels
        p_i = result.p_i[i]
        ps_i = p_i * result.s_i[i]

        for j in result.gray_levels
            p_j = result.p_i[j]
            p_sum = p_i + p_j

            # Skip if denominator is zero
            if p_sum > NGTDM_EPSILON
                ps_j = p_j * result.s_i[j]
                diff_ij = abs(i - j)
                total += diff_ij * (ps_i + ps_j) / p_sum
            end
        end
    end

    return total / N_v_p
end

"""
    ngtdm_strength(result::Union{NGTDMResult, NGTDMResult2D}) -> Float64

Compute the Strength feature from the NGTDM.

# Mathematical Formula
``\\text{Strength} = \\frac{\\sum_i \\sum_j (p_i + p_j)(i-j)^2}{\\sum_i s_i}``

# Description
Measures the primitives in an image. A high strength value indicates primitives
that are easily defined and visible (an image with slow change in intensity
but large coarse differences in gray level intensities).

# Edge Case
If Σ s_i = 0, returns 0.

# Returns
- `Float64`: Strength value

# References
- PyRadiomics: radiomics/ngtdm.py:getStrengthFeatureValue()
- IBSI: Section 3.6.6.5
- Amadasun & King (1989), DOI: 10.1109/21.44046
"""
function ngtdm_strength(result::Union{NGTDMResult, NGTDMResult2D})
    sum_s = sum(result.s_i)

    # Edge case: no differences
    if sum_s < NGTDM_EPSILON
        return 0.0
    end

    # Numerator: Σᵢ Σⱼ (p_i + p_j) × (i - j)²
    numerator = 0.0
    @inbounds for i in result.gray_levels
        for j in result.gray_levels
            p_sum = result.p_i[i] + result.p_i[j]
            diff_sq = (i - j)^2
            numerator += p_sum * diff_sq
        end
    end

    return numerator / sum_s
end

#==============================================================================#
# Convenience Functions for All NGTDM Features
#==============================================================================#

"""
    compute_all_ngtdm_features(result::Union{NGTDMResult, NGTDMResult2D}) -> NamedTuple

Compute all 5 NGTDM features at once.

# Returns
A NamedTuple with fields:
- `coarseness::Float64`
- `contrast::Float64`
- `busyness::Float64`
- `complexity::Float64`
- `strength::Float64`

# Example
```julia
result = compute_ngtdm(image, mask, binwidth=25.0)
features = compute_all_ngtdm_features(result)
println("Coarseness: ", features.coarseness)
```
"""
function compute_all_ngtdm_features(result::Union{NGTDMResult, NGTDMResult2D})
    return (
        coarseness = ngtdm_coarseness(result),
        contrast = ngtdm_contrast(result),
        busyness = ngtdm_busyness(result),
        complexity = ngtdm_complexity(result),
        strength = ngtdm_strength(result)
    )
end

"""
    compute_all_ngtdm_features(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                               binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                               distance::Int=1) -> NamedTuple

Compute all 5 NGTDM features directly from image and mask.

# Arguments
- `image`: Image array
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `distance::Int=1`: Chebyshev distance for neighborhood

# Returns
A NamedTuple with all 5 NGTDM features.

# Example
```julia
features = compute_all_ngtdm_features(image, mask, binwidth=25.0)
```
"""
function compute_all_ngtdm_features(image::AbstractArray{<:Real},
                                    mask::AbstractArray{Bool};
                                    binwidth::Real=25.0,
                                    bincount::Union{Int, Nothing}=nothing,
                                    distance::Int=1)
    result = compute_ngtdm(image, mask; binwidth=binwidth, bincount=bincount, distance=distance)
    return compute_all_ngtdm_features(result)
end
