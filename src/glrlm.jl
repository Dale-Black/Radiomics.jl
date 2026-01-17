# Gray Level Run Length Matrix (GLRLM) for Radiomics.jl
#
# This module implements GLRLM computation and features.
# GLRLM captures texture information by examining the length of consecutive
# voxels with the same gray level intensity along specific directions.
#
# Total: 16 features
#
# References:
# - PyRadiomics: radiomics/glrlm.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - Galloway, M.M. (1975). "Texture analysis using gray level run lengths"

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding log(0) - matches np.spacing(1)
const GLRLM_EPSILON = eps(Float64)  # ≈ 2.2e-16

"""
    GLRLM_DIRECTIONS_3D

The 13 unique directions in 3D for GLRLM computation.

These directions define the scanning vectors for detecting runs.
Only 13 directions are needed (hemisphere) because runs in opposite
directions would be counted separately.

Each tuple (dz, dy, dx) represents a direction offset where:
- dz: offset in the third dimension (z-axis, or slice direction)
- dy: offset in the second dimension (y-axis, or row direction)
- dx: offset in the first dimension (x-axis, or column direction)

Note: These are the same 13 directions used in GLCM.

Direction descriptions:
1. (0, 0, 1)   - along x-axis
2. (0, 1, 0)   - along y-axis
3. (1, 0, 0)   - along z-axis
4. (0, 1, 1)   - xy diagonal
5. (0, 1, -1)  - xy anti-diagonal
6. (1, 0, 1)   - xz diagonal
7. (1, 0, -1)  - xz anti-diagonal
8. (1, 1, 0)   - yz diagonal
9. (1, -1, 0)  - yz anti-diagonal
10. (1, 1, 1)  - body diagonal
11. (1, 1, -1) - body diagonal variant
12. (1, -1, 1) - body diagonal variant
13. (1, -1, -1)- body diagonal variant
"""
const GLRLM_DIRECTIONS_3D = (
    (0, 0, 1),   # 1: X-axis
    (0, 1, 0),   # 2: Y-axis
    (1, 0, 0),   # 3: Z-axis
    (0, 1, 1),   # 4: XY diagonal
    (0, 1, -1),  # 5: XY anti-diagonal
    (1, 0, 1),   # 6: XZ diagonal
    (1, 0, -1),  # 7: XZ anti-diagonal
    (1, 1, 0),   # 8: YZ diagonal
    (1, -1, 0),  # 9: YZ anti-diagonal
    (1, 1, 1),   # 10: Body diagonal
    (1, 1, -1),  # 11: Body diagonal variant
    (1, -1, 1),  # 12: Body diagonal variant
    (1, -1, -1), # 13: Body diagonal variant
)

"""
    GLRLM_DIRECTIONS_2D

The 4 unique directions in 2D for GLRLM computation.

Direction descriptions:
1. (0, 1)  - along x-axis (0 degrees)
2. (1, 1)  - diagonal (45 degrees)
3. (1, 0)  - along y-axis (90 degrees)
4. (1, -1) - anti-diagonal (135 degrees)
"""
const GLRLM_DIRECTIONS_2D = (
    (0, 1),   # 1: X-axis (0 degrees)
    (1, 1),   # 2: Diagonal (45 degrees)
    (1, 0),   # 3: Y-axis (90 degrees)
    (1, -1),  # 4: Anti-diagonal (135 degrees)
)

#==============================================================================#
# GLRLM Result Type
#==============================================================================#

"""
    GLRLMResult

Container for GLRLM computation results.

# Fields
- `matrices::Array{Float64, 3}`: GLRLM matrices, shape (Ng, Nr, num_directions)
- `Ng::Int`: Number of gray levels
- `Nr::Int`: Maximum run length (longest possible run in the ROI)
- `num_directions::Int`: Number of directions computed (13 for 3D, 4 for 2D)
- `directions`: Direction offsets used
- `Ns::Vector{Int}`: Number of runs per direction (Ns[d] = sum of matrix[:,:,d])
- `Np::Vector{Int}`: Number of voxels per direction (sum of run lengths)

# Notes
The matrices are stored as a 3D array where:
- First dimension: gray level (i), indexed 1:Ng
- Second dimension: run length (j), indexed 1:Nr
- Third dimension: direction index

P[i, j, d] is the count of runs with gray level i and length j in direction d.
Note: Unlike GLCM, GLRLM is NOT normalized by default. Features use the raw
counts and normalize by Ns (total number of runs) internally.
"""
struct GLRLMResult
    matrices::Array{Float64, 3}
    Ng::Int
    Nr::Int
    num_directions::Int
    directions::Any
    Ns::Vector{Int}  # Number of runs per direction
    Np::Vector{Int}  # Number of voxels per direction (sum of i,j contributions)
end

"""
    GLRLMResult2D

Container for 2D GLRLM computation results.

# Fields
- `matrices::Array{Float64, 3}`: GLRLM matrices, shape (Ng, Nr, 4)
- `Ng::Int`: Number of gray levels
- `Nr::Int`: Maximum run length
- `Ns::Vector{Int}`: Number of runs per direction
- `Np::Vector{Int}`: Number of voxels per direction
"""
struct GLRLMResult2D
    matrices::Array{Float64, 3}
    Ng::Int
    Nr::Int
    Ns::Vector{Int}
    Np::Vector{Int}
end

#==============================================================================#
# Run Detection Algorithm
#==============================================================================#

"""
    _detect_runs_3d(image::AbstractArray{<:Integer, 3},
                    mask::AbstractArray{Bool, 3},
                    direction::Tuple{Int, Int, Int},
                    Ng::Int, Nr::Int) -> (Matrix{Float64}, Int, Int)

Detect runs along a single direction in a 3D image.

# Algorithm
1. Find all starting points (voxels where the previous voxel in the direction
   is either outside the mask or outside the image bounds)
2. For each starting point, trace the run until:
   - The next voxel has a different gray level
   - The next voxel is outside the mask
   - The next voxel is outside the image bounds
3. Record the (gray level, run length) in the GLRLM matrix

# Arguments
- `image`: 3D array of discretized gray levels (1:Ng)
- `mask`: 3D boolean mask
- `direction`: Direction offset (dz, dy, dx)
- `Ng`: Number of gray levels
- `Nr`: Maximum possible run length

# Returns
- `P`: GLRLM matrix for this direction (Ng × Nr)
- `Ns`: Total number of runs
- `Np`: Total number of voxels in runs
"""
function _detect_runs_3d(image::AbstractArray{<:Integer, 3},
                         mask::AbstractArray{Bool, 3},
                         direction::Tuple{Int, Int, Int},
                         Ng::Int, Nr::Int)

    P = zeros(Float64, Ng, Nr)
    Ns = 0  # Total number of runs
    Np = 0  # Total number of voxels

    sz = size(image)
    offset = CartesianIndex(direction)

    # Track which voxels have been counted as part of a run
    # to avoid double counting
    visited = falses(sz)

    # Iterate through all voxels
    @inbounds for idx in CartesianIndices(mask)
        # Skip if not in mask or already visited
        mask[idx] || continue
        visited[idx] && continue

        # Check if this is a valid starting point
        # A starting point is where the previous voxel (idx - offset):
        # 1. Is outside the image bounds, OR
        # 2. Is outside the mask, OR
        # 3. Has a different gray level
        prev_idx = idx - offset
        is_start = !checkbounds(Bool, mask, prev_idx) ||
                   !mask[prev_idx] ||
                   image[prev_idx] != image[idx]

        is_start || continue

        # Found a starting point - trace the run
        gray_level = image[idx]

        # Validate gray level is in range
        (1 <= gray_level <= Ng) || continue

        run_length = 1
        visited[idx] = true
        current_idx = idx + offset

        # Extend run while conditions are met
        while checkbounds(Bool, mask, current_idx) &&
              mask[current_idx] &&
              image[current_idx] == gray_level

            visited[current_idx] = true
            run_length += 1
            current_idx += offset
        end

        # Clamp run length to Nr (shouldn't exceed, but safety check)
        run_length = min(run_length, Nr)

        # Record the run in the GLRLM matrix
        P[gray_level, run_length] += 1.0
        Ns += 1
        Np += run_length
    end

    return P, Ns, Np
end

"""
    _detect_runs_2d(image::AbstractArray{<:Integer, 2},
                    mask::AbstractArray{Bool, 2},
                    direction::Tuple{Int, Int},
                    Ng::Int, Nr::Int) -> (Matrix{Float64}, Int, Int)

Detect runs along a single direction in a 2D image.

Same algorithm as _detect_runs_3d but for 2D.
"""
function _detect_runs_2d(image::AbstractArray{<:Integer, 2},
                         mask::AbstractArray{Bool, 2},
                         direction::Tuple{Int, Int},
                         Ng::Int, Nr::Int)

    P = zeros(Float64, Ng, Nr)
    Ns = 0
    Np = 0

    sz = size(image)
    offset = CartesianIndex(direction)

    visited = falses(sz)

    @inbounds for idx in CartesianIndices(mask)
        mask[idx] || continue
        visited[idx] && continue

        prev_idx = idx - offset
        is_start = !checkbounds(Bool, mask, prev_idx) ||
                   !mask[prev_idx] ||
                   image[prev_idx] != image[idx]

        is_start || continue

        gray_level = image[idx]
        (1 <= gray_level <= Ng) || continue

        run_length = 1
        visited[idx] = true
        current_idx = idx + offset

        while checkbounds(Bool, mask, current_idx) &&
              mask[current_idx] &&
              image[current_idx] == gray_level

            visited[current_idx] = true
            run_length += 1
            current_idx += offset
        end

        run_length = min(run_length, Nr)

        P[gray_level, run_length] += 1.0
        Ns += 1
        Np += run_length
    end

    return P, Ns, Np
end

#==============================================================================#
# GLRLM Matrix Computation - 3D
#==============================================================================#

"""
    compute_glrlm(image::AbstractArray{<:Integer, 3}, mask::AbstractArray{Bool, 3};
                  Ng::Union{Int, Nothing}=nothing) -> GLRLMResult

Compute Gray Level Run Length Matrices for a 3D image.

The GLRLM P(i,j) counts the number of runs with gray level i and length j
along specific directions within the ROI. A "run" is a sequence of consecutive
voxels with the same gray level intensity.

# Arguments
- `image`: 3D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 3D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels. If nothing, auto-detected.

# Returns
- `GLRLMResult`: Struct containing:
  - `matrices`: GLRLM matrices, shape (Ng, Nr, 13)
  - `Ng`: Number of gray levels
  - `Nr`: Maximum run length
  - `num_directions`: 13
  - `directions`: Direction offsets used
  - `Ns`: Number of runs per direction
  - `Np`: Number of voxels per direction

# Notes
- Input image should be discretized (integer gray levels starting from 1)
- All voxels in a run must be within the mask
- Nr is computed as the maximum possible run length (diagonal of bounding box)
- Matrices are NOT normalized (raw counts). Features normalize internally.

# Example
```julia
# Discretize image first
discretized = discretize_image(image, mask, binwidth=25.0)
result = compute_glrlm(discretized.discretized, mask)

# Access GLRLM for direction 1 (x-axis)
P = result.matrices[:, :, 1]
```

# References
- PyRadiomics: radiomics/glrlm.py
- IBSI: Section 3.6.4 (Grey level run length based features)
"""
function compute_glrlm(image::AbstractArray{<:Integer, 3},
                       mask::AbstractArray{Bool, 3};
                       Ng::Union{Int, Nothing}=nothing)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))

    # Determine number of gray levels
    if isnothing(Ng)
        masked_values = image[mask]
        if isempty(masked_values)
            throw(ArgumentError("Mask is empty, no voxels to process"))
        end
        Ng = maximum(masked_values)
    end

    Ng > 0 || throw(ArgumentError("Ng must be positive, got $Ng"))

    # Compute maximum possible run length
    # This is the maximum diagonal of the bounding box
    sz = size(image)
    Nr = isqrt(sum(s^2 for s in sz)) + 1  # Ceiling of diagonal length

    # Actually, PyRadiomics uses the actual maximum extent in any direction
    # For simplicity, use the max dimension
    Nr = maximum(sz)

    num_directions = length(GLRLM_DIRECTIONS_3D)

    # Initialize storage
    matrices = zeros(Float64, Ng, Nr, num_directions)
    Ns = zeros(Int, num_directions)
    Np = zeros(Int, num_directions)

    # Compute GLRLM for each direction
    for (d, direction) in enumerate(GLRLM_DIRECTIONS_3D)
        P, ns, np = _detect_runs_3d(image, mask, direction, Ng, Nr)
        matrices[:, :, d] .= P
        Ns[d] = ns
        Np[d] = np
    end

    return GLRLMResult(matrices, Ng, Nr, num_directions, GLRLM_DIRECTIONS_3D, Ns, Np)
end

#==============================================================================#
# GLRLM Matrix Computation - 2D
#==============================================================================#

"""
    compute_glrlm_2d(image::AbstractArray{<:Integer, 2}, mask::AbstractArray{Bool, 2};
                     Ng::Union{Int, Nothing}=nothing) -> GLRLMResult2D

Compute Gray Level Run Length Matrices for a 2D image.

# Arguments
- `image`: 2D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 2D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)

# Returns
- `GLRLMResult2D`: Struct containing GLRLM matrices (shape: Ng × Nr × 4)

# Example
```julia
image_2d = discretize(image_slice, edges)
mask_2d = mask[:, :, 32]
result = compute_glrlm_2d(image_2d, mask_2d)
```
"""
function compute_glrlm_2d(image::AbstractArray{<:Integer, 2},
                          mask::AbstractArray{Bool, 2};
                          Ng::Union{Int, Nothing}=nothing)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions"
    ))

    # Determine number of gray levels
    if isnothing(Ng)
        masked_values = image[mask]
        if isempty(masked_values)
            throw(ArgumentError("Mask is empty"))
        end
        Ng = maximum(masked_values)
    end

    sz = size(image)
    Nr = maximum(sz)

    num_directions = length(GLRLM_DIRECTIONS_2D)

    matrices = zeros(Float64, Ng, Nr, num_directions)
    Ns = zeros(Int, num_directions)
    Np = zeros(Int, num_directions)

    for (d, direction) in enumerate(GLRLM_DIRECTIONS_2D)
        P, ns, np = _detect_runs_2d(image, mask, direction, Ng, Nr)
        matrices[:, :, d] .= P
        Ns[d] = ns
        Np[d] = np
    end

    return GLRLMResult2D(matrices, Ng, Nr, Ns, Np)
end

#==============================================================================#
# High-Level GLRLM Computation Interface
#==============================================================================#

"""
    compute_glrlm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                  binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing) -> GLRLMResult

Compute GLRLM from a non-discretized image (convenience wrapper).

This function discretizes the image before computing the GLRLM.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)

# Returns
- For 3D images: `GLRLMResult`
- For 2D images: `GLRLMResult2D`

# Example
```julia
result = compute_glrlm(image, mask, binwidth=25.0)
```
"""
function compute_glrlm(image::AbstractArray{<:Real, 3},
                       mask::AbstractArray{Bool, 3};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing)

    # Discretize the image
    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Compute GLRLM on discretized image
    return compute_glrlm(disc_result.discretized, mask)
end

function compute_glrlm(image::AbstractArray{<:Real, 2},
                       mask::AbstractArray{Bool, 2};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    return compute_glrlm_2d(disc_result.discretized, mask)
end

#==============================================================================#
# GLRLM with Settings
#==============================================================================#

"""
    compute_glrlm(image, mask, settings::Settings)

Compute GLRLM using parameters from a Settings object.

# Example
```julia
settings = Settings(binwidth=32.0)
result = compute_glrlm(image, mask, settings)
```
"""
function compute_glrlm(image::AbstractArray{<:Real},
                       mask::AbstractArray{Bool},
                       settings::Settings)

    bincount = settings.discretization_mode == FixedBinCount ? settings.bincount : nothing

    return compute_glrlm(image, mask;
                         binwidth=settings.binwidth,
                         bincount=bincount)
end

#==============================================================================#
# GLRLM Utility Functions
#==============================================================================#

"""
    glrlm_num_gray_levels(result::GLRLMResult) -> Int

Get the number of gray levels in the GLRLM.
"""
glrlm_num_gray_levels(result::GLRLMResult) = result.Ng
glrlm_num_gray_levels(result::GLRLMResult2D) = result.Ng

"""
    glrlm_num_directions(result::GLRLMResult) -> Int

Get the number of directions used in GLRLM computation.
"""
glrlm_num_directions(result::GLRLMResult) = result.num_directions
glrlm_num_directions(result::GLRLMResult2D) = length(GLRLM_DIRECTIONS_2D)

"""
    glrlm_max_run_length(result::Union{GLRLMResult, GLRLMResult2D}) -> Int

Get the maximum possible run length in the GLRLM.
"""
glrlm_max_run_length(result::GLRLMResult) = result.Nr
glrlm_max_run_length(result::GLRLMResult2D) = result.Nr

"""
    glrlm_num_runs(result::Union{GLRLMResult, GLRLMResult2D}, direction::Int) -> Int

Get the number of runs for a specific direction.
"""
function glrlm_num_runs(result::Union{GLRLMResult, GLRLMResult2D}, direction::Int)
    return result.Ns[direction]
end

"""
    glrlm_num_voxels(result::Union{GLRLMResult, GLRLMResult2D}, direction::Int) -> Int

Get the number of voxels counted in runs for a specific direction.
"""
function glrlm_num_voxels(result::Union{GLRLMResult, GLRLMResult2D}, direction::Int)
    return result.Np[direction]
end

"""
    get_merged_glrlm(result::GLRLMResult) -> Matrix{Float64}

Get merged (summed) GLRLM across all directions.

Alternative aggregation method where matrices are merged before feature computation.
"""
function get_merged_glrlm(result::GLRLMResult)
    return dropdims(sum(result.matrices, dims=3), dims=3)
end

#==============================================================================#
# GLRLM Marginal Distributions
#==============================================================================#

"""
    _glrlm_marginals(P::AbstractMatrix{Float64}) -> NamedTuple

Compute marginal distributions for a single GLRLM matrix.

# Arguments
- `P`: GLRLM matrix (Ng × Nr), raw counts (not normalized)

# Returns
NamedTuple with fields:
- `pg`: Gray level marginal, pg(i) = Σⱼ P(i,j)
- `pr`: Run length marginal, pr(j) = Σᵢ P(i,j)
- `Ns`: Total number of runs, Σᵢ Σⱼ P(i,j)
- `Np`: Total number of voxels, Σᵢ Σⱼ P(i,j)×j
"""
function _glrlm_marginals(P::AbstractMatrix{Float64})
    Ng, Nr = size(P)

    # Gray level marginal: pg(i) = Σⱼ P(i,j)
    pg = vec(sum(P, dims=2))

    # Run length marginal: pr(j) = Σᵢ P(i,j)
    pr = vec(sum(P, dims=1))

    # Total number of runs
    Ns = sum(P)

    # Total number of voxels in runs: Np = Σᵢ Σⱼ P(i,j) × j
    j_vals = collect(1:Nr)
    Np = sum(P[i, j] * j for i in 1:Ng, j in 1:Nr)

    return (pg=pg, pr=pr, Ns=Ns, Np=Np, Ng=Ng, Nr=Nr)
end

#==============================================================================#
# GLRLM Feature Computation - Helper Functions
#==============================================================================#

"""
    _glrlm_feature_single_direction(P::AbstractMatrix{Float64}, Ns::Int, Np::Int,
                                    Ng::Int, Nr::Int, feature_func::Function) -> Float64

Compute a GLRLM feature for a single direction.

# Arguments
- `P`: GLRLM matrix (Ng × Nr) for one direction
- `Ns`: Number of runs in this direction
- `Np`: Number of voxels in runs for this direction
- `Ng`: Number of gray levels
- `Nr`: Maximum run length
- `feature_func`: Function to compute the feature value

# Returns
Feature value, or NaN if Ns = 0
"""
function _glrlm_feature_single_direction(P::AbstractMatrix{Float64}, Ns::Int, Np::Int,
                                         Ng::Int, Nr::Int, feature_func::Function)
    if Ns == 0
        return NaN
    end
    return feature_func(P, Ns, Np, Ng, Nr)
end

"""
    _glrlm_feature_aggregated(result::Union{GLRLMResult, GLRLMResult2D},
                              feature_func::Function) -> Float64

Compute a GLRLM feature aggregated across all directions using nanmean.

This matches PyRadiomics' default aggregation approach: compute the feature
for each direction separately, then average using nanmean.

# Arguments
- `result`: GLRLM computation result
- `feature_func`: Function that takes (P, Ns, Np, Ng, Nr) and returns a Float64

# Returns
Mean feature value across all directions (NaN values excluded)
"""
function _glrlm_feature_aggregated(result::Union{GLRLMResult, GLRLMResult2D},
                                   feature_func::Function)
    Ng = result.Ng
    Nr = result.Nr
    num_dirs = result isa GLRLMResult ? result.num_directions : 4

    values = Float64[]
    for d in 1:num_dirs
        P = @view result.matrices[:, :, d]
        Ns = result.Ns[d]
        Np = result.Np[d]

        val = _glrlm_feature_single_direction(P, Ns, Np, Ng, Nr, feature_func)
        if !isnan(val)
            push!(values, val)
        end
    end

    return isempty(values) ? NaN : mean(values)
end

#==============================================================================#
# GLRLM Features - Run Length Emphasis Features
#==============================================================================#

"""
    glrlm_short_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Short Run Emphasis (SRE): Measures the distribution of short runs.

# Mathematical Formula
```
SRE = (1/Nr) × ΣᵢΣⱼ P(i,j) / j²
```

where:
- P(i,j) is the number of runs of gray level i with length j
- Nr is the total number of runs (Ns)

Higher values indicate more short runs and finer textures.

# References
- PyRadiomics: glrlm.py:getShortRunEmphasisFeatureValue
- IBSI: Grey level run length based features - Short runs emphasis
"""
function glrlm_short_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _sre(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for j in 1:Nr
            j_sq = j * j
            for i in 1:Ng
                total += P[i, j] / j_sq
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _sre)
end

"""
    glrlm_long_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Long Run Emphasis (LRE): Measures the distribution of long runs.

# Mathematical Formula
```
LRE = (1/Nr) × ΣᵢΣⱼ P(i,j) × j²
```

Higher values indicate more long runs and coarser textures.

# References
- PyRadiomics: glrlm.py:getLongRunEmphasisFeatureValue
- IBSI: Grey level run length based features - Long runs emphasis
"""
function glrlm_long_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _lre(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for j in 1:Nr
            j_sq = j * j
            for i in 1:Ng
                total += P[i, j] * j_sq
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _lre)
end

#==============================================================================#
# GLRLM Features - Non-Uniformity Features
#==============================================================================#

"""
    glrlm_gray_level_non_uniformity(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Gray Level Non-Uniformity (GLN): Measures variability of gray level intensity.

# Mathematical Formula
```
GLN = (1/Ns) × Σᵢ (Σⱼ P(i,j))²
```

where Σⱼ P(i,j) is the gray level marginal (pg).

Lower values indicate more homogeneous gray level distribution.

# References
- PyRadiomics: glrlm.py:getGrayLevelNonUniformityFeatureValue
- IBSI: Grey level run length based features - Grey level non-uniformity
"""
function glrlm_gray_level_non_uniformity(result::Union{GLRLMResult, GLRLMResult2D})
    function _gln(P, Ns, Np, Ng, Nr)
        # pg(i) = Σⱼ P(i,j) - gray level marginal
        total = 0.0
        @inbounds for i in 1:Ng
            pg_i = 0.0
            for j in 1:Nr
                pg_i += P[i, j]
            end
            total += pg_i * pg_i
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _gln)
end

"""
    glrlm_gray_level_non_uniformity_normalized(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Gray Level Non-Uniformity Normalized (GLNN): Normalized version of GLN.

# Mathematical Formula
```
GLNN = (1/Ns²) × Σᵢ (Σⱼ P(i,j))²
```

Normalizing by Ns² makes the feature less sensitive to the number of runs.

# References
- PyRadiomics: glrlm.py:getGrayLevelNonUniformityNormalizedFeatureValue
- IBSI: Grey level run length based features - Normalised grey level non-uniformity
"""
function glrlm_gray_level_non_uniformity_normalized(result::Union{GLRLMResult, GLRLMResult2D})
    function _glnn(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            pg_i = 0.0
            for j in 1:Nr
                pg_i += P[i, j]
            end
            total += pg_i * pg_i
        end
        return total / (Ns * Ns)
    end
    return _glrlm_feature_aggregated(result, _glnn)
end

"""
    glrlm_run_length_non_uniformity(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Run Length Non-Uniformity (RLN): Measures variability of run lengths.

# Mathematical Formula
```
RLN = (1/Ns) × Σⱼ (Σᵢ P(i,j))²
```

where Σᵢ P(i,j) is the run length marginal (pr).

Lower values indicate more homogeneous run length distribution.

# References
- PyRadiomics: glrlm.py:getRunLengthNonUniformityFeatureValue
- IBSI: Grey level run length based features - Run length non-uniformity
"""
function glrlm_run_length_non_uniformity(result::Union{GLRLMResult, GLRLMResult2D})
    function _rln(P, Ns, Np, Ng, Nr)
        # pr(j) = Σᵢ P(i,j) - run length marginal
        total = 0.0
        @inbounds for j in 1:Nr
            pr_j = 0.0
            for i in 1:Ng
                pr_j += P[i, j]
            end
            total += pr_j * pr_j
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _rln)
end

"""
    glrlm_run_length_non_uniformity_normalized(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Run Length Non-Uniformity Normalized (RLNN): Normalized version of RLN.

# Mathematical Formula
```
RLNN = (1/Ns²) × Σⱼ (Σᵢ P(i,j))²
```

# References
- PyRadiomics: glrlm.py:getRunLengthNonUniformityNormalizedFeatureValue
- IBSI: Grey level run length based features - Normalised run length non-uniformity
"""
function glrlm_run_length_non_uniformity_normalized(result::Union{GLRLMResult, GLRLMResult2D})
    function _rlnn(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for j in 1:Nr
            pr_j = 0.0
            for i in 1:Ng
                pr_j += P[i, j]
            end
            total += pr_j * pr_j
        end
        return total / (Ns * Ns)
    end
    return _glrlm_feature_aggregated(result, _rlnn)
end

#==============================================================================#
# GLRLM Features - Run Percentage
#==============================================================================#

"""
    glrlm_run_percentage(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Run Percentage (RP): Ratio of total runs to total voxels.

# Mathematical Formula
```
RP = Ns / Np
```

where:
- Ns is the total number of runs
- Np is the total number of voxels in the ROI

Higher values indicate more runs (shorter run lengths on average).
Maximum value is 1.0 when all runs have length 1.

# References
- PyRadiomics: glrlm.py:getRunPercentageFeatureValue
- IBSI: Grey level run length based features - Run percentage
"""
function glrlm_run_percentage(result::Union{GLRLMResult, GLRLMResult2D})
    function _rp(P, Ns, Np, Ng, Nr)
        if Np == 0
            return NaN
        end
        return Float64(Ns) / Float64(Np)
    end
    return _glrlm_feature_aggregated(result, _rp)
end

#==============================================================================#
# GLRLM Features - Variance Features
#==============================================================================#

"""
    glrlm_gray_level_variance(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Gray Level Variance (GLV): Measures variance in gray level intensities.

# Mathematical Formula
```
GLV = ΣᵢΣⱼ p(i,j) × (i - μ)²
where μ = ΣᵢΣⱼ p(i,j) × i
```

where p(i,j) = P(i,j) / Ns is the normalized GLRLM.

# References
- PyRadiomics: glrlm.py:getGrayLevelVarianceFeatureValue
- IBSI: Grey level run length based features - Grey level variance
"""
function glrlm_gray_level_variance(result::Union{GLRLMResult, GLRLMResult2D})
    function _glv(P, Ns, Np, Ng, Nr)
        # Compute mean gray level: μ = Σᵢ Σⱼ p(i,j) × i
        mu = 0.0
        @inbounds for i in 1:Ng
            for j in 1:Nr
                mu += (P[i, j] / Ns) * i
            end
        end

        # Compute variance: Σᵢ Σⱼ p(i,j) × (i - μ)²
        variance = 0.0
        @inbounds for i in 1:Ng
            diff_sq = (i - mu)^2
            for j in 1:Nr
                variance += (P[i, j] / Ns) * diff_sq
            end
        end

        return variance
    end
    return _glrlm_feature_aggregated(result, _glv)
end

"""
    glrlm_run_variance(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Run Variance (RV): Measures variance in run lengths.

# Mathematical Formula
```
RV = ΣᵢΣⱼ p(i,j) × (j - μ)²
where μ = ΣᵢΣⱼ p(i,j) × j
```

where p(i,j) = P(i,j) / Ns is the normalized GLRLM.

# References
- PyRadiomics: glrlm.py:getRunVarianceFeatureValue
- IBSI: Grey level run length based features - Run length variance
"""
function glrlm_run_variance(result::Union{GLRLMResult, GLRLMResult2D})
    function _rv(P, Ns, Np, Ng, Nr)
        # Compute mean run length: μ = Σᵢ Σⱼ p(i,j) × j
        mu = 0.0
        @inbounds for i in 1:Ng
            for j in 1:Nr
                mu += (P[i, j] / Ns) * j
            end
        end

        # Compute variance: Σᵢ Σⱼ p(i,j) × (j - μ)²
        variance = 0.0
        @inbounds for i in 1:Ng
            for j in 1:Nr
                variance += (P[i, j] / Ns) * (j - mu)^2
            end
        end

        return variance
    end
    return _glrlm_feature_aggregated(result, _rv)
end

#==============================================================================#
# GLRLM Features - Entropy
#==============================================================================#

"""
    glrlm_run_entropy(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Run Entropy (RE): Measures uncertainty/randomness in run distribution.

# Mathematical Formula
```
RE = -ΣᵢΣⱼ p(i,j) × log₂(p(i,j) + ε)
```

where p(i,j) = P(i,j) / Ns and ε ≈ 2.2×10⁻¹⁶ prevents log(0).

Higher values indicate more heterogeneous run distribution.

# References
- PyRadiomics: glrlm.py:getRunEntropyFeatureValue
- IBSI: Grey level run length based features - Run entropy
"""
function glrlm_run_entropy(result::Union{GLRLMResult, GLRLMResult2D})
    function _re(P, Ns, Np, Ng, Nr)
        entropy = 0.0
        @inbounds for i in 1:Ng
            for j in 1:Nr
                p_ij = P[i, j] / Ns
                if p_ij > 0
                    entropy -= p_ij * log2(p_ij + GLRLM_EPSILON)
                end
            end
        end
        return entropy
    end
    return _glrlm_feature_aggregated(result, _re)
end

#==============================================================================#
# GLRLM Features - Gray Level Emphasis Features
#==============================================================================#

"""
    glrlm_low_gray_level_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Low Gray Level Run Emphasis (LGLRE): Measures distribution of low gray levels.

# Mathematical Formula
```
LGLRE = (1/Ns) × ΣᵢΣⱼ P(i,j) / i²
```

Higher values indicate more runs with low gray levels.

# References
- PyRadiomics: glrlm.py:getLowGrayLevelRunEmphasisFeatureValue
- IBSI: Grey level run length based features - Low grey level run emphasis
"""
function glrlm_low_gray_level_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _lglre(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] / i_sq
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _lglre)
end

"""
    glrlm_high_gray_level_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

High Gray Level Run Emphasis (HGLRE): Measures distribution of high gray levels.

# Mathematical Formula
```
HGLRE = (1/Ns) × ΣᵢΣⱼ P(i,j) × i²
```

Higher values indicate more runs with high gray levels.

# References
- PyRadiomics: glrlm.py:getHighGrayLevelRunEmphasisFeatureValue
- IBSI: Grey level run length based features - High grey level run emphasis
"""
function glrlm_high_gray_level_run_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _hglre(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] * i_sq
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _hglre)
end

#==============================================================================#
# GLRLM Features - Combined Emphasis Features
#==============================================================================#

"""
    glrlm_short_run_low_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Short Run Low Gray Level Emphasis (SRLGLE): Joint distribution of short runs and low gray levels.

# Mathematical Formula
```
SRLGLE = (1/Ns) × ΣᵢΣⱼ P(i,j) / (i² × j²)
```

Higher values indicate more short runs with low gray levels.

# References
- PyRadiomics: glrlm.py:getShortRunLowGrayLevelEmphasisFeatureValue
- IBSI: Grey level run length based features - Short run low grey level emphasis
"""
function glrlm_short_run_low_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _srlgle(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] / (i_sq * j * j)
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _srlgle)
end

"""
    glrlm_short_run_high_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Short Run High Gray Level Emphasis (SRHGLE): Joint distribution of short runs and high gray levels.

# Mathematical Formula
```
SRHGLE = (1/Ns) × ΣᵢΣⱼ P(i,j) × i² / j²
```

Higher values indicate more short runs with high gray levels.

# References
- PyRadiomics: glrlm.py:getShortRunHighGrayLevelEmphasisFeatureValue
- IBSI: Grey level run length based features - Short run high grey level emphasis
"""
function glrlm_short_run_high_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _srhgle(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] * i_sq / (j * j)
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _srhgle)
end

"""
    glrlm_long_run_low_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Long Run Low Gray Level Emphasis (LRLGLE): Joint distribution of long runs and low gray levels.

# Mathematical Formula
```
LRLGLE = (1/Ns) × ΣᵢΣⱼ P(i,j) × j² / i²
```

Higher values indicate more long runs with low gray levels.

# References
- PyRadiomics: glrlm.py:getLongRunLowGrayLevelEmphasisFeatureValue
- IBSI: Grey level run length based features - Long run low grey level emphasis
"""
function glrlm_long_run_low_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _lrlgle(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] * (j * j) / i_sq
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _lrlgle)
end

"""
    glrlm_long_run_high_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D}) -> Float64

Long Run High Gray Level Emphasis (LRHGLE): Joint distribution of long runs and high gray levels.

# Mathematical Formula
```
LRHGLE = (1/Ns) × ΣᵢΣⱼ P(i,j) × i² × j²
```

Higher values indicate more long runs with high gray levels.

# References
- PyRadiomics: glrlm.py:getLongRunHighGrayLevelEmphasisFeatureValue
- IBSI: Grey level run length based features - Long run high grey level emphasis
"""
function glrlm_long_run_high_gray_level_emphasis(result::Union{GLRLMResult, GLRLMResult2D})
    function _lrhgle(P, Ns, Np, Ng, Nr)
        total = 0.0
        @inbounds for i in 1:Ng
            i_sq = i * i
            for j in 1:Nr
                total += P[i, j] * i_sq * (j * j)
            end
        end
        return total / Ns
    end
    return _glrlm_feature_aggregated(result, _lrhgle)
end

#==============================================================================#
# Exports
#==============================================================================#

# Export GLRLM computation functions
export compute_glrlm, compute_glrlm_2d
export GLRLMResult, GLRLMResult2D

# Export utility functions
export glrlm_num_gray_levels, glrlm_num_directions, glrlm_max_run_length
export glrlm_num_runs, glrlm_num_voxels, get_merged_glrlm

# Export direction constants
export GLRLM_DIRECTIONS_3D, GLRLM_DIRECTIONS_2D

# Export GLRLM feature functions
export glrlm_short_run_emphasis, glrlm_long_run_emphasis
export glrlm_gray_level_non_uniformity, glrlm_gray_level_non_uniformity_normalized
export glrlm_run_length_non_uniformity, glrlm_run_length_non_uniformity_normalized
export glrlm_run_percentage
export glrlm_gray_level_variance, glrlm_run_variance
export glrlm_run_entropy
export glrlm_low_gray_level_run_emphasis, glrlm_high_gray_level_run_emphasis
export glrlm_short_run_low_gray_level_emphasis, glrlm_short_run_high_gray_level_emphasis
export glrlm_long_run_low_gray_level_emphasis, glrlm_long_run_high_gray_level_emphasis
