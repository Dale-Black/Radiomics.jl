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
