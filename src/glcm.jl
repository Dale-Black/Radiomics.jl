# Gray Level Co-occurrence Matrix (GLCM) for Radiomics.jl
#
# This module implements GLCM computation and features.
# GLCM captures texture information by examining spatial relationships
# between pixels with specific gray level intensities.
#
# Total: 24 features (23 IBSI-compliant + 1 deprecated but included for parity)
#
# References:
# - PyRadiomics: radiomics/glcm.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - Haralick, R.M. et al. (1973). "Textural Features for Image Classification"

using LinearAlgebra

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding log(0) - matches np.spacing(1)
const GLCM_EPSILON = eps(Float64)  # ≈ 2.2e-16

"""
    GLCM_DIRECTIONS_3D

The 13 unique directions in 3D for GLCM computation.

These directions define the offset vectors for finding pixel pairs.
Only 13 directions are needed because the opposite directions would
produce symmetric contributions (counted via the symmetric option).

Each tuple (dz, dy, dx) represents a direction offset where:
- dz: offset in the third dimension (z-axis, or slice direction)
- dy: offset in the second dimension (y-axis, or row direction)
- dx: offset in the first dimension (x-axis, or column direction)

Note: PyRadiomics uses (z, y, x) ordering for arrays (C-order).
Julia uses column-major ordering, but we maintain the same offset
convention for compatibility.

Direction descriptions (for distance=1):
1. (0, 0, 1) - along x-axis
2. (0, 1, 0) - along y-axis
3. (1, 0, 0) - along z-axis
4. (0, 1, 1) - xy diagonal
5. (0, 1, -1) - xy anti-diagonal
6. (1, 0, 1) - xz diagonal
7. (1, 0, -1) - xz anti-diagonal
8. (1, 1, 0) - yz diagonal
9. (1, -1, 0) - yz anti-diagonal
10. (1, 1, 1) - body diagonal
11. (1, 1, -1) - body diagonal variant
12. (1, -1, 1) - body diagonal variant
13. (1, -1, -1) - body diagonal variant
"""
const GLCM_DIRECTIONS_3D = (
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
    GLCM_DIRECTIONS_2D

The 4 unique directions in 2D for GLCM computation.

Direction descriptions (for distance=1):
1. (0, 1) - along x-axis (0 degrees)
2. (1, 1) - diagonal (45 degrees)
3. (1, 0) - along y-axis (90 degrees)
4. (1, -1) - anti-diagonal (135 degrees)
"""
const GLCM_DIRECTIONS_2D = (
    (0, 1),   # 1: X-axis (0 degrees)
    (1, 1),   # 2: Diagonal (45 degrees)
    (1, 0),   # 3: Y-axis (90 degrees)
    (1, -1),  # 4: Anti-diagonal (135 degrees)
)

#==============================================================================#
# GLCM Result Type
#==============================================================================#

"""
    GLCMResult

Container for GLCM computation results.

# Fields
- `matrices::Array{Float64, 3}`: GLCM matrices, shape (Ng, Ng, num_directions)
- `Ng::Int`: Number of gray levels
- `num_directions::Int`: Number of directions computed
- `distance::Int`: Distance parameter used
- `symmetric::Bool`: Whether matrices are symmetric
- `directions`: Direction offsets used
- `counts::Vector{Int}`: Number of valid pairs per direction (before normalization)

# Notes
The matrices are stored as a 3D array where:
- First dimension: reference gray level (i)
- Second dimension: neighbor gray level (j)
- Third dimension: direction index

Each 2D slice `matrices[:, :, d]` is a normalized probability matrix
where P[i, j] is the probability of finding gray level j at distance d
from gray level i in direction d.
"""
struct GLCMResult
    matrices::Array{Float64, 3}
    Ng::Int
    num_directions::Int
    distance::Int
    symmetric::Bool
    directions::Any
    counts::Vector{Int}
end

"""
    GLCMResult2D

Container for 2D GLCM computation results.

# Fields
- `matrices::Array{Float64, 3}`: GLCM matrices, shape (Ng, Ng, 4)
- `Ng::Int`: Number of gray levels
- `distance::Int`: Distance parameter used
- `symmetric::Bool`: Whether matrices are symmetric
- `counts::Vector{Int}`: Number of valid pairs per direction
"""
struct GLCMResult2D
    matrices::Array{Float64, 3}
    Ng::Int
    distance::Int
    symmetric::Bool
    counts::Vector{Int}
end

#==============================================================================#
# GLCM Matrix Computation - 3D
#==============================================================================#

"""
    compute_glcm(image::AbstractArray{<:Integer, 3}, mask::AbstractArray{Bool, 3};
                 distance::Int=1, symmetric::Bool=true, Ng::Union{Int, Nothing}=nothing) -> GLCMResult

Compute Gray Level Co-occurrence Matrices for a 3D image.

The GLCM P(i,j) counts how often voxels with gray level i and gray level j
occur at a specified distance and direction from each other within the ROI.

# Arguments
- `image`: 3D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 3D boolean mask defining the ROI
- `distance::Int=1`: Distance between pixel pairs (PyRadiomics default: 1)
- `symmetric::Bool=true`: If true, P[i,j] and P[j,i] are both incremented (default in PyRadiomics)
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels. If nothing, auto-detected from image maximum.

# Returns
- `GLCMResult`: Struct containing:
  - `matrices`: Normalized GLCM matrices, shape (Ng, Ng, 13)
  - `Ng`: Number of gray levels
  - `distance`: Distance used
  - `symmetric`: Whether symmetric was applied
  - `directions`: Direction offsets used
  - `counts`: Raw pair counts per direction (before normalization)

# Notes
- Input image should be discretized (integer gray levels)
- Both voxels in a pair must be within the mask to be counted
- Each direction's matrix is independently normalized to sum to 1
- Directions with no valid pairs result in a zero matrix

# Example
```julia
# Discretize image first
discretized = discretize_image(image, mask, binwidth=25.0)
result = compute_glcm(discretized.discretized, mask)

# Access normalized GLCM for direction 1
P = result.matrices[:, :, 1]
```

# References
- PyRadiomics: radiomics/_cmatrices.c:calculate_glcm
- IBSI: Section 3.6.2 (Grey level co-occurrence based features)
"""
function compute_glcm(image::AbstractArray{<:Integer, 3},
                      mask::AbstractArray{Bool, 3};
                      distance::Int=1,
                      symmetric::Bool=true,
                      Ng::Union{Int, Nothing}=nothing)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive, got $distance"))

    # Determine number of gray levels
    if isnothing(Ng)
        # Get max from masked region only
        masked_values = image[mask]
        if isempty(masked_values)
            throw(ArgumentError("Mask is empty, no voxels to process"))
        end
        Ng = maximum(masked_values)
    end

    Ng > 0 || throw(ArgumentError("Ng must be positive, got $Ng"))

    num_directions = length(GLCM_DIRECTIONS_3D)

    # Initialize GLCM matrices (use Float64 for eventual normalization)
    matrices = zeros(Float64, Ng, Ng, num_directions)
    counts = zeros(Int, num_directions)

    # Image dimensions
    sz = size(image)

    # Compute GLCM for each direction
    for (d, direction) in enumerate(GLCM_DIRECTIONS_3D)
        offset = CartesianIndex(direction .* distance)

        # Count co-occurrences
        pair_count = 0

        @inbounds for idx in CartesianIndices(mask)
            # Skip if reference voxel not in mask
            mask[idx] || continue

            # Compute neighbor index
            neighbor_idx = idx + offset

            # Check bounds and mask for neighbor
            if checkbounds(Bool, mask, neighbor_idx) && mask[neighbor_idx]
                i = image[idx]
                j = image[neighbor_idx]

                # Only count valid gray levels
                if 1 <= i <= Ng && 1 <= j <= Ng
                    matrices[i, j, d] += 1.0
                    pair_count += 1
                end
            end
        end

        counts[d] = pair_count
    end

    # Make symmetric if requested (P[i,j] = P[i,j] + P[j,i])
    if symmetric
        for d in 1:num_directions
            @views matrices[:, :, d] .+= transpose(matrices[:, :, d])
        end
    end

    # Normalize each direction's matrix
    for d in 1:num_directions
        s = sum(@view matrices[:, :, d])
        if s > 0
            @views matrices[:, :, d] ./= s
        end
    end

    return GLCMResult(matrices, Ng, num_directions, distance, symmetric,
                      GLCM_DIRECTIONS_3D, counts)
end

#==============================================================================#
# GLCM Matrix Computation - 2D
#==============================================================================#

"""
    compute_glcm_2d(image::AbstractArray{<:Integer, 2}, mask::AbstractArray{Bool, 2};
                    distance::Int=1, symmetric::Bool=true, Ng::Union{Int, Nothing}=nothing) -> GLCMResult2D

Compute Gray Level Co-occurrence Matrices for a 2D image.

# Arguments
- `image`: 2D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 2D boolean mask defining the ROI
- `distance::Int=1`: Distance between pixel pairs
- `symmetric::Bool=true`: If true, count both directions
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)

# Returns
- `GLCMResult2D`: Struct containing normalized GLCM matrices (shape: Ng x Ng x 4)

# Example
```julia
image_2d = discretize(image_slice, edges)
mask_2d = mask[:, :, 32]
result = compute_glcm_2d(image_2d, mask_2d)
```
"""
function compute_glcm_2d(image::AbstractArray{<:Integer, 2},
                         mask::AbstractArray{Bool, 2};
                         distance::Int=1,
                         symmetric::Bool=true,
                         Ng::Union{Int, Nothing}=nothing)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive"))

    # Determine number of gray levels
    if isnothing(Ng)
        masked_values = image[mask]
        if isempty(masked_values)
            throw(ArgumentError("Mask is empty"))
        end
        Ng = maximum(masked_values)
    end

    num_directions = length(GLCM_DIRECTIONS_2D)
    matrices = zeros(Float64, Ng, Ng, num_directions)
    counts = zeros(Int, num_directions)

    # Compute GLCM for each direction
    for (d, direction) in enumerate(GLCM_DIRECTIONS_2D)
        offset = CartesianIndex(direction .* distance)
        pair_count = 0

        @inbounds for idx in CartesianIndices(mask)
            mask[idx] || continue

            neighbor_idx = idx + offset

            if checkbounds(Bool, mask, neighbor_idx) && mask[neighbor_idx]
                i = image[idx]
                j = image[neighbor_idx]

                if 1 <= i <= Ng && 1 <= j <= Ng
                    matrices[i, j, d] += 1.0
                    pair_count += 1
                end
            end
        end

        counts[d] = pair_count
    end

    # Make symmetric if requested
    if symmetric
        for d in 1:num_directions
            @views matrices[:, :, d] .+= transpose(matrices[:, :, d])
        end
    end

    # Normalize each direction's matrix
    for d in 1:num_directions
        s = sum(@view matrices[:, :, d])
        if s > 0
            @views matrices[:, :, d] ./= s
        end
    end

    return GLCMResult2D(matrices, Ng, distance, symmetric, counts)
end

#==============================================================================#
# High-Level GLCM Computation Interface
#==============================================================================#

"""
    compute_glcm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                 binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                 distance::Int=1, symmetric::Bool=true) -> GLCMResult

Compute GLCM from a non-discretized image (convenience wrapper).

This function discretizes the image before computing the GLCM.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `distance::Int=1`: GLCM distance
- `symmetric::Bool=true`: Whether to compute symmetric GLCM

# Returns
- For 3D images: `GLCMResult`
- For 2D images: `GLCMResult2D`

# Example
```julia
result = compute_glcm(image, mask, binwidth=25.0, distance=1)
```
"""
function compute_glcm(image::AbstractArray{<:Real, 3},
                      mask::AbstractArray{Bool, 3};
                      binwidth::Real=25.0,
                      bincount::Union{Int, Nothing}=nothing,
                      distance::Int=1,
                      symmetric::Bool=true)

    # Discretize the image
    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Compute GLCM on discretized image
    return compute_glcm(disc_result.discretized, mask;
                        distance=distance, symmetric=symmetric,
                        Ng=disc_result.nbins)
end

function compute_glcm(image::AbstractArray{<:Real, 2},
                      mask::AbstractArray{Bool, 2};
                      binwidth::Real=25.0,
                      bincount::Union{Int, Nothing}=nothing,
                      distance::Int=1,
                      symmetric::Bool=true)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    return compute_glcm_2d(disc_result.discretized, mask;
                           distance=distance, symmetric=symmetric,
                           Ng=disc_result.nbins)
end

#==============================================================================#
# GLCM with Settings
#==============================================================================#

"""
    compute_glcm(image, mask, settings::Settings)

Compute GLCM using parameters from a Settings object.

# Example
```julia
settings = Settings(binwidth=32.0, glcm_distance=2, symmetrical_glcm=true)
result = compute_glcm(image, mask, settings)
```
"""
function compute_glcm(image::AbstractArray{<:Real},
                      mask::AbstractArray{Bool},
                      settings::Settings)

    bincount = settings.discretization_mode == FixedBinCount ? settings.bincount : nothing

    return compute_glcm(image, mask;
                        binwidth=settings.binwidth,
                        bincount=bincount,
                        distance=settings.glcm_distance,
                        symmetric=settings.symmetrical_glcm)
end

#==============================================================================#
# Auxiliary Distributions for GLCM Features
#==============================================================================#

"""
    _glcm_marginals(P::AbstractMatrix{Float64}) -> NamedTuple

Compute marginal distributions and moments for a single GLCM matrix.

# Arguments
- `P`: Normalized GLCM matrix (Ng x Ng)

# Returns
NamedTuple with fields:
- `px`: Row marginal probabilities, p_x(i) = Σⱼ P(i,j)
- `py`: Column marginal probabilities, p_y(j) = Σᵢ P(i,j)
- `ux`: Mean of x (μₓ = Σᵢ i·p_x(i))
- `uy`: Mean of y (μᵧ = Σⱼ j·p_y(j))
- `sigx`: Std dev of x
- `sigy`: Std dev of y
- `Ng`: Number of gray levels

# Notes
For symmetric GLCM, px == py and ux == uy.
"""
function _glcm_marginals(P::AbstractMatrix{Float64})
    Ng = size(P, 1)

    # Marginal probabilities
    px = vec(sum(P, dims=2))  # Row marginal: p_x(i) = Σⱼ P(i,j)
    py = vec(sum(P, dims=1))  # Column marginal: p_y(j) = Σᵢ P(i,j)

    # Gray level indices (1:Ng)
    i_vals = collect(1:Ng)

    # Means
    ux = sum(i_vals .* px)  # μₓ = Σᵢ i·p_x(i)
    uy = sum(i_vals .* py)  # μᵧ = Σⱼ j·p_y(j)

    # Standard deviations
    varx = sum(((i_vals .- ux).^2) .* px)
    vary = sum(((i_vals .- uy).^2) .* py)
    sigx = sqrt(max(varx, 0.0))
    sigy = sqrt(max(vary, 0.0))

    return (px=px, py=py, ux=ux, uy=uy, sigx=sigx, sigy=sigy, Ng=Ng)
end

"""
    _glcm_sum_diff_distributions(P::AbstractMatrix{Float64}) -> NamedTuple

Compute sum (p_x+y) and difference (p_x-y) distributions for GLCM.

# Arguments
- `P`: Normalized GLCM matrix (Ng x Ng)

# Returns
NamedTuple with fields:
- `pxplusy`: Sum distribution p_{x+y}(k), k = 2 to 2Ng, indexed as [k-1]
- `pxminusy`: Difference distribution p_{x-y}(k), k = 0 to Ng-1, indexed as [k+1]

# Notes
- p_{x+y}(k) = Σ_{i+j=k} P(i,j) for k ∈ [2, 2Ng]
- p_{x-y}(k) = Σ_{|i-j|=k} P(i,j) for k ∈ [0, Ng-1]
"""
function _glcm_sum_diff_distributions(P::AbstractMatrix{Float64})
    Ng = size(P, 1)

    # Sum distribution: p_{x+y}(k) for k = 2 to 2*Ng
    # Array length = 2*Ng - 1 (indices from k=2 to k=2*Ng)
    pxplusy = zeros(Float64, 2*Ng - 1)

    # Difference distribution: p_{x-y}(k) for k = 0 to Ng-1
    # Array length = Ng (indices from k=0 to k=Ng-1)
    pxminusy = zeros(Float64, Ng)

    @inbounds for i in 1:Ng, j in 1:Ng
        pij = P[i, j]

        # Sum distribution: k = i + j (ranges from 2 to 2*Ng)
        # Store at index k - 1 (so k=2 is at index 1)
        k_sum = i + j
        pxplusy[k_sum - 1] += pij

        # Difference distribution: k = |i - j| (ranges from 0 to Ng-1)
        # Store at index k + 1 (so k=0 is at index 1)
        k_diff = abs(i - j)
        pxminusy[k_diff + 1] += pij
    end

    return (pxplusy=pxplusy, pxminusy=pxminusy)
end

#==============================================================================#
# GLCM Utility Functions
#==============================================================================#

"""
    get_averaged_glcm(result::GLCMResult) -> Matrix{Float64}

Get the averaged GLCM across all directions.

This is NOT the standard way to compute GLCM features in PyRadiomics.
PyRadiomics computes features per direction and then averages the features.

This function is provided for visualization and debugging purposes.
"""
function get_averaged_glcm(result::GLCMResult)
    return dropdims(mean(result.matrices, dims=3), dims=3)
end

"""
    get_merged_glcm(result::GLCMResult) -> Matrix{Float64}

Get merged (summed and normalized) GLCM across all directions.

Alternative aggregation method where matrices are merged before feature computation.
"""
function get_merged_glcm(result::GLCMResult)
    merged = dropdims(sum(result.matrices, dims=3), dims=3)
    s = sum(merged)
    return s > 0 ? merged ./ s : merged
end

"""
    glcm_num_gray_levels(result::GLCMResult) -> Int

Get the number of gray levels in the GLCM.
"""
glcm_num_gray_levels(result::GLCMResult) = result.Ng
glcm_num_gray_levels(result::GLCMResult2D) = result.Ng

"""
    glcm_num_directions(result::GLCMResult) -> Int

Get the number of directions used in GLCM computation.
"""
glcm_num_directions(result::GLCMResult) = result.num_directions
glcm_num_directions(result::GLCMResult2D) = length(GLCM_DIRECTIONS_2D)

#==============================================================================#
# Exports
#==============================================================================#

# Export GLCM computation functions
export compute_glcm, compute_glcm_2d
export GLCMResult, GLCMResult2D

# Export utility functions
export get_averaged_glcm, get_merged_glcm
export glcm_num_gray_levels, glcm_num_directions

# Export direction constants
export GLCM_DIRECTIONS_3D, GLCM_DIRECTIONS_2D
