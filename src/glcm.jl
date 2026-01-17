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
    # Let Ng be auto-detected as max(gray levels) to match PyRadiomics behavior.
    # PyRadiomics uses Ng = max(grayLevels), not nbins.
    return compute_glcm(disc_result.discretized, mask;
                        distance=distance, symmetric=symmetric)
end

function compute_glcm(image::AbstractArray{<:Real, 2},
                      mask::AbstractArray{Bool, 2};
                      binwidth::Real=25.0,
                      bincount::Union{Int, Nothing}=nothing,
                      distance::Int=1,
                      symmetric::Bool=true)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Let Ng be auto-detected as max(gray levels) to match PyRadiomics behavior.
    return compute_glcm_2d(disc_result.discretized, mask;
                           distance=distance, symmetric=symmetric)
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
# GLCM Feature Computation - Per Direction
#==============================================================================#

"""
    _glcm_features_single(P::AbstractMatrix{Float64}, Ng::Int) -> NamedTuple

Compute all GLCM features for a single direction's GLCM matrix.

This is an internal function that computes all 24 features from a single
normalized GLCM matrix. The main feature functions aggregate results
across directions using nanmean.

# Arguments
- `P`: Normalized GLCM matrix (Ng × Ng), must sum to 1
- `Ng`: Number of gray levels

# Returns
NamedTuple containing all 24 GLCM features for this direction.
"""
function _glcm_features_single(P::AbstractMatrix{Float64}, Ng::Int)
    # Get marginal distributions and moments
    marg = _glcm_marginals(P)
    px, py = marg.px, marg.py
    ux, uy = marg.ux, marg.uy
    sigx, sigy = marg.sigx, marg.sigy

    # Get sum/difference distributions
    sumdiff = _glcm_sum_diff_distributions(P)
    pxplusy = sumdiff.pxplusy  # Indexed: pxplusy[k-1] for k = 2 to 2Ng
    pxminusy = sumdiff.pxminusy  # Indexed: pxminusy[k+1] for k = 0 to Ng-1

    # Gray level indices
    i_vals = collect(1:Ng)
    j_vals = collect(1:Ng)

    # ==================== First-Order Statistics ====================

    # 1. Autocorrelation: Σᵢ Σⱼ P(i,j)·i·j
    autocorrelation = sum(P[i, j] * i * j for i in 1:Ng, j in 1:Ng)

    # 2. Joint Average: μₓ = Σᵢ Σⱼ P(i,j)·i
    joint_average = ux

    # 3. Joint Variance (Sum of Squares): Σᵢ Σⱼ (i-μₓ)²·P(i,j)
    sum_squares = sum(P[i, j] * (i - ux)^2 for i in 1:Ng, j in 1:Ng)

    # 4. Joint Entropy: -Σᵢ Σⱼ P(i,j)·log₂(P(i,j)+ε)
    joint_entropy = -sum(P[i, j] * log2(P[i, j] + GLCM_EPSILON) for i in 1:Ng, j in 1:Ng)

    # 5. Joint Energy (Angular Second Moment): Σᵢ Σⱼ P(i,j)²
    joint_energy = sum(abs2, P)

    # 6. Maximum Probability: max{P(i,j)}
    maximum_probability = maximum(P)

    # ==================== Contrast/Variation ====================

    # 7. Contrast: Σᵢ Σⱼ (i-j)²·P(i,j)
    contrast = sum(P[i, j] * (i - j)^2 for i in 1:Ng, j in 1:Ng)

    # ==================== Cluster Features ====================

    # 8. Cluster Prominence: Σᵢ Σⱼ (i+j-μₓ-μᵧ)⁴·P(i,j)
    cluster_prominence = sum(P[i, j] * (i + j - ux - uy)^4 for i in 1:Ng, j in 1:Ng)

    # 9. Cluster Shade: Σᵢ Σⱼ (i+j-μₓ-μᵧ)³·P(i,j)
    cluster_shade = sum(P[i, j] * (i + j - ux - uy)^3 for i in 1:Ng, j in 1:Ng)

    # 10. Cluster Tendency: Σᵢ Σⱼ (i+j-μₓ-μᵧ)²·P(i,j)
    cluster_tendency = sum(P[i, j] * (i + j - ux - uy)^2 for i in 1:Ng, j in 1:Ng)

    # ==================== Difference Features ====================

    # 11. Difference Average: Σₖ k·pₓ₋ᵧ(k), k=0 to Ng-1
    # pxminusy is indexed [k+1] for k = 0 to Ng-1
    diff_average = sum((k - 1) * pxminusy[k] for k in 1:Ng)

    # 12. Difference Variance: Σₖ (k-DA)²·pₓ₋ᵧ(k)
    diff_variance = sum((k - 1 - diff_average)^2 * pxminusy[k] for k in 1:Ng)

    # 13. Difference Entropy: -Σₖ pₓ₋ᵧ(k)·log₂(pₓ₋ᵧ(k)+ε)
    diff_entropy = -sum(pxminusy[k] * log2(pxminusy[k] + GLCM_EPSILON) for k in 1:Ng)

    # ==================== Sum Features ====================

    # 14. Sum Average: Σₖ k·pₓ₊ᵧ(k), k=2 to 2Ng
    # pxplusy is indexed [k-1] for k = 2 to 2Ng
    sum_average = sum(k * pxplusy[k - 1] for k in 2:(2*Ng))

    # 15. Sum Entropy: -Σₖ pₓ₊ᵧ(k)·log₂(pₓ₊ᵧ(k)+ε), k=2 to 2Ng
    sum_entropy = -sum(pxplusy[k - 1] * log2(pxplusy[k - 1] + GLCM_EPSILON) for k in 2:(2*Ng))

    # ==================== Homogeneity Features ====================

    # 16. Inverse Difference (ID): Σₖ pₓ₋ᵧ(k)/(1+k), k=0 to Ng-1
    inverse_diff = sum(pxminusy[k] / (1 + (k - 1)) for k in 1:Ng)

    # 17. Inverse Difference Normalized (IDN): Σₖ pₓ₋ᵧ(k)/(1+k/Ng)
    inverse_diff_normalized = sum(pxminusy[k] / (1 + (k - 1) / Ng) for k in 1:Ng)

    # 18. Inverse Difference Moment (IDM): Σₖ pₓ₋ᵧ(k)/(1+k²)
    idm = sum(pxminusy[k] / (1 + (k - 1)^2) for k in 1:Ng)

    # 19. Inverse Difference Moment Normalized (IDMN): Σₖ pₓ₋ᵧ(k)/(1+k²/Ng²)
    idmn = sum(pxminusy[k] / (1 + (k - 1)^2 / Ng^2) for k in 1:Ng)

    # 20. Inverse Variance: Σₖ pₓ₋ᵧ(k)/k², k=1 to Ng-1 (skip k=0)
    # pxminusy[k+1] for k=1 to Ng-1 → indices 2 to Ng
    inverse_variance = Ng > 1 ? sum(pxminusy[k + 1] / k^2 for k in 1:(Ng-1)) : 0.0

    # ==================== Correlation Features ====================

    # 21. Correlation: (Σᵢ Σⱼ P(i,j)·i·j - μₓ·μᵧ)/(σₓ·σᵧ)
    if sigx > 0 && sigy > 0
        correlation = (autocorrelation - ux * uy) / (sigx * sigy)
    else
        correlation = 1.0  # Flat region
    end

    # Marginal entropies for IMC features
    # HX = -Σᵢ pₓ(i)·log₂(pₓ(i)+ε)
    HX = -sum(px[i] * log2(px[i] + GLCM_EPSILON) for i in 1:Ng)
    # HY = -Σⱼ pᵧ(j)·log₂(pᵧ(j)+ε)
    HY = -sum(py[j] * log2(py[j] + GLCM_EPSILON) for j in 1:Ng)

    # HXY1 = -Σᵢ Σⱼ P(i,j)·log₂(pₓ(i)·pᵧ(j)+ε)
    HXY1 = -sum(P[i, j] * log2(px[i] * py[j] + GLCM_EPSILON) for i in 1:Ng, j in 1:Ng)

    # HXY2 = -Σᵢ Σⱼ pₓ(i)·pᵧ(j)·log₂(pₓ(i)·pᵧ(j)+ε)
    HXY2 = -sum(px[i] * py[j] * log2(px[i] * py[j] + GLCM_EPSILON) for i in 1:Ng, j in 1:Ng)

    # 22. IMC1: (HXY - HXY1)/max(HX, HY)
    max_HXY = max(HX, HY)
    if max_HXY > 0
        imc1 = (joint_entropy - HXY1) / max_HXY
    else
        imc1 = 0.0  # Flat region
    end

    # 23. IMC2: √(1 - exp(-2·(HXY2 - HXY)))
    diff_HXY = HXY2 - joint_entropy
    if diff_HXY >= 0
        imc2 = sqrt(1 - exp(-2 * diff_HXY))
    else
        imc2 = 0.0  # Return 0 when HXY > HXY2 (floating-point precision issue)
    end

    # 24. MCC: √(second largest eigenvalue of Q)
    # Q(i,k) = Σⱼ P(i,j)·P(k,j) / (pₓ(i)·pᵧ(j))
    mcc = _compute_mcc(P, px, py, Ng)

    return (
        autocorrelation = autocorrelation,
        joint_average = joint_average,
        cluster_prominence = cluster_prominence,
        cluster_shade = cluster_shade,
        cluster_tendency = cluster_tendency,
        contrast = contrast,
        correlation = correlation,
        difference_average = diff_average,
        difference_entropy = diff_entropy,
        difference_variance = diff_variance,
        joint_energy = joint_energy,
        joint_entropy = joint_entropy,
        imc1 = imc1,
        imc2 = imc2,
        idm = idm,
        idmn = idmn,
        id = inverse_diff,
        idn = inverse_diff_normalized,
        inverse_variance = inverse_variance,
        maximum_probability = maximum_probability,
        sum_average = sum_average,
        sum_entropy = sum_entropy,
        sum_squares = sum_squares,
        mcc = mcc
    )
end

"""
    _compute_mcc(P::AbstractMatrix{Float64}, px::Vector{Float64}, py::Vector{Float64}, Ng::Int) -> Float64

Compute the Maximal Correlation Coefficient (MCC) from GLCM.

MCC = √(second largest eigenvalue of Q)
where Q(i,k) = Σⱼ P(i,j)·P(k,j) / (pₓ(i)·pᵧ(j))

Note: The formula uses pₓ(i) and pᵧ(j), not pₓ(i)·pₓ(k).

# Edge Cases
- Returns 1.0 for a 1×1 GLCM (single gray level)
- Handles zero marginal probabilities
"""
function _compute_mcc(P::AbstractMatrix{Float64}, px::Vector{Float64}, py::Vector{Float64}, Ng::Int)
    # Edge case: single gray level
    if Ng == 1
        return 1.0
    end

    # Build Q matrix: Q(i,k) = Σⱼ P(i,j)·P(k,j) / (pₓ(i)·pᵧ(j))
    # Following PyRadiomics formula exactly
    Q = zeros(Float64, Ng, Ng)

    @inbounds for i in 1:Ng
        for k in 1:Ng
            accum = 0.0
            for j in 1:Ng
                denom = px[i] * py[j] + GLCM_EPSILON
                accum += P[i, j] * P[k, j] / denom
            end
            Q[i, k] = accum
        end
    end

    # Compute eigenvalues
    eigenvalues = eigvals(Q)

    # Filter to real parts (Q should be real symmetric but numerical issues)
    real_eigenvalues = real.(eigenvalues)

    # Sort in descending order
    sorted_eig = sort(real_eigenvalues, rev=true)

    # Get second largest eigenvalue
    if length(sorted_eig) >= 2
        second_largest = sorted_eig[2]
        # Clamp to valid range for sqrt
        second_largest = max(second_largest, 0.0)
        return sqrt(second_largest)
    else
        return 0.0
    end
end

#==============================================================================#
# GLCM Feature Aggregation - Main Interface
#==============================================================================#

"""
    _aggregate_glcm_features(result::Union{GLCMResult, GLCMResult2D}) -> NamedTuple

Compute all GLCM features aggregated across directions using nanmean.

This function computes features for each direction and then averages them,
matching PyRadiomics behavior (average features, not average matrix).

# Returns
NamedTuple with all 24 GLCM features (Float64 values).
"""
function _aggregate_glcm_features(result::Union{GLCMResult, GLCMResult2D})
    num_dirs = result isa GLCMResult ? result.num_directions : 4
    Ng = result.Ng

    # Initialize storage for features across directions
    features_per_direction = Vector{NamedTuple}(undef, num_dirs)

    # Compute features for each direction
    for d in 1:num_dirs
        P = @view result.matrices[:, :, d]
        features_per_direction[d] = _glcm_features_single(P, Ng)
    end

    # Aggregate using nanmean (ignore NaN values if any direction has issues)
    # Since all features have the same structure, we can compute means directly

    # Helper to compute nanmean of a field across directions
    function _nanmean(field::Symbol)
        values = [getfield(f, field) for f in features_per_direction]
        valid_values = filter(!isnan, values)
        return isempty(valid_values) ? NaN : mean(valid_values)
    end

    return (
        autocorrelation = _nanmean(:autocorrelation),
        joint_average = _nanmean(:joint_average),
        cluster_prominence = _nanmean(:cluster_prominence),
        cluster_shade = _nanmean(:cluster_shade),
        cluster_tendency = _nanmean(:cluster_tendency),
        contrast = _nanmean(:contrast),
        correlation = _nanmean(:correlation),
        difference_average = _nanmean(:difference_average),
        difference_entropy = _nanmean(:difference_entropy),
        difference_variance = _nanmean(:difference_variance),
        joint_energy = _nanmean(:joint_energy),
        joint_entropy = _nanmean(:joint_entropy),
        imc1 = _nanmean(:imc1),
        imc2 = _nanmean(:imc2),
        idm = _nanmean(:idm),
        idmn = _nanmean(:idmn),
        id = _nanmean(:id),
        idn = _nanmean(:idn),
        inverse_variance = _nanmean(:inverse_variance),
        maximum_probability = _nanmean(:maximum_probability),
        sum_average = _nanmean(:sum_average),
        sum_entropy = _nanmean(:sum_entropy),
        sum_squares = _nanmean(:sum_squares),
        mcc = _nanmean(:mcc)
    )
end

#==============================================================================#
# GLCM Feature Functions - Public API
#==============================================================================#

# Each feature function computes the feature from a GLCMResult,
# averaging across all directions.

"""
    glcm_autocorrelation(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Autocorrelation feature from GLCM.

# Formula
``\\text{Autocorrelation} = \\sum_i \\sum_j P(i,j) \\cdot i \\cdot j``

# Description
Measures the fineness and coarseness of texture. Higher values indicate
coarser texture.

# References
- IBSI: QWB0
- PyRadiomics: glcm.py:getAutocorrelationFeatureValue
"""
function glcm_autocorrelation(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.autocorrelation
end

"""
    glcm_joint_average(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Joint Average (μₓ) feature from GLCM.

# Formula
``\\mu_x = \\sum_i \\sum_j P(i,j) \\cdot i``

# Description
The mean gray level intensity of the i distribution. For symmetric GLCM,
μₓ = μᵧ.

# References
- IBSI: 60VM
- PyRadiomics: glcm.py:getJointAverageFeatureValue
"""
function glcm_joint_average(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.joint_average
end

"""
    glcm_cluster_prominence(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Cluster Prominence feature from GLCM.

# Formula
``\\text{ClusterProminence} = \\sum_i \\sum_j (i + j - \\mu_x - \\mu_y)^4 \\cdot P(i,j)``

# Description
Measures the asymmetry of the GLCM. A higher value indicates more asymmetry
about the mean.

# References
- IBSI: AE86
- PyRadiomics: glcm.py:getClusterProminenceFeatureValue
"""
function glcm_cluster_prominence(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.cluster_prominence
end

"""
    glcm_cluster_shade(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Cluster Shade feature from GLCM.

# Formula
``\\text{ClusterShade} = \\sum_i \\sum_j (i + j - \\mu_x - \\mu_y)^3 \\cdot P(i,j)``

# Description
Measures the skewness of the GLCM. A higher magnitude indicates more skewness.

# References
- IBSI: 7NFM
- PyRadiomics: glcm.py:getClusterShadeFeatureValue
"""
function glcm_cluster_shade(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.cluster_shade
end

"""
    glcm_cluster_tendency(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Cluster Tendency feature from GLCM.

# Formula
``\\text{ClusterTendency} = \\sum_i \\sum_j (i + j - \\mu_x - \\mu_y)^2 \\cdot P(i,j)``

# Description
Measures the tendency of pixels to cluster together. Higher values indicate
more clustering.

# References
- IBSI: DG8W
- PyRadiomics: glcm.py:getClusterTendencyFeatureValue
"""
function glcm_cluster_tendency(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.cluster_tendency
end

"""
    glcm_contrast(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Contrast feature from GLCM.

# Formula
``\\text{Contrast} = \\sum_i \\sum_j (i - j)^2 \\cdot P(i,j)``

# Description
Measures the amount of local intensity variation in the image. Higher values
indicate more variation (less homogeneity).

# References
- IBSI: ACUI
- PyRadiomics: glcm.py:getContrastFeatureValue
"""
function glcm_contrast(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.contrast
end

"""
    glcm_correlation(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Correlation feature from GLCM.

# Formula
``\\text{Correlation} = \\frac{\\sum_i \\sum_j P(i,j) \\cdot i \\cdot j - \\mu_x \\mu_y}{\\sigma_x \\sigma_y}``

# Description
Measures the linear dependency of gray level values on their respective voxels.
Range is [0, 1] where 1 indicates perfect correlation.

# Edge Cases
Returns 1 for flat regions where σ = 0.

# References
- IBSI: NI2N
- PyRadiomics: glcm.py:getCorrelationFeatureValue
"""
function glcm_correlation(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.correlation
end

"""
    glcm_difference_average(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Difference Average feature from GLCM.

# Formula
``\\text{DifferenceAverage} = \\sum_{k=0}^{N_g-1} k \\cdot p_{x-y}(k)``

# Description
The mean of the difference distribution p_{x-y}.

# References
- IBSI: TF7R
- PyRadiomics: glcm.py:getDifferenceAverageFeatureValue
"""
function glcm_difference_average(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.difference_average
end

"""
    glcm_difference_entropy(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Difference Entropy feature from GLCM.

# Formula
``\\text{DifferenceEntropy} = -\\sum_{k=0}^{N_g-1} p_{x-y}(k) \\cdot \\log_2(p_{x-y}(k) + \\epsilon)``

# Description
Measures the randomness/variability in the neighborhood intensity value differences.

# References
- IBSI: NTRS
- PyRadiomics: glcm.py:getDifferenceEntropyFeatureValue
"""
function glcm_difference_entropy(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.difference_entropy
end

"""
    glcm_difference_variance(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Difference Variance feature from GLCM.

# Formula
``\\text{DifferenceVariance} = \\sum_{k=0}^{N_g-1} (k - DA)^2 \\cdot p_{x-y}(k)``

where DA is the Difference Average.

# Description
Measures the heterogeneity that places higher weights on differing intensity
level pairs that deviate more from the mean.

# References
- IBSI: D3YU
- PyRadiomics: glcm.py:getDifferenceVarianceFeatureValue
"""
function glcm_difference_variance(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.difference_variance
end

"""
    glcm_joint_energy(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Joint Energy (Angular Second Moment) feature from GLCM.

# Formula
``\\text{JointEnergy} = \\sum_i \\sum_j P(i,j)^2``

# Description
Measures the orderliness of the image. High values indicate that the GLCM
has few dominant gray level transitions.

# References
- IBSI: 8ZQL (Angular Second Moment)
- PyRadiomics: glcm.py:getJointEnergyFeatureValue
"""
function glcm_joint_energy(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.joint_energy
end

"""
    glcm_joint_entropy(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute the Joint Entropy feature from GLCM.

# Formula
``\\text{JointEntropy} = -\\sum_i \\sum_j P(i,j) \\cdot \\log_2(P(i,j) + \\epsilon)``

# Description
Measures the uncertainty/randomness in the image values. Higher values indicate
more heterogeneity.

# References
- IBSI: TU9B
- PyRadiomics: glcm.py:getJointEntropyFeatureValue
"""
function glcm_joint_entropy(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.joint_entropy
end

"""
    glcm_imc1(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Informational Measure of Correlation 1 (IMC1) from GLCM.

# Formula
``\\text{IMC1} = \\frac{HXY - HXY1}{\\max(HX, HY)}``

where:
- HXY = Joint Entropy
- HXY1 = -Σᵢ Σⱼ P(i,j)·log₂(pₓ(i)·pᵧ(j)+ε)
- HX = -Σᵢ pₓ(i)·log₂(pₓ(i)+ε)
- HY = -Σⱼ pᵧ(j)·log₂(pᵧ(j)+ε)

# Edge Cases
Returns 0 when both HX and HY are 0 (flat regions).

# References
- IBSI: R8DG
- PyRadiomics: glcm.py:getImc1FeatureValue
"""
function glcm_imc1(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.imc1
end

"""
    glcm_imc2(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Informational Measure of Correlation 2 (IMC2) from GLCM.

# Formula
``\\text{IMC2} = \\sqrt{1 - e^{-2(HXY2 - HXY)}}``

where:
- HXY = Joint Entropy
- HXY2 = -Σᵢ Σⱼ pₓ(i)·pᵧ(j)·log₂(pₓ(i)·pᵧ(j)+ε)

# Edge Cases
Returns 0 when HXY > HXY2 (due to floating-point precision, would give complex number).
Range is [0, 1].

# References
- IBSI: JN9H
- PyRadiomics: glcm.py:getImc2FeatureValue
"""
function glcm_imc2(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.imc2
end

"""
    glcm_idm(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Inverse Difference Moment (IDM/Homogeneity 2) from GLCM.

# Formula
``\\text{IDM} = \\sum_{k=0}^{N_g-1} \\frac{p_{x-y}(k)}{1 + k^2}``

# Description
Measures the local homogeneity of an image. IDM weights are the inverse of the
Contrast weights.

# References
- IBSI: WF0Z
- PyRadiomics: glcm.py:getIdmFeatureValue
"""
function glcm_idm(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.idm
end

"""
    glcm_idmn(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Inverse Difference Moment Normalized (IDMN) from GLCM.

# Formula
``\\text{IDMN} = \\sum_{k=0}^{N_g-1} \\frac{p_{x-y}(k)}{1 + k^2/N_g^2}``

# Description
A normalized version of IDM. Normalizing factor reduces the effect of the
number of gray levels.

# References
- IBSI: 1QCO
- PyRadiomics: glcm.py:getIdmnFeatureValue
"""
function glcm_idmn(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.idmn
end

"""
    glcm_id(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Inverse Difference (ID/Homogeneity 1) from GLCM.

# Formula
``\\text{ID} = \\sum_{k=0}^{N_g-1} \\frac{p_{x-y}(k)}{1 + k}``

# Description
Measures the local homogeneity. Higher values indicate more homogeneous texture.

# References
- IBSI: IB1Z
- PyRadiomics: glcm.py:getIdFeatureValue
"""
function glcm_id(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.id
end

"""
    glcm_idn(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Inverse Difference Normalized (IDN) from GLCM.

# Formula
``\\text{IDN} = \\sum_{k=0}^{N_g-1} \\frac{p_{x-y}(k)}{1 + k/N_g}``

# Description
A normalized version of Inverse Difference.

# References
- IBSI: NDRX
- PyRadiomics: glcm.py:getIdnFeatureValue
"""
function glcm_idn(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.idn
end

"""
    glcm_inverse_variance(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Inverse Variance from GLCM.

# Formula
``\\text{InverseVariance} = \\sum_{k=1}^{N_g-1} \\frac{p_{x-y}(k)}{k^2}``

# Description
Note: k=0 is skipped to avoid division by zero.

# References
- IBSI: E8JP
- PyRadiomics: glcm.py:getInverseVarianceFeatureValue
"""
function glcm_inverse_variance(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.inverse_variance
end

"""
    glcm_maximum_probability(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Maximum Probability (Joint Maximum) from GLCM.

# Formula
``\\text{MaximumProbability} = \\max\\{P(i,j)\\}``

# Description
The maximum probability value in the GLCM. Higher values indicate that
one gray level transition dominates.

# References
- IBSI: GYBY (Joint Maximum)
- PyRadiomics: glcm.py:getMaximumProbabilityFeatureValue
"""
function glcm_maximum_probability(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.maximum_probability
end

"""
    glcm_sum_average(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Sum Average from GLCM.

# Formula
``\\text{SumAverage} = \\sum_{k=2}^{2N_g} k \\cdot p_{x+y}(k)``

# Description
The mean of the sum distribution p_{x+y}. Note: For symmetric GLCM,
SumAverage = 2 × JointAverage.

# References
- IBSI: ZGXS
- PyRadiomics: glcm.py:getSumAverageFeatureValue
"""
function glcm_sum_average(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.sum_average
end

"""
    glcm_sum_entropy(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Sum Entropy from GLCM.

# Formula
``\\text{SumEntropy} = -\\sum_{k=2}^{2N_g} p_{x+y}(k) \\cdot \\log_2(p_{x+y}(k) + \\epsilon)``

# Description
Measures the randomness of the sum distribution.

# References
- IBSI: P6QZ
- PyRadiomics: glcm.py:getSumEntropyFeatureValue
"""
function glcm_sum_entropy(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.sum_entropy
end

"""
    glcm_sum_squares(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Sum of Squares (Joint Variance) from GLCM.

# Formula
``\\text{SumSquares} = \\sum_i \\sum_j (i - \\mu_x)^2 \\cdot P(i,j)``

# Description
The variance of the i distribution (for symmetric GLCM, equals j variance).
Also known as Joint Variance in IBSI.

# References
- IBSI: UR99 (Joint Variance)
- PyRadiomics: glcm.py:getSumSquaresFeatureValue
"""
function glcm_sum_squares(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.sum_squares
end

"""
    glcm_mcc(result::Union{GLCMResult, GLCMResult2D}) -> Float64

Compute Maximal Correlation Coefficient (MCC) from GLCM.

# Formula
``\\text{MCC} = \\sqrt{\\text{second largest eigenvalue of } Q}``

where Q(i,k) = Σⱼ P(i,j)·P(k,j) / (pₓ(i)·pₓ(k))

# Description
Measures the complexity of the texture. Range is [0, 1].

# Edge Cases
- Returns 1.0 for single gray level (1×1 GLCM)

# References
- IBSI: QCDE
- PyRadiomics: glcm.py:getMCCFeatureValue
"""
function glcm_mcc(result::Union{GLCMResult, GLCMResult2D})
    features = _aggregate_glcm_features(result)
    return features.mcc
end

#==============================================================================#
# Convenience Function - Extract All GLCM Features
#==============================================================================#

"""
    glcm_features(result::Union{GLCMResult, GLCMResult2D}) -> NamedTuple

Extract all 24 GLCM features at once.

# Returns
NamedTuple with all GLCM features:
- `autocorrelation`
- `joint_average`
- `cluster_prominence`
- `cluster_shade`
- `cluster_tendency`
- `contrast`
- `correlation`
- `difference_average`
- `difference_entropy`
- `difference_variance`
- `joint_energy`
- `joint_entropy`
- `imc1`
- `imc2`
- `idm`
- `idmn`
- `id`
- `idn`
- `inverse_variance`
- `maximum_probability`
- `sum_average`
- `sum_entropy`
- `sum_squares`
- `mcc`

# Example
```julia
result = compute_glcm(image, mask)
features = glcm_features(result)
println("Contrast: ", features.contrast)
println("Correlation: ", features.correlation)
```
"""
function glcm_features(result::Union{GLCMResult, GLCMResult2D})
    return _aggregate_glcm_features(result)
end

"""
    glcm_features(image, mask; binwidth=25.0, bincount=nothing, distance=1, symmetric=true)

Convenience function to compute all GLCM features from raw image and mask.

# Arguments
- `image`: Image array (will be discretized if floating point)
- `mask`: Boolean mask for ROI
- `binwidth`: Bin width for discretization (default: 25.0)
- `bincount`: Fixed bin count (overrides binwidth if specified)
- `distance`: GLCM distance parameter (default: 1)
- `symmetric`: Whether to compute symmetric GLCM (default: true)

# Returns
NamedTuple with all 24 GLCM features.

# Example
```julia
features = glcm_features(image, mask, binwidth=32.0)
```
"""
function glcm_features(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing,
                       distance::Int=1,
                       symmetric::Bool=true)
    result = compute_glcm(image, mask; binwidth=binwidth, bincount=bincount,
                          distance=distance, symmetric=symmetric)
    return glcm_features(result)
end

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

# Export all GLCM feature functions
export glcm_autocorrelation, glcm_joint_average, glcm_cluster_prominence
export glcm_cluster_shade, glcm_cluster_tendency, glcm_contrast
export glcm_correlation, glcm_difference_average, glcm_difference_entropy
export glcm_difference_variance, glcm_joint_energy, glcm_joint_entropy
export glcm_imc1, glcm_imc2, glcm_idm, glcm_idmn, glcm_id, glcm_idn
export glcm_inverse_variance, glcm_maximum_probability
export glcm_sum_average, glcm_sum_entropy, glcm_sum_squares, glcm_mcc
export glcm_features
