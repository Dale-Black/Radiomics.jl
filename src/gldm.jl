# Gray Level Dependence Matrix (GLDM) for Radiomics.jl
#
# This module implements GLDM computation and features.
# GLDM quantifies gray level dependencies in an image, where a dependency is
# defined as the number of connected voxels within a specified distance that
# have similar gray levels (within alpha tolerance).
#
# The GLDM matrix P(i,j) counts voxels with:
# - Gray level i
# - Exactly j dependent neighbors (where |gray_level_neighbor - i| ≤ alpha)
#
# Total: 14 features
#
# References:
# - PyRadiomics: radiomics/gldm.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - Sun C, Wee WG. "Neighboring Gray Level Dependence Matrix for Texture Classification."
#   Computer Vision, Graphics, and Image Processing, 1983.

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding log(0) - matches np.spacing(1)
const GLDM_EPSILON = eps(Float64)  # ≈ 2.2e-16

# 13 unique 3D directions for neighbor checking (26-connectivity/2)
# Same as used in GLCM/GLRLM/NGTDM
const GLDM_DIRECTIONS_3D = [
    CartesianIndex(1, 0, 0),    # X-axis positive
    CartesianIndex(0, 1, 0),    # Y-axis positive
    CartesianIndex(0, 0, 1),    # Z-axis positive
    CartesianIndex(1, 1, 0),    # XY diagonal
    CartesianIndex(1, -1, 0),   # XY anti-diagonal
    CartesianIndex(1, 0, 1),    # XZ diagonal
    CartesianIndex(1, 0, -1),   # XZ anti-diagonal
    CartesianIndex(0, 1, 1),    # YZ diagonal
    CartesianIndex(0, 1, -1),   # YZ anti-diagonal
    CartesianIndex(1, 1, 1),    # Body diagonal 1
    CartesianIndex(1, 1, -1),   # Body diagonal 2
    CartesianIndex(1, -1, 1),   # Body diagonal 3
    CartesianIndex(1, -1, -1),  # Body diagonal 4
]

# 4 unique 2D directions for neighbor checking (8-connectivity/2)
const GLDM_DIRECTIONS_2D = [
    CartesianIndex(1, 0),    # X-axis positive
    CartesianIndex(0, 1),    # Y-axis positive
    CartesianIndex(1, 1),    # Diagonal
    CartesianIndex(1, -1),   # Anti-diagonal
]

#==============================================================================#
# GLDM Result Types
#==============================================================================#

"""
    GLDMResult

Container for GLDM computation results.

# Fields
- `P::Matrix{Float64}`: The GLDM matrix (gray levels × dependence sizes)
- `ivector::Vector{Int}`: Gray level values (indices) present in the ROI
- `jvector::Vector{Int}`: Dependence sizes present in the ROI
- `Nz::Int`: Total number of voxels in the ROI (= sum of all P elements)
- `Ng::Int`: Number of gray levels in discretized image
- `max_dependence::Int`: Maximum possible dependence count (26 for 3D with distance=1)

# Notes
The GLDM matrix is stored with dimensions Ng × (max_dependence + 1).
- Row i corresponds to gray level i
- Column j corresponds to dependence count j-1 (0-indexed dependence)
- Empty gray levels (not in ROI) contribute zero rows
- Empty dependence sizes contribute zero columns

Key property: Nz = Np (number of voxels) since every voxel has exactly one
dependence zone entry in the matrix.
"""
struct GLDMResult
    P::Matrix{Float64}         # GLDM matrix [Ng × (max_dependence+1)]
    ivector::Vector{Int}       # Non-empty gray level indices
    jvector::Vector{Int}       # Non-empty dependence size values
    Nz::Int                    # Total zones (= total voxels)
    Ng::Int                    # Number of gray levels
    max_dependence::Int        # Maximum possible dependence
end

"""
    GLDMResult2D

Container for 2D GLDM computation results.

# Fields
Same as GLDMResult, but for 2D images.
Maximum dependence is 8 for 2D with distance=1.
"""
struct GLDMResult2D
    P::Matrix{Float64}
    ivector::Vector{Int}
    jvector::Vector{Int}
    Nz::Int
    Ng::Int
    max_dependence::Int
end

#==============================================================================#
# GLDM Matrix Computation - 3D
#==============================================================================#

"""
    compute_gldm(image::AbstractArray{<:Integer, 3}, mask::AbstractArray{Bool, 3};
                 Ng::Union{Int, Nothing}=nothing,
                 alpha::Int=0,
                 distance::Int=1) -> GLDMResult

Compute Gray Level Dependence Matrix for a 3D image.

The GLDM matrix P(i,j) counts the number of voxels with:
- Gray level i
- Exactly j dependent neighbors

A neighbor is considered **dependent** if its gray level differs from the
center voxel's gray level by at most α (alpha).

# Arguments
- `image`: 3D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 3D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)
- `alpha::Int=0`: Coarseness parameter. Neighbors with |diff| ≤ α are dependent.
                  Default is 0 (only exact matches count as dependent).
- `distance::Int=1`: Chebyshev distance for neighborhood (default: 1)

# Returns
- `GLDMResult`: Struct containing:
  - `P`: GLDM matrix
  - `ivector`: Gray level values present in ROI
  - `jvector`: Dependence sizes present in ROI
  - `Nz`: Total number of voxels (zones)
  - `Ng`: Number of gray levels
  - `max_dependence`: Maximum possible dependence count

# Notes
- Input image should be discretized (integer gray levels starting from 1)
- All voxels in the mask are included, even boundary voxels with fewer neighbors
- Unlike NGTDM, GLDM does NOT exclude boundary voxels
- With distance=1, each voxel has at most 26 neighbors (13 directions × 2)
- Nz always equals the number of voxels in the mask

# Example
```julia
# Discretize image first
discretized = discretize_image(image, mask, binwidth=25.0)
result = compute_gldm(discretized.discretized, mask)

# Access the matrix
P = result.P
```

# References
- PyRadiomics: radiomics/gldm.py
- IBSI: Section 3.6.7 (Grey level dependence based features)
"""
function compute_gldm(image::AbstractArray{<:Integer, 3},
                      mask::AbstractArray{Bool, 3};
                      Ng::Union{Int, Nothing}=nothing,
                      alpha::Int=0,
                      distance::Int=1)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive, got $distance"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative, got $alpha"))

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

    # Get all neighbor offsets within Chebyshev distance
    # For distance=1: 26 neighbors (13 directions × 2)
    offsets = _get_gldm_offsets_3d(distance)
    max_dependence = length(offsets)  # Maximum possible dependence count

    # Initialize GLDM matrix: rows = gray levels (1:Ng), cols = dependence (0:max_dependence)
    # Column index j+1 corresponds to dependence count j
    P = zeros(Float64, Ng, max_dependence + 1)

    # Iterate over all voxels in the mask
    @inbounds for idx in CartesianIndices(image)
        # Skip if not in mask
        mask[idx] || continue

        gray_level = image[idx]
        if gray_level < 1 || gray_level > Ng
            continue  # Invalid gray level
        end

        # Count dependent neighbors
        dependence_count = 0

        for offset in offsets
            neighbor_idx = idx + offset

            # Check bounds
            if checkbounds(Bool, image, neighbor_idx)
                # Check if neighbor is in mask
                if mask[neighbor_idx]
                    neighbor_gray = image[neighbor_idx]
                    # Check if neighbor is dependent (within alpha tolerance)
                    if abs(gray_level - neighbor_gray) <= alpha
                        dependence_count += 1
                    end
                end
            end
        end

        # Increment GLDM matrix
        # Column index is dependence_count + 1 (1-indexed, so 0 dependence → column 1)
        P[gray_level, dependence_count + 1] += 1.0
    end

    # Compute derived values
    Nz = Int(sum(P))  # Total zones = total voxels

    # Find non-empty gray levels (rows with any non-zero entries)
    ivector = findall(i -> any(P[i, :] .> 0), 1:Ng)

    # Find non-empty dependence sizes (columns with any non-zero entries)
    # jvector contains actual dependence values (0-indexed)
    jvector = findall(j -> any(P[:, j] .> 0), 1:(max_dependence + 1)) .- 1

    return GLDMResult(P, ivector, jvector, Nz, Ng, max_dependence)
end

"""
    _get_gldm_offsets_3d(distance::Int) -> Vector{CartesianIndex{3}}

Get all neighbor offsets within Chebyshev distance in 3D.

For GLDM, we need ALL neighbors (both forward and backward directions),
not just the unique 13 directions used in GLCM.

For distance=1, returns 26 neighbors (3×3×3 cube minus center).

# Arguments
- `distance::Int`: Maximum Chebyshev distance (typically 1)

# Returns
- Vector of CartesianIndex offsets for all neighbors
"""
function _get_gldm_offsets_3d(distance::Int)
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

#==============================================================================#
# GLDM Matrix Computation - 2D
#==============================================================================#

"""
    compute_gldm_2d(image::AbstractArray{<:Integer, 2}, mask::AbstractArray{Bool, 2};
                    Ng::Union{Int, Nothing}=nothing,
                    alpha::Int=0,
                    distance::Int=1) -> GLDMResult2D

Compute Gray Level Dependence Matrix for a 2D image.

# Arguments
- `image`: 2D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 2D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)
- `alpha::Int=0`: Coarseness parameter. Neighbors with |diff| ≤ α are dependent.
- `distance::Int=1`: Chebyshev distance for neighborhood (default: 1)

# Returns
- `GLDMResult2D`: Struct containing GLDM matrix and metadata

# Example
```julia
image_2d = discretize(image_slice, edges)
mask_2d = mask[:, :, 32]
result = compute_gldm_2d(image_2d, mask_2d)
```
"""
function compute_gldm_2d(image::AbstractArray{<:Integer, 2},
                         mask::AbstractArray{Bool, 2};
                         Ng::Union{Int, Nothing}=nothing,
                         alpha::Int=0,
                         distance::Int=1)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions"
    ))
    distance > 0 || throw(ArgumentError("distance must be positive"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))

    # Get masked pixels
    masked_values = image[mask]
    if isempty(masked_values)
        throw(ArgumentError("Mask is empty"))
    end

    # Determine number of gray levels
    if isnothing(Ng)
        Ng = maximum(masked_values)
    end

    # Get all neighbor offsets
    offsets = _get_gldm_offsets_2d(distance)
    max_dependence = length(offsets)

    # Initialize GLDM matrix
    P = zeros(Float64, Ng, max_dependence + 1)

    # Iterate over all pixels in the mask
    @inbounds for idx in CartesianIndices(image)
        mask[idx] || continue

        gray_level = image[idx]
        if gray_level < 1 || gray_level > Ng
            continue
        end

        # Count dependent neighbors
        dependence_count = 0

        for offset in offsets
            neighbor_idx = idx + offset

            if checkbounds(Bool, image, neighbor_idx)
                if mask[neighbor_idx]
                    neighbor_gray = image[neighbor_idx]
                    if abs(gray_level - neighbor_gray) <= alpha
                        dependence_count += 1
                    end
                end
            end
        end

        # Increment GLDM matrix
        P[gray_level, dependence_count + 1] += 1.0
    end

    # Compute derived values
    Nz = Int(sum(P))
    ivector = findall(i -> any(P[i, :] .> 0), 1:Ng)
    jvector = findall(j -> any(P[:, j] .> 0), 1:(max_dependence + 1)) .- 1

    return GLDMResult2D(P, ivector, jvector, Nz, Ng, max_dependence)
end

"""
    _get_gldm_offsets_2d(distance::Int) -> Vector{CartesianIndex{2}}

Get all neighbor offsets within Chebyshev distance in 2D.

For distance=1, returns 8 neighbors (3×3 square minus center).

# Arguments
- `distance::Int`: Maximum Chebyshev distance (typically 1)

# Returns
- Vector of CartesianIndex offsets for all neighbors
"""
function _get_gldm_offsets_2d(distance::Int)
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
# High-Level GLDM Computation Interface
#==============================================================================#

"""
    compute_gldm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                 binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                 alpha::Int=0, distance::Int=1) -> GLDMResult

Compute GLDM from a non-discretized image (convenience wrapper).

This function discretizes the image before computing the GLDM.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `alpha::Int=0`: Coarseness parameter for dependence definition
- `distance::Int=1`: Chebyshev distance for neighborhood

# Returns
- For 3D images: `GLDMResult`
- For 2D images: `GLDMResult2D`

# Example
```julia
result = compute_gldm(image, mask, binwidth=25.0, alpha=0)
```
"""
function compute_gldm(image::AbstractArray{<:Real, 3},
                      mask::AbstractArray{Bool, 3};
                      binwidth::Real=25.0,
                      bincount::Union{Int, Nothing}=nothing,
                      alpha::Int=0,
                      distance::Int=1)

    # Discretize the image
    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Compute GLDM on discretized image
    return compute_gldm(disc_result.discretized, mask; alpha=alpha, distance=distance)
end

function compute_gldm(image::AbstractArray{<:Real, 2},
                      mask::AbstractArray{Bool, 2};
                      binwidth::Real=25.0,
                      bincount::Union{Int, Nothing}=nothing,
                      alpha::Int=0,
                      distance::Int=1)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    return compute_gldm_2d(disc_result.discretized, mask; alpha=alpha, distance=distance)
end

#==============================================================================#
# GLDM with Settings
#==============================================================================#

"""
    compute_gldm(image, mask, settings::Settings)

Compute GLDM using parameters from a Settings object.

# Example
```julia
settings = Settings(binwidth=32.0, gldm_alpha=0, gldm_distance=1)
result = compute_gldm(image, mask, settings)
```
"""
function compute_gldm(image::AbstractArray{<:Real},
                      mask::AbstractArray{Bool},
                      settings::Settings)

    bincount = settings.discretization_mode == FixedBinCount ? settings.bincount : nothing

    # Note: Settings struct needs gldm_alpha and gldm_distance fields
    # For now, use defaults if not present
    alpha = hasproperty(settings, :gldm_alpha) ? settings.gldm_alpha : 0
    dist = hasproperty(settings, :gldm_distance) ? settings.gldm_distance : 1

    return compute_gldm(image, mask;
                        binwidth=settings.binwidth,
                        bincount=bincount,
                        alpha=alpha,
                        distance=dist)
end

#==============================================================================#
# GLDM Utility Functions
#==============================================================================#

"""
    gldm_num_gray_levels(result::Union{GLDMResult, GLDMResult2D}) -> Int

Get the total number of gray levels in the GLDM.
"""
gldm_num_gray_levels(result::GLDMResult) = result.Ng
gldm_num_gray_levels(result::GLDMResult2D) = result.Ng

"""
    gldm_num_valid_gray_levels(result::Union{GLDMResult, GLDMResult2D}) -> Int

Get the number of non-empty gray levels (gray levels present in the ROI).
"""
gldm_num_valid_gray_levels(result::GLDMResult) = length(result.ivector)
gldm_num_valid_gray_levels(result::GLDMResult2D) = length(result.ivector)

"""
    gldm_num_zones(result::Union{GLDMResult, GLDMResult2D}) -> Int

Get the total number of zones (Nz), which equals the number of voxels.
"""
gldm_num_zones(result::GLDMResult) = result.Nz
gldm_num_zones(result::GLDMResult2D) = result.Nz

"""
    gldm_max_dependence(result::Union{GLDMResult, GLDMResult2D}) -> Int

Get the maximum possible dependence count.
"""
gldm_max_dependence(result::GLDMResult) = result.max_dependence
gldm_max_dependence(result::GLDMResult2D) = result.max_dependence

"""
    gldm_gray_levels(result::Union{GLDMResult, GLDMResult2D}) -> Vector{Int}

Get the vector of gray levels present in the ROI.
"""
gldm_gray_levels(result::GLDMResult) = result.ivector
gldm_gray_levels(result::GLDMResult2D) = result.ivector

"""
    gldm_dependence_sizes(result::Union{GLDMResult, GLDMResult2D}) -> Vector{Int}

Get the vector of dependence sizes present in the ROI.
"""
gldm_dependence_sizes(result::GLDMResult) = result.jvector
gldm_dependence_sizes(result::GLDMResult2D) = result.jvector

#==============================================================================#
# GLDM Features - Dependence Emphasis Features (1-2)
#==============================================================================#

"""
    gldm_small_dependence_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Small Dependence Emphasis (SDE): Measures the distribution of small dependencies.

# Mathematical Formula
```
SDE = (1/Nz) × Σᵢ Σⱼ P(i,j) / j²
```

where:
- P(i,j) is the GLDM matrix (gray level i, dependence column j)
- Nz is the total number of voxels (zones)
- j is the 1-indexed column position (matching PyRadiomics convention)

Higher values indicate smaller dependence and less homogeneous textures.

# Notes
- PyRadiomics uses jvector = [1, 2, 3, ..., Nd] for column indices
- Column 1 corresponds to dependence count 0, column 2 to count 1, etc.
- We use the same 1-indexed j values for parity

# References
- PyRadiomics: gldm.py:getSmallDependenceEmphasisFeatureValue
- IBSI Code: SODN
"""
function gldm_small_dependence_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    # In Julia, column index j is already 1-indexed, matching this convention
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] / j_sq
        end
    end

    return total / Nz
end

"""
    gldm_large_dependence_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Large Dependence Emphasis (LDE): Measures the distribution of large dependencies.

# Mathematical Formula
```
LDE = (1/Nz) × Σᵢ Σⱼ P(i,j) × j²
```

where j is the 1-indexed column position (matching PyRadiomics convention).

Higher values indicate larger dependence and more homogeneous textures.

# References
- PyRadiomics: gldm.py:getLargeDependenceEmphasisFeatureValue
- IBSI Code: IANU
"""
function gldm_large_dependence_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] * j_sq
        end
    end

    return total / Nz
end

#==============================================================================#
# GLDM Features - Non-Uniformity Features (3-5)
#==============================================================================#

"""
    gldm_gray_level_non_uniformity(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Gray Level Non-Uniformity (GLN): Measures variability of gray level intensity.

# Mathematical Formula
```
GLN = (1/Nz) × Σᵢ (Σⱼ P(i,j))²
```

Lower values indicate more homogeneous gray level distribution.

# References
- PyRadiomics: gldm.py:getGrayLevelNonUniformityFeatureValue
- IBSI Code: FP8K
"""
function gldm_gray_level_non_uniformity(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # pg(i) = Σⱼ P(i,j)
    total = 0.0
    @inbounds for i in 1:Ng
        pg_i = 0.0
        for j in 1:Nd
            pg_i += P[i, j]
        end
        total += pg_i * pg_i
    end

    return total / Nz
end

"""
    gldm_dependence_non_uniformity(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Dependence Non-Uniformity (DN): Measures variability of dependence sizes.

# Mathematical Formula
```
DN = (1/Nz) × Σⱼ (Σᵢ P(i,j))²
```

Lower values indicate more homogeneous dependence size distribution.

# References
- PyRadiomics: gldm.py:getDependenceNonUniformityFeatureValue
- IBSI Code: Z87G
"""
function gldm_dependence_non_uniformity(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # pd(j) = Σᵢ P(i,j)
    total = 0.0
    @inbounds for j in 1:Nd
        pd_j = 0.0
        for i in 1:Ng
            pd_j += P[i, j]
        end
        total += pd_j * pd_j
    end

    return total / Nz
end

"""
    gldm_dependence_non_uniformity_normalized(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Dependence Non-Uniformity Normalized (DNN): Normalized measure of dependence size variability.

# Mathematical Formula
```
DNN = (1/Nz²) × Σⱼ (Σᵢ P(i,j))²
```

Similar to DN but normalized by Nz², making it more comparable across different images.

# References
- PyRadiomics: gldm.py:getDependenceNonUniformityNormalizedFeatureValue
- IBSI Code: OKJI
"""
function gldm_dependence_non_uniformity_normalized(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # pd(j) = Σᵢ P(i,j)
    total = 0.0
    @inbounds for j in 1:Nd
        pd_j = 0.0
        for i in 1:Ng
            pd_j += P[i, j]
        end
        total += pd_j * pd_j
    end

    return total / (Nz * Nz)
end

#==============================================================================#
# GLDM Features - Variance and Entropy Features (6-8)
#==============================================================================#

"""
    gldm_gray_level_variance(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Gray Level Variance (GLV): Measures variance in gray level intensities.

# Mathematical Formula
```
GLV = Σᵢ Σⱼ p(i,j) × (i - μᵢ)²
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLDM)
- μᵢ = Σᵢ Σⱼ p(i,j) × i (mean gray level)

# References
- PyRadiomics: gldm.py:getGrayLevelVarianceFeatureValue
- IBSI Code: QK93
"""
function gldm_gray_level_variance(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # Compute mean gray level μᵢ = Σᵢ Σⱼ p(i,j) × i
    mu_i = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            mu_i += (P[i, j] / Nz) * i
        end
    end

    # Compute variance
    variance = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            variance += p_ij * (i - mu_i)^2
        end
    end

    return variance
end

"""
    gldm_dependence_variance(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Dependence Variance (DV): Measures variance in dependence sizes.

# Mathematical Formula
```
DV = Σᵢ Σⱼ p(i,j) × (j - μⱼ)²
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLDM)
- μⱼ = Σᵢ Σⱼ p(i,j) × j (mean dependence size)
- j is the 1-indexed column position (matching PyRadiomics convention)

# References
- PyRadiomics: gldm.py:getDependenceVarianceFeatureValue
- IBSI Code: 7162
"""
function gldm_dependence_variance(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # Compute mean dependence size μⱼ = Σᵢ Σⱼ p(i,j) × j
    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    mu_j = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            mu_j += (P[i, j] / Nz) * j  # Use 1-indexed column position
        end
    end

    # Compute variance
    variance = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            variance += p_ij * (j - mu_j)^2  # Use 1-indexed column position
        end
    end

    return variance
end

"""
    gldm_dependence_entropy(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Dependence Entropy (DE): Measures the randomness/heterogeneity of dependence distribution.

# Mathematical Formula
```
DE = -Σᵢ Σⱼ p(i,j) × log₂(p(i,j) + ε)
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLDM)
- ε ≈ 2.2×10⁻¹⁶ (machine epsilon to prevent log(0))

Higher values indicate more heterogeneous distributions.

# References
- PyRadiomics: gldm.py:getDependenceEntropyFeatureValue
- IBSI Code: GBDU
"""
function gldm_dependence_entropy(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    entropy = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            if p_ij > 0
                entropy -= p_ij * log2(p_ij + GLDM_EPSILON)
            end
        end
    end

    return entropy
end

#==============================================================================#
# GLDM Features - Gray Level Emphasis Features (9-10)
#==============================================================================#

"""
    gldm_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Low Gray Level Emphasis (LGLE): Measures the distribution of lower gray levels.

# Mathematical Formula
```
LGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) / i²
```

Higher values indicate a greater proportion of lower gray level values.

# References
- PyRadiomics: gldm.py:getLowGrayLevelEmphasisFeatureValue
- IBSI Code: 5W23
"""
function gldm_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            total += P[i, j] / (i * i)
        end
    end

    return total / Nz
end

"""
    gldm_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

High Gray Level Emphasis (HGLE): Measures the distribution of higher gray levels.

# Mathematical Formula
```
HGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i²
```

Higher values indicate a greater proportion of higher gray level values.

# References
- PyRadiomics: gldm.py:getHighGrayLevelEmphasisFeatureValue
- IBSI Code: DHV0
"""
function gldm_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Nd
        for i in 1:Ng
            total += P[i, j] * (i * i)
        end
    end

    return total / Nz
end

#==============================================================================#
# GLDM Features - Combined Emphasis Features (11-14)
#==============================================================================#

"""
    gldm_small_dependence_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Small Dependence Low Gray Level Emphasis (SDLGLE): Measures the joint distribution
of small dependencies and low gray levels.

# Mathematical Formula
```
SDLGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) / (i² × j²)
```

where j is the 1-indexed column position (matching PyRadiomics convention).

Higher values indicate a greater proportion of small dependencies with low gray levels.

# References
- PyRadiomics: gldm.py:getSmallDependenceLowGrayLevelEmphasisFeatureValue
- IBSI Code: RUVG
"""
function gldm_small_dependence_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] / (i * i * j_sq)
        end
    end

    return total / Nz
end

"""
    gldm_small_dependence_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Small Dependence High Gray Level Emphasis (SDHGLE): Measures the joint distribution
of small dependencies and high gray levels.

# Mathematical Formula
```
SDHGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i² / j²
```

where j is the 1-indexed column position (matching PyRadiomics convention).

Higher values indicate a greater proportion of small dependencies with high gray levels.

# References
- PyRadiomics: gldm.py:getSmallDependenceHighGrayLevelEmphasisFeatureValue
- IBSI Code: DKNJ
"""
function gldm_small_dependence_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] * (i * i) / j_sq
        end
    end

    return total / Nz
end

"""
    gldm_large_dependence_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Large Dependence Low Gray Level Emphasis (LDLGLE): Measures the joint distribution
of large dependencies and low gray levels.

# Mathematical Formula
```
LDLGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × j² / i²
```

where j is the 1-indexed column position (matching PyRadiomics convention).

Higher values indicate a greater proportion of large dependencies with low gray levels.

# References
- PyRadiomics: gldm.py:getLargeDependenceLowGrayLevelEmphasisFeatureValue
- IBSI Code: A7WM
"""
function gldm_large_dependence_low_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] * j_sq / (i * i)
        end
    end

    return total / Nz
end

"""
    gldm_large_dependence_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D}) -> Float64

Large Dependence High Gray Level Emphasis (LDHGLE): Measures the joint distribution
of large dependencies and high gray levels.

# Mathematical Formula
```
LDHGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i² × j²
```

where j is the 1-indexed column position (matching PyRadiomics convention).

Higher values indicate a greater proportion of large dependencies with high gray levels.

# References
- PyRadiomics: gldm.py:getLargeDependenceHighGrayLevelEmphasisFeatureValue
- IBSI Code: KLTH
"""
function gldm_large_dependence_high_gray_level_emphasis(result::Union{GLDMResult, GLDMResult2D})
    P = result.P
    Nz = result.Nz
    Ng = size(P, 1)
    Nd = size(P, 2)

    if Nz == 0
        return NaN
    end

    # PyRadiomics uses jvector = np.arange(1, Nd + 1), so j values are 1, 2, 3, ...
    total = 0.0
    @inbounds for j in 1:Nd
        j_sq = j * j  # Use 1-indexed column position directly
        for i in 1:Ng
            total += P[i, j] * (i * i) * j_sq
        end
    end

    return total / Nz
end
