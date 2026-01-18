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
