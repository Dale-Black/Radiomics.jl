# Gray Level Size Zone Matrix (GLSZM) for Radiomics.jl
#
# This module implements GLSZM computation and features.
# GLSZM captures texture information by examining connected regions (zones)
# of voxels with the same gray level intensity.
#
# Key difference from GLRLM: GLSZM is rotation-independent - only one matrix
# is computed for the entire ROI (not per-direction).
#
# Total: 16 features
#
# References:
# - PyRadiomics: radiomics/glszm.py
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - Thibault et al. (2009). "Texture Indexes and Gray Level Size Zone Matrix"

#==============================================================================#
# Constants
#==============================================================================#

# Machine epsilon for avoiding log(0) - matches np.spacing(1)
const GLSZM_EPSILON = eps(Float64)  # ≈ 2.2e-16

#==============================================================================#
# GLSZM Result Type
#==============================================================================#

"""
    GLSZMResult

Container for GLSZM computation results.

# Fields
- `matrix::Matrix{Float64}`: GLSZM matrix, shape (Ng, Ns)
- `Ng::Int`: Number of gray levels
- `Ns::Int`: Maximum zone size
- `Nz::Int`: Total number of zones (sum of all matrix entries)
- `Np::Int`: Total number of voxels in zones

# Notes
The matrix is stored as a 2D array where:
- First dimension: gray level (i), indexed 1:Ng
- Second dimension: zone size (j), indexed 1:Ns

P[i, j] is the count of zones with gray level i and size j.

Unlike GLCM and GLRLM, GLSZM is rotation-independent (only one matrix).
"""
struct GLSZMResult
    matrix::Matrix{Float64}
    Ng::Int      # Number of gray levels
    Ns::Int      # Maximum zone size
    Nz::Int      # Total number of zones
    Np::Int      # Total number of voxels in zones
end

"""
    GLSZMResult2D

Container for 2D GLSZM computation results.

# Fields
- `matrix::Matrix{Float64}`: GLSZM matrix, shape (Ng, Ns)
- `Ng::Int`: Number of gray levels
- `Ns::Int`: Maximum zone size
- `Nz::Int`: Total number of zones
- `Np::Int`: Total number of voxels in zones
"""
struct GLSZMResult2D
    matrix::Matrix{Float64}
    Ng::Int
    Ns::Int
    Nz::Int
    Np::Int
end

#==============================================================================#
# Connected Component Labeling - 3D
#==============================================================================#

"""
    _label_components_3d(binary_mask::AbstractArray{Bool, 3};
                         connectivity::Int=26) -> Array{Int, 3}

Label connected components in a 3D binary mask.

Uses iterative flood-fill with a stack to avoid stack overflow on large images.

# Arguments
- `binary_mask`: 3D boolean array
- `connectivity::Int=26`: Connectivity type (26 or 6)
  - 26: All 26 neighbors (including diagonals) - default for PyRadiomics
  - 6: Only face-adjacent neighbors

# Returns
- 3D array of integer labels (0 = background, 1+ = component labels)
"""
function _label_components_3d(binary_mask::AbstractArray{Bool, 3};
                              connectivity::Int=26)

    sz = size(binary_mask)
    labels = zeros(Int, sz)
    current_label = 0

    # Define neighbor offsets based on connectivity
    if connectivity == 26
        # All 26 neighbors (3x3x3 cube minus center)
        offsets = CartesianIndex{3}[]
        for dz in -1:1, dy in -1:1, dx in -1:1
            if !(dz == 0 && dy == 0 && dx == 0)
                push!(offsets, CartesianIndex(dx, dy, dz))
            end
        end
    elseif connectivity == 6
        # Only 6 face-adjacent neighbors
        offsets = [
            CartesianIndex(1, 0, 0), CartesianIndex(-1, 0, 0),
            CartesianIndex(0, 1, 0), CartesianIndex(0, -1, 0),
            CartesianIndex(0, 0, 1), CartesianIndex(0, 0, -1)
        ]
    else
        throw(ArgumentError("connectivity must be 6 or 26, got $connectivity"))
    end

    # Iterative flood-fill using a stack
    stack = CartesianIndex{3}[]

    @inbounds for start_idx in CartesianIndices(binary_mask)
        # Skip if not foreground or already labeled
        binary_mask[start_idx] || continue
        labels[start_idx] != 0 && continue

        # Start a new component
        current_label += 1
        push!(stack, start_idx)

        while !isempty(stack)
            idx = pop!(stack)

            # Skip if already labeled (may have been added multiple times)
            labels[idx] != 0 && continue

            # Label this voxel
            labels[idx] = current_label

            # Add unvisited neighbors to stack
            for offset in offsets
                neighbor = idx + offset
                if checkbounds(Bool, labels, neighbor) &&
                   binary_mask[neighbor] &&
                   labels[neighbor] == 0
                    push!(stack, neighbor)
                end
            end
        end
    end

    return labels
end

"""
    _label_components_2d(binary_mask::AbstractArray{Bool, 2};
                         connectivity::Int=8) -> Array{Int, 2}

Label connected components in a 2D binary mask.

Uses iterative flood-fill with a stack.

# Arguments
- `binary_mask`: 2D boolean array
- `connectivity::Int=8`: Connectivity type (8 or 4)
  - 8: All 8 neighbors (including diagonals) - default for PyRadiomics
  - 4: Only edge-adjacent neighbors

# Returns
- 2D array of integer labels (0 = background, 1+ = component labels)
"""
function _label_components_2d(binary_mask::AbstractArray{Bool, 2};
                              connectivity::Int=8)

    sz = size(binary_mask)
    labels = zeros(Int, sz)
    current_label = 0

    # Define neighbor offsets based on connectivity
    if connectivity == 8
        # All 8 neighbors (3x3 square minus center)
        offsets = CartesianIndex{2}[]
        for dy in -1:1, dx in -1:1
            if !(dy == 0 && dx == 0)
                push!(offsets, CartesianIndex(dx, dy))
            end
        end
    elseif connectivity == 4
        # Only 4 edge-adjacent neighbors
        offsets = [
            CartesianIndex(1, 0), CartesianIndex(-1, 0),
            CartesianIndex(0, 1), CartesianIndex(0, -1)
        ]
    else
        throw(ArgumentError("connectivity must be 4 or 8, got $connectivity"))
    end

    # Iterative flood-fill using a stack
    stack = CartesianIndex{2}[]

    @inbounds for start_idx in CartesianIndices(binary_mask)
        binary_mask[start_idx] || continue
        labels[start_idx] != 0 && continue

        current_label += 1
        push!(stack, start_idx)

        while !isempty(stack)
            idx = pop!(stack)
            labels[idx] != 0 && continue
            labels[idx] = current_label

            for offset in offsets
                neighbor = idx + offset
                if checkbounds(Bool, labels, neighbor) &&
                   binary_mask[neighbor] &&
                   labels[neighbor] == 0
                    push!(stack, neighbor)
                end
            end
        end
    end

    return labels
end

#==============================================================================#
# Zone Detection and Counting
#==============================================================================#

"""
    _compute_zone_sizes(labels::AbstractArray{Int}, max_label::Int) -> Vector{Int}

Compute the size (number of voxels) of each labeled component.

# Arguments
- `labels`: Array of integer labels (0 = background)
- `max_label`: Maximum label value

# Returns
- Vector of zone sizes, where sizes[i] = size of component with label i
"""
function _compute_zone_sizes(labels::AbstractArray{Int}, max_label::Int)
    sizes = zeros(Int, max_label)

    @inbounds for idx in eachindex(labels)
        label = labels[idx]
        if label > 0
            sizes[label] += 1
        end
    end

    return sizes
end

#==============================================================================#
# GLSZM Matrix Computation - 3D
#==============================================================================#

"""
    compute_glszm(image::AbstractArray{<:Integer, 3}, mask::AbstractArray{Bool, 3};
                  Ng::Union{Int, Nothing}=nothing,
                  connectivity::Int=26) -> GLSZMResult

Compute Gray Level Size Zone Matrix for a 3D image.

The GLSZM P(i,j) counts the number of zones (connected components) with
gray level i and size j within the ROI.

# Arguments
- `image`: 3D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 3D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels. If nothing, auto-detected.
- `connectivity::Int=26`: Connectivity for zone detection (26 or 6)

# Returns
- `GLSZMResult`: Struct containing:
  - `matrix`: GLSZM matrix, shape (Ng, Ns)
  - `Ng`: Number of gray levels
  - `Ns`: Maximum zone size
  - `Nz`: Total number of zones
  - `Np`: Total number of voxels

# Notes
- Input image should be discretized (integer gray levels starting from 1)
- All voxels in a zone must be within the mask
- Unlike GLRLM/GLCM, GLSZM is rotation-independent (single matrix)
- Zones are identified using connected component labeling

# Example
```julia
# Discretize image first
discretized = discretize_image(image, mask, binwidth=25.0)
result = compute_glszm(discretized.discretized, mask)

# Access the GLSZM matrix
P = result.matrix
```

# References
- PyRadiomics: radiomics/glszm.py
- IBSI: Section 3.6.5 (Grey level size zone based features)
"""
function compute_glszm(image::AbstractArray{<:Integer, 3},
                       mask::AbstractArray{Bool, 3};
                       Ng::Union{Int, Nothing}=nothing,
                       connectivity::Int=26)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))

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

    # Maximum possible zone size is the number of voxels in the mask
    max_possible_zone_size = sum(mask)

    # Initialize GLSZM matrix
    # We'll use a sparse-like approach: track actual max zone size found
    zone_counts = Dict{Tuple{Int, Int}, Int}()  # (gray_level, zone_size) -> count

    total_zones = 0
    total_voxels = 0
    max_zone_size = 0

    # Process each gray level
    for gray_level in 1:Ng
        # Create binary mask for this gray level
        gray_mask = (image .== gray_level) .& mask

        if !any(gray_mask)
            continue
        end

        # Label connected components
        labels = _label_components_3d(gray_mask; connectivity=connectivity)

        # Count components
        max_label = maximum(labels)
        if max_label == 0
            continue
        end

        # Compute zone sizes
        zone_sizes = _compute_zone_sizes(labels, max_label)

        # Record zones in the GLSZM
        for zone_size in zone_sizes
            if zone_size > 0
                key = (gray_level, zone_size)
                zone_counts[key] = get(zone_counts, key, 0) + 1
                total_zones += 1
                total_voxels += zone_size
                max_zone_size = max(max_zone_size, zone_size)
            end
        end
    end

    # Handle empty case
    if total_zones == 0
        return GLSZMResult(zeros(Float64, Ng, 1), Ng, 1, 0, 0)
    end

    # Build the dense GLSZM matrix
    Ns = max_zone_size
    P = zeros(Float64, Ng, Ns)

    for ((gray_level, zone_size), count) in zone_counts
        P[gray_level, zone_size] = Float64(count)
    end

    return GLSZMResult(P, Ng, Ns, total_zones, total_voxels)
end

#==============================================================================#
# GLSZM Matrix Computation - 2D
#==============================================================================#

"""
    compute_glszm_2d(image::AbstractArray{<:Integer, 2}, mask::AbstractArray{Bool, 2};
                     Ng::Union{Int, Nothing}=nothing,
                     connectivity::Int=8) -> GLSZMResult2D

Compute Gray Level Size Zone Matrix for a 2D image.

# Arguments
- `image`: 2D array of discretized gray levels (integers, typically 1:Ng)
- `mask`: 2D boolean mask defining the ROI
- `Ng::Union{Int, Nothing}=nothing`: Number of gray levels (auto-detected if nothing)
- `connectivity::Int=8`: Connectivity for zone detection (8 or 4)

# Returns
- `GLSZMResult2D`: Struct containing GLSZM matrix (shape: Ng × Ns)

# Example
```julia
image_2d = discretize(image_slice, edges)
mask_2d = mask[:, :, 32]
result = compute_glszm_2d(image_2d, mask_2d)
```
"""
function compute_glszm_2d(image::AbstractArray{<:Integer, 2},
                          mask::AbstractArray{Bool, 2};
                          Ng::Union{Int, Nothing}=nothing,
                          connectivity::Int=8)

    # Input validation
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions"
    ))

    # Get masked voxels
    masked_values = image[mask]
    if isempty(masked_values)
        throw(ArgumentError("Mask is empty"))
    end

    # Determine number of gray levels
    if isnothing(Ng)
        Ng = maximum(masked_values)
    end

    # Track zones
    zone_counts = Dict{Tuple{Int, Int}, Int}()
    total_zones = 0
    total_voxels = 0
    max_zone_size = 0

    # Process each gray level
    for gray_level in 1:Ng
        gray_mask = (image .== gray_level) .& mask

        if !any(gray_mask)
            continue
        end

        labels = _label_components_2d(gray_mask; connectivity=connectivity)
        max_label = maximum(labels)

        if max_label == 0
            continue
        end

        zone_sizes = _compute_zone_sizes(labels, max_label)

        for zone_size in zone_sizes
            if zone_size > 0
                key = (gray_level, zone_size)
                zone_counts[key] = get(zone_counts, key, 0) + 1
                total_zones += 1
                total_voxels += zone_size
                max_zone_size = max(max_zone_size, zone_size)
            end
        end
    end

    # Handle empty case
    if total_zones == 0
        return GLSZMResult2D(zeros(Float64, Ng, 1), Ng, 1, 0, 0)
    end

    # Build dense matrix
    Ns = max_zone_size
    P = zeros(Float64, Ng, Ns)

    for ((gray_level, zone_size), count) in zone_counts
        P[gray_level, zone_size] = Float64(count)
    end

    return GLSZMResult2D(P, Ng, Ns, total_zones, total_voxels)
end

#==============================================================================#
# High-Level GLSZM Computation Interface
#==============================================================================#

"""
    compute_glszm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                  binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                  connectivity::Int=26) -> GLSZMResult

Compute GLSZM from a non-discretized image (convenience wrapper).

This function discretizes the image before computing the GLSZM.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `connectivity::Int`: 26 for 3D, 8 for 2D (default based on dimensions)

# Returns
- For 3D images: `GLSZMResult`
- For 2D images: `GLSZMResult2D`

# Example
```julia
result = compute_glszm(image, mask, binwidth=25.0)
```
"""
function compute_glszm(image::AbstractArray{<:Real, 3},
                       mask::AbstractArray{Bool, 3};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing,
                       connectivity::Int=26)

    # Discretize the image
    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    # Compute GLSZM on discretized image
    return compute_glszm(disc_result.discretized, mask; connectivity=connectivity)
end

function compute_glszm(image::AbstractArray{<:Real, 2},
                       mask::AbstractArray{Bool, 2};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing,
                       connectivity::Int=8)

    disc_result = discretize_image(image, mask; binwidth=binwidth, bincount=bincount)

    return compute_glszm_2d(disc_result.discretized, mask; connectivity=connectivity)
end

#==============================================================================#
# GLSZM with Settings
#==============================================================================#

"""
    compute_glszm(image, mask, settings::Settings; connectivity::Int)

Compute GLSZM using parameters from a Settings object.

# Example
```julia
settings = Settings(binwidth=32.0)
result = compute_glszm(image, mask, settings)
```
"""
function compute_glszm(image::AbstractArray{<:Real},
                       mask::AbstractArray{Bool},
                       settings::Settings;
                       connectivity::Int = ndims(image) == 3 ? 26 : 8)

    bincount = settings.discretization_mode == FixedBinCount ? settings.bincount : nothing

    return compute_glszm(image, mask;
                         binwidth=settings.binwidth,
                         bincount=bincount,
                         connectivity=connectivity)
end

#==============================================================================#
# GLSZM Utility Functions
#==============================================================================#

"""
    glszm_num_gray_levels(result::Union{GLSZMResult, GLSZMResult2D}) -> Int

Get the number of gray levels in the GLSZM.
"""
glszm_num_gray_levels(result::GLSZMResult) = result.Ng
glszm_num_gray_levels(result::GLSZMResult2D) = result.Ng

"""
    glszm_max_zone_size(result::Union{GLSZMResult, GLSZMResult2D}) -> Int

Get the maximum zone size in the GLSZM.
"""
glszm_max_zone_size(result::GLSZMResult) = result.Ns
glszm_max_zone_size(result::GLSZMResult2D) = result.Ns

"""
    glszm_num_zones(result::Union{GLSZMResult, GLSZMResult2D}) -> Int

Get the total number of zones (Nz).
"""
glszm_num_zones(result::GLSZMResult) = result.Nz
glszm_num_zones(result::GLSZMResult2D) = result.Nz

"""
    glszm_num_voxels(result::Union{GLSZMResult, GLSZMResult2D}) -> Int

Get the total number of voxels represented in zones (Np).
"""
glszm_num_voxels(result::GLSZMResult) = result.Np
glszm_num_voxels(result::GLSZMResult2D) = result.Np

#==============================================================================#
# GLSZM Marginal Distributions
#==============================================================================#

"""
    _glszm_marginals(P::AbstractMatrix{Float64}, Nz::Int, Np::Int) -> NamedTuple

Compute marginal distributions and other values needed for GLSZM features.

# Arguments
- `P`: GLSZM matrix (Ng × Ns), raw counts (not normalized)
- `Nz`: Total number of zones
- `Np`: Total number of voxels in zones

# Returns
NamedTuple with fields:
- `pg`: Gray level marginal, pg(i) = Σⱼ P(i,j)
- `ps`: Zone size marginal, ps(j) = Σᵢ P(i,j)
- `Nz`: Total number of zones
- `Np`: Total number of voxels
- `Ng`: Number of gray levels
- `Ns`: Maximum zone size
"""
function _glszm_marginals(P::AbstractMatrix{Float64}, Nz::Int, Np::Int)
    Ng, Ns = size(P)

    # Gray level marginal: pg(i) = Σⱼ P(i,j)
    pg = vec(sum(P, dims=2))

    # Zone size marginal: ps(j) = Σᵢ P(i,j)
    ps = vec(sum(P, dims=1))

    return (pg=pg, ps=ps, Nz=Nz, Np=Np, Ng=Ng, Ns=Ns)
end

#==============================================================================#
# GLSZM Features - Zone Size Emphasis Features
#==============================================================================#

"""
    glszm_small_area_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Small Area Emphasis (SAE): Measures the distribution of small zones.

# Mathematical Formula
```
SAE = (1/Nz) × Σᵢ Σⱼ P(i,j) / j²
```

Higher values indicate finer textures with more small zones.

# References
- PyRadiomics: glszm.py:getSmallAreaEmphasisFeatureValue
- IBSI: Grey level size zone based features - Small zone emphasis
"""
function glszm_small_area_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            total += P[i, j] / j_sq
        end
    end

    return total / Nz
end

"""
    glszm_large_area_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Large Area Emphasis (LAE): Measures the distribution of large zones.

# Mathematical Formula
```
LAE = (1/Nz) × Σᵢ Σⱼ P(i,j) × j²
```

Higher values indicate coarser textures with more large zones.

# References
- PyRadiomics: glszm.py:getLargeAreaEmphasisFeatureValue
- IBSI: Grey level size zone based features - Large zone emphasis
"""
function glszm_large_area_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            total += P[i, j] * j_sq
        end
    end

    return total / Nz
end

#==============================================================================#
# GLSZM Features - Non-Uniformity Features
#==============================================================================#

"""
    glszm_gray_level_non_uniformity(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Gray Level Non-Uniformity (GLN): Measures variability of gray level intensity.

# Mathematical Formula
```
GLN = (1/Nz) × Σᵢ (Σⱼ P(i,j))²
```

Lower values indicate more homogeneous gray level distribution.

# References
- PyRadiomics: glszm.py:getGrayLevelNonUniformityFeatureValue
- IBSI: Grey level size zone based features - Grey level non-uniformity
"""
function glszm_gray_level_non_uniformity(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # pg(i) = Σⱼ P(i,j)
    total = 0.0
    @inbounds for i in 1:Ng
        pg_i = 0.0
        for j in 1:Ns
            pg_i += P[i, j]
        end
        total += pg_i * pg_i
    end

    return total / Nz
end

"""
    glszm_gray_level_non_uniformity_normalized(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Gray Level Non-Uniformity Normalized (GLNN): Normalized measure of gray level variability.

# Mathematical Formula
```
GLNN = (1/Nz²) × Σᵢ (Σⱼ P(i,j))²
```

Similar to GLN but normalized by Nz², making it more comparable across different images.

# References
- PyRadiomics: glszm.py:getGrayLevelNonUniformityNormalizedFeatureValue
- IBSI: Grey level size zone based features - Normalised grey level non-uniformity
"""
function glszm_gray_level_non_uniformity_normalized(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # pg(i) = Σⱼ P(i,j)
    total = 0.0
    @inbounds for i in 1:Ng
        pg_i = 0.0
        for j in 1:Ns
            pg_i += P[i, j]
        end
        total += pg_i * pg_i
    end

    return total / (Nz * Nz)
end

"""
    glszm_size_zone_non_uniformity(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Size Zone Non-Uniformity (SZN): Measures variability of zone sizes.

# Mathematical Formula
```
SZN = (1/Nz) × Σⱼ (Σᵢ P(i,j))²
```

Lower values indicate more homogeneous zone size distribution.

# References
- PyRadiomics: glszm.py:getSizeZoneNonUniformityFeatureValue
- IBSI: Grey level size zone based features - Zone size non-uniformity
"""
function glszm_size_zone_non_uniformity(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # ps(j) = Σᵢ P(i,j)
    total = 0.0
    @inbounds for j in 1:Ns
        ps_j = 0.0
        for i in 1:Ng
            ps_j += P[i, j]
        end
        total += ps_j * ps_j
    end

    return total / Nz
end

"""
    glszm_size_zone_non_uniformity_normalized(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Size Zone Non-Uniformity Normalized (SZNN): Normalized measure of zone size variability.

# Mathematical Formula
```
SZNN = (1/Nz²) × Σⱼ (Σᵢ P(i,j))²
```

Similar to SZN but normalized by Nz².

# References
- PyRadiomics: glszm.py:getSizeZoneNonUniformityNormalizedFeatureValue
- IBSI: Grey level size zone based features - Normalised zone size non-uniformity
"""
function glszm_size_zone_non_uniformity_normalized(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # ps(j) = Σᵢ P(i,j)
    total = 0.0
    @inbounds for j in 1:Ns
        ps_j = 0.0
        for i in 1:Ng
            ps_j += P[i, j]
        end
        total += ps_j * ps_j
    end

    return total / (Nz * Nz)
end

#==============================================================================#
# GLSZM Features - Zone Statistics Features
#==============================================================================#

"""
    glszm_zone_percentage(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Zone Percentage (ZP): Ratio of zones to voxels.

# Mathematical Formula
```
ZP = Nz / Np
```

where:
- Nz is the total number of zones
- Np is the total number of voxels in the ROI

Higher values indicate finer textures (more zones for same number of voxels).

# References
- PyRadiomics: glszm.py:getZonePercentageFeatureValue
- IBSI: Grey level size zone based features - Zone percentage
"""
function glszm_zone_percentage(result::Union{GLSZMResult, GLSZMResult2D})
    Nz = result.Nz
    Np = result.Np

    if Np == 0
        return NaN
    end

    return Float64(Nz) / Float64(Np)
end

"""
    glszm_gray_level_variance(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Gray Level Variance (GLV): Measures variance in gray level intensities.

# Mathematical Formula
```
GLV = Σᵢ Σⱼ p(i,j) × (i - μᵢ)²
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLSZM)
- μᵢ = Σᵢ Σⱼ p(i,j) × i (mean gray level)

# References
- PyRadiomics: glszm.py:getGrayLevelVarianceFeatureValue
- IBSI: Grey level size zone based features - Grey level variance
"""
function glszm_gray_level_variance(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # Compute mean gray level μᵢ = Σᵢ Σⱼ p(i,j) × i
    mu_i = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            mu_i += (P[i, j] / Nz) * i
        end
    end

    # Compute variance
    variance = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            variance += p_ij * (i - mu_i)^2
        end
    end

    return variance
end

"""
    glszm_zone_variance(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Zone Variance (ZV): Measures variance in zone sizes.

# Mathematical Formula
```
ZV = Σᵢ Σⱼ p(i,j) × (j - μⱼ)²
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLSZM)
- μⱼ = Σᵢ Σⱼ p(i,j) × j (mean zone size)

# References
- PyRadiomics: glszm.py:getZoneVarianceFeatureValue
- IBSI: Grey level size zone based features - Zone size variance
"""
function glszm_zone_variance(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    # Compute mean zone size μⱼ = Σᵢ Σⱼ p(i,j) × j
    mu_j = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            mu_j += (P[i, j] / Nz) * j
        end
    end

    # Compute variance
    variance = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            variance += p_ij * (j - mu_j)^2
        end
    end

    return variance
end

"""
    glszm_zone_entropy(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Zone Entropy (ZE): Measures the randomness/heterogeneity of zone distribution.

# Mathematical Formula
```
ZE = -Σᵢ Σⱼ p(i,j) × log₂(p(i,j) + ε)
```

where:
- p(i,j) = P(i,j) / Nz (normalized GLSZM)
- ε ≈ 2.2×10⁻¹⁶ (machine epsilon to prevent log(0))

Higher values indicate more heterogeneous distributions.

# References
- PyRadiomics: glszm.py:getZoneEntropyFeatureValue
- IBSI: Grey level size zone based features - Zone size entropy
"""
function glszm_zone_entropy(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    entropy = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            p_ij = P[i, j] / Nz
            if p_ij > 0
                entropy -= p_ij * log2(p_ij + GLSZM_EPSILON)
            end
        end
    end

    return entropy
end

#==============================================================================#
# GLSZM Features - Gray Level Emphasis Features
#==============================================================================#

"""
    glszm_low_gray_level_zone_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Low Gray Level Zone Emphasis (LGLZE): Measures the distribution of lower gray levels.

# Mathematical Formula
```
LGLZE = (1/Nz) × Σᵢ Σⱼ P(i,j) / i²
```

Higher values indicate a greater proportion of lower gray level values.

# References
- PyRadiomics: glszm.py:getLowGrayLevelZoneEmphasisFeatureValue
- IBSI: Grey level size zone based features - Low grey level zone emphasis
"""
function glszm_low_gray_level_zone_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            total += P[i, j] / (i * i)
        end
    end

    return total / Nz
end

"""
    glszm_high_gray_level_zone_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

High Gray Level Zone Emphasis (HGLZE): Measures the distribution of higher gray levels.

# Mathematical Formula
```
HGLZE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i²
```

Higher values indicate a greater proportion of higher gray level values.

# References
- PyRadiomics: glszm.py:getHighGrayLevelZoneEmphasisFeatureValue
- IBSI: Grey level size zone based features - High grey level zone emphasis
"""
function glszm_high_gray_level_zone_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        for i in 1:Ng
            total += P[i, j] * (i * i)
        end
    end

    return total / Nz
end

#==============================================================================#
# GLSZM Features - Joint Emphasis Features
#==============================================================================#

"""
    glszm_small_area_low_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Small Area Low Gray Level Emphasis (SALGLE): Measures the proportion of small zones
with low gray levels.

# Mathematical Formula
```
SALGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) / (i² × j²)
```

Higher values indicate more small zones with low intensities.

# References
- PyRadiomics: glszm.py:getSmallAreaLowGrayLevelEmphasisFeatureValue
- IBSI: Grey level size zone based features - Small zone low grey level emphasis
"""
function glszm_small_area_low_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            i_sq = i * i
            total += P[i, j] / (i_sq * j_sq)
        end
    end

    return total / Nz
end

"""
    glszm_small_area_high_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Small Area High Gray Level Emphasis (SAHGLE): Measures the proportion of small zones
with high gray levels.

# Mathematical Formula
```
SAHGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i² / j²
```

Higher values indicate more small zones with high intensities.

# References
- PyRadiomics: glszm.py:getSmallAreaHighGrayLevelEmphasisFeatureValue
- IBSI: Grey level size zone based features - Small zone high grey level emphasis
"""
function glszm_small_area_high_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            i_sq = i * i
            total += P[i, j] * i_sq / j_sq
        end
    end

    return total / Nz
end

"""
    glszm_large_area_low_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Large Area Low Gray Level Emphasis (LALGLE): Measures the proportion of large zones
with low gray levels.

# Mathematical Formula
```
LALGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × j² / i²
```

Higher values indicate more large zones with low intensities.

# References
- PyRadiomics: glszm.py:getLargeAreaLowGrayLevelEmphasisFeatureValue
- IBSI: Grey level size zone based features - Large zone low grey level emphasis
"""
function glszm_large_area_low_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            i_sq = i * i
            total += P[i, j] * j_sq / i_sq
        end
    end

    return total / Nz
end

"""
    glszm_large_area_high_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D}) -> Float64

Large Area High Gray Level Emphasis (LAHGLE): Measures the proportion of large zones
with high gray levels.

# Mathematical Formula
```
LAHGLE = (1/Nz) × Σᵢ Σⱼ P(i,j) × i² × j²
```

Higher values indicate more large zones with high intensities.

# References
- PyRadiomics: glszm.py:getLargeAreaHighGrayLevelEmphasisFeatureValue
- IBSI: Grey level size zone based features - Large zone high grey level emphasis
"""
function glszm_large_area_high_gray_level_emphasis(result::Union{GLSZMResult, GLSZMResult2D})
    P = result.matrix
    Nz = result.Nz
    Ng, Ns = size(P)

    if Nz == 0
        return NaN
    end

    total = 0.0
    @inbounds for j in 1:Ns
        j_sq = j * j
        for i in 1:Ng
            i_sq = i * i
            total += P[i, j] * i_sq * j_sq
        end
    end

    return total / Nz
end

#==============================================================================#
# GLSZM Feature Extraction
#==============================================================================#

"""
    glszm_feature_names() -> Vector{String}

Get the list of all GLSZM feature names in standard order.
"""
function glszm_feature_names()
    return [
        "SmallAreaEmphasis",
        "LargeAreaEmphasis",
        "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized",
        "SizeZoneNonUniformity",
        "SizeZoneNonUniformityNormalized",
        "ZonePercentage",
        "GrayLevelVariance",
        "ZoneVariance",
        "ZoneEntropy",
        "LowGrayLevelZoneEmphasis",
        "HighGrayLevelZoneEmphasis",
        "SmallAreaLowGrayLevelEmphasis",
        "SmallAreaHighGrayLevelEmphasis",
        "LargeAreaLowGrayLevelEmphasis",
        "LargeAreaHighGrayLevelEmphasis"
    ]
end

"""
    glszm_ibsi_features() -> Vector{String}

Get the list of IBSI-compliant GLSZM feature names.
All 16 GLSZM features are IBSI compliant.
"""
function glszm_ibsi_features()
    return glszm_feature_names()
end

"""
    extract_glszm(result::Union{GLSZMResult, GLSZMResult2D}) -> Dict{String, Float64}

Extract all 16 GLSZM features from a computed GLSZM result.

# Arguments
- `result`: GLSZM computation result from compute_glszm()

# Returns
Dict mapping feature names to values. All 16 GLSZM features:
- SmallAreaEmphasis
- LargeAreaEmphasis
- GrayLevelNonUniformity
- GrayLevelNonUniformityNormalized
- SizeZoneNonUniformity
- SizeZoneNonUniformityNormalized
- ZonePercentage
- GrayLevelVariance
- ZoneVariance
- ZoneEntropy
- LowGrayLevelZoneEmphasis
- HighGrayLevelZoneEmphasis
- SmallAreaLowGrayLevelEmphasis
- SmallAreaHighGrayLevelEmphasis
- LargeAreaLowGrayLevelEmphasis
- LargeAreaHighGrayLevelEmphasis

# Example
```julia
result = compute_glszm(image, mask, binwidth=25.0)
features = extract_glszm(result)
println(features["SmallAreaEmphasis"])
```

# References
- PyRadiomics: glszm.py
- IBSI: Grey level size zone based features
"""
function extract_glszm(result::Union{GLSZMResult, GLSZMResult2D})
    return Dict{String, Float64}(
        "SmallAreaEmphasis" => glszm_small_area_emphasis(result),
        "LargeAreaEmphasis" => glszm_large_area_emphasis(result),
        "GrayLevelNonUniformity" => glszm_gray_level_non_uniformity(result),
        "GrayLevelNonUniformityNormalized" => glszm_gray_level_non_uniformity_normalized(result),
        "SizeZoneNonUniformity" => glszm_size_zone_non_uniformity(result),
        "SizeZoneNonUniformityNormalized" => glszm_size_zone_non_uniformity_normalized(result),
        "ZonePercentage" => glszm_zone_percentage(result),
        "GrayLevelVariance" => glszm_gray_level_variance(result),
        "ZoneVariance" => glszm_zone_variance(result),
        "ZoneEntropy" => glszm_zone_entropy(result),
        "LowGrayLevelZoneEmphasis" => glszm_low_gray_level_zone_emphasis(result),
        "HighGrayLevelZoneEmphasis" => glszm_high_gray_level_zone_emphasis(result),
        "SmallAreaLowGrayLevelEmphasis" => glszm_small_area_low_gray_level_emphasis(result),
        "SmallAreaHighGrayLevelEmphasis" => glszm_small_area_high_gray_level_emphasis(result),
        "LargeAreaLowGrayLevelEmphasis" => glszm_large_area_low_gray_level_emphasis(result),
        "LargeAreaHighGrayLevelEmphasis" => glszm_large_area_high_gray_level_emphasis(result)
    )
end

"""
    extract_glszm(image::AbstractArray{<:Real}, mask::AbstractArray{Bool};
                  binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                  connectivity::Int=26) -> Dict{String, Float64}

Extract all GLSZM features from an image and mask.

This is a convenience function that computes the GLSZM and extracts all features.

# Arguments
- `image`: Image array (will be discretized)
- `mask`: Boolean mask for ROI
- `binwidth::Real=25.0`: Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing`: Bin count (overrides binwidth if specified)
- `connectivity::Int=26`: Connectivity for zone detection (26 or 6 for 3D, 8 or 4 for 2D)

# Returns
Dict mapping feature names to values.

# Example
```julia
features = extract_glszm(image, mask, binwidth=25.0)
```
"""
function extract_glszm(image::AbstractArray{<:Real, 3}, mask::AbstractArray{Bool, 3};
                       binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                       connectivity::Int=26)
    result = compute_glszm(image, mask; binwidth=binwidth, bincount=bincount,
                           connectivity=connectivity)
    return extract_glszm(result)
end

function extract_glszm(image::AbstractArray{<:Real, 2}, mask::AbstractArray{Bool, 2};
                       binwidth::Real=25.0, bincount::Union{Int, Nothing}=nothing,
                       connectivity::Int=8)
    result = compute_glszm(image, mask; binwidth=binwidth, bincount=bincount,
                           connectivity=connectivity)
    return extract_glszm(result)
end

"""
    extract_glszm_to_featureset!(fs::FeatureSet, result::Union{GLSZMResult, GLSZMResult2D};
                                  prefix::String="glszm_")

Extract all GLSZM features and add them to an existing FeatureSet.

# Arguments
- `fs`: FeatureSet to add features to
- `result`: GLSZM computation result
- `prefix::String="glszm_"`: Prefix for feature names

# Example
```julia
fs = FeatureSet()
result = compute_glszm(image, mask)
extract_glszm_to_featureset!(fs, result)
```
"""
function extract_glszm_to_featureset!(fs::FeatureSet, result::Union{GLSZMResult, GLSZMResult2D};
                                       prefix::String="glszm_")
    features = extract_glszm(result)
    for (name, value) in features
        key = Symbol(prefix * name)
        push!(fs.features, FeatureResult(key, value, "glszm"))
    end
    return fs
end

#==============================================================================#
# Exports
#==============================================================================#

# Export GLSZM computation functions
export compute_glszm, compute_glszm_2d
export GLSZMResult, GLSZMResult2D

# Export utility functions
export glszm_num_gray_levels, glszm_max_zone_size
export glszm_num_zones, glszm_num_voxels

# Export GLSZM feature functions
export glszm_small_area_emphasis, glszm_large_area_emphasis
export glszm_gray_level_non_uniformity, glszm_gray_level_non_uniformity_normalized
export glszm_size_zone_non_uniformity, glszm_size_zone_non_uniformity_normalized
export glszm_zone_percentage
export glszm_gray_level_variance, glszm_zone_variance, glszm_zone_entropy
export glszm_low_gray_level_zone_emphasis, glszm_high_gray_level_zone_emphasis
export glszm_small_area_low_gray_level_emphasis, glszm_small_area_high_gray_level_emphasis
export glszm_large_area_low_gray_level_emphasis, glszm_large_area_high_gray_level_emphasis

# Export extraction functions
export extract_glszm, extract_glszm_to_featureset!
export glszm_feature_names, glszm_ibsi_features
