# Image handling utilities for Radiomics.jl
# This module provides functions for extracting voxels from images and masks,
# image normalization, and preprocessing operations.

#==============================================================================#
# Voxel Extraction
#==============================================================================#

"""
    get_voxels(image, mask; label::Int=1) -> Vector{Float64}

Extract voxel intensities from an image within a masked region.

This is the primary function for extracting the intensity values needed for
first-order and texture feature computation. Only voxels where the mask equals
the specified label are included.

# Arguments
- `image`: The image data (Array, RadiomicsImage, or any AbstractArray)
- `mask`: The segmentation mask (Array, RadiomicsMask, BitArray, or any AbstractArray)
- `label::Int=1`: The label value in the mask defining the ROI (ignored for boolean masks)

# Returns
- `Vector{Float64}`: A vector of voxel intensity values within the ROI

# Throws
- `DimensionMismatch`: If image and mask have different dimensions
- `ArgumentError`: If no voxels are found in the mask with the specified label

# Example
```julia
# With plain arrays
image = rand(64, 64, 64)
mask = rand(64, 64, 64) .> 0.7
voxels = get_voxels(image, mask)

# With RadiomicsImage and RadiomicsMask
img = RadiomicsImage(image, (1.0, 1.0, 2.5))
m = RadiomicsMask(Int.(mask))
voxels = get_voxels(img, m)

# Multi-label mask
mask_multi = zeros(Int, 64, 64, 64)
mask_multi[20:40, 20:40, 20:40] .= 1
mask_multi[10:20, 10:20, 10:20] .= 2
voxels_label2 = get_voxels(image, mask_multi, label=2)
```

# Notes
- NaN values in the image are preserved in the output
- For texture features, the output should be discretized before matrix computation
- PyRadiomics reference: `radiomics/base.py` - `self.targetVoxelArray`

# See also
- [`get_voxels_with_coords`](@ref): Also returns voxel coordinates
- [`get_roi_mask`](@ref): Get boolean mask from any mask type
"""
function get_voxels(image::AbstractArray{T, N}, mask::AbstractArray{Bool, N};
                    label::Int=1) where {T<:Real, N}
    # Validate dimensions match
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask dimensions must match: got $(size(image)) and $(size(mask))"
    ))

    # Extract voxels where mask is true
    voxels = Float64.(image[mask])

    isempty(voxels) && throw(ArgumentError(
        "No voxels found in mask. Ensure mask contains true values."
    ))

    return voxels
end

# Handle integer masks with label
function get_voxels(image::AbstractArray{T, N}, mask::AbstractArray{<:Integer, N};
                    label::Int=1) where {T<:Real, N}
    roi_mask = mask .== label
    return get_voxels(image, roi_mask; label=label)
end

# Handle RadiomicsImage
function get_voxels(image::RadiomicsImage{T, N}, mask::AbstractArray;
                    label::Int=1) where {T, N}
    return get_voxels(image.data, mask; label=label)
end

# Handle RadiomicsMask
function get_voxels(image::AbstractArray, mask::RadiomicsMask{N};
                    label::Int=mask.label) where N
    roi_mask = get_roi_mask(mask)
    return get_voxels(image, roi_mask; label=label)
end

# Handle both RadiomicsImage and RadiomicsMask
function get_voxels(image::RadiomicsImage{T, N}, mask::RadiomicsMask{N};
                    label::Int=mask.label) where {T, N}
    return get_voxels(image.data, get_roi_mask(mask); label=label)
end

# Handle BitArray explicitly
function get_voxels(image::AbstractArray{T, N}, mask::BitArray{N};
                    label::Int=1) where {T<:Real, N}
    return get_voxels(image, convert(Array{Bool, N}, mask); label=label)
end

"""
    get_voxels_with_coords(image, mask; label::Int=1) -> Tuple{Vector{Float64}, Vector{CartesianIndex}}

Extract voxel intensities and their coordinates from an image within a masked region.

Returns both the intensity values and their corresponding CartesianIndex coordinates.
This is useful for texture matrix computation where spatial relationships matter.

# Arguments
- `image`: The image data
- `mask`: The segmentation mask
- `label::Int=1`: The label value defining the ROI

# Returns
- `Tuple{Vector{Float64}, Vector{CartesianIndex}}`: Voxel values and their coordinates

# Example
```julia
image = rand(32, 32, 32)
mask = rand(32, 32, 32) .> 0.8
voxels, coords = get_voxels_with_coords(image, mask)

# Access coordinates
for (val, coord) in zip(voxels, coords)
    println("Value \$val at position \$coord")
end
```

# See also
- [`get_voxels`](@ref): Extract only voxel values
"""
function get_voxels_with_coords(image::AbstractArray{T, N}, mask::AbstractArray{Bool, N};
                                 label::Int=1) where {T<:Real, N}
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask dimensions must match: got $(size(image)) and $(size(mask))"
    ))

    # Find indices where mask is true
    coords = findall(mask)

    isempty(coords) && throw(ArgumentError(
        "No voxels found in mask. Ensure mask contains true values."
    ))

    # Extract voxel values at those coordinates
    voxels = Float64[image[c] for c in coords]

    return voxels, coords
end

function get_voxels_with_coords(image::AbstractArray{T, N}, mask::AbstractArray{<:Integer, N};
                                 label::Int=1) where {T<:Real, N}
    roi_mask = mask .== label
    return get_voxels_with_coords(image, roi_mask; label=label)
end

function get_voxels_with_coords(image::RadiomicsImage, mask; label::Int=1)
    return get_voxels_with_coords(image.data, mask; label=label)
end

function get_voxels_with_coords(image::AbstractArray, mask::RadiomicsMask;
                                 label::Int=mask.label)
    return get_voxels_with_coords(image, get_roi_mask(mask); label=label)
end

#==============================================================================#
# Voxel Count and ROI Statistics
#==============================================================================#

"""
    count_voxels(mask; label::Int=1) -> Int

Count the number of voxels in the ROI.

# Arguments
- `mask`: The segmentation mask
- `label::Int=1`: The label value defining the ROI

# Returns
- `Int`: Number of voxels in the ROI

# Example
```julia
mask = rand(64, 64, 64) .> 0.9
n = count_voxels(mask)  # Number of true values
```
"""
function count_voxels(mask::AbstractArray{Bool}; label::Int=1)
    return count(mask)
end

function count_voxels(mask::AbstractArray{<:Integer}; label::Int=1)
    return count(==(label), mask)
end

function count_voxels(mask::RadiomicsMask; label::Int=mask.label)
    return count(==(label), mask.data)
end

"""
    voxel_volume(image::RadiomicsImage) -> Float64
    voxel_volume(spacing::NTuple{N, Float64}) -> Float64

Calculate the volume of a single voxel in mm³.

# Arguments
- `image::RadiomicsImage`: Image with spacing information
- `spacing::NTuple`: Voxel spacing tuple (dx, dy, dz) in mm

# Returns
- `Float64`: Volume in mm³

# Example
```julia
img = RadiomicsImage(rand(64, 64, 64), (1.0, 1.0, 2.5))
vol = voxel_volume(img)  # 2.5 mm³
```
"""
function voxel_volume(spacing::NTuple{N, Float64}) where N
    return prod(spacing)
end

function voxel_volume(image::RadiomicsImage)
    return voxel_volume(image.spacing)
end

function voxel_volume(image::AbstractArray{T, N}) where {T, N}
    # Default unit spacing
    return 1.0
end

"""
    roi_volume(image, mask; label::Int=1) -> Float64

Calculate the total volume of the ROI in mm³.

# Arguments
- `image`: Image with spacing information (or plain array for unit volume)
- `mask`: Segmentation mask
- `label::Int=1`: Label value defining the ROI

# Returns
- `Float64`: Total ROI volume in mm³

# Example
```julia
img = RadiomicsImage(rand(64, 64, 64), (1.0, 1.0, 2.0))
mask = img .> 0.5
vol = roi_volume(img, mask)  # Count * 2.0 mm³
```
"""
function roi_volume(image, mask; label::Int=1)
    nvoxels = count_voxels(mask; label=label)
    vvol = voxel_volume(image)
    return Float64(nvoxels) * vvol
end

function roi_volume(image, mask::RadiomicsMask)
    return roi_volume(image, mask; label=mask.label)
end

#==============================================================================#
# Image Normalization
#==============================================================================#

"""
    normalize_image(image, mask; scale::Float64=1.0, remove_outliers::Bool=false,
                    outlier_percentile::Float64=99.0, label::Int=1) -> Array

Normalize image intensities within the ROI using z-score normalization.

The normalization formula is: `f(x) = scale * (x - μ) / σ`

where μ and σ are computed from voxels within the mask.

# Arguments
- `image`: The input image
- `mask`: The segmentation mask
- `scale::Float64=1.0`: Scale factor applied after normalization
- `remove_outliers::Bool=false`: Whether to clip values outside the percentile range
- `outlier_percentile::Float64=99.0`: Percentile for outlier removal (clips to [100-p, p])
- `label::Int=1`: Label value in mask defining the ROI

# Returns
- `Array{Float64}`: Normalized image with same dimensions as input

# Example
```julia
image = rand(64, 64, 64) .* 1000  # High intensity range
mask = rand(64, 64, 64) .> 0.7
normalized = normalize_image(image, mask)
# Voxels within mask now have mean ≈ 0, std ≈ 1
```

# Notes
- Only voxels within the mask are used to compute mean and std
- All voxels in the image are normalized (not just masked region)
- PyRadiomics reference: `radiomics/imageoperations.py:normalizeImage()`
"""
function normalize_image(image::AbstractArray{T, N}, mask;
                         scale::Float64=1.0,
                         remove_outliers::Bool=false,
                         outlier_percentile::Float64=99.0,
                         label::Int=1) where {T, N}
    # Get voxels within mask for computing statistics
    roi_mask = _get_bool_mask(mask, label)
    voxels = Float64.(image[roi_mask])

    isempty(voxels) && throw(ArgumentError(
        "No voxels found in mask for normalization."
    ))

    # Compute mean and std from ROI
    μ = mean(voxels)
    σ = std(voxels)

    # Handle case of constant intensity (avoid division by zero)
    if σ < eps(Float64)
        σ = 1.0
    end

    # Normalize entire image
    normalized = scale .* (Float64.(image) .- μ) ./ σ

    # Optional outlier removal
    if remove_outliers
        low_p = 100.0 - outlier_percentile
        high_p = outlier_percentile

        # Compute percentiles from normalized ROI values
        norm_roi = normalized[roi_mask]
        low_val = quantile(norm_roi, low_p / 100.0)
        high_val = quantile(norm_roi, high_p / 100.0)

        # Clip values
        normalized = clamp.(normalized, low_val, high_val)
    end

    return normalized
end

"""
    normalize_image!(image::Array, mask; kwargs...)

In-place version of `normalize_image`.

Modifies the input image array directly. The input array element type must be
compatible with Float64 values.

# See also
- [`normalize_image`](@ref)
"""
function normalize_image!(image::Array{T, N}, mask;
                          scale::Float64=1.0,
                          remove_outliers::Bool=false,
                          outlier_percentile::Float64=99.0,
                          label::Int=1) where {T<:AbstractFloat, N}
    roi_mask = _get_bool_mask(mask, label)
    voxels = image[roi_mask]

    isempty(voxels) && throw(ArgumentError(
        "No voxels found in mask for normalization."
    ))

    μ = mean(voxels)
    σ = std(voxels)

    if σ < eps(Float64)
        σ = 1.0
    end

    # In-place normalization
    @. image = scale * (image - μ) / σ

    if remove_outliers
        low_p = 100.0 - outlier_percentile
        high_p = outlier_percentile
        norm_roi = image[roi_mask]
        low_val = quantile(norm_roi, low_p / 100.0)
        high_val = quantile(norm_roi, high_p / 100.0)
        clamp!(image, low_val, high_val)
    end

    return image
end

#==============================================================================#
# Image Information and Validation
#==============================================================================#

"""
    is_2d(image) -> Bool

Check if an image is 2D (or has a singleton dimension).

# Arguments
- `image`: Image array or RadiomicsImage

# Returns
- `Bool`: true if image is 2D or has any dimension of size 1

# Example
```julia
img2d = rand(64, 64)
img3d = rand(64, 64, 64)
img_pseudo_2d = rand(64, 64, 1)

is_2d(img2d)        # true
is_2d(img3d)        # false
is_2d(img_pseudo_2d) # true (has singleton dimension)
```
"""
function is_2d(image::AbstractArray)
    return ndims(image) == 2 || any(==(1), size(image))
end

function is_2d(image::RadiomicsImage)
    return is_2d(image.data)
end

"""
    is_3d(image) -> Bool

Check if an image is truly 3D (all dimensions > 1).

# Arguments
- `image`: Image array or RadiomicsImage

# Returns
- `Bool`: true if image has 3 dimensions all greater than 1
"""
function is_3d(image::AbstractArray)
    return ndims(image) == 3 && all(>(1), size(image))
end

function is_3d(image::RadiomicsImage)
    return is_3d(image.data)
end

"""
    effective_ndims(image) -> Int

Return the number of non-singleton dimensions.

# Example
```julia
effective_ndims(rand(64, 64))       # 2
effective_ndims(rand(64, 64, 64))   # 3
effective_ndims(rand(64, 64, 1))    # 2
effective_ndims(rand(1, 64, 64))    # 2
```
"""
function effective_ndims(image::AbstractArray)
    return count(>(1), size(image))
end

function effective_ndims(image::RadiomicsImage)
    return effective_ndims(image.data)
end

"""
    validate_image_mask(image, mask; label::Int=1)

Validate that image and mask are compatible for feature extraction.

# Checks performed
- Dimensions match
- Both are 2D or 3D
- Mask contains the specified label
- ROI is not empty (at least 1 voxel)
- ROI is not too small for meaningful analysis

# Throws
- `DimensionMismatch`: If dimensions don't match
- `ArgumentError`: If validation fails

# Example
```julia
image = rand(64, 64, 64)
mask = rand(64, 64, 64) .> 0.9
validate_image_mask(image, mask)  # Throws if invalid
```
"""
function validate_image_mask(image::AbstractArray, mask::AbstractArray; label::Int=1)
    # Check dimensions match
    ndims(image) == ndims(mask) || throw(DimensionMismatch(
        "Image and mask must have same number of dimensions: got $(ndims(image))D and $(ndims(mask))D"
    ))

    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same size: got $(size(image)) and $(size(mask))"
    ))

    # Check dimensions are 2D or 3D
    ndims(image) in (2, 3) || throw(ArgumentError(
        "Image must be 2D or 3D, got $(ndims(image))D"
    ))

    # Check mask contains the label
    nvoxels = count_voxels(mask; label=label)
    nvoxels > 0 || throw(ArgumentError(
        "Mask does not contain label $label. Available labels: $(unique(mask))"
    ))

    # Warn if ROI is very small (but don't throw)
    if nvoxels < 27  # Less than 3x3x3
        @warn "ROI is very small ($nvoxels voxels). Feature calculations may be unreliable."
    end

    return true
end

function validate_image_mask(image::RadiomicsImage, mask; label::Int=1)
    validate_image_mask(image.data, mask; label=label)
end

function validate_image_mask(image, mask::RadiomicsMask; label::Int=mask.label)
    validate_image_mask(image, mask.data; label=label)
end

function validate_image_mask(image::RadiomicsImage, mask::RadiomicsMask;
                             label::Int=mask.label)
    validate_image_mask(image.data, mask.data; label=label)
end

#==============================================================================#
# Image/Mask Conversion Utilities
#==============================================================================#

"""
    ensure_float64(image::AbstractArray) -> Array{Float64}

Convert image to Float64 array, preserving dimensions.

# Example
```julia
img_int = rand(Int16, 64, 64, 64)
img_float = ensure_float64(img_int)
eltype(img_float)  # Float64
```
"""
function ensure_float64(image::AbstractArray{T, N}) where {T, N}
    if T === Float64
        return convert(Array{Float64, N}, image)
    else
        return Float64.(image)
    end
end

function ensure_float64(image::RadiomicsImage{T, N}) where {T, N}
    if T === Float64
        return image
    else
        return RadiomicsImage(Float64.(image.data), image.spacing)
    end
end

"""
    squeeze_image(image::AbstractArray) -> AbstractArray

Remove singleton dimensions from an image.

# Example
```julia
img = rand(64, 64, 1)
squeezed = squeeze_image(img)  # Now 64×64
```
"""
function squeeze_image(image::AbstractArray)
    return dropdims(image; dims=tuple(findall(==(1), size(image))...))
end

"""
    get_slice(image::AbstractArray{T, 3}, slice_dim::Int, slice_idx::Int) -> Array{T, 2}

Extract a 2D slice from a 3D image.

# Arguments
- `image`: 3D image array
- `slice_dim`: Dimension along which to slice (1, 2, or 3)
- `slice_idx`: Index of the slice

# Example
```julia
img = rand(64, 64, 64)
axial_slice = get_slice(img, 3, 32)  # 64×64 slice at z=32
```
"""
function get_slice(image::AbstractArray{T, 3}, slice_dim::Int, slice_idx::Int) where T
    1 <= slice_dim <= 3 || throw(ArgumentError("slice_dim must be 1, 2, or 3"))
    1 <= slice_idx <= size(image, slice_dim) || throw(BoundsError(
        "slice_idx $slice_idx out of bounds for dimension $slice_dim with size $(size(image, slice_dim))"
    ))

    if slice_dim == 1
        return image[slice_idx, :, :]
    elseif slice_dim == 2
        return image[:, slice_idx, :]
    else
        return image[:, :, slice_idx]
    end
end

function get_slice(image::RadiomicsImage{T, 3}, slice_dim::Int, slice_idx::Int) where T
    return get_slice(image.data, slice_dim, slice_idx)
end

#==============================================================================#
# Spacing Utilities
#==============================================================================#

"""
    get_physical_size(image::RadiomicsImage) -> NTuple{N, Float64}
    get_physical_size(image::AbstractArray, spacing::NTuple) -> NTuple

Calculate the physical size of an image in mm.

# Returns
- `NTuple`: Physical dimensions (width, height, depth) in mm

# Example
```julia
img = RadiomicsImage(rand(100, 100, 50), (1.0, 1.0, 2.0))
phys_size = get_physical_size(img)  # (100.0, 100.0, 100.0) mm
```
"""
function get_physical_size(image::RadiomicsImage)
    return size(image) .* image.spacing
end

function get_physical_size(image::AbstractArray{T, N}, spacing::NTuple{N, Float64}) where {T, N}
    return size(image) .* spacing
end

function get_physical_size(image::AbstractArray)
    return Float64.(size(image))  # Assumes unit spacing
end

"""
    apply_spacing(coords::Vector{CartesianIndex{N}}, spacing::NTuple{N, Float64}) -> Vector{NTuple{N, Float64}}

Convert voxel coordinates to physical coordinates.

# Arguments
- `coords`: Vector of CartesianIndex voxel coordinates
- `spacing`: Voxel spacing in mm

# Returns
- `Vector{NTuple}`: Physical coordinates in mm

# Example
```julia
coords = [CartesianIndex(10, 20, 30), CartesianIndex(11, 21, 31)]
spacing = (1.0, 1.0, 2.0)
physical = apply_spacing(coords, spacing)
# [(10.0, 20.0, 60.0), (11.0, 21.0, 62.0)]
```
"""
function apply_spacing(coords::Vector{CartesianIndex{N}},
                       spacing::NTuple{N, Float64}) where N
    return [Tuple(c) .* spacing for c in coords]
end

"""
    get_centroid(mask; spacing=nothing, label::Int=1) -> NTuple

Calculate the centroid of the ROI.

# Arguments
- `mask`: Segmentation mask
- `spacing`: Optional voxel spacing for physical coordinates
- `label::Int=1`: Label value defining the ROI

# Returns
- `NTuple`: Centroid coordinates (voxel or physical depending on spacing)

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[30:35, 30:35, 30:35] .= true
centroid = get_centroid(mask)  # Approximately (32.5, 32.5, 32.5)
```
"""
function get_centroid(mask::AbstractArray{Bool, N};
                      spacing::Union{Nothing, NTuple{N, Float64}}=nothing,
                      label::Int=1) where N
    coords = findall(mask)
    isempty(coords) && throw(ArgumentError("Mask contains no voxels"))

    # Calculate mean position
    centroid = ntuple(N) do d
        mean(Float64(c[d]) for c in coords)
    end

    # Apply spacing if provided
    if spacing !== nothing
        centroid = centroid .* spacing
    end

    return centroid
end

function get_centroid(mask::AbstractArray{<:Integer, N};
                      spacing::Union{Nothing, NTuple{N, Float64}}=nothing,
                      label::Int=1) where N
    return get_centroid(mask .== label; spacing=spacing, label=label)
end

function get_centroid(mask::RadiomicsMask; spacing=nothing, label::Int=mask.label)
    return get_centroid(get_roi_mask(mask); spacing=spacing, label=label)
end

#==============================================================================#
# Internal Helper Functions
#==============================================================================#

"""
Internal helper to convert any mask type to boolean mask.
"""
function _get_bool_mask(mask::AbstractArray{Bool}, label::Int)
    return mask
end

function _get_bool_mask(mask::AbstractArray{<:Integer}, label::Int)
    return mask .== label
end

function _get_bool_mask(mask::RadiomicsMask, label::Int)
    return mask.data .== label
end

function _get_bool_mask(mask::BitArray, label::Int)
    return mask
end
