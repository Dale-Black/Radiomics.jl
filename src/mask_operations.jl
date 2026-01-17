# Mask operations for Radiomics.jl
# This module provides functions for mask validation, bounding box computation,
# cropping to mask, and related ROI operations.
#
# Reference: PyRadiomics imageoperations.py - checkMask(), cropToTumorMask()

#==============================================================================#
# Bounding Box Computation
#==============================================================================#

"""
    BoundingBox{N}

Represents the bounding box of a region of interest in N dimensions.

# Fields
- `lower::NTuple{N, Int}`: Lower bounds (inclusive, 1-indexed)
- `upper::NTuple{N, Int}`: Upper bounds (inclusive, 1-indexed)

# Example
```julia
bbox = BoundingBox((10, 10, 10), (50, 50, 30))
bbox.lower  # (10, 10, 10)
bbox.upper  # (50, 50, 30)
size(bbox)  # (41, 41, 21)
```
"""
struct BoundingBox{N}
    lower::NTuple{N, Int}
    upper::NTuple{N, Int}

    function BoundingBox(lower::NTuple{N, Int}, upper::NTuple{N, Int}) where N
        all(l <= u for (l, u) in zip(lower, upper)) || throw(ArgumentError(
            "Lower bounds must be <= upper bounds: got lower=$lower, upper=$upper"
        ))
        all(l >= 1 for l in lower) || throw(ArgumentError(
            "Lower bounds must be >= 1 (1-indexed): got $lower"
        ))
        new{N}(lower, upper)
    end
end

# Convenience constructor from tuples
BoundingBox(lower::Tuple, upper::Tuple) = BoundingBox(Int.(lower), Int.(upper))

# Size of bounding box
Base.size(bbox::BoundingBox) = bbox.upper .- bbox.lower .+ 1
Base.ndims(bbox::BoundingBox{N}) where N = N

"""
    bounding_box(mask; label::Int=1, pad::Int=0) -> BoundingBox

Compute the bounding box of the ROI in a mask.

The bounding box is the smallest axis-aligned box that contains all voxels
of the specified label.

# Arguments
- `mask`: Segmentation mask (Bool, Integer, or RadiomicsMask)
- `label::Int=1`: Label value defining the ROI (ignored for boolean masks)
- `pad::Int=0`: Padding to add around the bounding box

# Returns
- `BoundingBox`: The bounding box with lower and upper bounds (1-indexed, inclusive)

# Throws
- `ArgumentError`: If mask contains no voxels with the specified label

# Example
```julia
# 2D mask
mask = zeros(Bool, 64, 64)
mask[20:40, 30:50] .= true
bbox = bounding_box(mask)
# BoundingBox((20, 30), (40, 50))

# With padding
bbox_padded = bounding_box(mask; pad=5)
# BoundingBox((15, 25), (45, 55))

# 3D mask with labels
mask_3d = zeros(Int, 64, 64, 64)
mask_3d[10:30, 15:45, 20:50] .= 1
mask_3d[40:60, 40:60, 40:60] .= 2
bbox_label1 = bounding_box(mask_3d; label=1)
bbox_label2 = bounding_box(mask_3d; label=2)
```

# Notes
- Uses 1-indexed coordinates (Julia convention)
- Padding is clamped to image boundaries
- PyRadiomics reference: `LabelStatisticsImageFilter.GetBoundingBox()`

# See also
- [`crop_to_mask`](@ref): Crop image and mask to bounding box
"""
function bounding_box(mask::AbstractArray{Bool, N}; label::Int=1, pad::Int=0) where N
    # Find all true indices
    indices = findall(mask)

    isempty(indices) && throw(ArgumentError(
        "Mask contains no voxels (all false). Cannot compute bounding box."
    ))

    # Extract bounds for each dimension
    lower = ntuple(N) do d
        minimum(idx[d] for idx in indices)
    end

    upper = ntuple(N) do d
        maximum(idx[d] for idx in indices)
    end

    # Apply padding (clamped to valid range)
    if pad > 0
        mask_size = size(mask)
        lower = ntuple(N) do d
            max(1, lower[d] - pad)
        end
        upper = ntuple(N) do d
            min(mask_size[d], upper[d] + pad)
        end
    end

    return BoundingBox(lower, upper)
end

# Handle integer masks with label
function bounding_box(mask::AbstractArray{<:Integer, N}; label::Int=1, pad::Int=0) where N
    roi_mask = mask .== label
    any(roi_mask) || throw(ArgumentError(
        "Mask contains no voxels with label $label. Available labels: $(sort(unique(mask)))"
    ))
    return bounding_box(roi_mask; label=label, pad=pad)
end

# Handle RadiomicsMask
function bounding_box(mask::RadiomicsMask; label::Int=mask.label, pad::Int=0)
    return bounding_box(mask.data; label=label, pad=pad)
end

# Handle BitArray
function bounding_box(mask::BitArray{N}; label::Int=1, pad::Int=0) where N
    return bounding_box(convert(Array{Bool, N}, mask); label=label, pad=pad)
end

"""
    bounding_box_size(mask; label::Int=1) -> NTuple

Get the size of the ROI bounding box without creating a BoundingBox object.

This is a convenience function for when you only need the dimensions.

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[20:40, 20:40, 20:40] .= true
bounding_box_size(mask)  # (21, 21, 21)
```
"""
function bounding_box_size(mask; label::Int=1)
    bbox = bounding_box(mask; label=label)
    return size(bbox)
end

#==============================================================================#
# Crop to Mask
#==============================================================================#

"""
    crop_to_mask(image, mask; label::Int=1, pad::Int=0) -> Tuple{Array, Array}

Crop an image and its corresponding mask to the bounding box of the ROI.

Returns cropped views/copies of both the image and mask, reduced to the
minimum region containing all voxels of the specified label.

# Arguments
- `image`: The image to crop
- `mask`: The segmentation mask
- `label::Int=1`: Label value defining the ROI
- `pad::Int=0`: Padding to add around the bounding box

# Returns
- `Tuple{Array, Array}`: Cropped image and cropped mask

# Throws
- `DimensionMismatch`: If image and mask dimensions don't match
- `ArgumentError`: If mask contains no voxels with the specified label

# Example
```julia
# Create image and mask
image = rand(100, 100, 100)
mask = zeros(Bool, 100, 100, 100)
mask[40:60, 40:60, 40:60] .= true

# Crop to ROI
cropped_img, cropped_mask = crop_to_mask(image, mask)
size(cropped_img)  # (21, 21, 21)

# With padding
cropped_img, cropped_mask = crop_to_mask(image, mask; pad=5)
size(cropped_img)  # (31, 31, 31)
```

# Notes
- Returns copies, not views (to ensure memory contiguity)
- PyRadiomics reference: `cropToTumorMask()`
- Useful for reducing computation on large images with small ROIs

# See also
- [`bounding_box`](@ref): Get bounding box without cropping
- [`crop_to_bbox`](@ref): Crop using a pre-computed bounding box
"""
function crop_to_mask(image::AbstractArray{T, N}, mask::AbstractArray{<:Any, N};
                      label::Int=1, pad::Int=0) where {T, N}
    # Validate dimensions
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask dimensions must match: got $(size(image)) and $(size(mask))"
    ))

    # Get bounding box
    bool_mask = _get_bool_mask(mask, label)
    bbox = bounding_box(bool_mask; label=label, pad=pad)

    # Create index ranges for each dimension
    ranges = ntuple(N) do d
        bbox.lower[d]:bbox.upper[d]
    end

    # Crop image and mask
    cropped_image = image[ranges...]
    cropped_mask = mask[ranges...]

    return copy(cropped_image), copy(cropped_mask)
end

# Handle RadiomicsImage - returns plain arrays (not RadiomicsImage)
function crop_to_mask(image::RadiomicsImage{T, N}, mask::AbstractArray;
                      label::Int=1, pad::Int=0) where {T, N}
    return crop_to_mask(image.data, mask; label=label, pad=pad)
end

# Handle RadiomicsMask
function crop_to_mask(image::AbstractArray, mask::RadiomicsMask;
                      label::Int=mask.label, pad::Int=0)
    return crop_to_mask(image, mask.data; label=label, pad=pad)
end

# Handle both RadiomicsImage and RadiomicsMask
function crop_to_mask(image::RadiomicsImage, mask::RadiomicsMask;
                      label::Int=mask.label, pad::Int=0)
    return crop_to_mask(image.data, mask.data; label=label, pad=pad)
end

"""
    crop_to_bbox(image, bbox::BoundingBox) -> Array

Crop an image to a pre-computed bounding box.

# Arguments
- `image`: The image to crop
- `bbox::BoundingBox`: The bounding box to crop to

# Returns
- `Array`: Cropped image

# Example
```julia
image = rand(100, 100, 100)
bbox = BoundingBox((20, 20, 20), (80, 80, 80))
cropped = crop_to_bbox(image, bbox)
size(cropped)  # (61, 61, 61)
```
"""
function crop_to_bbox(image::AbstractArray{T, N}, bbox::BoundingBox{N}) where {T, N}
    ranges = ntuple(N) do d
        bbox.lower[d]:bbox.upper[d]
    end
    return copy(image[ranges...])
end

function crop_to_bbox(image::RadiomicsImage{T, N}, bbox::BoundingBox{N}) where {T, N}
    return crop_to_bbox(image.data, bbox)
end

#==============================================================================#
# Mask Validation
#==============================================================================#

"""
    validate_mask(mask; label::Int=1, check_connectivity::Bool=false,
                  min_voxels::Int=1) -> NamedTuple

Perform comprehensive validation of a segmentation mask.

# Arguments
- `mask`: The segmentation mask to validate
- `label::Int=1`: Label value defining the ROI
- `check_connectivity::Bool=false`: Whether to check for connected components
- `min_voxels::Int=1`: Minimum number of voxels required in the ROI

# Returns
- `NamedTuple` with fields:
  - `is_valid::Bool`: Whether the mask passes all checks
  - `nvoxels::Int`: Number of voxels in the ROI
  - `ndims_effective::Int`: Number of dimensions where ROI extent > 1
  - `bbox::BoundingBox`: Bounding box of the ROI
  - `is_binary::Bool`: Whether mask contains only 0 and label
  - `num_components::Union{Int, Nothing}`: Number of connected components (if checked)
  - `warnings::Vector{String}`: List of warnings generated

# Example
```julia
mask = zeros(Int, 64, 64, 64)
mask[20:40, 20:40, 20:40] .= 1
result = validate_mask(mask)

result.is_valid      # true
result.nvoxels       # 9261 (21³)
result.ndims_effective  # 3 (truly 3D)
```

# Notes
- Connectivity check requires additional computation
- PyRadiomics reference: `checkMask()` in imageoperations.py
"""
function validate_mask(mask::AbstractArray{<:Integer, N};
                       label::Int=1,
                       check_connectivity::Bool=false,
                       min_voxels::Int=1) where N
    warnings = String[]
    is_valid = true

    # Get ROI mask
    roi_mask = mask .== label

    # Count voxels
    nvoxels = count(roi_mask)

    if nvoxels == 0
        is_valid = false
        push!(warnings, "Mask contains no voxels with label $label")
        return (
            is_valid = false,
            nvoxels = 0,
            ndims_effective = 0,
            bbox = nothing,
            is_binary = false,
            num_components = nothing,
            warnings = warnings
        )
    end

    # Compute bounding box
    bbox = bounding_box(roi_mask; label=label)

    # Check dimensionality (how many dimensions have extent > 1)
    bbox_size = size(bbox)
    ndims_effective = count(>(1), bbox_size)

    if ndims_effective == 0
        push!(warnings, "ROI is a single voxel (0D)")
    elseif ndims_effective == 1
        push!(warnings, "ROI is a line (1D)")
    elseif ndims_effective == 2 && N == 3
        push!(warnings, "ROI is a surface (2D in 3D image)")
    end

    # Check minimum voxels
    if nvoxels < min_voxels
        is_valid = false
        push!(warnings, "ROI has $nvoxels voxels, less than minimum $min_voxels")
    end

    # Check if mask is binary (only 0 and label)
    unique_vals = unique(mask)
    is_binary = length(unique_vals) <= 2 && all(v -> v == 0 || v == label, unique_vals)
    if !is_binary
        push!(warnings, "Mask is not binary. Contains labels: $(unique_vals)")
    end

    # Connectivity check (simple version without full connected component analysis)
    num_components = nothing
    if check_connectivity
        num_components = _count_connected_components(roi_mask)
        if num_components > 1
            push!(warnings, "ROI has $num_components disconnected components")
        end
    end

    return (
        is_valid = is_valid,
        nvoxels = nvoxels,
        ndims_effective = ndims_effective,
        bbox = bbox,
        is_binary = is_binary,
        num_components = num_components,
        warnings = warnings
    )
end

function validate_mask(mask::AbstractArray{Bool, N}; kwargs...) where N
    return validate_mask(Int.(mask); label=1, kwargs...)
end

function validate_mask(mask::RadiomicsMask; label::Int=mask.label, kwargs...)
    return validate_mask(mask.data; label=label, kwargs...)
end

"""
    is_empty_mask(mask; label::Int=1) -> Bool

Check if a mask contains no voxels of the specified label.

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
is_empty_mask(mask)  # true

mask[32, 32, 32] = true
is_empty_mask(mask)  # false
```
"""
function is_empty_mask(mask::AbstractArray{Bool}; label::Int=1)
    return !any(mask)
end

function is_empty_mask(mask::AbstractArray{<:Integer}; label::Int=1)
    return !any(==(label), mask)
end

function is_empty_mask(mask::RadiomicsMask; label::Int=mask.label)
    return is_empty_mask(mask.data; label=label)
end

"""
    is_full_mask(mask; label::Int=1) -> Bool

Check if a mask contains the specified label at every voxel.

# Example
```julia
mask = trues(64, 64, 64)
is_full_mask(mask)  # true

mask[1, 1, 1] = false
is_full_mask(mask)  # false
```
"""
function is_full_mask(mask::AbstractArray{Bool}; label::Int=1)
    return all(mask)
end

function is_full_mask(mask::AbstractArray{<:Integer}; label::Int=1)
    return all(==(label), mask)
end

function is_full_mask(mask::RadiomicsMask; label::Int=mask.label)
    return is_full_mask(mask.data; label=label)
end

"""
    mask_extent(mask; label::Int=1) -> NTuple

Calculate the extent (size) of the ROI in each dimension.

This is equivalent to `size(bounding_box(mask))`.

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[10:30, 20:40, 25:50] .= true
mask_extent(mask)  # (21, 21, 26)
```
"""
function mask_extent(mask; label::Int=1)
    bbox = bounding_box(mask; label=label)
    return size(bbox)
end

"""
    mask_dimensionality(mask; label::Int=1) -> Int

Determine the effective dimensionality of the ROI.

Returns the number of dimensions where the ROI extent is greater than 1:
- 0: Single voxel
- 1: Line (1D)
- 2: Surface (2D)
- 3: Volume (3D)

# Example
```julia
# Single voxel
mask1 = zeros(Bool, 64, 64, 64)
mask1[32, 32, 32] = true
mask_dimensionality(mask1)  # 0

# Line
mask2 = zeros(Bool, 64, 64, 64)
mask2[30:40, 32, 32] .= true
mask_dimensionality(mask2)  # 1

# Volume
mask3 = zeros(Bool, 64, 64, 64)
mask3[20:40, 20:40, 20:40] .= true
mask_dimensionality(mask3)  # 3
```

# Notes
- PyRadiomics reference: `checkMask()` dimensionality assessment
"""
function mask_dimensionality(mask; label::Int=1)
    bbox = bounding_box(mask; label=label)
    return count(>(1), size(bbox))
end

#==============================================================================#
# Mask Manipulation
#==============================================================================#

"""
    dilate_mask(mask::AbstractArray{Bool, N}; radius::Int=1) -> Array{Bool, N}

Dilate a binary mask using a structuring element.

Each true voxel expands to include all neighbors within the given radius
(using a box/cuboid structuring element).

# Arguments
- `mask`: Binary mask to dilate
- `radius::Int=1`: Radius of dilation (1 = immediate neighbors)

# Returns
- `Array{Bool}`: Dilated mask

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[32, 32, 32] = true
dilated = dilate_mask(mask; radius=2)
count(dilated)  # 125 (5³ cube centered at 32,32,32)
```

# Notes
- Uses a simple box/cuboid structuring element
- For more sophisticated morphology, consider ImageMorphology.jl
"""
function dilate_mask(mask::AbstractArray{Bool, N}; radius::Int=1) where N
    radius >= 0 || throw(ArgumentError("radius must be non-negative, got $radius"))
    radius == 0 && return copy(mask)

    result = zeros(Bool, size(mask))
    mask_size = size(mask)

    # Iterate over all true voxels and expand
    for idx in findall(mask)
        # Create ranges for neighborhood
        ranges = ntuple(N) do d
            lo = max(1, idx[d] - radius)
            hi = min(mask_size[d], idx[d] + radius)
            lo:hi
        end

        # Set all voxels in neighborhood to true
        result[ranges...] .= true
    end

    return result
end

"""
    erode_mask(mask::AbstractArray{Bool, N}; radius::Int=1) -> Array{Bool, N}

Erode a binary mask using a structuring element.

A voxel remains true only if all neighbors within the given radius are also true.

# Arguments
- `mask`: Binary mask to erode
- `radius::Int=1`: Radius of erosion (1 = immediate neighbors)

# Returns
- `Array{Bool}`: Eroded mask

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[30:35, 30:35, 30:35] .= true  # 6x6x6 cube
eroded = erode_mask(mask; radius=1)
count(eroded)  # 64 (4x4x4 interior)
```

# Notes
- Uses a simple box/cuboid structuring element
- May result in empty mask if original is too small
"""
function erode_mask(mask::AbstractArray{Bool, N}; radius::Int=1) where N
    radius >= 0 || throw(ArgumentError("radius must be non-negative, got $radius"))
    radius == 0 && return copy(mask)

    result = zeros(Bool, size(mask))
    mask_size = size(mask)

    for idx in CartesianIndices(mask)
        if mask[idx]
            # Check if all neighbors within radius are true
            is_interior = true
            for offset in Iterators.product(ntuple(_ -> -radius:radius, N)...)
                neighbor = idx + CartesianIndex(offset)
                # Check bounds
                in_bounds = all(1 <= neighbor[d] <= mask_size[d] for d in 1:N)
                if !in_bounds || !mask[neighbor]
                    is_interior = false
                    break
                end
            end
            result[idx] = is_interior
        end
    end

    return result
end

"""
    fill_holes_2d(mask::AbstractArray{Bool, 2}) -> Array{Bool, 2}

Fill holes in a 2D binary mask.

A hole is defined as a region of false values completely surrounded by true values.

# Example
```julia
mask = trues(64, 64)
mask[30:35, 30:35] .= false  # Create a hole
filled = fill_holes_2d(mask)
all(filled)  # true
```

# Notes
- Only works on 2D masks
- For 3D hole filling, consider slice-by-slice application
"""
function fill_holes_2d(mask::AbstractArray{Bool, 2})
    # Simple flood fill from corners approach
    result = copy(mask)
    rows, cols = size(mask)

    # Create a padded version to handle border
    padded = ones(Bool, rows + 2, cols + 2)
    padded[2:end-1, 2:end-1] .= mask

    # Flood fill from corner (marks everything reachable from outside)
    # Use iterative approach to avoid stack overflow
    background = zeros(Bool, rows + 2, cols + 2)
    queue = CartesianIndex{2}[]
    push!(queue, CartesianIndex(1, 1))

    while !isempty(queue)
        idx = popfirst!(queue)
        if background[idx] || padded[idx]
            continue
        end
        background[idx] = true

        # Add neighbors
        for (di, dj) in ((0, 1), (0, -1), (1, 0), (-1, 0))
            ni, nj = idx[1] + di, idx[2] + dj
            if 1 <= ni <= rows + 2 && 1 <= nj <= cols + 2
                push!(queue, CartesianIndex(ni, nj))
            end
        end
    end

    # Anything not reachable from background and not original mask is a hole
    for i in 1:rows, j in 1:cols
        if !mask[i, j] && !background[i+1, j+1]
            result[i, j] = true
        end
    end

    return result
end

#==============================================================================#
# Connected Components (Simple Implementation)
#==============================================================================#

"""
    _count_connected_components(mask::AbstractArray{Bool, N}) -> Int

Count the number of connected components in a binary mask.

Uses 6-connectivity in 3D (face-adjacent) and 4-connectivity in 2D.

# Notes
- This is a simple implementation for validation purposes
- For full connected component labeling, use Images.jl or ImageMorphology.jl
"""
function _count_connected_components(mask::AbstractArray{Bool, N}) where N
    if !any(mask)
        return 0
    end

    # Create a label array
    labels = zeros(Int, size(mask))
    current_label = 0

    # Define connectivity offsets (6-connected for 3D, 4-connected for 2D)
    if N == 2
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elseif N == 3
        offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    else
        # Fallback for other dimensions
        return 1  # Assume single component
    end

    mask_size = size(mask)

    # BFS-based connected component labeling
    for idx in CartesianIndices(mask)
        if mask[idx] && labels[idx] == 0
            current_label += 1
            queue = CartesianIndex{N}[idx]
            labels[idx] = current_label

            while !isempty(queue)
                current = popfirst!(queue)
                for offset in offsets
                    neighbor = current + CartesianIndex(offset)

                    # Check bounds
                    in_bounds = all(1 <= neighbor[d] <= mask_size[d] for d in 1:N)
                    if in_bounds && mask[neighbor] && labels[neighbor] == 0
                        labels[neighbor] = current_label
                        push!(queue, neighbor)
                    end
                end
            end
        end
    end

    return current_label
end

"""
    largest_connected_component(mask::AbstractArray{Bool, N}) -> Array{Bool, N}

Extract the largest connected component from a binary mask.

# Arguments
- `mask`: Binary mask

# Returns
- `Array{Bool}`: Mask containing only the largest connected component

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[10:20, 10:20, 10:20] .= true  # Component 1 (11³ = 1331 voxels)
mask[40:45, 40:45, 40:45] .= true  # Component 2 (6³ = 216 voxels)

largest = largest_connected_component(mask)
count(largest)  # 1331 (only the larger component)
```
"""
function largest_connected_component(mask::AbstractArray{Bool, N}) where N
    if !any(mask)
        return copy(mask)
    end

    # Create a label array
    labels = zeros(Int, size(mask))
    component_sizes = Int[]
    current_label = 0

    # Define connectivity offsets
    if N == 2
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elseif N == 3
        offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    else
        return copy(mask)
    end

    mask_size = size(mask)

    # BFS-based connected component labeling
    for idx in CartesianIndices(mask)
        if mask[idx] && labels[idx] == 0
            current_label += 1
            component_size = 0
            queue = CartesianIndex{N}[idx]
            labels[idx] = current_label

            while !isempty(queue)
                current = popfirst!(queue)
                component_size += 1

                for offset in offsets
                    neighbor = current + CartesianIndex(offset)

                    in_bounds = all(1 <= neighbor[d] <= mask_size[d] for d in 1:N)
                    if in_bounds && mask[neighbor] && labels[neighbor] == 0
                        labels[neighbor] = current_label
                        push!(queue, neighbor)
                    end
                end
            end

            push!(component_sizes, component_size)
        end
    end

    if isempty(component_sizes)
        return copy(mask)
    end

    # Find largest component
    largest_label = argmax(component_sizes)

    return labels .== largest_label
end

#==============================================================================#
# Mask Statistics
#==============================================================================#

"""
    mask_surface_voxels(mask::AbstractArray{Bool, N}; connectivity::Int=6) -> Array{Bool, N}

Identify surface voxels (voxels at the boundary of the ROI).

A surface voxel is a true voxel that has at least one false neighbor.

# Arguments
- `mask`: Binary mask
- `connectivity::Int=6`: Number of neighbors to check (6 or 26 for 3D)

# Returns
- `Array{Bool}`: Mask of surface voxels only

# Example
```julia
mask = zeros(Bool, 64, 64, 64)
mask[20:40, 20:40, 20:40] .= true
surface = mask_surface_voxels(mask)

# Surface voxels are the outer shell of the cube
```
"""
function mask_surface_voxels(mask::AbstractArray{Bool, N}; connectivity::Int=6) where N
    result = zeros(Bool, size(mask))
    mask_size = size(mask)

    # Define neighbor offsets based on connectivity
    if N == 3
        if connectivity == 6
            offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        else  # 26-connectivity
            offsets = vec(collect(Iterators.product(-1:1, -1:1, -1:1)))
            filter!(o -> o != (0, 0, 0), offsets)
        end
    elseif N == 2
        if connectivity in (4, 6)
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else
            offsets = vec(collect(Iterators.product(-1:1, -1:1)))
            filter!(o -> o != (0, 0), offsets)
        end
    else
        return copy(mask)
    end

    for idx in CartesianIndices(mask)
        if mask[idx]
            # Check if any neighbor is false (or out of bounds)
            is_surface = false
            for offset in offsets
                neighbor = idx + CartesianIndex(offset)

                in_bounds = all(1 <= neighbor[d] <= mask_size[d] for d in 1:N)
                if !in_bounds || !mask[neighbor]
                    is_surface = true
                    break
                end
            end
            result[idx] = is_surface
        end
    end

    return result
end

"""
    mask_interior_voxels(mask::AbstractArray{Bool, N}; connectivity::Int=6) -> Array{Bool, N}

Identify interior voxels (voxels completely surrounded by other ROI voxels).

An interior voxel is a true voxel where all neighbors are also true.

# Arguments
- `mask`: Binary mask
- `connectivity::Int=6`: Number of neighbors to check

# Returns
- `Array{Bool}`: Mask of interior voxels only
"""
function mask_interior_voxels(mask::AbstractArray{Bool, N}; connectivity::Int=6) where N
    surface = mask_surface_voxels(mask; connectivity=connectivity)
    return mask .& .!surface
end
