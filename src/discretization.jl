# Gray Level Discretization for Radiomics.jl
#
# This module implements gray level discretization (binning) to reduce the number
# of gray levels before computing texture features. Matching PyRadiomics behavior
# exactly is critical for parity.
#
# References:
# - PyRadiomics imageoperations.py: getBinEdges(), binImage()
# - IBSI: Section 3.4 (Gray level discretisation)

#==============================================================================#
# Core Discretization Functions
#==============================================================================#

"""
    get_bin_edges(values; binwidth=25.0, bincount=nothing)

Compute histogram bin edges for gray level discretization.

# Arguments
- `values`: Vector of intensity values to discretize (typically voxels within ROI)
- `binwidth::Real=25.0`: Width of each bin (used for Fixed Bin Width mode)
- `bincount::Union{Int, Nothing}=nothing`: Number of bins (used for Fixed Bin Count mode)

# Returns
- `Vector{Float64}`: Bin edges for use with `discretize`

# Modes
1. **Fixed Bin Width** (default, when `bincount=nothing`):
   Bins are created with fixed width `W`. The formula follows PyRadiomics:
   ```
   X_b,i = floor(X_gl,i / W) - floor(min(X_gl) / W) + 1
   ```

2. **Fixed Bin Count** (when `bincount` is specified):
   Creates exactly `bincount` equal-width bins spanning the data range.

# Edge Cases
- If all values are identical, returns `[value - 0.5, value + 0.5]` (single bin)
- Empty input throws an ArgumentError

# Example
```julia
voxels = [10.0, 25.0, 30.0, 50.0, 75.0]

# Fixed bin width (default)
edges = get_bin_edges(voxels, binwidth=25.0)

# Fixed bin count
edges = get_bin_edges(voxels, bincount=4)
```

# References
- PyRadiomics: radiomics/imageoperations.py:getBinEdges
- IBSI: Section 3.4.3 (Fixed bin width) and 3.4.2 (Fixed bin number)
"""
function get_bin_edges(values::AbstractVector{<:Real};
                       binwidth::Real=25.0,
                       bincount::Union{Int, Nothing}=nothing)

    isempty(values) && throw(ArgumentError("Cannot compute bin edges for empty array"))
    binwidth > 0 || throw(ArgumentError("binwidth must be positive, got $binwidth"))

    # Get min and max, filtering NaN
    valid_values = filter(!isnan, values)
    isempty(valid_values) && throw(ArgumentError("All values are NaN"))

    minval = minimum(valid_values)
    maxval = maximum(valid_values)

    # Edge case: all values identical
    if minval == maxval
        return Float64[minval - 0.5, maxval + 0.5]
    end

    if isnothing(bincount)
        # Fixed Bin Width mode (PyRadiomics default)
        return _get_bin_edges_fixed_width(minval, maxval, binwidth)
    else
        # Fixed Bin Count mode
        bincount > 0 || throw(ArgumentError("bincount must be positive, got $bincount"))
        return _get_bin_edges_fixed_count(minval, maxval, bincount)
    end
end

"""
    _get_bin_edges_fixed_width(minval, maxval, binwidth)

Compute bin edges for Fixed Bin Width discretization.

The bin edges follow PyRadiomics convention exactly:
- Lower bound: minimum - (minimum % binwidth) for positive, or aligned to grid
- Upper bound: maximum + 2 * binwidth to ensure inclusion

This ensures bins are aligned at multiples of binwidth.
"""
function _get_bin_edges_fixed_width(minval::Real, maxval::Real, binwidth::Real)
    # Match PyRadiomics: lowBound = minimum - (minimum % binWidth)
    # For Julia, we use modulo with correct handling for negative values
    lower_bound = minval - mod(minval, binwidth)

    # Match PyRadiomics: highBound = maximum + 2 * binWidth
    upper_bound = maxval + 2 * binwidth

    # Generate edges using arange-like behavior
    edges = Float64[]
    edge = lower_bound
    while edge < upper_bound
        push!(edges, edge)
        edge += binwidth
    end

    # Ensure we have at least 2 edges (1 bin)
    if length(edges) < 2
        push!(edges, lower_bound + binwidth)
    end

    return edges
end

"""
    _get_bin_edges_fixed_count(minval, maxval, bincount)

Compute bin edges for Fixed Bin Count discretization.

Creates exactly `bincount` equal-width bins spanning [minval, maxval].
Following PyRadiomics, the final edge is extended by 1 unit to ensure
the maximum value is included in the last bin.
"""
function _get_bin_edges_fixed_count(minval::Real, maxval::Real, bincount::Int)
    # Calculate bin width
    bin_width = (maxval - minval) / bincount

    # Generate edges (one more edge than bins)
    edges = collect(range(minval, maxval, length=bincount + 1))

    # Extend final edge to include maximum (PyRadiomics convention)
    # This ensures digitize places maxval in the last bin, not beyond
    edges[end] += 1.0

    return Float64.(edges)
end

#==============================================================================#
# Discretization Functions
#==============================================================================#

"""
    discretize(values, edges)

Discretize values into integer bin indices using pre-computed bin edges.

Matches numpy.digitize behavior exactly:
- Bin i contains values where `edges[i] <= x < edges[i+1]`
- Result is 1-indexed (bins 1 to nbins)

# Arguments
- `values`: Array of intensity values to discretize
- `edges`: Bin edges from `get_bin_edges`

# Returns
- Array of the same shape as `values` with integer bin indices (1-indexed)

# Notes
- Values below first edge get bin 1
- Values >= last edge get bin nbins (clamped)
- Matches numpy.digitize(x, edges) behavior exactly

# Example
```julia
voxels = [10.0, 25.0, 30.0, 50.0, 75.0]
edges = get_bin_edges(voxels, binwidth=25.0)
bins = discretize(voxels, edges)
```
"""
function discretize(values::AbstractArray{T}, edges::AbstractVector{<:Real}) where {T<:Real}
    length(edges) >= 2 || throw(ArgumentError("edges must have at least 2 elements"))

    result = similar(values, Int)
    nbins = length(edges) - 1

    @inbounds for i in eachindex(values)
        if isnan(values[i])
            result[i] = 0  # NaN values get bin 0 (invalid)
        else
            # Match numpy.digitize: find bin where edges[bin] <= value < edges[bin+1]
            # searchsortedlast gives last index where edges[j] <= value
            bin = searchsortedlast(edges, values[i])
            # Clamp to valid range [1, nbins]
            result[i] = clamp(bin, 1, nbins)
        end
    end

    return result
end

"""
    discretize(values::AbstractVector, edges::AbstractVector)

Discretize a vector of values. Returns a Vector{Int}.
"""
function discretize(values::AbstractVector{T}, edges::AbstractVector{<:Real}) where {T<:Real}
    length(edges) >= 2 || throw(ArgumentError("edges must have at least 2 elements"))

    nbins = length(edges) - 1
    result = Vector{Int}(undef, length(values))

    @inbounds for i in eachindex(values)
        if isnan(values[i])
            result[i] = 0  # NaN values get bin 0 (invalid)
        else
            # Match numpy.digitize: searchsortedlast for left-closed intervals
            bin = searchsortedlast(edges, values[i])
            result[i] = clamp(bin, 1, nbins)
        end
    end

    return result
end

#==============================================================================#
# High-Level Discretization Functions
#==============================================================================#

"""
    discretize_image(image, mask; binwidth=25.0, bincount=nothing, label=1)

Discretize image intensities within the ROI defined by the mask.

# Arguments
- `image`: Image array (any numeric type)
- `mask`: Mask array (Bool, BitArray, or Integer)
- `binwidth::Real=25.0`: Bin width for Fixed Bin Width mode
- `bincount::Union{Int, Nothing}=nothing`: Number of bins for Fixed Bin Count mode
- `label::Int=1`: Label value for ROI in integer masks

# Returns
- `NamedTuple` with fields:
  - `discretized`: Discretized image (same shape as input, 0 outside ROI)
  - `edges`: Bin edges used for discretization
  - `nbins`: Number of bins (gray levels)
  - `min_val`: Minimum value in ROI (before discretization)
  - `max_val`: Maximum value in ROI (before discretization)

# Example
```julia
image = rand(64, 64, 64) * 100
mask = rand(Bool, 64, 64, 64)

# Fixed bin width
result = discretize_image(image, mask, binwidth=10.0)
println("Number of gray levels: ", result.nbins)

# Fixed bin count
result = discretize_image(image, mask, bincount=32)
```

# References
- PyRadiomics: radiomics/imageoperations.py:binImage
"""
function discretize_image(image::AbstractArray{<:Real},
                          mask::AbstractArray{Bool};
                          binwidth::Real=25.0,
                          bincount::Union{Int, Nothing}=nothing,
                          label::Int=1)

    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))

    # Extract voxels within ROI
    roi_voxels = image[mask]

    isempty(roi_voxels) && throw(ArgumentError("Mask is empty, no voxels to discretize"))

    # Get bin edges
    edges = get_bin_edges(roi_voxels; binwidth=binwidth, bincount=bincount)

    # Create output array (zeros outside ROI)
    discretized = zeros(Int, size(image))

    # Discretize only within ROI
    discretized[mask] = discretize(roi_voxels, edges)

    return (
        discretized = discretized,
        edges = edges,
        nbins = length(edges) - 1,
        min_val = minimum(roi_voxels),
        max_val = maximum(roi_voxels)
    )
end

# Method for integer masks
function discretize_image(image::AbstractArray{<:Real},
                          mask::AbstractArray{<:Integer};
                          binwidth::Real=25.0,
                          bincount::Union{Int, Nothing}=nothing,
                          label::Int=1)

    bool_mask = mask .== label
    return discretize_image(image, bool_mask; binwidth=binwidth, bincount=bincount, label=label)
end

# Method for BitArray masks
function discretize_image(image::AbstractArray{<:Real},
                          mask::BitArray;
                          binwidth::Real=25.0,
                          bincount::Union{Int, Nothing}=nothing,
                          label::Int=1)

    return discretize_image(image, convert(Array{Bool}, mask);
                           binwidth=binwidth, bincount=bincount, label=label)
end

# Method for RadiomicsImage and RadiomicsMask
function discretize_image(image::RadiomicsImage,
                          mask::RadiomicsMask;
                          binwidth::Real=25.0,
                          bincount::Union{Int, Nothing}=nothing,
                          label::Union{Int, Nothing}=nothing)

    # Use mask's label if not specified
    actual_label = isnothing(label) ? mask.label : label
    return discretize_image(image.data, mask.data;
                           binwidth=binwidth, bincount=bincount, label=actual_label)
end

#==============================================================================#
# Utility Functions
#==============================================================================#

"""
    discretize_voxels(voxels; binwidth=25.0, bincount=nothing)

Discretize a vector of voxel values directly.

This is a convenience function for discretizing voxels that have already been
extracted from an image. Useful for feature computation that operates on
voxel vectors rather than full images.

# Arguments
- `voxels`: Vector of voxel intensity values
- `binwidth::Real=25.0`: Bin width for Fixed Bin Width mode
- `bincount::Union{Int, Nothing}=nothing`: Number of bins for Fixed Bin Count mode

# Returns
- `NamedTuple` with fields:
  - `discretized`: Vector of discretized values (Int)
  - `edges`: Bin edges used
  - `nbins`: Number of bins

# Example
```julia
voxels = get_voxels(image, mask)
result = discretize_voxels(voxels, binwidth=25.0)
```
"""
function discretize_voxels(voxels::AbstractVector{<:Real};
                           binwidth::Real=25.0,
                           bincount::Union{Int, Nothing}=nothing)

    edges = get_bin_edges(voxels; binwidth=binwidth, bincount=bincount)
    discretized = discretize(voxels, edges)

    return (
        discretized = discretized,
        edges = edges,
        nbins = length(edges) - 1
    )
end

"""
    get_discretization_range(image, mask; label=1)

Get the intensity range (min, max) of voxels within the ROI.

Useful for determining appropriate binning parameters.

# Example
```julia
min_val, max_val = get_discretization_range(image, mask)
suggested_bins = ceil(Int, (max_val - min_val) / 25.0)
```
"""
function get_discretization_range(image::AbstractArray{<:Real},
                                  mask::AbstractArray;
                                  label::Int=1)

    # Convert mask to boolean if needed
    if eltype(mask) <: Integer
        bool_mask = mask .== label
    elseif eltype(mask) == Bool || mask isa BitArray
        bool_mask = mask
    else
        throw(ArgumentError("Unsupported mask type: $(typeof(mask))"))
    end

    roi_voxels = image[bool_mask]
    isempty(roi_voxels) && throw(ArgumentError("Mask is empty"))

    return (minimum(roi_voxels), maximum(roi_voxels))
end

"""
    suggest_bincount(image, mask; target_bins=64, label=1)

Suggest an appropriate bin count based on image intensity range.

# Arguments
- `image`: Image array
- `mask`: Mask array
- `target_bins::Int=64`: Target number of bins
- `label::Int=1`: Label for integer masks

# Returns
- Suggested bin count (clamped between 16 and 256)

# Notes
This is a heuristic. The optimal bin count depends on the application.
PyRadiomics recommends 30-130 bins for good reproducibility.
"""
function suggest_bincount(image::AbstractArray{<:Real},
                          mask::AbstractArray;
                          target_bins::Int=64,
                          label::Int=1)

    min_val, max_val = get_discretization_range(image, mask; label=label)

    if min_val == max_val
        return 1
    end

    # Use target_bins, clamped to reasonable range
    return clamp(target_bins, 16, 256)
end

"""
    suggest_binwidth(image, mask; target_bins=64, label=1)

Suggest an appropriate bin width based on image intensity range and target bin count.

# Arguments
- `image`: Image array
- `mask`: Mask array
- `target_bins::Int=64`: Target number of bins
- `label::Int=1`: Label for integer masks

# Returns
- Suggested bin width

# Example
```julia
binwidth = suggest_binwidth(image, mask, target_bins=50)
result = discretize_image(image, mask, binwidth=binwidth)
```
"""
function suggest_binwidth(image::AbstractArray{<:Real},
                          mask::AbstractArray;
                          target_bins::Int=64,
                          label::Int=1)

    min_val, max_val = get_discretization_range(image, mask; label=label)

    if min_val == max_val
        return 1.0
    end

    return (max_val - min_val) / target_bins
end

#==============================================================================#
# Settings-based Discretization
#==============================================================================#

"""
    discretize_image(image, mask, settings::Settings)

Discretize image using settings from a Settings object.

Uses settings.discretization_mode to determine the binning method:
- `FixedBinWidth`: Uses settings.binwidth
- `FixedBinCount`: Uses settings.bincount

# Example
```julia
settings = Settings(binwidth=32.0)
result = discretize_image(image, mask, settings)

settings = Settings(bincount=64, discretization_mode=FixedBinCount)
result = discretize_image(image, mask, settings)
```
"""
function discretize_image(image::AbstractArray{<:Real},
                          mask::AbstractArray,
                          settings::Settings)

    if settings.discretization_mode == FixedBinWidth
        return discretize_image(image, mask;
                               binwidth=settings.binwidth,
                               bincount=nothing,
                               label=settings.label)
    else  # FixedBinCount
        isnothing(settings.bincount) && throw(ArgumentError(
            "bincount must be specified in Settings when using FixedBinCount mode"
        ))
        return discretize_image(image, mask;
                               binwidth=settings.binwidth,
                               bincount=settings.bincount,
                               label=settings.label)
    end
end

#==============================================================================#
# Bin Information Functions
#==============================================================================#

"""
    count_gray_levels(discretized_image, mask; label=1)

Count the number of distinct gray levels in the discretized ROI.

Useful for verifying discretization and computing texture matrices.
"""
function count_gray_levels(discretized_image::AbstractArray{<:Integer},
                           mask::AbstractArray;
                           label::Int=1)

    if eltype(mask) <: Integer
        bool_mask = mask .== label
    else
        bool_mask = mask
    end

    roi_values = discretized_image[bool_mask]
    return length(unique(roi_values))
end

"""
    gray_level_histogram(discretized_image, mask; label=1, nbins=nothing)

Compute histogram of gray levels in the discretized ROI.

# Arguments
- `discretized_image`: Discretized image from `discretize_image`
- `mask`: Mask array
- `label::Int=1`: Label for integer masks
- `nbins::Union{Int, Nothing}=nothing`: Number of bins (auto-detected if nothing)

# Returns
- `NamedTuple` with:
  - `counts`: Histogram counts for each gray level
  - `levels`: Gray level values (1:nbins)
  - `probabilities`: Normalized probabilities (counts / sum(counts))
"""
function gray_level_histogram(discretized_image::AbstractArray{<:Integer},
                              mask::AbstractArray;
                              label::Int=1,
                              nbins::Union{Int, Nothing}=nothing)

    if eltype(mask) <: Integer
        bool_mask = mask .== label
    else
        bool_mask = mask
    end

    roi_values = discretized_image[bool_mask]

    # Determine number of bins
    if isnothing(nbins)
        nbins = maximum(roi_values)
    end

    # Count occurrences of each gray level
    counts = zeros(Int, nbins)
    for val in roi_values
        if 1 <= val <= nbins
            counts[val] += 1
        end
    end

    total = sum(counts)
    probabilities = total > 0 ? counts ./ total : zeros(Float64, nbins)

    return (
        counts = counts,
        levels = collect(1:nbins),
        probabilities = probabilities
    )
end
