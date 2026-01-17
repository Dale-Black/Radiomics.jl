# Core types for Radiomics.jl
# These types form the foundation of the feature extraction system.

#==============================================================================#
# Abstract Types
#==============================================================================#

"""
    AbstractRadiomicsFeature

Abstract base type for all radiomic feature computations.

All feature classes (FirstOrder, Shape, GLCM, etc.) inherit from this type
to enable dispatch-based feature extraction.

# Subtypes
- `AbstractFirstOrderFeature` - First-order statistical features
- `AbstractShapeFeature` - Shape/morphological features
- `AbstractTextureFeature` - Texture matrix features (GLCM, GLRLM, etc.)
"""
abstract type AbstractRadiomicsFeature end

"""
    AbstractFirstOrderFeature <: AbstractRadiomicsFeature

Abstract type for first-order statistical features.

These features are computed directly from voxel intensities without
considering spatial relationships.
"""
abstract type AbstractFirstOrderFeature <: AbstractRadiomicsFeature end

"""
    AbstractShapeFeature <: AbstractRadiomicsFeature

Abstract type for shape/morphological features.

These features describe the geometric properties of the ROI,
independent of voxel intensities.
"""
abstract type AbstractShapeFeature <: AbstractRadiomicsFeature end

"""
    AbstractTextureFeature <: AbstractRadiomicsFeature

Abstract type for texture features computed from texture matrices.

# Subtypes
- `AbstractGLCMFeature` - Gray Level Co-occurrence Matrix features
- `AbstractGLRLMFeature` - Gray Level Run Length Matrix features
- `AbstractGLSZMFeature` - Gray Level Size Zone Matrix features
- `AbstractNGTDMFeature` - Neighboring Gray Tone Difference Matrix features
- `AbstractGLDMFeature` - Gray Level Dependence Matrix features
"""
abstract type AbstractTextureFeature <: AbstractRadiomicsFeature end

abstract type AbstractGLCMFeature <: AbstractTextureFeature end
abstract type AbstractGLRLMFeature <: AbstractTextureFeature end
abstract type AbstractGLSZMFeature <: AbstractTextureFeature end
abstract type AbstractNGTDMFeature <: AbstractTextureFeature end
abstract type AbstractGLDMFeature <: AbstractTextureFeature end

#==============================================================================#
# Settings Type
#==============================================================================#

"""
    DiscretizationMode

Enum for gray level discretization modes.

# Values
- `FixedBinWidth`: Bins determined by (max - min) / binWidth (default in PyRadiomics)
- `FixedBinCount`: Fixed number of equal-width bins
"""
@enum DiscretizationMode begin
    FixedBinWidth
    FixedBinCount
end

"""
    Settings

Configuration settings for radiomic feature extraction.

# Fields
- `binwidth::Float64`: Bin width for fixed bin width discretization (default: 25.0)
- `bincount::Union{Int, Nothing}`: Number of bins for fixed bin count mode (default: nothing)
- `discretization_mode::DiscretizationMode`: Discretization mode (default: FixedBinWidth)
- `label::Int`: Label value in mask to use as ROI (default: 1)
- `resample_spacing::Union{NTuple{3, Float64}, Nothing}`: Target voxel spacing for resampling (default: nothing)
- `normalize::Bool`: Whether to normalize intensity values (default: false)
- `normalize_scale::Float64`: Scale factor for normalization (default: 1.0)
- `remove_outliers::Bool`: Whether to remove intensity outliers (default: false)
- `outlier_percentile::Float64`: Percentile for outlier removal (default: 99.0)
- `force_2d::Bool`: Force 2D feature extraction (default: false)
- `force_2d_dimension::Int`: Dimension to treat as slice in 2D mode (default: 3)
- `glcm_distance::Int`: Distance for GLCM computation (default: 1)
- `symmetrical_glcm::Bool`: Whether GLCM should be symmetrical (default: true)
- `gldm_alpha::Float64`: Alpha parameter for GLDM (default: 0.0)
- `ngtdm_distance::Int`: Distance for NGTDM neighborhood (default: 1)
- `preallocate::Bool`: Whether to preallocate arrays for performance (default: true)
- `voxel_array_shift::Int`: Shift for voxel values after discretization (default: 0, PyRadiomics uses 1)

# Example
```julia
# Default settings (matches PyRadiomics defaults)
settings = Settings()

# Custom bin width
settings = Settings(binwidth=32.0)

# Fixed bin count mode
settings = Settings(bincount=64, discretization_mode=FixedBinCount)
```

# References
- PyRadiomics settings: https://pyradiomics.readthedocs.io/en/latest/customization.html
"""
Base.@kwdef struct Settings
    # Discretization
    binwidth::Float64 = 25.0
    bincount::Union{Int, Nothing} = nothing
    discretization_mode::DiscretizationMode = FixedBinWidth

    # Mask/Label
    label::Int = 1

    # Preprocessing
    resample_spacing::Union{NTuple{3, Float64}, Nothing} = nothing
    normalize::Bool = false
    normalize_scale::Float64 = 1.0
    remove_outliers::Bool = false
    outlier_percentile::Float64 = 99.0

    # 2D/3D handling
    force_2d::Bool = false
    force_2d_dimension::Int = 3

    # Texture matrix parameters
    glcm_distance::Int = 1
    symmetrical_glcm::Bool = true
    gldm_alpha::Float64 = 0.0
    ngtdm_distance::Int = 1

    # Performance
    preallocate::Bool = true

    # PyRadiomics compatibility
    voxel_array_shift::Int = 0
end

# Validation function for Settings
function validate_settings(s::Settings)
    s.binwidth > 0 || throw(ArgumentError("binwidth must be positive, got $(s.binwidth)"))

    if s.discretization_mode == FixedBinCount
        isnothing(s.bincount) && throw(ArgumentError(
            "bincount must be specified when using FixedBinCount mode"
        ))
        s.bincount > 0 || throw(ArgumentError("bincount must be positive, got $(s.bincount)"))
    end

    s.label > 0 || throw(ArgumentError("label must be positive, got $(s.label)"))
    s.glcm_distance > 0 || throw(ArgumentError("glcm_distance must be positive, got $(s.glcm_distance)"))
    s.ngtdm_distance > 0 || throw(ArgumentError("ngtdm_distance must be positive, got $(s.ngtdm_distance)"))
    0.0 <= s.outlier_percentile <= 100.0 || throw(ArgumentError(
        "outlier_percentile must be between 0 and 100, got $(s.outlier_percentile)"
    ))
    1 <= s.force_2d_dimension <= 3 || throw(ArgumentError(
        "force_2d_dimension must be 1, 2, or 3, got $(s.force_2d_dimension)"
    ))

    return true
end

#==============================================================================#
# Image Wrapper Type
#==============================================================================#

"""
    RadiomicsImage{T<:Real, N}

Wrapper type for medical images with associated metadata.

This type wraps an N-dimensional array and stores voxel spacing information.
Using this wrapper is optional - feature extraction functions also accept
plain AbstractArrays with default spacing.

# Type Parameters
- `T`: Element type (must be a Real number)
- `N`: Number of dimensions (typically 2 or 3)

# Fields
- `data::Array{T, N}`: The image intensity data
- `spacing::NTuple{N, Float64}`: Voxel spacing in each dimension (mm)

# Example
```julia
# Create from array with spacing
data = rand(Float64, 64, 64, 64)
img = RadiomicsImage(data, (1.0, 1.0, 2.5))  # 1mm x 1mm x 2.5mm voxels

# Access data
img.data[32, 32, 32]  # Get voxel value
img.spacing           # (1.0, 1.0, 2.5)
```
"""
struct RadiomicsImage{T<:Real, N}
    data::Array{T, N}
    spacing::NTuple{N, Float64}

    function RadiomicsImage(data::Array{T, N}, spacing::NTuple{N, Float64}) where {T<:Real, N}
        N in (2, 3) || throw(ArgumentError(
            "RadiomicsImage only supports 2D or 3D arrays, got $(N)D"
        ))
        all(s -> s > 0, spacing) || throw(ArgumentError(
            "All spacing values must be positive, got $spacing"
        ))
        new{T, N}(data, spacing)
    end
end

# Convenience constructor with default spacing (1.0 for all dimensions)
function RadiomicsImage(data::Array{T, N}) where {T<:Real, N}
    spacing = ntuple(_ -> 1.0, N)
    RadiomicsImage(data, spacing)
end

# Allow AbstractArray access patterns
Base.size(img::RadiomicsImage) = size(img.data)
Base.size(img::RadiomicsImage, d) = size(img.data, d)
Base.ndims(img::RadiomicsImage) = ndims(img.data)
Base.eltype(img::RadiomicsImage) = eltype(img.data)
Base.getindex(img::RadiomicsImage, inds...) = getindex(img.data, inds...)
Base.setindex!(img::RadiomicsImage, v, inds...) = setindex!(img.data, v, inds...)

#==============================================================================#
# Mask Type
#==============================================================================#

"""
    RadiomicsMask{N}

Wrapper type for segmentation masks with metadata.

A mask defines the Region of Interest (ROI) for feature extraction.
Voxels with value equal to `label` are included in the ROI.

# Type Parameters
- `N`: Number of dimensions (must match image dimensions)

# Fields
- `data::Array{Int, N}`: The mask data (integer labels)
- `label::Int`: The label value that defines the ROI (default: 1)

# Example
```julia
# Create binary mask
mask_data = zeros(Int, 64, 64, 64)
mask_data[20:40, 20:40, 20:40] .= 1
mask = RadiomicsMask(mask_data)

# Multi-label mask
mask_data[10:20, 10:20, 10:20] .= 2
mask = RadiomicsMask(mask_data, label=2)  # Use label 2 as ROI
```
"""
struct RadiomicsMask{N}
    data::Array{Int, N}
    label::Int

    function RadiomicsMask(data::Array{Int, N}, label::Int=1) where N
        N in (2, 3) || throw(ArgumentError(
            "RadiomicsMask only supports 2D or 3D arrays, got $(N)D"
        ))
        label > 0 || throw(ArgumentError("label must be positive, got $label"))
        new{N}(data, label)
    end
end

# Constructor from Bool array
function RadiomicsMask(data::Array{Bool, N}) where N
    RadiomicsMask(Int.(data), 1)
end

# Constructor from BitArray
function RadiomicsMask(data::BitArray{N}) where N
    RadiomicsMask(Int.(data), 1)
end

# Allow AbstractArray access patterns
Base.size(mask::RadiomicsMask) = size(mask.data)
Base.size(mask::RadiomicsMask, d) = size(mask.data, d)
Base.ndims(mask::RadiomicsMask) = ndims(mask.data)
Base.eltype(mask::RadiomicsMask) = eltype(mask.data)
Base.getindex(mask::RadiomicsMask, inds...) = getindex(mask.data, inds...)
Base.setindex!(mask::RadiomicsMask, v, inds...) = setindex!(mask.data, v, inds...)

"""
    get_roi_mask(mask::RadiomicsMask) -> BitArray

Return a boolean mask where true indicates voxels in the ROI.

# Example
```julia
mask = RadiomicsMask(mask_data, label=2)
roi = get_roi_mask(mask)  # BitArray where mask_data .== 2
```
"""
function get_roi_mask(mask::RadiomicsMask)
    return mask.data .== mask.label
end

"""
    get_roi_mask(mask::AbstractArray{<:Integer}, label::Int=1) -> BitArray

Return a boolean mask from an integer array using specified label.
"""
function get_roi_mask(mask::AbstractArray{<:Integer}, label::Int=1)
    return mask .== label
end

"""
    get_roi_mask(mask::AbstractArray{Bool}) -> AbstractArray{Bool}

Return the boolean mask as-is.
"""
get_roi_mask(mask::AbstractArray{Bool}) = mask

#==============================================================================#
# Feature Result Type
#==============================================================================#

"""
    FeatureResult

Container for a single radiomic feature computation result.

# Fields
- `name::String`: Feature name (e.g., "Energy", "Entropy")
- `value::Float64`: Computed feature value
- `feature_class::String`: Feature class (e.g., "firstorder", "glcm")
- `image_type::String`: Image type used (e.g., "original", "wavelet")

# Example
```julia
result = FeatureResult("Energy", 12345.6, "firstorder", "original")
println(result)  # "firstorder_Energy: 12345.6"
```
"""
struct FeatureResult
    name::String
    value::Float64
    feature_class::String
    image_type::String
end

# Convenience constructor with defaults
FeatureResult(name::String, value::Float64, feature_class::String) =
    FeatureResult(name, value, feature_class, "original")

FeatureResult(name::String, value::Float64) =
    FeatureResult(name, value, "unknown", "original")

# Pretty printing
function Base.show(io::IO, r::FeatureResult)
    if r.image_type == "original"
        print(io, "$(r.feature_class)_$(r.name): $(r.value)")
    else
        print(io, "$(r.image_type)_$(r.feature_class)_$(r.name): $(r.value)")
    end
end

#==============================================================================#
# Feature Set Type
#==============================================================================#

"""
    FeatureSet

Collection of computed radiomic features.

Stores multiple `FeatureResult` objects and provides dictionary-like access.

# Fields
- `features::Vector{FeatureResult}`: Vector of feature results
- `settings::Settings`: Settings used for computation

# Example
```julia
fs = FeatureSet()
push!(fs, FeatureResult("Energy", 12345.6, "firstorder"))
push!(fs, FeatureResult("Entropy", 5.43, "firstorder"))

# Dictionary-like access
fs["firstorder_Energy"]  # 12345.6

# Convert to Dict
d = Dict(fs)
```
"""
mutable struct FeatureSet
    features::Vector{FeatureResult}
    settings::Settings

    FeatureSet() = new(FeatureResult[], Settings())
    FeatureSet(settings::Settings) = new(FeatureResult[], settings)
end

Base.push!(fs::FeatureSet, r::FeatureResult) = push!(fs.features, r)
Base.length(fs::FeatureSet) = length(fs.features)
Base.isempty(fs::FeatureSet) = isempty(fs.features)
Base.iterate(fs::FeatureSet) = iterate(fs.features)
Base.iterate(fs::FeatureSet, state) = iterate(fs.features, state)

# Feature name generation
function feature_key(r::FeatureResult)
    if r.image_type == "original"
        return "$(r.feature_class)_$(r.name)"
    else
        return "$(r.image_type)_$(r.feature_class)_$(r.name)"
    end
end

# Dictionary-like access
function Base.getindex(fs::FeatureSet, key::String)
    for r in fs.features
        if feature_key(r) == key
            return r.value
        end
    end
    throw(KeyError(key))
end

function Base.haskey(fs::FeatureSet, key::String)
    for r in fs.features
        if feature_key(r) == key
            return true
        end
    end
    return false
end

function Base.keys(fs::FeatureSet)
    return [feature_key(r) for r in fs.features]
end

function Base.values(fs::FeatureSet)
    return [r.value for r in fs.features]
end

function Base.Dict(fs::FeatureSet)
    return Dict(feature_key(r) => r.value for r in fs.features)
end

# Pretty printing
function Base.show(io::IO, fs::FeatureSet)
    n = length(fs.features)
    print(io, "FeatureSet with $n feature$(n == 1 ? "" : "s")")
end

function Base.show(io::IO, ::MIME"text/plain", fs::FeatureSet)
    n = length(fs.features)
    println(io, "FeatureSet with $n feature$(n == 1 ? "" : "s"):")
    for (i, r) in enumerate(fs.features)
        if i <= 10
            println(io, "  ", r)
        elseif i == 11
            println(io, "  ... and $(n - 10) more features")
            break
        end
    end
end

#==============================================================================#
# Type aliases for convenience
#==============================================================================#

"""
    ImageLike

Type alias for any array that can be used as an image.
"""
const ImageLike{T, N} = Union{Array{T, N}, RadiomicsImage{T, N}} where {T<:Real, N}

"""
    MaskLike

Type alias for any array that can be used as a mask.
"""
const MaskLike{N} = Union{Array{Bool, N}, BitArray{N}, Array{<:Integer, N}, RadiomicsMask{N}} where N

#==============================================================================#
# Utility Functions for Types
#==============================================================================#

"""
    get_data(img::RadiomicsImage) -> Array

Extract the raw data array from a RadiomicsImage.
"""
get_data(img::RadiomicsImage) = img.data

"""
    get_data(img::AbstractArray) -> AbstractArray

Return the array as-is when not wrapped.
"""
get_data(img::AbstractArray) = img

"""
    get_spacing(img::RadiomicsImage) -> NTuple

Get the voxel spacing from a RadiomicsImage.
"""
get_spacing(img::RadiomicsImage) = img.spacing

"""
    get_spacing(img::AbstractArray{T, N}) -> NTuple{N, Float64}

Return default unit spacing for plain arrays.
"""
get_spacing(img::AbstractArray{T, N}) where {T, N} = ntuple(_ -> 1.0, N)

"""
    get_mask_data(mask::RadiomicsMask) -> Array{Int}

Extract the raw data array from a RadiomicsMask.
"""
get_mask_data(mask::RadiomicsMask) = mask.data

"""
    get_mask_data(mask::AbstractArray) -> AbstractArray

Return the array as-is when not wrapped.
"""
get_mask_data(mask::AbstractArray) = mask
