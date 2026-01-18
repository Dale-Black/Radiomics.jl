# Feature Extractor for Radiomics.jl
# Provides a unified interface for extracting radiomic features from medical images.

#==============================================================================#
# Feature Class Enumeration
#==============================================================================#

"""
    FeatureClass

Enumeration of available radiomic feature classes.

# Values
- `FirstOrder` - First-order statistical features (19 features)
- `Shape` - Shape/morphological features (2D or 3D based on image dimensionality)
- `GLCM` - Gray Level Co-occurrence Matrix features (24 features)
- `GLRLM` - Gray Level Run Length Matrix features (16 features)
- `GLSZM` - Gray Level Size Zone Matrix features (16 features)
- `NGTDM` - Neighboring Gray Tone Difference Matrix features (5 features)
- `GLDM` - Gray Level Dependence Matrix features (14 features)
"""
@enum FeatureClass begin
    FirstOrder
    Shape
    GLCM
    GLRLM
    GLSZM
    NGTDM
    GLDM
end

# All feature classes
const ALL_FEATURE_CLASSES = Set([FirstOrder, Shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM])

#==============================================================================#
# Feature Extractor Struct
#==============================================================================#

"""
    RadiomicsFeatureExtractor

Main feature extraction interface for Radiomics.jl.

This struct provides a unified way to extract radiomic features from images,
with support for enabling/disabling feature classes and configuring extraction
parameters.

# Fields
- `enabled_classes::Set{FeatureClass}` - Set of enabled feature classes
- `settings::Settings` - Extraction settings (binning, distance, etc.)

# Example
```julia
# Create extractor with default settings (all features enabled)
extractor = RadiomicsFeatureExtractor()

# Create extractor with specific feature classes
extractor = RadiomicsFeatureExtractor(
    enabled_classes=Set([FirstOrder, GLCM, GLRLM])
)

# Create extractor with custom settings
extractor = RadiomicsFeatureExtractor(
    settings=Settings(binwidth=32.0, glcm_distance=2)
)

# Extract features
image = rand(64, 64, 64)
mask = image .> 0.5
features = extract(extractor, image, mask)

# Features is a Dict{String, Float64}
println("Energy: ", features["firstorder_Energy"])
println("Contrast: ", features["glcm_Contrast"])
```

# See also
- [`extract`](@ref) - Extract features from image and mask
- [`enable!`](@ref) - Enable a feature class
- [`disable!`](@ref) - Disable a feature class
- [`Settings`](@ref) - Extraction settings
"""
mutable struct RadiomicsFeatureExtractor
    enabled_classes::Set{FeatureClass}
    settings::Settings
end

# Constructor with enabled classes only
function RadiomicsFeatureExtractor(enabled_classes::Set{FeatureClass})
    RadiomicsFeatureExtractor(enabled_classes, Settings())
end

# Constructor with settings only
function RadiomicsFeatureExtractor(settings::Settings)
    RadiomicsFeatureExtractor(copy(ALL_FEATURE_CLASSES), settings)
end

# Keyword argument constructor (also serves as default constructor when called with no args)
function RadiomicsFeatureExtractor(;
    enabled_classes::Union{Set{FeatureClass}, Vector{FeatureClass}, Nothing}=nothing,
    settings::Settings=Settings()
)
    classes = if isnothing(enabled_classes)
        copy(ALL_FEATURE_CLASSES)
    elseif enabled_classes isa Vector
        Set(enabled_classes)
    else
        enabled_classes
    end
    RadiomicsFeatureExtractor(classes, settings)
end

#==============================================================================#
# Enable/Disable Feature Classes
#==============================================================================#

"""
    enable!(extractor::RadiomicsFeatureExtractor, class::FeatureClass)

Enable a feature class for extraction.

# Arguments
- `extractor` - The feature extractor to modify
- `class` - The feature class to enable

# Returns
The modified extractor (for chaining).

# Example
```julia
extractor = RadiomicsFeatureExtractor(enabled_classes=Set{FeatureClass}())
enable!(extractor, FirstOrder)
enable!(extractor, GLCM)
```
"""
function enable!(extractor::RadiomicsFeatureExtractor, class::FeatureClass)
    push!(extractor.enabled_classes, class)
    return extractor
end

"""
    enable!(extractor::RadiomicsFeatureExtractor, classes::Vector{FeatureClass})

Enable multiple feature classes at once.
"""
function enable!(extractor::RadiomicsFeatureExtractor, classes::Vector{FeatureClass})
    for class in classes
        push!(extractor.enabled_classes, class)
    end
    return extractor
end

"""
    disable!(extractor::RadiomicsFeatureExtractor, class::FeatureClass)

Disable a feature class from extraction.

# Arguments
- `extractor` - The feature extractor to modify
- `class` - The feature class to disable

# Returns
The modified extractor (for chaining).

# Example
```julia
extractor = RadiomicsFeatureExtractor()
disable!(extractor, Shape)  # Skip shape features
```
"""
function disable!(extractor::RadiomicsFeatureExtractor, class::FeatureClass)
    delete!(extractor.enabled_classes, class)
    return extractor
end

"""
    disable!(extractor::RadiomicsFeatureExtractor, classes::Vector{FeatureClass})

Disable multiple feature classes at once.
"""
function disable!(extractor::RadiomicsFeatureExtractor, classes::Vector{FeatureClass})
    for class in classes
        delete!(extractor.enabled_classes, class)
    end
    return extractor
end

"""
    enable_all!(extractor::RadiomicsFeatureExtractor)

Enable all feature classes.
"""
function enable_all!(extractor::RadiomicsFeatureExtractor)
    extractor.enabled_classes = copy(ALL_FEATURE_CLASSES)
    return extractor
end

"""
    disable_all!(extractor::RadiomicsFeatureExtractor)

Disable all feature classes.
"""
function disable_all!(extractor::RadiomicsFeatureExtractor)
    empty!(extractor.enabled_classes)
    return extractor
end

"""
    is_enabled(extractor::RadiomicsFeatureExtractor, class::FeatureClass) -> Bool

Check if a feature class is enabled.
"""
function is_enabled(extractor::RadiomicsFeatureExtractor, class::FeatureClass)
    return class in extractor.enabled_classes
end

"""
    enabled_classes(extractor::RadiomicsFeatureExtractor) -> Vector{FeatureClass}

Get a sorted list of enabled feature classes.
"""
function enabled_classes(extractor::RadiomicsFeatureExtractor)
    return sort(collect(extractor.enabled_classes), by=x->Int(x))
end

#==============================================================================#
# Feature Extraction - Internal Helpers
#==============================================================================#

# Extract first-order features
function _extract_firstorder_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings,
    spacing::NTuple
)
    voxel_vol = prod(spacing)
    fo_features = extract_firstorder(image, mask;
        label=1,
        shift=settings.voxel_array_shift,
        voxel_volume=voxel_vol
    )
    for (name, value) in fo_features
        features["firstorder_$name"] = value
    end
end

# Extract shape features
function _extract_shape_features!(
    features::Dict{String, Float64},
    mask::AbstractArray{Bool},
    spacing::NTuple
)
    N = ndims(mask)
    if N == 2
        shape_features = extract_shape_2d(mask, spacing)
        for (name, value) in shape_features
            features["shape_$name"] = value
        end
    elseif N == 3
        shape_features = extract_shape_3d(mask, spacing)
        for (name, value) in shape_features
            features["shape_$name"] = value
        end
    end
end

# Extract GLCM features
function _extract_glcm_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings
)
    binwidth = settings.binwidth
    bincount = settings.bincount
    distance = settings.glcm_distance
    symmetric = settings.symmetrical_glcm

    glcm_result = compute_glcm(image, mask;
        binwidth=binwidth,
        bincount=bincount,
        distance=distance,
        symmetric=symmetric
    )

    glcm_feats = glcm_features(glcm_result)

    # Map NamedTuple fields to feature names (PyRadiomics style)
    feature_names = Dict(
        :autocorrelation => "Autocorrelation",
        :joint_average => "JointAverage",
        :cluster_prominence => "ClusterProminence",
        :cluster_shade => "ClusterShade",
        :cluster_tendency => "ClusterTendency",
        :contrast => "Contrast",
        :correlation => "Correlation",
        :difference_average => "DifferenceAverage",
        :difference_entropy => "DifferenceEntropy",
        :difference_variance => "DifferenceVariance",
        :joint_energy => "JointEnergy",
        :joint_entropy => "JointEntropy",
        :imc1 => "Imc1",
        :imc2 => "Imc2",
        :idm => "Idm",
        :idmn => "Idmn",
        :id => "Id",
        :idn => "Idn",
        :inverse_variance => "InverseVariance",
        :maximum_probability => "MaximumProbability",
        :sum_average => "SumAverage",
        :sum_entropy => "SumEntropy",
        :sum_squares => "SumSquares",
        :mcc => "MCC"
    )

    for (field, name) in feature_names
        features["glcm_$name"] = getfield(glcm_feats, field)
    end
end

# Extract GLRLM features
function _extract_glrlm_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings
)
    binwidth = settings.binwidth
    bincount = settings.bincount

    glrlm_result = compute_glrlm(image, mask;
        binwidth=binwidth,
        bincount=bincount
    )

    # Extract all 16 GLRLM features
    features["glrlm_ShortRunEmphasis"] = glrlm_short_run_emphasis(glrlm_result)
    features["glrlm_LongRunEmphasis"] = glrlm_long_run_emphasis(glrlm_result)
    features["glrlm_GrayLevelNonUniformity"] = glrlm_gray_level_non_uniformity(glrlm_result)
    features["glrlm_GrayLevelNonUniformityNormalized"] = glrlm_gray_level_non_uniformity_normalized(glrlm_result)
    features["glrlm_RunLengthNonUniformity"] = glrlm_run_length_non_uniformity(glrlm_result)
    features["glrlm_RunLengthNonUniformityNormalized"] = glrlm_run_length_non_uniformity_normalized(glrlm_result)
    features["glrlm_RunPercentage"] = glrlm_run_percentage(glrlm_result)
    features["glrlm_GrayLevelVariance"] = glrlm_gray_level_variance(glrlm_result)
    features["glrlm_RunVariance"] = glrlm_run_variance(glrlm_result)
    features["glrlm_RunEntropy"] = glrlm_run_entropy(glrlm_result)
    features["glrlm_LowGrayLevelRunEmphasis"] = glrlm_low_gray_level_run_emphasis(glrlm_result)
    features["glrlm_HighGrayLevelRunEmphasis"] = glrlm_high_gray_level_run_emphasis(glrlm_result)
    features["glrlm_ShortRunLowGrayLevelEmphasis"] = glrlm_short_run_low_gray_level_emphasis(glrlm_result)
    features["glrlm_ShortRunHighGrayLevelEmphasis"] = glrlm_short_run_high_gray_level_emphasis(glrlm_result)
    features["glrlm_LongRunLowGrayLevelEmphasis"] = glrlm_long_run_low_gray_level_emphasis(glrlm_result)
    features["glrlm_LongRunHighGrayLevelEmphasis"] = glrlm_long_run_high_gray_level_emphasis(glrlm_result)
end

# Extract GLSZM features
function _extract_glszm_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings
)
    binwidth = settings.binwidth
    bincount = settings.bincount

    glszm_feats = extract_glszm(image, mask;
        binwidth=binwidth,
        bincount=bincount
    )

    for (name, value) in glszm_feats
        features["glszm_$name"] = value
    end
end

# Extract NGTDM features
function _extract_ngtdm_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings
)
    binwidth = settings.binwidth
    bincount = settings.bincount
    distance = settings.ngtdm_distance

    ngtdm_feats = compute_all_ngtdm_features(image, mask;
        binwidth=binwidth,
        bincount=bincount,
        distance=distance
    )

    # Map NamedTuple fields to feature names
    features["ngtdm_Coarseness"] = ngtdm_feats.coarseness
    features["ngtdm_Contrast"] = ngtdm_feats.contrast
    features["ngtdm_Busyness"] = ngtdm_feats.busyness
    features["ngtdm_Complexity"] = ngtdm_feats.complexity
    features["ngtdm_Strength"] = ngtdm_feats.strength
end

# Extract GLDM features
function _extract_gldm_features!(
    features::Dict{String, Float64},
    image::AbstractArray{<:Real},
    mask::AbstractArray{Bool},
    settings::Settings
)
    binwidth = settings.binwidth
    bincount = settings.bincount
    alpha = Int(settings.gldm_alpha)  # Convert Float64 to Int (alpha must be integer)

    gldm_result = compute_gldm(image, mask;
        binwidth=binwidth,
        bincount=bincount,
        alpha=alpha
    )

    # Extract all 14 GLDM features
    features["gldm_SmallDependenceEmphasis"] = gldm_small_dependence_emphasis(gldm_result)
    features["gldm_LargeDependenceEmphasis"] = gldm_large_dependence_emphasis(gldm_result)
    features["gldm_GrayLevelNonUniformity"] = gldm_gray_level_non_uniformity(gldm_result)
    features["gldm_DependenceNonUniformity"] = gldm_dependence_non_uniformity(gldm_result)
    features["gldm_DependenceNonUniformityNormalized"] = gldm_dependence_non_uniformity_normalized(gldm_result)
    features["gldm_GrayLevelVariance"] = gldm_gray_level_variance(gldm_result)
    features["gldm_DependenceVariance"] = gldm_dependence_variance(gldm_result)
    features["gldm_DependenceEntropy"] = gldm_dependence_entropy(gldm_result)
    features["gldm_LowGrayLevelEmphasis"] = gldm_low_gray_level_emphasis(gldm_result)
    features["gldm_HighGrayLevelEmphasis"] = gldm_high_gray_level_emphasis(gldm_result)
    features["gldm_SmallDependenceLowGrayLevelEmphasis"] = gldm_small_dependence_low_gray_level_emphasis(gldm_result)
    features["gldm_SmallDependenceHighGrayLevelEmphasis"] = gldm_small_dependence_high_gray_level_emphasis(gldm_result)
    features["gldm_LargeDependenceLowGrayLevelEmphasis"] = gldm_large_dependence_low_gray_level_emphasis(gldm_result)
    features["gldm_LargeDependenceHighGrayLevelEmphasis"] = gldm_large_dependence_high_gray_level_emphasis(gldm_result)
end

#==============================================================================#
# Main Extract Function
#==============================================================================#

"""
    extract(extractor::RadiomicsFeatureExtractor, image, mask;
            spacing=nothing) -> Dict{String, Float64}

Extract radiomic features from an image using the configured extractor.

# Arguments
- `extractor` - Configured feature extractor
- `image` - Image array (2D or 3D)
- `mask` - Boolean mask defining the ROI
- `spacing` - Voxel spacing tuple (optional, defaults to (1.0, 1.0, ...) if not provided)

# Returns
A `Dict{String, Float64}` containing all extracted features.
Feature names follow the pattern `"{class}_{feature}"`, e.g.:
- `"firstorder_Energy"`
- `"glcm_Contrast"`
- `"shape_Sphericity"`

# Example
```julia
extractor = RadiomicsFeatureExtractor()
image = rand(64, 64, 64)
mask = image .> 0.5

# Extract with default spacing
features = extract(extractor, image, mask)

# Extract with custom spacing
features = extract(extractor, image, mask; spacing=(1.0, 1.0, 2.5))

# Access specific features
println("Energy: ", features["firstorder_Energy"])
```

# Notes
- Only enabled feature classes are extracted
- The mask should be a Boolean array with the same dimensions as the image
- For texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM), the image is discretized
  using the settings in the extractor
"""
function extract(
    extractor::RadiomicsFeatureExtractor,
    image::AbstractArray{T, N},
    mask::AbstractArray{Bool, N};
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing
) where {T<:Real, N}

    # Validate inputs
    size(image) == size(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(size(image)) and $(size(mask))"
    ))
    N in (2, 3) || throw(ArgumentError(
        "Only 2D and 3D images are supported, got $(N)D"
    ))

    # Use default spacing if not provided
    actual_spacing = isnothing(spacing) ? ntuple(_ -> 1.0, N) : spacing

    # Initialize features dictionary
    features = Dict{String, Float64}()
    settings = extractor.settings

    # Extract enabled feature classes
    if is_enabled(extractor, FirstOrder)
        _extract_firstorder_features!(features, image, mask, settings, actual_spacing)
    end

    if is_enabled(extractor, Shape)
        _extract_shape_features!(features, mask, actual_spacing)
    end

    if is_enabled(extractor, GLCM)
        _extract_glcm_features!(features, image, mask, settings)
    end

    if is_enabled(extractor, GLRLM)
        _extract_glrlm_features!(features, image, mask, settings)
    end

    if is_enabled(extractor, GLSZM)
        _extract_glszm_features!(features, image, mask, settings)
    end

    if is_enabled(extractor, NGTDM)
        _extract_ngtdm_features!(features, image, mask, settings)
    end

    if is_enabled(extractor, GLDM)
        _extract_gldm_features!(features, image, mask, settings)
    end

    return features
end

# Support for RadiomicsImage type
function extract(
    extractor::RadiomicsFeatureExtractor,
    image::RadiomicsImage{T, N},
    mask::AbstractArray{Bool, N}
) where {T<:Real, N}
    return extract(extractor, image.data, mask; spacing=image.spacing)
end

# Support for integer masks (convert to Bool)
function extract(
    extractor::RadiomicsFeatureExtractor,
    image::AbstractArray{T, N},
    mask::AbstractArray{<:Integer, N};
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    label::Int=1
) where {T<:Real, N}
    bool_mask = mask .== label
    return extract(extractor, image, bool_mask; spacing=spacing)
end

#==============================================================================#
# Convenience Functions
#==============================================================================#

"""
    extract_all(image, mask; kwargs...) -> Dict{String, Float64}

Convenience function to extract all features with default settings.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI
- `spacing` - Voxel spacing (optional)
- `binwidth` - Bin width for discretization (default: 25.0)
- `bincount` - Fixed bin count (overrides binwidth)
- `glcm_distance` - Distance for GLCM (default: 1)
- `gldm_alpha` - Alpha for GLDM (default: 0.0)
- `ngtdm_distance` - Distance for NGTDM (default: 1)
- `label` - Mask label value (default: 1, for integer masks)

# Returns
Dict{String, Float64} with all radiomic features.

# Example
```julia
features = extract_all(image, mask)
features = extract_all(image, mask; binwidth=32.0)
```
"""
function extract_all(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    glcm_distance::Int=1,
    gldm_alpha::Real=0.0,
    ngtdm_distance::Int=1,
    label::Int=1
) where {T<:Real, N}

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount,
        glcm_distance=glcm_distance,
        gldm_alpha=Float64(gldm_alpha),
        ngtdm_distance=ngtdm_distance
    )

    extractor = RadiomicsFeatureExtractor(settings)

    # Convert mask to Bool if needed
    if eltype(mask) <: Integer
        bool_mask = mask .== label
    elseif eltype(mask) <: Bool
        bool_mask = mask
    else
        throw(ArgumentError("Mask must be Boolean or Integer array"))
    end

    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_firstorder_only(image, mask; kwargs...) -> Dict{String, Float64}

Extract only first-order features.
"""
function extract_firstorder_only(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    label::Int=1
) where {T<:Real, N}

    extractor = RadiomicsFeatureExtractor(enabled_classes=Set([FirstOrder]))

    if eltype(mask) <: Integer
        bool_mask = mask .== label
    else
        bool_mask = mask
    end

    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_shape_only(mask; spacing=nothing) -> Dict{String, Float64}

Extract only shape features.
"""
function extract_shape_only(
    mask::AbstractArray{T, N};
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    label::Int=1
) where {T, N}

    # Need a dummy image for the extractor (not used for shape)
    if T <: Bool
        bool_mask = mask
    elseif T <: Integer
        bool_mask = mask .== label
    else
        throw(ArgumentError("Mask must be Boolean or Integer array"))
    end

    actual_spacing = isnothing(spacing) ? ntuple(_ -> 1.0, N) : spacing

    features = Dict{String, Float64}()
    _extract_shape_features!(features, bool_mask, actual_spacing)
    return features
end

"""
    extract_texture_only(image, mask; kwargs...) -> Dict{String, Float64}

Extract only texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM).
"""
function extract_texture_only(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    glcm_distance::Int=1,
    gldm_alpha::Real=0.0,
    ngtdm_distance::Int=1,
    label::Int=1
) where {T<:Real, N}

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount,
        glcm_distance=glcm_distance,
        gldm_alpha=Float64(gldm_alpha),
        ngtdm_distance=ngtdm_distance
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([GLCM, GLRLM, GLSZM, NGTDM, GLDM]),
        settings=settings
    )

    if eltype(mask) <: Integer
        bool_mask = mask .== label
    else
        bool_mask = mask
    end

    return extract(extractor, image, bool_mask; spacing=spacing)
end

#==============================================================================#
# Display Methods
#==============================================================================#

function Base.show(io::IO, extractor::RadiomicsFeatureExtractor)
    n = length(extractor.enabled_classes)
    classes_str = join(sort([string(c) for c in extractor.enabled_classes]), ", ")
    print(io, "RadiomicsFeatureExtractor($n classes: $classes_str)")
end

function Base.show(io::IO, ::MIME"text/plain", extractor::RadiomicsFeatureExtractor)
    println(io, "RadiomicsFeatureExtractor")
    println(io, "  Enabled classes ($(length(extractor.enabled_classes))):")
    for class in sort(collect(extractor.enabled_classes), by=x->Int(x))
        println(io, "    - $class")
    end
    println(io, "  Settings:")
    println(io, "    binwidth: $(extractor.settings.binwidth)")
    if !isnothing(extractor.settings.bincount)
        println(io, "    bincount: $(extractor.settings.bincount)")
    end
    println(io, "    glcm_distance: $(extractor.settings.glcm_distance)")
    println(io, "    symmetrical_glcm: $(extractor.settings.symmetrical_glcm)")
    println(io, "    gldm_alpha: $(extractor.settings.gldm_alpha)")
    println(io, "    ngtdm_distance: $(extractor.settings.ngtdm_distance)")
end

#==============================================================================#
# Feature Count Utilities
#==============================================================================#

"""
    feature_count(class::FeatureClass) -> Int

Return the number of features in a feature class.
"""
function feature_count(class::FeatureClass)
    counts = Dict(
        FirstOrder => 19,
        Shape => 18,  # Mix of 2D (10) and 3D (18) - use max
        GLCM => 24,
        GLRLM => 16,
        GLSZM => 16,
        NGTDM => 5,
        GLDM => 14
    )
    return counts[class]
end

"""
    total_feature_count(extractor::RadiomicsFeatureExtractor; is_3d::Bool=true) -> Int

Return the total number of features that will be extracted.
"""
function total_feature_count(extractor::RadiomicsFeatureExtractor; is_3d::Bool=true)
    total = 0
    for class in extractor.enabled_classes
        if class == Shape
            total += is_3d ? 18 : 10
        else
            total += feature_count(class)
        end
    end
    return total
end

#==============================================================================#
# Feature Names Utilities
#==============================================================================#

"""
    feature_names(class::FeatureClass) -> Vector{String}

Return the names of all features in a feature class.
"""
function feature_names(class::FeatureClass)
    if class == FirstOrder
        return ["Energy", "TotalEnergy", "Entropy", "Minimum", "10Percentile",
                "90Percentile", "Maximum", "Mean", "Median", "InterquartileRange",
                "Range", "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation",
                "RootMeanSquared", "StandardDeviation", "Skewness", "Kurtosis",
                "Variance", "Uniformity"]
    elseif class == Shape
        return ["MeshVolume", "VoxelVolume", "SurfaceArea", "SurfaceVolumeRatio",
                "Sphericity", "Compactness1", "Compactness2", "SphericalDisproportion",
                "Maximum3DDiameter", "Maximum2DDiameterSlice", "Maximum2DDiameterColumn",
                "Maximum2DDiameterRow", "MajorAxisLength", "MinorAxisLength",
                "LeastAxisLength", "Elongation", "Flatness"]
    elseif class == GLCM
        return ["Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
                "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
                "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
                "Imc1", "Imc2", "Idm", "Idmn", "Id", "Idn", "InverseVariance",
                "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares", "MCC"]
    elseif class == GLRLM
        return ["ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
                "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity",
                "RunLengthNonUniformityNormalized", "RunPercentage", "GrayLevelVariance",
                "RunVariance", "RunEntropy", "LowGrayLevelRunEmphasis",
                "HighGrayLevelRunEmphasis", "ShortRunLowGrayLevelEmphasis",
                "ShortRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
                "LongRunHighGrayLevelEmphasis"]
    elseif class == GLSZM
        return ["SmallAreaEmphasis", "LargeAreaEmphasis", "GrayLevelNonUniformity",
                "GrayLevelNonUniformityNormalized", "SizeZoneNonUniformity",
                "SizeZoneNonUniformityNormalized", "ZonePercentage", "GrayLevelVariance",
                "ZoneVariance", "ZoneEntropy", "LowGrayLevelZoneEmphasis",
                "HighGrayLevelZoneEmphasis", "SmallAreaLowGrayLevelEmphasis",
                "SmallAreaHighGrayLevelEmphasis", "LargeAreaLowGrayLevelEmphasis",
                "LargeAreaHighGrayLevelEmphasis"]
    elseif class == NGTDM
        return ["Coarseness", "Contrast", "Busyness", "Complexity", "Strength"]
    elseif class == GLDM
        return ["SmallDependenceEmphasis", "LargeDependenceEmphasis",
                "GrayLevelNonUniformity", "DependenceNonUniformity",
                "DependenceNonUniformityNormalized", "GrayLevelVariance",
                "DependenceVariance", "DependenceEntropy", "LowGrayLevelEmphasis",
                "HighGrayLevelEmphasis", "SmallDependenceLowGrayLevelEmphasis",
                "SmallDependenceHighGrayLevelEmphasis", "LargeDependenceLowGrayLevelEmphasis",
                "LargeDependenceHighGrayLevelEmphasis"]
    else
        return String[]
    end
end

"""
    all_feature_names(extractor::RadiomicsFeatureExtractor) -> Vector{String}

Return all feature names that will be extracted by this extractor.
"""
function all_feature_names(extractor::RadiomicsFeatureExtractor)
    names = String[]
    for class in sort(collect(extractor.enabled_classes), by=x->Int(x))
        prefix = lowercase(string(class))
        for name in feature_names(class)
            push!(names, "$(prefix)_$name")
        end
    end
    return names
end
