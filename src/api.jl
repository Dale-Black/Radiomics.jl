# High-Level API for Radiomics.jl
#
# This module provides user-friendly convenience functions for feature extraction,
# image loading support, and enhanced error handling.
#
# The functions here complement the RadiomicsFeatureExtractor class by providing
# simple one-liner interfaces for common use cases.

#==============================================================================#
# Per-Class Extraction Helpers
#==============================================================================#

"""
    extract_glcm(image, mask; kwargs...) -> Dict{String, Float64}

Extract only GLCM (Gray Level Co-occurrence Matrix) features.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI

# Keyword Arguments
- `spacing` - Voxel spacing tuple (optional, defaults to (1.0, 1.0, ...))
- `binwidth::Real=25.0` - Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing` - Fixed bin count (overrides binwidth)
- `distance::Int=1` - Distance for GLCM computation
- `symmetric::Bool=true` - Whether to make GLCM symmetric
- `label::Int=1` - Mask label value (for integer masks)

# Returns
`Dict{String, Float64}` with 24 GLCM features.

# Example
```julia
image = rand(32, 32, 32)
mask = image .> 0.5
features = extract_glcm(image, mask)
println(features["glcm_Contrast"])
```

# See also
- [`extract_all`](@ref) - Extract all feature classes
- [`compute_glcm`](@ref) - Low-level GLCM matrix computation
"""
function extract_glcm(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    distance::Int=1,
    symmetric::Bool=true,
    label::Int=1
) where {T<:Real, N}

    # Validate inputs
    _validate_image_mask_api(image, mask, N)

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount,
        glcm_distance=distance,
        symmetrical_glcm=symmetric
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([GLCM]),
        settings=settings
    )

    bool_mask = _to_bool_mask(mask, label)
    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_glrlm(image, mask; kwargs...) -> Dict{String, Float64}

Extract only GLRLM (Gray Level Run Length Matrix) features.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI

# Keyword Arguments
- `spacing` - Voxel spacing tuple (optional)
- `binwidth::Real=25.0` - Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing` - Fixed bin count (overrides binwidth)
- `label::Int=1` - Mask label value (for integer masks)

# Returns
`Dict{String, Float64}` with 16 GLRLM features.

# Example
```julia
features = extract_glrlm(image, mask; binwidth=32.0)
println(features["glrlm_ShortRunEmphasis"])
```

# See also
- [`extract_all`](@ref) - Extract all feature classes
- [`compute_glrlm`](@ref) - Low-level GLRLM matrix computation
"""
function extract_glrlm(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    label::Int=1
) where {T<:Real, N}

    _validate_image_mask_api(image, mask, N)

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([GLRLM]),
        settings=settings
    )

    bool_mask = _to_bool_mask(mask, label)
    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_glszm(image, mask; kwargs...) -> Dict{String, Float64}

Extract only GLSZM (Gray Level Size Zone Matrix) features.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI

# Keyword Arguments
- `spacing` - Voxel spacing tuple (optional)
- `binwidth::Real=25.0` - Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing` - Fixed bin count (overrides binwidth)
- `label::Int=1` - Mask label value (for integer masks)

# Returns
`Dict{String, Float64}` with 16 GLSZM features.

# Example
```julia
features = extract_glszm(image, mask)
println(features["glszm_SmallAreaEmphasis"])
```

# See also
- [`extract_all`](@ref) - Extract all feature classes
- [`compute_glszm`](@ref) - Low-level GLSZM matrix computation
"""
function extract_glszm(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    label::Int=1
) where {T<:Real, N}

    _validate_image_mask_api(image, mask, N)

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([GLSZM]),
        settings=settings
    )

    bool_mask = _to_bool_mask(mask, label)
    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_ngtdm(image, mask; kwargs...) -> Dict{String, Float64}

Extract only NGTDM (Neighboring Gray Tone Difference Matrix) features.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI

# Keyword Arguments
- `spacing` - Voxel spacing tuple (optional)
- `binwidth::Real=25.0` - Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing` - Fixed bin count (overrides binwidth)
- `distance::Int=1` - Neighborhood distance
- `label::Int=1` - Mask label value (for integer masks)

# Returns
`Dict{String, Float64}` with 5 NGTDM features.

# Example
```julia
features = extract_ngtdm(image, mask)
println(features["ngtdm_Coarseness"])
```

# See also
- [`extract_all`](@ref) - Extract all feature classes
- [`compute_ngtdm`](@ref) - Low-level NGTDM computation
"""
function extract_ngtdm(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    distance::Int=1,
    label::Int=1
) where {T<:Real, N}

    _validate_image_mask_api(image, mask, N)

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount,
        ngtdm_distance=distance
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([NGTDM]),
        settings=settings
    )

    bool_mask = _to_bool_mask(mask, label)
    return extract(extractor, image, bool_mask; spacing=spacing)
end

"""
    extract_gldm(image, mask; kwargs...) -> Dict{String, Float64}

Extract only GLDM (Gray Level Dependence Matrix) features.

# Arguments
- `image` - Image array (2D or 3D)
- `mask` - Boolean or integer mask defining the ROI

# Keyword Arguments
- `spacing` - Voxel spacing tuple (optional)
- `binwidth::Real=25.0` - Bin width for discretization
- `bincount::Union{Int, Nothing}=nothing` - Fixed bin count (overrides binwidth)
- `alpha::Real=0.0` - Coarseness parameter (0 = strict equality)
- `label::Int=1` - Mask label value (for integer masks)

# Returns
`Dict{String, Float64}` with 14 GLDM features.

# Example
```julia
features = extract_gldm(image, mask; alpha=0)
println(features["gldm_SmallDependenceEmphasis"])
```

# See also
- [`extract_all`](@ref) - Extract all feature classes
- [`compute_gldm`](@ref) - Low-level GLDM computation
"""
function extract_gldm(
    image::AbstractArray{T, N},
    mask::AbstractArray;
    spacing::Union{NTuple{N, <:Real}, Nothing}=nothing,
    binwidth::Real=25.0,
    bincount::Union{Int, Nothing}=nothing,
    alpha::Real=0.0,
    label::Int=1
) where {T<:Real, N}

    _validate_image_mask_api(image, mask, N)

    settings = Settings(
        binwidth=Float64(binwidth),
        bincount=bincount,
        gldm_alpha=Float64(alpha)
    )

    extractor = RadiomicsFeatureExtractor(
        enabled_classes=Set([GLDM]),
        settings=settings
    )

    bool_mask = _to_bool_mask(mask, label)
    return extract(extractor, image, bool_mask; spacing=spacing)
end

#==============================================================================#
# Validation Helpers
#==============================================================================#

"""
    _validate_image_mask_api(image, mask, N)

Internal validation for high-level API functions.
Provides clear, user-friendly error messages.
"""
function _validate_image_mask_api(image::AbstractArray, mask::AbstractArray, N::Int)
    # Check dimensionality
    if N ∉ (2, 3)
        throw(ArgumentError(
            "Radiomics.jl only supports 2D and 3D images. " *
            "Got $(N)D array with size $(size(image)). " *
            "For 3D images, use arrays with shape (x, y, z). " *
            "For 2D images, use arrays with shape (x, y)."
        ))
    end

    # Check dimension match
    if ndims(mask) != N
        throw(DimensionMismatch(
            "Image and mask must have the same number of dimensions. " *
            "Image is $(N)D with size $(size(image)), but mask is $(ndims(mask))D " *
            "with size $(size(mask))."
        ))
    end

    # Check size match
    if size(image) != size(mask)
        throw(DimensionMismatch(
            "Image and mask must have the same size. " *
            "Image size: $(size(image)), Mask size: $(size(mask)). " *
            "Ensure both arrays represent the same spatial region."
        ))
    end

    # Check mask is not empty
    if eltype(mask) <: Bool
        if !any(mask)
            throw(ArgumentError(
                "Mask is empty (all false values). " *
                "At least one voxel must be true (inside the ROI) for feature extraction. " *
                "Check your mask generation or segmentation."
            ))
        end
    elseif eltype(mask) <: Integer
        # Will check after conversion; label value determines this
    end

    # Check for valid image values
    if any(isnan, image)
        throw(ArgumentError(
            "Image contains NaN values. " *
            "Please preprocess the image to remove or replace NaN values before extraction."
        ))
    end

    if any(isinf, image)
        throw(ArgumentError(
            "Image contains Inf values. " *
            "Please preprocess the image to remove or replace infinite values before extraction."
        ))
    end
end

"""
    _to_bool_mask(mask, label)

Convert mask to Boolean array.
"""
function _to_bool_mask(mask::AbstractArray, label::Int)
    if eltype(mask) <: Bool
        return mask
    elseif eltype(mask) <: Integer
        bool_mask = mask .== label
        if !any(bool_mask)
            throw(ArgumentError(
                "No voxels found with label value $label. " *
                "Available label values in mask: $(unique(mask)). " *
                "Specify the correct label using the 'label' keyword argument."
            ))
        end
        return bool_mask
    else
        throw(ArgumentError(
            "Mask must be Boolean or Integer array. " *
            "Got array with element type $(eltype(mask)). " *
            "Use a boolean mask (true/false) or an integer mask with label values."
        ))
    end
end

#==============================================================================#
# Feature Summary and Display
#==============================================================================#

"""
    summarize_features(features::Dict{String, Float64}; show_values::Bool=true)

Print a summary of extracted features organized by class.

# Arguments
- `features` - Dictionary of extracted features
- `show_values` - Whether to show feature values (default: true)

# Example
```julia
features = extract_all(image, mask)
summarize_features(features)
```
"""
function summarize_features(features::Dict{String, Float64}; show_values::Bool=true)
    # Group features by class
    classes = Dict{String, Vector{Pair{String, Float64}}}()

    for (name, value) in features
        parts = split(name, "_", limit=2)
        if length(parts) == 2
            class_name = parts[1]
            feature_name = parts[2]
            if !haskey(classes, class_name)
                classes[class_name] = Pair{String, Float64}[]
            end
            push!(classes[class_name], feature_name => value)
        end
    end

    # Print summary
    println("=" ^ 60)
    println("Radiomics Feature Summary")
    println("=" ^ 60)
    println("Total features: $(length(features))")
    println()

    for class_name in sort(collect(keys(classes)))
        class_features = classes[class_name]
        sort!(class_features, by=x->x[1])

        println("$(uppercase(class_name)) ($(length(class_features)) features)")
        println("-" ^ 40)

        if show_values
            for (fname, fval) in class_features
                println("  $fname: $fval")
            end
        else
            println("  ", join([p[1] for p in class_features], ", "))
        end
        println()
    end
end

"""
    features_to_dataframe(features::Dict{String, Float64})

Convert features dictionary to a format suitable for DataFrames.

Returns a NamedTuple with feature names as keys, suitable for
creating a DataFrame row.

# Example
```julia
using DataFrames
features = extract_all(image, mask)
row = features_to_dataframe(features)
df = DataFrame([row])
```
"""
function features_to_dataframe(features::Dict{String, Float64})
    # Sort keys for consistent ordering
    sorted_keys = sort(collect(keys(features)))
    names = Tuple(Symbol.(sorted_keys))
    values = Tuple(features[k] for k in sorted_keys)
    return NamedTuple{names}(values)
end

#==============================================================================#
# Batch Processing
#==============================================================================#

"""
    extract_batch(images, masks; kwargs...) -> Vector{Dict{String, Float64}}

Extract features from multiple image-mask pairs.

# Arguments
- `images` - Vector of image arrays
- `masks` - Vector of mask arrays (must match length of images)

# Keyword Arguments
- All keyword arguments from `extract_all` are supported
- `verbose::Bool=true` - Print progress information

# Returns
Vector of feature dictionaries, one per image-mask pair.

# Example
```julia
images = [rand(32, 32, 32) for _ in 1:10]
masks = [img .> 0.5 for img in images]
all_features = extract_batch(images, masks; verbose=true)
```

# Note
For parallel processing, consider using `Threads.@threads` or `pmap`:
```julia
all_features = [extract_all(img, msk) for (img, msk) in zip(images, masks)]
```
"""
function extract_batch(
    images::AbstractVector,
    masks::AbstractVector;
    verbose::Bool=true,
    kwargs...
)
    n = length(images)

    if length(masks) != n
        throw(ArgumentError(
            "Number of images ($n) must match number of masks ($(length(masks)))"
        ))
    end

    if n == 0
        return Dict{String, Float64}[]
    end

    results = Vector{Dict{String, Float64}}(undef, n)

    for i in 1:n
        if verbose
            println("Processing image $i/$n...")
        end
        results[i] = extract_all(images[i], masks[i]; kwargs...)
    end

    if verbose
        println("Done! Extracted features from $n images.")
    end

    return results
end

#==============================================================================#
# Feature Class Information
#==============================================================================#

"""
    list_feature_classes() -> Vector{String}

List all available feature classes with descriptions.
"""
function list_feature_classes()
    println("Available Feature Classes in Radiomics.jl")
    println("=" ^ 50)
    println()

    classes = [
        ("FirstOrder", 19, "Statistical features from voxel intensity distribution"),
        ("Shape", 18, "Morphological features describing ROI shape (3D)"),
        ("GLCM", 24, "Gray Level Co-occurrence Matrix texture features"),
        ("GLRLM", 16, "Gray Level Run Length Matrix texture features"),
        ("GLSZM", 16, "Gray Level Size Zone Matrix texture features"),
        ("NGTDM", 5, "Neighboring Gray Tone Difference Matrix features"),
        ("GLDM", 14, "Gray Level Dependence Matrix texture features"),
    ]

    total = 0
    for (name, count, desc) in classes
        println("$name ($count features)")
        println("  $desc")
        println()
        total += count
    end

    println("-" ^ 50)
    println("Total: $total features (for 3D images)")

    return [c[1] for c in classes]
end

"""
    describe_feature(feature_name::String)

Print description of a specific feature.

# Example
```julia
describe_feature("glcm_Contrast")
describe_feature("firstorder_Energy")
```
"""
function describe_feature(feature_name::String)
    # Parse feature name
    parts = split(feature_name, "_", limit=2)
    if length(parts) != 2
        println("Unknown feature format: $feature_name")
        println("Expected format: 'class_FeatureName' (e.g., 'glcm_Contrast')")
        return
    end

    class_name = lowercase(parts[1])
    fname = parts[2]

    # Feature descriptions (subset of most common)
    descriptions = Dict(
        # First Order
        "firstorder_energy" => "Sum of squared voxel values. Measures total magnitude of intensities.",
        "firstorder_entropy" => "Entropy of the voxel intensity histogram. Measures randomness.",
        "firstorder_mean" => "Mean voxel intensity value.",
        "firstorder_variance" => "Variance of voxel intensity values.",
        "firstorder_skewness" => "Asymmetry of the intensity distribution.",
        "firstorder_kurtosis" => "Peakedness of the intensity distribution.",

        # GLCM
        "glcm_contrast" => "Measures local intensity variation between neighboring pixels.",
        "glcm_correlation" => "Measures linear dependency of gray levels.",
        "glcm_energy" => "Also called Angular Second Moment. Measures texture uniformity.",
        "glcm_entropy" => "Measures randomness in the GLCM.",

        # Shape
        "shape_sphericity" => "Measures how spherical the ROI is (1.0 = perfect sphere).",
        "shape_volume" => "Volume of the ROI in physical units.",
        "shape_surfacearea" => "Surface area of the ROI in physical units.",
    )

    key = lowercase(feature_name)
    if haskey(descriptions, key)
        println("$feature_name:")
        println("  $(descriptions[key])")
    else
        println("$feature_name:")
        println("  (No detailed description available)")
        println("  Class: $(uppercase(class_name))")
    end
end
