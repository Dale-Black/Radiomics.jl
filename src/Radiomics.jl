"""
    Radiomics

A pure Julia port of PyRadiomics for extracting radiomic features from medical images.

This package provides feature extraction from image data with support for:
- First-order statistical features
- Shape features (2D and 3D)
- GLCM (Gray Level Co-occurrence Matrix) features
- GLRLM (Gray Level Run Length Matrix) features
- GLSZM (Gray Level Size Zone Matrix) features
- NGTDM (Neighboring Gray Tone Difference Matrix) features
- GLDM (Gray Level Dependence Matrix) features

# Example
```julia
using Radiomics

# Extract features from an image and mask
image = rand(64, 64, 64)
mask = image .> 0.5

# Extract all first-order features
features = extract_firstorder(image, mask)
```

See also: [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics)
"""
module Radiomics

using Statistics
using LinearAlgebra

# Package version
const VERSION = v"0.1.0"

# Core types
include("types.jl")

# Image handling utilities
include("image_handling.jl")

# Mask operations
include("mask_operations.jl")

# Gray level discretization
include("discretization.jl")

# First-order statistical features
include("firstorder.jl")

# Shape features (2D and 3D)
include("shape.jl")

# Module files will be included here as they are implemented
# include("utils.jl")
# include("glcm.jl")
# include("glrlm.jl")
# include("glszm.jl")
# include("ngtdm.jl")
# include("gldm.jl")
# include("extractor.jl")

# Export abstract types
export AbstractRadiomicsFeature,
       AbstractFirstOrderFeature,
       AbstractShapeFeature,
       AbstractTextureFeature,
       AbstractGLCMFeature,
       AbstractGLRLMFeature,
       AbstractGLSZMFeature,
       AbstractNGTDMFeature,
       AbstractGLDMFeature

# Export discretization mode
export DiscretizationMode, FixedBinWidth, FixedBinCount

# Export settings
export Settings, validate_settings

# Export image and mask types
export RadiomicsImage, RadiomicsMask
export get_roi_mask, get_data, get_spacing, get_mask_data

# Export feature result types
export FeatureResult, FeatureSet, feature_key

# Export type aliases
export ImageLike, MaskLike

# Export image handling functions
export get_voxels, get_voxels_with_coords
export count_voxels, voxel_volume, roi_volume
export normalize_image, normalize_image!
export is_2d, is_3d, effective_ndims
export validate_image_mask
export ensure_float64, squeeze_image, get_slice
export get_physical_size, apply_spacing, get_centroid

# Export mask operations
export BoundingBox
export bounding_box, bounding_box_size
export crop_to_mask, crop_to_bbox
export validate_mask, is_empty_mask, is_full_mask
export mask_extent, mask_dimensionality
export dilate_mask, erode_mask, fill_holes_2d
export largest_connected_component
export mask_surface_voxels, mask_interior_voxels

# Export discretization functions
export get_bin_edges, discretize, discretize_image, discretize_voxels
export get_discretization_range, suggest_bincount, suggest_binwidth
export count_gray_levels, gray_level_histogram

# Export first-order feature functions
export energy, total_energy, entropy
export fo_minimum, fo_maximum, fo_mean, fo_median, fo_variance, fo_range
export percentile_10, percentile_90
export interquartile_range, mean_absolute_deviation, robust_mean_absolute_deviation
export root_mean_squared, standard_deviation, skewness, kurtosis, uniformity
export extract_firstorder, extract_firstorder_to_featureset!
export firstorder_feature_names, firstorder_ibsi_features

# Export 2D shape feature functions
export perimeter_2d, mesh_surface_2d, pixel_surface_2d
export perimeter_surface_ratio_2d, sphericity_2d, spherical_disproportion_2d
export maximum_diameter_2d, major_axis_length_2d, minor_axis_length_2d, elongation_2d
export extract_shape_2d, extract_shape_2d_to_featureset!
export shape_2d_feature_names, shape_2d_ibsi_features

end # module Radiomics
