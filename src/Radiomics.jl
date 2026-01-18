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
using Meshing
using GeometryBasics

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

# Gray Level Co-occurrence Matrix (GLCM) features
include("glcm.jl")

# Gray Level Run Length Matrix (GLRLM) features
include("glrlm.jl")

# Gray Level Size Zone Matrix (GLSZM) features
include("glszm.jl")

# Neighbouring Gray Tone Difference Matrix (NGTDM) features
include("ngtdm.jl")

# Gray Level Dependence Matrix (GLDM) features
include("gldm.jl")

# Module files will be included here as they are implemented
# include("utils.jl")
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

# Export 3D shape feature functions
export mesh_volume, voxel_volume_3d, surface_area, surface_volume_ratio
export sphericity_3d, compactness1, compactness2, spherical_disproportion_3d
export maximum_3d_diameter, maximum_2d_diameter_slice, maximum_2d_diameter_column, maximum_2d_diameter_row
export major_axis_length_3d, minor_axis_length_3d, least_axis_length
export elongation_3d, flatness
export extract_shape_3d, extract_shape_3d_to_featureset!
export shape_3d_feature_names, shape_3d_ibsi_features

# Export GLCM computation functions and types
export GLCMResult, GLCMResult2D
export compute_glcm, compute_glcm_2d
export GLCM_DIRECTIONS_3D, GLCM_DIRECTIONS_2D
export get_averaged_glcm, get_merged_glcm
export glcm_num_gray_levels, glcm_num_directions

# Export GLRLM computation functions and types
export GLRLMResult, GLRLMResult2D
export compute_glrlm, compute_glrlm_2d
export GLRLM_DIRECTIONS_3D, GLRLM_DIRECTIONS_2D
export glrlm_num_gray_levels, glrlm_num_directions, glrlm_max_run_length
export glrlm_num_runs, glrlm_num_voxels

# Export GLSZM computation functions and types
export GLSZMResult, GLSZMResult2D
export compute_glszm, compute_glszm_2d
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

# Export GLSZM extraction functions
export extract_glszm, extract_glszm_to_featureset!
export glszm_feature_names, glszm_ibsi_features

# Export NGTDM computation functions and types
export NGTDMResult, NGTDMResult2D
export compute_ngtdm, compute_ngtdm_2d
export ngtdm_num_gray_levels, ngtdm_num_valid_gray_levels
export ngtdm_num_valid_voxels, ngtdm_sum_s

# Export NGTDM feature functions
export ngtdm_coarseness, ngtdm_contrast, ngtdm_busyness
export ngtdm_complexity, ngtdm_strength
export compute_all_ngtdm_features

# Export GLDM computation functions and types
export GLDMResult, GLDMResult2D
export compute_gldm, compute_gldm_2d
export GLDM_DIRECTIONS_3D, GLDM_DIRECTIONS_2D
export gldm_num_gray_levels, gldm_num_valid_gray_levels
export gldm_num_zones, gldm_max_dependence
export gldm_gray_levels, gldm_dependence_sizes

end # module Radiomics
