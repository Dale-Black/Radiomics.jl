# API Reference

## High-Level Functions

### Feature Extraction

```@docs
extract_all
extract
extract_firstorder_only
extract_shape_only
extract_texture_only
extract_glcm
extract_glrlm
extract_glszm
extract_ngtdm
extract_gldm
```

### Batch Processing

```@docs
extract_batch
```

### Results Handling

```@docs
summarize_features
features_to_dataframe
```

### Information Functions

```@docs
list_feature_classes
describe_feature
feature_count
total_feature_count
feature_names
all_feature_names
```

---

## Feature Extractor

```@docs
RadiomicsFeatureExtractor
enable!
disable!
enable_all!
disable_all!
is_enabled
enabled_classes
```

---

## Configuration Types

```@docs
Settings
validate_settings
DiscretizationMode
FeatureClass
```

---

## Image and Mask Types

```@docs
RadiomicsImage
RadiomicsMask
get_roi_mask
get_data
get_spacing
get_mask_data
ImageLike
MaskLike
```

---

## Result Types

```@docs
FeatureResult
FeatureSet
feature_key
```

---

## First-Order Features

### Extraction

```@docs
extract_firstorder
extract_firstorder_to_featureset!
firstorder_feature_names
firstorder_ibsi_features
```

### Individual Features

```@docs
energy
total_energy
entropy
fo_minimum
fo_maximum
fo_mean
fo_median
fo_variance
fo_range
percentile_10
percentile_90
interquartile_range
mean_absolute_deviation
robust_mean_absolute_deviation
root_mean_squared
standard_deviation
skewness
kurtosis
uniformity
```

---

## Shape Features

### 2D Shape

```@docs
extract_shape_2d
extract_shape_2d_to_featureset!
shape_2d_feature_names
shape_2d_ibsi_features
perimeter_2d
mesh_surface_2d
pixel_surface_2d
perimeter_surface_ratio_2d
sphericity_2d
spherical_disproportion_2d
maximum_diameter_2d
major_axis_length_2d
minor_axis_length_2d
elongation_2d
```

### 3D Shape

```@docs
extract_shape_3d
extract_shape_3d_to_featureset!
shape_3d_feature_names
shape_3d_ibsi_features
mesh_volume
voxel_volume_3d
surface_area
surface_volume_ratio
sphericity_3d
compactness1
compactness2
spherical_disproportion_3d
maximum_3d_diameter
maximum_2d_diameter_slice
maximum_2d_diameter_column
maximum_2d_diameter_row
major_axis_length_3d
minor_axis_length_3d
least_axis_length
elongation_3d
flatness
```

---

## GLCM Features

### Matrix Computation

```@docs
compute_glcm
compute_glcm_2d
GLCMResult
GLCMResult2D
get_averaged_glcm
get_merged_glcm
glcm_num_gray_levels
glcm_num_directions
GLCM_DIRECTIONS_3D
GLCM_DIRECTIONS_2D
```

---

## GLRLM Features

### Matrix Computation

```@docs
compute_glrlm
compute_glrlm_2d
GLRLMResult
GLRLMResult2D
glrlm_num_gray_levels
glrlm_num_directions
glrlm_max_run_length
glrlm_num_runs
glrlm_num_voxels
GLRLM_DIRECTIONS_3D
GLRLM_DIRECTIONS_2D
```

---

## GLSZM Features

### Matrix Computation

```@docs
compute_glszm
compute_glszm_2d
GLSZMResult
GLSZMResult2D
glszm_num_gray_levels
glszm_max_zone_size
glszm_num_zones
glszm_num_voxels
```

### Features

```@docs
glszm_small_area_emphasis
glszm_large_area_emphasis
glszm_gray_level_non_uniformity
glszm_gray_level_non_uniformity_normalized
glszm_size_zone_non_uniformity
glszm_size_zone_non_uniformity_normalized
glszm_zone_percentage
glszm_gray_level_variance
glszm_zone_variance
glszm_zone_entropy
glszm_low_gray_level_zone_emphasis
glszm_high_gray_level_zone_emphasis
glszm_small_area_low_gray_level_emphasis
glszm_small_area_high_gray_level_emphasis
glszm_large_area_low_gray_level_emphasis
glszm_large_area_high_gray_level_emphasis
extract_glszm_to_featureset!
glszm_feature_names
glszm_ibsi_features
```

---

## NGTDM Features

### Matrix Computation

```@docs
compute_ngtdm
compute_ngtdm_2d
NGTDMResult
NGTDMResult2D
ngtdm_num_gray_levels
ngtdm_num_valid_gray_levels
ngtdm_num_valid_voxels
ngtdm_sum_s
```

### Features

```@docs
ngtdm_coarseness
ngtdm_contrast
ngtdm_busyness
ngtdm_complexity
ngtdm_strength
compute_all_ngtdm_features
```

---

## GLDM Features

### Matrix Computation

```@docs
compute_gldm
compute_gldm_2d
GLDMResult
GLDMResult2D
gldm_num_gray_levels
gldm_num_valid_gray_levels
gldm_num_zones
gldm_max_dependence
gldm_gray_levels
gldm_dependence_sizes
GLDM_DIRECTIONS_3D
GLDM_DIRECTIONS_2D
```

### Features

```@docs
gldm_small_dependence_emphasis
gldm_large_dependence_emphasis
gldm_gray_level_non_uniformity
gldm_dependence_non_uniformity
gldm_dependence_non_uniformity_normalized
gldm_gray_level_variance
gldm_dependence_variance
gldm_dependence_entropy
gldm_low_gray_level_emphasis
gldm_high_gray_level_emphasis
gldm_small_dependence_low_gray_level_emphasis
gldm_small_dependence_high_gray_level_emphasis
gldm_large_dependence_low_gray_level_emphasis
gldm_large_dependence_high_gray_level_emphasis
```

---

## Image Handling Utilities

```@docs
get_voxels
get_voxels_with_coords
count_voxels
voxel_volume
roi_volume
normalize_image
normalize_image!
is_2d
is_3d
effective_ndims
validate_image_mask
ensure_float64
squeeze_image
get_slice
get_physical_size
apply_spacing
get_centroid
```

---

## Mask Operations

```@docs
BoundingBox
bounding_box
bounding_box_size
crop_to_mask
crop_to_bbox
validate_mask
is_empty_mask
is_full_mask
mask_extent
mask_dimensionality
dilate_mask
erode_mask
fill_holes_2d
largest_connected_component
mask_surface_voxels
mask_interior_voxels
```

---

## Discretization

```@docs
get_bin_edges
discretize
discretize_image
discretize_voxels
get_discretization_range
suggest_bincount
suggest_binwidth
count_gray_levels
gray_level_histogram
```

---

## Index

```@index
```
