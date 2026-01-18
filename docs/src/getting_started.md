# Getting Started

This guide will help you get started with Radiomics.jl for extracting radiomic features from medical images.

## Installation

Install Radiomics.jl using Julia's package manager:

```julia
using Pkg
Pkg.add("Radiomics")
```

## Basic Usage

### Extract All Features

The simplest way to extract features is using the `extract_all` function:

```julia
using Radiomics

# Your image data (2D or 3D array)
image = rand(64, 64, 64) * 100  # Example: 64x64x64 volume

# Binary mask defining the Region of Interest (ROI)
mask = rand(Bool, 64, 64, 64)

# Extract all radiomic features
features = extract_all(image, mask)

# Features is a Dict{String, Float64}
println("Number of features: ", length(features))
println("Energy: ", features["firstorder_Energy"])
```

### Extract Specific Feature Classes

If you only need certain feature types:

```julia
# Extract only first-order statistics
fo_features = extract_firstorder_only(image, mask)

# Extract only shape features
shape_features = extract_shape_only(mask)

# Extract only texture features (GLCM, GLRLM, GLSZM, NGTDM, GLDM)
texture_features = extract_texture_only(image, mask)

# Extract individual texture classes
glcm_features = extract_glcm(image, mask)
glrlm_features = extract_glrlm(image, mask)
glszm_features = extract_glszm(image, mask)
ngtdm_features = extract_ngtdm(image, mask)
gldm_features = extract_gldm(image, mask)
```

## Working with Medical Images

### Using Voxel Spacing

For accurate shape and volume measurements, provide voxel spacing:

```julia
# Image with 1mm x 1mm x 2.5mm voxels
spacing = (1.0, 1.0, 2.5)
features = extract_all(image, mask; spacing=spacing)
```

### Using RadiomicsImage Wrapper

For more structured image handling, use the `RadiomicsImage` type:

```julia
# Create image with metadata
img = RadiomicsImage(data, (1.0, 1.0, 2.5))  # data + spacing

# Create mask
msk = RadiomicsMask(mask_data)

# Extract features
features = extract_all(img.data, get_roi_mask(msk); spacing=img.spacing)
```

### Integer Label Masks

If your mask uses integer labels instead of boolean values:

```julia
# Mask with multiple labels (0=background, 1=tumor, 2=organ)
label_mask = rand(0:2, 64, 64, 64)

# Extract features from label 1 (tumor)
features = extract_all(image, label_mask; label=1)

# Extract features from label 2 (organ)
features = extract_all(image, label_mask; label=2)
```

## Customizing Extraction

### Using the Feature Extractor

For more control, use `RadiomicsFeatureExtractor`:

```julia
# Create extractor with custom settings
extractor = RadiomicsFeatureExtractor(
    settings=Settings(
        binwidth=32.0,        # Discretization bin width
        glcm_distance=2,      # GLCM distance parameter
        symmetrical_glcm=true # Symmetric GLCM
    )
)

# Extract features
features = extract(extractor, image, mask)
```

### Enable/Disable Feature Classes

```julia
# Start with all classes enabled
extractor = RadiomicsFeatureExtractor()

# Disable shape features (useful if only intensities matter)
disable!(extractor, Shape)

# Extract (will skip shape features)
features = extract(extractor, image, mask)

# Check what's enabled
println(enabled_classes(extractor))
```

### Select Specific Classes

```julia
# Only extract first-order and GLCM features
extractor = RadiomicsFeatureExtractor(
    enabled_classes=Set([FirstOrder, GLCM])
)

features = extract(extractor, image, mask)
```

## Working with Results

### Summarize Features

```julia
features = extract_all(image, mask)

# Print organized summary
summarize_features(features)

# Summary without values (just names)
summarize_features(features; show_values=false)
```

### Convert to DataFrame

```julia
using DataFrames

features = extract_all(image, mask)
row = features_to_dataframe(features)
df = DataFrame([row])
```

### Batch Processing

```julia
# Extract from multiple images
images = [rand(32, 32, 32) for _ in 1:10]
masks = [img .> 0.5 for img in images]

# Process all with progress output
all_features = extract_batch(images, masks; verbose=true)

# Convert to DataFrame
using DataFrames
df = DataFrame(features_to_dataframe.(all_features))
```

## 2D vs 3D Images

Radiomics.jl automatically handles 2D and 3D images:

```julia
# 3D image (most common for medical imaging)
image_3d = rand(64, 64, 64)
mask_3d = image_3d .> 0.5
features_3d = extract_all(image_3d, mask_3d)

# 2D image (single slice analysis)
image_2d = rand(256, 256)
mask_2d = image_2d .> 0.5
features_2d = extract_all(image_2d, mask_2d)
```

## Feature Naming Convention

All features follow the naming pattern `{class}_{feature}`:

| Class | Prefix | Example |
|-------|--------|---------|
| First Order | `firstorder_` | `firstorder_Energy` |
| Shape | `shape_` | `shape_Sphericity` |
| GLCM | `glcm_` | `glcm_Contrast` |
| GLRLM | `glrlm_` | `glrlm_ShortRunEmphasis` |
| GLSZM | `glszm_` | `glszm_SmallAreaEmphasis` |
| NGTDM | `ngtdm_` | `ngtdm_Coarseness` |
| GLDM | `gldm_` | `gldm_SmallDependenceEmphasis` |

## Next Steps

- Learn about all [Feature Classes](features.md)
- Understand [Configuration Options](configuration.md)
- Explore the full [API Reference](api.md)
