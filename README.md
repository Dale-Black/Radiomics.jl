# Radiomics.jl

[![CI](https://github.com/daleblack/Radiomics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/daleblack/Radiomics.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/daleblack/Radiomics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/daleblack/Radiomics.jl)

A **pure Julia** implementation of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) for extracting radiomic features from medical images.

## Features

Radiomics.jl provides **111+ radiomic features** organized into 7 feature classes:

| Feature Class | Features | Description |
|--------------|----------|-------------|
| **First Order** | 19 | Statistical features from intensity histogram |
| **Shape 2D** | 10 | Geometric features for 2D ROIs |
| **Shape 3D** | 17 | Geometric features for 3D ROIs |
| **GLCM** | 24 | Gray Level Co-occurrence Matrix texture features |
| **GLRLM** | 16 | Gray Level Run Length Matrix texture features |
| **GLSZM** | 16 | Gray Level Size Zone Matrix texture features |
| **NGTDM** | 5 | Neighboring Gray Tone Difference Matrix features |
| **GLDM** | 14 | Gray Level Dependence Matrix features |

### Key Highlights

- **Pure Julia** - No Python runtime dependencies
- **1:1 Parity with PyRadiomics** - Verified against PyRadiomics using comprehensive test suite
- **IBSI Compliant** - Follows [Image Biomarker Standardization Initiative](https://ibsi.readthedocs.io/) standards
- **Performant** - Type-stable, allocation-efficient code
- **Flexible** - Configurable discretization, texture parameters, and feature selection

## Installation

```julia
using Pkg
Pkg.add("Radiomics")  # Once registered in General Registry
```

For development version:

```julia
using Pkg
Pkg.add(url="https://github.com/daleblack/Radiomics.jl")
```

## Quick Start

```julia
using Radiomics

# Create sample data
image = rand(64, 64, 64)
mask = image .> 0.5

# Extract all features (default settings)
features = extract_all(image, mask)

# Access individual features
println("Energy: ", features[:firstorder_energy])
println("GLCM Contrast: ", features[:glcm_contrast])
println("Total features: ", length(features))
```

## Usage Examples

### Extract Specific Feature Classes

```julia
using Radiomics

image = rand(32, 32, 32)
mask = image .> 0.5

# Extract only first-order features
fo_features = extract_firstorder(image, mask)

# Extract only shape features (takes mask only, not image)
shape_features = extract_shape_3d(mask)

# Extract texture features
glcm_features = extract_glcm(image, mask)
glrlm_features = extract_glrlm(image, mask)
glszm_features = extract_glszm(image, mask)
ngtdm_features = extract_ngtdm(image, mask)
gldm_features = extract_gldm(image, mask)
```

### Using the Feature Extractor

```julia
using Radiomics

# Create a feature extractor with specific classes enabled
extractor = RadiomicsFeatureExtractor(
    enable_firstorder = true,
    enable_shape = true,
    enable_glcm = true,
    enable_glrlm = false,  # Disable GLRLM
    enable_glszm = false,  # Disable GLSZM
    enable_ngtdm = false,  # Disable NGTDM
    enable_gldm = false    # Disable GLDM
)

# Extract features
features = extract(extractor, image, mask)
```

### Custom Settings

```julia
using Radiomics

# Create custom settings
settings = Settings(
    binwidth = 25,           # Gray level bin width (default: 25)
    bincount = nothing,      # Or use fixed bin count
    glcm_distance = 1,       # GLCM pixel distance
    gldm_alpha = 0,          # GLDM coarseness parameter
    ngtdm_distance = 1       # NGTDM neighborhood distance
)

# Extract with custom settings
features = extract_all(image, mask; settings=settings)
```

### Working with Voxel Spacing

```julia
using Radiomics

# Medical images often have non-isotropic spacing
image = rand(128, 128, 64)
mask = image .> 0.5
spacing = (1.0, 1.0, 2.0)  # (x, y, z) in mm

# Extract shape features with correct spacing
shape_features = extract_shape_3d(mask, spacing)
```

### Integer Label Masks

```julia
using Radiomics

# Multi-label segmentation
labels = rand(0:3, 64, 64, 64)
image = rand(64, 64, 64)

# Extract features for label 2 only
features = extract_all(image, labels; label=2)
```

### Batch Processing

```julia
using Radiomics

# Process multiple images
images = [rand(32, 32, 32) for _ in 1:10]
masks = [img .> 0.5 for img in images]

# Extract features for all images
results = extract_batch(images, masks)

# Results is a Vector of feature dictionaries
for (i, features) in enumerate(results)
    println("Subject $i: $(length(features)) features extracted")
end
```

## Feature Documentation

### First-Order Features (19)

Statistical features computed directly from the intensity histogram:

- Energy, Total Energy
- Entropy
- Minimum, 10th Percentile, 90th Percentile, Maximum
- Mean, Median
- Interquartile Range, Range
- Mean Absolute Deviation, Robust Mean Absolute Deviation
- Root Mean Squared
- Skewness, Kurtosis
- Variance, Uniformity

### Shape Features (27 total: 10 2D + 17 3D)

Geometric features describing the ROI shape:

**3D Features:**
- Mesh Volume, Voxel Volume
- Surface Area, Surface to Volume Ratio
- Sphericity, Compactness 1/2, Spherical Disproportion
- Maximum 3D/2D Diameters
- Major/Minor/Least Axis Lengths
- Elongation, Flatness

**2D Features:**
- Perimeter, Pixel Surface
- Sphericity (Circularity)
- Maximum Diameter, Axis Lengths
- Elongation

### GLCM Features (24)

Gray Level Co-occurrence Matrix texture features:

- Autocorrelation, Joint Average
- Cluster Prominence/Shade/Tendency
- Contrast, Correlation
- Difference Average/Entropy/Variance
- Joint Energy, Joint Entropy
- IMC1, IMC2
- IDM, IDMN, ID, IDN
- Inverse Variance
- Maximum Probability
- Sum Average/Entropy, Sum Squares
- MCC (Maximal Correlation Coefficient)

### GLRLM Features (16)

Gray Level Run Length Matrix features:

- Short/Long Run Emphasis
- Gray Level Non-Uniformity (Normalized)
- Run Length Non-Uniformity (Normalized)
- Run Percentage
- Gray Level Variance, Run Variance, Run Entropy
- Low/High Gray Level Run Emphasis
- Short/Long Run Low/High Gray Level Emphasis

### GLSZM Features (16)

Gray Level Size Zone Matrix features:

- Small/Large Area Emphasis
- Gray Level Non-Uniformity (Normalized)
- Size Zone Non-Uniformity (Normalized)
- Zone Percentage
- Gray Level Variance, Zone Variance, Zone Entropy
- Low/High Gray Level Zone Emphasis
- Small/Large Area Low/High Gray Level Emphasis

### NGTDM Features (5)

Neighboring Gray Tone Difference Matrix features:

- Coarseness, Contrast, Busyness
- Complexity, Strength

### GLDM Features (14)

Gray Level Dependence Matrix features:

- Small/Large Dependence Emphasis
- Gray Level Non-Uniformity
- Dependence Non-Uniformity (Normalized)
- Gray Level Variance, Dependence Variance, Dependence Entropy
- Low/High Gray Level Emphasis
- Small/Large Dependence Low/High Gray Level Emphasis

## Configuration

### Discretization Modes

```julia
# Fixed Bin Width (default)
settings = Settings(binwidth=25)  # ~10 bins for typical CT data

# Fixed Bin Count
settings = Settings(bincount=32)  # Exactly 32 gray levels
```

### PyRadiomics Compatibility

The default settings match PyRadiomics defaults for maximum compatibility:

```julia
settings = Settings(
    binwidth = 25,      # Same as PyRadiomics default
    bincount = nothing, # Use binwidth mode
    glcm_distance = 1,
    gldm_alpha = 0,
    ngtdm_distance = 1
)
```

## Testing

The test suite verifies 1:1 parity with PyRadiomics using PythonCall.jl:

```julia
using Pkg
Pkg.test("Radiomics")
```

Tests cover:
- All 111+ features against PyRadiomics
- Multiple random seeds and array sizes
- Edge cases (small ROIs, single values, etc.)
- 2D and 3D images

## API Reference

### High-Level Functions

| Function | Description |
|----------|-------------|
| `extract_all(image, mask)` | Extract all features |
| `extract_firstorder(image, mask)` | First-order features only |
| `extract_shape_2d(mask)` | 2D shape features (mask only) |
| `extract_shape_3d(mask)` | 3D shape features (mask only) |
| `extract_glcm(image, mask)` | GLCM texture features |
| `extract_glrlm(image, mask)` | GLRLM texture features |
| `extract_glszm(image, mask)` | GLSZM texture features |
| `extract_ngtdm(image, mask)` | NGTDM texture features |
| `extract_gldm(image, mask)` | GLDM texture features |

### Feature Extractor

| Function | Description |
|----------|-------------|
| `RadiomicsFeatureExtractor(...)` | Create configurable extractor |
| `extract(extractor, image, mask)` | Extract using extractor settings |
| `enable!(extractor, class)` | Enable a feature class |
| `disable!(extractor, class)` | Disable a feature class |

### Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `binwidth` | 25 | Gray level bin width |
| `bincount` | nothing | Fixed number of bins (overrides binwidth) |
| `glcm_distance` | 1 | GLCM pixel pair distance |
| `gldm_alpha` | 0 | GLDM coarseness parameter |
| `ngtdm_distance` | 1 | NGTDM neighborhood distance |

## References

- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) - Original Python implementation
- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/) - Feature documentation
- [IBSI](https://ibsi.readthedocs.io/) - Image Biomarker Standardization Initiative
- van Griethuysen, J.J.M. et al. (2017). "Computational Radiomics System to Decode the Radiographic Phenotype". Cancer Research.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This package is a pure Julia port of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) by the Computational Imaging and Bioinformatics Lab at Harvard Medical School.
