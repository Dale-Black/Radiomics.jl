# Configuration

Radiomics.jl provides extensive configuration options through the `Settings` struct and `RadiomicsFeatureExtractor` class.

## Settings Struct

The `Settings` struct controls all extraction parameters:

```julia
using Radiomics

# Default settings
settings = Settings()

# Custom settings
settings = Settings(
    binwidth = 25.0,        # Discretization bin width
    bincount = nothing,      # Or use fixed bin count
    glcm_distance = 1,       # GLCM pixel distance
    symmetrical_glcm = true, # Symmetric GLCM matrix
    gldm_alpha = 0.0,        # GLDM coarseness parameter
    ngtdm_distance = 1       # NGTDM neighborhood distance
)
```

### Complete Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Discretization** ||||
| `binwidth` | Float64 | 25.0 | Width of gray level bins |
| `bincount` | Union{Int, Nothing} | nothing | Number of bins (overrides binwidth) |
| `discretization_mode` | DiscretizationMode | FixedBinWidth | Discretization method |
| **Mask/Label** ||||
| `label` | Int | 1 | Label value in mask for ROI |
| **Preprocessing** ||||
| `resample_spacing` | Union{NTuple{3,Float64}, Nothing} | nothing | Target voxel spacing |
| `normalize` | Bool | false | Normalize intensities |
| `normalize_scale` | Float64 | 1.0 | Normalization scale |
| `remove_outliers` | Bool | false | Remove intensity outliers |
| `outlier_percentile` | Float64 | 99.0 | Percentile for outlier removal |
| **2D/3D Handling** ||||
| `force_2d` | Bool | false | Force 2D extraction |
| `force_2d_dimension` | Int | 3 | Slice dimension for 2D mode |
| **Texture Parameters** ||||
| `glcm_distance` | Int | 1 | Distance for GLCM computation |
| `symmetrical_glcm` | Bool | true | Make GLCM symmetric |
| `gldm_alpha` | Float64 | 0.0 | GLDM coarseness parameter |
| `ngtdm_distance` | Int | 1 | NGTDM neighborhood distance |
| **Performance** ||||
| `preallocate` | Bool | true | Preallocate arrays |
| `voxel_array_shift` | Int | 0 | Shift for voxel values |

## Discretization

Gray level discretization is crucial for texture features. It maps continuous intensity values to discrete bins.

### Fixed Bin Width (Default)

Bins are created with a fixed width. The number of bins depends on the intensity range:

```julia
# Bin width of 25 intensity units
settings = Settings(binwidth=25.0)

# For intensity range [0, 100], this creates 4 bins:
# Bin 1: [0, 25), Bin 2: [25, 50), Bin 3: [50, 75), Bin 4: [75, 100]
```

### Fixed Bin Count

A fixed number of equal-width bins:

```julia
settings = Settings(
    bincount=64,
    discretization_mode=FixedBinCount
)
```

### Choosing Discretization

| Method | Use When | Advantages |
|--------|----------|------------|
| Fixed Bin Width | Comparing across images with different ranges | Consistent bin meaning |
| Fixed Bin Count | Need consistent feature space | Fixed number of gray levels |

**Typical values:**
- Binwidth: 5-50 (25 is common)
- Bincount: 16-256 (64-128 common)

## GLCM Configuration

The Gray Level Co-occurrence Matrix has several parameters:

### Distance

```julia
# Default: adjacent voxels
settings = Settings(glcm_distance=1)

# Larger distance for coarser textures
settings = Settings(glcm_distance=2)
```

### Symmetry

```julia
# Symmetric (recommended, matches PyRadiomics)
settings = Settings(symmetrical_glcm=true)

# Asymmetric (directional information preserved)
settings = Settings(symmetrical_glcm=false)
```

### Directions

GLCM is computed in 13 directions for 3D images (4 for 2D). Results are averaged across directions.

## GLDM Configuration

### Alpha Parameter

The alpha parameter controls what counts as a "dependent" voxel:

```julia
# Strict: only exact matches count as dependent
settings = Settings(gldm_alpha=0.0)

# Loose: neighbors within 5 gray levels are dependent
settings = Settings(gldm_alpha=5.0)
```

**Note:** Alpha refers to the raw gray level difference, not a probability.

## NGTDM Configuration

### Neighborhood Distance

```julia
# Default: immediate neighbors (26-connected in 3D)
settings = Settings(ngtdm_distance=1)

# Larger neighborhood
settings = Settings(ngtdm_distance=2)
```

## Feature Extractor Configuration

### Basic Setup

```julia
# All features with default settings
extractor = RadiomicsFeatureExtractor()

# All features with custom settings
extractor = RadiomicsFeatureExtractor(
    settings=Settings(binwidth=32.0)
)
```

### Selecting Feature Classes

```julia
# Only specific classes
extractor = RadiomicsFeatureExtractor(
    enabled_classes=Set([FirstOrder, GLCM, Shape])
)

# Start empty and add
extractor = RadiomicsFeatureExtractor(
    enabled_classes=Set{FeatureClass}()
)
enable!(extractor, FirstOrder)
enable!(extractor, GLCM)
```

### Enable/Disable Dynamically

```julia
extractor = RadiomicsFeatureExtractor()

# Disable specific classes
disable!(extractor, Shape)
disable!(extractor, GLDM)

# Check status
is_enabled(extractor, GLCM)  # true
enabled_classes(extractor)    # [FirstOrder, GLCM, GLRLM, GLSZM, NGTDM]

# Enable all
enable_all!(extractor)

# Disable all
disable_all!(extractor)
```

## Function-Level Configuration

High-level functions accept keyword arguments directly:

```julia
# extract_all with custom parameters
features = extract_all(image, mask;
    binwidth=32.0,
    glcm_distance=2,
    gldm_alpha=0.0,
    ngtdm_distance=1,
    spacing=(1.0, 1.0, 2.5),
    label=1
)

# Class-specific extraction
glcm_features = extract_glcm(image, mask;
    binwidth=25.0,
    distance=2,
    symmetric=true
)
```

## PyRadiomics Compatibility

Radiomics.jl aims for exact numerical parity with PyRadiomics. Default settings match PyRadiomics defaults:

| Parameter | Radiomics.jl | PyRadiomics |
|-----------|--------------|-------------|
| binwidth | 25.0 | 25.0 |
| glcm_distance | 1 | 1 |
| symmetrical_glcm | true | true |
| gldm_alpha | 0.0 | 0 |
| ngtdm_distance | 1 | 1 |

### Matching PyRadiomics Settings

```julia
# Equivalent to PyRadiomics default extraction
settings = Settings(
    binwidth=25.0,
    glcm_distance=1,
    symmetrical_glcm=true,
    gldm_alpha=0.0,
    ngtdm_distance=1,
    voxel_array_shift=0
)
```

## Validation

Settings are validated when used:

```julia
settings = Settings(
    binwidth=-1.0  # Invalid: will throw ArgumentError when used
)

# Explicit validation
validate_settings(settings)  # throws ArgumentError for invalid settings
```

### Validation Rules

- `binwidth` must be positive
- `bincount` must be positive (if specified)
- `label` must be positive
- `glcm_distance` must be positive
- `ngtdm_distance` must be positive
- `outlier_percentile` must be between 0 and 100
- `force_2d_dimension` must be 1, 2, or 3

## Best Practices

1. **Use consistent settings** across a study for comparable results
2. **Document your settings** for reproducibility
3. **Start with defaults** and adjust based on your data characteristics
4. **Consider your intensity range** when choosing binwidth
5. **Validate settings** before batch processing
