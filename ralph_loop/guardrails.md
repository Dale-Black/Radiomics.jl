# Radiomics.jl Guardrails

## Project Vision

Create a **pure Julia** implementation of PyRadiomics that:
- Has zero Python runtime dependencies (only test-time comparison)
- Matches PyRadiomics output exactly for all features
- Follows idiomatic Julia patterns
- Is performant and type-stable

## Current Phase

**Phase 1: Deep Research** - Understanding PyRadiomics architecture and planning Julia implementation.

## Settled Decisions (DO NOT RE-INVESTIGATE)

*This section will be populated as decisions are made during research phases.*

## Julia Coding Conventions

### Type System

```julia
# Use abstract types for extensibility
abstract type AbstractRadiomicsFeature end
abstract type AbstractTextureMatrix end

# Concrete types with fields
struct FirstOrderFeatures <: AbstractRadiomicsFeature
    image::AbstractArray
    mask::AbstractArray{Bool}
    settings::Settings
end
```

### Function Naming

- Use descriptive snake_case for functions: `compute_glcm`, `extract_features`
- Prefix internal functions with underscore: `_normalize_matrix`
- Feature functions match PyRadiomics names in PascalCase: `getEnergy`, `getEntropy`
  - OR use Julian snake_case and map in tests: `energy` → `getEnergy`

### Docstrings

```julia
"""
    energy(voxels::AbstractVector{<:Real}) -> Float64

Compute the energy (sum of squared values) of the input voxels.

# Mathematical Formula
``E = \\sum_{i=1}^{N} X_i^2``

# Arguments
- `voxels`: Vector of voxel intensity values within the ROI

# Returns
- Energy value as Float64

# References
- IBSI reference: 1.1
- PyRadiomics: firstorder.py:getEnergyFeatureValue
"""
function energy(voxels::AbstractVector{<:Real})
    return sum(abs2, voxels)
end
```

### Performance Guidelines

1. **Type stability**: All functions should be type-stable
2. **Avoid allocations in hot loops**: Pre-allocate arrays
3. **Use views**: `@view array[indices]` instead of copying
4. **Broadcast**: Use `.` for element-wise operations
5. **SIMD-friendly**: Structure loops for vectorization

### Error Handling

```julia
# Validate inputs early with informative messages
function compute_glcm(image, mask, distance=1)
    ndims(image) == ndims(mask) || throw(DimensionMismatch(
        "Image and mask must have same dimensions: got $(ndims(image)) and $(ndims(mask))"
    ))
    distance > 0 || throw(ArgumentError("Distance must be positive, got $distance"))
    # ... implementation
end
```

## Testing Conventions

### Test File Structure

```julia
using Test
using Radiomics
using PythonCall

include("test_utils.jl")

@testset "FirstOrder Features" begin
    @testset "Energy" begin
        # Test with multiple seeds
        for seed in [42, 123, 456]
            image, mask = random_image_mask(seed, (32, 32, 32))

            julia_result = Radiomics.energy(image, mask)
            python_result = pyradiomics_feature("firstorder", "Energy", image, mask)

            @test julia_result ≈ python_result rtol=1e-10
        end
    end
end
```

### Random Array Generation

```julia
using Random

function random_image_mask(seed::Int, size::Tuple;
                           intensity_range=(0, 255),
                           mask_fraction=0.3)
    rng = MersenneTwister(seed)

    # Generate image with random intensities
    image = rand(rng, intensity_range[1]:intensity_range[2], size)

    # Generate random mask (binary)
    mask = rand(rng, size) .< mask_fraction

    return image, mask
end
```

### Tolerance Guidelines

| Feature Type | Relative Tolerance | Absolute Tolerance |
|-------------|-------------------|-------------------|
| First Order | 1e-10 | 1e-12 |
| Shape | 1e-6 | 1e-8 |
| Texture (GLCM, etc.) | 1e-10 | 1e-12 |

Use `isapprox(a, b; rtol=..., atol=...)` or `@test a ≈ b rtol=...`

## PyRadiomics Feature Classes

### 1. First Order Statistics
- 19 features based on voxel intensity histogram
- No spatial information used
- Functions: Energy, Entropy, Mean, Median, Variance, etc.

### 2. Shape Features
- 2D: 14 features (perimeter, area, circularity, etc.)
- 3D: 16 features (volume, surface area, sphericity, etc.)
- Requires mesh generation for some features

### 3. GLCM (Gray Level Co-occurrence Matrix)
- 24 features
- Measures texture based on pixel pair relationships
- Configurable: distance, angles (13 directions in 3D)

### 4. GLRLM (Gray Level Run Length Matrix)
- 16 features
- Measures consecutive pixels with same gray level
- Directional (13 directions in 3D)

### 5. GLSZM (Gray Level Size Zone Matrix)
- 16 features
- Measures connected regions of same gray level
- Connectivity-based (6 or 26 connected in 3D)

### 6. NGTDM (Neighboring Gray Tone Difference Matrix)
- 5 features
- Measures difference between voxel and neighborhood average
- Neighborhood distance configurable

### 7. GLDM (Gray Level Dependence Matrix)
- 14 features
- Measures voxels with similar gray levels in neighborhood
- Alpha (coarseness) parameter configurable

## Discretization

PyRadiomics supports two discretization modes:

1. **Fixed Bin Width** (default): `binWidth` parameter
   - Bins = (max - min) / binWidth

2. **Fixed Bin Count**: `binCount` parameter
   - Equal-width bins from min to max

Our implementation MUST match PyRadiomics binning exactly.

## Git Commit Messages

Format: `STORY-ID: Brief description`

Examples:
- `RESEARCH-PYRADIOMICS-ARCH: Document PyRadiomics module structure`
- `IMPL-FIRSTORDER: Implement all 19 first-order features`
- `TEST-GLCM-PARITY: Add GLCM parity tests for all 24 features`

## Dependencies (Pure Julia)

### Core
- `Statistics` (stdlib)
- `LinearAlgebra` (stdlib)
- `StatsBase.jl` - Additional statistics (entropy, etc.)

### Image/Array
- `Images.jl` - Image processing (optional, for I/O)
- `ImageMorphology.jl` - Morphological operations
- `CoordinateTransformations.jl` - Geometry (maybe)

### Mesh/Shape
- `Meshes.jl` or `GeometryBasics.jl` - Mesh generation
- `MarchingCubes.jl` - Isosurface extraction

### Test Only
- `PythonCall.jl` - Call PyRadiomics
- `CondaPkg.jl` - Manage Python environment

## IBSI Compliance

The Image Biomarker Standardization Initiative (IBSI) defines standard feature calculations.
PyRadiomics is IBSI-compliant for most features. We should maintain this compliance.

Reference: https://ibsi.readthedocs.io/

## Performance Targets

- Feature extraction should be competitive with PyRadiomics
- Memory usage should be reasonable (no unnecessary copies)
- Support for large 3D volumes (512³ and larger)

## Notes for Research Phase

During RESEARCH-* stories, document:

1. **Exact function locations** in PyRadiomics repo
2. **Mathematical formulas** for each feature
3. **Edge cases** and special handling
4. **Dependencies** used by each feature
5. **Potential Julia equivalents** for Python libraries

Example research note format:

```markdown
## Feature: Energy (First Order)

**Location**: pyradiomics/firstorder.py:123-135

**Formula**: E = Σᵢ Xᵢ²

**Implementation Notes**:
- Uses numpy.sum with squared values
- No special handling for empty arrays
- Returns float64

**Julia Equivalent**:
```julia
energy(x) = sum(abs2, x)
```
```

## Questions to Resolve During Research

1. How does PyRadiomics handle NaN/Inf values?
2. What is the exact discretization algorithm?
3. How are GLCM directions defined (offsets)?
4. What mesh algorithm is used for shape features?
5. How are edge voxels handled in texture matrices?

These will be answered in RESEARCH-* stories and documented here.
