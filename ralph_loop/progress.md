# Radiomics.jl Development Progress

This file logs the progress of each Ralph loop iteration. Each iteration appends its findings, decisions, and accomplishments here.

---

## Project Initialized

**Date**: 2026-01-17

**Goal**: Port PyRadiomics to pure Julia with 1:1 test parity

**Approach**:
1. Deep research on PyRadiomics architecture and features
2. Set up Julia package with PythonCall test harness
3. Implement each feature class with comprehensive tests
4. Verify parity against PyRadiomics using deterministic random arrays

---

## Iteration Log

*Iterations will be logged below as the Ralph loop executes.*


### Iteration 1 - 2026-01-17

**Story**: RESEARCH-PYRADIOMICS-ARCH
**Status**: ✅ COMPLETED

---

## PyRadiomics Architecture Research Findings

### 1. Repository Directory Structure

**Source**: https://github.com/AIM-Harvard/pyradiomics

```
pyradiomics/
├── radiomics/           # Core package (main implementation)
│   ├── __init__.py      # Public API, version, feature class discovery
│   ├── base.py          # RadiomicsFeaturesBase abstract class
│   ├── featureextractor.py  # RadiomicsFeatureExtractor wrapper class
│   ├── imageoperations.py   # Image/mask handling, discretization, filters
│   ├── firstorder.py    # First-order statistical features (19 features)
│   ├── shape.py         # Shape features 2D/3D (17 features)
│   ├── glcm.py          # Gray Level Co-occurrence Matrix (24 features)
│   ├── glrlm.py         # Gray Level Run Length Matrix (16 features)
│   ├── glszm.py         # Gray Level Size Zone Matrix (16 features)
│   ├── ngtdm.py         # Neighboring Gray Tone Difference Matrix (5 features)
│   ├── gldm.py          # Gray Level Dependence Matrix (14 features)
│   ├── _cmatrices.cpp   # C++ extension for texture matrix computation
│   ├── _cshape.cpp      # C++ extension for shape calculations
│   └── deprecated.py    # Deprecation utilities
├── docs/                # Sphinx documentation
├── tests/               # Test suite
├── examples/            # Usage examples
├── notebooks/           # Jupyter notebooks
├── data/                # Sample test data
└── pyproject.toml       # Project configuration
```

### 2. Python Modules and Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `__init__.py` | Public API entry point | `getFeatureClasses()`, `getImageTypes()`, `setVerbosity()` |
| `base.py` | Abstract base class for all features | `RadiomicsFeaturesBase` |
| `featureextractor.py` | High-level extraction orchestration | `RadiomicsFeatureExtractor` |
| `imageoperations.py` | Image preprocessing, discretization | `checkMask()`, `getBinEdges()`, `binImage()`, filters |
| `firstorder.py` | First-order statistics | `RadiomicsFirstOrder` (19 features) |
| `shape.py` | Geometric shape features | `RadiomicsShape` (17 features, uses marching cubes) |
| `glcm.py` | GLCM texture features | `RadiomicsGLCM` (24 features) |
| `glrlm.py` | Run length texture features | `RadiomicsGLRLM` (16 features) |
| `glszm.py` | Size zone texture features | `RadiomicsGLSZM` (16 features) |
| `ngtdm.py` | Neighborhood difference features | `RadiomicsNGTDM` (5 features) |
| `gldm.py` | Gray level dependence features | `RadiomicsGLDM` (14 features) |

### 3. Core Base Classes and Inheritance Hierarchy

```
object
└── RadiomicsFeaturesBase (base.py)
    ├── RadiomicsFirstOrder (firstorder.py)
    ├── RadiomicsShape (shape.py)
    ├── RadiomicsGLCM (glcm.py)
    ├── RadiomicsGLRLM (glrlm.py)
    ├── RadiomicsGLSZM (glszm.py)
    ├── RadiomicsNGTDM (ngtdm.py)
    └── RadiomicsGLDM (gldm.py)

RadiomicsFeatureExtractor (standalone orchestration class)
```

**RadiomicsFeaturesBase Key Methods**:
- `__init__(inputImage, inputMask, **kwargs)` - Initialize with SimpleITK images
- `_initSegmentBasedCalculation()` - Prepare segment-level extraction
- `_initVoxelBasedCalculation()` - Prepare voxel-level extraction
- `_applyBinning(matrix)` - Discretize gray levels
- `getFeatureNames()` - Class method using reflection to find `get*FeatureValue` methods
- `enableFeatureByName()` / `enableAllFeatures()` - Feature toggling
- `execute()` - Run all enabled features

**Feature Method Naming Convention**: `get<FeatureName>FeatureValue()`

### 4. Image/Mask Handling Approach

**Image Loading** (via SimpleITK):
- Supports NRRD, NIfTI, DICOM, and other formats
- Uses SimpleITK image objects throughout
- Handles 2D and 3D images

**Mask Validation** (`imageoperations.checkMask`):
1. Validates geometry alignment between image and mask
2. Confirms label presence in mask
3. Verifies ROI dimensionality (prevents single-voxel extraction)
4. Optionally checks minimum ROI size
5. Returns bounding box coordinates

**Preprocessing Pipeline**:
1. `cropToTumorMask()` - Crop to ROI with optional padding
2. `resampleImage()` - Resample to specified spacing (B-spline interpolation)
3. `normalizeImage()` - Intensity normalization: `f(x) = (s(x - μ))/σ`
4. `resegmentMask()` - Re-segment based on intensity thresholds (absolute/relative/sigma)

**Discretization** (`imageoperations.getBinEdges`, `binImage`):
- **Fixed Bin Width** (default): `binWidth` parameter, bins = (max - min) / binWidth
- **Fixed Bin Count**: `binCount` parameter, equal-width bins
- Only processes voxels within segmentation mask

### 5. External Dependencies and Their Purposes

| Dependency | Version | Purpose |
|------------|---------|---------|
| **numpy** | ≥2.0 | Array operations, numerical computing |
| **SimpleITK** | ≥2.4.0 | Medical image I/O, spatial transformations |
| **PyWavelets** | ≥1.6.0 | Wavelet filter decomposition |
| **pykwalify** | ≥1.6.0 | YAML/JSON parameter validation |

**C Extensions** (built with scikit-build-core):
- `_cmatrices.cpp` - Fast texture matrix computation (GLCM, GLRLM, GLSZM, NGTDM, GLDM)
- `_cshape.cpp` - Mesh volume, surface area, diameter calculations

**Optional Dependencies**:
- `scikit-image` - LBP2D filter
- `scipy`, `trimesh` - LBP3D filter (spherical harmonics)

### 6. Feature Extraction Pipeline Flow

```
RadiomicsFeatureExtractor.execute(imageFilePath, maskFilePath)
│
├── 1. Load image and mask (SimpleITK)
│   └── loadImage() → validates, normalizes, resamples
│
├── 2. Validate ROI
│   └── checkMask() → bounding box, geometry check
│
├── 3. Add provenance (optional metadata)
│
├── 4. Calculate Shape Features
│   ├── computeShape() on cropped original image
│   └── Shape features are always from original (not filtered)
│
├── 5. Resegment mask (optional, based on intensity thresholds)
│
├── 6. For each enabled image type (filter):
│   │
│   ├── Apply filter (Original, LoG, Wavelet, Square, etc.)
│   │
│   └── For each enabled feature class:
│       │
│       ├── Instantiate feature class with filtered image
│       ├── _initCalculation() → discretization, setup
│       └── execute() → compute all enabled features
│
└── 7. Return OrderedDict
    └── Keys: "{imageType}_{featureClass}_{featureName}"
```

### 7. Entry Points and Public API

**Main Entry Point**: `RadiomicsFeatureExtractor`

```python
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)
result = extractor.execute(imagePath, maskPath)
```

**Configuration Methods**:
- `enableAllFeatures()` / `disableAllFeatures()`
- `enableFeatureClassByName(featureClass, enabled=True)`
- `enableFeaturesByName(**enabledFeatures)`
- `enableAllImageTypes()` / `disableAllImageTypes()`
- `enableImageTypeByName(imageType, enabled=True, customArgs=None)`
- `loadParams(paramsFile)` - Load YAML/JSON config

**Module-Level Functions** (`radiomics/__init__.py`):
- `getFeatureClasses()` - Returns dict of available feature classes
- `getImageTypes()` - Returns list of available image filters
- `setVerbosity(level)` - Control logging (10-60)
- `getTestCase(testCase)` - Get/download test data

**Feature Output Format**:
```python
{
    "original_firstorder_Energy": 1234.5,
    "original_firstorder_Entropy": 4.32,
    "original_shape_Volume": 5678.9,
    "wavelet-HHH_glcm_Contrast": 0.123,
    ...
}
```

### 8. Feature Class Summary

| Class | Features | Matrix/Method | C Extension |
|-------|----------|---------------|-------------|
| FirstOrder | 19 | Histogram statistics | No |
| Shape | 17 | Marching cubes mesh | `_cshape` |
| GLCM | 24 | Co-occurrence pairs, 13 directions | `_cmatrices` |
| GLRLM | 16 | Run length, 13 directions | `_cmatrices` |
| GLSZM | 16 | Connected components | `_cmatrices` |
| NGTDM | 5 | Neighborhood averaging | `_cmatrices` |
| GLDM | 14 | Dependence counting | `_cmatrices` |

**Total Features**: ~111 unique features (×8 image types default = ~1500 total)

### 9. Key Implementation Notes for Julia Port

1. **No SimpleITK needed**: Julia can use raw arrays; Images.jl for I/O
2. **Discretization is critical**: Must match PyRadiomics binning exactly
3. **Direction handling**: GLCM/GLRLM use 13 3D directions (26-connectivity ÷ 2)
4. **C extensions**: Will need pure Julia reimplementation (performance critical)
5. **Marching cubes**: Shape features require mesh generation
6. **Feature naming**: Use `get*FeatureValue` pattern for discovery, or create explicit mapping

### 10. References

- **Repository**: https://github.com/AIM-Harvard/pyradiomics
- **Documentation**: https://pyradiomics.readthedocs.io/
- **Base Class**: `radiomics/base.py`
- **Feature Extractor**: `radiomics/featureextractor.py`
- **Image Operations**: `radiomics/imageoperations.py`
- **IBSI Standard**: https://ibsi.readthedocs.io/

---


### Iteration 2 - 2026-01-17

**Story**: RESEARCH-PYRADIOMICS-FEATURES
**Status**: ✅ COMPLETED

---

## PyRadiomics Feature Classes - Complete Documentation

This section exhaustively documents every feature in every feature class of PyRadiomics, including mathematical formulas, function signatures, edge cases, and IBSI compliance notes.

### Total Feature Count Summary

| Feature Class | Features | IBSI Compliant | Notes |
|--------------|----------|----------------|-------|
| First Order | 19 | Yes (mostly) | Kurtosis differs by +3 from IBSI |
| Shape 3D | 17 | Yes | 3 deprecated features |
| Shape 2D | 10 | Yes | Subset of 3D features |
| GLCM | 24 | Yes | 4 deprecated aliases |
| GLRLM | 16 | Yes | Directional aggregation via mean |
| GLSZM | 16 | Yes | - |
| NGTDM | 5 | Yes | - |
| GLDM | 14 | Yes | - |
| **TOTAL** | **111** | - | Unique features to implement |

---

## 1. First Order Features (19 features)

**Source**: `radiomics/firstorder.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#first-order-features

First-order statistics describe the distribution of voxel intensities within the ROI without considering spatial relationships.

### 1.1 Energy
**Method**: `getEnergyFeatureValue()`
**Formula**:
```
Energy = Σᵢ(X(i) + c)²
```
Where `c` is the voxelArrayShift to prevent negative values.

**Implementation**: `np.nansum((targetVoxelArray + voxelArrayShift)**2)`
**Edge Cases**: Volume-confounded (depends on ROI size)
**IBSI**: Yes

### 1.2 Total Energy
**Method**: `getTotalEnergyFeatureValue()`
**Formula**:
```
Total Energy = V_voxel × Energy
```
Where `V_voxel` is the volume of a single voxel in mm³.

**Implementation**: `energy * voxel_volume`
**Edge Cases**: Not in IBSI standard
**IBSI**: No (PyRadiomics extension)

### 1.3 Entropy
**Method**: `getEntropyFeatureValue()`
**Formula**:
```
Entropy = -Σᵢ p(i) × log₂(p(i) + ε)
```
Where `p(i)` is the histogram probability and `ε ≈ 2.2×10⁻¹⁶`.

**Implementation**: `-np.sum(p_i * np.log2(p_i + epsilon))`
**Edge Cases**: Epsilon prevents log(0)
**IBSI**: Yes

### 1.4 Minimum
**Method**: `getMinimumFeatureValue()`
**Formula**: `Minimum = min(X)`
**Implementation**: `np.nanmin(targetVoxelArray)`
**IBSI**: Yes

### 1.5 10th Percentile
**Method**: `get10PercentileFeatureValue()`
**Formula**: 10th percentile of X
**Implementation**: `np.nanpercentile(targetVoxelArray, 10)`
**IBSI**: Yes

### 1.6 90th Percentile
**Method**: `get90PercentileFeatureValue()`
**Formula**: 90th percentile of X
**Implementation**: `np.nanpercentile(targetVoxelArray, 90)`
**IBSI**: Yes

### 1.7 Maximum
**Method**: `getMaximumFeatureValue()`
**Formula**: `Maximum = max(X)`
**Implementation**: `np.nanmax(targetVoxelArray)`
**IBSI**: Yes

### 1.8 Mean
**Method**: `getMeanFeatureValue()`
**Formula**:
```
Mean = (1/Nₚ) × Σᵢ X(i)
```
**Implementation**: `np.nanmean(targetVoxelArray)`
**IBSI**: Yes

### 1.9 Median
**Method**: `getMedianFeatureValue()`
**Formula**: Median gray level intensity
**Implementation**: `np.nanmedian(targetVoxelArray)`
**IBSI**: Yes

### 1.10 Interquartile Range
**Method**: `getInterquartileRangeFeatureValue()`
**Formula**:
```
IQR = P₇₅ - P₂₅
```
**Implementation**: `np.nanpercentile(75) - np.nanpercentile(25)`
**IBSI**: Yes

### 1.11 Range
**Method**: `getRangeFeatureValue()`
**Formula**:
```
Range = max(X) - min(X)
```
**Implementation**: `np.nanmax() - np.nanmin()`
**IBSI**: Yes

### 1.12 Mean Absolute Deviation
**Method**: `getMeanAbsoluteDeviationFeatureValue()`
**Formula**:
```
MAD = (1/Nₚ) × Σᵢ |X(i) - X̄|
```
**Implementation**: `np.nanmean(np.abs(targetVoxelArray - mean))`
**IBSI**: Yes

### 1.13 Robust Mean Absolute Deviation
**Method**: `getRobustMeanAbsoluteDeviationFeatureValue()`
**Formula**:
```
rMAD = (1/N₁₀₋₉₀) × Σᵢ |X₁₀₋₉₀(i) - X̄₁₀₋₉₀|
```
Only voxels between 10th and 90th percentile are considered.

**Implementation**: Filter to 10-90 percentile range, then compute MAD
**IBSI**: Yes

### 1.14 Root Mean Squared
**Method**: `getRootMeanSquaredFeatureValue()`
**Formula**:
```
RMS = √[(1/Nₚ) × Σᵢ(X(i) + c)²]
```
**Implementation**: `sqrt(np.nansum((shifted)**2) / count)`
**Edge Cases**: Returns 0 if no voxels; voxelArrayShift applied
**IBSI**: Yes

### 1.15 Standard Deviation
**Method**: `getStandardDeviationFeatureValue()`
**Formula**:
```
σ = √[(1/Nₚ) × Σᵢ(X(i) - X̄)²]
```
**Implementation**: `np.nanstd(targetVoxelArray)`
**Edge Cases**: Deprecated (correlated with variance)
**IBSI**: No (use Variance)

### 1.16 Skewness
**Method**: `getSkewnessFeatureValue()`
**Formula**:
```
Skewness = μ₃ / σ³
```
Where μ₃ is the third central moment.

**Implementation**: `m3 / (m2**1.5)` where m2, m3 are central moments
**Edge Cases**: Returns 0 for flat regions (prevents division by zero)
**IBSI**: Yes

### 1.17 Kurtosis
**Method**: `getKurtosisFeatureValue()`
**Formula**:
```
Kurtosis = μ₄ / σ⁴
```
Where μ₄ is the fourth central moment.

**Implementation**: `m4 / (m2**2)`
**Edge Cases**: Returns 0 for flat regions; **DIFFERS FROM IBSI BY +3** (excess kurtosis vs regular)
**IBSI**: Yes (with +3 offset)

### 1.18 Variance
**Method**: `getVarianceFeatureValue()`
**Formula**:
```
Variance = (1/Nₚ) × Σᵢ(X(i) - X̄)²
```
**Implementation**: `np.nanstd()**2`
**IBSI**: Yes

### 1.19 Uniformity
**Method**: `getUniformityFeatureValue()`
**Formula**:
```
Uniformity = Σᵢ p(i)²
```
**Implementation**: `np.nansum(p_i**2)`
**Edge Cases**: Higher values indicate more homogeneous distribution
**IBSI**: Yes

---

## 2. Shape Features 3D (17 features)

**Source**: `radiomics/shape.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#shape-features-3d

Shape features describe the geometric properties of the ROI. They are computed from a triangle mesh generated via marching cubes algorithm and do NOT depend on gray level values.

### 2.1 Mesh Volume
**Method**: `getMeshVolumeFeatureValue()`
**Formula**:
```
V = Σᵢ (Oaᵢ · (Obᵢ × Ocᵢ)) / 6
```
Signed volume calculation from mesh triangle faces.

**Implementation**: Uses triangle mesh vertices, origin-based tetrahedron calculation
**IBSI**: Yes

### 2.2 Voxel Volume
**Method**: `getVoxelVolumeFeatureValue()`
**Formula**:
```
V_voxel = N_v × v
```
Where N_v is voxel count and v is single voxel volume.

**Edge Cases**: Less precise than mesh volume; not used in other shape features
**IBSI**: Yes

### 2.3 Surface Area
**Method**: `getSurfaceAreaFeatureValue()`
**Formula**:
```
A = Σᵢ 0.5 × |aᵢbᵢ × aᵢcᵢ|
```
Sum of triangle areas in mesh (cross product magnitude / 2).

**IBSI**: Yes

### 2.4 Surface-to-Volume Ratio
**Method**: `getSurfaceVolumeRatioFeatureValue()`
**Formula**:
```
SVR = A / V
```
**Edge Cases**: Not dimensionless; lower values indicate more compact shapes
**IBSI**: Yes

### 2.5 Sphericity
**Method**: `getSphericityFeatureValue()`
**Formula**:
```
Sphericity = ∛(36πV²) / A
```
**Edge Cases**: Range: 0 < sphericity ≤ 1; perfect sphere = 1
**IBSI**: Yes

### 2.6 Compactness 1 (DEPRECATED)
**Method**: `getCompactness1FeatureValue()`
**Formula**:
```
Compactness1 = V / √(πA³)
```
**Edge Cases**: Range: 0 < value ≤ 1/(6π); redundant with Sphericity
**IBSI**: Deprecated

### 2.7 Compactness 2 (DEPRECATED)
**Method**: `getCompactness2FeatureValue()`
**Formula**:
```
Compactness2 = 36π(V²/A³)
```
**Edge Cases**: Equals Sphericity³; redundant
**IBSI**: Deprecated

### 2.8 Spherical Disproportion (DEPRECATED)
**Method**: `getSphericalDisproportionFeatureValue()`
**Formula**:
```
SphericalDisproportion = A / ∛(36πV²)
```
**Edge Cases**: Range ≥ 1; inverse of Sphericity
**IBSI**: Deprecated

### 2.9 Maximum 3D Diameter
**Method**: `getMaximum3DDiameterFeatureValue()`
**Formula**: Maximum pairwise Euclidean distance between mesh vertices (Feret diameter)
**IBSI**: Yes

### 2.10 Maximum 2D Diameter (Slice)
**Method**: `getMaximum2DDiameterSliceFeatureValue()`
**Formula**: Maximum distance in row-column (axial) plane projection
**IBSI**: Yes

### 2.11 Maximum 2D Diameter (Column)
**Method**: `getMaximum2DDiameterColumnFeatureValue()`
**Formula**: Maximum distance in row-slice (coronal) plane projection
**IBSI**: Yes

### 2.12 Maximum 2D Diameter (Row)
**Method**: `getMaximum2DDiameterRowFeatureValue()`
**Formula**: Maximum distance in column-slice (sagittal) plane projection
**IBSI**: Yes

### 2.13 Major Axis Length
**Method**: `getMajorAxisLengthFeatureValue()`
**Formula**:
```
MajorAxisLength = 4√λ_major
```
Where λ_major is the largest eigenvalue from PCA on physical voxel coordinates.

**Edge Cases**: Warns and returns NaN if eigenvalue < 0
**IBSI**: Yes

### 2.14 Minor Axis Length
**Method**: `getMinorAxisLengthFeatureValue()`
**Formula**:
```
MinorAxisLength = 4√λ_minor
```
Where λ_minor is the second-largest eigenvalue from PCA.

**Edge Cases**: Warns and returns NaN if eigenvalue < 0
**IBSI**: Yes

### 2.15 Least Axis Length
**Method**: `getLeastAxisLengthFeatureValue()`
**Formula**:
```
LeastAxisLength = 4√λ_least
```
Where λ_least is the smallest eigenvalue from PCA.

**Edge Cases**: May be 0 for 2D segmentations; warns if eigenvalue < 0
**IBSI**: Yes

### 2.16 Elongation
**Method**: `getElongationFeatureValue()`
**Formula**:
```
Elongation = √(λ_minor / λ_major)
```
**Edge Cases**: Range: 0-1; value of 1 indicates circular cross-section
**IBSI**: Yes

### 2.17 Flatness
**Method**: `getFlatnessFeatureValue()`
**Formula**:
```
Flatness = √(λ_least / λ_major)
```
**Edge Cases**: Range: 0-1; measures sphere-likeness vs. flatness
**IBSI**: Yes

---

## 3. Shape Features 2D (10 features)

Computed when `force2D=True`. Subset of 3D features adapted for 2D analysis.

| Feature | 2D Equivalent |
|---------|---------------|
| Mesh Surface | Signed area from perimeter vertices |
| Pixel Surface | Pixel count × pixel area |
| Perimeter | Sum of circumference line lengths |
| Perimeter-to-Surface Ratio | Perimeter / Area |
| Sphericity (2D Circularity) | 2√(πA) / P |
| Spherical Disproportion | P / 2√(πA) |
| Maximum 2D Diameter | Max pairwise vertex distance |
| Major Axis Length | 4√λ_major (2D PCA) |
| Minor Axis Length | 4√λ_minor (2D PCA) |
| Elongation | √(λ_minor / λ_major) |

---

## 4. GLCM Features (24 features)

**Source**: `radiomics/glcm.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#gray-level-co-occurrence-matrix-glcm-features

The Gray Level Co-occurrence Matrix (GLCM) is a second-order joint probability matrix that describes the probability of two voxels with gray levels i and j occurring at a specified distance and direction.

**Matrix Properties**:
- Computed for 13 directions in 3D (26-connectivity ÷ 2)
- Symmetric by default (P(i,j) = P(j,i))
- Normalized to sum to 1

**Notation**:
- P(i,j): GLCM matrix entry
- p_{x-y}(k): Diagonal probability (|i-j| = k)
- p_{x+y}(k): Anti-diagonal probability (i+j = k)
- μₓ, μᵧ: Marginal means
- σₓ, σᵧ: Marginal standard deviations
- HX, HY: Marginal entropies
- ε ≈ 2.2×10⁻¹⁶: Machine epsilon

### 4.1 Autocorrelation
**Method**: `getAutocorrelationFeatureValue()`
**Formula**:
```
Autocorrelation = ΣᵢΣⱼ p(i,j) × i × j
```
**IBSI**: Yes

### 4.2 Joint Average
**Method**: `getJointAverageFeatureValue()`
**Formula**:
```
JointAverage = μₓ = ΣᵢΣⱼ p(i,j) × i
```
**Edge Cases**: Assumes GLCM is symmetrical; warns if not
**IBSI**: Yes

### 4.3 Cluster Prominence
**Method**: `getClusterProminenceFeatureValue()`
**Formula**:
```
ClusterProminence = ΣᵢΣⱼ (i + j - μₓ - μᵧ)⁴ × p(i,j)
```
**IBSI**: Yes

### 4.4 Cluster Shade
**Method**: `getClusterShadeFeatureValue()`
**Formula**:
```
ClusterShade = ΣᵢΣⱼ (i + j - μₓ - μᵧ)³ × p(i,j)
```
**IBSI**: Yes

### 4.5 Cluster Tendency
**Method**: `getClusterTendencyFeatureValue()`
**Formula**:
```
ClusterTendency = ΣᵢΣⱼ (i + j - μₓ - μᵧ)² × p(i,j)
```
**IBSI**: Yes

### 4.6 Contrast
**Method**: `getContrastFeatureValue()`
**Formula**:
```
Contrast = ΣᵢΣⱼ (i - j)² × p(i,j)
```
**IBSI**: Yes

### 4.7 Correlation
**Method**: `getCorrelationFeatureValue()`
**Formula**:
```
Correlation = (ΣᵢΣⱼ p(i,j) × i × j - μₓμᵧ) / (σₓσᵧ)
```
**Edge Cases**: Returns 1 for flat regions where σ=0; range [0,1]
**IBSI**: Yes

### 4.8 Difference Average
**Method**: `getDifferenceAverageFeatureValue()`
**Formula**:
```
DifferenceAverage = Σₖ k × p_{x-y}(k)
```
**IBSI**: Yes

### 4.9 Difference Entropy
**Method**: `getDifferenceEntropyFeatureValue()`
**Formula**:
```
DifferenceEntropy = -Σₖ p_{x-y}(k) × log₂(p_{x-y}(k) + ε)
```
**IBSI**: Yes

### 4.10 Difference Variance
**Method**: `getDifferenceVarianceFeatureValue()`
**Formula**:
```
DifferenceVariance = Σₖ (k - DA)² × p_{x-y}(k)
```
Where DA is Difference Average.
**IBSI**: Yes

### 4.11 Joint Energy (Angular Second Moment)
**Method**: `getJointEnergyFeatureValue()`
**Formula**:
```
JointEnergy = ΣᵢΣⱼ (p(i,j))²
```
**IBSI**: Yes (IBSI name: Angular Second Moment)

### 4.12 Joint Entropy
**Method**: `getJointEntropyFeatureValue()`
**Formula**:
```
JointEntropy = -ΣᵢΣⱼ p(i,j) × log₂(p(i,j) + ε)
```
**IBSI**: Yes

### 4.13 IMC1 (Informational Measure of Correlation 1)
**Method**: `getImc1FeatureValue()`
**Formula**:
```
IMC1 = (HXY - HXY1) / max(HX, HY)
```
Where HXY is joint entropy, HXY1 = -ΣᵢΣⱼ p(i,j) × log₂(pₓ(i)pᵧ(j) + ε)

**Edge Cases**: Returns 0 when both HX and HY are 0 (flat regions)
**IBSI**: Yes

### 4.14 IMC2 (Informational Measure of Correlation 2)
**Method**: `getImc2FeatureValue()`
**Formula**:
```
IMC2 = √(1 - e^(-2(HXY2 - HXY)))
```
Where HXY2 = -ΣᵢΣⱼ pₓ(i)pᵧ(j) × log₂(pₓ(i)pᵧ(j) + ε)

**Edge Cases**: Returns 0 when HXY > HXY2 due to floating-point precision
**IBSI**: Yes

### 4.15 IDM (Inverse Difference Moment / Homogeneity 2)
**Method**: `getIdmFeatureValue()`
**Formula**:
```
IDM = Σₖ p_{x-y}(k) / (1 + k²)
```
**IBSI**: Yes

### 4.16 MCC (Maximal Correlation Coefficient)
**Method**: `getMCCFeatureValue()`
**Formula**:
```
MCC = √(second largest eigenvalue of Q)
```
Where Q(i,j) = Σₖ (p(i,k)p(j,k)) / (pₓ(i)pₓ(j))

**Edge Cases**: Returns 1 for flat regions with shape (1,1); range [0,1]
**IBSI**: Yes

### 4.17 IDMN (Inverse Difference Moment Normalized)
**Method**: `getIdmnFeatureValue()`
**Formula**:
```
IDMN = Σₖ p_{x-y}(k) / (1 + k²/Nᵧ²)
```
**IBSI**: Yes

### 4.18 ID (Inverse Difference / Homogeneity 1)
**Method**: `getIdFeatureValue()`
**Formula**:
```
ID = Σₖ p_{x-y}(k) / (1 + k)
```
**IBSI**: Yes

### 4.19 IDN (Inverse Difference Normalized)
**Method**: `getIdnFeatureValue()`
**Formula**:
```
IDN = Σₖ p_{x-y}(k) / (1 + k/Nᵧ)
```
**IBSI**: Yes

### 4.20 Inverse Variance
**Method**: `getInverseVarianceFeatureValue()`
**Formula**:
```
InverseVariance = Σₖ₌₁ p_{x-y}(k) / k²
```
**Edge Cases**: k=0 is skipped to avoid division by zero
**IBSI**: Yes

### 4.21 Maximum Probability
**Method**: `getMaximumProbabilityFeatureValue()`
**Formula**:
```
MaximumProbability = max(p(i,j))
```
**IBSI**: Yes (IBSI name: Joint Maximum)

### 4.22 Sum Average
**Method**: `getSumAverageFeatureValue()`
**Formula**:
```
SumAverage = Σₖ₌₂ p_{x+y}(k) × k
```
**Edge Cases**: Warning: Sum Average = 2 × Joint Average when symmetric; disabled by default
**IBSI**: Yes

### 4.23 Sum Entropy
**Method**: `getSumEntropyFeatureValue()`
**Formula**:
```
SumEntropy = -Σₖ p_{x+y}(k) × log₂(p_{x+y}(k) + ε)
```
**IBSI**: Yes

### 4.24 Sum of Squares (Variance)
**Method**: `getSumSquaresFeatureValue()`
**Formula**:
```
SumSquares = ΣᵢΣⱼ (i - μₓ)² × p(i,j)
```
**Edge Cases**: Assumes symmetrical GLCM
**IBSI**: Yes (IBSI name: Joint Variance)

**Deprecated GLCM Features** (aliases, not counted):
- Dissimilarity = Difference Average
- Homogeneity1 = Inverse Difference
- Homogeneity2 = Inverse Difference Moment
- SumVariance = Cluster Tendency

---

## 5. GLRLM Features (16 features)

**Source**: `radiomics/glrlm.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#gray-level-run-length-matrix-glrlm-features

The Gray Level Run Length Matrix (GLRLM) quantifies runs of consecutive voxels with the same gray level value along a given direction.

**Matrix Properties**:
- P(i,j|θ): Number of runs of gray level i with length j in direction θ
- Computed for 13 directions in 3D
- Nr(θ): Total number of runs in direction θ
- Nₚ: Total number of voxels in ROI

**Aggregation**: Features are averaged across all directions using `np.nanmean()`

### 5.1 Short Run Emphasis (SRE)
**Method**: `getShortRunEmphasisFeatureValue()`
**Formula**:
```
SRE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) / j²
```
**IBSI**: Yes

### 5.2 Long Run Emphasis (LRE)
**Method**: `getLongRunEmphasisFeatureValue()`
**Formula**:
```
LRE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) × j²
```
**IBSI**: Yes

### 5.3 Gray Level Non-Uniformity (GLN)
**Method**: `getGrayLevelNonUniformityFeatureValue()`
**Formula**:
```
GLN = (1/Nr(θ)) × Σᵢ (Σⱼ P(i,j|θ))²
```
**IBSI**: Yes

### 5.4 Gray Level Non-Uniformity Normalized (GLNN)
**Method**: `getGrayLevelNonUniformityNormalizedFeatureValue()`
**Formula**:
```
GLNN = (1/Nr(θ)²) × Σᵢ (Σⱼ P(i,j|θ))²
```
**IBSI**: Yes

### 5.5 Run Length Non-Uniformity (RLN)
**Method**: `getRunLengthNonUniformityFeatureValue()`
**Formula**:
```
RLN = (1/Nr(θ)) × Σⱼ (Σᵢ P(i,j|θ))²
```
**IBSI**: Yes

### 5.6 Run Length Non-Uniformity Normalized (RLNN)
**Method**: `getRunLengthNonUniformityNormalizedFeatureValue()`
**Formula**:
```
RLNN = (1/Nr(θ)²) × Σⱼ (Σᵢ P(i,j|θ))²
```
**IBSI**: Yes

### 5.7 Run Percentage (RP)
**Method**: `getRunPercentageFeatureValue()`
**Formula**:
```
RP = Nr(θ) / Nₚ
```
**Edge Cases**: Adjusts Nₚ for weighted matrices
**IBSI**: Yes

### 5.8 Gray Level Variance (GLV)
**Method**: `getGrayLevelVarianceFeatureValue()`
**Formula**:
```
GLV = ΣᵢΣⱼ p(i,j|θ) × (i - μ)²
where μ = ΣᵢΣⱼ p(i,j|θ) × i
```
**IBSI**: Yes

### 5.9 Run Variance (RV)
**Method**: `getRunVarianceFeatureValue()`
**Formula**:
```
RV = ΣᵢΣⱼ p(i,j|θ) × (j - μ)²
where μ = ΣᵢΣⱼ p(i,j|θ) × j
```
**IBSI**: Yes

### 5.10 Run Entropy (RE)
**Method**: `getRunEntropyFeatureValue()`
**Formula**:
```
RE = -ΣᵢΣⱼ p(i,j|θ) × log₂(p(i,j|θ) + ε)
```
**IBSI**: Yes

### 5.11 Low Gray Level Run Emphasis (LGLRE)
**Method**: `getLowGrayLevelRunEmphasisFeatureValue()`
**Formula**:
```
LGLRE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) / i²
```
**IBSI**: Yes

### 5.12 High Gray Level Run Emphasis (HGLRE)
**Method**: `getHighGrayLevelRunEmphasisFeatureValue()`
**Formula**:
```
HGLRE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) × i²
```
**IBSI**: Yes

### 5.13 Short Run Low Gray Level Emphasis (SRLGLE)
**Method**: `getShortRunLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SRLGLE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) / (i² × j²)
```
**IBSI**: Yes

### 5.14 Short Run High Gray Level Emphasis (SRHGLE)
**Method**: `getShortRunHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SRHGLE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) × i² / j²
```
**IBSI**: Yes

### 5.15 Long Run Low Gray Level Emphasis (LRLGLE)
**Method**: `getLongRunLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LRLGLE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) × j² / i²
```
**IBSI**: Yes

### 5.16 Long Run High Gray Level Emphasis (LRHGLE)
**Method**: `getLongRunHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LRHGLE = (1/Nr(θ)) × ΣᵢΣⱼ P(i,j|θ) × i² × j²
```
**IBSI**: Yes

---

## 6. GLSZM Features (16 features)

**Source**: `radiomics/glszm.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#gray-level-size-zone-matrix-glszm-features

The Gray Level Size Zone Matrix (GLSZM) quantifies connected regions (zones) of voxels with the same gray level intensity.

**Matrix Properties**:
- P(i,j): Number of zones with gray level i and size j
- Nz: Total number of zones
- Nₚ: Total number of voxels in ROI
- Connected components use 26-connectivity by default

### 6.1 Small Area Emphasis (SAE)
**Method**: `getSmallAreaEmphasisFeatureValue()`
**Formula**:
```
SAE = (1/Nz) × ΣᵢΣⱼ P(i,j) / j²
```
**IBSI**: Yes

### 6.2 Large Area Emphasis (LAE)
**Method**: `getLargeAreaEmphasisFeatureValue()`
**Formula**:
```
LAE = (1/Nz) × ΣᵢΣⱼ P(i,j) × j²
```
**IBSI**: Yes

### 6.3 Gray Level Non-Uniformity (GLN)
**Method**: `getGrayLevelNonUniformityFeatureValue()`
**Formula**:
```
GLN = (1/Nz) × Σᵢ (Σⱼ P(i,j))²
```
**IBSI**: Yes

### 6.4 Gray Level Non-Uniformity Normalized (GLNN)
**Method**: `getGrayLevelNonUniformityNormalizedFeatureValue()`
**Formula**:
```
GLNN = (1/Nz²) × Σᵢ (Σⱼ P(i,j))²
```
**IBSI**: Yes

### 6.5 Size-Zone Non-Uniformity (SZN)
**Method**: `getSizeZoneNonUniformityFeatureValue()`
**Formula**:
```
SZN = (1/Nz) × Σⱼ (Σᵢ P(i,j))²
```
**IBSI**: Yes

### 6.6 Size-Zone Non-Uniformity Normalized (SZNN)
**Method**: `getSizeZoneNonUniformityNormalizedFeatureValue()`
**Formula**:
```
SZNN = (1/Nz²) × Σⱼ (Σᵢ P(i,j))²
```
**IBSI**: Yes

### 6.7 Zone Percentage (ZP)
**Method**: `getZonePercentageFeatureValue()`
**Formula**:
```
ZP = Nz / Nₚ
```
**Edge Cases**: Nz and Nₚ set to 1 if equal to 0 (prevents division errors)
**IBSI**: Yes

### 6.8 Gray Level Variance (GLV)
**Method**: `getGrayLevelVarianceFeatureValue()`
**Formula**:
```
GLV = ΣᵢΣⱼ p(i,j) × (i - μ)²
where μ = ΣᵢΣⱼ p(i,j) × i
```
**IBSI**: Yes

### 6.9 Zone Variance (ZV)
**Method**: `getZoneVarianceFeatureValue()`
**Formula**:
```
ZV = ΣᵢΣⱼ p(i,j) × (j - μ)²
where μ = ΣᵢΣⱼ p(i,j) × j
```
**IBSI**: Yes

### 6.10 Zone Entropy (ZE)
**Method**: `getZoneEntropyFeatureValue()`
**Formula**:
```
ZE = -ΣᵢΣⱼ p(i,j) × log₂(p(i,j) + ε)
```
**IBSI**: Yes

### 6.11 Low Gray Level Zone Emphasis (LGLZE)
**Method**: `getLowGrayLevelZoneEmphasisFeatureValue()`
**Formula**:
```
LGLZE = (1/Nz) × ΣᵢΣⱼ P(i,j) / i²
```
**IBSI**: Yes

### 6.12 High Gray Level Zone Emphasis (HGLZE)
**Method**: `getHighGrayLevelZoneEmphasisFeatureValue()`
**Formula**:
```
HGLZE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i²
```
**IBSI**: Yes

### 6.13 Small Area Low Gray Level Emphasis (SALGLE)
**Method**: `getSmallAreaLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SALGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) / (i² × j²)
```
**IBSI**: Yes

### 6.14 Small Area High Gray Level Emphasis (SAHGLE)
**Method**: `getSmallAreaHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SAHGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i² / j²
```
**IBSI**: Yes

### 6.15 Large Area Low Gray Level Emphasis (LALGLE)
**Method**: `getLargeAreaLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LALGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × j² / i²
```
**IBSI**: Yes

### 6.16 Large Area High Gray Level Emphasis (LAHGLE)
**Method**: `getLargeAreaHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LAHGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i² × j²
```
**IBSI**: Yes

---

## 7. NGTDM Features (5 features)

**Source**: `radiomics/ngtdm.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#neighbouring-gray-tone-difference-matrix-ngtdm-features

The Neighboring Gray Tone Difference Matrix (NGTDM) quantifies the difference between a voxel's gray level and the average gray level of its neighbors.

**Matrix Properties**:
- pᵢ: Probability of gray level i occurring
- sᵢ: Sum of absolute differences |i - Ā| for all voxels with gray level i
- Nᵧ,ₚ: Number of discrete gray levels with pᵢ > 0
- Nᵥ,ₚ: Number of voxels with valid neighbors

### 7.1 Coarseness
**Method**: `getCoarsenessFeatureValue()`
**Formula**:
```
Coarseness = 1 / Σᵢ (pᵢ × sᵢ)
```
**Edge Cases**: Returns 10⁶ when denominator = 0 (completely homogeneous image)
**IBSI**: Yes

### 7.2 Contrast
**Method**: `getContrastFeatureValue()`
**Formula**:
```
Contrast = [1/(Nᵧ,ₚ(Nᵧ,ₚ-1)) × ΣᵢΣⱼ pᵢpⱼ(i-j)²] × [1/Nᵥ,ₚ × Σᵢ sᵢ]
```
(where pᵢ ≠ 0, pⱼ ≠ 0)

**Edge Cases**: Returns 0 when Nᵧ,ₚ = 1 (homogeneous image)
**IBSI**: Yes

### 7.3 Busyness
**Method**: `getBusynessFeatureValue()`
**Formula**:
```
Busyness = Σᵢ(pᵢ × sᵢ) / ΣᵢΣⱼ |i × pᵢ - j × pⱼ|
```
(where pᵢ ≠ 0, pⱼ ≠ 0)

**Edge Cases**: Returns 0 when Nᵧ,ₚ = 1 (produces 0/0)
**IBSI**: Yes

### 7.4 Complexity
**Method**: `getComplexityFeatureValue()`
**Formula**:
```
Complexity = (1/Nᵥ,ₚ) × ΣᵢΣⱼ [|i-j| × (pᵢsᵢ + pⱼsⱼ) / (pᵢ + pⱼ)]
```
(where pᵢ ≠ 0, pⱼ ≠ 0)

**Edge Cases**: Division by zero prevented by setting zero divisors to 1
**IBSI**: Yes

### 7.5 Strength
**Method**: `getStrengthFeatureValue()`
**Formula**:
```
Strength = ΣᵢΣⱼ [(pᵢ + pⱼ)(i-j)²] / Σᵢ sᵢ
```
(where pᵢ ≠ 0, pⱼ ≠ 0)

**Edge Cases**: Returns 0 when Σsᵢ = 0 (completely homogeneous image)
**IBSI**: Yes

---

## 8. GLDM Features (14 features)

**Source**: `radiomics/gldm.py`
**Documentation**: https://pyradiomics.readthedocs.io/en/latest/features.html#gray-level-dependence-matrix-gldm-features

The Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in the image. A voxel is considered dependent on its neighbor if the absolute difference in their gray levels is ≤ α (alpha, coarseness parameter).

**Matrix Properties**:
- P(i,j): Number of voxels with gray level i and j dependent neighbors
- Nz: Total number of dependency entries (sum of matrix)
- α: Coarseness parameter (default: 0)
- Neighborhood: 26-connectivity

### 8.1 Small Dependence Emphasis (SDE)
**Method**: `getSmallDependenceEmphasisFeatureValue()`
**Formula**:
```
SDE = (1/Nz) × ΣᵢΣⱼ P(i,j) / j²
```
**IBSI**: Yes

### 8.2 Large Dependence Emphasis (LDE)
**Method**: `getLargeDependenceEmphasisFeatureValue()`
**Formula**:
```
LDE = (1/Nz) × ΣᵢΣⱼ P(i,j) × j²
```
**IBSI**: Yes

### 8.3 Gray Level Non-Uniformity (GLN)
**Method**: `getGrayLevelNonUniformityFeatureValue()`
**Formula**:
```
GLN = (1/Nz) × Σᵢ (Σⱼ P(i,j))²
```
**IBSI**: Yes

### 8.4 Dependence Non-Uniformity (DN)
**Method**: `getDependenceNonUniformityFeatureValue()`
**Formula**:
```
DN = (1/Nz) × Σⱼ (Σᵢ P(i,j))²
```
**IBSI**: Yes

### 8.5 Dependence Non-Uniformity Normalized (DNN)
**Method**: `getDependenceNonUniformityNormalizedFeatureValue()`
**Formula**:
```
DNN = (1/Nz²) × Σⱼ (Σᵢ P(i,j))²
```
**IBSI**: Yes

### 8.6 Gray Level Variance (GLV)
**Method**: `getGrayLevelVarianceFeatureValue()`
**Formula**:
```
GLV = ΣᵢΣⱼ p(i,j) × (i - μ)²
where μ = ΣᵢΣⱼ p(i,j) × i
```
**IBSI**: Yes

### 8.7 Dependence Variance (DV)
**Method**: `getDependenceVarianceFeatureValue()`
**Formula**:
```
DV = ΣᵢΣⱼ p(i,j) × (j - μ)²
where μ = ΣᵢΣⱼ p(i,j) × j
```
**IBSI**: Yes

### 8.8 Dependence Entropy (DE)
**Method**: `getDependenceEntropyFeatureValue()`
**Formula**:
```
DE = -ΣᵢΣⱼ p(i,j) × log₂(p(i,j) + ε)
```
**IBSI**: Yes

### 8.9 Low Gray Level Emphasis (LGLE)
**Method**: `getLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) / i²
```
**IBSI**: Yes

### 8.10 High Gray Level Emphasis (HGLE)
**Method**: `getHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
HGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i²
```
**IBSI**: Yes

### 8.11 Small Dependence Low Gray Level Emphasis (SDLGLE)
**Method**: `getSmallDependenceLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SDLGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) / (i² × j²)
```
**IBSI**: Yes

### 8.12 Small Dependence High Gray Level Emphasis (SDHGLE)
**Method**: `getSmallDependenceHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
SDHGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i² / j²
```
**IBSI**: Yes

### 8.13 Large Dependence Low Gray Level Emphasis (LDLGLE)
**Method**: `getLargeDependenceLowGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LDLGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × j² / i²
```
**IBSI**: Yes

### 8.14 Large Dependence High Gray Level Emphasis (LDHGLE)
**Method**: `getLargeDependenceHighGrayLevelEmphasisFeatureValue()`
**Formula**:
```
LDHGLE = (1/Nz) × ΣᵢΣⱼ P(i,j) × i² × j²
```
**IBSI**: Yes

---

## 9. Discretization Methods

PyRadiomics uses discretization to reduce the number of gray levels before computing texture features.

### 9.1 Fixed Bin Width (FBW) - Default

**Formula**:
```
X_b,i = ⌊X_gl,i / W⌋ - ⌊min(X_gl) / W⌋ + 1
```
Where W is the bin width parameter.

**Implementation Notes**:
- Bins start from zero and extend to capture the full range
- Single-value regions get symmetric bin boundaries
- Default bin width: 25 (for CT images in HU)

### 9.2 Fixed Bin Count (FBC)

**Formula**:
```
For X < max(X): ⌊N_b × (X_gl,i - min(X_gl)) / (max(X_gl) - min(X_gl))⌋ + 1
For X = max(X): N_b
```
Where N_b is the number of bins.

**Implementation Notes**:
- Maximum value receives its own bin
- Recommended range: 30-130 bins for good reproducibility

---

## 10. IBSI Compliance Notes

PyRadiomics is largely IBSI-compliant with the following exceptions:

1. **Kurtosis**: PyRadiomics returns excess kurtosis (μ₄/σ⁴ - 3), while IBSI expects regular kurtosis (μ₄/σ⁴). The difference is +3.

2. **Total Energy**: PyRadiomics extension, not in IBSI standard.

3. **Standard Deviation**: Deprecated in favor of Variance (they are mathematically related).

4. **Deprecated Shape Features**: Compactness1, Compactness2, SphericalDisproportion are redundant with Sphericity.

---

## 11. Implementation Checklist for Julia

### First Order (19 features)
- [ ] Energy
- [ ] TotalEnergy
- [ ] Entropy
- [ ] Minimum
- [ ] 10Percentile
- [ ] 90Percentile
- [ ] Maximum
- [ ] Mean
- [ ] Median
- [ ] InterquartileRange
- [ ] Range
- [ ] MeanAbsoluteDeviation
- [ ] RobustMeanAbsoluteDeviation
- [ ] RootMeanSquared
- [ ] StandardDeviation (deprecated)
- [ ] Skewness
- [ ] Kurtosis
- [ ] Variance
- [ ] Uniformity

### Shape 3D (17 features)
- [ ] MeshVolume
- [ ] VoxelVolume
- [ ] SurfaceArea
- [ ] SurfaceVolumeRatio
- [ ] Sphericity
- [ ] Compactness1 (deprecated)
- [ ] Compactness2 (deprecated)
- [ ] SphericalDisproportion (deprecated)
- [ ] Maximum3DDiameter
- [ ] Maximum2DDiameterSlice
- [ ] Maximum2DDiameterColumn
- [ ] Maximum2DDiameterRow
- [ ] MajorAxisLength
- [ ] MinorAxisLength
- [ ] LeastAxisLength
- [ ] Elongation
- [ ] Flatness

### GLCM (24 features)
- [ ] Autocorrelation
- [ ] JointAverage
- [ ] ClusterProminence
- [ ] ClusterShade
- [ ] ClusterTendency
- [ ] Contrast
- [ ] Correlation
- [ ] DifferenceAverage
- [ ] DifferenceEntropy
- [ ] DifferenceVariance
- [ ] JointEnergy
- [ ] JointEntropy
- [ ] Imc1
- [ ] Imc2
- [ ] Idm
- [ ] MCC
- [ ] Idmn
- [ ] Id
- [ ] Idn
- [ ] InverseVariance
- [ ] MaximumProbability
- [ ] SumAverage
- [ ] SumEntropy
- [ ] SumSquares

### GLRLM (16 features)
- [ ] ShortRunEmphasis
- [ ] LongRunEmphasis
- [ ] GrayLevelNonUniformity
- [ ] GrayLevelNonUniformityNormalized
- [ ] RunLengthNonUniformity
- [ ] RunLengthNonUniformityNormalized
- [ ] RunPercentage
- [ ] GrayLevelVariance
- [ ] RunVariance
- [ ] RunEntropy
- [ ] LowGrayLevelRunEmphasis
- [ ] HighGrayLevelRunEmphasis
- [ ] ShortRunLowGrayLevelEmphasis
- [ ] ShortRunHighGrayLevelEmphasis
- [ ] LongRunLowGrayLevelEmphasis
- [ ] LongRunHighGrayLevelEmphasis

### GLSZM (16 features)
- [ ] SmallAreaEmphasis
- [ ] LargeAreaEmphasis
- [ ] GrayLevelNonUniformity
- [ ] GrayLevelNonUniformityNormalized
- [ ] SizeZoneNonUniformity
- [ ] SizeZoneNonUniformityNormalized
- [ ] ZonePercentage
- [ ] GrayLevelVariance
- [ ] ZoneVariance
- [ ] ZoneEntropy
- [ ] LowGrayLevelZoneEmphasis
- [ ] HighGrayLevelZoneEmphasis
- [ ] SmallAreaLowGrayLevelEmphasis
- [ ] SmallAreaHighGrayLevelEmphasis
- [ ] LargeAreaLowGrayLevelEmphasis
- [ ] LargeAreaHighGrayLevelEmphasis

### NGTDM (5 features)
- [ ] Coarseness
- [ ] Contrast
- [ ] Busyness
- [ ] Complexity
- [ ] Strength

### GLDM (14 features)
- [ ] SmallDependenceEmphasis
- [ ] LargeDependenceEmphasis
- [ ] GrayLevelNonUniformity
- [ ] DependenceNonUniformity
- [ ] DependenceNonUniformityNormalized
- [ ] GrayLevelVariance
- [ ] DependenceVariance
- [ ] DependenceEntropy
- [ ] LowGrayLevelEmphasis
- [ ] HighGrayLevelEmphasis
- [ ] SmallDependenceLowGrayLevelEmphasis
- [ ] SmallDependenceHighGrayLevelEmphasis
- [ ] LargeDependenceLowGrayLevelEmphasis
- [ ] LargeDependenceHighGrayLevelEmphasis

---

## 12. References

- **PyRadiomics Source**: https://github.com/AIM-Harvard/pyradiomics
- **PyRadiomics Docs**: https://pyradiomics.readthedocs.io/en/latest/features.html
- **IBSI Standard**: https://theibsi.github.io/
- **IBSI Reference Manual**: https://arxiv.org/pdf/1612.07003

---


### Iteration 3 - 2026-01-17

**Story**: RESEARCH-JULIA-PORTING
**Status**: ✅ COMPLETED

---

## Julia Porting Best Practices Research Findings

This section documents idiomatic Julia patterns, ecosystem packages, and best practices for porting PyRadiomics from Python to Julia.

### 1. Python-to-Julia Port Patterns (Lessons from MLJ.jl)

**Source**: [MLJ.jl FAQ](https://juliaai.github.io/MLJ.jl/stable/frequently_asked_questions/), [MLJ Blog Post](https://julialang.org/blog/2019/05/beyond-ml-pipelines-with-mlj/)

**Key Design Decisions**:

1. **Use Multiple Dispatch Instead of Inheritance**
   - Python scikit-learn uses class inheritance for feature classes
   - MLJ uses multiple dispatch on abstract types
   - Functions should work on abstract types, with specific methods for concrete types

2. **Handle Categorical Data Properly**
   - Julia has native categorical data types (CategoricalArrays.jl)
   - Unlike NumPy, Julia can represent categorical data without integer encoding
   - Our implementation should use proper types, not just raw integers for discretized values

3. **Composable Pipelines**
   - MLJ's design was predicated on flexible "learning network" API
   - Networks can be built and tested in stages
   - Features "smart" training (only necessary components retrained)

4. **Probabilistic API**
   - Define clear interfaces for outputs (Dict, NamedTuple)
   - Consistent return types across all feature functions

**Applicable Pattern for Radiomics.jl**:
```julia
# Abstract type hierarchy for features
abstract type AbstractRadiomicsFeature end
abstract type AbstractFirstOrderFeature <: AbstractRadiomicsFeature end
abstract type AbstractTextureFeature <: AbstractRadiomicsFeature end

# Multiple dispatch for feature computation
function compute(feature::AbstractFirstOrderFeature, voxels::AbstractVector)
    error("compute() not implemented for $(typeof(feature))")
end

# Concrete types carry parameters
struct Energy <: AbstractFirstOrderFeature end
struct Entropy <: AbstractFirstOrderFeature end

# Specific implementations
compute(::Energy, voxels::AbstractVector) = sum(abs2, voxels)
```

### 2. Julian Idioms for NumPy-Style Operations

**Sources**:
- [Julia Arrays Documentation](https://docs.julialang.org/en/v1/manual/arrays/)
- [Broadcasting and Dot Syntax](https://julialang.org/blog/2017/01/moredots/)

#### 2.1 Broadcasting (NumPy-style element-wise operations)

| NumPy | Julia |
|-------|-------|
| `np.sin(a)` | `sin.(a)` |
| `a * b` (element-wise) | `a .* b` |
| `a + b` (element-wise) | `a .+ b` |
| `a ** 2` | `a .^ 2` |
| `np.sum(a**2)` | `sum(abs2, a)` |

**Key Points**:
- Nested dot calls are fused: `sin.(cos.(x))` is one loop
- Use `@.` macro to dot everything: `@. sin(cos(x) + 1)`
- Broadcasting is automatic for scalars and arrays of different shapes

#### 2.2 Views vs. Copies

| NumPy (default view) | Julia (default copy) | Julia (view) |
|---------------------|---------------------|--------------|
| `a[1:10]` | `a[1:10]` (copy) | `@view a[1:10]` |
| `a[mask]` | `a[mask]` (copy) | `@view a[mask]` |

**Performance Tip**: Use `@views` macro to convert all slices in a block to views:
```julia
@views function process(arr)
    for i in 1:size(arr, 3)
        slice = arr[:, :, i]  # This is now a view, not a copy
        # process slice...
    end
end
```

#### 2.3 NumPy to Julia Function Mapping

| NumPy/SciPy | Julia |
|-------------|-------|
| `np.sum(a)` | `sum(a)` |
| `np.mean(a)` | `mean(a)` (Statistics) |
| `np.std(a)` | `std(a)` (Statistics) |
| `np.var(a)` | `var(a)` (Statistics) |
| `np.nansum(a)` | `nansum(a)` (NaNStatistics.jl) |
| `np.nanmean(a)` | `nanmean(a)` (NaNStatistics.jl) |
| `np.percentile(a, p)` | `percentile(a, p)` (StatsBase.jl) |
| `np.histogram(a)` | `fit(Histogram, a)` (StatsBase.jl) |
| `np.where(cond)` | `findall(cond)` |
| `np.argmax(a)` | `argmax(a)` |
| `np.abs(a)` | `abs.(a)` |
| `np.log2(a)` | `log2.(a)` |
| `np.sqrt(a)` | `sqrt.(a)` |
| `a.T` (transpose) | `transpose(a)` or `a'` (adjoint) |
| `a.flatten()` | `vec(a)` |
| `a.reshape(m, n)` | `reshape(a, m, n)` |
| `np.concatenate([a, b])` | `vcat(a, b)` or `[a; b]` |

### 3. Julia Equivalents for SciPy Functions

**Sources**:
- [StatsBase.jl](https://juliastats.org/StatsBase.jl/stable/)
- [NaNStatistics.jl](https://brenhinkeller.github.io/NaNStatistics.jl/dev/)

#### 3.1 Statistics (scipy.stats equivalents)

| SciPy | Julia Package | Julia Function |
|-------|---------------|----------------|
| `scipy.stats.skew` | Statistics | `skewness(x)` (StatsBase) |
| `scipy.stats.kurtosis` | Statistics | `kurtosis(x)` (StatsBase) |
| `scipy.stats.entropy` | StatsBase | `entropy(p)` |
| `scipy.stats.percentileofscore` | StatsBase | `percentilerank(x, v)` |

#### 3.2 Histograms

```julia
using StatsBase

# Basic histogram
h = fit(Histogram, data)

# With specified edges
h = fit(Histogram, data, 0:10:100)

# With specified number of bins
h = fit(Histogram, data, nbins=10)

# Access histogram data
h.edges   # bin edges
h.weights # counts per bin
```

#### 3.3 Percentiles

```julia
using Statistics, StatsBase

# Basic percentile
p10 = percentile(x, 10)
p90 = percentile(x, 90)

# Quantiles
q = quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])

# Interquartile range
iqr = percentile(x, 75) - percentile(x, 25)
# OR
iqr = iqr(x)  # from StatsBase
```

### 4. Type System Design Patterns for Feature Extraction

**Sources**:
- [Type-Dispatch Design](http://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/)
- [Julia Types Documentation](https://docs.julialang.org/en/v1/manual/types/)

#### 4.1 Abstract Type Hierarchy

```julia
# Root abstract type for all radiomics features
abstract type AbstractRadiomicsFeature end

# Feature class abstract types
abstract type AbstractFirstOrderFeature <: AbstractRadiomicsFeature end
abstract type AbstractShapeFeature <: AbstractRadiomicsFeature end
abstract type AbstractTextureFeature <: AbstractRadiomicsFeature end

# Texture matrix specific types
abstract type AbstractGLCMFeature <: AbstractTextureFeature end
abstract type AbstractGLRLMFeature <: AbstractTextureFeature end
abstract type AbstractGLSZMFeature <: AbstractTextureFeature end
abstract type AbstractNGTDMFeature <: AbstractTextureFeature end
abstract type AbstractGLDMFeature <: AbstractTextureFeature end
```

#### 4.2 Concrete Feature Types

```julia
# Singleton types for features (no state)
struct Energy <: AbstractFirstOrderFeature end
struct Entropy <: AbstractFirstOrderFeature end
struct Skewness <: AbstractFirstOrderFeature end

# Types with parameters
struct Kurtosis <: AbstractFirstOrderFeature
    excess::Bool  # Whether to return excess kurtosis
end
Kurtosis() = Kurtosis(true)  # Default constructor
```

#### 4.3 Multiple Dispatch for Feature Computation

```julia
# Generic interface
function compute end

# Specific implementations
compute(::Energy, voxels::AbstractVector{<:Real}) = sum(abs2, voxels)

compute(::Entropy, voxels::AbstractVector{<:Real}) = begin
    p = fit(Histogram, voxels, closed=:left).weights
    p = p ./ sum(p)
    -sum(p_i -> p_i > 0 ? p_i * log2(p_i) : 0.0, p)
end

# Batch computation using multiple dispatch
function compute_all(features::Vector{<:AbstractFirstOrderFeature}, voxels)
    Dict(nameof(typeof(f)) => compute(f, voxels) for f in features)
end
```

### 5. Julia Image Processing Ecosystem

**Sources**:
- [JuliaImages](https://juliaimages.org/)
- [MedImages.jl](https://github.com/JuliaHealth/MedImages.jl)

#### 5.1 Core Packages

| Package | Purpose | PyRadiomics Equivalent |
|---------|---------|----------------------|
| **Images.jl** | Umbrella package for image processing | SimpleITK (processing) |
| **FileIO.jl** | Generic I/O for images | SimpleITK (loading) |
| **MedImages.jl** | Medical imaging I/O (NRRD, NIfTI) | SimpleITK (medical formats) |
| **ImageMorphology.jl** | Morphological operations | scipy.ndimage |
| **Meshing.jl** | Marching cubes, isosurfaces | trimesh, vtk |
| **MarchingCubes.jl** | Efficient marching cubes | skimage.measure.marching_cubes |

#### 5.2 Working with 3D Medical Images

```julia
using Images

# Julia treats images as regular arrays
img = rand(100, 100, 50)  # 3D "image"
mask = rand(Bool, 100, 100, 50)  # 3D binary mask

# Extract voxels within mask
voxels = img[mask]

# Axis-specific operations
for z in axes(img, 3)
    slice = @view img[:, :, z]
    # process slice...
end
```

#### 5.3 Mesh Generation for Shape Features

```julia
using Meshing
using GeometryBasics

# Generate mesh from binary mask
mask = rand(Bool, 50, 50, 50)
vertices, faces = isosurface(Float64.(mask), MarchingCubes(iso=0.5))

# Alternatively, using MarchingCubes.jl
using MarchingCubes
mc = MC(Float64.(mask), Int)
march(mc)
mesh = makemesh(GeometryBasics, mc)
```

### 6. Performance Optimization Opportunities

**Sources**:
- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Modern Julia Workflows - Optimizing](https://modernjuliaworkflows.org/optimizing/)

#### 6.1 Type Stability (Critical!)

**Key Principle**: The return type must depend only on the types of the arguments, not their values.

```julia
# BAD: Type-unstable (return type depends on value)
function bad_max(a, b)
    a > b ? a : nothing  # Returns Union{T, Nothing}
end

# GOOD: Type-stable
function good_max(a::T, b::T) where T
    a > b ? a : b  # Always returns T
end

# Check type stability with @code_warntype
@code_warntype compute(Energy(), rand(100))
```

#### 6.2 Avoid Global Variables

```julia
# BAD
const_value = 25  # Mutable global
function bad_discretize(x)
    floor.(x ./ const_value)  # Type unstable!
end

# GOOD
const CONST_VALUE = 25  # Immutable global (const)
function good_discretize(x)
    floor.(x ./ CONST_VALUE)  # Type stable
end

# BETTER: Pass as argument
function best_discretize(x, bin_width)
    floor.(x ./ bin_width)
end
```

#### 6.3 Pre-allocate Arrays

```julia
# BAD: Allocates on every iteration
function bad_compute(data, n)
    results = []
    for i in 1:n
        push!(results, sum(data[:, i]))
    end
    results
end

# GOOD: Pre-allocate
function good_compute(data, n)
    results = Vector{Float64}(undef, n)
    for i in 1:n
        results[i] = sum(@view data[:, i])
    end
    results
end
```

#### 6.4 Use Views for Slicing

```julia
# BAD: Creates copies
function bad_slice(img, mask)
    for z in 1:size(img, 3)
        slice = img[:, :, z]  # Copy!
        # ...
    end
end

# GOOD: Use views
function good_slice(img, mask)
    @views for z in 1:size(img, 3)
        slice = img[:, :, z]  # View, no copy
        # ...
    end
end
```

#### 6.5 Benchmark with BenchmarkTools

```julia
using BenchmarkTools

# Basic benchmark
@btime compute(Energy(), $voxels)

# More detailed benchmark
@benchmark compute(Energy(), $voxels)

# Compare implementations
suite = BenchmarkGroup()
suite["energy_v1"] = @benchmarkable compute_v1($voxels)
suite["energy_v2"] = @benchmarkable compute_v2($voxels)
results = run(suite)
```

### 7. Multiple Dispatch Patterns for Feature Classes

**Source**: [The Art of Multiple Dispatch](https://scientificcoder.com/the-art-of-multiple-dispatch)

#### 7.1 Pattern: Trait-Based Dispatch

```julia
# Define traits
abstract type MatrixRequirement end
struct RequiresGLCM <: MatrixRequirement end
struct RequiresGLRLM <: MatrixRequirement end
struct RequiresNoMatrix <: MatrixRequirement end

# Trait function
matrix_requirement(::Type{<:AbstractGLCMFeature}) = RequiresGLCM()
matrix_requirement(::Type{<:AbstractFirstOrderFeature}) = RequiresNoMatrix()

# Dispatch on trait
function extract(feature::F, img, mask) where F <: AbstractRadiomicsFeature
    _extract(matrix_requirement(F), feature, img, mask)
end

# Trait-specific implementations
function _extract(::RequiresGLCM, feature, img, mask)
    glcm = compute_glcm(img, mask)
    compute(feature, glcm)
end

function _extract(::RequiresNoMatrix, feature, img, mask)
    voxels = img[mask]
    compute(feature, voxels)
end
```

#### 7.2 Pattern: Feature Registration

```julia
# Register all features in a feature class
const FIRSTORDER_FEATURES = [
    Energy(), TotalEnergy(), Entropy(), Minimum(),
    Percentile10(), Percentile90(), Maximum(), Mean(),
    Median(), InterquartileRange(), Range(),
    MeanAbsoluteDeviation(), RobustMeanAbsoluteDeviation(),
    RootMeanSquared(), Skewness(), Kurtosis(), Variance(),
    Uniformity()
]

# Feature name mapping
feature_name(::Energy) = "Energy"
feature_name(::Entropy) = "Entropy"
# ... etc

# PyRadiomics-compatible name
pyradiomics_name(f::AbstractFirstOrderFeature) = "firstorder_$(feature_name(f))"
```

### 8. Julia Testing Best Practices

**Sources**:
- [Julia Test Documentation](https://docs.julialang.org/en/v1/stdlib/Test/)
- [Julia Testing Best Practices](https://erikexplores.substack.com/p/julia-testing-best-pratice)

#### 8.1 Test File Structure

```
test/
├── runtests.jl          # Main entry point
├── test_utils.jl        # Shared test utilities
├── test_firstorder.jl   # First-order feature tests
├── test_shape.jl        # Shape feature tests
├── test_glcm.jl         # GLCM tests
├── test_glrlm.jl        # GLRLM tests
├── test_glszm.jl        # GLSZM tests
├── test_ngtdm.jl        # NGTDM tests
├── test_gldm.jl         # GLDM tests
└── test_full_parity.jl  # Comprehensive parity tests
```

#### 8.2 Test Organization

```julia
# runtests.jl
using Test
using Radiomics

@testset "Radiomics.jl" begin
    include("test_utils.jl")

    @testset "First Order Features" begin
        include("test_firstorder.jl")
    end

    @testset "Shape Features" begin
        include("test_shape.jl")
    end

    # ... more testsets
end
```

#### 8.3 Testing Floating-Point Values

```julia
using Test

# Exact equality (rarely appropriate for floats)
@test result == expected

# Approximate equality (preferred for floats)
@test result ≈ expected
@test isapprox(result, expected, rtol=1e-10)
@test isapprox(result, expected, atol=1e-12)

# Custom tolerance per feature type
const FIRSTORDER_RTOL = 1e-10
const SHAPE_RTOL = 1e-6  # Mesh computations are less precise

@test result ≈ expected rtol=FIRSTORDER_RTOL
```

#### 8.4 Reproducible Random Tests

```julia
using Random

function random_image_mask(seed::Int, size::Tuple;
                           intensity_range=(0.0, 255.0),
                           mask_fraction=0.3)
    rng = MersenneTwister(seed)

    # Generate random image
    lo, hi = intensity_range
    image = lo .+ (hi - lo) .* rand(rng, size...)

    # Generate random binary mask
    mask = rand(rng, size...) .< mask_fraction

    return image, mask
end

@testset "Energy with multiple seeds" begin
    for seed in [42, 123, 456, 789]
        image, mask = random_image_mask(seed, (32, 32, 32))

        julia_result = Radiomics.energy(image[mask])
        python_result = pyradiomics_energy(image, mask)

        @test julia_result ≈ python_result rtol=1e-10
    end
end
```

### 9. Recommended Package Dependencies

#### 9.1 Core Dependencies (src)

```toml
[deps]
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"  # stdlib
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"  # stdlib
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
```

#### 9.2 Optional Dependencies (for full functionality)

```toml
[deps]
Meshing = "e6723b0e-dd44-4e87-a2a7-2c0ac69b0d80"      # Marching cubes
GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"  # Mesh types
ImageMorphology = "787d08f9-d448-5407-9aad-5290dd7ab264"  # Morphology ops
```

#### 9.3 Test-Only Dependencies (test/Project.toml)

```toml
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
CondaPkg = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
```

#### 9.4 test/CondaPkg.toml (for PyRadiomics)

```toml
[deps]
pyradiomics = ""
numpy = ""
SimpleITK = ""
```

### 10. Implementation Recommendations Summary

1. **Use Multiple Dispatch**: Replace Python inheritance with Julia abstract types and multiple dispatch
2. **Type Stability**: Ensure all functions are type-stable for performance
3. **Views Over Copies**: Use `@views` and `@view` for array slicing
4. **Broadcasting**: Use dot syntax for element-wise operations
5. **Pre-allocation**: Pre-allocate output arrays in performance-critical code
6. **Pure Julia**: All runtime dependencies must be pure Julia (no Python/C bindings)
7. **Test Parity**: Every feature must be tested against PyRadiomics output
8. **Fixed Seeds**: Use deterministic random number generation for reproducible tests
9. **Appropriate Tolerances**: Use looser tolerances for mesh/shape features than statistics

### 11. References

- **Julia Documentation**: https://docs.julialang.org/en/v1/
- **JuliaImages**: https://juliaimages.org/
- **StatsBase.jl**: https://juliastats.org/StatsBase.jl/stable/
- **MLJ.jl**: https://juliaai.github.io/MLJ.jl/stable/
- **Meshing.jl**: https://juliageometry.github.io/Meshing.jl/stable/
- **MarchingCubes.jl**: https://github.com/JuliaGeometry/MarchingCubes.jl
- **BenchmarkTools.jl**: https://github.com/JuliaCI/BenchmarkTools.jl
- **PythonCall.jl**: https://github.com/JuliaPy/PythonCall.jl
- **NaNStatistics.jl**: https://brenhinkeller.github.io/NaNStatistics.jl/dev/

**Note**: Git commit pending - git repository not yet initialized (will be done in SETUP-GIT-REPO story).

---


### Iteration 4 - 2026-01-17

**Story**: RESEARCH-DEPENDENCIES
**Status**: ✅ COMPLETED

---

## Pure Julia Dependencies Research Findings

This section documents all Julia packages needed for Radiomics.jl, mapping PyRadiomics dependencies to Julia equivalents, and verifying pure Julia implementations are available for all required functionality.

### 1. PyRadiomics Dependencies → Julia Equivalents

| PyRadiomics Dependency | Purpose | Julia Equivalent | Pure Julia? |
|----------------------|---------|-----------------|-------------|
| **numpy** | Array operations | Base Julia arrays | ✅ Yes (stdlib) |
| **SimpleITK** | Medical image I/O | Images.jl, MedImages.jl | ✅ Yes |
| **scipy.stats** | Statistical functions | Statistics.jl, StatsBase.jl | ✅ Yes |
| **scipy.ndimage** | Image operations | ImageMorphology.jl | ✅ Yes |
| **PyWavelets** | Wavelet filters | Wavelets.jl | ✅ Yes (optional) |
| **trimesh/vtk** | Mesh operations | Meshing.jl, Meshes.jl | ✅ Yes |

### 2. Statistics Packages Verification

#### 2.1 Statistics.jl (stdlib)

**Source**: [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/Statistics/)

Built-in functions:
- `mean(x)` - Arithmetic mean
- `std(x)` - Standard deviation (with Bessel correction by default)
- `var(x)` - Variance
- `median(x)` - Median
- `quantile(x, p)` - Quantile/percentile
- `cor(x, y)` - Correlation
- `cov(x, y)` - Covariance

**Covers**: Mean, Median, Variance, Standard Deviation, basic percentiles

#### 2.2 StatsBase.jl

**Source**: [StatsBase.jl Documentation](https://juliastats.org/StatsBase.jl/stable/)

Additional functions needed for Radiomics:
- `percentile(x, p)` - Equivalent to `np.percentile`
- `iqr(x)` - Interquartile range
- `entropy(p)` - Shannon entropy: `-Σ p_i × log(p_i)`
- `skewness(x)` - Third standardized moment
- `kurtosis(x)` - Fourth standardized moment (excess kurtosis)
- `fit(Histogram, x)` - Histogram computation
- `mode(x)` - Most frequent value

**Covers**: Entropy, Skewness, Kurtosis, IQR, Percentiles, Histograms, Uniformity

#### 2.3 NaNStatistics.jl

**Source**: [NaNStatistics.jl](https://github.com/brenhinkeller/NaNStatistics.jl)

NaN-ignoring statistics (matching NumPy's nan* functions):
- `nanmean(x)` - Mean ignoring NaNs
- `nanstd(x)` - Std dev ignoring NaNs
- `nanvar(x)` - Variance ignoring NaNs
- `nansum(x)` - Sum ignoring NaNs
- `nanminimum(x)`, `nanmaximum(x)` - Min/Max ignoring NaNs
- `nanmedian(x)` - Median ignoring NaNs

**Decision**: Include NaNStatistics.jl for robustness, but primary implementation should avoid NaN values by proper masking. Can be made optional.

### 3. Linear Algebra (for Shape Features)

#### 3.1 LinearAlgebra.jl (stdlib)

**Source**: [LinearAlgebra Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)

Required for shape features (PCA on voxel coordinates):
- `eigen(A)` - Full eigendecomposition, returns `Eigen` factorization
- `eigvals(A)` - Eigenvalues only (faster when eigenvectors not needed)
- `eigvecs(A)` - Eigenvectors only
- `svd(A)` - Singular value decomposition
- `cross(a, b)` - Cross product (for mesh normal calculations)
- `norm(x)` - Vector norm
- `det(A)` - Determinant

**Covers**: MajorAxisLength, MinorAxisLength, LeastAxisLength, Elongation, Flatness (all require eigenvalues of covariance matrix)

### 4. Image Morphology & Connected Components

#### 4.1 ImageMorphology.jl

**Source**: [ImageMorphology.jl](https://juliaimages.org/ImageMorphology.jl/dev/reference/)

Critical functions for GLSZM:
- `label_components(A)` - Connected component labeling (6-connectivity default in 3D)
- `label_components(A, connectivity)` - With custom connectivity
- `component_lengths(labeled)` - Get size of each component (zone sizes!)
- `component_boxes(labeled)` - Bounding boxes per component
- `component_indices(labeled)` - Voxel indices per component

**Connectivity options**:
- `dims=(1,2,3)` - 6-connectivity (face-connected)
- Full 26-connectivity requires custom offset specification

**Covers**: GLSZM zone detection, connected component analysis

**Verification**: `label_components` works on any N-dimensional array including 3D volumes.

### 5. Mesh Generation for Shape Features

#### 5.1 Meshing.jl (Primary Choice)

**Source**: [Meshing.jl API](https://juliageometry.github.io/Meshing.jl/dev/api/)

Isosurface extraction from binary masks:
```julia
using Meshing, GeometryBasics

# Extract mesh from binary mask
vertices, faces = isosurface(Float64.(mask), MarchingCubes(iso=0.5))
```

**Key Features**:
- `MarchingCubes` - Default algorithm, good balance of performance and quality
- `MarchingTetrahedra` - Alternative algorithm
- `NaiveSurfaceNets` - Smoother surfaces
- Works directly with 3D arrays
- Returns vertices and faces compatible with GeometryBasics.jl

#### 5.2 GeometryBasics.jl (Mesh Types)

**Source**: [GeometryBasics.jl](https://juliageometry.github.io/GeometryBasics.jl/stable/)

Provides mesh data types:
- `Point3f`, `Point3d` - 3D point types
- `TriangleFace` - Triangle face type
- `Mesh` - Mesh container
- `coordinates(mesh)` - Get vertices
- `faces(mesh)` - Get faces

**Note**: GeometryBasics is the modern replacement for deprecated GeometryTypes.

#### 5.3 Meshes.jl (Alternative/Supplementary)

**Source**: [Meshes.jl](https://juliageometry.github.io/MeshesDocs/stable/)

Higher-level computational geometry:
- `measure(geometry)` - Area/volume computation
- Supports various polytopes
- More mathematical rigor

**Decision**: Use Meshing.jl + GeometryBasics.jl for mesh generation. May use Meshes.jl for additional geometry operations if needed.

### 6. Custom Implementation Requirements

The following PyRadiomics C++ functionality needs pure Julia implementation:

#### 6.1 Texture Matrix Computation (formerly `_cmatrices.cpp`)

| Matrix | Algorithm | Julia Implementation |
|--------|-----------|---------------------|
| GLCM | Co-occurrence counting with offsets | Loops with `@inbounds`, `@simd` |
| GLRLM | Run-length detection per direction | Direction-wise scanning |
| GLSZM | Zone labeling (connected components) | `ImageMorphology.label_components` |
| NGTDM | Neighborhood averaging | Convolution or explicit loops |
| GLDM | Dependence counting | Neighborhood iteration |

**Performance Strategy**:
- Use Julia's loop optimizations (`@inbounds`, `@simd`)
- Pre-allocate matrices
- Use views to avoid copying
- Consider LoopVectorization.jl for critical paths

#### 6.2 Mesh Calculations (formerly `_cshape.cpp`)

| Calculation | PyRadiomics Approach | Julia Implementation |
|-------------|---------------------|---------------------|
| Mesh Volume | Signed tetrahedron volumes | Sum of `(a · (b × c)) / 6` per triangle |
| Surface Area | Sum of triangle areas | Sum of `0.5 × |cross(e1, e2)|` per triangle |
| Max 3D Diameter | Pairwise vertex distances | Efficient nested loops or convex hull |

**Implementation Plan**: Create `src/mesh_utils.jl` with pure Julia mesh calculations.

### 7. Final Dependency List

#### 7.1 Runtime Dependencies (src/Radiomics.jl)

```toml
[deps]
# Standard Library (no version needed)
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

# Statistical Functions
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

# Image Operations
ImageMorphology = "787d08f9-d448-5407-9aad-5290dd7ab264"

# Mesh Generation (for shape features)
Meshing = "e6723b0e-dd44-4e87-a2a7-2c0ac69b0d80"
GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
```

#### 7.2 Optional Runtime Dependencies

```toml
[deps]
# NaN-handling (optional, for robustness)
NaNStatistics = "b946abbf-3ea7-4571-aef4-f84173b9ea0c"

# Medical image I/O (optional, users can use any loader)
# Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
```

#### 7.3 Test Dependencies (test/Project.toml)

```toml
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
CondaPkg = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
```

#### 7.4 Python Test Environment (test/CondaPkg.toml)

```toml
[deps]
pyradiomics = ""
numpy = ""
SimpleITK = ""
```

### 8. Verification Checklist

| Requirement | Package | Verified |
|-------------|---------|----------|
| Statistics (mean, std, var, median) | Statistics.jl (stdlib) | ✅ |
| Percentiles (10th, 90th, IQR) | StatsBase.jl | ✅ |
| Entropy | StatsBase.jl | ✅ |
| Skewness, Kurtosis | StatsBase.jl | ✅ |
| Histogram binning | StatsBase.jl | ✅ |
| Eigenvalues (PCA for shape) | LinearAlgebra.jl (stdlib) | ✅ |
| Cross product, norms | LinearAlgebra.jl (stdlib) | ✅ |
| Connected components | ImageMorphology.jl | ✅ |
| Marching cubes (mesh) | Meshing.jl | ✅ |
| Mesh data types | GeometryBasics.jl | ✅ |
| PythonCall for tests | PythonCall.jl | ✅ |
| Conda management for tests | CondaPkg.jl | ✅ |

### 9. Gaps Requiring Custom Implementation

| Gap | Solution |
|-----|----------|
| GLCM matrix computation | Pure Julia loops with optimization |
| GLRLM matrix computation | Pure Julia direction scanning |
| NGTDM matrix computation | Pure Julia neighborhood iteration |
| GLDM matrix computation | Pure Julia dependence counting |
| Mesh volume calculation | Cross product formula |
| Mesh surface area | Triangle area summation |
| Maximum 3D diameter | Pairwise distance (optimize later) |

**None of these require external C/Python libraries** - all can be implemented in pure Julia.

### 10. Package Version Constraints (Compatibility)

Based on current package ecosystems (as of 2026):

```toml
[compat]
julia = "1.10"
StatsBase = "0.34"
ImageMorphology = "0.4"
Meshing = "0.6"
GeometryBasics = "0.4"
```

### 11. References

- **StatsBase.jl**: https://juliastats.org/StatsBase.jl/stable/
- **NaNStatistics.jl**: https://brenhinkeller.github.io/NaNStatistics.jl/dev/
- **ImageMorphology.jl**: https://juliaimages.org/ImageMorphology.jl/dev/
- **Meshing.jl**: https://juliageometry.github.io/Meshing.jl/dev/
- **GeometryBasics.jl**: https://juliageometry.github.io/GeometryBasics.jl/stable/
- **Meshes.jl**: https://juliageometry.github.io/MeshesDocs/stable/
- **PythonCall.jl**: https://juliapy.github.io/PythonCall.jl/stable/
- **CondaPkg.jl**: https://github.com/JuliaPy/CondaPkg.jl

**Note**: Git commit pending - git repository not yet initialized (will be done in SETUP-GIT-REPO story).

---


### Iteration 5 - 2026-01-17

**Story**: RESEARCH-TEST-STRATEGY
**Status**: ✅ COMPLETED

---

## Test Strategy Research Findings

This section documents the comprehensive testing strategy for Radiomics.jl, using PythonCall.jl to verify 1:1 parity with PyRadiomics.

### 1. PythonCall.jl Setup for Test Environment

**Source**: [PythonCall.jl Documentation](https://juliapy.github.io/PythonCall.jl/stable/)

#### 1.1 Core Concepts

- PythonCall provides a symmetric interface between Julia and Python
- Uses `pyimport()` to import Python modules
- `Py` type wraps all Python objects
- `pyconvert()` converts Python objects to Julia types
- `PyArray` provides efficient numpy array interoperability (no copy when possible)

#### 1.2 Key Functions for Test Harness

```julia
using PythonCall

# Import Python modules
np = pyimport("numpy")
sitk = pyimport("SimpleITK")
radiomics = pyimport("radiomics")
featureextractor = pyimport("radiomics.featureextractor")

# Create SimpleITK images from numpy arrays
image_np = np.array(julia_array)
image_sitk = sitk.GetImageFromArray(image_np)
image_sitk.SetSpacing((1.0, 1.0, 1.0))  # Required for mask validation

# Call PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor()
result = extractor.execute(image_sitk, mask_sitk)

# Convert result to Julia
feature_value = pyconvert(Float64, result["original_firstorder_Energy"])
```

#### 1.3 Array Conversion Strategy

PyRadiomics requires SimpleITK image objects, not raw numpy arrays. The conversion path is:

```
Julia Array → numpy.ndarray → SimpleITK.Image → PyRadiomics
```

**Critical Notes**:
- SimpleITK uses (z, y, x) axis order (Fortran-style), while Julia uses (x, y, z)
- Must set matching spacing, origin, and direction on both image and mask
- PyRadiomics validates geometry between image and mask

```julia
function julia_to_sitk(arr::AbstractArray{T, N}, spacing=(1.0, 1.0, 1.0)) where {T, N}
    np = pyimport("numpy")
    sitk = pyimport("SimpleITK")

    # Julia is column-major, numpy is row-major
    # We need to reverse dimensions for proper alignment
    np_arr = np.array(permutedims(arr, reverse(1:N)))
    img = sitk.GetImageFromArray(np_arr.astype("float64"))
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    return img
end
```

### 2. CondaPkg.jl Configuration for PyRadiomics

**Source**: [CondaPkg.jl Repository](https://github.com/JuliaPy/CondaPkg.jl)

#### 2.1 Test Environment Setup

Create `test/CondaPkg.toml`:

```toml
channels = ["conda-forge"]

[deps]
python = ">=3.10"
numpy = ""
SimpleITK = ""

[pip.deps]
pyradiomics = ""
```

**Notes**:
- `pyradiomics` is installed via pip (not available on conda-forge)
- SimpleITK and numpy are available on conda-forge
- PythonCall automatically manages this environment

#### 2.2 Test Project.toml

Create `test/Project.toml`:

```toml
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
CondaPkg = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

### 3. Random Array Generation Strategy

**Source**: [Julia Random Documentation](https://docs.julialang.org/en/v1/stdlib/Random/)

#### 3.1 Design Principles

1. **Fixed seeds for reproducibility**: Every test uses explicit integer seeds
2. **Multiple seeds per test**: Test with seeds [42, 123, 456, 789] for robustness
3. **Varied array sizes**: Test small (16³), medium (32³), and large (64³) volumes
4. **Realistic intensity ranges**: Use [0, 255] or [-1000, 3000] (CT HU) ranges

#### 3.2 Seed Selection Strategy

| Seed | Purpose |
|------|---------|
| 42 | Primary test seed (classic) |
| 123 | Secondary seed for variation |
| 456 | Edge case testing |
| 789 | Large volume testing |
| 1234 | Stress testing |

#### 3.3 Standard Random Image/Mask Generator

```julia
using Random

"""
    random_image_mask(seed::Int, size::Tuple;
                      intensity_range=(0.0, 255.0),
                      mask_fraction=0.3,
                      ensure_contiguous=true)

Generate a random 3D image and binary mask for testing.

# Arguments
- `seed::Int`: Random seed for reproducibility
- `size::Tuple`: Dimensions of the array (x, y, z)
- `intensity_range`: (min, max) intensity values
- `mask_fraction`: Fraction of voxels to include in mask (0.0-1.0)
- `ensure_contiguous`: If true, generate a roughly spherical contiguous mask

# Returns
- `image::Array{Float64, 3}`: Random intensity image
- `mask::Array{Bool, 3}`: Binary mask
"""
function random_image_mask(seed::Int, size::Tuple{Int, Int, Int};
                           intensity_range::Tuple{Float64, Float64}=(0.0, 255.0),
                           mask_fraction::Float64=0.3,
                           ensure_contiguous::Bool=true)
    rng = MersenneTwister(seed)

    # Generate random image
    lo, hi = intensity_range
    image = lo .+ (hi - lo) .* rand(rng, Float64, size...)

    if ensure_contiguous
        # Generate roughly spherical mask for realistic ROI
        center = size .÷ 2
        radius = minimum(size) * sqrt(mask_fraction) / 2
        mask = [sqrt(sum(((i, j, k) .- center) .^ 2)) <= radius
                for i in 1:size[1], j in 1:size[2], k in 1:size[3]]
    else
        # Random scattered mask
        mask = rand(rng, Float64, size...) .< mask_fraction
    end

    return image, mask
end
```

#### 3.4 Stability Note

Julia's `MersenneTwister` output may change between Julia versions. For cross-version reproducibility, consider using `StableRNGs.jl`. However, for parity testing against PyRadiomics, this is not critical since we compare Julia vs Python results in the same run.

**Decision**: Use standard `MersenneTwister` for simplicity. If cross-version reproducibility becomes important, switch to `StableRNGs.jl`.

### 4. Tolerance Thresholds for Floating-Point Comparisons

#### 4.1 General Guidelines

- Use **relative tolerance (rtol)** as primary comparison method
- Add **absolute tolerance (atol)** for values near zero
- Different feature types may need different tolerances

#### 4.2 Tolerance Matrix

| Feature Class | rtol | atol | Rationale |
|--------------|------|------|-----------|
| First Order | 1e-10 | 1e-12 | Pure arithmetic, should match exactly |
| Shape | 1e-6 | 1e-8 | Mesh algorithms have inherent imprecision |
| GLCM | 1e-10 | 1e-12 | Matrix computation is deterministic |
| GLRLM | 1e-10 | 1e-12 | Matrix computation is deterministic |
| GLSZM | 1e-10 | 1e-12 | Connected component counting is exact |
| NGTDM | 1e-10 | 1e-12 | Neighborhood computation is deterministic |
| GLDM | 1e-10 | 1e-12 | Dependence counting is exact |

#### 4.3 Testing Function

```julia
const FEATURE_TOLERANCES = Dict(
    :firstorder => (rtol=1e-10, atol=1e-12),
    :shape => (rtol=1e-6, atol=1e-8),
    :glcm => (rtol=1e-10, atol=1e-12),
    :glrlm => (rtol=1e-10, atol=1e-12),
    :glszm => (rtol=1e-10, atol=1e-12),
    :ngtdm => (rtol=1e-10, atol=1e-12),
    :gldm => (rtol=1e-10, atol=1e-12),
)

function compare_feature(julia_val, python_val, feature_class::Symbol;
                        custom_rtol=nothing, custom_atol=nothing)
    tol = FEATURE_TOLERANCES[feature_class]
    rtol = isnothing(custom_rtol) ? tol.rtol : custom_rtol
    atol = isnothing(custom_atol) ? tol.atol : custom_atol
    return isapprox(julia_val, python_val; rtol=rtol, atol=atol)
end
```

#### 4.4 Known Edge Cases

1. **Kurtosis**: PyRadiomics returns excess kurtosis (μ₄/σ⁴ - 3), not regular kurtosis
2. **Entropy**: Uses ε ≈ 2.2×10⁻¹⁶ to prevent log(0)
3. **Division by zero**: Various features return 0 or special values when denominators are zero
4. **Empty masks**: May cause NaN or Inf values

### 5. Test File Organization

#### 5.1 Directory Structure

```
test/
├── Project.toml           # Test-only Julia dependencies
├── CondaPkg.toml          # Python dependencies (pyradiomics)
├── runtests.jl            # Main test runner
├── test_utils.jl          # PythonCall harness, random generators
├── test_core.jl           # Core infrastructure tests
├── test_discretization.jl # Binning/discretization tests
├── test_firstorder.jl     # First-order feature parity tests
├── test_shape.jl          # Shape feature parity tests
├── test_glcm.jl           # GLCM feature parity tests
├── test_glrlm.jl          # GLRLM feature parity tests
├── test_glszm.jl          # GLSZM feature parity tests
├── test_ngtdm.jl          # NGTDM feature parity tests
├── test_gldm.jl           # GLDM feature parity tests
├── test_integration.jl    # Full pipeline tests
└── test_full_parity.jl    # Comprehensive parity suite
```

#### 5.2 Test Runner Structure (runtests.jl)

```julia
using Test
using Radiomics

@testset "Radiomics.jl" begin
    include("test_utils.jl")

    @testset "Core Infrastructure" begin
        include("test_core.jl")
    end

    @testset "Discretization" begin
        include("test_discretization.jl")
    end

    @testset "First Order Features" begin
        include("test_firstorder.jl")
    end

    @testset "Shape Features" begin
        include("test_shape.jl")
    end

    @testset "GLCM Features" begin
        include("test_glcm.jl")
    end

    @testset "GLRLM Features" begin
        include("test_glrlm.jl")
    end

    @testset "GLSZM Features" begin
        include("test_glszm.jl")
    end

    @testset "NGTDM Features" begin
        include("test_ngtdm.jl")
    end

    @testset "GLDM Features" begin
        include("test_gldm.jl")
    end

    @testset "Integration" begin
        include("test_integration.jl")
    end

    @testset "Full Parity" begin
        include("test_full_parity.jl")
    end
end
```

### 6. Template Test Structure for Feature Parity Tests

#### 6.1 test_utils.jl - PythonCall Harness

```julia
using PythonCall
using Random

# Initialize Python modules (lazy loading)
const _np = Ref{Py}()
const _sitk = Ref{Py}()
const _radiomics = Ref{Py}()
const _featureextractor = Ref{Py}()

function ensure_python_initialized()
    if !isassigned(_np)
        _np[] = pyimport("numpy")
        _sitk[] = pyimport("SimpleITK")
        _radiomics[] = pyimport("radiomics")
        _featureextractor[] = pyimport("radiomics.featureextractor")
    end
end

np() = (ensure_python_initialized(); _np[])
sitk() = (ensure_python_initialized(); _sitk[])
featureextractor() = (ensure_python_initialized(); _featureextractor[])

"""
Convert Julia array to SimpleITK image for PyRadiomics.
"""
function julia_to_sitk(arr::AbstractArray{T, 3};
                       spacing::Tuple=(1.0, 1.0, 1.0)) where T
    ensure_python_initialized()

    # Convert to Float64 and permute to (z, y, x) for SimpleITK
    arr_f64 = Float64.(arr)
    arr_permuted = permutedims(arr_f64, (3, 2, 1))

    # Create numpy array and SimpleITK image
    np_arr = np().array(arr_permuted)
    img = sitk().GetImageFromArray(np_arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))

    return img
end

"""
Convert Julia Bool mask to SimpleITK label image.
"""
function mask_to_sitk(mask::AbstractArray{Bool, 3};
                      spacing::Tuple=(1.0, 1.0, 1.0))
    ensure_python_initialized()

    # Convert Bool to UInt8 (label 1 for ROI)
    mask_uint8 = UInt8.(mask)
    mask_permuted = permutedims(mask_uint8, (3, 2, 1))

    np_arr = np().array(mask_permuted).astype("uint8")
    img = sitk().GetImageFromArray(np_arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))

    return img
end

"""
Extract a single feature from PyRadiomics.

# Arguments
- `feature_class`: One of "firstorder", "shape", "glcm", "glrlm", "glszm", "ngtdm", "gldm"
- `feature_name`: The feature name (e.g., "Energy", "Entropy")
- `image`: Julia 3D array
- `mask`: Julia 3D Bool array
- `settings`: Dict of PyRadiomics settings (optional)

# Returns
- Float64 feature value
"""
function pyradiomics_feature(feature_class::String, feature_name::String,
                             image::AbstractArray{<:Real, 3},
                             mask::AbstractArray{Bool, 3};
                             settings::Dict=Dict())
    ensure_python_initialized()

    # Convert to SimpleITK
    image_sitk = julia_to_sitk(image)
    mask_sitk = mask_to_sitk(mask)

    # Create extractor with settings
    extractor = featureextractor().RadiomicsFeatureExtractor()

    # Disable all features first
    extractor.disableAllFeatures()

    # Enable only the target feature class and feature
    extractor.enableFeatureClassByName(feature_class)

    # Apply custom settings
    for (k, v) in settings
        extractor.settings[k] = v
    end

    # Execute extraction
    result = extractor.execute(image_sitk, mask_sitk)

    # Construct feature key
    key = "original_$(feature_class)_$(feature_name)"

    # Convert and return
    return pyconvert(Float64, result[key])
end

"""
Extract all features from a feature class.
"""
function pyradiomics_all_features(feature_class::String,
                                  image::AbstractArray{<:Real, 3},
                                  mask::AbstractArray{Bool, 3};
                                  settings::Dict=Dict())
    ensure_python_initialized()

    image_sitk = julia_to_sitk(image)
    mask_sitk = mask_to_sitk(mask)

    extractor = featureextractor().RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(feature_class)

    for (k, v) in settings
        extractor.settings[k] = v
    end

    result = extractor.execute(image_sitk, mask_sitk)

    # Extract all features with matching prefix
    prefix = "original_$(feature_class)_"
    features = Dict{String, Float64}()

    for key in result.keys()
        key_str = pyconvert(String, key)
        if startswith(key_str, prefix)
            feature_name = replace(key_str, prefix => "")
            features[feature_name] = pyconvert(Float64, result[key])
        end
    end

    return features
end

# Random image/mask generator (as documented above)
function random_image_mask(seed::Int, size::Tuple{Int, Int, Int};
                           intensity_range::Tuple{Float64, Float64}=(0.0, 255.0),
                           mask_fraction::Float64=0.3,
                           ensure_contiguous::Bool=true)
    rng = MersenneTwister(seed)
    lo, hi = intensity_range
    image = lo .+ (hi - lo) .* rand(rng, Float64, size...)

    if ensure_contiguous
        center = size .÷ 2
        radius = minimum(size) * sqrt(mask_fraction) / 2
        mask = [sqrt(sum(((i, j, k) .- center) .^ 2)) <= radius
                for i in 1:size[1], j in 1:size[2], k in 1:size[3]]
    else
        mask = rand(rng, Float64, size...) .< mask_fraction
    end

    return image, mask
end

# Standard test seeds and sizes
const TEST_SEEDS = [42, 123, 456, 789]
const TEST_SIZES = [(16, 16, 16), (32, 32, 32), (24, 32, 28)]
```

#### 6.2 Template for Feature Class Tests (e.g., test_firstorder.jl)

```julia
using Test
using Radiomics

# Test harness is included from runtests.jl
# include("test_utils.jl")

const FIRSTORDER_FEATURES = [
    "Energy", "TotalEnergy", "Entropy", "Minimum",
    "10Percentile", "90Percentile", "Maximum", "Mean",
    "Median", "InterquartileRange", "Range",
    "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation",
    "RootMeanSquared", "Skewness", "Kurtosis", "Variance",
    "Uniformity"
]

const FIRSTORDER_RTOL = 1e-10
const FIRSTORDER_ATOL = 1e-12

@testset "First Order Features" begin

    @testset "Energy" begin
        for seed in TEST_SEEDS
            for size in TEST_SIZES
                image, mask = random_image_mask(seed, size)

                julia_result = Radiomics.energy(image, mask)
                python_result = pyradiomics_feature("firstorder", "Energy", image, mask)

                @test isapprox(julia_result, python_result;
                              rtol=FIRSTORDER_RTOL, atol=FIRSTORDER_ATOL)
            end
        end
    end

    @testset "Entropy" begin
        for seed in TEST_SEEDS
            image, mask = random_image_mask(seed, (32, 32, 32))

            julia_result = Radiomics.entropy(image, mask)
            python_result = pyradiomics_feature("firstorder", "Entropy", image, mask)

            @test isapprox(julia_result, python_result;
                          rtol=FIRSTORDER_RTOL, atol=FIRSTORDER_ATOL)
        end
    end

    # ... similar blocks for each feature

    @testset "All First Order Features (Batch)" begin
        # Test all features together for comprehensive coverage
        image, mask = random_image_mask(42, (32, 32, 32))

        julia_results = Radiomics.extract_firstorder(image, mask)
        python_results = pyradiomics_all_features("firstorder", image, mask)

        for feature_name in FIRSTORDER_FEATURES
            @testset "$feature_name" begin
                julia_val = julia_results[feature_name]
                python_val = python_results[feature_name]
                @test isapprox(julia_val, python_val;
                              rtol=FIRSTORDER_RTOL, atol=FIRSTORDER_ATOL)
            end
        end
    end
end
```

### 7. CI Strategy for Python+Julia Tests

#### 7.1 GitHub Actions Workflow (.github/workflows/CI.yml)

```yaml
name: CI

on:
  push:
    branches:
      - main
    tags: ['*']
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}

jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.10'
          - '1.11'
        os:
          - ubuntu-latest
          - macos-latest
        arch:
          - x64
    steps:
      - uses: actions/checkout@v4

      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}

      - uses: julia-actions/cache@v2

      - uses: julia-actions/julia-buildpkg@v1

      # CondaPkg will automatically install Python dependencies
      # on first use of PythonCall

      - uses: julia-actions/julia-runtest@v1
        env:
          # Use conda-forge for dependencies
          JULIA_CONDAPKG_BACKEND: MicroMamba
        timeout-minutes: 60

      - uses: julia-actions/julia-processcoverage@v1

      - uses: codecov/codecov-action@v4
        with:
          files: lcov.info

  docs:
    name: Documentation
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'

      - uses: julia-actions/julia-buildpkg@v1

      - uses: julia-actions/julia-docdeploy@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### 7.2 CI Configuration Notes

1. **MicroMamba Backend**: Use `JULIA_CONDAPKG_BACKEND: MicroMamba` for faster conda environment setup in CI
2. **Timeout**: Set 60-minute timeout to allow for full Python dependency installation
3. **Caching**: Julia actions handle caching of Julia packages; CondaPkg caches conda environments
4. **Matrix Testing**: Test on Julia 1.10 and 1.11, Ubuntu and macOS

#### 7.3 Windows Considerations

Windows CI with Python dependencies can be problematic. Options:
1. **Skip Windows initially**: Focus on Linux/macOS where pyradiomics is well-tested
2. **Add Windows later**: Once Linux/macOS tests are stable
3. **Conda limitations**: Some packages may not work on Windows

**Recommendation**: Start with Linux and macOS only, add Windows in a later phase.

### 8. Summary and Recommendations

#### 8.1 Test Harness Design Summary

1. **PythonCall.jl** provides seamless Julia-Python interop
2. **CondaPkg.jl** manages pyradiomics and dependencies automatically
3. **SimpleITK conversion** handles array dimension ordering
4. **Wrapper functions** (`pyradiomics_feature`, `pyradiomics_all_features`) provide clean API
5. **Fixed seeds** ensure reproducibility within a test run

#### 8.2 Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RNG | MersenneTwister (stdlib) | Simple, fast, cross-version issues acceptable for parity tests |
| Default tolerance | rtol=1e-10 | Strict enough to catch bugs, loose enough for float variance |
| Shape tolerance | rtol=1e-6 | Mesh algorithms have inherent imprecision |
| Test organization | One file per feature class | Clear separation, easy to maintain |
| CI platforms | Linux + macOS | Windows pyradiomics support is limited |
| Conda backend | MicroMamba | Faster installation in CI |

#### 8.3 Implementation Checklist for SETUP-TEST-HARNESS Story

- [ ] Create test/Project.toml with dependencies
- [ ] Create test/CondaPkg.toml with pyradiomics
- [ ] Implement test/test_utils.jl with harness functions
- [ ] Verify pyradiomics imports successfully
- [ ] Create simple example parity test
- [ ] Test locally before CI setup

### 9. References

- **PythonCall.jl Documentation**: https://juliapy.github.io/PythonCall.jl/stable/
- **CondaPkg.jl Repository**: https://github.com/JuliaPy/CondaPkg.jl
- **PyRadiomics Documentation**: https://pyradiomics.readthedocs.io/en/latest/
- **PyRadiomics API Reference**: https://pyradiomics.readthedocs.io/en/latest/radiomics.html
- **Julia Random Module**: https://docs.julialang.org/en/v1/stdlib/Random/
- **Julia Test Module**: https://docs.julialang.org/en/v1/stdlib/Test/
- **StableRNGs.jl**: https://github.com/JuliaRandom/StableRNGs.jl
- **GitHub Actions for Julia**: https://github.com/julia-actions

---


### Iteration 6 - 2026-01-17 12:27:07

**Agent started** (Open: 43, Done: 5)


**Story**: SETUP-PACKAGE
**Status**: COMPLETED

### Accomplishments

Created the complete Julia package structure for Radiomics.jl:

1. **src/Radiomics.jl**: Main module file with package docstring, version constant, and placeholder includes for future feature modules

2. **Project.toml**: Package metadata with UUID `b38db8b5-554a-48bd-b1c6-be05a0ff3d6d`, dependencies on Statistics and LinearAlgebra, Julia 1.10 compat

3. **test/runtests.jl**: Basic test file that verifies package loads correctly

4. **test/Project.toml**: Test dependencies including PythonCall and CondaPkg for PyRadiomics parity testing

5. **test/CondaPkg.toml**: Conda dependencies for pyradiomics, numpy, and simpleitk

6. **LICENSE**: MIT License

7. **.gitignore**: Comprehensive gitignore for Julia projects, CondaPkg, IDE files, etc.

8. **README.md**: Already existed with appropriate content

### Verification

- Package loads successfully: `using Radiomics` works
- Tests pass: `Pkg.test("Radiomics")` passes with 1 test
- VERSION = 0.1.0 as expected

### Files Created

```
Radiomics.jl/
├── Project.toml          # Package metadata
├── src/
│   └── Radiomics.jl      # Main module
├── test/
│   ├── Project.toml      # Test dependencies
│   ├── CondaPkg.toml     # Python dependencies
│   └── runtests.jl       # Test runner
├── LICENSE               # MIT License
├── README.md             # (already existed)
└── .gitignore           # Julia gitignore
```

---


### Iteration 7 - 2026-01-17 12:31:19

**Agent started** (Open: 42, Done: 6)


### Iteration 8 - 2026-01-17

**Story**: SETUP-GIT-REPO
**Status**: ✅ COMPLETED

### Accomplishments

Initialized git repository for Radiomics.jl with proper configuration:

1. **Git initialization**: `git init` created repository on main branch

2. **Initial commit**: Created comprehensive initial commit including:
   - src/Radiomics.jl (main module)
   - Project.toml (package metadata)
   - test/ directory (test infrastructure)
   - LICENSE (MIT)
   - README.md
   - ralph_loop/ (agent orchestration)
   - .gitignore

3. **.gitignore verification**: Properly configured for:
   - Julia artifacts (*.jl.cov, *.jl.mem, Manifest.toml, docs/build/)
   - CondaPkg (Python environment: .CondaPkg/)
   - Build artifacts (deps/)
   - IDE files (.idea/, .vscode/, *.swp)
   - OS files (.DS_Store, Thumbs.db)
   - Test coverage (lcov.info)
   - Jupyter (.ipynb_checkpoints/)

4. **Commit message format**: Using conventional format `STORY-ID: Brief description`

5. **Clean working tree**: `git status` shows "nothing to commit, working tree clean"

### Verification

- Git repo initialized: ✓
- Initial commit created: e0ae70d
- Manifest.toml excluded: ✓ (not in commit)
- Working tree clean: ✓

---


### Iteration 8 - 2026-01-17 12:33:00

**Agent started** (Open: 41, Done: 7)



### Iteration 9 - 2026-01-17

**Story**: SETUP-CI
**Status**: ✅ COMPLETED

### Accomplishments

Created GitHub Actions CI workflow for Radiomics.jl:

1. **Created .github/workflows/CI.yml** with:
   - Julia version matrix: 1.10 and 1.11
   - OS matrix: ubuntu-latest and macos-latest
   - Python 3.11 setup for CondaPkg/PythonCall test dependencies
   - Concurrency control to cancel redundant PR builds
   - Coverage processing and Codecov upload on Julia 1.11/Ubuntu
   - Documentation job (set to continue-on-error until docs are fully set up)

2. **CI Workflow Features**:
   - Triggers on push to main, tags, and pull requests
   - Uses julia-actions for setup, caching, build, and test
   - Sets up Python 3.11 for pyradiomics dependencies via CondaPkg
   - JULIA_NUM_THREADS=2 for parallel test execution
   - Codecov integration for coverage reporting

3. **Added CI badge to README.md**:
   - Badge links to GitHub Actions workflow status

### Files Created/Modified

```
.github/
└── workflows/
    └── CI.yml          # GitHub Actions CI workflow

README.md               # Added CI badge
ralph_loop/prd.json     # Updated SETUP-CI status to done
```

### Verification

- CI workflow file created at .github/workflows/CI.yml
- README.md includes CI badge
- prd.json SETUP-CI status updated to "done"

---


### Iteration 9 - 2026-01-17 12:34:39

**Agent started** (Open: 40, Done: 8)


---

## Iteration 4: SETUP-TEST-HARNESS

**Date**: 2026-01-17
**Story**: SETUP-TEST-HARNESS - Implement PythonCall Test Harness
**Status**: COMPLETED

### Summary

Implemented a comprehensive test harness for PyRadiomics parity testing using PythonCall.jl. The harness allows Julia implementations to be verified against the reference PyRadiomics Python library.

### Completed Tasks

1. **Created test/test_utils.jl** with the following utilities:
   - `random_image_mask(seed, size)` - Generates deterministic random test images and masks
   - `random_image_mask_integer(seed, size)` - Integer version for discretization testing
   - `julia_array_to_sitk(image, mask)` - Converts Julia arrays to SimpleITK format
   - `pyradiomics_feature(class, name, image, mask)` - Extracts a single feature from PyRadiomics
   - `pyradiomics_extract(class, image, mask)` - Extracts all features from a feature class
   - `pyradiomics_extract_all(image, mask)` - Extracts all features from all classes
   - `compare_features(julia_val, python_val)` - Compares values with tolerance
   - `compare_feature_dicts(julia_dict, python_dict)` - Batch comparison
   - `verify_pyradiomics_available()` - Checks if PyRadiomics is installed
   - `get_tolerance(feature_class)` - Returns appropriate tolerance for each feature class

2. **Configured CondaPkg dependencies programmatically**:
   - Dependencies are added before PythonCall is loaded
   - pyradiomics 3.0.1 is installed via pip with --no-build-isolation
   - Python version constrained to 3.10-3.11 for compatibility
   - All dependencies (numpy, simpleitk, pywavelets, etc.) properly configured

3. **Updated test/runtests.jl** with test harness verification:
   - PyRadiomics availability check
   - Random data generation tests (reproducibility, dimensions, types)
   - PyRadiomics feature extraction test

4. **All 15 tests pass** via `Pkg.test()`:
   - Package loads correctly
   - Test harness works correctly
   - PyRadiomics integration verified

### Key Technical Decisions

1. **Dependencies installed programmatically**: Rather than relying on CondaPkg.toml in the test directory (which isn't picked up by Pkg.test() temp environments), dependencies are added via `CondaPkg.add()` before PythonCall is loaded.

2. **pyradiomics via pip**: pyradiomics is not available on conda-forge, so it's installed via pip after the conda environment is resolved. The `--no-build-isolation` flag ensures numpy is available during the build.

3. **Python version constraint**: Python 3.10-3.11 required for pyradiomics compatibility.

4. **Numpy type handling**: PyRadiomics returns numpy scalar types that need special handling (`pyfloat()`) for conversion to Julia Float64.

5. **Class name mapping**: PyRadiomics uses CamelCase class names (e.g., "RadiomicsFirstOrder") which differ from the lowercase module names.

### Files Modified

- `test/test_utils.jl` - NEW: Comprehensive test utilities (600+ lines)
- `test/runtests.jl` - Updated with test harness verification
- `test/Project.toml` - Removed explicit Radiomics dependency (handled by Pkg.test())
- `test/CondaPkg.toml` - Base conda dependencies (also added programmatically)
- `~/.julia/environments/v1.11/CondaPkg.toml` - Global environment configured for development

### References

- PythonCall.jl: https://github.com/JuliaPy/PythonCall.jl
- CondaPkg.jl: https://github.com/JuliaPy/CondaPkg.jl
- PyRadiomics: https://github.com/AIM-Harvard/pyradiomics


### Iteration 10 - 2026-01-17 12:57:17

**Agent started** (Open: 39, Done: 9)


---

## Iteration 11: IMPL-CORE-TYPES

**Date**: 2026-01-17
**Story**: IMPL-CORE-TYPES - Implement Core Types and Structs
**Status**: COMPLETED

### Summary

Implemented all core types for Radiomics.jl, establishing the foundation for the feature extraction system.

### Types Implemented

#### 1. Abstract Type Hierarchy

```
AbstractRadiomicsFeature
├── AbstractFirstOrderFeature
├── AbstractShapeFeature
└── AbstractTextureFeature
    ├── AbstractGLCMFeature
    ├── AbstractGLRLMFeature
    ├── AbstractGLSZMFeature
    ├── AbstractNGTDMFeature
    └── AbstractGLDMFeature
```

#### 2. Settings Struct

`Settings` with @kwdef for keyword construction:
- **Discretization**: `binwidth`, `bincount`, `discretization_mode`
- **Mask/Label**: `label` (default: 1)
- **Preprocessing**: `resample_spacing`, `normalize`, `normalize_scale`, `remove_outliers`, `outlier_percentile`
- **2D/3D handling**: `force_2d`, `force_2d_dimension`
- **Texture parameters**: `glcm_distance`, `symmetrical_glcm`, `gldm_alpha`, `ngtdm_distance`
- **Performance**: `preallocate`
- **PyRadiomics compatibility**: `voxel_array_shift`

`DiscretizationMode` enum: `FixedBinWidth`, `FixedBinCount`

#### 3. RadiomicsImage{T, N}

Wrapper for medical images with:
- `data::Array{T, N}`: Image intensity data
- `spacing::NTuple{N, Float64}`: Voxel spacing in mm
- Validation: only 2D/3D, positive spacing
- AbstractArray interface (size, getindex, etc.)
- Default unit spacing constructor

#### 4. RadiomicsMask{N}

Wrapper for segmentation masks with:
- `data::Array{Int, N}`: Mask labels
- `label::Int`: ROI label value (default: 1)
- Constructors from Int, Bool, and BitArray
- AbstractArray interface
- `get_roi_mask()` function for boolean extraction

#### 5. FeatureResult

Single feature result container:
- `name::String`: Feature name
- `value::Float64`: Computed value
- `feature_class::String`: Feature class
- `image_type::String`: Image filter type

#### 6. FeatureSet

Collection of FeatureResult with:
- Vector-based storage
- Settings reference
- Dict-like access by key (e.g., "firstorder_Energy")
- Conversion to Dict
- Pretty printing (shows first 10 features)

#### 7. Type Aliases

- `ImageLike{T, N}`: Union of Array and RadiomicsImage
- `MaskLike{N}`: Union of Bool/Int arrays and RadiomicsMask

### Utility Functions

- `validate_settings(s::Settings)`: Validates all settings parameters
- `get_data(img)`: Extract raw array from RadiomicsImage or pass through
- `get_spacing(img)`: Get spacing (default unit spacing for plain arrays)
- `get_roi_mask(mask)`: Get boolean ROI from any mask type
- `get_mask_data(mask)`: Extract raw array from RadiomicsMask
- `feature_key(r::FeatureResult)`: Generate dictionary key

### Files Created/Modified

- `src/types.jl` - NEW: All core types (~400 lines)
- `src/Radiomics.jl` - Updated: includes types.jl, exports all types
- `ralph_loop/prd.json` - Updated: IMPL-CORE-TYPES status to "done"

### Verification

All types tested successfully:
- Settings creation with defaults and custom values
- Settings validation catches invalid parameters
- RadiomicsImage creation with/without spacing
- RadiomicsMask from Int, Bool, and BitArray
- get_roi_mask extracts correct boolean mask
- FeatureResult creation and formatting
- FeatureSet with Dict-like access

