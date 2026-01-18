# Feature Classes

Radiomics.jl implements 7 feature classes with over 111 individual features. This page documents each class with mathematical formulas and descriptions.

## First-Order Features (19 features)

First-order features are computed directly from voxel intensity values within the ROI, without considering spatial relationships between voxels.

### Feature List

| Feature | IBSI ID | Formula | Description |
|---------|---------|---------|-------------|
| **Energy** | N8CA | ``E = \sum_{i=1}^{N_p} X(i)^2`` | Sum of squared voxel values |
| **TotalEnergy** | - | ``TE = V_{voxel} \times E`` | Energy weighted by voxel volume |
| **Entropy** | TLU2 | ``H = -\sum_{i=1}^{N_g} p(i) \log_2(p(i))`` | Randomness of intensity distribution |
| **Minimum** | 1GSF | ``\min(X)`` | Minimum intensity |
| **10Percentile** | QG58 | ``P_{10}`` | 10th percentile intensity |
| **90Percentile** | 8DWT | ``P_{90}`` | 90th percentile intensity |
| **Maximum** | 84IY | ``\max(X)`` | Maximum intensity |
| **Mean** | Q4LE | ``\bar{X} = \frac{1}{N_p} \sum_{i=1}^{N_p} X(i)`` | Average intensity |
| **Median** | Y12H | Median of X | Middle value |
| **InterquartileRange** | SALO | ``IQR = P_{75} - P_{25}`` | Spread of middle 50% |
| **Range** | 2OJQ | ``\max(X) - \min(X)`` | Total intensity range |
| **MeanAbsoluteDeviation** | 4FUA | ``MAD = \frac{1}{N_p} \sum |X(i) - \bar{X}|`` | Average absolute deviation from mean |
| **RobustMeanAbsoluteDeviation** | 1128 | MAD on 10-90 percentile range | Outlier-robust MAD |
| **RootMeanSquared** | 5ZWQ | ``RMS = \sqrt{\frac{1}{N_p} \sum X(i)^2}`` | Quadratic mean |
| **StandardDeviation** | - | ``\sigma = \sqrt{Var}`` | Standard deviation (deprecated) |
| **Skewness** | KE2A | ``\gamma_1 = \frac{\mu_3}{\sigma^3}`` | Distribution asymmetry |
| **Kurtosis** | IPH6 | ``\gamma_2 = \frac{\mu_4}{\sigma^4}`` | Distribution peakedness |
| **Variance** | ECT3 | ``\sigma^2 = \frac{1}{N_p} \sum (X(i) - \bar{X})^2`` | Intensity spread |
| **Uniformity** | BJ5W | ``U = \sum p(i)^2`` | Intensity uniformity |

### Usage

```julia
using Radiomics

# Extract all first-order features
features = extract_firstorder(image, mask)

# Or individual functions
e = energy(voxels)
H = entropy(voxels)
```

---

## Shape Features

Shape features describe the geometric properties of the ROI. Different features are available for 2D and 3D images.

### 3D Shape Features (17 features)

| Feature | Formula/Description |
|---------|---------------------|
| **MeshVolume** | Volume computed from triangular mesh |
| **VoxelVolume** | ``V = N_p \times V_{voxel}`` |
| **SurfaceArea** | Area from triangular mesh |
| **SurfaceVolumeRatio** | ``\frac{A}{V}`` |
| **Sphericity** | ``\psi = \frac{\sqrt[3]{36\pi V^2}}{A}`` |
| **Compactness1** | ``\frac{V}{\sqrt{\pi} A^{3/2}}`` |
| **Compactness2** | ``\frac{36\pi V^2}{A^3}`` |
| **SphericalDisproportion** | ``\frac{A}{\sqrt[3]{36\pi V^2}}`` |
| **Maximum3DDiameter** | Largest pairwise distance between surface voxels |
| **Maximum2DDiameterSlice** | Max 2D diameter in axial plane |
| **Maximum2DDiameterColumn** | Max 2D diameter in coronal plane |
| **Maximum2DDiameterRow** | Max 2D diameter in sagittal plane |
| **MajorAxisLength** | Longest principal axis |
| **MinorAxisLength** | Second principal axis |
| **LeastAxisLength** | Shortest principal axis |
| **Elongation** | ``\frac{Minor}{Major}`` |
| **Flatness** | ``\frac{Least}{Major}`` |

### 2D Shape Features (10 features)

| Feature | Description |
|---------|-------------|
| **Perimeter** | Contour length using marching squares |
| **MeshSurface** | Surface area from mesh |
| **PixelSurface** | Area as pixel count times pixel area |
| **PerimeterSurfaceRatio** | Perimeter / Area |
| **Sphericity** | Circularity: ``\frac{4\pi A}{P^2}`` |
| **SphericalDisproportion** | ``\frac{P^2}{4\pi A}`` |
| **MaximumDiameter** | Largest distance across ROI |
| **MajorAxisLength** | Longest principal axis |
| **MinorAxisLength** | Shortest principal axis |
| **Elongation** | Minor/Major axis ratio |

### Usage

```julia
# 3D shape features
shape_3d = extract_shape_3d(mask, spacing)

# 2D shape features
shape_2d = extract_shape_2d(mask_2d, spacing_2d)

# Or use high-level API (auto-detects 2D/3D)
shape = extract_shape_only(mask; spacing=spacing)
```

---

## GLCM Features (24 features)

The Gray Level Co-occurrence Matrix (GLCM) captures texture information by analyzing how often pairs of voxels with specific gray levels occur at a fixed spatial relationship.

### Matrix Definition

``P(i,j|\delta,\theta)`` = probability that gray levels i and j occur at distance ``\delta`` in direction ``\theta``.

### Feature List

| Feature | Formula | Description |
|---------|---------|-------------|
| **Autocorrelation** | ``\sum\sum i \cdot j \cdot P(i,j)`` | Linear dependency |
| **JointAverage** | ``\mu_x = \sum\sum i \cdot P(i,j)`` | Mean gray level |
| **ClusterProminence** | ``\sum\sum (i+j-\mu_x-\mu_y)^4 P(i,j)`` | Asymmetry measure |
| **ClusterShade** | ``\sum\sum (i+j-\mu_x-\mu_y)^3 P(i,j)`` | Skewness |
| **ClusterTendency** | ``\sum\sum (i+j-\mu_x-\mu_y)^2 P(i,j)`` | Grouping tendency |
| **Contrast** | ``\sum\sum |i-j|^2 P(i,j)`` | Local intensity variation |
| **Correlation** | ``\frac{\sum\sum (i-\mu_x)(j-\mu_y)P(i,j)}{\sigma_x\sigma_y}`` | Linear correlation |
| **DifferenceAverage** | ``\sum k \cdot P_{x-y}(k)`` | Mean of diagonal |
| **DifferenceEntropy** | ``-\sum P_{x-y}(k) \log_2 P_{x-y}(k)`` | Entropy of diagonal |
| **DifferenceVariance** | Variance of ``P_{x-y}`` | Spread of diagonal |
| **JointEnergy** | ``\sum\sum P(i,j)^2`` | Uniformity (ASM) |
| **JointEntropy** | ``-\sum\sum P(i,j) \log_2 P(i,j)`` | Randomness |
| **Imc1** | First correlation measure | Information measure 1 |
| **Imc2** | Second correlation measure | Information measure 2 |
| **Idm** | ``\sum\sum \frac{P(i,j)}{1+|i-j|^2}`` | Inverse difference moment |
| **Idmn** | Normalized IDM | Normalized for gray levels |
| **Id** | ``\sum\sum \frac{P(i,j)}{1+|i-j|}`` | Inverse difference |
| **Idn** | Normalized ID | Normalized for gray levels |
| **InverseVariance** | ``\sum\sum \frac{P(i,j)}{|i-j|^2}`` for i≠j | Inverse variance |
| **MaximumProbability** | ``\max(P(i,j))`` | Maximum entry |
| **SumAverage** | ``\sum k \cdot P_{x+y}(k)`` | Mean of anti-diagonal |
| **SumEntropy** | ``-\sum P_{x+y}(k) \log_2 P_{x+y}(k)`` | Entropy of anti-diagonal |
| **SumSquares** | ``\sum\sum (i-\mu)^2 P(i,j)`` | Variance |
| **MCC** | Maximal correlation coefficient | Complex correlation |

### Usage

```julia
# Compute GLCM and features
glcm_result = compute_glcm(image, mask; binwidth=25.0, distance=1)
features = extract_glcm(image, mask)
```

---

## GLRLM Features (16 features)

The Gray Level Run Length Matrix (GLRLM) quantifies gray level runs: consecutive voxels with the same gray level along a direction.

### Matrix Definition

``R(i,j)`` = number of runs of gray level i with length j.

### Feature List

| Feature | Description |
|---------|-------------|
| **ShortRunEmphasis** | Emphasizes short runs |
| **LongRunEmphasis** | Emphasizes long runs |
| **GrayLevelNonUniformity** | Gray level distribution |
| **GrayLevelNonUniformityNormalized** | Normalized GLNU |
| **RunLengthNonUniformity** | Run length distribution |
| **RunLengthNonUniformityNormalized** | Normalized RLNU |
| **RunPercentage** | Ratio of runs to voxels |
| **GrayLevelVariance** | Gray level variance in runs |
| **RunVariance** | Run length variance |
| **RunEntropy** | Entropy of run distribution |
| **LowGrayLevelRunEmphasis** | Emphasizes low intensity runs |
| **HighGrayLevelRunEmphasis** | Emphasizes high intensity runs |
| **ShortRunLowGrayLevelEmphasis** | Short, low intensity |
| **ShortRunHighGrayLevelEmphasis** | Short, high intensity |
| **LongRunLowGrayLevelEmphasis** | Long, low intensity |
| **LongRunHighGrayLevelEmphasis** | Long, high intensity |

### Usage

```julia
glrlm_result = compute_glrlm(image, mask; binwidth=25.0)
features = extract_glrlm(image, mask)
```

---

## GLSZM Features (16 features)

The Gray Level Size Zone Matrix (GLSZM) describes connected regions (zones) of the same gray level.

### Matrix Definition

``Z(i,j)`` = number of zones of gray level i with size j voxels.

### Feature List

| Feature | Description |
|---------|-------------|
| **SmallAreaEmphasis** | Emphasizes small zones |
| **LargeAreaEmphasis** | Emphasizes large zones |
| **GrayLevelNonUniformity** | Gray level distribution |
| **GrayLevelNonUniformityNormalized** | Normalized GLNU |
| **SizeZoneNonUniformity** | Zone size distribution |
| **SizeZoneNonUniformityNormalized** | Normalized SZNU |
| **ZonePercentage** | Ratio of zones to voxels |
| **GrayLevelVariance** | Gray level variance |
| **ZoneVariance** | Zone size variance |
| **ZoneEntropy** | Entropy of zone distribution |
| **LowGrayLevelZoneEmphasis** | Low intensity zones |
| **HighGrayLevelZoneEmphasis** | High intensity zones |
| **SmallAreaLowGrayLevelEmphasis** | Small, low intensity |
| **SmallAreaHighGrayLevelEmphasis** | Small, high intensity |
| **LargeAreaLowGrayLevelEmphasis** | Large, low intensity |
| **LargeAreaHighGrayLevelEmphasis** | Large, high intensity |

### Usage

```julia
glszm_result = compute_glszm(image, mask; binwidth=25.0)
features = extract_glszm(image, mask)
```

---

## NGTDM Features (5 features)

The Neighboring Gray Tone Difference Matrix (NGTDM) measures the difference between a voxel's gray level and the average of its neighbors.

### Matrix Definition

- ``s_i`` = sum of absolute differences for gray level i
- ``p_i`` = probability of gray level i

### Feature List

| Feature | Formula | Description |
|---------|---------|-------------|
| **Coarseness** | ``\frac{1}{\epsilon + \sum p_i s_i}`` | Texture coarseness |
| **Contrast** | ``\frac{1}{N_g(N_g-1)} \sum\sum p_i p_j (i-j)^2 \times \frac{\sum s_i}{N_v}`` | Local variation |
| **Busyness** | ``\frac{\sum p_i s_i}{\sum\sum |i p_i - j p_j|}`` | Spatial rate of change |
| **Complexity** | ``\sum\sum \frac{|i-j|(p_i s_i + p_j s_j)}{N_v(p_i + p_j)}`` | Texture complexity |
| **Strength** | ``\frac{\sum\sum (p_i + p_j)(i-j)^2}{\epsilon + \sum s_i}`` | Primitive strength |

### Usage

```julia
ngtdm_result = compute_ngtdm(image, mask; binwidth=25.0, distance=1)
features = extract_ngtdm(image, mask)
```

---

## GLDM Features (14 features)

The Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies: how many voxels in the neighborhood have similar gray levels.

### Matrix Definition

``D(i,j)`` = number of voxels with gray level i that have j dependent voxels in their neighborhood.

A voxel is "dependent" if ``|g_{center} - g_{neighbor}| \leq \alpha`` where ``\alpha`` is the coarseness parameter.

### Feature List

| Feature | Description |
|---------|-------------|
| **SmallDependenceEmphasis** | Emphasizes small dependencies |
| **LargeDependenceEmphasis** | Emphasizes large dependencies |
| **GrayLevelNonUniformity** | Gray level distribution |
| **DependenceNonUniformity** | Dependence distribution |
| **DependenceNonUniformityNormalized** | Normalized DNU |
| **GrayLevelVariance** | Gray level variance |
| **DependenceVariance** | Dependence variance |
| **DependenceEntropy** | Entropy of dependencies |
| **LowGrayLevelEmphasis** | Low intensity emphasis |
| **HighGrayLevelEmphasis** | High intensity emphasis |
| **SmallDependenceLowGrayLevelEmphasis** | Small dep., low intensity |
| **SmallDependenceHighGrayLevelEmphasis** | Small dep., high intensity |
| **LargeDependenceLowGrayLevelEmphasis** | Large dep., low intensity |
| **LargeDependenceHighGrayLevelEmphasis** | Large dep., high intensity |

### Usage

```julia
gldm_result = compute_gldm(image, mask; binwidth=25.0, alpha=0)
features = extract_gldm(image, mask; alpha=0)
```

---

## References

- [IBSI Documentation](https://ibsi.readthedocs.io/) - Standard feature definitions
- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/en/latest/features.html)
- Zwanenburg A, et al. (2020). The Image Biomarker Standardization Initiative
