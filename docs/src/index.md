# Radiomics.jl

A pure Julia implementation of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) for extracting radiomic features from medical images.

## Overview

Radiomics.jl provides comprehensive radiomic feature extraction capabilities, including:

- **First-order features** (19 features) - Statistical features from voxel intensity distributions
- **Shape features** (17 3D features, 10 2D features) - Morphological features describing ROI geometry
- **GLCM features** (24 features) - Gray Level Co-occurrence Matrix texture features
- **GLRLM features** (16 features) - Gray Level Run Length Matrix texture features
- **GLSZM features** (16 features) - Gray Level Size Zone Matrix texture features
- **NGTDM features** (5 features) - Neighboring Gray Tone Difference Matrix features
- **GLDM features** (14 features) - Gray Level Dependence Matrix texture features

**Total: 111+ radiomic features** with verified 1:1 parity against PyRadiomics.

## Features

- **Pure Julia** - No Python runtime dependencies
- **PyRadiomics Compatible** - Exact numerical parity with PyRadiomics
- **IBSI Compliant** - Follows Image Biomarker Standardization Initiative definitions
- **High Performance** - Leverages Julia's speed and type system
- **Flexible API** - From simple one-liners to full customization

## Quick Example

```julia
using Radiomics

# Create sample image and mask
image = rand(64, 64, 64) * 100
mask = rand(Bool, 64, 64, 64)

# Extract all features with one line
features = extract_all(image, mask)

# Access specific features
println("Energy: ", features["firstorder_Energy"])
println("Contrast: ", features["glcm_Contrast"])
println("Sphericity: ", features["shape_Sphericity"])
```

## Installation

```julia
using Pkg
Pkg.add("Radiomics")
```

Or from the Julia REPL package mode:
```
] add Radiomics
```

## Contents

```@contents
Pages = ["getting_started.md", "features.md", "configuration.md", "api.md"]
```

## References

- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) - Original Python implementation
- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
- [IBSI](https://ibsi.readthedocs.io/) - Image Biomarker Standardization Initiative

## License

Radiomics.jl is released under the MIT License.
