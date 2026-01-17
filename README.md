# Radiomics.jl

[![CI](https://github.com/daleblack/Radiomics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/daleblack/Radiomics.jl/actions/workflows/CI.yml)

A pure Julia implementation of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) for extracting radiomic features from medical images.

## Status

🚧 **Under Development** - This package is being developed using automated agent orchestration.

## Features

Radiomics.jl will implement all feature classes from PyRadiomics:

- [ ] First Order Statistics (19 features)
- [ ] Shape Features 2D/3D (30+ features)
- [ ] GLCM - Gray Level Co-occurrence Matrix (24 features)
- [ ] GLRLM - Gray Level Run Length Matrix (16 features)
- [ ] GLSZM - Gray Level Size Zone Matrix (16 features)
- [ ] NGTDM - Neighboring Gray Tone Difference Matrix (5 features)
- [ ] GLDM - Gray Level Dependence Matrix (14 features)

## Goals

1. **Pure Julia** - No Python runtime dependencies
2. **1:1 Parity** - Exact numerical match with PyRadiomics
3. **IBSI Compliant** - Following Image Biomarker Standardization Initiative
4. **Performant** - Type-stable, allocation-efficient code

## Installation

```julia
using Pkg
Pkg.add("Radiomics")  # Once registered
```

## Usage

```julia
using Radiomics

# Extract all features
features = extract_all(image, mask)

# Extract specific feature class
firstorder = extract_firstorder(image, mask)
glcm = extract_glcm(image, mask; distance=1)
```

## Testing

Tests verify parity against PyRadiomics using PythonCall.jl:

```julia
using Pkg
Pkg.test("Radiomics")
```

## References

- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics)
- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
- [IBSI](https://ibsi.readthedocs.io/) - Image Biomarker Standardization Initiative

## License

MIT License
