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

# Package version
const VERSION = v"0.1.0"

# Module files will be included here as they are implemented
# include("types.jl")
# include("utils.jl")
# include("discretization.jl")
# include("firstorder.jl")
# include("shape.jl")
# include("glcm.jl")
# include("glrlm.jl")
# include("glszm.jl")
# include("ngtdm.jl")
# include("gldm.jl")
# include("extractor.jl")

# Exports will be added as features are implemented

end # module Radiomics
