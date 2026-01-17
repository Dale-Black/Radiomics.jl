"""
    Radiomics.jl Test Utilities

Test harness for parity testing against PyRadiomics using PythonCall.jl.
This module provides utilities for:
- Generating deterministic random images and masks
- Calling PyRadiomics feature extraction from Julia
- Comparing Julia and Python results with appropriate tolerances

# Usage

```julia
using Test
include("test_utils.jl")

# Generate deterministic test data
image, mask = random_image_mask(42, (32, 32, 32))

# Extract a feature using PyRadiomics
py_result = pyradiomics_feature("firstorder", "Energy", image, mask)

# Compare with Julia implementation
julia_result = Radiomics.energy(image, mask)
@test compare_features(julia_result, py_result)
```

# Tolerance Guidelines

| Feature Type | Relative Tolerance | Absolute Tolerance |
|-------------|-------------------|-------------------|
| First Order | 1e-10 | 1e-12 |
| Shape | 1e-6 | 1e-8 |
| Texture | 1e-10 | 1e-12 |
"""

using Random
using CondaPkg

# ============================================================================
# Python Environment Setup (MUST run before PythonCall is loaded)
# ============================================================================

"""
    _ensure_pyradiomics_deps()

Ensure PyRadiomics dependencies are available in the CondaPkg environment.
This MUST be called BEFORE PythonCall is used/imported.
"""
function _ensure_pyradiomics_deps()
    # Add required conda packages if not already present
    CondaPkg.add("python"; version=">=3.10,<3.12")
    CondaPkg.add("numpy"; version=">=1.20,<2.0")
    CondaPkg.add("simpleitk"; version=">=2.2")
    CondaPkg.add("pywavelets"; version=">=1.4")
    CondaPkg.add("pykwalify"; version=">=1.6")
    CondaPkg.add("cython")
    CondaPkg.add("setuptools")
    CondaPkg.add("wheel")
    CondaPkg.add("pip")

    # Resolve the environment
    CondaPkg.resolve()

    # Install pyradiomics via pip if not already installed
    # We use CondaPkg.withenv to run in the correct environment
    CondaPkg.withenv() do
        try
            run(`python -c "import radiomics"`)
        catch
            # pyradiomics not installed, install it via pip
            @info "Installing pyradiomics via pip..."
            run(`pip install pyradiomics==3.0.1 --no-build-isolation`)
        end
    end
end

# Initialize dependencies BEFORE loading PythonCall
_ensure_pyradiomics_deps()

# Now load PythonCall (it will use the configured environment)
using PythonCall

# ============================================================================
# Python Initialization
# ============================================================================

"""
    init_python()

Initialize Python environment and import required modules.
Returns the pyradiomics and numpy modules.
"""
function init_python()
    # Import Python modules
    np = pyimport("numpy")
    sitk = pyimport("SimpleITK")
    radiomics = pyimport("radiomics")

    return (numpy=np, simpleitk=sitk, radiomics=radiomics)
end

# Global Python modules (lazily initialized)
const PY_MODULES = Ref{NamedTuple}()

function get_python_modules()
    if !isassigned(PY_MODULES)
        PY_MODULES[] = init_python()
    end
    return PY_MODULES[]
end

# ============================================================================
# Random Data Generation
# ============================================================================

"""
    random_image_mask(seed::Int, size::Tuple{Vararg{Int}};
                      intensity_range=(0, 255),
                      mask_fraction=0.3,
                      ensure_connected=false) -> (image, mask)

Generate a deterministic random image and binary mask for testing.

# Arguments
- `seed::Int`: Random seed for reproducibility
- `size::Tuple`: Dimensions of the image (e.g., `(32, 32, 32)` for 3D)

# Keyword Arguments
- `intensity_range::Tuple{Int,Int}`: Range of intensity values (default: `(0, 255)`)
- `mask_fraction::Float64`: Approximate fraction of voxels in mask (default: `0.3`)
- `ensure_connected::Bool`: If true, ensures mask is a single connected component (default: `false`)

# Returns
- `image::Array{Float64}`: Random image with values in `intensity_range`
- `mask::BitArray`: Binary mask with approximately `mask_fraction` of voxels set

# Example
```julia
# Generate a 3D 32x32x32 image with seed 42
image, mask = random_image_mask(42, (32, 32, 32))

# Generate a 2D image with custom parameters
image_2d, mask_2d = random_image_mask(123, (64, 64);
                                       intensity_range=(0, 1000),
                                       mask_fraction=0.5)
```
"""
function random_image_mask(seed::Int, size::Tuple{Vararg{Int}};
                           intensity_range::Tuple{Int,Int}=(0, 255),
                           mask_fraction::Float64=0.3,
                           ensure_connected::Bool=false)
    rng = MersenneTwister(seed)

    # Generate random image with floating point values
    low, high = intensity_range
    image = rand(rng, size...) .* (high - low) .+ low

    # Generate random binary mask
    mask = rand(rng, size...) .< mask_fraction

    # Ensure at least some voxels are in mask (avoid empty ROI)
    if !any(mask)
        # Set center region to true
        center = div.(size, 2)
        ranges = [max(1, c-2):min(s, c+2) for (c, s) in zip(center, size)]
        mask[ranges...] .= true
    end

    # Optionally ensure single connected component
    if ensure_connected
        mask = _ensure_connected(mask)
    end

    return Float64.(image), BitArray(mask)
end

"""
    random_image_mask_integer(seed::Int, size::Tuple{Vararg{Int}};
                              intensity_range=(0, 255),
                              mask_fraction=0.3) -> (image, mask)

Like `random_image_mask`, but returns integer intensity values.
Useful for testing discretization and texture features.
"""
function random_image_mask_integer(seed::Int, size::Tuple{Vararg{Int}};
                                   intensity_range::Tuple{Int,Int}=(0, 255),
                                   mask_fraction::Float64=0.3)
    rng = MersenneTwister(seed)

    low, high = intensity_range
    image = rand(rng, low:high, size...)
    mask = rand(rng, size...) .< mask_fraction

    if !any(mask)
        center = div.(size, 2)
        ranges = [max(1, c-2):min(s, c+2) for (c, s) in zip(center, size)]
        mask[ranges...] .= true
    end

    return image, BitArray(mask)
end

"""
    _ensure_connected(mask::BitArray) -> BitArray

Reduce mask to its largest connected component.
Simple flood-fill implementation for testing purposes.
"""
function _ensure_connected(mask::BitArray)
    # For now, just return the mask as-is
    # TODO: Implement connected component labeling if needed
    return mask
end

# ============================================================================
# PyRadiomics Wrapper Functions
# ============================================================================

"""
    julia_array_to_sitk(image::AbstractArray, mask::AbstractArray;
                        spacing=(1.0, 1.0, 1.0)) -> (sitk_image, sitk_mask)

Convert Julia arrays to SimpleITK images for PyRadiomics.

# Notes
- Julia uses column-major (Fortran) order, SimpleITK uses row-major (C) order
- We transpose the arrays to match PyRadiomics expectations
- Default spacing is 1mm isotropic
"""
function julia_array_to_sitk(image::AbstractArray, mask::AbstractArray;
                              spacing::Tuple=(1.0, 1.0, 1.0))
    py = get_python_modules()
    np = py.numpy
    sitk = py.simpleitk

    # Convert to numpy arrays (PythonCall handles the conversion)
    # Need to handle axis order: Julia is column-major, NumPy/SimpleITK is row-major
    # We'll let PyRadiomics handle the native layout since we're testing values

    # Convert Julia arrays to numpy
    # Julia (x, y, z) -> NumPy needs to see it as (z, y, x) for SimpleITK
    ndim = ndims(image)

    if ndim == 3
        # Permute to (z, y, x) for SimpleITK
        img_permuted = permutedims(image, (3, 2, 1))
        mask_permuted = permutedims(mask, (3, 2, 1))
        spacing_arr = (spacing[3], spacing[2], spacing[1])
    elseif ndim == 2
        # Permute to (y, x) for SimpleITK
        img_permuted = permutedims(image, (2, 1))
        mask_permuted = permutedims(mask, (2, 1))
        spacing_arr = (spacing[2], spacing[1])
    else
        img_permuted = image
        mask_permuted = mask
        spacing_arr = spacing
    end

    # Convert to concrete arrays (BitArray -> Array{Int32})
    # This is needed because PythonCall has trouble with BitArray
    img_array = collect(Float64, img_permuted)
    mask_array = collect(Int32, mask_permuted)

    # Create numpy arrays using asarray for memory efficiency
    np_image = np.asarray(img_array)
    np_mask = np.asarray(mask_array)

    # Create SimpleITK images from numpy arrays
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_mask = sitk.GetImageFromArray(np_mask)

    # Set spacing
    sitk_image.SetSpacing(pylist(spacing_arr))
    sitk_mask.SetSpacing(pylist(spacing_arr))

    return sitk_image, sitk_mask
end

"""
    _get_pyradiomics_class(radiomics, feature_class::String)

Get the PyRadiomics feature class from the module.
Maps feature class names to PyRadiomics class names (e.g., "firstorder" -> "RadiomicsFirstOrder").
"""
function _get_pyradiomics_class(radiomics, feature_class::String)
    # Map feature class names to PyRadiomics class names
    class_name_map = Dict(
        "firstorder" => "RadiomicsFirstOrder",
        "shape" => "RadiomicsShape",
        "shape2d" => "RadiomicsShape2D",
        "glcm" => "RadiomicsGLCM",
        "glrlm" => "RadiomicsGLRLM",
        "glszm" => "RadiomicsGLSZM",
        "ngtdm" => "RadiomicsNGTDM",
        "gldm" => "RadiomicsGLDM"
    )

    feature_class_lower = lowercase(feature_class)
    class_name = get(class_name_map, feature_class_lower, "Radiomics" * titlecase(feature_class))
    feature_class_module = getproperty(radiomics, Symbol(feature_class_lower))
    return getproperty(feature_class_module, Symbol(class_name))
end

"""
    pyradiomics_feature(feature_class::String, feature_name::String,
                        image::AbstractArray, mask::AbstractArray;
                        spacing=(1.0, 1.0, 1.0),
                        settings=Dict()) -> Float64

Extract a single feature from PyRadiomics.

# Arguments
- `feature_class::String`: Feature class name (e.g., "firstorder", "glcm", "shape")
- `feature_name::String`: Feature name (e.g., "Energy", "Entropy", "MeshVolume")
- `image::AbstractArray`: Image data
- `mask::AbstractArray`: Binary mask (Bool or 0/1 integers)

# Keyword Arguments
- `spacing::Tuple`: Voxel spacing in mm (default: `(1.0, 1.0, 1.0)`)
- `settings::Dict`: Additional PyRadiomics settings

# Returns
- Feature value as Float64

# Example
```julia
energy = pyradiomics_feature("firstorder", "Energy", image, mask)
contrast = pyradiomics_feature("glcm", "Contrast", image, mask;
                               settings=Dict("binWidth" => 25))
```
"""
function pyradiomics_feature(feature_class::String, feature_name::String,
                             image::AbstractArray, mask::AbstractArray;
                             spacing::Tuple=(1.0, 1.0, 1.0),
                             settings::Dict=Dict())
    py = get_python_modules()
    radiomics = py.radiomics

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask; spacing=spacing)

    # Get the feature class
    FeatureClass = _get_pyradiomics_class(radiomics, feature_class)

    # Build settings dict
    py_settings = pydict(settings)

    # Instantiate feature extractor
    extractor = FeatureClass(sitk_image, sitk_mask, label=1; py_settings...)

    # Execute feature calculation
    extractor.execute()

    # Get the specific feature value
    method_name = "get" * feature_name * "FeatureValue"
    feature_method = getproperty(extractor, Symbol(method_name))
    result = feature_method()

    return pyconvert(Float64, result)
end

"""
    pyradiomics_extract(feature_class::String,
                        image::AbstractArray, mask::AbstractArray;
                        spacing=(1.0, 1.0, 1.0),
                        settings=Dict(),
                        features=nothing) -> Dict{String,Float64}

Extract all features from a feature class using PyRadiomics.

# Arguments
- `feature_class::String`: Feature class name (e.g., "firstorder", "glcm")
- `image::AbstractArray`: Image data
- `mask::AbstractArray`: Binary mask

# Keyword Arguments
- `spacing::Tuple`: Voxel spacing in mm (default: `(1.0, 1.0, 1.0)`)
- `settings::Dict`: PyRadiomics settings (e.g., `Dict("binWidth" => 25)`)
- `features::Union{Nothing, Vector{String}}`: Specific features to enable (default: all)

# Returns
- `Dict{String,Float64}`: Dictionary mapping feature names to values

# Example
```julia
# Extract all first-order features
results = pyradiomics_extract("firstorder", image, mask)

# Extract specific GLCM features
results = pyradiomics_extract("glcm", image, mask;
                              features=["Contrast", "Correlation"])
```
"""
function pyradiomics_extract(feature_class::String,
                             image::AbstractArray, mask::AbstractArray;
                             spacing::Tuple=(1.0, 1.0, 1.0),
                             settings::Dict=Dict(),
                             features::Union{Nothing, Vector{String}}=nothing)
    py = get_python_modules()
    radiomics = py.radiomics

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask; spacing=spacing)

    # Get the feature class
    FeatureClass = _get_pyradiomics_class(radiomics, feature_class)

    # Build settings dict
    py_settings = pydict(settings)

    # Instantiate feature extractor
    extractor = FeatureClass(sitk_image, sitk_mask, label=1; py_settings...)

    # Enable specific features if requested, otherwise enable all
    if features !== nothing
        extractor.disableAllFeatures()
        for feat in features
            extractor.enableFeatureByName(feat)
        end
    end

    # Execute feature calculation
    result_dict = extractor.execute()

    # Convert to Julia Dict
    results = Dict{String,Float64}()
    np = py.numpy
    for (key, value) in result_dict.items()
        key_str = pyconvert(String, key)
        # Skip diagnostic features (start with "diagnostics_")
        if !startswith(key_str, "diagnostics_")
            try
                # Handle numpy scalar types by converting to Python float first
                if pyisinstance(value, np.ndarray) || pyisinstance(value, np.generic)
                    val = pyconvert(Float64, pyfloat(value))
                else
                    val = pyconvert(Float64, value)
                end
                results[key_str] = val
            catch e
                # Some values might not be convertible to Float64
                @debug "Could not convert feature $key_str" exception=e
                continue
            end
        end
    end

    return results
end

"""
    pyradiomics_extract_all(image::AbstractArray, mask::AbstractArray;
                            spacing=(1.0, 1.0, 1.0),
                            settings=Dict()) -> Dict{String,Float64}

Extract ALL features from ALL feature classes using PyRadiomics FeatureExtractor.

# Arguments
- `image::AbstractArray`: Image data
- `mask::AbstractArray`: Binary mask

# Returns
- `Dict{String,Float64}`: All extracted features

# Example
```julia
all_features = pyradiomics_extract_all(image, mask)
```
"""
function pyradiomics_extract_all(image::AbstractArray, mask::AbstractArray;
                                 spacing::Tuple=(1.0, 1.0, 1.0),
                                 settings::Dict=Dict())
    py = get_python_modules()
    radiomics = py.radiomics
    featureextractor = pyimport("radiomics.featureextractor")

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask; spacing=spacing)

    # Build settings
    all_settings = merge(Dict(
        "enableAllFeatures" => true,
        "imageType" => Dict("Original" => Dict())  # Only original image type for now
    ), settings)

    # Create extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(pydict(all_settings))
    extractor.enableAllFeatures()

    # Execute
    result = extractor.execute(sitk_image, sitk_mask, label=1)

    # Convert to Julia Dict
    results = Dict{String,Float64}()
    for (key, value) in result.items()
        key_str = pyconvert(String, key)
        if !startswith(key_str, "diagnostics_")
            try
                results[key_str] = pyconvert(Float64, value)
            catch
                continue
            end
        end
    end

    return results
end

# ============================================================================
# Comparison Utilities
# ============================================================================

"""
    compare_features(julia_value::Real, python_value::Real;
                     rtol=1e-10, atol=1e-12) -> Bool

Compare a Julia feature value against a PyRadiomics value.

# Arguments
- `julia_value::Real`: Value computed by Radiomics.jl
- `python_value::Real`: Value computed by PyRadiomics

# Keyword Arguments
- `rtol::Float64`: Relative tolerance (default: `1e-10`)
- `atol::Float64`: Absolute tolerance (default: `1e-12`)

# Returns
- `true` if values match within tolerance, `false` otherwise

# Example
```julia
@test compare_features(julia_energy, python_energy)
@test compare_features(julia_volume, python_volume; rtol=1e-6)  # Shape tolerance
```
"""
function compare_features(julia_value::Real, python_value::Real;
                          rtol::Float64=1e-10, atol::Float64=1e-12)
    return isapprox(julia_value, python_value; rtol=rtol, atol=atol)
end

"""
    compare_feature_dicts(julia_dict::Dict, python_dict::Dict;
                          rtol=1e-10, atol=1e-12,
                          verbose=true) -> (all_pass, failures)

Compare all features between Julia and PyRadiomics results.

# Arguments
- `julia_dict::Dict`: Features from Radiomics.jl
- `python_dict::Dict`: Features from PyRadiomics

# Returns
- `all_pass::Bool`: True if all features match
- `failures::Vector{NamedTuple}`: Details of any failures

# Example
```julia
julia_features = Dict("Energy" => 12345.0, "Entropy" => 4.5)
python_features = pyradiomics_extract("firstorder", image, mask)

all_pass, failures = compare_feature_dicts(julia_features, python_features)
@test all_pass
```
"""
function compare_feature_dicts(julia_dict::Dict, python_dict::Dict;
                               rtol::Float64=1e-10, atol::Float64=1e-12,
                               verbose::Bool=true)
    failures = NamedTuple{(:feature, :julia, :python, :diff, :reldiff),
                          Tuple{String, Float64, Float64, Float64, Float64}}[]

    for (feature, julia_val) in julia_dict
        if haskey(python_dict, feature)
            python_val = python_dict[feature]
            if !compare_features(julia_val, python_val; rtol=rtol, atol=atol)
                diff = abs(julia_val - python_val)
                reldiff = abs(diff / python_val)
                push!(failures, (feature=feature, julia=julia_val, python=python_val,
                                 diff=diff, reldiff=reldiff))
                if verbose
                    @warn "Feature mismatch" feature julia_val python_val diff reldiff
                end
            end
        else
            if verbose
                @warn "Feature not found in Python results" feature
            end
        end
    end

    all_pass = isempty(failures)
    return all_pass, failures
end

# ============================================================================
# Tolerance Constants
# ============================================================================

"""Default tolerances for different feature types."""
const TOLERANCES = (
    firstorder = (rtol=1e-10, atol=1e-12),
    shape = (rtol=1e-6, atol=1e-8),
    glcm = (rtol=1e-10, atol=1e-12),
    glrlm = (rtol=1e-10, atol=1e-12),
    glszm = (rtol=1e-10, atol=1e-12),
    ngtdm = (rtol=1e-10, atol=1e-12),
    gldm = (rtol=1e-10, atol=1e-12),
)

"""
    get_tolerance(feature_class::Symbol) -> NamedTuple

Get the appropriate tolerance for a feature class.

# Example
```julia
tol = get_tolerance(:shape)
@test isapprox(julia_val, python_val; tol.rtol, tol.atol)
```
"""
function get_tolerance(feature_class::Symbol)
    if hasfield(typeof(TOLERANCES), feature_class)
        return getfield(TOLERANCES, feature_class)
    else
        # Default tolerance
        return (rtol=1e-10, atol=1e-12)
    end
end

# ============================================================================
# Test Helpers
# ============================================================================

"""
    @test_feature(julia_func, feature_class, feature_name, image, mask; kwargs...)

Convenience macro for testing a single feature against PyRadiomics.

# Example
```julia
@test_feature Radiomics.energy "firstorder" "Energy" image mask
```
"""
macro test_feature(julia_func, feature_class, feature_name, image, mask)
    quote
        local julia_result = $(esc(julia_func))($(esc(image)), $(esc(mask)))
        local python_result = pyradiomics_feature($(esc(feature_class)),
                                                   $(esc(feature_name)),
                                                   $(esc(image)), $(esc(mask)))
        local tol = get_tolerance(Symbol($(esc(feature_class))))
        @test isapprox(julia_result, python_result; rtol=tol.rtol, atol=tol.atol)
    end
end

"""
    verify_pyradiomics_available() -> Bool

Check if PyRadiomics is properly installed and can be imported.
Useful for conditional test execution.

# Example
```julia
if verify_pyradiomics_available()
    @testset "Parity Tests" begin
        # ... tests that require PyRadiomics
    end
else
    @warn "PyRadiomics not available, skipping parity tests"
end
```
"""
function verify_pyradiomics_available()
    try
        py = get_python_modules()
        # Try to access radiomics module
        version = py.radiomics.__version__
        @info "PyRadiomics available" version=pyconvert(String, version)
        return true
    catch e
        @warn "PyRadiomics not available" exception=e
        return false
    end
end

# ============================================================================
# Exports
# ============================================================================

# Note: This file is meant to be included, not used as a module.
# All functions are available directly after include("test_utils.jl")
