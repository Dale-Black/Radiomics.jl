# NGTDM Feature Parity Tests for Radiomics.jl
#
# This file tests all 5 NGTDM (Neighbouring Gray Tone Difference Matrix) features
# against PyRadiomics to verify 1:1 parity.
#
# Story: TEST-NGTDM-PARITY
#
# NGTDM Features (5 total):
# 1. Coarseness - Measures spatial rate of change in intensity
# 2. Contrast   - Measures dynamic range and spatial intensity change
# 3. Busyness   - Measures rapid changes between pixels and neighbors
# 4. Complexity - Measures number of primitive components
# 5. Strength   - Measures visibility of primitives
#
# Tolerance Guidelines:
# - NGTDM Features: rtol=1e-10, atol=1e-12 (standard texture tolerance)
#
# Test Strategy:
# 1. Test each feature individually against PyRadiomics
# 2. Use multiple random seeds (42, 123, 456) for robustness
# 3. Use multiple array sizes (small, medium)
# 4. Test with different binwidth settings

using Test
using Radiomics
using Statistics
using Random

# Test utilities should already be loaded by runtests.jl
# If running standalone: include("test_utils.jl")

#==============================================================================#
# Test Configuration
#==============================================================================#

# Standard tolerance for NGTDM features
const NGTDM_RTOL = 1e-10
const NGTDM_ATOL = 1e-12

# Random seeds for reproducibility
const NGTDM_TEST_SEEDS = [42, 123, 456]

# Array sizes to test
const NGTDM_SMALL_SIZE = (16, 16, 16)
const NGTDM_MEDIUM_SIZE = (32, 32, 32)

#==============================================================================#
# PyRadiomics Feature Name Mapping
#==============================================================================#

# Map Julia feature function names to PyRadiomics feature names
const NGTDM_FEATURE_MAP = Dict(
    :coarseness => "Coarseness",
    :contrast => "Contrast",
    :busyness => "Busyness",
    :complexity => "Complexity",
    :strength => "Strength"
)

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    get_pyradiomics_ngtdm(image, mask; binwidth=25.0)

Extract all NGTDM features from PyRadiomics for comparison.
Returns a Dict with feature names as keys and values as Float64.
"""
function get_pyradiomics_ngtdm(image::AbstractArray, mask::AbstractArray;
                                binwidth::Real=25.0)
    py = get_python_modules()
    radiomics = py.radiomics
    np = py.numpy

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask)

    # Get the NGTDM feature class directly
    ngtdm_module = pyimport("radiomics.ngtdm")
    RadiomicsNGTDM = ngtdm_module.RadiomicsNGTDM

    # Instantiate feature extractor with settings
    extractor = RadiomicsNGTDM(sitk_image, sitk_mask;
                                label=1,
                                binWidth=binwidth)

    # Execute feature calculation
    result_dict = extractor.execute()

    # Convert to Julia Dict
    results = Dict{String,Float64}()
    for (key, value) in result_dict.items()
        key_str = pyconvert(String, key)
        if !startswith(key_str, "diagnostics_")
            try
                if pyisinstance(value, np.ndarray) || pyisinstance(value, np.generic)
                    val = pyconvert(Float64, pyfloat(value))
                else
                    val = pyconvert(Float64, value)
                end
                results[key_str] = val
            catch e
                @debug "Could not convert feature $key_str" exception=e
                continue
            end
        end
    end

    return results
end

"""
    get_julia_ngtdm_features(image, mask; binwidth=25.0)

Extract all NGTDM features from Julia for comparison.
Returns a NamedTuple with all 5 NGTDM features.
"""
function get_julia_ngtdm_features(image::AbstractArray, mask::AbstractArray;
                                   binwidth::Real=25.0)
    # Compute NGTDM
    result = compute_ngtdm(image, mask; binwidth=binwidth)

    # Extract all features
    return (
        coarseness = ngtdm_coarseness(result),
        contrast = ngtdm_contrast(result),
        busyness = ngtdm_busyness(result),
        complexity = ngtdm_complexity(result),
        strength = ngtdm_strength(result)
    )
end

#==============================================================================#
# Individual Feature Tests
#==============================================================================#

@testset "NGTDM Feature Parity" begin

    @testset "Test Environment Setup" begin
        @test verify_pyradiomics_available()
    end

    #--------------------------------------------------------------------------
    # Feature 1: Coarseness
    #--------------------------------------------------------------------------
    @testset "Coarseness" begin
        @testset "Multiple seeds" begin
            for seed in NGTDM_TEST_SEEDS
                image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

                julia_features = get_julia_ngtdm_features(image, mask)
                py_features = get_pyradiomics_ngtdm(image, mask)

                @test haskey(py_features, "Coarseness")
                @test julia_features.coarseness ≈ py_features["Coarseness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 2: Contrast
    #--------------------------------------------------------------------------
    @testset "Contrast" begin
        @testset "Multiple seeds" begin
            for seed in NGTDM_TEST_SEEDS
                image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

                julia_features = get_julia_ngtdm_features(image, mask)
                py_features = get_pyradiomics_ngtdm(image, mask)

                @test haskey(py_features, "Contrast")
                @test julia_features.contrast ≈ py_features["Contrast"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 3: Busyness
    #--------------------------------------------------------------------------
    @testset "Busyness" begin
        @testset "Multiple seeds" begin
            for seed in NGTDM_TEST_SEEDS
                image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

                julia_features = get_julia_ngtdm_features(image, mask)
                py_features = get_pyradiomics_ngtdm(image, mask)

                @test haskey(py_features, "Busyness")
                @test julia_features.busyness ≈ py_features["Busyness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 4: Complexity
    #--------------------------------------------------------------------------
    @testset "Complexity" begin
        @testset "Multiple seeds" begin
            for seed in NGTDM_TEST_SEEDS
                image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

                julia_features = get_julia_ngtdm_features(image, mask)
                py_features = get_pyradiomics_ngtdm(image, mask)

                @test haskey(py_features, "Complexity")
                @test julia_features.complexity ≈ py_features["Complexity"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 5: Strength
    #--------------------------------------------------------------------------
    @testset "Strength" begin
        @testset "Multiple seeds" begin
            for seed in NGTDM_TEST_SEEDS
                image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

                julia_features = get_julia_ngtdm_features(image, mask)
                py_features = get_pyradiomics_ngtdm(image, mask)

                @test haskey(py_features, "Strength")
                @test julia_features.strength ≈ py_features["Strength"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end
end

#==============================================================================#
# Comprehensive Parity Test - All Features at Once
#==============================================================================#

@testset "Comprehensive NGTDM Parity" begin

    @testset "All features - Multiple seeds" begin
        for seed in NGTDM_TEST_SEEDS
            image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)

            julia_features = get_julia_ngtdm_features(image, mask)
            py_features = get_pyradiomics_ngtdm(image, mask)

            # Test each feature
            for (julia_sym, py_name) in NGTDM_FEATURE_MAP
                if haskey(py_features, py_name)
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    @test julia_val ≈ py_val rtol=NGTDM_RTOL atol=NGTDM_ATOL
                end
            end
        end
    end

    @testset "All features - Different sizes" begin
        for sz in [NGTDM_SMALL_SIZE, NGTDM_MEDIUM_SIZE]
            image, mask = random_image_mask(42, sz)

            julia_features = get_julia_ngtdm_features(image, mask)
            py_features = get_pyradiomics_ngtdm(image, mask)

            # Count passing features
            n_features = 0
            n_passed = 0

            for (julia_sym, py_name) in NGTDM_FEATURE_MAP
                if haskey(py_features, py_name)
                    n_features += 1
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    if isapprox(julia_val, py_val; rtol=NGTDM_RTOL, atol=NGTDM_ATOL)
                        n_passed += 1
                    else
                        @warn "Mismatch" size=sz feature=py_name julia=julia_val python=py_val
                    end
                end
            end

            @test n_passed == n_features
        end
    end
end

#==============================================================================#
# Different Discretization Settings
#==============================================================================#

@testset "Discretization Settings" begin

    @testset "Different binwidth values" begin
        image, mask = random_image_mask(42, NGTDM_MEDIUM_SIZE)

        for binwidth in [16.0, 25.0, 32.0, 64.0]
            julia_features = get_julia_ngtdm_features(image, mask; binwidth=binwidth)
            py_features = get_pyradiomics_ngtdm(image, mask; binwidth=binwidth)

            # Test all features
            @test julia_features.coarseness ≈ py_features["Coarseness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_features.contrast ≈ py_features["Contrast"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_features.busyness ≈ py_features["Busyness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_features.complexity ≈ py_features["Complexity"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_features.strength ≈ py_features["Strength"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
        end
    end
end

#==============================================================================#
# Edge Cases
#==============================================================================#

@testset "Edge Cases" begin

    @testset "Small mask region" begin
        # Create image with only a small mask region
        rng = MersenneTwister(42)
        image = rand(rng, 32, 32, 32) .* 255
        mask = zeros(Bool, 32, 32, 32)
        mask[14:18, 14:18, 14:18] .= true  # 5x5x5 = 125 voxels

        julia_features = get_julia_ngtdm_features(image, mask)
        py_features = get_pyradiomics_ngtdm(image, mask)

        # Test all features
        for (julia_sym, py_name) in NGTDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    @testset "High intensity values" begin
        # CT-like intensity range
        rng = MersenneTwister(42)
        image = (rand(rng, 32, 32, 32) .* 2000) .- 1000  # Range -1000 to 1000 (like HU)
        mask = rand(rng, 32, 32, 32) .< 0.3

        # Ensure non-empty mask
        if !any(mask)
            mask[16, 16, 16] = true
        end

        julia_features = get_julia_ngtdm_features(image, mask)
        py_features = get_pyradiomics_ngtdm(image, mask)

        # Test all features
        for (julia_sym, py_name) in NGTDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end

    @testset "Integer image values" begin
        # Discrete integer values
        image_int, mask = random_image_mask_integer(42, NGTDM_MEDIUM_SIZE; intensity_range=(0, 255))
        image = Float64.(image_int)

        julia_features = get_julia_ngtdm_features(image, mask)
        py_features = get_pyradiomics_ngtdm(image, mask)

        # All features should match
        for (julia_sym, py_name) in NGTDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=NGTDM_RTOL atol=NGTDM_ATOL
            end
        end
    end
end

#==============================================================================#
# Feature Consistency Tests (Julia internal)
#==============================================================================#

@testset "Feature Consistency" begin

    @testset "NGTDM mathematical relationships" begin
        image, mask = random_image_mask(42, NGTDM_MEDIUM_SIZE)
        features = get_julia_ngtdm_features(image, mask)

        # Coarseness should be > 0 (unless perfectly homogeneous)
        @test features.coarseness > 0.0

        # Contrast should be >= 0
        @test features.contrast >= 0.0

        # Busyness should be >= 0
        @test features.busyness >= 0.0

        # Complexity should be >= 0
        @test features.complexity >= 0.0

        # Strength should be >= 0
        @test features.strength >= 0.0
    end

    @testset "Feature bounds" begin
        for seed in NGTDM_TEST_SEEDS
            image, mask = random_image_mask(seed, NGTDM_MEDIUM_SIZE)
            features = get_julia_ngtdm_features(image, mask)

            # All features should be finite
            for field in fieldnames(typeof(features))
                val = getfield(features, field)
                @test isfinite(val)
            end
        end
    end
end

#==============================================================================#
# 2D Image Tests
#==============================================================================#

@testset "2D Image Parity" begin
    @testset "2D NGTDM features" begin
        for seed in NGTDM_TEST_SEEDS
            # Generate 2D image
            rng = MersenneTwister(seed)
            image = rand(rng, 64, 64) .* 255
            mask = rand(rng, 64, 64) .< 0.3

            # Ensure non-empty mask
            if !any(mask)
                mask[32, 32] = true
            end

            # For PyRadiomics, we need 3D, so add a singleton dimension
            image_3d = reshape(image, size(image)..., 1)
            mask_3d = reshape(mask, size(mask)..., 1)

            # Julia can work with 2D directly
            result_2d = compute_ngtdm(image, mask; binwidth=25.0)
            julia_coarseness = ngtdm_coarseness(result_2d)
            julia_contrast = ngtdm_contrast(result_2d)
            julia_busyness = ngtdm_busyness(result_2d)
            julia_complexity = ngtdm_complexity(result_2d)
            julia_strength = ngtdm_strength(result_2d)

            # PyRadiomics needs 3D input
            py_features = get_pyradiomics_ngtdm(image_3d, mask_3d)

            # Test key features
            @test julia_coarseness ≈ py_features["Coarseness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_contrast ≈ py_features["Contrast"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_busyness ≈ py_features["Busyness"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_complexity ≈ py_features["Complexity"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
            @test julia_strength ≈ py_features["Strength"] rtol=NGTDM_RTOL atol=NGTDM_ATOL
        end
    end
end

#==============================================================================#
# Summary Report
#==============================================================================#

@testset "NGTDM Parity Summary" begin
    # Final comprehensive test with detailed reporting
    image, mask = random_image_mask(42, NGTDM_MEDIUM_SIZE)

    julia_features = get_julia_ngtdm_features(image, mask)
    py_features = get_pyradiomics_ngtdm(image, mask)

    # Count results
    n_tested = 0
    n_passed = 0
    n_missing = 0
    failures = []

    for (julia_sym, py_name) in NGTDM_FEATURE_MAP
        n_tested += 1
        if haskey(py_features, py_name)
            julia_val = getfield(julia_features, julia_sym)
            py_val = py_features[py_name]

            if isapprox(julia_val, py_val; rtol=NGTDM_RTOL, atol=NGTDM_ATOL)
                n_passed += 1
            else
                push!(failures, (name=py_name, julia=julia_val, python=py_val))
            end
        else
            n_missing += 1
            @warn "Feature missing from PyRadiomics results" feature=py_name
        end
    end

    # Report
    @info "NGTDM Parity Test Summary" n_tested=n_tested n_passed=n_passed n_missing=n_missing n_failed=length(failures)

    if !isempty(failures)
        for f in failures
            @warn "Failed: $(f.name)" julia=f.julia python=f.python
        end
    end

    # Final assertion - all features should pass
    @test n_passed == n_tested - n_missing
    @test length(failures) == 0
end
