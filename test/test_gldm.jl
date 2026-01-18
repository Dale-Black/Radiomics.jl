# GLDM Feature Parity Tests for Radiomics.jl
#
# This file tests all 14 GLDM (Gray Level Dependence Matrix) features
# against PyRadiomics to verify 1:1 parity.
#
# Story: TEST-GLDM-PARITY
#
# GLDM Features (14 total):
# 1. SmallDependenceEmphasis      - Measures distribution of small dependencies
# 2. LargeDependenceEmphasis      - Measures distribution of large dependencies
# 3. GrayLevelNonUniformity       - Measures variability of gray level intensities
# 4. DependenceNonUniformity      - Measures variability of dependence sizes
# 5. DependenceNonUniformityNormalized - Normalized dependence non-uniformity
# 6. GrayLevelVariance            - Variance in gray level intensities
# 7. DependenceVariance           - Variance in dependence sizes
# 8. DependenceEntropy            - Entropy of dependence distribution
# 9. LowGrayLevelEmphasis         - Distribution of lower gray levels
# 10. HighGrayLevelEmphasis       - Distribution of higher gray levels
# 11. SmallDependenceLowGrayLevelEmphasis  - Joint small dependence + low gray
# 12. SmallDependenceHighGrayLevelEmphasis - Joint small dependence + high gray
# 13. LargeDependenceLowGrayLevelEmphasis  - Joint large dependence + low gray
# 14. LargeDependenceHighGrayLevelEmphasis - Joint large dependence + high gray
#
# Tolerance Guidelines:
# - GLDM Features: rtol=1e-10, atol=1e-12 (standard texture tolerance)
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

# Standard tolerance for GLDM features
const GLDM_RTOL = 1e-10
const GLDM_ATOL = 1e-12

# Random seeds for reproducibility
const GLDM_TEST_SEEDS = [42, 123, 456]

# Array sizes to test
const GLDM_SMALL_SIZE = (16, 16, 16)
const GLDM_MEDIUM_SIZE = (32, 32, 32)

#==============================================================================#
# PyRadiomics Feature Name Mapping
#==============================================================================#

# Map Julia feature function names to PyRadiomics feature names
const GLDM_FEATURE_MAP = Dict(
    :small_dependence_emphasis => "SmallDependenceEmphasis",
    :large_dependence_emphasis => "LargeDependenceEmphasis",
    :gray_level_non_uniformity => "GrayLevelNonUniformity",
    :dependence_non_uniformity => "DependenceNonUniformity",
    :dependence_non_uniformity_normalized => "DependenceNonUniformityNormalized",
    :gray_level_variance => "GrayLevelVariance",
    :dependence_variance => "DependenceVariance",
    :dependence_entropy => "DependenceEntropy",
    :low_gray_level_emphasis => "LowGrayLevelEmphasis",
    :high_gray_level_emphasis => "HighGrayLevelEmphasis",
    :small_dependence_low_gray_level_emphasis => "SmallDependenceLowGrayLevelEmphasis",
    :small_dependence_high_gray_level_emphasis => "SmallDependenceHighGrayLevelEmphasis",
    :large_dependence_low_gray_level_emphasis => "LargeDependenceLowGrayLevelEmphasis",
    :large_dependence_high_gray_level_emphasis => "LargeDependenceHighGrayLevelEmphasis"
)

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    get_pyradiomics_gldm(image, mask; binwidth=25.0, gldm_a=0)

Extract all GLDM features from PyRadiomics for comparison.
Returns a Dict with feature names as keys and values as Float64.
"""
function get_pyradiomics_gldm(image::AbstractArray, mask::AbstractArray;
                               binwidth::Real=25.0, gldm_a::Int=0)
    py = get_python_modules()
    radiomics = py.radiomics
    np = py.numpy

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask)

    # Get the GLDM feature class directly
    gldm_module = pyimport("radiomics.gldm")
    RadiomicsGLDM = gldm_module.RadiomicsGLDM

    # Instantiate feature extractor with settings
    # gldm_a is the alpha (coarseness) parameter
    extractor = RadiomicsGLDM(sitk_image, sitk_mask;
                               label=1,
                               binWidth=binwidth,
                               gldm_a=gldm_a)

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
    get_julia_gldm_features(image, mask; binwidth=25.0, alpha=0)

Extract all GLDM features from Julia for comparison.
Returns a NamedTuple with all 14 GLDM features.
"""
function get_julia_gldm_features(image::AbstractArray, mask::AbstractArray;
                                  binwidth::Real=25.0, alpha::Int=0)
    # Compute GLDM
    result = compute_gldm(image, mask; binwidth=binwidth, alpha=alpha)

    # Extract all features
    return (
        small_dependence_emphasis = gldm_small_dependence_emphasis(result),
        large_dependence_emphasis = gldm_large_dependence_emphasis(result),
        gray_level_non_uniformity = gldm_gray_level_non_uniformity(result),
        dependence_non_uniformity = gldm_dependence_non_uniformity(result),
        dependence_non_uniformity_normalized = gldm_dependence_non_uniformity_normalized(result),
        gray_level_variance = gldm_gray_level_variance(result),
        dependence_variance = gldm_dependence_variance(result),
        dependence_entropy = gldm_dependence_entropy(result),
        low_gray_level_emphasis = gldm_low_gray_level_emphasis(result),
        high_gray_level_emphasis = gldm_high_gray_level_emphasis(result),
        small_dependence_low_gray_level_emphasis = gldm_small_dependence_low_gray_level_emphasis(result),
        small_dependence_high_gray_level_emphasis = gldm_small_dependence_high_gray_level_emphasis(result),
        large_dependence_low_gray_level_emphasis = gldm_large_dependence_low_gray_level_emphasis(result),
        large_dependence_high_gray_level_emphasis = gldm_large_dependence_high_gray_level_emphasis(result)
    )
end

#==============================================================================#
# Individual Feature Tests
#==============================================================================#

@testset "GLDM Feature Parity" begin

    @testset "Test Environment Setup" begin
        @test verify_pyradiomics_available()
    end

    #--------------------------------------------------------------------------
    # Feature 1: Small Dependence Emphasis
    #--------------------------------------------------------------------------
    @testset "SmallDependenceEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "SmallDependenceEmphasis")
                @test julia_features.small_dependence_emphasis ≈ py_features["SmallDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 2: Large Dependence Emphasis
    #--------------------------------------------------------------------------
    @testset "LargeDependenceEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "LargeDependenceEmphasis")
                @test julia_features.large_dependence_emphasis ≈ py_features["LargeDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 3: Gray Level Non-Uniformity
    #--------------------------------------------------------------------------
    @testset "GrayLevelNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "GrayLevelNonUniformity")
                @test julia_features.gray_level_non_uniformity ≈ py_features["GrayLevelNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 4: Dependence Non-Uniformity
    #--------------------------------------------------------------------------
    @testset "DependenceNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "DependenceNonUniformity")
                @test julia_features.dependence_non_uniformity ≈ py_features["DependenceNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 5: Dependence Non-Uniformity Normalized
    #--------------------------------------------------------------------------
    @testset "DependenceNonUniformityNormalized" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "DependenceNonUniformityNormalized")
                @test julia_features.dependence_non_uniformity_normalized ≈ py_features["DependenceNonUniformityNormalized"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 6: Gray Level Variance
    #--------------------------------------------------------------------------
    @testset "GrayLevelVariance" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "GrayLevelVariance")
                @test julia_features.gray_level_variance ≈ py_features["GrayLevelVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 7: Dependence Variance
    #--------------------------------------------------------------------------
    @testset "DependenceVariance" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "DependenceVariance")
                @test julia_features.dependence_variance ≈ py_features["DependenceVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 8: Dependence Entropy
    #--------------------------------------------------------------------------
    @testset "DependenceEntropy" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "DependenceEntropy")
                @test julia_features.dependence_entropy ≈ py_features["DependenceEntropy"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 9: Low Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "LowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "LowGrayLevelEmphasis")
                @test julia_features.low_gray_level_emphasis ≈ py_features["LowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 10: High Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "HighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "HighGrayLevelEmphasis")
                @test julia_features.high_gray_level_emphasis ≈ py_features["HighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 11: Small Dependence Low Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "SmallDependenceLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "SmallDependenceLowGrayLevelEmphasis")
                @test julia_features.small_dependence_low_gray_level_emphasis ≈ py_features["SmallDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 12: Small Dependence High Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "SmallDependenceHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "SmallDependenceHighGrayLevelEmphasis")
                @test julia_features.small_dependence_high_gray_level_emphasis ≈ py_features["SmallDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 13: Large Dependence Low Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "LargeDependenceLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "LargeDependenceLowGrayLevelEmphasis")
                @test julia_features.large_dependence_low_gray_level_emphasis ≈ py_features["LargeDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 14: Large Dependence High Gray Level Emphasis
    #--------------------------------------------------------------------------
    @testset "LargeDependenceHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLDM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

                julia_features = get_julia_gldm_features(image, mask)
                py_features = get_pyradiomics_gldm(image, mask)

                @test haskey(py_features, "LargeDependenceHighGrayLevelEmphasis")
                @test julia_features.large_dependence_high_gray_level_emphasis ≈ py_features["LargeDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end
end

#==============================================================================#
# Comprehensive Parity Test - All Features at Once
#==============================================================================#

@testset "Comprehensive GLDM Parity" begin

    @testset "All features - Multiple seeds" begin
        for seed in GLDM_TEST_SEEDS
            image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)

            julia_features = get_julia_gldm_features(image, mask)
            py_features = get_pyradiomics_gldm(image, mask)

            # Test each feature
            for (julia_sym, py_name) in GLDM_FEATURE_MAP
                if haskey(py_features, py_name)
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    @test julia_val ≈ py_val rtol=GLDM_RTOL atol=GLDM_ATOL
                end
            end
        end
    end

    @testset "All features - Different sizes" begin
        for sz in [GLDM_SMALL_SIZE, GLDM_MEDIUM_SIZE]
            image, mask = random_image_mask(42, sz)

            julia_features = get_julia_gldm_features(image, mask)
            py_features = get_pyradiomics_gldm(image, mask)

            # Count passing features
            n_features = 0
            n_passed = 0

            for (julia_sym, py_name) in GLDM_FEATURE_MAP
                if haskey(py_features, py_name)
                    n_features += 1
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    if isapprox(julia_val, py_val; rtol=GLDM_RTOL, atol=GLDM_ATOL)
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
        image, mask = random_image_mask(42, GLDM_MEDIUM_SIZE)

        for binwidth in [16.0, 25.0, 32.0, 64.0]
            julia_features = get_julia_gldm_features(image, mask; binwidth=binwidth)
            py_features = get_pyradiomics_gldm(image, mask; binwidth=binwidth)

            # Test all features
            @test julia_features.small_dependence_emphasis ≈ py_features["SmallDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.large_dependence_emphasis ≈ py_features["LargeDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.gray_level_non_uniformity ≈ py_features["GrayLevelNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.dependence_non_uniformity ≈ py_features["DependenceNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.dependence_non_uniformity_normalized ≈ py_features["DependenceNonUniformityNormalized"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.gray_level_variance ≈ py_features["GrayLevelVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.dependence_variance ≈ py_features["DependenceVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.dependence_entropy ≈ py_features["DependenceEntropy"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.low_gray_level_emphasis ≈ py_features["LowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.high_gray_level_emphasis ≈ py_features["HighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.small_dependence_low_gray_level_emphasis ≈ py_features["SmallDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.small_dependence_high_gray_level_emphasis ≈ py_features["SmallDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.large_dependence_low_gray_level_emphasis ≈ py_features["LargeDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_features.large_dependence_high_gray_level_emphasis ≈ py_features["LargeDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
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

        julia_features = get_julia_gldm_features(image, mask)
        py_features = get_pyradiomics_gldm(image, mask)

        # Test all features
        for (julia_sym, py_name) in GLDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLDM_RTOL atol=GLDM_ATOL
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

        julia_features = get_julia_gldm_features(image, mask)
        py_features = get_pyradiomics_gldm(image, mask)

        # Test all features
        for (julia_sym, py_name) in GLDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end

    @testset "Integer image values" begin
        # Discrete integer values
        image_int, mask = random_image_mask_integer(42, GLDM_MEDIUM_SIZE; intensity_range=(0, 255))
        image = Float64.(image_int)

        julia_features = get_julia_gldm_features(image, mask)
        py_features = get_pyradiomics_gldm(image, mask)

        # All features should match
        for (julia_sym, py_name) in GLDM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLDM_RTOL atol=GLDM_ATOL
            end
        end
    end
end

#==============================================================================#
# Feature Consistency Tests (Julia internal)
#==============================================================================#

@testset "Feature Consistency" begin

    @testset "GLDM mathematical relationships" begin
        image, mask = random_image_mask(42, GLDM_MEDIUM_SIZE)
        features = get_julia_gldm_features(image, mask)

        # Small dependence emphasis should be > 0
        @test features.small_dependence_emphasis > 0.0

        # Large dependence emphasis should be >= 0
        @test features.large_dependence_emphasis >= 0.0

        # Non-uniformity values should be >= 0
        @test features.gray_level_non_uniformity >= 0.0
        @test features.dependence_non_uniformity >= 0.0
        @test features.dependence_non_uniformity_normalized >= 0.0

        # Variance should be >= 0
        @test features.gray_level_variance >= 0.0
        @test features.dependence_variance >= 0.0

        # Entropy should be >= 0
        @test features.dependence_entropy >= 0.0

        # Gray level emphasis features should be >= 0
        @test features.low_gray_level_emphasis >= 0.0
        @test features.high_gray_level_emphasis >= 0.0

        # Combined features should be >= 0
        @test features.small_dependence_low_gray_level_emphasis >= 0.0
        @test features.small_dependence_high_gray_level_emphasis >= 0.0
        @test features.large_dependence_low_gray_level_emphasis >= 0.0
        @test features.large_dependence_high_gray_level_emphasis >= 0.0
    end

    @testset "Feature bounds" begin
        for seed in GLDM_TEST_SEEDS
            image, mask = random_image_mask(seed, GLDM_MEDIUM_SIZE)
            features = get_julia_gldm_features(image, mask)

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
    @testset "2D GLDM features" begin
        for seed in GLDM_TEST_SEEDS
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
            result_2d = compute_gldm(image, mask; binwidth=25.0)
            julia_sde = gldm_small_dependence_emphasis(result_2d)
            julia_lde = gldm_large_dependence_emphasis(result_2d)
            julia_gln = gldm_gray_level_non_uniformity(result_2d)
            julia_dn = gldm_dependence_non_uniformity(result_2d)
            julia_dnn = gldm_dependence_non_uniformity_normalized(result_2d)
            julia_glv = gldm_gray_level_variance(result_2d)
            julia_dv = gldm_dependence_variance(result_2d)
            julia_de = gldm_dependence_entropy(result_2d)
            julia_lgle = gldm_low_gray_level_emphasis(result_2d)
            julia_hgle = gldm_high_gray_level_emphasis(result_2d)
            julia_sdlgle = gldm_small_dependence_low_gray_level_emphasis(result_2d)
            julia_sdhgle = gldm_small_dependence_high_gray_level_emphasis(result_2d)
            julia_ldlgle = gldm_large_dependence_low_gray_level_emphasis(result_2d)
            julia_ldhgle = gldm_large_dependence_high_gray_level_emphasis(result_2d)

            # PyRadiomics needs 3D input
            py_features = get_pyradiomics_gldm(image_3d, mask_3d)

            # Test key features
            @test julia_sde ≈ py_features["SmallDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_lde ≈ py_features["LargeDependenceEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_gln ≈ py_features["GrayLevelNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_dn ≈ py_features["DependenceNonUniformity"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_dnn ≈ py_features["DependenceNonUniformityNormalized"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_glv ≈ py_features["GrayLevelVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_dv ≈ py_features["DependenceVariance"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_de ≈ py_features["DependenceEntropy"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_lgle ≈ py_features["LowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_hgle ≈ py_features["HighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_sdlgle ≈ py_features["SmallDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_sdhgle ≈ py_features["SmallDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_ldlgle ≈ py_features["LargeDependenceLowGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
            @test julia_ldhgle ≈ py_features["LargeDependenceHighGrayLevelEmphasis"] rtol=GLDM_RTOL atol=GLDM_ATOL
        end
    end
end

#==============================================================================#
# Summary Report
#==============================================================================#

@testset "GLDM Parity Summary" begin
    # Final comprehensive test with detailed reporting
    image, mask = random_image_mask(42, GLDM_MEDIUM_SIZE)

    julia_features = get_julia_gldm_features(image, mask)
    py_features = get_pyradiomics_gldm(image, mask)

    # Count results
    n_tested = 0
    n_passed = 0
    n_missing = 0
    failures = []

    for (julia_sym, py_name) in GLDM_FEATURE_MAP
        n_tested += 1
        if haskey(py_features, py_name)
            julia_val = getfield(julia_features, julia_sym)
            py_val = py_features[py_name]

            if isapprox(julia_val, py_val; rtol=GLDM_RTOL, atol=GLDM_ATOL)
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
    @info "GLDM Parity Test Summary" n_tested=n_tested n_passed=n_passed n_missing=n_missing n_failed=length(failures)

    if !isempty(failures)
        for f in failures
            @warn "Failed: $(f.name)" julia=f.julia python=f.python
        end
    end

    # Final assertion - all features should pass
    @test n_passed == n_tested - n_missing
    @test length(failures) == 0
end
