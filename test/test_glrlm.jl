# GLRLM Feature Parity Tests for Radiomics.jl
#
# This file tests all 16 GLRLM (Gray Level Run Length Matrix) features
# against PyRadiomics to verify 1:1 parity.
#
# Story: TEST-GLRLM-PARITY
#
# GLRLM Features (16 total):
# 1.  ShortRunEmphasis (SRE)              9.  RunVariance (RV)
# 2.  LongRunEmphasis (LRE)               10. RunEntropy (RE)
# 3.  GrayLevelNonUniformity (GLN)        11. LowGrayLevelRunEmphasis (LGLRE)
# 4.  GrayLevelNonUniformityNormalized    12. HighGrayLevelRunEmphasis (HGLRE)
# 5.  RunLengthNonUniformity (RLN)        13. ShortRunLowGrayLevelEmphasis
# 6.  RunLengthNonUniformityNormalized    14. ShortRunHighGrayLevelEmphasis
# 7.  RunPercentage (RP)                  15. LongRunLowGrayLevelEmphasis
# 8.  GrayLevelVariance (GLV)             16. LongRunHighGrayLevelEmphasis
#
# Tolerance Guidelines:
# - GLRLM Features: rtol=1e-10, atol=1e-12 (standard texture tolerance)
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

# Standard tolerance for GLRLM features
const GLRLM_RTOL = 1e-10
const GLRLM_ATOL = 1e-12

# Random seeds for reproducibility
const TEST_SEEDS = [42, 123, 456]

# Array sizes to test
const SMALL_SIZE = (16, 16, 16)
const MEDIUM_SIZE = (32, 32, 32)

#==============================================================================#
# PyRadiomics Feature Name Mapping
#==============================================================================#

# Map Julia feature function names to PyRadiomics feature names
const GLRLM_FEATURE_MAP = Dict(
    :short_run_emphasis => "ShortRunEmphasis",
    :long_run_emphasis => "LongRunEmphasis",
    :gray_level_non_uniformity => "GrayLevelNonUniformity",
    :gray_level_non_uniformity_normalized => "GrayLevelNonUniformityNormalized",
    :run_length_non_uniformity => "RunLengthNonUniformity",
    :run_length_non_uniformity_normalized => "RunLengthNonUniformityNormalized",
    :run_percentage => "RunPercentage",
    :gray_level_variance => "GrayLevelVariance",
    :run_variance => "RunVariance",
    :run_entropy => "RunEntropy",
    :low_gray_level_run_emphasis => "LowGrayLevelRunEmphasis",
    :high_gray_level_run_emphasis => "HighGrayLevelRunEmphasis",
    :short_run_low_gray_level_emphasis => "ShortRunLowGrayLevelEmphasis",
    :short_run_high_gray_level_emphasis => "ShortRunHighGrayLevelEmphasis",
    :long_run_low_gray_level_emphasis => "LongRunLowGrayLevelEmphasis",
    :long_run_high_gray_level_emphasis => "LongRunHighGrayLevelEmphasis"
)

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    get_pyradiomics_glrlm(image, mask; binwidth=25.0)

Extract all GLRLM features from PyRadiomics for comparison.
Returns a Dict with feature names as keys and values as Float64.
"""
function get_pyradiomics_glrlm(image::AbstractArray, mask::AbstractArray;
                                binwidth::Real=25.0)
    py = get_python_modules()
    radiomics = py.radiomics
    np = py.numpy

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask)

    # Get the GLRLM feature class directly
    glrlm_module = pyimport("radiomics.glrlm")
    RadiomicsGLRLM = glrlm_module.RadiomicsGLRLM

    # Instantiate feature extractor with settings
    extractor = RadiomicsGLRLM(sitk_image, sitk_mask;
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
    get_julia_glrlm_features(image, mask; binwidth=25.0)

Extract all GLRLM features from Julia for comparison.
Returns a NamedTuple with all 16 GLRLM features.
"""
function get_julia_glrlm_features(image::AbstractArray, mask::AbstractArray;
                                   binwidth::Real=25.0)
    # Compute GLRLM
    result = compute_glrlm(image, mask; binwidth=binwidth)

    # Extract all features
    return (
        short_run_emphasis = glrlm_short_run_emphasis(result),
        long_run_emphasis = glrlm_long_run_emphasis(result),
        gray_level_non_uniformity = glrlm_gray_level_non_uniformity(result),
        gray_level_non_uniformity_normalized = glrlm_gray_level_non_uniformity_normalized(result),
        run_length_non_uniformity = glrlm_run_length_non_uniformity(result),
        run_length_non_uniformity_normalized = glrlm_run_length_non_uniformity_normalized(result),
        run_percentage = glrlm_run_percentage(result),
        gray_level_variance = glrlm_gray_level_variance(result),
        run_variance = glrlm_run_variance(result),
        run_entropy = glrlm_run_entropy(result),
        low_gray_level_run_emphasis = glrlm_low_gray_level_run_emphasis(result),
        high_gray_level_run_emphasis = glrlm_high_gray_level_run_emphasis(result),
        short_run_low_gray_level_emphasis = glrlm_short_run_low_gray_level_emphasis(result),
        short_run_high_gray_level_emphasis = glrlm_short_run_high_gray_level_emphasis(result),
        long_run_low_gray_level_emphasis = glrlm_long_run_low_gray_level_emphasis(result),
        long_run_high_gray_level_emphasis = glrlm_long_run_high_gray_level_emphasis(result),
    )
end

#==============================================================================#
# Individual Feature Tests
#==============================================================================#

@testset "GLRLM Feature Parity" begin

    @testset "Test Environment Setup" begin
        @test verify_pyradiomics_available()
    end

    #--------------------------------------------------------------------------
    # Feature 1: ShortRunEmphasis
    #--------------------------------------------------------------------------
    @testset "ShortRunEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "ShortRunEmphasis")
                @test julia_features.short_run_emphasis ≈ py_features["ShortRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 2: LongRunEmphasis
    #--------------------------------------------------------------------------
    @testset "LongRunEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "LongRunEmphasis")
                @test julia_features.long_run_emphasis ≈ py_features["LongRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 3: GrayLevelNonUniformity
    #--------------------------------------------------------------------------
    @testset "GrayLevelNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "GrayLevelNonUniformity")
                @test julia_features.gray_level_non_uniformity ≈ py_features["GrayLevelNonUniformity"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 4: GrayLevelNonUniformityNormalized
    #--------------------------------------------------------------------------
    @testset "GrayLevelNonUniformityNormalized" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "GrayLevelNonUniformityNormalized")
                @test julia_features.gray_level_non_uniformity_normalized ≈ py_features["GrayLevelNonUniformityNormalized"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 5: RunLengthNonUniformity
    #--------------------------------------------------------------------------
    @testset "RunLengthNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "RunLengthNonUniformity")
                @test julia_features.run_length_non_uniformity ≈ py_features["RunLengthNonUniformity"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 6: RunLengthNonUniformityNormalized
    #--------------------------------------------------------------------------
    @testset "RunLengthNonUniformityNormalized" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "RunLengthNonUniformityNormalized")
                @test julia_features.run_length_non_uniformity_normalized ≈ py_features["RunLengthNonUniformityNormalized"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 7: RunPercentage
    #--------------------------------------------------------------------------
    @testset "RunPercentage" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "RunPercentage")
                @test julia_features.run_percentage ≈ py_features["RunPercentage"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 8: GrayLevelVariance
    #--------------------------------------------------------------------------
    @testset "GrayLevelVariance" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "GrayLevelVariance")
                @test julia_features.gray_level_variance ≈ py_features["GrayLevelVariance"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 9: RunVariance
    #--------------------------------------------------------------------------
    @testset "RunVariance" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "RunVariance")
                @test julia_features.run_variance ≈ py_features["RunVariance"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 10: RunEntropy
    #--------------------------------------------------------------------------
    @testset "RunEntropy" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "RunEntropy")
                @test julia_features.run_entropy ≈ py_features["RunEntropy"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 11: LowGrayLevelRunEmphasis
    #--------------------------------------------------------------------------
    @testset "LowGrayLevelRunEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "LowGrayLevelRunEmphasis")
                @test julia_features.low_gray_level_run_emphasis ≈ py_features["LowGrayLevelRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 12: HighGrayLevelRunEmphasis
    #--------------------------------------------------------------------------
    @testset "HighGrayLevelRunEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "HighGrayLevelRunEmphasis")
                @test julia_features.high_gray_level_run_emphasis ≈ py_features["HighGrayLevelRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 13: ShortRunLowGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "ShortRunLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "ShortRunLowGrayLevelEmphasis")
                @test julia_features.short_run_low_gray_level_emphasis ≈ py_features["ShortRunLowGrayLevelEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 14: ShortRunHighGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "ShortRunHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "ShortRunHighGrayLevelEmphasis")
                @test julia_features.short_run_high_gray_level_emphasis ≈ py_features["ShortRunHighGrayLevelEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 15: LongRunLowGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "LongRunLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "LongRunLowGrayLevelEmphasis")
                @test julia_features.long_run_low_gray_level_emphasis ≈ py_features["LongRunLowGrayLevelEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 16: LongRunHighGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "LongRunHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)

                julia_features = get_julia_glrlm_features(image, mask)
                py_features = get_pyradiomics_glrlm(image, mask)

                @test haskey(py_features, "LongRunHighGrayLevelEmphasis")
                @test julia_features.long_run_high_gray_level_emphasis ≈ py_features["LongRunHighGrayLevelEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end
end

#==============================================================================#
# Comprehensive Parity Test - All Features at Once
#==============================================================================#

@testset "Comprehensive GLRLM Parity" begin

    @testset "All features - Multiple seeds" begin
        for seed in TEST_SEEDS
            image, mask = random_image_mask(seed, MEDIUM_SIZE)

            julia_features = get_julia_glrlm_features(image, mask)
            py_features = get_pyradiomics_glrlm(image, mask)

            # Test each feature
            for (julia_sym, py_name) in GLRLM_FEATURE_MAP
                if haskey(py_features, py_name)
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    @test julia_val ≈ py_val rtol=GLRLM_RTOL atol=GLRLM_ATOL
                end
            end
        end
    end

    @testset "All features - Different sizes" begin
        for sz in [SMALL_SIZE, MEDIUM_SIZE]
            image, mask = random_image_mask(42, sz)

            julia_features = get_julia_glrlm_features(image, mask)
            py_features = get_pyradiomics_glrlm(image, mask)

            # Count passing features
            n_features = 0
            n_passed = 0

            for (julia_sym, py_name) in GLRLM_FEATURE_MAP
                if haskey(py_features, py_name)
                    n_features += 1
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    if isapprox(julia_val, py_val; rtol=GLRLM_RTOL, atol=GLRLM_ATOL)
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
        image, mask = random_image_mask(42, MEDIUM_SIZE)

        for binwidth in [16.0, 25.0, 32.0, 64.0]
            julia_features = get_julia_glrlm_features(image, mask; binwidth=binwidth)
            py_features = get_pyradiomics_glrlm(image, mask; binwidth=binwidth)

            # Test key features
            @test julia_features.short_run_emphasis ≈ py_features["ShortRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            @test julia_features.long_run_emphasis ≈ py_features["LongRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            @test julia_features.run_entropy ≈ py_features["RunEntropy"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
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

        julia_features = get_julia_glrlm_features(image, mask)
        py_features = get_pyradiomics_glrlm(image, mask)

        # Test key features
        for feature_sym in [:short_run_emphasis, :long_run_emphasis, :run_entropy, :run_percentage]
            py_name = GLRLM_FEATURE_MAP[feature_sym]
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, feature_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLRLM_RTOL atol=GLRLM_ATOL
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

        julia_features = get_julia_glrlm_features(image, mask)
        py_features = get_pyradiomics_glrlm(image, mask)

        # Test key features
        @test julia_features.short_run_emphasis ≈ py_features["ShortRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
        @test julia_features.run_entropy ≈ py_features["RunEntropy"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
    end

    @testset "Integer image values" begin
        # Discrete integer values
        image_int, mask = random_image_mask_integer(42, MEDIUM_SIZE; intensity_range=(0, 255))
        image = Float64.(image_int)

        julia_features = get_julia_glrlm_features(image, mask)
        py_features = get_pyradiomics_glrlm(image, mask)

        # All features should match
        for (julia_sym, py_name) in GLRLM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLRLM_RTOL atol=GLRLM_ATOL
            end
        end
    end
end

#==============================================================================#
# Feature Consistency Tests (Julia internal)
#==============================================================================#

@testset "Feature Consistency" begin

    @testset "GLRLM mathematical relationships" begin
        image, mask = random_image_mask(42, MEDIUM_SIZE)
        features = get_julia_glrlm_features(image, mask)

        # ShortRunEmphasis should be > 0 and typically < 1
        @test features.short_run_emphasis > 0.0

        # LongRunEmphasis should be > 0
        @test features.long_run_emphasis > 0.0

        # Run percentage should be in (0, 1]
        @test 0.0 < features.run_percentage <= 1.0

        # NonUniformity features should be non-negative
        @test features.gray_level_non_uniformity >= 0.0
        @test features.run_length_non_uniformity >= 0.0

        # Normalized NonUniformity features should be in [0, 1]
        @test 0.0 <= features.gray_level_non_uniformity_normalized <= 1.0
        @test 0.0 <= features.run_length_non_uniformity_normalized <= 1.0

        # Variance features should be non-negative
        @test features.gray_level_variance >= 0.0
        @test features.run_variance >= 0.0

        # Entropy should be non-negative
        @test features.run_entropy >= 0.0

        # Gray level emphasis features should be > 0
        @test features.low_gray_level_run_emphasis > 0.0
        @test features.high_gray_level_run_emphasis > 0.0
    end

    @testset "Feature bounds" begin
        for seed in TEST_SEEDS
            image, mask = random_image_mask(seed, MEDIUM_SIZE)
            features = get_julia_glrlm_features(image, mask)

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
    @testset "2D GLRLM features" begin
        for seed in TEST_SEEDS
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
            result_2d = compute_glrlm(image, mask; binwidth=25.0)
            julia_sre = glrlm_short_run_emphasis(result_2d)
            julia_lre = glrlm_long_run_emphasis(result_2d)
            julia_re = glrlm_run_entropy(result_2d)

            # PyRadiomics needs 3D input
            py_features = get_pyradiomics_glrlm(image_3d, mask_3d)

            # Test key features
            @test julia_sre ≈ py_features["ShortRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            @test julia_lre ≈ py_features["LongRunEmphasis"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
            @test julia_re ≈ py_features["RunEntropy"] rtol=GLRLM_RTOL atol=GLRLM_ATOL
        end
    end
end

#==============================================================================#
# Summary Report
#==============================================================================#

@testset "GLRLM Parity Summary" begin
    # Final comprehensive test with detailed reporting
    image, mask = random_image_mask(42, MEDIUM_SIZE)

    julia_features = get_julia_glrlm_features(image, mask)
    py_features = get_pyradiomics_glrlm(image, mask)

    # Count results
    n_tested = 0
    n_passed = 0
    n_missing = 0
    failures = []

    for (julia_sym, py_name) in GLRLM_FEATURE_MAP
        n_tested += 1
        if haskey(py_features, py_name)
            julia_val = getfield(julia_features, julia_sym)
            py_val = py_features[py_name]

            if isapprox(julia_val, py_val; rtol=GLRLM_RTOL, atol=GLRLM_ATOL)
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
    @info "GLRLM Parity Test Summary" n_tested=n_tested n_passed=n_passed n_missing=n_missing n_failed=length(failures)

    if !isempty(failures)
        for f in failures
            @warn "Failed: $(f.name)" julia=f.julia python=f.python
        end
    end

    # Final assertion - all features should pass
    @test n_passed == n_tested - n_missing
    @test length(failures) == 0
end
