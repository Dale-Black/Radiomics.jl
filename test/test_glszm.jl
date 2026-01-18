# GLSZM Feature Parity Tests for Radiomics.jl
#
# This file tests all 16 GLSZM (Gray Level Size Zone Matrix) features
# against PyRadiomics to verify 1:1 parity.
#
# Story: TEST-GLSZM-PARITY
#
# GLSZM Features (16 total):
# 1.  SmallAreaEmphasis (SAE)              9.  ZoneVariance (ZV)
# 2.  LargeAreaEmphasis (LAE)              10. ZoneEntropy (ZE)
# 3.  GrayLevelNonUniformity (GLN)         11. LowGrayLevelZoneEmphasis (LGLZE)
# 4.  GrayLevelNonUniformityNormalized     12. HighGrayLevelZoneEmphasis (HGLZE)
# 5.  SizeZoneNonUniformity (SZN)          13. SmallAreaLowGrayLevelEmphasis
# 6.  SizeZoneNonUniformityNormalized      14. SmallAreaHighGrayLevelEmphasis
# 7.  ZonePercentage (ZP)                  15. LargeAreaLowGrayLevelEmphasis
# 8.  GrayLevelVariance (GLV)              16. LargeAreaHighGrayLevelEmphasis
#
# Tolerance Guidelines:
# - GLSZM Features: rtol=1e-10, atol=1e-12 (standard texture tolerance)
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

# Standard tolerance for GLSZM features
const GLSZM_RTOL = 1e-10
const GLSZM_ATOL = 1e-12

# Random seeds for reproducibility
const GLSZM_TEST_SEEDS = [42, 123, 456]

# Array sizes to test
const GLSZM_SMALL_SIZE = (16, 16, 16)
const GLSZM_MEDIUM_SIZE = (32, 32, 32)

#==============================================================================#
# PyRadiomics Feature Name Mapping
#==============================================================================#

# Map Julia feature function names to PyRadiomics feature names
const GLSZM_FEATURE_MAP = Dict(
    :small_area_emphasis => "SmallAreaEmphasis",
    :large_area_emphasis => "LargeAreaEmphasis",
    :gray_level_non_uniformity => "GrayLevelNonUniformity",
    :gray_level_non_uniformity_normalized => "GrayLevelNonUniformityNormalized",
    :size_zone_non_uniformity => "SizeZoneNonUniformity",
    :size_zone_non_uniformity_normalized => "SizeZoneNonUniformityNormalized",
    :zone_percentage => "ZonePercentage",
    :gray_level_variance => "GrayLevelVariance",
    :zone_variance => "ZoneVariance",
    :zone_entropy => "ZoneEntropy",
    :low_gray_level_zone_emphasis => "LowGrayLevelZoneEmphasis",
    :high_gray_level_zone_emphasis => "HighGrayLevelZoneEmphasis",
    :small_area_low_gray_level_emphasis => "SmallAreaLowGrayLevelEmphasis",
    :small_area_high_gray_level_emphasis => "SmallAreaHighGrayLevelEmphasis",
    :large_area_low_gray_level_emphasis => "LargeAreaLowGrayLevelEmphasis",
    :large_area_high_gray_level_emphasis => "LargeAreaHighGrayLevelEmphasis"
)

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    get_pyradiomics_glszm(image, mask; binwidth=25.0)

Extract all GLSZM features from PyRadiomics for comparison.
Returns a Dict with feature names as keys and values as Float64.
"""
function get_pyradiomics_glszm(image::AbstractArray, mask::AbstractArray;
                                binwidth::Real=25.0)
    py = get_python_modules()
    radiomics = py.radiomics
    np = py.numpy

    # Convert to SimpleITK format
    sitk_image, sitk_mask = julia_array_to_sitk(image, mask)

    # Get the GLSZM feature class directly
    glszm_module = pyimport("radiomics.glszm")
    RadiomicsGLSZM = glszm_module.RadiomicsGLSZM

    # Instantiate feature extractor with settings
    extractor = RadiomicsGLSZM(sitk_image, sitk_mask;
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
    get_julia_glszm_features(image, mask; binwidth=25.0)

Extract all GLSZM features from Julia for comparison.
Returns a NamedTuple with all 16 GLSZM features.
"""
function get_julia_glszm_features(image::AbstractArray, mask::AbstractArray;
                                   binwidth::Real=25.0)
    # Compute GLSZM
    result = compute_glszm(image, mask; binwidth=binwidth)

    # Extract all features
    return (
        small_area_emphasis = glszm_small_area_emphasis(result),
        large_area_emphasis = glszm_large_area_emphasis(result),
        gray_level_non_uniformity = glszm_gray_level_non_uniformity(result),
        gray_level_non_uniformity_normalized = glszm_gray_level_non_uniformity_normalized(result),
        size_zone_non_uniformity = glszm_size_zone_non_uniformity(result),
        size_zone_non_uniformity_normalized = glszm_size_zone_non_uniformity_normalized(result),
        zone_percentage = glszm_zone_percentage(result),
        gray_level_variance = glszm_gray_level_variance(result),
        zone_variance = glszm_zone_variance(result),
        zone_entropy = glszm_zone_entropy(result),
        low_gray_level_zone_emphasis = glszm_low_gray_level_zone_emphasis(result),
        high_gray_level_zone_emphasis = glszm_high_gray_level_zone_emphasis(result),
        small_area_low_gray_level_emphasis = glszm_small_area_low_gray_level_emphasis(result),
        small_area_high_gray_level_emphasis = glszm_small_area_high_gray_level_emphasis(result),
        large_area_low_gray_level_emphasis = glszm_large_area_low_gray_level_emphasis(result),
        large_area_high_gray_level_emphasis = glszm_large_area_high_gray_level_emphasis(result),
    )
end

#==============================================================================#
# Individual Feature Tests
#==============================================================================#

@testset "GLSZM Feature Parity" begin

    @testset "Test Environment Setup" begin
        @test verify_pyradiomics_available()
    end

    #--------------------------------------------------------------------------
    # Feature 1: SmallAreaEmphasis
    #--------------------------------------------------------------------------
    @testset "SmallAreaEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "SmallAreaEmphasis")
                @test julia_features.small_area_emphasis ≈ py_features["SmallAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 2: LargeAreaEmphasis
    #--------------------------------------------------------------------------
    @testset "LargeAreaEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "LargeAreaEmphasis")
                @test julia_features.large_area_emphasis ≈ py_features["LargeAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 3: GrayLevelNonUniformity
    #--------------------------------------------------------------------------
    @testset "GrayLevelNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "GrayLevelNonUniformity")
                @test julia_features.gray_level_non_uniformity ≈ py_features["GrayLevelNonUniformity"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 4: GrayLevelNonUniformityNormalized
    #--------------------------------------------------------------------------
    @testset "GrayLevelNonUniformityNormalized" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "GrayLevelNonUniformityNormalized")
                @test julia_features.gray_level_non_uniformity_normalized ≈ py_features["GrayLevelNonUniformityNormalized"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 5: SizeZoneNonUniformity
    #--------------------------------------------------------------------------
    @testset "SizeZoneNonUniformity" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "SizeZoneNonUniformity")
                @test julia_features.size_zone_non_uniformity ≈ py_features["SizeZoneNonUniformity"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 6: SizeZoneNonUniformityNormalized
    #--------------------------------------------------------------------------
    @testset "SizeZoneNonUniformityNormalized" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "SizeZoneNonUniformityNormalized")
                @test julia_features.size_zone_non_uniformity_normalized ≈ py_features["SizeZoneNonUniformityNormalized"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 7: ZonePercentage
    #--------------------------------------------------------------------------
    @testset "ZonePercentage" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "ZonePercentage")
                @test julia_features.zone_percentage ≈ py_features["ZonePercentage"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 8: GrayLevelVariance
    #--------------------------------------------------------------------------
    @testset "GrayLevelVariance" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "GrayLevelVariance")
                @test julia_features.gray_level_variance ≈ py_features["GrayLevelVariance"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 9: ZoneVariance
    #--------------------------------------------------------------------------
    @testset "ZoneVariance" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "ZoneVariance")
                @test julia_features.zone_variance ≈ py_features["ZoneVariance"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 10: ZoneEntropy
    #--------------------------------------------------------------------------
    @testset "ZoneEntropy" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "ZoneEntropy")
                @test julia_features.zone_entropy ≈ py_features["ZoneEntropy"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 11: LowGrayLevelZoneEmphasis
    #--------------------------------------------------------------------------
    @testset "LowGrayLevelZoneEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "LowGrayLevelZoneEmphasis")
                @test julia_features.low_gray_level_zone_emphasis ≈ py_features["LowGrayLevelZoneEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 12: HighGrayLevelZoneEmphasis
    #--------------------------------------------------------------------------
    @testset "HighGrayLevelZoneEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "HighGrayLevelZoneEmphasis")
                @test julia_features.high_gray_level_zone_emphasis ≈ py_features["HighGrayLevelZoneEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 13: SmallAreaLowGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "SmallAreaLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "SmallAreaLowGrayLevelEmphasis")
                @test julia_features.small_area_low_gray_level_emphasis ≈ py_features["SmallAreaLowGrayLevelEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 14: SmallAreaHighGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "SmallAreaHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "SmallAreaHighGrayLevelEmphasis")
                @test julia_features.small_area_high_gray_level_emphasis ≈ py_features["SmallAreaHighGrayLevelEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 15: LargeAreaLowGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "LargeAreaLowGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "LargeAreaLowGrayLevelEmphasis")
                @test julia_features.large_area_low_gray_level_emphasis ≈ py_features["LargeAreaLowGrayLevelEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 16: LargeAreaHighGrayLevelEmphasis
    #--------------------------------------------------------------------------
    @testset "LargeAreaHighGrayLevelEmphasis" begin
        @testset "Multiple seeds" begin
            for seed in GLSZM_TEST_SEEDS
                image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

                julia_features = get_julia_glszm_features(image, mask)
                py_features = get_pyradiomics_glszm(image, mask)

                @test haskey(py_features, "LargeAreaHighGrayLevelEmphasis")
                @test julia_features.large_area_high_gray_level_emphasis ≈ py_features["LargeAreaHighGrayLevelEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end
end

#==============================================================================#
# Comprehensive Parity Test - All Features at Once
#==============================================================================#

@testset "Comprehensive GLSZM Parity" begin

    @testset "All features - Multiple seeds" begin
        for seed in GLSZM_TEST_SEEDS
            image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)

            julia_features = get_julia_glszm_features(image, mask)
            py_features = get_pyradiomics_glszm(image, mask)

            # Test each feature
            for (julia_sym, py_name) in GLSZM_FEATURE_MAP
                if haskey(py_features, py_name)
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    @test julia_val ≈ py_val rtol=GLSZM_RTOL atol=GLSZM_ATOL
                end
            end
        end
    end

    @testset "All features - Different sizes" begin
        for sz in [GLSZM_SMALL_SIZE, GLSZM_MEDIUM_SIZE]
            image, mask = random_image_mask(42, sz)

            julia_features = get_julia_glszm_features(image, mask)
            py_features = get_pyradiomics_glszm(image, mask)

            # Count passing features
            n_features = 0
            n_passed = 0

            for (julia_sym, py_name) in GLSZM_FEATURE_MAP
                if haskey(py_features, py_name)
                    n_features += 1
                    julia_val = getfield(julia_features, julia_sym)
                    py_val = py_features[py_name]

                    if isapprox(julia_val, py_val; rtol=GLSZM_RTOL, atol=GLSZM_ATOL)
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
        image, mask = random_image_mask(42, GLSZM_MEDIUM_SIZE)

        for binwidth in [16.0, 25.0, 32.0, 64.0]
            julia_features = get_julia_glszm_features(image, mask; binwidth=binwidth)
            py_features = get_pyradiomics_glszm(image, mask; binwidth=binwidth)

            # Test key features
            @test julia_features.small_area_emphasis ≈ py_features["SmallAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            @test julia_features.large_area_emphasis ≈ py_features["LargeAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            @test julia_features.zone_entropy ≈ py_features["ZoneEntropy"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
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

        julia_features = get_julia_glszm_features(image, mask)
        py_features = get_pyradiomics_glszm(image, mask)

        # Test key features
        for feature_sym in [:small_area_emphasis, :large_area_emphasis, :zone_entropy, :zone_percentage]
            py_name = GLSZM_FEATURE_MAP[feature_sym]
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, feature_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLSZM_RTOL atol=GLSZM_ATOL
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

        julia_features = get_julia_glszm_features(image, mask)
        py_features = get_pyradiomics_glszm(image, mask)

        # Test key features
        @test julia_features.small_area_emphasis ≈ py_features["SmallAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
        @test julia_features.zone_entropy ≈ py_features["ZoneEntropy"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
    end

    @testset "Integer image values" begin
        # Discrete integer values
        image_int, mask = random_image_mask_integer(42, GLSZM_MEDIUM_SIZE; intensity_range=(0, 255))
        image = Float64.(image_int)

        julia_features = get_julia_glszm_features(image, mask)
        py_features = get_pyradiomics_glszm(image, mask)

        # All features should match
        for (julia_sym, py_name) in GLSZM_FEATURE_MAP
            if haskey(py_features, py_name)
                julia_val = getfield(julia_features, julia_sym)
                py_val = py_features[py_name]

                @test julia_val ≈ py_val rtol=GLSZM_RTOL atol=GLSZM_ATOL
            end
        end
    end
end

#==============================================================================#
# Feature Consistency Tests (Julia internal)
#==============================================================================#

@testset "Feature Consistency" begin

    @testset "GLSZM mathematical relationships" begin
        image, mask = random_image_mask(42, GLSZM_MEDIUM_SIZE)
        features = get_julia_glszm_features(image, mask)

        # SmallAreaEmphasis should be > 0 and typically < 1
        @test features.small_area_emphasis > 0.0

        # LargeAreaEmphasis should be > 0
        @test features.large_area_emphasis > 0.0

        # Zone percentage should be in (0, 1]
        @test 0.0 < features.zone_percentage <= 1.0

        # NonUniformity features should be non-negative
        @test features.gray_level_non_uniformity >= 0.0
        @test features.size_zone_non_uniformity >= 0.0

        # Normalized NonUniformity features should be in [0, 1]
        @test 0.0 <= features.gray_level_non_uniformity_normalized <= 1.0
        @test 0.0 <= features.size_zone_non_uniformity_normalized <= 1.0

        # Variance features should be non-negative
        @test features.gray_level_variance >= 0.0
        @test features.zone_variance >= 0.0

        # Entropy should be non-negative
        @test features.zone_entropy >= 0.0

        # Gray level emphasis features should be > 0
        @test features.low_gray_level_zone_emphasis > 0.0
        @test features.high_gray_level_zone_emphasis > 0.0
    end

    @testset "Feature bounds" begin
        for seed in GLSZM_TEST_SEEDS
            image, mask = random_image_mask(seed, GLSZM_MEDIUM_SIZE)
            features = get_julia_glszm_features(image, mask)

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
    @testset "2D GLSZM features" begin
        for seed in GLSZM_TEST_SEEDS
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
            result_2d = compute_glszm(image, mask; binwidth=25.0)
            julia_sae = glszm_small_area_emphasis(result_2d)
            julia_lae = glszm_large_area_emphasis(result_2d)
            julia_ze = glszm_zone_entropy(result_2d)

            # PyRadiomics needs 3D input
            py_features = get_pyradiomics_glszm(image_3d, mask_3d)

            # Test key features
            @test julia_sae ≈ py_features["SmallAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            @test julia_lae ≈ py_features["LargeAreaEmphasis"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
            @test julia_ze ≈ py_features["ZoneEntropy"] rtol=GLSZM_RTOL atol=GLSZM_ATOL
        end
    end
end

#==============================================================================#
# Summary Report
#==============================================================================#

@testset "GLSZM Parity Summary" begin
    # Final comprehensive test with detailed reporting
    image, mask = random_image_mask(42, GLSZM_MEDIUM_SIZE)

    julia_features = get_julia_glszm_features(image, mask)
    py_features = get_pyradiomics_glszm(image, mask)

    # Count results
    n_tested = 0
    n_passed = 0
    n_missing = 0
    failures = []

    for (julia_sym, py_name) in GLSZM_FEATURE_MAP
        n_tested += 1
        if haskey(py_features, py_name)
            julia_val = getfield(julia_features, julia_sym)
            py_val = py_features[py_name]

            if isapprox(julia_val, py_val; rtol=GLSZM_RTOL, atol=GLSZM_ATOL)
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
    @info "GLSZM Parity Test Summary" n_tested=n_tested n_passed=n_passed n_missing=n_missing n_failed=length(failures)

    if !isempty(failures)
        for f in failures
            @warn "Failed: $(f.name)" julia=f.julia python=f.python
        end
    end

    # Final assertion - all features should pass
    @test n_passed == n_tested - n_missing
    @test length(failures) == 0
end
