# First Order Feature Parity Tests for Radiomics.jl
#
# This file tests all 19 first-order features against PyRadiomics to verify 1:1 parity.
# Tests use deterministic random arrays with fixed seeds for reproducibility.
#
# Story: TEST-FIRSTORDER-PARITY
#
# Tolerance Guidelines:
# - First Order Features: rtol=1e-10, atol=1e-12 (standard)
# - Exception: Entropy/Uniformity may need slightly relaxed tolerance due to
#   histogram-based computation with floating-point probability accumulation
#
# Test Strategy:
# 1. Test each feature individually against PyRadiomics
# 2. Use multiple random seeds (42, 123, 456) for robustness
# 3. Use multiple array sizes (small: 16³, medium: 32³, large: 64³)
# 4. Test edge cases (uniform values, single voxel, etc.)
#
# IMPORTANT: PyRadiomics computes Entropy and Uniformity on DISCRETIZED voxels.
# The discretization uses binWidth (default 25) to reduce continuous values to
# discrete gray levels. This significantly affects Entropy/Uniformity values.
# Our tests must discretize voxels first to match PyRadiomics behavior.

using Test
using Radiomics
using Statistics
using Random

# Test utilities should already be loaded by runtests.jl
# If running standalone: include("test_utils.jl")

#==============================================================================#
# Test Configuration
#==============================================================================#

# Tolerance for first-order features
const FO_RTOL = 1e-10
const FO_ATOL = 1e-12

# Slightly relaxed tolerance for histogram-based features (entropy, uniformity)
# due to floating-point accumulation in probability computation
const HIST_RTOL = 1e-9
const HIST_ATOL = 1e-11

# Random seeds for reproducibility
const TEST_SEEDS = [42, 123, 456]

# Array sizes to test
const SMALL_SIZE = (16, 16, 16)
const MEDIUM_SIZE = (32, 32, 32)
const LARGE_SIZE = (64, 64, 64)

#==============================================================================#
# PyRadiomics Feature Name Mapping
#==============================================================================#

# Map Julia feature names to PyRadiomics feature names
const FEATURE_NAME_MAP = Dict(
    "Energy" => "Energy",
    "TotalEnergy" => "TotalEnergy",
    "Entropy" => "Entropy",
    "Minimum" => "Minimum",
    "10Percentile" => "10Percentile",
    "90Percentile" => "90Percentile",
    "Maximum" => "Maximum",
    "Mean" => "Mean",
    "Median" => "Median",
    "InterquartileRange" => "InterquartileRange",
    "Range" => "Range",
    "MeanAbsoluteDeviation" => "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation" => "RobustMeanAbsoluteDeviation",
    "RootMeanSquared" => "RootMeanSquared",
    "StandardDeviation" => "StandardDeviation",  # Deprecated but still tested
    "Skewness" => "Skewness",
    "Kurtosis" => "Kurtosis",
    "Variance" => "Variance",
    "Uniformity" => "Uniformity"
)

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    get_pyradiomics_firstorder(image, mask)

Extract all first-order features from PyRadiomics for comparison.
Returns a Dict with feature names as keys and values as Float64.
"""
function get_pyradiomics_firstorder(image::AbstractArray, mask::AbstractArray)
    return pyradiomics_extract("firstorder", image, mask)
end

"""
    get_julia_firstorder_with_discretization(image, mask; binwidth=25.0)

Extract all first-order features from Julia with proper discretization for
histogram-based features (Entropy, Uniformity).

This matches PyRadiomics behavior which:
- Uses raw voxels for most features (Energy, Mean, Variance, etc.)
- Uses DISCRETIZED voxels for Entropy and Uniformity

Returns a Dict with feature names as keys and values as Float64.
"""
function get_julia_firstorder_with_discretization(image::AbstractArray, mask::AbstractArray;
                                                   binwidth::Real=25.0, voxel_volume::Real=1.0)
    voxels = get_voxels(image, mask)

    # For histogram-based features, discretize first
    disc_result = discretize_voxels(voxels; binwidth=binwidth)
    discretized = Float64.(disc_result.discretized)

    return Dict{String, Float64}(
        "Energy" => energy(voxels),
        "TotalEnergy" => total_energy(voxels, voxel_volume),
        "Entropy" => entropy(discretized),  # Uses discretized values!
        "Minimum" => fo_minimum(voxels),
        "10Percentile" => percentile_10(voxels),
        "90Percentile" => percentile_90(voxels),
        "Maximum" => fo_maximum(voxels),
        "Mean" => fo_mean(voxels),
        "Median" => fo_median(voxels),
        "InterquartileRange" => interquartile_range(voxels),
        "Range" => fo_range(voxels),
        "MeanAbsoluteDeviation" => mean_absolute_deviation(voxels),
        "RobustMeanAbsoluteDeviation" => robust_mean_absolute_deviation(voxels),
        "RootMeanSquared" => root_mean_squared(voxels),
        "StandardDeviation" => standard_deviation(voxels),
        "Skewness" => skewness(voxels),
        "Kurtosis" => kurtosis(voxels),
        "Variance" => fo_variance(voxels),
        "Uniformity" => uniformity(discretized)  # Uses discretized values!
    )
end

"""
    compare_firstorder_feature(julia_value, py_features, feature_name; rtol=FO_RTOL, atol=FO_ATOL)

Compare a single Julia feature value against PyRadiomics result.
Returns (passed, julia_val, py_val, diff, reldiff).
"""
function compare_firstorder_feature(julia_value::Real, py_features::Dict, feature_name::String;
                                     rtol::Float64=FO_RTOL, atol::Float64=FO_ATOL)
    if !haskey(py_features, feature_name)
        @warn "Feature not found in PyRadiomics results" feature_name
        return (false, julia_value, NaN, NaN, NaN)
    end

    py_value = py_features[feature_name]
    passed = isapprox(julia_value, py_value; rtol=rtol, atol=atol)
    diff = abs(julia_value - py_value)
    reldiff = py_value != 0 ? abs(diff / py_value) : (diff == 0 ? 0.0 : Inf)

    return (passed, julia_value, py_value, diff, reldiff)
end

#==============================================================================#
# Individual Feature Tests
#==============================================================================#

@testset "First Order Feature Parity" begin

    @testset "Test Environment Setup" begin
        @test verify_pyradiomics_available()
    end

    #--------------------------------------------------------------------------
    # Feature 1: Energy
    #--------------------------------------------------------------------------
    @testset "Energy" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = energy(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Energy")
                @test julia_result ≈ py_features["Energy"] rtol=FO_RTOL atol=FO_ATOL
            end
        end

        @testset "Different sizes" begin
            for sz in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE]
                image, mask = random_image_mask(42, sz)
                voxels = get_voxels(image, mask)

                julia_result = energy(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test julia_result ≈ py_features["Energy"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 2: Total Energy
    #--------------------------------------------------------------------------
    @testset "TotalEnergy" begin
        @testset "Multiple seeds with unit spacing" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                # Default spacing is (1,1,1), so voxel_volume = 1
                julia_result = total_energy(voxels, 1.0)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "TotalEnergy")
                @test julia_result ≈ py_features["TotalEnergy"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 3: Entropy
    # NOTE: PyRadiomics computes Entropy on DISCRETIZED voxels (binWidth=25 default)
    #--------------------------------------------------------------------------
    @testset "Entropy" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                # Discretize voxels to match PyRadiomics behavior (default binWidth=25)
                disc_result = discretize_voxels(voxels; binwidth=25.0)
                julia_result = entropy(Float64.(disc_result.discretized))

                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Entropy")
                # Use relaxed tolerance for histogram-based feature
                @test julia_result ≈ py_features["Entropy"] rtol=HIST_RTOL atol=HIST_ATOL
            end
        end

        @testset "Different sizes" begin
            for sz in [SMALL_SIZE, MEDIUM_SIZE]
                image, mask = random_image_mask(42, sz)
                voxels = get_voxels(image, mask)

                # Discretize voxels to match PyRadiomics behavior
                disc_result = discretize_voxels(voxels; binwidth=25.0)
                julia_result = entropy(Float64.(disc_result.discretized))

                py_features = get_pyradiomics_firstorder(image, mask)

                @test julia_result ≈ py_features["Entropy"] rtol=HIST_RTOL atol=HIST_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 4: Minimum
    #--------------------------------------------------------------------------
    @testset "Minimum" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_minimum(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Minimum")
                @test julia_result ≈ py_features["Minimum"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 5: 10th Percentile
    #--------------------------------------------------------------------------
    @testset "10Percentile" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = percentile_10(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "10Percentile")
                @test julia_result ≈ py_features["10Percentile"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 6: 90th Percentile
    #--------------------------------------------------------------------------
    @testset "90Percentile" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = percentile_90(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "90Percentile")
                @test julia_result ≈ py_features["90Percentile"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 7: Maximum
    #--------------------------------------------------------------------------
    @testset "Maximum" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_maximum(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Maximum")
                @test julia_result ≈ py_features["Maximum"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 8: Mean
    #--------------------------------------------------------------------------
    @testset "Mean" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_mean(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Mean")
                @test julia_result ≈ py_features["Mean"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 9: Median
    #--------------------------------------------------------------------------
    @testset "Median" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_median(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Median")
                @test julia_result ≈ py_features["Median"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 10: Interquartile Range
    #--------------------------------------------------------------------------
    @testset "InterquartileRange" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = interquartile_range(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "InterquartileRange")
                @test julia_result ≈ py_features["InterquartileRange"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 11: Range
    #--------------------------------------------------------------------------
    @testset "Range" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_range(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Range")
                @test julia_result ≈ py_features["Range"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 12: Mean Absolute Deviation
    #--------------------------------------------------------------------------
    @testset "MeanAbsoluteDeviation" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = mean_absolute_deviation(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "MeanAbsoluteDeviation")
                @test julia_result ≈ py_features["MeanAbsoluteDeviation"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 13: Robust Mean Absolute Deviation
    #--------------------------------------------------------------------------
    @testset "RobustMeanAbsoluteDeviation" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = robust_mean_absolute_deviation(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "RobustMeanAbsoluteDeviation")
                @test julia_result ≈ py_features["RobustMeanAbsoluteDeviation"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 14: Root Mean Squared
    #--------------------------------------------------------------------------
    @testset "RootMeanSquared" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = root_mean_squared(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "RootMeanSquared")
                @test julia_result ≈ py_features["RootMeanSquared"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 15: Standard Deviation (deprecated but tested)
    #--------------------------------------------------------------------------
    @testset "StandardDeviation" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = standard_deviation(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                # StandardDeviation is deprecated in PyRadiomics but should still be present
                if haskey(py_features, "StandardDeviation")
                    @test julia_result ≈ py_features["StandardDeviation"] rtol=FO_RTOL atol=FO_ATOL
                end
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 16: Skewness
    #--------------------------------------------------------------------------
    @testset "Skewness" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = skewness(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Skewness")
                @test julia_result ≈ py_features["Skewness"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 17: Kurtosis
    #--------------------------------------------------------------------------
    @testset "Kurtosis" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = kurtosis(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Kurtosis")
                @test julia_result ≈ py_features["Kurtosis"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 18: Variance
    #--------------------------------------------------------------------------
    @testset "Variance" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                julia_result = fo_variance(voxels)
                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Variance")
                @test julia_result ≈ py_features["Variance"] rtol=FO_RTOL atol=FO_ATOL
            end
        end
    end

    #--------------------------------------------------------------------------
    # Feature 19: Uniformity
    # NOTE: PyRadiomics computes Uniformity on DISCRETIZED voxels (binWidth=25 default)
    #--------------------------------------------------------------------------
    @testset "Uniformity" begin
        @testset "Multiple seeds" begin
            for seed in TEST_SEEDS
                image, mask = random_image_mask(seed, MEDIUM_SIZE)
                voxels = get_voxels(image, mask)

                # Discretize voxels to match PyRadiomics behavior (default binWidth=25)
                disc_result = discretize_voxels(voxels; binwidth=25.0)
                julia_result = uniformity(Float64.(disc_result.discretized))

                py_features = get_pyradiomics_firstorder(image, mask)

                @test haskey(py_features, "Uniformity")
                # Use relaxed tolerance for histogram-based feature
                @test julia_result ≈ py_features["Uniformity"] rtol=HIST_RTOL atol=HIST_ATOL
            end
        end
    end
end

#==============================================================================#
# Comprehensive Parity Test - All Features at Once
#==============================================================================#

@testset "Comprehensive First Order Parity" begin

    @testset "All features - extract_firstorder vs PyRadiomics" begin
        for seed in TEST_SEEDS
            image, mask = random_image_mask(seed, MEDIUM_SIZE)

            # Julia extraction WITH proper discretization for Entropy/Uniformity
            julia_features = get_julia_firstorder_with_discretization(image, mask)

            # PyRadiomics extraction
            py_features = get_pyradiomics_firstorder(image, mask)

            # Test each feature
            for (julia_name, py_name) in FEATURE_NAME_MAP
                if haskey(py_features, py_name)
                    julia_val = julia_features[julia_name]
                    py_val = py_features[py_name]

                    # Use appropriate tolerance
                    rtol = (py_name in ["Entropy", "Uniformity"]) ? HIST_RTOL : FO_RTOL
                    atol = (py_name in ["Entropy", "Uniformity"]) ? HIST_ATOL : FO_ATOL

                    @test julia_val ≈ py_val rtol=rtol atol=atol
                end
            end
        end
    end

    @testset "All features - Multiple array sizes" begin
        for sz in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE]
            image, mask = random_image_mask(42, sz)

            # Julia extraction WITH proper discretization for Entropy/Uniformity
            julia_features = get_julia_firstorder_with_discretization(image, mask)
            py_features = get_pyradiomics_firstorder(image, mask)

            # Count passing features
            n_features = 0
            n_passed = 0

            for (julia_name, py_name) in FEATURE_NAME_MAP
                if haskey(py_features, py_name)
                    n_features += 1
                    julia_val = julia_features[julia_name]
                    py_val = py_features[py_name]

                    rtol = (py_name in ["Entropy", "Uniformity"]) ? HIST_RTOL : FO_RTOL
                    atol = (py_name in ["Entropy", "Uniformity"]) ? HIST_ATOL : FO_ATOL

                    if isapprox(julia_val, py_val; rtol=rtol, atol=atol)
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
# Edge Cases
#==============================================================================#

@testset "Edge Cases" begin

    @testset "Small mask region" begin
        # Create image with only a small mask region
        rng = MersenneTwister(42)
        image = rand(rng, 32, 32, 32) .* 255
        mask = zeros(Bool, 32, 32, 32)
        mask[14:18, 14:18, 14:18] .= true  # 5x5x5 = 125 voxels

        # Use helper with proper discretization
        julia_features = get_julia_firstorder_with_discretization(image, mask)
        py_features = get_pyradiomics_firstorder(image, mask)

        # All features should still match
        for (julia_name, py_name) in FEATURE_NAME_MAP
            if haskey(py_features, py_name)
                julia_val = julia_features[julia_name]
                py_val = py_features[py_name]

                rtol = (py_name in ["Entropy", "Uniformity"]) ? HIST_RTOL : FO_RTOL
                atol = (py_name in ["Entropy", "Uniformity"]) ? HIST_ATOL : FO_ATOL

                @test julia_val ≈ py_val rtol=rtol atol=atol
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

        # Use helper with proper discretization
        julia_features = get_julia_firstorder_with_discretization(image, mask)
        py_features = get_pyradiomics_firstorder(image, mask)

        # Check key features
        @test julia_features["Energy"] ≈ py_features["Energy"] rtol=FO_RTOL atol=FO_ATOL
        @test julia_features["Mean"] ≈ py_features["Mean"] rtol=FO_RTOL atol=FO_ATOL
        @test julia_features["Variance"] ≈ py_features["Variance"] rtol=FO_RTOL atol=FO_ATOL
    end

    @testset "Near-uniform intensity" begin
        # Image with very little variation
        rng = MersenneTwister(42)
        image = 100.0 .+ rand(rng, 32, 32, 32) .* 0.01  # Very small range
        mask = rand(rng, 32, 32, 32) .< 0.3

        # Ensure non-empty mask
        if !any(mask)
            mask[16, 16, 16] = true
        end

        # Use helper with proper discretization
        julia_features = get_julia_firstorder_with_discretization(image, mask)
        py_features = get_pyradiomics_firstorder(image, mask)

        # Mean should match closely
        @test julia_features["Mean"] ≈ py_features["Mean"] rtol=FO_RTOL

        # Range should be very small
        @test julia_features["Range"] ≈ py_features["Range"] rtol=FO_RTOL
    end

    @testset "Integer image values" begin
        # Discrete integer values (like quantized medical images)
        image_int, mask = random_image_mask_integer(42, MEDIUM_SIZE; intensity_range=(0, 255))
        image = Float64.(image_int)

        # Use helper with proper discretization
        julia_features = get_julia_firstorder_with_discretization(image, mask)
        py_features = get_pyradiomics_firstorder(image, mask)

        # All features should match
        for (julia_name, py_name) in FEATURE_NAME_MAP
            if haskey(py_features, py_name)
                julia_val = julia_features[julia_name]
                py_val = py_features[py_name]

                rtol = (py_name in ["Entropy", "Uniformity"]) ? HIST_RTOL : FO_RTOL
                atol = (py_name in ["Entropy", "Uniformity"]) ? HIST_ATOL : FO_ATOL

                @test julia_val ≈ py_val rtol=rtol atol=atol
            end
        end
    end
end

#==============================================================================#
# Feature Consistency Tests (Julia internal)
#==============================================================================#

@testset "Feature Consistency" begin

    @testset "Energy relationships" begin
        image, mask = random_image_mask(42, MEDIUM_SIZE)
        voxels = get_voxels(image, mask)

        # TotalEnergy with unit spacing should equal Energy
        @test total_energy(voxels, 1.0) ≈ energy(voxels)

        # TotalEnergy with 2.0 spacing should be 2x Energy
        @test total_energy(voxels, 2.0) ≈ 2.0 * energy(voxels)

        # RMS² × N should equal Energy (without shift)
        n = length(voxels)
        @test root_mean_squared(voxels)^2 * n ≈ energy(voxels) rtol=1e-10
    end

    @testset "Range relationships" begin
        image, mask = random_image_mask(42, MEDIUM_SIZE)
        voxels = get_voxels(image, mask)

        # Range = Maximum - Minimum
        @test fo_range(voxels) ≈ fo_maximum(voxels) - fo_minimum(voxels)

        # IQR = P75 - P25
        p25 = quantile(voxels, 0.25)
        p75 = quantile(voxels, 0.75)
        @test interquartile_range(voxels) ≈ p75 - p25
    end

    @testset "Standard deviation and variance" begin
        image, mask = random_image_mask(42, MEDIUM_SIZE)
        voxels = get_voxels(image, mask)

        # Std = sqrt(Variance)
        @test standard_deviation(voxels) ≈ sqrt(fo_variance(voxels))
    end

    @testset "Entropy and uniformity relationship" begin
        image, mask = random_image_mask(42, MEDIUM_SIZE)
        voxels = get_voxels(image, mask)

        ent = entropy(voxels)
        uni = uniformity(voxels)

        # For uniform distribution (all same), entropy=0 and uniformity=1
        # For diverse distribution, high entropy and low uniformity
        # They should be inversely related in general
        @test ent >= 0
        @test 0 <= uni <= 1
    end
end

#==============================================================================#
# 2D Image Tests
#==============================================================================#

@testset "2D Image Parity" begin
    @testset "2D images" begin
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
            voxels = image[mask]

            # Discretize for entropy (to match PyRadiomics behavior)
            disc_result = discretize_voxels(voxels; binwidth=25.0)
            discretized = Float64.(disc_result.discretized)

            julia_features = Dict(
                "Energy" => energy(voxels),
                "Mean" => fo_mean(voxels),
                "Variance" => fo_variance(voxels),
                "Entropy" => entropy(discretized)  # Uses discretized values
            )

            # PyRadiomics needs 3D input
            py_features = get_pyradiomics_firstorder(image_3d, mask_3d)

            # Test key features
            @test julia_features["Energy"] ≈ py_features["Energy"] rtol=FO_RTOL
            @test julia_features["Mean"] ≈ py_features["Mean"] rtol=FO_RTOL
            @test julia_features["Variance"] ≈ py_features["Variance"] rtol=FO_RTOL
            @test julia_features["Entropy"] ≈ py_features["Entropy"] rtol=HIST_RTOL
        end
    end
end

#==============================================================================#
# Summary Report
#==============================================================================#

@testset "First Order Parity Summary" begin
    # Final comprehensive test with detailed reporting
    image, mask = random_image_mask(42, MEDIUM_SIZE)

    # Use helper with proper discretization for Entropy/Uniformity
    julia_features = get_julia_firstorder_with_discretization(image, mask)
    py_features = get_pyradiomics_firstorder(image, mask)

    # Count results
    n_tested = 0
    n_passed = 0
    n_missing = 0
    failures = []

    for (julia_name, py_name) in FEATURE_NAME_MAP
        n_tested += 1
        if haskey(py_features, py_name)
            julia_val = julia_features[julia_name]
            py_val = py_features[py_name]

            rtol = (py_name in ["Entropy", "Uniformity"]) ? HIST_RTOL : FO_RTOL
            atol = (py_name in ["Entropy", "Uniformity"]) ? HIST_ATOL : FO_ATOL

            if isapprox(julia_val, py_val; rtol=rtol, atol=atol)
                n_passed += 1
            else
                push!(failures, (name=py_name, julia=julia_val, python=py_val))
            end
        else
            n_missing += 1
        end
    end

    # Report
    @info "First Order Parity Test Summary" n_tested=n_tested n_passed=n_passed n_missing=n_missing n_failed=length(failures)

    if !isempty(failures)
        for f in failures
            @warn "Failed: $(f.name)" julia=f.julia python=f.python
        end
    end

    # Final assertion - all features should pass
    @test n_passed == n_tested - n_missing
    @test length(failures) == 0
end
