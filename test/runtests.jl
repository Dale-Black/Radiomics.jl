using Test
using Radiomics

# Load test utilities for PyRadiomics parity testing
include("test_utils.jl")

@testset "Radiomics.jl" begin
    @testset "Package loads" begin
        @test Radiomics.VERSION == v"0.1.0"
    end

    @testset "Test Harness" begin
        @testset "PyRadiomics available" begin
            @test verify_pyradiomics_available()
        end

        @testset "Random data generation" begin
            # Test reproducibility
            img1, mask1 = random_image_mask(42, (16, 16, 16))
            img2, mask2 = random_image_mask(42, (16, 16, 16))
            @test img1 == img2
            @test mask1 == mask2

            # Test dimensions
            @test size(img1) == (16, 16, 16)
            @test size(mask1) == (16, 16, 16)

            # Test types
            @test eltype(img1) == Float64
            @test eltype(mask1) == Bool

            # Test mask is non-empty
            @test any(mask1)
        end

        @testset "PyRadiomics feature extraction" begin
            # Generate test data
            image, mask = random_image_mask(42, (16, 16, 16))

            # Extract first-order features from PyRadiomics
            features = pyradiomics_extract("firstorder", image, mask)

            # Should have 18 first-order features
            @test length(features) >= 18

            # Check some expected features exist
            @test haskey(features, "Energy")
            @test haskey(features, "Mean")
            @test haskey(features, "Entropy")

            # Check feature values are reasonable
            @test features["Mean"] > 0
            @test features["Energy"] > 0
        end
    end

    # Core infrastructure tests
    include("test_core.jl")

    # Feature class tests will be included as they are implemented
    # include("test_firstorder.jl")
    # include("test_shape.jl")
    # include("test_glcm.jl")
    # include("test_glrlm.jl")
    # include("test_glszm.jl")
    # include("test_ngtdm.jl")
    # include("test_gldm.jl")
    # include("test_integration.jl")
    # include("test_full_parity.jl")
end
