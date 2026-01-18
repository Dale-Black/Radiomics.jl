# Integration Tests for Radiomics.jl
#
# This test file verifies the complete feature extraction pipeline works correctly
# across various scenarios: different image sizes, mask shapes, configuration options,
# and feature class combinations.

# Helper macro for capturing output (must be defined before use)
macro capture_out(ex)
    quote
        buf = IOBuffer()
        old_stdout = stdout
        redirect_stdout(buf)
        try
            $(esc(ex))
        finally
            redirect_stdout(old_stdout)
        end
        String(take!(buf))
    end
end

@testset "Integration Tests" begin

    #==========================================================================#
    # Full Extraction Pipeline Tests
    #==========================================================================#

    @testset "Full Extraction Pipeline" begin

        @testset "Default extractor with all features" begin
            # Create test data
            image, mask = random_image_mask(42, (32, 32, 32))

            # Create extractor with default settings
            extractor = RadiomicsFeatureExtractor()

            # Verify all classes are enabled by default
            @test length(enabled_classes(extractor)) == 7

            # Extract all features
            features = extract(extractor, image, mask)

            # Should have all feature classes
            @test !isempty(features)

            # Check for expected feature prefixes
            has_firstorder = any(k -> startswith(k, "firstorder_"), keys(features))
            has_shape = any(k -> startswith(k, "shape_"), keys(features))
            has_glcm = any(k -> startswith(k, "glcm_"), keys(features))
            has_glrlm = any(k -> startswith(k, "glrlm_"), keys(features))
            has_glszm = any(k -> startswith(k, "glszm_"), keys(features))
            has_ngtdm = any(k -> startswith(k, "ngtdm_"), keys(features))
            has_gldm = any(k -> startswith(k, "gldm_"), keys(features))

            @test has_firstorder
            @test has_shape
            @test has_glcm
            @test has_glrlm
            @test has_glszm
            @test has_ngtdm
            @test has_gldm

            # Verify feature count is reasonable (should be ~111+ for 3D)
            @test length(features) >= 100

            # All values should be finite
            @test all(isfinite, values(features))
        end

        @testset "Selective feature class extraction" begin
            image, mask = random_image_mask(123, (24, 24, 24))

            # Only FirstOrder features
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set([FirstOrder]))
            features = extract(extractor, image, mask)

            @test length(features) == 19  # FirstOrder has 19 features
            @test all(k -> startswith(k, "firstorder_"), keys(features))

            # Only Shape features
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set([Shape]))
            features = extract(extractor, image, mask)

            @test length(features) == 17  # Shape 3D has 17 features
            @test all(k -> startswith(k, "shape_"), keys(features))

            # Only GLCM features
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set([GLCM]))
            features = extract(extractor, image, mask)

            @test length(features) == 24  # GLCM has 24 features
            @test all(k -> startswith(k, "glcm_"), keys(features))

            # Multiple texture classes
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set([GLCM, GLRLM, GLSZM]))
            features = extract(extractor, image, mask)

            @test length(features) == 24 + 16 + 16  # GLCM + GLRLM + GLSZM
        end

        @testset "Enable/disable class operations" begin
            image, mask = random_image_mask(456, (20, 20, 20))

            # Start with empty extractor
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set{FeatureClass}())
            @test length(enabled_classes(extractor)) == 0

            # Enable single class
            enable!(extractor, FirstOrder)
            @test is_enabled(extractor, FirstOrder)
            @test !is_enabled(extractor, GLCM)

            features = extract(extractor, image, mask)
            @test length(features) == 19

            # Enable another class
            enable!(extractor, NGTDM)
            @test is_enabled(extractor, NGTDM)

            features = extract(extractor, image, mask)
            @test length(features) == 19 + 5  # FirstOrder + NGTDM

            # Disable a class
            disable!(extractor, FirstOrder)
            @test !is_enabled(extractor, FirstOrder)

            features = extract(extractor, image, mask)
            @test length(features) == 5  # Only NGTDM

            # Enable all
            enable_all!(extractor)
            @test length(enabled_classes(extractor)) == 7

            # Disable all
            disable_all!(extractor)
            @test length(enabled_classes(extractor)) == 0
        end

        @testset "Chained enable/disable operations" begin
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set{FeatureClass}())

            # Chain operations
            enable!(enable!(enable!(extractor, FirstOrder), GLCM), Shape)

            @test is_enabled(extractor, FirstOrder)
            @test is_enabled(extractor, GLCM)
            @test is_enabled(extractor, Shape)
            @test length(enabled_classes(extractor)) == 3
        end

    end

    #==========================================================================#
    # Various Image Sizes
    #==========================================================================#

    @testset "Various Image Sizes" begin

        @testset "Small images (16x16x16)" begin
            image, mask = random_image_mask(100, (16, 16, 16))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Medium images (32x32x32)" begin
            image, mask = random_image_mask(101, (32, 32, 32))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Large images (64x64x64)" begin
            image, mask = random_image_mask(102, (64, 64, 64))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Non-cubic images (32x48x24)" begin
            image, mask = random_image_mask(103, (32, 48, 24))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Thin slabs (64x64x8)" begin
            image, mask = random_image_mask(104, (64, 64, 8))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "2D images (64x64)" begin
            image, mask = random_image_mask(105, (64, 64))
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))

            # 2D should have shape2d features, not shape3d
            # (Note: depending on implementation, prefix may still be "shape_")
        end

    end

    #==========================================================================#
    # Various Mask Shapes
    #==========================================================================#

    @testset "Various Mask Shapes" begin

        @testset "Random scattered mask" begin
            image, mask = random_image_mask(200, (32, 32, 32); mask_fraction=0.2)
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Dense mask (80% fill)" begin
            image, mask = random_image_mask(201, (32, 32, 32); mask_fraction=0.8)
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Sparse mask (5% fill)" begin
            image, mask = random_image_mask(202, (32, 32, 32); mask_fraction=0.05)
            extractor = RadiomicsFeatureExtractor()

            features = extract(extractor, image, mask)
            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Cubic ROI (central cube)" begin
            image = rand(MersenneTwister(203), 32, 32, 32) .* 255.0
            mask = falses(32, 32, 32)
            mask[10:22, 10:22, 10:22] .= true

            extractor = RadiomicsFeatureExtractor()
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))

            # Volume should be approximately (13^3) * voxel_volume
            # With default spacing of 1.0, should be close to 2197
            @test features["shape_VoxelVolume"] ≈ 13^3 rtol=0.01
        end

        @testset "Spherical ROI (approximate sphere)" begin
            image = rand(MersenneTwister(204), 32, 32, 32) .* 255.0
            mask = falses(32, 32, 32)

            # Create approximate sphere centered at (16,16,16) with radius 8
            center = (16, 16, 16)
            radius = 8
            for i in 1:32, j in 1:32, k in 1:32
                d = sqrt((i - center[1])^2 + (j - center[2])^2 + (k - center[3])^2)
                if d <= radius
                    mask[i, j, k] = true
                end
            end

            extractor = RadiomicsFeatureExtractor()
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))

            # Sphericity should be close to 1 for a sphere
            @test features["shape_Sphericity"] > 0.8
        end

        @testset "Elongated ROI (rod shape)" begin
            image = rand(MersenneTwister(205), 32, 32, 32) .* 255.0
            mask = falses(32, 32, 32)
            # Create elongated rod along z-axis
            mask[14:18, 14:18, 2:30] .= true

            extractor = RadiomicsFeatureExtractor()
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))

            # Elongation should be high (close to 0 means elongated)
            # Actually in IBSI, Elongation = minor/major so elongated shapes have low values
            @test features["shape_Elongation"] < 0.5
        end

        @testset "Flat ROI (pancake shape)" begin
            image = rand(MersenneTwister(206), 32, 32, 32) .* 255.0
            mask = falses(32, 32, 32)
            # Create flat disk
            mask[4:28, 4:28, 14:18] .= true

            extractor = RadiomicsFeatureExtractor()
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))

            # Flatness should be low (0 = flat, 1 = not flat)
            @test features["shape_Flatness"] < 0.5
        end

    end

    #==========================================================================#
    # Configuration Options
    #==========================================================================#

    @testset "Configuration Options" begin

        @testset "Custom bin width" begin
            image, mask = random_image_mask(300, (24, 24, 24))

            # Default bin width (25)
            extractor1 = RadiomicsFeatureExtractor(settings=Settings(binwidth=25.0))
            features1 = extract(extractor1, image, mask)

            # Different bin width (50)
            extractor2 = RadiomicsFeatureExtractor(settings=Settings(binwidth=50.0))
            features2 = extract(extractor2, image, mask)

            # First-order features should be the same (not affected by discretization)
            # But texture features should differ
            @test features1["firstorder_Energy"] ≈ features2["firstorder_Energy"]
            @test features1["firstorder_Mean"] ≈ features2["firstorder_Mean"]

            # Texture features will differ (but both should be valid)
            @test isfinite(features1["glcm_Contrast"])
            @test isfinite(features2["glcm_Contrast"])
            # They may or may not be equal depending on how discretization affects them
        end

        @testset "Fixed bin count mode" begin
            image, mask = random_image_mask(301, (24, 24, 24))

            # Fixed bin count mode
            settings = Settings(
                bincount=32,
                discretization_mode=FixedBinCount
            )
            extractor = RadiomicsFeatureExtractor(settings=settings)
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Custom GLCM distance" begin
            image, mask = random_image_mask(302, (24, 24, 24))

            # Distance 1 (default)
            extractor1 = RadiomicsFeatureExtractor(
                enabled_classes=Set([GLCM]),
                settings=Settings(glcm_distance=1)
            )
            features1 = extract(extractor1, image, mask)

            # Distance 2
            extractor2 = RadiomicsFeatureExtractor(
                enabled_classes=Set([GLCM]),
                settings=Settings(glcm_distance=2)
            )
            features2 = extract(extractor2, image, mask)

            # Both should produce valid features
            @test all(isfinite, values(features1))
            @test all(isfinite, values(features2))

            # Features should generally differ
            # (Though for some random images they might be similar)
            @test length(features1) == length(features2) == 24
        end

        @testset "Custom NGTDM distance" begin
            image, mask = random_image_mask(303, (24, 24, 24))

            # Distance 1
            extractor1 = RadiomicsFeatureExtractor(
                enabled_classes=Set([NGTDM]),
                settings=Settings(ngtdm_distance=1)
            )
            features1 = extract(extractor1, image, mask)

            # Distance 2
            extractor2 = RadiomicsFeatureExtractor(
                enabled_classes=Set([NGTDM]),
                settings=Settings(ngtdm_distance=2)
            )
            features2 = extract(extractor2, image, mask)

            @test all(isfinite, values(features1))
            @test all(isfinite, values(features2))
            @test length(features1) == length(features2) == 5
        end

        @testset "Custom GLDM alpha" begin
            image, mask = random_image_mask(304, (24, 24, 24))

            # Alpha 0 (default)
            extractor1 = RadiomicsFeatureExtractor(
                enabled_classes=Set([GLDM]),
                settings=Settings(gldm_alpha=0.0)
            )
            features1 = extract(extractor1, image, mask)

            # Alpha 1
            extractor2 = RadiomicsFeatureExtractor(
                enabled_classes=Set([GLDM]),
                settings=Settings(gldm_alpha=1.0)
            )
            features2 = extract(extractor2, image, mask)

            @test all(isfinite, values(features1))
            @test all(isfinite, values(features2))
            @test length(features1) == length(features2) == 14
        end

        @testset "Custom voxel spacing" begin
            image, mask = random_image_mask(305, (24, 24, 24))

            # Default spacing (1.0, 1.0, 1.0)
            extractor = RadiomicsFeatureExtractor(enabled_classes=Set([Shape, FirstOrder]))
            features1 = extract(extractor, image, mask)

            # Custom spacing (0.5, 0.5, 0.5) - smaller voxels
            features2 = extract(extractor, image, mask; spacing=(0.5, 0.5, 0.5))

            # Volume should scale with spacing^3
            # With 0.5 spacing, volume should be 1/8 of default
            @test features2["shape_VoxelVolume"] ≈ features1["shape_VoxelVolume"] / 8 rtol=0.01

            # Total Energy scales with voxel volume
            @test features2["firstorder_TotalEnergy"] ≈ features1["firstorder_TotalEnergy"] / 8 rtol=0.01

            # Mean should be the same (not affected by spacing)
            @test features1["firstorder_Mean"] ≈ features2["firstorder_Mean"]
        end

        @testset "Settings validation" begin
            # Valid settings should not throw
            @test validate_settings(Settings()) == true
            @test validate_settings(Settings(binwidth=50.0)) == true
            @test validate_settings(Settings(bincount=32, discretization_mode=FixedBinCount)) == true

            # Invalid settings should throw
            @test_throws ArgumentError Settings(binwidth=-1.0)
            @test_throws ArgumentError Settings(glcm_distance=0)
            @test_throws ArgumentError Settings(ngtdm_distance=-1)
        end

    end

    #==========================================================================#
    # High-Level API Functions
    #==========================================================================#

    @testset "High-Level API" begin

        @testset "extract_all convenience function" begin
            image, mask = random_image_mask(400, (24, 24, 24))

            features = extract_all(image, mask)

            @test !isempty(features)
            @test length(features) >= 100
            @test all(isfinite, values(features))
        end

        @testset "extract_firstorder_only" begin
            image, mask = random_image_mask(401, (24, 24, 24))

            features = extract_firstorder_only(image, mask)

            @test length(features) == 19
            @test all(k -> startswith(k, "firstorder_"), keys(features))
        end

        @testset "extract_shape_only" begin
            mask = falses(24, 24, 24)
            mask[8:16, 8:16, 8:16] .= true

            features = extract_shape_only(mask)

            @test length(features) == 17  # 3D shape
            @test all(k -> startswith(k, "shape_"), keys(features))
        end

        @testset "extract_texture_only" begin
            image, mask = random_image_mask(402, (24, 24, 24))

            features = extract_texture_only(image, mask)

            # Should have GLCM + GLRLM + GLSZM + NGTDM + GLDM
            @test length(features) == 24 + 16 + 16 + 5 + 14  # 75 total

            # Should not have FirstOrder or Shape
            @test !any(k -> startswith(k, "firstorder_"), keys(features))
            @test !any(k -> startswith(k, "shape_"), keys(features))
        end

        @testset "Per-class extraction helpers" begin
            image, mask = random_image_mask(403, (24, 24, 24))

            # extract_glcm
            glcm_features = extract_glcm(image, mask)
            @test length(glcm_features) == 24
            @test all(k -> startswith(k, "glcm_"), keys(glcm_features))

            # extract_glrlm
            glrlm_features = extract_glrlm(image, mask)
            @test length(glrlm_features) == 16
            @test all(k -> startswith(k, "glrlm_"), keys(glrlm_features))

            # extract_glszm
            glszm_features = extract_glszm(image, mask)
            @test length(glszm_features) == 16
            @test all(k -> startswith(k, "glszm_"), keys(glszm_features))

            # extract_ngtdm
            ngtdm_features = extract_ngtdm(image, mask)
            @test length(ngtdm_features) == 5
            @test all(k -> startswith(k, "ngtdm_"), keys(ngtdm_features))

            # extract_gldm
            gldm_features = extract_gldm(image, mask)
            @test length(gldm_features) == 14
            @test all(k -> startswith(k, "gldm_"), keys(gldm_features))
        end

        @testset "feature_count and feature_names" begin
            @test feature_count(FirstOrder) == 19
            @test feature_count(Shape) == 17  # 3D
            @test feature_count(GLCM) == 24
            @test feature_count(GLRLM) == 16
            @test feature_count(GLSZM) == 16
            @test feature_count(NGTDM) == 5
            @test feature_count(GLDM) == 14

            # total_feature_count
            extractor = RadiomicsFeatureExtractor()
            @test total_feature_count(extractor) >= 100

            # feature_names
            fo_names = feature_names(FirstOrder)
            @test length(fo_names) == 19
            @test "Energy" in fo_names
            @test "Mean" in fo_names
            @test "Entropy" in fo_names
        end

        @testset "list_feature_classes" begin
            classes = list_feature_classes()
            @test length(classes) == 7
            @test "FirstOrder" in classes
            @test "Shape" in classes
            @test "GLCM" in classes
            @test "GLRLM" in classes
            @test "GLSZM" in classes
            @test "NGTDM" in classes
            @test "GLDM" in classes
        end

        @testset "summarize_features" begin
            image, mask = random_image_mask(404, (24, 24, 24))
            features = extract_all(image, mask)

            # Should not throw
            output = @capture_out summarize_features(features)
            @test !isempty(output)
            @test occursin("firstorder", lowercase(output))
        end

        @testset "extract_batch" begin
            # Create batch of test data
            images = [random_image_mask(500 + i, (16, 16, 16))[1] for i in 1:3]
            masks = [random_image_mask(500 + i, (16, 16, 16))[2] for i in 1:3]

            results = extract_batch(images, masks)

            @test length(results) == 3
            @test all(!isempty, results)
            @test all(r -> all(isfinite, values(r)), results)
        end

    end

    #==========================================================================#
    # Memory and Edge Cases
    #==========================================================================#

    @testset "Memory and Edge Cases" begin

        @testset "Multiple sequential extractions" begin
            # Run multiple extractions to check for memory leaks
            extractor = RadiomicsFeatureExtractor()

            for i in 1:10
                image, mask = random_image_mask(600 + i, (20, 20, 20))
                features = extract(extractor, image, mask)
                @test !isempty(features)
            end

            # If we got here without OOM, memory handling is reasonable
            @test true
        end

        @testset "Reusing extractor instance" begin
            extractor = RadiomicsFeatureExtractor()

            # First extraction
            image1, mask1 = random_image_mask(700, (20, 20, 20))
            features1 = extract(extractor, image1, mask1)

            # Second extraction with different data
            image2, mask2 = random_image_mask(701, (24, 24, 24))
            features2 = extract(extractor, image2, mask2)

            # Third extraction
            image3, mask3 = random_image_mask(702, (18, 18, 18))
            features3 = extract(extractor, image3, mask3)

            # All should succeed
            @test !isempty(features1)
            @test !isempty(features2)
            @test !isempty(features3)

            # Different images should give different features
            @test features1["firstorder_Mean"] != features2["firstorder_Mean"]
        end

        @testset "Minimum viable mask" begin
            # Very small mask (just a few voxels)
            image = rand(MersenneTwister(800), 32, 32, 32) .* 255.0
            mask = falses(32, 32, 32)
            mask[15:17, 15:17, 15:17] .= true  # 27 voxels

            extractor = RadiomicsFeatureExtractor()
            features = extract(extractor, image, mask)

            @test !isempty(features)
            @test all(isfinite, values(features))
        end

        @testset "Single row/column/slice in mask" begin
            image = rand(MersenneTwister(801), 32, 32, 32) .* 255.0

            # Row mask
            mask_row = falses(32, 32, 32)
            mask_row[16, :, 16] .= true

            # Column mask
            mask_col = falses(32, 32, 32)
            mask_col[:, 16, 16] .= true

            # Slice mask
            mask_slice = falses(32, 32, 32)
            mask_slice[:, :, 16] .= true

            extractor = RadiomicsFeatureExtractor()

            # These should all work (though some features may be degenerate)
            features_row = extract(extractor, image, mask_row)
            features_col = extract(extractor, image, mask_col)
            features_slice = extract(extractor, image, mask_slice)

            @test !isempty(features_row)
            @test !isempty(features_col)
            @test !isempty(features_slice)
        end

        @testset "Integer mask labels" begin
            image, _ = random_image_mask(802, (24, 24, 24))

            # Create integer mask with multiple labels
            int_mask = zeros(Int, 24, 24, 24)
            int_mask[4:12, 4:12, 4:12] .= 1
            int_mask[14:20, 14:20, 14:20] .= 2

            # Should work with explicit label
            features1 = extract_firstorder_only(image, int_mask .== 1)
            features2 = extract_firstorder_only(image, int_mask .== 2)

            @test !isempty(features1)
            @test !isempty(features2)

            # Different ROIs should give different features
            @test features1["firstorder_Mean"] != features2["firstorder_Mean"]
        end

        @testset "Float64 vs Float32 images" begin
            mask = falses(24, 24, 24)
            mask[8:16, 8:16, 8:16] .= true

            # Float64 image
            image64 = rand(MersenneTwister(803), Float64, 24, 24, 24) .* 255.0
            features64 = extract_all(image64, mask)

            # Float32 image (should be automatically converted)
            image32 = Float32.(image64)
            features32 = extract_all(image32, mask)

            @test !isempty(features64)
            @test !isempty(features32)

            # Results should be very close
            @test features64["firstorder_Mean"] ≈ features32["firstorder_Mean"] rtol=1e-6
        end

        @testset "Constant intensity region" begin
            # All voxels have the same intensity
            image = fill(128.0, 24, 24, 24)
            mask = falses(24, 24, 24)
            mask[8:16, 8:16, 8:16] .= true

            extractor = RadiomicsFeatureExtractor(
                enabled_classes=Set([FirstOrder])
            )
            features = extract(extractor, image, mask)

            @test !isempty(features)

            # Variance should be 0 for constant region
            @test features["firstorder_Variance"] ≈ 0 atol=1e-10

            # Mean should equal the constant value
            @test features["firstorder_Mean"] ≈ 128.0
        end

    end

    #==========================================================================#
    # Consistency Tests
    #==========================================================================#

    @testset "Consistency Tests" begin

        @testset "Deterministic results with same seed" begin
            # Same seed should give identical results
            image1, mask1 = random_image_mask(900, (24, 24, 24))
            image2, mask2 = random_image_mask(900, (24, 24, 24))

            @test image1 == image2
            @test mask1 == mask2

            extractor = RadiomicsFeatureExtractor()
            features1 = extract(extractor, image1, mask1)
            features2 = extract(extractor, image2, mask2)

            # All features should be identical
            for (k, v1) in features1
                @test v1 ≈ features2[k] rtol=1e-12
            end
        end

        @testset "Different seeds give different results" begin
            image1, mask1 = random_image_mask(901, (24, 24, 24))
            image2, mask2 = random_image_mask(902, (24, 24, 24))

            extractor = RadiomicsFeatureExtractor()
            features1 = extract(extractor, image1, mask1)
            features2 = extract(extractor, image2, mask2)

            # Features should differ
            @test features1["firstorder_Mean"] != features2["firstorder_Mean"]
        end

        @testset "Feature values are reasonable" begin
            image, mask = random_image_mask(903, (24, 24, 24))
            features = extract_all(image, mask)

            # Energy should be positive
            @test features["firstorder_Energy"] > 0

            # Entropy should be non-negative
            @test features["firstorder_Entropy"] >= 0

            # Mean should be within intensity range
            voxels = image[mask]
            @test features["firstorder_Mean"] >= minimum(voxels) - 1e-10
            @test features["firstorder_Mean"] <= maximum(voxels) + 1e-10

            # Shape metrics should be positive
            @test features["shape_VoxelVolume"] > 0
            @test features["shape_SurfaceArea"] > 0

            # Sphericity should be between 0 and 1
            @test 0 <= features["shape_Sphericity"] <= 1

            # Uniformity should be between 0 and 1
            @test 0 <= features["firstorder_Uniformity"] <= 1
        end

    end

end
