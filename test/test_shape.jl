# Shape Feature Parity Tests for Radiomics.jl
#
# This file tests 2D and 3D shape features against PyRadiomics to verify parity.
# Tests use deterministic masks with known shapes for reproducibility.
#
# Story: TEST-SHAPE-PARITY
#
# Tolerance Guidelines:
# - Shape Features: rtol=1e-4 to 1e-2 (relaxed due to mesh algorithm differences)
# - Marching cubes algorithms may produce different meshes between implementations.
#
# Note on Shape Feature Comparison:
# Shape features in PyRadiomics are computed from the mask geometry, NOT image intensities.
# Some features (especially mesh-based) may have small differences due to:
# - Different marching cubes/squares implementations
# - Floating-point accumulation in mesh calculations
# - Different vertex handling at boundaries
#
# Note on Deprecated Features:
# PyRadiomics v3.0 no longer exports deprecated features (Compactness1, Compactness2,
# SphericalDisproportion) by default. These are skipped in parity tests.

using Test
using Radiomics
using Statistics
using Random

# Test utilities should already be loaded by runtests.jl

#==============================================================================#
# Test Configuration
#==============================================================================#

# Tolerances for shape features (relaxed due to mesh algorithm differences)
const SHAPE_RTOL = 1e-4  # 0.01% relative tolerance
const SHAPE_ATOL = 1e-6  # Absolute tolerance for small values

# More relaxed tolerance for mesh-derived features
const MESH_RTOL = 0.02   # 2% tolerance for mesh-derived features
const MESH_ATOL = 1e-4

# Sizes for test masks
const SIZE_3D_MEDIUM = (24, 24, 24)
const SIZE_2D_MEDIUM = (40, 40)

#==============================================================================#
# Helper Functions - Mask Generation
#==============================================================================#

"""Create a 3D spherical mask for testing."""
function create_sphere_mask_3d(size::NTuple{3,Int}, center::NTuple{3,<:Real}, radius::Real)
    mask = falses(size...)
    for i in 1:size[1], j in 1:size[2], k in 1:size[3]
        dist = sqrt((i - center[1])^2 + (j - center[2])^2 + (k - center[3])^2)
        if dist <= radius
            mask[i, j, k] = true
        end
    end
    return mask
end

"""Create a 3D cubic mask for testing."""
function create_cube_mask_3d(size::NTuple{3,Int}, corner::NTuple{3,Int}, side_length::Int)
    mask = falses(size...)
    for i in corner[1]:min(corner[1]+side_length-1, size[1])
        for j in corner[2]:min(corner[2]+side_length-1, size[2])
            for k in corner[3]:min(corner[3]+side_length-1, size[3])
                mask[i, j, k] = true
            end
        end
    end
    return mask
end

"""Create a 3D ellipsoidal mask for testing elongated shapes."""
function create_ellipsoid_mask_3d(size::NTuple{3,Int}, center::NTuple{3,<:Real}, radii::NTuple{3,<:Real})
    mask = falses(size...)
    for i in 1:size[1], j in 1:size[2], k in 1:size[3]
        val = ((i - center[1])/radii[1])^2 + ((j - center[2])/radii[2])^2 + ((k - center[3])/radii[3])^2
        if val <= 1.0
            mask[i, j, k] = true
        end
    end
    return mask
end

"""Create a 2D circular mask for testing."""
function create_circle_mask_2d(size::NTuple{2,Int}, center::NTuple{2,<:Real}, radius::Real)
    mask = falses(size...)
    for i in 1:size[1], j in 1:size[2]
        dist = sqrt((i - center[1])^2 + (j - center[2])^2)
        if dist <= radius
            mask[i, j] = true
        end
    end
    return mask
end

"""Create a 2D square mask for testing."""
function create_square_mask_2d(size::NTuple{2,Int}, corner::NTuple{2,Int}, side_length::Int)
    mask = falses(size...)
    for i in corner[1]:min(corner[1]+side_length-1, size[1])
        for j in corner[2]:min(corner[2]+side_length-1, size[2])
            mask[i, j] = true
        end
    end
    return mask
end

"""Create a 2D elliptical mask for testing elongated shapes."""
function create_ellipse_mask_2d(size::NTuple{2,Int}, center::NTuple{2,<:Real}, radii::NTuple{2,<:Real})
    mask = falses(size...)
    for i in 1:size[1], j in 1:size[2]
        val = ((i - center[1])/radii[1])^2 + ((j - center[2])/radii[2])^2
        if val <= 1.0
            mask[i, j] = true
        end
    end
    return mask
end

#==============================================================================#
# Helper Functions - PyRadiomics Extraction
#==============================================================================#

"""
Extract all 3D shape features from PyRadiomics for comparison.
Creates a dummy image (shape features don't use intensity values).
"""
function get_pyradiomics_shape3d(mask::AbstractArray{Bool,3}; spacing::Tuple=(1.0, 1.0, 1.0))
    image = ones(Float64, size(mask)...)
    return pyradiomics_extract("shape", image, mask; spacing=spacing)
end

"""
Extract all 2D shape features from PyRadiomics for comparison.
PyRadiomics Shape2D expects a 3D image with one dimension=1 (a single slice).
"""
function get_pyradiomics_shape2d(mask::AbstractMatrix{Bool}; spacing::Tuple=(1.0, 1.0))
    # Convert 2D mask to 3D by adding a singleton dimension (z=1)
    mask_3d = reshape(mask, size(mask)..., 1)
    image_3d = ones(Float64, size(mask_3d)...)

    # Spacing for 3D: (x, y, z) where z is the slice thickness
    spacing_3d = (spacing[1], spacing[2], 1.0)

    py = get_python_modules()
    radiomics = py.radiomics

    sitk_image, sitk_mask = julia_array_to_sitk(image_3d, mask_3d; spacing=spacing_3d)

    # Get the Shape2D feature class
    shape2d_class = radiomics.shape2D.RadiomicsShape2D

    # Instantiate feature extractor
    # Shape2D needs force2D=true and force2Ddimension=0 (the z dimension in NumPy order)
    # since we have a 3D array with z=1 slice
    extractor = shape2d_class(sitk_image, sitk_mask; force2D=true, force2Ddimension=0)

    # Execute feature calculation
    result_dict = extractor.execute()

    # Convert to Julia Dict
    results = Dict{String,Float64}()
    np = py.numpy
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

"""Extract all 3D shape features from Julia implementation."""
function get_julia_shape3d(mask::AbstractArray{Bool,3}; spacing::NTuple{3,<:Real}=(1.0, 1.0, 1.0))
    return extract_shape_3d(mask, spacing)
end

"""Extract all 2D shape features from Julia implementation."""
function get_julia_shape2d(mask::AbstractMatrix{Bool}; spacing::NTuple{2,<:Real}=(1.0, 1.0))
    return extract_shape_2d(mask, spacing)
end

#==============================================================================#
# Feature Comparison Helper
#==============================================================================#

"""
Compare a shape feature between Julia and PyRadiomics.
Returns true if values match within tolerance, missing if feature not found.
"""
function test_shape_feature(julia_features::Dict, py_features::Dict, feature_name::String;
                            rtol::Float64=SHAPE_RTOL, atol::Float64=SHAPE_ATOL)
    if !haskey(julia_features, feature_name) || !haskey(py_features, feature_name)
        return missing
    end

    julia_val = julia_features[feature_name]
    py_val = py_features[feature_name]

    # Handle NaN values
    if isnan(julia_val) && isnan(py_val)
        return true
    elseif isnan(julia_val) || isnan(py_val)
        return false
    end

    return isapprox(julia_val, py_val; rtol=rtol, atol=atol)
end

#==============================================================================#
# 3D Shape Feature Tests
#==============================================================================#

@testset "3D Shape Features" begin

    @testset "VoxelVolume (Exact)" begin
        # VoxelVolume is a simple count - should match exactly
        @testset "Cubic mask" begin
            mask = create_cube_mask_3d(SIZE_3D_MEDIUM, (5, 5, 5), 10)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "VoxelVolume";
                                        rtol=1e-10, atol=1e-12)
            @test result === true || @info "VoxelVolume" julia=julia_features["VoxelVolume"] python=py_features["VoxelVolume"]
        end

        @testset "Spherical mask" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "VoxelVolume";
                                        rtol=1e-10, atol=1e-12)
            @test result === true || @info "VoxelVolume" julia=julia_features["VoxelVolume"] python=py_features["VoxelVolume"]
        end
    end

    @testset "MeshVolume" begin
        @testset "Cubic mask" begin
            mask = create_cube_mask_3d(SIZE_3D_MEDIUM, (5, 5, 5), 10)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "MeshVolume";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MeshVolume" julia=julia_features["MeshVolume"] python=py_features["MeshVolume"]
        end

        @testset "Spherical mask" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "MeshVolume";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MeshVolume" julia=julia_features["MeshVolume"] python=py_features["MeshVolume"]
        end
    end

    @testset "SurfaceArea" begin
        @testset "Cubic mask" begin
            mask = create_cube_mask_3d(SIZE_3D_MEDIUM, (5, 5, 5), 10)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "SurfaceArea";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "SurfaceArea" julia=julia_features["SurfaceArea"] python=py_features["SurfaceArea"]
        end

        @testset "Spherical mask" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "SurfaceArea";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "SurfaceArea" julia=julia_features["SurfaceArea"] python=py_features["SurfaceArea"]
        end
    end

    @testset "SurfaceVolumeRatio" begin
        mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
        julia_features = get_julia_shape3d(mask)
        py_features = get_pyradiomics_shape3d(mask)

        result = test_shape_feature(julia_features, py_features, "SurfaceVolumeRatio";
                                    rtol=MESH_RTOL, atol=MESH_ATOL)
        @test result === true || @info "SurfaceVolumeRatio" julia=julia_features["SurfaceVolumeRatio"] python=py_features["SurfaceVolumeRatio"]
    end

    @testset "Sphericity" begin
        @testset "Spherical mask (should be close to 1.0)" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Sphericity";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Sphericity" julia=julia_features["Sphericity"] python=py_features["Sphericity"]
        end

        @testset "Cubic mask" begin
            mask = create_cube_mask_3d(SIZE_3D_MEDIUM, (5, 5, 5), 10)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Sphericity";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Sphericity" julia=julia_features["Sphericity"] python=py_features["Sphericity"]
        end
    end

    @testset "Maximum3DDiameter" begin
        @testset "Spherical mask" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Maximum3DDiameter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Maximum3DDiameter" julia=julia_features["Maximum3DDiameter"] python=py_features["Maximum3DDiameter"]
        end

        @testset "Ellipsoidal mask" begin
            mask = create_ellipsoid_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), (10.0, 5.0, 3.0))
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Maximum3DDiameter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Maximum3DDiameter" julia=julia_features["Maximum3DDiameter"] python=py_features["Maximum3DDiameter"]
        end
    end

    @testset "Maximum 2D Diameters" begin
        mask = create_ellipsoid_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), (10.0, 7.0, 5.0))
        julia_features = get_julia_shape3d(mask)
        py_features = get_pyradiomics_shape3d(mask)

        for feature_name in ["Maximum2DDiameterSlice", "Maximum2DDiameterColumn", "Maximum2DDiameterRow"]
            result = test_shape_feature(julia_features, py_features, feature_name;
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
        end
    end

    @testset "PCA-based Axis Lengths" begin
        @testset "Spherical mask" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            for feature_name in ["MajorAxisLength", "MinorAxisLength", "LeastAxisLength"]
                result = test_shape_feature(julia_features, py_features, feature_name;
                                            rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
                @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
            end
        end

        @testset "Ellipsoidal mask" begin
            mask = create_ellipsoid_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), (10.0, 5.0, 3.0))
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            for feature_name in ["MajorAxisLength", "MinorAxisLength", "LeastAxisLength"]
                result = test_shape_feature(julia_features, py_features, feature_name;
                                            rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
                @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
            end
        end
    end

    @testset "Elongation and Flatness" begin
        @testset "Spherical mask (≈ 1)" begin
            mask = create_sphere_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), 8.0)
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            for feature_name in ["Elongation", "Flatness"]
                result = test_shape_feature(julia_features, py_features, feature_name;
                                            rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
                @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
            end
        end

        @testset "Elongated mask" begin
            mask = create_ellipsoid_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), (10.0, 4.0, 4.0))
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Elongation";
                                        rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
            @test result === true || @info "Elongation" julia=julia_features["Elongation"] python=py_features["Elongation"]
            @test julia_features["Elongation"] < 1.0  # Should be elongated
        end

        @testset "Flat mask" begin
            mask = create_ellipsoid_mask_3d(SIZE_3D_MEDIUM, (12.0, 12.0, 12.0), (8.0, 8.0, 3.0))
            julia_features = get_julia_shape3d(mask)
            py_features = get_pyradiomics_shape3d(mask)

            result = test_shape_feature(julia_features, py_features, "Flatness";
                                        rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
            @test result === true || @info "Flatness" julia=julia_features["Flatness"] python=py_features["Flatness"]
            @test julia_features["Flatness"] < 1.0  # Should be flat
        end
    end

    @testset "Non-isotropic Spacing" begin
        spacing = (1.0, 1.5, 2.0)  # Anisotropic voxels
        mask = create_sphere_mask_3d((16, 16, 16), (8.0, 8.0, 8.0), 5.0)
        julia_features = get_julia_shape3d(mask; spacing=spacing)
        py_features = get_pyradiomics_shape3d(mask; spacing=spacing)

        # Test a subset of features with non-isotropic spacing
        for feature_name in ["VoxelVolume", "MeshVolume", "SurfaceArea", "Sphericity"]
            rtol = feature_name == "VoxelVolume" ? 1e-10 : MESH_RTOL
            result = test_shape_feature(julia_features, py_features, feature_name; rtol=rtol)
            @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
        end
    end
end

#==============================================================================#
# 2D Shape Feature Tests
#==============================================================================#

@testset "2D Shape Features" begin

    @testset "PixelSurface (Exact)" begin
        @testset "Square mask" begin
            mask = create_square_mask_2d(SIZE_2D_MEDIUM, (10, 10), 15)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "PixelSurface";
                                        rtol=1e-10, atol=1e-12)
            @test result === true || @info "PixelSurface" julia=julia_features["PixelSurface"] python=py_features["PixelSurface"]
        end

        @testset "Circular mask" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "PixelSurface";
                                        rtol=1e-10, atol=1e-12)
            @test result === true || @info "PixelSurface" julia=julia_features["PixelSurface"] python=py_features["PixelSurface"]
        end
    end

    @testset "MeshSurface" begin
        @testset "Square mask" begin
            mask = create_square_mask_2d(SIZE_2D_MEDIUM, (10, 10), 15)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "MeshSurface";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MeshSurface" julia=julia_features["MeshSurface"] python=py_features["MeshSurface"]
        end

        @testset "Circular mask" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "MeshSurface";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MeshSurface" julia=julia_features["MeshSurface"] python=py_features["MeshSurface"]
        end
    end

    @testset "Perimeter" begin
        @testset "Square mask" begin
            mask = create_square_mask_2d(SIZE_2D_MEDIUM, (10, 10), 15)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Perimeter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Perimeter" julia=julia_features["Perimeter"] python=py_features["Perimeter"]
        end

        @testset "Circular mask" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Perimeter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Perimeter" julia=julia_features["Perimeter"] python=py_features["Perimeter"]
        end
    end

    @testset "PerimeterSurfaceRatio" begin
        mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
        julia_features = get_julia_shape2d(mask)
        py_features = get_pyradiomics_shape2d(mask)

        result = test_shape_feature(julia_features, py_features, "PerimeterSurfaceRatio";
                                    rtol=MESH_RTOL, atol=MESH_ATOL)
        @test result === true || @info "PerimeterSurfaceRatio" julia=julia_features["PerimeterSurfaceRatio"] python=py_features["PerimeterSurfaceRatio"]
    end

    @testset "Sphericity (Circularity)" begin
        @testset "Circular mask (should be close to 1.0)" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Sphericity";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Sphericity" julia=julia_features["Sphericity"] python=py_features["Sphericity"]
        end

        @testset "Square mask" begin
            mask = create_square_mask_2d(SIZE_2D_MEDIUM, (10, 10), 15)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Sphericity";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "Sphericity" julia=julia_features["Sphericity"] python=py_features["Sphericity"]
        end
    end

    @testset "MaximumDiameter" begin
        @testset "Circular mask" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "MaximumDiameter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MaximumDiameter" julia=julia_features["MaximumDiameter"] python=py_features["MaximumDiameter"]
        end

        @testset "Elliptical mask" begin
            mask = create_ellipse_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), (15.0, 8.0))
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "MaximumDiameter";
                                        rtol=MESH_RTOL, atol=MESH_ATOL)
            @test result === true || @info "MaximumDiameter" julia=julia_features["MaximumDiameter"] python=py_features["MaximumDiameter"]
        end
    end

    @testset "PCA-based Axis Lengths" begin
        @testset "Circular mask" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            for feature_name in ["MajorAxisLength", "MinorAxisLength"]
                result = test_shape_feature(julia_features, py_features, feature_name;
                                            rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
                @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
            end
        end

        @testset "Elliptical mask" begin
            mask = create_ellipse_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), (15.0, 6.0))
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            for feature_name in ["MajorAxisLength", "MinorAxisLength"]
                result = test_shape_feature(julia_features, py_features, feature_name;
                                            rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
                @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
            end
        end
    end

    @testset "Elongation" begin
        @testset "Circular mask (≈ 1)" begin
            mask = create_circle_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), 12.0)
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Elongation";
                                        rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
            @test result === true || @info "Elongation" julia=julia_features["Elongation"] python=py_features["Elongation"]
        end

        @testset "Elongated mask" begin
            mask = create_ellipse_mask_2d(SIZE_2D_MEDIUM, (20.0, 20.0), (15.0, 5.0))
            julia_features = get_julia_shape2d(mask)
            py_features = get_pyradiomics_shape2d(mask)

            result = test_shape_feature(julia_features, py_features, "Elongation";
                                        rtol=SHAPE_RTOL, atol=SHAPE_ATOL)
            @test result === true || @info "Elongation" julia=julia_features["Elongation"] python=py_features["Elongation"]
            @test julia_features["Elongation"] < 1.0  # Should be elongated
        end
    end

    @testset "Non-isotropic Spacing" begin
        spacing = (1.0, 2.0)  # Anisotropic pixels
        mask = create_circle_mask_2d((20, 20), (10.0, 10.0), 6.0)
        julia_features = get_julia_shape2d(mask; spacing=spacing)
        py_features = get_pyradiomics_shape2d(mask; spacing=spacing)

        # Test a subset of features with non-isotropic spacing
        for feature_name in ["PixelSurface", "MeshSurface", "Perimeter", "Sphericity"]
            rtol = feature_name == "PixelSurface" ? 1e-10 : MESH_RTOL
            result = test_shape_feature(julia_features, py_features, feature_name; rtol=rtol)
            @test result === true || @info feature_name julia=julia_features[feature_name] python=py_features[feature_name]
        end
    end
end

#==============================================================================#
# Edge Cases
#==============================================================================#

@testset "Edge Cases" begin

    @testset "Single voxel mask (3D)" begin
        mask = falses(10, 10, 10)
        mask[5, 5, 5] = true

        # Should not crash
        julia_features = get_julia_shape3d(mask)

        # VoxelVolume should be 1.0
        @test julia_features["VoxelVolume"] == 1.0
    end

    @testset "Single pixel mask (2D)" begin
        mask = falses(10, 10)
        mask[5, 5] = true

        # Should not crash
        julia_features = get_julia_shape2d(mask)

        # PixelSurface should be 1.0
        @test julia_features["PixelSurface"] == 1.0
    end

    @testset "Small cubic mask (2x2x2)" begin
        mask = falses(10, 10, 10)
        mask[4:5, 4:5, 4:5] .= true

        julia_features = get_julia_shape3d(mask)

        # VoxelVolume should be 8.0
        @test julia_features["VoxelVolume"] == 8.0
    end

    @testset "Full mask" begin
        size = (8, 8, 8)
        mask = trues(size...)

        julia_features = get_julia_shape3d(mask)

        # VoxelVolume should be total voxels
        @test julia_features["VoxelVolume"] == prod(size)

        # Elongation should be close to 1.0 (isotropic)
        @test isapprox(julia_features["Elongation"], 1.0; rtol=0.01)
    end
end

#==============================================================================#
# Feature Count Verification
#==============================================================================#

@testset "Feature Count Verification" begin
    # Verify we have the expected number of features
    @test length(shape_3d_feature_names()) == 17
    @test length(shape_2d_feature_names()) == 10

    # Verify IBSI-compliant features (excludes deprecated)
    @test length(shape_3d_ibsi_features()) == 14
    @test length(shape_2d_ibsi_features()) == 9
end
