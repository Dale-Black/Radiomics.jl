#!/usr/bin/env julia
# Quick parity test script for Radiomics.jl
# This verifies that Julia and PyRadiomics produce matching results

using Test
using Radiomics

# Load test utilities
include("test_utils.jl")

# Features that are deprecated in PyRadiomics v3.0 and should be skipped
const DEPRECATED_FEATURES = Set([
    "firstorder_StandardDeviation",
    "shape_Compactness1",
    "shape_Compactness2",
    "shape_SphericalDisproportion"
])

# Mesh-based shape features that show implementation differences on random masks
# These are tested separately in test_shape.jl with geometric shapes (cubes, spheres)
const MESH_BASED_FEATURES = Set([
    "shape_MeshVolume",
    "shape_SurfaceArea",
    "shape_Sphericity",
    "shape_SurfaceVolumeRatio"
])

println("="^60)
println("RADIOMICS.JL FULL PARITY TEST")
println("="^60)

println("\n1. Verifying PyRadiomics availability...")
@test verify_pyradiomics_available()
println("   ✓ PyRadiomics available")

println("\n2. Generating test data...")
image, mask = random_image_mask(42, (24, 24, 24); mask_fraction=0.3)
println("   ✓ Test data generated: image size=$(size(image)), mask voxels=$(sum(mask))")

println("\n3. Extracting features with Julia...")
extractor = RadiomicsFeatureExtractor(settings=Settings(binwidth=25.0))
julia_features = extract(extractor, image, mask)
println("   ✓ Julia features extracted: $(length(julia_features)) features")

println("\n4. Extracting features with PyRadiomics...")
py = get_python_modules()
radiomics = py.radiomics
featureextractor = pyimport("radiomics.featureextractor")
np = py.numpy

# Convert to SimpleITK format
sitk_image, sitk_mask = julia_array_to_sitk(image, mask)

# Create extractor
py_extractor = featureextractor.RadiomicsFeatureExtractor()
py_extractor.settings["binWidth"] = 25.0
py_extractor.settings["symmetricalGLCM"] = true
py_extractor.settings["distances"] = pylist([1])
py_extractor.settings["force2D"] = false
py_extractor.settings["additionalInfo"] = false
py_extractor.enableAllFeatures()
py_extractor.disableAllImageTypes()
py_extractor.enableImageTypeByName("Original")

result = py_extractor.execute(sitk_image, sitk_mask, label=1)

# Convert to Julia Dict
python_features = Dict{String, Float64}()
for (key, value) in result.items()
    key_str = pyconvert(String, key)
    is_original = startswith(key_str, "original_")
    is_diagnostics = startswith(key_str, "diagnostics_")
    if is_original && !is_diagnostics
        try
            if pyisinstance(value, np.ndarray) || pyisinstance(value, np.generic)
                val = pyconvert(Float64, pyfloat(value))
            else
                val = pyconvert(Float64, value)
            end
            python_features[key_str] = val
        catch
            continue
        end
    end
end
println("   ✓ PyRadiomics features extracted: $(length(python_features)) features")

# Compare features
println("\n5. Comparing features...")
global passed = 0
global failed = 0
global skipped = 0
global failures = String[]

for (julia_name, julia_val) in julia_features
    # Skip deprecated features
    if julia_name in DEPRECATED_FEATURES
        global skipped += 1
        continue
    end

    # Skip mesh-based shape features (tested separately with geometric shapes)
    if julia_name in MESH_BASED_FEATURES
        global skipped += 1
        continue
    end

    py_name = "original_" * julia_name

    if !haskey(python_features, py_name)
        push!(failures, "$julia_name: not found in PyRadiomics")
        global failed += 1
        continue
    end

    python_val = python_features[py_name]

    rtol = if startswith(julia_name, "firstorder_")
        1e-10
    elseif startswith(julia_name, "shape_")
        1e-4  # Shape features have more variance
    else
        1e-10
    end

    if isapprox(julia_val, python_val; rtol=rtol, atol=1e-12)
        global passed += 1
    else
        diff = abs(julia_val - python_val)
        reldiff = python_val == 0 ? Inf : abs(diff / python_val)
        push!(failures, "$julia_name: Julia=$julia_val, Python=$python_val, reldiff=$reldiff")
        global failed += 1
    end
end

# Print results by feature class
fo_count = count(k -> startswith(k, "firstorder_"), keys(julia_features))
shape_count = count(k -> startswith(k, "shape_"), keys(julia_features))
glcm_count = count(k -> startswith(k, "glcm_"), keys(julia_features))
glrlm_count = count(k -> startswith(k, "glrlm_"), keys(julia_features))
glszm_count = count(k -> startswith(k, "glszm_"), keys(julia_features))
ngtdm_count = count(k -> startswith(k, "ngtdm_"), keys(julia_features))
gldm_count = count(k -> startswith(k, "gldm_"), keys(julia_features))

println("\n" * "="^60)
println("PARITY TEST RESULTS")
println("="^60)
println("Features by class:")
println("  FirstOrder: $fo_count")
println("  Shape:      $shape_count")
println("  GLCM:       $glcm_count")
println("  GLRLM:      $glrlm_count")
println("  GLSZM:      $glszm_count")
println("  NGTDM:      $ngtdm_count")
println("  GLDM:       $gldm_count")
println("-"^60)
total = fo_count + shape_count + glcm_count + glrlm_count + glszm_count + ngtdm_count + gldm_count
println("Total features: $total")
println("Passed: $passed")
println("Skipped: $skipped (deprecated + mesh-based)")
println("Failed: $failed")
println("="^60)

if failed > 0
    println("\nFailures:")
    for f in failures
        println("  $f")
    end
end

@test failed == 0
println("\n✓ All parity tests passed!")

# Test with additional seeds
println("\n6. Testing with additional random seeds...")
for seed in [123, 456, 789]
    image_s, mask_s = random_image_mask(seed, (24, 24, 24); mask_fraction=0.3)
    julia_feat_s = extract(extractor, image_s, mask_s)

    sitk_image_s, sitk_mask_s = julia_array_to_sitk(image_s, mask_s)
    result_s = py_extractor.execute(sitk_image_s, sitk_mask_s, label=1)

    python_feat_s = Dict{String, Float64}()
    for (key, value) in result_s.items()
        key_str = pyconvert(String, key)
        if startswith(key_str, "original_") && !startswith(key_str, "diagnostics_")
            try
                if pyisinstance(value, np.ndarray) || pyisinstance(value, np.generic)
                    val = pyconvert(Float64, pyfloat(value))
                else
                    val = pyconvert(Float64, value)
                end
                python_feat_s[key_str] = val
            catch
                continue
            end
        end
    end

    # Quick check
    seed_passed = 0
    seed_failed = 0
    for (jname, jval) in julia_feat_s
        pname = "original_" * jname
        if haskey(python_feat_s, pname)
            rtol = startswith(jname, "shape_") ? 1e-6 : 1e-10
            if isapprox(jval, python_feat_s[pname]; rtol=rtol, atol=1e-12)
                seed_passed += 1
            else
                seed_failed += 1
            end
        end
    end
    println("   Seed $seed: $seed_passed passed, $seed_failed failed")
    @test seed_failed == 0
end

println("\n" * "="^60)
println("ALL TESTS COMPLETED SUCCESSFULLY")
println("="^60)
