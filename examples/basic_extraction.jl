# Basic Feature Extraction Example
# Demonstrates the fundamental usage of Radiomics.jl for extracting radiomic features.

using Radiomics
using Random

# ==============================================================================
# Example 1: Simple Feature Extraction
# ==============================================================================

println("=" ^ 60)
println("Example 1: Simple Feature Extraction")
println("=" ^ 60)

# Create a simple 3D image with random intensities
Random.seed!(42)
image = rand(32, 32, 32) .* 100  # Random values from 0-100

# Create a spherical mask (ROI) in the center
mask = falses(32, 32, 32)
center = (16, 16, 16)
radius = 10
for i in 1:32, j in 1:32, k in 1:32
    if sqrt((i - center[1])^2 + (j - center[2])^2 + (k - center[3])^2) <= radius
        mask[i, j, k] = true
    end
end

# Extract all features using the convenience function
features = extract_all(image, mask)

# Display some results
println("\nExtracted $(length(features)) features")
println("\nSample first-order features:")
println("  Energy: $(features["firstorder_Energy"])")
println("  Mean: $(features["firstorder_Mean"])")
println("  Entropy: $(features["firstorder_Entropy"])")
println("  Skewness: $(features["firstorder_Skewness"])")

println("\nSample shape features:")
println("  VoxelVolume: $(features["shape_VoxelVolume"])")
println("  Sphericity: $(features["shape_Sphericity"])")
println("  MajorAxisLength: $(features["shape_MajorAxisLength"])")

println("\nSample GLCM features:")
println("  Contrast: $(features["glcm_Contrast"])")
println("  Correlation: $(features["glcm_Correlation"])")
println("  JointEntropy: $(features["glcm_JointEntropy"])")

# ==============================================================================
# Example 2: Using the Feature Extractor
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 2: Using RadiomicsFeatureExtractor")
println("=" ^ 60)

# Create a feature extractor with default settings
extractor = RadiomicsFeatureExtractor()

# Display extractor configuration
println("\nExtractor configuration:")
println(extractor)

# Extract features
features2 = extract(extractor, image, mask)
println("\nExtracted $(length(features2)) features")

# ==============================================================================
# Example 3: Extracting Only Specific Feature Classes
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 3: Specific Feature Classes")
println("=" ^ 60)

# Create extractor with only first-order and shape features
extractor_limited = RadiomicsFeatureExtractor(
    enabled_classes=Set([FirstOrder, Shape])
)
println("\nExtractor with limited classes:")
println(extractor_limited)

features_limited = extract(extractor_limited, image, mask)
println("\nExtracted $(length(features_limited)) features (first-order + shape only)")

# Alternative: disable unwanted classes
extractor_no_texture = RadiomicsFeatureExtractor()
disable!(extractor_no_texture, [GLCM, GLRLM, GLSZM, NGTDM, GLDM])
features_no_texture = extract(extractor_no_texture, image, mask)
println("Without texture: $(length(features_no_texture)) features")

# ==============================================================================
# Example 4: Working with Voxel Spacing
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 4: Voxel Spacing")
println("=" ^ 60)

# Extract with specific voxel spacing (e.g., 1mm x 1mm x 2.5mm slices)
spacing = (1.0, 1.0, 2.5)
features_spaced = extract_all(image, mask; spacing=spacing)

println("\nWith voxel spacing $spacing:")
println("  VoxelVolume: $(features_spaced["shape_VoxelVolume"]) mm³")
println("  SurfaceArea: $(features_spaced["shape_SurfaceArea"]) mm²")

# Compare with default unit spacing
println("\nWith unit spacing (1.0, 1.0, 1.0):")
println("  VoxelVolume: $(features["shape_VoxelVolume"]) voxels³")
println("  SurfaceArea: $(features["shape_SurfaceArea"]) voxels²")

# ==============================================================================
# Example 5: Integer Label Masks
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 5: Integer Label Masks")
println("=" ^ 60)

# Multi-label mask (e.g., from segmentation software)
label_mask = zeros(Int, 32, 32, 32)
# Label 1: small sphere
for i in 1:32, j in 1:32, k in 1:32
    d = sqrt((i - 10)^2 + (j - 10)^2 + (k - 16)^2)
    if d <= 5
        label_mask[i, j, k] = 1
    end
end
# Label 2: larger sphere
for i in 1:32, j in 1:32, k in 1:32
    d = sqrt((i - 22)^2 + (j - 22)^2 + (k - 16)^2)
    if d <= 8
        label_mask[i, j, k] = 2
    end
end

# Extract features for label 1
features_label1 = extract_all(image, label_mask; label=1)
println("\nFeatures for label 1 (small sphere):")
println("  VoxelVolume: $(features_label1["shape_VoxelVolume"])")

# Extract features for label 2
features_label2 = extract_all(image, label_mask; label=2)
println("\nFeatures for label 2 (larger sphere):")
println("  VoxelVolume: $(features_label2["shape_VoxelVolume"])")

# ==============================================================================
# Example 6: Convenience Functions for Single Classes
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 6: Single-Class Extraction")
println("=" ^ 60)

# Extract only first-order features
fo_features = extract_firstorder_only(image, mask)
println("\nFirst-order only: $(length(fo_features)) features")

# Extract only shape features (doesn't need intensity image)
shape_features = extract_shape_only(mask)
println("Shape only: $(length(shape_features)) features")

# Extract only texture features
texture_features = extract_texture_only(image, mask)
println("Texture only: $(length(texture_features)) features (GLCM+GLRLM+GLSZM+NGTDM+GLDM)")

# ==============================================================================
# Summary
# ==============================================================================

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("""
Key functions demonstrated:
  - extract_all(image, mask; kwargs...)     # All features, simple API
  - extract(extractor, image, mask)         # Full control via extractor
  - extract_firstorder_only(image, mask)    # First-order features only
  - extract_shape_only(mask)                # Shape features only
  - extract_texture_only(image, mask)       # All texture features

Feature classes available:
  - FirstOrder (19 features)
  - Shape (17 features for 3D, 10 for 2D)
  - GLCM (24 features)
  - GLRLM (16 features)
  - GLSZM (16 features)
  - NGTDM (5 features)
  - GLDM (14 features)

Total: 111 features for 3D images
""")

println("\nExample completed successfully!")
