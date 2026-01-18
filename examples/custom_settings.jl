# Custom Settings Example
# Demonstrates how to configure radiomic feature extraction with various settings.

using Radiomics
using Random
using Printf

# ==============================================================================
# Setup: Create test data
# ==============================================================================

Random.seed!(42)

# Create a realistic-looking medical image (simulating CT HU values)
# Range: -1000 (air) to 3000 (bone/metal)
image = zeros(Float64, 40, 40, 40)

# Background (air-like, around -1000)
image .= -900 .+ randn(40, 40, 40) .* 50

# Add some tissue (soft tissue HU around 30-50)
for i in 10:30, j in 10:30, k in 10:30
    image[i, j, k] = 40 + randn() * 20
end

# Add a denser region (like calcification, HU > 100)
for i in 18:22, j in 18:22, k in 18:22
    image[i, j, k] = 200 + randn() * 30
end

# Create mask for the tissue region
mask = falses(40, 40, 40)
mask[10:30, 10:30, 10:30] .= true

println("Test image created: $(size(image))")
println("Image range: $(minimum(image)) to $(maximum(image))")
println("Voxels in ROI: $(sum(mask))")

# ==============================================================================
# Example 1: Default Settings
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 1: Default Settings")
println("=" ^ 60)

settings_default = Settings()
extractor_default = RadiomicsFeatureExtractor(settings=settings_default)

println("\nDefault settings:")
println("  binwidth: $(settings_default.binwidth)")
println("  bincount: $(settings_default.bincount)")
println("  glcm_distance: $(settings_default.glcm_distance)")
println("  symmetrical_glcm: $(settings_default.symmetrical_glcm)")
println("  gldm_alpha: $(settings_default.gldm_alpha)")
println("  ngtdm_distance: $(settings_default.ngtdm_distance)")

features_default = extract(extractor_default, image, mask)
println("\nExtracted $(length(features_default)) features")

# ==============================================================================
# Example 2: Custom Bin Width
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 2: Custom Bin Width")
println("=" ^ 60)

# Smaller bin width = more gray levels = finer texture analysis
settings_fine = Settings(binwidth=10.0)
features_fine = extract_all(image, mask; binwidth=10.0)

# Larger bin width = fewer gray levels = coarser texture
settings_coarse = Settings(binwidth=50.0)
features_coarse = extract_all(image, mask; binwidth=50.0)

println("\nEffect of bin width on texture features:")
println("\n                    binwidth=10  binwidth=25  binwidth=50")
println("-" ^ 60)

for feat in ["glcm_JointEntropy", "glcm_Contrast", "glrlm_RunEntropy"]
    v_fine = features_fine[feat]
    v_default = features_default[feat]
    v_coarse = features_coarse[feat]
    name = split(feat, "_")[2]
    @printf("  %-16s  %10.4f   %10.4f   %10.4f\n", name, v_fine, v_default, v_coarse)
end

println("\nSmaller binwidth → more gray levels → higher entropy")
println("Larger binwidth → fewer gray levels → lower entropy")

# ==============================================================================
# Example 3: Fixed Bin Count Mode
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 3: Fixed Bin Count Mode")
println("=" ^ 60)

# Use exactly 64 bins regardless of intensity range
settings_bincount = Settings(bincount=64)
features_bincount = extract_all(image, mask; bincount=64)

println("\nWith fixed bincount=64:")
println("  JointEntropy: $(features_bincount["glcm_JointEntropy"])")

# Use 32 bins
features_bincount32 = extract_all(image, mask; bincount=32)
println("\nWith fixed bincount=32:")
println("  JointEntropy: $(features_bincount32["glcm_JointEntropy"])")

# ==============================================================================
# Example 4: GLCM Distance Parameter
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 4: GLCM Distance Parameter")
println("=" ^ 60)

# Distance = 1 (immediate neighbors, default)
features_d1 = extract_all(image, mask; glcm_distance=1)

# Distance = 2 (neighbors 2 voxels apart)
features_d2 = extract_all(image, mask; glcm_distance=2)

println("\nGLCM features at different distances:")
println("\n                    distance=1   distance=2")
println("-" ^ 50)
for feat in ["glcm_Contrast", "glcm_Correlation", "glcm_Idm"]
    v1 = features_d1[feat]
    v2 = features_d2[feat]
    name = split(feat, "_")[2]
    @printf("  %-16s  %10.4f   %10.4f\n", name, v1, v2)
end

println("\nLarger distance captures coarser texture patterns")

# ==============================================================================
# Example 5: GLDM Alpha Parameter
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 5: GLDM Alpha (Coarseness) Parameter")
println("=" ^ 60)

# Alpha = 0 (default, exact matching)
features_a0 = extract_all(image, mask; gldm_alpha=0.0)

# Alpha = 4 (more tolerant matching)
features_a4 = extract_all(image, mask; gldm_alpha=4.0)

println("\nGLDM features with different alpha:")
println("\n                         alpha=0      alpha=4")
println("-" ^ 55)
for feat in ["gldm_SmallDependenceEmphasis", "gldm_LargeDependenceEmphasis",
             "gldm_DependenceEntropy"]
    v0 = features_a0[feat]
    v4 = features_a4[feat]
    name = split(feat, "_")[2]
    @printf("  %-22s  %10.4f   %10.4f\n", name, v0, v4)
end

println("\nHigher alpha = more voxels counted as dependent = larger zones")

# ==============================================================================
# Example 6: NGTDM Distance Parameter
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 6: NGTDM Distance Parameter")
println("=" ^ 60)

# Distance = 1 (immediate neighborhood)
features_nd1 = extract_all(image, mask; ngtdm_distance=1)

# Distance = 2 (larger neighborhood)
features_nd2 = extract_all(image, mask; ngtdm_distance=2)

println("\nNGTDM features at different distances:")
println("\n                    distance=1   distance=2")
println("-" ^ 50)
for feat in ["ngtdm_Coarseness", "ngtdm_Contrast", "ngtdm_Busyness"]
    v1 = features_nd1[feat]
    v2 = features_nd2[feat]
    name = split(feat, "_")[2]
    @printf("  %-16s  %10.6f   %10.6f\n", name, v1, v2)
end

# ==============================================================================
# Example 7: Combining Multiple Settings
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 7: Combined Custom Settings")
println("=" ^ 60)

# Settings optimized for a specific use case
# (e.g., high-resolution analysis of small structures)
custom_settings = Settings(
    binwidth=15.0,          # Finer gray level discretization
    glcm_distance=1,        # Immediate neighbors (appropriate for small ROI)
    symmetrical_glcm=true,  # Standard GLCM symmetry
    gldm_alpha=0.0,         # Strict dependence matching
    ngtdm_distance=1        # Small neighborhood
)

extractor_custom = RadiomicsFeatureExtractor(settings=custom_settings)
println("\nCustom extractor:")
println(extractor_custom)

features_custom = extract(extractor_custom, image, mask)
println("\nExtracted $(length(features_custom)) features with custom settings")

# ==============================================================================
# Example 8: PyRadiomics Compatibility Settings
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 8: PyRadiomics Compatibility")
println("=" ^ 60)

# Settings that match PyRadiomics defaults exactly
pyradiomics_compat = Settings(
    binwidth=25.0,
    glcm_distance=1,
    symmetrical_glcm=true,
    gldm_alpha=0.0,
    ngtdm_distance=1,
    voxel_array_shift=0  # PyRadiomics internal detail
)

println("\nPyRadiomics-compatible settings:")
println("  These settings ensure 1:1 parity with PyRadiomics output")

features_compat = extract_all(image, mask; binwidth=25.0)
println("\nExtracted features will match PyRadiomics output within floating-point tolerance")

# ==============================================================================
# Example 9: Accessing Individual Settings Properties
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 9: Working with Settings Object")
println("=" ^ 60)

# Create settings
s = Settings(binwidth=20.0, glcm_distance=2)

# Access properties
println("\nSettings properties:")
println("  binwidth: $(s.binwidth)")
println("  bincount: $(isnothing(s.bincount) ? "not set" : s.bincount)")
println("  glcm_distance: $(s.glcm_distance)")
println("  symmetrical_glcm: $(s.symmetrical_glcm)")
println("  gldm_alpha: $(s.gldm_alpha)")
println("  ngtdm_distance: $(s.ngtdm_distance)")
println("  discretization_mode: $(s.discretization_mode)")

# Note: Settings is immutable, create a new one to change values
s_modified = Settings(
    binwidth=s.binwidth * 2,  # Double the bin width
    glcm_distance=s.glcm_distance,
    gldm_alpha=s.gldm_alpha
)
println("\nModified binwidth: $(s_modified.binwidth)")

# ==============================================================================
# Summary Table
# ==============================================================================

println("\n" * "=" ^ 60)
println("Settings Summary")
println("=" ^ 60)
println("""
Key settings and their effects:

Parameter           Default    Effect
---------           -------    ------
binwidth            25.0       Gray level discretization width
                               Smaller = more gray levels = finer texture

bincount            nothing    Fixed number of bins (overrides binwidth)
                               Use for normalized discretization

glcm_distance       1          Pixel offset for co-occurrence
                               Larger = coarser texture patterns

symmetrical_glcm    true       Make GLCM symmetric (P + P')
                               Standard for most applications

gldm_alpha          0.0        Dependence tolerance
                               Larger = more tolerant matching

ngtdm_distance      1          Neighborhood size for NGTDM
                               Larger = larger context

discretization_mode FixedBinWidth
                               FixedBinWidth or FixedBinCount

Tip: For PyRadiomics compatibility, use default settings (binwidth=25.0)
""")

println("\nExample completed successfully!")
