# Batch Processing Example
# Demonstrates how to process multiple images/ROIs efficiently with Radiomics.jl.

using Radiomics
using Random
using Printf

# ==============================================================================
# Setup: Generate simulated dataset
# ==============================================================================

println("=" ^ 60)
println("Generating Simulated Dataset")
println("=" ^ 60)

# Simulate a dataset with multiple subjects, each having an image and mask
function generate_subject_data(seed::Int; size=(48, 48, 32))
    rng = MersenneTwister(seed)

    # Generate image with variable intensity patterns
    image = rand(rng, size...) .* 200 .+ 50  # Values 50-250

    # Add some structure
    cx, cy, cz = size .÷ 2
    for i in 1:size[1], j in 1:size[2], k in 1:size[3]
        d = sqrt((i-cx)^2 + (j-cy)^2 + (k-cz)^2)
        if d < min(size...) / 3
            image[i, j, k] += rand(rng) * 100
        end
    end

    # Generate spherical mask
    mask = falses(size...)
    radius = min(size...) / 4 + rand(rng) * 4  # Vary radius slightly
    for i in 1:size[1], j in 1:size[2], k in 1:size[3]
        if sqrt((i-cx)^2 + (j-cy)^2 + (k-cz)^2) <= radius
            mask[i, j, k] = true
        end
    end

    return image, mask
end

# Generate dataset
n_subjects = 5
subjects = Dict{String, Tuple{Array{Float64,3}, BitArray{3}}}()

for i in 1:n_subjects
    subjects["Subject_$(lpad(i, 3, '0'))"] = generate_subject_data(1000 + i)
end

println("\nGenerated $(n_subjects) subjects")
for name in sort(collect(keys(subjects)))
    img, msk = subjects[name]
    println("  $name: image $(size(img)), ROI voxels: $(sum(msk))")
end

# ==============================================================================
# Example 1: Sequential Batch Processing
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 1: Sequential Batch Processing")
println("=" ^ 60)

# Process all subjects sequentially
results = Dict{String, Dict{String, Float64}}()

@time begin
    for (name, (image, mask)) in subjects
        features = extract_all(image, mask)
        results[name] = features
    end
end

println("\nProcessed $(length(results)) subjects")
println("Features per subject: $(length(first(values(results))))")

# ==============================================================================
# Example 2: Batch Processing with Progress Tracking
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 2: Batch Processing with Progress")
println("=" ^ 60)

function batch_extract_with_progress(subjects::Dict;
                                      feature_classes=nothing,
                                      binwidth=25.0)
    results = Dict{String, Dict{String, Float64}}()
    subject_names = sort(collect(keys(subjects)))
    n_total = length(subject_names)

    # Create extractor
    if isnothing(feature_classes)
        extractor = RadiomicsFeatureExtractor(settings=Settings(binwidth=binwidth))
    else
        extractor = RadiomicsFeatureExtractor(
            enabled_classes=Set(feature_classes),
            settings=Settings(binwidth=binwidth)
        )
    end

    for (idx, name) in enumerate(subject_names)
        image, mask = subjects[name]

        print("\rProcessing $name ($idx/$n_total)...")

        try
            features = extract(extractor, image, mask)
            results[name] = features
        catch e
            println("\n  Error processing $name: $e")
            results[name] = Dict{String, Float64}()
        end
    end

    println("\rProcessing complete!                    ")
    return results
end

results_tracked = batch_extract_with_progress(subjects)
println("Successfully processed: $(count(r -> !isempty(r), values(results_tracked))) subjects")

# ==============================================================================
# Example 3: Extracting Specific Features Only
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 3: Extracting Specific Feature Classes")
println("=" ^ 60)

# Only first-order and shape (faster for large batches)
results_fo_shape = batch_extract_with_progress(
    subjects,
    feature_classes=[FirstOrder, Shape]
)

println("Features extracted: $(length(first(values(results_fo_shape))))")
println("  (FirstOrder + Shape only)")

# ==============================================================================
# Example 4: Results to Table Format
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 4: Results to Table Format")
println("=" ^ 60)

function results_to_table(results::Dict{String, Dict{String, Float64}};
                          features::Union{Vector{String}, Nothing}=nothing)
    subject_names = sort(collect(keys(results)))

    if isnothing(features)
        # Get all feature names from first non-empty result
        for name in subject_names
            if !isempty(results[name])
                features = sort(collect(keys(results[name])))
                break
            end
        end
    end

    # Create header
    println("\n", "-" ^ 80)
    @printf("%-15s", "Subject")
    for feat in features[1:min(4, length(features))]
        name = split(feat, "_")[2][1:min(12, length(split(feat, "_")[2]))]
        @printf("%14s", name)
    end
    println("  ...")
    println("-" ^ 80)

    # Print rows
    for name in subject_names
        @printf("%-15s", name)
        for feat in features[1:min(4, length(features))]
            value = get(results[name], feat, NaN)
            @printf("%14.4f", value)
        end
        println("  ...")
    end
    println("-" ^ 80)
end

results_to_table(results)

# ==============================================================================
# Example 5: Feature Statistics Across Subjects
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 5: Feature Statistics Across Subjects")
println("=" ^ 60)

function compute_feature_statistics(results::Dict{String, Dict{String, Float64}})
    # Get feature names
    feature_names = String[]
    for (_, features) in results
        if !isempty(features)
            feature_names = collect(keys(features))
            break
        end
    end

    stats = Dict{String, NamedTuple{(:mean, :std, :min, :max), Tuple{Float64, Float64, Float64, Float64}}}()

    for feat in feature_names
        values = Float64[]
        for (_, features) in results
            if haskey(features, feat)
                push!(values, features[feat])
            end
        end

        if !isempty(values)
            μ = sum(values) / length(values)
            σ = sqrt(sum((v - μ)^2 for v in values) / length(values))
            stats[feat] = (mean=μ, std=σ, min=minimum(values), max=maximum(values))
        end
    end

    return stats
end

stats = compute_feature_statistics(results)

# Display statistics for selected features
println("\nFeature Statistics ($(n_subjects) subjects):")
println("-" ^ 70)
@printf("%-35s %12s %12s %12s\n", "Feature", "Mean", "Std", "Range")
println("-" ^ 70)

selected_features = [
    "firstorder_Mean",
    "firstorder_Entropy",
    "shape_VoxelVolume",
    "shape_Sphericity",
    "glcm_Contrast",
    "glcm_Correlation"
]

for feat in selected_features
    if haskey(stats, feat)
        s = stats[feat]
        @printf("%-35s %12.4f %12.4f %6.2f-%.2f\n",
                feat, s.mean, s.std, s.min, s.max)
    end
end
println("-" ^ 70)

# ==============================================================================
# Example 6: Export to CSV-like Format
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 6: Export to CSV Format")
println("=" ^ 60)

function results_to_csv_string(results::Dict{String, Dict{String, Float64}})
    subject_names = sort(collect(keys(results)))

    # Get feature names
    feature_names = String[]
    for name in subject_names
        if !isempty(results[name])
            feature_names = sort(collect(keys(results[name])))
            break
        end
    end

    lines = String[]

    # Header
    push!(lines, "Subject," * join(feature_names, ","))

    # Data rows
    for name in subject_names
        values = [string(get(results[name], feat, "NaN")) for feat in feature_names]
        push!(lines, name * "," * join(values, ","))
    end

    return join(lines, "\n")
end

csv_output = results_to_csv_string(results)

# Show first few lines
println("\nCSV output preview (first 5 lines):")
for (i, line) in enumerate(split(csv_output, "\n"))
    if i <= 5
        # Truncate long lines
        println(length(line) > 80 ? line[1:77] * "..." : line)
    end
end
println("...")

# To save to file:
# open("radiomics_features.csv", "w") do f
#     write(f, csv_output)
# end

# ==============================================================================
# Example 7: Processing Multiple ROIs per Image
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 7: Multiple ROIs per Image")
println("=" ^ 60)

# Generate image with multiple labeled ROIs
Random.seed!(42)
multi_roi_image = rand(50, 50, 40) .* 200

# Create multi-label mask
multi_roi_mask = zeros(Int, 50, 50, 40)

# ROI 1: Upper left sphere
for i in 1:50, j in 1:50, k in 1:40
    if sqrt((i-15)^2 + (j-15)^2 + (k-20)^2) <= 8
        multi_roi_mask[i, j, k] = 1
    end
end

# ROI 2: Upper right sphere
for i in 1:50, j in 1:50, k in 1:40
    if sqrt((i-35)^2 + (j-15)^2 + (k-20)^2) <= 10
        multi_roi_mask[i, j, k] = 2
    end
end

# ROI 3: Lower center sphere
for i in 1:50, j in 1:50, k in 1:40
    if sqrt((i-25)^2 + (j-35)^2 + (k-20)^2) <= 7
        multi_roi_mask[i, j, k] = 3
    end
end

println("Multi-ROI image: $(size(multi_roi_image))")
println("Labels present: $(sort(unique(multi_roi_mask[multi_roi_mask .> 0])))")

# Extract features for each ROI
roi_results = Dict{Int, Dict{String, Float64}}()

for label in 1:3
    println("Processing ROI $label ($(sum(multi_roi_mask .== label)) voxels)...")
    roi_results[label] = extract_all(multi_roi_image, multi_roi_mask; label=label)
end

# Compare ROIs
println("\nComparison of ROIs:")
println("-" ^ 60)
@printf("%-25s %12s %12s %12s\n", "Feature", "ROI 1", "ROI 2", "ROI 3")
println("-" ^ 60)

compare_features = ["firstorder_Mean", "shape_VoxelVolume", "glcm_Contrast"]
for feat in compare_features
    @printf("%-25s %12.4f %12.4f %12.4f\n",
            feat, roi_results[1][feat], roi_results[2][feat], roi_results[3][feat])
end
println("-" ^ 60)

# ==============================================================================
# Example 8: Memory-Efficient Processing
# ==============================================================================

println("\n" * "=" ^ 60)
println("Example 8: Memory-Efficient Processing Pattern")
println("=" ^ 60)

# For very large datasets, process and save immediately
function process_and_save_pattern(subject_generator, n_subjects::Int, output_fn)
    """
    Pattern for memory-efficient batch processing:
    1. Generate/load one subject at a time
    2. Extract features
    3. Save immediately
    4. Release memory
    """
    extractor = RadiomicsFeatureExtractor()

    for i in 1:n_subjects
        # In real code: load image from disk
        image, mask = subject_generator(i)

        # Extract
        features = extract(extractor, image, mask)

        # Save (in real code: append to file or database)
        output_fn(i, features)

        # Memory is released when image/mask go out of scope
    end
end

# Demo with simple generator and printer
saved_count = Ref(0)
process_and_save_pattern(
    i -> generate_subject_data(2000 + i, size=(30, 30, 20)),
    3,
    (i, features) -> begin
        saved_count[] += 1
        println("  Processed subject $i: $(length(features)) features")
    end
)
println("Processed $(saved_count[]) subjects with minimal memory footprint")

# ==============================================================================
# Summary
# ==============================================================================

println("\n" * "=" ^ 60)
println("Batch Processing Summary")
println("=" ^ 60)
println("""
Key patterns demonstrated:

1. Sequential Processing
   - Simple loop over subjects
   - Use Dict to store results

2. Progress Tracking
   - Print progress during processing
   - Handle errors gracefully

3. Selective Feature Extraction
   - Enable only needed feature classes
   - Faster for large batches

4. Results to Table
   - Convert Dict results to tabular format
   - Export to CSV for analysis

5. Feature Statistics
   - Compute mean, std, range across subjects
   - Quality control and outlier detection

6. Multiple ROIs per Image
   - Use integer label masks
   - Specify label parameter

7. Memory-Efficient Processing
   - Load/process/save one at a time
   - Suitable for very large datasets

Tips for Production:
- Pre-compute extractor outside loops
- Use try/catch for error handling
- Log failures for review
- Consider parallel processing (Threads.@threads)
""")

println("\nExample completed successfully!")
