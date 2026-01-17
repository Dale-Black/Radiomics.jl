# Shape Features for Radiomics.jl
#
# Shape features describe the geometric properties of the ROI (Region of Interest).
# They are computed from the mask geometry and do NOT depend on voxel gray level values.
#
# This module implements both 2D and 3D shape features.
# - 2D: 10 features (for single slices or 2D images)
# - 3D: 17 features (for volumetric images)
#
# References:
# - PyRadiomics: radiomics/shape.py, radiomics/shape2D.py
# - PyRadiomics C extension: radiomics/src/cshape.c
# - IBSI: https://ibsi.readthedocs.io/en/latest/03_Image_features.html
# - PyRadiomics docs: https://pyradiomics.readthedocs.io/en/latest/features.html

using LinearAlgebra
using Statistics

#==============================================================================#
# 2D Marching Squares Algorithm (Contour/Perimeter Generation)
#==============================================================================#

"""
    _marching_squares_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real})

Generate 2D contour vertices and edges from a binary mask using the Marching Squares algorithm.

# Arguments
- `mask`: 2D binary mask (Bool matrix)
- `spacing`: Pixel spacing as (row_spacing, column_spacing) in mm

# Returns
- `vertices`: Vector of vertex positions as (row, col) tuples in physical coordinates (mm)
- `edges`: Vector of (v1_idx, v2_idx) index pairs defining line segments

# Algorithm
The algorithm traverses the mask with a 2×2 sliding window. Each configuration of
4 corner pixels maps to 0, 1, or 2 line segments through a lookup table. Vertices
are placed at edge midpoints and scaled by pixel spacing.

# References
- PyRadiomics: radiomics/src/cshape.c (2D marching squares implementation)
"""
function _marching_squares_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real})
    rows, cols = size(mask)

    # Grid angles for 2×2 cell corners: [bottom-left, bottom-right, top-right, top-left]
    # Using (row, col) convention
    grid_angles = [(0, 0), (0, 1), (1, 1), (1, 0)]

    # Vertex positions for edge midpoints (relative to cell origin)
    # Edges: 0=bottom, 1=right, 2=top, 3=left
    vertex_list = [
        (0.0, 0.5),   # Edge 0: bottom (between corners 0-1)
        (0.5, 1.0),   # Edge 1: right (between corners 1-2)
        (1.0, 0.5),   # Edge 2: top (between corners 2-3)
        (0.5, 0.0)    # Edge 3: left (between corners 3-0)
    ]

    # Lookup table: index → [(edge1, edge2), ...] line segments
    # Index is 4-bit value from corner occupancy: bit i = corner i inside mask
    # -1 indicates no segment
    line_table = [
        [(-1, -1), (-1, -1)],  # 0: all outside
        [(0, 3), (-1, -1)],    # 1: corner 0 inside
        [(0, 1), (-1, -1)],    # 2: corner 1 inside
        [(1, 3), (-1, -1)],    # 3: corners 0,1 inside
        [(1, 2), (-1, -1)],    # 4: corner 2 inside
        [(0, 1), (2, 3)],      # 5: corners 0,2 inside (ambiguous - two segments)
        [(0, 2), (-1, -1)],    # 6: corners 1,2 inside
        [(2, 3), (-1, -1)],    # 7: corners 0,1,2 inside
        [(2, 3), (-1, -1)],    # 8: corner 3 inside
        [(0, 2), (-1, -1)],    # 9: corners 0,3 inside
        [(0, 3), (1, 2)],      # 10: corners 1,3 inside (ambiguous - two segments)
        [(1, 2), (-1, -1)],    # 11: corners 0,1,3 inside
        [(1, 3), (-1, -1)],    # 12: corners 2,3 inside
        [(0, 1), (-1, -1)],    # 13: corners 0,2,3 inside
        [(0, 3), (-1, -1)],    # 14: corners 1,2,3 inside
        [(-1, -1), (-1, -1)]   # 15: all inside
    ]

    vertices = NTuple{2,Float64}[]
    edges = NTuple{2,Int}[]

    # Pad mask with zeros to handle boundary
    padded = zeros(Bool, rows + 2, cols + 2)
    padded[2:end-1, 2:end-1] .= mask

    # Traverse with 2×2 sliding window
    for r in 1:(rows + 1)
        for c in 1:(cols + 1)
            # Compute cell index from corner occupancy
            idx = 0
            for (i, (dr, dc)) in enumerate(grid_angles)
                if padded[r + dr, c + dc]
                    idx += 2^(i - 1)
                end
            end

            # Skip if all corners same (no boundary)
            if idx == 0 || idx == 15
                continue
            end

            # Add line segments for this cell configuration
            for (e1, e2) in line_table[idx + 1]
                if e1 == -1
                    continue
                end

                # Get vertex positions for this edge pair
                v1_local = vertex_list[e1 + 1]
                v2_local = vertex_list[e2 + 1]

                # Convert to physical coordinates (accounting for padding offset)
                # Position is (row-1, col-1) to account for 0-indexed cells, minus 1 for padding
                v1 = ((r - 1 + v1_local[1] - 0.5) * spacing[1],
                      (c - 1 + v1_local[2] - 0.5) * spacing[2])
                v2 = ((r - 1 + v2_local[1] - 0.5) * spacing[1],
                      (c - 1 + v2_local[2] - 0.5) * spacing[2])

                # Add vertices and edge
                push!(vertices, v1)
                push!(vertices, v2)
                push!(edges, (length(vertices) - 1, length(vertices)))
            end
        end
    end

    return vertices, edges
end

#==============================================================================#
# 2D Shape Feature: Perimeter
#==============================================================================#

"""
    perimeter_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the perimeter of a 2D ROI using the Marching Squares contour.

# Mathematical Formula
```
P = Σᵢ √[(x_{i+1} - xᵢ)² + (y_{i+1} - yᵢ)²]
```
Sum of all line segment lengths along the contour.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Perimeter in mm (or pixels if spacing is (1.0, 1.0))

# Notes
- Uses Marching Squares algorithm to generate contour
- Line segments are placed at pixel edge midpoints
- Handles multiple disconnected regions (sums all perimeters)

# Example
```julia
mask = falses(10, 10)
mask[3:7, 3:7] .= true  # 5×5 square
p = perimeter_2d(mask)  # ≈ 20.0 pixels
```

# References
- PyRadiomics: radiomics/shape2D.py:getPerimeterFeatureValue
- IBSI: Perimeter
"""
function perimeter_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    vertices, edges = _marching_squares_2d(mask, spacing)

    if isempty(edges)
        return 0.0
    end

    # Sum lengths of all line segments
    perimeter = 0.0
    for (i1, i2) in edges
        v1 = vertices[i1]
        v2 = vertices[i2]
        perimeter += sqrt((v2[1] - v1[1])^2 + (v2[2] - v1[2])^2)
    end

    return perimeter
end

#==============================================================================#
# 2D Shape Feature: Mesh Surface (Area from contour)
#==============================================================================#

"""
    mesh_surface_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the surface area of a 2D ROI using the Marching Squares mesh.

# Mathematical Formula
```
A = ½ |Σᵢ (xᵢ × y_{i+1} - x_{i+1} × yᵢ)|
```
Shoelace formula for polygon area using signed triangle areas.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Area in mm² (or pixels² if spacing is (1.0, 1.0))

# Notes
- Uses signed area calculation from mesh triangles
- More accurate than pixel counting for smooth boundaries
- Handles holes correctly (subtracts inner contour areas)

# Example
```julia
mask = falses(10, 10)
mask[3:7, 3:7] .= true  # 5×5 square
a = mesh_surface_2d(mask)  # ≈ 25.0 pixels²
```

# References
- PyRadiomics: radiomics/shape2D.py:getMeshSurfaceFeatureValue
"""
function mesh_surface_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    vertices, edges = _marching_squares_2d(mask, spacing)

    if isempty(edges)
        return 0.0
    end

    # Use shoelace formula: A = ½|Σ(xᵢyᵢ₊₁ - xᵢ₊₁yᵢ)|
    # For line segments, compute signed area of triangles with origin
    area = 0.0
    for (i1, i2) in edges
        v1 = vertices[i1]
        v2 = vertices[i2]
        # Cross product (v1 × v2) for 2D = v1[1]*v2[2] - v1[2]*v2[1]
        # This gives signed area of triangle with origin
        area += v1[1] * v2[2] - v1[2] * v2[1]
    end

    return abs(area) / 2.0
end

#==============================================================================#
# 2D Shape Feature: Pixel Surface (Area from pixel count)
#==============================================================================#

"""
    pixel_surface_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the surface area of a 2D ROI by counting pixels.

# Mathematical Formula
```
A = Nₚ × Aₚᵢₓₑₗ
```
where:
- Nₚ = number of pixels in mask
- Aₚᵢₓₑₗ = area of single pixel = spacing[1] × spacing[2]

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Area in mm² (or pixels² if spacing is (1.0, 1.0))

# Notes
- Simple counting method, less accurate than mesh-based for smooth shapes
- Identical to mesh area for rectangular shapes aligned with pixel grid

# Example
```julia
mask = falses(10, 10)
mask[3:7, 3:7] .= true  # 5×5 = 25 pixels
a = pixel_surface_2d(mask)  # 25.0
```

# References
- PyRadiomics: radiomics/shape2D.py:getPixelSurfaceFeatureValue
"""
function pixel_surface_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    n_pixels = count(mask)
    pixel_area = spacing[1] * spacing[2]
    return Float64(n_pixels) * pixel_area
end

#==============================================================================#
# 2D Shape Feature: Perimeter-to-Surface Ratio
#==============================================================================#

"""
    perimeter_surface_ratio_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the ratio of perimeter to surface area.

# Mathematical Formula
```
P/A = Perimeter / PixelSurface
```

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Perimeter-to-area ratio in mm⁻¹ (NOT dimensionless)

# Notes
- Higher values indicate more complex/irregular shapes
- A circle has the minimum P/A ratio for a given area
- NOT dimensionless - has units of 1/length

# Example
```julia
mask = falses(10, 10)
mask[3:7, 3:7] .= true  # 5×5 square
ratio = perimeter_surface_ratio_2d(mask)  # ≈ 0.8 (20/25)
```

# References
- PyRadiomics: radiomics/shape2D.py:getPerimeterSurfaceRatioFeatureValue
"""
function perimeter_surface_ratio_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    p = perimeter_2d(mask, spacing)
    a = pixel_surface_2d(mask, spacing)

    if a == 0.0
        return NaN
    end

    return p / a
end

#==============================================================================#
# 2D Shape Feature: Sphericity (Circularity)
#==============================================================================#

"""
    sphericity_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the sphericity (circularity) of a 2D ROI.

# Mathematical Formula
```
Sphericity = (2√(πA)) / P
```
where:
- A = Pixel Surface area
- P = Perimeter

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Sphericity value in range (0, 1], where 1 = perfect circle

# Notes
- Measures how circular the shape is
- A perfect circle has sphericity = 1
- More irregular shapes have lower sphericity
- Uses pixel surface for area (matches PyRadiomics)

# Example
```julia
# A rough approximation of a circle
mask = [sqrt((i-15)^2 + (j-15)^2) <= 10 for i in 1:30, j in 1:30]
s = sphericity_2d(mask)  # Close to 1.0
```

# References
- PyRadiomics: radiomics/shape2D.py:getSphericityFeatureValue
- IBSI: Circularity (2D equivalent of sphericity)
"""
function sphericity_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    a = pixel_surface_2d(mask, spacing)
    p = perimeter_2d(mask, spacing)

    if p == 0.0
        return NaN
    end

    return (2.0 * sqrt(π * a)) / p
end

#==============================================================================#
# 2D Shape Feature: Spherical Disproportion (DEPRECATED)
#==============================================================================#

"""
    spherical_disproportion_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the spherical disproportion of a 2D ROI.

# Mathematical Formula
```
SphericalDisproportion = P / (2√(πA)) = 1 / Sphericity
```

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Spherical disproportion value ≥ 1, where 1 = perfect circle

# Notes
- **DEPRECATED**: This is the inverse of sphericity
- Included for PyRadiomics compatibility
- A perfect circle has spherical disproportion = 1
- More irregular shapes have higher values

# References
- PyRadiomics: radiomics/shape2D.py:getSphericalDisproportionFeatureValue
"""
function spherical_disproportion_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    s = sphericity_2d(mask, spacing)

    if s == 0.0 || isnan(s)
        return NaN
    end

    return 1.0 / s
end

#==============================================================================#
# 2D Shape Feature: Maximum 2D Diameter (Feret Diameter)
#==============================================================================#

"""
    maximum_diameter_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the maximum 2D diameter (Feret diameter) of the ROI.

# Mathematical Formula
```
MaxDiameter = max_{i,j} ||Vᵢ - Vⱼ||
```
Maximum Euclidean distance between any two contour vertices.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Maximum diameter in mm (or pixels if spacing is (1.0, 1.0))

# Notes
- Also known as Feret diameter or caliper length
- Computed by finding the maximum distance between all pairs of contour vertices
- O(n²) algorithm where n is number of contour vertices

# Example
```julia
mask = falses(10, 10)
mask[3:7, 3:7] .= true  # 5×5 square
d = maximum_diameter_2d(mask)  # ≈ √(4² + 4²) ≈ 5.66 (diagonal)
```

# References
- PyRadiomics: radiomics/shape2D.py:getMaximumDiameterFeatureValue
- IBSI: Maximum 2D diameter
"""
function maximum_diameter_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    vertices, edges = _marching_squares_2d(mask, spacing)

    if length(vertices) < 2
        return 0.0
    end

    # Find maximum distance between all vertex pairs
    max_dist_sq = 0.0
    n = length(vertices)

    for i in 1:n
        for j in (i+1):n
            v1 = vertices[i]
            v2 = vertices[j]
            dist_sq = (v2[1] - v1[1])^2 + (v2[2] - v1[2])^2
            if dist_sq > max_dist_sq
                max_dist_sq = dist_sq
            end
        end
    end

    return sqrt(max_dist_sq)
end

#==============================================================================#
# 2D Shape Feature: PCA-based Axis Lengths and Elongation
#==============================================================================#

"""
    _compute_eigenvalues_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real})

Compute eigenvalues of the 2D covariance matrix for PCA-based shape features.

# Algorithm
1. Get physical coordinates of all ROI pixels
2. Center coordinates at mean
3. Normalize by √N
4. Compute covariance matrix: C = X'X
5. Compute eigenvalues and sort ascending

# Arguments
- `mask`: 2D binary mask
- `spacing`: Pixel spacing as (row_spacing, col_spacing)

# Returns
- `(λ_minor, λ_major)`: Eigenvalues sorted ascending

# References
- PyRadiomics: radiomics/shape.py (PCA approach)
"""
function _compute_eigenvalues_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real})
    # Get indices of all mask voxels
    indices = findall(mask)
    n = length(indices)

    if n < 2
        return (0.0, 0.0)
    end

    # Convert to physical coordinates
    coords = Matrix{Float64}(undef, n, 2)
    for (i, idx) in enumerate(indices)
        coords[i, 1] = (idx[1] - 0.5) * spacing[1]  # row → physical
        coords[i, 2] = (idx[2] - 0.5) * spacing[2]  # col → physical
    end

    # Center at mean
    μ = mean(coords, dims=1)
    coords .-= μ

    # Normalize by √N
    coords ./= sqrt(n)

    # Compute covariance matrix
    cov_matrix = coords' * coords

    # Compute eigenvalues
    eigenvalues = eigvals(Symmetric(cov_matrix))

    # Handle small negative eigenvalues from numerical precision
    eigenvalues = [max(λ, 0.0) for λ in eigenvalues]

    # Sort ascending: [λ_minor, λ_major]
    sort!(eigenvalues)

    return (eigenvalues[1], eigenvalues[2])
end

"""
    major_axis_length_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the major axis length of the 2D ROI using PCA.

# Mathematical Formula
```
MajorAxisLength = 4√λ_major
```
where λ_major is the largest eigenvalue of the covariance matrix.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Major axis length in mm

# Notes
- Represents the length of the major axis of an equivalent ellipse
- Based on principal component analysis (PCA) of pixel coordinates
- The factor of 4 converts the eigenvalue to the diameter of an enclosing ellipsoid

# Example
```julia
mask = falses(20, 10)
mask[5:15, 3:7] .= true  # Elongated rectangle
major = major_axis_length_2d(mask)  # Length along long axis
```

# References
- PyRadiomics: radiomics/shape2D.py:getMajorAxisLengthFeatureValue
- IBSI: Major axis length
"""
function major_axis_length_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    λ_minor, λ_major = _compute_eigenvalues_2d(mask, spacing)
    return 4.0 * sqrt(λ_major)
end

"""
    minor_axis_length_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the minor axis length of the 2D ROI using PCA.

# Mathematical Formula
```
MinorAxisLength = 4√λ_minor
```
where λ_minor is the smallest eigenvalue of the covariance matrix.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Minor axis length in mm

# Notes
- Represents the length of the minor axis of an equivalent ellipse
- Based on principal component analysis (PCA) of pixel coordinates
- The factor of 4 converts the eigenvalue to the diameter of an enclosing ellipsoid

# Example
```julia
mask = falses(20, 10)
mask[5:15, 3:7] .= true  # Elongated rectangle
minor = minor_axis_length_2d(mask)  # Length along short axis
```

# References
- PyRadiomics: radiomics/shape2D.py:getMinorAxisLengthFeatureValue
- IBSI: Minor axis length
"""
function minor_axis_length_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    λ_minor, λ_major = _compute_eigenvalues_2d(mask, spacing)
    return 4.0 * sqrt(λ_minor)
end

"""
    elongation_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0)) -> Float64

Compute the elongation of a 2D ROI using PCA eigenvalues.

# Mathematical Formula
```
Elongation = √(λ_minor / λ_major)
```
where λ_minor and λ_major are eigenvalues of the covariance matrix.

# Arguments
- `mask`: 2D binary mask defining the ROI
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))

# Returns
- `Float64`: Elongation value in range [0, 1], where 1 = circular (isotropic)

# Notes
- Measures how elongated the shape is
- A perfect circle has elongation = 1
- More elongated shapes have values closer to 0
- Equivalent to MinorAxisLength / MajorAxisLength

# Example
```julia
# Square (isotropic)
mask = falses(10, 10)
mask[3:7, 3:7] .= true
e = elongation_2d(mask)  # Close to 1.0

# Rectangle (elongated)
mask = falses(20, 10)
mask[5:15, 3:7] .= true
e = elongation_2d(mask)  # < 1.0
```

# References
- PyRadiomics: radiomics/shape2D.py:getElongationFeatureValue
- IBSI: Elongation
"""
function elongation_2d(mask::AbstractMatrix{Bool}, spacing::NTuple{2,<:Real}=(1.0, 1.0))
    λ_minor, λ_major = _compute_eigenvalues_2d(mask, spacing)

    if λ_major == 0.0
        return NaN
    end

    return sqrt(λ_minor / λ_major)
end

#==============================================================================#
# High-Level 2D Extraction Function
#==============================================================================#

"""
    extract_shape_2d(mask, spacing=(1.0, 1.0); label::Int=1) -> Dict{String, Float64}

Extract all 2D shape features from a mask.

# Arguments
- `mask`: 2D array (Bool, BitArray, or Integer with label)
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))
- `label::Int=1`: Label value in mask defining the ROI (for integer masks)

# Returns
- `Dict{String, Float64}`: Dictionary of feature names to values

# Feature Names
The returned dictionary uses PyRadiomics-compatible names:
- "MeshSurface" - Area from mesh
- "PixelSurface" - Area from pixel count
- "Perimeter" - Contour length
- "PerimeterSurfaceRatio" - P/A ratio
- "Sphericity" - Circularity (2√(πA)/P)
- "SphericalDisproportion" - 1/Sphericity (deprecated)
- "MaximumDiameter" - Feret diameter
- "MajorAxisLength" - PCA major axis
- "MinorAxisLength" - PCA minor axis
- "Elongation" - √(λ_minor/λ_major)

# Example
```julia
mask = rand(Bool, 64, 64)
features = extract_shape_2d(mask)
println(features["Sphericity"])
println(features["Elongation"])
```

# Notes
- SphericalDisproportion is deprecated but included for PyRadiomics compatibility

# See also
- Individual feature functions: `perimeter_2d`, `sphericity_2d`, etc.
"""
function extract_shape_2d(mask::AbstractMatrix{T}, spacing::NTuple{2,<:Real}=(1.0, 1.0);
                          label::Int=1) where T
    # Convert to boolean mask
    if T <: Bool
        bool_mask = mask
    elseif T <: Integer
        bool_mask = mask .== label
    else
        throw(ArgumentError("Mask must be Bool or Integer type, got $T"))
    end

    # Ensure we have at least some pixels
    if !any(bool_mask)
        return Dict{String, Float64}(
            "MeshSurface" => 0.0,
            "PixelSurface" => 0.0,
            "Perimeter" => 0.0,
            "PerimeterSurfaceRatio" => NaN,
            "Sphericity" => NaN,
            "SphericalDisproportion" => NaN,
            "MaximumDiameter" => 0.0,
            "MajorAxisLength" => 0.0,
            "MinorAxisLength" => 0.0,
            "Elongation" => NaN
        )
    end

    return Dict{String, Float64}(
        "MeshSurface" => mesh_surface_2d(bool_mask, spacing),
        "PixelSurface" => pixel_surface_2d(bool_mask, spacing),
        "Perimeter" => perimeter_2d(bool_mask, spacing),
        "PerimeterSurfaceRatio" => perimeter_surface_ratio_2d(bool_mask, spacing),
        "Sphericity" => sphericity_2d(bool_mask, spacing),
        "SphericalDisproportion" => spherical_disproportion_2d(bool_mask, spacing),
        "MaximumDiameter" => maximum_diameter_2d(bool_mask, spacing),
        "MajorAxisLength" => major_axis_length_2d(bool_mask, spacing),
        "MinorAxisLength" => minor_axis_length_2d(bool_mask, spacing),
        "Elongation" => elongation_2d(bool_mask, spacing)
    )
end

# Convenience method for BitArray
function extract_shape_2d(mask::BitMatrix, spacing::NTuple{2,<:Real}=(1.0, 1.0); label::Int=1)
    bool_mask = convert(Matrix{Bool}, mask)
    return extract_shape_2d(bool_mask, spacing; label=label)
end

"""
    extract_shape_2d_to_featureset!(fs::FeatureSet, mask, spacing=(1.0, 1.0);
                                     label::Int=1, image_type::String="original")

Extract 2D shape features and add them to an existing FeatureSet.

# Arguments
- `fs`: FeatureSet to add features to
- `mask`: 2D array (Bool, BitArray, or Integer with label)
- `spacing`: Pixel spacing as (row_spacing, col_spacing) in mm (default: (1.0, 1.0))
- `label::Int=1`: Label value in mask defining the ROI
- `image_type::String="original"`: Image type label for feature keys

# Returns
- Modified FeatureSet (also modifies in-place)

# Example
```julia
fs = FeatureSet()
extract_shape_2d_to_featureset!(fs, mask)
println(fs["shape_Sphericity"])
```
"""
function extract_shape_2d_to_featureset!(fs::FeatureSet, mask::AbstractMatrix,
                                          spacing::NTuple{2,<:Real}=(1.0, 1.0);
                                          label::Int=1, image_type::String="original")
    features = extract_shape_2d(mask, spacing; label=label)

    for (name, value) in features
        push!(fs, FeatureResult(name, value, "shape", image_type))
    end

    return fs
end

#==============================================================================#
# Feature List and Names
#==============================================================================#

"""
    shape_2d_feature_names() -> Vector{String}

Return a list of all 2D shape feature names.

# Returns
- `Vector{String}`: Names of all 10 2D shape features

# Example
```julia
names = shape_2d_feature_names()
println(length(names))  # 10
```
"""
function shape_2d_feature_names()
    return [
        "MeshSurface",
        "PixelSurface",
        "Perimeter",
        "PerimeterSurfaceRatio",
        "Sphericity",
        "SphericalDisproportion",
        "MaximumDiameter",
        "MajorAxisLength",
        "MinorAxisLength",
        "Elongation"
    ]
end

"""
    shape_2d_ibsi_features() -> Vector{String}

Return a list of IBSI-compliant 2D shape features.

# Returns
- `Vector{String}`: Names of IBSI-compliant features (excludes deprecated SphericalDisproportion)
"""
function shape_2d_ibsi_features()
    return [
        "MeshSurface",
        "PixelSurface",
        "Perimeter",
        "PerimeterSurfaceRatio",
        "Sphericity",
        "MaximumDiameter",
        "MajorAxisLength",
        "MinorAxisLength",
        "Elongation"
    ]
end
