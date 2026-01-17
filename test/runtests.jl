using Test
using Radiomics

@testset "Radiomics.jl" begin
    @testset "Package loads" begin
        @test Radiomics.VERSION == v"0.1.0"
    end

    # Feature class tests will be included as they are implemented
    # include("test_utils.jl")
    # include("test_core.jl")
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
