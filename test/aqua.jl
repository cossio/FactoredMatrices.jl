import Aqua
import LowRankMatrices

using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(LowRankMatrices)
end
