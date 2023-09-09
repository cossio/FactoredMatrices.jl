import Aqua
import FactoredMatrices

using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(FactoredMatrices)
end
