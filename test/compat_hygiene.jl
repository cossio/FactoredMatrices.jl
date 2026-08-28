import FactoredMatrices
import TestCompatHygiene

using Test: @testset

@testset "TestCompatHygiene" begin
    TestCompatHygiene.test_all(FactoredMatrices)
end
