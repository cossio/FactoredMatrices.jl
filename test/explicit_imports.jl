import ExplicitImports
import FactoredMatrices

using Test: @testset

@testset "ExplicitImports" begin
    ExplicitImports.test_explicit_imports(FactoredMatrices)
end
