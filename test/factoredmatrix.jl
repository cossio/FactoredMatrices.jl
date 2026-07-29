using Test: @testset, @test, @test_throws, @inferred
using LinearAlgebra: rank
using FactoredMatrices: FactoredMatrix

@test iszero(FactoredMatrix(zeros(10, 3), zeros(5, 3)).u)
@test iszero(FactoredMatrix(zeros(10, 3), zeros(5, 3)).v)
@test iszero(Matrix(FactoredMatrix(zeros(10, 3), zeros(5, 3))))

A = FactoredMatrix(randn(20, 4), randn(12, 4))
@test Matrix(@inferred 2A) ≈ Matrix(@inferred A * 2) ≈ 2Matrix(A)

A = FactoredMatrix(randn(20, 4), randn(12, 4))
B = FactoredMatrix(randn(12, 2), randn(7, 2))
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test size(Matrix(A * B)) == size(A * B)
@test rank(A * B) == 2
@test Matrix(A) * Matrix(B) ≈ Matrix(A * Matrix(B)) ≈ Matrix(Matrix(A) * B) ≈ Matrix(A * B)
@test A * B isa FactoredMatrix
@inferred A * B

A = FactoredMatrix(randn(20, 2), randn(12, 2))
B = FactoredMatrix(randn(12, 4), randn(7, 4))
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test rank(A * B) == 2
@test A * B isa FactoredMatrix
@inferred A * B

A = FactoredMatrix(randn(20, 2), randn(12, 2))
B = FactoredMatrix(randn(10, 2), randn(14, 2))
@test_throws DimensionMismatch A * B
@test_throws DimensionMismatch B * A

A = FactoredMatrix(randn(20, 2), randn(12, 2))
B = randn(12, 14)
@test A * B isa FactoredMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test Matrix(A) * B ≈ Matrix(A * B)

A = randn(20, 12)
B = FactoredMatrix(randn(12, 2), randn(14, 2))
@test A * B isa FactoredMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test A * Matrix(B) ≈ Matrix(A * B)

A = FactoredMatrix(randn(20, 2), randn(12, 2))
v = randn(12)
@test A * v ≈ Matrix(A) * v

A = FactoredMatrix(randn(20, 2), randn(12, 2))
v = randn(20)
@test v' * A ≈ v' * Matrix(A)

for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4))
    @test @inferred(adjoint(A)) isa FactoredMatrix
    @test @inferred(transpose(A)) isa FactoredMatrix
    @test Matrix(A') ≈ Matrix(A)'
    @test Matrix(transpose(A)) ≈ transpose(Matrix(A))
    @test Matrix((A')') ≈ Matrix(A)
    @test Matrix(transpose(transpose(A))) ≈ Matrix(A)
end
