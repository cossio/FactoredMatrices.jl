using Test: @testset, @test, @test_throws, @inferred
using LinearAlgebra: rank
using LowRankMatrices: LowRankMatrix

@test iszero(LowRankMatrix(zeros(10,3), zeros(5,3)).u)
@test iszero(LowRankMatrix(zeros(10,3), zeros(5,3)).v)
@test iszero(Matrix(LowRankMatrix(zeros(10,3), zeros(5,3))))

A = LowRankMatrix(randn(20, 4), randn(12, 4))
@test Matrix(@inferred 2A) ≈ Matrix(@inferred A * 2) ≈ 2Matrix(A)

A = LowRankMatrix(randn(20, 4), randn(12, 4))
B = LowRankMatrix(randn(12, 2), randn(7, 2))
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test size(Matrix(A * B)) == size(A * B)
@test rank(A * B) == 2
@test Matrix(A) * Matrix(B) ≈ Matrix(A * Matrix(B)) ≈ Matrix(Matrix(A) * B) ≈ Matrix(A * B)
@test A * B isa LowRankMatrix
@inferred A * B

A = LowRankMatrix(randn(20, 2), randn(12, 2))
B = LowRankMatrix(randn(12, 4), randn(7, 4))
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test rank(A * B) == 2
@test A * B isa LowRankMatrix
@inferred A * B

A = LowRankMatrix(randn(20, 2), randn(12, 2))
B = LowRankMatrix(randn(10, 2), randn(14, 2))
@test_throws DimensionMismatch A * B
@test_throws DimensionMismatch B * A

A = LowRankMatrix(randn(20, 2), randn(12, 2))
B = randn(12,14)
@test A * B isa LowRankMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test Matrix(A) * B ≈ Matrix(A * B)

A = randn(20, 12)
B = LowRankMatrix(randn(12, 2), randn(14, 2))
@test A * B isa LowRankMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test A * Matrix(B) ≈ Matrix(A * B)

A = LowRankMatrix(randn(20, 2), randn(12, 2))
v = randn(12)
@test A * v ≈ Matrix(A) * v

A = LowRankMatrix(randn(20, 2), randn(12, 2))
v = randn(20)
@test v' * A ≈ v' * Matrix(A)
