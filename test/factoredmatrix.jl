using Test: @testset, @test, @test_throws, @inferred
using LinearAlgebra: rank, pinv, Adjoint, Diagonal, UpperTriangular, LowerTriangular
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

# u and v must have the same number of columns
@test_throws ArgumentError FactoredMatrix(randn(5, 2), randn(4, 3))

# vector arguments are reshaped to single-column matrices
u = randn(5)
v = randn(4)
U = randn(5, 1)
V = randn(4, 1)
@test Matrix(@inferred FactoredMatrix(u, V)) ≈ u * vec(V)'
@test Matrix(@inferred FactoredMatrix(U, v)) ≈ vec(U) * v'
@test Matrix(@inferred FactoredMatrix(u, v)) ≈ u * v'
@test rank(FactoredMatrix(u, V)) == rank(FactoredMatrix(U, v)) == rank(FactoredMatrix(u, v)) == 1

# iszero
@test @inferred iszero(FactoredMatrix(zeros(10, 3), randn(5, 3)))
@test iszero(FactoredMatrix(randn(10, 3), zeros(5, 3)))
@test !iszero(FactoredMatrix(randn(10, 3), randn(5, 3)))
# 0 * NaN = NaN, so these are not zero matrices
@test !iszero(FactoredMatrix(zeros(10, 3), fill(NaN, 5, 3)))
@test !iszero(FactoredMatrix(fill(NaN, 10, 3), zeros(5, 3)))

A = FactoredMatrix(randn(20, 4), randn(12, 4))
@test Array(A) == Matrix(A)
B = @inferred copy(A)
@test B isa FactoredMatrix
@test Matrix(B) == Matrix(A)
@test B.u == A.u && B.u !== A.u
@test B.v == A.v && B.v !== A.v

# left division: square invertible case
A = FactoredMatrix(randn(4, 4), randn(4, 4))
b = randn(4)
B = randn(4, 3)
@test A * (A \ b) ≈ b
@test A \ b ≈ Matrix(A) \ b
@test A \ B ≈ Matrix(A) \ B

# left division: low-rank case gives the pseudoinverse solution
A = FactoredMatrix(randn(12, 3), randn(15, 3))
b = randn(12)
B = randn(12, 5)
@test A \ b ≈ pinv(Matrix(A)) * b
@test A \ B ≈ pinv(Matrix(A)) * B

# products with Diagonal and triangular matrices (dispatch ambiguity resolutions)
A = FactoredMatrix(randn(20, 2), randn(12, 2))
D = Diagonal(randn(12))
@test @inferred(A * D) isa FactoredMatrix
@test Matrix(A * D) ≈ Matrix(A) * D
D = Diagonal(randn(20))
@test @inferred(D * A) isa FactoredMatrix
@test Matrix(D * A) ≈ D * Matrix(A)
T = UpperTriangular(randn(12, 12))
@test @inferred(A * T) isa FactoredMatrix
@test Matrix(A * T) ≈ Matrix(A) * T
T = LowerTriangular(randn(20, 20))
@test @inferred(T * A) isa FactoredMatrix
@test Matrix(T * A) ≈ T * Matrix(A)

# transposed vector times FactoredMatrix (dispatch ambiguity resolution)
x = randn(20)
@test transpose(x) * A isa FactoredMatrix
@test Matrix(transpose(x) * A) ≈ transpose(x) * Matrix(A)

# product with an explicit Adjoint wrapper keeps the factored structure
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4))
    local B = FactoredMatrix(randn(T, 7, 2), randn(T, 12, 2))
    @test @inferred(A * Adjoint(B)) isa FactoredMatrix
    @test Matrix(A * Adjoint(B)) ≈ Matrix(A) * Matrix(B)'
    @test rank(A * Adjoint(B)) == 2 # min(rank(A), rank(B)), not rank(A)
end

# show
A = FactoredMatrix(randn(3, 2), randn(5, 2))
@test sprint(show, MIME("text/plain"), A) == "FactoredMatrix{Float64} of rank 2."
