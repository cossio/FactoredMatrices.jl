using Test: @testset, @test, @test_throws, @inferred
using LinearAlgebra: Adjoint, Diagonal, Factorization, I, LowerTriangular, Transpose, UpperTriangular, dot, norm, pinv, rank, tr
using FactoredMatrices: FactoredMatrix

# the constructor requires the second factor adjointed: FactoredMatrix(u, v') is u * v'
@test_throws ArgumentError FactoredMatrix(randn(5, 2), randn(4, 2))
@test_throws ArgumentError FactoredMatrix(randn(5), randn(4))
u = randn(5, 2)
v = randn(4, 2)
A = @inferred FactoredMatrix(u, v')
@test A isa Factorization{Float64}
@test Matrix(A) == u * v'
@test A.u === u && A.v === v # factors are stored, not copied

# transposed second factor, and complex element types
for T in (Float64, ComplexF64)
    local u = randn(T, 5, 2)
    local v = randn(T, 4, 2)
    @test Matrix(@inferred FactoredMatrix(u, v')) == u * v'
    @test Matrix(@inferred FactoredMatrix(u, transpose(v))) ≈ u * transpose(v)
end

# u and v must have the same number of columns
@test_throws ArgumentError FactoredMatrix(randn(5, 2), randn(4, 3)')

# vector arguments are treated as one-column matrices
x = randn(5)
y = randn(4)
V = randn(4, 1)
U = randn(5, 1)
@test Matrix(@inferred FactoredMatrix(x, V')) ≈ x * vec(V)'
@test Matrix(@inferred FactoredMatrix(U, y')) ≈ vec(U) * y'
@test Matrix(@inferred FactoredMatrix(x, y')) ≈ x * y'
@test rank(FactoredMatrix(x, V')) == rank(FactoredMatrix(U, y')) == rank(FactoredMatrix(x, y')) == 1

# mixed element types promote
A = @inferred FactoredMatrix(randn(Float32, 5, 2), randn(Float64, 4, 2)')
@test eltype(A) == Float64
@test A.u isa Matrix{Float64} && A.v isa Matrix{Float64}

# size, length, rank, getindex
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
@test size(A) == (20, 12)
@test size(A, 1) == 20 && size(A, 2) == 12 && size(A, 3) == 1
@test_throws ArgumentError size(A, 0)
@test length(A) == 20 * 12
@test rank(A) == 4
M = Matrix(A)
@test all(A[i, j] ≈ M[i, j] for i in 1:20, j in 1:12)
@test_throws BoundsError A[0, 1]
@test_throws BoundsError A[1, 13]

# iszero
@test iszero(Matrix(FactoredMatrix(zeros(10, 3), zeros(5, 3)')))
@test @inferred iszero(FactoredMatrix(zeros(10, 3), randn(5, 3)'))
@test iszero(FactoredMatrix(randn(10, 3), zeros(5, 3)'))
@test !iszero(FactoredMatrix(randn(10, 3), randn(5, 3)'))
# 0 * NaN = NaN and 0 * Inf = NaN, so these are not zero matrices
@test !iszero(FactoredMatrix(zeros(10, 3), fill(NaN, 5, 3)'))
@test !iszero(FactoredMatrix(fill(NaN, 10, 3), zeros(5, 3)'))
@test !iszero(FactoredMatrix(zeros(10, 3), fill(Inf, 5, 3)'))
@test !iszero(FactoredMatrix(fill(-Inf, 10, 3), zeros(5, 3)'))
# empty represented matrices are zero regardless of the (unused) factor values
@test iszero(FactoredMatrix(zeros(0, 1), fill(Inf, 2, 1)'))
@test iszero(FactoredMatrix(fill(NaN, 2, 1), zeros(0, 1)'))

# adjoint and transpose
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4)')
    @test @inferred(adjoint(A)) isa FactoredMatrix
    @test @inferred(transpose(A)) isa FactoredMatrix
    @test Matrix(A') ≈ Matrix(A)'
    @test Matrix(transpose(A)) ≈ transpose(Matrix(A))
    @test Matrix((A')') ≈ Matrix(A)
    @test Matrix(transpose(transpose(A))) ≈ Matrix(A)
end

# conversion and copy
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
@test Array(A) == Matrix(A)
B = @inferred copy(A)
@test B isa FactoredMatrix
@test Matrix(B) == Matrix(A)
@test B.u == A.u && B.u !== A.u
@test B.v == A.v && B.v !== A.v

# show
A = FactoredMatrix(randn(3, 2), randn(5, 2)')
@test sprint(show, A) == "FactoredMatrix{Float64} of size (3, 5) and rank 2"
@test sprint(show, MIME("text/plain"), A) == sprint(show, A)

# scalar multiples and division
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
@test Matrix(@inferred 2A) ≈ Matrix(@inferred A * 2) ≈ 2Matrix(A)
@test Matrix(@inferred A / 2) ≈ Matrix(A) / 2
@test Matrix(@inferred 2 \ A) ≈ 2 \ Matrix(A)
@test Matrix(@inferred -A) ≈ -Matrix(A)
@test (+A) === A
a = 1 + 2im
A = FactoredMatrix(randn(ComplexF64, 20, 4), randn(ComplexF64, 12, 4)')
@test Matrix(@inferred a * A) ≈ a * Matrix(A)
@test Matrix(@inferred A * a) ≈ Matrix(A) * a # scalar must not be conjugated by the implicit adjoint
@test Matrix(@inferred A / a) ≈ Matrix(A) / a
@test Matrix(@inferred a \ A) ≈ a \ Matrix(A)

# a type-promoting scalar works because the factors are promoted to a common type
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
a = 1 + im
@test Matrix(@inferred a * A) ≈ a * Matrix(A)
@test Matrix(@inferred A * a) ≈ Matrix(A) * a

# UniformScaling products reduce to scalar multiplication
A = FactoredMatrix(randn(6, 2), randn(5, 2)')
@test A * I isa FactoredMatrix
@test Matrix(A * I) ≈ Matrix(A)
@test Matrix(I * A) ≈ Matrix(A)
@test Matrix(A * (2I)) ≈ 2 * Matrix(A)
@test Matrix((3I) * A) ≈ 3 * Matrix(A)
@test Matrix(Adjoint(A) * (2I)) ≈ 2 * Matrix(A)'
@test Matrix((2I) * Transpose(A)) ≈ 2 * transpose(Matrix(A))

# lazy sums and differences by factor concatenation
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4)')
    local B = FactoredMatrix(randn(T, 20, 2), randn(T, 12, 2)')
    @test @inferred(A + B) isa FactoredMatrix
    @test @inferred(A - B) isa FactoredMatrix
    @test Matrix(A + B) ≈ Matrix(A) + Matrix(B)
    @test Matrix(A - B) ≈ Matrix(A) - Matrix(B)
    @test rank(A + B) == rank(A - B) == rank(A) + rank(B)
end
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
B = FactoredMatrix(randn(21, 4), randn(12, 4)')
@test_throws DimensionMismatch A + B
@test_throws DimensionMismatch A - FactoredMatrix(randn(20, 4), randn(13, 4)')

# products of two factored matrices keep the smaller rank
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
B = FactoredMatrix(randn(12, 2), randn(7, 2)')
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test size(Matrix(A * B)) == size(A * B)
@test rank(A * B) == 2
@test Matrix(A) * Matrix(B) ≈ Matrix(A * Matrix(B)) ≈ Matrix(Matrix(A) * B) ≈ Matrix(A * B)
@test A * B isa FactoredMatrix
@inferred A * B

A = FactoredMatrix(randn(20, 2), randn(12, 2)')
B = FactoredMatrix(randn(12, 4), randn(7, 4)')
@test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
@test rank(A * B) == 2
@test A * B isa FactoredMatrix
@inferred A * B

A = FactoredMatrix(randn(20, 2), randn(12, 2)')
B = FactoredMatrix(randn(10, 2), randn(14, 2)')
@test_throws DimensionMismatch A * B
@test_throws DimensionMismatch B * A

# products over an empty contracted dimension are exactly zero, even when the unused
# factor entries are not finite (the reassociation must not manufacture 0 * Inf = NaN)
Le = FactoredMatrix(fill(Inf, 2, 1), zeros(0, 1)') # 2 × 0
Me = FactoredMatrix(zeros(0, 1), fill(Inf, 3, 1)') # 0 × 3
@test size(Le * Me) == (2, 3)
@test iszero(Matrix(Le * Me))
@test iszero(Matrix(Le * zeros(0, 3)))
@test iszero(Matrix(zeros(3, 0) * Me))
@test iszero(Le * zeros(0))
@test iszero(zeros(0)' * Me)
@test_throws DimensionMismatch Le * FactoredMatrix(randn(5, 1), randn(3, 1)')

# products with dense matrices and vectors
A = FactoredMatrix(randn(20, 2), randn(12, 2)')
B = randn(12, 14)
@test A * B isa FactoredMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test Matrix(A) * B ≈ Matrix(A * B)

A = randn(20, 12)
B = FactoredMatrix(randn(12, 2), randn(14, 2)')
@test A * B isa FactoredMatrix
@test rank(A * B) == size((A * B).u, 2) == 2
@test A * Matrix(B) ≈ Matrix(A * B)

A = FactoredMatrix(randn(20, 2), randn(12, 2)')
x = randn(12)
@test A * x ≈ Matrix(A) * x
y = randn(20)
@test y' * A ≈ y' * Matrix(A)
@test transpose(y) * A isa FactoredMatrix
@test Matrix(transpose(y) * A) ≈ transpose(y) * Matrix(A)

# products with Diagonal and triangular matrices go through the generic method
A = FactoredMatrix(randn(20, 2), randn(12, 2)')
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

# products with explicit Adjoint/Transpose wrappers keep the factored structure
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4)') # 20 × 12
    local B = FactoredMatrix(randn(T, 7, 2), randn(T, 12, 2)') # 7 × 12
    local B3 = FactoredMatrix(randn(T, 9, 2), randn(T, 20, 2)') # 9 × 20
    local C = randn(T, 12, 20)
    local x = randn(T, 20)
    local y = randn(T, 7)
    local z = randn(T, 12)
    @test @inferred(A * Adjoint(B)) isa FactoredMatrix
    @test Matrix(A * Adjoint(B)) ≈ Matrix(A) * Matrix(B)'
    @test rank(A * Adjoint(B)) == 2 # min(rank(A), rank(B)), not rank(A)
    @test Matrix(A * Transpose(B)) ≈ Matrix(A) * transpose(Matrix(B))
    @test Matrix(Adjoint(A) * Adjoint(B3)) ≈ Matrix(A)' * Matrix(B3)'
    @test Matrix(Adjoint(A) * B3') ≈ Matrix(A)' * Matrix(B3)'
    @test Matrix(A' * Adjoint(B3)) ≈ Matrix(A)' * Matrix(B3)'
    @test Matrix(Adjoint(A)) ≈ Matrix(A)'
    @test Matrix(Transpose(A)) ≈ transpose(Matrix(A))
    @test Matrix(Adjoint(A) * C') ≈ Matrix(A)' * C'
    @test Matrix(C' * Adjoint(A)) ≈ C' * Matrix(A)'
    @test Adjoint(A) * x ≈ Matrix(A)' * x
    @test y' * Adjoint(B') ≈ y' * Matrix(B')'
    @test Matrix(transpose(z) * Transpose(A)) ≈ transpose(z) * transpose(Matrix(A))
end

# left division: square invertible case
A = FactoredMatrix(randn(4, 4), randn(4, 4)')
b = randn(4)
B = randn(4, 3)
@test A * (A \ b) ≈ b
@test A \ b ≈ Matrix(A) \ b
@test A \ B ≈ Matrix(A) \ B

# left division: complex right-hand side with a real factorization (disambiguation
# against the LinearAlgebra Factorization fallback)
bc = randn(ComplexF64, 4)
Bc = randn(ComplexF64, 4, 3)
@test A \ bc ≈ Matrix(A) \ bc
@test A \ Bc ≈ Matrix(A) \ Bc

# left division: low-rank case gives the pseudoinverse solution
A = FactoredMatrix(randn(12, 3), randn(15, 3)')
b = randn(12)
B = randn(12, 5)
@test A \ b ≈ pinv(Matrix(A)) * b
@test A \ B ≈ pinv(Matrix(A)) * B

# left division: factors with dependent columns (as produced by the lazy sums) still
# give the pseudoinverse solution
A1 = FactoredMatrix(fill(1.0, 1, 1), fill(1.0, 1, 1)')
B1 = FactoredMatrix(fill(2.0, 1, 1), fill(1.0, 1, 1)')
@test (A1 + B1) \ [1.0] ≈ [1 / 3]
u9 = randn(9, 2)
v9 = randn(7, 2)
S2 = FactoredMatrix(u9, v9') + FactoredMatrix(u9, v9') # storage rank 4, actual rank 2
b9 = randn(9)
B9 = randn(9, 3)
@test S2 \ b9 ≈ pinv(Matrix(S2)) * b9
@test S2 \ B9 ≈ pinv(Matrix(S2)) * B9

# left division: zero and rank-0 matrices have the all-zero minimum-norm solution
@test FactoredMatrix(zeros(3, 0), zeros(2, 0)') \ ones(3) == zeros(2)
@test FactoredMatrix(zeros(3, 2), zeros(2, 2)') \ ones(3, 4) == zeros(2, 4)
@test_throws DimensionMismatch A1 \ ones(2)

# dot, sum(abs2), norm, tr
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4)')
    local B = FactoredMatrix(randn(T, 20, 2), randn(T, 12, 2)')
    local M = randn(T, 20, 12)
    @test dot(A, B) ≈ dot(Matrix(A), Matrix(B))
    @test dot(A, A) ≈ sum(abs2, Matrix(A))
    @test dot(A, M) ≈ dot(Matrix(A), M)
    @test dot(M, A) ≈ dot(M, Matrix(A))
    @test sum(abs2, A) ≈ sum(abs2, Matrix(A))
    @test norm(A) ≈ norm(Matrix(A))
    @test norm(A, 2) ≈ norm(Matrix(A))
    @test_throws ArgumentError norm(A, 1)
    local S = FactoredMatrix(randn(T, 9, 3), randn(T, 9, 3)')
    @test tr(S) ≈ tr(Matrix(S))
    @test_throws DimensionMismatch tr(A)
end
@test iszero(norm(FactoredMatrix(zeros(5, 2), zeros(4, 2)')))

# empty factorizations dot to zero (an empty sum) even with non-finite factor values,
# and self-dots use the stable nonnegative reduction instead of the Gram form
E1 = FactoredMatrix(zeros(0, 1), fill(Inf, 2, 1)')
@test iszero(dot(E1, E1))
Afc = FactoredMatrix([1.0 1.0], [1.0e8 -nextfloat(1.0e8)]')
@test dot(Afc, Afc) ≥ 0
@test dot(Afc, Afc) ≈ sum(abs2, Matrix(Afc))
# equal factors (not just identical objects) also take the stable self-dot path
@test dot(Afc, copy(Afc)) ≥ 0
@test dot(Afc, copy(Afc)) == dot(Afc, Afc)
# but only for matching element types: mixed precisions accumulate differently, so
# ==-equal factors of different precisions must evaluate the mixed product
A32 = FactoredMatrix(Float32[4097 4096], Float32[4097 -4096]') # entry rounds to 8192f0
A64 = FactoredMatrix(Float64[4097 4096], Float64[4097 -4096]') # entry is exactly 8193.0
@test A32.u == A64.u && A32.v == A64.v
@test dot(A32, A64) == 8193.0^2 # the exact mixed Gram product, not sum(abs2, A32)

# non-finite factors with well-defined represented entries: norm, sum(abs2) and the
# self-dot reduce the entries directly, since a QR of the factors would produce NaNs
An = FactoredMatrix([Inf 1.0; Inf 2.0], [1.0 1.0]')
@test norm(An) == Inf
@test sum(abs2, An) == Inf
@test dot(An, An) == Inf
@test isnan(norm(FactoredMatrix(fill(NaN, 2, 1), ones(1, 1)')))
Ac2 = FactoredMatrix(randn(ComplexF64, 6, 2), randn(ComplexF64, 5, 2)')
@test dot(Ac2, Ac2) isa ComplexF64
@test dot(Ac2, Ac2) ≈ dot(Matrix(Ac2), Matrix(Ac2))
# self-dots widen to the accumulation type, like the dense inner product
Lb = FactoredMatrix(trues(1, 2), trues(1, 2)')
@test dot(Lb, Lb) === 4
# mixed factored/dense dots of empty matrices are empty (zero) sums, not 0 * Inf
@test iszero(dot(Le, zeros(2, 0)))
@test iszero(dot(zeros(2, 0), Le))

# integer factorizations keep the exact integer accumulation of sum(abs2, Matrix(A)),
# beyond Float64 precision (unlike the floating-point QR route used by norm)
Ai = FactoredMatrix([134217729;;], [1;;]')
@test sum(abs2, Ai) === 134217729^2
@test sum(abs2, Ai) == sum(abs2, Matrix(Ai))

# floating-point factors use the stable QR route instead: the Gram closed form can
# catastrophically cancel (even to a negative value) for nearly-cancelling columns
Af = FactoredMatrix([1.0 1.0], [1.0e8 -nextfloat(1.0e8)]')
@test sum(abs2, Af) ≥ 0
@test sum(abs2, Af) ≈ sum(abs2, Matrix(Af))
@test sum(abs2, FactoredMatrix(randn(ComplexF64, 5, 2), randn(ComplexF64, 4, 2)')) ≥ 0
