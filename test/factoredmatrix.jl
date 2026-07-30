using Test: @testset, @test, @test_throws, @inferred
using LinearAlgebra: Diagonal, Factorization, Hermitian, I, LowerTriangular, Symmetric, Transpose, UpperTriangular, dot, norm, pinv, rank, tr
using FactoredMatrices: FactoredMatrix, FactoredMatrices

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

# structured second factors whose adjoint is eager (not an Adjoint wrapper) are
# accepted and taken verbatim as the right multiplicand
u52 = randn(5, 2)
uc52 = randn(ComplexF64, 5, 2)
d2 = Diagonal([1.0, 2.0])
@test Matrix(FactoredMatrix(u52, d2')) ≈ u52 * Matrix(d2)'
dc2 = Diagonal([1.0 + 2im, 3.0 - im])
@test Matrix(FactoredMatrix(uc52, dc2')) ≈ uc52 * Matrix(dc2)'
Ut2 = UpperTriangular(randn(2, 2))
@test Matrix(FactoredMatrix(u52, Ut2')) ≈ u52 * Matrix(Ut2)'
Hc2 = Hermitian(randn(ComplexF64, 2, 2))
@test Matrix(FactoredMatrix(uc52, Hc2')) ≈ uc52 * Matrix(Hc2)
Sr2 = Symmetric(randn(2, 2))
@test Matrix(FactoredMatrix(u52, Sr2')) ≈ u52 * Matrix(Sr2)
# a complex Symmetric's adjoint is a lazy wrapper, so the plain form stays ambiguous
@test_throws ArgumentError FactoredMatrix(uc52, Symmetric(randn(ComplexF64, 2, 2)))
@test Matrix(FactoredMatrix(uc52, Symmetric(randn(ComplexF64, 2, 2))')) isa Matrix

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

# size, length, rank; like stdlib Factorization types, indexing is not public API
A = FactoredMatrix(randn(20, 4), randn(12, 4)')
@test size(A) == (20, 12)
@test size(A, 1) == 20 && size(A, 2) == 12 && size(A, 3) == 1
@test_throws ArgumentError size(A, 0)
@test length(A) == 20 * 12
@test rank(A) == 4
@test_throws MethodError A[1, 1]
M = Matrix(A)
# the internal O(rank) entry accessor backs isapprox, iszero, norm, dot, ...
@test all(FactoredMatrices._entry(A, i, j) ≈ M[i, j] for i in 1:20, j in 1:12)

# iszero
@test iszero(Matrix(FactoredMatrix(zeros(10, 3), zeros(5, 3)')))
@test @inferred iszero(FactoredMatrix(zeros(10, 3), randn(5, 3)'))
@test iszero(FactoredMatrix(randn(10, 3), zeros(5, 3)'))
@test !iszero(FactoredMatrix(randn(10, 3), randn(5, 3)'))

# nonzero factors can cancel: A - A represents the zero matrix. Integer-valued factors
# keep every product and partial sum exact, so the cancellation survives FMA contraction.
A1z = FactoredMatrix(Float64[1 2; 3 4; 5 6], Float64[1 2; 3 4]')
@test @inferred iszero(A1z - A1z)
@test !iszero(A1z + A1z)
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
# empty products of Bool factors accumulate into Int, like ordinary matrix products
LbE = FactoredMatrix(trues(2, 1), trues(0, 1)') # 2 × 0
MbE = FactoredMatrix(trues(0, 1), trues(3, 1)') # 0 × 3
@test eltype(LbE * MbE) == Int
@test iszero(Matrix(LbE * MbE))
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
@test transpose(y) * A isa Transpose{<:Any, <:AbstractVector}
@test transpose(y) * A ≈ transpose(y) * Matrix(A)

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

# adjointed/transposed operands (plain FactoredMatrixes) keep the factored structure
for T in (Float64, ComplexF64)
    local A = FactoredMatrix(randn(T, 20, 4), randn(T, 12, 4)') # 20 × 12
    local B = FactoredMatrix(randn(T, 7, 2), randn(T, 12, 2)') # 7 × 12
    local B3 = FactoredMatrix(randn(T, 9, 2), randn(T, 20, 2)') # 9 × 20
    local C = randn(T, 12, 20)
    local x = randn(T, 20)
    local z = randn(T, 12)
    @test @inferred(A * B') isa FactoredMatrix
    @test Matrix(A * B') ≈ Matrix(A) * Matrix(B)'
    @test rank(A * B') == 2 # min(rank(A), rank(B)), not rank(A)
    @test Matrix(A * transpose(B)) ≈ Matrix(A) * transpose(Matrix(B))
    @test Matrix(A' * B3') ≈ Matrix(A)' * Matrix(B3)'
    @test Matrix(A' * C') ≈ Matrix(A)' * C'
    @test Matrix(C' * A') ≈ C' * Matrix(A)'
    @test A' * x ≈ Matrix(A)' * x
    # row-vector left operands keep their row-vector shape, without spurious conjugation
    @test transpose(x) * A isa Transpose{<:Any, <:AbstractVector}
    @test transpose(x) * A ≈ transpose(x) * Matrix(A)
    @test transpose(z) * transpose(A) isa Transpose{<:Any, <:AbstractVector}
    @test transpose(z) * transpose(A) ≈ transpose(z) * transpose(Matrix(A))
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

# left division: singular values below pinv's dimension-scaled cutoff are treated as
# zero (eps * min(m, n) * S[1] here discards 1e-15; an unscaled eps * S[1] would keep
# it and blow the solution up to order 1e15)
uc = zeros(100, 2)
uc[1, 1] = uc[2, 2] = 1
vc = zeros(100, 2)
vc[1, 1] = 1
vc[2, 2] = 1.0e-15
Lc = FactoredMatrix(uc, vc')
bc2 = zeros(100)
bc2[2] = 1
@test Lc \ bc2 ≈ pinv(Matrix(Lc)) * bc2 atol = 1.0e-8

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
# but only for matching element types: each operand's entries round in its own
# precision, so mixed-precision dots are evaluated entrywise, matching the dense
# mixed dot
A32 = FactoredMatrix(Float32[4097 4096], Float32[4097 -4096]') # entry rounds to 8192f0
A64 = FactoredMatrix(Float64[4097 4096], Float64[4097 -4096]') # entry is exactly 8193.0
@test A32.u == A64.u && A32.v == A64.v
@test dot(A32, A64) == 8192.0f0 * 8193.0 == dot(Matrix(A32), Matrix(A64))
@test dot(A32, Matrix(A64)) == dot(Matrix(A32), Matrix(A64)) # mixed dense operand too
@test dot(Matrix(A64), A32) == dot(Matrix(A64), Matrix(A32))
# mismatched shapes throw instead of silently indexing into the larger operand
@test_throws DimensionMismatch dot(A32, FactoredMatrix(randn(2, 1), randn(2, 1)'))
@test_throws DimensionMismatch dot(A32, randn(2, 2))

# the default isapprox rtol matches dense: max of the per-eltype defaults for mixed
# precisions, and 0 (exact) for integer factorizations
@test isapprox(A32, A64) == isapprox(Matrix(A32), Matrix(A64)) == true
@test !isapprox(FactoredMatrix([100_000_000;;], [1;;]'), FactoredMatrix([100_000_001;;], [1;;]'))

# mixed precision where the narrower operand overflows: promoting the factors would
# erase the Inf32 entry, so the distance is evaluated in each operand's own arithmetic
# (dense compares Inf32 against the finite Float64 entry)
Aof = FactoredMatrix([floatmax(Float32);;], [2.0f0;;]')
Bof = FactoredMatrix([Float64(floatmax(Float32));;], [2.0;;]')
@test Matrix(Aof) == [Inf32;;]
@test isapprox(Aof, Bof) == isapprox(Bof, Aof) == isapprox(Matrix(Aof), Matrix(Bof)) == false

# non-finite factors with well-defined represented entries: norm, sum(abs2) and the
# self-dot reduce the entries directly, since a QR of the factors would produce NaNs
An = FactoredMatrix([Inf 1.0; Inf 2.0], [1.0 1.0]')
@test norm(An) == Inf
@test sum(abs2, An) == Inf
@test dot(An, An) == Inf
@test isnan(norm(FactoredMatrix(fill(NaN, 2, 1), ones(1, 1)')))
# dots with non-finite factors fall back to entrywise evaluation, so dormant zero
# factor entries cannot manufacture 0 * Inf = NaN
Anf = FactoredMatrix([Inf 0.0], [1.0 0.0]') # represents [Inf]
Bnf = FactoredMatrix([0.0 1.0], [0.0 1.0]') # represents [1.0]
@test dot(Anf, Bnf) == dot(Matrix(Anf), Matrix(Bnf)) == Inf
@test dot(Bnf, Anf) == Inf
@test dot(Anf, Matrix(Bnf)) == Inf
@test dot(Matrix(Bnf), Anf) == Inf
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
