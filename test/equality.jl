using Test: @testset, @test, @test_throws
using FactoredMatrices: FactoredMatrix

u = randn(10, 3)
v = randn(8, 3)
A = FactoredMatrix(u, v')

# ==, isequal and hash are LinearAlgebra's field-wise Factorization fallbacks: they
# compare the stored factors, not the represented matrices. Use ≈ for the latter.
@test A == FactoredMatrix(copy(u), copy(v)')
@test hash(A) == hash(FactoredMatrix(copy(u), copy(v)'))
@test A == A

# different factorizations of the same matrix compare unequal (field-wise), but the
# represented matrices agree, which ≈ detects
B = FactoredMatrix(2 * u, (v / 2)')
@test A != B
@test A ≈ B

# permuting factor columns preserves the represented matrix, but reassociation and fma
# can change the rounded entries, so only approximate equality is guaranteed
P = [3, 1, 2]
C = FactoredMatrix(u[:, P], v[:, P]')
@test A ≈ C

# different factors, and different sizes, are not equal
@test A != FactoredMatrix(randn(10, 3), randn(8, 3)')
@test A != FactoredMatrix(u[1:9, :], v')
@test A != FactoredMatrix(u, v[1:7, :]')

# NaN factor entries are never ==, matching array semantics on the fields
N = FactoredMatrix(fill(NaN, 2, 1), ones(2, 1)')
@test N != N

# field-wise: equal factors compare equal even when the represented entries are NaN
# (0 * Inf), just like lu(A) == lu(A) ignores what the factors reassemble to
Z = FactoredMatrix(zeros(2, 1), fill(Inf, 2, 1)')
@test Z == Z
@test Z == FactoredMatrix(zeros(2, 1), fill(Inf, 2, 1)')

# signed zeros: 0.0 == -0.0 field-wise, while isequal distinguishes them (array semantics)
Zp = FactoredMatrix(fill(0.0, 2, 1), ones(2, 1)')
Zm = FactoredMatrix(fill(-0.0, 2, 1), ones(2, 1)')
@test Zp == Zm
@test !isequal(Zp, Zm)

# isequal and hash use array semantics on the fields (NaN equal to itself), so
# factorizations work as hashed-collection keys
@test isequal(N, N)
@test isequal(Z, copy(Z))
@test hash(Z) == hash(copy(Z))
@test Dict(Z => 1)[copy(Z)] == 1
@test isequal(A, FactoredMatrix(copy(u), copy(v)'))
@test !isequal(A, FactoredMatrix(u[1:9, :], v')) # different u

# empty matrices hash and compare without error
E = FactoredMatrix(zeros(0, 2), zeros(4, 2)')
@test E == FactoredMatrix(zeros(0, 2), zeros(4, 2)')
@test E != FactoredMatrix(zeros(0, 2), randn(4, 2)') # same (empty) matrix, different v
@test hash(E) isa UInt

# isapprox: closed-form Frobenius distance, no dense matrices materialized
D = FactoredMatrix(3 * u, (v / 3)') # same matrix up to roundoff, but not exactly
@test A ≈ D
@test A ≈ A
@test !(A ≈ 2A)
@test !(A ≈ FactoredMatrix(randn(10, 3), randn(8, 3)'))
@test A ≈ A + FactoredMatrix(1.0e-14 * randn(10, 1), randn(8, 1)')
@test isapprox(A, 2A; rtol = 2) # coarse tolerance
@test isapprox(A, A + FactoredMatrix(fill(1.0e-3, 10, 1), fill(1.0e-3, 8, 1)'); atol = 1)
@test !isapprox(A, 2A; atol = 1.0e-3, rtol = 0)
@test_throws DimensionMismatch isapprox(A, FactoredMatrix(randn(9, 3), randn(8, 3)'))

# non-finite entries: the distance is not finite, so isapprox falls back to elementwise
# comparison, treating exactly equal infinities as equal (like isapprox for arrays)
I1 = FactoredMatrix(fill(Inf, 1, 1), ones(1, 1)')
@test I1 ≈ I1
@test I1 ≈ FactoredMatrix(fill(Inf, 1, 1), ones(1, 1)')
@test !(I1 ≈ -I1)
N1 = FactoredMatrix(fill(NaN, 1, 1), ones(1, 1)')
@test !(N1 ≈ N1)
@test isapprox(N1, N1; nans = true)

# complex case
uc = randn(ComplexF64, 6, 2)
vc = randn(ComplexF64, 5, 2)
Ac = FactoredMatrix(uc, vc')
@test Ac == FactoredMatrix(copy(uc), copy(vc)')
@test Ac != FactoredMatrix(2 * uc, (vc / 2)')
@test Ac ≈ FactoredMatrix(2 * uc, (vc / 2)')
@test Ac ≈ FactoredMatrix(3 * uc, (vc / 3)')
@test !(Ac ≈ im * Ac)
