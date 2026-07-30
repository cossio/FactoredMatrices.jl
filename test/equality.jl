using Test: @testset, @test, @test_throws
using FactoredMatrices: FactoredMatrix

u = randn(10, 3)
v = randn(8, 3)
A = FactoredMatrix(u, v')

# equal factors give equal matrices
@test A == FactoredMatrix(copy(u), copy(v)')
@test hash(A) == hash(FactoredMatrix(copy(u), copy(v)'))
@test A == A

# `==` compares the represented matrices, not the factors: scaling a factor by a power
# of two and the other by its inverse is exact in floating point
B = FactoredMatrix(2 * u, (v / 2)')
@test A == B
@test hash(A) == hash(B)

# permuting factor columns preserves the represented matrix, but reassociation and fma
# can change the rounded entries, so only approximate equality is guaranteed
P = [3, 1, 2]
C = FactoredMatrix(u[:, P], v[:, P]')
@test A ≈ C

# different matrices, and different sizes, are not equal
@test A != FactoredMatrix(randn(10, 3), randn(8, 3)')
@test A != FactoredMatrix(u[1:9, :], v')
@test A != FactoredMatrix(u, v[1:7, :]')

# NaN entries are never equal, matching dense array semantics
N = FactoredMatrix(fill(NaN, 2, 1), ones(2, 1)')
@test N != N

# equal factors can still represent NaN entries (0 * Inf = NaN), which compare unequal;
# there must be no equal-factor shortcut
Z = FactoredMatrix(zeros(2, 1), fill(Inf, 2, 1)')
@test Z != Z
@test Z != FactoredMatrix(zeros(2, 1), fill(Inf, 2, 1)')

# signed zeros: 0.0 == -0.0, so the hashes must also agree
Zp = FactoredMatrix(fill(0.0, 2, 1), ones(2, 1)')
Zm = FactoredMatrix(fill(-0.0, 2, 1), ones(2, 1)')
@test Zp == Zm
@test hash(Zp) == hash(Zm)

# isequal compares the represented entries with dense-array semantics: NaN entries are
# equal to themselves, so factorizations with NaN entries work as Dict keys. (No ±0.0
# test: whether an entry comes out as -0.0 depends on the dot accumulation order.)
@test isequal(N, N)
@test isequal(Z, copy(Z))
@test hash(Z) == hash(copy(Z))
@test Dict(Z => 1)[copy(Z)] == 1
@test isequal(A, FactoredMatrix(copy(u), copy(v)'))
@test !isequal(A, FactoredMatrix(u[1:9, :], v')) # size mismatch

# empty matrices hash and compare without error
E = FactoredMatrix(zeros(0, 2), zeros(4, 2)')
@test E == FactoredMatrix(zeros(0, 2), randn(4, 2)')
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
@test Ac == FactoredMatrix(2 * uc, (vc / 2)')
@test hash(Ac) == hash(FactoredMatrix(2 * uc, (vc / 2)'))
@test Ac ≈ FactoredMatrix(3 * uc, (vc / 3)')
@test !(Ac ≈ im * Ac)
