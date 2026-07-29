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

# signed zeros: 0.0 == -0.0, so the hashes must also agree
Zp = FactoredMatrix(fill(0.0, 2, 1), ones(2, 1)')
Zm = FactoredMatrix(fill(-0.0, 2, 1), ones(2, 1)')
@test Zp == Zm
@test hash(Zp) == hash(Zm)

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

# complex case
uc = randn(ComplexF64, 6, 2)
vc = randn(ComplexF64, 5, 2)
Ac = FactoredMatrix(uc, vc')
@test Ac == FactoredMatrix(2 * uc, (vc / 2)')
@test hash(Ac) == hash(FactoredMatrix(2 * uc, (vc / 2)'))
@test Ac ≈ FactoredMatrix(3 * uc, (vc / 3)')
@test !(Ac ≈ im * Ac)
