using Test: @test, @testset
using LinearAlgebra: qr, svd
using FactoredMatrices: FactoredMatrix
using KrylovKit: svdsolve

@testset "svd" begin
    A = FactoredMatrix(randn(20, 3), randn(12, 3)')
    F = svd(Matrix(A))
    F1 = svd(A)

    @test F1.S ≈ F.S[1:3]
    @test F1.U .* sign.(F1.U[1:1, :]) ≈ F.U[:, 1:3] .* sign.(F.U[1:1, 1:3])
    @test F1.Vt .* sign.(F1.Vt[:, 1:1]) ≈ F.Vt[1:3, :] .* sign.(F.Vt[1:3, 1:1])

    # the factorization is reduced: only min(m, n, r) singular values
    @test length(F1.S) == 3
    @test size(F1.U) == (20, 3)
    @test size(F1.Vt) == (3, 12)
end

@testset "Krylov" begin
    A = FactoredMatrix(randn(20, 3), randn(12, 3)')

    # A Factorization is passed to svdsolve as a function computing A * x and A' * x.
    fA(x, ::Val{false}) = A * x
    fA(x, ::Val{true}) = A' * x
    vals, lvecs, rvecs, info = svdsolve(fA, randn(20), 3)
    F = svd(Matrix(A))
    @test length(vals) ≥ 3
    @test vals[1:3] ≈ F.S[1:3]

    for k in 1:3
        @test lvecs[k] * sign(lvecs[k][1]) ≈ F.U[:, k] * sign(F.U[1, k])
        @test rvecs[k] * sign(rvecs[k][1]) ≈ F.V[:, k] * sign(F.V[1, k])
    end
end
