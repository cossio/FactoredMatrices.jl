using KrylovKit: svdsolve
using LinearAlgebra: svd
using FactoredMatrices: FactoredMatrix
using Test: @test, @testset

@testset "Krylov" begin
    A = FactoredMatrix(randn(20, 3), randn(12, 3))

    vals, lvecs, rvecs, info = svdsolve(A)
    F = svd(Matrix(A))
    @test vals[1:3] ≈ F.S[1:3]
    @test length(vals) == 4
    @test abs(vals[4]) < 1e-10
    @test abs(F.S[4]) < 1e-10

    for k = 1:3
        @test lvecs[k] * sign(lvecs[k][1]) ≈ F.U[:,k] * sign(F.U[1,k])
        @test rvecs[k] * sign(rvecs[k][1]) ≈ F.V[:,k] * sign(F.V[1,k])
    end
end
