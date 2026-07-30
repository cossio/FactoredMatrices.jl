"""
    FactoredMatrices

Store and operate on low-rank matrices in factored form `A = u * v'` without
materializing the full dense matrix. The main type is [`FactoredMatrix`](@ref).

For allocation-free repeated multiplications, create a
[`FactoredMatrices.Workspace`](@ref) and pass it to `mul!` via the `cache` keyword,
or bundle it with the matrix in a [`FactoredMatrices.CachedFactoredMatrix`](@ref).
"""
module FactoredMatrices

import LinearAlgebra
using LinearAlgebra: Adjoint, Bidiagonal, Diagonal, Factorization, Hermitian, LowerTriangular, SVD,
    SymTridiagonal, Symmetric, Transpose, Tridiagonal, UniformScaling, UnitLowerTriangular,
    UnitUpperTriangular, UpperTriangular, dot, mul!, norm, qr, rank, svd

export FactoredMatrix
public Workspace, CachedFactoredMatrix

include("factoredmatrix.jl")
include("mul.jl")

end
