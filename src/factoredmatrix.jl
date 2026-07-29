# based on: https://github.com/JuliaLinearAlgebra/LowRankApprox.jl/blob/master/src/lowrankmatrix.jl

"""
    FactoredMatrix(u, v)

A lazy matrix stored as the product `u * v'`, where `u` is `m × r` and `v` is `n × r`
(typically with `r < m, n`). The resulting matrix has size `m × n` and rank at most `r`.

The full dense matrix can be recovered with `Matrix(A)`. Operations such as scalar
multiplication, matrix products, `adjoint`, `transpose`, and `svd` exploit the factored
representation and return `FactoredMatrix` results where possible.

If `u` or `v` are vectors, they are treated as `m × 1` or `n × 1` matrices, respectively.
If `u` and `v` have different element types, both are converted to their promoted
element type.
"""
struct FactoredMatrix{T, U, V} <: AbstractMatrix{T}
    u::U # m x r Matrix
    v::V # n x r Matrix
    function FactoredMatrix{T, U, V}(u::U, v::V) where {T, U <: AbstractMatrix{T}, V <: AbstractMatrix{T}}
        if size(u, 2) ≠ size(v, 2)
            throw(ArgumentError("u and v must have same number of columns"))
        end
        return new{T, U, V}(u, v)
    end
end

FactoredMatrix(u::AbstractMatrix{T}, v::AbstractMatrix{T}) where {T} = FactoredMatrix{T, typeof(u), typeof(v)}(u, v)

function FactoredMatrix(u::AbstractMatrix, v::AbstractMatrix)
    T = promote_type(eltype(u), eltype(v))
    return FactoredMatrix(convert(AbstractMatrix{T}, u), convert(AbstractMatrix{T}, v))
end
FactoredMatrix(u::AbstractVector, v::AbstractMatrix) = FactoredMatrix(reshape(u, :, 1), v)
FactoredMatrix(u::AbstractMatrix, v::AbstractVector) = FactoredMatrix(u, reshape(v, :, 1))
FactoredMatrix(u::AbstractVector, v::AbstractVector) = FactoredMatrix(reshape(u, :, 1), reshape(v, :, 1))

Base.size(L::FactoredMatrix) = (size(L.u, 1), size(L.v, 1))
Base.iszero(L::FactoredMatrix) = iszero(L.u) && all(!isnan, L.v) || all(!isnan, L.u) && iszero(L.v)
LinearAlgebra.rank(L::FactoredMatrix) = size(L.u, 2)
LinearAlgebra.adjoint(L::FactoredMatrix) = FactoredMatrix(L.v, L.u)
LinearAlgebra.transpose(L::FactoredMatrix) = FactoredMatrix(conj(L.v), conj(L.u))

Base.Matrix(L::FactoredMatrix) = L.u * L.v'
Base.Array(L::FactoredMatrix) = Matrix(L)
Base.copy(L::FactoredMatrix) = FactoredMatrix(copy(L.u), copy(L.v))

Base.:(*)(a::Number, L::FactoredMatrix) = FactoredMatrix(a * L.u, L.v)
Base.:(*)(L::FactoredMatrix, a::Number) = FactoredMatrix(L.u, conj(a) * L.v)
Base.:(*)(A::FactoredMatrix, B::Adjoint{<:Any, <:FactoredMatrix}) = A * adjoint(parent(B)) # override default

function Base.:(*)(L::FactoredMatrix, M::FactoredMatrix)
    if rank(L) ≤ rank(M)
        return FactoredMatrix(copy(L.u), M.v * (M.u' * L.v))
    else
        return FactoredMatrix(L.u * (L.v' * M.u), copy(M.v))
    end
end

Base.:(*)(L::FactoredMatrix, A::AbstractMatrix) = FactoredMatrix(copy(L.u), A' * L.v)
Base.:(*)(A::AbstractMatrix, L::FactoredMatrix) = FactoredMatrix(A * L.u, copy(L.v))

Base.:(*)(L::FactoredMatrix, x::AbstractVector) = L.u * (L.v' * x)
Base.:(*)(x::Adjoint{<:Any, <:AbstractVector}, L::FactoredMatrix) = (x * L.u) * L.v'

Base.:(\)(L::FactoredMatrix, b::AbstractMatrix) = L.v' \ (L.u \ b)
Base.:(\)(L::FactoredMatrix, b::AbstractVector) = L.v' \ (L.u \ b)

# Resolve ambiguities
Base.:(*)(L::FactoredMatrix, A::Diagonal) = invoke(*, Tuple{FactoredMatrix, AbstractMatrix}, L, A)
Base.:(*)(L::FactoredMatrix, A::AbstractTriangular) = invoke(*, Tuple{FactoredMatrix, AbstractMatrix}, L, A)
Base.:(*)(A::Diagonal, L::FactoredMatrix) = invoke(*, Tuple{AbstractMatrix, FactoredMatrix}, A, L)
Base.:(*)(A::AbstractTriangular, L::FactoredMatrix) = invoke(*, Tuple{AbstractMatrix, FactoredMatrix}, A, L)
Base.:(*)(A::Transpose{T, <:AbstractVector}, L::FactoredMatrix) where {T} = invoke(*, Tuple{AbstractMatrix, FactoredMatrix}, A, L)

Base.show(io::IO, ::MIME"text/plain", M::FactoredMatrix) = print(io, "FactoredMatrix{", eltype(M), "} of rank ", rank(M), ".")

"""
    svd(A::FactoredMatrix)

Compute the singular value decomposition of `A`, exploiting the factored representation.
Only QR decompositions of the factors and an SVD of the `min(m, r) × min(n, r)` core are
required, which is cheaper than an SVD of the materialized full matrix when `r < m, n`.

Returns a reduced `LinearAlgebra.SVD` factorization object with `min(m, n, r)` singular
values, where `U` is `m × min(m, n, r)` and `Vt` is `min(m, n, r) × n`. When
`r < min(m, n)` this is smaller than the factorization returned by `svd(Matrix(A))`,
which has `min(m, n)` singular values; the omitted singular values are all zero.
"""
function LinearAlgebra.svd(A::FactoredMatrix)
    qru = qr(A.u)
    qrv = qr(A.v)
    Fqr = svd(qru.R * qrv.R')
    U = qru.Q * Fqr.U
    Vt = Fqr.Vt * qrv.Q'
    return SVD(U, Fqr.S, Vt)
end
