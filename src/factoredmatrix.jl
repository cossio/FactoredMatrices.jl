# based on: https://github.com/JuliaLinearAlgebra/LowRankApprox.jl/blob/master/src/lowrankmatrix.jl

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
Base.:(*)(L::FactoredMatrix, a::Number) = FactoredMatrix(L.u, a * L.v)
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

function LinearAlgebra.svd(A::FactoredMatrix)
    qru = qr(A.u)
    qrv = qr(A.v)
    Fqr = svd(qru.R * qrv.R')
    U = qru.Q * Fqr.U
    Vt = Fqr.Vt * qrv.Q'
    return SVD(U, Fqr.S, Vt)
end
