# based on: https://github.com/JuliaLinearAlgebra/LowRankApprox.jl/blob/master/src/lowrankmatrix.jl

struct LowRankMatrix{T,U,V} <: AbstractMatrix{T}
    u::U # m x r Matrix
    v::V # n x r Matrix
    function LowRankMatrix{T,U,V}(u::U, v::V) where {T, U<:AbstractMatrix{T}, V<:AbstractMatrix{T}}
        if size(u, 2) ≠ size(v, 2)
            throw(ArgumentError("u and v must have same number of columns"))
        end
        return new{T,U,V}(u, v)
    end
end

LowRankMatrix(u::AbstractMatrix{T}, v::AbstractMatrix{T}) where {T} = LowRankMatrix{T,typeof(u),typeof(v)}(u,v)
LowRankMatrix(u::AbstractVector, v::AbstractMatrix) = LowRankMatrix(reshape(u,:,1), v)
LowRankMatrix(u::AbstractMatrix, v::AbstractVector) = LowRankMatrix(u, reshape(v,:,1))
LowRankMatrix(u::AbstractVector, v::AbstractVector) = LowRankMatrix(reshape(u,:,1), reshape(v,:,1))

Base.size(L::LowRankMatrix) = (size(L.u, 1), size(L.v, 1))
Base.iszero(L::LowRankMatrix) = iszero(L.u) && all(!isnan, L.v) || all(!isnan, L.u) && iszero(L.v)
LinearAlgebra.rank(L::LowRankMatrix) = size(L.u, 2)
LinearAlgebra.transpose(L::LowRankMatrix) = LowRankMatrix(L.v, L.u)
LinearAlgebra.adjoint(L::LowRankMatrix) = LowRankMatrix(conj(L.v), conj(L.u))

Base.Matrix(L::LowRankMatrix) = L.u * L.v'
Base.Array(L) = Matrix(L)
Base.copy(L::LowRankMatrix) = LowRankMatrix(copy(L.u), copy(L.v))

Base.:(*)(a::Number, L::LowRankMatrix) = LowRankMatrix(a * L.u, L.v)
Base.:(*)(L::LowRankMatrix, a::Number) = LowRankMatrix(L.u, a * L.v)
Base.:(*)(A::LowRankMatrix, B::Adjoint{<:Any, LowRankMatrix}) = A * adjoint(B) # override default

function Base.:(*)(L::LowRankMatrix, M::LowRankMatrix)
    if rank(L) ≤ rank(M)
        return LowRankMatrix(copy(L.u), M.v * (M.u' * L.v))
    else
        return LowRankMatrix(L.u * (L.v' * M.u), copy(M.v))
    end
end

Base.:(*)(L::LowRankMatrix, A::AbstractMatrix) = LowRankMatrix(copy(L.u), A' * L.v)
Base.:(*)(A::AbstractMatrix, L::LowRankMatrix) = LowRankMatrix(A * L.u, copy(L.v))

Base.:(*)(L::LowRankMatrix, x::AbstractVector) = L.u * (L.v' * x)
Base.:(*)(x::Adjoint{<:Any, <:AbstractVector}, L::LowRankMatrix) = (x * L.u) * L.v'

Base.:(\)(L::LowRankMatrix, b::AbstractMatrix) = L.v' \ (L.u \ b)
Base.:(\)(L::LowRankMatrix, b::AbstractVector) = L.v' \ (L.u \ b)

# Resolve ambiguities
Base.:(*)(L::LowRankMatrix, A::Diagonal) = invoke(LowRankMatrix, Tuple{LowRankMatrix,AbstractMatrix}, L, A)
Base.:(*)(L::LowRankMatrix, A::AbstractTriangular) = invoke(LowRankMatrix, Tuple{LowRankMatrix,AbstractMatrix}, L, A)
Base.:(*)(A::Diagonal, L::LowRankMatrix) = invoke(LowRankMatrix, Tuple{AbstractMatrix, LowRankMatrix}, A, L)
Base.:(*)(A::AbstractTriangular, L::LowRankMatrix) = invoke(LowRankMatrix, Tuple{AbstractMatrix, LowRankMatrix}, A, L)
Base.:(*)(A::Transpose{T, <:AbstractVector}, L::LowRankMatrix) where {T} = invoke(LowRankMatrix, Tuple{AbstractMatrix, LowRankMatrix}, A, L)

Base.show(io::IO, ::MIME"text/plain", M::LowRankMatrix) = print(io, "LowRankMatrix{", eltype(M), "} of rank ", rank(M), ".")
