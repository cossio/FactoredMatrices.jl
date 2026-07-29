#=== Workspace: pre-allocated buffers for allocation-free mul! ===#

"""
    Workspace{T}(r::Integer, p::Integer)
    Workspace(L::FactoredMatrix, p::Integer)

Pre-allocated scratch buffers for allocation-free `mul!` with a [`FactoredMatrix`](@ref),
passed via the `cache` keyword: `mul!(C, L, B; cache = ws)`. Here `r` is the storage
rank of the factorization and `p` is the outer size of the other operand:

  - `mul!(C, L, B; cache = ws)` needs `p = size(B, 2)` (`p = 1` for a vector `B`);
  - `mul!(C, A, L; cache = ws)` needs `p = size(A, 1)`;
  - `mul!(C, L, M; cache = ws)` with both operands factored needs `r = rank(L)` and
    `p = rank(M)`, and is fully allocation-free when `C` is a `FactoredMatrix` (one
    small intermediate is still allocated when `C` is dense).

The same `Workspace` can be reused across calls (including adjoint/transpose variants,
which have the same rank) as long as the operand sizes do not change. Create one
`Workspace` per thread/task when multiplying concurrently against the same matrix.
See also [`FactoredMatrices.CachedFactoredMatrix`](@ref) to bundle the buffers with the
matrix.
"""
struct Workspace{T}
    left::Matrix{T} # r × p buffer, holds v' * B when the factored matrix is the left operand
    right::Matrix{T} # p × r buffer, holds A * u when the factored matrix is the right operand
end

Workspace{T}(r::Integer, p::Integer) where {T} = Workspace{T}(Matrix{T}(undef, r, p), Matrix{T}(undef, p, r))
Workspace(L::FactoredMatrix{T}, p::Integer) where {T} = Workspace{T}(rank(L), p)

const MaybeWorkspace = Union{Workspace, Nothing}

# Small intermediates, written into the workspace buffers when a cache is provided.
_lbuf(::Nothing, L::FactoredMatrix, B) = L.v' * B
_lbuf(ws::Workspace, L::FactoredMatrix, B::AbstractMatrix) = mul!(ws.left, L.v', B)
_lbuf(ws::Workspace, L::FactoredMatrix, b::AbstractVector) = mul!(view(ws.left, :, 1), L.v', b)
_rbuf(::Nothing, A::AbstractMatrix, M::FactoredMatrix) = A * M.u
_rbuf(ws::Workspace, A::AbstractMatrix, M::FactoredMatrix) = mul!(ws.right, A, M.u)

#=== mul! into a dense output ===#

# C = α * L * B + β * C = α * u * (v' * B) + β * C
function _mul!(C::AbstractVecOrMat, L::FactoredMatrix, B::AbstractVecOrMat, α::Number, β::Number, cache::MaybeWorkspace)
    return mul!(C, L.u, _lbuf(cache, L, B), α, β)
end

# C = α * A * M + β * C = α * (A * u) * v' + β * C
function _mul!(C::AbstractMatrix, A::AbstractMatrix, M::FactoredMatrix, α::Number, β::Number, cache::MaybeWorkspace)
    return mul!(C, _rbuf(cache, A, M), M.v', α, β)
end

# C = α * L * M + β * C with t = v_L' * u_M of size rank(L) × rank(M); associate so that
# the second (allocated) intermediate is the smaller of rank(L) × n and m × rank(M).
function _mul!(C::AbstractMatrix, L::FactoredMatrix, M::FactoredMatrix, α::Number, β::Number, cache::MaybeWorkspace)
    t = _lbuf(cache, L, M.u)
    if rank(L) * size(M.v, 1) ≤ size(L.u, 1) * rank(M)
        return mul!(C, L.u, t * M.v', α, β)
    else
        return mul!(C, L.u * t, M.v', α, β)
    end
end

#=== mul! into a FactoredMatrix output (3-arg only; β * C is not representable) ===#

# One factor of the product is copied verbatim, the other absorbs t = v_L' * u_M; which
# one depends on whether C was allocated with the rank of L or of M. Allocation-free
# when a cache is provided.
function _fmul!(C::FactoredMatrix, L::FactoredMatrix, M::FactoredMatrix, cache::MaybeWorkspace)
    t = _lbuf(cache, L, M.u)
    if rank(C) == rank(L)
        C.u .= L.u
        mul!(C.v, M.v, t') # C.v' = t * M.v'
    elseif rank(C) == rank(M)
        mul!(C.u, L.u, t)
        C.v .= M.v
    else
        throw(DimensionMismatch("rank of C ($(rank(C))) must equal rank of L ($(rank(L))) or rank of M ($(rank(M)))"))
    end
    return C
end

# C = L * B = u * (v' * B), so C.u = u and C.v = B' * v. Allocation-free, no cache needed.
function _fmul!(C::FactoredMatrix, L::FactoredMatrix, B::AbstractMatrix, ::MaybeWorkspace)
    C.u .= L.u
    mul!(C.v, B', L.v)
    return C
end

# C = A * M = (A * u) * v', so C.u = A * u and C.v = v. Allocation-free, no cache needed.
function _fmul!(C::FactoredMatrix, A::AbstractMatrix, M::FactoredMatrix, ::MaybeWorkspace)
    mul!(C.u, A, M.u)
    C.v .= M.v
    return C
end

# Repack a plain product of dense factors: C = A * B, so C.u = A and C.v = B'.
function _fmul!(C::FactoredMatrix, A::AbstractMatrix, B::AbstractMatrix, ::MaybeWorkspace)
    C.u .= A
    C.v .= B'
    return C
end

#=== public mul! methods ===#

"""
    mul!(C, A, B, [α, β]; cache::Union{Nothing, FactoredMatrices.Workspace} = nothing)

Three- and five-argument `mul!` where `A`, `B` and/or `C` are [`FactoredMatrix`](@ref)es
(or their `Adjoint`/`Transpose` wrappers), exploiting the factored form. Pass a
pre-allocated [`FactoredMatrices.Workspace`](@ref) as `cache` to make repeated products
allocation-free.
"""
mul!

# Signatures must name FactoredMatrix or the (AbstractMatrix-subtyped) wrapper union
# explicitly — never a Union of the two — so that each method is strictly more specific
# than the LinearAlgebra generics and no dispatch ambiguities arise.
for FM in (:FactoredMatrix, :AdjOrTransFM)
    @eval begin
        function LinearAlgebra.mul!(C::AbstractMatrix, A::$FM, B::AbstractMatrix, α::Number, β::Number; cache::MaybeWorkspace = nothing)
            return _mul!(C, rewrap(A), B, α, β, cache)
        end
        function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::$FM, α::Number, β::Number; cache::MaybeWorkspace = nothing)
            return _mul!(C, A, rewrap(B), α, β, cache)
        end
        function LinearAlgebra.mul!(y::AbstractVector, A::$FM, x::AbstractVector, α::Number, β::Number; cache::MaybeWorkspace = nothing)
            return _mul!(y, rewrap(A), x, α, β, cache)
        end
        function LinearAlgebra.mul!(C::AbstractMatrix, A::$FM, B::AbstractMatrix; cache::MaybeWorkspace = nothing)
            return _mul!(C, rewrap(A), B, true, false, cache)
        end
        function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::$FM; cache::MaybeWorkspace = nothing)
            return _mul!(C, A, rewrap(B), true, false, cache)
        end
        function LinearAlgebra.mul!(y::AbstractVector, A::$FM, x::AbstractVector; cache::MaybeWorkspace = nothing)
            return _mul!(y, rewrap(A), x, true, false, cache)
        end
        function LinearAlgebra.mul!(C::FactoredMatrix, A::$FM, B::AbstractMatrix; cache::MaybeWorkspace = nothing)
            return _fmul!(C, rewrap(A), B, cache)
        end
        function LinearAlgebra.mul!(C::FactoredMatrix, A::AbstractMatrix, B::$FM; cache::MaybeWorkspace = nothing)
            return _fmul!(C, A, rewrap(B), cache)
        end
    end
end

for FMA in (:FactoredMatrix, :AdjOrTransFM), FMB in (:FactoredMatrix, :AdjOrTransFM)
    @eval begin
        function LinearAlgebra.mul!(C::AbstractMatrix, A::$FMA, B::$FMB, α::Number, β::Number; cache::MaybeWorkspace = nothing)
            return _mul!(C, rewrap(A), rewrap(B), α, β, cache)
        end
        function LinearAlgebra.mul!(C::AbstractMatrix, A::$FMA, B::$FMB; cache::MaybeWorkspace = nothing)
            return _mul!(C, rewrap(A), rewrap(B), true, false, cache)
        end
        function LinearAlgebra.mul!(C::FactoredMatrix, A::$FMA, B::$FMB; cache::MaybeWorkspace = nothing)
            return _fmul!(C, rewrap(A), rewrap(B), cache)
        end
    end
end

function LinearAlgebra.mul!(C::FactoredMatrix, A::AbstractMatrix, B::AbstractMatrix; cache::MaybeWorkspace = nothing)
    return _fmul!(C, A, B, cache)
end

#=== CachedFactoredMatrix: a FactoredMatrix bundled with its Workspace ===#

"""
    CachedFactoredMatrix(M::FactoredMatrix, ws::Workspace)
    CachedFactoredMatrix(M::FactoredMatrix, p::Integer)

Bundle a [`FactoredMatrix`](@ref) with a pre-allocated [`FactoredMatrices.Workspace`](@ref)
so they travel together through an API that takes a single matrix-like argument.
Products against a `CachedFactoredMatrix` automatically use the bundled buffers as their
`mul!` cache, sparing callers from threading a `cache` argument through every call site.

The bundled `Workspace` carries a single outer size `p`, which must equal `size(B, 2)`
for `cfm * B` and `size(A, 1)` for `A * cfm`. Use one `CachedFactoredMatrix` per task
when multiplying concurrently.
"""
struct CachedFactoredMatrix{T, F <: FactoredMatrix{T}, W <: Workspace{T}} <: Factorization{T}
    M::F
    ws::W
end

CachedFactoredMatrix(M::FactoredMatrix, p::Integer) = CachedFactoredMatrix(M, Workspace(M, p))

Base.size(C::CachedFactoredMatrix) = size(C.M)
Base.size(C::CachedFactoredMatrix, d::Integer) = size(C.M, d)
Base.length(C::CachedFactoredMatrix) = length(C.M)
LinearAlgebra.rank(C::CachedFactoredMatrix) = rank(C.M)
Base.Matrix(C::CachedFactoredMatrix) = Matrix(C.M)
Base.Array(C::CachedFactoredMatrix) = Matrix(C.M)
Base.sum(::typeof(abs2), C::CachedFactoredMatrix) = sum(abs2, C.M)
Base.show(io::IO, C::CachedFactoredMatrix) = print(io, "Cached", C.M)
Base.show(io::IO, ::MIME"text/plain", C::CachedFactoredMatrix) = show(io, C)

LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::AbstractMatrix) = mul!(C, A.M, B; cache = A.ws)
LinearAlgebra.mul!(y::AbstractVector, A::CachedFactoredMatrix, x::AbstractVector) = mul!(y, A.M, x; cache = A.ws)
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::CachedFactoredMatrix) = mul!(C, A, B.M; cache = B.ws)
function LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::AbstractMatrix, α::Number, β::Number)
    return mul!(C, A.M, B, α, β; cache = A.ws)
end
function LinearAlgebra.mul!(y::AbstractVector, A::CachedFactoredMatrix, x::AbstractVector, α::Number, β::Number)
    return mul!(y, A.M, x, α, β; cache = A.ws)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::CachedFactoredMatrix, α::Number, β::Number)
    return mul!(C, A, B.M, α, β; cache = B.ws)
end

# Allocating products: only the output is allocated; the intermediates use the buffers.
function Base.:(*)(A::CachedFactoredMatrix{T}, B::AbstractMatrix) where {T}
    return mul!(Matrix{T}(undef, size(A, 1), size(B, 2)), A.M, B; cache = A.ws)
end
function Base.:(*)(A::CachedFactoredMatrix{T}, x::AbstractVector) where {T}
    return mul!(Vector{T}(undef, size(A, 1)), A.M, x; cache = A.ws)
end
function Base.:(*)(A::AbstractMatrix, B::CachedFactoredMatrix{T}) where {T}
    return mul!(Matrix{T}(undef, size(A, 1), size(B, 2)), A, B.M; cache = B.ws)
end
