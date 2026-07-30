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

# Element type of a matrix product with operand element types T and S. Products
# accumulate, so this can grow beyond promote_type: e.g. Bool * Bool entries sum to Int.
_prodtype(::Type{T}, ::Type{S}) where {T, S} = typeof(zero(T) * zero(S) + zero(T) * zero(S))

# Whether a buffer with element type T can hold intermediates of element type S.
_fits(::Type{T}, ::Type{S}) where {T, S} = promote_type(T, S) === T

# Small intermediates, written into the workspace buffers when a cache is provided.
# A buffer that cannot hold the intermediate's element type (e.g. a real buffer with a
# complex operand, which would throw InexactError) falls back to allocating, so that a
# cache never changes results. The branches are resolved at compile time.
_lbuf(::Nothing, L::FactoredMatrix, B) = L.v' * B
function _lbuf(ws::Workspace{T}, L::FactoredMatrix, B::AbstractMatrix) where {T}
    if _fits(T, _prodtype(eltype(L), eltype(B)))
        return mul!(ws.left, L.v', B)
    else
        return L.v' * B
    end
end
function _lbuf(ws::Workspace{T}, L::FactoredMatrix, b::AbstractVector) where {T}
    if _fits(T, _prodtype(eltype(L), eltype(b)))
        return mul!(view(ws.left, :, 1), L.v', b)
    else
        return L.v' * b
    end
end
_rbuf(::Nothing, A::AbstractMatrix, M::FactoredMatrix) = A * M.u
function _rbuf(ws::Workspace{T}, A::AbstractMatrix, M::FactoredMatrix) where {T}
    if _fits(T, _prodtype(eltype(A), eltype(M)))
        return mul!(ws.right, A, M.u)
    else
        return A * M.u
    end
end

#=== mul! into a dense output ===#

# When the contracted dimension is empty, the product is exactly zero even if unused
# factor entries are Inf/NaN; multiplying those factors into the exactly-zero rank-sized
# intermediate would manufacture NaNs (0 * Inf). Passing α = false makes mul! scale C by
# β without referencing the factors (a BLAS-level guarantee), keeping the shape checks.

# C = α * L * B + β * C = α * u * (v' * B) + β * C
Base.@inline function _mul!(C::AbstractVecOrMat, L::FactoredMatrix, B::AbstractVecOrMat, α::Number, β::Number, cache::MaybeWorkspace)
    t = _lbuf(cache, L, B)
    if size(L.v, 1) == 0
        return mul!(C, L.u, t, false, β)
    end
    return mul!(C, L.u, t, α, β)
end

# C = α * A * M + β * C = α * (A * u) * v' + β * C
Base.@inline function _mul!(C::AbstractMatrix, A::AbstractMatrix, M::FactoredMatrix, α::Number, β::Number, cache::MaybeWorkspace)
    t = _rbuf(cache, A, M)
    if size(M.u, 1) == 0
        return mul!(C, t, M.v', false, β)
    end
    return mul!(C, t, M.v', α, β)
end

# C = α * L * M + β * C with t = v_L' * u_M of size rank(L) × rank(M); associate so that
# the second (allocated) intermediate is the smaller of rank(L) × n and m × rank(M).
Base.@inline function _mul!(C::AbstractMatrix, L::FactoredMatrix, M::FactoredMatrix, α::Number, β::Number, cache::MaybeWorkspace)
    t = _lbuf(cache, L, M.u)
    if size(L.v, 1) == 0
        return mul!(C, L.u * t, M.v', false, β)
    elseif rank(L) * size(M.v, 1) ≤ size(L.u, 1) * rank(M)
        return mul!(C, L.u, t * M.v', α, β)
    else
        return mul!(C, L.u * t, M.v', α, β)
    end
end

#=== mul! into a FactoredMatrix output (3-arg only; β * C is not representable) ===#

# Verbatim factor copy with a strict shape check: broadcasting `.=` would silently
# expand singleton dimensions (e.g. copying an m × 1 factor into an m × 2 one).
function _copyfactor!(dst::AbstractMatrix, src::AbstractMatrix)
    if size(dst) ≠ size(src)
        throw(DimensionMismatch("output factor has size $(size(dst)), expected $(size(src))"))
    end
    return copyto!(dst, src)
end

# One factor of the product is copied verbatim, the other absorbs t = v_L' * u_M; which
# one depends on whether C was allocated with the rank of L or of M. Allocation-free
# when a cache is provided.
function _fmul!(C::FactoredMatrix, L::FactoredMatrix, M::FactoredMatrix, cache::MaybeWorkspace)
    t = _lbuf(cache, L, M.u)
    if size(L.v, 1) == 0 # exactly zero product; avoid 0 * Inf from unused factor values
        if size(C.u, 1) ≠ size(L.u, 1) || size(C.v, 1) ≠ size(M.v, 1)
            throw(DimensionMismatch("C has size $(size(C)), expected $((size(L.u, 1), size(M.v, 1)))"))
        end
        fill!(C.u, zero(eltype(C.u)))
        fill!(C.v, zero(eltype(C.v)))
        return C
    elseif rank(C) == rank(L)
        _copyfactor!(C.u, L.u)
        mul!(C.v, M.v, t') # C.v' = t * M.v'
    elseif rank(C) == rank(M)
        mul!(C.u, L.u, t)
        _copyfactor!(C.v, M.v)
    else
        throw(DimensionMismatch("rank of C ($(rank(C))) must equal rank of L ($(rank(L))) or rank of M ($(rank(M)))"))
    end
    return C
end

# C = L * B = u * (v' * B), so C.u = u and C.v = B' * v. Allocation-free, no cache needed.
function _fmul!(C::FactoredMatrix, L::FactoredMatrix, B::AbstractMatrix, ::MaybeWorkspace)
    _copyfactor!(C.u, L.u)
    mul!(C.v, B', L.v)
    return C
end

# C = A * M = (A * u) * v', so C.u = A * u and C.v = v. Allocation-free, no cache needed.
function _fmul!(C::FactoredMatrix, A::AbstractMatrix, M::FactoredMatrix, ::MaybeWorkspace)
    mul!(C.u, A, M.u)
    _copyfactor!(C.v, M.v)
    return C
end

# Repack a plain product of dense factors: C = A * B, so C.u = A and C.v = B'.
function _fmul!(C::FactoredMatrix, A::AbstractMatrix, B::AbstractMatrix, ::MaybeWorkspace)
    _copyfactor!(C.u, A)
    if size(C.v) ≠ (size(B, 2), size(B, 1))
        throw(DimensionMismatch("output factor has size $(size(C.v)), expected $((size(B, 2), size(B, 1)))"))
    end
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

The bundled `Workspace` carries a single outer size `p` (e.g. `size(B, 2)` for
`cfm * B`, or `size(A, 1)` for `A * cfm`). The buffers are used only for products they
fit, in shape and element type; other products still work, falling back to allocating
their intermediates. Use one `CachedFactoredMatrix` per task when multiplying
concurrently.
"""
struct CachedFactoredMatrix{T, F <: FactoredMatrix{T}, W <: Workspace{T}} <: Factorization{T}
    M::F
    ws::W
end

CachedFactoredMatrix(M::FactoredMatrix, p::Integer) = CachedFactoredMatrix(M, Workspace(M, p))

# The adjoint/transpose share the buffers: the rank is unchanged, so products against
# the adjointed matrix use the same intermediate sizes.
Base.adjoint(C::CachedFactoredMatrix) = CachedFactoredMatrix(adjoint(C.M), C.ws)
Base.transpose(C::CachedFactoredMatrix) = CachedFactoredMatrix(transpose(C.M), C.ws)

Base.size(C::CachedFactoredMatrix) = size(C.M)
Base.size(C::CachedFactoredMatrix, d::Integer) = size(C.M, d)
Base.length(C::CachedFactoredMatrix) = length(C.M)
LinearAlgebra.rank(C::CachedFactoredMatrix) = rank(C.M)
Base.Matrix(C::CachedFactoredMatrix) = Matrix(C.M)
Base.Array(C::CachedFactoredMatrix) = Matrix(C.M)
Base.sum(::typeof(abs2), C::CachedFactoredMatrix) = sum(abs2, C.M)
Base.show(io::IO, C::CachedFactoredMatrix) = print(io, "Cached", C.M)
Base.show(io::IO, ::MIME"text/plain", C::CachedFactoredMatrix) = show(io, C)

# The bundled buffers are used only when they fit the operation: the intermediate's
# element type must not grow beyond the buffer's, and the buffer must have the shape the
# operation needs (it carries a single outer size `p`). Otherwise the product falls back
# to allocating its intermediates, preserving ordinary multiplication semantics.
_pdim(B::AbstractMatrix) = size(B, 2)
_pdim(B::FactoredMatrix) = rank(B)
_pdim(B::AdjOrTransFM) = rank(parent(B))

function _left_cache(A::CachedFactoredMatrix{T}, B::Union{AbstractMatrix, FactoredMatrix}) where {T}
    return _prodtype(T, eltype(B)) === T && size(A.ws.left) == (rank(A.M), _pdim(B)) ? A.ws : nothing
end
function _left_cache(A::CachedFactoredMatrix{T}, b::AbstractVector) where {T}
    ok = _prodtype(T, eltype(b)) === T && size(A.ws.left, 1) == rank(A.M) && size(A.ws.left, 2) ≥ 1
    return ok ? A.ws : nothing
end
function _right_cache(B::CachedFactoredMatrix{T}, A::AbstractMatrix) where {T}
    return _prodtype(T, eltype(A)) === T && size(B.ws.right) == (size(A, 1), rank(B.M)) ? B.ws : nothing
end
# For factored × cached products the FactoredMatrix × FactoredMatrix path uses the left
# buffer, with shape rank(A) × rank(B.M). When the ranks differ, the bundled right
# buffer may have exactly that shape instead — hand it over as the left buffer then.
function _right_cache(B::CachedFactoredMatrix{T}, A::Union{FactoredMatrix, AdjOrTransFM}) where {T}
    _prodtype(T, eltype(A)) === T || return nothing
    shape = (rank(rewrap(A)), rank(B.M))
    if size(B.ws.left) == shape
        return B.ws
    elseif size(B.ws.right) == shape
        return Workspace{T}(B.ws.right, B.ws.left)
    else
        return nothing
    end
end

# The forwarding methods call the positional internals directly: routing the bundled
# buffers through the `cache` keyword would box the Union-typed value on Julia 1.11.
Base.@inline LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::AbstractMatrix) = _mul!(C, A.M, B, true, false, _left_cache(A, B))
Base.@inline LinearAlgebra.mul!(y::AbstractVector, A::CachedFactoredMatrix, x::AbstractVector) = _mul!(y, A.M, x, true, false, _left_cache(A, x))
Base.@inline LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::CachedFactoredMatrix) = _mul!(C, A, B.M, true, false, _right_cache(B, A))
Base.@inline function LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::AbstractMatrix, α::Number, β::Number)
    return _mul!(C, A.M, B, α, β, _left_cache(A, B))
end
Base.@inline function LinearAlgebra.mul!(y::AbstractVector, A::CachedFactoredMatrix, x::AbstractVector, α::Number, β::Number)
    return _mul!(y, A.M, x, α, β, _left_cache(A, x))
end
Base.@inline function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::CachedFactoredMatrix, α::Number, β::Number)
    return _mul!(C, A, B.M, α, β, _right_cache(B, A))
end

# Factored operands (FactoredMatrix, its wrappers, or another CachedFactoredMatrix)
# are supported like on a plain FactoredMatrix.
for BT in (:FactoredMatrix, :AdjOrTransFM)
    @eval begin
        LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::$BT) = _mul!(C, A.M, rewrap(B), true, false, _left_cache(A, B))
        LinearAlgebra.mul!(C::FactoredMatrix, A::CachedFactoredMatrix, B::$BT) = _fmul!(C, A.M, rewrap(B), _left_cache(A, B))
        LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::$BT, α::Number, β::Number) = _mul!(C, A.M, rewrap(B), α, β, _left_cache(A, B))
        LinearAlgebra.mul!(C::AbstractMatrix, A::$BT, B::CachedFactoredMatrix) = _mul!(C, rewrap(A), B.M, true, false, _right_cache(B, A))
        LinearAlgebra.mul!(C::FactoredMatrix, A::$BT, B::CachedFactoredMatrix) = _fmul!(C, rewrap(A), B.M, _right_cache(B, A))
        LinearAlgebra.mul!(C::AbstractMatrix, A::$BT, B::CachedFactoredMatrix, α::Number, β::Number) = _mul!(C, rewrap(A), B.M, α, β, _right_cache(B, A))
    end
end
LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::CachedFactoredMatrix) = _mul!(C, A.M, B.M, true, false, _left_cache(A, B.M))
LinearAlgebra.mul!(C::FactoredMatrix, A::CachedFactoredMatrix, B::CachedFactoredMatrix) = _fmul!(C, A.M, B.M, _left_cache(A, B.M))
function LinearAlgebra.mul!(C::AbstractMatrix, A::CachedFactoredMatrix, B::CachedFactoredMatrix, α::Number, β::Number)
    return _mul!(C, A.M, B.M, α, β, _left_cache(A, B.M))
end

# Products with factored operands stay factored (like FactoredMatrix products), so no
# large intermediates arise and the buffers are not needed.
Base.:(*)(A::CachedFactoredMatrix, B::FactoredMatrix) = A.M * B
Base.:(*)(A::FactoredMatrix, B::CachedFactoredMatrix) = A * B.M
Base.:(*)(A::CachedFactoredMatrix, B::CachedFactoredMatrix) = A.M * B.M
Base.:(*)(A::CachedFactoredMatrix, B::AdjOrTransFM) = A.M * rewrap(B)
Base.:(*)(A::AdjOrTransFM, B::CachedFactoredMatrix) = rewrap(A) * B.M

# Allocating products with dense operands: the output has the element type of the
# ordinary matrix product; the intermediates use the buffers whenever they fit.
function Base.:(*)(A::CachedFactoredMatrix{T}, B::AbstractMatrix) where {T}
    S = _prodtype(T, eltype(B))
    return _mul!(Matrix{S}(undef, size(A, 1), size(B, 2)), A.M, B, true, false, _left_cache(A, B))
end
function Base.:(*)(A::CachedFactoredMatrix{T}, x::AbstractVector) where {T}
    S = _prodtype(T, eltype(x))
    return _mul!(Vector{S}(undef, size(A, 1)), A.M, x, true, false, _left_cache(A, x))
end
function Base.:(*)(A::AbstractMatrix, B::CachedFactoredMatrix{T}) where {T}
    S = _prodtype(T, eltype(A))
    return _mul!(Matrix{S}(undef, size(A, 1), size(B, 2)), A, B.M, true, false, _right_cache(B, A))
end
