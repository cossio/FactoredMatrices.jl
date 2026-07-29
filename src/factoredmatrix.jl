# based on: https://github.com/JuliaLinearAlgebra/LowRankApprox.jl/blob/master/src/lowrankmatrix.jl
# with ideas from: https://github.com/HolyLab/FactoredMatrices.jl

"""
    FactoredMatrix(u, v')

The `m × n` matrix `u * v'` stored in factored form, where `u` is `m × r` and `v` is
`n × r` (typically `r < min(m, n)`).

The second factor must be passed adjointed (or transposed), so that the call reads as
the product it represents: `FactoredMatrix(u, v')` is the matrix `u * v'`. Passing two
plain matrices throws an `ArgumentError`. Vector arguments are treated as one-column
matrices, so `FactoredMatrix(x, y')` is the outer product of the vectors `x` and `y`.
If the factors have different element types, both are converted to their promoted
element type.

`FactoredMatrix <: Factorization`, so it does not support iteration or the generic
`AbstractMatrix` fallbacks. The supported operations (`*`, `mul!`, `+`, `-`, `dot`,
`norm`, `svd`, `\\`, ...) exploit the factors and never materialize the dense `m × n`
matrix. Single entries can be read with `A[i, j]` at `O(r)` cost.

The factors are stored (not copied) in the fields `u` (`m × r`) and `v` (`n × r`).
"""
struct FactoredMatrix{T, U, V} <: Factorization{T}
    u::U # m × r matrix
    v::V # n × r matrix; the represented matrix is u * v'
    function FactoredMatrix{T, U, V}(u::U, v::V) where {T, U <: AbstractMatrix{T}, V <: AbstractMatrix{T}}
        if size(u, 2) ≠ size(v, 2)
            throw(ArgumentError("u and v must have same number of columns"))
        end
        return new{T, U, V}(u, v)
    end
end

# Internal constructor from the stored factors, with `v` in column (n × r) form.
_FactoredMatrix(u::AbstractMatrix{T}, v::AbstractMatrix{T}) where {T} = FactoredMatrix{T, typeof(u), typeof(v)}(u, v)
function _FactoredMatrix(u::AbstractMatrix, v::AbstractMatrix)
    T = promote_type(eltype(u), eltype(v))
    return _FactoredMatrix(convert(AbstractMatrix{T}, u), convert(AbstractMatrix{T}, v))
end

_colform(A::AbstractMatrix) = A
_colform(x::AbstractVector) = reshape(x, :, 1)

FactoredMatrix(u::AbstractVecOrMat, vt::Adjoint{<:Any, <:AbstractVecOrMat}) = _FactoredMatrix(_colform(u), _colform(parent(vt)))
FactoredMatrix(u::AbstractVecOrMat, vt::Transpose{<:Any, <:AbstractVecOrMat}) = _FactoredMatrix(_colform(u), conj(_colform(parent(vt))))
function FactoredMatrix(u::AbstractVecOrMat, v::AbstractVecOrMat)
    throw(
        ArgumentError(
            "FactoredMatrix(u, v) is ambiguous about the product it represents; " *
                "pass the second factor adjointed, as in FactoredMatrix(u, v'), " *
                "which reads as (and represents) the product u * v'"
        )
    )
end

Base.size(L::FactoredMatrix) = (size(L.u, 1), size(L.v, 1))
function Base.size(L::FactoredMatrix, d::Integer)
    d ≥ 1 || throw(ArgumentError("dimension out of range: $d"))
    return d == 1 ? size(L.u, 1) : d == 2 ? size(L.v, 1) : 1
end
Base.length(L::FactoredMatrix) = prod(size(L))

"""
    rank(L::FactoredMatrix)

The storage rank of the factorization: the number of columns of the factors. This is an
upper bound of (not necessarily equal to) the numerical rank of the represented matrix.
"""
LinearAlgebra.rank(L::FactoredMatrix) = size(L.u, 2)

# Single entries are cheap (O(rank)); full indexing semantics are deliberately absent.
Base.getindex(L::FactoredMatrix, i::Integer, j::Integer) = dot(view(L.v, j, :), view(L.u, i, :))

# 0 * Inf = NaN and 0 * NaN = NaN, so a zero factor only gives a zero matrix if the
# other factor is finite.
Base.iszero(L::FactoredMatrix) = iszero(L.u) && all(isfinite, L.v) || all(isfinite, L.u) && iszero(L.v)

Base.adjoint(L::FactoredMatrix) = _FactoredMatrix(L.v, L.u)
Base.transpose(L::FactoredMatrix) = _FactoredMatrix(conj(L.v), conj(L.u))

Base.Matrix(L::FactoredMatrix) = L.u * L.v'
Base.Array(L::FactoredMatrix) = Matrix(L)
Base.copy(L::FactoredMatrix) = _FactoredMatrix(copy(L.u), copy(L.v))

Base.show(io::IO, L::FactoredMatrix) = print(io, "FactoredMatrix{", eltype(L), "} of size ", size(L), " and rank ", rank(L))
Base.show(io::IO, ::MIME"text/plain", L::FactoredMatrix) = show(io, L)

#=== equality ===#

"""
    ==(A::FactoredMatrix, B::FactoredMatrix)

Exact elementwise equality of the represented matrices. Entries are compared one at a
time from the factors (`O(m * n * rank)` work, `O(1)` memory), so the dense matrices are
never materialized. Since floating-point roundoff makes exact equality of two different
factorizations of the same matrix unlikely, [`isapprox`](@ref) is usually what you want.
"""
function Base.:(==)(A::FactoredMatrix, B::FactoredMatrix)
    size(A) == size(B) || return false
    if A.u == B.u && A.v == B.v
        return true # identical factors give identical entries
    end
    return all(A[i, j] == B[i, j] for i in axes(A.u, 1), j in axes(A.v, 1))
end

# Entries that are `==` must hash equally; collapse ±0.0, which hash differently.
_canonicalzero(x) = iszero(x) ? zero(x) : x

# Hash from the represented content (size and sampled entries), never from the factors,
# so that `==` matrices with different factorizations hash equally.
function Base.hash(A::FactoredMatrix, h::UInt)
    h = hash(:FactoredMatrix, h)
    h = hash(size(A), h)
    m, n = size(A)
    if m > 0 && n > 0
        h = hash(_canonicalzero(A[1, 1]), hash(_canonicalzero(A[m, n]), h))
    end
    return h
end

_rtoldefault(A, B, atol) = iszero(atol) ? √eps(float(promote_type(real(eltype(A)), real(eltype(B))))) : 0

"""
    isapprox(A::FactoredMatrix, B::FactoredMatrix; atol = 0, rtol)

Approximate equality `norm(A - B) ≤ max(atol, rtol * max(norm(A), norm(B)))` in the
Frobenius norm, like `isapprox` for arrays. The difference `A - B` is itself low-rank
(rank at most `rank(A) + rank(B)`), so its norm is evaluated from thin QR factorizations
of the concatenated factors at `O((m + n) * (rank(A) + rank(B))²)` cost, without ever
materializing the dense matrices.
"""
function Base.isapprox(A::FactoredMatrix, B::FactoredMatrix; atol::Real = 0, rtol::Real = _rtoldefault(A, B, atol))
    return norm(A - B) ≤ max(atol, rtol * max(norm(A), norm(B)))
end

#=== scalar multiples and low-rank sums ===#

Base.:(*)(a::Number, L::FactoredMatrix) = _FactoredMatrix(a * L.u, L.v)
Base.:(*)(L::FactoredMatrix, a::Number) = _FactoredMatrix(L.u, conj(a) * L.v) # u * v' * a = u * (conj(a) * v)'
Base.:(/)(L::FactoredMatrix, a::Number) = _FactoredMatrix(L.u, L.v / conj(a))
Base.:(\)(a::Number, L::FactoredMatrix) = _FactoredMatrix(a \ L.u, L.v)

Base.:(+)(L::FactoredMatrix) = L
Base.:(-)(L::FactoredMatrix) = _FactoredMatrix(-L.u, L.v)

"""
    +(A::FactoredMatrix, B::FactoredMatrix)
    -(A::FactoredMatrix, B::FactoredMatrix)

Lazy sum (difference) by concatenating the factors: the result is a `FactoredMatrix` of
storage rank `rank(A) + rank(B)`. Ranks add up under repeated summation; the dense
matrices are never formed.
"""
Base.:(+)(A::FactoredMatrix, B::FactoredMatrix) = _FactoredMatrix([A.u B.u], [A.v B.v])
Base.:(-)(A::FactoredMatrix, B::FactoredMatrix) = _FactoredMatrix([A.u B.u], [A.v -B.v])

#=== products (allocating, structure-preserving) ===#

# The product of two factored matrices is factored, with the rank of the lower-rank
# operand: u_L (v_L' u_M) v_M' is absorbed into the factor pair that keeps rank smallest.
function Base.:(*)(L::FactoredMatrix, M::FactoredMatrix)
    if rank(L) ≤ rank(M)
        return _FactoredMatrix(copy(L.u), M.v * (M.u' * L.v))
    else
        return _FactoredMatrix(L.u * (L.v' * M.u), copy(M.v))
    end
end

Base.:(*)(L::FactoredMatrix, A::AbstractMatrix) = _FactoredMatrix(copy(L.u), A' * L.v)
Base.:(*)(A::AbstractMatrix, L::FactoredMatrix) = _FactoredMatrix(A * L.u, copy(L.v))

Base.:(*)(L::FactoredMatrix, x::AbstractVector) = L.u * (L.v' * x)
Base.:(*)(x::Adjoint{<:Any, <:AbstractVector}, L::FactoredMatrix) = (x * L.u) * L.v'

# Explicit Adjoint/Transpose wrappers around a FactoredMatrix are re-wrapped into plain
# FactoredMatrixes (adjoint/transpose of a FactoredMatrix is again a FactoredMatrix).
const AdjOrTransFM = Union{Adjoint{<:Any, <:FactoredMatrix}, Transpose{<:Any, <:FactoredMatrix}}

rewrap(L::FactoredMatrix) = L
rewrap(L::Adjoint{<:Any, <:FactoredMatrix}) = adjoint(parent(L))
rewrap(L::Transpose{<:Any, <:FactoredMatrix}) = transpose(parent(L))

Base.Matrix(L::AdjOrTransFM) = Matrix(rewrap(L))
Base.:(*)(A::FactoredMatrix, B::AdjOrTransFM) = A * rewrap(B)
Base.:(*)(A::AdjOrTransFM, B::FactoredMatrix) = rewrap(A) * B
Base.:(*)(A::AdjOrTransFM, B::AdjOrTransFM) = rewrap(A) * rewrap(B)
Base.:(*)(A::AdjOrTransFM, B::AbstractMatrix) = rewrap(A) * B
Base.:(*)(A::AbstractMatrix, B::AdjOrTransFM) = A * rewrap(B)
Base.:(*)(A::AdjOrTransFM, x::AbstractVector) = rewrap(A) * x
Base.:(*)(x::Adjoint{<:Any, <:AbstractVector}, B::AdjOrTransFM) = x * rewrap(B)
Base.:(*)(x::Transpose{<:Any, <:AbstractVector}, B::AdjOrTransFM) = x * rewrap(B)

#=== least squares ===#

"""
    \\(L::FactoredMatrix, b)

Solve `L * x = b` in the least-squares sense, factor by factor. When `L` is rank
deficient this gives the pseudoinverse (minimum-norm least-squares) solution.
"""
Base.:(\)(L::FactoredMatrix, b::AbstractVecOrMat) = L.v' \ (L.u \ b)

# Disambiguate against LinearAlgebra's real-Factorization/complex-RHS fallback.
Base.:(\)(L::FactoredMatrix{T}, b::VecOrMat{Complex{T}}) where {T <: Union{Float32, Float64}} = L.v' \ (L.u \ b)

#=== reductions: dot, norm, trace ===#

"""
    dot(A::FactoredMatrix, B::FactoredMatrix)

Frobenius inner product `sum(conj(A[i, j]) * B[i, j])`, evaluated in closed form from
`rank(A) × rank(B)` Gram matrices at `O((m + n) * rank(A) * rank(B))` cost, without
materializing the dense matrices.
"""
LinearAlgebra.dot(A::FactoredMatrix, B::FactoredMatrix) = sum((A.u' * B.u) .* conj.(A.v' * B.v))
LinearAlgebra.dot(A::FactoredMatrix, B::AbstractMatrix) = tr(A.u' * (B * A.v))
LinearAlgebra.dot(A::AbstractMatrix, B::FactoredMatrix) = conj(dot(B, A))

Base.sum(::typeof(abs2), A::FactoredMatrix) = norm(A)^2

"""
    norm(A::FactoredMatrix, p = 2)

Frobenius norm of the represented matrix, computed from the factors at `O((m + n) * rank²)`
cost without materializing the dense matrix. Only `p = 2` is supported.

By unitary invariance `‖u * v'‖ = ‖R_u * R_v'‖` with `R_u`, `R_v` the triangular QR
factors of `u` and `v`. Unlike the Gram-matrix closed form `tr((u'u)(v'v))`, this does
not suffer catastrophic cancellation when the represented matrix is much smaller than
its factors (e.g. the difference `A - B` of two nearly-equal factored matrices), which
is what makes the [`isapprox`](@ref) comparison reliable.
"""
function LinearAlgebra.norm(A::FactoredMatrix, p::Real = 2)
    p == 2 || throw(ArgumentError("only the Frobenius norm (p = 2) is supported"))
    return norm(qr(A.u).R * qr(A.v).R')
end

function LinearAlgebra.tr(A::FactoredMatrix)
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are ($m, $n)"))
    return dot(A.v, A.u) # tr(u * v') = sum(u .* conj(v))
end

#=== svd ===#

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
