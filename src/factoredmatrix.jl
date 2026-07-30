# based on: https://github.com/JuliaLinearAlgebra/LowRankApprox.jl/blob/master/src/lowrankmatrix.jl
# with ideas from: https://github.com/HolyLab/FactoredMatrices.jl

"""
    FactoredMatrix(u, v')

The `m × n` matrix `u * v'` stored in factored form, where `u` is `m × r` and `v` is
`n × r` (typically `r < min(m, n)`).

The second factor must be passed adjointed (or transposed), so that the call reads as
the product it represents: `FactoredMatrix(u, v')` is the matrix `u * v'`. Passing two
plain matrices throws an `ArgumentError`. Vectors are treated as one-column matrices,
so `FactoredMatrix(x, y')` is the outer product of `x` and `y`. Structured matrices
whose adjoint is eager (`Diagonal`, `Hermitian`, triangular, ...) are accepted directly
and taken verbatim as the right multiplicand. Factors with different element types are
promoted to a common element type.

`FactoredMatrix <: Factorization`, so — like the standard-library `Factorization`
types — it supports neither indexing nor iteration. The supported operations (`*`,
`mul!`, `+`, `-`, `dot`, `norm`, `svd`, `\\`, ...) exploit the factors and never
materialize the dense `m × n` matrix.

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

# Types whose adjoint is eager (adjoint(::Diagonal) is a Diagonal, not an Adjoint
# wrapper), so the wrapper convention cannot apply; the second argument is taken
# verbatim as the right multiplicand: the represented matrix is `u * X`.
const EagerAdjointFactor = Union{
    Bidiagonal, Diagonal, Hermitian, LowerTriangular, SymTridiagonal, Symmetric{<:Real},
    Tridiagonal, UnitLowerTriangular, UnitUpperTriangular, UpperTriangular,
}
FactoredMatrix(u::AbstractVecOrMat, X::EagerAdjointFactor) = _FactoredMatrix(_colform(u), _colform(X'))

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

# Internal O(rank) entry accessor, backing the entrywise fallbacks of ==, isapprox,
# iszero, norm, dot and hash. Deliberately not getindex: like the standard-library
# Factorization types, FactoredMatrix exposes no indexing API.
_entry(L::FactoredMatrix, i::Integer, j::Integer) = dot(view(L.v, j, :), view(L.u, i, :))

# Fast path: a zero factor with a finite cofactor gives an exactly zero product
# (0 * Inf = NaN). Nonzero factors can still cancel (e.g. A - A), so otherwise check
# the represented entries, short-circuiting at the first nonzero one.
function Base.iszero(L::FactoredMatrix)
    if iszero(L.u) && all(isfinite, L.v) || all(isfinite, L.u) && iszero(L.v)
        return true
    end
    return all(iszero(_entry(L, i, j)) for i in axes(L.u, 1), j in axes(L.v, 1))
end

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

Exact elementwise equality of the represented matrices, one entry at a time from the
factors (`O(m * n * rank)` work, `O(1)` memory). Exact equality of different
factorizations is unlikely in floating point; [`isapprox`](@ref) is usually what you
want. There is deliberately no equal-factor shortcut: equal factors can still represent
`NaN` entries (e.g. from `0 * Inf`), which compare unequal like in dense arrays.
"""
function Base.:(==)(A::FactoredMatrix, B::FactoredMatrix)
    size(A) == size(B) || return false
    return all(_entry(A, i, j) == _entry(B, i, j) for i in axes(A.u, 1), j in axes(A.v, 1))
end

# isequal uses dense-array semantics (NaN entries equal to themselves), so that
# factorizations work as hashed-collection keys even with NaN entries.
function Base.isequal(A::FactoredMatrix, B::FactoredMatrix)
    size(A) == size(B) || return false
    return all(isequal(_entry(A, i, j), _entry(B, i, j)) for i in axes(A.u, 1), j in axes(A.v, 1))
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
        h = hash(_canonicalzero(_entry(A, 1, 1)), hash(_canonicalzero(_entry(A, m, n)), h))
    end
    return h
end

# Default rtol with Base.rtoldefault's semantics (the larger of the per-eltype
# defaults, e.g. √eps(Float32) for a Float32/Float64 comparison; 0 for integers),
# so the default matches dense isapprox exactly.
_rtoldefault(::Type{T}) where {T <: AbstractFloat} = √eps(T)
_rtoldefault(::Type{<:Real}) = 0
_rtoldefault(A, B, atol) = iszero(atol) ? max(_rtoldefault(real(eltype(A))), _rtoldefault(real(eltype(B)))) : 0

"""
    isapprox(A::FactoredMatrix, B::FactoredMatrix; atol = 0, rtol, nans = false)

Approximate equality `norm(A - B) ≤ max(atol, rtol * max(norm(A), norm(B)))` in the
Frobenius norm, like `isapprox` for arrays. `A - B` is itself low-rank, so its norm is
evaluated from thin QR factorizations of the concatenated factors at
`O((m + n) * (rank(A) + rank(B))²)` cost, never materializing the dense matrices.
When the distance is not finite, falls back to elementwise comparison, like arrays.
"""
function Base.isapprox(
        A::FactoredMatrix, B::FactoredMatrix;
        atol::Real = 0, rtol::Real = _rtoldefault(A, B, atol), nans::Bool = false
    )
    d = norm(A - B)
    if isfinite(d)
        return d ≤ max(atol, rtol * max(norm(A), norm(B)))
    else
        # entrywise fallback, matching isapprox for AbstractArray
        return all(isapprox(_entry(A, i, j), _entry(B, i, j); atol, rtol, nans) for i in axes(A.u, 1), j in axes(A.v, 1))
    end
end

#=== scalar multiples and low-rank sums ===#

# Scalars are folded into one factor. For non-finite scalars the represented entries
# can differ from dense entrywise scaling (0 * Inf = NaN from dormant factor terms);
# this is inherent to the factored form.
Base.:(*)(a::Number, L::FactoredMatrix) = _FactoredMatrix(a * L.u, L.v)
Base.:(*)(L::FactoredMatrix, a::Number) = _FactoredMatrix(L.u, conj(a) * L.v) # u * v' * a = u * (conj(a) * v)'
Base.:(/)(L::FactoredMatrix, a::Number) = _FactoredMatrix(L.u, L.v / conj(a))
Base.:(\)(a::Number, L::FactoredMatrix) = _FactoredMatrix(a \ L.u, L.v)

Base.:(+)(L::FactoredMatrix) = L
Base.:(-)(L::FactoredMatrix) = _FactoredMatrix(-L.u, L.v)

"""
    +(A::FactoredMatrix, B::FactoredMatrix)
    -(A::FactoredMatrix, B::FactoredMatrix)

Lazy sum (difference) by concatenating the factors: the result is a `FactoredMatrix`
of storage rank `rank(A) + rank(B)`. The dense matrices are never formed.
"""
Base.:(+)(A::FactoredMatrix, B::FactoredMatrix) = _FactoredMatrix([A.u B.u], [A.v B.v])
Base.:(-)(A::FactoredMatrix, B::FactoredMatrix) = _FactoredMatrix([A.u B.u], [A.v -B.v])

#=== products (allocating, structure-preserving) ===#

# The product of two factored matrices keeps the smaller rank: u_L (v_L' u_M) v_M' is
# absorbed into one factor pair. Products over an empty contracted dimension are exactly
# zero; the unused (possibly non-finite) factors are replaced by zeros so that
# 0 * Inf = NaN cannot leak into the result.
function Base.:(*)(L::FactoredMatrix, M::FactoredMatrix)
    if size(L.v, 1) == size(M.u, 1) == 0
        r = min(rank(L), rank(M))
        T = _prodtype(eltype(L), eltype(M)) # the accumulation type, like the nonempty branches
        return _FactoredMatrix(zeros(T, size(L.u, 1), r), zeros(T, size(M.v, 1), r))
    elseif rank(L) ≤ rank(M)
        return _FactoredMatrix(copy(L.u), M.v * (M.u' * L.v))
    else
        return _FactoredMatrix(L.u * (L.v' * M.u), copy(M.v))
    end
end

function Base.:(*)(L::FactoredMatrix, A::AbstractMatrix)
    w = A' * L.v
    return _FactoredMatrix(size(L.v, 1) == 0 ? zero(L.u) : copy(L.u), w)
end
function Base.:(*)(A::AbstractMatrix, L::FactoredMatrix)
    u = A * L.u
    return _FactoredMatrix(u, size(L.u, 1) == 0 ? zero(L.v) : copy(L.v))
end

function Base.:(*)(L::FactoredMatrix, x::AbstractVector)
    t = L.v' * x
    return size(L.v, 1) == 0 ? zeros(_prodtype(eltype(L), eltype(x)), size(L.u, 1)) : L.u * t
end
function Base.:(*)(x::Adjoint{<:Any, <:AbstractVector}, L::FactoredMatrix)
    t = x * L.u
    return size(L.u, 1) == 0 ? adjoint(zeros(_prodtype(eltype(x), eltype(L)), size(L.v, 1))) : t * L.v'
end
# xᵀ * L = (Lᵀ * x)ᵀ keeps the row-vector shape of the dense product (without this the
# generic matrix method above would return a 1 × n FactoredMatrix instead).
Base.:(*)(x::Transpose{<:Any, <:AbstractVector}, L::FactoredMatrix) = transpose(transpose(L) * parent(x))

# UniformScaling products reduce to scalar multiplication.
Base.:(*)(L::FactoredMatrix, J::UniformScaling) = L * J.λ
Base.:(*)(J::UniformScaling, L::FactoredMatrix) = J.λ * L

#=== least squares ===#

"""
    \\(L::FactoredMatrix, b)

Minimum-norm least-squares (pseudoinverse) solution of `L * x = b`, via `svd(L)` — the
dense matrix is never materialized. Rank deficiency (including dependent factor columns
from the lazy `+`/`-`) is handled with the same dimension-scaled singular-value cutoff
as `pinv`.
"""
Base.:(\)(L::FactoredMatrix, b::AbstractVecOrMat) = _lssolve(L, b)

# Disambiguate against LinearAlgebra's real-Factorization/complex-RHS fallback.
Base.:(\)(L::FactoredMatrix{T}, b::VecOrMat{Complex{T}}) where {T <: Union{Float32, Float64}} = _lssolve(L, b)

# Solving factor by factor (v' \ (u \ b)) would require both factors to have full
# column rank, which the lazy sums routinely break; the SVD handles any rank profile.
function _lssolve(L::FactoredMatrix, b::AbstractVecOrMat)
    if size(L, 1) ≠ size(b, 1)
        throw(DimensionMismatch("matrix has $(size(L, 1)) rows, right-hand side has $(size(b, 1))"))
    end
    F = svd(L)
    # Singular values at or below pinv's cutoff contribute nothing; with k = 0 (e.g.
    # empty or all-zero S, where tol == 0) the products below yield the zero solution.
    tol = eps(eltype(F.S)) * minimum(size(L)) * (isempty(F.S) ? zero(eltype(F.S)) : first(F.S))
    k = count(>(tol), F.S)
    y = view(F.U, :, 1:k)' * b
    y ./= view(F.S, 1:k)
    return view(F.Vt, 1:k, :)' * y
end

#=== reductions: dot, norm, trace ===#

"""
    dot(A::FactoredMatrix, B::FactoredMatrix)

Frobenius inner product `sum(conj(A[i, j]) * B[i, j])`, evaluated in closed form from
`rank(A) × rank(B)` Gram matrices at `O((m + n) * rank(A) * rank(B))` cost, without
materializing the dense matrices. Mixed element types are evaluated entrywise
(`O(m * n * rank)`), preserving each operand's represented-entry arithmetic.

The closed form reassociates the sum, so relative accuracy degrades for strongly
cancelling factorizations; self-inner products take the QR-stable `sum(abs2, A)` path
instead.
"""
function LinearAlgebra.dot(A::FactoredMatrix, B::FactoredMatrix)
    if size(A) ≠ size(B)
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    end
    T = _prodtype(eltype(A), eltype(B)) # the accumulation type of the inner product
    if length(A) == 0
        return zero(T) # empty sum; the Gram factors could still hold 0 * Inf garbage
    elseif A === B || (eltype(A) === eltype(B) && A.u == B.u && A.v == B.v)
        # same represented matrix, same arithmetic: use the stable nonnegative self-dot
        return convert(T, sum(abs2, A))
    elseif eltype(A) === eltype(B)
        return sum((A.u' * B.u) .* conj.(A.v' * B.v))
    end
    # Mixed element types: promoting the factors would change the represented entries
    # (each operand rounds in its own precision), so evaluate the entrywise sum.
    return sum(dot(_entry(A, i, j), _entry(B, i, j)) for i in axes(A.u, 1), j in axes(A.v, 1))
end
function LinearAlgebra.dot(A::FactoredMatrix, B::AbstractMatrix)
    if size(A) ≠ size(B)
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    end
    if length(A) == 0
        return zero(_prodtype(eltype(A), eltype(B))) # empty sum; avoid 0 * Inf garbage
    elseif eltype(A) === eltype(B)
        return tr(A.u' * (B * A.v))
    end
    return sum(dot(_entry(A, i, j), B[i, j]) for i in axes(A.u, 1), j in axes(A.v, 1))
end
LinearAlgebra.dot(A::AbstractMatrix, B::FactoredMatrix) = conj(dot(B, A))

# Gram-matrix closed form ‖u * v'‖² = sum((u'u) .* conj(v'v)): keeps the accumulation
# type of sum(abs2, Matrix(A)) and stays exact for integer factors (barring overflow).
Base.sum(::typeof(abs2), A::FactoredMatrix) = real(sum((A.u' * A.u) .* conj.(A.v' * A.v)))
# For floating-point factors the Gram form can catastrophically cancel (even below
# zero) when factor columns nearly cancel; the QR-based norm is stable.
Base.sum(::typeof(abs2), A::FactoredMatrix{<:Union{AbstractFloat, Complex{<:AbstractFloat}}}) = norm(A)^2

"""
    norm(A::FactoredMatrix, p = 2)

Frobenius norm of the represented matrix, computed from the factors at `O((m + n) * rank²)`
cost without materializing the dense matrix. Only `p = 2` is supported.

By unitary invariance `‖u * v'‖ = ‖R_u * R_v'‖` with `R_u`, `R_v` the triangular QR
factors. Unlike the Gram form `tr((u'u)(v'v))`, this does not cancel catastrophically
when the represented matrix is much smaller than its factors (e.g. `A - B` of two
nearly-equal matrices), which is what makes [`isapprox`](@ref) reliable.
"""
function LinearAlgebra.norm(A::FactoredMatrix, p::Real = 2)
    p == 2 || throw(ArgumentError("only the Frobenius norm (p = 2) is supported"))
    if length(A) > 0 && !(all(isfinite, A.u) && all(isfinite, A.v))
        # a QR of non-finite factors would contaminate R with NaNs even when the
        # represented entries are well-defined; reduce the entries directly instead
        return norm(_entry(A, i, j) for i in axes(A.u, 1), j in axes(A.v, 1))
    end
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

Singular value decomposition from QRs of the factors and an SVD of the small
`min(m, r) × min(n, r)` core — cheaper than `svd(Matrix(A))` when `r < m, n`.

Returns a reduced `LinearAlgebra.SVD` with `min(m, n, r)` singular values (`U` is
`m × k`, `Vt` is `k × n`); the singular values omitted relative to `svd(Matrix(A))`
are all zero.
"""
function LinearAlgebra.svd(A::FactoredMatrix)
    qru = qr(A.u)
    qrv = qr(A.v)
    Fqr = svd(qru.R * qrv.R')
    U = qru.Q * Fqr.U
    Vt = Fqr.Vt * qrv.Q'
    return SVD(U, Fqr.S, Vt)
end
