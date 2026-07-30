# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Three- and five-argument `mul!` methods for all combinations of `FactoredMatrix`, dense matrices and vectors — including `mul!` into a pre-allocated `FactoredMatrix` output. This makes `FactoredMatrix` usable with iterative solvers that require `mul!`.
- `FactoredMatrices.Workspace`: pre-allocated buffers passed to `mul!` via the `cache` keyword, making repeated products allocation-free.
- `FactoredMatrices.CachedFactoredMatrix`: bundles a `FactoredMatrix` with its `Workspace` so `*` and `mul!` use the buffers automatically.
- Closed-form `dot` (Frobenius inner product) and `tr` from small Gram matrices, and `norm`/`sum(abs2, A)` from the Gram closed form guarded against cancellation and underflow, falling back to thin QR of the factors when a guard trips — as fast as the plain closed form on generic inputs, yet stable even when the represented matrix is much smaller than its factors (e.g. `A - B` of nearly-equal matrices). All without materializing the dense matrix.
- Lazy `+` and `-` by factor concatenation (storage ranks add up), and `-A`, `A / a`, `a \ A` scalar operations.
- `isapprox`, comparing the *represented* matrices in the Frobenius norm, evaluated in closed form from the factors at `O((m + n) * (rank(A) + rank(B))²)` cost. (`==`, `isequal` and `hash` follow the standard field-wise `Factorization` fallback, comparing the stored factors.)
- `size(A, d)`, `length`, and docstrings for the public API.
- `FactoredMatrix` factors with different element types are now promoted to a common element type. In particular, multiplication by a type-promoting scalar (e.g. `(1 + im) * A` for a real `A`) now works instead of throwing a `MethodError`.

### Changed

- `FactoredMatrix` is now a `Factorization` instead of an `AbstractMatrix`. It never supported `getindex`-based iteration anyway, so the `AbstractMatrix` generic fallbacks (printing, `==`, reductions, broadcasting) either errored or would have been silently `O(m * n * rank)`; subtyping `Factorization` makes the multiplication-oriented contract explicit. Like the standard-library `Factorization` types, entry indexing is not provided.
- The constructor now requires the second factor to be passed adjointed (or transposed), so the call reads as the product it represents: `FactoredMatrix(u, v')` is the matrix `u * v'`. Passing two plain matrices throws an `ArgumentError` explaining this. (Suggested by Tim Holy.) Structured matrices whose adjoint is computed eagerly rather than as an `Adjoint` wrapper (`Diagonal`, `Hermitian`, triangular, ...) are accepted directly as the second argument and taken verbatim as the right multiplicand.
- `FactoredMatrix` is now exported.
- Minimum supported Julia version is now 1.11 (was 1.9), as required by the ExplicitImports publicness check added to the test suite (`public` markers exist only on Julia 1.11+).

### Fixed

- Right-multiplication by a scalar (`A * a`) conjugated non-real scalars: it scaled the `v` factor by `a`, whose adjoint appears in the represented matrix `u * v'`, so the result was `conj(a) * A` instead of `a * A`. Real scalars were unaffected.
- `adjoint` and `transpose` of a `FactoredMatrix` were swapped for complex element types: `adjoint` conjugated the factors (computing the transpose) and `transpose` did not (computing the adjoint). Real element types were unaffected.
- `iszero` returned `true` for a zero factor paired with a factor containing `Inf` entries, although `0 * Inf = NaN` means the represented matrix is not zero. (`NaN` entries were already handled.)
- `iszero` returned `false` for represented matrices that are zero by cancellation, without any stored factor being zero (e.g. `A - A`, whose factor columns cancel pairwise). When no factor is zero it now checks the represented entries, short-circuiting at the first nonzero one, so nonzero matrices stay cheap.
- `A \ b` solved factor by factor (`v' \ (u \ b)`), which is the pseudoinverse solution only when both factors have full column rank — a condition the lazy `+`/`-` routinely break by concatenating factor columns. It now solves through `svd(A)` (still without materializing the dense matrix), returning the minimum-norm least-squares solution for any rank profile.
