# FactoredMatrices

[![CI](https://github.com/cossio/FactoredMatrices.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cossio/FactoredMatrices.jl/actions/workflows/CI.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/FactoredMatrices.jl/stable)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/FactoredMatrices.jl/dev)
[![codecov](https://codecov.io/gh/cossio/FactoredMatrices.jl/graph/badge.svg)](https://codecov.io/gh/cossio/FactoredMatrices.jl)

Store and operate on low-rank matrices in factored form `A = u * v'` without materializing
the full dense matrix, where `u` is `m × r` and `v` is `n × r` (typically `r < min(m, n)`).
Products cost `O((m + n) * r)` per column instead of `O(m * n)`, and many whole-matrix
operations (`dot`, `norm`, `svd`, approximate equality) have closed forms in the factors
that cost far less than forming the `m × n` matrix.

## Installation

This package is not yet registered. Install it with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/FactoredMatrices.jl")
```

(To be registered soon: https://github.com/JuliaRegistries/General/pull/166064)

## Usage

### Construction and products

The second factor is passed adjointed, so the call reads as the product it represents:

```julia
using FactoredMatrices

u = randn(100, 5)
v = randn(80, 5)
A = FactoredMatrix(u, v')   # represents the 100 × 80 matrix u * v'

x = randn(80)
y = A * x                   # 100-element vector, no 100 × 80 matrix formed

B = randn(80, 7)
A * B                       # FactoredMatrix (rank 5): the factored structure is preserved
A' * randn(100, 3)          # adjoints/transposes are again FactoredMatrixes
A * FactoredMatrix(randn(80, 2), randn(30, 2)')  # rank-2 FactoredMatrix
```

`FactoredMatrix` is a `LinearAlgebra.Factorization`, not an `AbstractMatrix`: like the
standard-library `Factorization` types it supports neither indexing nor iteration, and
instead implements `*`, `mul!`, `+`, `-`, scalar multiples, `dot`, `norm`, `tr`, `svd`,
`\` (least squares / pseudoinverse solution) and conversion via `Matrix(A)`.

### Sums, equality, norms — all in factored form

```julia
using LinearAlgebra

C = A + A               # lazy sum by factor concatenation (storage rank 10)
norm(A)                 # Frobenius norm, closed form in the factors
dot(A, C)               # Frobenius inner product, closed form
A ≈ C - A               # closed-form Frobenius distance: no dense matrix is ever built
svd(A)                  # compact SVD from QRs of the factors, O((m + n) r²)
```

`isapprox` works because the difference of two factored matrices is itself low-rank, so
`norm(A - B)` reduces to small Gram-matrix computations of size `rank(A) + rank(B)`.

### Allocation-free repeated multiplication

For tight loops, pre-allocate a `Workspace` and pass it via the `cache` keyword of `mul!`:

```julia
B = randn(80, 7)
C = zeros(100, 7)
ws = FactoredMatrices.Workspace(A, size(B, 2))

for _ in 1:1000
    mul!(C, A, B; cache = ws)    # no allocation
end

mul!(C, A, B, 2.0, 1.0; cache = ws)  # five-argument C = α A B + β C, also supported
```

Or bundle the buffers with the matrix, so any generic code calling `*` or `mul!` uses
them automatically:

```julia
cfm = FactoredMatrices.CachedFactoredMatrix(A, size(B, 2))
cfm * B         # only the output is allocated
mul!(C, cfm, B) # no allocation
```

## Related packages

- [HolyLab/FactoredMatrices.jl](https://github.com/HolyLab/FactoredMatrices.jl): same
  idea with a `U * V` (not `u * v'`) storage convention; several features here originated
  there. See [this benchmark report](https://claude.ai/code/artifact/b7cb20d6-128d-4f57-bbaf-7d8609b812e8)
  for a feature, performance and allocation comparison of the two packages (and a dense
  baseline): the shared `mul!` core performs identically in both, and they differ in
  structure-preserving products, numerically stable norms, and operation coverage.
- [LowRankApprox.jl](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl): low-rank
  approximation algorithms; inspired the original implementation of this package.
- [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl),
  [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl),
  [SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl): general lazy linear
  operator frameworks that can compose factored products, without the factored-form
  closed forms provided here.

## Acknowledgments

Thanks to [Tim Holy](https://github.com/timholy) for suggesting the
`FactoredMatrix(U, V')` constructor, where the second factor is passed adjointed so that
the call reads as the product it represents. `mul!` and various other features here are
ported from his package,
[HolyLab/FactoredMatrices.jl](https://github.com/HolyLab/FactoredMatrices.jl).
