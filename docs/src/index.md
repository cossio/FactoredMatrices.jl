```@meta
CurrentModule = FactoredMatrices
```

# FactoredMatrices.jl

A Julia package implementing a [`FactoredMatrix`](@ref) type: a lazy matrix stored as a product `u * v'` of two factor matrices, `u` (of size `m × r`) and `v` (of size `n × r`), typically with `r < m, n`. Storing the factors instead of the full `m × n` matrix saves memory and makes many operations cheaper when the rank `r` is small.

`FactoredMatrix` is a `LinearAlgebra.Factorization`, not an `AbstractMatrix`: it does not support iteration or the dense generic fallbacks, and instead implements `*`, `mul!`, `+`, `-`, scalar multiples, `dot`, `norm`, `tr`, `svd`, `\` (least-squares / pseudoinverse solution), `isapprox`, and conversion via `Matrix(A)` — all exploiting the factored form.

## Installation

This package is registered in the Julia General registry. Install it with:

```julia
import Pkg
Pkg.add("FactoredMatrices")
```

## Quick start

The second factor is passed adjointed, so the constructor call reads as the product it represents:

```julia
using LinearAlgebra
using FactoredMatrices

u = randn(100, 5)
v = randn(50, 5)
A = FactoredMatrix(u, v') # lazy representation of u * v', of size 100 x 50

size(A) # (100, 50)
rank(A) # 5 (storage rank: the number of factor columns)

x = randn(50)
A * x # efficient: never forms the full 100 x 50 matrix

B = A' * A # products of factored matrices stay factored
C = A + A  # lazy sums by factor concatenation (storage rank 10)
A ≈ C - A  # closed-form Frobenius distance, no dense matrix is built
F = svd(A) # SVD exploiting the factored form

Matrix(A) # materialize the full dense matrix
```

Operations such as scalar multiplication, matrix products, adjoint/transpose, and [`svd`](@ref LinearAlgebra.svd) exploit the factored representation and return `FactoredMatrix` results where possible, without materializing the full dense matrix.

## Allocation-free repeated multiplication

For tight loops, pre-allocate a [`FactoredMatrices.Workspace`](@ref) and pass it via the `cache` keyword of [`mul!`](@ref LinearAlgebra.mul!):

```julia
B = randn(50, 7)
C = zeros(100, 7)
ws = FactoredMatrices.Workspace(A, size(B, 2))

mul!(C, A, B; cache = ws)           # no allocation
mul!(C, A, B, 2.0, 1.0; cache = ws) # five-argument form, also supported
```

Or bundle the buffers with the matrix in a [`FactoredMatrices.CachedFactoredMatrix`](@ref), so any generic code calling `*` or `mul!` uses them automatically.

See the [Reference](@ref) section for the full API documentation.

## Related packages

- [HolyLab/FactoredMatrices.jl](https://github.com/HolyLab/FactoredMatrices.jl)
- [LowRankApprox.jl](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl)
