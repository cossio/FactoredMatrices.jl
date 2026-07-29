```@meta
CurrentModule = FactoredMatrices
```

# FactoredMatrices.jl

A Julia package implementing a [`FactoredMatrix`](@ref) type: a lazy matrix stored as a product `u * v'` of two factor matrices, `u` (of size `m × r`) and `v` (of size `n × r`), typically with `r < m, n`. Storing the factors instead of the full `m × n` matrix saves memory and makes many operations cheaper when the rank `r` is small.

## Installation

This package is not registered. Install with:

```julia
import Pkg
Pkg.add(url = "https://github.com/cossio/FactoredMatrices.jl")
```

## Quick start

```julia
using LinearAlgebra
using FactoredMatrices: FactoredMatrix

u = randn(100, 5)
v = randn(50, 5)
A = FactoredMatrix(u, v) # lazy representation of u * v', of size 100 x 50

size(A) # (100, 50)
rank(A) # 5

x = randn(50)
A * x # efficient: never forms the full 100 x 50 matrix

B = A' * A # products of factored matrices stay factored
F = svd(A) # SVD exploiting the factored form

Matrix(A) # materialize the full dense matrix
```

Operations such as scalar multiplication, matrix products, adjoint/transpose, and [`svd`](@ref LinearAlgebra.svd) exploit the factored representation and return `FactoredMatrix` results where possible, without materializing the full dense matrix.

See the [Reference](@ref) section for the full API documentation.

## Related packages

- [HolyLab/FactoredMatrices.jl](https://github.com/HolyLab/FactoredMatrices.jl)
- [LowRankApprox.jl](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl)
