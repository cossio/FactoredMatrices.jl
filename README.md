# FactoredMatrices

[![CI](https://github.com/cossio/FactoredMatrices.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cossio/FactoredMatrices.jl/actions/workflows/CI.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/FactoredMatrices.jl/stable)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/FactoredMatrices.jl/dev)
[![codecov](https://codecov.io/gh/cossio/FactoredMatrices.jl/graph/badge.svg)](https://codecov.io/gh/cossio/FactoredMatrices.jl)

Implements a `FactoredMatrix` type. This is a matrix that is stored as a product of two matrices, `U` and `V`, where `U` is `m` by `r` and `V` is `r` by `n` (typically `r < m, n`).

See also: https://github.com/HolyLab/FactoredMatrices.jl.
