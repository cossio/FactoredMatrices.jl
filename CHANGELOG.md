# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 2.1.0

### Added

- `svd` defined for `FactoredMatrix` using `qr` decomposition.

## 2.0.0

### Breaking changes

- Package is now called `FactoredMatrices`, to avoid conflict with [LowRankMatrices.jl](https://github.com/JuliaLinearAlgebra/LowRankMatrices.jl).
- `LowRankMatrix` is now called `FactoredMatrix`.

## 1.0.0 - yanked

This version was yanked because of a name conflict with [LowRankMatrices.jl].

### Added

- This CHANGELOG file.
- Register at https://github.com/cossio/CossioJuliaRegistry (yanked).
- `LowRankMatrix` type.