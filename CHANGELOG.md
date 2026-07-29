# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `FactoredMatrix` factors with different element types are now promoted to a common element type. In particular, multiplication by a type-promoting scalar (e.g. `(1 + im) * A` for a real `A`) now works instead of throwing a `MethodError`.

### Changed

- Minimum supported Julia version is now 1.11 (was 1.9), as required by the ExplicitImports publicness check added to the test suite (`public` markers exist only on Julia 1.11+).

### Fixed

- Right-multiplication by a scalar (`A * a`) conjugated non-real scalars: it scaled the `v` factor by `a`, whose adjoint appears in the represented matrix `u * v'`, so the result was `conj(a) * A` instead of `a * A`. Real scalars were unaffected.
- `adjoint` and `transpose` of a `FactoredMatrix` were swapped for complex element types: `adjoint` conjugated the factors (computing the transpose) and `transpose` did not (computing the adjoint). Real element types were unaffected.
