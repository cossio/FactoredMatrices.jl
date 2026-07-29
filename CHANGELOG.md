# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Documentation built with Documenter.jl and deployed to GitHub Pages, including docstrings for `FactoredMatrix` and `svd`.
- CI and documentation badges in the README.

### Changed

- Minimum supported Julia version is now 1.11 (was 1.9), as required by the ExplicitImports publicness check added to the test suite (`public` markers exist only on Julia 1.11+).

### Fixed

- `adjoint` and `transpose` of a `FactoredMatrix` were swapped for complex element types: `adjoint` conjugated the factors (computing the transpose) and `transpose` did not (computing the adjoint). Real element types were unaffected.
