# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Minimum supported Julia version is now 1.11 (was 1.9), as required by the ExplicitImports publicness check added to the test suite (`public` markers exist only on Julia 1.11+).
- Test dependencies are now managed through a Pkg workspace: `Project.toml` declares `test` as a workspace project, and `test/Project.toml` lists FactoredMatrices itself as a dependency sourced from the package root. This is the recommended setup on Julia 1.12+, where all workspace projects resolve into a single root manifest; on Julia 1.11 testing still works through the standalone test project fallback.

### Fixed

- `adjoint` and `transpose` of a `FactoredMatrix` were swapped for complex element types: `adjoint` conjugated the factors (computing the transpose) and `transpose` did not (computing the adjoint). Real element types were unaffected.
