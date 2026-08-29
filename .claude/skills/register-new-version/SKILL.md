---
name: register-new-version
description: Release a new version of FactoredMatrices.jl and register it in the Julia General registry, suggesting a ColPrac-compliant version number while leaving the final version decision to the user. Use when asked to release, register, tag, or publish a new package version.
---

# Releasing a new version

During development `version` in Project.toml stays at the last released number and changes accumulate under an `## Unreleased` section in CHANGELOG.md. The release commit is what bumps the version.

This package's release history was reset before `v1.0.0`: the older tags and GitHub releases were deleted, and `v1.0.0` (registered in the Julia General registry) is the first release under the new history — no tags older than `v1.0.0` exist.

## Choosing the version number (ColPrac semver)

Before anything else, analyze the changes and ask the user to choose the version. Never make the final version decision for the user.

1. Review the actual changes since the last registered version — the `## Unreleased` CHANGELOG entries **and** the commit history/diff since the previous release tag (`git log vLAST..main`), since the CHANGELOG may be incomplete. Run `git fetch --tags` first, since a shallow clone may have no tag refs.
2. Classify every change using [ColPrac's extension of SemVer for Julia packages](https://docs.sciml.ai/ColPrac/stable/#Guidance-on-Package-Releases):
   - **Post-1.0:** bump major for breaking changes, minor for non-breaking features, and patch for bug fixes.
   - **Pre-1.0:** bump minor for breaking changes and patch for every non-breaking feature or bug fix.
   - Treat all documented APIs as public, including unexported names documented for normal use. Introducing a deprecation is non-breaking; removing one is breaking.
   - Treat dependency or Julia compatibility changes as non-breaking features, unless a dependency API exposed through this package makes the user-facing change breaking. Treat a compatibility change made solely to fix a bug as a bug fix.
   - Treat a correction to clearly broken behavior as a bug fix even when behavior changes incompatibly. Do not classify internal implementation changes, replacing an exception with non-error behavior, unspecified exception types or messages, floating-point details, new exports or supertypes, or textual representations as breaking solely for that reason.
3. Derive one suggested version from the highest bump required by the accumulated changes.
4. Present the suggestion with a brief explanation identifying the changes that drive the bump, then ask the user to confirm it or choose another version. Do not edit release files, commit, push, or trigger registration until the user explicitly chooses the final version.
5. If the **user** proposes a number, still perform the same review rather than accepting it blindly. If it conflicts with ColPrac, push back once with a brief explanation and ask them to confirm or revise. The user's decision is always final, including for borderline classifications.

## Procedure

1. **Release commit.** After the user explicitly chooses `X.Y.Z`, make a single commit titled `vX.Y.Z` that sets `version = "X.Y.Z"` in Project.toml, renames the `## Unreleased` CHANGELOG section to `## X.Y.Z`, and adds a fresh empty `## Unreleased` section above it. Land it on `main` (directly, or via a PR).

2. **Wait for CI to pass on the release commit.** Push the release commit and let `CI.yml` finish on that exact SHA before going further. Registration is immutable, and neither Registrator nor the registry's AutoMerge runs this package's test suite — so registering ahead of CI can publish a permanently broken version. Check with:

   ```bash
   gh api repos/cossio/FactoredMatrices.jl/commits/<sha>/check-runs \
     --jq '.check_runs[] | "\(.name)\t\(.status)\t\(.conclusion)"'
   ```

   Note that `CI.yml` triggers only on pushes to `main`, so a release commit sitting on a PR branch has no CI result yet; wait until it has landed on `main`.

3. **Trigger Registrator on the release commit.** Open that exact commit on GitHub (the merge commit if the release landed via PR) and post this comment directly on it:

   ```markdown
   @JuliaRegistrator register

   Release notes:

   ## Breaking changes

   - blah
   ```

   Use the CHANGELOG entries for this version as the release notes. A commit comment pins registration to that commit, so no release branch or registration issue is needed. Post through the GitHub commit page or the commit-comments API (`gh api repos/cossio/FactoredMatrices.jl/commits/<sha>/comments -f body="..."`). Registrator replies on the commit with a link to the General registry PR; the notes flow into that PR and the GitHub release.

4. **Monitor the registry PR until it merges.** AutoMerge normally merges it within ~15–30 minutes. Watch for AutoMerge failures (version-increment, compat, or project-file checks) and comments from registry maintainers. If changes are needed, commit the fixes to `main` while keeping Project.toml at `X.Y.Z`, then post a new Registrator comment on the corrected commit. Registrator updates the registration to that commit. If the GitHub tooling in the session cannot read the General repo directly, read the public registry PR page.

5. **Tag and GitHub release.** Once the registry PR merges, TagBot creates the `vX.Y.Z` tag at the registered commit and the GitHub release with the notes automatically — no action needed. This is handled by `.github/workflows/TagBot.yml`, which needs the `DOCUMENTER_KEY` secret to be present; if the tag has not appeared within ~30 minutes of the registry PR merging, check that workflow's runs before tagging by hand.

6. **Confirm the cycle is closed.** Verify the registry PR merged, the `vX.Y.Z` tag exists, and CHANGELOG.md carries an empty `## Unreleased` section ready for the next development cycle.

Worked example (sibling package, same procedure): [the RestrictedBoltzmannMachines.jl v5.3.2 registration comment](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a4dcb8cee859c752881c6c1bb6051edaffcecf84#commitcomment-188488609).
