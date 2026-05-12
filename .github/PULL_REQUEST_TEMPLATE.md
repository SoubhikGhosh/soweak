<!--
Before submitting, please read CONTRIBUTING.md. soweak has architectural
rules that aren't obvious from the file tree — PRs that violate them
will be sent back for revision.
-->

## What

One or two sentences describing the change.

## Why

The user-visible problem this solves. If it's a new defense, cite the
attack class or OWASP category.

## Architectural fit

- [ ] The defense runs at the **right boundary** (see `CONTRIBUTING.md`).
- [ ] Any partial / heuristic claims are labelled as such in code AND in
      `THREAT_MODEL.md`.
- [ ] New optional dependencies are lazy-imported and gated behind an
      `[extra]` install group.
- [ ] Public API additions are re-exported from `soweak.__init__` and
      registered in `soweak.config` if they belong in the YAML DSL.

## Tests

- [ ] New tests added for the changed code path.
- [ ] `pytest -W error::ResourceWarning` passes locally.
- [ ] `mypy soweak` passes locally.
- [ ] `ruff check soweak tests` passes locally.

## Docs

- [ ] `CHANGELOG.md` updated under the unreleased section.
- [ ] `ROADMAP.md` updated if this changes OWASP coverage status.
- [ ] `THREAT_MODEL.md` updated if this changes in-scope / out-of-scope.
- [ ] `MIGRATION.md` updated if there's a backward-incompatible change.

## Breaking changes

- [ ] No
- [ ] Yes (described in `MIGRATION.md` and called out in the PR description)
