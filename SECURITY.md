# Security Policy

## Reporting a vulnerability

If you believe you've found a security issue in soweak — a bug that
allows an attacker to bypass a documented defense, defeat the audit log,
escalate privilege through a `@guarded_tool`, or compromise the integrity
of stored budgets — **please do not file a public GitHub issue**.

Instead, email **99ghoshsoubhik@gmail.com** with:

* A description of the issue and the impact you believe it has.
* The minimum reproduction (a Python snippet that triggers it, or a YAML
  policy + input that demonstrates the bypass).
* The soweak version (`python -c "import soweak; print(soweak.__version__)"`)
  and the Python version.
* Whether you'd like to be credited; if yes, the name / handle to use.

You'll get an acknowledgement within **3 working days** and a triage
verdict within **10 working days**. If we agree it's a security issue
we'll work with you on a disclosure timeline; the default is 90 days from
the acknowledgement, or sooner if a fix is available.

If you don't hear back within 3 working days, please re-send: mail
occasionally goes astray.

## What's in scope

We treat these as security issues:

* **Bypasses of documented defenses.** A regex that fails to match a
  variant of a documented prompt-injection probe is a bug; a regex that a
  carefully encoded payload sidesteps is in scope.
* **`@guarded_tool` privilege escalation.** Any input that makes a guarded
  tool execute with scopes the caller's context did not grant, or that
  bypasses the human-approval handler, is in scope.
* **`BudgetEnforcer` / `RateLimitEnforcer` evasion.** Inputs or
  concurrent-access patterns that let a single scope consume more than
  its limit are in scope. (Multi-host stores are out of scope for the
  bundled implementations — see below.)
* **Audit-log tamper / drop.** Any pipeline path that fails to record an
  `AuditEvent` for a decision that should have been recorded is in scope.
* **`AuditEvent` data corruption.** A payload that causes
  `JsonLinesAuditLog` or `OpenTelemetryAuditLog` to emit malformed data
  or to crash the host process is in scope.
* **Dependency or build issues.** A malformed wheel that lets an attacker
  inject code at install time is in scope. So is a vendored dependency
  with a known CVE that we ship by default.

## What's out of scope

* **The "soweak doesn't catch a novel jailbreak"** category. Pattern packs
  are heuristics; please file these as feature requests on the public
  tracker.
* **Plausible fabrications that pass `GroundingDetector` /
  `EmbeddingGroundingDetector`**. These are heuristic, not fact-checkers.
  Open an issue if you want to discuss the heuristic; only treat as
  security if the detector silently emits *false positives* on documented
  benign retrieval contexts.
* **Multi-host concurrency on the bundled `InMemory*` stores.** They are
  documented as in-process. The `Sqlite*` stores are documented as
  single-host. For multi-host correctness, implement
  `CounterStore` / `WindowStore` against your distributed backend (Redis,
  Postgres, etc.). A bug in soweak's contract docs is in scope; a bug in
  your subclass is not.
* **Optional-extra integrations failing when the extra isn't installed.**
  We raise `ImportError` with a clear install hint; that's the contract.
* **Anything in `tests/` or `examples/`.** These ship for completeness; we
  don't promise they're hardened.
* **Output sanitization beyond the documented baseline.** `sanitize_html`
  delegates to `bleach` when `pip install soweak[output]` is present. If
  bleach has a CVE, please report it upstream; if soweak invokes bleach
  with insecure arguments, that's in scope.

## Supported versions

We support the **two most recent minor releases** with security fixes —
currently `3.10.x` and `3.11.x`. Older releases get the fix in the
release notes only.

| Version  | Supported          |
| -------- | ------------------ |
| 3.11.x   | ✅                 |
| 3.10.x   | ✅                 |
| 3.0–3.9  | Backports on request |
| < 3.0    | ❌                 |

## What you'll get

* Acknowledgement of receipt within 3 working days.
* A triage verdict (in scope / out of scope / needs more info) within 10
  working days.
* A coordinated-disclosure plan if the issue is in scope.
* Public credit in the release notes (unless you ask otherwise).
* A CVE assignment if the severity warrants one.

## What we'd appreciate

* Don't probe systems you don't own.
* Don't run tests against PyPI or the github.com infrastructure.
* Give us a reasonable disclosure window before publishing.
* Send PoC code as gists, plaintext, or attachments — not links to
  third-party paste services.
