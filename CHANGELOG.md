# Changelog

All notable changes to soweak are documented here. The project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.11.0] — 2026-05-12

Production-hardening release. Closes every blocking / production-gap
issue identified in the v3.10 internal review. CI-ready, mypy-clean,
ResourceWarning-free.

### Added

- **Anthropic adapter** — `soweak.adapters.anthropic.SecureAnthropic`
  wraps the official `anthropic` client. Install with
  `pip install soweak[anthropic]`. Runnable example in
  `examples/anthropic_example.py`. Closes the README claim of "Anthropic
  support" that v3.10 made but did not deliver.
- **`[output]` extras** — installs `bleach` + `sqlparse`. When present,
  `sanitize_html` delegates to `bleach` (charset-aware, mutation-XSS
  resistant) and `is_safe_sql` delegates to `sqlparse` (real SQL
  grammar). Behaviour is transparent: callers don't change anything;
  the optional path runs silently when the deps are installed.
- **Library logger** — `logging.getLogger("soweak")` is created at
  import with a `NullHandler` per stdlib best practice. Callers
  configure handlers as they see fit.
- **`Budget` protocol** — `runtime_checkable` `Protocol` shared by
  `TokenBudget`, `CostBudget`, and any user-defined budget.
  `BudgetEnforcer` types its `budget` parameter against the protocol
  so custom budgets compose cleanly.
- **`PatternPack.require_version("MAJOR.MINOR")`** — callers can pin
  pack version requirements (raises `ValueError` if older).
- **`Decision` factory methods** — `Decision.warn(...)`,
  `Decision.block(...)`, `Decision.redact(...)`,
  `Decision.transform(...)` round out `Decision.allow(...)` so custom
  enforcers don't need to name `Action` constants directly.
- **`gather_retrieval()`, `split_sentences()`, `tokenize()`** — promoted
  from private helpers in `soweak.grounding` to public API. Aliases for
  the underscored names remain for back-compat in the v3.x line.
- **`@guarded_tool(rate_limit_store=..., rate_limit_window_seconds=...)`**
  — per-tool rate limits now share state with the framework's
  `WindowStore` infrastructure. Pass `SqliteWindowStore` (or any
  custom backend) for multi-replica deployments.
- **`AuditLog.arecord()`** — async path on the audit-log ABC. Default
  runs sync `record()` in the loop's default executor so blocking
  sinks don't stall `Pipeline.arun`.
- **Numpy fast-path in `cosine_similarity()`** — uses numpy when
  available, falls back to pure Python. No new required dependency.
- **`transformers_classifier(...).warmup()`** — pre-runs one inference
  so the first real call doesn't take the cold-start hit. Attached to
  the returned callable.
- **i18n-friendly grounding tokenizer** — handles CJK, Cyrillic,
  Arabic, Urdu. `split_sentences()` recognises Unicode terminators
  (`。 ! ? ؟ ۔`) including for scripts that don't use whitespace
  between sentences.
- **6 new test modules** (456 tests total, ↑ 138):
  - `test_adapters.py` — mock-based adapter tests for all four SDKs
    (OpenAI, Anthropic, Gemini, LangChain). Adapter coverage went from
    0% to fully exercised.
  - `test_robustness.py` — ReDoS guard (every pack handles 10K-char
    adversarial inputs in < 1.5s), i18n tokenizer tests, pack version
    validation.
  - `test_cli_errors.py` — CLI error-path coverage (bad model specs,
    missing files, wrong policy types, etc.).
  - `test_config_errors.py` — `build_policy` error-message coverage.
  - Concurrency tests in `test_storage.py` — 8-thread hammer against
    `TokenBudget.charge` and `RateLimiter.allow` verifies the
    atomicity claim.
- **`SECURITY.md`** — vulnerability disclosure policy with response
  SLAs.
- **`THREAT_MODEL.md`** — explicit in-scope / out-of-scope list, per
  OWASP category, plus operating assumptions and known limitations.
- **`MIGRATION.md`** — version-to-version upgrade notes for v3.6+.
- **`.github/workflows/ci.yml`** — pytest matrix across Python
  3.10/3.11/3.12/3.13 plus mypy, ruff, twine check.
- **`.github/workflows/release.yml`** — tag-triggered TestPyPI / PyPI
  release.
- **`.github/ISSUE_TEMPLATE/`** + **`PULL_REQUEST_TEMPLATE.md`** — bug
  report, feature request, and PR templates.

### Fixed

- **SQLite stores no longer leak connections.** `SqliteCounterStore`
  and `SqliteWindowStore` hold one connection for the store's lifetime,
  lock-serialised, with explicit `close()` and context-manager
  protocol. Previously, every operation opened a new connection and
  Python 3.13's `sqlite3` context-manager semantics meant the
  connections were never closed (just committed). Under load this
  exhausted file descriptors. Fix verified by running the test suite
  under `pytest -W error::ResourceWarning`.
- **`JsonLinesAuditLog` no longer reopens its file per record.** Holds
  the file handle open for the log's lifetime, `flush()`es per record,
  context-manager-protocol for clean shutdown.
- **`mypy soweak` runs clean.** The strict-mode config in
  `pyproject.toml` is now actually obeyed. Fixed 5 latent errors
  including a `PackageMetadata.get()` call that was wrong on every
  supported Python version and a missing return annotation on
  `SecurityError.signals`.
- **`Pipeline.arun` captures the original boundary** before enforcers
  mutate the payload, so audit emissions reflect where the rule ran,
  not where the (possibly transformed) payload ended up.
- **`@guarded_tool` rate limiter is no longer hard-coded in-process.**
  Accepts an optional `rate_limit_store=WindowStore`, matching the
  v3.7 storage infrastructure that other limiters already used.

### Changed

- **`Severity` is now an `IntEnum`.** Sorts, compares, and works with
  plain-int callers without `.value` reach-through. `.label` and
  `.weight` properties are unchanged.
- **`Pipeline.arun` uses `AuditLog.arecord`** instead of sync `record`,
  so blocking audit sinks don't stall the event loop.
- **`sanitize_html` and `is_safe_sql` are tiered**: stdlib-only
  baseline always, stronger `bleach` / `sqlparse` backends when
  `[output]` is installed. Same Python signature in both cases.
- **`pyproject.toml` mypy `[[overrides]]`** expanded to cover every
  optional third-party module the library lazy-imports.
- **`grounding` regex internals** — `_TOKEN_RE` is now Unicode-aware
  (`\w{3,}`), `split_sentences` handles CJK/Arabic/Urdu terminators.
- **Examples are tested via mocks in CI** rather than relying on a
  reviewer to manually run them against real APIs.

### Honest framing

- The earlier v3.0 — v3.10 series shipped over a compressed
  development period and reflects the initial public publication of
  soweak rather than ten months of incremental releases. v3.11 is the
  first release that has CI, has been mypy-audited end-to-end, and
  has had its production-gap list closed out.
- `THREAT_MODEL.md` now documents explicit out-of-scope items rather
  than leaving them implicit.

### Migration

See `MIGRATION.md`. No breaking changes for code that already used the
documented public API. Recommended: switch SQLite-store and
`JsonLinesAuditLog` callers to context-manager form (`with ... as`)
for clean shutdown.

---

## [3.0.0 – 3.10.0] — initial publication (2026-05-11)

soweak's initial public surface, shipped as ten minor versions to keep
each capability shift on a clean semver boundary. All ten versions were
cut on the same day; release order matches the architectural roadmap.

* **3.0.0** — rewrite as an LLM-security middleware framework. Replaces
  the v2 input-regex scanner with the boundary/policy/pipeline model.
  Ships LLM01 (prompt injection), LLM02 input DLP, LLM07 system-prompt
  leakage detectors and the LangChain / OpenAI / Gemini adapters.
* **3.1.0** — bidirectional LLM02 (output DLP) and the LLM05 output
  toolkit: HTML / SQL / shell pattern packs, `sanitize_html`,
  `URLAllowlist`, `is_safe_sql`, `html_sanitizer_enforcer`.
* **3.2.0** — LLM06 tool authorization (`@guarded_tool`, `authorize`,
  scopes / approval / rate limit), LLM10 budgets (`TokenBudget`,
  `CostBudget`, `BudgetEnforcer`, `RateLimitEnforcer`), streaming
  `RepetitionDetector`.
* **3.3.0** — LLM08 RAG defenses (`IndirectInjectionDetector`,
  `TenantIsolationDetector`, `ProvenanceDetector`,
  `RetrievalAnomalyDetector`) and LLM09 grounding heuristics
  (`CitationRequiredDetector`, lexical `GroundingDetector`).
* **3.4.0** — LLM03 supply-chain + LLM04 canary CLI: `soweak audit
  model / deps / canaries / policy`. Pure build-time tooling.
* **3.5.0** — observability and red-team CLI:
  `OpenTelemetryAuditLog`, `soweak.redteam` probe corpus + coverage
  report, `soweak redteam` CLI.
* **3.6.0** — async surface and `StreamingPipeline`. `Detector.ainspect`
  / `Enforcer.adecide` defaults; `Pipeline.arun` + `acheck_*` helpers.
* **3.7.0** — pluggable state persistence for budgets and rate limits.
  `CounterStore` / `WindowStore` interfaces + `InMemory*` + `Sqlite*`
  backends.
* **3.8.0** — optional ML classifier detector for LLM01.
  `MLClassifierDetector` (dep-free), `transformers_classifier`
  factory under `[ml]` extras.
* **3.9.0** — declarative YAML / JSON policy DSL. `load_policy`,
  `build_policy`, full detector/enforcer registry.
* **3.10.0** — more ML model integrations: toxicity classifier,
  LLM-as-judge adapter, embedding-based grounding
  (`EmbeddingGroundingDetector` + sentence-transformers factory under
  `[embeddings]`).

For per-version specifics see the git log for the corresponding tag.

---

## [2.x and earlier]

Pre-rewrite. soweak v2 was a single-file regex-on-input scanner with
ten "OWASP detectors" that scanned user prompts for problems whose
real defense lives at other boundaries. v3 was a clean break; v2.x is
unsupported and not on PyPI.
