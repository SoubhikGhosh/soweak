# Changelog

All notable changes to soweak are documented here. The project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.2.0] — 2026-05-11

Phase 2 of the OWASP roadmap: agent and runtime controls — LLM06 tool
authorization plus LLM10 budgets, rate limits, and streaming repetition
detection.

### Added

- **`soweak.agent`** — LLM06 tool authorization:
  - `@guarded_tool(scopes=[...], approval="auto"|"human", rate_limit_per_minute=N,
    approval_handler=...)` decorator.
  - `authorize(ctx)` context manager (contextvars-based, async-safe).
  - `current_context()` accessor.
  - `ToolCall` / `ToolCallEvent` records.
  - `ApprovalRequired` exception.
  - Audit callback via `ctx.metadata["tool_audit_callback"]`.
  - Granted-scopes via `ctx.metadata["granted_scopes"]`.
- **`soweak.budget`** — LLM10 budgets and rate limits:
  - `TokenBudget(limit)` — per-scope integer token tracker.
  - `CostBudget(limit_usd, pricing=...)` — USD spend tracker with
    `ModelPricing` table and `DEFAULT_PRICING` for common models.
  - `BudgetExceededError` raised on charge overrun.
  - `BudgetEnforcer(budget, scope_attr=...)` — Pipeline enforcer that
    blocks when a scope's remaining budget is zero.
  - `RateLimiter` / `RateLimitEnforcer(requests_per_minute=N)` —
    sliding-window per-scope rate limit.
- **`soweak.streaming`** — `RepetitionDetector` flags output stuck in a
  substring loop (LLM10 quality + cost).
- 36 new tests covering tool auth (scopes, approval, rate limit, audit,
  async-safety), budgets (token + cost + enforcer behaviour), and
  repetition detection.

### Changed

- README and ROADMAP coverage tables updated to reflect Phase 2 status.

---

## [3.1.0] — 2026-05-11

Phase 1 of the OWASP roadmap: bidirectional LLM02 and a real LLM05 toolkit.

### Added

- **LLM02 output DLP** — `output_dlp_detector()` factory and
  `OUTPUT_DLP_PACK` covering every input DLP pattern plus output-specific
  markers (RFC1918 IPs, internal hostnames, database connection strings,
  JWT tokens, AWS ARNs).
- **LLM05 output-handling detectors** —
  `output_html_detector()` / `OUTPUT_HTML_PACK`,
  `output_sql_detector()` / `OUTPUT_SQL_PACK`,
  `output_shell_detector()` / `OUTPUT_SHELL_PACK` flag risky HTML, SQL and
  shell content in model output.
- **`soweak.output`** module:
  - `sanitize_html()` — stdlib-only HTML sanitizer (allowlist tags, strip
    `on*` handlers, drop dangerous URL schemes).
  - `URLAllowlist` — predicate-style scheme/host validator.
  - `is_safe_sql()` — heuristic SQL safety check.
  - `html_sanitizer_enforcer()` — `TransformEnforcer` factory.
- Top-level re-exports for the four new public symbols above.
- 49 new tests covering output detectors, sanitisers and allowlists.

### Changed

- README and ROADMAP coverage tables updated to reflect Phase 1 status.
- v2.x output handling is no longer "Phase 1, planned" — it's shipped.

---

## [3.0.0] — 2026-05-11

**Complete rewrite.** soweak v3 is a middleware framework, not a prompt
scanner. Old APIs are gone.

### Added

- `Boundary` enum (`INPUT`, `RETRIEVAL`, `TOOL_CALL`, `OUTPUT`, `STREAM`).
- `Detector` ABC and `Signal` dataclass — the signal-producer contract.
- `Enforcer` ABC, `Action`, `Decision` — the action-taker contract.
- `Policy`, `Rule`, `PolicyBuilder` — fluent policy definition.
- `Pipeline` — runs a policy at a boundary, emits audit events.
- `AuditEvent`, `AuditLog`, `InMemoryAuditLog`, `JsonLinesAuditLog`.
- `PatternMatchDetector` — regex-driven detector loaded from versioned
  `PatternPack`s.
- Pattern packs for LLM01 prompt injection (input), LLM02 input DLP (PII /
  secrets / API keys), and LLM07 system-prompt extraction attempts (input).
- `CanaryDetector` — output-boundary detector for LLM07 system-prompt
  leakage via canary tokens.
- Enforcers: `BlockEnforcer`, `RedactEnforcer`, `LogOnlyEnforcer`,
  `ThresholdEnforcer`, `TransformEnforcer`.
- Adapters in `soweak.adapters.{langchain,openai,gemini}` plus a shared
  `SecurityError` exception.
- New CLI: `soweak scan`, `soweak list`, `soweak version`.
- `ROADMAP.md` — phased coverage plan for the rest of the OWASP LLM Top 10.

### Removed (breaking)

- `analyze_prompt`, `is_prompt_safe`, `get_risk_score` top-level functions.
- `PromptAnalyzer`, `AnalysisResult` (replaced by `Pipeline` + `Decision`).
- `RiskScorer`, `RiskLevel` (replaced by `ThresholdEnforcer`).
- v2.x detector classes: `PromptInjectionDetector`, `SensitiveInfoDetector`,
  `DataPoisoningDetector`, `OutputHandlingDetector`,
  `ExcessiveAgencyDetector`, `SystemPromptLeakageDetector`,
  `RAGWeaknessDetector`, `MisinformationDetector`,
  `UnboundedConsumptionDetector`, `SupplyChainDetector`.
  Their input-regex approach was not the right defense for 7 of the 10
  OWASP categories. See `ROADMAP.md` for the replacement plan at the
  correct boundary.

### Changed

- Minimum Python is now **3.10** (was 3.8).
- License remains Apache-2.0.
- Project description rewritten to reflect middleware framing.
- Build excludes `tests/` and `examples/` from the wheel.

### Migration

See the "Migrating from v2.x" section in `README.md` for a side-by-side
mapping of old → new APIs.

---

## [2.0.1] — 2026-01-20

- Fixed missing detection for "Ignore all instructions" variant.

## [2.0.0] — 2026-01-19

- Added `SupplyChainDetector` (LLM03 input regex).
- Expanded pattern libraries for LLM05/08/09/10.

## [1.2.0] — 2026-01-19

- LangChain / OpenAI / Google framework integrations added as examples.

## [1.1.0] — 2026-01-19

- License changed from MIT to Apache-2.0.

## [1.0.0] — 2025-01-19

- Initial release: regex-based scanner across all 10 OWASP LLM categories.
