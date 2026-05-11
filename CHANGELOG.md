# Changelog

All notable changes to soweak are documented here. The project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
