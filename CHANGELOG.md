# Changelog

All notable changes to soweak are documented here. The project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.8.0] — 2026-05-11

ML classifier as an optional Detector. Regex misses paraphrased and
novel injection prompts; a learned classifier fills that gap.

### Added

- **`soweak.ml`** — `MLClassifierDetector(classifier, threshold, ...)`
  accepts any ``Callable[[str], float]`` returning an injection
  probability. Yields a Signal when the probability is at or above
  ``threshold``. The signal's ``confidence`` is the classifier's
  probability.
- `soweak.ml.transformers_classifier(model, device, ...)` — Hugging
  Face factory. Loads tokenizer + model at construction time and
  returns a thread-safe callable for use with `MLClassifierDetector`.
  Requires `pip install soweak[ml]` (pulls `transformers` + `torch`).
- New extras: `[ml]` (transformers + torch), `[yaml]` (PyYAML;
  unused in v3.8, reserved for v3.9 policy loader).

### Notes

- The classifier protocol is intentionally dependency-free. Users can
  plug in a small sklearn pipeline, an HTTP service, or an ONNX model
  without installing `[ml]`.
- `[ml]` is **not** in `[all]` — torch is hundreds of MB and we don't
  want to surprise users opting into LangChain or OpenAI extras.

---

## [3.7.0] — 2026-05-11

State persistence for budgets and rate limits. Production deployments
with multiple replicas can now share state via a SQLite file, and any
single-replica deployment can survive restarts without losing budgets.

### Added

- **`soweak.storage`** — two pluggable interfaces:
  - `CounterStore.add(key, delta, limit=None)` atomic add-or-reject.
  - `WindowStore.record(key, ts, window)` sliding-window event store.
- Backends:
  - `InMemoryCounterStore` / `InMemoryWindowStore` (default; reset on restart).
  - `SqliteCounterStore(path)` / `SqliteWindowStore(path)` —
    file-backed, restart-survival, single-host. Multi-host deployments
    can subclass either ABC for Redis / Postgres / DynamoDB.
- `TokenBudget`, `CostBudget` accept a `store=CounterStore(...)` argument.
- `RateLimiter`, `RateLimitEnforcer` accept `store=WindowStore(...)` and
  `window_seconds=` (default 60).
- Top-level re-exports of all six storage symbols.

### Changed

- `TokenBudget` / `CostBudget` internal state moved out of in-class
  dicts and into the store. Default backend (in-memory) is unchanged
  semantically — existing code keeps working without modification.
- Atomic check-and-charge now goes through `CounterStore.add(limit=...)`,
  which avoids the rollback race that existed in the previous
  in-class-lock implementation.

---

## [3.6.0] — 2026-05-11

Async surface and streaming-output guard.

### Added

- `Detector.ainspect()` and `Enforcer.adecide()` default async methods on
  the ABCs. Both delegate to the existing sync methods, so all built-in
  detectors and enforcers work in async pipelines without changes.
- `Pipeline.arun()` plus `acheck_input` / `acheck_output` /
  `acheck_retrieval` / `acheck_tool_call`.
- `StreamingPipeline(pipeline, scan_every_chars=200, boundary=...)` —
  guards an async iterator of text chunks (e.g. an LLM streaming
  response). Re-scans the accumulating buffer every N chars and once
  more on stream completion. Raises `SecurityError` the moment any
  STREAM (or OUTPUT, by config) rule blocks.
- `pytest-asyncio` added to dev extras; `asyncio_mode = "auto"`.

### Changed

- `soweak` re-exports `StreamingPipeline`.

---

## [3.5.0] — 2026-05-11

Phase 5 of the OWASP roadmap: observability and a red-team CLI. soweak
now covers every defendable layer of the OWASP LLM Top 10.

### Added

- **`soweak.observability`** — `OpenTelemetryAuditLog`: bridges every
  `AuditEvent` into an OTEL span (one span per boundary invocation,
  signals attached as span events). Opt-in via `pip install soweak[otel]`.
  Matched text is **not** recorded by default (it often contains the
  sensitive value you're trying not to leak); set
  `record_matched_text=True` to include it.
- **`soweak.redteam`** — probe runner:
  - Bundled `DEFAULT_PROBES` covering LLM01 / LLM02 / LLM07 attack
    surface (11 probes).
  - `Probe`, `ProbeResult`, `CategoryCoverage` dataclasses.
  - `run_probes(pipeline, probes=DEFAULT_PROBES)` — execute each probe
    at its declared boundary.
  - `coverage_report(results)` — per-category blocked/total/rate.
  - `load_corpus(path)` — JSON corpus loader (validates category +
    boundary).
- **CLI: `soweak redteam`** — replay the probe corpus against the
  default policy or any importable `MODULE:ATTR` Policy. Outputs a
  per-category coverage table or JSON. Useful as the final smoke test
  in CI before shipping a policy change.
- New `[otel]` install extra; added to `[all]`.
- 20 new tests (red-team runner, coverage, custom corpus / custom
  policy via CLI, OTEL adapter — gracefully skips when SDK absent).

### Coverage milestone

soweak v3.5 ships defenses at the **correct boundary** for **all 10
OWASP LLM categories**, with honest framing in the README where any
defense is partial. See `ROADMAP.md` for the per-category status.

---

## [3.4.0] — 2026-05-11

Phase 4 of the OWASP roadmap: build- and deploy-time audit tooling for
LLM03 (Supply Chain) and LLM04 (Data/Model Poisoning), plus a policy
linter.

### Added

- **`soweak.audit_tools`** — pure-Python audit primitives:
  - `hash_file(path, algorithm="sha256")` — streamed digest, fine for
    multi-GB weights.
  - `verify_against_manifest(path, manifest)` — verify a model artifact
    against a `{filename: sha256}` manifest.
  - `list_python_packages()` / `check_packages_against_blocklist()` —
    enumerate installed distributions and flag any on a blocklist.
  - `Canary`, `CanaryResult`, `run_canaries(canaries, call_model)` —
    deploy-time behavioural canary battery for LLM04 drift detection.
  - `lint_policy(policy)` — static checks on a soweak Policy (empty,
    missing INPUT/OUTPUT boundaries, rules without detectors,
    duplicate detector classes).
- **CLI: `soweak audit <subcmd>`** —
  - `soweak audit model PATH [--manifest JSON]`
  - `soweak audit deps [--blocklist FILE] [--json]`
  - `soweak audit canaries --corpus FILE --model MODULE:FUNC [--json]`
  - `soweak audit policy MODULE:ATTR [--json]`
- 30 new tests covering hashing, manifest verify, dep listing,
  blocklist filter, canary runner, policy linter, and all four CLI
  subcommands (including module-import paths for canary models and
  policies).

### Honest scope

- LLM03 and LLM04 cannot be defended at inference time. The audit
  CLI is positioned as **build/CI tooling** that runs before deploy —
  not as runtime detection.

---

## [3.3.0] — 2026-05-11

Phase 3 of the OWASP roadmap: RAG-layer defenses (LLM08) and grounding /
citation checks (LLM09).

### Added

- **`soweak.rag`** — retrieval boundary detectors:
  - `IndirectInjectionDetector` — runs the prompt-injection pack against
    each retrieved document; flags 2nd-order injection payloads coming
    from the corpus.
  - `TenantIsolationDetector(tenant_key="tenant_id")` — flags retrieved
    documents whose tenant key doesn't match `ctx.tenant_id` (or is
    missing). Critical-severity on mismatch.
  - `ProvenanceDetector(required_keys=...)` — flags documents lacking
    any of source/url/uri/doc_id.
  - `RetrievalAnomalyDetector(max_deviation=3.0)` — flags score outliers
    (median-MAD based, robust to skew).
  - All accept dict, LangChain-style objects, or plain strings.
- **`soweak.grounding`** — output boundary detectors for LLM09:
  - `CitationRequiredDetector` — signals when long output contains no
    citation marker.
  - `GroundingDetector` — heuristic lexical-overlap check between output
    sentences and the retrieval context. Reads context from
    `ctx.metadata["retrieved_text"]` or `["retrieved_documents"]`.
- 26 new tests covering RAG and grounding paths, including pipeline
  integration that blocks cross-tenant retrieval.

### Notes

- LLM09 coverage is honestly partial. `GroundingDetector` cannot detect
  plausible fabrication that shares vocabulary with the source. Treat
  signals as "worth a human look", not "definitely false."

---

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
