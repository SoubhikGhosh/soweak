# soweak roadmap

This document describes how soweak evolves from a regex-on-input scanner into a
real LLM-security middleware framework — and which OWASP LLM Top 10 (2025)
categories we can credibly defend versus which are outside the scope of any
runtime library.

The honest premise: **only 1 of the 10 OWASP LLM categories is solvable by
scanning a user prompt with regex.** The other 9 require being in the *right
place* in the pipeline — retrieval, tool-call, output, or build-time. soweak's
architecture is a hookable middleware framework so you can put a defense at
the boundary where it actually matters.

## Architectural model

```text
       ┌──────────────────────────────────────────────────────────────┐
       │                          your app                            │
       │                                                              │
user ──┼──▶ on_input ──▶ retriever ──▶ on_retrieval ──▶ LLM ──▶ tool? │
       │      │              │                          │       │     │
       │      ▼              ▼                          ▼       ▼     │
       │   pipeline       pipeline                  on_output  on_tool_call
       │      │              │                          │       │     │
       │      ▼              ▼                          ▼       ▼     │
       │   decision       decision                   decision  decision
       └──────────────────────────────────────────────────────────────┘
```

Every boundary runs the same primitive — a `Pipeline` that asks `Detector`s for
`Signal`s and asks an `Enforcer` for a `Decision` (allow, warn, redact,
transform, require-approval, block). All decisions are recorded in an
`AuditLog`.

## OWASP LLM coverage plan

| OWASP                          | Right defense layer                                  | soweak phase |
| ------------------------------ | ---------------------------------------------------- | ------------ |
| LLM01 Prompt Injection         | Input scan + indirect-injection scan of retrieved/tool text | **v3.0**     |
| LLM02 Sensitive Info           | Bidirectional DLP (scrub on input, scan on output)   | **v3.1**     |
| LLM03 Supply Chain             | Build/CI tooling — model hash, SBOM, manifest checks | v3.4 (CLI)   |
| LLM04 Data & Model Poisoning   | Behavioral canary harness at deploy time             | v3.4 (advisory) |
| LLM05 Improper Output Handling | Output sanitizer toolkit (HTML/SQL/shell/URL)        | **v3.1**     |
| LLM06 Excessive Agency         | Tool authorization framework + human-in-the-loop     | **v3.2**     |
| LLM07 System Prompt Leakage    | Canary tokens + output leak detector                 | **v3.0**     |
| LLM08 Vector & Embedding       | Retriever middleware + tenant isolation              | v3.3         |
| LLM09 Misinformation           | Grounding/citation checks (partial — no silver bullet) | v3.3       |
| LLM10 Unbounded Consumption    | Budgets, rate limits, repetition detection           | **v3.2**     |

Bold rows are what v3.0 ships. The rest are phased.

## Phase 0 — Foundations (v3.0.0) ✅

What ships:

- `Policy`, `PolicyBuilder`, `Rule`, `Pipeline` — the boundary-hook engine.
- `Detector` ABC, `Signal` dataclass — signal producers.
- `Enforcer` ABC, `Decision`, `Action` — action takers.
- `AuditEvent`, `AuditLog`, `InMemoryAuditLog`, `JsonLinesAuditLog`.
- `PatternMatchDetector` driven by curated `PatternPack`s for:
  - LLM01 prompt injection (input)
  - LLM02 input DLP (PII / secrets / API keys)
  - LLM07 system-prompt extraction attempts (input)
- `CanaryDetector` for LLM07 output-side leakage detection.
- Enforcers: `BlockEnforcer`, `RedactEnforcer`, `LogOnlyEnforcer`,
  `ThresholdEnforcer`, `TransformEnforcer`.
- `soweak scan` CLI driven by a `Policy`.
- Adapter examples: LangChain, OpenAI, Google Gemini.

What we explicitly **dropped** from v2.x because it was theatre:

- `MisinformationDetector` — keyword blocklist that didn't address LLM09.
- `DataPoisoningDetector` — poisoning happens at training time, not inference.
- `SupplyChainDetector` — supply-chain risk is a build-time concern.
- `ExcessiveAgencyDetector` — agency is granted by architecture, not detected in a prompt.
- `RAGWeaknessDetector` — needs to wrap the retriever, not scan the prompt.
- `UnboundedConsumptionDetector` — solved by rate limits, not regex.
- `OutputHandlingDetector` (input-side) — output handling needs to sanitize *output*.

These all return in later phases at the correct boundary.

## Phase 1 — Bidirectional I/O (v3.1) ✅

What shipped:

- **LLM02 bidirectional DLP**: `output_dlp_detector()` extends the input
  pack with internal IPs, hostnames, connection strings, JWTs, and AWS
  ARNs.
- **LLM05 output toolkit**: `sanitize_html()`, `URLAllowlist`,
  `is_safe_sql()`, and the `html_sanitizer_enforcer()` `TransformEnforcer`
  factory; plus `output_html_detector()`, `output_sql_detector()`,
  `output_shell_detector()` for raising signals at the output boundary.

What deferred to v3.1.x or later:

- **LLM01 ML classifier**: optional local model via `extras=["ml"]` for
  injection cases that regex can't catch.
- Optional Presidio backend via `extras=["pii"]`.
- Spotlighting / delimiter encoding helpers for untrusted input spans.

## Phase 2 — Agent & runtime (v3.2) ✅

What shipped:

- **`soweak.agent`** — `@guarded_tool` decorator with scopes, optional
  human approval, per-(tool, user) rate limit; `authorize(ctx)`
  contextvars-based context manager; `ToolCall` / `ToolCallEvent` audit
  records via `ctx.metadata["tool_audit_callback"]`.
- **`soweak.budget`** — `TokenBudget`, `CostBudget` (with `ModelPricing`
  table + `DEFAULT_PRICING` for common models), `BudgetEnforcer`,
  `RateLimiter`, `RateLimitEnforcer`. All thread-safe.
- **`soweak.streaming`** — `RepetitionDetector` for output-loop pathology
  (LLM10 cost + quality).

Deferred to later minor releases:

- Streaming pipeline integration (back-pressure / circuit-breaker tying
  budgets to in-flight requests).

## Phase 3 — RAG & grounding (v3.3)

- **LLM08 retriever middleware**: `on_retrieval` hook enforces tenant
  isolation, tags documents with provenance, runs indirect-injection scan on
  retrieved content, detects retrieval anomalies.
- Vector-store adapters: pgvector, Pinecone, Weaviate, Chroma, Qdrant.
- **LLM09 grounding checks** (partial): citation requirement, grounding
  score (output vs. retrieved context), optional LLM-as-judge backend.

## Phase 4 — Build / CI tooling (v3.4)

Not a runtime concern — ships as `soweak audit` subcommand.

- **LLM03 supply-chain audit**: model SHA-256 / sigstore signature checks at
  load time, ML dependency SBOM check, plugin manifest validator.
- **LLM04 behavioral canaries**: a battery of known prompts run at deploy
  time, alert on output drift (only realistic LLM04 hook for an inference
  library).
- Policy linter (validate `Policy` configs in CI).

## Phase 5 — Observability & ecosystem (v3.5)

- OpenTelemetry exporter for `AuditEvent`s.
- Prebuilt Grafana / Datadog dashboards.
- Policy-as-code repo template.
- Red-team CLI: replay an OWASP probe corpus against any policy, produce a
  coverage report.

## Honest coverage after roadmap

| OWASP  | After roadmap                                                |
| ------ | ------------------------------------------------------------ |
| LLM01  | Strong — regex + indirect scan + optional ML classifier      |
| LLM02  | Strong — bidirectional DLP                                   |
| LLM03  | Partial — CLI audit only, never runtime                      |
| LLM04  | Weak / advisory — canaries, can't fix at inference           |
| LLM05  | Strong — output toolkit                                      |
| LLM06  | Strong — tool authorization framework                        |
| LLM07  | Strong — canary + output scan                                |
| LLM08  | Strong — retriever middleware                                |
| LLM09  | Medium — grounding + citations, not a misinformation cure    |
| LLM10  | Strong — budgets + limits                                    |

Target: 8 strong, 1 medium, 1 honestly-partial — versus v2.x's
"1 strong, 9 checkbox."

## Stability commitment

- **Public API** (anything importable from `soweak.*` top level) follows
  semver. Breaking changes only on major versions.
- **Internal modules** (`soweak.core.*`, `soweak.detectors.*` internals) may
  change in minor versions; pin to a minor if you depend on them.
- **Pattern packs** are versioned data and may gain patterns in minor
  releases. Removals only on major versions.
