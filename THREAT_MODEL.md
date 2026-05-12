# soweak threat model

This document tells you what soweak is built to defend against, what it
explicitly is not, and which OWASP LLM Top 10 categories ship with
"strong", "partial", or "build-time only" coverage. Read this before
deploying soweak as a security control.

## Operating assumptions

soweak runs **as middleware inside your application process**. It assumes:

1. The application code (the policy definition, the call sites of
   `pipeline.check_input` etc.) is trusted. If the attacker can modify
   your Python, soweak provides no value.
2. The LLM provider is honest-but-curious. We don't try to defend
   against a malicious provider returning forged tool calls in a
   structured-output stream.
3. The host environment is not under active root-level compromise.
4. Optional dependencies (bleach, sqlparse, sentence-transformers,
   transformers) are honest implementations of their advertised APIs.

If any of these is false, soweak's guarantees don't hold.

## Trust boundaries

```text
                                    ┌──────────────────────────────┐
                                    │     UNTRUSTED                │
   ┌─────────┐                      │                              │
   │  user   │ ──── prompt ───────▶ │  on_input boundary           │
   └─────────┘                      │   (pattern, ML, DLP)         │
                                    │                              │
                                    │  on_retrieval boundary       │
   ┌────────────┐                   │   (tenant, provenance,       │
   │ corpus /   │ ──── docs ──────▶ │    indirect injection)       │
   │ vector DB  │                   │                              │
   └────────────┘                   │                              │
                                    │  on_tool_call boundary       │
                                    │   (scopes, approval, rate)   │
                                    │                              │
                                    │  on_output boundary          │
   ┌────────────┐                   │   (DLP, sanitize, canary,    │
   │  the LLM   │ ──── tokens ────▶ │    toxicity, grounding)      │
   └────────────┘                   │                              │
                                    └──────────────────────────────┘
                                            │
                                            ▼
                                       TRUSTED app code
```

Anything entering the box from the left is **untrusted by default**.
Anything that has passed through a boundary's enforcer is **trusted only
for the decision the enforcer made**: a BLOCK is reliable, an ALLOW means
"no rule in this policy fired", not "verified safe".

## In scope — soweak addresses

| OWASP                                    | Defense                                                                     | Strength |
| ---------------------------------------- | --------------------------------------------------------------------------- | -------- |
| LLM01 Prompt Injection (direct)          | Pattern packs at `on_input`; ML classifier (opt-in via `[ml]`)              | Strong   |
| LLM01 Prompt Injection (indirect / 2nd-order) | `IndirectInjectionDetector` at `on_retrieval`                          | Strong   |
| LLM02 Sensitive Info (input PII / keys)  | `input_dlp_detector` + `RedactEnforcer`                                     | Strong   |
| LLM02 Sensitive Info (output leakage)    | `output_dlp_detector` at `on_output`                                        | Strong   |
| LLM05 Improper Output Handling           | `output_html` / `output_sql` / `output_shell` detectors; `sanitize_html`; `is_safe_sql` (bleach + sqlparse with `[output]`) | Strong   |
| LLM06 Excessive Agency                   | `@guarded_tool` (scopes, rate limit, human approval, audit)                 | Strong   |
| LLM07 System Prompt Leakage              | Extraction-pattern pack at `on_input`; `CanaryDetector` at `on_output`      | Strong   |
| LLM08 Vector & Embedding                 | `TenantIsolationDetector`, `ProvenanceDetector`, `RetrievalAnomalyDetector` | Strong   |
| LLM10 Unbounded Consumption              | `TokenBudget`, `CostBudget`, `RateLimiter`, `RepetitionDetector`            | Strong   |

| OWASP                              | Defense                                              | Strength       |
| ---------------------------------- | ---------------------------------------------------- | -------------- |
| LLM03 Supply Chain                 | `soweak audit model` / `audit deps`                  | Build-time only |
| LLM04 Data & Model Poisoning       | `soweak audit canaries` deploy-time battery          | Advisory       |
| LLM09 Misinformation               | `CitationRequiredDetector`, `GroundingDetector`, `EmbeddingGroundingDetector` | Partial (heuristic) |

## Out of scope — soweak does NOT defend against

### Categorically out of scope

* **Hallucinations that share vocabulary with the source.** The grounding
  heuristics check lexical or embedding overlap; a plausible paraphrase
  that mirrors the source's wording will pass. Use a fact-checking
  pipeline if you need real-claim verification.
* **Novel jailbreaks not yet in the pattern packs or the ML classifier's
  training data.** Pattern packs grow per release; we make no claim of
  exhaustiveness.
* **Steganographic / multi-modal attacks** outside the text channel
  (audio-prompt injection, image OCR injection). v3.x is text-only.
* **Side-channel attacks on the LLM provider.** Timing, token counts,
  cost variance, retry behaviour. Not in scope.
* **Membership inference / model extraction.** Out of scope; these are
  training-time properties.
* **Adversarial examples in the embedding space.** Crafted text whose
  embedding lies inside the retrieval cluster despite carrying a
  malicious payload. Out of scope.
* **Compromise of the underlying model.** A backdoored model that
  responds correctly to canary prompts and maliciously to a hidden trigger
  is undetectable from this layer.
* **Compromise of the host or of your application code.** soweak runs
  in-process; an attacker with code execution has already won.
* **Cross-tenant data leakage outside the retrieval boundary.** Soweak
  enforces tenant isolation only on documents flowing through
  `on_retrieval`. Tenant mixing inside your own database queries is your
  problem.

### Specific known limitations

* **In-process stores are in-process.** `InMemoryCounterStore` and
  `InMemoryWindowStore` reset on restart and do not share state across
  replicas. Use `SqliteCounterStore` / `SqliteWindowStore` for a single
  host, or implement the `CounterStore` / `WindowStore` ABCs against
  Redis/Postgres for multi-host.
* **`sanitize_html` baseline (without `[output]`) is intentionally
  minimal.** It strips event-handler attributes and dangerous URL schemes
  but doesn't normalise character sets. For production HTML rendering,
  install `[output]` (which delegates to `bleach`).
* **`is_safe_sql` baseline** is regex-only. For any execution path that
  takes generated SQL, install `[output]` (which adds `sqlparse`).
* **`StreamingPipeline.guard` scans at `scan_every_chars` boundaries.**
  An injection payload entirely contained in a single chunk smaller than
  `scan_every_chars` may briefly be yielded before the next scan; the
  scan that catches it will raise `SecurityError` on the *next* yield.
* **`@guarded_tool(approval="human")` blocks the calling thread.** The
  default handler runs synchronously. For non-blocking approval flows,
  supply your own `approval_handler` that queues the request and returns
  False until approved out-of-band.
* **Pattern packs are English-focused.** LLM01 patterns target English
  jailbreak phrases; the i18n grounding tokenizer handles non-Latin
  scripts but the injection detector will miss equivalent payloads in,
  say, Mandarin. Localised pattern packs are welcome as PRs.
* **The OWASP probe corpus in `soweak.redteam` is a smoke test, not an
  evaluation suite.** It checks that the default policy catches a small
  set of canonical probes; passing it does *not* certify the policy.

## Defense in depth

soweak is **one** layer of a complete LLM-security stack. It does not
replace:

* Provider-side moderation / safety endpoints.
* Application-level authz on tools and resources (soweak's scope
  framework is a *complement* to your real authz, not a replacement).
* HTTP-level rate limiting at the edge.
* Output-channel sanitization at the application's rendering / execution
  point (we recommend our `[output]` extras alongside, not instead of,
  framework-native sanitizers).
* Logging, monitoring, and incident response.

Treat any single soweak decision as advisory until you've layered other
controls behind it.

## Threat-model changes

Material changes to this document are listed in `CHANGELOG.md` under the
release that introduced them. The "Out of scope" list in particular is
expected to shrink over time as new defenses ship; see `ROADMAP.md` for
what's planned.
