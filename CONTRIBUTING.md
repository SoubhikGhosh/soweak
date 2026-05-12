# Contributing to soweak

Thanks for considering a contribution. Read this whole document before
opening a PR — soweak has architectural rules that aren't obvious from
the file tree.

## Architectural rules

These aren't style preferences. They're the reason soweak's coverage of
the OWASP LLM Top 10 is honest instead of theatre.

### 1. New defenses go at the right boundary

Every defense lives at exactly one of:

* `Boundary.INPUT`     — user prompts
* `Boundary.RETRIEVAL` — retrieved documents
* `Boundary.TOOL_CALL` — LLM-requested tool invocations
* `Boundary.OUTPUT`    — model responses
* `Boundary.STREAM`    — streaming chunks

If you have a defense for an output-side problem (e.g. detecting
toxic generation), it goes at `Boundary.OUTPUT`. Do not bolt it onto
`Boundary.INPUT` because that's easier. A PR that scans user input for
output-side problems will be rejected.

### 2. Honest framing beats checkbox coverage

If your defense is partial, document the gap in the docstring **and** in
[`THREAT_MODEL.md`](THREAT_MODEL.md). "Partial" labels are not weakness;
overselling a heuristic as a guarantee is.

### 3. Optional dependencies are optional

If your code needs `transformers` / `bleach` / `sentence-transformers`
/etc, lazy-import inside a factory and raise a clear `ImportError` with
the matching `pip install soweak[X]` hint. The core library is
zero-dependency on import.

### 4. mypy must stay clean

CI runs `mypy soweak` in strict-ish mode. New code must type-check.
`from __future__ import annotations` is standard at the top of every
module.

### 5. No `ResourceWarning`

CI runs `pytest -W error::ResourceWarning`. If your code opens a file or
network handle, it must close it cleanly — even on exception paths.

## Module map

```
soweak/
├── __init__.py            # public API (top-level re-exports)
├── core/                  # Boundary, Detector, Enforcer, Policy, Pipeline, AuditLog
├── detectors/             # PatternMatchDetector, CanaryDetector, pattern packs
├── enforcers.py           # Block, Redact, LogOnly, Threshold, Transform
├── storage.py             # CounterStore / WindowStore (in-memory + SQLite)
├── budget.py              # TokenBudget, CostBudget, RateLimiter + their enforcers
├── agent.py               # @guarded_tool, authorize, scopes/approval/rate
├── streaming.py           # RepetitionDetector
├── rag.py                 # IndirectInjection, TenantIsolation, Provenance, RetrievalAnomaly
├── grounding.py           # CitationRequired, GroundingDetector, public tokenizer helpers
├── output.py              # sanitize_html, URLAllowlist, is_safe_sql (+ optional bleach/sqlparse)
├── ml.py                  # MLClassifierDetector + HF / toxicity / llm-judge factories
├── embeddings.py          # EmbeddingGroundingDetector + sentence-transformers factory
├── observability.py       # OpenTelemetryAuditLog
├── audit_tools.py         # build-time: hash_file, canaries, lint_policy
├── redteam.py             # OWASP probe corpus + run_probes + coverage_report
├── config.py              # YAML/JSON policy loader + detector/enforcer registries
├── cli.py                 # soweak {scan, list, audit, redteam, version}
└── adapters/              # SecureOpenAI, SecureAnthropic, SecureGemini, LangChain guard
```

Adding a new module means deciding which directory it belongs in:

* Boundary-specific detectors → `soweak/detectors/`, `soweak/rag.py`,
  `soweak/grounding.py`, etc.
* Boundary-agnostic infrastructure → `soweak/core/`.
* Optional-extra integrations → top-level (e.g., `soweak/ml.py`).
* Adapters around third-party SDKs → `soweak/adapters/`.

## Setup

```bash
git clone https://github.com/SoubhikGhosh/soweak.git
cd soweak
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,all]"

# Optional extras for tests that need them:
pip install -e ".[ml,embeddings,output]"

# Run the suite the way CI does:
pytest -W error::ResourceWarning
mypy soweak
ruff check .
black --check .
isort --check .
```

## What good contributions look like

### A new pattern

Add the `Pattern` entry to the matching pack in
[`soweak/detectors/patterns.py`](soweak/detectors/patterns.py). Then:

1. Add at least one positive and one negative test in
   [`tests/test_detectors.py`](tests/test_detectors.py).
2. Bump the pack's `version` (`"1.0"` → `"1.1"`).
3. Note the addition in `CHANGELOG.md` under the unreleased section.

Confidence values:

* `0.95+`  — well-known token formats (AWS keys, GitHub PATs, JWT shape).
* `0.85`   — case-insensitive English phrases with low FP risk.
* `0.6-0.8` — heuristic phrases that may produce some FPs; document the
  tradeoff in the description.

Severity values:

* `CRITICAL` — defaults to BLOCK; reserved for unambiguous attacks.
* `HIGH`     — defaults to BLOCK in most policies; high confidence
  threats.
* `MEDIUM`   — usually WARN or REDACT; signals worth attention.
* `LOW` / `INFO` — log-only telemetry.

### A new detector

Subclass `soweak.Detector` in a new file or in the most appropriate
existing module. Required:

```python
class MyDetector(Detector):
    @property
    def name(self) -> str: ...
    @property
    def category(self) -> OwaspCategory: ...
    @property
    def boundaries(self) -> tuple[Boundary, ...]: ...  # default (Boundary.INPUT,)
    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]: ...
```

Override `ainspect` only if your detector performs real I/O. Add the
detector's `type` to `DEFAULT_DETECTOR_REGISTRY` in `soweak/config.py`
so it's usable from YAML/JSON policies.

### A new enforcer

Subclass `soweak.Enforcer` in `soweak/enforcers.py` (general-purpose) or
in a topical module (e.g. `soweak/budget.py` for budget-related). Wire
into `DEFAULT_ENFORCER_REGISTRY` if it makes sense as a YAML primitive.

### A new adapter

Add `soweak/adapters/<name>.py`. Requirements:

* Lazy-import the third-party SDK and raise a clear `ImportError` with
  the matching `pip install soweak[<name>]` hint if missing.
* Raise `soweak.adapters.errors.SecurityError` on a BLOCK decision.
* Cover input *and* output boundaries when both are meaningful.
* Add a mock-based test suite in `tests/test_adapters.py` — no live API
  calls in CI.
* Add `soweak/adapters/<name>.py` exports to `soweak/adapters/__init__.py`
  is *not* required (lazy import keeps `[<name>]` opt-in).
* Add a runnable example in `examples/<name>_example.py`.
* Add the `[<name>]` extra to `pyproject.toml`.

### A new storage backend

Subclass `CounterStore` or `WindowStore` (or both). Document concurrency
guarantees in the docstring. Add a fixture-driven test suite that runs
the same contract tests against your backend.

### Pattern-pack version bumps

* `MAJOR` (`1.x` → `2.x`): patterns removed, semantics changed, default
  threshold raised in a way that breaks calling code.
* `MINOR` (`1.0` → `1.1`): patterns added, descriptions clarified.

Callers can pin with `pack.require_version("1.1")`.

## Code style

* `black .` — line length 100.
* `isort --profile=black` — line length 100.
* `ruff check .` — `pyproject.toml` lists the rules.
* `mypy soweak` — must pass.
* Public APIs have type hints; internal helpers may be inferred.
* No comments that restate what the code does. Comments are for *why*.
* `from __future__ import annotations` at the top of every module.

## Tests

* `pytest -W error::ResourceWarning` is the contract — no leaks.
* Tests live in `tests/`, mirroring the source layout.
* No network-dependent tests; mock SDKs at the adapter boundary.
* New regex patterns require both a positive and a negative case.
* New detectors require at least one pipeline-integration test.
* Concurrency claims must be verified with `ThreadPoolExecutor` in
  `tests/test_storage.py` or equivalent.

## Documentation

* Update `CHANGELOG.md` under the latest version's unreleased section.
* Update `ROADMAP.md` if your change moves an OWASP category between
  Strong / Partial / Build-time status.
* Update `THREAT_MODEL.md` if your defense changes the in-scope /
  out-of-scope boundary.
* Update `MIGRATION.md` if you introduce a backward-incompatible change.
* Update the relevant README section for any new public API.

## Pull requests

* One logical change per PR. Squash before merge.
* CI must be green. Do not use `--no-verify` on pre-commit hooks.
* Reviews focus on architectural fit (right boundary?), honest framing
  (claims match capability?), and test coverage. Code style is enforced
  by the formatter — please run it locally before submitting.
* PRs touching `core/`, `storage.py`, the `Detector` / `Enforcer` ABCs,
  or the public API surface need explicit reviewer approval. Anything
  else can ship faster.

## License

Contributions are licensed under Apache-2.0, the same as the project.
By submitting a PR you certify that you have the right to license your
contribution under those terms.

## Security disclosures

See [`SECURITY.md`](SECURITY.md). **Do not file security bugs as public
issues.**
