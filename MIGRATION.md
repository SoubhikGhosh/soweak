# Migration guide

Per-version notes for everyone upgrading inside the v3.x line.

## To v3.11

### Storage backends now close cleanly (recommended action)

`SqliteCounterStore` and `SqliteWindowStore` hold a single open
connection for the store's lifetime. Always close them, either via
`store.close()` or context manager:

```python
# Old (worked, but leaked descriptors per operation):
budget = TokenBudget(limit=10_000, store=SqliteCounterStore("/var/lib/x.db"))

# v3.11 (recommended):
with SqliteCounterStore("/var/lib/x.db") as store:
    budget = TokenBudget(limit=10_000, store=store)
    # ... use budget ...
# store auto-closes here.

# Or explicitly:
store = SqliteCounterStore("/var/lib/x.db")
try:
    budget = TokenBudget(limit=10_000, store=store)
    # ...
finally:
    store.close()
```

Existing code that never called `close()` will still work, but a
`__del__`-time cleanup will run when the store is garbage-collected.
Long-running processes that constructed many stores will see file
descriptor counts drop after upgrading.

### `JsonLinesAuditLog` now keeps the file handle open

Previously, each `record()` reopened the file. v3.11 keeps a single
handle and `flush()`es per record. Always close it on shutdown:

```python
with JsonLinesAuditLog("/var/log/soweak.jsonl") as audit:
    pipeline = Pipeline(policy, audit=audit)
    # ...
```

### `Severity` is now an `IntEnum`

Comparisons and sorting work without `.value`:

```python
# Both v3.x and v3.11:
if signal.severity >= Severity.HIGH:
    ...

# v3.11 also supports this — used to need key=lambda s: s.severity.value
sorted_signals = sorted(signals, key=lambda s: s.severity)
```

`Severity.HIGH.label`, `Severity.HIGH.weight`, and `Severity.HIGH.value`
are unchanged. Code that compared `Severity` to plain ints
(`severity == 3`) now works too — previously it would have raised.

### `BudgetEnforcer` accepts any `Budget`

The parameter type is now `Budget` (a `Protocol`) instead of
`TokenBudget | CostBudget`. User-defined budgets that implement
`consumed`, `remaining`, `reset`, and `name` work directly with
`BudgetEnforcer`.

### New optional extras

* **`soweak[output]`** — `bleach` + `sqlparse`. `sanitize_html` and
  `is_safe_sql` transparently delegate to the stronger backends when
  this extra is installed.
* **`soweak[anthropic]`** — Anthropic SDK adapter
  (`soweak.adapters.anthropic.SecureAnthropic`).

### `Pipeline.arun` now uses `AuditLog.arecord`

Async pipelines no longer block the event loop on slow sinks. The
default `arecord` runs `record` in the loop's default executor, so all
existing `AuditLog` subclasses work unchanged. Override `arecord` to add
native-async writes.

### `@guarded_tool` accepts a `WindowStore`

Per-(tool, user) rate limits can now share state across replicas:

```python
from soweak import SqliteWindowStore
from soweak.agent import guarded_tool

rate_store = SqliteWindowStore("/var/lib/soweak/tool-rl.db")

@guarded_tool(scopes=["email:send"], rate_limit_per_minute=5, rate_limit_store=rate_store)
def send_email(...): ...
```

### Public grounding helpers

`soweak.grounding._tokenize` / `_split_sentences` / `_gather_retrieval`
moved to `soweak.grounding.tokenize` / `split_sentences` /
`gather_retrieval`. The private names remain as aliases for back-compat
in the v3.x line and will be removed in v4.0.

### `Decision` factory methods

`Decision.allow(...)` was already a classmethod. v3.11 adds the
parallel `Decision.warn(...)`, `Decision.block(...)`,
`Decision.redact(...)`, `Decision.transform(...)` so custom enforcers can
construct decisions without naming the `Action` enum directly.

## To v3.10

Renamed `transformers_classifier(injection_label_index=...)` →
`positive_label_index=...`. The factory now auto-applies known-model
defaults from `KNOWN_INJECTION_MODELS`. If you pinned to a non-default
model and passed `injection_label_index`, rename the kwarg.

`MLClassifierDetector` continues to take any `Callable[[str], float]` —
no change.

Added `soweak.embeddings` with `EmbeddingGroundingDetector` and the
`sentence_transformer_embedder` factory. Requires
`pip install soweak[embeddings]`.

## To v3.9

YAML / JSON policies via `soweak.config.load_policy`. Existing Python
`PolicyBuilder` policies keep working.

```python
from soweak import Pipeline, load_policy

pipeline = Pipeline(load_policy("policy.yaml"))
```

## To v3.7

`TokenBudget` and `CostBudget` accept an optional `store=` argument.
Existing code that doesn't pass `store=` still works (defaults to
in-process). Use `SqliteCounterStore` for restart-survival or implement
`CounterStore` for multi-host backends.

## To v3.6

Async surface added. Sync API is unchanged. To migrate:

```python
# Sync (still works):
decision = pipeline.check_input(text, ctx)

# Async (new):
decision = await pipeline.acheck_input(text, ctx)
```

Existing sync detectors and enforcers work in async pipelines without
modification — the ABC's default `ainspect` / `adecide` delegate to the
sync impl.

For streaming output:

```python
from soweak import StreamingPipeline

stream = StreamingPipeline(pipeline, scan_every_chars=200)
async for chunk in stream.guard(llm_stream, ctx):
    yield chunk
```

## v3.0 — v3.5

These are the initial framework releases. See `CHANGELOG.md` for what
each version added; there are no in-line API breaks to migrate across.
