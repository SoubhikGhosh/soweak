# Contributing to soweak

Thanks for your interest in contributing.

## Where new work lands

Read [`ROADMAP.md`](ROADMAP.md) first. Every phase ships a specific set of
capabilities at specific boundaries. Two principles:

1. **New defenses go at the right boundary.** If a check belongs on the
   output side, do not bolt it onto the input boundary just because input
   scanning is easier. The whole point of the v3 architecture is that we
   stopped doing that.
2. **Honest framing beats checkbox coverage.** If a defense is partial or
   only covers part of an OWASP category, document the gap. We'd rather
   ship "medium" coverage clearly labelled than "strong" coverage that
   isn't real.

## Setup

```bash
git clone https://github.com/SoubhikGhosh/soweak.git
cd soweak
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,all]"
pytest
```

## What good contributions look like

### New pattern (LLM01 / LLM02 / LLM07 input)

Add to the relevant pack in [`soweak/detectors/patterns.py`](soweak/detectors/patterns.py):

- Use the lowest severity that still represents a real threat.
- Use `re.IGNORECASE` unless case matters.
- Set `confidence` honestly — 0.95+ is reserved for well-known token formats
  (AWS keys, GitHub PATs). Heuristic phrases should sit around 0.6–0.85.
- Add at least one positive and one negative test in `tests/test_detectors.py`.

### New detector

Subclass `soweak.Detector` in a new module under `soweak/detectors/`. Wire
it up in `soweak/detectors/__init__.py` only if it's general enough to be a
default.

### New enforcer

Subclass `soweak.Enforcer` and add to `soweak/enforcers.py`. Re-export
from the top-level `soweak.__init__` only for the general cases.

### New adapter

Add `soweak/adapters/<name>.py` plus an example in `examples/<name>_example.py`.
The adapter MUST:

- Lazy-import the third-party library and raise an `ImportError` with the
  correct `pip install soweak[<extra>]` hint if missing.
- Raise `soweak.adapters.errors.SecurityError` on a BLOCK decision.
- Cover input *and* output boundaries when both are meaningful.

## Code style

- `black .` (line length 100)
- `isort .` (profile = black)
- `ruff check .`
- `mypy soweak`
- Public API has type hints; internal helpers may be inferred.

## Tests

- Use `pytest`. Tests live in `tests/`, mirror the module layout.
- Don't add network-dependent tests; mock SDKs at the adapter boundary.
- New patterns require coverage in `tests/test_detectors.py`.

## Pull requests

- One logical change per PR.
- Update `CHANGELOG.md` under the unreleased section.
- Update `ROADMAP.md` if your change shifts coverage in or out of a phase.
- A passing CI is required; do not skip pre-commit hooks (`--no-verify`).

## License

Contributions are licensed under Apache-2.0, the same license as the project.
