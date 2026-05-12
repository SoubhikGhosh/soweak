---
name: Feature request
about: A new defense, a new adapter, a new pattern, a new detector, a new pack.
title: ""
labels: enhancement
assignees: ""
---

## Problem

What real attack / class of bug / operational pain are you trying to
address? Be specific: "soweak doesn't catch X" is fine; please include a
representative example.

## Where does it belong?

soweak's architecture splits defenses by boundary. Which boundary does
this defense run at?

- [ ] `Boundary.INPUT`     — user prompts
- [ ] `Boundary.RETRIEVAL` — retrieved documents
- [ ] `Boundary.TOOL_CALL` — LLM-requested tool invocations
- [ ] `Boundary.OUTPUT`    — model responses
- [ ] `Boundary.STREAM`    — streaming chunks
- [ ] Build / deploy time  — `soweak audit` subcommand

If you're not sure, that's fine — we'll discuss it on the issue. But "the
attack happens at boundary X so the defense should run there too" is
usually the right framing.

## Proposed API or pattern

If applicable, sketch the public API:

```python
# what the user writes
```

## Alternatives considered

What else have you tried? Are there existing tools / patterns that
partially solve this?

## OWASP mapping

If this addresses an OWASP LLM Top 10 category, which one? See
`THREAT_MODEL.md` for current coverage status.

## Anything else

Links, papers, related discussion.
