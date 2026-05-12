---
name: Bug report
about: A defense doesn't behave as documented, an error trace, or a crash.
title: ""
labels: bug
assignees: ""
---

<!--
SECURITY ISSUES (a defense bypass, a privilege escalation, a way to defeat
the audit log) — please do NOT use this template. See SECURITY.md for the
private disclosure channel.

For everything else, please fill out this report carefully. We close
incomplete reports automatically.
-->

## Summary

One sentence describing the bug.

## Soweak / environment versions

- soweak: <!-- output of: `python -c "import soweak; print(soweak.__version__)"` -->
- Python: <!-- output of: `python --version` -->
- OS: <!-- linux/macos/windows + version -->
- Extras installed: <!-- e.g. `soweak[langchain,ml]` -->

## Reproduction

A minimal Python snippet (or YAML policy + driver) that reproduces the
problem. Less than ~50 lines if possible.

```python
# code here
```

## Expected behaviour

What you expected to see.

## Actual behaviour

What you saw. Include the full traceback if there was one.

```
# traceback or output here
```

## Anything else

Context, related issues, workarounds you've tried.
