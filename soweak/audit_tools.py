"""Build- and deploy-time audit tooling.

This module addresses two OWASP categories that *cannot* be defended at
inference time:

* **LLM03 Supply Chain** — surface model and dependency integrity
  problems before deployment. Functions: :func:`hash_file`,
  :func:`verify_against_manifest`, :func:`list_python_packages`,
  :func:`check_packages_against_blocklist`.

* **LLM04 Data & Model Poisoning** — run a battery of canary prompts
  through a model callable at deploy time; flag output drift from
  expected behaviour. Types: :class:`Canary`, :class:`CanaryResult`;
  function :func:`run_canaries`.

A :func:`lint_policy` helper checks a :class:`~soweak.Policy` for common
misconfigurations (no rules at a critical boundary, rules with no
detectors, etc.).
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from soweak.core.policy import Policy
from soweak.core.types import Boundary


# ---------------------------------------------------------------------------
# LLM03 — model integrity & dependency audit
# ---------------------------------------------------------------------------


def hash_file(path: str | Path, algorithm: str = "sha256", block_size: int = 1 << 16) -> str:
    """Return ``algorithm`` digest of the file at ``path`` as a hex string.

    Streams the file in ``block_size`` chunks; works on multi-GB model
    weights without exhausting memory.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    digest = hashlib.new(algorithm)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    filename: str
    sha256: str


def load_manifest(path: str | Path) -> dict[str, str]:
    """Load a manifest of ``{filename: sha256}``. Accepts a JSON file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {p}")
    return {k: str(v) for k, v in data.items()}


def verify_against_manifest(
    artifact: str | Path,
    manifest: Mapping[str, str] | str | Path,
) -> tuple[bool, str]:
    """Verify ``artifact`` against a manifest.

    Returns ``(ok, message)``. ``manifest`` is either a dict
    ``{filename: sha256}`` or a path to a JSON file in that format.
    """
    if isinstance(manifest, (str, Path)):
        manifest = load_manifest(manifest)
    p = Path(artifact)
    expected = manifest.get(p.name)
    if not expected:
        return False, f"manifest has no entry for {p.name!r}"
    actual = hash_file(p)
    if actual.lower() != expected.lower():
        return False, f"hash mismatch: expected {expected!r}, got {actual!r}"
    return True, f"{p.name}: ok"


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    version: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


def list_python_packages() -> list[InstalledPackage]:
    """Enumerate installed Python distributions in the current interpreter."""
    seen: dict[str, str] = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        version = (dist.version or "").strip()
        # Distributions can be reported multiple times if installed in
        # several site-packages dirs — last one wins, which mirrors what
        # actually imports.
        seen[name.lower()] = version
    return [InstalledPackage(name=n, version=v) for n, v in sorted(seen.items())]


def check_packages_against_blocklist(
    blocklist: Iterable[str] | None = None,
    packages: Sequence[InstalledPackage] | None = None,
) -> list[InstalledPackage]:
    """Return installed packages whose name is in ``blocklist``.

    Comparison is case-insensitive. ``packages`` defaults to
    :func:`list_python_packages`. The framework ships an empty default
    blocklist — supply your own (e.g. from a vetted supply-chain feed).
    """
    blocked = {b.lower() for b in (blocklist or ())}
    pkgs = packages if packages is not None else list_python_packages()
    return [p for p in pkgs if p.name.lower() in blocked]


# ---------------------------------------------------------------------------
# LLM04 — behavioural canaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Canary:
    """One canary prompt and the expectations its output must meet.

    Either ``expect_contains`` (every substring must appear, case-insensitive),
    ``expect_not_contains`` (none may appear), or both. Empty tuples mean no
    check.
    """

    prompt: str
    expect_contains: tuple[str, ...] = ()
    expect_not_contains: tuple[str, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class CanaryResult:
    canary: Canary
    output: str
    passed: bool
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "canary": {
                "name": self.canary.name,
                "prompt": self.canary.prompt,
                "expect_contains": list(self.canary.expect_contains),
                "expect_not_contains": list(self.canary.expect_not_contains),
            },
            "output": self.output,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def run_canaries(
    canaries: Iterable[Canary],
    call_model: Callable[[str], str],
) -> list[CanaryResult]:
    """Run each canary's prompt through ``call_model`` and check expectations."""
    results: list[CanaryResult] = []
    for c in canaries:
        output = call_model(c.prompt)
        lowered = output.lower()
        failures: list[str] = []
        for sub in c.expect_contains:
            if sub.lower() not in lowered:
                failures.append(f"missing expected: {sub!r}")
        for sub in c.expect_not_contains:
            if sub.lower() in lowered:
                failures.append(f"contains forbidden: {sub!r}")
        results.append(
            CanaryResult(
                canary=c,
                output=output,
                passed=not failures,
                failures=tuple(failures),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Policy linter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintIssue:
    severity: str  # "error" | "warning"
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "message": self.message}


def lint_policy(policy: Policy) -> list[LintIssue]:
    """Static-check a :class:`Policy` for common misconfigurations.

    Returns a list of issues. Empty list means clean.
    """
    issues: list[LintIssue] = []
    if not policy.rules:
        issues.append(LintIssue("error", "policy has no rules"))
        return issues

    boundaries = {r.boundary for r in policy.rules}
    if Boundary.INPUT not in boundaries:
        issues.append(
            LintIssue("warning", "no rules at the INPUT boundary — user prompts unscanned")
        )
    if Boundary.OUTPUT not in boundaries:
        issues.append(
            LintIssue(
                "warning",
                "no rules at the OUTPUT boundary — leakage and harmful generation uncovered",
            )
        )

    for rule in policy.rules:
        if not rule.detectors:
            issues.append(
                LintIssue(
                    "warning",
                    f"rule {rule.name!r} at {rule.boundary.value} has no detectors — enforcer runs unconditionally",
                )
            )
        # Same boundary + duplicate detector type can mask intent — warn.
        types = [type(d).__name__ for d in rule.detectors]
        dups = {t for t in types if types.count(t) > 1}
        if dups:
            issues.append(
                LintIssue(
                    "warning",
                    f"rule {rule.name!r} contains duplicate detector classes: {sorted(dups)}",
                )
            )

    # Multiple rules at the same boundary with a BLOCK-style enforcer first
    # followed by something else can never reach the later rules. We can't
    # introspect enforcer behaviour generically, but flag rule ordering for
    # the boundaries where any rule's name suggests blocking.
    return issues


__all__ = [
    "Canary",
    "CanaryResult",
    "InstalledPackage",
    "LintIssue",
    "ManifestEntry",
    "check_packages_against_blocklist",
    "hash_file",
    "lint_policy",
    "list_python_packages",
    "load_manifest",
    "run_canaries",
    "verify_against_manifest",
]
