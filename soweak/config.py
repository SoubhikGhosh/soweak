"""Declarative policy loader.

Load a :class:`~soweak.Policy` from a YAML or JSON file::

    from soweak.config import load_policy

    policy = load_policy("policy.yaml")

Schema (version 1)::

    version: 1
    rules:
      - name: prompt-injection
        boundary: input
        detectors:
          - type: prompt_injection
        enforcer:
          type: block
          min_severity: high

      - name: input-dlp
        boundary: input
        detectors:
          - type: input_dlp
        enforcer:
          type: redact
          min_severity: high
          placeholder: "[REDACTED]"

      - name: output-canary
        boundary: output
        detectors:
          - type: canary
            tokens: ["x7K2-PRODSEC-9F4E"]
        enforcer:
          type: block
          min_severity: critical

Each detector and enforcer ``type`` is resolved through a registry. Pass
``detector_registry=`` / ``enforcer_registry=`` to extend or override the
defaults with your own factories.

YAML support requires ``pip install soweak[yaml]``. JSON works with no
extras.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from soweak.core.detector import Detector
from soweak.core.enforcer import Enforcer
from soweak.core.policy import Policy, PolicyBuilder
from soweak.core.types import Boundary, Severity
from soweak.detectors import (
    CanaryDetector,
    PatternMatchDetector,
    input_dlp_detector,
    output_dlp_detector,
    output_html_detector,
    output_shell_detector,
    output_sql_detector,
    prompt_injection_detector,
    system_prompt_extraction_detector,
)
from soweak.detectors.patterns import Pattern, PatternPack
from soweak.core.types import OwaspCategory
from soweak.enforcers import (
    BlockEnforcer,
    LogOnlyEnforcer,
    RedactEnforcer,
    ThresholdEnforcer,
)
from soweak.grounding import CitationRequiredDetector, GroundingDetector
from soweak.rag import (
    IndirectInjectionDetector,
    ProvenanceDetector,
    RetrievalAnomalyDetector,
    TenantIsolationDetector,
)
from soweak.streaming import RepetitionDetector


#: Type alias for the detector / enforcer factory protocol.
DetectorFactory = Callable[..., Detector]
EnforcerFactory = Callable[..., Enforcer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _severity(value: str | Severity) -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity[value.upper()]
    except KeyError as e:
        raise ValueError(
            f"invalid severity {value!r}; expected one of "
            f"{[s.label for s in Severity]}"
        ) from e


def _pattern_pack(spec: Mapping[str, Any]) -> PatternPack:
    """Build a :class:`PatternPack` from a mapping (for inline custom packs)."""
    patterns = tuple(
        Pattern(
            regex=p["regex"],
            severity=_severity(p.get("severity", "medium")),
            description=p.get("description", ""),
            confidence=float(p.get("confidence", 0.85)),
            attack_type=p.get("attack_type", ""),
        )
        for p in spec.get("patterns", [])
    )
    category = OwaspCategory(spec.get("category", "LLM01"))
    return PatternPack(
        name=spec.get("name", "custom"),
        category=category,
        patterns=patterns,
        version=spec.get("version", "1.0"),
    )


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------


def _make_canary(*, tokens: list[str], **kwargs: Any) -> CanaryDetector:
    return CanaryDetector(tokens=tokens, **kwargs)


def _make_pattern_match(*, pack: Mapping[str, Any], **kwargs: Any) -> PatternMatchDetector:
    return PatternMatchDetector(_pattern_pack(pack), **kwargs)


def _make_indirect_injection(**kwargs: Any) -> IndirectInjectionDetector:
    return IndirectInjectionDetector(**kwargs)


def _make_tenant_isolation(**kwargs: Any) -> TenantIsolationDetector:
    return TenantIsolationDetector(**kwargs)


def _make_provenance(*, required_keys: list[str] | None = None, **kwargs: Any) -> ProvenanceDetector:
    if required_keys is not None:
        return ProvenanceDetector(required_keys=tuple(required_keys), **kwargs)
    return ProvenanceDetector(**kwargs)


def _make_retrieval_anomaly(**kwargs: Any) -> RetrievalAnomalyDetector:
    return RetrievalAnomalyDetector(**kwargs)


def _make_citation_required(*, severity: str | None = None, **kwargs: Any) -> CitationRequiredDetector:
    if severity is not None:
        return CitationRequiredDetector(severity=_severity(severity), **kwargs)
    return CitationRequiredDetector(**kwargs)


def _make_grounding(*, severity: str | None = None, **kwargs: Any) -> GroundingDetector:
    if severity is not None:
        return GroundingDetector(severity=_severity(severity), **kwargs)
    return GroundingDetector(**kwargs)


def _make_repetition(**kwargs: Any) -> RepetitionDetector:
    return RepetitionDetector(**kwargs)


#: Default mapping from string ``type`` to detector factory. Each factory
#: accepts only keyword arguments matched from the YAML/JSON spec.
DEFAULT_DETECTOR_REGISTRY: dict[str, DetectorFactory] = {
    # Input (LLM01 / LLM02 / LLM07)
    "prompt_injection": lambda: prompt_injection_detector(),
    "input_dlp": lambda: input_dlp_detector(),
    "system_prompt_extraction": lambda: system_prompt_extraction_detector(),
    # Output (LLM02 output / LLM05 / LLM07)
    "output_dlp": lambda: output_dlp_detector(),
    "output_html": lambda: output_html_detector(),
    "output_sql": lambda: output_sql_detector(),
    "output_shell": lambda: output_shell_detector(),
    "canary": _make_canary,
    # RAG (LLM08)
    "indirect_injection": _make_indirect_injection,
    "tenant_isolation": _make_tenant_isolation,
    "provenance": _make_provenance,
    "retrieval_anomaly": _make_retrieval_anomaly,
    # Grounding (LLM09)
    "citation_required": _make_citation_required,
    "grounding": _make_grounding,
    # Streaming (LLM10)
    "repetition": _make_repetition,
    # Generic
    "pattern_match": _make_pattern_match,
}


# ---------------------------------------------------------------------------
# Enforcer registry
# ---------------------------------------------------------------------------


def _make_block(*, min_severity: str | None = None, **kwargs: Any) -> BlockEnforcer:
    if min_severity is not None:
        return BlockEnforcer(min_severity=_severity(min_severity), **kwargs)
    return BlockEnforcer(**kwargs)


def _make_redact(
    *,
    min_severity: str | None = None,
    placeholder: str | None = None,
    **kwargs: Any,
) -> RedactEnforcer:
    if min_severity is not None:
        kwargs["min_severity"] = _severity(min_severity)
    if placeholder is not None:
        kwargs["placeholder"] = placeholder
    return RedactEnforcer(**kwargs)


def _make_log_only(**kwargs: Any) -> LogOnlyEnforcer:
    return LogOnlyEnforcer(**kwargs)


def _make_threshold(**kwargs: Any) -> ThresholdEnforcer:
    return ThresholdEnforcer(**kwargs)


DEFAULT_ENFORCER_REGISTRY: dict[str, EnforcerFactory] = {
    "block": _make_block,
    "redact": _make_redact,
    "log_only": _make_log_only,
    "threshold": _make_threshold,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


SUPPORTED_VERSIONS: tuple[int, ...] = (1,)


def build_policy(
    data: Mapping[str, Any],
    detector_registry: Mapping[str, DetectorFactory] | None = None,
    enforcer_registry: Mapping[str, EnforcerFactory] | None = None,
) -> Policy:
    """Build a :class:`Policy` from a plain-dict spec.

    The spec mirrors the YAML/JSON schema; use :func:`load_policy` to parse
    a file. Pass custom registries to add your own detector / enforcer
    types.
    """
    version = data.get("version", 1)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported policy version {version!r}; supported: {SUPPORTED_VERSIONS}"
        )

    det_reg: dict[str, DetectorFactory] = dict(DEFAULT_DETECTOR_REGISTRY)
    if detector_registry:
        det_reg.update(detector_registry)
    enf_reg: dict[str, EnforcerFactory] = dict(DEFAULT_ENFORCER_REGISTRY)
    if enforcer_registry:
        enf_reg.update(enforcer_registry)

    builder = PolicyBuilder()
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("policy.rules must be a list")

    for i, raw in enumerate(rules):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rules[{i}] must be a mapping, got {type(raw).__name__}")
        name = raw.get("name", f"rule-{i}")
        boundary_str = raw.get("boundary")
        if not boundary_str:
            raise ValueError(f"rules[{i}] missing 'boundary'")
        try:
            boundary = Boundary(boundary_str)
        except ValueError as e:
            raise ValueError(
                f"rules[{i}] invalid boundary {boundary_str!r}; expected one of "
                f"{[b.value for b in Boundary]}"
            ) from e

        detector_specs = raw.get("detectors", [])
        if not isinstance(detector_specs, list):
            raise ValueError(f"rules[{i}].detectors must be a list")
        detectors: list[Detector] = []
        for j, spec in enumerate(detector_specs):
            if not isinstance(spec, Mapping):
                raise ValueError(
                    f"rules[{i}].detectors[{j}] must be a mapping"
                )
            d_type = spec.get("type")
            if not d_type:
                raise ValueError(f"rules[{i}].detectors[{j}] missing 'type'")
            factory = det_reg.get(d_type)
            if factory is None:
                raise ValueError(
                    f"rules[{i}].detectors[{j}] unknown type {d_type!r}; "
                    f"registered: {sorted(det_reg)}"
                )
            kwargs = {k: v for k, v in spec.items() if k != "type"}
            try:
                detectors.append(factory(**kwargs))
            except TypeError as e:
                raise ValueError(
                    f"rules[{i}].detectors[{j}] ({d_type!r}): {e}"
                ) from e

        enf_spec = raw.get("enforcer")
        if not enf_spec or not isinstance(enf_spec, Mapping):
            raise ValueError(f"rules[{i}].enforcer must be a mapping")
        e_type = enf_spec.get("type")
        if not e_type:
            raise ValueError(f"rules[{i}].enforcer missing 'type'")
        e_factory = enf_reg.get(e_type)
        if e_factory is None:
            raise ValueError(
                f"rules[{i}].enforcer unknown type {e_type!r}; "
                f"registered: {sorted(enf_reg)}"
            )
        enf_kwargs = {k: v for k, v in enf_spec.items() if k != "type"}
        try:
            enforcer = e_factory(**enf_kwargs)
        except TypeError as e:
            raise ValueError(
                f"rules[{i}].enforcer ({e_type!r}): {e}"
            ) from e

        clause = {
            Boundary.INPUT: builder.on_input,
            Boundary.RETRIEVAL: builder.on_retrieval,
            Boundary.TOOL_CALL: builder.on_tool_call,
            Boundary.OUTPUT: builder.on_output,
            Boundary.STREAM: builder.on_stream,
        }[boundary](name)
        if detectors:
            clause = clause.detect(*detectors)
        clause.enforce(enforcer)

    return builder.build()


def load_policy(
    path: str | Path,
    format: str | None = None,
    detector_registry: Mapping[str, DetectorFactory] | None = None,
    enforcer_registry: Mapping[str, EnforcerFactory] | None = None,
) -> Policy:
    """Load a Policy from a YAML or JSON file.

    The file format is auto-detected from the suffix (``.yaml`` / ``.yml``
    → YAML, anything else → JSON). YAML support requires
    ``pip install soweak[yaml]``.
    """
    p = Path(path)
    if format is None:
        fmt = "yaml" if p.suffix.lower() in (".yaml", ".yml") else "json"
    else:
        fmt = format.lower()
    text = p.read_text(encoding="utf-8")
    if fmt == "yaml":
        try:
            import yaml  # type: ignore[import-not-found,import-untyped]
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "YAML policy support requires `pip install soweak[yaml]`."
            ) from e
        data = yaml.safe_load(text)
    elif fmt == "json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported policy format: {fmt!r}")
    if not isinstance(data, Mapping):
        raise ValueError(f"policy file root must be a mapping, got {type(data).__name__}")
    return build_policy(data, detector_registry, enforcer_registry)


__all__ = [
    "DEFAULT_DETECTOR_REGISTRY",
    "DEFAULT_ENFORCER_REGISTRY",
    "DetectorFactory",
    "EnforcerFactory",
    "SUPPORTED_VERSIONS",
    "build_policy",
    "load_policy",
]
