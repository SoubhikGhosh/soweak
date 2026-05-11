"""soweak — OWASP LLM Top 10 security middleware framework.

See :doc:`ROADMAP <ROADMAP>` for the full architectural model and which
OWASP categories each release covers.

Quick start::

    from soweak import (
        Pipeline,
        PolicyBuilder,
        BlockEnforcer,
        RedactEnforcer,
        Severity,
    )
    from soweak.detectors import (
        prompt_injection_detector,
        input_dlp_detector,
    )

    policy = (
        PolicyBuilder()
        .on_input("user-prompt")
            .detect(prompt_injection_detector())
            .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .on_input("dlp")
            .detect(input_dlp_detector())
            .enforce(RedactEnforcer())
        .build()
    )

    pipeline = Pipeline(policy)
    decision = pipeline.check_input(user_text)
    if decision.blocked:
        raise SecurityError(decision.reason)
"""

from soweak.core import (
    Action,
    AuditEvent,
    AuditLog,
    Boundary,
    Context,
    Decision,
    Detector,
    Enforcer,
    InMemoryAuditLog,
    JsonLinesAuditLog,
    OwaspCategory,
    Payload,
    Pipeline,
    Policy,
    PolicyBuilder,
    Rule,
    Severity,
    Signal,
)
from soweak.enforcers import (
    BlockEnforcer,
    LogOnlyEnforcer,
    RedactEnforcer,
    ThresholdEnforcer,
    TransformEnforcer,
)
from soweak.output import (
    URLAllowlist,
    html_sanitizer_enforcer,
    is_safe_sql,
    sanitize_html,
)

__version__ = "3.1.0"

__all__ = [
    # version
    "__version__",
    # core types
    "Action",
    "AuditEvent",
    "AuditLog",
    "Boundary",
    "Context",
    "Decision",
    "Detector",
    "Enforcer",
    "InMemoryAuditLog",
    "JsonLinesAuditLog",
    "OwaspCategory",
    "Payload",
    "Pipeline",
    "Policy",
    "PolicyBuilder",
    "Rule",
    "Severity",
    "Signal",
    # built-in enforcers
    "BlockEnforcer",
    "LogOnlyEnforcer",
    "RedactEnforcer",
    "ThresholdEnforcer",
    "TransformEnforcer",
    # output sanitisation (LLM05)
    "URLAllowlist",
    "html_sanitizer_enforcer",
    "is_safe_sql",
    "sanitize_html",
]
