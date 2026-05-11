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
from soweak.agent import (
    ApprovalRequired,
    ToolCall,
    ToolCallEvent,
    authorize,
    current_context,
    guarded_tool,
)
from soweak.budget import (
    BudgetEnforcer,
    BudgetExceededError,
    CostBudget,
    ModelPricing,
    RateLimitEnforcer,
    RateLimiter,
    TokenBudget,
)
from soweak.streaming import RepetitionDetector
from soweak.rag import (
    IndirectInjectionDetector,
    ProvenanceDetector,
    RetrievalAnomalyDetector,
    TenantIsolationDetector,
)
from soweak.grounding import (
    CitationRequiredDetector,
    GroundingDetector,
)

__version__ = "3.4.0"

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
    # tool authorization (LLM06)
    "ApprovalRequired",
    "ToolCall",
    "ToolCallEvent",
    "authorize",
    "current_context",
    "guarded_tool",
    # budgets & rate limits (LLM10)
    "BudgetEnforcer",
    "BudgetExceededError",
    "CostBudget",
    "ModelPricing",
    "RateLimitEnforcer",
    "RateLimiter",
    "TokenBudget",
    # streaming (LLM10)
    "RepetitionDetector",
    # RAG (LLM08)
    "IndirectInjectionDetector",
    "ProvenanceDetector",
    "RetrievalAnomalyDetector",
    "TenantIsolationDetector",
    # grounding (LLM09)
    "CitationRequiredDetector",
    "GroundingDetector",
]
