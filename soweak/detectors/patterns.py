"""Curated pattern packs for the built-in detectors.

A Pattern is a regex plus the severity, OWASP category, confidence and
human-readable description we want to attach when it fires. A PatternPack is a
named collection of patterns sharing one OWASP category. Pattern packs are
versioned data: minor releases may add patterns; removals only happen on
majors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from soweak.core.types import OwaspCategory, Severity


@dataclass(frozen=True)
class Pattern:
    """One regex pattern with associated metadata."""

    regex: str
    severity: Severity
    description: str
    confidence: float = 0.85
    attack_type: str = ""
    flags: int = re.IGNORECASE


@dataclass(frozen=True)
class PatternPack:
    """A named bundle of patterns for one OWASP category.

    ``version`` follows the ``MAJOR.MINOR`` convention. Minor bumps add or
    refine patterns without removing any; major bumps may remove patterns or
    change semantics. Callers that pin against a pack version should use
    :meth:`require_version`.
    """

    name: str
    category: OwaspCategory
    patterns: tuple[Pattern, ...] = field(default_factory=tuple)
    version: str = "1.0"

    def require_version(self, minimum: str) -> None:
        """Raise :class:`ValueError` if this pack's version is older than
        ``minimum`` (semver-style ``MAJOR.MINOR`` comparison).

        Example::

            PROMPT_INJECTION_PACK.require_version("1.0")  # ok in v3.x
        """
        actual = _parse_pack_version(self.version)
        wanted = _parse_pack_version(minimum)
        if actual < wanted:
            raise ValueError(
                f"pattern pack {self.name!r} is version {self.version}; "
                f"caller requires >= {minimum}"
            )


def _parse_pack_version(value: str) -> tuple[int, int]:
    try:
        parts = value.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError) as e:
        raise ValueError(f"invalid pack version {value!r}; expected MAJOR.MINOR") from e


# ---------------------------------------------------------------------------
# LLM01 — Prompt Injection (input boundary)
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS: tuple[Pattern, ...] = (
    # Instruction override
    Pattern(
        regex=r"ignore\s+(all\s+|any\s+|every\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?"
        r"(instructions?|rules?|guidelines?|prompts?|constraints?)",
        severity=Severity.CRITICAL,
        description="Instruction override attempt",
        attack_type="instruction_override",
    ),
    Pattern(
        regex=r"disregard\s+(?:all\s+|any\s+|the\s+|every\s+)?"
        r"(?:previous|prior|above|earlier|preceding)"
        r"(?:\s+(?:instructions?|rules?|text|prompts?|context|messages?))?",
        severity=Severity.CRITICAL,
        description="Instruction disregard attempt",
        attack_type="instruction_override",
    ),
    Pattern(
        regex=r"forget\s+(everything|all|what|your)\s+(you\s+)?(know|learned|were\s+told|instructions?)",
        severity=Severity.CRITICAL,
        description="Memory wipe attempt",
        attack_type="instruction_override",
    ),
    Pattern(
        regex=r"override\s+(your\s+)?(instructions?|rules?|guidelines?|programming|system)",
        severity=Severity.CRITICAL,
        description="System override attempt",
        attack_type="instruction_override",
    ),
    Pattern(
        regex=r"bypass\s+(your\s+|the\s+)?(safety|security|filters?|restrictions?|rules?|guardrails?)",
        severity=Severity.CRITICAL,
        description="Safety bypass attempt",
        attack_type="safety_bypass",
    ),
    # Developer / admin mode
    Pattern(
        regex=r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug|test|root|maintenance)\s+mode",
        severity=Severity.CRITICAL,
        description="Developer/admin mode activation",
        attack_type="mode_escalation",
    ),
    Pattern(
        regex=r"(enter|enable|activate)\s+(developer|admin|sudo|root|debug|god)\s+(mode|access|privileges?)",
        severity=Severity.CRITICAL,
        description="Privileged mode request",
        attack_type="mode_escalation",
    ),
    Pattern(
        regex=r"switch\s+to\s+(unrestricted|unlimited|unfiltered|uncensored)\s+mode",
        severity=Severity.CRITICAL,
        description="Unrestricted mode request",
        attack_type="mode_escalation",
    ),
    # Role manipulation / jailbreak personas
    Pattern(
        regex=r"act\s+as\s+(if\s+)?(you\s+)?(have\s+)?(no|without)\s+(restrictions?|limits?|rules?|guardrails?)",
        severity=Severity.CRITICAL,
        description="Restriction removal via role play",
        attack_type="role_manipulation",
    ),
    Pattern(
        regex=r"roleplay\s+as\s+\S+.*(unrestricted|evil|malicious|unfiltered|uncensored)",
        severity=Severity.CRITICAL,
        description="Malicious roleplay request",
        attack_type="role_manipulation",
    ),
    Pattern(
        regex=r"you\s+are\s+(now\s+)?(DAN|STAN|DUDE|KEVIN|JAILBREAK|AIM|DUDE)\b",
        severity=Severity.CRITICAL,
        description="Known jailbreak persona",
        attack_type="jailbreak_persona",
    ),
    Pattern(
        regex=r"\bdo\s+anything\s+now\b",
        severity=Severity.CRITICAL,
        description="DAN jailbreak phrase",
        attack_type="jailbreak_persona",
    ),
    Pattern(
        regex=r"no\s+(ethical|moral|safety)\s+(guidelines?|restrictions?|rules?|considerations?)",
        severity=Severity.CRITICAL,
        description="Ethics bypass",
        attack_type="safety_bypass",
    ),
    Pattern(
        regex=r"without\s+(any\s+)?(restrictions?|limitations?|filters?|safeguards?|censorship)",
        severity=Severity.HIGH,
        description="Safeguard removal",
        attack_type="safety_bypass",
    ),
    # Output forcing
    Pattern(
        regex=r"(say|respond|reply|output|print)\s+['\"][^'\"]{1,40}['\"]\s+and\s+nothing\s+else",
        severity=Severity.MEDIUM,
        description="Forced response",
        attack_type="output_forcing",
    ),
    Pattern(
        regex=r"your\s+(only\s+|entire\s+)?response\s+(should|must|will)\s+be",
        severity=Severity.MEDIUM,
        description="Response constraint",
        attack_type="output_forcing",
    ),
    # Persistent / structural injection
    Pattern(
        regex=r"from\s+now\s+on[,\s]+(you\s+)?(will|must|shall|should)\b",
        severity=Severity.HIGH,
        description="Persistent instruction injection",
        attack_type="context_injection",
    ),
    Pattern(
        regex=r"\[(system|admin|developer|root|instructions?)\]",
        severity=Severity.HIGH,
        description="Fake system tag",
        attack_type="structural_injection",
    ),
    Pattern(
        regex=r"<\s*/?\s*(system|admin|root|developer|instructions?)\s*>",
        severity=Severity.HIGH,
        description="Fake XML system tag",
        attack_type="structural_injection",
    ),
    # Indirect injection — hidden in untrusted markup
    Pattern(
        regex=r"<!--[^>]*?(ignore|override|bypass|system|jailbreak)[^>]*?-->",
        severity=Severity.HIGH,
        description="Hidden HTML comment injection",
        attack_type="indirect_injection",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    Pattern(
        regex=r"/\*[^*]*?(ignore|override|bypass|system|jailbreak)[^*]*?\*/",
        severity=Severity.HIGH,
        description="Hidden code-comment injection",
        attack_type="indirect_injection",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    # Encoding obfuscation
    Pattern(
        regex="[​-‏‪-‮⁠-⁯]{3,}",
        severity=Severity.HIGH,
        description="Invisible Unicode (zero-width/bidi) cluster",
        confidence=0.95,
        attack_type="encoding_obfuscation",
        flags=0,
    ),
    Pattern(
        regex=r"(?:\\u[0-9a-fA-F]{4}){5,}",
        severity=Severity.MEDIUM,
        description="Unicode escape sequence cluster",
        attack_type="encoding_obfuscation",
        flags=0,
    ),
    Pattern(
        regex=r"(?:\\x[0-9a-fA-F]{2}){5,}",
        severity=Severity.MEDIUM,
        description="Hex escape sequence cluster",
        attack_type="encoding_obfuscation",
        flags=0,
    ),
)

PROMPT_INJECTION_PACK = PatternPack(
    name="prompt_injection",
    category=OwaspCategory.LLM01_PROMPT_INJECTION,
    patterns=_PROMPT_INJECTION_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM02 — Sensitive Information Disclosure (input DLP)
# ---------------------------------------------------------------------------
# These patterns detect secrets / PII *being submitted* to the LLM. They're
# the input-side half of LLM02; the output-side scan ships in v3.1.

_INPUT_DLP_PATTERNS: tuple[Pattern, ...] = (
    # Cloud / vendor API keys (high-confidence well-known prefixes)
    Pattern(
        regex=r"\bAKIA[0-9A-Z]{16}\b",
        severity=Severity.CRITICAL,
        description="AWS access key ID",
        confidence=0.98,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\b(?:sk|rk)-[A-Za-z0-9]{20,}\b",
        severity=Severity.CRITICAL,
        description="OpenAI-style API key",
        confidence=0.95,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\bgh[ps]_[A-Za-z0-9]{36,}\b",
        severity=Severity.CRITICAL,
        description="GitHub personal access token",
        confidence=0.98,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
        severity=Severity.CRITICAL,
        description="GitHub fine-grained PAT",
        confidence=0.98,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        severity=Severity.CRITICAL,
        description="Slack token",
        confidence=0.95,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\bya29\.[A-Za-z0-9_-]{30,}\b",
        severity=Severity.CRITICAL,
        description="Google OAuth token",
        confidence=0.95,
        attack_type="secret_leak",
        flags=0,
    ),
    Pattern(
        regex=r"-----BEGIN\s+(RSA|EC|OPENSSH|PGP|DSA|PRIVATE)\s+PRIVATE\s+KEY-----",
        severity=Severity.CRITICAL,
        description="Private key block",
        confidence=0.99,
        attack_type="secret_leak",
    ),
    # Loose API key / password keywords (lower confidence)
    Pattern(
        regex=r"(?:api[_-]?key|secret|password|passwd|pwd|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        severity=Severity.HIGH,
        description="Hardcoded credential assignment",
        confidence=0.7,
        attack_type="secret_leak",
    ),
    # PII
    Pattern(
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        severity=Severity.HIGH,
        description="US Social Security Number (SSN)",
        confidence=0.85,
        attack_type="pii",
        flags=0,
    ),
    Pattern(
        regex=r"\b(?:\d[ -]*?){13,19}\b",
        severity=Severity.MEDIUM,
        description="Possible credit card number",
        confidence=0.55,
        attack_type="pii",
        flags=0,
    ),
    Pattern(
        regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        severity=Severity.LOW,
        description="Email address",
        confidence=0.9,
        attack_type="pii",
        flags=0,
    ),
    Pattern(
        regex=r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        severity=Severity.LOW,
        description="Possible US phone number",
        confidence=0.6,
        attack_type="pii",
        flags=0,
    ),
)

INPUT_DLP_PACK = PatternPack(
    name="input_dlp",
    category=OwaspCategory.LLM02_SENSITIVE_INFO,
    patterns=_INPUT_DLP_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM07 — System Prompt Leakage (input-side extraction attempts)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_EXTRACTION_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        regex=r"(reveal|show|display|print|output|tell\s+me|give\s+me)\s+(your\s+)?(full\s+|exact\s+|original\s+|initial\s+)?(system\s+)?(prompt|instructions?|rules?|directives?)",
        severity=Severity.HIGH,
        description="System prompt extraction attempt",
        attack_type="prompt_extraction",
    ),
    Pattern(
        regex=r"what\s+(are|were)\s+(your\s+)?(exact\s+|original\s+|initial\s+|first\s+)?(instructions?|prompt|rules?|directives?)",
        severity=Severity.HIGH,
        description="Instruction extraction question",
        attack_type="prompt_extraction",
    ),
    Pattern(
        regex=r"repeat\s+(the\s+|all\s+)?(text|instructions?|prompt|content)\s+(above|before|starting|preceding)",
        severity=Severity.HIGH,
        description="Prompt repeat request",
        attack_type="prompt_extraction",
    ),
    Pattern(
        regex=r"(copy|paste)\s+(and\s+paste\s+)?(your\s+)?(system|initial)\s+(prompt|message|instructions?)",
        severity=Severity.HIGH,
        description="System prompt copy request",
        attack_type="prompt_extraction",
    ),
    Pattern(
        regex=r"print\s+(everything|all|the\s+text)\s+(above|before|preceding)",
        severity=Severity.HIGH,
        description="Bulk preceding-text request",
        attack_type="prompt_extraction",
    ),
)

SYSTEM_PROMPT_EXTRACTION_PACK = PatternPack(
    name="system_prompt_extraction",
    category=OwaspCategory.LLM07_SYSTEM_PROMPT_LEAKAGE,
    patterns=_SYSTEM_PROMPT_EXTRACTION_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM02 — Sensitive Information Disclosure (output DLP)
# ---------------------------------------------------------------------------
# Patterns for scanning model OUTPUT. Shares the input pack's secret/PII
# patterns and adds output-specific markers that don't typically appear in
# user prompts (internal network references, connection strings, JWTs, etc.).

_OUTPUT_DLP_EXTRA_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        regex=r"\b10(?:\.\d{1,3}){3}\b",
        severity=Severity.MEDIUM,
        description="Private IP (10.0.0.0/8) in output",
        confidence=0.7,
        attack_type="internal_network_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b",
        severity=Severity.MEDIUM,
        description="Private IP (172.16.0.0/12) in output",
        confidence=0.7,
        attack_type="internal_network_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\b192\.168(?:\.\d{1,3}){2}\b",
        severity=Severity.MEDIUM,
        description="Private IP (192.168.0.0/16) in output",
        confidence=0.7,
        attack_type="internal_network_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\b[A-Za-z0-9-]+\.(?:internal|corp|local|lan|intranet)\b",
        severity=Severity.MEDIUM,
        description="Internal hostname in output",
        confidence=0.75,
        attack_type="internal_network_leak",
    ),
    Pattern(
        regex=r"\b(?:jdbc|mongodb|mysql|postgres|postgresql|redis)://[^\s'\"]+",
        severity=Severity.HIGH,
        description="Database connection string in output",
        confidence=0.9,
        attack_type="connection_string_leak",
    ),
    Pattern(
        regex=r"\beyJ[A-Za-z0-9_=-]+\.eyJ[A-Za-z0-9_=-]+\.[A-Za-z0-9_.+/=-]+",
        severity=Severity.HIGH,
        description="JWT token in output",
        confidence=0.9,
        attack_type="token_leak",
        flags=0,
    ),
    Pattern(
        regex=r"\b(?:arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d{12}:[A-Za-z0-9_/.+-]+)",
        severity=Severity.HIGH,
        description="AWS ARN with account ID in output",
        confidence=0.95,
        attack_type="cloud_resource_leak",
        flags=0,
    ),
)

OUTPUT_DLP_PACK = PatternPack(
    name="output_dlp",
    category=OwaspCategory.LLM02_SENSITIVE_INFO,
    patterns=_INPUT_DLP_PATTERNS + _OUTPUT_DLP_EXTRA_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM05 — Improper Output Handling: risky HTML in model output
# ---------------------------------------------------------------------------
# These detectors flag *the presence* of dangerous HTML. Pair them with
# BlockEnforcer to refuse delivery, or with the html_sanitizer_enforcer in
# soweak.output to scrub the output in-place.

_OUTPUT_HTML_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        regex=r"<\s*script\b[^>]*>",
        severity=Severity.CRITICAL,
        description="<script> tag in output",
        confidence=0.95,
        attack_type="xss_script",
    ),
    Pattern(
        regex=r"\son[a-z]+\s*=\s*['\"]",
        severity=Severity.HIGH,
        description="HTML event-handler attribute (on*=) in output",
        confidence=0.9,
        attack_type="xss_event_handler",
    ),
    Pattern(
        regex=r"\b(?:javascript|vbscript|data)\s*:",
        severity=Severity.HIGH,
        description="Dangerous URL scheme in output",
        confidence=0.85,
        attack_type="xss_url_scheme",
    ),
    Pattern(
        regex=r"<\s*(?:iframe|object|embed|applet)\b[^>]*>",
        severity=Severity.HIGH,
        description="Embedded-frame tag in output",
        confidence=0.9,
        attack_type="xss_embed",
    ),
    Pattern(
        regex=r"<\s*meta\s+[^>]*http-equiv\s*=\s*['\"]refresh",
        severity=Severity.MEDIUM,
        description="<meta refresh> redirect in output",
        confidence=0.85,
        attack_type="xss_meta_refresh",
    ),
    Pattern(
        regex=r"expression\s*\(",
        severity=Severity.HIGH,
        description="CSS expression() in output",
        confidence=0.85,
        attack_type="xss_css_expression",
    ),
)

OUTPUT_HTML_PACK = PatternPack(
    name="output_html",
    category=OwaspCategory.LLM05_OUTPUT_HANDLING,
    patterns=_OUTPUT_HTML_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM05 — Improper Output Handling: risky SQL in model output
# ---------------------------------------------------------------------------
# Flags model output that, if executed by a downstream system, would be
# destructive or indicative of SQL injection. This is a heuristic — pair with
# a real SQL parser (soweak.output.is_safe_sql, sqlparse, etc.) when you have
# the full SQL string in hand.

_OUTPUT_SQL_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        regex=r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX|VIEW|USER|ROLE)\b",
        severity=Severity.CRITICAL,
        description="SQL DROP statement in output",
        confidence=0.95,
        attack_type="sql_ddl",
    ),
    Pattern(
        regex=r"\bTRUNCATE\s+(?:TABLE\s+)?\w+",
        severity=Severity.CRITICAL,
        description="SQL TRUNCATE in output",
        confidence=0.95,
        attack_type="sql_ddl",
    ),
    Pattern(
        regex=r"\bGRANT\s+|\bREVOKE\s+",
        severity=Severity.HIGH,
        description="SQL GRANT/REVOKE in output",
        confidence=0.85,
        attack_type="sql_acl",
    ),
    Pattern(
        regex=r"\bUNION\s+(?:ALL\s+)?SELECT\b",
        severity=Severity.HIGH,
        description="UNION-based SQL injection pattern",
        confidence=0.85,
        attack_type="sqli_union",
    ),
    Pattern(
        regex=r"\bOR\s+['\"]?\s*1\s*['\"]?\s*=\s*['\"]?\s*1",
        severity=Severity.HIGH,
        description="SQL tautology (' OR 1=1)",
        confidence=0.9,
        attack_type="sqli_tautology",
    ),
    Pattern(
        regex=r"\bxp_cmdshell\b",
        severity=Severity.CRITICAL,
        description="SQL Server xp_cmdshell RCE",
        confidence=0.98,
        attack_type="sqli_rce",
    ),
    Pattern(
        regex=r"\binformation_schema\.",
        severity=Severity.MEDIUM,
        description="information_schema reconnaissance",
        confidence=0.7,
        attack_type="sqli_recon",
    ),
    Pattern(
        regex=r"--\s*(?:\r?\n|$)",
        severity=Severity.LOW,
        description="SQL comment terminator at end of line",
        confidence=0.6,
        attack_type="sqli_comment",
    ),
)

OUTPUT_SQL_PACK = PatternPack(
    name="output_sql",
    category=OwaspCategory.LLM05_OUTPUT_HANDLING,
    patterns=_OUTPUT_SQL_PATTERNS,
)


# ---------------------------------------------------------------------------
# LLM05 — Improper Output Handling: risky shell commands in model output
# ---------------------------------------------------------------------------

_OUTPUT_SHELL_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        regex=r"\brm\s+(?:-[rRf]+\s+)+(?:/|~|\$HOME|--no-preserve-root)",
        severity=Severity.CRITICAL,
        description="Destructive rm command",
        confidence=0.95,
        attack_type="shell_destructive",
    ),
    Pattern(
        regex=r"(?:curl|wget|fetch)\s+[^\s|;&]+\s*\|\s*(?:sudo\s+)?(?:bash|sh|zsh)\b",
        severity=Severity.CRITICAL,
        description="curl|bash remote-code execution",
        confidence=0.97,
        attack_type="shell_pipe_exec",
    ),
    Pattern(
        regex=r"[;&|]\s*(?:bash|sh|zsh|python(?:\d?)|perl|ruby)\s+-c\s+['\"]",
        severity=Severity.HIGH,
        description="Inline shell -c exec",
        confidence=0.9,
        attack_type="shell_inline_exec",
    ),
    Pattern(
        regex=r"\bdd\s+(?:[^|;]*\s+)?of\s*=\s*/dev/(?:sda|hda|nvme|disk)",
        severity=Severity.CRITICAL,
        description="dd to raw disk",
        confidence=0.95,
        attack_type="shell_disk_wipe",
    ),
    Pattern(
        regex=r"\bmkfs(?:\.\w+)?\s+/dev/",
        severity=Severity.CRITICAL,
        description="Filesystem format on raw device",
        confidence=0.97,
        attack_type="shell_disk_wipe",
    ),
    Pattern(
        regex=r"\bchmod\s+(?:0?777|a\+w)\b",
        severity=Severity.MEDIUM,
        description="Permissive chmod",
        confidence=0.8,
        attack_type="shell_perms",
    ),
    Pattern(
        regex=r"\bkill\s+-9?\s+1\b",
        severity=Severity.HIGH,
        description="kill init / PID 1",
        confidence=0.85,
        attack_type="shell_kill_init",
    ),
    Pattern(
        regex=r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:?\s*&?\s*\}\s*;\s*:",
        severity=Severity.CRITICAL,
        description="Fork-bomb pattern",
        confidence=0.95,
        attack_type="shell_fork_bomb",
    ),
    Pattern(
        regex=r"\bnc\s+(?:-l|--listen|-e)\b",
        severity=Severity.HIGH,
        description="netcat listener / -e reverse shell",
        confidence=0.85,
        attack_type="shell_reverse_shell",
    ),
)

OUTPUT_SHELL_PACK = PatternPack(
    name="output_shell",
    category=OwaspCategory.LLM05_OUTPUT_HANDLING,
    patterns=_OUTPUT_SHELL_PATTERNS,
)
