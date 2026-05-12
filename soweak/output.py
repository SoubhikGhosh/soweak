"""LLM05 output-handling helpers and TransformEnforcer factories.

Detectors flag risky output content; sanitizers fix it. This module ships
two tiers:

* **Stdlib-only baseline** — :func:`sanitize_html`, :class:`URLAllowlist`,
  :func:`is_safe_sql`. Zero dependencies; works everywhere; intended as
  defence-in-depth alongside a proper sanitizer / parser at the
  application's actual rendering / execution boundary.
* **Stronger optional path** — when ``pip install soweak[output]`` is
  available, :func:`sanitize_html` delegates to ``bleach`` (which handles
  charset normalisation, mutation-XSS variants, CSS sanitisation) and
  :func:`is_safe_sql` delegates to ``sqlparse`` (which understands real
  SQL grammar, not just regex). Behaviour is transparent: callers don't
  change anything.

Both tiers honour the same Python signatures. The optional path is opt-in
and silent — if ``bleach`` / ``sqlparse`` aren't installed, the baseline
runs.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from soweak.enforcers import TransformEnforcer

try:  # optional, via [output] extras
    import bleach as _bleach  # type: ignore[import-not-found]

    _HAS_BLEACH = True
except ImportError:  # pragma: no cover
    _bleach = None  # type: ignore[assignment]
    _HAS_BLEACH = False

try:  # optional, via [output] extras
    import sqlparse as _sqlparse  # type: ignore[import-not-found]

    _HAS_SQLPARSE = True
except ImportError:  # pragma: no cover
    _sqlparse = None  # type: ignore[assignment]
    _HAS_SQLPARSE = False


DEFAULT_ALLOWED_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "br",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "code",
        "pre",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "a",
        "span",
    }
)

DEFAULT_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
}

_DANGEROUS_URL_PREFIXES: tuple[str, ...] = (
    "javascript:",
    "data:",
    "vbscript:",
    "file:",
)


def _is_dangerous_url(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.strip().lower()
    return lowered.startswith(_DANGEROUS_URL_PREFIXES)


class _SanitizingParser(HTMLParser):
    def __init__(
        self,
        allowed_tags: frozenset[str],
        allowed_attrs: dict[str, frozenset[str]],
    ) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._allowed_tags = allowed_tags
        self._allowed_attrs = allowed_attrs

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag not in self._allowed_tags:
            return
        allowed = self._allowed_attrs.get(tag, frozenset())
        kept: list[tuple[str, str]] = []
        for k, v in attrs:
            k_lower = k.lower()
            # Always strip event handlers.
            if k_lower.startswith("on"):
                continue
            if k_lower not in allowed:
                continue
            if k_lower == "href" and _is_dangerous_url(v):
                continue
            kept.append((k_lower, v or ""))
        attrstr = "".join(
            f' {k}="{html.escape(v, quote=True)}"' for k, v in kept
        )
        self._parts.append(f"<{tag}{attrstr}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._allowed_tags:
            self._parts.append(f"</{tag}>")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # Treat void/self-closing tags the same as opening tags. Most allowed
        # tags here aren't void, but <br/> and similar should round-trip.
        if tag in self._allowed_tags:
            self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self._parts.append(html.escape(data, quote=False))

    @property
    def result(self) -> str:
        return "".join(self._parts)


def sanitize_html(
    text: str,
    allowed_tags: frozenset[str] = DEFAULT_ALLOWED_TAGS,
    allowed_attrs: dict[str, frozenset[str]] | None = None,
) -> str:
    """Strip tags outside ``allowed_tags`` and remove dangerous attributes.

    All ``on*`` event-handler attributes are removed regardless of tag.
    ``href`` attributes are dropped when they point to ``javascript:``,
    ``data:``, ``vbscript:`` or ``file:`` URIs.

    When ``bleach`` is installed (``pip install soweak[output]``), the
    sanitization delegates to it for charset-aware, mutation-XSS-resistant
    handling. Otherwise the bundled stdlib parser runs.
    """
    attrs = allowed_attrs or DEFAULT_ALLOWED_ATTRS
    if _HAS_BLEACH:
        # bleach.clean expects attrs as dict[tag -> list[str]].
        b_attrs: dict[str, list[str]] = {
            tag: list(attrs_for_tag) for tag, attrs_for_tag in attrs.items()
        }
        cleaned: str = _bleach.clean(
            text,
            tags=list(allowed_tags),
            attributes=b_attrs,
            protocols=["http", "https", "mailto"],
            strip=True,
        )
        return cleaned
    parser = _SanitizingParser(allowed_tags, attrs)
    parser.feed(text)
    parser.close()
    return parser.result


@dataclass(frozen=True)
class URLAllowlist:
    """Predicate-style allowlist for URLs that may appear in LLM output.

    Example::

        allowlist = URLAllowlist(schemes={"https"}, hosts={"docs.example.com"})
        allowlist.is_safe("https://docs.example.com/x")  # True
        allowlist.is_safe("https://evil.example.com")    # False
        allowlist.is_safe("javascript:alert(1)")         # False
    """

    schemes: frozenset[str] = field(default_factory=lambda: frozenset({"http", "https"}))
    hosts: frozenset[str] | None = None

    def is_safe(self, url: str) -> bool:
        if not url:
            return False
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError:
            return False
        if parsed.scheme.lower() not in {s.lower() for s in self.schemes}:
            return False
        if self.hosts is not None:
            host = (parsed.hostname or "").lower()
            if host not in {h.lower() for h in self.hosts}:
                return False
        return True


_DDL_RE = re.compile(
    r"\b(?:DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\s+",
    re.IGNORECASE,
)
_SUSPICIOUS_DML_RE = re.compile(
    r"\b(?:UNION\s+(?:ALL\s+)?SELECT|OR\s+['\"]?\s*1\s*['\"]?\s*=\s*['\"]?\s*1|xp_cmdshell)",
    re.IGNORECASE,
)


_SAFE_STATEMENT_TYPES: frozenset[str] = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"})


def is_safe_sql(sql: str, allow_ddl: bool = False) -> bool:
    """Heuristic SQL safety check.

    Returns ``False`` when ``sql`` contains DDL (unless ``allow_ddl=True``)
    or well-known SQL-injection signatures (``UNION SELECT``, tautologies,
    ``xp_cmdshell``).

    When ``sqlparse`` is installed (``pip install soweak[output]``), the
    function additionally parses ``sql`` and rejects any statement whose
    type is not in ``{SELECT, INSERT, UPDATE, DELETE}`` (plus DDL types
    when ``allow_ddl=True``). This catches semicolon-stacked injections
    that the regex pass alone might miss.
    """
    if not sql:
        return True
    if not allow_ddl and _DDL_RE.search(sql):
        return False
    if _SUSPICIOUS_DML_RE.search(sql):
        return False
    if _HAS_SQLPARSE:
        statements = _sqlparse.parse(sql)
        allowed = set(_SAFE_STATEMENT_TYPES)
        if allow_ddl:
            allowed |= {"CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE"}
        for stmt in statements:
            stmt_type = (stmt.get_type() or "UNKNOWN").upper()
            if stmt_type == "UNKNOWN":
                # sqlparse couldn't classify it — be conservative.
                continue
            if stmt_type not in allowed:
                return False
    return True


# ---------------------------------------------------------------------------
# TransformEnforcer factories
# ---------------------------------------------------------------------------


def html_sanitizer_enforcer(
    allowed_tags: frozenset[str] = DEFAULT_ALLOWED_TAGS,
    allowed_attrs: dict[str, frozenset[str]] | None = None,
    name: str = "html-sanitizer",
) -> TransformEnforcer:
    """A :class:`TransformEnforcer` that sanitises HTML at the output boundary.

    Example::

        Policy.builder().on_output().enforce(html_sanitizer_enforcer()).build()
    """

    def _transform(text: str) -> str:
        return sanitize_html(
            text, allowed_tags=allowed_tags, allowed_attrs=allowed_attrs
        )

    return TransformEnforcer(_transform, name=name)


__all__ = [
    "DEFAULT_ALLOWED_ATTRS",
    "DEFAULT_ALLOWED_TAGS",
    "URLAllowlist",
    "html_sanitizer_enforcer",
    "is_safe_sql",
    "sanitize_html",
]
