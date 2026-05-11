"""LLM05 output-handling helpers and TransformEnforcer factories.

Detectors flag risky output content; sanitizers fix it. This module ships
the latter — small, stdlib-only helpers that can be used standalone or
plugged into a :class:`~soweak.TransformEnforcer` at the output boundary.

For HTML, the bundled :func:`sanitize_html` is intentionally minimal: it
removes tags outside an allowlist and strips event-handler attributes and
dangerous URL schemes. For richer sanitisation (e.g. preserving inline
styles, CSS sanitisation) use bleach + an external transform.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser

from soweak.enforcers import TransformEnforcer


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
    """
    parser = _SanitizingParser(allowed_tags, allowed_attrs or DEFAULT_ALLOWED_ATTRS)
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


def is_safe_sql(sql: str, allow_ddl: bool = False) -> bool:
    """Heuristic SQL safety check.

    Returns ``False`` when ``sql`` contains DDL (unless ``allow_ddl=True``)
    or well-known SQL-injection signatures (``UNION SELECT``, tautologies,
    ``xp_cmdshell``). Intentionally conservative — use a real parser
    (``sqlparse`` etc.) for stronger guarantees.
    """
    if not sql:
        return True
    if not allow_ddl and _DDL_RE.search(sql):
        return False
    if _SUSPICIOUS_DML_RE.search(sql):
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
