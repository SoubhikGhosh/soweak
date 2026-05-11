"""Built-in detectors and pattern packs.

The framework is detector-agnostic — anything implementing
:class:`soweak.core.Detector` works. This module ships the baseline
detectors that v3.x covers: a configurable :class:`PatternMatchDetector`,
the :class:`CanaryDetector` for output-side system-prompt leakage, and a
set of curated :class:`PatternPack` s for each OWASP category we cover.
"""

from soweak.core.types import Boundary
from soweak.detectors.canary import CanaryDetector
from soweak.detectors.pattern_match import PatternMatchDetector
from soweak.detectors.patterns import (
    INPUT_DLP_PACK,
    OUTPUT_DLP_PACK,
    OUTPUT_HTML_PACK,
    OUTPUT_SHELL_PACK,
    OUTPUT_SQL_PACK,
    PROMPT_INJECTION_PACK,
    SYSTEM_PROMPT_EXTRACTION_PACK,
    Pattern,
    PatternPack,
)


# ---------- input boundary (LLM01 / LLM02 / LLM07) ----------


def prompt_injection_detector() -> PatternMatchDetector:
    """Default LLM01 (prompt injection) detector for the input boundary."""
    return PatternMatchDetector(PROMPT_INJECTION_PACK)


def input_dlp_detector() -> PatternMatchDetector:
    """Default LLM02 input DLP detector (PII / secrets / credentials)."""
    return PatternMatchDetector(INPUT_DLP_PACK)


def system_prompt_extraction_detector() -> PatternMatchDetector:
    """Default LLM07 detector for system-prompt extraction attempts on input."""
    return PatternMatchDetector(SYSTEM_PROMPT_EXTRACTION_PACK)


# ---------- output boundary (LLM02 output, LLM05) ----------


def output_dlp_detector() -> PatternMatchDetector:
    """LLM02 output-side DLP: scans model responses for leaked secrets / PII,
    plus internal network references and connection strings the model
    typically shouldn't echo."""
    return PatternMatchDetector(OUTPUT_DLP_PACK, boundaries=(Boundary.OUTPUT,))


def output_html_detector() -> PatternMatchDetector:
    """LLM05: flag risky HTML in model output (script tags, event handlers,
    dangerous URL schemes, iframe/object/embed)."""
    return PatternMatchDetector(OUTPUT_HTML_PACK, boundaries=(Boundary.OUTPUT,))


def output_sql_detector() -> PatternMatchDetector:
    """LLM05: flag risky SQL in model output (DDL, SQLi tautologies,
    information_schema reconnaissance, xp_cmdshell, etc.)."""
    return PatternMatchDetector(OUTPUT_SQL_PACK, boundaries=(Boundary.OUTPUT,))


def output_shell_detector() -> PatternMatchDetector:
    """LLM05: flag risky shell commands in model output (rm -rf /,
    curl|bash, dd to disk, fork bombs, reverse shells)."""
    return PatternMatchDetector(OUTPUT_SHELL_PACK, boundaries=(Boundary.OUTPUT,))


__all__ = [
    # core types
    "CanaryDetector",
    "Pattern",
    "PatternMatchDetector",
    "PatternPack",
    # input packs
    "INPUT_DLP_PACK",
    "PROMPT_INJECTION_PACK",
    "SYSTEM_PROMPT_EXTRACTION_PACK",
    # output packs
    "OUTPUT_DLP_PACK",
    "OUTPUT_HTML_PACK",
    "OUTPUT_SHELL_PACK",
    "OUTPUT_SQL_PACK",
    # input factories
    "input_dlp_detector",
    "prompt_injection_detector",
    "system_prompt_extraction_detector",
    # output factories
    "output_dlp_detector",
    "output_html_detector",
    "output_shell_detector",
    "output_sql_detector",
]
