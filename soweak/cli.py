"""soweak CLI.

Subcommands:

  ``soweak scan``     run a Policy against text or files
  ``soweak version``  print the package version
  ``soweak list``     list built-in pattern packs and their patterns

The default policy applied to ``scan`` is suitable for ad-hoc auditing of
prompts. For production use, define your own Policy in code and call
``pipeline.run`` from your application.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from soweak import (
    BlockEnforcer,
    Decision,
    Pipeline,
    PolicyBuilder,
    RedactEnforcer,
    Severity,
    __version__,
)
from soweak.detectors import (
    INPUT_DLP_PACK,
    PROMPT_INJECTION_PACK,
    SYSTEM_PROMPT_EXTRACTION_PACK,
    input_dlp_detector,
    prompt_injection_detector,
    system_prompt_extraction_detector,
)
from soweak.detectors.patterns import PatternPack


def _default_pipeline() -> Pipeline:
    policy = (
        PolicyBuilder()
        .on_input("prompt-injection")
        .detect(prompt_injection_detector(), system_prompt_extraction_detector())
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .on_input("input-dlp")
        .detect(input_dlp_detector())
        .enforce(RedactEnforcer(min_severity=Severity.HIGH))
        .build()
    )
    return Pipeline(policy)


def _iter_inputs(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    """Yield (source, text) pairs from the CLI args."""
    if args.text:
        yield "<arg>", args.text
    for path in args.file:
        yield path, Path(path).read_text(encoding="utf-8")
    if args.stdin:
        yield "<stdin>", sys.stdin.read()


def _format_human(source: str, text: str, decision: Decision) -> str:
    lines = [f"── {source} ──", f"action: {decision.action.value}"]
    if decision.reason:
        lines.append(f"reason: {decision.reason}")
    for s in decision.signals:
        snippet = (s.matched_text or "")[:80]
        lines.append(
            f"  • [{s.category.value} {s.severity.label}] "
            f"{s.message}  confidence={s.confidence:.2f}"
            + (f"  match={snippet!r}" if snippet else "")
        )
    if decision.action.value == "redact" and decision.payload.text != text:
        lines.append(f"redacted: {decision.payload.text}")
    return "\n".join(lines)


def _format_json(source: str, text: str, decision: Decision) -> str:
    return json.dumps(
        {
            "source": source,
            "action": decision.action.value,
            "reason": decision.reason,
            "input": text,
            "output": decision.payload.text,
            "signals": [
                {
                    "detector": s.detector,
                    "category": s.category.value,
                    "severity": s.severity.label,
                    "confidence": s.confidence,
                    "message": s.message,
                    "matched_text": s.matched_text,
                    "span": list(s.span) if s.span else None,
                    "attack_type": s.metadata.get("attack_type"),
                }
                for s in decision.signals
            ],
        },
        ensure_ascii=False,
    )


def _cmd_scan(args: argparse.Namespace) -> int:
    pipeline = _default_pipeline()
    inputs = list(_iter_inputs(args))
    if not inputs:
        print(
            "error: no input provided (use --stdin, --file, or pass text)",
            file=sys.stderr,
        )
        return 2
    any_blocked = False
    formatter = _format_json if args.json else _format_human
    for source, text in inputs:
        decision = pipeline.check_input(text)
        print(formatter(source, text, decision))
        if decision.blocked:
            any_blocked = True
    return 1 if any_blocked else 0


def _cmd_list(args: argparse.Namespace) -> int:
    packs: list[PatternPack] = [
        PROMPT_INJECTION_PACK,
        INPUT_DLP_PACK,
        SYSTEM_PROMPT_EXTRACTION_PACK,
    ]
    for pack in packs:
        print(
            f"# {pack.name} ({pack.category.value}, v{pack.version}) — "
            f"{len(pack.patterns)} patterns"
        )
        if args.verbose:
            for p in pack.patterns:
                print(f"  [{p.severity.label}] {p.description}")
                print(f"    regex: {p.regex}")
        print()
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soweak",
        description="OWASP LLM Top 10 security middleware framework — CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan text/files against the default policy")
    scan.add_argument("text", nargs="?", help="Text to scan (positional)")
    scan.add_argument(
        "-f",
        "--file",
        action="append",
        default=[],
        metavar="FILE",
        help="Read text from FILE (may be repeated)",
    )
    scan.add_argument("--stdin", action="store_true", help="Read text from stdin")
    scan.add_argument("--json", action="store_true", help="Emit JSON output")
    scan.set_defaults(func=_cmd_scan)

    listcmd = sub.add_parser("list", help="List built-in pattern packs")
    listcmd.add_argument(
        "-v", "--verbose", action="store_true", help="Show individual patterns"
    )
    listcmd.set_defaults(func=_cmd_list)

    ver = sub.add_parser("version", help="Print the soweak version")
    ver.set_defaults(func=_cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
