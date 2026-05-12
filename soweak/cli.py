"""soweak CLI.

Subcommands:

  ``soweak scan``           run a Policy against text or files
  ``soweak list``           list built-in pattern packs and their patterns
  ``soweak version``        print the package version
  ``soweak audit model``    hash a file or verify against a manifest (LLM03)
  ``soweak audit deps``     enumerate installed packages, check blocklist (LLM03)
  ``soweak audit canaries`` run a canary corpus through a model callable (LLM04)
  ``soweak audit policy``   lint a soweak Policy for misconfigurations
  ``soweak redteam``        replay the OWASP probe corpus, report coverage

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


# ---------------------------------------------------------------------------
# `soweak audit` subcommands (LLM03/04 build-time tooling)
# ---------------------------------------------------------------------------


def _cmd_audit_model(args: argparse.Namespace) -> int:
    from soweak.audit_tools import hash_file, verify_against_manifest

    if args.manifest:
        ok, msg = verify_against_manifest(args.path, args.manifest)
        print(msg)
        return 0 if ok else 1
    digest = hash_file(args.path, algorithm=args.algorithm)
    print(f"{args.algorithm}  {args.path}  {digest}")
    return 0


def _cmd_audit_deps(args: argparse.Namespace) -> int:
    from soweak.audit_tools import (
        check_packages_against_blocklist,
        list_python_packages,
    )

    blocklist: list[str] = []
    if args.blocklist:
        blocklist = [
            line.strip()
            for line in Path(args.blocklist).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    pkgs = list_python_packages()
    flagged = check_packages_against_blocklist(blocklist, pkgs)
    if args.json:
        print(
            json.dumps(
                {
                    "total": len(pkgs),
                    "flagged": [p.as_dict() for p in flagged],
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"{len(pkgs)} packages installed")
        if blocklist:
            if flagged:
                print(f"{len(flagged)} flagged:")
                for p in flagged:
                    print(f"  {p.name} {p.version}")
            else:
                print("0 flagged against blocklist")
        else:
            print("(no blocklist supplied; pass --blocklist FILE to check)")
    return 1 if flagged else 0


def _cmd_audit_canaries(args: argparse.Namespace) -> int:
    """Run a canary corpus through a model callable loaded from MODULE:FUNC.

    The corpus file is JSON: a list of objects with keys ``prompt``,
    ``expect_contains``, ``expect_not_contains``, ``name``.
    The model spec is ``module.path:function_name``; the function must
    accept a single ``str`` prompt and return a ``str``.
    """
    import importlib

    from soweak.audit_tools import Canary, run_canaries

    corpus_raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    canaries = [
        Canary(
            prompt=item["prompt"],
            expect_contains=tuple(item.get("expect_contains", [])),
            expect_not_contains=tuple(item.get("expect_not_contains", [])),
            name=item.get("name", ""),
        )
        for item in corpus_raw
    ]

    module_path, _, func_name = args.model.rpartition(":")
    if not module_path or not func_name:
        print(f"error: --model must be MODULE:FUNC, got {args.model!r}", file=sys.stderr)
        return 2
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)

    results = run_canaries(canaries, fn)
    failed = [r for r in results if not r.passed]
    if args.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False))
    else:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            name = r.canary.name or r.canary.prompt[:40]
            print(f"[{status}] {name}")
            for fail in r.failures:
                print(f"    {fail}")
        print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 1


def _cmd_redteam(args: argparse.Namespace) -> int:
    """Replay an OWASP probe corpus through a Policy; report coverage."""
    import importlib

    from soweak.core.policy import Policy
    from soweak.redteam import (
        DEFAULT_PROBES,
        Pipeline as _Pipeline,
        coverage_report,
        load_corpus,
        run_probes,
    )

    if args.policy:
        module_path, _, attr = args.policy.rpartition(":")
        if not module_path or not attr:
            print(f"error: --policy must be MODULE:ATTR, got {args.policy!r}", file=sys.stderr)
            return 2
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
        if callable(obj):
            obj = obj()
        if not isinstance(obj, Policy):
            print(f"error: {args.policy!r} did not yield a Policy", file=sys.stderr)
            return 2
        pipeline = _Pipeline(obj)
    else:
        pipeline = _default_pipeline()

    probes = load_corpus(args.corpus) if args.corpus else list(DEFAULT_PROBES)
    results = run_probes(pipeline, probes)
    coverage = coverage_report(results)

    if args.json:
        print(
            json.dumps(
                {
                    "results": [r.as_dict() for r in results],
                    "coverage": [c.as_dict() for c in coverage],
                },
                ensure_ascii=False,
            )
        )
    else:
        for r in results:
            status = "BLOCKED" if r.blocked else "passed"
            name = r.probe.name or r.probe.prompt[:40]
            print(f"[{status:7}] {r.probe.category.value} {name}")
        print()
        print(f"{'category':8} {'blocked/total':>13}  {'rate':>6}")
        for c in coverage:
            print(
                f"{c.category.value:8} {c.blocked:>5}/{c.total:<5}  {c.rate * 100:>5.0f}%"
            )
    return 0


def _cmd_audit_policy(args: argparse.Namespace) -> int:
    import importlib

    from soweak.audit_tools import lint_policy
    from soweak.core.policy import Policy

    module_path, _, attr = args.policy.rpartition(":")
    if not module_path or not attr:
        print(f"error: --policy must be MODULE:ATTR, got {args.policy!r}", file=sys.stderr)
        return 2
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    if callable(obj):
        obj = obj()
    if not isinstance(obj, Policy):
        print(f"error: {args.policy!r} did not yield a Policy", file=sys.stderr)
        return 2

    issues = lint_policy(obj)
    if args.json:
        print(json.dumps([i.as_dict() for i in issues], ensure_ascii=False))
        return 1 if any(i.severity == "error" for i in issues) else 0
    if not issues:
        print("ok — no issues")
        return 0
    for i in issues:
        print(f"[{i.severity}] {i.message}")
    return 1 if any(i.severity == "error" for i in issues) else 0


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

    audit = sub.add_parser(
        "audit", help="Build/deploy-time checks (LLM03 supply chain, LLM04 canaries, policy lint)"
    )
    audit_sub = audit.add_subparsers(dest="subcmd", required=True)

    am = audit_sub.add_parser("model", help="Hash a file or verify against a manifest")
    am.add_argument("path", help="Path to the model / weights file")
    am.add_argument(
        "--manifest",
        metavar="JSON",
        help="JSON file {filename: sha256}; verify rather than just hash",
    )
    am.add_argument(
        "--algorithm",
        default="sha256",
        help="hashlib algorithm (default: sha256)",
    )
    am.set_defaults(func=_cmd_audit_model)

    ad = audit_sub.add_parser(
        "deps", help="Enumerate installed Python packages; optionally check a blocklist"
    )
    ad.add_argument(
        "--blocklist",
        metavar="FILE",
        help="Path to a newline-delimited list of package names to flag",
    )
    ad.add_argument("--json", action="store_true", help="Emit JSON output")
    ad.set_defaults(func=_cmd_audit_deps)

    ac = audit_sub.add_parser(
        "canaries",
        help="Run a canary corpus (JSON) through a model callable (MODULE:FUNC)",
    )
    ac.add_argument("--corpus", required=True, help="Path to the JSON canary corpus")
    ac.add_argument(
        "--model",
        required=True,
        metavar="MODULE:FUNC",
        help="Importable callable accepting prompt -> output",
    )
    ac.add_argument("--json", action="store_true", help="Emit JSON results")
    ac.set_defaults(func=_cmd_audit_canaries)

    ap = audit_sub.add_parser("policy", help="Lint a soweak Policy")
    ap.add_argument(
        "policy",
        metavar="MODULE:ATTR",
        help="Importable attribute that is (or returns) a Policy",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON issues")
    ap.set_defaults(func=_cmd_audit_policy)

    redteam = sub.add_parser(
        "redteam",
        help="Replay an OWASP probe corpus against a Policy; report coverage",
    )
    redteam.add_argument(
        "--policy",
        metavar="MODULE:ATTR",
        help="Importable Policy (or callable returning one); default is the built-in",
    )
    redteam.add_argument(
        "--corpus",
        metavar="JSON",
        help="Path to a JSON probe corpus; default is the bundled set",
    )
    redteam.add_argument("--json", action="store_true", help="Emit JSON output")
    redteam.set_defaults(func=_cmd_redteam)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
