"""LLM03/04 audit-tool tests (file hashing, deps, canaries, policy lint)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soweak import (
    BlockEnforcer,
    Policy,
    PolicyBuilder,
    Severity,
)
from soweak.audit_tools import (
    Canary,
    CanaryResult,
    check_packages_against_blocklist,
    hash_file,
    InstalledPackage,
    lint_policy,
    list_python_packages,
    load_manifest,
    run_canaries,
    verify_against_manifest,
)
from soweak.detectors import (
    input_dlp_detector,
    prompt_injection_detector,
)


# ---------------- hash_file ----------------


def test_hash_file_sha256(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello soweak")
    h = hash_file(p)
    # sha256("hello soweak") precomputed:
    assert h == "9beadd5b3ade2116f3093475b31f3063d3b0348f1156cb4f0e1d46baeb233d18"


def test_hash_file_alternate_algorithm(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    h = hash_file(p, algorithm="md5")
    assert h == "900150983cd24fb0d6963f7d28e17f72"


def test_hash_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nope")


# ---------------- verify_against_manifest ----------------


def test_verify_against_manifest_ok(tmp_path: Path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    digest = hash_file(p)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model.bin": digest}))
    ok, msg = verify_against_manifest(p, manifest)
    assert ok
    assert "ok" in msg.lower()


def test_verify_against_manifest_mismatch(tmp_path: Path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model.bin": "deadbeef" * 8}))
    ok, msg = verify_against_manifest(p, manifest)
    assert not ok
    assert "mismatch" in msg


def test_verify_against_manifest_unknown_file(tmp_path: Path):
    p = tmp_path / "other.bin"
    p.write_bytes(b"x")
    ok, msg = verify_against_manifest(p, {"different.bin": "abc"})
    assert not ok
    assert "no entry" in msg


def test_load_manifest_rejects_non_dict(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_manifest(p)


# ---------------- list_python_packages ----------------


def test_list_python_packages_includes_pytest():
    pkgs = list_python_packages()
    names = {p.name.lower() for p in pkgs}
    assert "pytest" in names
    # Every entry has a version string.
    assert all(p.version for p in pkgs)


def test_check_packages_against_blocklist_case_insensitive():
    fake = [
        InstalledPackage("requests", "2.0.0"),
        InstalledPackage("EvilLib", "1.0.0"),
        InstalledPackage("safe-pkg", "0.1.0"),
    ]
    flagged = check_packages_against_blocklist(["EVILLIB"], fake)
    assert len(flagged) == 1
    assert flagged[0].name == "EvilLib"


def test_check_packages_empty_blocklist_returns_empty():
    fake = [InstalledPackage("requests", "2.0.0")]
    assert check_packages_against_blocklist([], fake) == []


# ---------------- canaries ----------------


def test_run_canaries_passes_when_expectations_met():
    canaries = [
        Canary(prompt="2+2?", expect_contains=("4",), name="math"),
        Canary(prompt="say no secret", expect_not_contains=("secret",)),
    ]
    results = run_canaries(canaries, lambda p: "Answer is 4." if "2+2" in p else "OK")
    assert all(r.passed for r in results)


def test_run_canaries_reports_missing_substring():
    canaries = [Canary(prompt="x", expect_contains=("hello",))]
    results = run_canaries(canaries, lambda p: "goodbye")
    assert not results[0].passed
    assert "missing expected" in results[0].failures[0]


def test_run_canaries_reports_forbidden_substring():
    canaries = [Canary(prompt="x", expect_not_contains=("secret",))]
    results = run_canaries(canaries, lambda p: "this leaks the SECRET")
    assert not results[0].passed
    assert "contains forbidden" in results[0].failures[0]


def test_canary_result_as_dict():
    c = Canary(prompt="hi", expect_contains=("hello",), name="greet")
    r = CanaryResult(canary=c, output="hello!", passed=True, failures=())
    d = r.as_dict()
    assert d["passed"] is True
    assert d["canary"]["name"] == "greet"


# ---------------- lint_policy ----------------


def test_lint_empty_policy_is_error():
    issues = lint_policy(Policy())
    assert any(i.severity == "error" for i in issues)


def test_lint_warns_when_input_unscanned():
    p = (
        PolicyBuilder()
        .on_output()
        .detect(input_dlp_detector())
        .enforce(BlockEnforcer())
        .build()
    )
    issues = lint_policy(p)
    assert any("INPUT" in i.message for i in issues)


def test_lint_warns_when_output_unscanned():
    p = (
        PolicyBuilder()
        .on_input()
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer())
        .build()
    )
    issues = lint_policy(p)
    assert any("OUTPUT" in i.message for i in issues)


def test_lint_warns_on_rule_with_no_detectors():
    p = (
        PolicyBuilder()
        .on_input()
        .enforce(BlockEnforcer())
        .on_output()
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer())
        .build()
    )
    issues = lint_policy(p)
    assert any("no detectors" in i.message for i in issues)


def test_lint_warns_on_duplicate_detector_classes():
    p = (
        PolicyBuilder()
        .on_input()
        .detect(prompt_injection_detector(), prompt_injection_detector())
        .enforce(BlockEnforcer())
        .on_output()
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer())
        .build()
    )
    issues = lint_policy(p)
    assert any("duplicate" in i.message for i in issues)


def test_lint_clean_policy_has_no_issues():
    p = (
        PolicyBuilder()
        .on_input()
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .on_output()
        .detect(input_dlp_detector())
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .build()
    )
    issues = lint_policy(p)
    assert issues == []
