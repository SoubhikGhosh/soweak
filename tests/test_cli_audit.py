"""CLI ``soweak audit`` integration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from soweak.cli import main


def test_audit_model_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    code = main(["audit", "model", str(p)])
    assert code == 0
    out = capsys.readouterr().out
    assert "sha256" in out
    assert str(p) in out


def test_audit_model_verify_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    manifest = tmp_path / "m.json"
    from soweak.audit_tools import hash_file

    manifest.write_text(json.dumps({"model.bin": hash_file(p)}))
    code = main(["audit", "model", str(p), "--manifest", str(manifest)])
    assert code == 0


def test_audit_model_verify_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "model.bin"
    p.write_bytes(b"weights")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"model.bin": "00" * 32}))
    code = main(["audit", "model", str(p), "--manifest", str(manifest)])
    assert code == 1
    assert "mismatch" in capsys.readouterr().out


def test_audit_deps_no_blocklist(capsys: pytest.CaptureFixture[str]):
    code = main(["audit", "deps"])
    assert code == 0
    out = capsys.readouterr().out
    assert "packages installed" in out
    assert "no blocklist" in out.lower()


def test_audit_deps_with_blocklist_no_hits(tmp_path: Path):
    bl = tmp_path / "bl.txt"
    bl.write_text("totally-not-a-real-package\n")
    code = main(["audit", "deps", "--blocklist", str(bl)])
    assert code == 0


def test_audit_deps_json(capsys: pytest.CaptureFixture[str]):
    code = main(["audit", "deps", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "total" in parsed
    assert parsed["flagged"] == []


def test_audit_policy_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
):
    """Write a tiny policy module to a tmp dir, point sys.path at it, lint."""
    mod = tmp_path / "_audit_policy_mod_clean.py"
    mod.write_text(
        "from soweak import PolicyBuilder, BlockEnforcer\n"
        "from soweak.detectors import prompt_injection_detector, input_dlp_detector\n"
        "policy = (PolicyBuilder()"
        ".on_input().detect(prompt_injection_detector()).enforce(BlockEnforcer())"
        ".on_output().detect(input_dlp_detector()).enforce(BlockEnforcer())"
        ".build())\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(["audit", "policy", "_audit_policy_mod_clean:policy"])
    assert code == 0
    assert "ok" in capsys.readouterr().out.lower()


def test_audit_policy_lints_issues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
):
    mod = tmp_path / "_audit_policy_mod_issues.py"
    mod.write_text(
        "from soweak import PolicyBuilder, BlockEnforcer\n"
        "from soweak.detectors import prompt_injection_detector\n"
        "policy = (PolicyBuilder()"
        ".on_input().detect(prompt_injection_detector()).enforce(BlockEnforcer())"
        ".build())\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(["audit", "policy", "_audit_policy_mod_issues:policy"])
    assert code == 0  # warnings only
    out = capsys.readouterr().out
    assert "OUTPUT" in out


def test_audit_canaries_runs_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
):
    """Provide a model callable in a temp module."""
    mod = tmp_path / "_audit_canary_model.py"
    mod.write_text("def call(prompt): return 'answer is 4'\n")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            [
                {"prompt": "2+2?", "expect_contains": ["4"], "name": "math"},
                {"prompt": "leak?", "expect_not_contains": ["secret"], "name": "no-leak"},
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(
        [
            "audit",
            "canaries",
            "--corpus",
            str(corpus),
            "--model",
            "_audit_canary_model:call",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "2/2 passed" in out


def test_audit_canaries_reports_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
):
    mod = tmp_path / "_audit_canary_bad_model.py"
    mod.write_text("def call(prompt): return 'wrong'\n")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps([{"prompt": "x", "expect_contains": ["right"], "name": "t"}])
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(
        [
            "audit",
            "canaries",
            "--corpus",
            str(corpus),
            "--model",
            "_audit_canary_bad_model:call",
            "--json",
        ]
    )
    assert code == 1
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["passed"] is False
