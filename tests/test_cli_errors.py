"""CLI error-path coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soweak.cli import main


def test_scan_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["nope"])


def test_audit_model_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hashing a non-existent path must raise rather than silently succeed."""
    with pytest.raises(FileNotFoundError):
        main(["audit", "model", str(tmp_path / "absent.bin")])


def test_audit_model_manifest_missing_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "weights.bin"
    p.write_bytes(b"x")
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"otherfile.bin": "00" * 32}))
    code = main(["audit", "model", str(p), "--manifest", str(m)])
    assert code == 1
    assert "no entry" in capsys.readouterr().out


def test_audit_canaries_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(
            [
                "audit",
                "canaries",
                "--corpus",
                str(tmp_path / "absent.json"),
                "--model",
                "soweak:soweak.__version__",
            ]
        )


def test_audit_canaries_bad_model_spec(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "c.json"
    corpus.write_text("[]")
    code = main(
        ["audit", "canaries", "--corpus", str(corpus), "--model", "not_a_spec"]
    )
    assert code == 2
    assert "MODULE:FUNC" in capsys.readouterr().err


def test_audit_policy_bad_spec(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["audit", "policy", "not_a_spec"])
    assert code == 2
    assert "MODULE:ATTR" in capsys.readouterr().err


def test_audit_policy_wrong_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    mod = tmp_path / "_audit_wrong.py"
    mod.write_text("not_a_policy = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(["audit", "policy", "_audit_wrong:not_a_policy"])
    assert code == 2
    assert "did not yield a Policy" in capsys.readouterr().err


def test_audit_deps_blocklist_with_comments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bl = tmp_path / "bl.txt"
    bl.write_text("# comment\nnot-installed-pkg\n\n  # also comment\n")
    code = main(["audit", "deps", "--blocklist", str(bl)])
    assert code == 0  # nothing actually flagged


def test_redteam_bad_policy_spec(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["redteam", "--policy", "bad_spec_no_colon"])
    assert code == 2
    assert "MODULE:ATTR" in capsys.readouterr().err


def test_redteam_policy_wrong_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    mod = tmp_path / "_redteam_wrong.py"
    mod.write_text("not_a_policy = 'hi'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main(["redteam", "--policy", "_redteam_wrong:not_a_policy"])
    assert code == 2
    assert "did not yield a Policy" in capsys.readouterr().err


def test_scan_file_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["scan", "--file", str(tmp_path / "absent.txt")])


def test_list_verbose_shows_individual_patterns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["list", "--verbose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "regex:" in out
