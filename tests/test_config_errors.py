"""config.py error-path coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soweak import build_policy, load_policy


def test_build_policy_rules_must_be_list() -> None:
    with pytest.raises(ValueError, match="rules must be a list"):
        build_policy({"version": 1, "rules": "not a list"})


def test_build_policy_rule_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        build_policy({"version": 1, "rules": ["just-a-string"]})


def test_build_policy_rule_missing_boundary() -> None:
    with pytest.raises(ValueError, match="missing 'boundary'"):
        build_policy(
            {
                "version": 1,
                "rules": [
                    {
                        "name": "x",
                        "detectors": [{"type": "prompt_injection"}],
                        "enforcer": {"type": "block"},
                    }
                ],
            }
        )


def test_build_policy_detectors_must_be_list() -> None:
    with pytest.raises(ValueError, match="detectors must be a list"):
        build_policy(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "input",
                        "detectors": "not a list",
                        "enforcer": {"type": "block"},
                    }
                ]
            }
        )


def test_build_policy_detector_spec_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        build_policy(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "input",
                        "detectors": ["string-not-dict"],
                        "enforcer": {"type": "block"},
                    }
                ]
            }
        )


def test_build_policy_detector_missing_type() -> None:
    with pytest.raises(ValueError, match="missing 'type'"):
        build_policy(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "input",
                        "detectors": [{"no_type": "x"}],
                        "enforcer": {"type": "block"},
                    }
                ]
            }
        )


def test_build_policy_enforcer_required() -> None:
    with pytest.raises(ValueError, match="enforcer must be a mapping"):
        build_policy(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "input",
                        "detectors": [{"type": "prompt_injection"}],
                    }
                ]
            }
        )


def test_build_policy_detector_kwarg_mismatch() -> None:
    """If a detector factory rejects supplied kwargs, the error references the rule."""
    with pytest.raises(ValueError, match="canary"):
        build_policy(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "output",
                        "detectors": [{"type": "canary"}],  # missing required tokens
                        "enforcer": {"type": "block"},
                    }
                ]
            }
        )


def test_load_policy_format_override(tmp_path: Path) -> None:
    p = tmp_path / "policy.txt"
    p.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "x",
                        "boundary": "input",
                        "detectors": [{"type": "prompt_injection"}],
                        "enforcer": {"type": "block"},
                    }
                ]
            }
        )
    )
    pol = load_policy(p, format="json")
    assert len(pol.rules) == 1
