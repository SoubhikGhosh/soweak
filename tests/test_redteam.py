"""Red-team probe runner + coverage report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soweak import (
    BlockEnforcer,
    LogOnlyEnforcer,
    OwaspCategory,
    Pipeline,
    PolicyBuilder,
    Severity,
)
from soweak.detectors import (
    input_dlp_detector,
    prompt_injection_detector,
    system_prompt_extraction_detector,
)
from soweak.redteam import (
    Boundary,
    CategoryCoverage,
    DEFAULT_PROBES,
    Probe,
    coverage_report,
    load_corpus,
    run_probes,
)


@pytest.fixture
def blocking_pipeline() -> Pipeline:
    return Pipeline(
        PolicyBuilder()
        .on_input()
        .detect(
            prompt_injection_detector(),
            input_dlp_detector(),
            system_prompt_extraction_detector(),
        )
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .build()
    )


@pytest.fixture
def permissive_pipeline() -> Pipeline:
    return Pipeline(
        PolicyBuilder()
        .on_input()
        .detect(prompt_injection_detector())
        .enforce(LogOnlyEnforcer())
        .build()
    )


def test_default_corpus_has_each_category():
    cats = {p.category for p in DEFAULT_PROBES}
    assert OwaspCategory.LLM01_PROMPT_INJECTION in cats
    assert OwaspCategory.LLM02_SENSITIVE_INFO in cats
    assert OwaspCategory.LLM07_SYSTEM_PROMPT_LEAKAGE in cats


def test_run_probes_with_blocking_policy_high_block_rate(blocking_pipeline):
    results = run_probes(blocking_pipeline)
    blocked = sum(1 for r in results if r.blocked)
    assert blocked >= len(results) - 1  # at most one falls through


def test_run_probes_with_permissive_policy_blocks_nothing(permissive_pipeline):
    results = run_probes(permissive_pipeline)
    assert all(not r.blocked for r in results)


def test_coverage_report_buckets_by_category(blocking_pipeline):
    results = run_probes(blocking_pipeline)
    coverage = coverage_report(results)
    assert all(isinstance(c, CategoryCoverage) for c in coverage)
    total_from_buckets = sum(c.total for c in coverage)
    assert total_from_buckets == len(results)
    # Each rate between 0 and 1
    assert all(0.0 <= c.rate <= 1.0 for c in coverage)


def test_coverage_report_sorted_by_category(blocking_pipeline):
    results = run_probes(blocking_pipeline)
    coverage = coverage_report(results)
    values = [c.category.value for c in coverage]
    assert values == sorted(values)


def test_load_corpus_round_trip(tmp_path: Path):
    corpus = [
        {"prompt": "test", "category": "LLM01", "name": "t1"},
        {"prompt": "leak", "category": "LLM02", "boundary": "input", "name": "t2"},
    ]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(corpus))
    probes = load_corpus(p)
    assert len(probes) == 2
    assert probes[0].category is OwaspCategory.LLM01_PROMPT_INJECTION
    assert probes[1].boundary is Boundary.INPUT
    assert probes[0].name == "t1"


def test_load_corpus_rejects_non_list(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"prompt": "x"}')
    with pytest.raises(ValueError, match="list"):
        load_corpus(p)


def test_load_corpus_rejects_bad_category(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"prompt": "x", "category": "XYZ"}]))
    with pytest.raises(ValueError, match="category"):
        load_corpus(p)


def test_load_corpus_rejects_bad_boundary(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"prompt": "x", "category": "LLM01", "boundary": "elsewhere"}]))
    with pytest.raises(ValueError, match="boundary"):
        load_corpus(p)


def test_probe_result_as_dict(blocking_pipeline):
    results = run_probes(blocking_pipeline, probes=[DEFAULT_PROBES[0]])
    d = results[0].as_dict()
    assert "blocked" in d
    assert "probe" in d
    assert d["probe"]["category"] == "LLM01"


def test_category_coverage_rate_zero_when_empty():
    c = CategoryCoverage(category=OwaspCategory.LLM01_PROMPT_INJECTION, total=0, blocked=0)
    assert c.rate == 0.0


def test_custom_probe_with_explicit_boundary(blocking_pipeline):
    custom = [Probe(prompt="hello", category=OwaspCategory.LLM01_PROMPT_INJECTION)]
    results = run_probes(blocking_pipeline, probes=custom)
    assert len(results) == 1
    assert not results[0].blocked  # benign prompt should not match
