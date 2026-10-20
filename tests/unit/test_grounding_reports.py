"""Fixed evidence and real scoring-loop/workflow controls; no provider calls."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from attune_rag.eval.reporting import CONTEXT_REFUSAL, run_manifest, summarize_answers


def row(answer="a fact", supported=None, unsupported=None, **kwargs):
    supported = ["fact"] if supported is None else supported
    unsupported = [] if unsupported is None else unsupported
    total = len(supported) + len(unsupported)
    return dict(
        status="completed",
        answer=answer,
        supported_claims=supported,
        unsupported_claims=unsupported,
        score=len(supported) / total if total else 1.0,
        **kwargs,
    )


def test_distinct_denominators_do_not_hide_refusals_or_missing_verdicts():
    rows = [
        row(supported=["a", "b"], unsupported=["c"]),
        row(CONTEXT_REFUSAL, supported=[], abstained=True),
        row("", supported=[]),
        row("Let me help", supported=[]),
        {"status": "failed", "answer": "unjudged"},
    ]
    summary = summarize_answers(rows, 6)
    assert summary["complete"] is False
    assert summary["observed_mean_per_answer_faithfulness"] == pytest.approx(11 / 12)
    assert summary["answers_with_unsupported_claims"] == 1
    assert summary["unsupported_answer_denominator"] == 4
    assert summary["unsupported_answer_rate"] == 0.25
    assert summary["supported_claims"] == 2
    assert summary["unsupported_claims"] == 1
    assert summary["generated_answers"] == 5
    assert summary["requested_answers"] == 6
    assert summary["scored_answers"] == 4
    assert summary["unscored_answers"] == 2
    assert summary["zero_claim_answers"] == 3
    assert summary["exact_context_refusals"] == 1
    assert summary["empty_answers"] == 1
    assert summary["other_zero_claim_answers"] == 1
    assert summary["pipeline_abstentions"] == 1
    assert summary["pipeline_abstention_denominator"] == 5
    assert summary["refusal_appropriateness"] == summary["answer_usefulness"] == "not_assessed"
    assert summarize_answers(rows[:1], 1)["complete"] is True


@pytest.mark.parametrize(
    "patch",
    [
        {"score": float("nan")},
        {"score": True},
        {"score": 0.5},
        {"supported_claims": "not a list"},
        {"unsupported_claims": [None]},
    ],
)
def test_malformed_completed_verdict_is_rejected(patch):
    sample = row()
    sample.update(patch)
    with pytest.raises(ValueError):
        summarize_answers([sample], 1)


def test_no_verdict_has_no_observed_score():
    result = summarize_answers([], 40)
    assert result["observed_mean_per_answer_faithfulness"] is None
    assert result["unsupported_answer_rate"] is None
    assert result["unscored_answers"] == 40
    assert result["complete"] is False


def test_manifest_fingerprints_exact_queries_and_whitelists_environment(tmp_path, monkeypatch):
    import hashlib

    path = tmp_path / "queries.yaml"
    path.write_bytes(b"synthetic exact bytes\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DO_NOT_CAPTURE")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("ATTUNE_BENCH_ATTEMPT", "1")
    manifest = run_manifest(path)
    assert manifest["queries_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["run_id"] == "123"
    assert manifest["run_attempt"] == "2"
    assert manifest["benchmark_attempt"] == "1"
    assert manifest["schema_version"] == 1
    assert manifest["dependencies"]
    assert "DO_NOT_CAPTURE" not in json.dumps(manifest)


@pytest.mark.asyncio
async def test_partial_loop_retains_answer_before_judge_failure(tmp_path, monkeypatch):
    import attune_rag
    from attune_rag import benchmark, eval
    from attune_rag.eval.faithfulness import FaithfulnessResult

    result = SimpleNamespace(
        context="ACTUAL source tail",
        used_native_citations=False,
        claim_citations=(),
        fallback_used=False,
        abstained=False,
    )
    pipeline = SimpleNamespace(
        corpus=SimpleNamespace(name="synthetic", version="v1"),
        run_and_generate=AsyncMock(return_value=("a generated answer", result)),
    )
    judge = SimpleNamespace(
        _model="judge-stub",
        score=AsyncMock(
            side_effect=[
                FaithfulnessResult(
                    0.5, ["supported"], ["unsupported"], "fixed reasoning", "judge-stub"
                ),
                ValueError("DO_NOT_CAPTURE_EXCEPTION_BODY"),
            ]
        ),
    )
    monkeypatch.setattr(attune_rag, "RagPipeline", lambda: pipeline)
    monkeypatch.setattr(eval, "FaithfulnessJudge", lambda **kw: judge)
    snapshots = []
    output = tmp_path / "partial.json"

    def progress(evidence):
        snapshots.append(copy.deepcopy(evidence))
        benchmark._dump_json(output, evidence)

    queries = [{"id": f"q{i}", "query": "synthetic"} for i in range(3)]
    with pytest.raises(ValueError, match="DO_NOT_CAPTURE_EXCEPTION_BODY"):
        await benchmark._score_faithfulness(queries, 3, progress=progress)
    final = json.loads(output.read_text())
    assert final["status"] == "incomplete"
    assert final["summary"]["scored_answers"] == 1
    assert final["summary"]["unscored_answers"] == 2
    assert final["summary"]["complete"] is False
    assert final["per_query"][0]["unsupported_claims"] == ["unsupported"]
    pending = final["per_query"][1]
    assert pending["status"] == "failed"
    assert pending["failed_stage"] == "judge"
    assert pending["answer"] == "a generated answer"
    assert pending["context"] == "ACTUAL source tail"
    assert "score" not in pending
    assert pending["actual_model"] is None
    assert pending["judge_actual_model"] is None
    assert pending["prompt_variant"] == "citation"
    assert any(s["per_query"] and s["per_query"][-1]["status"] == "generated" for s in snapshots)
    assert "DO_NOT_CAPTURE_EXCEPTION_BODY" not in output.read_text()
    assert pipeline.run_and_generate.await_count == 2
    assert judge.score.call_args.kwargs["passages"] == result.context


def test_available_corpus_fingerprint_is_explicitly_body_only(tmp_path):
    from attune_rag.corpus import DirectoryCorpus
    from attune_rag.corpus.attune_help import AttuneHelpCorpus
    from attune_rag.eval.reporting import corpus_provenance

    (tmp_path / "sample.md").write_text("A synthetic source body.")
    directory = DirectoryCorpus(tmp_path)
    direct = corpus_provenance(directory)
    assert direct["content_fingerprint"] == directory.version
    assert direct["fingerprint_status"] == "available"
    assert "excludes aliases and summaries" in direct["fingerprint_scope"]
    bundled = AttuneHelpCorpus(SimpleNamespace(templates_root=tmp_path, version="synthetic-v1"))
    adapted = corpus_provenance(bundled)
    assert adapted["version"] == "synthetic-v1"
    assert adapted["content_fingerprint"] == direct["content_fingerprint"]
    unavailable = corpus_provenance(SimpleNamespace(name="other", version="v1"))
    assert unavailable["content_fingerprint"] is None
    assert unavailable["fingerprint_status"] == "not_exposed_by_corpus_interface"


@pytest.mark.asyncio
async def test_native_pipeline_report_and_judge_share_tail_evidence(monkeypatch):
    import attune_rag
    from attune_rag import benchmark, eval, providers
    from attune_rag.eval.faithfulness import FaithfulnessResult
    from tests.unit.test_grounding_contracts import Provider, entry, pipeline

    pipe, _ = pipeline([entry("x" * 25_000), entry("  FINAL_TAIL_FACT Ω ", "tail.md")])
    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda name: provider)
    monkeypatch.setattr(attune_rag, "RagPipeline", lambda: pipe)
    judge = SimpleNamespace(
        _model="judge-stub",
        score=AsyncMock(
            return_value=FaithfulnessResult(1.0, ["tail fact"], [], "tail exists", "judge-stub")
        ),
    )
    monkeypatch.setattr(eval, "FaithfulnessJudge", lambda **kw: judge)
    records = []
    await benchmark._score_faithfulness(
        [{"id": "tail", "query": "tail"}],
        3,
        use_native_citations=True,
        progress=lambda e: records.append(copy.deepcopy(e)),
    )
    report = records[-1]
    row = report["per_query"][0]
    assert report["status"] == "completed" and report["summary"]["complete"]
    assert row["generation_documents"][-1] == {"title": "tail.md", "text": "  FINAL_TAIL_FACT Ω "}
    assert row["claim_citations"][0]["document_index"] == 1
    assert row["retrieval_sources"][-1]["template_path"] == "tail.md"
    assert row["context"] == judge.score.call_args.kwargs["passages"]
    assert row["generation_documents"][-1]["text"] in row["context"]
    assert row["abstained"] is False and row["fallback_used"] is False


def test_installed_reporter_cannot_inherit_consuming_application_git_sha(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from attune_rag.eval import reporting

    wheel_module = (
        tmp_path / "other-app/.venv/lib/python3.13/site-packages/attune_rag/eval/reporting.py"
    )
    wheel_module.parent.mkdir(parents=True)
    wheel_module.write_text("# installed reporter layout")
    monkeypatch.setattr(reporting, "__file__", str(wheel_module))
    unrelated_git = Mock(return_value=SimpleNamespace(stdout="b" * 40))
    monkeypatch.setattr(reporting.subprocess, "run", unrelated_git)
    queries = tmp_path / "queries.yaml"
    queries.write_text("synthetic")
    manifest = reporting.run_manifest(queries)
    assert manifest["code_sha"] is None
    assert manifest["tracked_changes"] is None
    unrelated_git.assert_not_called()


def test_source_layout_requires_reporter_in_reported_git_revision(tmp_path, monkeypatch):
    import subprocess

    from attune_rag.eval import reporting

    source = tmp_path / "source/src/attune_rag/eval/reporting.py"
    source.parent.mkdir(parents=True)
    source.write_text("# untracked source-layout reporter")
    root = source.parents[3]
    (root / ".git").mkdir()
    monkeypatch.setattr(reporting, "__file__", str(source))
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        if args[1] == "cat-file":
            raise subprocess.CalledProcessError(128, args)
        if "--show-toplevel" in args:
            return SimpleNamespace(stdout=str(root))
        return SimpleNamespace(stdout="b" * 40 if "HEAD" in args else "")

    monkeypatch.setattr(reporting.subprocess, "run", git)
    query = tmp_path / "queries.yaml"
    query.write_text("synthetic")
    result = reporting.run_manifest(query)
    assert result["code_sha"] is None and result["tracked_changes"] is None
    assert result["code_sha_status"] == "unavailable_or_unverified_source_revision"
    assert any("cat-file" in args for args in calls)
