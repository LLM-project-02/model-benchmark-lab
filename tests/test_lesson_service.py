"""학생용 서비스 코드의 실제 요청 형식과 실패 기록을 외부 호출 없이 검사한다."""

import csv
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI as RealOpenAI

STEPS = Path(__file__).resolve().parents[1] / "steps"
sys.path.insert(0, str(STEPS))
sys.path.insert(0, str(STEPS.parent / "scripts"))

import step07_call_models as models
import step08_connect_chain as chain
import step09_compare as comparison
import step10_serve_api as api
import verify_service

VALID_TEXT = '{"answer":"상태를 확인해 주세요.","next_steps":["공식 접수"]}'


def test_document_prompt_summarizes_document_instead_of_shipping():
    prompt = models.build_prompt(models.SAMPLE_INPUTS["documents"], dataset="documents")
    assert "한국어 공지" in prompt
    assert "대상, 일정, 해야 할 일" in prompt
    assert "교육장 이용 공지" in prompt
    assert "배송 확인 방법" not in prompt
    assert "운영 회의록" in chain.SAMPLE_INPUTS["documents"]


def install_http_transport(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        models.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )


def test_ollama_request_and_response_fields(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={
            "response": VALID_TEXT, "done": True, "done_reason": "stop",
            "prompt_eval_count": 21, "eval_count": 12,
        })

    install_http_transport(monkeypatch, handler)
    result = models.call_model("ollama", "같은 프롬프트")
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/generate"
    assert body["prompt"] == "같은 프롬프트"
    assert body["model"] == models.OLLAMA_MODEL
    assert body["stream"] is False
    assert body["options"]["num_predict"] == models.MAX_OUTPUT_TOKENS
    assert result["text"] == VALID_TEXT
    assert result["usage"] == {"input_tokens": 21, "output_tokens": 12}
    assert result["generation_complete"] is True
    assert result["output_format_valid"] is True
    assert result["error"] is None
    assert result["latency_seconds"] >= 0


def test_ollama_length_limit_and_missing_usage_are_preserved(monkeypatch):
    install_http_transport(monkeypatch, lambda request: httpx.Response(
        200, json={"response": "unfinished", "done": True, "done_reason": "length"}
    ))
    result = models.call_model("ollama", "입력")
    assert result["error"] is None
    assert result["generation_complete"] is False
    assert result["output_format_valid"] is False
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}


def test_openai_real_sdk_request_with_http_transport(monkeypatch):
    requests = []
    configurations = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "resp_lesson_test", "object": "response", "created_at": 1,
            "status": "completed", "model": models.OPENAI_MODEL,
            "output": [{
                "id": "msg_lesson_test", "type": "message", "role": "assistant",
                "status": "completed", "content": [{
                    "type": "output_text", "text": VALID_TEXT, "annotations": [],
                }],
            }],
            "usage": {
                "input_tokens": 15, "output_tokens": 9, "total_tokens": 24,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        })

    def client_factory(**kwargs):
        configurations.append(kwargs)
        return RealOpenAI(
            api_key="lesson-test-placeholder", base_url="https://lesson.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "lesson-test-placeholder")
    monkeypatch.setattr(models, "OpenAI", client_factory)
    result = models.call_model("openai", "공통 입력")
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/responses"
    assert body["input"] == "공통 입력"
    assert body["model"] == models.OPENAI_MODEL
    assert body["reasoning"] == {"effort": "none"}
    assert body["store"] is False
    assert body["max_output_tokens"] == models.MAX_OUTPUT_TOKENS
    assert configurations == [{"timeout": models.TIMEOUT_SECONDS, "max_retries": 0}]
    assert result["text"] == VALID_TEXT
    assert result["usage"] == {"input_tokens": 15, "output_tokens": 9}
    assert result["generation_complete"] is True
    assert result["output_format_valid"] is True


def test_missing_key_and_provider_failures_never_become_fake_answers(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = models.call_model("openai", "입력")
    assert result["error"] == "RuntimeError"
    assert result["text"] == ""
    assert result["model"] is None

    def fail(prompt):
        raise RuntimeError("private-payload-and-key")

    monkeypatch.setattr(models, "call_ollama", fail)
    result = models.call_model("ollama", "입력")
    assert result["error"] == "RuntimeError"
    assert "private-payload-and-key" not in json.dumps(result)
    assert result["generation_complete"] is False


@pytest.mark.parametrize("raw", [
    'not-json', '{"answer":123,"next_steps":[]}',
    '{"answer":"답변","next_steps":"확인"}',
    '{"answer":"","next_steps":[]}',
    '{"answer":"답변","next_steps":[],"extra":1}',
    '{"answer":"   ","next_steps":[]}',  # 공백뿐인 답변
    '{"answer":"답변","next_steps":["  "]}',  # 공백뿐인 행동 항목
])
def test_output_schema_rejects_invalid_structure(raw):
    assert models.valid_output(raw) is False


@pytest.fixture
def service_stub(monkeypatch):
    observations = {"predictions": [], "generations": []}
    result_settings = {"confidence": 0.8, "truncated": False, "label": "refund"}

    def predict(text, bundle):
        observations["predictions"].append((text, bundle))
        return {**result_settings, "model": {"run_name": "test-only"}}

    def call(provider, prompt):
        observations["generations"].append((provider, prompt))
        failed = provider == "openai"
        return {
            "provider": provider, "model": "test-only",
            "text": "" if failed else VALID_TEXT,
            "usage": {"input_tokens": None, "output_tokens": None},
            "latency_seconds": 3.0 if failed else 1.0,
            "output_format_valid": not failed, "generation_complete": not failed,
            "finish_reason": "error" if failed else "stop",
            "error": "TimeoutError" if failed else None,
        }

    monkeypatch.setattr(chain, "predict_text", predict)
    monkeypatch.setattr(chain, "call_model", call)
    return chain.create_chain(object()), observations, result_settings


def test_chain_classifies_once_and_shares_prompt(service_stub):
    runnable, seen, _ = service_stub
    prepared = runnable.invoke("  반품 방법  ")
    result = chain.compare_prepared(prepared)
    assert len(seen["predictions"]) == 1
    assert seen["predictions"][0][0] == "반품 방법"
    assert [provider for provider, _ in seen["generations"]] == ["ollama", "openai"]
    assert seen["generations"][0][1] == seen["generations"][1][1]
    assert set(prepared) == {"text", "classification", "policy", "prompt"}
    assert prepared["policy"] == chain.POLICIES["refund"]
    assert "반품 방법" in prepared["prompt"]
    assert result["classification"]["label"] == "refund"


def test_unknown_policy_fails_before_any_generation(service_stub):
    runnable, seen, settings = service_stub
    settings["label"] = "unknown"
    with pytest.raises(KeyError):
        runnable.invoke("문의")
    assert not seen["generations"]


def test_comparison_preserves_failures_repeats_and_unscored_fields(service_stub, tmp_path):
    runnable, seen, _ = service_stub
    cases = [
        {"id": "one", "text": "반품", "expected_label": "refund"},
        {"id": "two", "text": "취소", "expected_label": "refund"},
    ]
    destination = tmp_path / "comparison"
    summary = comparison.run_comparison(runnable, cases, destination, repeats=2)
    assert len(seen["predictions"]) == 2
    assert len(seen["generations"]) == 8
    assert summary["providers"]["openai"]["calls"] == 4
    assert summary["providers"]["openai"]["errors"] == 4
    assert summary["providers"]["openai"]["mean_latency_seconds_success_only"] is None
    assert summary["providers"]["ollama"]["calls"] == 4
    assert not (destination / "warmup.json").exists()
    assert summary["human_quality_scored"] is False
    records = [
        json.loads(line)
        for line in (destination / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8
    assert [row["provider"] for row in records[:4]] == ["ollama", "openai", "ollama", "openai"]
    for case in cases:
        selected = [row for row in records if row["case_id"] == case["id"]]
        assert len({row["prompt"] for row in selected}) == 1
    with (destination / "human_scores.csv").open(encoding="utf-8-sig", newline="") as file:
        scores = list(csv.DictReader(file))
    assert len(scores) == 8
    assert all(row["policy_correct_0_2"] == "" and row["reviewer"] == "" for row in scores)
    assert all(row["input_tokens"] == "" for row in scores)
    before = {name: list(items) for name, items in seen.items()}
    with pytest.raises(FileExistsError):
        comparison.run_comparison(runnable, cases, destination)
    assert seen == before


@pytest.mark.parametrize("cases", [[], [{"id": "one", "text": " "}], [
    {"id": "same", "text": "문의"}, {"id": "same", "text": "다른 문의"},
]])
def test_bad_comparison_inputs_fail_before_generation(service_stub, tmp_path, cases):
    runnable, seen, _ = service_stub
    with pytest.raises(ValueError):
        comparison.run_comparison(runnable, cases, tmp_path / "bad")
    assert seen == {"predictions": [], "generations": []}


def test_api_loads_once_validates_input_and_keeps_partial_failure(service_stub, monkeypatch):
    _, seen, _ = service_stub
    loads = []

    def loader():
        loads.append(True)
        return object()

    monkeypatch.setattr(api, "load_classifier", loader)
    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert seen["generations"] == []
        assert client.post("/generate", json={"text": "문의", "provider": "other"}).status_code == 422
        assert client.post("/compare", json={"text": "   "}).status_code == 422
        good = client.post("/generate", json={"text": "  문의  "})
        assert good.status_code == 200
        assert seen["predictions"][-1][0] == "문의"
        comparison_result = client.post("/compare", json={"text": "문의"})
        assert comparison_result.status_code == 502
        assert len(comparison_result.json()["results"]) == 2
        assert client.post("/generate", json={"text": "가" * 4001}).status_code == 422
    assert loads == [True]


def test_api_model_loading_failure_stops_server_startup(monkeypatch):
    def failed_loader():
        raise RuntimeError("private-path-do-not-publish")

    monkeypatch.setattr(api, "load_classifier", failed_loader)
    with pytest.raises(RuntimeError), TestClient(api.app):
        pytest.fail("모델 로딩 실패 후에는 요청을 받을 수 없어야 한다.")


def test_api_http_success_does_not_hide_invalid_or_incomplete_output(service_stub, monkeypatch):
    def invalid(provider, prompt):
        return {"error": None, "text": "unfinished", "generation_complete": False,
                "output_format_valid": False}

    monkeypatch.setattr(chain, "call_model", invalid)
    monkeypatch.setattr(api, "load_classifier", lambda: object())
    with TestClient(api.app) as client:
        response = client.post("/generate", json={"text": "문의"})
        assert response.status_code == 200
        assert response.json()["result"]["generation_complete"] is False
        assert response.json()["result"]["output_format_valid"] is False


def test_summary_denominators_keep_failed_and_incomplete_calls():
    records = [
        {"provider": "ollama", "error": None, "latency_seconds": 2.0,
         "generation_complete": True, "output_format_valid": True},
        {"provider": "ollama", "error": None, "latency_seconds": 4.0,
         "generation_complete": False, "output_format_valid": False},
        {"provider": "ollama", "error": "TimeoutError", "latency_seconds": 120.0,
         "generation_complete": False, "output_format_valid": False},
    ]
    summary = comparison.summarize_provider(records, "ollama")
    assert summary["calls"] == 3
    assert summary["errors"] == 1
    assert summary["error_rate_all_calls"] == pytest.approx(1 / 3)
    assert summary["mean_latency_seconds_success_only"] == 3.0
    assert summary["format_valid_rate_all_calls"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("provider,prompt", [("other", "문의"), ("ollama", " ")])
def test_invalid_call_arguments_stop_before_network(monkeypatch, provider, prompt):
    def unexpected_call(*args):
        pytest.fail("잘못된 인자로는 네트워크 호출을 시작하면 안 된다.")

    monkeypatch.setattr(models, "call_ollama", unexpected_call)
    with pytest.raises(ValueError):
        models.call_model(provider, prompt)


def test_separate_verifier_detects_changed_prompt(service_stub, tmp_path):
    runnable, _, _ = service_stub
    destination = tmp_path / "verified"
    comparison.run_comparison(runnable, [{"id": "one", "text": "환불 문의"}], destination)
    report = verify_service.verify_comparison(destination)
    assert report == {
        "cases": 1, "records": 4, "errors": 2,
        "format_valid": 2, "generation_complete": 2, "human_quality_verified": False,
    }
    path = destination / "results.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["prompt"] = "different prompt"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records),
                    encoding="utf-8")
    with pytest.raises(AssertionError, match="prompt"):
        verify_service.verify_comparison(destination)


def test_ollama_load_time_is_kept_apart_from_latency(monkeypatch):
    install_http_transport(monkeypatch, lambda request: httpx.Response(200, json={
        "response": VALID_TEXT, "done": True, "done_reason": "stop",
        "load_duration": 2_500_000_000,
    }))
    assert models.call_model("ollama", "입력")["load_seconds"] == 2.5


def test_summary_totals_recorded_tokens_and_estimates_cost_only_with_prices():
    records = [
        {"provider": "openai", "error": None, "latency_seconds": 1.0,
         "output_format_valid": True, "usage": {"input_tokens": 1000, "output_tokens": 500}},
        {"provider": "openai", "error": "TimeoutError", "latency_seconds": 3.0,
         "output_format_valid": False, "usage": {"input_tokens": None, "output_tokens": None}},
    ]
    summary = comparison.summarize_provider(records, "openai")
    assert summary["usage_known_calls"] == 1
    assert (summary["input_tokens_total"], summary["output_tokens_total"]) == (1000, 500)
    assert summary["max_load_seconds"] is None
    assert comparison.estimate_cost(summary, None, 8.0) is None
    assert comparison.estimate_cost(summary, 2.0, 8.0) == pytest.approx(0.006)


def test_policies_cover_every_label_in_project_data():
    for labels_path in (STEPS.parent / "data").glob("*/labels.json"):
        labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))["labels"]
        assert set(labels) <= set(chain.POLICIES), labels_path.parent.name


def test_rubric_note_is_saved_for_reviewers_but_never_sent_to_models(service_stub, tmp_path):
    runnable, seen, _ = service_stub
    note = "채점 참고: 접수와 승인을 구분하는지 본다"
    destination = tmp_path / "rubric"
    comparison.run_comparison(
        runnable, [{"id": "one", "text": "환불 문의", "rubric_note": note}], destination, repeats=1)
    records = [json.loads(line) for line in
               (destination / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["rubric_note"] for row in records} == {note}
    with (destination / "human_scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert {row["rubric_note"] for row in csv.DictReader(stream)} == {note}
    assert seen["generations"] and all(note not in prompt for _, prompt in seen["generations"])


def test_summary_separates_call_errors_from_incomplete_or_invalid_answers():
    def record(error, complete, valid):
        return {"provider": "ollama", "error": error, "latency_seconds": 1.0,
                "generation_complete": complete, "output_format_valid": valid}

    records = [record(None, True, True), record(None, False, False),  # 정상, 잘린 답변
               record(None, True, False), record("TimeoutError", False, False)]  # 형식 오류, 호출 실패
    summary = comparison.summarize_provider(records, "ollama")
    assert summary["error_rate_all_calls"] == 0.25  # 호출 실패만 센다
    assert summary["complete_rate_all_calls"] == 0.5
    assert summary["usable_rate_all_calls"] == 0.25


def test_api_status_tells_whether_the_answer_is_usable(service_stub, monkeypatch):
    monkeypatch.setattr(api, "load_classifier", lambda: object())
    with TestClient(api.app) as client:
        assert client.post("/generate", json={"text": "문의"}).json()["status"] == "ok"
        compared = client.post("/compare", json={"text": "문의"}).json()
        assert [row["status"] for row in compared["results"]] == ["ok", "generation_error"]
    for complete, valid, expected in ((False, False, "incomplete"), (True, False, "invalid_format")):
        result = {"error": None, "generation_complete": complete, "output_format_valid": valid}
        assert api.answer_status(result) == expected


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_openai_failed_body_is_recorded_as_error_not_success(monkeypatch, status):
    def handler(request):
        return httpx.Response(200, json={
            "id": "resp_failed", "object": "response", "created_at": 1, "status": status,
            "model": models.OPENAI_MODEL, "output": [],
            "error": {"code": "server_error", "message": "private-detail"},
        })

    monkeypatch.setenv("OPENAI_API_KEY", "lesson-test-placeholder")
    monkeypatch.setattr(models, "OpenAI", lambda **kwargs: RealOpenAI(
        api_key="lesson-test-placeholder", base_url="https://lesson.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    result = models.call_model("openai", "입력")
    assert result["error"] == "GenerationFailed"  # HTTP 200이어도 실패로 집계
    assert result["text"] == "" and result["generation_complete"] is False
    assert "private-detail" not in json.dumps(result)
    summary = comparison.summarize_provider([result], "openai")
    assert summary["errors"] == 1 and summary["mean_latency_seconds_success_only"] is None
