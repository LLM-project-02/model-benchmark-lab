"""보고서 연구용 세 경로 비교(nli/research_pipeline.py, research_api.py, run_research_compare.py)의 계약 테스트.

실제 모델과 LLM은 부르지 않는다(tests/conftest.py가 막는다). 분류기 예측과 생성 호출은 가짜 함수로 바꾸고,
공통 엔진·서비스 지시·프롬프트 조립·파서는 실제 코드를 쓴다. 로지스틱 회귀 적재는 임시 소형 분류기로 확인한다.
"""

import csv
import io
import json
import socket
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nli"))

import common_adapter
import explainer
import research_api
import research_pipeline as research
import run_research_compare as runner

EVIDENCE = "출시일은 11월 10일로 확정되었다."
CLAIM = "출시일은 11월 20일로 확정되었다."
LLM = research.DEFAULT_LLM
PROBS = {
    "logreg": {"entailment": 0.2, "neutral": 0.25, "contradiction": 0.55},
    "roberta": {"entailment": 0.05, "neutral": 0.1, "contradiction": 0.85},
}


def payload(label, explanation="원문과 날짜가 다르다.", suggestion="출시일은 11월 10일로 확정되었다."):
    return json.dumps({"label": label, "explanation": explanation, "suggestion": suggestion}, ensure_ascii=False)


class FakeRegistry(research.ClassifierRegistry):
    """가짜 분류기를 쓰는 레지스트리. 적재 호출과 예측 호출을 센다."""

    def __init__(self, available=("logreg", "roberta"), probs=PROBS, labels=research.LABELS, broken=()):
        self.loads, self.predictions = [], []

        def loader_for(name):
            def load(run_dir):
                self.loads.append(name)
                if name not in available:
                    raise research.ModelNotReady(f"필요한 모델 파일이 없습니다: {run_dir}")
                if name in broken:
                    raise ValueError("적재 실패")

                def predict(evidence, claim):
                    self.predictions.append((name, evidence, claim))
                    return dict(probs[name])

                return {"labels": list(labels), "predict": predict,
                        "identity": {"model_id": f"fake/{name}", "run_id": f"{name}-run", "kind": name,
                                     "temperature": None}}

            return load

        super().__init__(
            runs={"logreg": Path("/fake/logreg"), "roberta": Path("/fake/roberta")},
            loaders={name: loader_for(name) for name in ("logreg", "roberta")},
        )
        self.available = set(available)

    def info(self, name):  # 파일 검사 대신 지정한 사용 가능 여부를 쓴다
        info = super().info(name)
        if name not in self.bundles and name not in self.failures:
            ok = name in self.available
            info.update(state="not_loaded" if ok else "model_not_ready", available=ok,
                        unavailable_code=None if ok else "model_not_ready",
                        error=None if ok else "필요한 모델 파일이 없습니다")
        return info


def fake_call(replies, prompts=None, complete=True, error=None, tokens=(120, 30)):
    """경로별로 정해진 응답을 돌려주는 가짜 생성 호출. 프롬프트로 경로를 가린다."""

    def call(llm_key, prompt):
        if prompts is not None:
            prompts.append(prompt)
        user = json.loads(prompt)[1]["content"]
        if "[라벨 정의와 라벨별 지시]" in user:
            key = "llm_only"
        elif "확신도 0.55" in user:
            key = "logreg"
        else:
            key = "roberta"
        return {
            "text": replies[key], "model": research.REGISTERED_LLMS[llm_key]["model"],
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            "generation_complete": complete, "finish_reason": "stop" if complete else "length",
            "provider": "ollama", "latency_seconds": 0.5, "error": error, "error_detail": None,
        }

    return call


REPLIES = {key: payload("contradiction") for key in ("llm_only", "logreg", "roberta")}


def by_path(record):
    return {item["path"]: item for item in record["paths"]}


def test_three_paths_run_through_common_pipeline_with_same_llm():
    registry, prompts = FakeRegistry(), []
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, registry, call=fake_call(REPLIES, prompts))
    paths = by_path(record)
    assert record["schema"] == research.SCHEMA and record["executed"] == list(research.PATH_IDS)
    assert {p["status"] for p in record["paths"]} == {"success"}
    assert {p["result"]["llm"] for p in record["paths"]} == {LLM}
    assert record["input"] == {"evidence": EVIDENCE, "claim": CLAIM}
    assert record["pipeline"]["service_id"] == research.SERVICE_ID
    assert record["generation_config"]["retries"] == 0 and record["generation_config"]["fallback"] is None
    # 분류기 경로는 실제 예측 확률과 모델 식별자를 그대로 남긴다
    roberta = paths["roberta"]
    assert roberta["classifier_verdict"]["probabilities"] == PROBS["roberta"]
    assert roberta["classifier_verdict"]["confidence"] == 0.85
    assert roberta["result"]["classification"]["model_id"] == "fake/roberta"
    assert roberta["result"]["classification"]["run_id"] == "roberta-run"
    assert roberta["result"]["service_output"]["suggestion"] == "출시일은 11월 10일로 확정되었다."
    assert roberta["result"]["generation"]["input_tokens"] == 120
    assert roberta["result"]["generation"]["request_attempts"] == 1
    assert roberta["prompt"] in prompts and roberta["llm_call"]["finish_reason"] == "stop"
    assert sorted(registry.loads) == ["logreg", "roberta"]  # 경로마다 한 번씩만 불러온다


def test_llm_only_needs_no_classifier_and_never_calls_one():
    registry = FakeRegistry(available=())  # 체크포인트가 하나도 없다
    record = research.run_comparison(
        EVIDENCE, CLAIM, LLM, registry, selected=["llm_only"], call=fake_call(REPLIES)
    )
    (item,) = record["paths"]
    assert item["status"] == "success" and item["classifier_called"] is False
    assert item["classifier_verdict"] is None and item["result"]["classification"] is None
    assert registry.loads == [] and registry.predictions == []
    assert record["classifiers"] == {}


def test_only_selected_paths_load_models_and_are_recorded():
    registry = FakeRegistry()
    record = research.run_comparison(
        EVIDENCE, CLAIM, LLM, registry, selected=["roberta"], call=fake_call(REPLIES)
    )
    assert [p["path"] for p in record["paths"]] == ["roberta"] and record["executed"] == ["roberta"]
    assert registry.loads == ["roberta"]  # 로지스틱 회귀는 불러오지 않는다
    assert [name for name, *_ in registry.predictions] == ["roberta", "roberta"]  # 워밍업 1 + 실제 1
    assert record["classifier_calls"] == {"roberta": 1}


def test_order_controls_execution_and_results_map_back():
    record = research.run_comparison(
        EVIDENCE, CLAIM, LLM, FakeRegistry(), order=["roberta", "llm_only", "logreg"], call=fake_call(REPLIES)
    )
    paths = by_path(record)
    assert record["executed"] == ["roberta", "llm_only", "logreg"]
    assert [paths[p]["executed_order"] for p in ("roberta", "llm_only", "logreg")] == [1, 2, 3]
    assert paths["logreg"]["classifier_verdict"]["probabilities"] == PROBS["logreg"]
    with pytest.raises(ValueError):
        research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(), order=["llm_only"], call=fake_call(REPLIES))


def test_llm_only_prompt_has_all_definitions_and_no_fake_verdict():
    prompts = []
    research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=["llm_only"],
                            call=fake_call(REPLIES, prompts))
    system, user = (message["content"] for message in json.loads(prompts[0]))
    for label in research.LABELS:
        assert common_adapter.SERVICE["label_definitions"][label] in user
        assert explainer.POLICIES[label] in user
    assert "확신도" not in user and "[판정]" not in user and "[판정 라벨]" not in user
    assert "분류 모델이 내린" not in system and research.LLM_ONLY_RULE_2 in system
    assert research.RESEARCH_FORMAT in system and research.SERVICE_FORMAT not in system
    assert f"[원문 근거] {EVIDENCE}" in user and f"[보고서 문장] {CLAIM}" in user


def test_classifier_prompt_uses_predicted_policy_and_real_probability():
    prompts = []
    research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=["logreg"],
                            call=fake_call(REPLIES, prompts))
    system, user = (message["content"] for message in json.loads(prompts[0]))
    assert "[판정] 모순됨 (확신도 0.55)" in user and "[판정 라벨] contradiction" in user
    assert explainer.POLICIES["contradiction"] in user
    assert explainer.POLICIES["entailment"] not in user and explainer.POLICIES["neutral"] not in user
    assert research.SERVICE_RULE_2 in system  # 분류기 경로는 서비스 규칙을 그대로 쓴다
    # 서비스 프롬프트와 출력 형식 줄·판정 라벨 줄만 다르다
    service_prompt = common_adapter.build_prompt(
        {"evidence": EVIDENCE, "claim": CLAIM},
        {"label": "contradiction", "policies": {"contradiction": explainer.POLICIES["contradiction"]},
         "classification": {"probabilities": PROBS["logreg"]}},
    )
    service_system, service_user = (message["content"] for message in json.loads(service_prompt))
    assert system == service_system.replace(research.SERVICE_FORMAT, research.RESEARCH_FORMAT)
    assert user.replace("[판정 라벨] contradiction (label에는 이 값을 그대로 적는다)\n", "") == service_user


def test_label_mismatch_is_recorded_without_correction():
    replies = dict(REPLIES, roberta=payload("neutral", suggestion="출처를 알려 주실 수 있나요?"))
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=["roberta"],
                                     call=fake_call(replies))
    (item,) = record["paths"]
    assert item["status"] == "error" and item["result"]["error"]["stage"] == "label"
    assert item["raw_label"] == "neutral" and item["applied_policy_label"] == "contradiction"
    assert item["result"]["service_output"] is None
    assert item["result"]["generation"]["raw_text"] == replies["roberta"]  # 원본 응답 보존


@pytest.mark.parametrize("raw", [
    '{"explanation": "설명", "suggestion": null}',  # label 없음
    '{"label": "contradiction", "explanation": "설명", "suggestion": null, "extra": 1}',  # 추가 필드
    '{"label": "모순", "explanation": "설명", "suggestion": null}',  # 등록되지 않은 라벨
    '{"label": "contradiction", "explanation": "  ", "suggestion": null}',  # 빈 설명
    '{"label": "contradiction", "explanation": "설명"}',  # suggestion 없음
    "JSON이 아닌 응답",
])
def test_parse_failures_are_errors_with_raw_text_kept(raw):
    for path in research.PATH_IDS:
        record = research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=[path],
                                         call=fake_call({key: raw for key in research.PATH_IDS}))
        (item,) = record["paths"]
        assert item["status"] == "error" and item["result"]["error"]["stage"] == "parse_output"
        assert item["result"]["generation"]["raw_text"] == raw
        assert item["result"]["service_output"] is None


def test_wrapped_json_and_null_text_follow_service_parser():
    raw = '```json\n{"label": "neutral", "explanation": "확인할 수 없다.", "suggestion": "null"}\n```'
    parsed = research.parse_output(raw, {})
    assert parsed["service_output"] == {"label": "neutral", "explanation": "확인할 수 없다.", "suggestion": None}


def test_incomplete_generation_and_llm_failure_are_kept_as_states():
    incomplete = research.run_comparison(
        EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=["llm_only"],
        call=fake_call({"llm_only": '{"label": "contradic'}, complete=False),
    )
    (item,) = incomplete["paths"]
    assert item["status"] == "incomplete" and item["result"]["error"]["stage"] == "generation"
    assert item["result"]["generation"]["raw_text"] == '{"label": "contradic'
    assert item["llm_call"]["finish_reason"] == "length"

    failed = research.run_comparison(
        EVIDENCE, CLAIM, LLM, FakeRegistry(), selected=["roberta"],
        call=fake_call(REPLIES, error="APIConnectionError"),
    )
    (item,) = failed["paths"]
    assert item["status"] == "error" and item["result"]["error"]["stage"] == "llm"
    assert item["llm_calls"] == 1 and item["llm_call"]["error"] == "APIConnectionError"
    assert item["classifier_verdict"]["probabilities"] == PROBS["roberta"]  # 분류는 실제로 했다


def test_missing_models_are_unavailable_and_not_replaced():
    registry = FakeRegistry(available=("roberta",))
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, registry, call=fake_call(REPLIES))
    paths = by_path(record)
    assert paths["logreg"]["status"] == "unavailable"
    assert paths["logreg"]["unavailable_code"] == "model_not_ready"
    assert paths["logreg"]["result"] is None and paths["logreg"]["llm_calls"] == 0
    assert record["executed"] == ["llm_only", "roberta"]
    summary = research.summarize([record])
    assert summary["logreg"]["excluded"] is True and summary["logreg"]["llm_seconds"] is None
    assert summary["logreg"]["status_counts"] is None and summary["roberta"]["runs"] == 1


def test_all_selected_paths_unavailable_skips_engine_without_llm_call():
    calls = []
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(available=()),
                                     selected=["logreg", "roberta"],
                                     call=lambda *args: calls.append(args))
    assert calls == [] and record["executed"] == [] and record["pipeline"] is None
    assert {p["status"] for p in record["paths"]} == {"unavailable"}


def test_load_failure_and_label_order_are_load_failed():
    broken = FakeRegistry(broken=("logreg",))
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, broken, selected=["logreg"], call=fake_call(REPLIES))
    assert record["paths"][0]["unavailable_code"] == "load_failed"
    reordered = FakeRegistry(labels=("contradiction", "neutral", "entailment"))
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, reordered, selected=["roberta"], call=fake_call(REPLIES))
    assert record["paths"][0]["unavailable_code"] == "load_failed"
    assert "라벨 순서" in record["paths"][0]["unavailable_reason"]


def test_real_registry_reports_model_not_ready_from_missing_files(tmp_path):
    registry = research.ClassifierRegistry(runs={"logreg": tmp_path / "baseline", "roberta": tmp_path / "rob"})
    infos = registry.infos()
    assert {info["unavailable_code"] for info in infos.values()} == {"model_not_ready"}
    assert "baseline.joblib" in infos["logreg"]["error"] and "model.safetensors" in infos["roberta"]["error"]
    assert research.available_path_ids(registry) == ["llm_only"]
    assert registry.get("roberta") is None  # 파일 확인에서 멈추고 from_pretrained를 부르지 않는다
    assert registry.info("roberta")["unavailable_code"] == "model_not_ready"


def train_tiny_baseline(run_dir):
    """train_baseline.py와 같은 입력·라벨 번호·저장 형식으로 만든 임시 소형 분류기 (실제 모델 아님)."""
    import joblib
    from features import PairTfidf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    labels = list(research.LABELS)
    pairs = [
        ("회의는 10일이다.", "회의는 10일이다.", 0), ("예산은 3억이다.", "예산은 3억이다.", 0),
        ("회의는 10일이다.", "참석자는 5명이다.", 1), ("예산은 3억이다.", "담당자는 김 팀장이다.", 1),
        ("회의는 10일이다.", "회의는 20일이다.", 2), ("예산은 3억이다.", "예산은 5억이다.", 2),
    ]
    model = make_pipeline(PairTfidf(ngram_range=(1, 2), max_features=200),
                          LogisticRegression(max_iter=200, random_state=0))
    model.fit([(p, h) for p, h, _ in pairs], [y for *_, y in pairs])
    run_dir.mkdir(parents=True)
    joblib.dump(model, run_dir / "baseline.joblib")
    (run_dir / "config.json").write_text(json.dumps({"kind": "baseline", "dataset": "klue_nli", "labels": labels}),
                                         encoding="utf-8")
    return model


def test_logreg_loader_follows_baseline_format_with_tiny_model(tmp_path):
    model = train_tiny_baseline(tmp_path / "baseline")
    bundle = research.load_logreg(tmp_path / "baseline")
    probabilities = bundle["predict"]("예산은 3억이다.", "예산은 5억이다.")
    expected = model.predict_proba([("예산은 3억이다.", "예산은 5억이다.")])[0]
    assert list(probabilities) == list(research.LABELS)
    assert probabilities == pytest.approx(dict(zip(research.LABELS, expected)))
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert bundle["identity"]["kind"] == "baseline" and bundle["identity"]["run_id"] == "baseline"
    # 레지스트리·엔진을 거쳐도 같은 확률로 판정한다
    registry = research.ClassifierRegistry(runs={"logreg": tmp_path / "baseline", "roberta": tmp_path / "none"})
    record = research.run_comparison("예산은 3억이다.", "예산은 5억이다.", LLM, registry, selected=["logreg"],
                                     call=fake_call(REPLIES))
    verdict = record["paths"][0]["classifier_verdict"]
    assert verdict["probabilities"] == pytest.approx(dict(zip(research.LABELS, expected)))


def test_logreg_loader_rejects_comment_model_config(tmp_path):
    train_tiny_baseline(tmp_path / "baseline")
    (tmp_path / "baseline" / "config.json").write_text(
        json.dumps({"kind": "baseline", "dataset": "comments", "labels": ["none", "offensive", "hate"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        research.load_logreg(tmp_path / "baseline")


def test_roberta_loader_uses_selected_checker_and_temperature(tmp_path, monkeypatch):
    run_dir = tmp_path / "roberta_lr2e5_ep6"
    for path in research.required_files("roberta", run_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    seen = {}

    class FakeChecker:  # predictor.Checker와 같은 속성·메서드
        def __init__(self, run_name):
            seen["run_name"] = run_name
            self.labels, self.temperature, self.device = list(research.LABELS), 2.77, "cpu"

        def predict_pairs(self, premises, hypotheses):
            seen["pairs"] = (premises, hypotheses)
            return [[0.1, 0.2, 0.7]]

    import predictor

    monkeypatch.setattr(predictor, "Checker", FakeChecker)
    bundle = research.load_roberta(run_dir)
    assert bundle["predict"](EVIDENCE, CLAIM) == pytest.approx(
        {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7})
    assert seen["pairs"] == ([EVIDENCE], [CLAIM]) and seen["run_name"] == str(run_dir)
    assert bundle["identity"]["temperature"] == 2.77


def test_default_runs_point_to_selected_models():
    assert research.LABELS == ("entailment", "neutral", "contradiction")
    import predictor

    assert research.PATHS[2]["classifier"] == "roberta" and predictor.SELECTED_RUN == "roberta_lr2e5_ep6"


def make_client(registry=None, call=None):
    app = research_api.create_app(registry_factory=lambda: registry or FakeRegistry(available=()),
                                  call=call or fake_call(REPLIES), preload=False)
    return TestClient(app)


def test_api_starts_without_models_and_reports_not_ready(tmp_path):
    registry = research.ClassifierRegistry(runs={"logreg": tmp_path / "a", "roberta": tmp_path / "b"})
    with make_client(registry) as client:
        health = client.get("/research/health")
        assert health.status_code == 200 and health.json()["service"] == "nli"
        options = client.get("/research/options").json()
        assert options["available_paths"] == ["llm_only"]
        assert options["input_fields"] == ["evidence", "claim"]
        assert {c["unavailable_code"] for c in options["classifiers"].values()} == {"model_not_ready"}
        assert options["generation_config"][LLM]["retries"] == 0
        response = client.post("/research/compare", json={"evidence": EVIDENCE, "claim": CLAIM})
        assert response.status_code == 200
        statuses = {p["path"]: p["status"] for p in response.json()["paths"]}
        assert statuses == {"llm_only": "success", "logreg": "unavailable", "roberta": "unavailable"}


def test_api_compare_paths_and_validation():
    registry = FakeRegistry()
    with make_client(registry) as client:
        body = client.post("/research/compare", json={"evidence": EVIDENCE, "claim": CLAIM,
                                                      "paths": ["llm_only"]}).json()
        assert [p["path"] for p in body["paths"]] == ["llm_only"] and registry.loads == []
        assert client.post("/research/compare", json={"evidence": " ", "claim": CLAIM}).status_code == 422
        assert client.post("/research/compare", json={"evidence": EVIDENCE}).status_code == 422
        assert client.post("/research/compare", json={"evidence": EVIDENCE, "claim": CLAIM,
                                                      "paths": ["kcelectra"]}).status_code == 400
        assert client.post("/research/compare", json={"evidence": EVIDENCE, "claim": CLAIM,
                                                      "llm": "gpt"}).status_code == 400
        assert client.post("/research/compare", json={"evidence": EVIDENCE, "claim": CLAIM,
                                                      "order": ["llm_only"]}).status_code == 400
        too_long = "가" * (research_api.MAX_INPUT_CHARS + 1)
        assert client.post("/research/compare", json={"evidence": too_long, "claim": CLAIM}).status_code == 422


def test_csv_rows_keep_raw_response_and_blank_human_evaluation():
    registry = FakeRegistry(available=("roberta",))
    replies = dict(REPLIES, llm_only="깨진 응답")
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, registry, call=fake_call(replies))
    record["example"] = {"id": "pair-01", "reference_label": "contradiction", "source_text": "원문 전체"}
    rows = list(csv.DictReader(io.StringIO(research.to_csv([record]))))
    assert list(rows[0]) == research.CSV_COLUMNS
    paths = {row["path"]: row for row in rows}
    assert paths["llm_only"]["status"] == "error" and paths["llm_only"]["raw_text"] == "깨진 응답"
    assert paths["llm_only"]["has_output"] == "no" and paths["llm_only"]["explanation"] == ""
    assert paths["logreg"]["status"] == "unavailable" and paths["logreg"]["classifier_confidence"] == ""
    assert paths["roberta"]["has_output"] == "yes"
    assert json.loads(paths["roberta"]["classifier_probabilities"]) == PROBS["roberta"]
    assert paths["roberta"]["suggestion"] == "출시일은 11월 10일로 확정되었다."
    for row in rows:
        assert all(row[column] == "" for column in research.EVAL_COLUMNS)
        assert row["evidence"] == EVIDENCE and row["source_text"] == "원문 전체"


def test_runner_rotates_order_and_keeps_reference_out_of_prompts(tmp_path):
    examples = tmp_path / "pairs.jsonl"
    rows = [
        {"id": f"pair-{i}", "evidence": f"근거 문장 {i}입니다.", "claim": f"주장 문장 {i}입니다.",
         "reference_label": "neutral", "source_text": f"참고용원문표시{i}"}
        for i in range(3)
    ]
    examples.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    prompts = []
    out = runner.run("t01", paths=research.PATH_IDS, examples_file=examples, output_root=tmp_path / "out",
                     registry=FakeRegistry(available=("roberta",)), call=fake_call(REPLIES, prompts),
                     log=lambda *_: None)
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    orders = [record["executed"] for record in saved["records"]]
    assert orders == [["llm_only", "roberta"], ["roberta", "llm_only"], ["llm_only", "roberta"]]
    assert saved["summary"]["logreg"]["excluded"] is True and saved["warmup"]["paths"]
    assert all("참고용원문표시" not in prompt and "neutral" not in json.loads(prompt)[1]["content"].split("[원문 근거]")[1]
               for prompt in prompts)
    with open(out / "results.csv", encoding="utf-8-sig", newline="") as file:
        assert len(list(csv.DictReader(file))) == 9  # 3 예제 × 3 경로
    with pytest.raises(SystemExit):
        runner.run("t01", examples_file=examples, output_root=tmp_path / "out", registry=FakeRegistry(),
                   call=fake_call(REPLIES), log=lambda *_: None)


def test_rotated_order_keeps_unavailable_last():
    assert runner.rotated_order(1, ["llm_only", "roberta"]) == ["roberta", "llm_only", "logreg"]
    assert runner.rotated_order(2) == ["roberta", "llm_only", "logreg"]


def test_service_adapter_and_api_are_unchanged():
    # 연구 기능은 서비스 설정을 바꾸지 않는다: 서비스 프롬프트는 여전히 LLM 단독을 거부한다.
    with pytest.raises(ValueError):
        common_adapter.build_prompt({"evidence": EVIDENCE, "claim": CLAIM}, {"classification": None})
    assert common_adapter.SERVICE["id"] == "nli_report_checker" and common_adapter.SERVICE["policy_version"] == "v1"
    service = research.build_service()
    assert service["policies"] == explainer.POLICIES and service["id"] != common_adapter.SERVICE["id"]


@pytest.mark.allow_blocked_attempts
def test_real_calls_are_blocked(block_real_calls):
    # 기본 생성 호출(실제 Ollama)을 쓰면 막히고 기록된다
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, FakeRegistry(available=()), selected=["llm_only"])
    assert record["paths"][0]["status"] == "error"
    with pytest.raises(RuntimeError):
        socket.create_connection(("127.0.0.1", 11434))
    assert "research_pipeline.call_llm" in block_real_calls and "socket.create_connection" in block_real_calls


def test_research_pair_files_are_reproducible_and_separate(tmp_path):
    import build_research_pairs as builder

    builder.write(tmp_path)
    for name in ("dev_pairs.jsonl", "final_eval_candidates.jsonl"):
        committed = (ROOT / "data" / "nli_research" / name).read_text(encoding="utf-8")
        assert (tmp_path / name).read_text(encoding="utf-8") == committed, name
    dev = research.load_examples(ROOT / "data" / "nli_research" / "dev_pairs.jsonl")
    final = [json.loads(line) for line in (ROOT / "data" / "nli_research" / "final_eval_candidates.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert len({row["id"] for row in final}) == len(final) and len({row["id"] for row in dev}) == len(dev)
    pairs = lambda rows: {(row["evidence"], row["claim"]) for row in rows}
    assert not pairs(dev) & pairs(final)  # 개발 예제와 최종 후보는 겹치지 않는다
    klue = [row for row in final if row["candidate_set"] == "klue_test_sample"]
    assert len({row["origin"]["group_id"] for row in klue}) == len(klue) == 30
    assert all(row["origin"]["file"] == "data/klue_nli/test.csv" for row in klue)
    assert all("사용함" in row["prior_usage"] for row in final)  # 사용 이력을 미사용으로 쓰지 않는다
    assert {row["reference_label"] for row in final} <= set(research.LABELS)


def test_examples_endpoint_keeps_reference_out_of_inputs():
    with make_client() as client:
        examples = client.get("/research/examples").json()["examples"]
    assert examples and set(examples[0]) == {"id", "evidence", "claim", "reference_label", "source_text"}


def test_eval_sheet_blinds_paths_and_leaves_no_output_rows_blank():
    import build_research_eval_sheet as sheet_builder

    registry = FakeRegistry(available=("roberta",))
    record = research.run_comparison(EVIDENCE, CLAIM, LLM, registry,
                                     call=fake_call(dict(REPLIES, llm_only="깨진 응답")))
    record["example"] = {"id": "pair-01", "reference_label": "contradiction", "source_text": "원문 전체"}
    sheet, key = sheet_builder.build({"records": [record]}, blind=True, seed=3)
    assert {row["output_code"] for row in sheet} == {"A", "B", "C"}
    assert all(row["output_code"] not in research.PATH_IDS for row in sheet)
    assert "path" not in sheet_builder.SHEET_COLUMNS and "reference_label" not in sheet_builder.SHEET_COLUMNS
    by_code = {row["output_code"]: row for row in sheet}
    paths = {row["output_code"]: row["path"] for row in key}
    for code, row in by_code.items():
        assert row["evidence"] == EVIDENCE and row["claim"] == CLAIM and row["source_text"] == "원문 전체"
        assert all(row[column] == "" for column in research.EVAL_COLUMNS)
        if paths[code] == "roberta":
            assert row["eval_target"] == "yes" and row["output_label"] == "contradiction"
        else:
            assert row["eval_target"].startswith("no") and row["explanation"] == ""
    assert {row["reference_label"] for row in key} == {"contradiction"}


def test_runner_candidate_set_filters_final_candidates(tmp_path):
    path = ROOT / "data" / "nli_research" / "final_eval_candidates.jsonl"
    assert len(research.load_examples(path, "klue_test_sample")) == 30
    with pytest.raises(ValueError):
        research.load_examples(path, "없는 묶음")
    out = runner.run("final01", paths=["llm_only"], examples_file=path, candidate_set="stress2_eval2", limit=2,
                     output_root=tmp_path, registry=FakeRegistry(available=()), call=fake_call(REPLIES),
                     log=lambda *_: None)
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["candidate_set"] == "stress2_eval2" and len(saved["records"]) == 2
    assert saved["records"][0]["example"]["id"].startswith("stress2-")


def test_summarize_eval_counts_yes_no_and_rejects_edited_content(tmp_path):
    import build_research_eval_sheet as sheet_builder
    import summarize_research_eval as summarizer

    examples = tmp_path / "pairs.jsonl"
    examples.write_text("\n".join(json.dumps(
        {"id": f"p{i}", "evidence": EVIDENCE, "claim": f"{CLAIM} {i}", "reference_label": "contradiction"},
        ensure_ascii=False) for i in range(2)), encoding="utf-8")
    replies = dict(REPLIES, llm_only="깨진 응답")  # LLM 단독은 출력 없음
    out = runner.run("s01", examples_file=examples, output_root=tmp_path, registry=FakeRegistry(),
                     call=fake_call(replies), log=lambda *_: None)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    sheet, key = sheet_builder.build(results, blind=True, seed=1)
    def as_rows(rows):  # CSV로 읽은 것처럼 문자열로
        return [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]

    original, key = as_rows(sheet), as_rows(key)
    paths = {row["sheet_row"]: row["path"] for row in key}
    filled = []
    for row in original:
        row = dict(row)
        if paths[row["sheet_row"]] == "roberta":
            row.update(eval_evidence_faithful="yes", eval_new_facts="no")
        elif paths[row["sheet_row"]] == "logreg":
            row.update(eval_evidence_faithful="no", eval_new_facts="yes", eval_suggestion_valid="maybe")
        else:
            row.update(eval_evidence_faithful="yes")  # 출력이 없는 행의 값은 집계하지 않는다
        filled.append(row)
    summary = summarizer.summarize(original, filled, key)
    roberta, logreg, llm_only = (summary["per_path"][p] for p in ("roberta", "logreg", "llm_only"))
    assert roberta["criteria"]["eval_evidence_faithful"] == {**roberta["criteria"]["eval_evidence_faithful"],
                                                             "yes": 2, "no": 0, "n": 2, "rate": 1.0}
    assert logreg["criteria"]["eval_new_facts"]["yes"] == 2 and logreg["criteria"]["eval_suggestion_valid"]["n"] == 0
    assert llm_only["with_output"] == 0 and llm_only["criteria"]["eval_evidence_faithful"]["n"] == 0
    assert len(summary["issues"]["values_without_output"]) == 2 and len(summary["issues"]["invalid_values"]) == 2
    assert roberta["reference_label_agreement"] == {**roberta["reference_label_agreement"], "n": 2, "agree": 2}
    # LLM 단독이 출력을 못 낸 예제는 세 경로 공통 비교에서 빠진다
    assert summary["common"]["eval_evidence_faithful"]["example_ids"] == []
    edited = [dict(row) for row in filled]
    edited[0]["explanation"] = "고친 설명"
    with pytest.raises(ValueError):
        summarizer.summarize(original, edited, key)
