"""실제 모델·네트워크 없이 공통 계약, 실패 격리, 정책과 측정 범위를 검증한다."""

import json
from copy import deepcopy

import pytest

from common_pipeline import compare


@pytest.fixture
def service():
    return {
        "id": "comment", "policy_version": "v1",
        "label_definitions": {"abusive": "욕설이나 모욕", "normal": "일반 댓글"},
        "common_policy": "입력에 없는 사실을 추가하지 마세요.",
        "policies": {"abusive": "정중하게 고쳐 쓰세요.", "normal": "원문 뜻을 유지하세요."},
    }


def generation(label="abusive", **overrides):
    return {
        "raw_text": json.dumps({"label": label, "generated_text": "정중한 문장입니다."}),
        "model_id": "mock-llm", "input_tokens": None, "output_tokens": None,
        "generation_complete": True, "request_attempts": 1,
        **overrides,
    }


def run(service, *, input_data=None, classifiers=None, llms=None, combinations=None):
    return compare(
        {"comment": "원본 댓글"} if input_data is None else input_data,
        service=service,
        classifiers={"mock": lambda data: {"label": "abusive"}}
        if classifiers is None else classifiers,
        llms={"mock": lambda prompt: generation()} if llms is None else llms,
        combinations=[("mock", "mock")] if combinations is None else combinations,
    )


def default_context(prompt):
    # 기본 프롬프트의 JSON 부분을 읽어 실제 전달된 입력과 정책을 확인한다.
    return json.loads(prompt.split("\n", 1)[1])


def test_one_field_classifier_and_optional_defaults(service):
    seen = []

    def predict(data):
        seen.append(data)
        return {"label": "abusive"}

    result = run(service, classifiers={"mock": predict})
    row = result["results"][0]
    assert seen == [{"comment": "원본 댓글"}]
    assert row["success"] is True and row["status"] == "success"
    assert row["classification"] == {
        "label": "abusive", "probabilities": None, "model_id": None, "run_id": None,
    }
    assert row["label"] == "abusive"
    assert row["generated_text"] == "정중한 문장입니다."
    assert result["service_id"] == "comment" and result["policy_version"] == "v1"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_two_field_input_preserves_names_and_structure(service):
    seen = []
    data = {"evidence": "회의는 금요일에 진행된다.", "claim": "회의는 월요일이다."}
    service.update(
        id="report",
        label_definitions={"entailment": "근거가 주장을 지지", "contradiction": "근거와 주장 모순"},
        policies={"entailment": "지지 근거를 설명하세요.", "contradiction": "모순 근거를 설명하세요."},
    )

    def predict(item):
        assert item == data
        return {"label": "contradiction"}

    def generate(prompt):
        seen.append(default_context(prompt))
        return generation("contradiction")

    run(service, input_data=data, classifiers={"mock": predict}, llms={"mock": generate})
    assert seen[0]["input_data"] == data
    assert set(seen[0]["input_data"]) == {"evidence", "claim"}


def test_llm_only_gets_all_criteria_and_policies_and_calls_once(service):
    seen = []

    def generate(prompt):
        seen.append(default_context(prompt))
        return generation("normal")

    row = run(
        service, classifiers={}, llms={"mock": generate}, combinations=[(None, "mock")]
    )["results"][0]
    assert len(seen) == 1
    assert seen[0]["label_definitions"] == service["label_definitions"]
    assert seen[0]["policies"] == service["policies"]
    assert seen[0]["common_policy"] == service["common_policy"]
    assert seen[0]["label"] is None
    assert row["mode"] == "llm_only" and row["label"] == "normal"
    assert row["classification"] is None
    assert row["timings"]["classification_seconds"] is None
    assert row["llm_invocations"] == 1 and row["success"] is True


@pytest.mark.parametrize("label", ["abusive", "normal"])
def test_classifier_selects_common_and_only_predicted_policy(service, label):
    seen = []

    def generate(prompt):
        seen.append(default_context(prompt))
        # 분류기 경로의 LLM 출력에는 label이 없어도 된다.
        return generation(raw_text=json.dumps({"generated_text": "결과"}))

    row = run(
        service, classifiers={"mock": lambda data: {"label": label}},
        llms={"mock": generate},
    )["results"][0]
    assert row["success"] is True and row["label"] == label
    assert seen[0]["common_policy"] == service["common_policy"]
    assert seen[0]["policies"] == {label: service["policies"][label]}
    assert seen[0]["label_definitions"] == {label: service["label_definitions"][label]}


def test_only_requested_combinations_execute_sequentially(service):
    calls = []

    def predict(data):
        calls.append("predict")
        return {"label": "abusive"}

    def generate_a(prompt):
        calls.append("a")
        return generation()

    def generate_b(prompt):
        calls.append("b")
        return generation()

    def unrequested(*args):
        pytest.fail("명시하지 않은 함수가 호출됨")

    result = run(
        service, classifiers={"mock": predict, "unused": unrequested},
        llms={"a": generate_a, "b": generate_b, "unused": unrequested},
        combinations=[(None, "b"), ("mock", "a"), ("mock", "b")],
    )
    assert calls == ["b", "predict", "a", "predict", "b"]
    assert [(row["classifier"], row["llm"]) for row in result["results"]] == [
        (None, "b"), ("mock", "a"), ("mock", "b"),
    ]
    assert all(row["success"] for row in result["results"])


@pytest.mark.parametrize("label", ["unknown", "", 3, None])
def test_bad_classifier_label_prevents_llm_call(service, label):
    row = run(
        service, classifiers={"mock": lambda data: {"label": label}},
        llms={"mock": lambda prompt: pytest.fail("잘못된 라벨로 LLM 호출")},
    )["results"][0]
    assert row["status"] == "error" and row["success"] is False
    assert row["error"]["stage"] == "classification_contract"
    assert row["llm_invocations"] == 0 and row["timings"]["llm_seconds"] is None


@pytest.mark.parametrize("raw", [
    "not-json", '```json\n{"generated_text":"결과"}\n```', "[]",
    '{"generated_text":"   "}', '{"generated_text":42}',
    '{"label":"abusive"}', '{"generated_text":"결과","extra":true}',
    '{"generated_text":"첫째","generated_text":"둘째"}',
])
def test_parsing_failure_keeps_raw_response(service, raw):
    row = run(service, llms={"mock": lambda prompt: generation(raw_text=raw)})["results"][0]
    assert row["status"] == "error" and row["error"]["stage"] == "parse"
    assert row["generation"]["raw_text"] == raw
    assert row["generated_text"] is None and row["label"] == "abusive"


@pytest.mark.parametrize("raw", [
    '{"generated_text":"결과"}', '{"label":"unknown","generated_text":"결과"}',
])
def test_llm_only_requires_known_label(service, raw):
    row = run(
        service, classifiers={}, combinations=[(None, "mock")],
        llms={"mock": lambda prompt: generation(raw_text=raw)},
    )["results"][0]
    assert row["success"] is False and row["label"] is None
    assert row["error"]["stage"] in ("parse", "label")


def test_llm_cannot_overwrite_classifier_label(service):
    row = run(service, llms={"mock": lambda prompt: generation("normal")})["results"][0]
    assert row["label"] == "abusive"
    assert row["success"] is False and row["error"]["stage"] == "label"


def test_one_failed_combination_does_not_stop_the_next(service):
    def fail(prompt):
        raise TimeoutError("private-request-and-api-key")

    rows = run(
        service, llms={"bad": fail, "good": lambda prompt: generation()},
        combinations=[("mock", "bad"), (None, "good")],
    )["results"]
    assert [row["status"] for row in rows] == ["error", "success"]
    assert rows[0]["error"]["type"] == "TimeoutError"
    assert rows[0]["error"]["stage"] == "llm"
    assert rows[0]["generation"]["request_attempts"] is None  # 실제 요청 횟수는 알 수 없다.
    assert rows[0]["timings"]["llm_seconds"] >= 0
    assert "private-request-and-api-key" not in json.dumps(rows)


def test_classifier_exception_does_not_stop_llm_only(service):
    def fail(data):
        raise RuntimeError("모델 실패")

    rows = run(
        service, classifiers={"mock": fail},
        combinations=[("mock", "mock"), (None, "mock")],
    )["results"]
    assert rows[0]["error"]["stage"] == "classification"
    assert rows[0]["timings"]["classification_seconds"] >= 0
    assert rows[0]["timings"]["llm_seconds"] is None
    assert rows[1]["success"] is True


def test_incomplete_generation_remains_incomplete_even_with_valid_json(service):
    rows = run(
        service,
        llms={"short": lambda prompt: generation(generation_complete=False),
              "good": lambda prompt: generation()},
        combinations=[(None, "short"), (None, "good")],
    )["results"]
    assert rows[0]["status"] == "incomplete" and rows[0]["success"] is False
    assert rows[0]["error"]["stage"] == "generation"
    assert rows[0]["generation"]["raw_text"] and rows[0]["generated_text"] is None
    assert rows[1]["success"] is True


def test_measured_times_and_unknown_usage_are_not_invented(service, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("common_pipeline.pipeline.time.perf_counter", lambda: clock[0])

    def predict(data):
        clock[0] += 0.25
        return {"label": "abusive"}

    def generate(prompt):
        clock[0] += 0.75
        return generation(request_attempts=2, input_tokens=0)

    rows = run(
        service, classifiers={"mock": predict}, llms={"mock": generate},
        combinations=[("mock", "mock"), (None, "mock")],
    )["results"]
    assert rows[0]["timings"] == {
        "classification_seconds": 0.25, "llm_seconds": 0.75, "total_seconds": 1.0,
    }
    assert rows[1]["timings"] == {
        "classification_seconds": None, "llm_seconds": 0.75, "total_seconds": 0.75,
    }
    assert rows[0]["generation"]["input_tokens"] == 0
    assert rows[0]["generation"]["output_tokens"] is None
    assert rows[0]["generation"]["request_attempts"] == 2
    assert rows[0]["llm_invocations"] == 1  # 재시도는 어댑터 보고값; 엔진 호출은 한 번


def test_input_and_service_are_not_mutated_or_shared_between_combinations(service):
    original = {"comment": "원본", "metadata": {"tags": ["원본"]}}
    before_input, before_service = deepcopy(original), deepcopy(service)
    seen = []

    def validate(data):
        data["metadata"]["tags"].append("검증 중 변경")

    def predict(data):
        assert data == before_input
        data["metadata"]["tags"].append("분류 중 변경")
        return {"label": "abusive"}

    def builder(data, context):
        assert data == before_input
        seen.append(deepcopy(context))
        data["metadata"]["tags"].append("프롬프트 중 변경")
        context["policies"].clear()
        return "사용자 정의 프롬프트"

    service.update(validate_input=validate, build_prompt=builder)
    result = run(
        service, input_data=original, classifiers={"mock": predict},
        combinations=[("mock", "mock"), (None, "mock")],
    )
    assert original == before_input and result["input_data"] == before_input
    assert service["policies"] == before_service["policies"]
    assert seen[0]["policies"] == {"abusive": before_service["policies"]["abusive"]}
    assert seen[1]["policies"] == before_service["policies"]
    assert all(row["success"] for row in result["results"])


def test_custom_builder_is_used_and_runtime_failure_is_isolated(service):
    def builder(data, context):
        if context["mode"] == "classifier_llm":
            raise RuntimeError("프롬프트 실패")
        return f"질문: {data['comment']}"

    seen = []
    service["build_prompt"] = builder

    def generate(prompt):
        seen.append(prompt)
        return generation()

    rows = run(
        service, llms={"mock": generate},
        combinations=[("mock", "mock"), (None, "mock")],
    )["results"]
    assert rows[0]["error"]["stage"] == "prompt"
    assert rows[0]["llm_invocations"] == 0
    assert seen == ["질문: 원본 댓글"] and rows[1]["success"] is True


def test_service_input_validation_happens_before_callbacks(service):
    def validate(data):
        if not isinstance(data.get("comment"), str):
            raise TypeError("comment 필수")

    service["validate_input"] = validate
    with pytest.raises(ValueError, match="입력 검증"):
        run(service, input_data={"claim": "다른 필드"},
            llms={"mock": lambda prompt: pytest.fail("잘못된 입력으로 호출")})


@pytest.mark.parametrize("data", [
    "문자열", {1: "정수 키"}, {"tuple": (1, 2)}, {"nan": float("nan")},
    {"infinity": float("inf")}, {"nested": [{"object": object()}]},
])
def test_invalid_input_is_rejected_before_callbacks(service, data):
    with pytest.raises((TypeError, ValueError), match="JSON 호환"):
        run(service, input_data=data,
            llms={"mock": lambda prompt: pytest.fail("잘못된 입력으로 호출")})


@pytest.mark.parametrize("change", [
    {"id": ""}, {"policy_version": None}, {"common_policy": None},
    {"label_definitions": {}}, {"policies": {"abusive": "하나만 존재"}},
    {"policies": {"abusive": "", "normal": "일반"}}, {"build_prompt": "함수 아님"},
    {"validate_input": "함수 아님"},
])
def test_service_configuration_errors_are_detected_before_calls(service, change):
    service.update(change)
    with pytest.raises((TypeError, ValueError)):
        run(service, llms={"mock": lambda prompt: pytest.fail("설정 오류 중 호출")})


@pytest.mark.parametrize("combinations", [
    [], [(None, "missing")], [("missing", "mock")], [(None, "mock"), ("missing", "mock")],
    [[None, "mock"]], [(None,)], [(None, "")], [(42, "mock")],
])
def test_all_combinations_are_preflighted_before_any_call(service, combinations):
    with pytest.raises((TypeError, ValueError)):
        run(service, combinations=combinations,
            llms={"mock": lambda prompt: pytest.fail("전체 조합 검사 전 호출")})


def test_probabilities_and_model_identity_are_preserved(service):
    prediction = {
        "label": "abusive", "probabilities": {"abusive": 0.7, "normal": 0.3},
        "model_id": "mock-roberta", "run_id": "experiment-1",
    }
    row = run(service, classifiers={"mock": lambda data: prediction})["results"][0]
    assert row["classification"] == prediction
    assert row["classification"]["probabilities"] is not prediction["probabilities"]


@pytest.mark.parametrize("probabilities", [
    {"abusive": 0.7}, {"abusive": 0.8, "normal": 0.8},
    {"abusive": float("nan"), "normal": 0.3}, {"abusive": True, "normal": False},
])
def test_invalid_probabilities_are_rejected(service, probabilities):
    row = run(service, classifiers={"mock": lambda data: {
        "label": "abusive", "probabilities": probabilities,
    }})["results"][0]
    assert row["error"]["stage"] == "classification_contract"
    assert row["llm_invocations"] == 0


@pytest.mark.parametrize("change", [
    {"model_id": None}, {"input_tokens": -1}, {"output_tokens": True},
    {"generation_complete": "true"}, {"request_attempts": 0}, {"request_attempts": True},
])
def test_llm_metadata_contract_is_checked(service, change):
    row = run(service, llms={"mock": lambda prompt: generation(**change)})["results"][0]
    assert row["success"] is False and row["error"]["stage"] == "llm_contract"
    assert row["generation"]["raw_text"]


def test_missing_metadata_is_not_filled_with_fake_values(service):
    row = run(service, llms={"mock": lambda prompt: {
        "raw_text": '{"generated_text":"결과"}',
    }})["results"][0]
    assert row["error"]["stage"] == "llm_contract"
    assert row["generation"]["input_tokens"] is None
    assert row["generation"]["request_attempts"] is None


def test_success_is_structural_not_a_quality_score(service):
    row = run(service, llms={"mock": lambda prompt: generation(
        raw_text='{"generated_text":"품질을 평가하지 않은 문장"}'
    )})["results"][0]
    assert row["success"] is True
    assert "quality_score" not in row
