"""NLI 연결 계층의 입력 검증·파서·미완료 처리를 확인한다."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nli"))

from common_adapter import SERVICE
from common_pipeline import compare


VALID_INPUT = {
    "evidence": "출시일은 11월 10일이다.",
    "claim": "출시일은 11월 20일이다.",
}


def run_case(input_data, raw_text, complete=True):
    calls = {"classifier": 0, "llm": 0}

    def predict(data):
        calls["classifier"] += 1
        return {
            "label": "contradiction",
            "probabilities": {
                "entailment": 0.05,
                "neutral": 0.05,
                "contradiction": 0.9,
            },
            "model_id": "mock-classifier",
            "run_id": "mock-run",
        }

    def generate(prompt):
        calls["llm"] += 1
        return {
            "raw_text": raw_text,
            "model_id": "mock-llm",
            "input_tokens": None,
            "output_tokens": None,
            "generation_complete": complete,
            "request_attempts": 1,
        }

    result = compare(
        input_data,
        service=SERVICE,
        classifiers={"mock": predict},
        llms={"mock": generate},
        combinations=[("mock", "mock")],
    )
    return result["results"][0], calls


def test_valid_output_preserves_original():
    raw = (
        '{"explanation":"원문과 날짜가 다릅니다.",'
        '"suggestion":"출시일은 11월 10일이다."}'
    )
    result, calls = run_case(VALID_INPUT, raw)

    assert result["status"] == "success"
    assert result["generated_text"] == "원문과 날짜가 다릅니다."
    assert result["service_output"]["suggestion"] == "출시일은 11월 10일이다."
    assert result["generation"]["raw_text"] == raw
    assert calls == {"classifier": 1, "llm": 1}


@pytest.mark.parametrize(
    "input_data",
    [
        {"evidence": "", "claim": "주장"},
        {"evidence": "근거", "claim": "   "},
        {"source": "근거", "claim": "주장"},
        {"evidence": 123, "claim": "주장"},
    ],
)
def test_invalid_input_never_calls_models(input_data):
    def must_not_run(value):
        pytest.fail("잘못된 입력인데 모델이 호출되었습니다.")

    with pytest.raises((ValueError, TypeError)):
        compare(
            input_data,
            service=SERVICE,
            classifiers={"mock": must_not_run},
            llms={"mock": must_not_run},
            combinations=[("mock", "mock")],
        )


@pytest.mark.parametrize(
    "raw",
    [
        "JSON이 아닌 응답입니다.",
        '{"suggestion":"수정안만 있습니다."}',
        '{"explanation":"","suggestion":null}',
        '{"explanation":123,"suggestion":null}',
        '{"explanation":"설명","suggestion":123}',
    ],
)
def test_invalid_output_is_parse_failure(raw):
    result, _ = run_case(VALID_INPUT, raw)

    assert result["status"] == "error"
    assert result["error"]["stage"] == "parse_output"
    assert result["generated_text"] is None
    assert result["service_output"] is None
    assert result["generation"]["raw_text"] == raw


def test_incomplete_output_is_not_parsed():
    # JSON이 정상이어도 생성이 미완료라면 성공 처리하면 안 된다.
    raw = '{"explanation":"설명입니다.","suggestion":null}'
    result, _ = run_case(VALID_INPUT, raw, complete=False)

    assert result["status"] == "incomplete"
    assert result["error"]["stage"] == "generation"
    assert result["generated_text"] is None
    assert result["service_output"] is None
    assert result["generation"]["raw_text"] == raw