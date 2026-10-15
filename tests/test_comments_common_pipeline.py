"""악성댓글 서비스를 공통 파이프라인에 연결한 코드(scripts/comments_common_pipeline.py)의 테스트.

실제 모델은 부르지 않는다. 분류 함수와 생성 호출은 가짜로 바꾸고, 프롬프트와 파서는 서비스 코드를 그대로 쓴다.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comments_common_pipeline as connect
import step08_connect_chain as chain

from common_pipeline import compare

ANSWER = {"answer": "연기가 아쉬워서 보다가 껐어요.", "next_steps": []}
PROBABILITIES = {"none": 0.1, "offensive": 0.7, "hate": 0.2}


def fake_predict(input_data):
    return {
        "label": "offensive",
        "probabilities": PROBABILITIES,
        "model_id": "bench01/test-run",
        "run_id": "run-test",
    }


def fake_call(calls, text=None, error=None, complete=True):
    """step07 call_model과 같은 모양을 돌려주는 가짜 생성 호출."""

    def call(provider, prompt):
        calls.append((provider, prompt))
        return {
            "text": json.dumps(ANSWER, ensure_ascii=False) if text is None else text,
            "model": f"{provider}-test",
            "usage": {"input_tokens": 12, "output_tokens": 7},
            "generation_complete": complete,
            "error": error,
        }

    return call


def run(llm, text=" 연기를 왜 이렇게 못하냐 ", combinations=None):
    return compare(
        {"text": text},
        service=connect.build_service(),
        classifiers={"kcelectra": fake_predict},
        llms={"ollama": llm},
        combinations=combinations or [("kcelectra", "ollama")],
    )


def test_service_prompt_and_existing_parser_are_used():
    calls = []
    output = run(connect.make_llm("ollama", call=fake_call(calls)))
    result = output["results"][0]
    assert output["schema_version"] == 2
    assert result["status"] == "success"
    assert result["label"] == "offensive"
    assert result["classification"]["probabilities"] == PROBABILITIES
    assert result["generated_text"] == ANSWER["answer"]
    assert result["service_output"] == ANSWER
    assert result["generation"]["request_attempts"] == 1
    assert result["generation"]["input_tokens"] == 12
    # FastAPI 서비스와 같은 프롬프트(step08 apply_policy)를 같은 공백 처리로 보낸다
    expected = chain.apply_policy(
        {"text": "연기를 왜 이렇게 못하냐", "classification": {"label": "offensive"}}
    )["prompt"]
    assert calls == [("ollama", expected)]


@pytest.mark.parametrize(
    "raw_text",
    [
        "```json\n" + json.dumps(ANSWER, ensure_ascii=False) + "\n```",  # 코드 블록
        json.dumps({**ANSWER, "label": "offensive"}, ensure_ascii=False),  # 허용하지 않은 필드
        json.dumps({"answer": "  ", "next_steps": []}),  # 빈 답
    ],
)
def test_existing_parser_rejects_bad_answers_and_raw_text_is_kept(raw_text):
    result = run(connect.make_llm("ollama", call=fake_call([], text=raw_text)))["results"][0]
    assert result["status"] == "error"
    assert result["error"]["stage"] == "parse_output"
    assert result["generated_text"] is None
    assert result["generation"]["raw_text"] == raw_text


def test_llm_only_is_not_supported_and_does_not_call_the_model():
    calls = []
    output = run(connect.make_llm("ollama", call=fake_call(calls)), combinations=[(None, "ollama")])
    result = output["results"][0]
    assert result["error"]["stage"] == "prompt"
    assert result["llm_invocations"] == 0
    assert calls == []


def test_generation_failure_and_incomplete_answer_are_recorded():
    failed = run(connect.make_llm("ollama", call=fake_call([], error="ConnectError")))
    assert failed["results"][0]["error"]["stage"] == "llm"
    assert failed["results"][0]["error"]["type"] == "RuntimeError"
    cut = run(connect.make_llm("ollama", call=fake_call([], complete=False)))["results"][0]
    assert cut["status"] == "incomplete"
    assert cut["error"]["stage"] == "generation"
    assert cut["service_output"] is None


@pytest.mark.parametrize(
    "data",
    [{"text": "   "}, {"text": "가" * (chain.MAX_INPUT_CHARS + 1)}, {"comment": "다른 필드"}],
)
def test_bad_input_fails_before_any_model_call(data):
    calls = []
    with pytest.raises(ValueError):
        compare(
            data,
            service=connect.build_service(),
            classifiers={"kcelectra": fake_predict},
            llms={"ollama": connect.make_llm("ollama", call=fake_call(calls))},
            combinations=[("kcelectra", "ollama")],
        )
    assert calls == []


def test_classifier_wrapper_passes_real_probabilities_and_checkpoint_identity(monkeypatch):
    seen = []

    def predict_text(text, bundle):
        seen.append(text)
        return {
            "label": "none",
            "confidence": 0.6,
            "probabilities": {"none": 0.6, "offensive": 0.3, "hate": 0.1},
            "truncated": False,
        }

    monkeypatch.setattr(connect, "predict_text", predict_text)
    bundle = {
        "identity": {
            "learner": "bench01",
            "run_name": "kcelectra-base_lr2e5_s44",
            "run_id": "0fdf19bd",
        }
    }
    predict = connect.make_classifier(bundle)
    assert predict({"text": "댓글"}) == {
        "label": "none",
        "probabilities": {"none": 0.6, "offensive": 0.3, "hate": 0.1},
        "model_id": "bench01/kcelectra-base_lr2e5_s44",
        "run_id": "0fdf19bd",
    }
    assert seen == ["댓글"]
