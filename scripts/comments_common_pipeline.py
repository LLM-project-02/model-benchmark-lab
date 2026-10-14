"""악성댓글 서비스(comments)를 공통 파이프라인(common_pipeline.compare)에 연결한다.

엔진에 넘기는 함수는 모두 서비스의 기존 코드를 감싼 것이다. 새 파서나 새 프롬프트를 만들지 않는다.
- 분류: step06 predict_text (selected.json의 KcELECTRA, CPU). 세 라벨의 확률을 반올림 전 값으로 넘긴다.
- 프롬프트: step08 apply_policy가 만드는 서비스 프롬프트 그대로 (FastAPI 서비스와 같은 문자열)
- LLM: step07 call_model. ollama = 로컬 gemma4:e4b, openai = gpt-6-luna
- 파서: step07 AnswerPayload (answer, next_steps). 형식이 틀리면 예외가 나고 엔진이 parse_output 실패로 기록한다.

LLM 단독(분류기 없이 LLM이 라벨까지 고르기)은 지원하지 않는다. 서비스 프롬프트가 라벨별 정책을 전제로 하고
AnswerPayload에 label 필드가 없어서다. 그 조합은 프롬프트 단계에서 실패로 기록된다(docs/COMMON_PIPELINE.md).
서비스의 사람 검토 검사(확신도, 다시 검사, 뜻 검사)는 FastAPI(step10)에만 있고 이 연결에는 넣지 않았다.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # common_pipeline
sys.path.insert(0, str(ROOT / "steps"))

import step07_call_models as models
from step06_predict import predict_text
from step07_call_models import AnswerPayload
from step08_connect_chain import COMMENT_RULES, MAX_INPUT_CHARS, POLICIES, apply_policy

SERVICE_ID = "comments"
# step08의 정책과 프롬프트를 고정한 개발 비교의 이름. 정책을 바꾸면 함께 바꾼다.
POLICY_VERSION = "bench01-development03"
# 라벨 기준은 원본 데이터의 주석 지침을 요약한 data/comments/README.md에서 옮겼다.
LABEL_DEFINITIONS = {
    "none": "혐오나 모욕이 없는 댓글",
    "offensive": "혐오나 모욕 수준은 아니지만 대상이나 읽는 사람을 불쾌하게 하는 댓글. "
    "무례하거나 공격적인 말, 비꼼, 비윤리적 표현, 확인되지 않은 소문",
    "hate": "개인이나 집단을 향한 강한 혐오 또는 모욕. 성별, 인종, 국적, 정치 성향, 종교, 나이, "
    "외모, 직업 등에 대한 적대적 표현, 평판을 심하게 해치는 모욕, 성희롱",
}
INPUT_FIELDS = {"text"}  # 모델이 볼 입력 필드. 정답 라벨 같은 평가용 필드는 넣지 않는다
REQUEST_ATTEMPTS = 1  # call_model은 다시 시도하지 않는다(OpenAI max_retries=0, Ollama 요청 한 번)


def validate_input(input_data):
    """FastAPI 서비스(step10 TextRequest)와 같은 기준으로 입력을 검사한다. 정상이면 None."""
    if set(input_data) != INPUT_FIELDS:
        raise ValueError("입력은 text 필드 하나만 가져야 합니다.")
    text = input_data["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text는 공백이 아닌 문자열이어야 합니다.")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"text는 {MAX_INPUT_CHARS}자 이하여야 합니다.")


def build_prompt(input_data, context):
    """서비스 프롬프트를 그대로 만든다. 분류 라벨이 없으면(LLM 단독) 정책을 고를 수 없어 실패한다."""
    if context["label"] is None:
        raise ValueError("댓글 서비스 프롬프트는 분류 라벨이 있어야 만들 수 있습니다.")
    item = {"text": input_data["text"].strip(), "classification": {"label": context["label"]}}
    return apply_policy(item)["prompt"]


def parse_output(raw_text, context):
    """기존 응답 검사 모델(AnswerPayload)로 해석한다. 형식이 틀리면 ValidationError가 난다."""
    data = AnswerPayload.model_validate_json(raw_text).model_dump()
    return {"generated_text": data["answer"], "service_output": data}


def build_service():
    return {
        "id": SERVICE_ID,
        "policy_version": POLICY_VERSION,
        "label_definitions": LABEL_DEFINITIONS,
        "common_policy": COMMENT_RULES,
        "policies": {label: POLICIES[label] for label in LABEL_DEFINITIONS},
        "validate_input": validate_input,
        "build_prompt": build_prompt,
        "parse_output": parse_output,
    }


def make_classifier(bundle):
    """미리 불러온 분류 모델(step06 load_classifier)을 엔진의 분류 함수로 감싼다."""
    identity = bundle["identity"]
    model_id = f"{identity['learner']}/{identity['run_name']}"  # 미세조정한 체크포인트

    def predict(input_data):
        verdict = predict_text(input_data["text"], bundle)
        return {
            "label": verdict["label"],
            "probabilities": verdict["probabilities"],
            "model_id": model_id,
            "run_id": identity["run_id"],
        }

    return predict


def make_llm(provider, call=None):
    """step07 call_model을 엔진의 LLM 함수로 감싼다. 호출이 실패하면 예외를 낸다(엔진이 llm 단계로 기록)."""
    call = call or models.call_model

    def generate(prompt):
        result = call(provider, prompt)
        if result["error"]:
            raise RuntimeError(f"{provider} 생성 호출 실패: {result['error']}")
        return {
            "raw_text": result["text"],  # 원본 응답 그대로
            "model_id": result["model"],
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
            "generation_complete": result["generation_complete"],
            "request_attempts": REQUEST_ATTEMPTS,
        }

    return generate


def summary_line(result):
    """결과 항목 하나를 사람이 읽을 한 줄로 줄인다."""
    if result["success"]:
        answer = result["generated_text"]
        return f"{result['status']} · {result['label']} · {json.dumps(answer, ensure_ascii=False)}"
    error = result["error"]
    return f"{result['status']} · {error['stage']} · {error['type']}"
