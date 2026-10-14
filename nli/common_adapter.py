"""기존 NLI 분류기·프롬프트·파서를 공통 파이프라인에 연결한다."""

import argparse
import json
import sys
from pathlib import Path

# 파일을 직접 실행해도 프로젝트 루트의 common_pipeline을 찾는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from common_pipeline import compare
from explainer import (
    POLICIES,
    PROMPT,
    ROLE,
    SYSTEM,
    make_client,
    parse_output as existing_parse_output,
)
from predictor import Checker, LABEL_KO


def validate_input(input_data):
    """모델 호출 전에 입력 필드·자료형·길이를 검사한다."""
    if set(input_data) != {"evidence", "claim"}:
        raise ValueError("evidence와 claim 필드만 필요합니다.")

    for name in ("evidence", "claim"):
        value = input_data[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")
        if len(value) > config.MAX_INPUT_CHARS:
            raise ValueError(f"{name}의 최대 길이를 초과했습니다.")


def make_predict(checker):
    """미리 로딩한 모델로 근거·주장 한 쌍을 실제 추론한다."""
    checkpoint = (
        config.LEARNER_DIR / checker.run_name / "checkpoint"
    ).relative_to(ROOT).as_posix()

    def predict(input_data):
        p = checker.predict_pairs(
            [input_data["evidence"]],
            [input_data["claim"]],
        )[0]

        label = checker.labels[int(p.argmax())]
        return {
            "label": label,
            "probabilities": {
                name: float(value)
                for name, value in zip(checker.labels, p)
            },
            # 실제 로딩한 체크포인트 경로를 모델 식별자로 사용한다.
            "model_id": checkpoint,
            "run_id": checker.run_name,
        }

    return predict


def build_prompt(input_data, context):
    """기존 메시지 구조와 예측 라벨의 정책을 유지한다."""
    if context["classification"] is None:
        raise ValueError("현재 연결은 분류기 + LLM만 지원합니다.")

    label = context["label"]
    probabilities = context["classification"]["probabilities"]
    if probabilities is None:
        raise ValueError("확신도에 사용할 실제 확률이 필요합니다.")

    messages = PROMPT.format_messages(
        label_ko=LABEL_KO[label],
        confidence=round(probabilities[label], 3),  # 기존 Checker.check와 같은 세 자리 표시
        policy=context["policies"][label],
        evidence=input_data["evidence"],
        claim=input_data["claim"],
    )
    return json.dumps(
        [
            {"role": ROLE[message.type], "content": message.content}
            for message in messages
        ],
        ensure_ascii=False,
    )


def make_generate(provider):
    """기존 클라이언트 설정을 재사용하고 호출 정보를 반환한다."""
    base_client, model = make_client(provider)

    # 자동 재시도를 끄므로 실제 요청 시도 수를 1로 기록할 수 있다.
    client = base_client.with_options(max_retries=0)

    def generate(prompt):
        if provider == "ollama":
            options = {
                "max_tokens": config.MAX_OUTPUT_TOKENS,
                "temperature": 0,
            }
        else:
            options = {
                "max_completion_tokens": config.MAX_OUTPUT_TOKENS,
            }

        response = client.chat.completions.create(
            model=model,
            messages=json.loads(prompt),
            **options,
        )
        choice = response.choices[0]
        usage = response.usage

        return {
            "raw_text": choice.message.content or "",
            "model_id": response.model or model,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "generation_complete": choice.finish_reason == "stop",
            "request_attempts": 1,
        }

    return generate


def parse_output(raw_text, context):
    """기존 파서를 재사용하되 실패 응답은 성공으로 넘기지 않는다."""
    parsed = existing_parse_output({"text": raw_text})

    if not parsed["parse_ok"]:
        raise ValueError("설명 JSON을 해석할 수 없습니다.")

    explanation = parsed["explanation"]
    suggestion = parsed["suggestion"]

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation은 비어 있지 않은 문자열이어야 합니다.")
    if suggestion is not None and not isinstance(suggestion, str):
        raise ValueError("suggestion은 문자열 또는 None이어야 합니다.")

    return {
        "generated_text": explanation,
        "service_output": {
            "explanation": explanation,
            "suggestion": suggestion,
        },
    }


SERVICE = {
    "id": "nli_report_checker",
    "policy_version": "v1",
    "label_definitions": {
        "entailment": "원문 근거가 보고서의 주장을 뒷받침한다.",
        "neutral": "원문 근거만으로 보고서의 주장을 확인할 수 없다.",
        "contradiction": "보고서의 주장이 원문 근거와 충돌한다.",
    },
    "common_policy": SYSTEM,
    "policies": POLICIES,
    "validate_input": validate_input,
    "build_prompt": build_prompt,
    "parse_output": parse_output,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["ollama", "openai"], default="ollama"
    )
    args = parser.parse_args()

    input_data = {
        "evidence": "출시일은 11월 10일로 확정되었다.",
        "claim": "출시일은 11월 20일로 확정되었다.",
    }

    # 로딩 시간은 공통 파이프라인의 추론 시간에서 제외한다.
    checker = Checker()
    generate = make_generate(args.provider)

    result = compare(
        input_data,
        service=SERVICE,
        classifiers={"selected_classifier": make_predict(checker)},
        llms={args.provider: generate},
        combinations=[("selected_classifier", args.provider)],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()