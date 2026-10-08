"""09. 같은 입력과 프롬프트로 두 모델을 호출하고 비교 기록을 저장한다.

개발 입력(generation_development.jsonl)으로 코드와 프롬프트를 확인한 뒤 최종 비교 시
INPUT_PATH를 변경한다. 최종 생성 평가 데이터의 답변을 보며 프롬프트를 고치지 않는다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step09_compare.py
"""

import csv
import json
import statistics
from pathlib import Path

from lesson_settings import (
    DATA_DIR,
    DEVELOPMENT_INPUT,
    LEARNER_DIR,
    OPENAI_INPUT_USD_PER_1M,
    OPENAI_OUTPUT_USD_PER_1M,
)
from step06_predict import load_classifier
from step07_call_models import generation_config
from step08_connect_chain import MAX_INPUT_CHARS, create_chain, generate_prepared

# 최종 비교를 시작할 때 INPUT_PATH = FINAL_INPUT_PATH로 바꾼다.
FINAL_INPUT_PATH = DATA_DIR / "generation_eval.jsonl"
INPUT_PATH = DEVELOPMENT_INPUT
REPEATS = 2  # 같은 입력을 반복 호출해 응답이 얼마나 달라지는지 확인
OUTPUT_NAME = "development01"  # 실행마다 새 이름 (기존 결과 보호)
OUTPUT_DIR = LEARNER_DIR / "comparisons" / OUTPUT_NAME


def read_cases(path):
    cases = [
        json.loads(line)  # JSONL: 한 줄에 JSON 객체 하나
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    validate_cases(cases)
    return cases


def validate_cases(cases):
    # 빈 입력이나 중복 id를 발견하면 유료 호출 전에 멈춘다.
    if not cases:
        raise ValueError("비교 입력이 비어 있습니다.")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not str(value).strip() for value in ids):
        raise ValueError("입력 id는 내용이 있어야 하고 서로 달라야 합니다.")
    for case in cases:
        text = case["text"]
        if not text.strip() or len(text) > MAX_INPUT_CHARS:
            raise ValueError("비교 입력의 text와 길이를 확인하세요.")


def summarize_provider(records, provider):
    subset = [record for record in records if record["provider"] == provider]
    success = [record for record in subset if not record["error"]]
    calls = len(subset)
    errors = calls - len(success)
    valid = sum(record["output_format_valid"] for record in subset)  # True를 1로 세어 합산
    complete = sum(bool(record.get("generation_complete")) for record in subset)
    # 그대로 쓸 수 있는 답변: 호출 예외가 없고, 끝까지 생성됐고, 약속한 JSON 형식을 지킴
    usable = sum(not record["error"] and bool(record.get("generation_complete"))
                 and record["output_format_valid"] for record in subset)
    # 토큰 수는 응답에 기록된 호출만 더한다. 몇 번의 호출이 기록됐는지 함께 남긴다.
    usage = [record.get("usage") or {} for record in subset]
    known = [item for item in usage
             if item.get("input_tokens") is not None and item.get("output_tokens") is not None]
    loads = [record["load_seconds"] for record in subset if record.get("load_seconds") is not None]
    return {
        "calls": calls,
        "errors": errors,
        "error_rate_all_calls": errors / calls if calls else None,
        "mean_latency_seconds_success_only": statistics.mean(
            record["latency_seconds"] for record in success
        ) if success else None,
        "format_valid_rate_all_calls": valid / calls if calls else None,
        "complete_rate_all_calls": complete / calls if calls else None,
        "usable_rate_all_calls": usable / calls if calls else None,
        "usage_known_calls": len(known),
        "input_tokens_total": sum(item["input_tokens"] for item in known) if known else None,
        "output_tokens_total": sum(item["output_tokens"] for item in known) if known else None,
        # 로컬 모델을 메모리에 올린 시간의 최댓값. 첫 호출 지연을 따로 설명할 때 쓴다.
        "max_load_seconds": max(loads) if loads else None,
    }


def estimate_cost(provider_summary, input_usd_per_1m, output_usd_per_1m):
    """기록된 토큰 수와 단가로 계산한 추정 비용(USD). 단가나 토큰 기록이 없으면 None."""
    if None in (input_usd_per_1m, output_usd_per_1m, provider_summary["input_tokens_total"]):
        return None
    return (provider_summary["input_tokens_total"] * input_usd_per_1m
            + provider_summary["output_tokens_total"] * output_usd_per_1m) / 1_000_000


def write_human_scores(path, records):
    fields = [
        "case_id", "repeat", "provider", "model",
        "predicted_label", "expected_label", "latency_seconds", "input_tokens",
        "output_tokens", "output_format_valid", "generation_complete", "finish_reason",
        "error", "text", "response", "required_points", "forbidden_claims", "rubric_note",
        "reviewer", "policy_correct_0_2", "required_points_0_2",
        "no_unsupported_claims_0_2", "clarity_0_2", "notes",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row.update(
                predicted_label=record["classification"]["label"],
                text=record["input_text"], response=record["response_text"],
                input_tokens=record["usage"]["input_tokens"],
                output_tokens=record["usage"]["output_tokens"],
                required_points=json.dumps(record["required_points"], ensure_ascii=False),
                forbidden_claims=json.dumps(record["forbidden_claims"], ensure_ascii=False),
            )
            # 점수는 빈칸으로 남긴다. 사람이 원문과 답변을 비교한 뒤 채운다.
            writer.writerow(row)


def run_comparison(chain, cases, output_dir, repeats=REPEATS):
    validate_cases(cases)
    if repeats < 1:
        raise ValueError("REPEATS는 1 이상이어야 합니다.")

    # 폴더가 이미 있으면 멈춘다. 앞선 결과를 덮어쓰지 않는다.
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("출력 폴더가 있습니다. OUTPUT_NAME을 새 이름으로 바꾸세요.")
    # 분류와 프롬프트는 입력마다 한 번만 만들어 모든 반복·제공자가 공유한다.
    prepared = [chain.invoke(case["text"]) for case in cases]
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case_index, (case, fixed) in enumerate(zip(cases, prepared, strict=True)):
            for repeat in range(repeats):
                # 준비한 프롬프트를 매번 그대로 사용한다. 예열 호출은 하지 않는다.
                for provider in ("ollama", "openai"):
                    generated = generate_prepared(fixed, provider)
                    record = {
                        "case_id": case["id"], "repeat": repeat + 1,
                        "input_text": fixed["text"],
                        "expected_label": case.get("expected_label"),
                        "classification": fixed["classification"],
                        "policy": fixed["policy"], "prompt": fixed["prompt"],
                        # 채점 기준은 사람이 답변을 평가할 때만 쓴다. 프롬프트에는 넣지 않는다.
                        "required_points": case.get("required_points", []),
                        "forbidden_claims": case.get("forbidden_claims", []),
                        "rubric_note": case.get("rubric_note", ""),
                        **generated, "response_text": generated["text"],
                    }
                    # 예외가 생긴 호출도 저장한다. 중단되더라도 끝난 호출 기록은 남긴다.
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()  # 바로 파일에 기록
                    records.append(record)
                print(f"입력 {case_index + 1}/{len(cases)}, 반복 {repeat + 1} 저장")

    write_human_scores(output_dir / "human_scores.csv", records)
    providers = {
        provider: summarize_provider(records, provider) for provider in ("ollama", "openai")
    }
    summary = {
        "cases": len(cases), "repeats": repeats,
        "generation_config": generation_config(),
        "human_quality_scored": False,  # 사람 채점은 아직 하지 않음
        "providers": providers,
        "metric_notes": {
            "error_rate_all_calls": "호출 자체가 실패(예외)한 비율. 답변 품질은 포함하지 않는다.",
            "complete_rate_all_calls": "토큰 한도 등으로 잘리지 않고 끝까지 생성된 비율",
            "usable_rate_all_calls": "예외 없음 + 끝까지 생성 + JSON 형식 준수를 모두 만족한 비율",
        },
        # 비용은 실측이 아닌 추정값이다. 로컬 모델은 API 요금이 없고 장비·전력 비용은 계산하지 않는다.
        "estimated_cost_usd": {
            "ollama": None,
            "openai": estimate_cost(
                providers["openai"], OPENAI_INPUT_USD_PER_1M, OPENAI_OUTPUT_USD_PER_1M
            ),
        },
        "cost_basis": {
            "openai_input_usd_per_1m": OPENAI_INPUT_USD_PER_1M,
            "openai_output_usd_per_1m": OPENAI_OUTPUT_USD_PER_1M,
            "note": "기록된 토큰 수 x .env 단가. 단가가 없으면 계산하지 않는다.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    cases = read_cases(INPUT_PATH)
    classifier = load_classifier()
    chain = create_chain(classifier)
    summary = run_comparison(chain, cases, OUTPUT_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("저장 위치:", OUTPUT_DIR)
    print("human_scores.csv에 원문과 답변을 비교한 점수와 이유를 적으세요.")


if __name__ == "__main__":
    main()
