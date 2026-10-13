"""자유주제(악성댓글) 순화에 쓸 로컬 생성 모델을 개발 입력으로 비교한다.

uv run python scripts/compare_comments_local_models.py local_models01

설치된 Ollama 모델마다 step08에서 동결한 프롬프트로 개발 입력 15건을 2번씩 호출하고
comparisons/<이름>/results.jsonl에 남긴다. OpenAI는 부르지 않는다.
모델을 고르는 단계이므로 최종 입력(generation_eval.jsonl)은 쓰지 않는다.

조건은 두 가지다.
- prompt_only: step07과 같은 요청. JSON 형식은 프롬프트로만 요구한다.
- json_schema: 같은 요청에 Ollama의 format(JSON 스키마)을 더해 형식을 강제한다.
두 조건 모두 생각(thinking) 단계는 끈다. OpenAI를 reasoning effort none으로 부르는 것과 맞춘 것이다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import step07_call_models as models
import step09_compare as compare
from lesson_settings import DATASET, DEVELOPMENT_INPUT, LEARNER_DIR
from step06_predict import load_classifier
from step08_connect_chain import create_chain

NUM_CTX = 4096  # step07과 같은 문맥 길이
# 형식을 강제할 때는 step07이 응답을 검사하는 스키마(AnswerPayload)를 그대로 Ollama에 넘긴다.
ANSWER_SCHEMA = models.AnswerPayload.model_json_schema()
CONDITIONS = {"prompt_only": None, "json_schema": ANSWER_SCHEMA}


def post(client, path, body):
    response = client.post(models.OLLAMA_URL.rstrip("/") + path, json=body)
    response.raise_for_status()
    return response.json()


def installed_models(client):
    """설치된 모델을 파일 크기가 작은 순서로 돌려준다."""
    response = client.get(models.OLLAMA_URL.rstrip("/") + "/api/tags")
    response.raise_for_status()
    return sorted(response.json()["models"], key=lambda model: model["size"])


def call_local(client, model, prompt, schema):
    """step07.call_model과 같은 필드를 돌려준다. 요청에서 다른 점은 think와 format뿐이다."""
    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # 생각 단계 없이 바로 답변
        "keep_alive": "5m",
        "options": {
            "temperature": models.TEMPERATURE,
            "num_predict": models.MAX_OUTPUT_TOKENS,
            "num_ctx": NUM_CTX,
        },
    }
    if schema is not None:
        request_body["format"] = schema
    output = {
        "text": "",
        "model": model,
        "usage": {"input_tokens": None, "output_tokens": None},
        "generation_complete": False,
        "finish_reason": "error",
        "load_seconds": None,
    }
    error = None
    started = time.perf_counter()
    try:
        body = post(client, "/api/generate", request_body)
        output = {
            "text": body["response"],
            "model": model,
            "usage": {
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
            },
            "generation_complete": bool(body.get("done")) and body.get("done_reason") != "length",
            "finish_reason": body.get("done_reason"),
            "load_seconds": (
                body["load_duration"] / 1e9 if body.get("load_duration") is not None else None
            ),
        }
    except Exception as exc:  # noqa: BLE001 - 실패한 호출도 기록으로 남긴다.
        error = type(exc).__name__
    return {
        **output,
        "provider": "ollama",
        "latency_seconds": time.perf_counter() - started,
        "output_format_valid": models.valid_output(output["text"]),
        "error": error,
    }


def load_model(client, model):
    """모델을 메모리에 올리고 올리는 데 걸린 시간과 메모리 사용량을 돌려준다.

    이 호출은 기록하지 않는다. 뒤의 응답 시간에 모델을 올리는 시간이 섞이지 않게 하려는 것이다.
    """
    body = post(
        client,
        "/api/generate",
        {
            "model": model,
            "prompt": "준비 확인",
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"num_predict": 1, "num_ctx": NUM_CTX},
        },
    )
    running = client.get(models.OLLAMA_URL.rstrip("/") + "/api/ps").json().get("models", [])
    loaded = next((item for item in running if item["name"] == model), {})
    return {
        "load_seconds": body.get("load_duration", 0) / 1e9,
        "loaded_gb": loaded.get("size", 0) / 2**30,
        "vram_gb": loaded.get("size_vram", 0) / 2**30,
    }


def unload_model(client, model):
    """다음 모델이 GPU 메모리를 다 쓸 수 있게 내린다."""
    post(client, "/api/generate", {"model": model, "keep_alive": 0})
    for _ in range(40):
        if not client.get(models.OLLAMA_URL.rstrip("/") + "/api/ps").json().get("models"):
            return
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # comparisons 아래에 만들 결과 폴더 이름
    parser.add_argument("--models", nargs="+")  # 생략하면 설치된 모델 전부
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    output_dir = LEARNER_DIR / "comparisons" / args.name
    if output_dir.exists():
        raise FileExistsError("출력 폴더가 있습니다. 새 이름으로 실행하세요.")

    cases = compare.read_cases(DEVELOPMENT_INPUT)
    chain = create_chain(load_classifier())
    # 분류와 프롬프트는 입력마다 한 번만 만들어 모든 모델·조건·반복이 공유한다.
    prepared = [chain.invoke(case["text"]) for case in cases]

    with httpx.Client(timeout=models.TIMEOUT_SECONDS, trust_env=False) as client:
        installed = installed_models(client)
        if args.models:
            installed = [model for model in installed if model["name"] in args.models]
            missing = set(args.models) - {model["name"] for model in installed}
            if missing:
                raise ValueError(f"설치되지 않은 모델: {sorted(missing)}")
        version = client.get(models.OLLAMA_URL.rstrip("/") + "/api/version").json()["version"]

        output_dir.mkdir(parents=True, exist_ok=False)
        records, model_rows = [], []
        with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
            for model_index, model in enumerate(installed, start=1):
                name = model["name"]
                memory = load_model(client, name)
                for condition, schema in CONDITIONS.items():
                    for case, fixed in zip(cases, prepared, strict=True):
                        for repeat in range(compare.REPEATS):
                            generated = call_local(client, name, fixed["prompt"], schema)
                            record = {
                                "case_id": case["id"],
                                "repeat": repeat + 1,
                                "condition": condition,
                                "input_text": fixed["text"],
                                "expected_label": case.get("expected_label"),
                                "classification": fixed["classification"],
                                "policy": fixed["policy"],
                                "prompt": fixed["prompt"],
                                "required_points": case.get("required_points", []),
                                "forbidden_claims": case.get("forbidden_claims", []),
                                "rubric_note": case.get("rubric_note", ""),
                                **generated,
                                "response_text": generated["text"],
                            }
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                            records.append(record)
                unload_model(client, name)
                own = [record for record in records if record["model"] == name]
                model_rows.append(
                    {
                        "name": name,
                        "digest": model["digest"][:12],
                        "parameter_size": model["details"].get("parameter_size"),
                        "quantization": model["details"].get("quantization_level"),
                        "file_gb": model["size"] / 2**30,
                        **memory,
                        "conditions": {
                            condition: compare.summarize_provider(
                                [record for record in own if record["condition"] == condition],
                                "ollama",
                            )
                            for condition in CONDITIONS
                        },
                    }
                )
                usable = {
                    condition: summary["usable_rate_all_calls"]
                    for condition, summary in model_rows[-1]["conditions"].items()
                }
                print(f"모델 {model_index}/{len(installed)} {name} 저장, 쓸 수 있는 답변 {usable}")

    summary = {
        "input": DEVELOPMENT_INPUT.name,
        "cases": len(cases),
        "repeats": compare.REPEATS,
        "ollama_version": version,
        "generation_config": {
            "temperature": models.TEMPERATURE,
            "max_output_tokens": models.MAX_OUTPUT_TOKENS,
            "timeout_seconds": models.TIMEOUT_SECONDS,
            "num_ctx": NUM_CTX,
            "think": False,
            "stream": False,
            "keep_alive": "5m",
            "conditions": {
                "prompt_only": "format 없음. JSON 형식은 프롬프트로만 요구한다 (step07과 같음).",
                "json_schema": "format에 step07의 AnswerPayload 스키마를 넣어 형식을 강제한다.",
            },
            "warmup": "모델마다 기록하지 않는 호출 한 번으로 먼저 메모리에 올린다.",
        },
        "models": model_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("저장 위치:", output_dir)


if __name__ == "__main__":
    main()
