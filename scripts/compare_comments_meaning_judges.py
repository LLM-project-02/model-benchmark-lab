"""자유주제(악성댓글) 서비스의 '뜻 검사'에 쓸 판정 모델을 고르고, 고른 모델로 최종 비교를 한 번 잰다.

uv run python scripts/compare_comments_meaning_judges.py develop review_rules01
uv run python scripts/compare_comments_meaning_judges.py final review_rules01 --judge 모델이름

뜻 검사는 생성 모델이 고친 댓글을 원문과 함께 판정 모델에 보여 주고, 원문의 대상과 평가를
지켰는지 묻는다(프롬프트는 step08의 meaning_prompt). 판정 대상은 분류 라벨이 offensive나 hate이고
게시 불가 안내가 아닌 답이다. none은 원문을 그대로 게시하고, 게시 불가 안내는 고친 문장이 아니다.

- develop: local_models01에서 채점해 둔 답을 설치된 Ollama 모델 전부와 OpenAI(참고값)에 판정시킨다.
- final: final02의 답을 develop에서 고른 Ollama 모델 하나와 OpenAI(참고값)로 한 번 판정한다.

정답은 Claude가 모델 이름을 가리고 매긴 '근거 없는 주장 없음' 점수다(0 = 뜻이 바뀜, 2 = 지킴).
판정이 매번 같게 나오도록 temperature는 0으로 두고, 생각 단계는 끄고, JSON 형식은 스키마로 강제한다.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))
sys.path.insert(0, str(ROOT / "scripts"))

import step07_call_models as models
from compare_comments_local_models import installed_models, load_model, post, unload_model
from lesson_settings import DATASET, LEARNER_DIR
from step08_connect_chain import MeaningVerdict, meaning_prompt

SOURCES = {"develop": "local_models01", "final": "final02"}  # 이미 채점한 비교 폴더
JUDGE_SCHEMA = MeaningVerdict.model_json_schema()
TEMPERATURE = 0  # 같은 답에 같은 판정
MAX_TOKENS = 256  # 이유 한 문장과 판정이면 충분하다
NUM_CTX = 4096  # 생성 비교와 같은 문맥 길이
REFUSAL = "게시하기 어렵"  # hate 정책의 게시 불가 안내 (step08 POLICIES)


def read_pairs(name):
    """채점한 비교에서 뜻 검사 대상인 (원문, 고친 댓글) 쌍을 중복 없이 모은다."""
    path = LEARNER_DIR / "comparisons" / name / "llm_scores.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs = {}
    for row in rows:
        if row["usable"] != "True" or row["predicted_label"] not in ("offensive", "hate"):
            continue
        answer = json.loads(row["response"])["answer"]
        if REFUSAL in answer:
            continue
        score = int(row["no_unsupported_claims_0_2"])
        pair = pairs.setdefault(
            (row["case_id"], answer),
            {
                "case_id": row["case_id"],
                "text": row["text"],
                "answer": answer,
                "predicted_label": row["predicted_label"],
                "expected_label": row["expected_label"],
                "meaning_score": score,  # 정답: 0 = 뜻이 바뀜, 1 = 가벼운 변형, 2 = 지킴
                "answers": 0,  # 같은 답이 나온 횟수 (모델·조건·반복)
            },
        )
        if pair["meaning_score"] != score:
            raise ValueError(f"같은 답의 점수가 다릅니다: {row['case_id']}")
        pair["answers"] += 1
    return list(pairs.values())


def parse_verdict(text):
    """판정 JSON을 읽는다. 형식이 틀리면 (None, None)."""
    try:
        verdict = MeaningVerdict.model_validate_json(text)
    except ValidationError:
        return None, None
    return verdict.meaning_kept, verdict.reason


def judge_local(client, model, prompt):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # 생각 단계 없이 바로 판정
        "format": JUDGE_SCHEMA,  # 판정 JSON 모양을 강제한다
        "keep_alive": "5m",
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
    }
    started = time.perf_counter()
    try:
        response = post(client, "/api/generate", body)
        text, error = response["response"], None
        usage = response.get("prompt_eval_count"), response.get("eval_count")
    except Exception as exc:  # noqa: BLE001 - 실패한 호출도 기록으로 남긴다.
        text, error, usage = "", type(exc).__name__, (None, None)
    return text, error, time.perf_counter() - started, usage


def judge_openai(client, prompt):
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=models.OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": "none"},  # 생성 비교와 같게 추론 단계 없이
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            store=False,  # 요청·응답을 OpenAI에 저장하지 않음
            # 로컬과 같게 판정 JSON 모양을 스키마로 강제한다.
            text={
                "format": {
                    "type": "json_schema",
                    "name": "meaning_verdict",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                }
            },
        )
        text = response.output_text
        error = None if response.status == "completed" else f"status:{response.status}"
        usage = (
            (response.usage.input_tokens, response.usage.output_tokens)
            if response.usage
            else (None, None)
        )
    except Exception as exc:  # noqa: BLE001 - 예외 원문에는 비밀값이 섞일 수 있어 타입만 남긴다.
        text, error, usage = "", type(exc).__name__, (None, None)
    return text, error, time.perf_counter() - started, usage


def record(mode, judge, provider, pair, text, error, seconds, usage):
    meaning_kept, reason = parse_verdict(text)
    return {
        "mode": mode,
        "judge": judge,
        "provider": provider,
        **pair,
        "response_text": text,
        "valid": meaning_kept is not None,
        "meaning_kept": meaning_kept,
        "reason": reason,
        "error": error,
        "latency_seconds": seconds,
        "input_tokens": usage[0],
        "output_tokens": usage[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=SOURCES)
    parser.add_argument("name")  # comparisons 아래 결과 폴더 이름
    parser.add_argument("--judge")  # final에서 쓸 Ollama 모델 (develop에서 고른 것)
    parser.add_argument("--models", nargs="+")  # develop에서 일부 모델만 (생략하면 설치된 전부)
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    if args.mode == "final" and not args.judge:
        raise ValueError("final에는 --judge에 develop에서 고른 모델을 적으세요.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            ".env에 OPENAI_API_KEY를 설정하세요. OpenAI는 참고값으로 함께 판정합니다."
        )
    folder = LEARNER_DIR / "comparisons" / args.name
    results_path = folder / f"meaning_{args.mode}.jsonl"
    if results_path.exists():
        raise FileExistsError(f"결과 파일이 있습니다: {results_path}")
    pairs = read_pairs(SOURCES[args.mode])

    with httpx.Client(timeout=models.TIMEOUT_SECONDS, trust_env=False) as client:
        installed = installed_models(client)
        wanted = [args.judge] if args.mode == "final" else args.models
        if wanted:
            installed = [model for model in installed if model["name"] in wanted]
            missing = set(wanted) - {model["name"] for model in installed}
            if missing:
                raise ValueError(f"설치되지 않은 모델: {sorted(missing)}")
        version = client.get(models.OLLAMA_URL.rstrip("/") + "/api/version").json()["version"]

        folder.mkdir(parents=True, exist_ok=True)
        judges = []
        with results_path.open("w", encoding="utf-8") as stream:
            # 키나 요금 문제는 먼저 드러나도록 OpenAI(참고값)부터 판정한다.
            with OpenAI(timeout=models.TIMEOUT_SECONDS, max_retries=0) as openai_client:
                for pair in pairs:
                    prompt = meaning_prompt(pair["text"], pair["answer"])
                    row = record(
                        args.mode,
                        models.OPENAI_MODEL,
                        "openai",
                        pair,
                        *judge_openai(openai_client, prompt),
                    )
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            judges.append({"name": models.OPENAI_MODEL, "provider": "openai", "role": "참고값"})
            print(f"OpenAI {models.OPENAI_MODEL} 판정 {len(pairs)}건 저장")

            for index, model in enumerate(installed, start=1):
                name = model["name"]
                memory = load_model(client, name)  # 기록하지 않는 호출로 먼저 올린다
                for pair in pairs:
                    prompt = meaning_prompt(pair["text"], pair["answer"])
                    row = record(
                        args.mode, name, "ollama", pair, *judge_local(client, name, prompt)
                    )
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                unload_model(client, name)
                judges.append(
                    {
                        "name": name,
                        "provider": "ollama",
                        "digest": model["digest"][:12],
                        "parameter_size": model["details"].get("parameter_size"),
                        "quantization": model["details"].get("quantization_level"),
                        **memory,
                    }
                )
                print(f"모델 {index}/{len(installed)} {name} 판정 {len(pairs)}건 저장")

    summary = {
        "mode": args.mode,
        "source": SOURCES[args.mode],
        "pairs": len(pairs),
        "answers": sum(pair["answers"] for pair in pairs),
        "ollama_version": version,
        "judge_config": {
            "prompt": "steps/step08_connect_chain.py meaning_prompt",
            "schema": "MeaningVerdict(reason, meaning_kept)",
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
            "ollama": {"num_ctx": NUM_CTX, "think": False, "format": "json_schema"},
            "openai": {"reasoning_effort": "none", "text_format": "json_schema strict"},
            "warmup": "Ollama 모델마다 기록하지 않는 호출 한 번으로 먼저 메모리에 올린다.",
        },
        "judges": judges,
    }
    (folder / f"meaning_{args.mode}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
