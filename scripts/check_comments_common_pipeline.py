"""악성댓글 서비스를 공통 파이프라인에 연결한 뒤 정상 요청과 오류 요청을 실제로 보내 결과를 남긴다.

uv run python scripts/check_comments_common_pipeline.py common_pipeline01

- 정상 요청: 라벨마다 개발 입력의 첫 댓글을 분류기(KcELECTRA) + 로컬, 분류기 + 상용 조합으로 보낸다.
  OpenAI 호출은 3번이다.
- 실행 전에 막히는 오류: 빈 입력, 글자 수 초과, 입력 필드 이름이 다름, 등록하지 않은 LLM 이름
- 실행 중 오류(조합별 기록): LLM 단독, Ollama 연결 실패, 형식 강제를 끈 응답(코드 블록), 토큰 한도로 잘린 응답
결과는 artifacts/.../<LEARNER>/service_checks/<이름>/에 results.json과 check.md로 남긴다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "steps"))
sys.path.insert(0, str(ROOT / "scripts"))

import comments_common_pipeline as connect
import step07_call_models as models
from compare_comments_local_models import call_local
from lesson_settings import DATASET, DEVELOPMENT_INPUT, LEARNER_DIR, MAX_INPUT_CHARS
from step06_predict import load_classifier
from step08_connect_chain import REFUSAL_PHRASE

from common_pipeline import compare

ENGINE_COMMIT = "44c8944"  # feature/common-pipeline-parser-support
CLOSED_OLLAMA_URL = "http://127.0.0.1:9"  # 아무것도 듣지 않는 포트
LABELS = ["none", "offensive", "hate"]


def first_cases():
    """라벨마다 개발 입력의 첫 댓글. 결과를 보고 고르지 않도록 순서로만 정한다."""
    cases = [
        json.loads(line)
        for line in DEVELOPMENT_INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picked = {}
    for case in cases:
        picked.setdefault(case["expected_label"], case)
    return [picked[label] for label in LABELS]


def prompt_only(provider, prompt):
    """형식 강제(format)를 뺀 로컬 호출. gemma4:e4b는 JSON을 코드 블록으로 감싸 파서가 실패한다."""
    with httpx.Client(timeout=models.TIMEOUT_SECONDS, trust_env=False) as client:
        return call_local(client, models.OLLAMA_MODEL, prompt, None)


def cut_short(provider, prompt):
    """최대 출력 토큰을 8로 줄인 로컬 호출. 답이 끝나기 전에 멈춰 생성 미완료가 된다."""
    body = {
        "model": models.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": models.AnswerPayload.model_json_schema(),
        "options": {"temperature": models.TEMPERATURE, "num_predict": 8, "num_ctx": 4096},
    }
    with httpx.Client(timeout=models.TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(models.OLLAMA_URL.rstrip("/") + "/api/generate", json=body)
        response.raise_for_status()
        data = response.json()
    return {
        "text": data["response"],
        "model": models.OLLAMA_MODEL,
        "usage": {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        },
        "generation_complete": bool(data.get("done")) and data.get("done_reason") != "length",
        "error": None,
    }


def closed_ollama(provider, prompt):
    """Ollama 주소만 닫힌 포트로 바꿔 부른다. 켜져 있는 Ollama는 끄지 않는다."""
    original = models.OLLAMA_URL
    models.OLLAMA_URL = CLOSED_OLLAMA_URL
    try:
        return models.call_model(provider, prompt)
    finally:
        models.OLLAMA_URL = original


def run_before_errors(service, classifiers, llms, text):
    """실행 전에 막혀야 하는 요청. 모델 함수가 불리지 않았는지도 센다."""
    calls = {"count": 0}

    def counting(function):
        def wrapped(*args):
            calls["count"] += 1
            return function(*args)

        return wrapped

    watched_classifiers = {name: counting(function) for name, function in classifiers.items()}
    watched_llms = {name: counting(function) for name, function in llms.items()}
    cases = [
        ("빈 입력 (공백만)", {"text": "   "}, [("kcelectra", "ollama")]),
        (
            f"글자 수 초과 ({MAX_INPUT_CHARS + 1}자)",
            {"text": "가" * (MAX_INPUT_CHARS + 1)},
            [("kcelectra", "ollama")],
        ),
        ("입력 필드 이름이 다름 (comment)", {"comment": text}, [("kcelectra", "ollama")]),
        ("등록하지 않은 LLM 이름", {"text": text}, [("kcelectra", "claude")]),
    ]
    records = []
    for name, data, combinations in cases:
        before = calls["count"]
        try:
            compare(
                data,
                service=service,
                classifiers=watched_classifiers,
                llms=watched_llms,
                combinations=combinations,
            )
            outcome = {"raised": None, "message": None}
        except (TypeError, ValueError) as exc:
            outcome = {"raised": type(exc).__name__, "message": str(exc)}
        records.append(
            {
                "name": name,
                **outcome,
                "model_calls": calls["count"] - before,
                "passed": outcome["raised"] is not None and calls["count"] == before,
            }
        )
    return records


def run_during_errors(service, classifiers, text):
    """실행 중 실패해 조합별 결과에 남아야 하는 요청."""
    llms = {
        "ollama": connect.make_llm("ollama"),
        "ollama_closed": connect.make_llm("ollama", call=closed_ollama),
        "ollama_prompt_only": connect.make_llm("ollama", call=prompt_only),
        "ollama_cut_short": connect.make_llm("ollama", call=cut_short),
    }
    expected = [
        ("LLM 단독 (분류기 없음)", (None, "ollama"), "error", "prompt"),
        ("Ollama 연결 실패", ("kcelectra", "ollama_closed"), "error", "llm"),
        ("형식 강제를 끈 응답", ("kcelectra", "ollama_prompt_only"), "error", "parse_output"),
        ("토큰 한도로 잘린 응답", ("kcelectra", "ollama_cut_short"), "incomplete", "generation"),
    ]
    output = compare(
        {"text": text},
        service=service,
        classifiers=classifiers,
        llms=llms,
        combinations=[combination for _, combination, _, _ in expected],
    )
    checks = []
    for (name, _, status, stage), result in zip(expected, output["results"], strict=True):
        checks.append(
            {
                "name": name,
                "expected": f"{status} / {stage}",
                "actual": f"{result['status']} / {(result['error'] or {}).get('stage')}",
                "error_type": (result["error"] or {}).get("type"),
                "raw_text_kept": result["generation"]["raw_text"] is not None,
                "passed": result["status"] == status
                and (result["error"] or {}).get("stage") == stage,
            }
        )
    return output, checks


def short(text, limit=60):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def handling(result, original):
    """고친 댓글을 문서에 그대로 옮기지 않고 처리 방식만 적는다(실제 댓글의 실명·혐오 표현을 피함)."""
    if not result["success"]:
        return "-"
    answer = result["generated_text"]
    if " ".join(answer.split()) == " ".join(original.split()):
        return "원문 그대로"
    return "게시 불가 안내" if REFUSAL_PHRASE in answer else "고쳐 씀"


def write_markdown(path, name, normal, before, during_checks, identity):
    lines = [
        f"# 공통 파이프라인 연결 확인 ({name})",
        "",
        (
            f"악성댓글 서비스를 공통 파이프라인(`common_pipeline`, {ENGINE_COMMIT})에 연결하고 "
            "정상 요청과 오류 요청을 실제로 보낸 기록입니다. 결과 원문은 [results.json](results.json)에 "
            "있습니다. 연결 코드는 `scripts/comments_common_pipeline.py`입니다."
        ),
        "",
        "## 설정",
        "",
        (
            f"- 서비스: `{connect.SERVICE_ID}`, 정책 버전 `{connect.POLICY_VERSION}`, "
            f"결과 형식 `schema_version` {normal[0]['schema_version']}"
        ),
        (
            f"- 분류기 `kcelectra`: `{identity['learner']}/{identity['run_name']}` "
            f"(beomi/KcELECTRA-base-v2022 미세조정, run_id `{identity['run_id']}`), CPU"
        ),
        f"- LLM `ollama`: `{models.OLLAMA_MODEL}`, `openai`: `{models.OPENAI_MODEL}`",
        "- 프롬프트와 파서: 서비스의 step08 `apply_policy`와 step07 `AnswerPayload`를 그대로 씀",
        "- 로컬 모델은 기록하지 않는 호출 한 번으로 먼저 메모리에 올렸습니다.",
        "",
        "## 정상 요청",
        "",
        "| 입력 | 조합 | 결과 | 라벨 (확률) | 처리 (generated_text) | 분류 | LLM | 전체 |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for case_id, output in normal_rows(normal):
        original = output["input_data"]["text"]
        for result in output["results"]:
            probability = (result["classification"] or {}).get("probabilities") or {}
            label = result["label"]
            timings = result["timings"]
            lines.append(
                f"| {case_id} | {result['classifier']} + {result['llm']} | {result['status']} | "
                f"{label} ({probability.get(label, 0):.2f}) | {handling(result, original)} | "
                f"{timings['classification_seconds']:.2f}초 | {timings['llm_seconds']:.2f}초 | "
                f"{timings['total_seconds']:.2f}초 |"
            )
    lines += [
        "",
        (
            "- **success의 뜻:** 호출과 형식 검사(분류 결과, 생성 응답, `AnswerPayload`)가 통과했다는 "
            "뜻입니다. 고친 댓글의 내용이 적절한지는 엔진이 판단하지 않습니다. 내용 품질은 "
            "`comparisons/final02`의 채점에, 사람 검토 검사(확신도, 다시 검사, 뜻 검사)는 FastAPI "
            "서비스(step10)에 있습니다."
        ),
        (
            "- **내용 주의:** 입력은 실제 댓글이라 실명과 혐오 표현이 들어 있어, 고친 댓글은 이 문서에 "
            "옮기지 않고 처리 방식만 적었습니다. 원문과 고친 댓글은 results.json에 있습니다."
        ),
        "- 첫 분류 호출은 CPU에서 모델을 처음 실행하느라 다른 호출보다 조금 더 걸립니다.",
        "",
        "## 실행 전에 막히는 오류",
        "",
        "입력이나 설정이 잘못되면 `compare()`가 모델을 부르기 전에 예외를 냅니다.",
        "",
        "| 상황 | 예외 | 모델 호출 수 | 맞음 |",
        "|---|---|---:|:---:|",
    ]
    for record in before:
        lines.append(
            f"| {record['name']} | {record['raised']}: {short(record['message'], 50)} | "
            f"{record['model_calls']} | {'예' if record['passed'] else '아니오'} |"
        )
    lines += [
        "",
        "## 실행 중 오류 (조합별로 기록하고 다음 조합을 계속함)",
        "",
        "| 상황 | 기대 (status / stage) | 결과 | 예외 종류 | 원본 응답 보존 | 맞음 |",
        "|---|---|---|---|:---:|:---:|",
    ]
    for check in during_checks:
        lines.append(
            f"| {check['name']} | {check['expected']} | {check['actual']} | {check['error_type']} | "
            f"{'예' if check['raw_text_kept'] else '-'} | {'예' if check['passed'] else '아니오'} |"
        )
    lines += [
        "",
        (
            "- LLM 단독: 서비스 프롬프트가 라벨별 정책을 전제로 하고 `AnswerPayload`에 label 필드가 "
            "없어 지원하지 않습니다. 프롬프트 단계에서 실패하고 LLM은 부르지 않습니다."
        ),
        (
            "- 형식 강제를 끈 응답: gemma4:e4b가 JSON을 코드 블록으로 감싸 `AnswerPayload`가 "
            "거부합니다. 서비스가 Ollama에 형식 강제(format)를 쓰는 이유입니다. 원본 응답은 "
            "`generation.raw_text`에 남습니다."
        ),
        (
            "- 토큰 한도로 잘린 응답: 생성이 끝나지 않았으므로 엔진이 파서를 부르지 않고 "
            "`incomplete`로 기록합니다."
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def normal_rows(normal):
    return [(output["case_id"], output) for output in normal]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")  # service_checks 아래에 만들 결과 폴더 이름
    args = parser.parse_args()
    if DATASET != "comments":
        raise ValueError('lesson_settings.py의 DATASET이 "comments"일 때만 실행합니다.')
    folder = LEARNER_DIR / "service_checks" / args.name
    if folder.exists():
        raise FileExistsError("결과 폴더가 있습니다. 새 이름으로 실행하세요.")

    bundle = load_classifier()
    service = connect.build_service()
    classifiers = {"kcelectra": connect.make_classifier(bundle)}
    llms = {"ollama": connect.make_llm("ollama"), "openai": connect.make_llm("openai")}
    cases = first_cases()

    models.call_model("ollama", "준비 확인")  # 로컬 모델을 먼저 올린다 (기록하지 않음)
    normal = []
    for case in cases:
        output = compare(
            {"text": case["text"]},
            service=service,
            classifiers=classifiers,
            llms=llms,
            combinations=[("kcelectra", "ollama"), ("kcelectra", "openai")],
        )
        normal.append({"case_id": case["id"].split("development-")[-1], **output})
    before = run_before_errors(service, classifiers, llms, cases[1]["text"])
    during, during_checks = run_during_errors(service, classifiers, cases[1]["text"])

    folder.mkdir(parents=True)
    record = {
        "engine_commit": ENGINE_COMMIT,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "normal": normal,
        "before_errors": before,
        "during_errors": during,
        "during_checks": during_checks,
    }
    (folder / "results.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(
        folder / "check.md", args.name, normal, before, during_checks, bundle["identity"]
    )
    ok_normal = all(result["success"] for output in normal for result in output["results"])
    ok_errors = all(item["passed"] for item in before + during_checks)
    print(f"정상 요청 {sum(len(o['results']) for o in normal)}개 모두 성공: {ok_normal}")
    print(f"오류 요청 {len(before) + len(during_checks)}개 모두 기대대로: {ok_errors}")
    for output in normal:
        for result in output["results"]:
            print(" ", output["case_id"], result["llm"], connect.summary_line(result))
    for item in before + during_checks:
        print(" ", "예" if item["passed"] else "아니오", item["name"])
    print("저장 위치:", folder)


if __name__ == "__main__":
    main()
