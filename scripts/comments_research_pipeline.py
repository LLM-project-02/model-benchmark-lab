"""연구용: 같은 댓글·같은 LLM으로 세 경로를 공통 파이프라인(common_pipeline.compare)으로 비교한다.

경로
- llm_only:  LLM 단독. 분류 모델을 부르지 않고, 세 라벨의 정의와 정책을 모두 주어 LLM이 라벨을 고른다.
- logreg:    로지스틱 회귀 + LLM. step03 기준 모델(TF-IDF+LogisticRegression, research_lr01/baseline)의
             판정과 그 라벨의 정책을 준다.
- kcelectra: KcELECTRA + LLM. 선택 모델(bench01/kcelectra-base_lr2e5_s44)의 판정과 그 라벨의 정책을 준다.

서비스 코드와 공통 엔진은 바꾸지 않는다. 연구용으로 새로 둔 것은 다음뿐이다.
- 프롬프트: 서비스 프롬프트(step08 apply_policy)에서 출력 형식 두 줄만 {label, answer, next_steps}로 바꾼다.
  분류기 경로는 분류 결과 한 줄을 정책 앞에 더하고, LLM 단독은 정책 한 줄을 세 라벨의 정의·정책으로 바꾼다.
- 파서: ResearchPayload로 검사해 label, generated_text(answer), service_output으로 바꾼다. 보정하지 않는다.
  분류기 경로에서 LLM의 label이 분류기 판정과 다르면 엔진이 label 단계 오류로 기록한다.
- 생성 호출: step07 call_ollama와 같은 생성 설정(온도, 출력 토큰, 문맥 길이, think, keep_alive)을 쓰고,
  모델 이름과 응답 스키마(ResearchPayload)만 다르다. 자동 재시도와 다른 LLM으로의 대체(fallback)는 없다.
"""

import csv
import io
import json
import os
import statistics
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # common_pipeline
sys.path.insert(0, str(ROOT / "steps"))

import comments_common_pipeline as connect  # 서비스 연결 코드: 입력 검사, 분류 함수 감싸기, 라벨 정의
import step07_call_models as models
from step06_predict import load_run, predict_text
from step07_call_models import NonBlankText
from step08_connect_chain import COMMENT_RULES, POLICIES, apply_policy

from common_pipeline import compare

SCHEMA = "comments-research-compare/v1"
SERVICE_ID = "comments-research"
POLICY_VERSION = f"research-v1+{connect.POLICY_VERSION}"  # 정책 내용은 서비스와 같고 출력 형식만 다르다
LABELS = ("none", "offensive", "hate")

# 비교에 쓸 수 있는 LLM. 키는 화면과 기록에 쓰는 이름이다.
REGISTERED_LLMS = {
    "qwen3-4b": {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
}
DEFAULT_LLM = "qwen3-4b"

# 분류기 실험 폴더(config.json과 가중치가 있는 폴더). 가중치는 git에 없으므로 .env로 바꿀 수 있다.
COMMENTS_ROOT = ROOT / "artifacts" / "step_by_step" / "comments"
LEARNER_ROOT = COMMENTS_ROOT / "bench01"
CLASSIFIER_RUNS = {
    # bench01/baseline은 학습 기록만 있고 모델 파일이 없다. 같은 데이터·설정으로 step03을 다시 실행한
    # research_lr01/baseline을 쓴다(bench01의 학습 결과·포인터·선택 모델은 그대로 둔다).
    "logreg": Path(os.getenv("RESEARCH_LOGREG_RUN") or COMMENTS_ROOT / "research_lr01" / "baseline"),
    "kcelectra": Path(os.getenv("RESEARCH_KCELECTRA_RUN") or LEARNER_ROOT / "kcelectra-base_lr2e5_s44"),
}

PATHS = (
    {"id": "llm_only", "title": "LLM 단독", "classifier": None},
    {"id": "logreg", "title": "로지스틱 회귀 + LLM", "classifier": "logreg"},
    {"id": "kcelectra", "title": "KcELECTRA + LLM", "classifier": "kcelectra"},
)
PATH_IDS = tuple(path["id"] for path in PATHS)

DEV_EXAMPLES = ROOT / "data" / "comments" / "generation_development.jsonl"  # validation에서 뽑은 개발용

# ---------- 프롬프트 ----------

SERVICE_FORMAT = (
    "출력은 Markdown 없이 JSON 객체 하나이며 필드는 answer와 next_steps뿐이다.\n"
    "answer는 한국어 답변 문자열이고 next_steps는 문자열 배열이다.\n"
)
RESEARCH_FORMAT = (
    "출력은 Markdown 없이 JSON 객체 하나이며 필드는 label, answer, next_steps뿐이다.\n"
    "label은 none, offensive, hate 중 하나이고, answer는 한국어 답변 문자열이며, "
    "next_steps는 문자열 배열이다.\n"
)


def _replace_once(prompt, old, new):
    """서비스 프롬프트의 정해진 부분만 바꾼다. 서비스 문구가 바뀌어 찾지 못하면 실패로 알린다."""
    if prompt.count(old) != 1:
        raise RuntimeError("서비스 프롬프트 형식이 바뀌어 연구용 프롬프트를 만들 수 없습니다.")
    return prompt.replace(old, new)


def _service_prompt(text, label):
    prompt = apply_policy({"text": text, "classification": {"label": label}})["prompt"]
    return _replace_once(prompt, SERVICE_FORMAT, RESEARCH_FORMAT)


def build_prompt(input_data, context):
    text = input_data["text"].strip()  # 서비스(step10)와 같은 공백 처리
    if context["mode"] == "llm_only":
        # 서비스 프롬프트의 역할·주입 방어·출력 규칙은 그대로 두고, 정책 한 줄만 세 라벨의 정의·정책으로 바꾼다.
        lines = [
            (
                "라벨 정의와 라벨별 정책: 먼저 댓글이 아래 라벨 중 어디에 해당하는지 판단해 label에 적고, "
                "그 라벨의 정책만 따른다.\n"
            )
        ]
        for label in LABELS:
            lines.append(
                f"- {label}: 정의: {context['label_definitions'][label]} "
                f"정책: {context['policies'][label]}\n"
            )
        prompt = _service_prompt(text, "none")
        return _replace_once(prompt, f"정책: {POLICIES['none']}\n", "".join(lines))

    label = context["label"]
    # 엔진이 넘긴 정책이 서비스 정책과 같은지 확인한다(분류기 경로에는 예측 라벨의 정책만 온다).
    if context["policies"] != {label: POLICIES[label]}:
        raise ValueError("엔진이 넘긴 정책이 서비스 정책과 다릅니다.")
    prompt = _service_prompt(text, label)
    policy_line = f"정책: {POLICIES[label]}\n"
    classified = f"분류 결과(분류 모델 판정): {label}. label에는 이 값을 그대로 적는다.\n"
    return _replace_once(prompt, policy_line, classified + policy_line)


# ---------- 파서 ----------


class ResearchPayload(BaseModel):
    """연구용 응답 형식. 세 경로가 같은 형식을 쓴다."""

    model_config = ConfigDict(extra="forbid")  # 정해진 필드 외 추가 필드 금지
    label: Literal["none", "offensive", "hate"]
    answer: NonBlankText  # 서비스 AnswerPayload와 같은 규칙
    next_steps: list[NonBlankText]


def parse_output(raw_text, context):
    """형식이 틀리면 ValidationError가 나고 엔진이 parse_output 실패로 기록한다. 고쳐서 받지 않는다."""
    data = ResearchPayload.model_validate_json(raw_text).model_dump()
    return {"label": data["label"], "generated_text": data["answer"], "service_output": data}


def build_service():
    return {
        "id": SERVICE_ID,
        "policy_version": POLICY_VERSION,
        "label_definitions": connect.LABEL_DEFINITIONS,
        "common_policy": COMMENT_RULES,
        "policies": {label: POLICIES[label] for label in LABELS},
        "validate_input": connect.validate_input,  # 서비스와 같은 입력 검사
        "build_prompt": build_prompt,
        "parse_output": parse_output,
    }


# ---------- 생성 호출 ----------


def ollama_request(model, prompt):
    """step07 call_ollama와 같은 생성 설정. 모델 이름과 응답 스키마만 다르다."""
    service = models.generation_config()["ollama"]
    return {
        "model": model,
        "prompt": prompt,
        "stream": service["stream"],
        "think": service["think"],
        "format": ResearchPayload.model_json_schema(),
        "keep_alive": service["keep_alive"],
        "options": {
            "temperature": models.TEMPERATURE,
            "num_predict": models.MAX_OUTPUT_TOKENS,
            "num_ctx": service["num_ctx"],
        },
    }


def generation_config(llm_key):
    """기록에 남길 실제 생성 설정."""
    llm = REGISTERED_LLMS[llm_key]
    request = ollama_request(llm["model"], "")
    return {
        "provider": llm["provider"],
        "model": llm["model"],
        "endpoint": "/api/generate",
        "stream": request["stream"],
        "think": request["think"],
        "keep_alive": request["keep_alive"],
        **request["options"],
        "format": "json_schema(ResearchPayload)",
        "timeout_seconds": models.TIMEOUT_SECONDS,
        "retries": 0,
        "fallback": None,
    }


def call_llm(llm_key, prompt):
    """step07 call_model과 같은 모양으로 결과와 실패를 돌려준다. 다시 시도하지 않는다."""
    llm = REGISTERED_LLMS[llm_key]
    started = time.perf_counter()
    output = {
        "text": "",
        "model": None,
        "usage": {"input_tokens": None, "output_tokens": None},
        "generation_complete": False,
        "finish_reason": "error",
        "load_seconds": None,
    }
    error = detail = None
    try:
        with httpx.Client(timeout=models.TIMEOUT_SECONDS, trust_env=False) as client:
            response = client.post(
                models.OLLAMA_URL.rstrip("/") + "/api/generate",
                json=ollama_request(llm["model"], prompt),
            )
            response.raise_for_status()
            body = response.json()
        output = {
            "text": body["response"],  # 원본 응답 그대로
            "model": llm["model"],
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
    except Exception as exc:  # noqa: BLE001 - 실패도 호출 기록으로 남긴다
        error = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):  # 로컬 Ollama의 오류 본문 (비밀값 없음)
            detail = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
    return {
        **output,
        "provider": llm["provider"],
        "latency_seconds": time.perf_counter() - started,
        "error": error,
        "error_detail": detail,
    }


def make_llm(llm_key, call=call_llm, log=None):
    """생성 호출을 엔진의 LLM 함수로 감싼다. log에는 경로별 프롬프트와 호출 결과를 순서대로 남긴다."""

    def generate(prompt):
        result = call(llm_key, prompt)
        if log is not None:
            log.append({"prompt": prompt, "call": result})
        if result["error"]:
            raise RuntimeError(f"{llm_key} 생성 호출 실패: {result['error']}")
        return {
            "raw_text": result["text"],
            "model_id": result["model"],
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
            "generation_complete": result["generation_complete"],
            "request_attempts": 1,
        }

    return generate


# ---------- 분류기 ----------


def load_classifiers(runs=None, loader=load_run):
    """분류기를 미리 불러온다. 파일이 없거나 라벨이 다르면 그 경로만 사용할 수 없음으로 기록한다.

    unavailable_code: model_not_ready(가중치 파일 없음 — 아직 학습·전달 전), load_failed(그 밖의 적재 실패).
    로지스틱 회귀는 step03 기준 모델 폴더(config.json + baseline.joblib)를 RESEARCH_LOGREG_RUN으로 주면 연결된다.
    baseline.joblib은 TfidfVectorizer와 LogisticRegression을 묶은 sklearn Pipeline이라 학습 때 전처리를 그대로 쓴다.
    """
    loaded = {}
    for name, run in (CLASSIFIER_RUNS if runs is None else runs).items():
        info = {"name": name, "run_dir": str(run), "available": False, "error": None,
                "unavailable_code": None}
        try:
            bundle = loader(run)
            if list(bundle["labels"]) != list(LABELS):
                raise ValueError(f"라벨 순서가 연구 기준과 다릅니다: {bundle['labels']}")
            identity = bundle["identity"]
            info.update(
                available=True,
                kind=identity["kind"],
                run_id=identity["run_id"],
                model_id=f"{identity['learner']}/{identity['run_name']}",
                labels=list(bundle["labels"]),
            )
            loaded[name] = {"bundle": bundle, "info": info}
        except Exception as exc:  # noqa: BLE001 - 다른 경로는 계속 쓸 수 있게 기록만 한다
            info["error"] = f"{type(exc).__name__}: {exc}"
            info["unavailable_code"] = (
                "model_not_ready" if isinstance(exc, FileNotFoundError) else "load_failed"
            )
            loaded[name] = {"bundle": None, "info": info}
    return loaded


def warm_up_classifiers(classifiers, predict=predict_text):
    """첫 추론의 준비 시간을 측정에서 빼려고 한 번씩 미리 추론한다. 결과는 쓰지 않는다."""
    for entry in classifiers.values():
        if entry["bundle"] is not None:
            predict("워밍업용 문장입니다.", entry["bundle"])


# ---------- 비교 실행 ----------

RUN_LOCK = threading.Lock()  # 동시에 들어온 비교가 서로의 시간 측정에 섞이지 않게 하나씩 실행한다


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _raw_label(raw_text):
    """원본 응답에 적힌 label 값(참고용). 형식 검사와 별개이며 결과 판정에 쓰지 않는다."""
    try:
        value = json.loads(raw_text).get("label")
    except (TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def run_comparison(
    text, llm_key, classifiers, order=None, call=call_llm, predict=predict_text, selected=None
):
    """세 경로를 compare() 한 번으로 실행하고 경로별 기록을 돌려준다.

    selected를 주면 그 경로만 실행하고 기록한다(예: 이미 실행한 경로를 빼고 한 경로만 추가 실행).
    선택하지 않은 경로는 기록의 paths에 넣지 않는다. 생략하면 세 경로 모두다.
    """
    if llm_key not in REGISTERED_LLMS:
        raise KeyError(f"등록되지 않은 LLM입니다: {llm_key}")
    order = list(order or PATH_IDS)
    if sorted(order) != sorted(PATH_IDS):
        raise ValueError(f"order는 {', '.join(PATH_IDS)}를 한 번씩 담아야 합니다.")
    selected = list(PATH_IDS) if selected is None else [p for p in PATH_IDS if p in selected]
    if not selected:
        raise ValueError(f"selected에는 {', '.join(PATH_IDS)} 중 하나 이상이 있어야 합니다.")
    paths = {path["id"]: path for path in PATHS}

    def usable(path_id):
        name = paths[path_id]["classifier"]
        return name is None or (name in classifiers and classifiers[name]["bundle"] is not None)

    executed = [path_id for path_id in order if path_id in selected and usable(path_id)]
    verdicts = {name: [] for name in classifiers}

    def counting(name):
        def predict_fn(value, bundle):
            verdict = predict(value, bundle)
            verdicts[name].append(verdict)
            return verdict

        return predict_fn

    used = {paths[path_id]["classifier"] for path_id in executed} - {None}
    classifier_fns = {
        name: connect.make_classifier(classifiers[name]["bundle"], counting(name)) for name in used
    }
    llm_log = []
    created_at = _now()
    with RUN_LOCK:
        output = compare(
            {"text": text},
            service=build_service(),
            classifiers=classifier_fns,
            llms={llm_key: make_llm(llm_key, call, llm_log)},
            combinations=[(paths[path_id]["classifier"], llm_key) for path_id in executed],
        )

    records = {}
    calls = iter(llm_log)  # 엔진은 조합을 순서대로 실행하고 LLM을 조합마다 최대 한 번 부른다
    for position, (path_id, result) in enumerate(zip(executed, output["results"], strict=True)):
        logged = next(calls) if result["llm_invocations"] else None
        name = paths[path_id]["classifier"]
        classification = result["classification"]
        if name is None:
            applied = result["label"] if result["status"] == "success" else None
        else:
            applied = classification["label"] if classification else None
        call_info = None
        if logged:
            call_info = {key: value for key, value in logged["call"].items() if key != "text"}
        records[path_id] = {
            "path": path_id,
            "title": paths[path_id]["title"],
            "classifier": name,
            "executed_order": position + 1,
            "status": result["status"],
            "unavailable_code": None,
            "unavailable_reason": None,
            # 이 조합에서 엔진이 분류 함수를 불렀는지 (LLM 단독은 항상 False여야 한다)
            "classifier_called": result["timings"]["classification_seconds"] is not None,
            "classifier_verdict": (
                {key: verdicts[name][0][key] for key in ("label", "confidence", "truncated")}
                if name and verdicts.get(name)
                else None
            ),
            "llm_calls": result["llm_invocations"],
            "applied_policy_label": applied,
            "applied_policy": POLICIES[applied] if applied else None,
            "raw_label": _raw_label(result["generation"]["raw_text"]),
            "prompt": logged["prompt"] if logged else None,
            "llm_call": call_info,
            "result": result,  # 공통 파이프라인 결과 항목 그대로
        }
    for path_id in selected:
        if path_id not in records:
            name = paths[path_id]["classifier"]
            info = classifiers.get(name, {}).get("info", {})
            records[path_id] = {
                "path": path_id,
                "title": paths[path_id]["title"],
                "classifier": name,
                "executed_order": None,
                "status": "unavailable",
                "unavailable_code": info.get("unavailable_code") or "load_failed",
                "unavailable_reason": info.get("error") or "분류기를 불러오지 못했습니다.",
                "classifier_called": False,
                "classifier_verdict": None,
                "llm_calls": 0,
                "applied_policy_label": None,
                "applied_policy": None,
                "raw_label": None,
                "prompt": None,
                "llm_call": None,
                "result": None,
            }

    return {
        "schema": SCHEMA,
        "run_id": uuid.uuid4().hex,
        "created_at": created_at,
        "input": {"text": text},
        "llm": {"key": llm_key, **REGISTERED_LLMS[llm_key]},
        "generation_config": generation_config(llm_key),
        "policy_version": POLICY_VERSION,
        "service_policy_version": connect.POLICY_VERSION,
        "classifiers": {name: entry["info"] for name, entry in classifiers.items()},
        "order": order,
        "selected": selected,
        "executed": executed,
        "classifier_calls": {name: len(items) for name, items in verdicts.items()},
        "pipeline": {
            key: output[key] for key in ("schema_version", "service_id", "policy_version", "total_seconds")
        },
        "paths": [records[path_id] for path_id in selected],
    }


def available_path_ids(classifiers):
    """실행할 수 있는 경로 (분류기를 불러오지 못한 경로는 빠진다)."""
    return [
        path["id"]
        for path in PATHS
        if path["classifier"] is None
        or classifiers.get(path["classifier"], {}).get("bundle") is not None
    ]


METRICS = (
    ("classification_seconds", "timings", "classification_seconds"),
    ("llm_seconds", "timings", "llm_seconds"),
    ("total_seconds", "timings", "total_seconds"),
    ("input_tokens", "generation", "input_tokens"),
    ("output_tokens", "generation", "output_tokens"),
)


def summarize(records):
    """경로별 비교 통계. 실행하지 않은 경로(unavailable)는 통계에서 빼고 값은 None(해당 없음)으로 둔다.

    품질 점수는 만들지 않는다. 상태별 개수와 측정값의 평균·중앙값만 낸다(값이 있는 기록만).
    """
    summary = {}
    for path in PATHS:
        rows = [
            item for record in records for item in record["paths"]
            if item["path"] == path["id"] and item["status"] != "unavailable"
        ]
        skipped = sum(
            1 for record in records for item in record["paths"]
            if item["path"] == path["id"] and item["status"] == "unavailable"
        )
        entry = {"title": path["title"], "runs": len(rows), "unavailable": skipped,
                 "excluded": not rows}
        statuses = {}
        for item in rows:
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
        entry["status_counts"] = statuses if rows else None
        for name, group, key in METRICS:
            values = [item["result"][group][key] for item in rows if item["result"][group][key] is not None]
            entry[name] = (
                {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values)}
                if values else None
            )
        summary[path["id"]] = entry
    return summary


# ---------- 기록 내보내기 ----------

CSV_COLUMNS = [
    "run_id", "created_at", "example_id", "reference_label", "input_text",
    "path", "path_title", "executed_order", "status", "error_stage", "error_type", "error_message",
    "unavailable_code", "unavailable_reason", "llm_key", "llm_model", "generation_model_id", "generation_config",
    "policy_version", "classifier_model_id", "classifier_run_id", "classifier_label",
    "classifier_confidence", "llm_label", "raw_label", "applied_policy_label", "answer",
    "next_steps", "raw_text", "generation_complete", "request_attempts", "classification_seconds",
    "llm_seconds", "total_seconds", "input_tokens", "output_tokens", "load_seconds",
    "eval_policy_compliance", "eval_meaning_preserved", "eval_new_facts", "eval_memo",
]


def csv_rows(record):
    """비교 기록 하나를 경로별 행으로 바꾼다. 사람 평가 열은 비워 둔다(자동 점수 없음)."""
    example = record.get("example") or {}
    rows = []
    for path in record["paths"]:
        result = path["result"] or {}
        error = result.get("error") or {}
        classification = result.get("classification") or {}
        generation = result.get("generation") or {}
        timings = result.get("timings") or {}
        output = result.get("service_output") or {}
        verdict = path["classifier_verdict"] or {}
        rows.append({
            "run_id": record["run_id"],
            "created_at": record["created_at"],
            "example_id": example.get("id", ""),
            "reference_label": example.get("reference_label", ""),  # 사람 확인용, 모델 입력 아님
            "input_text": record["input"]["text"],
            "path": path["path"],
            "path_title": path["title"],
            "executed_order": path["executed_order"],
            "status": path["status"],
            "error_stage": error.get("stage"),
            "error_type": error.get("type"),
            "error_message": error.get("message"),
            "unavailable_code": path["unavailable_code"],
            "unavailable_reason": path["unavailable_reason"],
            "llm_key": record["llm"]["key"],
            "llm_model": record["llm"]["model"],
            "generation_model_id": generation.get("model_id"),
            "generation_config": json.dumps(record["generation_config"], ensure_ascii=False),
            "policy_version": record["policy_version"],
            "classifier_model_id": classification.get("model_id"),
            "classifier_run_id": classification.get("run_id"),
            "classifier_label": classification.get("label"),
            "classifier_confidence": verdict.get("confidence"),
            "llm_label": output.get("label"),
            "raw_label": path["raw_label"],
            "applied_policy_label": path["applied_policy_label"],
            "answer": output.get("answer"),
            "next_steps": json.dumps(output["next_steps"], ensure_ascii=False) if output else None,
            "raw_text": generation.get("raw_text"),
            "generation_complete": generation.get("generation_complete"),
            "request_attempts": generation.get("request_attempts"),
            "classification_seconds": timings.get("classification_seconds"),
            "llm_seconds": timings.get("llm_seconds"),
            "total_seconds": timings.get("total_seconds"),
            "input_tokens": generation.get("input_tokens"),
            "output_tokens": generation.get("output_tokens"),
            "load_seconds": (path["llm_call"] or {}).get("load_seconds"),
            "eval_policy_compliance": "",
            "eval_meaning_preserved": "",
            "eval_new_facts": "",
            "eval_memo": "",
        })
    return rows


def to_csv(records):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for record in records:
        for row in csv_rows(record):
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    return buffer.getvalue()


def load_dev_examples(path=DEV_EXAMPLES):
    """개발용 고정 댓글. 정답 라벨과 평가 메모는 모델 입력에 넣지 않는다(text만 넘긴다)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"id": row["id"], "text": row["text"], "reference_label": row["expected_label"]} for row in rows]
