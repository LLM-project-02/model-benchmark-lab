"""연구용: 같은 근거·주장 한 쌍과 같은 LLM으로 보고서 검수의 세 경로를 공통 파이프라인(compare)으로 비교한다.

경로
- llm_only: LLM 단독. 분류기를 부르지 않고 세 라벨의 정의와 지시를 모두 주어 LLM이 label을 고른다.
            가짜 분류 결과나 확신도는 넣지 않는다.
- logreg:   로지스틱 회귀 + LLM. train_baseline.py의 기준 모델(PairTfidf + LogisticRegression,
            wanwi/baseline/baseline.joblib)의 실제 예측 라벨·확률과 그 라벨의 지시를 준다.
- roberta:  RoBERTa + LLM. 선택 모델(predictor.SELECTED_RUN = roberta_lr2e5_ep6)의 체크포인트와
            온도 보정값(temperature.json)으로 계산한 실제 예측 라벨·확률과 그 라벨의 지시를 준다.

비교 단위는 사람이 직접 준 근거 문장(evidence)과 검증할 주장(claim) 한 쌍이다. 문서 전체 검수 서비스
(serve_api.py /check)처럼 근거를 검색하지 않으므로 세 경로가 같은 근거를 본다.

서비스 코드(common_adapter.py, serve_api.py)와 공통 엔진(common_pipeline)은 바꾸지 않는다. 연구용으로 둔 것:
- 프롬프트: 서비스 프롬프트(explainer.PROMPT)에서 출력 형식 줄만 {label, explanation, suggestion}으로 바꾼다.
  분류기 경로는 판정 라벨 한 줄을 더하고, LLM 단독은 [판정]·[지시] 두 줄을 세 라벨의 정의·지시로,
  분류 모델을 전제한 규칙 2를 "판정은 네가 내린다"로 바꾼다.
- 파서: 서비스 파서와 같은 방식(첫 '{'부터 마지막 '}')으로 JSON을 꺼낸 뒤 ResearchPayload로 엄격히 검사한다.
  필드가 빠지거나 더 있거나 라벨이 다르면 실패로 기록하고 고쳐서 받지 않는다.
- 생성 호출: 서비스(common_adapter.make_generate)와 같은 OpenAI 호환 chat.completions, temperature 0,
  max_tokens = config.MAX_OUTPUT_TOKENS. 자동 재시도(max_retries=0)와 다른 LLM으로의 대체는 없다.
- 분류기: 선택한 경로에 필요한 모델만 처음 쓸 때 불러온다. 파일이 없으면 model_not_ready로 기록하고
  다른 모델이나 Mock 결과로 대신하지 않는다.
"""

import csv
import io
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

NLI_DIR = Path(__file__).resolve().parent
ROOT = NLI_DIR.parent
sys.path.insert(0, str(ROOT))  # common_pipeline
sys.path.insert(0, str(NLI_DIR))  # joblib이 기준 모델을 풀 때 features.PairTfidf를 찾는다

import config
import explainer
from common_adapter import SERVICE as NLI_SERVICE
from common_adapter import validate_input
from predictor import LABEL_KO, SELECTED_RUN
from read_data import read_labels

from common_pipeline import compare

SCHEMA = "nli-research-compare/v1"
SERVICE_ID = "nli-research"
# 정책·지시 내용은 서비스(common_adapter v1)와 같고 출력 형식과 LLM 단독 안내만 다르다.
POLICY_VERSION = f"research-v1+{NLI_SERVICE['id']}-{NLI_SERVICE['policy_version']}"
LABELS = tuple(read_labels(config.DATA_DIR))  # labels.json 순서 = 학습·예측의 라벨 번호 순서

# 비교에 쓸 수 있는 LLM. 키는 화면과 기록에 쓰는 이름이다. 상용 LLM은 비용 때문에 기본 등록하지 않는다.
REGISTERED_LLMS = {
    "qwen3-4b": {"provider": "ollama", "model": config.OLLAMA_MODEL},
}
DEFAULT_LLM = "qwen3-4b"

# 분류기 실험 폴더. 가중치는 git에 없으므로 환경 변수로 바꿀 수 있다.
CLASSIFIER_RUNS = {
    "logreg": Path(os.getenv("NLI_RESEARCH_LOGREG_RUN") or config.LEARNER_DIR / "baseline"),
    "roberta": Path(os.getenv("NLI_RESEARCH_ROBERTA_RUN") or config.LEARNER_DIR / SELECTED_RUN),
}

PATHS = (
    {"id": "llm_only", "title": "LLM 단독", "classifier": None},
    {"id": "logreg", "title": "로지스틱 회귀 + LLM", "classifier": "logreg"},
    {"id": "roberta", "title": "RoBERTa + LLM", "classifier": "roberta"},
)
PATH_IDS = tuple(path["id"] for path in PATHS)

DEV_EXAMPLES = ROOT / "data" / "nli_research" / "dev_pairs.jsonl"

# ---------- 프롬프트 ----------

SERVICE_FORMAT = '{"explanation": "설명", "suggestion": "수정안 또는 확인 질문, 없으면 null"}'
RESEARCH_FORMAT = (
    '{"label": "판정 라벨", "explanation": "설명", "suggestion": "수정안 또는 확인 질문, 없으면 null"}\n'
    "label은 entailment, neutral, contradiction 중 하나의 영문 값이다."
)
SERVICE_RULE_2 = (
    "2. 판정은 분류 모델이 내린 것이다. 판정이 근거와 맞지 않아 보이면 "
    "explanation 끝에 '(판정 재검토 필요)'를 덧붙인다."
)
LLM_ONLY_RULE_2 = (
    "2. 판정은 네가 내린다. [원문 근거]와 [보고서 문장]을 비교해 아래 라벨 정의 중 하나를 골라 label에 적는다."
)
SENTINEL = "␟연구용자리표시␟"  # 입력 문장에 나올 수 없는 자리 표시 (LLM 단독 프롬프트 조립용)


def _replace_once(text, old, new):
    """서비스 프롬프트의 정해진 부분만 바꾼다. 서비스 문구가 바뀌어 찾지 못하면 실패로 알린다."""
    if text.count(old) != 1:
        raise RuntimeError("서비스 프롬프트 형식이 바뀌어 연구용 프롬프트를 만들 수 없습니다.")
    return text.replace(old, new)


def _messages(system, human):
    """서비스(common_adapter.build_prompt)와 같은 메시지 목록 JSON."""
    return json.dumps(
        [{"role": "system", "content": system}, {"role": "user", "content": human}],
        ensure_ascii=False,
    )


def build_prompt(input_data, context):
    evidence, claim = input_data["evidence"], input_data["claim"]
    if context["mode"] == "llm_only":
        if context["classification"] is not None:
            raise ValueError("LLM 단독 경로에 분류 결과가 들어왔습니다.")
        system, human = (
            message.content
            for message in explainer.PROMPT.format_messages(
                label_ko=SENTINEL, confidence=SENTINEL, policy=SENTINEL, evidence=evidence, claim=claim
            )
        )
        system = _replace_once(system, SERVICE_RULE_2, LLM_ONLY_RULE_2)
        lines = ["[라벨 정의와 라벨별 지시] 먼저 label을 고르고, 고른 라벨의 지시만 따른다.\n"]
        for label in LABELS:
            lines.append(
                f"- {label} ({LABEL_KO[label]}): 정의: {context['label_definitions'][label]} "
                f"지시: {context['policies'][label]}\n"
            )
        human = _replace_once(
            human, f"[판정] {SENTINEL} (확신도 {SENTINEL})\n[지시] {SENTINEL}\n", "".join(lines)
        )
    else:
        label = context["label"]
        # 엔진이 넘긴 정책이 서비스 지시와 같은지 확인한다(분류기 경로에는 예측 라벨의 지시만 온다).
        if context["policies"] != {label: explainer.POLICIES[label]}:
            raise ValueError("엔진이 넘긴 정책이 서비스 지시와 다릅니다.")
        probabilities = context["classification"]["probabilities"]
        if probabilities is None:
            raise ValueError("확신도에 사용할 실제 확률이 필요합니다.")
        system, human = (
            message.content
            for message in explainer.PROMPT.format_messages(
                label_ko=LABEL_KO[label],
                confidence=round(probabilities[label], 3),  # 서비스와 같은 세 자리 표시
                policy=context["policies"][label],
                evidence=evidence,
                claim=claim,
            )
        )
        judged = f"[판정] {LABEL_KO[label]} (확신도 {round(probabilities[label], 3)})\n"
        human = _replace_once(
            human, judged, judged + f"[판정 라벨] {label} (label에는 이 값을 그대로 적는다)\n"
        )
    system = _replace_once(system, SERVICE_FORMAT, RESEARCH_FORMAT)
    return _messages(system, human)


# ---------- 파서 ----------

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=False, min_length=1, pattern=r"\S")]


class ResearchPayload(BaseModel):
    """연구용 응답 형식. 세 경로가 같은 형식을 쓴다."""

    model_config = ConfigDict(extra="forbid", strict=True)
    label: Literal["entailment", "neutral", "contradiction"]
    explanation: NonBlankText
    suggestion: NonBlankText | None


def extract_json(raw_text):
    """서비스 파서(explainer.parse_output)와 같은 범위로 JSON 부분을 꺼낸다. 없으면 None."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    return match.group(0) if match else None


def parse_output(raw_text, context):
    """JSON을 꺼내 엄격히 검사한다. 형식이 틀리면 예외가 나고 엔진이 parse_output 실패로 기록한다.

    서비스와 달리 explanation만 있는 응답이나 추가 필드를 허용하지 않고, 원문을 설명으로 대신 쓰지 않는다.
    suggestion의 "null" 문자열·빈 문자열은 서비스와 같이 null로 본다.
    """
    body = extract_json(raw_text)
    if body is None:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")
    data = json.loads(body)
    if isinstance(data, dict) and data.get("suggestion") in ("", "null"):
        data["suggestion"] = None
    payload = ResearchPayload.model_validate(data).model_dump()
    return {"label": payload["label"], "generated_text": payload["explanation"], "service_output": payload}


def build_service():
    return {
        "id": SERVICE_ID,
        "policy_version": POLICY_VERSION,
        "label_definitions": dict(NLI_SERVICE["label_definitions"]),
        "common_policy": explainer.SYSTEM,
        "policies": {label: explainer.POLICIES[label] for label in LABELS},
        "validate_input": validate_input,  # 서비스와 같은 입력 검사 (evidence·claim만, 최대 길이)
        "build_prompt": build_prompt,
        "parse_output": parse_output,
    }


# ---------- 생성 호출 ----------


def generation_config(llm_key):
    """기록에 남길 실제 생성 설정 (서비스 common_adapter.make_generate의 ollama 설정과 같다)."""
    llm = REGISTERED_LLMS[llm_key]
    return {
        "provider": llm["provider"],
        "model": llm["model"],
        "endpoint": config.OLLAMA_URL.rstrip("/") + "/v1/chat/completions",
        "temperature": 0,
        "max_tokens": config.MAX_OUTPUT_TOKENS,
        "response_format": None,  # 서비스와 같이 JSON 모드를 쓰지 않는다
        "timeout_seconds": config.TIMEOUT_SECONDS,
        "retries": 0,
        "fallback": None,
    }


def call_llm(llm_key, prompt):
    """서비스와 같은 클라이언트·설정으로 한 번만 호출한다. 실패도 호출 기록으로 돌려준다."""
    llm = REGISTERED_LLMS[llm_key]
    started = time.perf_counter()
    output = {
        "text": "", "model": None, "usage": {"input_tokens": None, "output_tokens": None},
        "generation_complete": False, "finish_reason": "error",
    }
    error = detail = None
    try:
        client, model = explainer.make_client(llm["provider"])
        response = client.with_options(max_retries=0).chat.completions.create(
            model=model,
            messages=json.loads(prompt),
            max_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=0,
        )
        choice = response.choices[0]
        usage = response.usage
        output = {
            "text": choice.message.content or "",  # 원본 응답 그대로
            "model": response.model or model,
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            },
            "generation_complete": choice.finish_reason == "stop",
            "finish_reason": choice.finish_reason,
        }
    except Exception as exc:  # noqa: BLE001 - 실패도 호출 기록으로 남긴다
        error = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if status is not None:  # 로컬 Ollama의 HTTP 상태 (비밀값 없음)
            detail = f"HTTP {status}"
    return {
        **output,
        "provider": llm["provider"],
        "latency_seconds": time.perf_counter() - started,
        "error": error,
        "error_detail": detail,
    }


def make_llm(llm_key, call=None, log=None):
    """생성 호출을 엔진의 LLM 함수로 감싼다. log에는 경로별 프롬프트와 호출 결과를 순서대로 남긴다."""
    call = call or call_llm  # 실행 때 찾는다 (테스트의 실제 호출 차단이 적용되게)

    def generate(prompt):
        entry = {"prompt": prompt, "call": None}  # 호출이 예외로 끝나도 프롬프트는 기록에 남는다
        if log is not None:
            log.append(entry)
        result = entry["call"] = call(llm_key, prompt)
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


class ModelNotReady(FileNotFoundError):
    """필요한 모델 파일이 없다 (아직 학습·전달 전)."""


def _require(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise ModelNotReady("필요한 모델 파일이 없습니다: " + ", ".join(missing))


def required_files(name, run_dir):
    run_dir = Path(run_dir)
    if name == "logreg":
        return [run_dir / "config.json", run_dir / "baseline.joblib"]
    if name == "roberta":
        checkpoint = run_dir / "checkpoint"
        return [run_dir / "config.json", run_dir / "temperature.json", checkpoint / "config.json",
                checkpoint / "model.safetensors"]
    raise KeyError(name)


def _relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load_logreg(run_dir):
    """train_baseline.py가 저장한 기준 모델을 서비스 학습 때와 같은 입력·라벨 순서로 불러온다."""
    import joblib

    run_dir = Path(run_dir)
    _require(required_files("logreg", run_dir))
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if run_config.get("kind") != "baseline" or run_config.get("dataset") != "klue_nli":
        raise ValueError("NLI 기준 모델(kind=baseline, dataset=klue_nli)이 아닙니다.")
    labels = list(run_config["labels"])
    model = joblib.load(run_dir / "baseline.joblib")  # 팀이 학습해 저장한 파일만 연결한다
    classes = [int(value) for value in model.classes_]
    if sorted(classes) != list(range(len(labels))):
        raise ValueError(f"모델 클래스 번호가 라벨 목록과 맞지 않습니다: {classes}")

    def predict(evidence, claim):
        # train_baseline.to_pairs와 같은 (premise, hypothesis) 쌍 입력
        values = model.predict_proba([(evidence, claim)])[0]
        return {labels[index]: float(value) for index, value in zip(classes, values)}

    return {
        "labels": labels,
        "predict": predict,
        "identity": {"model_id": _relative(run_dir / "baseline.joblib"), "run_id": run_dir.name,
                     "kind": "baseline", "temperature": None},
    }


def load_roberta(run_dir):
    """선택 체크포인트와 온도 보정값으로 서비스(predictor.Checker)와 같은 확률을 계산한다."""
    from predictor import Checker

    run_dir = Path(run_dir)
    _require(required_files("roberta", run_dir))
    run_name = run_dir.name if run_dir.resolve().parent == config.LEARNER_DIR.resolve() else str(run_dir)
    checker = Checker(run_name=run_name)

    def predict(evidence, claim):
        values = checker.predict_pairs([evidence], [claim])[0]
        return {label: float(value) for label, value in zip(checker.labels, values)}

    return {
        "labels": list(checker.labels),
        "predict": predict,
        "identity": {"model_id": _relative(run_dir / "checkpoint"), "run_id": run_dir.name,
                     "kind": "transformer", "temperature": checker.temperature,
                     "device": checker.device},
    }


LOADERS = {"logreg": load_logreg, "roberta": load_roberta}


class ClassifierRegistry:
    """분류기를 처음 쓸 때 불러 둔다. 선택하지 않은 경로의 모델은 불러오지 않는다.

    상태(state): not_loaded(파일 있음, 아직 안 불러옴), ready(불러옴), model_not_ready(파일 없음),
    load_failed(파일은 있지만 불러오기·라벨 확인 실패).
    """

    def __init__(self, runs=None, loaders=None):
        self.runs = dict(CLASSIFIER_RUNS if runs is None else runs)
        self.loaders = dict(LOADERS if loaders is None else loaders)
        self.bundles = {}
        self.failures = {}
        self.lock = threading.Lock()

    def info(self, name):
        run = self.runs[name]
        info = {"name": name, "run_dir": _relative(run), "state": "not_loaded", "available": True,
                "error": None, "unavailable_code": None}
        if name in self.bundles:
            bundle = self.bundles[name]
            info.update(state="ready", labels=bundle["labels"], **bundle["identity"])
        elif name in self.failures:
            code, error = self.failures[name]
            info.update(state=code, available=False, error=error, unavailable_code=code)
        else:
            missing = [_relative(path) for path in required_files(name, run) if not Path(path).is_file()]
            if missing:
                info.update(state="model_not_ready", available=False, unavailable_code="model_not_ready",
                            error="필요한 모델 파일이 없습니다: " + ", ".join(missing))
        return info

    def infos(self):
        return {name: self.info(name) for name in self.runs}

    def get(self, name):
        """불러온 분류기(bundle) 또는 None(사용할 수 없음). 실패는 다시 시도하지 않고 기록을 돌려준다."""
        with self.lock:
            if name in self.bundles:
                return self.bundles[name]
            if name in self.failures:
                return None
            try:
                bundle = self.loaders[name](self.runs[name])
                if list(bundle["labels"]) != list(LABELS):
                    raise ValueError(f"라벨 순서가 연구 기준과 다릅니다: {bundle['labels']}")
                bundle["predict"]("워밍업용 근거 문장입니다.", "워밍업용 주장입니다.")  # 첫 추론 준비 시간 제외
                self.bundles[name] = bundle
                return bundle
            except Exception as exc:  # noqa: BLE001 - 다른 경로는 계속 쓸 수 있게 기록만 한다
                code = "model_not_ready" if isinstance(exc, FileNotFoundError) else "load_failed"
                self.failures[name] = (code, f"{type(exc).__name__}: {exc}")
                return None


def make_classifier(name, bundle, verdicts):
    """불러온 모델로 근거·주장 한 쌍을 실제 추론하는 엔진용 분류 함수."""

    def predict(input_data):
        probabilities = bundle["predict"](input_data["evidence"], input_data["claim"])
        label = max(probabilities, key=probabilities.get)
        verdict = {"label": label, "confidence": probabilities[label], "probabilities": probabilities}
        verdicts.append(verdict)
        return {
            "label": label,
            "probabilities": probabilities,
            "model_id": bundle["identity"]["model_id"],
            "run_id": bundle["identity"]["run_id"],
        }

    return predict


# ---------- 비교 실행 ----------

RUN_LOCK = threading.Lock()  # 동시에 들어온 비교가 서로의 시간 측정에 섞이지 않게 하나씩 실행한다


def _now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _raw_label(raw_text):
    """원본 응답에 적힌 label 값(참고용). 형식 검사와 별개이며 결과 판정에 쓰지 않는다."""
    try:
        value = json.loads(extract_json(raw_text or "") or "null").get("label")
    except (TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def run_comparison(evidence, claim, llm_key, registry, order=None, selected=None, call=None):
    """선택한 경로를 compare() 한 번으로 실행하고 경로별 기록을 돌려준다.

    selected를 생략하면 세 경로 모두다. 선택하지 않은 경로는 기록하지 않고 모델도 불러오지 않는다.
    분류기를 쓸 수 없는 경로는 unavailable로 기록하고 실행하지 않는다(다른 모델로 대신하지 않는다).
    """
    if llm_key not in REGISTERED_LLMS:
        raise KeyError(f"등록되지 않은 LLM입니다: {llm_key}")
    order = list(order or PATH_IDS)
    if sorted(order) != sorted(PATH_IDS):
        raise ValueError(f"order는 {', '.join(PATH_IDS)}를 한 번씩 담아야 합니다.")
    selected = list(PATH_IDS) if selected is None else [p for p in PATH_IDS if p in selected]
    if not selected:
        raise ValueError(f"selected에는 {', '.join(PATH_IDS)} 중 하나 이상이 있어야 합니다.")
    input_data = {"evidence": evidence, "claim": claim}
    validate_input(dict(input_data))  # 모델을 불러오기 전에 서비스와 같은 입력 검사
    paths = {path["id"]: path for path in PATHS}

    bundles = {}
    for path_id in selected:
        name = paths[path_id]["classifier"]
        if name is not None and name not in bundles:
            bundles[name] = registry.get(name)  # 선택한 경로의 모델만 불러온다
    executed = [
        path_id for path_id in order
        if path_id in selected
        and (paths[path_id]["classifier"] is None or bundles[paths[path_id]["classifier"]] is not None)
    ]
    verdicts = {name: [] for name in bundles}
    classifier_fns = {
        name: make_classifier(name, bundle, verdicts[name])
        for name, bundle in bundles.items() if bundle is not None
        and any(paths[path_id]["classifier"] == name for path_id in executed)
    }
    llm_log = []
    created_at = _now()
    output = None
    if executed:
        with RUN_LOCK:
            output = compare(
                input_data,
                service=build_service(),
                classifiers=classifier_fns,
                llms={llm_key: make_llm(llm_key, call, llm_log)},
                combinations=[(paths[path_id]["classifier"], llm_key) for path_id in executed],
            )

    records = {}
    calls = iter(llm_log)  # 엔진은 조합을 순서대로 실행하고 LLM을 조합마다 최대 한 번 부른다
    results = output["results"] if output else []
    for position, (path_id, result) in enumerate(zip(executed, results, strict=True)):
        logged = next(calls) if result["llm_invocations"] else None
        name = paths[path_id]["classifier"]
        classification = result["classification"]
        if name is None:
            applied = result["label"] if result["status"] == "success" else None
        else:
            applied = classification["label"] if classification else None
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
            "classifier_verdict": verdicts[name][0] if name and verdicts.get(name) else None,
            "llm_calls": result["llm_invocations"],
            "applied_policy_label": applied,
            "applied_policy": explainer.POLICIES[applied] if applied else None,
            "raw_label": _raw_label(result["generation"]["raw_text"]),
            "prompt": logged["prompt"] if logged else None,
            "llm_call": (
                {k: v for k, v in logged["call"].items() if k != "text"} if logged and logged["call"] else None
            ),
            "result": result,  # 공통 파이프라인 결과 항목 그대로
        }
    for path_id in selected:
        if path_id not in records:
            name = paths[path_id]["classifier"]
            info = registry.info(name)
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
        "input": input_data,
        "llm": {"key": llm_key, **REGISTERED_LLMS[llm_key]},
        "generation_config": generation_config(llm_key),
        "policy_version": POLICY_VERSION,
        "service_policy_version": NLI_SERVICE["policy_version"],
        "classifiers": {name: registry.info(name) for name in bundles},
        "order": order,
        "selected": selected,
        "executed": executed,
        "classifier_calls": {name: len(items) for name, items in verdicts.items()},
        "pipeline": (
            {key: output[key] for key in ("schema_version", "service_id", "policy_version", "total_seconds")}
            if output else None
        ),
        "paths": [records[path_id] for path_id in selected],
    }


def available_path_ids(registry):
    """실행할 수 있는 경로. 모델 파일이 없거나 불러오지 못한 경로는 빠진다(파일이 있으면 아직 안 불러왔어도 포함)."""
    infos = registry.infos()
    return [
        path["id"] for path in PATHS
        if path["classifier"] is None or infos[path["classifier"]]["available"]
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
        entry = {"title": path["title"], "runs": len(rows), "unavailable": skipped, "excluded": not rows}
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

# 사람 평가 열: docs/NLI_RESEARCH_EVAL.md의 기준. 자동 점수는 만들지 않고 비워 둔다.
EVAL_COLUMNS = [
    "eval_evidence_faithful", "eval_label_explanation_consistent", "eval_suggestion_valid",
    "eval_new_facts", "eval_memo",
]
CSV_COLUMNS = [
    "run_id", "created_at", "example_id", "reference_label", "source_text", "evidence", "claim",
    "path", "path_title", "executed_order", "status", "has_output", "error_stage", "error_type",
    "error_message", "unavailable_code", "unavailable_reason", "llm_key", "llm_model",
    "generation_model_id", "generation_config", "policy_version", "classifier_model_id",
    "classifier_run_id", "classifier_label", "classifier_confidence", "classifier_probabilities",
    "llm_label", "raw_label", "applied_policy_label", "explanation", "suggestion", "raw_text",
    "generation_complete", "finish_reason", "request_attempts", "classification_seconds",
    "llm_seconds", "total_seconds", "input_tokens", "output_tokens", *EVAL_COLUMNS,
]


def csv_rows(record):
    """비교 기록 하나를 경로별 행으로 바꾼다. 사람 평가 열은 비워 둔다(자동 점수 없음).

    has_output은 형식 검사를 통과한 설명이 있는지다. no인 행은 평가할 출력이 없으므로 평가 칸을 비워 둔다.
    reference_label·source_text는 사람 확인용이며 모델 입력이 아니다.
    """
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
            "reference_label": example.get("reference_label", ""),
            "source_text": example.get("source_text", ""),
            "evidence": record["input"]["evidence"],
            "claim": record["input"]["claim"],
            "path": path["path"],
            "path_title": path["title"],
            "executed_order": path["executed_order"],
            "status": path["status"],
            "has_output": "yes" if output else "no",
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
            "classifier_probabilities": (
                json.dumps(classification["probabilities"], ensure_ascii=False)
                if classification.get("probabilities") else None
            ),
            "llm_label": output.get("label"),
            "raw_label": path["raw_label"],
            "applied_policy_label": path["applied_policy_label"],
            "explanation": output.get("explanation"),
            "suggestion": output.get("suggestion"),
            "raw_text": generation.get("raw_text"),
            "generation_complete": generation.get("generation_complete"),
            "finish_reason": (path["llm_call"] or {}).get("finish_reason"),
            "request_attempts": generation.get("request_attempts"),
            "classification_seconds": timings.get("classification_seconds"),
            "llm_seconds": timings.get("llm_seconds"),
            "total_seconds": timings.get("total_seconds"),
            "input_tokens": generation.get("input_tokens"),
            "output_tokens": generation.get("output_tokens"),
            **{column: "" for column in EVAL_COLUMNS},
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


def load_examples(path=DEV_EXAMPLES):
    """근거·주장 쌍 예제. evidence·claim만 모델 입력이고, 참고 라벨·원문·메모는 기록에만 남긴다."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        {
            "id": row["id"],
            "evidence": row["evidence"],
            "claim": row["claim"],
            "reference_label": row.get("reference_label"),
            "source_text": row.get("source_text"),
        }
        for row in rows
    ]
