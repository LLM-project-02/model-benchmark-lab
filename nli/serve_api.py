"""보고서 검수 API (FastAPI). 문장 분리·근거 검색 후 문장마다 팀 공통 파이프라인으로 판정·설명한다.

흐름: 원문+보고서 → 문장 분리·근거 검색(checker.check) → 문장별 compare() → 응답
- 공통 파이프라인(common_pipeline)은 수정하지 않고, common_adapter.py로 연결한다.
- 설명 LLM 기본: 상용(OpenAI). 상용이 LLM 단계에서 실패하면 로컬(Ollama)로 다시 실행한다.
- 로컬이 설명한 답은 화면용 설명·수정안에만 후처리를 적용한다. 공통 기록의 원본 응답은 바꾸지 않는다.

실행: uv run python nli/serve_api.py
사용자 화면: http://127.0.0.1:8000/
API 문서:    http://127.0.0.1:8000/docs
"""

import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
from common_adapter import SERVICE, make_generate, make_predict
from common_pipeline import compare
from postprocess import soften
from predictor import LABEL_KO, Checker
from web_page import PAGE

STATE = {}  # 서버가 켜져 있는 동안 분류기와 LLM 연결을 보관
DEFAULT_PROVIDER = "openai"   # 서비스 기본 LLM
BACKUP_PROVIDER = "ollama"    # 대체 경로 (local_4b + 후처리)
CLASSIFIER_NAME = "roberta"   # compare()에 등록하는 분류 함수 이름 (실제 모델은 model_id에 기록)
# 이 단계에서 실패하면 LLM 쪽 문제로 보고 대체 경로로 다시 실행한다 (분류·입력·프롬프트 실패는 제외)
LLM_STAGES = {"llm", "llm_contract", "generation", "parse", "parse_output", "output_contract", "label"}


@asynccontextmanager
async def lifespan(app):
    checker = Checker()  # 분류기는 서버가 켜질 때 한 번만 불러 둔다
    STATE["checker"] = checker
    STATE["predict"] = make_predict(checker)
    STATE["generates"] = {}
    yield
    STATE.clear()


app = FastAPI(
    title="AI 보고서 검수관",
    description=("원문과 보고서를 받아 문장마다 근거를 찾고, 팀 공통 파이프라인으로 "
                 "학습한 분류기의 판정과 LLM의 설명·수정안을 만듭니다."),
    lifespan=lifespan,
)


class CheckRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=config.MAX_INPUT_CHARS,
                        description="원문 (회의록, 기사 등)")
    report: str = Field(..., min_length=1, max_length=config.MAX_INPUT_CHARS,
                        description="검토할 보고서")
    provider: Literal["openai", "ollama"] = Field(
        DEFAULT_PROVIDER, description="설명 LLM (기본: openai, 비공개 문서는 ollama)")
    with_explanation: bool = Field(True, description="False면 판정만 돌려준다 (LLM 호출 없음)")
    fallback: bool = Field(True, description="상용 LLM이 실패하면 로컬 LLM으로 대신 설명")


def get_generate(provider):
    """LLM 연결은 처음 쓸 때 한 번만 만든다. 만들 수 없으면 RuntimeError (예: API 키 없음)."""
    if provider not in STATE["generates"]:
        STATE["generates"][provider] = make_generate(provider)
    return STATE["generates"][provider]


def run_pair(row, provider):
    """근거·주장 한 쌍을 공통 파이프라인으로 처리하고 결과 항목 하나를 돌려준다."""
    result = compare(
        {"evidence": row["evidence"], "claim": row["claim"]},
        service=SERVICE,
        classifiers={CLASSIFIER_NAME: STATE["predict"]},
        llms={provider: get_generate(provider)},
        combinations=[(CLASSIFIER_NAME, provider)],
    )
    return result["results"][0]


def explain_row(row, request):
    """문장 하나를 공통 파이프라인으로 처리하고, 기존 화면 필드와 공통 기록을 채운다."""
    primary = request.provider
    backup = BACKUP_PROVIDER if request.fallback and primary != BACKUP_PROVIDER else None
    attempts, item, used, reason = [], None, primary, None

    # 1. 요청한 LLM으로 실행. LLM 연결을 만들 수 없거나 LLM 단계에서 실패하면 대체 경로로
    try:
        item = run_pair(row, primary)
        attempts.append(item)
        if item["status"] != "success" and item["error"]["stage"] in LLM_STAGES and backup:
            reason = f"{primary} {item['error']['stage']} 단계 실패 ({item['status']})"
    except RuntimeError as error:
        if backup is None:
            raise
        reason = f"{primary} 사용 불가: {error}"

    if reason:
        item = run_pair(row, backup)
        attempts.append(item)
        used = backup

    # 2. 공통 파이프라인의 분류 결과를 공식 판정으로 사용 (확률은 반올림 전 값이 기록에 남는다)
    classification = item["classification"]
    if classification:
        label = classification["label"]
        row["label"], row["label_ko"] = label, LABEL_KO[label]
        if classification["probabilities"]:
            row["confidence"] = round(classification["probabilities"][label], 3)

    # 3. 화면용 설명·수정안. 로컬 성공 답에만 후처리 (공통 기록의 raw_text는 그대로)
    output = item["service_output"] or {}
    explanation, suggestion = output.get("explanation"), output.get("suggestion")
    postprocessed = used == BACKUP_PROVIDER and item["status"] == "success"
    if postprocessed:
        explanation, suggestion = soften(explanation), soften(suggestion)

    generation = item.get("generation") or {}
    row["explanation"] = explanation
    row["suggestion"] = suggestion
    row["status"] = item["status"]  # success / error / incomplete
    row["llm"] = {
        "model": generation.get("model_id"),
        "seconds": item["timings"].get("llm_seconds"),
        "input_tokens": generation.get("input_tokens"),
        "output_tokens": generation.get("output_tokens"),
        "parse_ok": item["status"] == "success",
        "error": item["error"],
        "requested_provider": primary,
        "used_provider": used,
        "fallback_reason": reason,
        "postprocessed": postprocessed,
    }
    row["pipeline"] = attempts  # 공통 파이프라인 실행 기록 (대체 경로를 탔으면 두 개)
    return used != primary


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """사용자 화면: 원문과 보고서를 넣고 판정별로 정리된 결과를 본다."""
    return PAGE


@app.get("/health")
def health():
    """서버와 모델 상태 확인."""
    checker = STATE.get("checker")
    if checker is None:
        return {"status": "loading"}
    return {"status": "ok", "classifier": checker.run_name, "temperature": checker.temperature,
            "device": checker.device, "default_llm": DEFAULT_PROVIDER, "backup_llm": BACKUP_PROVIDER,
            "pipeline": {"service_id": SERVICE["id"], "policy_version": SERVICE["policy_version"]}}


@app.post("/check")
def check(request: CheckRequest):
    """원문과 보고서를 검수한다."""
    checker = STATE["checker"]
    started = time.perf_counter()

    # 1. 문장 분리 + 문장별 근거 찾기 (판정만 요청이면 이 결과를 그대로 쓴다)
    try:
        rows = checker.check(request.source, request.report)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    classify_seconds = time.perf_counter() - started

    # 2. 문장마다 공통 파이프라인으로 판정·설명
    fallback_count = 0
    for row in rows:
        if not request.with_explanation:
            row["status"] = "skipped"
            continue
        try:
            fallback_count += explain_row(row, request)
        except RuntimeError as error:  # 대체 경로 없이 LLM 연결을 만들 수 없음
            raise HTTPException(status_code=503, detail=str(error))
        except (TypeError, ValueError) as error:  # 공통 파이프라인의 실행 전 입력 검사 실패
            row.update(explanation=None, suggestion=None, status="error", pipeline=[],
                       llm={"error": {"stage": "input", "type": type(error).__name__,
                                      "message": str(error)}})

    counts = Counter(row["label_ko"] for row in rows)
    return {
        "classifier": checker.run_name,
        "temperature": checker.temperature,
        "provider": request.provider if request.with_explanation else None,
        "fallback_used": fallback_count,
        "pipeline": {"service_id": SERVICE["id"], "policy_version": SERVICE["policy_version"],
                     "classifier_name": CLASSIFIER_NAME},
        "summary": {
            "sentences": len(rows),
            "needs_review": sum(1 for r in rows if r["label"] != "entailment"),
            "by_label": dict(counts),
            "by_status": dict(Counter(r["status"] for r in rows)),
        },
        "notice": "판정은 학습한 모델의 예측이며 확정된 오류가 아닙니다. 근거 문장과 함께 검토하세요.",
        "seconds": {"classify": round(classify_seconds, 2),
                    "total": round(time.perf_counter() - started, 2)},
        "results": rows,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)