"""보고서 검수 API (FastAPI). 판정기와 LLM 설명을 하나의 요청으로 제공한다.

- 설명 LLM 기본: 상용(OpenAI). 대체 경로: 로컬(Ollama)
- 상용을 쓸 수 없거나 호출이 실패하면 자동으로 로컬로 설명한다 (fallback=true, 기본값)
- 비공개 문서는 provider를 "ollama"로 지정해 로컬에서만 처리한다

실행: uv run python nli/serve_api.py
문서 화면(시연용): http://127.0.0.1:8000/docs
"""

import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from explainer import explain, make_chain
from predictor import Checker

STATE = {}  # 서버가 켜져 있는 동안 모델과 LLM 연결을 보관
DEFAULT_PROVIDER = "openai"   # 서비스 기본 LLM (model_selection.md의 생성 모델 선정)
BACKUP_PROVIDER = "ollama"    # 대체 경로


@asynccontextmanager
async def lifespan(app):
    STATE["checker"] = Checker()  # 분류기는 서버가 켜질 때 한 번만 불러 둔다
    STATE["chains"] = {}
    yield
    STATE.clear()


app = FastAPI(
    title="AI 보고서 검수관",
    description=("원문과 보고서를 받아 문장마다 근거를 찾고, 학습한 분류기로 판정한 뒤 "
                 "LLM이 설명과 수정안을 제시합니다. 설명 LLM은 상용이 기본이며, "
                 "상용을 쓸 수 없으면 로컬 LLM으로 대신합니다."),
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
    fallback: bool = Field(True, description="상용 LLM을 쓸 수 없으면 로컬 LLM으로 대신 설명")


def get_chain(provider):
    """LLM 연결은 처음 쓸 때 한 번만 만든다. 만들 수 없으면 RuntimeError."""
    if provider not in STATE["chains"]:
        STATE["chains"][provider] = make_chain(provider)
    return STATE["chains"][provider]


@app.get("/health")
def health():
    """서버와 모델 상태 확인."""
    checker = STATE.get("checker")
    if checker is None:
        return {"status": "loading"}
    return {"status": "ok", "classifier": checker.run_name, "temperature": checker.temperature,
            "device": checker.device, "default_llm": DEFAULT_PROVIDER, "backup_llm": BACKUP_PROVIDER}


@app.post("/check")
def check(request: CheckRequest):
    """원문과 보고서를 검수한다."""
    checker = STATE["checker"]
    started = time.perf_counter()

    # 1. 문장별 근거 찾기 + 분류기 판정
    try:
        results = checker.check(request.source, request.report)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    classify_seconds = time.perf_counter() - started

    # 2. 판정에 따라 LLM 설명 (분류기의 실제 예측으로 지시문을 고른다)
    fallback_count = 0
    if request.with_explanation:
        primary = request.provider
        backup = BACKUP_PROVIDER if primary != BACKUP_PROVIDER else None

        # 2-1. 요청한 LLM에 연결. 안 되면 대체 경로로
        try:
            chain, used, reason = get_chain(primary), primary, None
        except RuntimeError as error:
            if not (request.fallback and backup):
                raise HTTPException(status_code=503, detail=str(error))
            chain, used, reason = get_chain(backup), backup, f"{primary} 사용 불가: {error}"

        # 2-2. 문장마다 설명. 호출이 실패하면 그 문장만 대체 경로로 다시
        for result in results:
            output = explain(chain, result)
            row_used, row_reason = used, reason
            if output.get("error") and request.fallback and used == primary and backup:
                row_reason = f"{primary} 호출 실패: {output['error']}"
                output = explain(get_chain(backup), result)
                row_used = backup
            if row_used != primary:
                fallback_count += 1
            result["explanation"] = output.get("explanation")
            result["suggestion"] = output.get("suggestion")
            result["llm"] = {
                **{key: output.get(key) for key in ("model", "seconds", "input_tokens",
                                                    "output_tokens", "parse_ok", "error")},
                "requested_provider": primary,
                "used_provider": row_used,
                "fallback_reason": row_reason,
            }

    counts = Counter(result["label_ko"] for result in results)
    return {
        "classifier": checker.run_name,
        "temperature": checker.temperature,
        "provider": request.provider if request.with_explanation else None,
        "fallback_used": fallback_count,
        "summary": {
            "sentences": len(results),
            "needs_review": sum(1 for r in results if r["label"] != "entailment"),
            "by_label": dict(counts),
        },
        "notice": "판정은 학습한 모델의 예측이며 확정된 오류가 아닙니다. 근거 문장과 함께 검토하세요.",
        "seconds": {"classify": round(classify_seconds, 2),
                    "total": round(time.perf_counter() - started, 2)},
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)