"""보고서 검수 API (FastAPI). 판정기와 LLM 설명을 하나의 요청으로 제공한다.

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


@asynccontextmanager
async def lifespan(app):
    # 서버가 켜질 때 분류기를 한 번만 불러 둔다. 요청마다 부르면 매우 느려진다.
    STATE["checker"] = Checker()
    STATE["chains"] = {}
    yield
    STATE.clear()


app = FastAPI(
    title="AI 보고서 검수관",
    description="원문과 보고서를 받아 문장마다 근거를 찾고, 학습한 분류기로 판정한 뒤 LLM이 설명과 수정안을 제시합니다.",
    lifespan=lifespan,
)


class CheckRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=config.MAX_INPUT_CHARS,
                        description="원문 (회의록, 기사 등)")
    report: str = Field(..., min_length=1, max_length=config.MAX_INPUT_CHARS,
                        description="검토할 보고서")
    provider: Literal["ollama", "openai"] = Field("ollama", description="설명을 작성할 LLM")
    with_explanation: bool = Field(True, description="False면 판정만 돌려준다 (LLM 호출 없음)")


def get_chain(provider):
    """LLM 연결은 처음 쓸 때 한 번만 만든다."""
    if provider not in STATE["chains"]:
        try:
            STATE["chains"][provider] = make_chain(provider)
        except RuntimeError as error:  # 예: OpenAI 키 없음
            raise HTTPException(status_code=503, detail=str(error))
    return STATE["chains"][provider]


@app.get("/health")
def health():
    """서버와 모델 상태 확인."""
    checker = STATE.get("checker")
    if checker is None:
        return {"status": "loading"}
    return {"status": "ok", "classifier": checker.run_name,
            "temperature": checker.temperature, "device": checker.device}


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
    if request.with_explanation:
        chain = get_chain(request.provider)
        for result in results:
            output = explain(chain, result)
            result["explanation"] = output.get("explanation")
            result["suggestion"] = output.get("suggestion")
            result["llm"] = {key: output.get(key) for key in (
                "provider", "model", "seconds", "input_tokens", "output_tokens",
                "parse_ok", "error")}

    counts = Counter(result["label_ko"] for result in results)
    return {
        "classifier": checker.run_name,
        "temperature": checker.temperature,
        "provider": request.provider if request.with_explanation else None,
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