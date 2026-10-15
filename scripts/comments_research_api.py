"""연구용 API: 같은 댓글·같은 LLM으로 세 경로(LLM 단독, 로지스틱 회귀 + LLM, KcELECTRA + LLM)를 비교한다.

기존 서비스 API(steps/step10_serve_api.py)는 바꾸지 않고 따로 띄운다. 모델과 LLM 호출은 이 서버에서 실행한다.
분류기는 서버가 켜질 때 한 번 불러오고 한 번씩 미리 추론해 둔다. 요청마다 실제 추론을 다시 한다.

실행(저장소 루트): uv run python scripts/comments_research_api.py
분류기 폴더는 .env의 RESEARCH_KCELECTRA_RUN, RESEARCH_LOGREG_RUN으로 바꿀 수 있다.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import comments_research_pipeline as research

MAX_INPUT_CHARS = research.connect.MAX_INPUT_CHARS  # 서비스(step08)와 같은 최대 글자 수
HOST = "127.0.0.1"
PORT = int(os.getenv("RESEARCH_PORT", "8002"))  # 서비스 API(8000)와 겹치지 않게 한다


class CompareRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)  # 서비스와 같은 입력 기준
    llm: str = research.DEFAULT_LLM
    # 경로 실행 순서. 생략하면 llm_only, logreg, kcelectra. 실험에서는 예제마다 번갈아 준다.
    order: list[str] | None = None

    @field_validator("text")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("공백만 입력할 수 없습니다.")
        return text.strip()


def create_app(load=research.load_classifiers, call=research.call_llm, warm_up=True):
    """테스트에서 분류기 적재와 생성 호출을 바꿔 끼울 수 있게 앱을 만든다."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.classifiers = load()
        if warm_up:
            research.warm_up_classifiers(app.state.classifiers)
        yield

    app = FastAPI(title="댓글 파이프라인 비교 연구 API", lifespan=lifespan)
    # 같은 PC의 화면만 부를 수 있게 로컬 주소만 허용한다 (서비스 API와 같은 기준).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/research/health")
    def health(request: Request):
        return {
            "status": "ok",
            "classifiers": {
                name: entry["info"] for name, entry in request.app.state.classifiers.items()
            },
        }

    @app.get("/research/options")
    def options(request: Request):
        return {
            "schema": research.SCHEMA,
            "llms": [{"key": key, **llm} for key, llm in research.REGISTERED_LLMS.items()],
            "default_llm": research.DEFAULT_LLM,
            "paths": list(research.PATHS),
            "available_paths": research.available_path_ids(request.app.state.classifiers),
            "classifiers": {
                name: entry["info"] for name, entry in request.app.state.classifiers.items()
            },
            "policy_version": research.POLICY_VERSION,
            "generation_config": {
                key: research.generation_config(key) for key in research.REGISTERED_LLMS
            },
            "max_input_chars": MAX_INPUT_CHARS,
        }

    @app.get("/research/examples")
    def examples():
        # reference_label은 사람이 확인하는 값이다. 비교 실행에는 text만 넘긴다.
        return {"examples": research.load_dev_examples()}

    @app.post("/research/compare")
    def run(body: CompareRequest, request: Request):
        if body.llm not in research.REGISTERED_LLMS:
            raise HTTPException(400, f"등록되지 않은 LLM입니다: {body.llm}")
        try:
            return research.run_comparison(
                body.text, body.llm, request.app.state.classifiers, order=body.order, call=call
            )
        except ValueError as error:  # 잘못된 order, 서비스 입력 검사 실패
            raise HTTPException(400, str(error)) from error

    return app


app = create_app()


def main():
    print(f"연구용 API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)


if __name__ == "__main__":
    main()
