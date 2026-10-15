"""연구용 API: 같은 근거·주장 한 쌍과 같은 LLM으로 세 경로(LLM 단독, 로지스틱 회귀 + LLM, RoBERTa + LLM)를 비교한다.

기존 보고서 검수 API(serve_api.py /check)는 바꾸지 않고 따로 띄운다. 댓글 연구 API(8002)와도 겹치지 않게
기본 포트는 8003이다. 경로(/research/*)는 댓글 연구 API와 같은 모양이라 화면이 같은 방식으로 부른다.

- 모델 파일이 없어도 서버와 설정 조회는 시작된다. 없는 경로는 model_not_ready로 알린다.
- 분류기는 서버가 켜질 때 불러오지 않고, 그 경로를 처음 실행할 때 한 번 불러와 미리 추론해 둔다.
  LLM 단독만 실행하면 분류기 체크포인트가 없어도 된다. NLI_RESEARCH_PRELOAD=1이면 켤 때 모두 불러온다.

실행(저장소 루트): uv run python nli/research_api.py
분류기 폴더는 .env의 NLI_RESEARCH_LOGREG_RUN, NLI_RESEARCH_ROBERTA_RUN으로 바꿀 수 있다.
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

import config
import research_pipeline as research

MAX_INPUT_CHARS = config.MAX_INPUT_CHARS  # 서비스(common_adapter.validate_input)와 같은 최대 글자 수
HOST = "127.0.0.1"
PORT = int(os.getenv("NLI_RESEARCH_PORT", "8003"))  # 서비스(8001)·댓글 연구(8002)와 겹치지 않게 한다


class CompareRequest(BaseModel):
    evidence: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="근거 문장 (원문에서 고른 문장)")
    claim: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="검증할 주장 (보고서 문장)")
    llm: str = research.DEFAULT_LLM
    # 경로 실행 순서. 생략하면 llm_only, logreg, roberta. 실험에서는 예제마다 번갈아 준다.
    order: list[str] | None = None
    # 실행할 경로. 생략하면 세 경로 모두. 선택하지 않은 경로는 기록하지 않고 모델도 불러오지 않는다.
    paths: list[str] | None = None

    @field_validator("evidence", "claim")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("공백만 입력할 수 없습니다.")
        return text


def create_app(registry_factory=None, call=None, preload=None):
    """테스트에서 분류기 적재와 생성 호출을 바꿔 끼울 수 있게 앱을 만든다."""
    preload = os.getenv("NLI_RESEARCH_PRELOAD") == "1" if preload is None else preload

    @asynccontextmanager
    async def lifespan(app):
        app.state.registry = (registry_factory or research.ClassifierRegistry)()
        if preload:
            for name in app.state.registry.runs:
                app.state.registry.get(name)
        yield

    app = FastAPI(title="보고서 파이프라인 비교 연구 API", lifespan=lifespan)
    # 같은 PC의 화면만 부를 수 있게 로컬 주소만 허용한다 (서비스 API와 같은 기준).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/research/health")
    def health(request: Request):
        return {"status": "ok", "service": "nli", "classifiers": request.app.state.registry.infos()}

    @app.get("/research/options")
    def options(request: Request):
        registry = request.app.state.registry
        return {
            "schema": research.SCHEMA,
            "service": "nli",
            "input_fields": ["evidence", "claim"],
            "labels": list(research.LABELS),
            "llms": [{"key": key, **llm} for key, llm in research.REGISTERED_LLMS.items()],
            "default_llm": research.DEFAULT_LLM,
            "paths": list(research.PATHS),
            "available_paths": research.available_path_ids(registry),
            "classifiers": registry.infos(),
            "policy_version": research.POLICY_VERSION,
            "generation_config": {key: research.generation_config(key) for key in research.REGISTERED_LLMS},
            "max_input_chars": MAX_INPUT_CHARS,
        }

    @app.get("/research/examples")
    def examples():
        # reference_label·source_text는 사람이 확인하는 값이다. 비교 실행에는 evidence·claim만 넘긴다.
        try:
            return {"examples": research.load_examples()}
        except FileNotFoundError:
            return {"examples": []}

    @app.post("/research/compare")
    def run(body: CompareRequest, request: Request):
        if body.llm not in research.REGISTERED_LLMS:
            raise HTTPException(400, f"등록되지 않은 LLM입니다: {body.llm}")
        if body.paths is not None and not set(body.paths) <= set(research.PATH_IDS):
            raise HTTPException(400, f"paths는 {', '.join(research.PATH_IDS)} 중에서 고릅니다.")
        try:
            return research.run_comparison(
                body.evidence, body.claim, body.llm, request.app.state.registry,
                order=body.order, selected=body.paths, call=call,
            )
        except ValueError as error:  # 잘못된 order·paths, 서비스 입력 검사 실패
            raise HTTPException(400, str(error)) from error

    return app


app = create_app()


def main():
    print(f"보고서 연구용 API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)


if __name__ == "__main__":
    main()
