"""10. 학습한 분류기와 생성 체인을 FastAPI 요청으로 실행한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step10_serve_api.py
"""

from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from step06_predict import load_classifier
from step08_connect_chain import (
    MAX_INPUT_CHARS,
    compare_prepared,
    create_chain,
    generate_prepared,
)

HOST = "127.0.0.1"
PORT = 8000
# provider를 생략한 요청이 쓰는 생성 모델. step09 비교를 마치면 선정한 제공자로 바꾼다.
DEFAULT_PROVIDER = "ollama"


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)  # 위반 시 422 응답

    @field_validator("text")  # 공백만 있는 입력을 추가로 거부
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("공백만 입력할 수 없습니다.")
        return text.strip()


class GenerateRequest(TextRequest):
    provider: Literal["ollama", "openai"] = DEFAULT_PROVIDER  # 두 값만 허용


@asynccontextmanager  # yield가 하나 있는 함수를 "시작할 때 할 일 / 끝날 때 할 일"을 가진 객체로 바꿔 줍니다.
async def lifespan(app):
    # 서버가 시작될 때 한 번 읽는다. 모델을 읽지 못하면 시작이 실패한다.
    classifier = load_classifier()
    app.state.chain = create_chain(classifier)
    yield  # 여기서부터 서버가 요청을 처리한다


app = FastAPI(title="2차 프로젝트 단계별 API", lifespan=lifespan)


@app.get("/health")
def health():
    # 이 응답은 서버 동작만 확인한다. 생성 모델 연결을 시험하지 않는다.
    return {"status": "ok"}


def answer_status(result):
    """HTTP 코드와 별도로, 답변을 그대로 써도 되는지 한 단어로 알려 준다."""
    if result["error"]:
        return "generation_error"  # 생성 호출 자체가 실패 (HTTP 502)
    if not result["generation_complete"]:
        return "incomplete"  # 토큰 한도 등으로 답변이 잘림
    if not result["output_format_valid"]:
        return "invalid_format"  # 약속한 JSON 형식이 아님
    return "ok"


@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    # 요청 검증에 실패하면 함수 실행 전 422를 반환한다.
    prepared = request.app.state.chain.invoke(body.text)  # 분류 + 프롬프트 준비
    result = generate_prepared(prepared, body.provider)
    # HTTP 200은 모델이 응답했다는 뜻이다. 답변을 쓸 수 있는지는 status로 확인한다.
    response = {"status": answer_status(result),
                "classification": prepared["classification"], "result": result}
    return JSONResponse(response, status_code=502 if result["error"] else 200)  # 생성 실패 시 502


@app.post("/compare")
def compare(body: TextRequest, request: Request):
    # 한 번 분류해서 만든 프롬프트를 두 생성 모델이 공유한다.
    prepared = request.app.state.chain.invoke(body.text)
    result = compare_prepared(prepared)
    result["results"] = [{**row, "status": answer_status(row)} for row in result["results"]]
    failed = any(row["error"] for row in result["results"])
    return JSONResponse(result, status_code=502 if failed else 200)


def main():
    print(f"API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)  # 서버 실행 (Ctrl+C로 종료)


if __name__ == "__main__":
    main()
