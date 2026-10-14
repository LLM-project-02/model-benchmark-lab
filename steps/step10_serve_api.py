"""10. 학습한 분류기와 생성 모델을 팀 공통 파이프라인으로 연결해 FastAPI 요청으로 실행한다.

요청 하나를 공통 파이프라인(common_pipeline.compare)이 분류 → 프롬프트 → 생성 → 응답 검사 순서로 실행한다.
악성댓글 서비스를 파이프라인에 넘기는 연결 코드는 scripts/comments_common_pipeline.py에 있다.
요청·응답 형식은 docs/COMMENTS_API.md에 있다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step10_serve_api.py
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from step06_predict import load_classifier, predict_text
from step07_call_models import call_model
from step08_connect_chain import (
    MAX_INPUT_CHARS,
    review_answer,
    review_classification,
    review_meaning,
)

ROOT = Path(__file__).resolve().parents[1]
# 댓글 서비스 연결 코드. 이 모듈이 저장소 루트(common_pipeline)도 경로에 더한다.
sys.path.insert(0, str(ROOT / "scripts"))

import comments_common_pipeline as connect

from common_pipeline import compare as run_pipeline

HOST = "127.0.0.1"
PORT = 8000
# provider를 생략한 요청이 쓰는 생성 모델. step09 비교를 마치면 선정한 제공자로 바꾼다.
# 악성댓글 주제는 comparisons/final02 비교 뒤 로컬(Ollama gemma4:e4b)로 정했다.
DEFAULT_PROVIDER = "ollama"
# 공통 파이프라인에 등록하는 분류 함수 이름. 실제 모델은 selected.json의 모델이고(현재 KcELECTRA),
# 결과의 model_id, run_id로 확인한다.
CLASSIFIER = "kcelectra"
# 파이프라인이 기록한 실패 단계 중 생성 호출 전 단계. 답을 만들 수 없는 서버 오류(500)로 알린다.
SERVER_STAGES = {"classification", "classification_contract", "prompt"}
GENERATION_STAGES = {"llm", "llm_contract"}  # 생성 호출 실패 (502)


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
    app.state.classifier = load_classifier()  # 고친 댓글을 다시 판정할 때도 같은 모델을 쓴다
    app.state.service = connect.build_service()  # 파이프라인에 넘길 정책, 프롬프트, 파서
    yield  # 여기서부터 서버가 요청을 처리한다


app = FastAPI(title="2차 프로젝트 단계별 API", lifespan=lifespan)
# 같은 PC에서 띄운 프론트엔드 화면(브라우저)이 부를 수 있게 로컬 주소만 허용한다.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    # 이 응답은 서버 동작만 확인한다. 생성 모델 연결을 시험하지 않는다.
    return {"status": "ok"}


def answer_status(item):
    """파이프라인 결과 항목 하나로, 답변을 그대로 써도 되는지 한 단어로 알려 준다."""
    if item["status"] == "success":
        return "ok"
    if item["status"] == "incomplete":
        return "incomplete"  # 토큰 한도 등으로 답변이 잘림
    if item["error"]["stage"] in GENERATION_STAGES:
        return "generation_error"  # 생성 호출 자체가 실패 (HTTP 502)
    return "invalid_format"  # 서비스 파서(AnswerPayload)가 받아들이지 않은 답


def run(request, text, providers):
    """공통 파이프라인으로 providers의 생성 모델마다 분류 → 프롬프트 → 생성 → 응답 검사를 실행한다.

    파이프라인 결과에 없는 기존 응답 필드(확신도, 잘림 여부, 걸린 시간, 예외 종류 등)를 채우려고
    파이프라인 안에서 부른 분류·생성 결과를 함께 모은다. 모델을 따로 다시 부르지 않는다.
    (파이프라인 결과, 분류 결과, 생성 모델별 결과 목록)을 돌려준다.
    """
    verdicts, calls = [], {}

    def predict(value, bundle):
        verdicts.append(predict_text(value, bundle))
        return verdicts[-1]

    def call(provider, prompt):
        calls[provider] = call_model(provider, prompt)
        return calls[provider]

    classifier = request.app.state.classifier
    output = run_pipeline(
        {"text": text},
        service=request.app.state.service,
        classifiers={CLASSIFIER: connect.make_classifier(classifier, predict)},
        llms={name: connect.make_llm(name, call) for name in providers},
        combinations=[(CLASSIFIER, name) for name in providers],  # 적은 순서대로 실행
    )
    for item in output["results"]:
        error = item["error"]
        if error and error["stage"] in SERVER_STAGES:
            raise HTTPException(500, f"{error['stage']} 단계에서 실패했습니다 ({error['type']}).")

    # 두 생성 모델을 비교할 때도 같은 입력을 같은 모델로 분류하므로 판정과 프롬프트가 같다.
    classification = verdicts[0]
    label = classification["label"]
    results = [
        {
            **calls[item["llm"]],
            "status": answer_status(item),
            # 댓글 주제의 검사 두 가지. 답을 막지 않고 사람이 확인할지만 알린다. 다른 주제는 None
            "answer_check": review_answer(calls[item["llm"]], label, classifier),
            "meaning_check": review_meaning(calls[item["llm"]], label, text),
        }
        for item in output["results"]
    ]
    return output, classification, results


@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    # 요청 검증에 실패하면 함수 실행 전 422를 반환한다.
    output, classification, results = run(request, body.text, [body.provider])
    result = results[0]
    status = result.pop("status")
    # HTTP 200은 모델이 응답했다는 뜻이다. 답변을 쓸 수 있는지는 status로 확인한다.
    response = {
        "status": status,
        "classification": classification,
        "classification_check": review_classification(classification),
        "result": result,
        "pipeline": output,  # 팀 공통 파이프라인 결과 원본 (schema_version 2)
    }
    return JSONResponse(response, status_code=502 if status == "generation_error" else 200)


@app.post("/compare")
def compare(body: TextRequest, request: Request):
    # 같은 판정과 같은 프롬프트를 두 생성 모델에 보낸다.
    output, classification, results = run(request, body.text, ["ollama", "openai"])
    response = {
        "classification": classification,
        "classification_check": review_classification(classification),
        "results": results,
        "pipeline": output,
    }
    failed = any(row["status"] == "generation_error" for row in results)
    return JSONResponse(response, status_code=502 if failed else 200)  # 생성 실패 시 502


def main():
    print(f"API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)  # 서버 실행 (Ctrl+C로 종료)


if __name__ == "__main__":
    main()
