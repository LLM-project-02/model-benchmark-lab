"""판정 결과를 받아 LLM이 설명과 수정안을 쓰게 한다. (LangChain으로 연결)

흐름: 판정 결과 → 판정별 지시문(정책) 선택 → 프롬프트 → LLM 호출 → JSON 해석
- 로컬: Ollama (OpenAI 호환 주소 /v1 사용)
- 상용: OpenAI API

실행
  uv run python nli/explainer.py                    # 로컬 (Ollama)
  uv run python nli/explainer.py --provider openai  # 상용 (.env에 OPENAI_API_KEY 필요)
"""

import argparse
import json
import os
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

import config
from predictor import Checker

# 판정별 지시문. 분류기의 실제 예측으로 고른다 (정답 라벨이 아님).
POLICIES = {
    "entailment": (
        "보고서 문장이 원문과 일치한다고 판정되었다. 원문의 어느 부분이 뒷받침하는지 "
        "한두 문장으로 설명하라. 수정안은 필요 없으므로 suggestion은 null로 둔다."
    ),
    "contradiction": (
        "보고서 문장이 원문과 충돌한다고 판정되었다. 원문과 다른 부분(숫자, 날짜, 주체, "
        "부정 표현 등)을 짚어 설명하고, 원문에 맞게 고친 보고서 문장을 suggestion에 적어라."
    ),
    "neutral": (
        "원문만으로는 보고서 문장을 확인할 수 없다고 판정되었다. '원문과 다르다'고 쓰지 말고 "
        "'원문에서 확인할 수 없다'고 설명하라. 보고서 작성자에게 이 내용의 출처나 근거 자료를 "
        "묻는 짧은 질문 한 문장을 suggestion에 적어라."
    ),
}

# 중괄호 { }는 LangChain에서 빈칸 표시라, 글자 그대로 쓰려면 {{ }}로 두 번 쓴다.
SYSTEM = """너는 보고서 검수를 돕는 보조자다.
규칙:
1. 반드시 [원문 근거]에 있는 내용만 사용한다. 원문에 없는 사실을 지어내지 않는다.
2. 판정은 분류 모델이 내린 것이다. 판정이 근거와 맞지 않아 보이면 explanation 끝에 '(판정 재검토 필요)'를 덧붙인다.
3. 한국어로 짧고 분명하게 쓴다.
4. '명확한 오류', '틀렸다'처럼 단정하지 않고, '원문과 다르다', '확인이 필요하다'처럼 쓴다. 확신도는 설명에 언급하지 않는다.
5. 아래 JSON 형식으로만 답한다. 다른 글은 쓰지 않는다.
{{"explanation": "설명", "suggestion": "수정안 또는 확인 질문, 없으면 null"}}"""

HUMAN = """[판정] {label_ko} (확신도 {confidence})
[지시] {policy}
[원문 근거] {evidence}
[보고서 문장] {claim}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
ROLE = {"system": "system", "human": "user", "ai": "assistant"}


def make_client(provider):
    """provider에 맞는 LLM 연결과 모델 이름을 돌려준다."""
    if provider == "ollama":
        client = OpenAI(base_url=config.OLLAMA_URL.rstrip("/") + "/v1",
                        api_key="ollama", timeout=config.TIMEOUT_SECONDS)
        return client, config.OLLAMA_MODEL
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")
        return OpenAI(timeout=config.TIMEOUT_SECONDS), config.OPENAI_MODEL
    raise ValueError(f"알 수 없는 provider: {provider}")


def parse_output(raw):
    """LLM 답에서 JSON을 꺼낸다. 형식이 깨지면 원문 그대로 설명으로 쓰고 실패를 기록한다."""
    match = re.search(r"\{.*\}", raw["text"], re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    ok = isinstance(data, dict) and "explanation" in data
    suggestion = data.get("suggestion") if ok else None
    raw["parse_ok"] = ok
    raw["explanation"] = data["explanation"] if ok else raw["text"].strip()
    raw["suggestion"] = None if suggestion in (None, "", "null") else suggestion
    return raw


def make_chain(provider):
    """프롬프트 → LLM 호출 → JSON 해석을 LangChain으로 잇는다."""
    client, model = make_client(provider)

    def call_llm(prompt_value):
        messages = [{"role": ROLE[m.type], "content": m.content}
                    for m in prompt_value.to_messages()]
        if provider == "ollama":
            options = {"max_tokens": config.MAX_OUTPUT_TOKENS, "temperature": 0}
        else:
            options = {"max_completion_tokens": config.MAX_OUTPUT_TOKENS}
        started = time.perf_counter()
        response = client.chat.completions.create(model=model, messages=messages, **options)
        usage = response.usage
        return {
            "provider": provider,
            "model": model,
            "text": response.choices[0].message.content or "",
            "seconds": round(time.perf_counter() - started, 2),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }

    return PROMPT | RunnableLambda(call_llm) | RunnableLambda(parse_output)


def explain(chain, result):
    """판정 결과 하나를 설명한다. LLM 호출이 실패해도 서비스가 멈추지 않게 오류를 기록한다."""
    try:
        return chain.invoke({
            "label_ko": result["label_ko"],
            "confidence": result["confidence"],
            "policy": POLICIES[result["label"]],
            "evidence": result["evidence"],
            "claim": result["claim"],
        })
    except Exception as error:  # 연결 실패, 시간 초과 등
        return {"error": f"{type(error).__name__}: {error}", "explanation": None,
                "suggestion": None, "parse_ok": False}


# 직접 작성한 가상 회의록 (외부 API로 보내도 되는 공유 가능한 문장, 화면 예시 1과 같음)
DEMO_SOURCE = (
    "2026년 10월 7일 제품기획팀 회의에서 신규 앱 '오늘장보기'의 출시일을 11월 10일로 확정했다. "
    "마케팅 예산은 3억 원으로 승인되었다. "
    "디자인팀은 10월 24일까지 최종 화면을 개발팀에 전달하기로 했다. "
    "유료 구독 기능은 이번 출시에 포함하지 않기로 했다."
)
DEMO_REPORT = (
    "마케팅 예산은 5억 원으로 승인되었다. "
    "디자인팀은 10월 24일까지 최종 화면을 개발팀에 넘기기로 했다. "
    "출시 첫 달 목표 가입자는 10만 명이다."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    checker = Checker()
    chain = make_chain(args.provider)
    print(f"분류기: {checker.run_name} / LLM: {args.provider}\n")

    for i, result in enumerate(checker.check(DEMO_SOURCE, DEMO_REPORT), 1):
        output = explain(chain, result)
        print(f"[{i}] 보고서 문장: {result['claim']}")
        print(f"    판정: {result['label_ko']} (확신도 {result['confidence']})")
        if output.get("error"):
            print(f"    LLM 오류: {output['error']}\n")
            continue
        print(f"    설명: {output['explanation']}")
        print(f"    제안: {output['suggestion']}")
        print(f"    형식 정상: {output['parse_ok']} / {output['seconds']}초 / "
              f"토큰 입력 {output['input_tokens']}, 출력 {output['output_tokens']}\n")


if __name__ == "__main__":
    main()