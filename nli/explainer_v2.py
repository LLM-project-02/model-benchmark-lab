"""지시문 v2: v1(explainer.py)은 그대로 두고 바뀐 부분만 정의한다.

v1에서 바뀐 점 (두 가지만)
1. 재검토가 필요하면 판정과 관계없이 원문 근거에 맞게 고친 문장을 suggestion에 적는다.
   - 공통 규칙 2에 추가
   - 뒷받침함·확인 불가 지시 끝에 같은 예외 문장 추가 (v1에서는 이 두 지시가 수정안을 막았다)
2. 공통 규칙 4의 금지 표현 예시를 자동 확인 목록과 같게 늘린다.

바뀌지 않은 것: 모순됨 지시, 공통 규칙 1·3·5, 질문 틀(HUMAN), LLM 연결, JSON 해석
"""

import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import config
import explainer
from explainer import HUMAN, ROLE, make_client, parse_output

PROMPT_VERSION = "v2"

# ① 재검토가 필요할 때의 예외 (뒷받침함·확인 불가 지시 끝에 붙인다)
REVIEW_EXCEPTION = (
    " 단, 원문 근거가 보고서 문장과 맞지 않아 재검토가 필요하면 "
    "원문 근거에 맞게 고친 보고서 문장을 suggestion에 적는다."
)

POLICIES_V2 = {
    "entailment": (
        "보고서 문장이 원문과 일치한다고 판정되었다. 원문의 어느 부분이 뒷받침하는지 "
        "한두 문장으로 설명하라. 근거가 뒷받침하면 suggestion은 null로 둔다."
        + REVIEW_EXCEPTION
    ),
    "contradiction": explainer.POLICIES["contradiction"],  # v1과 같음
    "neutral": explainer.POLICIES["neutral"] + REVIEW_EXCEPTION,
}

SYSTEM_V2 = """너는 보고서 검수를 돕는 보조자다.
규칙:
1. 반드시 [원문 근거]에 있는 내용만 사용한다. 원문에 없는 사실을 지어내지 않는다.
2. 판정은 분류 모델이 내린 것이다. 판정이 근거와 맞지 않아 보이면 explanation 끝에 '(판정 재검토 필요)'를 덧붙이고, 판정과 관계없이 원문 근거에 맞게 고친 보고서 문장을 suggestion에 적는다.
3. 한국어로 짧고 분명하게 쓴다.
4. '명확한 오류', '명백한 오류', '틀렸다', '틀린 내용'처럼 단정하지 않고, '원문과 다르다', '확인이 필요하다'처럼 쓴다. 확신도는 설명에 언급하지 않는다.
5. 아래 JSON 형식으로만 답한다. 다른 글은 쓰지 않는다.
{{"explanation": "설명", "suggestion": "수정안 또는 확인 질문, 없으면 null"}}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", HUMAN)])


def make_chain(provider):
    """v1의 make_chain과 같고, 프롬프트만 v2를 쓴다."""
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

    return PROMPT_V2 | RunnableLambda(call_llm) | RunnableLambda(parse_output)


def explain(chain, result):
    """v1의 explain과 같고, 판정별 지시만 v2를 쓴다."""
    try:
        return chain.invoke({
            "label_ko": result["label_ko"],
            "confidence": result["confidence"],
            "policy": POLICIES_V2[result["label"]],
            "evidence": result["evidence"],
            "claim": result["claim"],
        })
    except Exception as error:  # 연결 실패, 시간 초과 등
        return {"error": f"{type(error).__name__}: {error}", "explanation": None,
                "suggestion": None, "parse_ok": False}