"""악성댓글 분류 + 순화 서비스의 사용자 화면 (도전 실습 B, Streamlit).

터미널 1: uv run python steps/step10_serve_api.py
터미널 2: uv run --group ui streamlit run scripts/comments_streamlit_app.py

이 화면은 step10 FastAPI 서버에 요청을 보내고 결과를 보여 주기만 한다. 분류와 생성은 서버가 한다.
- 댓글 하나: 분류 결과, 적용한 정책, 고친 댓글을 보여 준다. 두 생성 모델을 나란히 비교할 수도 있다.
- CSV 여러 건: 한 열의 댓글을 차례로 보내고, 결과 표를 CSV로 내려받는다.
"""

import io
import json

import httpx
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"  # step10_serve_api.py의 HOST, PORT
TIMEOUT_SECONDS = 300  # 서버를 켠 뒤 첫 요청은 로컬 모델을 올리느라 오래 걸린다
LABELS = {
    "none": ("문제 없음", "원문을 그대로 둡니다."),
    "offensive": ("불쾌한 표현", "욕설, 비속어, 비꼼을 걷어 내고 정중한 문장으로 고쳐 씁니다."),
    "hate": (
        "혐오 표현",
        "혐오, 차별, 모욕 표현을 지웁니다. 남길 의견이 없으면 게시하기 어렵다고 알립니다.",
    ),
}
STATUS = {
    "ok": "그대로 쓸 수 있는 답변",
    "incomplete": "답변이 중간에 잘림",
    "invalid_format": "약속한 JSON 형식이 아님",
    "generation_error": "생성 모델 호출 실패",
}
PROVIDERS = {"로컬 (Ollama)": "ollama", "상용 (OpenAI)": "openai"}
COMPARE = "두 모델 비교"


def call_api(url, path, payload=None):
    """서버에 요청하고 {code, body, error}를 돌려준다. 서버에 닿지 못하면 code는 None이다."""
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False) as client:
            if payload is None:
                response = client.get(url.rstrip("/") + path)
            else:
                response = client.post(url.rstrip("/") + path, json=payload)
    except httpx.HTTPError as exc:
        return {"code": None, "body": None, "error": type(exc).__name__}
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text[:500]}
    return {"code": response.status_code, "body": body, "error": None}


def parse_answer(result):
    """생성 결과의 JSON 문자열에서 (answer, next_steps)를 꺼낸다. 형식이 틀리면 원문을 그대로 돌려준다."""
    try:
        payload = json.loads(result.get("text") or "")
        return str(payload["answer"]), [str(step) for step in payload.get("next_steps", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return result.get("text") or "", []


# 서버가 422와 함께 알려 주는 오류 종류(FastAPI, 영어 문장)를 화면에 보일 한국어 문장으로 바꾼다.
INPUT_ERRORS = {
    "string_too_short": "댓글을 입력해 주세요.",
    "missing": "댓글을 입력해 주세요.",
    "string_too_long": "댓글은 4,000자까지 입력할 수 있습니다.",
    "literal_error": "생성 모델은 ollama 또는 openai만 고를 수 있습니다.",
    "json_invalid": "요청 본문이 JSON 형식이 아닙니다.",
}


def input_error(body):
    """422 응답에서 사람이 읽을 오류 문장을 만든다."""
    details = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(details, list):
        return "요청 형식이 맞지 않습니다."
    messages = []
    for item in details:
        # value_error는 서버가 쓴 한국어 문장(예: 공백만 입력할 수 없습니다.)을 그대로 쓴다
        message = INPUT_ERRORS.get(item.get("type"), str(item.get("msg", "")))
        messages.append(message.removeprefix("Value error, "))
    return " / ".join(dict.fromkeys(messages))  # 같은 문장은 한 번만


def read_csv_bytes(data):
    """올린 CSV를 읽는다. 엑셀에서 저장한 한국어 CSV(cp949)도 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽지 못했습니다. UTF-8로 저장해 다시 올려 주세요.")


def batch_row(text, response):
    """CSV 처리 결과 한 줄."""
    row = {
        "원문": text,
        "라벨": "",
        "분류 확신도": None,
        "고친 댓글": "",
        "다음 안내": "",
        "상태": "",
        "응답 시간(초)": None,
        "생성 모델": "",
        "오류": "",
    }
    body = response["body"] or {}
    if response["code"] is None:
        row["오류"] = f"서버에 연결하지 못함 ({response['error']})"
        return row
    if response["code"] == 422:
        row["오류"] = "입력 오류: " + input_error(body)
        return row
    classification = body.get("classification") or {}
    result = body.get("result") or {}
    answer, next_steps = parse_answer(result)
    row.update(
        {
            "라벨": classification.get("label", ""),
            "분류 확신도": round(classification["confidence"], 3)
            if "confidence" in classification
            else None,
            "고친 댓글": answer,
            "다음 안내": " / ".join(next_steps),
            "상태": body.get("status", ""),
            "응답 시간(초)": round(result["latency_seconds"], 2)
            if "latency_seconds" in result
            else None,
            "생성 모델": result.get("model") or "",
            "오류": result.get("error") or "",
        }
    )
    return row


def show_classification(classification):
    label = classification["label"]
    title, policy = LABELS.get(label, (label, ""))
    left, right = st.columns(2)
    left.metric("분류 결과", f"{title} ({label})")
    right.metric("분류 확신도", f"{classification['confidence']:.0%}")
    # 확신도는 분류 모델이 낸 값이다. 생성 모델은 이 라벨을 받기만 하므로 두 모델 비교에서도 하나다.
    st.caption(
        f"적용한 정책: {policy} · 확신도는 분류 모델(KcELECTRA)이 이 라벨을 얼마나 확신하는지이고, "
        "생성 모델과는 관계없습니다."
    )


def show_result(result, status):
    """생성 모델 하나의 결과."""
    if result.get("error"):
        st.error(f"생성 모델 호출에 실패했습니다 ({result['error']}). 분류 결과는 위와 같습니다.")
        return
    answer, next_steps = parse_answer(result)
    st.markdown("**고친 댓글**")
    if status == "ok":
        st.success(answer)
    else:
        st.warning(answer or "(빈 답변)")
    for step in next_steps:
        st.markdown(f"- {step}")
    st.caption(
        f"{STATUS.get(status, status)} · {result.get('model')} · {result['latency_seconds']:.2f}초"
    )


def single_tab(url):
    text = st.text_area("댓글", height=120, max_chars=4000, placeholder="확인할 댓글을 입력하세요.")
    mode = st.radio("생성 모델", [*PROVIDERS, COMPARE], horizontal=True)
    if st.button("확인", type="primary"):
        with st.spinner("분류하고 고쳐 쓰는 중입니다."):
            if mode == COMPARE:
                response = call_api(url, "/compare", {"text": text})
            else:
                response = call_api(url, "/generate", {"text": text, "provider": PROVIDERS[mode]})
        st.session_state["single"] = response

    response = st.session_state.get("single")
    if not response:
        return
    body = response["body"] or {}
    if response["code"] is None:
        st.error(
            f"서버에 연결하지 못했습니다 ({response['error']}). 왼쪽 안내대로 서버를 먼저 켜 주세요."
        )
        return
    if response["code"] == 422:
        st.warning("입력을 확인해 주세요: " + input_error(body))
        return
    if "classification" not in body:
        st.error(f"예상하지 못한 응답입니다 (HTTP {response['code']}).")
        return
    show_classification(body["classification"])
    if "results" in body:  # 두 모델 비교
        columns = st.columns(len(body["results"]))
        for column, result in zip(columns, body["results"], strict=True):
            with column:
                st.subheader("로컬" if result["provider"] == "ollama" else "상용")
                show_result(result, result.get("status"))
    else:
        show_result(body["result"], body.get("status"))


def batch_tab(url):
    uploaded = st.file_uploader("CSV 파일", type="csv")
    if not uploaded:
        st.caption("댓글이 한 열에 들어 있는 CSV를 올리면 한 줄씩 분류하고 고쳐 씁니다.")
        return
    try:
        frame = read_csv_bytes(uploaded.getvalue())
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return
    if frame.empty:
        st.warning("CSV에 행이 없습니다.")
        return
    columns = list(frame.columns)
    column = st.selectbox(
        "댓글이 든 열", columns, index=columns.index("text") if "text" in columns else 0
    )
    limit = st.number_input(
        "처리할 최대 건수", min_value=1, max_value=len(frame), value=min(len(frame), 50)
    )
    provider = st.radio("생성 모델", list(PROVIDERS), horizontal=True, key="batch_provider")
    st.caption(
        f"{len(frame):,}행 중 앞에서 {int(limit):,}건을 처리합니다. 로컬 모델은 한 건에 1~2초쯤 걸립니다."
    )
    if st.button("처리 시작", type="primary"):
        texts = frame[column].head(int(limit)).fillna("").astype(str).tolist()
        progress = st.progress(0.0)
        rows = []
        for index, text in enumerate(texts, start=1):
            response = call_api(url, "/generate", {"text": text, "provider": PROVIDERS[provider]})
            rows.append(batch_row(text, response))
            progress.progress(index / len(texts), text=f"{index}/{len(texts)}건 처리")
        st.session_state["batch"] = pd.DataFrame(rows)

    result = st.session_state.get("batch")
    if result is None:
        return
    counts = result["라벨"].replace("", "처리 못 함").value_counts()
    st.markdown(
        "**결과:** "
        + ", ".join(f"{label} {count}건" for label, count in counts.items())
        + f" · 그대로 쓸 수 있는 답변 {(result['상태'] == 'ok').sum()}/{len(result)}건"
    )
    st.dataframe(result, width="stretch", hide_index=True)
    st.download_button(
        "결과 CSV 내려받기",
        result.to_csv(index=False).encode("utf-8-sig"),  # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
        file_name="comments_result.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="악성댓글 분류 + 순화", layout="wide")
    st.title("악성댓글 분류 + 순화")
    st.caption(
        "학습한 분류 모델(KcELECTRA)이 댓글을 none·offensive·hate로 나누고, "
        "생성 모델이 라벨별 정책에 따라 댓글을 고쳐 씁니다."
    )
    with st.sidebar:
        url = st.text_input("서버 주소", API_URL)
        health = call_api(url, "/health")
        if health["code"] == 200:
            st.success("서버에 연결됨")
        else:
            st.error("서버가 꺼져 있습니다.")
            st.code("uv run python steps/step10_serve_api.py", language="powershell")
            st.caption("다른 터미널에서 위 명령으로 서버를 먼저 켜 주세요.")
        st.caption("서버를 켠 뒤 첫 요청은 로컬 모델을 메모리에 올리느라 7초쯤 걸립니다.")
    single, batch = st.tabs(["댓글 하나", "CSV 여러 건"])
    with single:
        single_tab(url)
    with batch:
        batch_tab(url)


if __name__ == "__main__":
    main()
