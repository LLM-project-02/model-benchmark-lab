"""악성댓글 분류 + 순화 서비스의 업무용 콘솔 화면 (도전 실습 B, Streamlit).

터미널 1: uv run python steps/step10_serve_api.py
터미널 2: uv run --group ui streamlit run scripts/comments_streamlit_app.py

이 화면은 step10 FastAPI 서버에 요청을 보내고 결과를 보여 주기만 한다. 분류와 생성은 서버가 한다.
- 단건 검수: 판정, 권장 조치, 게시할 댓글을 카드로 보여 준다. 로컬과 상용 결과를 나란히 비교할 수도 있다.
- 대량 검수: CSV의 댓글을 여러 개씩 동시에 보내 처리하고, 요약 지표와 결과 표를 CSV로 내려받는다.
- 정책, API 연동: 라벨별 처리 기준과 서버를 직접 부르는 방법을 보여 준다.
화면 색과 글꼴은 .streamlit/config.toml의 [theme]에 있다.
"""

import io
import json
import os
import re
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd
import streamlit as st

# step10_serve_api.py의 HOST, PORT. 다른 주소의 서버를 쓰려면 COMMENTS_API_URL 환경 변수로 바꾼다.
API_URL = os.getenv("COMMENTS_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 300  # 서버를 켠 뒤 첫 요청은 로컬 모델을 올리느라 오래 걸린다
# 로컬 모델에 4개씩 보내면 한 줄씩 보낼 때보다 1.5배쯤 빨랐다(8건 10.9초 → 7.1초)
DEFAULT_WORKERS = 4
# 라벨: (화면 이름, 배지 색, 처리 정책)
LABELS = {
    "none": ("문제 없음", "green", "원문을 그대로 둡니다."),
    "offensive": (
        "불쾌한 표현",
        "orange",
        "욕설, 비속어, 비꼼을 걷어 내고 의견은 남긴 채 정중한 문장으로 고쳐 씁니다.",
    ),
    "hate": (
        "혐오 표현",
        "red",
        "혐오, 차별, 모욕 표현을 지웁니다. 남길 의견이 없으면 게시하기 어렵다고 알립니다.",
    ),
}
# 권장 조치: (배지 색, 아이콘). 화면이 API 응답으로 정하는 운영자용 안내다.
ACTIONS = {
    "그대로 게시": ("green", ":material/check_circle:"),
    "순화 후 게시": ("blue", ":material/edit_note:"),
    "게시 보류": ("red", ":material/block:"),
    "사람 검토": ("orange", ":material/person_search:"),
    "처리 실패": ("gray", ":material/error:"),
}
STATUS = {
    "ok": "그대로 쓸 수 있는 답변",
    "incomplete": "답변이 중간에 잘림",
    "invalid_format": "약속한 JSON 형식이 아님",
    "generation_error": "생성 모델 호출 실패",
}
PROVIDERS = {"로컬": "ollama", "상용": "openai"}
PROVIDER_MODELS = {"ollama": "Ollama gemma4:e4b", "openai": "OpenAI gpt-6-luna"}
JUDGE_MODEL = "Ollama qwen3:4b-instruct"  # 뜻 검사 판정 모델 (comparisons/review_rules01)
REFUSAL = "게시하기 어렵"  # hate 지침의 게시 불가 안내에 들어가는 말 (step08 REFUSAL_PHRASE)
# 서버가 422와 함께 알려 주는 오류 종류(FastAPI, 영어 문장)를 화면에 보일 한국어 문장으로 바꾼다.
INPUT_ERRORS = {
    "string_too_short": "댓글을 입력해 주세요.",
    "missing": "댓글을 입력해 주세요.",
    "string_too_long": "댓글은 4,000자까지 입력할 수 있습니다.",
    "literal_error": "생성 모델은 ollama 또는 openai만 고를 수 있습니다.",
    "json_invalid": "요청 본문이 JSON 형식이 아닙니다.",
}


def call_api(url, path, payload=None):
    """서버에 요청하고 {code, body, error, seconds}를 돌려준다. 서버에 닿지 못하면 code는 None이다.

    seconds는 요청을 보내고 응답을 받기까지 걸린 시간이다(분류, 생성, 검사를 모두 포함).
    """
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False) as client:
            if payload is None:
                response = client.get(url.rstrip("/") + path)
            else:
                response = client.post(url.rstrip("/") + path, json=payload)
    except httpx.HTTPError as exc:
        return {
            "code": None,
            "body": None,
            "error": type(exc).__name__,
            "seconds": time.perf_counter() - started,
        }
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text[:500]}
    return {
        "code": response.status_code,
        "body": body,
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def parse_answer(result):
    """생성 결과의 JSON 문자열에서 (answer, next_steps)를 꺼낸다. 형식이 틀리면 원문을 그대로 돌려준다."""
    try:
        payload = json.loads(result.get("text") or "")
        return str(payload["answer"]), [str(step) for step in payload.get("next_steps", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return result.get("text") or "", []


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


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def plain(text):
    """댓글을 마크다운으로 보여 줄 때 ~~, *, # 같은 기호가 서식으로 바뀌지 않게 막는다."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!~|>:$])", r"\\\1", text)


def unclear_answer(label, result):
    """생성 모델의 답이 고친 댓글인지 게시 불가 안내인지 알 수 없는지 본다.

    지침상 고친 댓글은 next_steps가 비어 있고, 게시 불가 안내는 '게시하기 어렵다'를 쓴다.
    안내 문구 없이 다시 쓰라는 안내(next_steps)만 붙은 답은 둘 중 무엇인지 알 수 없어서,
    서버도 다시 검사와 뜻 검사를 하지 않는다(step08 rewritten_answer).
    """
    answer, next_steps = parse_answer(result)
    return label != "none" and REFUSAL not in answer and bool(next_steps)


def recommended_action(label, answer, result, classification_check=None):
    """운영자에게 보일 권장 조치를 정한다.

    문제 없음(none)으로 판정되면 원문을 그대로 올린다. 생성 모델이 문장을 바꿨어도 원문을 쓴다.
    다만 분류 확신도가 낮으면(classification_check) 사람 검토로 보낸다.
    나머지는 게시 불가 안내면 게시 보류, 고친 댓글인지 안내인지 알 수 없으면 사람 검토,
    고친 댓글 다시 검사나 뜻 검사에 걸리면 사람 검토, 그 밖에는 순화 후 게시다.
    """
    if not result or result.get("error") or not result.get("output_format_valid"):
        return "처리 실패"
    if label == "none":
        return "사람 검토" if (classification_check or {}).get("needs_review") else "그대로 게시"
    if REFUSAL in answer:
        return "게시 보류"
    if unclear_answer(label, result):
        return "사람 검토"
    if any(
        (result.get(key) or {}).get("needs_review") for key in ("answer_check", "meaning_check")
    ):
        return "사람 검토"
    return "순화 후 게시"


def review_reasons(label, result, classification_check=None):
    """사람 검토로 보낸 이유. 권장 조치가 사람 검토일 때 화면과 결과 표에 보여 준다."""
    reasons = []
    if label == "none":
        if (classification_check or {}).get("needs_review"):
            reasons.append(f"분류 확신도 {classification_check['threshold']:.0%} 미만")
        return reasons
    if unclear_answer(label, result):
        reasons.append("고친 댓글인지 안내인지 불분명")
    check = result.get("answer_check") or {}
    if check.get("needs_review"):
        reasons.append(f"고친 댓글도 {check['label']}로 판정")
    meaning = result.get("meaning_check") or {}
    if meaning.get("needs_review"):
        reasons.append(
            "뜻이 바뀌었을 수 있음" if meaning["meaning_kept"] is False else "뜻 검사 실패"
        )
    return reasons


def meaning_summary(meaning):
    """뜻 검사 결과를 표에 넣을 짧은 문장으로 만든다. 검사하지 않았으면 빈 문자열."""
    if not meaning:
        return ""
    if meaning["meaning_kept"] is None:
        return "판정 실패"
    return "뜻을 지킴" if meaning["meaning_kept"] else f"바뀌었을 수 있음: {meaning['reason']}"


def posted_text(original, answer, action, label=None):
    """권장 조치대로 하면 실제로 올라갈 문장. 게시 보류나 처리 실패면 빈 문자열.

    none 판정은 고쳐 쓰지 않으므로, 사람 검토로 가도 원문이 게시 후보다.
    """
    if action == "그대로 게시" or (action == "사람 검토" and label == "none"):
        return original
    if action in ("순화 후 게시", "사람 검토"):
        return answer
    return ""


def read_csv_bytes(data):
    """올린 CSV를 읽는다. 엑셀에서 저장한 한국어 CSV(cp949)도 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽지 못했습니다. UTF-8로 저장해 다시 올려 주세요.")


def batch_row(text, response):
    """대량 검수 결과 한 줄."""
    row = {
        "원문": text,
        "권장 조치": "처리 실패",
        "검토 이유": "",
        "라벨": "",
        "분류 확신도": None,
        "게시할 댓글": "",
        "글쓴이 안내": "",
        "다시 검사": "",
        "뜻 검사": "",
        "생성 모델 답": "",
        "처리 시간(초)": round(response["seconds"], 2),
        "생성 모델": "",
        "상태": "",
        "오류": "",
    }
    body = response["body"] or {}
    if response["code"] is None:
        row["오류"] = f"서버에 연결하지 못함 ({response['error']})"
        return row
    if response["code"] == 422:
        row["오류"] = "입력 오류: " + input_error(body)
        return row
    if "classification" not in body:  # 분류·프롬프트 단계 실패(500) 등
        row["오류"] = f"서버 오류 (HTTP {response['code']}) {body.get('detail', '')}".strip()
        return row
    classification = body.get("classification") or {}
    classification_check = body.get("classification_check")  # none인데 확신도가 낮은지
    result = body.get("result") or {}
    answer, next_steps = parse_answer(result)
    check = result.get("answer_check")  # 고친 댓글을 분류 모델로 다시 판정한 결과
    label = classification.get("label", "")
    action = recommended_action(label, answer, result, classification_check)
    notice = " / ".join(next_steps)
    if action == "게시 보류":  # 게시하지 않으므로 생성 모델의 안내 문장을 글쓴이에게 보낸다
        notice = " / ".join([answer, *next_steps])
    row.update(
        {
            "권장 조치": action,
            "검토 이유": " / ".join(review_reasons(label, result, classification_check))
            if action == "사람 검토"
            else "",
            "라벨": label,
            "분류 확신도": classification.get("confidence"),
            # 고친 댓글인지 안내인지 알 수 없는 답은 게시할 댓글 칸을 비운다(답은 '생성 모델 답' 칸에)
            "게시할 댓글": ""
            if unclear_answer(label, result)
            else posted_text(text, answer, action, label),
            "글쓴이 안내": notice,
            "다시 검사": f"{check['label']} ({check['confidence']:.0%})"
            if check and label != "none"
            else "",
            "뜻 검사": meaning_summary(result.get("meaning_check")),
            "생성 모델 답": answer,
            "생성 모델": result.get("model") or "",
            "상태": body.get("status", ""),
            "오류": result.get("error") or "",
        }
    )
    return row


def process_batch(url, texts, provider, workers, on_progress=None):
    """댓글 여러 개를 workers개씩 동시에 보내고, 입력 순서대로 결과 줄을 돌려준다."""
    rows = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(call_api, url, "/generate", {"text": text, "provider": provider}): index
            for index, text in enumerate(texts)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            rows[index] = batch_row(texts[index], future.result())
            if on_progress:
                on_progress(done, len(texts))
    return rows


def action_badge(action):
    color, icon = ACTIONS[action]
    st.badge(action, icon=icon, color=color)


def show_classification(classification, check=None):
    title, color, policy = LABELS.get(
        classification["label"], (classification["label"], "gray", "")
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"{title} ({classification['label']})", color=color)
        st.badge(f"분류 확신도 {classification['confidence']:.0%}", color="gray")
        if check and check["needs_review"]:
            st.badge("확신도 낮음", icon=":material/person_search:", color="orange")
    # 확신도는 분류 모델이 낸 값이다. 생성 모델은 이 라벨을 받기만 하므로 두 모델 비교에서도 하나다.
    st.caption(
        f"적용한 정책: {policy} 확신도는 분류 모델(KcELECTRA)이 이 라벨을 얼마나 확신하는지이고, "
        "생성 모델과는 관계없습니다."
    )
    if check and check["needs_review"]:
        st.warning(
            f"분류 모델은 문제 없음으로 봤지만 확신도가 {classification['confidence']:.0%}로 기준"
            f"({check['threshold']:.0%})보다 낮습니다. 비꼼처럼 분류 모델이 놓치기 쉬운 댓글일 수 "
            "있으니 게시하기 전에 사람이 확인해 주세요.",
            icon=":material/person_search:",
        )


def show_checks(label, result, action):
    """고친 댓글을 검사한 두 결과(다시 검사, 뜻 검사)를 보여 준다."""
    if label == "none" or action == "게시 보류":
        return
    if unclear_answer(label, result):  # 고친 댓글인지 알 수 없어 서버가 검사하지 않은 답
        st.warning(
            "생성 모델의 답이 지침과 맞지 않습니다. 게시 불가 안내 문구('게시하기 어렵다') 없이 다시 "
            "쓰라는 안내를 붙여서, 위 문장이 고친 댓글인지 글쓴이에게 보낼 안내인지 알 수 없습니다. "
            "그래서 다시 검사와 뜻 검사를 하지 않았습니다. 사람이 확인해 주세요.",
            icon=":material/help:",
        )
        return
    check = result.get("answer_check")  # 고친 댓글을 분류 모델로 다시 판정한 결과
    if check and check["needs_review"]:
        check_title = LABELS.get(check["label"], (check["label"],))[0]
        st.warning(
            f"다시 검사: 고친 댓글도 분류 모델이 {check_title}({check['label']}, "
            f"{check['confidence']:.0%})로 봤습니다. 게시하기 전에 사람이 확인해 주세요.",
            icon=":material/person_search:",
        )
    elif check:
        st.caption(f"다시 검사: 고친 댓글은 문제 없음(none, {check['confidence']:.0%})")
    meaning = result.get("meaning_check")  # 고친 댓글이 원문의 뜻을 지켰는지 판정 모델이 본 결과
    if meaning and meaning["meaning_kept"] is False:
        st.warning(
            f"뜻 검사: 고친 댓글이 원문의 뜻과 달라졌을 수 있습니다. 판정 이유: {plain(meaning['reason'])}",
            icon=":material/compare_arrows:",
        )
    elif meaning and meaning["needs_review"]:
        st.warning(
            "뜻 검사를 하지 못했습니다(판정 모델 응답 오류). 게시하기 전에 사람이 확인해 주세요.",
            icon=":material/help:",
        )
    elif meaning:
        st.caption("뜻 검사: 원문의 뜻을 지켰다고 판정했습니다. 판정 모델도 틀릴 수 있습니다.")


def show_result(original, label, result, status, title=None, classification_check=None):
    """생성 모델 하나의 결과 카드."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        if result.get("error"):
            action_badge("처리 실패")
            st.error(
                f"생성 모델 호출에 실패했습니다 ({result['error']}). 분류 결과는 위와 같습니다.",
                icon=":material/error:",
            )
            return
        answer, next_steps = parse_answer(result)
        action = recommended_action(label, answer, result, classification_check)
        action_badge(action)
        if action == "게시 보류":
            st.markdown("**글쓴이에게 보낼 안내**")
            with st.container(border=True):
                st.markdown(plain(answer))
                for step in next_steps:
                    st.markdown(f":material/arrow_right: {plain(step)}")
        elif unclear_answer(label, result):  # 안내일 수도 있으므로 게시할 댓글로 보이지 않는다
            st.markdown("**생성 모델의 답** (게시할 댓글인지 글쓴이 안내인지 사람이 정합니다)")
            with st.container(border=True):
                st.markdown(plain(answer) or "(빈 답변)")
                for step in next_steps:
                    st.markdown(f":material/arrow_right: {plain(step)}")
        else:
            st.markdown("**게시할 댓글**")
            with st.container(border=True):
                st.markdown(plain(posted_text(original, answer, action, label)) or "(빈 답변)")
            for step in next_steps:
                st.markdown(f":material/arrow_right: {plain(step)}")
        if label == "none" and normalized(answer) != normalized(original):
            # 생성 모델이 문제 없는 댓글까지 고친 경우. 원문을 그대로 쓴다.
            st.caption(
                f'생성 모델이 "{plain(answer)}"(으)로 바꿨지만, 문제 없는 댓글이라 원문을 그대로 올립니다.'
            )
        show_checks(label, result, action)
        meaning = result.get("meaning_check")
        times = f"{result['latency_seconds']:.2f}초"
        if meaning:
            times += f" · 뜻 검사 {meaning['latency_seconds']:.2f}초"
        st.caption(f"{STATUS.get(status, status)} · {result.get('model')} · {times}")


def single_tab(url, provider):
    left, right = st.columns([5, 7], gap="large")
    with left, st.container(border=True):
        st.markdown("**검수할 댓글**")
        text = st.text_area(
            "검수할 댓글",
            height=180,
            max_chars=4000,
            placeholder="게시 전에 확인할 댓글을 붙여 넣으세요.",
            label_visibility="collapsed",
        )
        compare = st.toggle("로컬과 상용 결과 나란히 비교", key="compare")
        st.caption(
            "비교를 켜면 두 생성 모델에 같은 판정과 같은 지침을 보냅니다."
            if compare
            else f"생성 모델: {PROVIDER_MODELS[provider]} (왼쪽 메뉴에서 바꿀 수 있습니다)"
        )
        if st.button("검수하기", type="primary", icon=":material/fact_check:", width="stretch"):
            with st.spinner("분류하고 고쳐 쓰는 중입니다."):
                if compare:
                    response = call_api(url, "/compare", {"text": text})
                else:
                    response = call_api(url, "/generate", {"text": text, "provider": provider})
            st.session_state["single"] = {"text": text.strip(), "response": response}

    with right:
        saved = st.session_state.get("single")
        if not saved:
            with st.container(border=True):
                st.markdown("**검수 결과**")
                st.caption(
                    "댓글을 넣고 검수하기를 누르면 판정, 권장 조치, 게시할 댓글이 여기에 나옵니다."
                )
            return
        response, original = saved["response"], saved["text"]
        body = response["body"] or {}
        if response["code"] is None:
            st.error(
                f"서버에 연결하지 못했습니다 ({response['error']}). 왼쪽 메뉴의 안내대로 서버를 먼저 켜 주세요.",
                icon=":material/cloud_off:",
            )
            return
        if response["code"] == 422:
            st.warning("입력을 확인해 주세요: " + input_error(body), icon=":material/edit:")
            return
        if "classification" not in body:  # 분류·프롬프트 단계 실패(500) 등
            st.error(
                f"서버가 처리하지 못했습니다 (HTTP {response['code']}). {body.get('detail', '')}",
                icon=":material/error:",
            )
            return
        label = body["classification"]["label"]
        check = body.get("classification_check")  # none인데 확신도가 낮은지
        with st.container(border=True):
            st.markdown("**판정**")
            show_classification(body["classification"], check)
        if "results" in body:  # 두 모델 비교
            columns = st.columns(len(body["results"]))
            for column, result in zip(columns, body["results"], strict=True):
                with column:
                    name = "로컬" if result["provider"] == "ollama" else "상용"
                    show_result(original, label, result, result.get("status"), name, check)
        else:
            show_result(original, label, body["result"], body.get("status"), None, check)


def batch_tab(url, provider):
    with st.container(border=True):
        st.markdown("**CSV 올리기**")
        uploaded = st.file_uploader(
            "CSV 파일",
            type="csv",
            help="댓글이 한 열에 들어 있는 CSV. 엑셀에서 저장한 파일도 됩니다.",
            label_visibility="collapsed",
        )
        if not uploaded:
            st.caption("댓글이 한 열에 들어 있는 CSV를 올리면 줄마다 판정하고 고쳐 씁니다.")
        else:
            try:
                frame = read_csv_bytes(uploaded.getvalue())
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
                return
            if frame.empty:
                st.warning("CSV에 행이 없습니다.")
                return
            columns = list(frame.columns)
            column_box, limit_box, workers_box = st.columns(3)
            column = column_box.selectbox(
                "댓글 열", columns, index=columns.index("text") if "text" in columns else 0
            )
            limit = limit_box.number_input(
                "처리할 건수", min_value=1, max_value=len(frame), value=min(len(frame), 50)
            )
            workers = workers_box.number_input(
                "동시 처리 수",
                min_value=1,
                max_value=8,
                value=DEFAULT_WORKERS,
                help="서버에 한 번에 보내는 요청 수입니다. 로컬 모델은 4개일 때 한 줄씩보다 1.5배쯤 빨랐습니다.",
            )
            st.caption(
                f"{len(frame):,}행 중 앞에서 {int(limit):,}건을 {PROVIDER_MODELS[provider]}(으)로 처리합니다."
            )
            if st.button("대량 검수 시작", type="primary", icon=":material/play_arrow:"):
                texts = frame[column].head(int(limit)).fillna("").astype(str).tolist()
                progress = st.progress(0.0, text="처리를 시작합니다.")
                started = time.perf_counter()
                rows = process_batch(
                    url,
                    texts,
                    provider,
                    int(workers),
                    lambda done, total: progress.progress(
                        done / total, text=f"{done}/{total}건 처리"
                    ),
                )
                progress.empty()
                st.session_state["batch"] = pd.DataFrame(rows)
                st.session_state["batch_run"] = {
                    "seconds": time.perf_counter() - started,
                    "workers": int(workers),
                }
                st.toast(f"{len(rows)}건을 처리했습니다.", icon=":material/task_alt:")

    result = st.session_state.get("batch")
    if result is None:
        return
    run = st.session_state.get("batch_run", {})
    counts = result["권장 조치"].value_counts()
    latencies = [value for value in result["처리 시간(초)"] if pd.notna(value)]
    with st.container(horizontal=True):
        st.metric("처리 건수", f"{len(result):,}", border=True)
        for action in ("그대로 게시", "순화 후 게시", "게시 보류", "사람 검토"):
            st.metric(action, f"{int(counts.get(action, 0)):,}", border=True)
        st.metric(
            "처리 시간",
            f"{run['seconds']:.1f}초" if run else "-",
            border=True,
            help=(
                f"동시 {run.get('workers', '-')}개로 보낸 전체 시간입니다. "
                + (
                    f"한 건이 끝나기까지 평균 {statistics.mean(latencies):.2f}초 걸렸습니다"
                    "(서버에서 차례를 기다린 시간 포함)."
                    if latencies
                    else ""
                )
            ),
        )
    chart, table = st.columns([1, 2], gap="medium")
    with chart, st.container(border=True):
        st.markdown("**권장 조치별 건수**")
        order = [action for action in ACTIONS if counts.get(action, 0)]
        st.bar_chart(
            pd.DataFrame({"권장 조치": order, "건수": [int(counts[action]) for action in order]}),
            x="권장 조치",
            y="건수",
            x_label="",
            y_label="",
            sort=False,
            horizontal=True,
            alt="권장 조치별 댓글 건수",
        )
    with table, st.container(border=True):
        st.markdown("**결과 표**")
        st.caption(
            "사람 검토로 보낸 이유는 '검토 이유' 열에 있습니다. 분류 확신도가 낮은 none 판정, "
            "고친 댓글에도 문제 표현이 남은 줄, 뜻이 바뀌었을 수 있는 줄입니다."
        )
        st.dataframe(
            result,
            hide_index=True,
            alt="대량 검수 결과",
            # 줄이 적으면 스크롤 없이 모두 보이게 한다
            height="content" if len(result) <= 20 else 600,
            column_config={
                "원문": st.column_config.TextColumn(pinned=True),
                "검토 이유": st.column_config.TextColumn(width="medium"),
                "분류 확신도": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0
                ),
                "처리 시간(초)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "결과 CSV 내려받기",
            result.to_csv(index=False).encode(
                "utf-8-sig"
            ),  # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
            file_name="comments_result.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def policy_tab():
    with st.container(border=True):
        st.markdown("**라벨별 처리 기준**")
        st.table(
            {f"{title} ({label})": policy for label, (title, _, policy) in LABELS.items()},
            border="horizontal",
            alt="라벨별 처리 정책",
        )
    with st.container(border=True):
        st.markdown("**권장 조치 기준**")
        st.table(
            {
                "그대로 게시": "문제 없음으로 판정된 댓글. 생성 모델이 문장을 바꿨어도 원문을 그대로 올림",
                "순화 후 게시": "문제 표현을 고쳐 쓴 댓글",
                "게시 보류": "남길 의견이 없어 게시하기 어렵다고 안내한 댓글",
                "사람 검토": "아래 세 검사 가운데 하나라도 걸린 댓글. 생성 모델이 '게시하기 어렵다' "
                "없이 다시 쓰라는 안내를 붙여 고친 댓글인지 안내인지 알 수 없는 답도 사람 검토",
                "처리 실패": "생성 모델 호출 실패, 잘린 답변, 형식 오류",
            },
            border="horizontal",
            alt="권장 조치 기준",
        )
    with st.container(border=True):
        st.markdown("**사람 검토로 보내는 검사**")
        st.table(
            {
                "분류 확신도": "문제 없음(none)으로 판정했지만 확신도가 93% 미만. "
                "none은 고치지 않고 그대로 게시되므로 먼저 확인",
                "다시 검사": "고친 댓글을 같은 분류 모델로 다시 판정해 none이 아님",
                "뜻 검사": f"판정 모델({JUDGE_MODEL})이 원문과 고친 댓글을 비교해 뜻이 바뀌었다고 봄",
            },
            border="horizontal",
            alt="사람 검토로 보내는 검사",
        )
    st.caption(
        "판정은 KcELECTRA 분류 모델(test Macro F1 0.666)이 합니다. 최종 비교에서 세 검사는 나쁜 답 "
        "20개 중 16개를 게시 전에 잡았고, 좋은 답 54개 중 7개를 사람 검토로 보냈습니다. 검사도 틀릴 "
        "수 있으니 중요한 게시물은 사람이 한 번 더 봐 주세요."
    )


def api_tab(url):
    with st.container(border=True):
        st.markdown("**서버 직접 호출 (Python)**")
        st.code(
            "import json, httpx\n\n"
            f'response = httpx.post("{url.rstrip("/")}/generate",\n'
            '                      json={"text": "검수할 댓글", "provider": "ollama"}, timeout=180)\n'
            "body = response.json()\n"
            'body["status"]                              # ok, incomplete, invalid_format, generation_error\n'
            'body["classification"]["label"]             # none, offensive, hate\n'
            'json.loads(body["result"]["text"])["answer"]  # 게시할 댓글이나 안내 문장\n'
            "# 아래 셋 중 하나라도 True면 사람 검토. 검사하지 않은 항목은 None\n"
            'body["classification_check"]["needs_review"]   # none인데 확신도가 낮음\n'
            'body["result"]["answer_check"]["needs_review"]   # 고친 댓글에도 문제 표현\n'
            'body["result"]["meaning_check"]["needs_review"]  # 뜻이 바뀌었을 수 있음\n'
            "# 두 검사는 고친 댓글에만 함. answer에 '게시하기 어렵'이 없는데 next_steps가 있으면\n"
            "# 고친 댓글인지 안내인지 알 수 없어 검사하지 않음(None). 이 답도 사람 검토로 보냄\n"
            "# 팀 공통 파이프라인 결과 원본 (schema_version 2)\n"
            'body["pipeline"]["results"][0]["service_output"]  # answer, next_steps',
            language="python",
        )
    with st.container(border=True):
        st.markdown("**엔드포인트와 응답 코드**")
        st.table(
            {
                "GET /health": "서버가 켜져 있는지 확인",
                "POST /generate": "댓글 하나를 판정하고 고쳐 씀. provider는 ollama(기본) 또는 openai",
                "POST /compare": "같은 판정과 지침으로 두 생성 모델의 결과를 함께 돌려줌",
                "200": "판정과 생성이 끝남. 답을 그대로 써도 되는지는 status로 확인",
                "422": "잘못된 입력(빈 입력, 4,000자 초과, 없는 provider). 판정 전에 거절",
                "500": "서버 안의 분류·프롬프트 단계 실패. detail에 실패한 단계를 알려 줌",
                "502": "생성 모델 호출 실패. 판정 결과는 함께 돌려줌",
            },
            border="horizontal",
            alt="API 엔드포인트와 응답 코드",
        )
    st.caption(
        f"API 문서 화면: {url.rstrip('/')}/docs · 응답 형식 전체: 저장소의 docs/COMMENTS_API.md"
    )


def main():
    st.set_page_config(page_title="댓글 검수 콘솔", page_icon=":material/shield:", layout="wide")
    with st.sidebar:
        st.markdown("**API 서버**")
        url = st.text_input("서버 주소", API_URL)
        health = call_api(url, "/health")
        if health["code"] == 200:
            st.badge("연결됨", icon=":material/check_circle:", color="green")
        else:
            st.badge("연결 안 됨", icon=":material/error:", color="red")
            st.code("uv run python steps/step10_serve_api.py", language="powershell")
            st.caption("다른 터미널에서 위 명령으로 서버를 먼저 켜 주세요.")
        st.markdown("**생성 모델**")
        choice = st.segmented_control(
            "생성 모델",
            list(PROVIDERS),
            default="로컬",
            required=True,
            key="provider",
            label_visibility="collapsed",
        )
        provider = PROVIDERS[choice]
        st.caption(f"{PROVIDER_MODELS[provider]}로 고쳐 씁니다.")
        st.caption(
            "서버를 켠 뒤 첫 요청은 생성 모델과 뜻 검사 모델을 메모리에 올리느라 12초쯤 걸립니다."
        )

    st.title("댓글 검수 콘솔", icon=":material/shield:")
    st.caption(
        "악성댓글을 판정하고, 의견은 살린 채 순화합니다. "
        "판정은 직접 학습한 분류 모델(KcELECTRA), 순화는 생성 모델이 맡습니다."
    )
    single, batch, policy, api = st.tabs(
        [
            ":material/fact_check: 단건 검수",
            ":material/table_view: 대량 검수",
            ":material/policy: 처리 기준",
            ":material/code: API 연동",
        ]
    )
    with single:
        single_tab(url, provider)
    with batch:
        batch_tab(url, provider)
    with policy:
        policy_tab()
    with api:
        api_tab(url)


if __name__ == "__main__":
    main()
