"""댓글 검수 콘솔(scripts/comments_streamlit_app.py)의 권장 조치 규칙.

Streamlit은 선택 설치 묶음(ui)에만 있으므로 없으면 건너뛴다.
uv run --group ui python -m pytest tests/test_comments_console.py
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comments_streamlit_app as app

LOW = {"threshold": 0.93, "needs_review": True}  # none인데 확신도가 낮음
SURE = {"threshold": 0.93, "needs_review": False}
CLEAN = {"label": "none", "confidence": 0.9, "needs_review": False}  # 다시 검사 통과
TOXIC = {"label": "hate", "confidence": 0.8, "needs_review": True}  # 다시 검사에 걸림
KEPT = {
    "meaning_kept": True,
    "reason": "대상과 평가가 같다.",
    "needs_review": False,
    "model": "judge-only",
    "latency_seconds": 1.2,
    "error": None,
}
CHANGED = {**KEPT, "meaning_kept": False, "reason": "불만이 감사로 바뀌었다.", "needs_review": True}


def generated(answer="고친 댓글", answer_check=None, meaning_check=None, next_steps=()):
    return {
        "error": None,
        "output_format_valid": True,
        "text": json.dumps({"answer": answer, "next_steps": list(next_steps)}, ensure_ascii=False),
        "answer_check": answer_check,
        "meaning_check": meaning_check,
        "latency_seconds": 1.0,
        "model": "test-only",
    }


def test_none_is_posted_as_is_unless_confidence_is_low():
    original = generated("원문")
    assert app.recommended_action("none", "원문", original, SURE) == "그대로 게시"
    assert app.recommended_action("none", "원문", original, LOW) == "사람 검토"
    assert app.review_reasons("none", original, LOW) == ["분류 확신도 93% 미만"]
    # none은 고쳐 쓰지 않으므로 사람 검토로 가도 원문이 게시 후보다
    assert app.posted_text("원문", "바뀐 문장", "사람 검토", "none") == "원문"


def test_rewrite_goes_to_review_when_either_check_flags_it():
    passed = generated(answer_check=CLEAN, meaning_check=KEPT)
    changed = generated(answer_check=CLEAN, meaning_check=CHANGED)
    both = generated(answer_check=TOXIC, meaning_check=CHANGED)
    assert app.recommended_action("offensive", "고친 댓글", passed, SURE) == "순화 후 게시"
    assert app.recommended_action("offensive", "고친 댓글", changed, SURE) == "사람 검토"
    assert app.review_reasons("offensive", both) == [
        "고친 댓글도 hate로 판정",
        "뜻이 바뀌었을 수 있음",
    ]
    unreadable = generated(meaning_check={**KEPT, "meaning_kept": None, "needs_review": True})
    assert app.review_reasons("hate", unreadable) == ["뜻 검사 실패"]
    # 게시 불가 안내는 검사 결과와 관계없이 게시 보류
    refusal = "이 댓글은 그대로 게시하기 어렵습니다."
    assert app.recommended_action("hate", refusal, generated(refusal, TOXIC), SURE) == "게시 보류"
    guided = generated(refusal, next_steps=["표현을 고쳐 다시 써 주세요."])
    assert app.recommended_action("hate", refusal, guided, SURE) == "게시 보류"


def test_answer_with_next_steps_but_no_refusal_goes_to_review():
    # 지침상 고친 댓글은 next_steps가 비어 있다. 안내 문구 없이 다시 쓰라는 안내만 붙으면
    # 고친 댓글인지 안내인지 알 수 없으므로, 서버는 검사하지 않고(None) 화면은 사람 검토로 보낸다
    advice = "운전 실력보다는 사고 원인에 대해 이야기하는 것이 좋겠습니다."
    unclear = generated(advice, next_steps=["댓글을 다시 작성해 주세요."])
    assert app.unclear_answer("hate", unclear) is True
    assert app.recommended_action("hate", advice, unclear, SURE) == "사람 검토"
    assert app.review_reasons("hate", unclear) == ["고친 댓글인지 안내인지 불분명"]
    body = {
        "status": "ok",
        "classification": {"label": "hate", "confidence": 0.68},
        "classification_check": SURE,
        "result": unclear,
    }
    row = app.batch_row("원문", {"code": 200, "body": body, "error": None, "seconds": 1.5})
    assert row["권장 조치"] == "사람 검토"
    assert row["검토 이유"] == "고친 댓글인지 안내인지 불분명"
    assert row["다시 검사"] == ""
    assert row["글쓴이 안내"] == "댓글을 다시 작성해 주세요."
    # none은 고쳐 쓰지 않으므로 next_steps가 있어도 이 규칙과 상관없다
    assert app.unclear_answer("none", unclear) is False
    assert app.unclear_answer("hate", generated(advice)) is False


def test_batch_row_explains_why_a_comment_needs_review():
    body = {
        "status": "ok",
        "classification": {"label": "offensive", "confidence": 0.8},
        "classification_check": SURE,
        "result": generated(answer_check=CLEAN, meaning_check=CHANGED),
    }
    row = app.batch_row("원문", {"code": 200, "body": body, "error": None, "seconds": 2.5})
    assert row["권장 조치"] == "사람 검토"
    assert row["검토 이유"] == "뜻이 바뀌었을 수 있음"
    assert row["뜻 검사"] == "바뀌었을 수 있음: 불만이 감사로 바뀌었다."
    assert row["게시할 댓글"] == "고친 댓글"
    assert row["처리 시간(초)"] == 2.5
    offline = app.batch_row(
        "원문", {"code": None, "body": None, "error": "ConnectError", "seconds": 0.1}
    )
    assert offline["권장 조치"] == "처리 실패"
    assert "ConnectError" in offline["오류"]
    # 서버 안의 분류 단계 실패(500)는 판정 없이 실패로 남기고 이유를 보여 준다
    detail = "classification 단계에서 실패했습니다 (RuntimeError)."
    failed = app.batch_row(
        "원문", {"code": 500, "body": {"detail": detail}, "error": None, "seconds": 0.2}
    )
    assert failed["권장 조치"] == "처리 실패"
    assert failed["오류"] == f"서버 오류 (HTTP 500) {detail}"
