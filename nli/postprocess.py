"""LLM 답변 후처리: 지시문이 금지한 단정 표현을 바꾼다.

대체 경로 확인 실험(stress_compare.py)에서 사용한 규칙과 같다.
지시문 파일(explainer.py)은 건드리지 않기 위해 따로 둔다.
"""

import re

SOFTEN = [
    (r"명[확백]한 오류(입니다|이다|다)?", "원문과 다른 부분입니다"),
    (r"틀린 내용", "원문과 다른 내용"),
    (r"틀렸다", "원문과 다르다"),
]


def soften(text):
    """금지 표현을 바꾼 문장을 돌려준다. 빈 값은 그대로 둔다."""
    if not text:
        return text
    for pattern, replacement in SOFTEN:
        text = re.sub(pattern, replacement, text)
    return text