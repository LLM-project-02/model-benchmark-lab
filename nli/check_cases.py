"""LLM 비교용 문제 파일이 올바른 형식인지 확인한다.

실행: uv run python nli/check_cases.py
"""

import json
import re
from collections import Counter

import config

CASES_PATH = config.ROOT / "data" / "llm_compare" / "cases.jsonl"
REQUIRED = {"id", "split", "source", "report", "error_type", "expected_label",
            "required_points", "forbidden_claims"}
LABELS = {"entailment", "neutral", "contradiction"}


def main():
    lines = [line for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases, problems = [], []
    for number, line in enumerate(lines, 1):
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"{number}번째 줄: JSON 형식 오류 ({error})")
            continue
        missing = REQUIRED - set(case)
        if missing:
            problems.append(f"{case.get('id', number)}: 빠진 칸 {missing}")
        if case.get("expected_label") not in LABELS:
            problems.append(f"{case.get('id')}: 기대 판정이 올바르지 않음 ({case.get('expected_label')})")
        # 숫자가 들어간 필수 내용은 원문에 실제로 있어야 한다 (정답 오타 확인)
        for point in case.get("required_points", []):
            if re.search(r"\d", point) and point.replace(" ", "") not in case["source"].replace(" ", ""):
                problems.append(f"{case['id']}: 필수 내용 '{point}'이 원문에 없음")
        cases.append(case)

    ids = [case["id"] for case in cases]
    duplicated = [i for i, n in Counter(ids).items() if n > 1]
    if duplicated:
        problems.append(f"중복된 id: {duplicated}")

    print(f"문제 수: {len(cases)}건")
    print("분할별:", dict(Counter(case["split"] for case in cases)))
    print("기대 판정별:", dict(Counter(case["expected_label"] for case in cases)))
    print("오류 유형별:", dict(Counter(case["error_type"] for case in cases)))
    if problems:
        print("\n문제 발견:")
        for problem in problems:
            print("  -", problem)
    else:
        print("\n형식 검사 통과")


if __name__ == "__main__":
    main()