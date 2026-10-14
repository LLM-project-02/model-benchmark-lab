"""Claude가 블라인드로 채점한 점수를 eval_blind_scoring.csv에 채워 넣는다.

채점 방식: 모델 이름을 가린 X/Y 표를 Claude(LLM)가 내용만 보고 채점, 사용자가 검토
실행: uv run python nli/apply_scores.py
"""

import csv

import config

PATH = config.ROOT / "results" / "llm_compare" / "eval_blind_scoring.csv"
COLUMNS = ["근거충실성(0/1)", "필수내용(0/1)", "금지내용없음(0/1)"]

# (case_id, 답변): (근거충실성, 필수내용, 금지내용없음, 메모)
SCORES = {
    ("eval-01", "X"): (1, 0, 1, "근거가 비용 문장이 아니어서 42억 원을 알 수 없음 (근거 찾기 실패)"),
    ("eval-01", "Y"): (1, 0, 1, "근거가 비용 문장이 아니어서 42억 원을 알 수 없음 (근거 찾기 실패)"),
    ("eval-02", "X"): (1, 1, 1, "판정 오류를 알아채고 재검토 표시, 91%로 수정"),
    ("eval-02", "Y"): (1, 1, 1, "판정 오류를 알아채고 재검토 표시"),
    ("eval-03", "X"): (1, 1, 1, "확인 불가라면서 원문과 다르다고 함 (지시 위반), 내용은 정확"),
    ("eval-03", "Y"): (0, 0, 1, "근거에 150명이 있는데 200명을 확인할 수 없다고 함"),
    ("eval-04", "X"): (1, 1, 1, ""),
    ("eval-04", "Y"): (1, 1, 1, "명확한 오류 단정 표현 (지시 위반)"),
    ("eval-05", "X"): (1, 1, 1, ""),
    ("eval-05", "Y"): (1, 1, 1, "보고서에 없는 주체·장소를 일치한다고 덧붙임 (경미)"),
    ("eval-06", "X"): (1, 1, 1, ""),
    ("eval-06", "Y"): (1, 1, 1, "명확한 오류 단정 표현 (지시 위반)"),
    ("eval-07", "X"): (1, 0, 1, "판정대로 날짜를 확인할 수 없다고 설명"),
    ("eval-07", "Y"): (0, 0, 1, "근거에 8천만 원이 있는데 그런 내용이 없다고 설명 (사실과 다름)"),
    ("eval-08", "X"): (0, 0, 1, "근거가 포함되지 않았다고 명시하는데 확인할 수 없다고 함"),
    ("eval-08", "Y"): (1, 1, 1, "확인 불가 판정이지만 반대 내용임을 정확히 짚음"),
    ("eval-09", "X"): (0, 1, 1, "근거를 인용했지만 모순을 알아채지 못함"),
    ("eval-09", "Y"): (1, 0, 1, "재검토 표시, 근거 내용은 언급 안 함"),
    ("eval-10", "X"): (1, 1, 1, "판정 오류를 알아채고 재검토 표시, 수정안 제시"),
    ("eval-10", "Y"): (1, 1, 1, "재검토 표시는 없지만 반대 내용임을 짚음"),
    ("eval-11", "X"): (1, 1, 1, ""),
    ("eval-11", "Y"): (1, 1, 1, ""),
    ("eval-12", "X"): (0, 0, 1, "근거가 데이터팀이 제공이라고 명시하는데 확인할 수 없다고 함"),
    ("eval-12", "Y"): (1, 1, 1, "주체 차이를 정확히 짚고 확인 질문"),
    ("eval-13", "X"): (1, 1, 1, "판정 오류를 알아채고 재검토 표시, 수정안 제시"),
    ("eval-13", "Y"): (1, 1, 1, "재검토 표시는 없지만 방향이 반대임을 짚음"),
    ("eval-14", "X"): (1, 1, 1, ""),
    ("eval-14", "Y"): (1, 1, 1, ""),
    ("eval-15", "X"): (1, 1, 1, ""),
    ("eval-15", "Y"): (1, 1, 1, ""),
    ("eval-16", "X"): (1, 1, 1, ""),
    ("eval-16", "Y"): (1, 1, 1, ""),
    ("eval-17", "X"): (1, 1, 1, ""),
    ("eval-17", "Y"): (1, 1, 1, ""),
    ("eval-18", "X"): (1, 1, 1, ""),
    ("eval-18", "Y"): (1, 1, 1, ""),
    ("eval-19", "X"): (1, 0, 1, "정보가 원문 두 문장에 나뉘어 근거 한 문장으로는 확인 불가 (판정 오류)"),
    ("eval-19", "Y"): (1, 0, 1, "정보가 원문 두 문장에 나뉘어 근거 한 문장으로는 확인 불가 (판정 오류)"),
    ("eval-20", "X"): (1, 1, 1, ""),
    ("eval-20", "Y"): (1, 0, 1, "근거 문장에 업체명이 없어 재검토 요청 (근거 한 문장의 한계)"),
}


def main():
    with PATH.open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filled = 0
    for row in rows:
        key = (row["case_id"], row["답변"])
        if key not in SCORES:
            print("채점 없음:", key)
            continue
        *values, memo = SCORES[key]
        for column, value in zip(COLUMNS, values):
            row[column] = str(value)
        row["메모"] = memo
        filled += 1

    with PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{filled}/{len(rows)}개 답변 채점 입력 완료:", PATH)


if __name__ == "__main__":
    main()