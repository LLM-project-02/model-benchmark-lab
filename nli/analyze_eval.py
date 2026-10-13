"""eval에서 분류기 판정이 틀린 문제를 근거 문장과 함께 보여준다.

LLM 답변은 보여주지 않으므로 블라인드 채점에 영향을 주지 않는다.

실행: uv run python nli/analyze_eval.py
"""

import json

import config

CASES_PATH = config.ROOT / "data" / "llm_compare" / "cases.jsonl"
RAW_PATH = config.ROOT / "results" / "llm_compare" / "eval_raw.jsonl"
OUT_PATH = config.ROOT / "results" / "llm_compare" / "eval_classifier_errors.md"


def main():
    cases = {c["id"]: c for c in (json.loads(line) for line in
             CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip())}
    records = [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]

    # 두 LLM이 같은 판정을 받았으므로 문제당 한 번만 본다
    seen, rows = set(), []
    for r in records:
        if r["case_id"] in seen:
            continue
        seen.add(r["case_id"])
        rows.append(r)

    lines = ["# eval 분류기 판정 분석", "",
             "| id | 오류 유형 | 스타일 | 기대 | 판정 (확신도) | 맞음 | 보고서 문장 | 찾은 근거 문장 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    wrong = 0
    for r in rows:
        mark = "O" if r["classifier_correct"] else "X"
        lines.append(f"| {r['case_id']} | {r['error_type']} | {r['style']} | {r['expected_label']} "
                     f"| {r['classifier_label']} ({r['confidence']}) | {mark} "
                     f"| {r['report']} | {r['evidence']} |")
        if not r["classifier_correct"]:
            wrong += 1
            print(f"[{r['case_id']}] {r['error_type']} / {r['style']}")
            print(f"  기대 {r['expected_label']} → 판정 {r['classifier_label']} (확신도 {r['confidence']})")
            print(f"  보고서: {r['report']}")
            print(f"  근거:   {r['evidence']}")
            print(f"  메모:   {cases[r['case_id']].get('rubric_note')}\n")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"틀린 문제 {wrong}/{len(rows)}건")
    print("저장:", OUT_PATH)


if __name__ == "__main__":
    main()