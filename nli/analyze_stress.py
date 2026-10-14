"""대체 경로 확인 실험 추가 분석 (결정에는 영향 없음, 해석용).

1. 수정안 제시: 위험 사례에서 suggestion 칸에 고친 문장을 실제로 냈는지 (설명 속 인용 제외)
2. JSON 형식이 깨진 답변이 어느 문제였는지
3. 분류기가 실제로 틀린 5건(주체 바뀜)에서 각 LLM의 재검토 표시

실행: uv run python nli/analyze_stress.py
"""

import json

import config
from compare_llms import normalize

RAW = config.ROOT / "results" / "llm_compare" / "stress_compare_raw.jsonl"
CASES = config.ROOT / "data" / "llm_compare" / "stress_cases.jsonl"
OUT = config.ROOT / "results" / "llm_compare" / "stress_compare_extra.md"
VARIANTS = ["openai", "local_4b", "local_8b"]
CLASSIFIER_MISSED = ["s07", "s08", "s27", "s28", "s30"]  # compare_two.md: 공식 분류기의 위험한 오답


def main():
    records = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    required = {c["id"]: c["required_any"] for c in
                (json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip())}
    by = {v: {r["id"]: r for r in records if r["variant"] == v} for v in VARIANTS}

    lines = ["# 대체 경로 확인 실험 추가 분석 (해석용, 결정에는 영향 없음)", "",
             "## 1. 위험 사례 40건에서 수정안(suggestion)을 실제로 냈는가", "",
             "| 항목 | " + " | ".join(VARIANTS) + " |", "| --- |" + " --- |" * len(VARIANTS)]
    for name, check in (
        ("수정안을 냄", lambda r: bool(r["suggestion"])),
        ("수정안에 바른 내용이 들어 있음", lambda r: bool(r["suggestion"]) and any(
            normalize(k) in normalize(r["suggestion"]) for k in required[r["id"]])),
    ):
        cells = []
        for v in VARIANTS:
            danger = [r for r in by[v].values() if r["group"] == "위험"]
            cells.append(f"{sum(check(r) for r in danger)}/{len(danger)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## 2. JSON 형식이 깨진 답변", ""]
    for v in VARIANTS:
        broken = [r["id"] for r in by[v].values() if not r["parse_ok"]]
        lines.append(f"- {v}: {', '.join(broken) if broken else '없음'}")

    lines += ["", "## 3. 분류기가 실제로 틀린 5건에서 재검토 표시", "",
              "| id | " + " | ".join(VARIANTS) + " |", "| --- |" + " --- |" * len(VARIANTS)]
    for case_id in CLASSIFIER_MISSED:
        lines.append(f"| {case_id} | " + " | ".join(
            "경고" if by[v][case_id]["flagged"] else "-" for v in VARIANTS) + " |")
    lines.append("| 경고 없이 통과 | " + " | ".join(
        str(sum(not by[v][c]["flagged"] for c in CLASSIFIER_MISSED)) for v in VARIANTS) + " |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()