"""후처리 전후 비교: 같은 답에서 후처리로 실제로 바뀐 문장만 보여준다.

대체 경로 확인 실험 결과(stress_compare_raw.jsonl)를 읽어서,
local_4b와 local_4b+후처리(8B도 같이)의 설명·수정안이 다른 경우만 모은다.

실행: uv run python nli/show_postprocess.py
결과: results/llm_compare/postprocess_diff.md
"""

import json

import config

RAW = config.ROOT / "results" / "llm_compare" / "stress_compare_raw.jsonl"
OUT = config.ROOT / "results" / "llm_compare" / "postprocess_diff.md"
PAIRS = [("local_4b", "local_4b+후처리"), ("local_8b", "local_8b+후처리")]


def main():
    records = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["variant"], r["id"]): r for r in records}

    lines = ["# 후처리 전후 비교 (대체 경로 확인 실험 60건)", "",
             "- 같은 LLM 답에 후처리만 적용한 것이라, 바뀐 곳은 금지 표현뿐이다.",
             "- 후처리 규칙: '명확한/명백한 오류' → '원문과 다른 부분입니다', "
             "'틀린 내용' → '원문과 다른 내용', '틀렸다' → '원문과 다르다'", ""]
    total = 0
    for before_name, after_name in PAIRS:
        ids = sorted({key[1] for key in by if key[0] == before_name})
        changed = []
        for case_id in ids:
            before, after = by[(before_name, case_id)], by[(after_name, case_id)]
            for field, label in (("explanation", "설명"), ("suggestion", "수정안")):
                if (before[field] or "") != (after[field] or ""):
                    changed.append((case_id, before["report"], label, before[field], after[field]))
        total += len(changed)
        lines += [f"## {before_name} → {after_name}: 바뀐 곳 {len(changed)}개", ""]
        if not changed:
            lines += ["(바뀐 곳 없음)", ""]
            continue
        for case_id, report, label, old, new in changed:
            lines += [f"### {case_id} ({label})", "",
                      f"- 보고서 문장: {report}",
                      f"- **후처리 전:** {old}",
                      f"- **후처리 후:** {new}", ""]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"후처리로 바뀐 곳: 총 {total}개")
    print("저장:", OUT)


if __name__ == "__main__":
    main()