"""블라인드 채점 결과를 정답 열쇠와 합쳐 모델별 점수를 계산한다.

- 채점표(eval_blind_scoring.csv)의 0/1을 열쇠(eval_blind_key.csv)로 모델에 연결한다.
- 원본 결과(eval_raw.jsonl)에서 분류기 정답 여부, 시간, 비용, 지시 위반을 가져온다.
- 분류기가 판정을 맞힌 문제만 따로 집계해 공정하게 비교한다.

실행: uv run python nli/score_llms.py
결과: results/llm_compare/eval_final_scores.md
"""

import csv
import json
from statistics import mean

import config

OUT_DIR = config.ROOT / "results" / "llm_compare"
CRITERIA = ["근거충실성(0/1)", "필수내용(0/1)", "금지내용없음(0/1)"]


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    scores = read_csv(OUT_DIR / "eval_blind_scoring.csv")
    key = {(r["case_id"], r["답변"]): r["provider"] for r in read_csv(OUT_DIR / "eval_blind_key.csv")}
    raw = {(r["case_id"], r["provider"]): r for r in
           (json.loads(line) for line in (OUT_DIR / "eval_raw.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip())}

    # 1. 채점 칸 확인: 모두 0 또는 1이어야 한다
    problems = []
    for row in scores:
        for c in CRITERIA:
            if (row.get(c) or "").strip() not in ("0", "1"):
                problems.append(f"{row['case_id']} {row['답변']} {c}: '{row.get(c)}'")
    if problems:
        print("채점이 덜 됐거나 0/1이 아닌 칸이 있습니다:")
        for p in problems:
            print("  -", p)
        return

    # 2. 열쇠로 모델에 연결
    merged = []
    for row in scores:
        provider = key[(row["case_id"], row["답변"])]
        r = raw[(row["case_id"], provider)]
        merged.append({
            "case_id": row["case_id"], "provider": provider, "model": r["model"],
            "error_type": r["error_type"], "classifier_correct": r["classifier_correct"],
            "grounded": int(row[CRITERIA[0]]), "required": int(row[CRITERIA[1]]),
            "no_forbidden": int(row[CRITERIA[2]]),
            "violations": len(r.get("violations") or []), "seconds": r["seconds"],
            "cost": r.get("cost_usd"), "flagged": "재검토" in (r.get("explanation") or ""),
            "memo": row.get("메모", ""),
        })

    def avg(rows, field):
        return f"{mean(r[field] for r in rows):.2f}" if rows else "-"

    providers = sorted({m["provider"] for m in merged})
    lines = [
        "# 로컬 vs 상용 LLM 최종 비교 (eval 20건, 블라인드 채점)",
        "",
        "- 채점: 모델 이름을 가린 X/Y 표를 Claude(LLM)가 내용만 보고 채점, 사용자가 검토",
        "- 기준: 근거충실성, 필수내용, 금지내용없음 (각 0/1, 아래 값은 평균)",
        "- '분류기 맞힌 문제만': 분류기 판정이 정답과 같은 문제만 집계 (판정 오류의 영향 제거)",
        "",
        "| 항목 | " + " | ".join(providers) + " |",
        "| --- |" + " --- |" * len(providers),
    ]
    by = {p: [m for m in merged if m["provider"] == p] for p in providers}
    ok = {p: [m for m in by[p] if m["classifier_correct"]] for p in providers}
    wrong = {p: [m for m in by[p] if not m["classifier_correct"]] for p in providers}
    rows = [
        ("모델", lambda p: by[p][0]["model"]),
        ("채점 답변 수", lambda p: str(len(by[p]))),
        ("근거충실성 (전체)", lambda p: avg(by[p], "grounded")),
        ("필수내용 (전체)", lambda p: avg(by[p], "required")),
        ("금지내용없음 (전체)", lambda p: avg(by[p], "no_forbidden")),
        ("근거충실성 (분류기 맞힌 문제만)", lambda p: avg(ok[p], "grounded")),
        ("필수내용 (분류기 맞힌 문제만)", lambda p: avg(ok[p], "required")),
        ("근거충실성 (분류기 틀린 문제만)", lambda p: avg(wrong[p], "grounded")),
        ("지시 위반 답변 수", lambda p: str(sum(1 for m in by[p] if m["violations"]))),
        ("판정 오류 문제에서 '재검토' 표시", lambda p: f"{sum(m['flagged'] for m in wrong[p])}/{len(wrong[p])}"),
        ("응답 시간 평균", lambda p: f"{mean(m['seconds'] for m in by[p]):.2f}초"),
        ("API 비용 합계", lambda p: (f"${sum(m['cost'] for m in by[p] if m['cost']):.6f}"
                                if any(m["cost"] for m in by[p]) else "없음 (로컬)")),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(p) for p in providers) + " |")

    lines += ["", "## 오류 유형별 종합 점수 (세 기준 평균)", "",
              "| 오류 유형 | " + " | ".join(providers) + " |", "| --- |" + " --- |" * len(providers)]
    for etype in dict.fromkeys(m["error_type"] for m in merged):
        cells = []
        for p in providers:
            rs = [m for m in by[p] if m["error_type"] == etype]
            cells.append(f"{mean((m['grounded'] + m['required'] + m['no_forbidden']) / 3 for m in rs):.2f}")
        lines.append(f"| {etype} | " + " | ".join(cells) + " |")

    memos = [m for m in merged if m["memo"].strip()]
    if memos:
        lines += ["", "## 채점 메모", ""]
        lines += [f"- {m['case_id']} ({m['provider']}): {m['memo']}" for m in memos]

    path = OUT_DIR / "eval_final_scores.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n저장:", path)


if __name__ == "__main__":
    main()