"""LLM 비교 실험(dev~eval)의 조건·결과·분석을 한 보고서로 모은다.

블라인드 채점을 지키기 위해 모델별 답변 내용은 넣지 않는다.
(자동 집계, 실행 조건, 분류기 분석, 파일 목록만 포함)

실행: uv run python nli/report_llm_compare.py
결과: results/llm_compare/REPORT.md
"""

import csv
import json
from collections import Counter

import config

CASES_PATH = config.ROOT / "data" / "llm_compare" / "cases.jsonl"
OUT_DIR = config.ROOT / "results" / "llm_compare"
REPORT_PATH = OUT_DIR / "REPORT.md"

# 분류기 오답 원인 (eval_classifier_errors.md를 보고 사람이 분류, 2026-10-13)
ERROR_CAUSES = {
    "A. 근거 찾기 실패 (엉뚱한 문장을 근거로 고름)": ["eval-01"],
    "B. 정보가 원문 두 문장에 나뉨 (근거를 한 문장만 사용)": ["eval-03", "eval-07", "eval-19"],
    "C. 분류기 판단 실패 (근거는 맞게 찾음)": ["eval-02", "eval-08", "eval-09",
                                         "eval-10", "eval-12", "eval-13"],
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else None


def body_of(markdown):
    """맨 앞 제목 줄을 빼고 본문만 돌려준다 (보고서 안에 끼워 넣기 위해)."""
    lines = markdown.splitlines()
    return "\n".join(lines[1:]).strip() if lines and lines[0].startswith("#") else markdown


def scoring_status():
    """블라인드 채점표가 얼마나 채워졌는지 센다."""
    path = OUT_DIR / "eval_blind_scoring.csv"
    if not path.exists():
        return "채점표 없음"
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    keys = ["근거충실성(0/1)", "필수내용(0/1)", "금지내용없음(0/1)"]
    done = sum(all((row.get(k) or "").strip() for k in keys) for row in rows)
    return f"{done}/{len(rows)}개 답변 채점 완료"


def main():
    cases = read_jsonl(CASES_PATH)
    lines = [
        "# LLM 비교 실험 보고서 (로컬 vs 상용)",
        "",
        "이 문서는 `nli/report_llm_compare.py`가 결과 파일들을 읽어 자동으로 만든다.",
        "블라인드 채점을 지키기 위해 모델별 답변 내용은 넣지 않았다.",
        "",
        "## 1. 실험 설계",
        "",
        "- 목적: 같은 입력과 같은 지시문으로 로컬 LLM과 상용 LLM의 설명·수정안 품질, 속도, 비용을 비교",
        "- 흐름: 원문 + 보고서 문장 → 분류기 판정 → 판정별 지시문 → 각 LLM이 설명·수정안 작성",
        "- 두 LLM은 같은 분류기 판정을 받는다. 채점용 정답은 LLM에 주지 않는다",
        "- 데이터: 외부 API 전송이 가능하도록 직접 작성한 가상 회의록 (실제 문서가 아님)",
        "- dev(5건)는 연결·형식 점검용으로 반복 실행, eval(20건)은 조건을 고정한 뒤 1회만 실행",
        "- dev와 eval은 원문이 겹치지 않는다 (dev: A~E, eval: F~J)",
        "",
        "### 문제 구성",
        "",
        "| 구분 | dev | eval |",
        "| --- | --- | --- |",
    ]
    for key, title in (("expected_label", "기대 판정"), ("error_type", "오류 유형"), ("style", "문장 스타일")):
        values = sorted({c.get(key) or "-" for c in cases})
        for value in values:
            counts = [sum(1 for c in cases if c["split"] == s and (c.get(key) or "-") == value)
                      for s in ("dev", "eval")]
            lines.append(f"| {title}: {value} | {counts[0]} | {counts[1]} |")

    # 2. 실행 조건
    lines += ["", "## 2. 실행 조건", ""]
    for split in ("dev", "eval"):
        info = read_text(OUT_DIR / f"{split}_run_info.json")
        if info:
            data = json.loads(info)
            lines += [
                f"### {split}",
                "",
                f"- 실행 시각: {data.get('run_at')}",
                f"- 분류기: {data.get('classifier')} (T = {data.get('temperature')})",
                f"- LLM: {data.get('providers')}",
                f"- 최대 출력 토큰: {data.get('max_output_tokens')}",
                f"- 단가(USD/100만 토큰): {data.get('price_usd_per_1m')}",
                f"- 문제 파일 지문: `{data.get('cases_file_sha256', '')[:16]}...`",
                f"- 지시문 파일 지문: `{data.get('prompt_file_sha256', '')[:16]}...`",
                "",
            ]
    dev_info = read_text(OUT_DIR / "dev_run_info.json")
    eval_info = read_text(OUT_DIR / "eval_run_info.json")
    if dev_info and eval_info:
        same = (json.loads(dev_info)["prompt_file_sha256"] == json.loads(eval_info)["prompt_file_sha256"])
        lines.append(f"- dev와 eval의 지시문 파일 지문 일치: **{'예' if same else '아니오'}**")

    # 3. 자동 집계
    for split, title in (("dev", "3. dev 결과 (점검용)"), ("eval", "4. eval 결과 (최종 비교, 1회)")):
        summary = read_text(OUT_DIR / f"{split}_summary.md")
        lines += ["", f"## {title}", ""]
        lines.append(body_of(summary) if summary else "(결과 없음)")

    # 5. 분류기 분석
    lines += ["", "## 5. eval 분류기 오답 분석", ""]
    raw = OUT_DIR / "eval_raw.jsonl"
    if raw.exists():
        first = {}
        for r in read_jsonl(raw):
            first.setdefault(r["case_id"], r)
        wrong = [r for r in first.values() if not r["classifier_correct"]]
        dangerous = [r["case_id"] for r in wrong
                     if r["expected_label"] == "contradiction" and r["classifier_label"] == "entailment"]
        lines += [f"- 틀린 문제: {len(wrong)}/{len(first)}건", "", "| 원인 | 문제 | 수 |", "| --- | --- | --- |"]
        for cause, ids in ERROR_CAUSES.items():
            lines.append(f"| {cause} | {', '.join(ids)} | {len(ids)} |")
        lines += [
            "",
            f"- **위험한 오답 (모순인데 '뒷받침함'으로 통과): {', '.join(dangerous) or '없음'}**",
            "- 나머지 오답은 '확인 불가'로 판정되어 사용자 검토 대상으로 표시됨 (안전한 실패)",
            "- 오답의 확신도가 0.80~0.92로 높음: KLUE에서 맞춘 보정(T)이 회의록 문체에서는 통하지 않음",
            "- 상세 표: eval_classifier_errors.md",
        ]

    # 6. 블라인드 채점 상태
    lines += [
        "", "## 6. 블라인드 채점", "",
        f"- 상태: {scoring_status()}",
        "- 채점표: eval_blind_scoring.csv (모델 이름 대신 X/Y, 문제마다 순서 무작위)",
        "- 정답 열쇠: eval_blind_key.csv (채점이 끝난 뒤에만 연다)",
        "- 기준: 근거충실성, 필수내용, 금지내용없음 (각 0/1)",
        "- 한계: 채점자 1명, 말투(~습니다 / ~다)로 모델을 짐작할 수 있음",
    ]

    # 7. 파일 목록
    lines += ["", "## 7. 관련 파일", "", "| 파일 | 내용 |", "| --- | --- |"]
    files = [
        ("data/llm_compare/cases.jsonl", "문제 25건 (원문, 보고서 문장, 정답 기준)"),
        ("nli/explainer.py", "판정별 지시문과 LLM 호출"),
        ("nli/compare_llms.py", "비교 실험 코드"),
        ("results/llm_compare/dev_*.*", "dev 결과 (원본 응답, 채점표, 실행 조건, 요약)"),
        ("results/llm_compare/eval_*.*", "eval 결과 (원본 응답, 채점표, 실행 조건, 요약)"),
        ("results/llm_compare/eval_classifier_errors.md", "분류기 판정 분석 표"),
    ]
    for name, desc in files:
        lines.append(f"| `{name}` | {desc} |")
    archive = OUT_DIR / "archive"
    if archive.exists():
        for folder in sorted(p.name for p in archive.iterdir() if p.is_dir()):
            lines.append(f"| `results/llm_compare/archive/{folder}/` | 이전 dev 실행 결과 (보관) |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("보고서 저장:", REPORT_PATH)
    print("블라인드 채점 상태:", scoring_status())


if __name__ == "__main__":
    main()