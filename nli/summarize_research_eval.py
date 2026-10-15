"""연구용: 사람이 채운 보고서 비교 평가 시트를 집계한다 (모델·LLM 호출 없음). 기준: docs/NLI_RESEARCH_EVAL.md

- 평가본(eval_sheet_<평가자>.csv)을 같은 실행 폴더의 원래 시트(eval_sheet.csv)·key(eval_key.csv)와 sheet_row로 맞춘다.
  평가 열 밖의 내용(근거·주장·출력·상태)이 원래 시트와 다르면 멈춘다. 평가값은 고치지 않는다.
- yes/no만 비율에 넣는다. 빈칸은 제외 건수로, 그 밖의 값은 잘못된 값으로 따로 센다.
  출력이 없는 행(eval_target이 no)에 적힌 평가값은 집계하지 않고 따로 알린다.
- 경로 사이 비교는 같은 example_id에서 모든 경로가 출력을 냈고 그 항목에 모두 yes/no가 있는 예제만 쓴다.
- 판정이 참고 라벨과 같은지는 사람 평가가 아니라 key의 reference_label로 따로 센다(출력이 있는 행만).
- 품질 점수를 새로 만들지 않는다.

실행(저장소 루트):
  uv run python nli/summarize_research_eval.py dev_qwen3_01 --eval-file eval_sheet_rater1.csv
결과: <실행 폴더>/human_eval_summary.json (있으면 멈춘다. --output으로 다른 이름)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_research_eval_sheet as sheets
import research_pipeline as research
import run_research_compare as runner

CRITERIA = {
    "eval_evidence_faithful": "근거 충실성 (yes 비율)",
    "eval_label_explanation_consistent": "판정과 설명의 일치 (yes 비율)",
    "eval_suggestion_valid": "수정안 타당성 (yes 비율)",
    "eval_new_facts": "새 사실 추가 (yes = 문제 발생 비율)",
}
CHECKED = [column for column in sheets.SHEET_COLUMNS if column not in research.EVAL_COLUMNS]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def summarize(original, filled, key):
    """원래 시트·평가본·key 행 목록으로 집계 결과를 만든다."""
    by_row = {row["sheet_row"]: row for row in original}
    keys = {row["sheet_row"]: row for row in key}
    if len(filled) != len(original) or set(by_row) != {row["sheet_row"] for row in filled} or set(keys) != set(by_row):
        raise ValueError("평가본·원래 시트·key의 행이 서로 맞지 않습니다.")
    rows, issues = [], {"content_diffs": [], "invalid_values": [], "values_without_output": []}
    for row in filled:
        base = by_row[row["sheet_row"]]
        diffs = [column for column in CHECKED if row.get(column, "") != base[column]]
        if diffs:
            issues["content_diffs"].append({"sheet_row": row["sheet_row"], "columns": diffs})
        target = base["eval_target"] == "yes"
        values = {}
        for column in CRITERIA:
            value = (row.get(column) or "").strip().lower()
            if value and not target:
                issues["values_without_output"].append({"sheet_row": row["sheet_row"], "column": column})
                value = ""
            if value not in ("", "yes", "no"):
                issues["invalid_values"].append({"sheet_row": row["sheet_row"], "column": column, "value": value})
                value = ""
            values[column] = value
        rows.append({"base": base, "key": keys[row["sheet_row"]], "target": target, "values": values})
    if issues["content_diffs"]:
        raise ValueError(f"평가 열 밖의 내용이 원래 시트와 다릅니다: {issues['content_diffs'][:5]}")

    paths = [path["id"] for path in research.PATHS if any(item["key"]["path"] == path["id"] for item in rows)]
    per_path = {}
    for path in paths:
        items = [item for item in rows if item["key"]["path"] == path]
        with_output = [item for item in items if item["target"]]
        entry = {"rows": len(items), "with_output": len(with_output), "criteria": {}}
        for column, title in CRITERIA.items():
            values = [item["values"][column] for item in with_output]
            yes, no = values.count("yes"), values.count("no")
            entry["criteria"][column] = {
                "title": title, "yes": yes, "no": no, "blank": values.count(""), "n": yes + no,
                "rate": yes / (yes + no) if yes + no else None,
            }
        labeled = [item for item in with_output if item["key"]["reference_label"]]
        agree = sum(1 for item in labeled if item["base"]["output_label"] == item["key"]["reference_label"])
        entry["reference_label_agreement"] = {
            "n": len(labeled), "agree": agree, "rate": agree / len(labeled) if labeled else None,
            "note": "출력 판정이 참고 라벨과 같은 비율 (자동 계산, 사람 평가 아님)",
        }
        per_path[path] = entry

    examples = sorted({item["key"]["example_id"] for item in rows})
    common = {}
    for column, title in CRITERIA.items():
        ids = [
            example for example in examples
            if all(
                any(item["key"]["example_id"] == example and item["target"] and item["values"][column]
                    for item in rows if item["key"]["path"] == path)
                for path in paths
            )
        ]
        common[column] = {
            "title": title, "example_ids": ids,
            "yes": {
                path: sum(1 for item in rows if item["key"]["path"] == path and item["key"]["example_id"] in ids
                          and item["values"][column] == "yes")
                for path in paths
            },
        }
    return {"paths": paths, "per_path": per_path, "common": common, "issues": issues,
            "rows": len(rows), "examples": len(examples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="비교 실행 폴더 이름")
    parser.add_argument("--eval-file", required=True, help="실행 폴더 안의 채운 평가본")
    parser.add_argument("--output", default="human_eval_summary.json", help="결과 파일 (실행 폴더 안)")
    args = parser.parse_args()

    run_dir = runner.OUTPUT_ROOT / args.name
    out = run_dir / args.output
    if out.exists():
        raise SystemExit(f"이미 있습니다(덮어쓰지 않음): {out}")
    summary = summarize(
        read_csv(run_dir / "eval_sheet.csv"), read_csv(run_dir / args.eval_file), read_csv(run_dir / "eval_key.csv")
    )
    summary["eval_file"] = args.eval_file
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for path in summary["paths"]:
        entry = summary["per_path"][path]
        rates = ", ".join(
            f"{c['title'].split(' (')[0]} {c['yes']}/{c['n']}" for c in entry["criteria"].values()
        )
        agreement = entry["reference_label_agreement"]
        print(f"{path}: 출력 {entry['with_output']}/{entry['rows']} · {rates} · 참고 라벨 일치 {agreement['agree']}/{agreement['n']}")
    for name, items in summary["issues"].items():
        if items:
            print(f"주의 {name}: {len(items)}건")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
