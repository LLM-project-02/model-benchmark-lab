"""연구용: 비교 실행 결과(results.json)로 사람 평가 시트를 만든다 (모델·LLM 호출 없음).

- 평가자가 원문(있으면)·근거·주장·출력(판정 라벨·설명·수정안)을 한 행에서 함께 보게 한다.
- 평가 기준은 docs/NLI_RESEARCH_EVAL.md. 평가 열(eval_*)은 비워 둔다. 자동 점수는 만들지 않는다.
- 형식 검사를 통과한 출력이 없는 행(오류·미완료·미실행)은 eval_target=no로 표시하고 평가 칸을 비워 둔다.
- --blind(기본)이면 경로 이름을 예제마다 섞은 코드(A·B·C)로 가리고, 경로·참고 라벨은 따로 key 파일에 둔다.
  섞는 순서는 --seed로 고정한다.

실행(저장소 루트):
  uv run python nli/build_research_eval_sheet.py dev_qwen3_01
결과: <실행 폴더>/eval_sheet.csv, eval_key.csv (--no-blind면 eval_sheet.csv 하나에 경로를 함께 적는다)
"""

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_pipeline as research
import run_research_compare as runner

CODES = "ABC"
SHEET_COLUMNS = [
    "sheet_row", "example_id", "output_code", "source_text", "evidence", "claim", "output_label",
    "explanation", "suggestion", "status", "eval_target", *research.EVAL_COLUMNS,
]
KEY_COLUMNS = ["sheet_row", "example_id", "output_code", "path", "run_id", "reference_label",
               "classifier_label", "classifier_confidence"]


def build(results, blind=True, seed=0):
    """results.json 내용으로 (시트 행, key 행)을 만든다."""
    rng = random.Random(seed)
    sheet, key = [], []
    for record in results["records"]:
        example = record.get("example") or {}
        paths = list(record["paths"])
        if blind:
            rng.shuffle(paths)
        for index, path in enumerate(paths):
            result = path["result"] or {}
            output = result.get("service_output") or {}
            verdict = path["classifier_verdict"] or {}
            row_no = len(sheet) + 1
            code = CODES[index] if blind else path["path"]
            target = "yes" if output else f"no ({path['status']})"
            sheet.append({
                "sheet_row": row_no,
                "example_id": example.get("id", ""),
                "output_code": code,
                "source_text": example.get("source_text") or "",
                "evidence": record["input"]["evidence"],
                "claim": record["input"]["claim"],
                "output_label": output.get("label", ""),
                "explanation": output.get("explanation", ""),
                "suggestion": output.get("suggestion") or "",
                "status": path["status"],
                "eval_target": target,
                **{column: "" for column in research.EVAL_COLUMNS},  # 출력이 없는 행도 비워 둔다
            })
            key.append({
                "sheet_row": row_no,
                "example_id": example.get("id", ""),
                "output_code": code,
                "path": path["path"],
                "run_id": record["run_id"],
                "reference_label": example.get("reference_label") or "",
                "classifier_label": verdict.get("label", ""),
                "classifier_confidence": verdict.get("confidence", ""),
            })
    return sheet, key


def to_csv(rows, columns):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row[column] is None else row[column] for column in columns})
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="비교 실행 폴더 이름")
    parser.add_argument("--no-blind", action="store_true", help="경로 이름을 그대로 적는다")
    parser.add_argument("--seed", type=int, default=0, help="경로 코드를 섞는 순서")
    args = parser.parse_args()

    run_dir = runner.OUTPUT_ROOT / args.name
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    sheet, key = build(results, blind=not args.no_blind, seed=args.seed)
    for name, rows, columns in (("eval_sheet.csv", sheet, SHEET_COLUMNS), ("eval_key.csv", key, KEY_COLUMNS)):
        if name == "eval_key.csv" and args.no_blind:
            continue
        if (run_dir / name).exists():
            raise SystemExit(f"이미 있습니다(덮어쓰지 않음): {run_dir / name}")
        with open(run_dir / name, "w", encoding="utf-8-sig", newline="") as file:
            file.write(to_csv(rows, columns))
        print(f"저장: {run_dir / name} ({len(rows)}행)")


if __name__ == "__main__":
    main()
