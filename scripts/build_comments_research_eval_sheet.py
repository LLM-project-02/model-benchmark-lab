"""연구용: 두 비교 실행의 결과를 합쳐 사람 평가용 CSV 하나를 만든다 (새 LLM 호출 없음).

- 기준 실행(--base)에서 고른 경로(--base-paths)의 행은 원본 results.csv 값에, 평가본(results_eval.csv)의 eval_* 값을
  example_id·path로 맞춰 그대로 붙인다. 입력·상태·답변이 원본과 다르면 멈춘다(평가본은 시트를 거쳐 숫자 형식이 바뀔 수 있어
  평가 열만 가져온다).
- 추가 실행(--add)에서 고른 경로(--add-paths)의 행은 새 결과 그대로 두고 eval_* 네 열은 비운다.
- 기준 실행에서 고르지 않은 경로의 행(예: 출력이 없던 logreg 행)과 그 평가값은 가져오지 않는다.
- 맨 앞 source_run 열에 행이 나온 실행 이름을 적는다.

실행(저장소 루트):
  uv run python scripts/build_comments_research_eval_sheet.py
결과: <추가 실행 폴더>/three_paths_eval.csv
"""

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import comments_research_pipeline as research

OUTPUT_ROOT = research.LEARNER_ROOT / "research" / "pipeline_compare"
EVAL_COLUMNS = ("eval_policy_compliance", "eval_meaning_preserved", "eval_new_facts", "eval_memo")
CHECKED = ("example_id", "input_text", "status", "answer")  # 평가본이 원본과 같은 행인지 확인하는 열


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def keyed(rows, where):
    table = {}
    for row in rows:
        key = (row["example_id"], row["path"])
        if key in table:
            raise SystemExit(f"{where}에 같은 example_id·path 행이 두 번 있습니다: {key}")
        table[key] = row
    return table


def build(base_dir, add_dir, base_paths, add_paths, eval_file="results_eval.csv"):
    base_rows = read_csv(base_dir / "results.csv")
    evaluated = keyed(read_csv(base_dir / eval_file), f"{base_dir.name}/{eval_file}")
    original = keyed(base_rows, f"{base_dir.name}/results.csv")
    if set(evaluated) != set(original):
        raise SystemExit("평가본과 원본 results.csv의 행이 맞지 않습니다.")
    for key, row in original.items():
        for column in CHECKED:
            if evaluated[key][column] != row[column]:
                raise SystemExit(f"평가본의 {column} 값이 원본과 다릅니다: {key}")
    added = keyed(read_csv(add_dir / "results.csv"), f"{add_dir.name}/results.csv")

    order = list(dict.fromkeys(row["example_id"] for row in base_rows))
    inputs = {row["example_id"]: row["input_text"] for row in base_rows}
    if sorted({key[0] for key in added}) != sorted(order):
        raise SystemExit("두 실행의 예제가 같지 않습니다.")

    rows = []
    for example_id in order:
        for path_id in research.PATH_IDS:
            if path_id in base_paths:
                row = dict(original[(example_id, path_id)])
                row.update({column: evaluated[(example_id, path_id)][column] for column in EVAL_COLUMNS})
                source = base_dir.name
            elif path_id in add_paths:
                row = dict(added[(example_id, path_id)])
                if row["input_text"] != inputs[example_id]:
                    raise SystemExit(f"두 실행의 입력 문장이 다릅니다: {example_id}")
                row.update({column: "" for column in EVAL_COLUMNS})  # 새 행은 평가하지 않은 상태
                source = add_dir.name
            else:
                continue
            rows.append({"source_run": source, **row})
    return rows


def to_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["source_run", *research.CSV_COLUMNS])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="dev15_qwen3_01", help="평가본이 있는 기준 실행")
    parser.add_argument("--base-paths", nargs="+", default=["llm_only", "kcelectra"], choices=research.PATH_IDS)
    parser.add_argument("--add", default="dev15_qwen3_logreg01", help="새로 더할 실행")
    parser.add_argument("--add-paths", nargs="+", default=["logreg"], choices=research.PATH_IDS)
    parser.add_argument("--output", default="three_paths_eval.csv", help="추가 실행 폴더 안에 만들 파일 이름")
    args = parser.parse_args()
    if set(args.base_paths) & set(args.add_paths):
        raise SystemExit("한 경로를 두 실행에서 함께 가져올 수 없습니다.")

    add_dir = OUTPUT_ROOT / args.add
    out = add_dir / args.output
    if out.exists():
        raise SystemExit(f"이미 있습니다(덮어쓰지 않음): {out}")
    rows = build(OUTPUT_ROOT / args.base, add_dir, args.base_paths, args.add_paths)
    # 엑셀·구글 시트에서 한글이 깨지지 않게 BOM을 붙인다.
    with open(out, "w", encoding="utf-8-sig", newline="") as file:
        file.write(to_csv(rows))
    filled = sum(1 for row in rows if any(row[column] for column in EVAL_COLUMNS))
    print(f"행 {len(rows)}개 (예제 {len({row['example_id'] for row in rows})}개), 평가값이 있는 행 {filled}개")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
