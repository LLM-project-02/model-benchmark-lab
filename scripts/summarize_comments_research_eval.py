"""연구용: 파이프라인 비교 결과(results.json)와 사람 평가본을 대조해 집계한다.

- 평가본의 행은 source_run·example_id·path로 원본 기록에 대응시킨다. source_run 열이 없으면 평가본이 있는
  실험 폴더의 실행으로 본다. 평가 열 밖의 값은 원본 results.csv와 비교해 표시 형식 차이(TRUE/True, 소수 자릿수)와
  내용 차이를 나눠 센다. 내용 차이가 있으면 멈춘다.
- 평가값(eval_*)은 고치지 않고 그대로 읽는다. yes/no만 비율에 넣고, 빈칸과 그 밖의 값은 제외 건수와 사유로 따로 센다.
  빈칸의 사유는 기록에서 알 수 있는 것(출력 없음 등)만 적고, 알 수 없으면 "사유 기록 없음"으로 둔다(메모로 추정하지 않음).
- 출력 품질은 고른 경로(--paths)가 모두 success인 예제끼리(같은 example_id) 비교한다. 항목마다 모든 경로에
  yes/no가 있는 예제(common)만 비율에 넣는다.
- 정책 준수·의미 보존은 yes 비율, 새 사실 추가는 yes가 문제 발생 비율이다.
- 시간·토큰은 평가본이 아니라 각 행의 source_run에 해당하는 원본 results.json에서 읽는다. 실행 시각이 다른 경로의
  시간은 같은 조건의 측정이 아니므로 실행 시각을 함께 남긴다.
- answer가 입력과 같은지(unchanged_output)는 문자열 비교로 표시만 한다. 품질 점수를 새로 만들지 않는다.
- 새 LLM 호출이나 분류 추론은 하지 않는다.

실행(저장소 루트):
  uv run python scripts/summarize_comments_research_eval.py dev15_qwen3_01 --paths llm_only kcelectra
  uv run python scripts/summarize_comments_research_eval.py dev15_qwen3_logreg01 --eval-file three_paths_eval_final.csv
결과: artifacts/step_by_step/comments/bench01/research/pipeline_compare/<이름>/human_eval_summary.json
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts" / "step_by_step" / "comments" / "bench01" / "research" / "pipeline_compare"

PATH_IDS = ("llm_only", "logreg", "kcelectra")
EVAL_COLUMNS = {
    "eval_policy_compliance": "정책 준수 (yes 비율)",
    "eval_meaning_preserved": "의미 보존 (yes 비율)",
    "eval_new_facts": "새 사실 추가 (yes = 문제 발생 비율)",
}
EVAL_FIELDS = (*EVAL_COLUMNS, "eval_memo")
METRICS = (
    ("input_tokens", "generation", "input_tokens"),
    ("output_tokens", "generation", "output_tokens"),
    ("classification_seconds", "timings", "classification_seconds"),
    ("llm_seconds", "timings", "llm_seconds"),
    ("total_seconds", "timings", "total_seconds"),
)
NO_REASON = "사유 기록 없음"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_run(run_dir):
    """실행 하나의 원본 기록: (example_id, path) → 경로 기록, 예제 순서, 원본 CSV 행."""
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    items = {}
    order = []
    for record in data["records"]:
        example = record["example"]
        order.append(example["id"])
        for path in record["paths"]:
            items[(example["id"], path["path"])] = {
                "example": example,
                "input_text": record["input"]["text"],
                "created_at": record["created_at"],
                "path": path,
            }
    csv_rows = {(row["example_id"], row["path"]): row for row in read_csv(run_dir / "results.csv")}
    return {"data": data, "items": items, "order": order, "csv": csv_rows}


def compare_value(original, sheet):
    """원본 CSV 값과 평가본 값: same, format(표시 형식만 다름), content(내용이 다름)."""
    if original == sheet:
        return "same"
    if original.lower() == sheet.lower():  # 시트가 True를 TRUE로 바꾼 경우
        return "format"
    try:
        a, b = float(original), float(sheet)
    except ValueError:
        return "content"
    return "format" if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) else "content"  # 소수 자릿수 축소


def match_rows(rows, default_run, paths):
    """평가본 행을 원본 기록에 대응시키고 평가 열 밖의 차이를 센다."""
    names = sorted({row.get("source_run") or default_run for row in rows})
    runs = {name: load_run(OUTPUT_ROOT / name) for name in names}
    entries = {}
    format_diffs, content_diffs = {}, []
    for row in rows:
        source = row.get("source_run") or default_run
        key = (row["example_id"], row["path"])
        if key in entries:
            raise SystemExit(f"평가본에 같은 example_id·path 행이 두 번 이상 있습니다: {key}")
        run = runs[source]
        if key not in run["items"]:
            raise SystemExit(f"{source}의 원본 기록에 없는 행입니다: {key}")
        for column, value in run["csv"][key].items():
            if column in EVAL_FIELDS:
                continue
            kind = compare_value(value, row[column])
            if kind == "format":
                format_diffs[column] = format_diffs.get(column, 0) + 1
            elif kind == "content":
                content_diffs.append({"source_run": source, "example_id": key[0], "path": key[1], "column": column})
        entries[key] = {"row": row, "source": source, "item": run["items"][key]}
    if content_diffs:
        raise SystemExit(f"평가본의 평가 열 밖 내용이 원본과 다릅니다: {content_diffs[:5]}")
    for path_id in paths:
        if not any(key[1] == path_id for key in entries):
            raise SystemExit(f"평가본에 {path_id} 행이 없습니다.")
    alignment = {
        "rows": len(rows),
        "matched_by": ["source_run", "example_id", "path"],
        "source_runs": names,
        "content_diffs": content_diffs,
        "format_only_diffs": format_diffs,  # 열 이름 → 표시 형식만 다른 칸 수
        "row_order_same_as_source": [
            (row.get("source_run") or default_run, row["example_id"], row["path"]) for row in rows
        ] == sorted(
            [(row.get("source_run") or default_run, row["example_id"], row["path"]) for row in rows],
            key=lambda key: (runs[key[0]]["order"].index(key[1]), PATH_IDS.index(key[2])),
        ),
    }
    return runs, entries, alignment


def exact_mcnemar_p(only_a, only_b):
    """짝지은 yes/no의 엇갈린 건수로 구한 정확 McNemar(양측) p값. 표본이 작아 참고용이다."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def value_of(entries, example_id, path_id, column):
    return entries[(example_id, path_id)]["row"][column]


def summarize_column(column, entries, paths, compared_ids):
    per_path = {}
    blanks = []
    for path_id in paths:
        counts = {"yes": 0, "no": 0, "blank": 0, "other": 0}
        for example_id in compared_ids:
            value = value_of(entries, example_id, path_id, column)
            if value in ("yes", "no"):
                counts[value] += 1
                continue
            kind = "blank" if not value.strip() else "other"
            counts[kind] += 1
            blanks.append({
                "example_id": example_id,
                "path": path_id,
                "value": value,
                "reason": NO_REASON if kind == "blank" else "yes/no가 아닌 값이라 비율에서 제외",
                "memo": entries[(example_id, path_id)]["row"]["eval_memo"],  # 그대로 (빈칸 사유로 해석하지 않음)
            })
        evaluated = counts["yes"] + counts["no"]
        per_path[path_id] = {**counts, "evaluated": evaluated,
                             "yes_rate": counts["yes"] / evaluated if evaluated else None}

    # 고른 경로 모두에 yes/no가 있는 예제끼리 비교한다.
    common_ids = [
        example_id for example_id in compared_ids
        if all(value_of(entries, example_id, p, column) in ("yes", "no") for p in paths)
    ]
    common = {"n": len(common_ids), "example_ids": common_ids,
              "excluded_example_ids": [i for i in compared_ids if i not in common_ids]}
    for path_id in paths:
        yes = sum(1 for i in common_ids if value_of(entries, i, path_id, column) == "yes")
        common[path_id] = {"yes": yes, "yes_rate": yes / len(common_ids) if common_ids else None}

    pairwise = {}
    for a, b in itertools.combinations(paths, 2):
        counts = {"both_yes": 0, f"only_{a}_yes": 0, f"only_{b}_yes": 0, "both_no": 0}
        for i in common_ids:
            va, vb = value_of(entries, i, a, column), value_of(entries, i, b, column)
            if va == vb:
                counts["both_yes" if va == "yes" else "both_no"] += 1
            else:
                counts[f"only_{a}_yes" if va == "yes" else f"only_{b}_yes"] += 1
        pairwise[f"{a}~{b}"] = {
            "n": len(common_ids), **counts,
            "exact_mcnemar_p": exact_mcnemar_p(counts[f"only_{a}_yes"], counts[f"only_{b}_yes"]),
        }
    return {"title": EVAL_COLUMNS[column], "per_path": per_path, "common": common,
            "pairwise": pairwise, "blanks": blanks}


def describe(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def summarize_metrics(entries, paths, compared_ids):
    """같은 비교 대상의 토큰·시간 (경로별로 그 행의 원본 results.json 값). 분류 시간이 없는 경로는 None."""
    result = {}
    for name, group, key in METRICS:
        result[name] = {}
        for path_id in paths:
            values = [
                entries[(i, path_id)]["item"]["path"]["result"][group][key] for i in compared_ids
            ]
            result[name][path_id] = describe([value for value in values if value is not None])
    return result


def is_unchanged(entry):
    output = (entry["item"]["path"]["result"] or {}).get("service_output") or {}
    answer = output.get("answer")
    return answer is not None and answer.strip() == entry["item"]["input_text"].strip()


def unchanged_outputs(entries, paths, compared_ids):
    """answer가 입력 댓글과 글자까지 같은 경우. 참고 라벨이 offensive·hate면 고치지 않고 돌려준 것이다."""
    result = {}
    for path_id in paths:
        items = []
        for i in compared_ids:
            entry = entries[(i, path_id)]
            if is_unchanged(entry):
                row = entry["row"]
                items.append({
                    "example_id": i,
                    "reference_label": entry["item"]["example"]["reference_label"],
                    "applied_policy_label": entry["item"]["path"]["applied_policy_label"],
                    **{column: row[column] for column in EVAL_COLUMNS},
                })
        by_reference = {}
        for item in items:
            by_reference[item["reference_label"]] = by_reference.get(item["reference_label"], 0) + 1
        result[path_id] = {"count": len(items), "by_reference_label": by_reference, "items": items}
    return result


def policy_meaning_crosstab(entries, paths, compared_ids):
    """정책 준수 × 의미 보존. 고른 경로 모두에 두 항목의 yes/no가 있는 예제끼리 센다.

    원문을 고치지 않고 돌려주면 의미 보존은 yes가 되기 쉬우므로, 의미 보존 yes만으로 좋은 출력이라고 볼 수 없음을 보이는 표다.
    """
    columns = ("eval_policy_compliance", "eval_meaning_preserved")
    ids = [
        i for i in compared_ids
        if all(entries[(i, p)]["row"][c] in ("yes", "no") for p in paths for c in columns)
    ]
    result = {"n": len(ids), "example_ids": ids}
    for path_id in paths:
        table = {}
        for i in ids:
            row = entries[(i, path_id)]["row"]
            name = f"policy_{row[columns[0]]}__meaning_{row[columns[1]]}"
            table[name] = table.get(name, 0) + 1
        result[path_id] = table
    return result


def by_reference_label(entries, paths, compared_ids):
    """항목별 yes/평가 수를 참고 라벨(데이터의 reference_label)별로 나눈다. 모델 입력·채점에는 쓰지 않은 값이다."""
    result = {}
    for column in EVAL_COLUMNS:
        result[column] = {}
        for path_id in paths:
            groups = {}
            for i in compared_ids:
                entry = entries[(i, path_id)]
                value = entry["row"][column]
                if value not in ("yes", "no"):
                    continue
                label = entry["item"]["example"]["reference_label"]
                group = groups.setdefault(label, {"yes": 0, "evaluated": 0})
                group["evaluated"] += 1
                group["yes"] += value == "yes"
            result[column][path_id] = groups
    return result


def out_of_scope_values(entries, paths, compared_ids):
    """비교 대상이 아닌 행에 들어 있는 평가값. 집계에 넣지 않고 따로 알린다."""
    found = []
    for (example_id, path_id), entry in entries.items():
        if path_id in paths and example_id in compared_ids:
            continue
        row = entry["row"]
        values = {column: row[column] for column in EVAL_FIELDS if row[column].strip()}
        if not values:
            continue
        status = entry["item"]["path"]["status"]
        if status == "unavailable":
            reason = "실행 안 함(모델 미준비), 출력 없음"
        elif path_id not in paths:
            reason = "비교에 고르지 않은 경로"
        elif status != "success":
            reason = f"{status}, 비교 대상 아님"
        else:
            reason = "다른 경로가 완료되지 않아 비교 대상 아님"
        found.append({"source_run": entry["source"], "example_id": example_id, "path": path_id,
                      "status": status, "reason": reason, "values": values})
    return found


def per_example(entries, paths, order, compared_ids):
    table = []
    for example_id in order:
        first = entries[(example_id, paths[0])]["item"]
        item = {"example_id": example_id, "reference_label": first["example"]["reference_label"],
                "input_text": first["input_text"], "compared": example_id in compared_ids}
        for path_id in paths:
            entry = entries[(example_id, path_id)]
            path = entry["item"]["path"]
            verdict = path["classifier_verdict"] or {}
            output = (path["result"] or {}).get("service_output") or {}
            item[path_id] = {
                "source_run": entry["source"],
                "status": path["status"],
                "classifier_label": verdict.get("label"),
                "classifier_confidence": verdict.get("confidence"),
                "llm_label": path["raw_label"],
                "applied_policy_label": path["applied_policy_label"],
                "answer": output.get("answer"),
                "next_steps": output.get("next_steps"),
                "unchanged_output": is_unchanged(entry),
                **{column: entry["row"][column] for column in EVAL_FIELDS},
            }
        table.append(item)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="평가본이 있는 실험 폴더 이름 (예: dev15_qwen3_logreg01)")
    parser.add_argument("--eval-file", default="results_eval.csv", help="실험 폴더 안의 사람 평가본")
    parser.add_argument("--paths", nargs="+", default=list(PATH_IDS), choices=PATH_IDS, help="비교할 경로")
    parser.add_argument("--output", default="human_eval_summary.json",
                        help="결과 파일 (상대 경로면 실험 폴더 안)")
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.name
    paths = [path_id for path_id in PATH_IDS if path_id in args.paths]
    rows = read_csv(run_dir / args.eval_file)
    runs, entries, alignment = match_rows(rows, args.name, paths)

    order = runs[entries[next(k for k in entries if k[1] == paths[0])]["source"]]["order"]
    for path_id in paths:
        if sorted(i for i, p in entries if p == path_id) != sorted(order):
            raise SystemExit(f"{path_id} 행의 예제가 다른 경로와 같지 않습니다.")

    statuses = {}
    for path_id in paths:
        counts = {}
        for example_id in order:
            status = entries[(example_id, path_id)]["item"]["path"]["status"]
            counts[status] = counts.get(status, 0) + 1
        ran = len(order) - counts.get("unavailable", 0)
        statuses[path_id] = {
            "source_run": entries[(order[0], path_id)]["source"],
            "counts": counts,
            "completed": counts.get("success", 0),
            "ran": ran,
            "completion_rate": counts.get("success", 0) / ran if ran else None,
            "not_success": [
                {"example_id": i, "status": entries[(i, path_id)]["item"]["path"]["status"],
                 "error": (entries[(i, path_id)]["item"]["path"]["result"] or {}).get("error")}
                for i in order if entries[(i, path_id)]["item"]["path"]["status"] not in ("success", "unavailable")
            ],
        }

    compared_ids = [
        i for i in order if all(entries[(i, p)]["item"]["path"]["status"] == "success" for p in paths)
    ]
    source_runs = {}
    for name, run in runs.items():
        times = [record["created_at"] for record in run["data"]["records"]]
        source_runs[name] = {
            "first_record_at": min(times),
            "last_record_at": max(times),
            "llm": run["data"]["llm"],
            "generation_config": run["data"]["generation_config"],
            "policy_version": run["data"]["policy_version"],
        }
    summary = {
        "run": args.name,
        "eval_file": args.eval_file,
        "alignment": alignment,
        "paths_compared": paths,
        "source_runs": source_runs,
        "statuses": statuses,
        "compared_example_ids": compared_ids,
        "human_eval": {column: summarize_column(column, entries, paths, compared_ids) for column in EVAL_COLUMNS},
        "policy_meaning_crosstab": policy_meaning_crosstab(entries, paths, compared_ids),
        "human_eval_by_reference_label": by_reference_label(entries, paths, compared_ids),
        "unchanged_outputs": unchanged_outputs(entries, paths, compared_ids),
        "out_of_scope_values": out_of_scope_values(entries, paths, compared_ids),
        "metrics": summarize_metrics(entries, paths, compared_ids),
        "per_example": per_example(entries, paths, order, compared_ids),
        "notes": [
            "평가자 1명, 개발 예제 15개의 예비 결과다.",
            "비율의 분모는 항목마다 비교 경로 모두에 yes/no가 적힌 예제 수(common.n)다. 빈칸은 no로 세지 않는다.",
            "exact_mcnemar_p는 표본이 작아 참고용이다.",
            "실행 시각이 다른 source_run의 시간 값은 같은 조건의 측정이 아니다.",
        ],
    }
    out = Path(args.output)
    out = out if out.is_absolute() else run_dir / out
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"대응 확인: {alignment['rows']}행, 내용 차이 0건, 표시 형식만 다른 열 {alignment['format_only_diffs']}")
    print(f"비교 대상(경로 {', '.join(paths)} 모두 성공): {len(compared_ids)}개")
    for entry in summary["human_eval"].values():
        common = entry["common"]
        rates = ", ".join(f"{p} {common[p]['yes']}/{common['n']}" for p in paths)
        print(f"{entry['title']}: 비교 {common['n']}개 — {rates}")
    print(f"비교 대상 밖 평가값: {len(summary['out_of_scope_values'])}행")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
