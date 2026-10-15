"""연구용: 개발용 고정 댓글로 세 경로를 비교하고 JSON·CSV로 저장한다.

- 입력: data/comments/generation_development.jsonl (validation에서 뽑은 15건). 최종 test 데이터는 쓰지 않는다.
- 모델 입력은 댓글 text뿐이다. 정답 라벨(reference_label)은 기록에만 남긴다.
- 워밍업 1회는 따로 기록하고 측정에서 뺀다.
- 예제마다 실행할 수 있는 경로의 순서를 돌려 가며 바꾼다. 실행하지 못하는 경로(모델 미준비)는 순서에서 빼고
  unavailable로 기록하며 summary 통계에서도 뺀다.
- 사람 평가 열(정책 준수, 의미 보존, 새 사실 추가, 메모)은 비워 둔다. 자동 품질 점수는 만들지 않는다.

- --paths로 일부 경로만 실행할 수 있다(예: 이미 실행한 비교에 logreg만 더할 때). 선택하지 않은 경로는 기록하지 않는다.

실행(저장소 루트):
  uv run python scripts/run_comments_research_compare.py check01 --limit 2
  uv run python scripts/run_comments_research_compare.py dev15_qwen3_logreg01 --paths logreg
결과: artifacts/step_by_step/comments/bench01/research/pipeline_compare/<이름>/results.json, results.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import comments_research_pipeline as research

OUTPUT_ROOT = research.LEARNER_ROOT / "research" / "pipeline_compare"
WARMUP_TEXT = "워밍업용 댓글입니다. 측정에 넣지 않습니다."


def rotated_order(index, available=research.PATH_IDS):
    """예제 번호에 따라 실행할 수 있는 경로의 순서를 한 칸씩 돌린다.

    실행하지 못하는 경로(모델 미준비 등)는 회전에서 빼고 뒤에 붙인다. 두 경로만 있으면 예제마다 앞뒤가 바뀐다.
    """
    runnable = [path_id for path_id in research.PATH_IDS if path_id in available]
    shift = index % len(runnable)
    rest = [path_id for path_id in research.PATH_IDS if path_id not in runnable]
    return runnable[shift:] + runnable[:shift] + rest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="결과 폴더 이름 (이미 있으면 멈춘다)")
    parser.add_argument("--llm", default=research.DEFAULT_LLM, choices=list(research.REGISTERED_LLMS))
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 실행")
    parser.add_argument("--paths", nargs="+", choices=research.PATH_IDS, default=list(research.PATH_IDS),
                        help="실행할 경로 (기본: 세 경로 모두)")
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.name
    if out_dir.exists():
        raise SystemExit(f"결과 폴더가 이미 있습니다: {out_dir}")

    selected = [path_id for path_id in research.PATH_IDS if path_id in args.paths]
    examples = research.load_dev_examples()[: args.limit]
    # 선택한 경로에 필요한 분류기만 불러온다.
    needed = {path["classifier"] for path in research.PATHS if path["id"] in selected} - {None}
    classifiers = research.load_classifiers(
        {name: run for name, run in research.CLASSIFIER_RUNS.items() if name in needed}
    )
    print("실행 경로:", ", ".join(selected))
    for name, entry in classifiers.items():
        info = entry["info"]
        print(f"분류기 {name}: {'사용' if info['available'] else '사용 불가'} {info.get('model_id') or info['error']}")
    research.warm_up_classifiers(classifiers)

    warmup = research.run_comparison(WARMUP_TEXT, args.llm, classifiers, selected=selected)
    print("워밍업:", ", ".join(f"{p['path']}={p['status']}" for p in warmup["paths"]))

    available = [p for p in research.available_path_ids(classifiers) if p in selected]
    records = []
    for index, example in enumerate(examples):
        record = research.run_comparison(
            example["text"], args.llm, classifiers, order=rotated_order(index, available),
            selected=selected,
        )
        record["example"] = {"id": example["id"], "reference_label": example["reference_label"]}
        records.append(record)
        print(f"[{index + 1}/{len(examples)}] {example['id']} 순서={'→'.join(record['executed'])} "
              + ", ".join(f"{p['path']}={p['status']}" for p in record["paths"]))

    out_dir.mkdir(parents=True)
    summary = {
        "schema": research.SCHEMA,
        "examples_file": str(research.DEV_EXAMPLES.relative_to(research.ROOT)),
        "llm": {"key": args.llm, **research.REGISTERED_LLMS[args.llm]},
        "selected_paths": selected,  # 이 실행에서 실행·기록한 경로
        "generation_config": research.generation_config(args.llm),
        "policy_version": research.POLICY_VERSION,
        "classifiers": {name: entry["info"] for name, entry in classifiers.items()},
        "warmup": warmup,  # 측정에서 뺀 실행
        # 경로별 상태 개수와 시간·토큰 평균·중앙값. 실행하지 않은 경로는 excluded, 값은 null(해당 없음)
        "summary": research.summarize(records),
        "records": records,
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다. csv 모듈이 줄 끝을 정하므로 newline=""로 쓴다.
    with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as file:
        file.write(research.to_csv(records))
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
